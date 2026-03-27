"""
MPC-Only Controller for Docking6D

Gradient-based MPC controller using differentiable shooting with multi-start
Adam optimization. No learned value function is used -- this serves as a
pure-MPC baseline for comparison against the BRAT controller.

At each simulation step the controller:
  1. Optimises control sequences over the full planning horizon via
     differentiable rollout + Adam (multi-start).
  2. Evaluates a combined cost: analytical reach-avoid + goal-directed
     quadratic regularisation (to compensate for tanh gradient saturation
     in reach_fn at distant states).
  3. Applies the first control and warm-starts the next step.

Usage:
    controller = MPCController(
        checkpoint_path='./runs/Docking6D_RA/training/checkpoints/model_final.pth',
        planning_horizon_sec=20.0,
        mpc_dt=0.5,
    )
    result = controller.simulate_docking(initial_state, max_sim_time=30.0)
"""

import time
import torch
import numpy as np
import pickle
import os
import sys
import inspect
from tqdm import tqdm

# Add project root to path (3 levels up: controllers -> utils -> project_root)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

from utils.gradient_mpc import GradientMPC
from dynamics import dynamics as dynamics_module
from utils.controllers.safety_filter import SafetyFilter
from utils.controllers import clip_state_for_execution


class MPCController:
    """
    Gradient-based MPC controller for Docking6D.

    At each simulation step the controller:
      1. Optimises control sequences via differentiable shooting + Adam
         with ``num_restarts`` parallel random restarts.
      2. Evaluates a combined cost: analytical reach-avoid (from dynamics)
         plus a quadratic goal-directed term for gradient guidance.
      3. Applies the first control and warm-starts the next step.

    No learned value function is used; cost evaluation relies solely on
    analytical reach_fn / avoid_fn plus goal-directed regularisation.
    """

    def __init__(self, checkpoint_path, planning_horizon_sec=20.0,
                 mpc_dt=0.5, dt=0.1, device='cuda',
                 gradient_lr=1.0, gradient_iters=80, num_restarts=16,
                 goal_weight=0.01, safety_filter=None):
        """
        Args:
            checkpoint_path:      Path to a trained model checkpoint (used only
                                  to load the dynamics config from orig_opt.pickle).
            planning_horizon_sec: MPC planning horizon in seconds (default 20).
            mpc_dt:               MPC planning timestep in seconds (default 0.5).
            dt:                   Simulation integration timestep (default 0.1).
            device:               Torch device ('cuda' or 'cpu').
            gradient_lr:          Adam learning rate (default 1.0).
            gradient_iters:       Adam iterations per MPC step (default 80).
            num_restarts:         Parallel random restarts (default 16).
            goal_weight:          Weight for goal-directed regularisation (default 0.01).
        """
        self.checkpoint_path = checkpoint_path
        self.planning_horizon_sec = planning_horizon_sec
        self.mpc_dt = mpc_dt
        self.dt = dt
        self.planning_horizon_steps = int(planning_horizon_sec / mpc_dt)
        self.device = device
        self.goal_weight = goal_weight

        # Derive experiment directory from checkpoint path
        self.experiment_dir = os.path.dirname(
            os.path.dirname(os.path.dirname(checkpoint_path)))

        # Load dynamics
        self._load_dynamics()

        # Instantiate the gradient MPC solver
        self.gradient_mpc = GradientMPC(
            dt=self.mpc_dt,
            horizon=self.planning_horizon_steps,
            dynamics=self.dynamics,
            device=self.device,
            num_iters=gradient_iters,
            lr=gradient_lr,
            num_restarts=num_restarts,
        )

        # Safety filter (no-op when mode=0 or None)
        self.safety_filter = safety_filter or SafetyFilter(mode=0)

        # Control state
        self.reset()

        print(f"MPCController initialised  |  horizon={self.planning_horizon_sec}s  "
              f"mpc_dt={self.mpc_dt}s  sim_dt={self.dt}s  "
              f"horizon_steps={self.planning_horizon_steps}  "
              f"iters={gradient_iters}  restarts={num_restarts}  "
              f"lr={gradient_lr}  goal_weight={goal_weight}")

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _load_dynamics(self):
        """Load the dynamics class from the experiment's saved options."""
        opt_path = os.path.join(self.experiment_dir, 'orig_opt.pickle')
        with open(opt_path, 'rb') as f:
            self.orig_opt = pickle.load(f)

        dynamics_class = getattr(dynamics_module, self.orig_opt.dynamics_class)
        sig = inspect.signature(dynamics_class)
        kwargs = {}
        for param_name in sig.parameters.keys():
            if param_name != 'self' and hasattr(self.orig_opt, param_name):
                kwargs[param_name] = getattr(self.orig_opt, param_name)

        self.dynamics = dynamics_class(**kwargs)
        self.dynamics.set_model(self.orig_opt.deepReach_model)

        # Fallback: ensure normalization matches training even if
        # state_range was not passed through the constructor
        if hasattr(self.orig_opt, 'state_range') and self.orig_opt.state_range is not None:
            self.dynamics.override_state_range(self.orig_opt.state_range)

    def reset(self):
        """Reset controller state for a new simulation."""
        self.state_history = []
        self.control_history = []
        self.value_history = []
        self.sim_time_history = []
        self._warm_controls = None
        self.safety_filter.reset()

    # ------------------------------------------------------------------
    # Cost function
    # ------------------------------------------------------------------

    def _build_cost_fn(self):
        """Build cost function for baseline gradient MPC.

        Combines:
          1. Analytical reach-avoid cost (dynamics.cost_fn)
          2. Quadratic goal-directed regularisation — provides useful gradients
             far from the goal where reach_fn's tanh saturates.
        """
        goal = self.dynamics.goal_state.to(self.device).float()
        goal_weight = self.goal_weight

        def cost_fn(trajectory, controls):
            # 1. Analytical reach-avoid cost
            reach_avoid = self.dynamics.cost_fn(trajectory)  # (K,)

            # 2. Goal-directed regularisation (provides gradient far from goal)
            terminal = trajectory[:, -1, :]  # (K, 6)

            pos_err = terminal[:, :2] - goal[:2]
            vel_err = terminal[:, 2:4] - goal[2:4]
            theta_err = torch.atan2(
                torch.sin(terminal[:, 4] - goal[4]),
                torch.cos(terminal[:, 4] - goal[4]))
            omega_err = terminal[:, 5] - goal[5]

            # Weights from dynamics.Q = diag([3,3,10,10,5,5])
            goal_cost = (3.0 * (pos_err ** 2).sum(-1)
                         + 10.0 * (vel_err ** 2).sum(-1)
                         + 5.0 * theta_err ** 2
                         + 5.0 * omega_err ** 2)

            return reach_avoid + goal_weight * goal_cost

        return cost_fn

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def simulate_docking(self, initial_state, max_sim_time, dynamics_fn=None):
        """
        Run a full docking simulation using gradient-based MPC.

        The MPC replans at every simulation step (dt).

        Args:
            initial_state: Initial state [px, py, vx, vy, theta, omega].
            max_sim_time:  Maximum simulation duration in seconds.
            dynamics_fn:   Optional f(state, control) -> state_dot.
                           Defaults to Docking6D CW equations.

        Returns:
            dict with the shared result format.
        """
        self.reset()
        t_wall_start = time.perf_counter()

        state = np.array(initial_state, dtype=np.float64)
        num_steps = int(max_sim_time / self.dt) + 1

        if dynamics_fn is None:
            dynamics_fn = self._default_dynamics_fn

        print(f"[MPC] Starting from state: {state}")

        docked = False
        collided = False
        n_clipped = 0

        pbar = tqdm(range(num_steps), desc="[MPC] Simulating",
                   unit="step", leave=True)
        for step in pbar:
            sim_time = step * self.dt

            # --- MPC replan every step ---
            control, cost_val = self._mpc_step(state)

            # Post-process through safety filter (no-op when mode=0)
            control = self.safety_filter.apply(state, control)

            # Record history
            self.state_history.append(state.copy())
            self.control_history.append(control.copy())
            self.value_history.append(cost_val)
            self.sim_time_history.append(sim_time)

            # Update progress bar
            dist = float(np.sqrt(state[0]**2 + state[1]**2))
            pbar.set_postfix(t=f"{sim_time:.1f}s", dist=f"{dist:.3f}m",
                             cost=f"{cost_val:.4f}")

            # Termination checks
            if self._check_docked(state):
                docked = True
                pbar.set_postfix(t=f"{sim_time:.1f}s", status="DOCKED")
                print(f"\n[MPC] Docking successful at t={sim_time:.2f}s")
                break

            if self._check_collision(state):
                collided = True
                pbar.set_postfix(t=f"{sim_time:.1f}s", status="COLLISION")
                print(f"\n[MPC] Collision detected at t={sim_time:.2f}s")
                break

            # Integrate dynamics at fine dt (Euler)
            state_dot = dynamics_fn(state, control)
            state = state + self.dt * state_dot
            state, clipped = clip_state_for_execution(state, self.dynamics)
            if clipped:
                n_clipped += 1
        pbar.close()

        wall_time = time.perf_counter() - t_wall_start

        # Compute control effort: sum(||u_k||_2 * dt)
        controls_arr = np.array(self.control_history)
        control_effort = float(
            np.sum(np.linalg.norm(controls_arr, axis=-1)) * self.dt)

        result = {
            'trajectory': np.array(self.state_history),
            'controls': controls_arr,
            'values': np.array(self.value_history),
            'times': np.array(self.sim_time_history),
            'success': docked and not collided,
            'collision': collided,
            'docked': docked,
            'final_state': state,
            'controller_type': 'mpc',
            'control_effort': control_effort,
            'wall_time': wall_time,
            # --- safety filter ---
            'safety_filter_mode': self.safety_filter.mode,
            'safety_filter_log': self.safety_filter.get_log(),
            # --- execution clamping ---
            'n_clipped_steps': n_clipped,
        }
        return result

    # ------------------------------------------------------------------
    # Batched simulation
    # ------------------------------------------------------------------

    def simulate_docking_batch(self, initial_states_np, max_sim_time):
        """Run docking simulations for multiple ICs in parallel on GPU.

        Uses batch_optimize to process all ICs simultaneously each step,
        giving a large speedup over sequential simulate_docking calls.

        Args:
            initial_states_np: (B, 6) numpy array of initial conditions.
            max_sim_time: Maximum simulation time (seconds).

        Returns:
            list of B result dicts (same format as simulate_docking).
        """
        B = len(initial_states_np)
        num_steps = int(max_sim_time / self.dt) + 1
        cdim = 3  # 6D control dim
        H = self.planning_horizon_steps

        t_wall_start = time.perf_counter()

        states = torch.tensor(
            initial_states_np, dtype=torch.float32, device=self.device)

        # Cache state test range for batch wrapping
        sr = self.dynamics.state_test_range()

        # Pre-allocate storage (timestep-major)
        state_buf = torch.zeros(num_steps, B, 6)
        ctrl_buf = torch.zeros(num_steps, B, cdim)
        cost_buf = torch.zeros(num_steps, B)
        sf_active_buf = torch.zeros(num_steps, B, dtype=torch.bool)

        # Per-IC tracking
        active = torch.ones(B, dtype=torch.bool, device=self.device)
        docked = torch.zeros(B, dtype=torch.bool, device=self.device)
        collided = torch.zeros(B, dtype=torch.bool, device=self.device)
        dock_step = torch.full((B,), num_steps, dtype=torch.long,
                               device=self.device)
        final_step = torch.zeros(B, dtype=torch.long, device=self.device)
        post_dock_steps = int(1.0 / self.dt)

        # Per-IC wall time
        ic_wall_time = np.zeros(B, dtype=np.float64)
        ic_terminated = np.zeros(B, dtype=bool)

        # Warm-start controls (B, H, cdim)
        warm = torch.zeros(B, H, cdim, device=self.device)

        cost_fn = self._build_cost_fn()

        for step in range(num_steps):
            # Coast check
            coast_done = docked & ((step - dock_step) >= post_dock_steps)
            active = active & ~coast_done & ~collided

            if not active.any():
                break

            # Record state
            state_buf[step] = states.detach().cpu()
            final_step[active] = step

            # --- MPC optimisation (all ICs simultaneously) ---
            best_controls, best_costs, _ = self.gradient_mpc.batch_optimize(
                states, cost_fn, warm)

            controls = best_controls[:, 0, :]  # (B, cdim)
            controls = controls * active.unsqueeze(-1).float()

            # --- Safety filter (vectorised, no-op when mode=0) ---
            controls, sf_active_step = self.safety_filter.batch_apply(
                states, controls, active_mask=active)
            sf_active_buf[step] = sf_active_step.cpu()

            ctrl_buf[step] = controls.detach().cpu()
            cost_buf[step] = best_costs.detach().cpu()

            # --- Termination checks ---
            with torch.no_grad():
                # Docking: reach_fn <= 0
                reach_vals = self.dynamics.reach_fn(states)
                newly_docked = active & ~docked & (reach_vals <= 0)
                docked = docked | newly_docked
                dock_step = torch.where(
                    newly_docked,
                    torch.tensor(step, device=self.device),
                    dock_step)

                # Collision: oriented 4-corner box
                coll_mask = _batch_check_collision_6d(states, self.dynamics)
                newly_collided = active & ~docked & coll_mask
                collided = collided | newly_collided

            # Record wall time for ICs that just terminated
            _now = time.perf_counter() - t_wall_start
            newly_done = (newly_docked | newly_collided).cpu().numpy()
            for idx in np.where(newly_done & ~ic_terminated)[0]:
                ic_wall_time[idx] = _now
                ic_terminated[idx] = True

            # --- Euler integration ---
            with torch.no_grad():
                state_dot = self.dynamics.dsdt(states, controls, None)
                states = states + self.dt * state_dot
                states = _batch_wrap_state_6d(states, sr)

            # Shift warm-start
            warm = torch.cat([
                best_controls[:, 1:, :].detach(),
                torch.zeros(B, 1, cdim, device=self.device),
            ], dim=1)

            if step % 50 == 0:
                n_a = int(active.sum().item())
                n_d = int(docked.sum().item())
                n_c = int(collided.sum().item())
                print(f'  [Batch MPC] step={step} t={step*self.dt:.1f}s  '
                      f'active={n_a} docked={n_d} coll={n_c}')

        wall_time = time.perf_counter() - t_wall_start

        # Assign wall time for ICs that never terminated (timed out)
        ic_wall_time[~ic_terminated] = wall_time

        # --- Build per-IC result dicts ---
        docked_np = docked.cpu().numpy()
        collided_np = collided.cpu().numpy()
        final_step_np = final_step.cpu().numpy()
        sf_active_np = sf_active_buf.numpy()

        results = []
        for i in range(B):
            n = int(final_step_np[i]) + 1
            traj_i = state_buf[:n, i].numpy()
            ctrl_i = ctrl_buf[:n, i].numpy()
            cost_i = cost_buf[:n, i].numpy()
            times_i = np.arange(n) * self.dt
            sf_i = sf_active_np[:n, i]

            effort = float(np.sum(np.linalg.norm(ctrl_i, axis=-1)) * self.dt)

            sf_log_i = [{'filter_active': bool(sf_i[s])} for s in range(n)]

            results.append({
                'trajectory': traj_i,
                'controls': ctrl_i,
                'values': cost_i,
                'times': times_i,
                'success': bool(docked_np[i] and not collided_np[i]),
                'collision': bool(collided_np[i]),
                'docked': bool(docked_np[i]),
                'final_state': traj_i[-1],
                'controller_type': 'mpc',
                'control_effort': effort,
                'wall_time': float(ic_wall_time[i]),
                'safety_filter_mode': self.safety_filter.mode,
                'safety_filter_log': sf_log_i,
                'n_clipped_steps': 0,
            })

        mean_ic_wall = float(np.mean(ic_wall_time))
        print(f'  [Batch MPC] Done: {B} ICs in {wall_time:.1f}s '
              f'(mean {mean_ic_wall:.2f}s/IC)  '
              f'dock={int(docked_np.sum())} coll={int(collided_np.sum())}')

        return results

    # ------------------------------------------------------------------
    # Core MPC logic (gradient-based)
    # ------------------------------------------------------------------

    def _mpc_step(self, state):
        """Run one gradient-based MPC step and return (first_control, best_cost)."""
        state_tensor = torch.tensor(
            state, dtype=torch.float32, device=self.device)

        cost_fn = self._build_cost_fn()

        # Warm-start: shift previous solution
        warm = None
        if self._warm_controls is not None:
            warm = torch.cat([
                self._warm_controls[1:],
                torch.zeros(1, 3, device=self.device)
            ], dim=0)

        best_controls, best_cost, best_traj = self.gradient_mpc.optimize(
            state_tensor, cost_fn, warm_start=warm)

        self._warm_controls = best_controls.detach()
        first_control = best_controls[0].cpu().numpy()
        return first_control, best_cost

    def _default_dynamics_fn(self, state, control):
        """Evaluate Docking6D CW equations."""
        s = torch.tensor(state, dtype=torch.float32,
                         device=self.device).unsqueeze(0)
        u = torch.tensor(control, dtype=torch.float32,
                         device=self.device).unsqueeze(0)
        return self.dynamics.dsdt(s, u, None).squeeze().cpu().numpy()

    def _check_docked(self, state):
        """Check if all docking tolerances are met."""
        d = self.dynamics
        px, py, vx, vy, theta, omega = state
        x_ok = np.abs(px) <= d.eps_p
        y_ok = d.goal_y_min <= py <= d.goal_y_max
        pos_ok = x_ok and y_ok
        vel_ok = np.sqrt(vx**2 + vy**2) <= d.eps_v
        theta_goal = d.goal_state[4].item()
        omega_goal = d.goal_state[5].item()
        theta_diff = np.abs(
            np.arctan2(np.sin(theta - theta_goal),
                       np.cos(theta - theta_goal)))
        theta_ok = theta_diff <= d.eps_theta
        omega_ok = np.abs(omega - omega_goal) <= d.eps_omega
        return pos_ok and vel_ok and theta_ok and omega_ok

    def _check_collision(self, state):
        """Orientation-aware collision check (actual chaser corners)."""
        return self.dynamics.check_collision_oriented(state)


# ------------------------------------------------------------------
# Module-level helpers for 6D batch simulation
# ------------------------------------------------------------------

def _batch_wrap_state_6d(states, sr):
    """Wrap theta and clamp velocities for (B, 6) tensor.

    Args:
        states: (B, 6) tensor on device.
        sr: state_test_range() list — sr[i] = (lo, hi).
    Returns:
        (B, 6) wrapped tensor.
    """
    return torch.stack([
        states[:, 0],
        states[:, 1],
        torch.clamp(states[:, 2], sr[2][0], sr[2][1]),
        torch.clamp(states[:, 3], sr[3][0], sr[3][1]),
        torch.atan2(torch.sin(states[:, 4]), torch.cos(states[:, 4])),
        torch.clamp(states[:, 5], sr[5][0], sr[5][1]),
    ], dim=-1)


def _batch_check_collision_6d(states, dynamics):
    """Vectorised 4-corner oriented box collision check for 6D.

    Args:
        states: (B, 6) tensor on device.
        dynamics: Docking6D instance.
    Returns:
        (B,) bool tensor — True where any chaser corner is inside obstacle.
    """
    px = states[:, 0]
    py = states[:, 1]
    theta = states[:, 4]
    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)
    hw = dynamics.w_c / 2.0
    hh = dynamics.h_c / 2.0

    # 4 corners in world frame: (B, 4)
    cx = torch.stack([
        px + hw * cos_t - hh * sin_t,
        px - hw * cos_t - hh * sin_t,
        px - hw * cos_t + hh * sin_t,
        px + hw * cos_t + hh * sin_t,
    ], dim=-1)
    cy = torch.stack([
        py + hw * sin_t + hh * cos_t,
        py - hw * sin_t + hh * cos_t,
        py - hw * sin_t - hh * cos_t,
        py + hw * sin_t - hh * cos_t,
    ], dim=-1)

    # Target body (y in [0, h_t])
    half_w = dynamics.w_t / 2.0
    in_body = ((torch.abs(cx) <= half_w)
               & (cy >= 0) & (cy <= dynamics.h_t))

    # Docking post (y in [-post_length, 0])
    in_post = ((torch.abs(cx) <= dynamics.post_hw_x)
               & (cy >= -dynamics.post_length) & (cy <= 0))

    return (in_body | in_post).any(dim=-1)  # (B,)

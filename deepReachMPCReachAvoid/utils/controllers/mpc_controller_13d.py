"""
MPC-Only Controller for Docking13D

Direct-style MPC controller using random-shooting optimisation with the
analytical reachability cost function.  No learned value function is used —
this serves as a pure-MPC baseline for comparison.

At each simulation step the controller:
  1. Rolls out ``num_samples`` perturbed control sequences over the full
     planning horizon via ``rollout_dynamics``.
  2. Evaluates the analytical reachability cost for each trajectory.
  3. Selects the best sample and updates the control plan.
  4. Applies the first control and warm-starts the next step.

Usage:
    controller = MPCController13D(
        checkpoint_path='./runs/Docking13D_RA/training/checkpoints/model_final.pth',
        planning_horizon_sec=20.0,
        mpc_dt=0.5,
    )
    result = controller.simulate_docking(initial_state, max_sim_time=60.0)
"""

import time
import torch
import numpy as np
import pickle
import os
import sys
import inspect

# Add project root to path
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_THIS_DIR, '..', '..'))

from utils.gradient_mpc import GradientMPC
from dynamics import dynamics as dynamics_module
from utils.controllers.docking13d_mixin import Docking13DControllerMixin
from utils.controllers.safety_filter import SafetyFilter


class MPCController13D(Docking13DControllerMixin):
    """Gradient-based MPC controller for 13D docking (no learned VF)."""

    def __init__(self, checkpoint_path, planning_horizon_sec=20.0,
                 mpc_dt=0.5, dt=0.1, device='cuda',
                 gradient_lr=1.0, gradient_iters=80, num_restarts=16,
                 goal_weight=0.01, safety_filter=None):
        """
        Args:
            checkpoint_path:      Path to trained checkpoint (used only to
                                  load dynamics config from orig_opt.pickle).
            planning_horizon_sec: MPC planning horizon (seconds).
            mpc_dt:               MPC planning timestep (seconds).
            dt:                   Simulation integration timestep (seconds).
            device:               Torch device.
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

        self.experiment_dir = os.path.dirname(
            os.path.dirname(os.path.dirname(checkpoint_path)))

        self._load_dynamics()

        self.gradient_mpc = GradientMPC(
            dt=self.mpc_dt,
            horizon=self.planning_horizon_steps,
            dynamics=self.dynamics,
            device=self.device,
            num_iters=gradient_iters,
            lr=gradient_lr,
            num_restarts=num_restarts,
        )

        self.safety_filter = safety_filter or SafetyFilter(mode=0)

        self.reset()

        print(f"MPCController13D initialised | horizon={self.planning_horizon_sec}s "
              f"mpc_dt={self.mpc_dt}s  sim_dt={self.dt}s  "
              f"horizon_steps={self.planning_horizon_steps}  "
              f"iters={gradient_iters}  restarts={num_restarts}  "
              f"lr={gradient_lr}  goal_weight={goal_weight}")

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _load_dynamics(self):
        """Load dynamics class from saved experiment options."""
        opt_path = os.path.join(self.experiment_dir, 'orig_opt.pickle')
        with open(opt_path, 'rb') as f:
            self.orig_opt = pickle.load(f)

        dynamics_class = getattr(dynamics_module, self.orig_opt.dynamics_class)
        sig = inspect.signature(dynamics_class)
        kwargs = {}
        for param_name in sig.parameters:
            if param_name != 'self' and hasattr(self.orig_opt, param_name):
                kwargs[param_name] = getattr(self.orig_opt, param_name)

        self.dynamics = dynamics_class(**kwargs)
        self.dynamics.set_model(self.orig_opt.deepReach_model)

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
    # Simulation
    # ------------------------------------------------------------------

    def simulate_docking(self, initial_state, max_sim_time, dynamics_fn=None):
        """Run a full docking simulation using direct-style MPC.

        Returns:
            dict with the shared result format.
        """
        self.reset()
        t_wall_start = time.perf_counter()

        state = np.array(initial_state, dtype=np.float64)
        num_steps = int(max_sim_time / self.dt) + 1

        if dynamics_fn is None:
            dynamics_fn = self._default_dynamics_fn_13d

        print(f"  [MPC13D] Starting  dist={np.linalg.norm(state[:3]):.3f}m")

        docked = False
        collided = False
        dock_time = None
        post_dock_duration = 1.0  # seconds to continue after docking

        # Per-component reach_fn tracking
        reach_fn_comp_history = {
            'position': [], 'vlat': [], 'vax': [],
            'attitude': [], 'omega_py': [], 'omega_roll': [],
        }

        for step in range(num_steps):
            sim_time = step * self.dt

            # Stop after post-dock coast period
            if docked and (sim_time - dock_time) >= post_dock_duration:
                break

            # Keep MPC active even after docking so chaser can
            # converge closer to the goal point.
            control, cost_val = self._mpc_step(state)

            # Post-process through safety filter (no-op when mode=0)
            control = self.safety_filter.apply(state, control)

            # Record
            self.state_history.append(state.copy())
            self.control_history.append(control.copy())
            self.value_history.append(cost_val)
            self.sim_time_history.append(sim_time)

            # Record per-component reach_fn values
            with torch.no_grad():
                s_t = torch.tensor(state, dtype=torch.float32,
                                   device=self.device)
                comps = self.dynamics.reach_fn_components(s_t)
                for k in reach_fn_comp_history:
                    reach_fn_comp_history[k].append(float(comps[k]))

            # Termination (only before docking)
            if not docked and self._check_docked_13d(state):
                docked = True
                dock_time = sim_time
                print(f"  [MPC13D] Docking at t={sim_time:.2f}s")

            if not docked and self._check_collision_13d(state):
                collided = True
                print(f"  [MPC13D] Collision at t={sim_time:.2f}s")
                break

            # Euler integration
            state_dot = dynamics_fn(state, control)
            state = state + self.dt * state_dot

            # Quaternion normalization
            state = self._wrap_state_13d(state)

        wall_time = time.perf_counter() - t_wall_start

        controls_arr = np.array(self.control_history)
        control_effort = float(
            np.sum(np.linalg.norm(controls_arr, axis=-1)) * self.dt)

        return {
            'trajectory': np.array(self.state_history),
            'controls': controls_arr,
            'values': np.array(self.value_history),
            'times': np.array(self.sim_time_history),
            'success': docked and not collided,
            'collision': collided,
            'docked': docked,
            'final_state': state,
            'controller_type': 'mpc_13d',
            'control_effort': control_effort,
            'wall_time': wall_time,
            'safety_filter_mode': self.safety_filter.mode,
            'safety_filter_log': self.safety_filter.get_log(),
            # --- per-component reach_fn breakdown ---
            'reach_fn_components': {
                k: np.array(v) for k, v in reach_fn_comp_history.items()
            },
        }

    # ------------------------------------------------------------------
    # Batched simulation
    # ------------------------------------------------------------------

    def simulate_docking_batch(self, initial_states_np, max_sim_time):
        """Run docking simulations for multiple ICs in parallel on GPU.

        Uses batch_optimize to process all ICs simultaneously each step,
        giving a large speedup over sequential simulate_docking calls.

        Args:
            initial_states_np: (B, 13) numpy array of initial conditions.
            max_sim_time: Maximum simulation time (seconds).

        Returns:
            list of B result dicts (same format as simulate_docking).
        """
        B = len(initial_states_np)
        num_steps = int(max_sim_time / self.dt) + 1
        cdim = self.dynamics.control_dim
        H = self.planning_horizon_steps

        t_wall_start = time.perf_counter()

        states = torch.tensor(
            initial_states_np, dtype=torch.float32, device=self.device)

        # Pre-allocate storage (timestep-major)
        state_buf = torch.zeros(num_steps, B, 13)
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

        # Per-IC wall time: recorded when each IC terminates
        ic_wall_time = np.zeros(B, dtype=np.float64)
        ic_terminated = np.zeros(B, dtype=bool)

        # Warm-start controls (B, H, cdim)
        warm = torch.zeros(B, H, cdim, device=self.device)

        cost_fn = self._build_cost_fn()

        for step in range(num_steps):
            # Coast check: stop ICs that docked long enough ago
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

            # --- Termination checks (vectorised, matching sequential) ---
            with torch.no_grad():
                # Docking: reach_fn <= 0 (equivalent to _check_docked_13d)
                reach_vals = self.dynamics.reach_fn(states)
                newly_docked = active & ~docked & (reach_vals <= 0)
                docked = docked | newly_docked
                dock_step = torch.where(
                    newly_docked,
                    torch.tensor(step, device=self.device),
                    dock_step)

                # Collision: oriented 8-corner box (matches sequential)
                coll_mask = self._batch_check_collision_oriented(states)
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
                # Quaternion-only normalisation (matches sequential _wrap_state_13d)
                states = self._batch_wrap_quat(states)

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

            # Build safety filter log (per-step dicts matching sequential format)
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
                'controller_type': 'mpc_13d',
                'control_effort': effort,
                'wall_time': float(ic_wall_time[i]),
                'safety_filter_mode': self.safety_filter.mode,
                'safety_filter_log': sf_log_i,
            })

        mean_ic_wall = float(np.mean(ic_wall_time))
        print(f'  [Batch MPC] Done: {B} ICs in {wall_time:.1f}s '
              f'(mean {mean_ic_wall:.2f}s/IC)  '
              f'dock={int(docked_np.sum())} coll={int(collided_np.sum())}')

        return results

    # ------------------------------------------------------------------
    # Core MPC logic
    # ------------------------------------------------------------------

    def _build_cost_fn(self):
        """Build cost function for gradient MPC.

        Combines analytical reach-avoid cost with quadratic goal-directed
        regularisation for useful gradients far from the goal.
        """
        dynamics = self.dynamics
        goal_weight = self.goal_weight
        device = self.device

        # Goal state for 13D: [pos(3), vel(3), quat(4), omega(3)]
        q_goal = dynamics.q_goal.to(device).float()

        def cost_fn(trajectory, controls):
            # 1. Analytical reach-avoid cost
            reach_avoid = dynamics.cost_fn(trajectory)  # (K,)

            # 2. Goal-directed regularisation
            if goal_weight > 0:
                terminal = trajectory[:, -1, :]  # (K, 13)

                # Position error (goal at origin / goal_y_center)
                goal_pos = torch.zeros(3, device=device)
                if hasattr(dynamics, 'goal_y_center'):
                    goal_pos[1] = dynamics.goal_y_center
                pos_err = terminal[:, :3] - goal_pos
                vel_err = terminal[:, 3:6]

                # Quaternion error: 1 - |q . q_goal|^2
                q = terminal[:, 6:10]
                dot = torch.sum(q * q_goal, dim=-1)
                quat_err = 1.0 - dot ** 2

                omega_err = terminal[:, 10:13]

                goal_cost = (3.0 * (pos_err ** 2).sum(-1)
                             + 10.0 * (vel_err ** 2).sum(-1)
                             + 5.0 * quat_err
                             + 5.0 * (omega_err ** 2).sum(-1))

                reach_avoid = reach_avoid + goal_weight * goal_cost

            return reach_avoid

        return cost_fn

    def _mpc_step(self, state):
        """Run one gradient-based MPC step and return (first_control, best_cost)."""
        state_tensor = torch.tensor(
            state, dtype=torch.float32, device=self.device)

        cost_fn = self._build_cost_fn()

        # Warm-start: shift previous solution
        warm = None
        if self._warm_controls is not None:
            cdim = self.dynamics.control_dim
            warm = torch.cat([
                self._warm_controls[1:],
                torch.zeros(1, cdim, device=self.device)
            ], dim=0)

        best_controls, best_cost, best_traj = self.gradient_mpc.optimize(
            state_tensor, cost_fn, warm_start=warm)

        self._warm_controls = best_controls.detach()
        first_control = best_controls[0].cpu().numpy()
        return first_control, best_cost

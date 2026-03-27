"""
MPC + Learned Terminal Cost Controller for Docking13D

Receding-horizon MPC with a short effective planning horizon, augmented by
the learned DeepReach value function as terminal cost.

Two-phase terminal cost strategy:
  Phase 1 (Approach): V(x, tMax) — static terminal cost while outside BRT.
  Phase 2 (Tracking): V(x, t_remaining) — time-varying cost once inside BRT.
                       One-way transition: timer never resets.

Graduated stagnation-escape:
  NORMAL -> EXPLORING (larger eps_var) -> BRT_FALLBACK (pure gradient control).
  If progress resumes, reverts to NORMAL.

Usage:
    controller = MPCTerminalController13D(
        checkpoint_path='./runs/Docking13D_RA/training/checkpoints/model_final.pth',
        effective_horizon_sec=2.0,
        tMax=14.0,
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

from utils import modules
from utils import diff_operators  # noqa: F401
from utils.gradient_mpc import GradientMPC, DifferentiableValueFunction13D
from dynamics import dynamics as dynamics_module
from utils.controllers.docking13d_mixin import Docking13DControllerMixin
from utils.controllers.safety_filter import SafetyFilter


class MPCTerminalController13D(Docking13DControllerMixin):
    """Short-horizon gradient MPC + learned terminal cost for 13D docking."""

    def __init__(self, checkpoint_path, effective_horizon_sec=2.0, tMax=14.0,
                 dt=0.1, device='cuda',
                 effort_weight=0.0,
                 exploration_factor=3.0, exploration_patience=2,
                 escape_thresh=0.5,
                 gradient_lr=1.0, gradient_iters=50, num_restarts=8,
                 goal_weight=0.01, safety_filter=None,
                 safety_margin_phase1=0.1, safety_margin_phase2=0.02):
        """
        Args:
            checkpoint_path:       Path to trained model checkpoint.
            effective_horizon_sec: Short MPC planning horizon (seconds).
            tMax:                  Time at which V(x, tMax) is queried.
            dt:                    Control / integration timestep (seconds).
            device:                Torch device.
            effort_weight:         Weight for control effort penalty (0 = off).
            exploration_factor:    eps_var multiplier in EXPLORING mode.
            exploration_patience:  Stagnation windows before BRT fallback.
            escape_thresh:         Distance improvement (m) to escape local min.
            gradient_lr:           Adam learning rate (default 1.0).
            gradient_iters:        Adam iterations per MPC step (default 50).
            num_restarts:          Parallel random restarts (default 8).
            goal_weight:           Weight for goal-directed regularisation (default 0.01).
        """
        self.checkpoint_path = checkpoint_path
        self.effective_horizon_sec = effective_horizon_sec
        self.effective_horizon = int(effective_horizon_sec / dt)
        self.tMax = tMax
        self.dt = dt
        self.device = device
        self.effort_weight = effort_weight
        self.exploration_factor_setting = exploration_factor
        self.exploration_patience = exploration_patience
        self.escape_thresh = escape_thresh
        self._goal_weight = goal_weight
        self._near_obstacle = False

        self.experiment_dir = os.path.dirname(
            os.path.dirname(os.path.dirname(checkpoint_path)))

        self._load_dynamics_and_model()

        self.gradient_mpc = GradientMPC(
            dt=self.dt,
            horizon=self.effective_horizon,
            dynamics=self.dynamics,
            device=self.device,
            num_iters=gradient_iters,
            lr=gradient_lr,
            num_restarts=num_restarts,
        )

        # Differentiable value function (bypasses SingleBVPNet's .detach())
        self.diff_value_fn = DifferentiableValueFunction13D(
            self.model, self.dynamics, self.device)

        # Safety filter (no-op when mode=0 or None)
        self.safety_filter = safety_filter or SafetyFilter(mode=0)
        self.safety_margin_phase1 = safety_margin_phase1
        self.safety_margin_phase2 = safety_margin_phase2

        self.reset()

        effort_str = (f"  effort_weight={self.effort_weight}"
                      if self.effort_weight > 0 else "")
        print(f"MPCTerminalController13D initialised | "
              f"horizon={self.effective_horizon_sec}s  tMax={self.tMax}s  "
              f"iters={gradient_iters}  restarts={num_restarts}  "
              f"lr={gradient_lr}  goal_weight={goal_weight}"
              f"{effort_str}")

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _load_dynamics_and_model(self):
        """Load dynamics and trained SIREN model."""
        opt_path = os.path.join(self.experiment_dir, 'orig_opt.pickle')
        with open(opt_path, 'rb') as f:
            self.orig_opt = pickle.load(f)

        # Dynamics
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

        # Model
        self.model = modules.SingleBVPNet(
            in_features=self.dynamics.input_dim,
            out_features=1,
            type=self.orig_opt.model,
            mode=self.orig_opt.model_mode,
            final_layer_factor=1.,
            hidden_features=self.orig_opt.num_nl,
            num_hidden_layers=self.orig_opt.num_hl,
            periodic_transform_fn=self.dynamics.periodic_transform_fn,
        )
        ckpt = torch.load(
            self.checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt['model'])
        self.model.to(self.device)
        self.model.eval()

    def reset(self):
        """Reset controller state for a new simulation."""
        self.state_history = []
        self.control_history = []
        self.value_history = []
        self.sim_time_history = []
        self.phase_history = []
        self.t_remaining_history = []
        self._warm_controls = None
        self.safety_filter.reset()

        # Phase tracking (one-way)
        self.in_brt = False
        self.t_remaining = self.tMax
        self.brt_entry_time = None

        # Diagnostics
        self._last_reach_avoid = 0.0
        self._last_terminal = 0.0
        self._last_t_query = self.tMax

        # Graduated stagnation-escape
        self._stagnation_count = 0
        self._control_mode = 'normal'
        self._exploration_factor = 1.0
        self._mode_entry_dist = None

    # ------------------------------------------------------------------
    # Value-function queries
    # ------------------------------------------------------------------

    def _is_in_brt(self, state_np):
        """True if V(state, tMax) <= 0."""
        s = torch.tensor(
            state_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        time_col = torch.full(
            (1, 1), self.tMax, dtype=torch.float32, device=self.device)
        coords = torch.cat([time_col, s], dim=-1)
        model_input = self.dynamics.coord_to_input(coords)
        with torch.no_grad():
            result = self.model({'coords': model_input})
            output = result['model_out'].squeeze()
        value = self.dynamics.io_to_value(model_input, output)
        return value.item() <= 0

    def get_value(self, state_np, time_query):
        """Query V(state, time_query) for a single state."""
        s = torch.tensor(
            state_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        time_col = torch.full(
            (1, 1), time_query, dtype=torch.float32, device=self.device)
        coords = torch.cat([time_col, s], dim=-1)
        model_input = self.dynamics.coord_to_input(coords)
        with torch.no_grad():
            result = self.model({'coords': model_input})
            output = result['model_out'].squeeze()
        return self.dynamics.io_to_value(model_input, output).item()

    # ------------------------------------------------------------------
    # Terminal cost evaluation
    # ------------------------------------------------------------------

    def _evaluate_terminal_values(self, terminal_states, t_query=None):
        """Batch-evaluate V(x, t_query) for terminal states.

        Args:
            terminal_states: (A, N, state_dim) tensor.
            t_query:         Scalar time (defaults to tMax).

        Returns:
            (A, N) tensor of terminal values.
        """
        if t_query is None:
            t_query = self.tMax

        A, N, D = terminal_states.shape
        flat = terminal_states.reshape(A * N, D)

        test_range = torch.tensor(
            self.dynamics.state_test_range(), device=self.device)
        flat_clamped = torch.clamp(flat, test_range[..., 0], test_range[..., 1])

        time_col = torch.full(
            (A * N, 1), t_query, dtype=torch.float32, device=self.device)
        coords = torch.cat([time_col, flat_clamped], dim=-1)
        model_input = self.dynamics.coord_to_input(coords)

        with torch.no_grad():
            result = self.model({'coords': model_input})
            output = result['model_out'].squeeze(-1)

        values = self.dynamics.io_to_value(model_input, output)
        return values.reshape(A, N)

    # ------------------------------------------------------------------
    # Phase tracking
    # ------------------------------------------------------------------

    def _update_phase(self, state, sim_time):
        """One-way phase transition: Phase 1 -> Phase 2 when entering BRT."""
        if not self.in_brt:
            if self._is_in_brt(state):
                self.in_brt = True
                self.t_remaining = self.tMax
                self.brt_entry_time = sim_time
                print(f"  [MPC+T] Entered BRT at t={sim_time:.2f}s -> Phase 2")

        if self.in_brt:
            t_query = max(self.t_remaining, 0.01)
            phase = 2
        else:
            t_query = self.tMax
            phase = 1

        self.phase_history.append(phase)
        self.t_remaining_history.append(
            self.t_remaining if self.in_brt else self.tMax)
        self._last_t_query = t_query

        return phase, t_query

    # ------------------------------------------------------------------
    # BRT optimal-control fallback (13D-specific)
    # ------------------------------------------------------------------

    def _compute_brt_control(self, state):
        """Bang-bang optimal control from value-function gradient."""
        s = torch.tensor(state, dtype=torch.float32,
                         device=self.device).unsqueeze(0)
        time_col = torch.full(
            (1, 1), self._last_t_query, dtype=torch.float32,
            device=self.device)
        coord = torch.cat([time_col, s], dim=-1)
        model_input = self.dynamics.coord_to_input(coord)

        result = self.model({'coords': model_input})
        output = result['model_out'].squeeze()
        model_in = result['model_in']

        dv = self.dynamics.io_to_dv(model_in, output)
        dvds = dv[0, 1:].detach().cpu().numpy()

        return self._compute_brt_control_13d(dvds, state)

    # ------------------------------------------------------------------
    # Batched simulation
    # ------------------------------------------------------------------

    def _batch_is_in_brt(self, states):
        """Vectorised BRT membership check. Returns (B,) bool tensor."""
        B = states.shape[0]
        time_col = torch.full(
            (B, 1), self.tMax, dtype=torch.float32, device=self.device)
        coords = torch.cat([time_col, states], dim=-1)
        model_input = self.dynamics.coord_to_input(coords)
        with torch.no_grad():
            result = self.model({'coords': model_input})
            output = result['model_out'].squeeze(-1)
        values = self.dynamics.io_to_value(model_input, output)
        return values <= 0

    def _batch_compute_brt_control_batch(self, states, t_queries):
        """Batch bang-bang control from value function gradient for 13D fallback ICs.

        Uses reach-avoid sign convention (MINIMISE V), matching
        ``_compute_brt_control_13d`` from the mixin.

        Args:
            states:    (N, 13) states of fallback ICs on device.
            t_queries: (N,) per-IC time queries on device.

        Returns:
            (N, 6) bang-bang controls on device.
        """
        N = states.shape[0]
        time_col = t_queries.unsqueeze(-1)  # (N, 1)
        coords = torch.cat([time_col, states.float()], dim=-1)
        model_input = self.dynamics.coord_to_input(coords)

        result = self.model({'coords': model_input})
        output = result['model_out'].squeeze(-1)
        model_in = result['model_in']

        dv = self.dynamics.io_to_dv(model_in, output)  # (N, 14)
        dvds = dv[:, 1:].detach()                       # (N, 13)

        # Rotation-aware force allocation (reach-avoid convention: MINIMISE V)
        q = states[:, 6:10]
        q = q / (q.norm(dim=-1, keepdim=True) + 1e-12)
        R = self.dynamics.quat_to_R_LVLH_to_body(q)  # (N, 3, 3) LVLH→body

        # Force: body coefficients = R @ p_v / mc
        p_v = dvds[:, 3:6].unsqueeze(-1)              # (N, 3, 1)
        mc = self.dynamics.mc
        coeff_body = (torch.bmm(R, p_v).squeeze(-1)) / mc  # (N, 3)
        F_bar = self.dynamics.F_bar
        # Reach-avoid convention: OPPOSITE of safety filter
        F = torch.where(coeff_body > 0, -F_bar, F_bar)  # (N, 3)

        # Torque: coeff_tau = I^{-T} @ p_omega
        I_np = self.dynamics.I.detach().cpu().numpy()
        I_inv_T = torch.tensor(
            np.linalg.inv(I_np).T, dtype=torch.float32,
            device=states.device)
        p_omega = dvds[:, 10:13]                        # (N, 3)
        coeff_tau = p_omega @ I_inv_T.T                  # (N, 3)
        tau_bar = self.dynamics.tau_bar
        tau = torch.where(coeff_tau > 0, -tau_bar, tau_bar)  # (N, 3)

        return torch.cat([F, tau], dim=-1)               # (N, 6)

    def _build_batch_cost_fn(self, t_queries, goal_weights=None,
                             near_obstacle_mask=None):
        """Build cost function with per-IC time queries for batch_optimize.

        Args:
            t_queries:          (B,) tensor of per-IC time queries.
            goal_weights:       (B,) tensor of per-IC goal weights, or None
                                to use the scalar ``self._goal_weight``.
            near_obstacle_mask: (B,) bool tensor, or None (no suppression).
        """
        diff_value_fn = self.diff_value_fn
        dynamics = self.dynamics
        effort_weight = self.effort_weight
        dt = self.dt
        device = self.device

        # Determine whether to use per-IC or scalar goal weight
        if goal_weights is not None:
            per_ic_goal = True
            _goal_weights = goal_weights
            _near_mask = near_obstacle_mask
        else:
            per_ic_goal = False
            _scalar_goal_weight = self._goal_weight

        def cost_fn(trajectory, controls):
            # trajectory: (B, H+1, 13), controls: (B, H, 6)

            # 1. Short-horizon reach-avoid cost
            reach_avoid = dynamics.cost_fn(trajectory)  # (B,)

            # 2. Differentiable terminal cost with per-IC time
            terminal_states = trajectory[:, -1, :]  # (B, 13)
            terminal_values = diff_value_fn(terminal_states, t_queries)

            # 3. Combine
            combined = torch.minimum(reach_avoid, terminal_values)
            avoid_max = torch.max(
                -dynamics.avoid_fn(trajectory), dim=-1).values
            combined = torch.maximum(combined, avoid_max)

            # 4. Control effort penalty
            if effort_weight > 0:
                control_norms = torch.norm(controls, dim=-1)
                effort = torch.sum(control_norms, dim=-1) * dt
                on_track = (combined <= 0).float()
                combined = combined + effort_weight * effort * on_track

            # 5. Goal-directed regularisation
            if per_ic_goal:
                active_goal = _goal_weights * (
                    ~_near_mask).float() if _near_mask is not None else _goal_weights
                if active_goal.sum() > 0:
                    goal_cost = _goal_directed_cost_13d(
                        trajectory, dynamics, device)
                    combined = combined + active_goal * goal_cost
            elif _scalar_goal_weight > 0:
                combined = combined + _scalar_goal_weight * _goal_directed_cost_13d(
                    trajectory, dynamics, device)

            return combined

        return cost_fn

    def simulate_docking_batch(self, initial_states_np, max_sim_time):
        """Run docking simulations for multiple ICs in parallel on GPU.

        Uses batch_optimize to process all ICs simultaneously each step.
        Phase tracking (BRT entry) is handled per-IC via masking.
        Per-IC stagnation detection with graduated escalation
        (NORMAL -> EXPLORING -> BRT_FALLBACK).

        Args:
            initial_states_np: (B, 13) numpy array of initial conditions.
            max_sim_time: Maximum simulation time (seconds).

        Returns:
            list of B result dicts (same format as simulate_docking).
        """
        B = len(initial_states_np)
        num_steps = int(max_sim_time / self.dt) + 1
        cdim = self.dynamics.control_dim
        H = self.effective_horizon

        t_wall_start = time.perf_counter()

        states = torch.tensor(
            initial_states_np, dtype=torch.float32, device=self.device)

        # Pre-allocate storage
        state_buf = torch.zeros(num_steps, B, 13)
        ctrl_buf = torch.zeros(num_steps, B, cdim)
        cost_buf = torch.zeros(num_steps, B)
        phase_buf = torch.ones(num_steps, B, dtype=torch.long)
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

        # Phase tracking (per-IC)
        in_brt = torch.zeros(B, dtype=torch.bool, device=self.device)
        t_remaining = torch.full((B,), self.tMax, dtype=torch.float32,
                                 device=self.device)
        brt_entry_step = torch.full((B,), -1, dtype=torch.long,
                                    device=self.device)

        # Per-IC stagnation detection state
        MODE_NORMAL, MODE_EXPLORING, MODE_FALLBACK = 0, 1, 2
        control_mode = np.zeros(B, dtype=np.int32)
        stagnation_count = np.zeros(B, dtype=np.int32)
        goal_weights_np = np.full(B, self._goal_weight, dtype=np.float64)
        mode_entry_dist = np.full(B, np.inf, dtype=np.float64)
        prev_log_dist = np.full(B, np.inf, dtype=np.float64)
        first_log = np.ones(B, dtype=bool)

        log_interval = 50
        stagnation_thresh = 0.1

        # Warm-start controls (B, H, cdim)
        warm = torch.zeros(B, H, cdim, device=self.device)

        for step in range(num_steps):
            # Coast check
            coast_done = docked & ((step - dock_step) >= post_dock_steps)
            active = active & ~coast_done & ~collided

            if not active.any():
                break

            # Record state
            state_buf[step] = states.detach().cpu()
            final_step[active] = step

            # --- Phase update (one-way transition) ---
            with torch.no_grad():
                newly_in_brt = active & ~in_brt & self._batch_is_in_brt(states)
                in_brt = in_brt | newly_in_brt
                brt_entry_step = torch.where(
                    newly_in_brt,
                    torch.tensor(step, device=self.device),
                    brt_entry_step)

            # Build per-IC t_query
            t_queries = torch.where(
                in_brt,
                torch.clamp(t_remaining, min=0.01),
                torch.full_like(t_remaining, self.tMax))

            phase_buf[step] = torch.where(in_brt, 2, 1).cpu()

            # --- Per-IC stagnation check (every log_interval steps) ---
            if step % log_interval == 0:
                with torch.no_grad():
                    dists = torch.norm(
                        states[:, :3], dim=-1).cpu().numpy()

                active_np = active.cpu().numpy()
                for i in range(B):
                    if not active_np[i]:
                        continue
                    if first_log[i]:
                        first_log[i] = False
                        prev_log_dist[i] = dists[i]
                        continue

                    d_dist = prev_log_dist[i] - dists[i]

                    if d_dist >= stagnation_thresh:
                        # Making progress — check if escaped
                        if (control_mode[i] != MODE_NORMAL
                                and mode_entry_dist[i] != np.inf):
                            if (mode_entry_dist[i] - dists[i]
                                    >= self.escape_thresh):
                                control_mode[i] = MODE_NORMAL
                                goal_weights_np[i] = self._goal_weight
                                stagnation_count[i] = 0
                    else:
                        # Stagnating
                        stagnation_count[i] += 1
                        if control_mode[i] == MODE_NORMAL:
                            control_mode[i] = MODE_EXPLORING
                            goal_weights_np[i] = 0.1
                            mode_entry_dist[i] = dists[i]
                        elif control_mode[i] == MODE_EXPLORING:
                            goal_weights_np[i] = min(
                                goal_weights_np[i]
                                * self.exploration_factor_setting, 1.0)
                            if (stagnation_count[i]
                                    >= self.exploration_patience):
                                control_mode[i] = MODE_FALLBACK

                    prev_log_dist[i] = dists[i]

                # Print batch stagnation summary
                n_a = int(active_np.sum())
                n_exploring = int(
                    (control_mode[active_np] == MODE_EXPLORING).sum())
                n_fallback = int(
                    (control_mode[active_np] == MODE_FALLBACK).sum())
                n_d = int(docked.sum().item())
                n_c = int(collided.sum().item())
                n_p2 = int(in_brt.sum().item())
                print(f'  [Batch MPC+T] step={step} '
                      f't={step*self.dt:.1f}s  '
                      f'active={n_a} docked={n_d} coll={n_c} '
                      f'phase2={n_p2} '
                      f'exploring={n_exploring} '
                      f'fallback={n_fallback}')

            # --- Near-obstacle check (batch) ---
            with torch.no_grad():
                avoid_vals = self.dynamics.avoid_fn(states)
                near_obstacle_mask = (avoid_vals < 1.0)  # proximity margin

            # --- Build per-IC goal weights and cost function ---
            goal_weights_t = torch.tensor(
                goal_weights_np, dtype=torch.float32,
                device=self.device)

            cost_fn = self._build_batch_cost_fn(
                t_queries, goal_weights=goal_weights_t,
                near_obstacle_mask=near_obstacle_mask)

            # --- MPC optimisation (all ICs simultaneously) ---
            best_controls, best_costs, _ = self.gradient_mpc.batch_optimize(
                states, cost_fn, warm)

            controls = best_controls[:, 0, :]
            controls = controls * active.unsqueeze(-1).float()

            # --- Override controls for BRT_FALLBACK ICs ---
            fallback_mask = (torch.tensor(
                control_mode == MODE_FALLBACK,
                device=self.device) & active)
            if fallback_mask.any():
                fb_controls = self._batch_compute_brt_control_batch(
                    states[fallback_mask], t_queries[fallback_mask])
                controls[fallback_mask] = fb_controls

            # --- Safety filter (phase-aware per-IC margin, no-op when mode=0) ---
            sf_margins = torch.where(
                in_brt,
                torch.tensor(self.safety_margin_phase2, device=self.device),
                torch.tensor(self.safety_margin_phase1, device=self.device))
            controls, sf_active_step = self.safety_filter.batch_apply(
                states, controls, active_mask=active, margins=sf_margins)
            sf_active_buf[step] = sf_active_step.cpu()

            ctrl_buf[step] = controls.detach().cpu()
            cost_buf[step] = best_costs.detach().cpu()

            # Decrement timer for phase-2 ICs
            t_remaining = torch.where(
                in_brt, t_remaining - self.dt, t_remaining)

            # --- Termination checks ---
            with torch.no_grad():
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

        wall_time = time.perf_counter() - t_wall_start

        # Assign wall time for ICs that never terminated (timed out)
        ic_wall_time[~ic_terminated] = wall_time

        # --- Build per-IC result dicts ---
        docked_np = docked.cpu().numpy()
        collided_np = collided.cpu().numpy()
        final_step_np = final_step.cpu().numpy()
        brt_entry_np = brt_entry_step.cpu().numpy()

        sf_active_np = sf_active_buf.numpy()

        results = []
        for i in range(B):
            n = int(final_step_np[i]) + 1
            traj_i = state_buf[:n, i].numpy()
            ctrl_i = ctrl_buf[:n, i].numpy()
            cost_i = cost_buf[:n, i].numpy()
            phase_i = phase_buf[:n, i].numpy()
            times_i = np.arange(n) * self.dt
            sf_i = sf_active_np[:n, i]

            effort = float(
                np.sum(np.linalg.norm(ctrl_i, axis=-1)) * self.dt)

            brt_t = (float(brt_entry_np[i] * self.dt)
                     if brt_entry_np[i] >= 0 else None)

            sf_log_i = [{'filter_active': bool(sf_i[s])} for s in range(n)]

            results.append({
                'trajectory': traj_i,
                'controls': ctrl_i,
                'values': cost_i,
                'times': times_i,
                'phases': phase_i,
                'success': bool(docked_np[i] and not collided_np[i]),
                'collision': bool(collided_np[i]),
                'docked': bool(docked_np[i]),
                'final_state': traj_i[-1],
                'controller_type': 'mpc_terminal_13d',
                'control_effort': effort,
                'wall_time': float(ic_wall_time[i]),
                'brt_entry_time': brt_t,
                'safety_filter_mode': self.safety_filter.mode,
                'safety_filter_log': sf_log_i,
            })

        mean_ic_wall = float(np.mean(ic_wall_time))
        print(f'  [Batch MPC+T] Done: {B} ICs in {wall_time:.1f}s '
              f'(mean {mean_ic_wall:.2f}s/IC)  '
              f'dock={int(docked_np.sum())} coll={int(collided_np.sum())}')

        return results

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def simulate_docking(self, initial_state, max_sim_time, dynamics_fn=None):
        """Run full docking simulation with MPC + terminal cost.

        Returns:
            dict with the shared result format.
        """
        self.reset()
        t_wall_start = time.perf_counter()

        state = np.array(initial_state, dtype=np.float64)
        num_steps = int(max_sim_time / self.dt) + 1

        if dynamics_fn is None:
            dynamics_fn = self._default_dynamics_fn_13d

        print(f"[MPC+Terminal13D] Starting  dist={np.linalg.norm(state[:3]):.3f}m")

        docked = False
        collided = False
        dock_time = None
        post_dock_duration = 1.0  # seconds to continue after docking

        # Stagnation detection
        log_interval = 50
        stagnation_thresh = 0.1
        prev_log_dist = None
        prev_log_V = None

        for step in range(num_steps):
            sim_time = step * self.dt

            # Stop after post-dock coast period
            if docked and (sim_time - dock_time) >= post_dock_duration:
                break

            # --- Control selection (mode-aware) ---
            # Keep control active even after docking so chaser can
            # converge closer to the goal point.
            if self._control_mode == 'brt_fallback':
                phase, t_query = self._update_phase(state, sim_time)
                control = self._compute_brt_control(state)
                cost_val = self.get_value(state, self._last_t_query)
                if self.in_brt:
                    self.t_remaining -= self.dt
            else:
                control, cost_val = self._mpc_step(state, sim_time)

            # Post-process through safety filter (no-op when mode=0)
            self.safety_filter.set_margin(
                self.safety_margin_phase2 if self.in_brt else self.safety_margin_phase1)
            control = self.safety_filter.apply(state, control)

            # Record
            self.state_history.append(state.copy())
            self.control_history.append(control.copy())
            self.value_history.append(cost_val)
            self.sim_time_history.append(sim_time)


            # --- Periodic logging + graduated stagnation ---
            if step % log_interval == 0:
                dist = float(np.linalg.norm(state[:3]))
                V_now = self.get_value(state, self._last_t_query)
                phase = self.phase_history[-1]
                ra = self._last_reach_avoid
                tv = self._last_terminal
                dominates = 'terminal' if tv <= ra else 'reach_avoid'

                delta_str = ''
                stag_flag = ''
                if prev_log_dist is not None:
                    d_dist = prev_log_dist - dist
                    d_V = prev_log_V - V_now
                    delta_str = (f'  delta_dist={d_dist:+.4f}m  '
                                 f'delta_V={d_V:+.4f}')
                    if d_dist < stagnation_thresh:
                        stag_flag = '  ** STAGNATING **'

                    if d_dist >= stagnation_thresh:
                        if (self._control_mode != 'normal'
                                and self._mode_entry_dist is not None):
                            if self._mode_entry_dist - dist >= self.escape_thresh:
                                self._control_mode = 'normal'
                                self._exploration_factor = 1.0
                                self._stagnation_count = 0
                                self._warm_controls = None
                                print('    -> Escaped local min, mode=NORMAL')
                    else:
                        self._stagnation_count += 1
                        if self._control_mode == 'normal':
                            self._control_mode = 'exploring'
                            self._exploration_factor = self.exploration_factor_setting
                            self._mode_entry_dist = dist
                            print(f'    -> mode=EXPLORING '
                                  f'(eps x{self._exploration_factor})')
                        elif (self._control_mode == 'exploring'
                              and self._stagnation_count >= self.exploration_patience):
                            self._control_mode = 'brt_fallback'
                            print('    -> Exploration failed, mode=BRT_FALLBACK')

                mode_tag = self._control_mode.upper()
                print(
                    f'  [MPC+T] t={sim_time:5.1f}s  Ph{phase} {mode_tag:<12} '
                    f'dist={dist:.3f}m  V={V_now:.4f}  '
                    f'best={cost_val:.4f}  '
                    f'({dominates}){delta_str}{stag_flag}')

                prev_log_dist = dist
                prev_log_V = V_now

            # Termination (only before docking)
            if not docked and self._check_docked_13d(state):
                docked = True
                dock_time = sim_time
                print(f"  [MPC+T] Docking at t={sim_time:.2f}s")

            if not docked and self._check_collision_13d(state):
                collided = True
                print(f"  [MPC+T] Collision at t={sim_time:.2f}s")
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
            'phases': np.array(self.phase_history),
            't_remaining': np.array(self.t_remaining_history),
            'success': docked and not collided,
            'collision': collided,
            'docked': docked,
            'final_state': state,
            'controller_type': 'mpc_terminal_13d',
            'control_effort': control_effort,
            'wall_time': wall_time,
            'brt_entry_time': self.brt_entry_time,
            'safety_filter_mode': self.safety_filter.mode,
            'safety_filter_log': self.safety_filter.get_log(),
        }

    # ------------------------------------------------------------------
    # Cost function construction
    # ------------------------------------------------------------------

    def _build_cost_fn(self, phase, t_query):
        """Build the cost function for gradient MPC.

        Combines:
          1. Short-horizon reach-avoid cost (analytical)
          2. Terminal cost from learned value function (differentiable through SIREN)
          3. Control effort penalty (optional)
          4. Goal-directed regularisation (stagnation escape)
        """
        diff_value_fn = self.diff_value_fn
        dynamics = self.dynamics
        effort_weight = self.effort_weight
        dt = self.dt
        goal_weight = self._goal_weight
        device = self.device

        def cost_fn(trajectory, controls):
            # 1. Short-horizon reach-avoid cost
            reach_avoid = dynamics.cost_fn(trajectory)  # (K,)

            # 2. Terminal cost (differentiable through SIREN)
            terminal_states = trajectory[:, -1, :]  # (K, 13)
            terminal_values = diff_value_fn(terminal_states, t_query)  # (K,)

            # 3. Combine (reach-avoid formulation)
            combined = torch.minimum(reach_avoid, terminal_values)
            avoid_max = torch.max(
                -dynamics.avoid_fn(trajectory), dim=-1).values  # (K,)
            combined = torch.maximum(combined, avoid_max)

            # 4. Control effort penalty
            if effort_weight > 0:
                control_norms = torch.norm(controls, dim=-1)  # (K, H)
                effort = torch.sum(control_norms, dim=-1) * dt  # (K,)
                on_track = (combined <= 0).float()
                combined = combined + effort_weight * effort * on_track

            # 5. Goal-directed regularisation (stagnation escape)
            if goal_weight > 0:
                combined = combined + goal_weight * _goal_directed_cost_13d(
                    trajectory, dynamics, device)

            return combined

        return cost_fn

    # ------------------------------------------------------------------
    # Core MPC + terminal cost logic (gradient-based)
    # ------------------------------------------------------------------

    def _mpc_step(self, state, sim_time):
        """Run one gradient-based MPC step with phase-aware terminal cost.

        Phase 1 (Approach): terminal cost = V(x, tMax)
        Phase 2 (Tracking): terminal cost = V(x, t_remaining)

        Returns:
            (first_control, best_combined_cost)
        """
        phase, t_query = self._update_phase(state, sim_time)

        state_tensor = torch.tensor(
            state, dtype=torch.float32, device=self.device)

        cost_fn = self._build_cost_fn(phase, t_query)

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

        # Diagnostics (matching existing interface)
        with torch.no_grad():
            self._last_reach_avoid = self.dynamics.cost_fn(
                best_traj.unsqueeze(0)).item()
            self._last_terminal = self.diff_value_fn(
                best_traj[-1].unsqueeze(0), t_query).item()

        # Decrement timer (Phase 2 only)
        if self.in_brt:
            self.t_remaining -= self.dt

        first_control = best_controls[0].cpu().numpy()
        return first_control, best_cost


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------

def _goal_directed_cost_13d(trajectory, dynamics, device):
    """Quadratic goal-directed cost for 13D stagnation escape.

    Provides useful gradients everywhere, unlike the tanh-saturated reach_fn.
    """
    terminal = trajectory[:, -1, :]  # (K, 13)

    # Position error (goal at origin / goal_y_center)
    goal_pos = torch.zeros(3, device=device)
    if hasattr(dynamics, 'goal_y_center'):
        goal_pos[1] = dynamics.goal_y_center
    pos_err = terminal[:, :3] - goal_pos
    vel_err = terminal[:, 3:6]

    # Quaternion error: 1 - |q . q_goal|^2
    q = terminal[:, 6:10]
    q_goal = dynamics.q_goal.to(device).float()
    dot = torch.sum(q * q_goal, dim=-1)
    quat_err = 1.0 - dot ** 2

    omega_err = terminal[:, 10:13]

    return (3.0 * (pos_err ** 2).sum(-1)
            + 10.0 * (vel_err ** 2).sum(-1)
            + 5.0 * quat_err
            + 5.0 * (omega_err ** 2).sum(-1))

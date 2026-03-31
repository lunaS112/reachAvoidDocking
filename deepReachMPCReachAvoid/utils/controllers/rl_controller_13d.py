"""
RL-Based Controller for Docking13D

Loads a trained DDQNSingle Q-network and implements the standard
simulate_docking() interface for the 13D controller comparison framework.
Optionally applies a BRT least-restrictive safety filter.

Usage:
    controller = RLController13D(
        rl_checkpoint_path='../RLBaseline/experiments/.../model/Q-800000.pth',
        architecture=[256, 256],
    )
    result = controller.simulate_docking(initial_state, max_sim_time=30.0)
"""

import itertools
import os
import sys
import time as _time

import numpy as np
import torch
from tqdm import tqdm

# Project root for dynamics
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

from dynamics.dynamics import Docking13D
from utils.controllers.docking13d_mixin import Docking13DControllerMixin
from utils.controllers.safety_filter import SafetyFilter

# RL model
_RL_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', '..', 'RLBaseline'))
if _RL_ROOT not in sys.path:
    sys.path.insert(0, _RL_ROOT)

from RARL.model import Model


class RLController13D(Docking13DControllerMixin):
    """13D docking controller using a trained DDQN reach-avoid Q-network.

    Inherits docking/collision checks from Docking13DControllerMixin.
    Implements simulate_docking() for drop-in use with run_controller_13d.py.
    """

    def __init__(self, rl_checkpoint_path, dt=0.1, device='cuda',
                 safety_filter=None, architecture=None, activation='Tanh',
                 pd_attitude=False, pd_kp=1.0, pd_kd=0.5):
        """
        Args:
            rl_checkpoint_path: Path to Q-network .pth file.
            dt: Control timestep (must match training).
            device: Torch device.
            safety_filter: SafetyFilter instance or None.
            architecture: Hidden layer dims (must match training).
            activation: Activation function (must match training).
            pd_attitude: If True, Q-network only outputs force actions (27)
                         and torques are computed by a PD attitude controller.
            pd_kp: PD proportional gain for attitude control.
            pd_kd: PD derivative gain for attitude control.
        """
        if architecture is None:
            architecture = [256, 256]

        self.dt = dt
        self.device = device
        self.pd_attitude = pd_attitude
        self.safety_filter = safety_filter or SafetyFilter(mode=0)

        # Dynamics (read-only, required by Docking13DControllerMixin)
        self.dynamics = Docking13D(set_mode='reach_avoid')

        # Build discrete action set
        F_bar = float(self.dynamics.F_bar)
        tau_bar = float(self.dynamics.tau_bar)
        force_levels = [-F_bar, 0.0, F_bar]

        if pd_attitude:
            # Forces only: 3^3 = 27 actions; torques from PD controller
            self.discrete_controls = np.array(
                list(itertools.product(
                    force_levels, force_levels, force_levels)),
                dtype=np.float64)
            self._pd_kp = pd_kp
            self._pd_kd = pd_kd
            self._tau_bar = tau_bar
            self._q_goal_np = self.dynamics.q_goal.cpu().numpy()
        else:
            # Full 6D: 3^6 = 729 actions
            torque_levels = [-tau_bar, 0.0, tau_bar]
            self.discrete_controls = np.array(
                list(itertools.product(
                    force_levels, force_levels, force_levels,
                    torque_levels, torque_levels, torque_levels)),
                dtype=np.float64)

        # Build and load Q-network
        state_dim = 13
        action_num = len(self.discrete_controls)
        dim_list = [state_dim] + list(architecture) + [action_num]
        self.Q_network = Model(dim_list, actType=activation)
        self.Q_network.load_state_dict(
            torch.load(rl_checkpoint_path, map_location=device, weights_only=True))
        self.Q_network.to(device)
        self.Q_network.eval()

        self._last_q_value = 0.0

        print(f"[RLController13D] Loaded {rl_checkpoint_path}")
        print(f"  arch={dim_list}, actions={action_num}, dt={dt}, "
              f"pd_attitude={pd_attitude}")

    def reset(self):
        """Reset controller state for a new simulation."""
        self.safety_filter.reset()
        self._last_q_value = 0.0

    def u_fn(self, state, sim_time):
        """Compute control action from state using the Q-network.

        Includes PD attitude torques when ``self.pd_attitude`` is True.
        Caches the min Q-value in ``self._last_q_value`` for diagnostic
        logging without requiring a second forward pass.
        """
        s_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_vals = self.Q_network(s_t)
            self._last_q_value = q_vals.min(dim=1)[0].item()
            action_idx = q_vals.min(dim=1)[1].item()
        raw_control = self.discrete_controls[action_idx].copy()
        if self.pd_attitude:
            torques = self._pd_torque(state)
            return np.concatenate([raw_control, torques])
        return raw_control

    def _pd_torque(self, state):
        """PD attitude controller: drives quaternion toward q_goal, damps omega.

        Returns (3,) torque array clipped to [-tau_bar, tau_bar].
        """
        q = state[6:10].copy()
        q /= np.linalg.norm(q) + 1e-12
        omega = state[10:13]

        # Error quaternion: q_err = q_goal^{-1} * q  (scalar-first)
        qg = self._q_goal_np
        qg_conj = np.array([qg[0], -qg[1], -qg[2], -qg[3]])
        a0, a1, a2, a3 = qg_conj
        b0, b1, b2, b3 = q
        q_err = np.array([
            a0*b0 - a1*b1 - a2*b2 - a3*b3,
            a0*b1 + a1*b0 + a2*b3 - a3*b2,
            a0*b2 - a1*b3 + a2*b0 + a3*b1,
            a0*b3 + a1*b2 - a2*b1 + a3*b0,
        ])
        if q_err[0] < 0:
            q_err = -q_err

        err_vec = 2.0 * q_err[1:4]
        tau = -self._pd_kp * err_vec - self._pd_kd * omega
        return np.clip(tau, -self._tau_bar, self._tau_bar)

    def simulate_docking(self, initial_state, max_sim_time, dynamics_fn=None):
        """Run docking simulation using the RL policy.

        Args:
            initial_state: (13,) numpy array.
            max_sim_time: Maximum simulation time (seconds).
            dynamics_fn: Optional f(state, control) -> state_dot.

        Returns:
            dict with standardized comparison fields.
        """
        self.safety_filter.reset()
        t_wall_start = _time.perf_counter()

        state = np.array(initial_state, dtype=np.float64)
        num_steps = int(max_sim_time / self.dt) + 1

        if dynamics_fn is None:
            dynamics_fn = self._default_dynamics_fn

        states, controls, times, values = [], [], [], []
        docked = False
        collided = False

        for step in range(num_steps):
            sim_time = step * self.dt

            control = self.u_fn(state, sim_time)
            value = self._last_q_value

            # Apply safety filter
            control = self.safety_filter.apply(state, control)

            # Record
            states.append(state.copy())
            controls.append(control.copy())
            times.append(sim_time)
            values.append(value)

            # Termination checks (using mixin methods)
            if self._check_docked_13d(state):
                docked = True
                break

            if self._check_collision_13d(state):
                collided = True
                break

            # Integrate
            state_dot = dynamics_fn(state, control)
            state = state + self.dt * state_dot

            # Normalize quaternion and clamp
            self._wrap_state_13d(state)

        wall_time = _time.perf_counter() - t_wall_start

        controls_arr = np.array(controls)
        control_effort = float(
            np.sum(np.linalg.norm(controls_arr, axis=-1)) * self.dt)

        return {
            'trajectory': np.array(states),
            'controls': controls_arr,
            'times': np.array(times),
            'values': np.array(values),
            'success': docked and not collided,
            'docked': docked,
            'collision': collided,
            'final_state': state,
            'control_effort': control_effort,
            'wall_time': wall_time,
            'controller_type': 'rl_13d',
            'safety_filter_mode': self.safety_filter.mode,
            'safety_filter_log': self.safety_filter.get_log(),
            'n_clipped_steps': 0,
        }

    def _pd_torque_batch(self, states):
        """Vectorised PD attitude controller for a batch of states.

        Mirrors _pd_torque exactly but operates on (B, 13) torch tensors.
        Only called when self.pd_attitude is True.

        Args:
            states: (B, 13) float32 torch tensor on self.device.

        Returns:
            (B, 3) float32 torch tensor of clamped torques.
        """
        q = states[:, 6:10].float()
        q = q / (torch.norm(q, dim=-1, keepdim=True) + 1e-12)   # (B, 4)
        omega = states[:, 10:13].float()                          # (B, 3)

        qg = torch.tensor(self._q_goal_np, dtype=torch.float32,
                          device=self.device)                     # (4,)
        # Conjugate of q_goal (scalar-first convention)
        qg_conj = qg * torch.tensor(
            [1., -1., -1., -1.], dtype=torch.float32, device=self.device)

        a0, a1, a2, a3 = qg_conj[0], qg_conj[1], qg_conj[2], qg_conj[3]
        b0, b1, b2, b3 = q[:, 0], q[:, 1], q[:, 2], q[:, 3]    # each (B,)

        # Hamilton product qg_conj * q, matching _pd_torque scalar logic
        q_err = torch.stack([
            a0*b0 - a1*b1 - a2*b2 - a3*b3,
            a0*b1 + a1*b0 + a2*b3 - a3*b2,
            a0*b2 - a1*b3 + a2*b0 + a3*b1,
            a0*b3 + a1*b2 - a2*b1 + a3*b0,
        ], dim=-1)                                                # (B, 4)

        # Flip sign where scalar part < 0 (matches `if q_err[0] < 0: q_err = -q_err`)
        flip = (q_err[:, 0] < 0).unsqueeze(-1).float()
        q_err = q_err * (1.0 - 2.0 * flip)

        err_vec = 2.0 * q_err[:, 1:4]                            # (B, 3)
        tau = -self._pd_kp * err_vec - self._pd_kd * omega
        return torch.clamp(tau, -self._tau_bar, self._tau_bar)   # (B, 3)

    def simulate_docking_batch(self, initial_states_np, max_sim_time):
        """Run RL docking simulations for all ICs in parallel on the GPU.

        Replaces the sequential per-IC loop in run_compare with a single
        simulation loop that advances all B trajectories simultaneously:

          - Q-network forward pass: (B, 13) -> (B, n_actions), one call/step
          - Action lookup: argmin over action dim, index into discrete table
          - PD torque (pd_attitude only): vectorised Hamilton product
          - Safety filter: SafetyFilter.batch_apply (no-op when mode=0)
          - Euler integration: dynamics.dsdt batched
          - Termination: reach_fn / _batch_check_collision_oriented

        No post-dock coast period (matches sequential simulate_docking which
        breaks immediately on docking detection).

        Per-IC wall time and final-state capture follow the same pattern used
        by MPC and BRT batch controllers for consistent metric computation.

        Args:
            initial_states_np: (B, 13) numpy array of initial conditions.
            max_sim_time: Maximum simulation time in seconds.

        Returns:
            list of B result dicts with all fields required by run_compare /
            compute_metrics: docked, collision, success, final_state, times,
            control_effort, wall_time, safety_filter_mode, n_clipped_steps.
        """
        import time as _t
        B = len(initial_states_np)
        num_steps = int(max_sim_time / self.dt) + 1

        t_wall_start = _t.perf_counter()

        # ---- State on GPU -----------------------------------------------
        states = torch.tensor(
            initial_states_np, dtype=torch.float32, device=self.device)

        # Pre-compute discrete action table on device once
        discrete_t = torch.tensor(
            self.discrete_controls, dtype=torch.float32, device=self.device)
        # discrete_t: (27, 3) if pd_attitude else (729, 6)

        # ---- Per-IC tracking --------------------------------------------
        active   = torch.ones(B, dtype=torch.bool, device=self.device)
        docked   = torch.zeros(B, dtype=torch.bool, device=self.device)
        collided = torch.zeros(B, dtype=torch.bool, device=self.device)
        final_step = torch.zeros(B, dtype=torch.long, device=self.device)

        ctrl_effort = np.zeros(B, dtype=np.float64)

        # final_states_gpu: frozen to each IC's post-integration state at its
        # last active step.  Inactive ICs' entries are not overwritten.
        final_states_gpu = states.clone()

        # Per-IC wall time: recorded the moment each IC terminates
        ic_wall_time  = np.zeros(B, dtype=np.float64)
        ic_terminated = np.zeros(B, dtype=bool)

        print(f"  [RL batch] {B} ICs  "
              f"max_sim={max_sim_time}s  dt={self.dt}s  "
              f"device={self.device}  pd_attitude={self.pd_attitude}")

        pbar = tqdm(range(num_steps), desc="[RL batch]   ", unit="step",
                    leave=True,
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} steps"
                                " [{elapsed}<{remaining}]  {postfix}")
        for step in pbar:
            # Deactivate docked and collided ICs (no coast for RL)
            active = active & ~docked & ~collided

            if not active.any():
                break

            active_at_step_start = active.clone()
            final_step[active] = step

            # === Q-network forward pass ==================================
            with torch.no_grad():
                q_vals = self.Q_network(states.float())  # (B, n_actions)
            action_idx = q_vals.argmin(dim=1)            # (B,)

            # === Build control tensor ====================================
            if self.pd_attitude:
                forces  = discrete_t[action_idx]           # (B, 3)
                torques = self._pd_torque_batch(states)    # (B, 3)
                controls = torch.cat([forces, torques], dim=-1)  # (B, 6)
            else:
                controls = discrete_t[action_idx]          # (B, 6)

            # Zero out inactive ICs (safety: inactive controls must be 0)
            controls = controls * active_at_step_start.unsqueeze(-1).float()

            # === Safety filter (vectorised, no-op when mode=0) ===========
            if self.safety_filter.mode != 0:
                controls, _ = self.safety_filter.batch_apply(
                    states, controls, active_mask=active_at_step_start)

            # === Accumulate control effort ================================
            # Inactive controls are 0, so no masking needed
            ctrl_np = controls.detach().cpu().numpy()
            ctrl_effort += np.linalg.norm(ctrl_np, axis=-1) * self.dt

            # === Termination checks BEFORE integration ====================
            # Matches sequential simulate_docking which calls
            # _check_docked_13d / _check_collision_13d before the Euler step.
            with torch.no_grad():
                reach_vals = self.dynamics.reach_fn(states)
                newly_docked = (active_at_step_start & ~docked
                                & (reach_vals <= 0))
                docked = docked | newly_docked

                newly_collided = (active_at_step_start & ~docked
                                  & self._batch_check_collision_oriented(states))
                collided = collided | newly_collided

            # Record wall time for ICs that just terminated
            _now = _t.perf_counter() - t_wall_start
            newly_done = (newly_docked | newly_collided).cpu().numpy()
            for idx in np.where(newly_done & ~ic_terminated)[0]:
                ic_wall_time[idx] = _now
                ic_terminated[idx] = True

            # Update progress bar
            sim_time = step * self.dt
            pbar.set_postfix(
                dock=f"{int(docked.sum())}/{B}",
                coll=f"{int(collided.sum())}/{B}",
                active=int(active.sum()),
                t=f"{sim_time:.1f}s",
                refresh=False)

            # === Euler integration =======================================
            with torch.no_grad():
                state_dot = self.dynamics.dsdt(states, controls, None)
                states = states + self.dt * state_dot
            states = self._batch_wrap_quat(states)

            # Freeze each IC's final state to its post-integration state
            # at its last active step
            final_states_gpu = torch.where(
                active_at_step_start.unsqueeze(-1), states, final_states_gpu)

        wall_total = _t.perf_counter() - t_wall_start
        # Timed-out ICs get total wall time (consistent with MPC/BRT batch)
        ic_wall_time[~ic_terminated] = wall_total

        final_states_np = final_states_gpu.detach().cpu().numpy()
        docked_np     = docked.cpu().numpy()
        collided_np   = collided.cpu().numpy()
        final_step_np = final_step.cpu().numpy()

        results = []
        for i in range(B):
            fsim = float(final_step_np[i]) * self.dt
            results.append({
                'trajectory':         None,  # not stored in batch mode
                'controls':           None,
                'values':             None,
                'times':              np.array([0.0, fsim]),
                'success':            bool(docked_np[i] and not collided_np[i]),
                'docked':             bool(docked_np[i]),
                'collision':          bool(collided_np[i]),
                'final_state':        final_states_np[i],
                'controller_type':    'rl_13d',
                'control_effort':     float(ctrl_effort[i]),
                'wall_time':          float(ic_wall_time[i]),
                'safety_filter_mode': self.safety_filter.mode,
                'n_clipped_steps':    0,
            })

        n_dock = int(docked_np.sum())
        n_coll = int(collided_np.sum())
        mean_wall = float(np.mean(ic_wall_time))
        print(f"  [RL batch] done  {n_dock}/{B} docked  "
              f"{n_coll}/{B} collision  "
              f"total_wall={wall_total:.1f}s  mean_per_ic={mean_wall*1000:.1f}ms")
        return results

    def _default_dynamics_fn(self, state, control):
        """Euler dynamics using Docking13D.dsdt."""
        s_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        c_t = torch.tensor(control, dtype=torch.float32, device=self.device).unsqueeze(0)
        dsdt = self.dynamics.dsdt(s_t, c_t, None)
        return dsdt.squeeze().cpu().numpy()

"""
Safety Filter for Docking6D / Docking13D Controllers

Toggleable post-processing filter that overrides or modifies the nominal
controller output to enforce collision avoidance using a separately trained
avoid-only BRT value function.

Modes:
    0 -- Disabled.  Zero overhead, no model loaded.
    1 -- Least-restrictive.  Hard switch to avoid-optimal bang-bang control
         when V_avoid(x, tMax) <= delta.
    2 -- CBF-QP.  Solve min ||u - u_nom||^2  s.t. grad_V · f(x,u) + gamma*V >= 0
         and box control bounds.  (6D only for now.)

Usage:
    sf = SafetyFilter(mode=1, checkpoint_path='runs/Docking6D_RA_avoid', ...)
    u_filtered = sf.apply(state, u_nominal)
"""

import glob
import inspect
import os
import pickle
import re

import numpy as np
import torch

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

from utils import modules
from dynamics import dynamics as dynamics_module


class SafetyFilter:
    """Post-processing safety filter backed by an avoid-only BRT value function.

    Designed as a composed object: each controller owns a SafetyFilter instance
    and calls ``apply(state, u_nominal)`` after computing its nominal control.

    When *mode=0* no model is loaded and ``apply`` returns ``u_nominal``
    unchanged (one integer comparison of overhead).
    """

    def __init__(self, mode=0, checkpoint_path=None, tMax=None,
                 margin=0.1, gamma=0.2, device='cuda'):
        """
        Args:
            mode: 0=disabled, 1=least-restrictive, 2=CBF-QP.
            checkpoint_path: Path to the avoid-only BRT experiment directory
                *or* a specific ``.pth`` checkpoint file.  Ignored when mode=0.
            tMax: Time horizon for V_avoid queries.  ``None`` = use the avoid
                model's own ``orig_opt.tMax``.
            margin: Activation threshold delta for Mode 1 (meters, same units
                as ``avoid_fn`` signed distance).  Can be updated dynamically
                via ``set_margin()``.
            gamma: CBF decay rate for Mode 2 (default 0.2, from ComboControl).
            device: Torch device string.
        """
        self.mode = mode
        self.margin = margin
        self.gamma = gamma
        self.device = device

        if mode == 0:
            self.avoid_dynamics = None
            self.avoid_model = None
            self.avoid_tMax = None
            self._is_13d = False
            self._log = []
            return

        self._load_avoid_model(checkpoint_path, tMax)
        self._log = []

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_checkpoint(path):
        """Accept a directory or a ``.pth`` file and return the full ``.pth`` path.

        If *path* is a directory, the latest checkpoint (by epoch number) in
        ``<path>/training/checkpoints/`` is returned.
        """
        if os.path.isfile(path) and path.endswith('.pth'):
            return path

        ckpt_dir = os.path.join(path, 'training', 'checkpoints')
        if not os.path.isdir(ckpt_dir):
            raise FileNotFoundError(
                f"Cannot find checkpoints in {ckpt_dir}")

        final = os.path.join(ckpt_dir, 'model_final.pth')
        if os.path.isfile(final):
            return final

        pth_files = glob.glob(os.path.join(ckpt_dir, 'model_epoch_*.pth'))
        if not pth_files:
            raise FileNotFoundError(
                f"No checkpoint files found in {ckpt_dir}")

        def _epoch(p):
            m = re.search(r'model_epoch_(\d+)\.pth$', p)
            return int(m.group(1)) if m else -1

        pth_files.sort(key=_epoch)
        return pth_files[-1]

    def _load_avoid_model(self, checkpoint_path, tMax):
        """Load the avoid-only dynamics and SIREN model from *checkpoint_path*."""
        ckpt_file = self._resolve_checkpoint(checkpoint_path)
        experiment_dir = os.path.dirname(
            os.path.dirname(os.path.dirname(ckpt_file)))

        opt_path = os.path.join(experiment_dir, 'orig_opt.pickle')
        with open(opt_path, 'rb') as f:
            orig_opt = pickle.load(f)

        assert orig_opt.set_mode == 'avoid', (
            f"Safety filter requires set_mode='avoid', got '{orig_opt.set_mode}'")

        dynamics_class = getattr(dynamics_module, orig_opt.dynamics_class)
        sig = inspect.signature(dynamics_class)
        kwargs = {}
        for pn in sig.parameters.keys():
            if pn != 'self' and hasattr(orig_opt, pn):
                kwargs[pn] = getattr(orig_opt, pn)

        self.avoid_dynamics = dynamics_class(**kwargs)
        self.avoid_dynamics.set_model(orig_opt.deepReach_model)

        if hasattr(orig_opt, 'state_range') and orig_opt.state_range is not None:
            self.avoid_dynamics.override_state_range(orig_opt.state_range)

        self.avoid_model = modules.SingleBVPNet(
            in_features=self.avoid_dynamics.input_dim,
            out_features=1,
            type=orig_opt.model,
            mode=orig_opt.model_mode,
            final_layer_factor=1.,
            hidden_features=orig_opt.num_nl,
            num_hidden_layers=orig_opt.num_hl,
            periodic_transform_fn=self.avoid_dynamics.periodic_transform_fn,
        )
        checkpoint = torch.load(ckpt_file, map_location=self.device,
                                weights_only=False)
        self.avoid_model.load_state_dict(checkpoint['model'])
        self.avoid_model.to(self.device)
        self.avoid_model.eval()

        self.avoid_tMax = tMax if tMax is not None else orig_opt.tMax
        self._is_13d = (self.avoid_dynamics.state_dim == 13)

        print(f"[SafetyFilter] mode={self.mode}  "
              f"dynamics={self.avoid_dynamics.name}  "
              f"checkpoint={os.path.basename(ckpt_file)}  "
              f"tMax={self.avoid_tMax}  margin={self.margin}  "
              f"gamma={self.gamma}")

    # ------------------------------------------------------------------
    # Reset / log access
    # ------------------------------------------------------------------

    def set_margin(self, margin):
        """Update the activation margin dynamically (e.g. per control phase)."""
        self.margin = margin

    def reset(self):
        """Clear per-simulation log.  Call from the host controller's reset()."""
        self._log = []

    def get_log(self):
        """Return the per-timestep log (list of dicts)."""
        return self._log

    # ------------------------------------------------------------------
    # Value / gradient queries
    # ------------------------------------------------------------------

    def _get_avoid_value(self, state):
        """Query V_avoid(x, tMax) cheaply (no grad)."""
        if isinstance(state, np.ndarray):
            state_t = torch.tensor(state, dtype=torch.float32)
        else:
            state_t = state.clone().detach().float()
        state_t = state_t.to(self.device)
        if state_t.dim() == 1:
            state_t = state_t.unsqueeze(0)

        time_col = torch.full(
            (1, 1), self.avoid_tMax, dtype=torch.float32, device=self.device)
        coord = torch.cat([time_col, state_t], dim=-1)
        model_input = self.avoid_dynamics.coord_to_input(coord)

        with torch.no_grad():
            result = self.avoid_model({'coords': model_input})
            output = result['model_out'].squeeze()

        value = self.avoid_dynamics.io_to_value(model_input, output)
        return value.item()

    def _get_avoid_gradient(self, state):
        """Query spatial gradient dV_avoid/ds (6-element vector)."""
        if isinstance(state, np.ndarray):
            state_t = torch.tensor(state, dtype=torch.float32)
        else:
            state_t = state.clone().detach().float()
        state_t = state_t.to(self.device)
        if state_t.dim() == 1:
            state_t = state_t.unsqueeze(0)

        time_col = torch.full(
            (1, 1), self.avoid_tMax, dtype=torch.float32, device=self.device)
        coord = torch.cat([time_col, state_t], dim=-1)
        model_input = self.avoid_dynamics.coord_to_input(coord)

        result = self.avoid_model({'coords': model_input})
        output = result['model_out'].squeeze()
        model_in = result['model_in']

        dv = self.avoid_dynamics.io_to_dv(model_in, output)
        dvds = dv[0, 1:].detach().cpu().numpy()
        return dvds

    def _get_avoid_value_and_gradient(self, state):
        """Query V_avoid and dV/ds in a single forward pass (requires grad)."""
        if isinstance(state, np.ndarray):
            state_t = torch.tensor(state, dtype=torch.float32)
        else:
            state_t = state.clone().detach().float()
        state_t = state_t.to(self.device)
        if state_t.dim() == 1:
            state_t = state_t.unsqueeze(0)

        time_col = torch.full(
            (1, 1), self.avoid_tMax, dtype=torch.float32, device=self.device)
        coord = torch.cat([time_col, state_t], dim=-1)
        model_input = self.avoid_dynamics.coord_to_input(coord)

        result = self.avoid_model({'coords': model_input})
        output = result['model_out'].squeeze()
        model_in = result['model_in']

        value = self.avoid_dynamics.io_to_value(
            model_input.detach(), output.detach())
        dv = self.avoid_dynamics.io_to_dv(model_in, output)
        dvds = dv[0, 1:].detach().cpu().numpy()

        return value.item(), dvds

    # ------------------------------------------------------------------
    # Safety control
    # ------------------------------------------------------------------

    def _compute_safety_control(self, dvds, state=None):
        """Avoid-mode optimal bang-bang control from the value function gradient.

        Maximises V_avoid (pushes state away from the failure set).
        Sign convention is OPPOSITE of reach-avoid:
            u_i = +u_max * sign(dV/dv_i)

        Dispatches to 6D or 13D allocation based on loaded dynamics.
        """
        if self._is_13d:
            return self._compute_safety_control_13d(dvds, state)
        return self._compute_safety_control_6d(dvds)

    def _compute_safety_control_6d(self, dvds):
        """6D bang-bang: 3-element [Fx, Fy, τ]."""
        u_bar = self.avoid_dynamics.u_bar
        u_theta_bar = self.avoid_dynamics.u_theta_bar

        u_x = u_bar if dvds[2] > 0 else -u_bar
        u_y = u_bar if dvds[3] > 0 else -u_bar
        u_theta = u_theta_bar if dvds[5] > 0 else -u_theta_bar

        return np.array([u_x, u_y, u_theta])

    def _compute_safety_control_13d(self, dvds, state):
        """13D bang-bang: 6-element [Fx, Fy, Fz, τx, τy, τz].

        Uses rotation-aware force allocation (same logic as
        Docking13DControllerMixin._compute_brt_control_13d) but with
        OPPOSITE sign convention — maximise V_avoid instead of minimise.
        """
        q = np.asarray(state[6:10], dtype=np.float64)
        q_norm = np.linalg.norm(q)
        if q_norm > 1e-12:
            q = q / q_norm
        R = self.avoid_dynamics._quat_to_R_np(q)

        # Force: body coefficients = R @ p_v / mc
        p_v = np.asarray(dvds[3:6], dtype=np.float64)
        coeff_body = (R @ p_v) / self.avoid_dynamics.mc
        # Maximise V_avoid → sign is +F_bar when coeff > 0
        F = np.where(coeff_body > 0, self.avoid_dynamics.F_bar,
                     -self.avoid_dynamics.F_bar)

        # Torque: effective coefficients = I^{-T} @ p_omega
        I_np = self.avoid_dynamics.I.detach().cpu().numpy()
        p_omega = np.asarray(dvds[10:13], dtype=np.float64)
        coeff_tau = np.linalg.solve(I_np.T, p_omega)
        tau = np.where(coeff_tau > 0, self.avoid_dynamics.tau_bar,
                       -self.avoid_dynamics.tau_bar)

        return np.concatenate([F, tau])

    # ------------------------------------------------------------------
    # Mode dispatch
    # ------------------------------------------------------------------

    def apply(self, state, u_nominal):
        """Post-process *u_nominal* through the safety filter.

        Args:
            state: numpy (6,) current state [px, py, vx, vy, theta, omega].
            u_nominal: numpy (3,) nominal control [ux, uy, u_theta].

        Returns:
            numpy (3,) filtered control.
        """
        if self.mode == 0:
            return u_nominal
        if self.mode == 1:
            return self._apply_least_restrictive(state, u_nominal)
        if self.mode == 2:
            return self._apply_cbf_qp(state, u_nominal)
        raise ValueError(f"Unknown safety filter mode {self.mode}")

    # ------------------------------------------------------------------
    # Batched application (Mode 1 only, for GPU batch simulation)
    # ------------------------------------------------------------------

    def batch_apply(self, states, controls, active_mask=None, margins=None):
        """Vectorised least-restrictive filter for a batch of states.

        Only supports mode 1.  Mode 0 returns controls unchanged.

        Args:
            states:      (B, 13) torch tensor on device.
            controls:    (B, cdim) torch tensor on device.
            active_mask: (B,) bool torch tensor — only process active ICs.
                         None → process all.
            margins:     (B,) float torch tensor — per-IC activation margin.
                         None → use ``self.margin`` for all ICs.

        Returns:
            filtered_controls: (B, cdim) torch tensor.
            filter_active:     (B,) bool torch tensor — True where filter fired.
        """
        B = states.shape[0]
        device = states.device
        filter_active = torch.zeros(B, dtype=torch.bool, device=device)

        if self.mode == 0:
            return controls, filter_active

        if self.mode != 1:
            raise NotImplementedError(
                "batch_apply only supports mode 0 (disabled) and 1 "
                f"(least-restrictive), got mode={self.mode}")

        if active_mask is None:
            active_mask = torch.ones(B, dtype=torch.bool, device=device)

        active_idx = torch.where(active_mask)[0]
        if len(active_idx) == 0:
            return controls, filter_active

        # --- Batch value query (no grad) -------------------------------- #
        active_states = states[active_idx]                       # (Na, 13)
        Na = active_states.shape[0]
        time_col = torch.full(
            (Na, 1), self.avoid_tMax, dtype=torch.float32, device=device)
        coord = torch.cat([time_col, active_states.float()], dim=-1)
        model_input = self.avoid_dynamics.coord_to_input(coord)

        with torch.no_grad():
            result = self.avoid_model({'coords': model_input})
            output = result['model_out'].squeeze(-1)
        V = self.avoid_dynamics.io_to_value(model_input, output)  # (Na,)

        # --- Identify ICs that need safety override --------------------- #
        if margins is not None:
            active_margins = margins[active_idx]                 # (Na,)
        else:
            active_margins = torch.full((Na,), self.margin, device=device)
        needs_override = V <= active_margins                     # (Na,)
        if not needs_override.any():
            return controls, filter_active

        override_local = torch.where(needs_override)[0]          # indices in active_states
        override_global = active_idx[override_local]              # indices in full batch
        filter_active[override_global] = True

        # --- Batch gradient query (requires grad) ----------------------- #
        ovr_states = active_states[override_local]               # (No, 13)
        No = ovr_states.shape[0]
        time_col_o = torch.full(
            (No, 1), self.avoid_tMax, dtype=torch.float32, device=device)
        coord_o = torch.cat([time_col_o, ovr_states.float()], dim=-1)
        model_input_o = self.avoid_dynamics.coord_to_input(coord_o)

        result_o = self.avoid_model({'coords': model_input_o})
        output_o = result_o['model_out'].squeeze(-1)
        model_in_o = result_o['model_in']

        dv = self.avoid_dynamics.io_to_dv(model_in_o, output_o)  # (No, 14)
        dvds = dv[:, 1:].detach()                                # (No, 13)

        # --- Batch bang-bang safety control ------------------------------ #
        u_safety = self._batch_compute_safety_control_13d(
            dvds, ovr_states)                                    # (No, 6)

        # Replace controls for overridden ICs
        filtered = controls.clone()
        filtered[override_global] = u_safety.to(controls.dtype)

        return filtered, filter_active

    def _batch_compute_safety_control_13d(self, dvds, states):
        """Batched 13D bang-bang safety control (maximise V_avoid).

        Args:
            dvds:   (N, 13) gradient tensor on device.
            states: (N, 13) state tensor on device.

        Returns:
            (N, 6) safety control tensor on device.
        """
        device = states.device
        q = states[:, 6:10]                                      # (N, 4)
        q = q / (q.norm(dim=-1, keepdim=True) + 1e-12)

        # Quaternion to rotation matrix (batched)
        q0, q1, q2, q3 = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        R = torch.stack([
            torch.stack([1 - 2*(q2*q2 + q3*q3), 2*(q1*q2 + q0*q3),     2*(q1*q3 - q0*q2)], dim=-1),
            torch.stack([2*(q1*q2 - q0*q3),     1 - 2*(q1*q1 + q3*q3),  2*(q2*q3 + q0*q1)], dim=-1),
            torch.stack([2*(q1*q3 + q0*q2),     2*(q2*q3 - q0*q1),      1 - 2*(q1*q1 + q2*q2)], dim=-1),
        ], dim=-2)                                                # (N, 3, 3)

        # Force: coeff_body = R @ p_v / mc
        p_v = dvds[:, 3:6].unsqueeze(-1)                        # (N, 3, 1)
        mc = self.avoid_dynamics.mc
        coeff_body = (torch.bmm(R, p_v).squeeze(-1)) / mc       # (N, 3)
        F_bar = self.avoid_dynamics.F_bar
        F = torch.where(coeff_body > 0, F_bar, -F_bar)          # (N, 3)

        # Torque: coeff_tau = I^{-T} @ p_omega
        I_np = self.avoid_dynamics.I.detach().cpu().numpy()
        I_inv_T = torch.tensor(
            np.linalg.inv(I_np).T, dtype=torch.float32, device=device)
        p_omega = dvds[:, 10:13]                                 # (N, 13→10:13)
        coeff_tau = p_omega @ I_inv_T.T                          # (N, 3)
        tau_bar = self.avoid_dynamics.tau_bar
        tau = torch.where(coeff_tau > 0, tau_bar, -tau_bar)      # (N, 3)

        return torch.cat([F, tau], dim=-1)                       # (N, 6)

    # ------------------------------------------------------------------
    # Mode 1: Least-restrictive
    # ------------------------------------------------------------------

    def _apply_least_restrictive(self, state, u_nominal):
        """Hard switch: override with avoid-optimal control when V <= delta.

        The Hamilton-Jacobi least-restrictive safety filter:  the nominal
        controller has full authority when V_avoid > delta.  When V_avoid drops
        to delta or below, the avoid-BRT's optimal bang-bang control is applied
        (it maximises dV/dt, pushing the state away from the failure set).
        """
        V = self._get_avoid_value(state)

        if V > self.margin:
            self._log.append({
                'V_avoid': V,
                'filter_active': False,
            })
            return u_nominal

        dvds = self._get_avoid_gradient(state)
        u_safety = self._compute_safety_control(dvds, state)

        self._log.append({
            'V_avoid': V,
            'filter_active': True,
            'u_nominal': u_nominal.copy(),
            'u_safety': u_safety.copy(),
        })
        return u_safety

    # ------------------------------------------------------------------
    # Mode 2: CBF-QP  (replicates ComboControl filter_mode=2)
    # ------------------------------------------------------------------

    def _apply_cbf_qp(self, state, u_nominal):
        """Solve  min ||u - u_nom||^2  s.t. CBF constraint + box bounds.

        The CBF constraint  grad_V · f(x, u) + gamma * V >= 0  ensures the
        avoid-BRT value function decreases at most at rate gamma, providing a
        continuous safety guarantee.  Adapted from ComboControl.py
        qp_controller_4D / qp_controller_2D, unified for the 6D system.
        """
        import cvxpy as cp

        V, dvds = self._get_avoid_value_and_gradient(state)

        px, py, vx, vy, theta, omega = state
        n = float(self.avoid_dynamics.n)
        mc = float(self.avoid_dynamics.mc)
        jc = float(self.avoid_dynamics.jc)
        u_bar = float(self.avoid_dynamics.u_bar)
        u_theta_bar = float(self.avoid_dynamics.u_theta_bar)

        # Lie derivative of V along the drift (open-loop) dynamics
        Lf_V = (dvds[0] * vx + dvds[1] * vy
                + dvds[2] * (3 * n**2 * px + 2 * n * vy)
                + dvds[3] * (-2 * n * vx)
                + dvds[4] * omega)

        # Control-gradient coefficients  a = dV/ds · (df/du)
        a = np.array([dvds[2] / mc, dvds[3] / mc, dvds[5] / jc])

        # CBF constraint:  a @ u >= -(Lf_V + gamma * V)
        rhs = -(Lf_V + self.gamma * V)

        u = cp.Variable(3)
        u_min = np.array([-u_bar, -u_bar, -u_theta_bar])
        u_max = np.array([u_bar, u_bar, u_theta_bar])

        constraints = [
            a @ u >= rhs,
            u >= u_min,
            u <= u_max,
        ]
        objective = cp.Minimize(cp.sum_squares(u - u_nominal))
        prob = cp.Problem(objective, constraints)

        try:
            prob.solve(solver=cp.OSQP, warm_start=True)
        except cp.SolverError:
            try:
                prob.solve(solver=cp.SCS)
            except cp.SolverError:
                pass

        if prob.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            u_filtered = np.asarray(u.value).flatten()
        else:
            u_filtered = self._compute_safety_control(dvds, state)

        cbf_margin = float(a @ u_filtered - rhs)

        u_safety = self._compute_safety_control(dvds, state)
        denom = np.linalg.norm(u_safety - u_nominal)
        alpha_eff = (np.linalg.norm(u_filtered - u_nominal) / denom
                     if denom > 1e-8 else 0.0)

        self._log.append({
            'V_avoid': V,
            'Lf_V': float(Lf_V),
            'cbf_margin': cbf_margin,
            'alpha_effective': alpha_eff,
            'u_nominal': u_nominal.copy(),
            'u_safety': u_safety.copy(),
            'u_filtered': u_filtered.copy(),
            'qp_status': prob.status,
        })
        return u_filtered

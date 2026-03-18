"""
Gradient-based MPC via differentiable shooting with multi-start Adam.

Two classes:
  - DifferentiableValueFunction: evaluates V(x, t) with full gradient flow
    through a frozen SIREN network, bypassing SingleBVPNet.forward()'s .detach().
  - GradientMPC: multi-start trajectory optimization shared by both the baseline
    MPC controller and the MPC+terminal controller.
"""

import torch
import torch.nn as nn
import math


class DifferentiableValueFunction:
    """Evaluate V(x, t) with full gradient flow through a frozen SIREN.

    SingleBVPNet.forward() detaches the input (modules.py:124-125), breaking
    the computation graph. This class bypasses that by calling model.net
    (FCBlock) directly with a functional normalization + periodic transform.
    Network weights are frozen so gradients flow through the network for
    dV/dx but do not accumulate on the weights themselves.
    """

    def __init__(self, model, dynamics, device):
        """
        Args:
            model:    SingleBVPNet instance (weights will be frozen).
            dynamics: Docking6D instance.
            device:   torch device.
        """
        self.net = model.net  # FCBlock (SIREN backbone)
        self.device = device

        # Freeze SIREN weights
        for p in self.net.parameters():
            p.requires_grad_(False)

        # Cache normalization constants on device
        self.state_mean = dynamics.state_mean.to(device).float()  # (6,)
        self.state_var = dynamics.state_var.to(device).float()    # (6,)
        self.value_var = float(dynamics.value_var)
        self.value_normto = float(dynamics.value_normto)
        self.theta_var = float(dynamics.state_var[4].item())      # pi

        # State bounds for clamping (prevents out-of-distribution queries)
        sr = dynamics.state_range_.float().to(device)  # (6, 2)
        self.state_lo = sr[:, 0]
        self.state_hi = sr[:, 1]

        self.dynamics = dynamics
        self.deepReach_model = dynamics.deepReach_model

    def __call__(self, states, t_query):
        """V(states, t_query) — fully differentiable w.r.t. states.

        Args:
            states:  (..., 6) raw (unnormalized) state tensor.
            t_query: scalar time value.
        Returns:
            (...,) value tensor.
        """
        # 1. Clamp to training domain (torch.clamp has valid subgradients)
        states_clamped = torch.clamp(states, self.state_lo, self.state_hi)

        # 2. Normalize: (state - mean) / var
        norm = (states_clamped - self.state_mean) / self.state_var

        # 3. Extract components for periodic transform
        prefix = norm[..., :4]          # [norm_x, norm_y, norm_vx, norm_vy]
        norm_theta = norm[..., 4:5]
        norm_omega = norm[..., 5:6]

        # 4. Periodic transform (functional, no CPU allocation)
        #    periodic_transform_fn multiplies normalized theta by state_var[4]
        #    which recovers the raw theta, then takes sin/cos
        sin_theta = torch.sin(norm_theta * self.theta_var)
        cos_theta = torch.cos(norm_theta * self.theta_var)

        time_col = torch.full(
            (*states.shape[:-1], 1), t_query,
            dtype=states.dtype, device=self.device)

        # 8D input: [time, norm_x, norm_y, norm_vx, norm_vy, sin, cos, norm_omega]
        net_input = torch.cat(
            [time_col, prefix, sin_theta, cos_theta, norm_omega], dim=-1)

        # 5. Forward through SIREN
        raw_output = self.net(net_input).squeeze(-1)

        # 6. Denormalize: V = output * t * value_var / value_normto + boundary_fn(x)
        if self.deepReach_model == 'exact':
            value = (raw_output * t_query * self.value_var / self.value_normto
                     + self.dynamics.boundary_fn(states_clamped))
        elif self.deepReach_model == 'diff':
            value = (raw_output * self.value_var / self.value_normto
                     + self.dynamics.boundary_fn(states_clamped))
        else:
            value = raw_output * self.value_var / self.value_normto + self.dynamics.value_mean

        return value


class GradientMPC:
    """Multi-start gradient-based trajectory optimization via differentiable shooting.

    Shared by both the baseline MPC controller (analytical cost) and the
    MPC+terminal controller (analytical + learned terminal cost). The cost
    function is injected as a callable.
    """

    def __init__(self, dt, horizon, dynamics, device,
                 num_iters=50, lr=1.0, num_restarts=8):
        """
        Args:
            dt:           Planning timestep in seconds.
            horizon:      Number of planning steps.
            dynamics:     Docking6D instance.
            device:       torch device.
            num_iters:    Adam iterations per MPC step.
            lr:           Adam learning rate.
            num_restarts: Number of parallel random restarts.
        """
        self.dt = dt
        self.horizon = horizon
        self.device = device
        self.num_iters = num_iters
        self.lr = lr
        self.num_restarts = num_restarts

        # Cache dynamics constants on device
        self.n = float(dynamics.n)
        self.mc = float(dynamics.mc)
        self.jc = float(dynamics.jc)

        # Velocity limit (for acceleration zeroing)
        sr = dynamics.state_range_.float().to(device)
        self.v_max = float(sr[2:4, 1].max().item())
        self.vx_lo = float(sr[2, 0].item())
        self.vx_hi = float(sr[2, 1].item())
        self.vy_lo = float(sr[3, 0].item())
        self.vy_hi = float(sr[3, 1].item())
        self.omega_lo = float(sr[5, 0].item())
        self.omega_hi = float(sr[5, 1].item())

        # Control bounds as device tensors for broadcasting
        cr = dynamics.control_range_.float().to(device)
        self.u_lo = cr[:, 0]  # (3,)
        self.u_hi = cr[:, 1]  # (3,)

    def _differentiable_step(self, state, control):
        """One Euler step using functional ops only (autograd-safe).

        Args:
            state:   (..., 6) tensor [px, py, vx, vy, theta, omega].
            control: (..., 3) tensor [ux, uy, u_theta].
        Returns:
            (..., 6) next state tensor.
        """
        px, py, vx, vy, theta, omega = state.unbind(-1)
        ux, uy, u_th = control.unbind(-1)

        # CW dynamics
        dpx = vx
        dpy = vy
        dvx = 3 * self.n**2 * px + 2 * self.n * vy + ux / self.mc
        dvy = -2 * self.n * vx + uy / self.mc
        dtheta = omega
        domega = u_th / self.jc

        # Velocity-limit acceleration zeroing (matches dsdt lines 731-739)
        z = torch.zeros_like(dvx)
        dvx = torch.where((vx >= self.v_max) & (dvx > 0), z, dvx)
        dvx = torch.where((vx <= -self.v_max) & (dvx < 0), z, dvx)
        dvy = torch.where((vy >= self.v_max) & (dvy > 0), z, dvy)
        dvy = torch.where((vy <= -self.v_max) & (dvy < 0), z, dvy)

        # Euler integration
        next_px = px + dpx * self.dt
        next_py = py + dpy * self.dt
        next_vx = vx + dvx * self.dt
        next_vy = vy + dvy * self.dt
        next_theta = theta + dtheta * self.dt
        next_omega = omega + domega * self.dt

        # State wrapping (matches equivalent_wrapped_state lines 674-689)
        next_theta = torch.atan2(torch.sin(next_theta), torch.cos(next_theta))
        next_vx = torch.clamp(next_vx, self.vx_lo, self.vx_hi)
        next_vy = torch.clamp(next_vy, self.vy_lo, self.vy_hi)
        next_omega = torch.clamp(next_omega, self.omega_lo, self.omega_hi)

        return torch.stack(
            [next_px, next_py, next_vx, next_vy, next_theta, next_omega],
            dim=-1)

    def _differentiable_rollout(self, initial_state, controls):
        """Roll out a trajectory through differentiable dynamics.

        Args:
            initial_state: (K, 6) tensor.
            controls:      (K, H, 3) tensor.
        Returns:
            (K, H+1, 6) trajectory tensor.
        """
        traj = [initial_state]
        state = initial_state
        for k in range(self.horizon):
            state = self._differentiable_step(state, controls[:, k, :])
            traj.append(state)
        return torch.stack(traj, dim=1)

    def optimize(self, initial_state, cost_fn, warm_start=None):
        """Run multi-start gradient optimization.

        Args:
            initial_state: (6,) tensor.
            cost_fn:       callable(trajectory (K,H+1,6), controls (K,H,3)) -> (K,) costs.
            warm_start:    optional (H, 3) tensor from previous step.
        Returns:
            best_controls: (H, 3), best_cost: float, best_traj: (H+1, 6).
        """
        K = self.num_restarts

        # Initialize: slot 0 = warm-start or zeros, rest = uniform random
        controls = torch.empty(K, self.horizon, 3, device=self.device)
        if warm_start is not None:
            controls[0] = warm_start
        else:
            controls[0] = 0.0
        if K > 1:
            controls[1:] = (
                self.u_lo
                + (self.u_hi - self.u_lo)
                * torch.rand(K - 1, self.horizon, 3, device=self.device)
            )

        init_batch = initial_state.unsqueeze(0).expand(K, -1)

        controls_param = nn.Parameter(controls)
        optimizer = torch.optim.Adam([controls_param], lr=self.lr)

        best_cost = float('inf')
        best_controls = controls[0].detach().clone()
        best_traj = None

        for _ in range(self.num_iters):
            optimizer.zero_grad()

            # Differentiable clamp for gradient flow
            clamped = torch.clamp(controls_param, self.u_lo, self.u_hi)
            trajectory = self._differentiable_rollout(init_batch, clamped)
            costs = cost_fn(trajectory, clamped)

            # Track best across ALL iterations and restarts
            with torch.no_grad():
                min_cost, min_idx = costs.min(0)
                if min_cost.item() < best_cost:
                    best_cost = min_cost.item()
                    best_controls = clamped[min_idx].detach().clone()
                    best_traj = trajectory[min_idx].detach().clone()

            costs.sum().backward()
            optimizer.step()

            # Hard projection after step for box constraint
            with torch.no_grad():
                controls_param.data.clamp_(self.u_lo, self.u_hi)

        return best_controls, best_cost, best_traj

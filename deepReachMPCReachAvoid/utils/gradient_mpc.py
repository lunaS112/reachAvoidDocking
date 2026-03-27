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
            dynamics: Dynamics instance (currently Docking6D-specific layout).
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


class DifferentiableValueFunction13D:
    """Evaluate V(x, t) with full gradient flow for 13D dynamics.

    Like DifferentiableValueFunction but handles the 13D periodic transform
    (quaternion sign canonicalization, no dimension expansion) instead of the
    6D sin/cos theta expansion.
    """

    def __init__(self, model, dynamics, device):
        self.net = model.net  # FCBlock (SIREN backbone)
        self.device = device

        # Freeze SIREN weights
        for p in self.net.parameters():
            p.requires_grad_(False)

        # Cache normalization constants
        self.state_mean = dynamics.state_mean.to(device).float()  # (13,)
        self.state_var = dynamics.state_var.to(device).float()    # (13,)
        self.value_var = float(dynamics.value_var)
        self.value_normto = float(dynamics.value_normto)

        # State bounds for clamping
        sr = dynamics.state_range_.float().to(device)  # (13, 2)
        self.state_lo = sr[:, 0]
        self.state_hi = sr[:, 1]

        self.dynamics = dynamics
        self.deepReach_model = dynamics.deepReach_model

    def __call__(self, states, t_query):
        """V(states, t_query) — fully differentiable w.r.t. states.

        Args:
            states:  (..., 13) raw (unnormalized) state tensor.
            t_query: scalar float **or** (...,) tensor of per-sample times.
        Returns:
            (...,) value tensor.
        """
        # 1. Clamp to training domain
        states_clamped = torch.clamp(states, self.state_lo, self.state_hi)

        # 2. Normalize: (state - mean) / var
        norm = (states_clamped - self.state_mean) / self.state_var

        # 3. Prepend time (scalar or per-sample)
        if isinstance(t_query, (int, float)):
            time_col = torch.full(
                (*states.shape[:-1], 1), t_query,
                dtype=states.dtype, device=self.device)
        else:
            time_col = t_query.unsqueeze(-1)  # (..., 1)
        net_input = torch.cat([time_col, norm], dim=-1)  # (..., 14)

        # 4. Quaternion sign canonicalization (matching periodic_transform_fn)
        q0 = net_input[..., 7:8]
        sign = torch.sign(q0)
        sign = torch.where(sign == 0, torch.ones_like(sign), sign)
        net_input = torch.cat([
            net_input[..., :7],           # time + pos(3) + vel(3)
            net_input[..., 7:11] * sign,  # quaternion (canonicalized)
            net_input[..., 11:],          # angular velocity(3)
        ], dim=-1)

        # 5. Forward through SIREN
        raw_output = self.net(net_input).squeeze(-1)

        # 6. Denormalize (t_query broadcasts for both scalar and tensor)
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
            dynamics:     Dynamics instance.
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

        self.dynamics = dynamics

        cr = dynamics.control_range_.float().to(device)
        self.u_lo = cr[:, 0]  # (control_dim,)
        self.u_hi = cr[:, 1]  # (control_dim,)

    def _differentiable_step(self, state, control):
        """One Euler step — delegates to dynamics.differentiable_step()."""
        return self.dynamics.differentiable_step(state, control, self.dt)

    def _differentiable_rollout(self, initial_state, controls):
        """Roll out a trajectory through differentiable dynamics.

        Args:
            initial_state: (K, state_dim) tensor.
            controls:      (K, H, control_dim) tensor.
        Returns:
            (K, H+1, state_dim) trajectory tensor.
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
            initial_state: (state_dim,) tensor.
            cost_fn:       callable(traj (K,H+1,state_dim), controls (K,H,control_dim)) -> (K,).
            warm_start:    optional (H, control_dim) tensor from previous step.
        Returns:
            best_controls: (H, control_dim), best_cost: float, best_traj: (H+1, state_dim).
        """
        K = self.num_restarts

        # Initialize: slot 0 = warm-start or zeros, rest = uniform random
        cdim = self.dynamics.control_dim
        controls = torch.empty(K, self.horizon, cdim, device=self.device)
        if warm_start is not None:
            controls[0] = warm_start
        else:
            controls[0] = 0.0
        if K > 1:
            controls[1:] = (
                self.u_lo
                + (self.u_hi - self.u_lo)
                * torch.rand(K - 1, self.horizon, cdim, device=self.device)
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

    def batch_optimize(self, initial_states, cost_fn, warm_start_controls):
        """Optimize B states simultaneously with a single Adam optimizer.

        Unlike optimize() which takes a single state with K restarts,
        this uses the batch dimension directly: one state per slot, each
        warm-started from its own control sequence.

        Args:
            initial_states:       (B, state_dim) tensor of initial conditions.
            cost_fn:              callable(traj (B,H+1,state_dim), controls (B,H,control_dim)) -> (B,).
            warm_start_controls:  (B, H, control_dim) tensor of initial control sequences.
        Returns:
            best_controls (B, H, control_dim), best_costs (B,), best_trajs (B, H+1, state_dim).
        """
        B = initial_states.shape[0]

        controls_param = nn.Parameter(warm_start_controls.clone())
        optimizer = torch.optim.Adam([controls_param], lr=self.lr)

        best_costs = torch.full((B,), float('inf'), device=self.device)
        best_controls = warm_start_controls.clone()
        best_trajs = None

        for _ in range(self.num_iters):
            optimizer.zero_grad()

            clamped = torch.clamp(controls_param, self.u_lo, self.u_hi)
            trajectory = self._differentiable_rollout(initial_states, clamped)
            costs = cost_fn(trajectory, clamped)
            costs = torch.nan_to_num(costs, nan=float('inf'))

            # Track per-state best across all iterations
            with torch.no_grad():
                improved = costs < best_costs
                if improved.any():
                    best_costs[improved] = costs[improved]
                    best_controls[improved] = clamped[improved].detach().clone()
                    if best_trajs is None:
                        best_trajs = trajectory.detach().clone()
                    else:
                        best_trajs[improved] = trajectory[improved].detach().clone()

            costs.sum().backward()
            optimizer.step()

            with torch.no_grad():
                controls_param.data.clamp_(self.u_lo, self.u_hi)

        # If no iteration improved (shouldn't happen), use warm-start trajectory
        if best_trajs is None:
            with torch.no_grad():
                clamped = torch.clamp(warm_start_controls, self.u_lo, self.u_hi)
                best_trajs = self._differentiable_rollout(initial_states, clamped)

        return best_controls, best_costs, best_trajs


# ---------------------------------------------------------------------------
# Gradient-based MPC label generation for refinement stage
# ---------------------------------------------------------------------------

def _generate_warm_start(model, dynamics, initial_states, dt, horizon, device):
    """Generate bang-bang control sequences by rolling out the learned policy.

    Adapted from MPC.rollout_with_policy(). For each state, computes the
    value function gradient via io_to_dv, extracts bang-bang control via
    optimal_control, and steps the dynamics forward.

    Args:
        model:          SingleBVPNet (current learned model).
        dynamics:       Dynamics instance.
        initial_states: (B, state_dim) tensor on device.
        dt:             Planning timestep in seconds.
        horizon:        Number of planning steps.
        device:         torch device.
    Returns:
        controls: (B, H, control_dim) bang-bang control sequences.
    """
    B = initial_states.shape[0]
    controls = torch.zeros(B, horizon, dynamics.control_dim, device=device)
    state = initial_states.clone()
    time_remaining = torch.full((B, 1), horizon * dt, device=device)

    for k in range(horizon):
        # Build coord: [time_remaining, state] -> (B, 7)
        coord = torch.cat([time_remaining, state], dim=-1)
        norm_coord = dynamics.coord_to_input(coord)

        # Query model (internally handles .detach().requires_grad_(True))
        result = model({'coords': norm_coord})

        # Value gradient via Jacobian
        dvs = dynamics.io_to_dv(
            result['model_in'],
            result['model_out'].squeeze(dim=-1)
        ).detach()

        # Bang-bang control from gradient sign
        u = dynamics.optimal_control(state, dvs[..., 1:])
        u = dynamics.clamp_control(state, u)
        controls[:, k, :] = u

        # Step dynamics forward
        dsdt_val = dynamics.dsdt(state, u, None)
        state = dynamics.equivalent_wrapped_state(state + dsdt_val * dt)
        time_remaining = time_remaining - dt

    return controls


def _bootstrap_labels(trajectories, dynamics, T, dt):
    """Extract (time, state) -> cost-to-go labels from optimized trajectories.

    Adapted from MPC.get_batch_data(). Computes cost-to-go at each timestep
    along the trajectory and filters to in-range states. Supports reach_avoid,
    avoid, and reach modes.

    Args:
        trajectories: (B, H+1, state_dim) optimized state trajectories.
        dynamics:     Dynamics instance.
        T:            Total time horizon (for time coordinate).
        dt:           Timestep.
    Returns:
        (coords, value_labels) -- CPU tensors. coords normalized via coord_to_input.
    """
    device = trajectories.device
    B, Hp1, D = trajectories.shape
    H = Hp1 - 1

    # Mode-aware cost computation (mirrors MPC.get_batch_data lines 55-71)
    if dynamics.set_mode == 'reach_avoid':
        reach_values = dynamics.reach_fn(trajectories)   # (B, H+1)
        avoid_values = dynamics.avoid_fn(trajectories)   # (B, H+1)
        # cummax is the mathematically correct running-max formulation;
        # cost_fn uses global-max, yielding a slightly different validity mask
        cummax_neg_avoid = torch.cummax(-avoid_values, dim=-1).values
        ra_cost = torch.clamp(reach_values, min=cummax_neg_avoid)
        _, min_idx = torch.min(ra_cost, dim=-1)
    elif dynamics.set_mode in ('avoid', 'reach'):
        lxs = dynamics.boundary_fn(trajectories)         # (B, H+1)
        _, min_idx = torch.min(lxs, dim=-1)
    else:
        raise NotImplementedError(
            f"set_mode '{dynamics.set_mode}' not supported")

    coords_list = []
    values_list = []
    state_lo = dynamics.state_range_[:, 0].to(device) - 0.01
    state_hi = dynamics.state_range_[:, 1].to(device) + 0.01

    for i in range(H):
        valid = (min_idx > i)  # (B,) bool — cost-to-go is valid before optimum
        if valid.sum() == 0:
            continue

        # Coord: [T - i*dt, state_at_step_i]
        coord_i = torch.zeros(valid.sum(), D + 1, device=device)
        coord_i[:, 0] = T - i * dt
        coord_i[:, 1:] = trajectories[valid, i, :]

        # Cost-to-go from step i onward
        if dynamics.set_mode == 'reach_avoid':
            ra_cost_i = torch.clamp(
                reach_values[valid, i:],
                min=torch.max(-avoid_values[valid, i:],
                              dim=-1).values.unsqueeze(-1))
            value_i = torch.min(ra_cost_i, dim=-1).values
        elif dynamics.set_mode in ('avoid', 'reach'):
            value_i = torch.min(lxs[valid, i:], dim=-1).values

        # In-range filter
        in_range = (
            torch.all(coord_i[:, 1:] >= state_lo, dim=-1)
            & torch.all(coord_i[:, 1:] <= state_hi, dim=-1)
            & ~torch.isnan(value_i)
        )
        if in_range.sum() == 0:
            continue

        coords_list.append(dynamics.coord_to_input(coord_i[in_range]))
        values_list.append(value_i[in_range])

    if len(coords_list) == 0:
        return torch.empty(0, D + 1), torch.empty(0)

    coords = torch.cat(coords_list, dim=0)
    value_labels = torch.cat(values_list, dim=0)
    return coords.detach().cpu(), value_labels.detach().cpu()


def generate_gradient_mpc_dataset(model, dynamics, dataset, device,
                                  gradient_batch_size=256, num_iters=15, lr=1.0,
                                  num_batches=None):
    """Generate MPC training labels using gradient-based trajectory optimization.

    Warm-starts from the learned policy's bang-bang controls, then refines
    via differentiable shooting + Adam. Produces labels in the same format
    as generate_MPC_dataset().

    Args:
        model:                SingleBVPNet (current learned model).
        dynamics:             Dynamics instance.
        dataset:              ReachabilityDataset (for sampling and config).
        device:               torch device string or object.
        gradient_batch_size:  States per GPU chunk (controls peak memory).
        num_iters:            Adam iterations per batch.
        lr:                   Adam learning rate.
        num_batches:          IC batches for gradient refinement.
                              None falls back to dataset.num_MPC_batches.
    Returns:
        (MPC_inputs, MPC_values) as CPU tensors in same format as
        generate_MPC_dataset().
    """
    from tqdm.autonotebook import tqdm as tqdm_auto

    effective_dt = dataset.MPC_dt
    if dataset.refinement_dt is not None:
        effective_dt = dataset.refinement_dt
        print(f"  Gradient refinement: using dt={effective_dt}")

    tMax = dataset.tMax
    horizon = math.ceil(tMax / effective_dt)

    gmpc = GradientMPC(
        dt=effective_dt,
        horizon=horizon,
        dynamics=dynamics,
        device=device,
        num_iters=num_iters,
        lr=lr,
        num_restarts=1,  # unused by batch_optimize, but keeps init consistent
    )

    # Cost function matching dynamics.cost_fn (reachability reach-avoid)
    def cost_fn(trajectory, controls):
        return dynamics.cost_fn(trajectory)

    all_inputs = []
    all_values = []

    effective_num_batches = num_batches if num_batches is not None else dataset.num_MPC_batches

    was_training = model.training
    model.eval()
    try:
        for batch_idx in tqdm_auto(range(effective_num_batches),
                                   desc="Gradient MPC labels"):
            states = dataset.sample_init_state().to(device)
            B = states.shape[0]

            for chunk_start in range(0, B, gradient_batch_size):
                chunk_end = min(chunk_start + gradient_batch_size, B)
                chunk_states = states[chunk_start:chunk_end]

                warm_controls = _generate_warm_start(
                    model, dynamics, chunk_states, effective_dt, horizon, device)

                best_controls, best_costs, best_trajs = gmpc.batch_optimize(
                    chunk_states, cost_fn, warm_controls)

                coords, values = _bootstrap_labels(
                    best_trajs, dynamics, tMax, effective_dt)

                if coords.shape[0] > 0:
                    all_inputs.append(coords)
                    all_values.append(values)

                del warm_controls, best_controls, best_costs, best_trajs
                torch.cuda.empty_cache()
    finally:
        if was_training:
            model.train()

    if len(all_inputs) == 0:
        print("WARNING: Gradient MPC generated 0 labels")
        return torch.empty(0, dynamics.state_dim + 1), torch.empty(0)

    MPC_inputs = torch.cat(all_inputs, dim=0)
    MPC_values = torch.cat(all_values, dim=0)
    return MPC_inputs, MPC_values

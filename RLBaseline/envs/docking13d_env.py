"""
Gym environment wrapping Docking13D dynamics for reach-avoid RL training.

The environment exposes the 13-state 3D CW + rigid body attitude docking
dynamics from deepReachMPCReachAvoid/dynamics/dynamics.py as a standard
gym.Env with a discrete action space compatible with DDQNSingle.

State:  [x,y,z,vx,vy,vz,q0,q1,q2,q3,wx,wy,wz]  (13,)
Action: Discrete(729) — 6 control axes × {-max, 0, +max}
Info:   {"g_x": float, "l_x": float}

Sign conventions (matching safety_rl):
    g_x = -avoid_fn(state)   positive inside obstacle (unsafe)
    l_x =  reach_fn(state)   negative inside target   (docked)
"""

import itertools
import os
import sys

import gym
import gym.spaces
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import random
import torch

# Add project root so we can import dynamics
_PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'deepReachMPCReachAvoid'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from dynamics.dynamics import Docking13D


# IC sampling bounds (matching run_controller_13d.py without BRAT filter)
SAMPLING_BOUNDS_13D = {
    'pos': (-13.0, 13.0),
    'vel': (-0.15, 0.15),
    'omega': (-0.3, 0.3),
}


def _random_unit_quaternion(rng):
    """Sample a uniform random unit quaternion (scalar-first)."""
    u = rng.randn(4)
    u /= np.linalg.norm(u)
    if u[0] < 0:
        u = -u  # canonical form q0 >= 0
    return u


class Docking13DEnv(gym.Env):
    """Gym environment for 13D 3D docking reach-avoid training."""

    def __init__(self, device, mode="RA", doneType="TF",
                 sample_inside_obs=False, dt=0.1, importance_sampler=None):
        """
        Args:
            device: torch device string or object.
            mode: 'RA' for reach-avoid.
            doneType: 'TF' (target/fail), 'fail', or 'toEnd'.
            sample_inside_obs: allow reset() to sample inside obstacles.
            dt: integration timestep (seconds).
            importance_sampler: ImportanceSampler instance or None.
        """
        self.set_seed(0)

        self.device = device
        self.mode = mode
        self.doneType = doneType
        self.sample_inside_obs = sample_inside_obs
        self.dt = dt
        self.importance_sampler = importance_sampler

        # Dynamics (read-only, kept on CPU — step() works in numpy)
        self.dynamics = Docking13D(set_mode='reach_avoid')
        # Ensure inertia tensor and q_goal are on CPU to match step() tensors
        self.dynamics.I = self.dynamics.I.cpu()
        self.dynamics.q_goal = self.dynamics.q_goal.cpu()

        # State bounds from dynamics
        sr = self.dynamics.state_range_.cpu().numpy()  # (13, 2)
        self.bounds = sr
        self.low = sr[:, 0].copy()
        self.high = sr[:, 1].copy()

        # Observation normalization: obs = (state - center) / half_range → [-1, 1]
        self._obs_center = (self.high + self.low) / 2.0
        self._obs_half_range = (self.high - self.low) / 2.0
        self._obs_half_range = np.maximum(self._obs_half_range, 1e-8)

        # Gym spaces (normalized to [-1, 1])
        self.observation_space = gym.spaces.Box(
            -np.ones(13, dtype=np.float32), np.ones(13, dtype=np.float32))

        # Discrete action space: 6 axes × {-max, 0, +max} = 729
        F_bar = float(self.dynamics.F_bar)
        tau_bar = float(self.dynamics.tau_bar)
        force_levels = [-F_bar, 0.0, F_bar]
        torque_levels = [-tau_bar, 0.0, tau_bar]
        self.discrete_controls = np.array(
            list(itertools.product(
                force_levels, force_levels, force_levels,
                torque_levels, torque_levels, torque_levels)),
            dtype=np.float64)
        self.action_space = gym.spaces.Discrete(len(self.discrete_controls))

        # Internal state
        self.state = np.zeros(13, dtype=np.float64)

        # Cost parameters
        self.penalty = 1.0
        self.reward = -1.0

        print(f"Docking13DEnv: mode={mode}, doneType={doneType}, "
              f"dt={dt}, actions={self.action_space.n}")

    def set_sampling_range(self, sampling_bounds):
        """Override IC sampling bounds. sampling_bounds: dict with 'pos', 'vel', 'omega'."""
        self._sampling_bounds = sampling_bounds

    # ------------------------------------------------------------------
    # Seeding
    # ------------------------------------------------------------------

    def set_seed(self, seed):
        """Set random seeds for reproducibility."""
        self.seed_val = seed
        self._rng = np.random.RandomState(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        random.seed(seed)

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self, start=None):
        """Reset environment to a valid initial state.

        If start is None, samples either from the importance sampler
        (targeted goal-centric) or uniformly, rejecting states inside
        the obstacle or already docked.
        """
        if start is not None:
            self.state = np.array(start, dtype=np.float64)
            return self._normalize_obs(np.copy(self.state))

        use_targeted = (self.importance_sampler is not None
                        and self.importance_sampler.is_targeted())

        sb = getattr(self, '_sampling_bounds', SAMPLING_BOUNDS_13D)
        for _ in range(10000):
            if use_targeted:
                s = self.importance_sampler.sample_targeted()
            else:
                s = np.zeros(13, dtype=np.float64)
                s[0:3] = self._rng.uniform(sb['pos'][0], sb['pos'][1], 3)
                s[3:6] = self._rng.uniform(sb['vel'][0], sb['vel'][1], 3)
                s[6:10] = _random_unit_quaternion(self._rng)
                s[10:13] = self._rng.uniform(sb['omega'][0], sb['omega'][1], 3)

            s = np.clip(s, self.low, self.high)
            s = self._wrap_state_np(s)

            s_t = torch.FloatTensor(s).unsqueeze(0)
            avoid_val = self.dynamics.avoid_fn(s_t).item()
            reach_val = self.dynamics.reach_fn(s_t).item()

            in_obstacle = avoid_val < 0
            at_target = reach_val <= 0

            if self.sample_inside_obs:
                if not at_target:
                    break
            else:
                if not in_obstacle and not at_target:
                    break

        self.state = s
        return self._normalize_obs(np.copy(self.state))

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(self, action):
        """Advance one timestep.

        Args:
            action: int index into discrete_controls (body-frame forces/torques).

        Returns:
            (state, cost, done, info)
        """
        control = self.discrete_controls[action]

        # Euler integration via dynamics.dsdt (tensor in/out)
        s_t = torch.FloatTensor(self.state).unsqueeze(0)
        c_t = torch.FloatTensor(control).unsqueeze(0)
        dsdt = self.dynamics.dsdt(s_t, c_t, None).squeeze(0).detach().numpy()
        self.state = self.state + self.dt * dsdt

        # Normalize quaternion, clamp velocities
        self.state = self._wrap_state_np(self.state)

        # Compute margins (sign convention bridge)
        s_t = torch.FloatTensor(self.state).unsqueeze(0)
        l_x = self.dynamics.reach_fn(s_t).item()       # < 0 at target
        g_x = -self.dynamics.avoid_fn(s_t).item()       # > 0 in obstacle

        fail = g_x > 0
        success = l_x <= 0

        # Cost
        if fail:
            cost = self.penalty
        elif success:
            cost = self.reward
        else:
            cost = 0.0

        # Done
        if self.doneType == "toEnd":
            done = not self._check_within_bounds(self.state)
        elif self.doneType == "TF":
            done = fail or success
        elif self.doneType == "fail":
            done = fail
        else:
            raise ValueError(f"Invalid doneType: {self.doneType}")

        info = {"g_x": g_x, "l_x": l_x}
        return self._normalize_obs(np.copy(self.state)), cost, done, info

    # ------------------------------------------------------------------
    # Margin functions
    # ------------------------------------------------------------------

    def safety_margin(self, state):
        """g_x: positive inside failure set."""
        s_t = torch.FloatTensor(state).unsqueeze(0)
        return -self.dynamics.avoid_fn(s_t).item()

    def target_margin(self, state):
        """l_x: negative inside target set."""
        s_t = torch.FloatTensor(state).unsqueeze(0)
        return self.dynamics.reach_fn(s_t).item()

    # ------------------------------------------------------------------
    # Warmup examples
    # ------------------------------------------------------------------

    def get_warmup_examples(self, num_warmup_samples=200):
        """Sample states with heuristic values for Q-network warmup."""
        states = np.zeros((num_warmup_samples, 13))
        heuristic_v = np.zeros((num_warmup_samples, self.action_space.n))

        for i in range(num_warmup_samples):
            s = self._rng.uniform(self.low, self.high)
            s[6:10] = _random_unit_quaternion(self._rng)
            s = self._wrap_state_np(s)
            l_x = self.target_margin(s)
            g_x = self.safety_margin(s)
            heuristic_v[i, :] = max(l_x, g_x)
            states[i, :] = self._normalize_obs(s)

        return states, heuristic_v

    # ------------------------------------------------------------------
    # Trajectory simulation
    # ------------------------------------------------------------------

    def simulate_one_trajectory(self, q_func, T=100, state=None, theta=None,
                                sample_inside_obs=True, sample_inside_tar=True,
                                toEnd=False):
        """Roll out q_func from an initial state.

        Returns:
            traj:   (L, 13) array
            result: 1=success, -1=failure, 0=unfinished
            minV:   minimum reach-avoid value
            info:   dict with valueList, gxList, lxList
        """
        if state is None:
            self.reset()  # sets self.state; return value is normalized
            state = np.copy(self.state)  # raw state for dynamics
        else:
            state = np.array(state, dtype=np.float64)

        traj = []
        result = 0
        valueList, gxList, lxList = [], [], []

        for t in range(T):
            traj.append(state.copy())

            g_x = self.safety_margin(state)
            l_x = self.target_margin(state)

            if t == 0:
                maxG = g_x
                current = max(l_x, maxG)
                minV = current
            else:
                maxG = max(maxG, g_x)
                current = max(l_x, maxG)
                minV = min(current, minV)

            valueList.append(minV)
            gxList.append(g_x)
            lxList.append(l_x)

            if toEnd:
                if not self._check_within_bounds(state):
                    result = 1
                    break
            else:
                if g_x > 0:
                    result = -1
                    break
                elif l_x <= 0:
                    result = 1
                    break

            # Query policy (Q-network expects normalized observations)
            q_func.eval()
            s_norm = torch.FloatTensor(
                self._normalize_obs(state)).to(self.device).unsqueeze(0)
            with torch.no_grad():
                action_idx = q_func(s_norm).min(dim=1)[1].item()
            control = self.discrete_controls[action_idx]

            s_t = torch.FloatTensor(state).unsqueeze(0)
            c_t = torch.FloatTensor(control).unsqueeze(0)
            dsdt = self.dynamics.dsdt(s_t, c_t, None).squeeze(0).detach().numpy()
            state = state + self.dt * dsdt
            state = self._wrap_state_np(state)

        traj = np.array(traj)
        info = {"valueList": valueList, "gxList": gxList, "lxList": lxList}
        return traj, result, minV, info

    def simulate_trajectories(self, q_func, T=100, num_rnd_traj=None,
                              states=None, toEnd=False):
        """Simulate multiple trajectories for evaluation."""
        assert (num_rnd_traj is not None or states is not None)

        if states is None:
            states = []
            for _ in range(num_rnd_traj):
                self.reset()  # sets self.state
                states.append(np.copy(self.state))  # raw states for dynamics

        trajectories = []
        results = np.empty(len(states), dtype=int)
        minVs = np.empty(len(states), dtype=float)

        for idx, s in enumerate(states):
            traj, result, minV, _ = self.simulate_one_trajectory(
                q_func, T=T, state=s, toEnd=toEnd)
            trajectories.append(traj)
            results[idx] = result
            minVs[idx] = minV

        return trajectories, results, minVs

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def visualize(self, q_func, vmin=-1, vmax=1, nx=31, ny=31,
                  cmap="seismic", addBias=False, labels=None, boolPlot=False):
        """2D value heatmap at fixed state (slice through x-y plane)."""
        matplotlib.use('Agg')
        q_goal = self.dynamics.q_goal.cpu().numpy()
        fixed_vel = [0.0, 0.0, 0.0]
        fixed_q = list(q_goal)
        fixed_omega = [0.0, 0.0, 0.0]

        xs = np.linspace(self.bounds[0, 0], self.bounds[0, 1], nx)
        ys = np.linspace(self.bounds[1, 0], self.bounds[1, 1], ny)
        V = np.zeros((nx, ny))

        for i, x in enumerate(xs):
            for j, y in enumerate(ys):
                state = np.array(
                    [x, y, 0.0] + fixed_vel + fixed_q + fixed_omega)
                s_norm = torch.FloatTensor(
                    self._normalize_obs(state)).to(self.device).unsqueeze(0)
                with torch.no_grad():
                    V[i, j] = q_func(s_norm).min(dim=1)[0].item()

        if boolPlot:
            V = (V <= 0).astype(float)
            cmap, vmin, vmax = "coolwarm", 0, 1

        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        im = ax.imshow(
            V.T, interpolation='none',
            extent=[xs[0], xs[-1], ys[0], ys[-1]],
            origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
        fig.colorbar(im, ax=ax, shrink=0.9)
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.set_title('Q-value (x-y slice, z=0, goal attitude)')
        ax.set_aspect('equal')
        fig.tight_layout()
        return fig

    def get_axes(self):
        """Return axes bounds for plotting compatibility."""
        axes = np.array([
            self.bounds[0, 0], self.bounds[0, 1],
            self.bounds[1, 0], self.bounds[1, 1],
        ])
        aspect = ((self.bounds[0, 1] - self.bounds[0, 0])
                  / (self.bounds[1, 1] - self.bounds[1, 0]))
        return [axes, aspect]

    def get_value(self, q_func, theta=0, nx=31, ny=31, addBias=False):
        """Get 2D value grid for plotting (x-y slice)."""
        q_goal = self.dynamics.q_goal.cpu().numpy()
        xs = np.linspace(self.bounds[0, 0], self.bounds[0, 1], nx)
        ys = np.linspace(self.bounds[1, 0], self.bounds[1, 1], ny)
        V = np.zeros((nx, ny))
        for i, x in enumerate(xs):
            for j, y in enumerate(ys):
                state = np.array(
                    [x, y, 0.0, 0.0, 0.0, 0.0] + list(q_goal) + [0.0, 0.0, 0.0])
                s_norm = torch.FloatTensor(
                    self._normalize_obs(state)).to(self.device).unsqueeze(0)
                with torch.no_grad():
                    V[i, j] = q_func(s_norm).min(dim=1)[0].item()
        return V

    # ------------------------------------------------------------------
    # Observation normalization
    # ------------------------------------------------------------------

    def _normalize_obs(self, state):
        """Map raw state to [-1, 1] using dynamics state_range bounds."""
        return (state - self._obs_center) / self._obs_half_range

    def _denormalize_obs(self, obs):
        """Map normalized observation back to raw state."""
        return obs * self._obs_half_range + self._obs_center

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _wrap_state_np(self, state):
        """Normalize quaternion (canonical q0 >= 0) and clamp velocities."""
        s = state.copy()
        # Normalize quaternion
        q_norm = np.linalg.norm(s[6:10])
        if q_norm > 1e-12:
            s[6:10] /= q_norm
        # Canonical form: q0 >= 0
        if s[6] < 0:
            s[6:10] = -s[6:10]
        # Clamp velocities to dynamics bounds
        s[3:6] = np.clip(s[3:6], self.low[3:6], self.high[3:6])
        s[10:13] = np.clip(s[10:13], self.low[10:13], self.high[10:13])
        return s

    def _check_within_bounds(self, state):
        """Check if state is within the dynamics state range."""
        return np.all(state[0:6] >= self.low[0:6]) and np.all(state[0:6] <= self.high[0:6])

    def render(self):
        pass

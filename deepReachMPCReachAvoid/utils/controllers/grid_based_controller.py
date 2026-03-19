"""Grid-based 4D+2D HJ reach-avoid controller wrapper.

Wraps the ComboController from gridBased6DImplementation to provide the
same ``simulate_docking()`` interface as BRATController, MPCController,
and MPCTerminalController for unified comparison.

Dependencies (beyond the deepReach stack):
    jax, jax[cuda], hj_reachability, cvxpy, scipy
"""

import os
import sys
import time

import numpy as np
from scipy.interpolate import interpn

# ---------------------------------------------------------------------------
# Import ComboController from the grid-based implementation.
# We temporarily add its directory to sys.path, then remove it to avoid
# shadowing the deepReach ``utils/`` package.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(__file__), '..', '..', '..'))
_GRID_DIR = os.path.join(_PROJECT_ROOT, 'gridBased6DImplementation')

sys.path.insert(0, _GRID_DIR)
from ComboControl import ComboController as _ComboController  # noqa: E402
sys.path.remove(_GRID_DIR)


# ---------------------------------------------------------------------------
# Minimal dynamics proxy for plotting compatibility
# ---------------------------------------------------------------------------
class _DynamicsProxy:
    """Exposes the geometry attributes that controller_animation.py needs."""
    pass


def _build_dynamics_proxy(combo):
    """Populate a proxy from ComboController attributes."""
    d = _DynamicsProxy()
    d.w_t = combo.w_t
    d.h_t = combo.h_t
    d.post_hw_x = combo.post_hw_x
    d.post_length = combo.post_length
    d.w_c = combo.w_c
    d.h_c = combo.h_c
    d.eps_p = combo.eps_p

    # Goal band (identical formula to Docking6D dynamics)
    cb = np.sqrt(combo.w_c**2 + combo.h_c**2) / 2
    goal_clearance = -0.007
    goal_band_height = 0.5
    d.goal_y_max = -(combo.post_length + cb + goal_clearance)
    d.goal_y_min = d.goal_y_max - goal_band_height
    return d


# ---------------------------------------------------------------------------
# CW dynamics builder
# ---------------------------------------------------------------------------
def _build_cw_dynamics(combo):
    """Return a CW dynamics function derived from ComboController params."""
    n = float(combo.docking_problem_4D.n)
    mc = float(combo.mc)
    jc = float(combo.docking_problem_2D.jc)

    def dynamics(state, control):
        px, py, vx, vy, _theta, omega = state
        ux, uy, utheta = control
        return np.array([
            vx,
            vy,
            3 * n**2 * px + 2 * n * vy + ux / mc,
            -2 * n * vx + uy / mc,
            omega,
            utheta / jc,
        ])
    return dynamics


# ---------------------------------------------------------------------------
# GridBasedController
# ---------------------------------------------------------------------------
class GridBasedController:
    """Grid-based HJ reach-avoid controller for unified comparison.

    Uses the ComboController's 4D+2D decomposed value functions for
    optimal bang-bang control from the BRAT gradient, with CW dynamics
    integration and deepReach-compatible docking/collision criteria.

    The BRAT time horizon is set equal to ``max_sim_time`` so that
    any IC inside the BRAT is guaranteed reachable within the
    simulation budget.

    Args:
        dt: Integration / control timestep (s).
        max_sim_time: Simulation time budget (s). The HJ backward
            horizon is set to ``-max_sim_time``.
        cache_dir: Directory for caching solved HJ value functions.
            ``None`` forces a fresh solve every run.
        filter_mode: Control strategy for ComboController.
            ``None`` = optimal bang-bang from BRAT gradient (default),
            1 = least-restrictive, 2 = smooth-blend QP, 3 = nominal LQR.
    """

    def __init__(self, dt=0.1, max_sim_time=30.0, cache_dir=None,
                 filter_mode=None):
        self.dt = dt

        # JAX GPU diagnostic
        import jax
        print(f"  JAX backend: {jax.default_backend()}  "
              f"devices: {jax.devices()}")

        # Instantiate the grid-based HJ controller with BRAT horizon = max_sim_time
        self.combo = _ComboController(
            mc=200.0, orbit_alt=400,
            post_hw_x=0.6, post_length=0.2,
            w_t=6, h_t=3, w_c=1.0, h_c=1.0,
            eps_p=0.1, eps_v=0.1, eps_theta=0.04, eps_omega=0.05,
            u_bar_4D=20.0, u_bar_2D=1.5,
            d_bar_4D=0.0, d_bar_2D=0.0,
            final_time=-max_sim_time, dt=dt,
            cache_dir=cache_dir, filter_mode=filter_mode,
        )

        # CW dynamics closure (parametrically consistent with ComboController)
        self._cw_dynamics = _build_cw_dynamics(self.combo)

        # Plotting-compatible dynamics proxy
        self.dynamics = _build_dynamics_proxy(self.combo)

        # Docking tolerances (from ComboController, verified == deepReach)
        self._eps_p = self.combo.eps_p
        self._eps_v = self.combo.eps_v
        self._eps_theta = self.combo.eps_theta
        self._eps_omega = self.combo.eps_omega
        self._goal_y_min = self.dynamics.goal_y_min
        self._goal_y_max = self.dynamics.goal_y_max

        # Collision geometry
        self._hw = self.combo.w_c / 2.0
        self._hh = self.combo.h_c / 2.0
        self._body_half_w = self.combo.w_t / 2.0
        self._body_h = self.combo.h_t
        self._post_hw = self.combo.post_hw_x
        self._post_len = self.combo.post_length

        # Pre-compute interpolation coordinates for batch BRAT queries
        self._coords_4d = [np.array(v) for v
                           in self.combo.grid_4D.coordinate_vectors]
        self._coords_2d = [np.array(v) for v
                           in self.combo.grid_2D.coordinate_vectors]

        # Convert full value arrays from JAX DeviceArrays to numpy once.
        # This eliminates GPU→CPU transfers during simulation (the main
        # bottleneck when values live on GPU).
        print("  Converting value arrays to numpy ...")
        self._np_values_4d = np.asarray(self.combo.values_4D)  # (T, N1, N2, N3, N4)
        self._np_values_2d = np.asarray(self.combo.values_2D)  # (T, M1, M2)
        self._values_4d_terminal = self._np_values_4d[-1]
        self._values_2d_terminal = self._np_values_2d[-1]

        # Grid spacings for numpy gradient computation
        self._dx_4d = [float(c[1] - c[0]) for c in self._coords_4d]
        self._dx_2d = [float(c[1] - c[0]) for c in self._coords_2d]

        self._u_bar_4d = float(self.combo.u_bar_4D)
        self._u_bar_2d = float(self.combo.u_bar_2D)

    # ------------------------------------------------------------------
    # Batch BRAT value query (for IC filtering)
    # ------------------------------------------------------------------
    def get_brat_values_batch(self, states):
        """Query the grid-based BRAT value for a batch of 6D states.

        A state is inside the BRAT when **both** the 4D translational
        and 2D rotational sub-values are <= 0.  The combined value is
        ``max(V_4D, V_2D)`` so that combined <= 0  iff  both <= 0.

        Args:
            states: (N, 6) numpy array of states
                    [px, py, vx, vy, theta, omega].

        Returns:
            (N,) numpy array of combined BRAT values.
            Entries <= 0 are inside the BRAT.
        """
        v_4d = interpn(self._coords_4d, self._values_4d_terminal,
                       states[:, :4], method='linear',
                       bounds_error=False, fill_value=np.inf)
        v_2d = interpn(self._coords_2d, self._values_2d_terminal,
                       states[:, 4:6], method='linear',
                       bounds_error=False, fill_value=np.inf)
        return np.maximum(v_4d, v_2d)

    # ------------------------------------------------------------------
    # Min-time BRT index search (numpy-only, no JAX transfers)
    # ------------------------------------------------------------------
    def _min_time_idx_4d(self, s_4d):
        """Find smallest time index where interpolated V_4D <= 0."""
        cell_idx = []
        frac = []
        for dim in range(4):
            grid = self._coords_4d[dim]
            x = float(s_4d[dim])
            if x <= grid[0]:
                cell_idx.append(0); frac.append(0.0)
            elif x >= grid[-1]:
                cell_idx.append(len(grid) - 2); frac.append(1.0)
            else:
                i = int(np.searchsorted(grid, x)) - 1
                i = max(0, min(i, len(grid) - 2))
                cell_idx.append(i)
                frac.append(float((x - grid[i]) / (grid[i + 1] - grid[i])))

        result = np.zeros(self._np_values_4d.shape[0])
        for d0 in range(2):
            w0 = frac[0] if d0 else (1.0 - frac[0])
            for d1 in range(2):
                w01 = w0 * (frac[1] if d1 else (1.0 - frac[1]))
                for d2 in range(2):
                    w012 = w01 * (frac[2] if d2 else (1.0 - frac[2]))
                    for d3 in range(2):
                        w = w012 * (frac[3] if d3 else (1.0 - frac[3]))
                        corner = self._np_values_4d[
                            :, cell_idx[0]+d0, cell_idx[1]+d1,
                            cell_idx[2]+d2, cell_idx[3]+d3]
                        result += w * corner

        neg_mask = result <= 0
        if np.any(neg_mask):
            return int(np.argmax(neg_mask))
        return int(np.argmin(result))

    def _min_time_idx_2d(self, s_2d):
        """Find smallest time index where interpolated V_2D <= 0."""
        cell_idx = []
        frac = []
        for dim in range(2):
            grid = self._coords_2d[dim]
            x = float(s_2d[dim])
            if x <= grid[0]:
                cell_idx.append(0); frac.append(0.0)
            elif x >= grid[-1]:
                cell_idx.append(len(grid) - 2); frac.append(1.0)
            else:
                i = int(np.searchsorted(grid, x)) - 1
                i = max(0, min(i, len(grid) - 2))
                cell_idx.append(i)
                frac.append(float((x - grid[i]) / (grid[i + 1] - grid[i])))

        result = np.zeros(self._np_values_2d.shape[0])
        for d0 in range(2):
            w0 = frac[0] if d0 else (1.0 - frac[0])
            for d1 in range(2):
                w = w0 * (frac[1] if d1 else (1.0 - frac[1]))
                corner = self._np_values_2d[
                    :, cell_idx[0]+d0, cell_idx[1]+d1]
                result += w * corner

        neg_mask = result <= 0
        if np.any(neg_mask):
            return int(np.argmax(neg_mask))
        return int(np.argmin(result))

    # ------------------------------------------------------------------
    # Gradient at a point via numpy central differences
    # ------------------------------------------------------------------
    def _grad_at_point_4d(self, s_4d, tidx, grad_field_cache):
        """Compute 4D gradient at a point, caching the gradient field by tidx."""
        cached_tidx, cached_fields = grad_field_cache
        if cached_tidx != tidx:
            vals = self._np_values_4d[tidx]
            fields = np.gradient(vals, *self._dx_4d)
            grad_field_cache[0] = tidx
            grad_field_cache[1] = fields
        else:
            fields = cached_fields
        pt = np.atleast_2d(s_4d)
        return [interpn(self._coords_4d, f, pt, method='linear',
                        bounds_error=False, fill_value=0.0).item()
                for f in fields]

    def _grad_at_point_2d(self, s_2d, tidx, grad_field_cache):
        """Compute 2D gradient at a point, caching the gradient field by tidx."""
        cached_tidx, cached_fields = grad_field_cache
        if cached_tidx != tidx:
            vals = self._np_values_2d[tidx]
            fields = np.gradient(vals, *self._dx_2d)
            grad_field_cache[0] = tidx
            grad_field_cache[1] = fields
        else:
            fields = cached_fields
        pt = np.atleast_2d(s_2d)
        return [interpn(self._coords_2d, f, pt, method='linear',
                        bounds_error=False, fill_value=0.0).item()
                for f in fields]

    # ------------------------------------------------------------------
    # Simulation interface
    # ------------------------------------------------------------------
    def simulate_docking(self, initial_state, max_sim_time):
        """Run grid-based docking from *initial_state*.

        Returns a result dict compatible with the deepReach comparison
        framework (same keys as BRATController / MPCController).
        """
        self.combo.reset()
        t_wall_start = time.perf_counter()

        state = np.array(initial_state, dtype=np.float64)
        num_steps = int(max_sim_time / self.dt)

        state_history = [state.copy()]
        control_history = []
        value_history = []
        time_history = [0.0]

        docked = False
        collision = False

        # Gradient field caches: [tidx, list_of_grad_arrays]
        # Recomputed only when the min-time index changes.
        cache_4d = [None, None]
        cache_2d = [None, None]

        for step in range(num_steps):
            s_4d = state[:4]
            s_2d = state[4:6]

            # Min-time search: tightest BRT boundary for each subsystem
            tidx_4d = self._min_time_idx_4d(s_4d)
            tidx_2d = self._min_time_idx_2d(s_2d)

            # Gradient at the min-time BRT boundary (cached by tidx)
            g4 = self._grad_at_point_4d(s_4d, tidx_4d, cache_4d)
            g2 = self._grad_at_point_2d(s_2d, tidx_2d, cache_2d)

            # Bang-bang optimal control from gradient sign
            ux = -self._u_bar_4d if g4[2] > 0 else self._u_bar_4d
            uy = -self._u_bar_4d if g4[3] > 0 else self._u_bar_4d
            ut = -self._u_bar_2d if g2[1] > 0 else self._u_bar_2d
            control = np.array([ux, uy, ut])
            control_history.append(control)

            # Combined decomposed value at terminal time
            v_4d = interpn(self._coords_4d, self._values_4d_terminal,
                           np.atleast_2d(s_4d), method='linear',
                           bounds_error=False, fill_value=np.inf).item()
            v_2d = interpn(self._coords_2d, self._values_2d_terminal,
                           np.atleast_2d(s_2d), method='linear',
                           bounds_error=False, fill_value=np.inf).item()
            value_history.append(max(v_4d, v_2d))

            # Integrate CW dynamics
            state = state + self.dt * self._cw_dynamics(state, control)
            state[4] = np.arctan2(np.sin(state[4]), np.cos(state[4]))

            state_history.append(state.copy())
            time_history.append((step + 1) * self.dt)

            # Termination checks (collision first, matching deepReach order)
            if self._check_collision(state):
                collision = True
                break
            if self._check_docked(state):
                docked = True
                break

        # Final value for the terminal state
        v_4d = interpn(self._coords_4d, self._values_4d_terminal,
                       np.atleast_2d(state[:4]), method='linear',
                       bounds_error=False, fill_value=np.inf).item()
        v_2d = interpn(self._coords_2d, self._values_2d_terminal,
                       np.atleast_2d(state[4:6]), method='linear',
                       bounds_error=False, fill_value=np.inf).item()
        value_history.append(max(v_4d, v_2d))

        wall_time = time.perf_counter() - t_wall_start

        controls_arr = np.array(control_history) if control_history else np.zeros((0, 3))
        control_effort = float(
            np.sum(np.linalg.norm(controls_arr, axis=-1)) * self.dt
        ) if len(control_history) > 0 else 0.0

        return {
            'trajectory':         np.array(state_history),
            'controls':           controls_arr,
            'values':             np.array(value_history),
            'times':              np.array(time_history),
            'success':            docked and not collision,
            'docked':             docked,
            'collision':          collision,
            'final_state':        state.copy(),
            'controller_type':    'grid_based',
            'control_effort':     control_effort,
            'wall_time':          wall_time,
            'safety_filter_mode': 0,
            'safety_filter_log':  [],
            'n_clipped_steps':    0,
        }

    # ------------------------------------------------------------------
    # Docking check (matches BRATController._check_docked exactly)
    # ------------------------------------------------------------------
    def _check_docked(self, state):
        px, py, vx, vy, theta, omega = state
        x_ok = np.abs(px) <= self._eps_p
        y_ok = self._goal_y_min <= py <= self._goal_y_max
        vel_ok = np.sqrt(vx**2 + vy**2) <= self._eps_v
        theta_diff = np.abs(
            np.arctan2(np.sin(theta - np.pi / 2), np.cos(theta - np.pi / 2)))
        theta_ok = theta_diff <= self._eps_theta
        omega_ok = np.abs(omega) <= self._eps_omega
        return x_ok and y_ok and vel_ok and theta_ok and omega_ok

    # ------------------------------------------------------------------
    # Collision check (matches Docking6D.check_collision_oriented exactly)
    # ------------------------------------------------------------------
    def _check_collision(self, state):
        px, py, theta = float(state[0]), float(state[1]), float(state[4])
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        hw, hh = self._hw, self._hh

        corners = [
            (px + hw * cos_t - hh * sin_t, py + hw * sin_t + hh * cos_t),
            (px - hw * cos_t - hh * sin_t, py - hw * sin_t + hh * cos_t),
            (px - hw * cos_t + hh * sin_t, py - hw * sin_t - hh * cos_t),
            (px + hw * cos_t + hh * sin_t, py + hw * sin_t - hh * cos_t),
        ]
        for cx, cy in corners:
            # Target body
            if abs(cx) <= self._body_half_w and 0.0 <= cy <= self._body_h:
                return True
            # Docking post
            if abs(cx) <= self._post_hw and -self._post_len <= cy <= 0.0:
                return True
        return False

import jax
import jax.numpy as jnp
import numpy as np
import os
import cvxpy as cp
import hj_reachability as hj
import matplotlib.pyplot as plt
from hj_reachability import dynamics
from hj_reachability import sets
from scipy.interpolate import interpn

import hashlib

class Docking_translational(dynamics.ControlAndDisturbanceAffineDynamics):
    def __init__(self, mc, orbit_alt, u_bar = 20.0, d_bar = 0.01):
        # Mass and orbital parameters
        self.mc = mc  # Mass of chaser spacecraft (kg)
        self.orbit_alt = orbit_alt * 1e3  # Altitude in meters

        # Control and disturbance bounds
        self.u_bar = u_bar
        self.d_bar = d_bar
        self.u_min = jnp.array([-u_bar, -u_bar])
        self.u_max = jnp.array([u_bar, u_bar])
        self.d_min = jnp.array([-d_bar, -d_bar])
        self.d_max = jnp.array([d_bar, d_bar])

        # Derived orbital parameters
        self.n = self._mean_motion()  # Mean motion (rad/s)

        # Call base class constructor
        control_mode = 'min'         # Controller minimizes value
        disturbance_mode = 'max'     # Disturbance maximizes value
        control_space = sets.Box(self.u_min, self.u_max)
        disturbance_space = sets.Box(self.d_min, self.d_max)
        super().__init__(control_mode, disturbance_mode, control_space, disturbance_space)
    
    def _mean_motion(self):
        mu = 3.986004418e14  # Earth's gravitational parameter (m^3/s^2)
        r_earth = 6371e3     # Earth's radius (m)
        r = r_earth + self.orbit_alt
        return jnp.sqrt(mu / (r**3))
        
    def open_loop_dynamics(self, state, time=None):
        """Dynamics without control or disturbance inputs."""
        px, py, vx, vy = state
        dp_x = vx
        dp_y = vy
        dv_x = 3 * (self.n**2) * px + 2 * self.n * vy
        dv_y = -2 * self.n * vx
        return jnp.stack([dp_x, dp_y, dv_x, dv_y]) 
    
    def control_jacobian(self, state, time=None):
        """Jacobian of dynamics w.r.t. control input u = [u_x, u_y]."""
        return jnp.array([
            [0.0,       0.0],
            [0.0,       0.0],
            [1.0/self.mc, 0.0],
            [0.0,       1.0/self.mc]
        ])
    
    def disturbance_jacobian(self, state, time=None):
        """Jacobian of dynamics w.r.t. disturbance d = [d_x, d_y]."""
        # Disturbance enters same channels as control but without mass scaling
        return jnp.array([
            [0.0, 0.0],
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0]
        ])

class Docking_rotational(dynamics.ControlAndDisturbanceAffineDynamics):
    def __init__(self, mc, u_bar = 1.5, d_bar = 0.01):
        # Defineing dynamic parameters
        self.u_bar = u_bar  # Maximum control input
        self.d_bar = d_bar  # Maximum disturbance
        
        # Define Chaser spacecraft (Planar)
        self.mc = mc    # Mass of chaser spacecraft (kg)
        self.w_c = 1.0  # width of chaser spacecraft (m) (along x-axis)
        self.h_c = 1.0  # height of chaser spacecraft (m) (along y-axis)
        
        # Calculate derived parameters
        self.jc = self.moment_of_inertia()    # Moment of inertia of chaser spacecraft (kg*m^2)
        
        # Define control and disturbance spaces
        control_mode = 'min'
        disturbance_mode = 'max'
        control_space = sets.Box(jnp.array([-self.u_bar]), jnp.array([self.u_bar]))
        disturbance_space = sets.Box(jnp.array([-self.d_bar]), jnp.array([self.d_bar]))
        super().__init__(control_mode, disturbance_mode, control_space, disturbance_space)
    
    def moment_of_inertia(self):
        """Calculate the moment of inertia for the chaser spacecraft."""
        # Assuming a rectangular shape for the chaser spacecraft
        return 1.0/12.0 * (self.mc * (self.w_c**2 + self.h_c**2)) 
    
    def open_loop_dynamics(self, state, time):
        theta, theta_dot = state
        return jnp.array([theta_dot,
                          0.0]) 
    
    def control_jacobian(self, state, time):
        return jnp.array([[0.0],
                          [1/self.jc]])
    
    def disturbance_jacobian(self, state, time):
        return jnp.array([[0.0],
                          [1.0/self.jc]])

class ComboController:
    def __init__(self, mc=200.0, orbit_alt=400, post_hw_x=0.6, post_length=0.2,
                 w_t=6, h_t=3, w_c=1.0, h_c=1.0,
                 eps_p=0.1, eps_v=0.1, eps_theta=0.04, eps_omega=0.05,
                 u_bar_4D=20.0, u_bar_2D=1.5,
                 d_bar_4D=0.0, d_bar_2D=0.0,
                 final_time=-30.0, dt=0.1,
                 grid_resolution_4D=(51, 51, 31, 31),
                 grid_resolution_2D=(361, 141),
                 cache_dir=None, filter_mode=2):
        """Grid-based 4D+2D decomposed HJ reach-avoid controller.

        Args:
            mc: Chaser mass (kg).
            orbit_alt: Orbital altitude (km).
            post_hw_x: Docking post half-width in x (m).
            post_length: How far the docking post extends in -y (m).
            w_t, h_t: Target spacecraft width/height (m).
            w_c, h_c: Chaser spacecraft width/height (m).
            eps_p, eps_v: Translational position/velocity tolerances (m, m/s).
            eps_theta, eps_omega: Rotational position/velocity tolerances (rad, rad/s).
            u_bar_4D, u_bar_2D: Translational/rotational control bounds (N, N·m).
            d_bar_4D, d_bar_2D: Translational/rotational disturbance bounds.
            final_time: Backward time horizon (negative seconds, e.g. -30).
            dt: Time step for HJ solver (s).
            grid_resolution_4D: 4-tuple grid resolution for translational subsystem.
            grid_resolution_2D: 2-tuple grid resolution for rotational subsystem.
            cache_dir: Directory for caching solved value functions. None = no caching.
            filter_mode: 1=least-restrictive, 2=smooth-blend QP, 3=nominal, None=optimal-safety.
        """
        self.reset()
        # Define required parameters
        self.final_time = final_time  
        
        # 1: for least restrictive safety filter
        # 2: for smooth blending filter
        # 3: for nominal controller 
        # None: for optimal safety controller only
        self.FILTER = filter_mode
        
        # Define constants
        self.mc = mc  # Mass of chaser spacecraft (kg)
        self.orbit_alt = orbit_alt  # Orbital altitude (km)
        
        # Define target spacecraft (Planar)
        self.w_t = w_t  # width of target spacecraft (m) (along x-axis)
        self.h_t = h_t  # height of target spacecraft (m) (along y-axis)
        self.post_hw_x = post_hw_x    # docking post half-width in x (m)
        self.post_length = post_length  # docking post extent in -y (m)
        
        # Define the body of the chaser spacecraft 
        self.w_c = w_c  # width of chaser spacecraft (m) (along x-axis)
        self.h_c = h_c  # height of chaser spacecraft (m) (along y-axis)
                
        # Define docking parameters
        self.eps_p = eps_p # Position tolerance for docking (m)
        self.eps_v = eps_v # Velocity tolerance for docking (m/s)
        self.eps_theta = eps_theta # Angular position tolerance for docking (rad)
        self.eps_omega = eps_omega # Angular velocity tolerance for docking (rad/s)

        # Define 4D system
        self.state_domain_4D = hj.sets.Box(lo=jnp.array([-15.0 , -15.0, -1.5, -1.5]), hi=jnp.array([15.0, 15.0, 1.5, 1.5]))
        self.grid_resolution_4D = grid_resolution_4D
        self.grid_4D = hj.Grid.from_lattice_parameters_and_boundary_conditions(self.state_domain_4D, self.grid_resolution_4D)

        # Define 2D system
        self.state_domain_2D = hj.sets.Box(lo = jnp.array([-np.pi, -2.0]), hi = jnp.array([np.pi, 2.0]))
        self.grid_resolution_2D = grid_resolution_2D
        self.periodic_dims = [0]
        self.grid_2D = hj.Grid.from_lattice_parameters_and_boundary_conditions(
            self.state_domain_2D, 
            self.grid_resolution_2D,
            periodic_dims=self.periodic_dims)
    
        # Define target and failure sets for 4D system
        self.target_values_4D = self.target_set_4D(self.grid_4D.states) 
        self.failure_values_4D = self.failure_set(self.grid_4D.states)
        self.terminal_values_4D = jnp.maximum(self.target_values_4D, -self.failure_values_4D)
        
        # Define target and failure sets for 2D system
        self.target_values_2D = self.target_set_2D(self.grid_2D.states)
        self.terminal_values_2D = jnp.maximum(self.target_values_2D, -jnp.inf)
        
        # Define the 4D and 2D dynamics
        self.docking_problem_4D = Docking_translational(self.mc, self.orbit_alt, u_bar=u_bar_4D, d_bar=d_bar_4D)
        self.docking_problem_2D = Docking_rotational(self.mc, u_bar=u_bar_2D, d_bar=d_bar_2D)
        
        # Define the solver settings
        self.solver_settings_4D = hj.SolverSettings.with_accuracy("very_high",
                                                          hamiltonian_postprocessor=lambda x: jnp.minimum(x, 0), 
                                                          value_postprocessor=lambda t, x: jnp.maximum(x, -self.failure_values_4D))
        self.solver_settings_2D = hj.SolverSettings.with_accuracy("very_high",
                                                          hamiltonian_postprocessor=lambda x: jnp.minimum(x, 0),
                                                          value_postprocessor=lambda t, x: jnp.maximum(x, -jnp.inf))
        
        # Solve for value functions (with optional caching)
        self.dt = dt
        self.nt = int(abs(self.final_time / self.dt)) + 1
        self.times = jnp.linspace(0, self.final_time, self.nt, endpoint=True)

        # Define control bounds (used by optimal / QP controllers and cache fingerprint)
        self.u_bar_4D = u_bar_4D
        self.u_bar_2D = u_bar_2D

        loaded = self._try_load_cache(cache_dir)
        if not loaded:
            print("Solving 4D HJ PDE ...")
            self.values_4D = hj.solve(self.solver_settings_4D, self.docking_problem_4D, self.grid_4D, self.times, self.terminal_values_4D)
            print("Solving 2D HJ PDE ...")
            self.values_2D = hj.solve(self.solver_settings_2D, self.docking_problem_2D, self.grid_2D, self.times, self.terminal_values_2D)
            self._save_cache(cache_dir)

        # Obtain values from 4D value function
        self.values_4D_T = self.values_4D[-1]
        self.grads_4D_T = self.grid_4D.grad_values(self.values_4D_T, self.solver_settings_4D.upwind_scheme)
        
        # Obtain values from 2D value function
        self.values_2D_T = self.values_2D[-1]
        self.grads_2D_T = self.grid_2D.grad_values(self.values_2D_T, self.solver_settings_2D.upwind_scheme)

    # ------------------------------------------------------------------ #
    #  Caching helpers                                                     #
    # ------------------------------------------------------------------ #
    def _cache_fingerprint(self):
        """Return a deterministic hash of every parameter that affects the
        solved value functions, so cache files are invalidated when params change."""
        key_parts = [
            self.mc, self.orbit_alt, self.post_hw_x, self.post_length,
            self.w_t, self.h_t, self.w_c, self.h_c,
            self.eps_p, self.eps_v, self.eps_theta, self.eps_omega,
            self.u_bar_4D, self.u_bar_2D,
            float(self.docking_problem_4D.d_bar),
            float(self.docking_problem_2D.d_bar),
            self.final_time, self.dt,
            self.grid_resolution_4D, self.grid_resolution_2D,
            tuple(float(x) for x in self.state_domain_4D.lo),
            tuple(float(x) for x in self.state_domain_4D.hi),
            tuple(float(x) for x in self.state_domain_2D.lo),
            tuple(float(x) for x in self.state_domain_2D.hi),
            0.143, 0.2,  # goal_clearance, goal_band_height (for cache invalidation)
        ]
        raw = str(key_parts).encode()
        return hashlib.sha256(raw).hexdigest()[:16]

    def _try_load_cache(self, cache_dir):
        """Attempt to load cached value arrays.  Returns True on success."""
        if cache_dir is None:
            return False
        fp = self._cache_fingerprint()
        path_4d = os.path.join(cache_dir, f'values_4D_{fp}.npy')
        path_2d = os.path.join(cache_dir, f'values_2D_{fp}.npy')
        if os.path.isfile(path_4d) and os.path.isfile(path_2d):
            print(f"Loading cached grid values  (fingerprint={fp}) ...")
            self.values_4D = jnp.array(np.load(path_4d))
            self.values_2D = jnp.array(np.load(path_2d))
            print("  Cache loaded successfully.")
            return True
        return False

    def _save_cache(self, cache_dir):
        """Save solved value arrays to cache directory."""
        if cache_dir is None:
            return
        os.makedirs(cache_dir, exist_ok=True)
        fp = self._cache_fingerprint()
        path_4d = os.path.join(cache_dir, f'values_4D_{fp}.npy')
        path_2d = os.path.join(cache_dir, f'values_2D_{fp}.npy')
        np.save(path_4d, np.array(self.values_4D))
        np.save(path_2d, np.array(self.values_2D))
        print(f"Grid values cached to {cache_dir}  (fingerprint={fp})")

    def reset(self):
        self.s_history = []
        self.t_history = []
        self.u_history = []    
        
    def target_set_4D(self, state):
        """Signed distance <= 0 if within docking position/velocity tolerances.

        Position: |px| <= eps_p AND py inside goal band [goal_y_min, goal_y_max].
        Velocity: L2 norm.
        """
        px = state[..., 0]
        py = state[..., 1]
        vx = state[..., 2]
        vy = state[..., 3]

        cb = np.sqrt(self.w_c**2 + self.h_c**2) / 2
        goal_clearance = 0.143
        goal_band_height = 0.2
        goal_y_max = -(self.post_length + cb + goal_clearance)
        goal_y_min = goal_y_max - goal_band_height

        x_dist = jnp.abs(px) - self.eps_p
        y_dist = jnp.maximum(goal_y_min - py, py - goal_y_max)
        pos_dist = jnp.maximum(x_dist, y_dist)

        vel_dist = jnp.sqrt(vx**2 + vy**2 + 1e-8) - self.eps_v

        return jnp.maximum(pos_dist, vel_dist)
    
    def target_set_2D(self, state):
        """Signed distance <= 0 if within docking position,velocity,angular tolerances."""
        theta = state[..., 0]
        omega = state[..., 1]
        # Wrapped angular distance to goal orientation (handles periodicity)
        theta_error = jnp.arctan2(jnp.sin(theta - np.pi/2), jnp.cos(theta - np.pi/2))
        gi = jnp.stack([
            jnp.abs(theta_error) - self.eps_theta,
            jnp.abs(omega) - self.eps_omega],
            axis=-1)
        return jnp.max(gi, axis=-1)
            
    def failure_set(self, state):
        """Obstacle = target body + docking post (2D simplification of 13D).

        Returns: negative inside obstacle, positive outside (safe).
        """
        px = state[..., 0]
        py = state[..., 1]

        self.chaser_buffer = np.sqrt(self.w_c**2 + self.h_c**2) / 2
        cb = self.chaser_buffer

        # Body SDF (inflated rectangle)
        s_body = jnp.maximum(
            jnp.abs(px) - (self.w_t / 2.0 + cb),
            jnp.maximum(-(py + cb), py - (self.h_t + cb))
        )

        # Post SDF (inflated rectangle)
        s_post = jnp.maximum(
            jnp.abs(px) - (self.post_hw_x + cb),
            jnp.maximum(-(py + self.post_length + cb), py - cb)
        )

        # Union of body + post
        return jnp.minimum(s_body, s_post)
    
    def value_at_state_4D(self, x, t):
        time_values = self.values_4D[t]
        v = interpn(
            ([np.array(v) for v in self.grid_4D.coordinate_vectors]), 
            np.array(time_values),                      
            np.atleast_2d(x),                            
            method='linear',
            bounds_error=False,
            fill_value=np.inf)
        return v.item()
    
    def value_at_state_2D(self, x, t):
        time_values = self.values_2D[t]
        v = interpn(
            ([np.array(v) for v in self.grid_2D.coordinate_vectors]), 
            np.array(time_values),                      
            np.atleast_2d(x),                            
            method='linear',
            bounds_error=False,
            fill_value=np.inf
        )
        return v.item()

    def grad_at_state_4D(self, x, t):
        time_values = self.values_4D[t]
        time_grad = self.grid_4D.grad_values(time_values, self.solver_settings_4D.upwind_scheme)
        return np.array([
            interpn(
                [np.array(v) for v in self.grid_4D.coordinate_vectors],
                np.array(time_grad[..., i]),
                np.atleast_2d(x),
                method='linear',
                bounds_error=False,
                fill_value=np.inf).item()
            for i in range(4)
        ])

    def grad_at_state_2D(self, x, t):
        time_values = self.values_2D[t]
        time_grad = self.grid_2D.grad_values(time_values, self.solver_settings_2D.upwind_scheme)
        return np.array([
            interpn(
                [np.array(v) for v in self.grid_2D.coordinate_vectors],
                np.array(time_grad[..., i]),
                np.atleast_2d(x),
                method='linear',
                bounds_error=False,
                fill_value=np.inf).item()
            for i in range(2)
        ])

    def _find_min_time_index_4D(self, s_4D):
        """Find minimum time index where V_4D(s_4D, t) <= 0.

        Sweeps from smallest horizon (t=0) to largest via vectorized
        linear interpolation across all time slices at once.
        Returns first index where V <= 0, or argmin index if V > 0 everywhere.
        """
        coords = [np.array(v) for v in self.grid_4D.coordinate_vectors]
        cell_idx = []
        frac = []
        for dim in range(4):
            grid = coords[dim]
            x = float(s_4D[dim])
            if x <= grid[0]:
                cell_idx.append(0); frac.append(0.0)
            elif x >= grid[-1]:
                cell_idx.append(len(grid) - 2); frac.append(1.0)
            else:
                i = int(np.searchsorted(grid, x)) - 1
                i = max(0, min(i, len(grid) - 2))
                cell_idx.append(i)
                frac.append(float((x - grid[i]) / (grid[i + 1] - grid[i])))

        result = np.zeros(len(self.times))
        for d0 in range(2):
            w0 = frac[0] if d0 else (1.0 - frac[0])
            for d1 in range(2):
                w01 = w0 * (frac[1] if d1 else (1.0 - frac[1]))
                for d2 in range(2):
                    w012 = w01 * (frac[2] if d2 else (1.0 - frac[2]))
                    for d3 in range(2):
                        w = w012 * (frac[3] if d3 else (1.0 - frac[3]))
                        corner = np.asarray(self.values_4D[
                            :, cell_idx[0]+d0, cell_idx[1]+d1,
                            cell_idx[2]+d2, cell_idx[3]+d3])
                        result += w * corner

        neg_mask = result <= 0
        if np.any(neg_mask):
            return int(np.argmax(neg_mask))
        return int(np.argmin(result))

    def _find_min_time_index_2D(self, s_2D):
        """Find minimum time index where V_2D(s_2D, t) <= 0."""
        coords = [np.array(v) for v in self.grid_2D.coordinate_vectors]
        cell_idx = []
        frac = []
        for dim in range(2):
            grid = coords[dim]
            x = float(s_2D[dim])
            if x <= grid[0]:
                cell_idx.append(0); frac.append(0.0)
            elif x >= grid[-1]:
                cell_idx.append(len(grid) - 2); frac.append(1.0)
            else:
                i = int(np.searchsorted(grid, x)) - 1
                i = max(0, min(i, len(grid) - 2))
                cell_idx.append(i)
                frac.append(float((x - grid[i]) / (grid[i + 1] - grid[i])))

        result = np.zeros(len(self.times))
        for d0 in range(2):
            w0 = frac[0] if d0 else (1.0 - frac[0])
            for d1 in range(2):
                w = w0 * (frac[1] if d1 else (1.0 - frac[1]))
                corner = np.asarray(self.values_2D[
                    :, cell_idx[0]+d0, cell_idx[1]+d1])
                result += w * corner

        neg_mask = result <= 0
        if np.any(neg_mask):
            return int(np.argmax(neg_mask))
        return int(np.argmin(result))

    def u_fn(self, s, t):
        self.s_history.append(s)
        self.t_history.append(t)
        
        # Define state vectors for 4D and 2D systems
        s_4D = np.array([s[0], s[1], s[2], s[3]])
        s_2D = np.array([s[4], s[5]])
        
        # Min-time search: find tightest BRT boundary for each subsystem
        tidx_4D = self._find_min_time_index_4D(s_4D)
        tidx_2D = self._find_min_time_index_2D(s_2D)

        if self.FILTER == 1:
            value_4D = self.value_at_state_4D(s_4D, tidx_4D)
            if value_4D > -0.2:
                ux, uy = self.U_4D(s_4D, tidx_4D)
                u_theta = self.U_2D(s_2D, tidx_2D)
                u_safe = np.array([ux, uy, u_theta])
                u = u_safe
            else:
                u = self.U_nom(s, t)
        elif self.FILTER == 2:
            u_QP_4D = self.qp_controller_4D(s, tidx_4D, gamma=0.2)
            u_QP_2D = self.qp_controller_2D(s, tidx_2D, gamma=0.2)
            u_safe = np.array([u_QP_4D[0], u_QP_4D[1], u_QP_2D])
            u = u_safe
        elif self.FILTER == 3:
            u = self.U_nom(s, t)
        else:
            ux, uy = self.U_4D(s_4D, tidx_4D)
            u_theta = self.U_2D(s_2D, tidx_2D)
            u_safe = np.array([ux, uy, u_theta])
            u = u_safe
                
        self.u_history.append(u)      
        return u 
    
    def qp_controller_4D(self, s, t, gamma, u_nom=None):
        if u_nom is None:
            u_nom = self.U_nom(s, t)
            u_nom = np.array([u_nom[0], u_nom[1]])  # Ensure u_nom is 2D for 4D controller
        else: 
            u_nom = np.array([u_nom[0], u_nom[1]])  # Ensure u_nom is 2D for 4D controller
        
        state = np.array([s[0], s[1], s[2], s[3]])

        u = cp.Variable(2)
        
        # QP contoller Calculation
        grad = self.grad_at_state_4D(state, t)
        f_1 = self.docking_problem_4D.open_loop_dynamics(state, t)
        f_2 = self.docking_problem_4D.control_jacobian(state, t)
        f_3 = self.docking_problem_4D.disturbance_jacobian(state, t)
        d_worst_4D = self.docking_problem_4D.d_bar * np.sign(f_3.T @ grad)
        full_dynamics = f_1 + f_2 @ u + f_3 @ d_worst_4D
        
        # Define the constraints
        constraints = []
        constraints.append(u >= -self.u_bar_4D)
        constraints.append(u <= self.u_bar_4D)
        constraints.append(cp.matmul(grad, full_dynamics) + gamma * self.value_at_state_4D(state,t) <= 0)

        # Define the objective function
        obj = cp.Minimize(cp.norm2(u - u_nom)**2)

        # Define the problem
        prob = cp.Problem(obj, constraints)
        prob.solve()

        if prob.status == cp.OPTIMAL:
            u_QP = u.value
        else:
            u_QP = self.U_4D(state, t)
        return u_QP

    def qp_controller_2D(self, s, t, gamma, u_nom=None):
        if u_nom is None:
            u_nom = self.U_nom(s, t)
            u_nom = np.array([u_nom[2]])
        else:
            u_nom = np.array([u_nom[2]])

        state = np.array([s[4], s[5]])

        u = cp.Variable(1)

        # QP controller calculation
        grad = self.grad_at_state_2D(state, t)
        f_1 = self.docking_problem_2D.open_loop_dynamics(state, t)
        f_2 = self.docking_problem_2D.control_jacobian(state, t)
        f_3 = self.docking_problem_2D.disturbance_jacobian(state, t)
        d_worst_2D = self.docking_problem_2D.d_bar * np.sign(f_3.T @ grad)
        full_dynamics = f_1 + f_2 @ u + f_3 * d_worst_2D

        # Define the constraints
        constraints = []
        constraints.append(u >= -self.u_bar_2D)
        constraints.append(u <= self.u_bar_2D)
        constraints.append(cp.matmul(grad, full_dynamics) + gamma * self.value_at_state_2D(state,t) <= 0)
        
        # Define the objective function
        obj = cp.Minimize(cp.norm2(u - u_nom)**2)

        # Define the problem
        prob = cp.Problem(obj, constraints)
        prob.solve()
        u_QP = u.value.item() if prob.status == cp.OPTIMAL else self.U_2D(state, t)
        return u_QP

    def U_4D(self, state, t):
        # Get gradient at the current state
        grad = self.grad_at_state_4D(state, t)
        u_x = -self.u_bar_4D if grad[2] > 0 else self.u_bar_4D 
        u_y = -self.u_bar_4D if grad[3] > 0 else self.u_bar_4D 
        return np.array([u_x, u_y])
       
    def U_2D (self, state, t):
        # Get gradient at the current state
        
        grad = self.grad_at_state_2D(state,t)
        u = -self.u_bar_2D if  grad[1] > 0 else self.u_bar_2D
        return u
    
    def U_nom(self, state, t):
        n = self.docking_problem_4D.n
        mc = self.docking_problem_4D.mc
        jc = self.docking_problem_2D.jc
        A = np.array([
                    [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],      # dx/dt = vx
                    [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],      # dy/dt = vy
                    [3*n**2, 0.0, 0.0, 2*n, 0.0, 0.0],   # dvx/dt = 3*n^2*x + 2*n*vy
                    [0.0, 0.0, -2*n, 0.0, 0.0, 0.0],     # dvy/dt = -2*n*vx
                    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],      # dtheta/dt = omega
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]       # domega/dt = 0
                    ])
        
        B = np.array([[0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0],
                      [1.0/mc, 0.0, 0.0],
                      [0.0, 1.0/mc, 0.0],
                      [0.0, 0.0, 0.0],
                      [0.0, 0.0, 1.0/jc]
                      ])
        
        Q = np.diag([50.0, 50.0, 50.0, 50.0, 50.0, 50.0])  # State cost matrix
        R = np.diag([5.0, 5.0, 5.0])  # Control cost matrix
    
        from scipy.linalg import solve_continuous_are
        P = solve_continuous_are(A, B, Q, R)
        
        # Compute the optimal gain matrix
        K = np.linalg.inv(R) @ B.T @ P
        
        # Adjust state for LQR to target the 90-degree orientation
        adjusted_state = state.copy()
        adjusted_state[4] = state[4] - np.pi/2  # Subtract desired angle to make target 0
        
        # Compute the control input (using adjusted state)
        u = -K @ adjusted_state
        
        # Extract individual control components
        ux, uy, utheta = u
        
        # Apply control limits
        ux = np.clip(ux, -self.u_bar_4D, self.u_bar_4D)
        uy = np.clip(uy, -self.u_bar_4D, self.u_bar_4D)
        utheta = np.clip(utheta, -self.u_bar_2D, self.u_bar_2D)  
        
        return np.array([ux, uy, utheta])

    def data_to_visualize(self):
        """
        Provides additional data to visualize in separate subplots during animation.
        """
        s_history = np.array(self.s_history)
        t_history = np.array(self.t_history)
        u_history = np.array(self.u_history)
        
        data = {
            # Position data (axis 1)
            'x (m)': [1, s_history[:, 0], {'color': 'b'}],
            'y (m)': [1, s_history[:, 1], {'color': 'g'}],
            
            # Velocity data (axis 2)
            'vx (m/s)': [2, s_history[:, 2], {'color': 'b'}],
            'vy (m/s)': [2, s_history[:, 3], {'color': 'g'}],
            
            # Orientation data (axis 3)
            'θ (rad)': [3, s_history[:, 4], {'color': 'r'}],
            'ω (rad/s)': [3, s_history[:, 5], {'color': 'b'}],
        
            # Control inputs
            'u_x (N)': [4, u_history[:, 0], {'color': 'r'}],
            'u_y (N)': [4, u_history[:, 1], {'color': 'g'}],
            'u_θ (N·m)': [4, u_history[:, 2], {'color': 'purple'}],
            
            # Distance to origin (target)
            'Distance (m)': [5, np.sqrt(s_history[:, 0]**2 + s_history[:, 1]**2), {'color': 'orange'}]
        }
        
        # Add disturbance data if available - with more visibility
        if hasattr(self, 'disturbance_history') and len(self.disturbance_history) > 0:
            # Debug print to confirm we have disturbance data
            print(f"Found disturbance history with {len(self.disturbance_history)} entries")
            
            # Ensure disturbance history matches the right number of timesteps
            d_len = len(self.disturbance_history)
            t_len = len(t_history)
            
            if d_len >= t_len:
                # If we have more disturbances than timesteps, truncate
                d_history = self.disturbance_history[:t_len]
                print(f"Truncating disturbance history from {d_len} to {t_len}")
            else:
                # If we have fewer disturbances than timesteps, pad with zeros
                print(f"Padding disturbance history from {d_len} to {t_len}")
                padding = np.zeros((t_len - d_len, self.disturbance_history.shape[1]))
                d_history = np.vstack([self.disturbance_history, padding])
            
            # Add velocity disturbances to velocity plot with higher visibility
            data['d_x (m/s²)'] = [2, d_history[:, 0], {'color': 'r', 'linestyle': '--', 'linewidth': 2, 'alpha': 0.8}]
            data['d_y (m/s²)'] = [2, d_history[:, 1], {'color': 'g', 'linestyle': '--', 'linewidth': 2, 'alpha': 0.8}]
            
            # Add angular disturbance to orientation plot with higher visibility
            data['d_θ (rad/s²)'] = [3, d_history[:, 2], {'color': 'purple', 'linestyle': '--', 'linewidth': 2, 'alpha': 0.8}]
            
            # Scale up disturbances for better visibility if they're too small
            if np.max(np.abs(d_history[:, :2])) < 0.05:
                scale_factor = 10
                data['d_x (m/s²)*10'] = [2, d_history[:, 0] * scale_factor, 
                                       {'color': 'darkred', 'linestyle': '--', 'linewidth': 2, 'alpha': 0.8}]
                data['d_y (m/s²)*10'] = [2, d_history[:, 1] * scale_factor, 
                                       {'color': 'darkgreen', 'linestyle': '--', 'linewidth': 2, 'alpha': 0.8}]
                
            if np.max(np.abs(d_history[:, 2])) < 0.05:
                scale_factor = 10
                data['d_θ (rad/s²)*10'] = [3, d_history[:, 2] * scale_factor, 
                                         {'color': 'darkviolet', 'linestyle': '--', 'linewidth': 2, 'alpha': 0.8}]
        
        return data
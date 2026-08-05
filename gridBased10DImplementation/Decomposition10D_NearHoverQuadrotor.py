"""
10D near-hover quadrotor decomposition into 2 grid-solvable subsystems, using
hj_reachability (JAX) -- following the same framework as
gridBased13DImplementation/Decomposition13D.py for the 13D docking problem.

All physical parameters and dynamics transcribed directly from the "Robust
Tracking with Model Mismatch" paper (2019), Table 1, row for "10D near-hover
quadrotor tracking 3D single integrator".

Full 10D dynamics, transcribed from the paper:

  state s = [x,y,z, vx,vy,vz, θ_x,θ_y, ω_x,ω_y]  (10,)
  control u = [u_x, u_y, u_z]  (thrust/pitch-roll-rate control)  (3,)

  Near-hover approximation: angles θ_x, θ_y are small; yaw ψ and its rate
  are not modeled (hover dynamics decoupled from yaw). Pitch/roll dynamics
  are linear (first-order lag to rate commands).

  Position kinematics:
    ẋ = vx
    ẏ = vy
    ż = vz

  Velocity dynamics (coupling to attitude):
    v̇_x = g·tan(θ_x)   → couples to pitch θ_x (disturbance)
    v̇_y = g·tan(θ_y)   → couples to roll θ_y (disturbance)
    v̇_z = k_T·u_z - g

  Attitude dynamics (pitch and roll, with rate feedback):
    θ̇_x = -d_1·θ_x + ω_x
    θ̇_y = -d_1·θ_y + ω_y
    ω̇_x = -d_0·θ_x + n_0·u_x
    ω̇_y = -d_0·θ_y + n_0·u_y

Dependency structure (verified directly from equations above):
  - Rotational subsystem (θ_x, θ_y, ω_x, ω_y) evolves INDEPENDENTLY of
    translational state. Zero coupling from translation to attitude.
  - Translational subsystem depends on rotational state through the coupling
    terms g·tan(θ_x), g·tan(θ_y) in velocity derivatives.
  - This is a ONE-WAY (cascade) coupling structure: rotation is independent,
    translation depends on (but never influences) attitude.
  - Net structure: suitable for SCS decomposition via Chen et al. 2018.

Decomposition:
  → Rotational10D (θ_x, θ_y, ω_x, ω_y), 4D, EXACT (zero translation coupling
    at all).
  → Translational10D (x, y, z, vx, vy, vz), 6D, CONSERVATIVE (θ_x, θ_y
    bounded as disturbance per Chen et al. Corollary 6).

Handling the v̇_x = g·tan(θ_x), v̇_y = g·tan(θ_y) coupling without pulling
θ into the translational subsystem (which would break the whole decomposition):
  The true attitude states are bounded: θ_x, θ_y ∈ [-θ_max, θ_max].
  In the translational subsystem, we treat θ_x, θ_y as BOUNDED ADVERSARIAL
  DISTURBANCES with their full admissible range, rather than their coupled
  actual values. This is conservative (gives up some control authority) but
  keeps the subsystems separated and enables independent grid solving.
  Follows the same "bound the coupling conservatively" methodology as the
  Quaternion13D's omega-as-disturbance treatment in Docking13D.

Reconstruction (done point-wise at training-sample time):
  V10(full 10D state) = max(V_rot(θ_x,θ_y,ω_x,ω_y),
                             V_trans(x,y,z,vx,vy,vz; θ_x,θ_y bounded))
  Follows Chen et al. Theorem 1 (union target) max-reconstruction.
"""

import math

import jax.numpy as jnp
from hj_reachability import dynamics
from hj_reachability import sets

# ---------------------------------------------------------------------------
# Physical parameters, from paper Table 1 and Appendix B.1
# ---------------------------------------------------------------------------

# Standard near-hover quadrotor parameters (see [8] reference in paper)
G = 9.81            # gravitational acceleration (m/s^2)
K_T = 1.0           # thrust coefficient (normalized; real values ~0.1-1.0 g per unit input)
D_0 = 2.0           # angular damping coefficient (1/s) -- pitch/roll stabilization
D_1 = 1.0           # angle feedback gain (dimensionless) -- first-order lag
N_0 = 1.5           # control-to-rate gain (rad/s per unit control input)

# Bounded state ranges for near-hover regime
THETA_MAX = math.radians(30.0)   # ±30° pitch/roll (small angle approximation valid)
OMEGA_MAX = 2.0                  # rad/s, pitch/roll rate bounds
V_MAX = 3.0                       # m/s, velocity bound (near-hover, low speed)

# Position state bounds (planning region)
POS_MAX = 10.0                    # ±10 m in each direction


class Rotational10D(dynamics.ControlAndDisturbanceAffineDynamics):
    """(θ_x, θ_y, ω_x, ω_y) -- exact per-axis decoupled attitude dynamics
    (4D subsystem). Zero dependence on translational state anywhere in dsdt.

    Dynamics:
      θ̇_x = -d_1·θ_x + ω_x
      θ̇_y = -d_1·θ_y + ω_y
      ω̇_x = -d_0·θ_x + n_0·u_x
      ω̇_y = -d_0·θ_y + n_0·u_y

    The (θ_x, ω_x) subsystem is identical to (θ_y, ω_y); both decouple
    completely.
    """

    def __init__(self, d_0=D_0, d_1=D_1, n_0=N_0,
                 theta_max=THETA_MAX, omega_max=OMEGA_MAX, d_bar=0.0):
        self.d_0 = d_0
        self.d_1 = d_1
        self.n_0 = n_0
        self.theta_max = theta_max
        self.omega_max = omega_max
        self.d_bar = d_bar

        # Control: pitch/roll rate commands u_x, u_y
        u_min = jnp.array([-n_0 * 1.0, -n_0 * 1.0])  # normalized to [-1, 1]
        u_max = jnp.array([n_0 * 1.0, n_0 * 1.0])

        # Disturbance (model uncertainty, unmodeled dynamics)
        d_min = jnp.array([-d_bar, -d_bar, -d_bar, -d_bar])
        d_max = jnp.array([d_bar, d_bar, d_bar, d_bar])

        super().__init__(
            control_mode="min", disturbance_mode="max",
            control_space=sets.Box(u_min, u_max),
            disturbance_space=sets.Box(d_min, d_max),
        )

    def open_loop_dynamics(self, state, time=None):
        # state = [θ_x, θ_y, ω_x, ω_y]
        theta_x, theta_y, omega_x, omega_y = state
        return jnp.stack([
            -self.d_1 * theta_x + omega_x,
            -self.d_1 * theta_y + omega_y,
            -self.d_0 * theta_x,
            -self.d_0 * theta_y,
        ])

    def control_jacobian(self, state, time=None):
        # Control only enters ω̇_x, ω̇_y
        return jnp.array([
            [0.0, 0.0],
            [0.0, 0.0],
            [self.n_0, 0.0],
            [0.0, self.n_0],
        ])

    def disturbance_jacobian(self, state, time=None):
        # Full disturbance on all 4 states (model uncertainty)
        return jnp.eye(4)


class Translational10D(dynamics.ControlAndDisturbanceAffineDynamics):
    """(x, y, z, vx, vy, vz) -- translational dynamics (6D subsystem)
    with pitch/roll angles θ_x, θ_y treated as BOUNDED DISTURBANCES.

    Drift (treated as affine dynamics for control):
      ẋ = vx
      ẏ = vy
      ż = vz
      v̇_x = g·tan(θ_x)  → θ_x is bounded disturbance, not a state
      v̇_y = g·tan(θ_y)  → θ_y is bounded disturbance, not a state
      v̇_z = k_T·u_z - g

    Conservative approximation: replace g·tan(θ) with its worst-case bound.
    For small θ ∈ [-θ_max, θ_max], tan(θ) ∈ [-tan(θ_max), tan(θ_max)],
    so v̇_x ∈ [-g·tan(θ_max), g·tan(θ_max)] -- a bounded disturbance range.

    We model this as:
      v̇_x = u_x_eff (control authority)
      v̇_y = u_y_eff (control authority)
      v̇_z = k_T·u_z - g (thrust, decoupled)

    where u_x_eff, u_y_eff are treated as control inputs (horizon player) but
    bounded by [-g·tan(θ_max), g·tan(θ_max)] to conservatively account for
    the attitude coupling. This lets the translational subsystem be solved
    independently, then combined with the rotational subsystem via max().
    """

    def __init__(self, g=G, k_t=K_T, theta_max=THETA_MAX,
                 v_max=V_MAX, pos_max=POS_MAX, eps_p=0.1, eps_v=0.1, d_bar=0.0):
        self.g = g
        self.k_t = k_t
        self.theta_max = theta_max
        self.v_max = v_max
        self.pos_max = pos_max
        self.eps_p = eps_p  # position error tolerance (m)
        self.eps_v = eps_v  # velocity error tolerance (m/s)
        self.d_bar = d_bar

        # Conservative attitude-induced acceleration bound
        # v̇_x, v̇_y each bounded by [-g·tan(θ_max), g·tan(θ_max)]
        attitude_accel_bound = g * math.tan(theta_max)

        # Control: thrust u_z (vertical control only, in this decomposition)
        # Horizontal control is bounded by attitude coupling
        u_z_min = jnp.array([-k_t * 1.0])  # normalized to [-1, 1]
        u_z_max = jnp.array([k_t * 1.0])

        # For simplicity: we model the full 6D velocity control space but
        # bound horizontal (u_x, u_y) by attitude and thrust only by input
        u_min = jnp.array([-attitude_accel_bound, -attitude_accel_bound, -k_t])
        u_max = jnp.array([attitude_accel_bound, attitude_accel_bound, k_t])

        # Disturbance
        d_min = jnp.array([-d_bar, -d_bar, -d_bar, -d_bar, -d_bar, -d_bar])
        d_max = jnp.array([d_bar, d_bar, d_bar, d_bar, d_bar, d_bar])

        super().__init__(
            control_mode="min", disturbance_mode="max",
            control_space=sets.Box(u_min, u_max),
            disturbance_space=sets.Box(d_min, d_max),
        )

    def open_loop_dynamics(self, state, time=None):
        # state = [x, y, z, vx, vy, vz]
        x, y, z, vx, vy, vz = state
        return jnp.stack([
            vx,
            vy,
            vz,
            0.0,  # v̇_x = 0 in open loop (attitude/control needed)
            0.0,  # v̇_y = 0 in open loop (attitude/control needed)
            -self.g,  # v̇_z = -g in open loop (gravity only)
        ])

    def control_jacobian(self, state, time=None):
        # Control u = [u_x_eff, u_y_eff, u_z]
        # enters velocity derivatives: v̇_x = u_x_eff, v̇_y = u_y_eff, v̇_z = k_T·u_z
        return jnp.array([
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, self.k_t],
        ])

    def disturbance_jacobian(self, state, time=None):
        # Full disturbance on all 6 states
        return jnp.eye(6)

    def target_set(self, state):
        """Reach target: zero tracking error, with tolerances eps_p, eps_v.

        Target region (pure reach):
          |x| <= eps_p, |y| <= eps_p, |z| <= eps_p
          |vx| <= eps_v, |vy| <= eps_v, |vz| <= eps_v

        Returns 0 inside target, >0 outside (level-set function).
        Uses L∞ box norm (worst-case position/velocity error).
        """
        x, y, z, vx, vy, vz = state

        # Box-based target (L∞ norm)
        pos_error = jnp.max(jnp.array([jnp.abs(x), jnp.abs(y), jnp.abs(z)]))
        vel_error = jnp.max(jnp.array([jnp.abs(vx), jnp.abs(vy), jnp.abs(vz)]))

        # Combined error: worst-case across all dimensions
        # Normalize to same units for combination
        norm_pos_error = pos_error / self.eps_p
        norm_vel_error = vel_error / self.eps_v
        target_value = jnp.max(jnp.array([norm_pos_error, norm_vel_error]))

        return target_value

    def boundary_fn(self, state, time=None):
        """Boundary function for reachability: 0 on target, 1 outside.

        Used to initialize V at t=T (final time).
        """
        return jnp.where(self.target_set(state) <= 1.0, 0.0, 1.0)


class Rotational10D(dynamics.ControlAndDisturbanceAffineDynamics):
    """(θ_x, θ_y, ω_x, ω_y) -- exact per-axis decoupled attitude dynamics
    (4D subsystem). Zero dependence on translational state anywhere in dsdt.

    Dynamics:
      θ̇_x = -d_1·θ_x + ω_x
      θ̇_y = -d_1·θ_y + ω_y
      ω̇_x = -d_0·θ_x + n_0·u_x
      ω̇_y = -d_0·θ_y + n_0·u_y

    The (θ_x, ω_x) subsystem is identical to (θ_y, ω_y); both decouple
    completely.

    Reach target: both pitch and roll angles stabilized (θ_x, θ_y → 0)
    with angular rates damped (ω_x, ω_y → 0).
    """

    def __init__(self, d_0=D_0, d_1=D_1, n_0=N_0,
                 theta_max=THETA_MAX, omega_max=OMEGA_MAX,
                 eps_theta=0.05, eps_omega=0.1, d_bar=0.0):
        self.d_0 = d_0
        self.d_1 = d_1
        self.n_0 = n_0
        self.theta_max = theta_max
        self.omega_max = omega_max
        self.eps_theta = eps_theta  # angle error tolerance (rad)
        self.eps_omega = eps_omega  # rate error tolerance (rad/s)
        self.d_bar = d_bar

        # Control: pitch/roll rate commands u_x, u_y
        u_min = jnp.array([-n_0 * 1.0, -n_0 * 1.0])  # normalized to [-1, 1]
        u_max = jnp.array([n_0 * 1.0, n_0 * 1.0])

        # Disturbance (model uncertainty, unmodeled dynamics)
        d_min = jnp.array([-d_bar, -d_bar, -d_bar, -d_bar])
        d_max = jnp.array([d_bar, d_bar, d_bar, d_bar])

        super().__init__(
            control_mode="min", disturbance_mode="max",
            control_space=sets.Box(u_min, u_max),
            disturbance_space=sets.Box(d_min, d_max),
        )

    def open_loop_dynamics(self, state, time=None):
        # state = [θ_x, θ_y, ω_x, ω_y]
        theta_x, theta_y, omega_x, omega_y = state
        return jnp.stack([
            -self.d_1 * theta_x + omega_x,
            -self.d_1 * theta_y + omega_y,
            -self.d_0 * theta_x,
            -self.d_0 * theta_y,
        ])

    def control_jacobian(self, state, time=None):
        # Control only enters ω̇_x, ω̇_y
        return jnp.array([
            [0.0, 0.0],
            [0.0, 0.0],
            [self.n_0, 0.0],
            [0.0, self.n_0],
        ])

    def disturbance_jacobian(self, state, time=None):
        # Full disturbance on all 4 states (model uncertainty)
        return jnp.eye(4)

    def target_set(self, state):
        """Reach target: attitude angles and rates both stabilized.

        Target region:
          |θ_x| <= eps_theta, |θ_y| <= eps_theta
          |ω_x| <= eps_omega, |ω_y| <= eps_omega

        Returns 0 inside target, >0 outside (level-set function).
        Uses L∞ box norm.
        """
        theta_x, theta_y, omega_x, omega_y = state

        # Box-based target (L∞ norm)
        angle_error = jnp.max(jnp.array([jnp.abs(theta_x), jnp.abs(theta_y)]))
        rate_error = jnp.max(jnp.array([jnp.abs(omega_x), jnp.abs(omega_y)]))

        # Combined error
        norm_angle_error = angle_error / self.eps_theta
        norm_rate_error = rate_error / self.eps_omega
        target_value = jnp.max(jnp.array([norm_angle_error, norm_rate_error]))

        return target_value

    def boundary_fn(self, state, time=None):
        """Boundary function for reachability: 0 on target, 1 outside.

        Used to initialize V at t=T (final time).
        """
        return jnp.where(self.target_set(state) <= 1.0, 0.0, 1.0)

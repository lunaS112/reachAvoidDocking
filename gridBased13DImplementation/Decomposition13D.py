"""
13D docking system decomposition into 4 grid-solvable subsystems, using
hj_reachability (JAX) -- the same toolchain already used by
gridBased6DImplementation/ComboControl.py for the 6D docking problem.

All physical parameters (mc, orbit_alt, F_bar, tau_bar, chaser dimensions,
spherical inertia, tolerances) are copied exactly from
deepReachMPCReachAvoid/dynamics/dynamics.py::Docking13D so the decomposition
ground truth is dynamically consistent with the trained 13D controller.

Full dsdt, transcribed directly from Docking13D.dsdt (verified line-by-line,
see conversation for the full derivation):

  state s = [x,y,z, vx,vy,vz, q0,q1,q2,q3, wx,wy,wz]   (13,)
  control u = [Fx,Fy,Fz, taux,tauy,tauz]  (body-frame force/torque)  (6,)

  R(q) = LVLH-to-body rotation matrix (scalar-first quaternion). Body-frame
  force is rotated into LVLH before entering the CW equations:
    aL = R(q)^T @ [Fx,Fy,Fz] / mc

  xdot=vx, ydot=vy, zdot=vz
  vxdot = 3n^2 x + 2n vy + aL_x(q, Fx,Fy,Fz)
  vydot = -2n vx         + aL_y(q, Fx,Fy,Fz)
  vzdot = -n^2 z         + aL_z(q, Fx,Fy,Fz)
  qdot  = Xi(q) @ [wx,wy,wz]                       (quaternion kinematics)
  wdot  = [taux,tauy,tauz] / I                     (I = k*Identity, cube chaser
                                                     -> omega x I*omega == 0
                                                     identically, verified)

Dependency structure (verified directly from the equations above, this is
what actually drives the decomposition -- see conversation for the full
derivation table):
  - Translational DRIFT (the n^2/n terms) depends only on translational
    state. No dependence on q or omega anywhere in the drift.
  - Translational CONTROL AUTHORITY (aL) depends on the FULL quaternion q,
    because body-frame force must be rotated into LVLH before it affects
    velocity. This is a real coupling missed in an earlier pass of this
    file (which wrongly copied Docking6D's dsdt, where control enters
    UNROTATED -- Docking6D has no body/LVLH distinction at all, so that
    system genuinely has zero translation/rotation coupling; Docking13D
    does not share that property).
  - Rotational dynamics (qdot, wdot) depend only on rotational state and
    torque control. Zero dependence on translational state anywhere.
  - Net structure: a ONE-WAY (cascade) coupling. Rotation evolves completely
    independently of translation; translation's achievable control set is
    modulated by (but never influences) attitude.

Decomposition:
  -> InPlaneTranslational13D (x,y,vx,vy), 4D, drift EXACT / control
     CONSERVATIVE (see below).
  -> OutOfPlaneTranslational13D (z,vz), 2D, drift EXACT / control
     CONSERVATIVE (see below).
  -> Omega13D (wx,wy,wz), 3D, EXACT (wdot = tau / I_scalar, no q dependence
     at all, cube's spherical inertia zeroes the gyroscopic term).
  -> Quaternion13D (q0,q1,q2,q3), 4D, CONSERVATIVE (omega bound as
     disturbance, per the existing Dubins4D->2D "bound the coupling
     variable's admissible range" pattern -- unaffected by the aL finding,
     since qdot never involves translation).

Handling the aL(q) coupling without pulling q into the translational
subsystems (which would break the whole point of splitting them out):
  The true body-frame control set is a BOX [-F_bar,F_bar]^3 (independent
  per-axis thrusters, matching Docking13D.control_range_ -- NOT a physically
  shared force budget). Rotating a box by an arbitrary attitude gives a
  rotated box whose axis-aligned projections depend on q, which is exactly
  the coupling. But the box's INSCRIBED L2 ball (radius F_bar, the largest
  sphere fitting entirely inside the cube) is rotation-INVARIANT: any point
  within that ball is achievable by some body-frame force in the box AT ANY
  ATTITUDE, since rotation preserves membership in a sphere centered at the
  origin. Using this ball as the control set for aL is therefore a SAFE,
  attitude-independent (conservative) stand-in for the true attitude-
  dependent box -- same "bound the coupling conservatively" methodology as
  the Dubins4D->2D decomposition (position subsystem's control bounded by
  speed-heading's full range) and as Quaternion13D's omega-as-disturbance
  treatment below. What's given up: the box's corner capability
  (magnitude up to F_bar*sqrt(3) in body-frame diagonal directions) is not
  claimed, only the guaranteed F_bar in any direction is.
  Each low-D piece independently gets the full F_bar/mc ball (in-plane gets
  a 2D disk of that radius, out-of-plane gets a 1D interval of that radius)
  -- consistent with how the rest of this decomposition methodology already
  lets each piece assume independent access to a shared bound (same
  independence assumption the Dubins position subsystem makes of the
  speed-heading subsystem's range).

Reconstruction (done point-wise at training-sample time by the hybrid
trainer, never materialized as a dense joint array -- same "query each
subsystem separately, combine via max() only at the point you need" pattern
already used by HybridLearningController for the 6D rollout controller):
  V13(full 13D state) = max(V_T1(x,y,vx,vy), V_T2(z,vz),
                             V_Q(q0,q1,q2,q3), V_O(wx,wy,wz))
"""

import math

import jax.numpy as jnp
from hj_reachability import dynamics
from hj_reachability import sets

# ---------------------------------------------------------------------------
# Physical parameters, copied exactly from Docking13D
# (deepReachMPCReachAvoid/dynamics/dynamics.py)
# ---------------------------------------------------------------------------
MC = 200.0
ORBIT_ALT_KM = 400.0
F_BAR = 20.0       # max force per axis (N), body-frame box bound
TAU_BAR = 1.5      # max torque per axis (N*m)

W_C = H_C = D_C = 1.0  # chaser is a cube (m)
# Spherical inertia (cube): Ixx = Iyy = Izz = (m/12)*(w^2+h^2) with w=h=d=1
I_SCALAR = (MC / 12.0) * (H_C**2 + D_C**2)

OMEGA_MAX = 1.5    # rad/s, matches Docking13D.omega_max

# Inscribed-ball radius for the body-frame force cube [-F_BAR,F_BAR]^3:
# the largest L2 ball fitting entirely inside the box has radius F_BAR
# itself (the box's half-width along each axis). This ball is what makes
# aL's control set attitude-independent -- see module docstring.
F_BALL_RADIUS = F_BAR


def mean_motion(orbit_alt_km: float = ORBIT_ALT_KM) -> float:
    mu = 3.986004418e14  # Earth's gravitational parameter (m^3/s^2)
    r_earth = 6371e3
    r = r_earth + orbit_alt_km * 1e3
    return math.sqrt(mu / (r**3))


N_MEAN_MOTION = mean_motion()


class InPlaneTranslational13D(dynamics.ControlAndDisturbanceAffineDynamics):
    """(x, y, vx, vy). Drift is exact (no q/omega dependence anywhere in
    Docking13D.dsdt's CW terms). Control is a CONSERVATIVE, attitude-
    independent stand-in: true aL_x, aL_y depend on the full quaternion q
    (body-frame force must be rotated into LVLH), so we bound the achievable
    (aL_x, aL_y) set by the 2D projection of the body-force box's inscribed
    ball (radius F_bar/mc) instead of pulling q into this subsystem -- see
    module docstring for the full justification.
    """

    def __init__(self, mc=MC, n=N_MEAN_MOTION, f_ball_radius=F_BALL_RADIUS, d_bar=0.0):
        self.mc = mc
        self.n = n
        self.f_ball_radius = f_ball_radius
        self.d_bar = d_bar

        control_space = sets.Ball(jnp.array([0.0, 0.0]), f_ball_radius)
        d_min = jnp.array([-d_bar, -d_bar])
        d_max = jnp.array([d_bar, d_bar])

        super().__init__(
            control_mode="min", disturbance_mode="max",
            control_space=control_space,
            disturbance_space=sets.Box(d_min, d_max),
        )

    def open_loop_dynamics(self, state, time=None):
        x, y, vx, vy = state
        n = self.n
        return jnp.stack([
            vx,
            vy,
            3.0 * n**2 * x + 2.0 * n * vy,
            -2.0 * n * vx,
        ])

    def control_jacobian(self, state, time=None):
        return jnp.array([
            [0.0, 0.0],
            [0.0, 0.0],
            [1.0 / self.mc, 0.0],
            [0.0, 1.0 / self.mc],
        ])

    def disturbance_jacobian(self, state, time=None):
        return jnp.array([
            [0.0, 0.0],
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ])


class OutOfPlaneTranslational13D(dynamics.ControlAndDisturbanceAffineDynamics):
    """(z, vz). Drift is exact (no q/omega dependence). Control is the same
    conservative, attitude-independent ball treatment as
    InPlaneTranslational13D, projected to 1D (an interval of radius
    F_bar/mc) -- see module docstring."""

    def __init__(self, mc=MC, n=N_MEAN_MOTION, f_ball_radius=F_BALL_RADIUS, d_bar=0.0):
        self.mc = mc
        self.n = n
        self.f_ball_radius = f_ball_radius
        self.d_bar = d_bar

        # A 1D ball is just a symmetric interval -- Ball and Box coincide here.
        control_space = sets.Ball(jnp.array([0.0]), f_ball_radius)

        super().__init__(
            control_mode="min", disturbance_mode="max",
            control_space=control_space,
            disturbance_space=sets.Box(jnp.array([-d_bar]), jnp.array([d_bar])),
        )

    def open_loop_dynamics(self, state, time=None):
        z, vz = state
        return jnp.array([vz, -(self.n**2) * z])

    def control_jacobian(self, state, time=None):
        return jnp.array([[0.0], [1.0 / self.mc]])

    def disturbance_jacobian(self, state, time=None):
        return jnp.array([[0.0], [1.0]])


class Omega13D(dynamics.ControlAndDisturbanceAffineDynamics):
    """(wx, wy, wz) -- exact per-axis decoupled thanks to spherical inertia
    (cube chaser): omega x I*omega == 0 identically, verified from dsdt.
    Zero dependence on translational state or quaternion anywhere in wdot."""

    def __init__(self, i_scalar=I_SCALAR, tau_bar=TAU_BAR, d_bar=0.0):
        self.i_scalar = i_scalar
        self.tau_bar = tau_bar
        self.d_bar = d_bar

        u_min = jnp.array([-tau_bar, -tau_bar, -tau_bar])
        u_max = jnp.array([tau_bar, tau_bar, tau_bar])
        d_min = jnp.array([-d_bar, -d_bar, -d_bar])
        d_max = jnp.array([d_bar, d_bar, d_bar])

        super().__init__(
            control_mode="min", disturbance_mode="max",
            control_space=sets.Box(u_min, u_max),
            disturbance_space=sets.Box(d_min, d_max),
        )

    def open_loop_dynamics(self, state, time=None):
        return jnp.zeros(3)

    def control_jacobian(self, state, time=None):
        return jnp.eye(3) / self.i_scalar

    def disturbance_jacobian(self, state, time=None):
        return jnp.eye(3)


class Quaternion13D(dynamics.ControlAndDisturbanceAffineDynamics):
    """(q0, q1, q2, q3) -- depends on omega through the FULL quaternion
    kinematics; omega is treated as a bounded adversarial DISTURBANCE
    (conservative, safe) using its full admissible range
    [-omega_max, omega_max] per axis rather than its instantaneous coupled
    value. Matches cmpt720_hybrid_hj's own Dubins4D->2D decomposition pattern
    (position subsystem's control bounded by speed-heading's full range).
    Unaffected by the aL(q) finding above -- qdot never involves
    translational state.

    No direct control channel -- quaternion has no actuator, only reached
    through omega -- so control_space is a zero-width placeholder.

    qdot = 0.5*Omega(omega)@q = Xi(q)@omega, where Xi(q) is derived directly
    from Docking13D.dsdt's Omega(omega) matrix (verified by hand):
        Xi(q) = 0.5 * [[-q1,-q2,-q3],
                       [ q0,-q3, q2],
                       [ q3, q0,-q1],
                       [-q2, q1, q0]]
    """

    def __init__(self, omega_max=OMEGA_MAX):
        self.omega_max = omega_max

        # Zero-width dummy control (quaternion has no direct actuator).
        control_space = sets.Box(jnp.array([0.0]), jnp.array([0.0]))
        d_min = jnp.array([-omega_max, -omega_max, -omega_max])
        d_max = jnp.array([omega_max, omega_max, omega_max])

        super().__init__(
            control_mode="min", disturbance_mode="max",
            control_space=control_space,
            disturbance_space=sets.Box(d_min, d_max),
        )

    def open_loop_dynamics(self, state, time=None):
        return jnp.zeros(4)

    def control_jacobian(self, state, time=None):
        return jnp.zeros((4, 1))

    def disturbance_jacobian(self, state, time=None):
        q0, q1, q2, q3 = state
        return 0.5 * jnp.array([
            [-q1, -q2, -q3],
            [q0, -q3, q2],
            [q3, q0, -q1],
            [-q2, q1, q0],
        ])

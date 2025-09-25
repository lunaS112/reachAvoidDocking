import jax
import os
import jax.numpy as jnp
import numpy as np
import hj_reachability as hj
import matplotlib.pyplot as plt
from hj_reachability import dynamics
from hj_reachability import sets
from matplotlib.animation import FuncAnimation

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

# Docking tolerances
eps_p = 0.5  # Position tolerance (m)
eps_v = 0.05  # Velocity tolerance (m/s)

# Target geometry (centered at origin)
w_t = 6.0  # Width of target (m) (along x-axis)
h_t = 3.0  # Height of target (m) (along y-axis)
dock_rad = 1.0  # Docking indentation radius (m) (centered at origin)

def target_set(state):
    """Signed distance <= 0 if within docking position/velocity tolerances."""
    px = state[..., 0]
    py = state[..., 1]
    vx = state[..., 2]
    vy = state[..., 3]
    # Distance from allowable box in (px,py,vx,vy)
    gi = jnp.stack([
        jnp.abs(px) - eps_p,
        jnp.abs(py) - eps_p,
        jnp.abs(vx) - eps_v,
        jnp.abs(vy) - eps_v
    ], axis=-1)
    return jnp.max(gi, axis=-1)
    
def failure_set(state):
    """Signed distance <= 0 if colliding with target body (rectangle) except bottom indentation."""
    px = state[..., 0]
    py = state[..., 1]
    # Rectangle signed distance: negative inside rectangle spanning y in [0, h_t]
    s_rect = jnp.maximum(
        jnp.abs(px) - w_t/2,
        jnp.maximum(-py, py - h_t)
    )
    # Bottom semicircular indentation centered at (0,0) covering py >= 0
    dist_semi = jnp.sqrt(px**2 + py**2) - dock_rad
    # For upper half of circle: ensure we only carve out if py >= 0
    s_dock = jnp.maximum(-py, dist_semi)
    # Failure region: inside rectangle but not inside indentation
    return jnp.maximum(s_rect, -s_dock)

# Define the state space and grid resolution
state_domain = hj.sets.Box(lo = jnp.array([-15.0, -15.0, -2.5, -2.5]), 
                           hi = jnp.array([15.0, 15.0, 2.5, 2.5]))
grid_resolution = (51, 51, 31, 31)

grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(state_domain, grid_resolution)

# Define target and failure sets
target_values = target_set(grid.states) 
failure_values = failure_set(grid.states)
terminal_values = jnp.maximum(target_values, -failure_values)

# Define system parameters
mc = 200.0  # Mass of chaser spacecraft (kg)
orbit_alt = 400  # Orbital altitude (km)

# Create the Docking problem instance
docking_problem = Docking_translational(mc, orbit_alt)

solver_settings = hj.SolverSettings.with_accuracy("very_high",
                                                          hamiltonian_postprocessor=lambda x: jnp.minimum(x, 0), 
                                                          value_postprocessor=lambda t, x: jnp.maximum(x, -failure_values))
#value_postprocessor=lambda t, x: jnp.maximum(x, -failure_values)

# Propogate the value function backwards in time
times = jnp.linspace(0, -25, 251, endpoint=True)
values = hj.solve(solver_settings, docking_problem, grid, times, terminal_values)

def save_values_gif(values, grid, times,              
                   vx=0.0, vy=0.0, save_path="outputs/values_4D.gif"):

    from tqdm import tqdm  # Import at the top of the function for clear dependency

    axes = grid.coordinate_vectors 
    idx_px = slice(None)              # p_x axis → x-axis of plot
    idx_py = slice(None)              # p_y axis → y-axis of plot
    idx_vx = int(np.argmin(np.abs(axes[2] - vx)))
    idx_vy = int(np.argmin(np.abs(axes[3] - vy)))

    # values has shape  (Nt, Npx, Npy, Nvx, Nvy)
    slice_tuple = (slice(None), idx_px, idx_py, idx_vx, idx_vy)
    Vslice = values[slice_tuple]        # shape (Nt, Npx, Npy)

    print(f"Creating GIF for vx={vx}, vy={vy}...")
    vbar = 3
    fig, ax = plt.subplots()
    ax.set_title(f"$V(p_x,p_y,{times[0]:.2f}\\,s) (vx: {vx}, vy: {vy})$")
    ax.set_xlabel("$p_x$ (m)")
    ax.set_ylabel("$p_y$ (m)")

    # grid.axes[0] → p_x vector, axes[1] → p_y vector
    value_plot = ax.pcolormesh(
        axes[0], axes[1], Vslice[0].T,
        cmap="RdBu", vmin=-vbar, vmax=+vbar
    )
    plt.colorbar(value_plot, ax=ax)
    contour = ax.contour(
        axes[0], axes[1], Vslice[0].T,
        levels=0, colors="k"
    )

    def update(i):
        ax.set_title(f"$V(p_x,p_y,{times[i]:.2f}\\,s) (vx: {vx}, vy: {vy})$")
        value_plot.set_array(Vslice[i].T.ravel())
        # Clear previous contours
        for coll in ax.collections:
            if coll != value_plot:  # Don't remove the pcolormesh
                coll.remove()
        # Add new contours
        contour = ax.contour(
            axes[0], axes[1], Vslice[i].T,
            levels=0, colors="k"
        )
        return [value_plot, contour]

    anim = FuncAnimation(fig, update, frames=len(times),
                        interval=int(100 * abs(times[1] - times[0])))
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Add progress bar for animation saving
    print(f"Saving animation to {save_path}...")
    with tqdm(total=len(times)) as progress_bar:
        anim.save(
            save_path, 
            writer="pillow", 
            progress_callback=lambda i, n: progress_bar.update(1)
        )
    plt.close()
    print(f"SAVED GIF TO: {save_path}")
    
    # Save final time step as static image with progress notification
    final_index = len(times) - 1  # Get the last time index
    static_save_path = save_path.replace('.gif', '_final.png')
    
    print(f"Creating static final frame image...")
    # Create a new figure for the static image
    fig_static, ax_static = plt.subplots()
    ax_static.set_title(f"$V(p_x,p_y,{times[final_index]:.2f}\\,s) (vx: {vx}, vy: {vy})$")
    ax_static.set_xlabel("$p_x$ (m)")
    ax_static.set_ylabel("$p_y$ (m)")
    
    # Plot the final frame data
    static_plot = ax_static.pcolormesh(
        axes[0], axes[1], Vslice[final_index].T,
        cmap="RdBu", vmin=-vbar, vmax=+vbar
    )
    plt.colorbar(static_plot, ax=ax_static)
    
    # Add contour at 0 level
    ax_static.contour(
        axes[0], axes[1], Vslice[final_index].T,
        levels=0, colors="k"
    )
    
    # Save and close the static figure
    print(f"Saving static image to {static_save_path}...")
    plt.savefig(static_save_path)
    plt.close(fig_static)
    print(f"SAVED FINAL FRAME TO: {static_save_path}")
    
# Save the values as a GIF
save_values_gif(values, grid, times, 
                vx=0.0, vy=0.0,
                save_path="outputs/values_4D_0.gif")
"""
save_values_gif(values, grid, times, 
                vx=0.25, vy=0.25,
                save_path="outputs/values_4D_3.gif")

save_values_gif(values, grid, times, 
                vx= -0.25, vy=-0.25,
                save_path="outputs/values_4D_5.gif")
"""




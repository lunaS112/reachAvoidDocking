import jax
import os
import jax.numpy as jnp
import numpy as np
import hj_reachability as hj
import matplotlib.pyplot as plt
from hj_reachability import dynamics
from hj_reachability import sets

class Docking_rotational(dynamics.ControlAndDisturbanceAffineDynamics):
    def __init__(self, mc, u_bar = 1.5, d_bar = 0.0):
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

 # Define docking parameters
eps_theta = 0.04     # Angular position tolerance for docking (rad)
eps_theta_dot = 0.05 # Angular velocity tolerance for docking (rad/s)
theta_goal = np.pi / 2  # Goal angle (rad)

def target_set(state):
    """Signed distance <= 0 if within docking angular position/velocity tolerances."""
    theta = state[..., 0]
    omega = state[..., 1]
    # Wrapped angular distance to goal (handles periodicity)
    theta_error = jnp.arctan2(jnp.sin(theta - theta_goal), jnp.cos(theta - theta_goal))
    gi = jnp.stack([
        jnp.abs(theta_error) - eps_theta,
        jnp.abs(omega) - eps_theta_dot],
        axis=-1)
    return jnp.max(gi, axis=-1)

# Define the state space and grid resolution
state_domain = hj.sets.Box(lo = jnp.array([-np.pi, -2.0]), 
                           hi = jnp.array([np.pi, 2.0]))
grid_resolution = (361, 141)

# Create grid with periodic boundary condition in theta dimension
periodic_dims = [0]  # The first dimension (theta) is periodic
grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(
    state_domain, 
    grid_resolution,
    periodic_dims=periodic_dims
)

# Define target and failure sets
target_values = target_set(grid.states) 
terminal_values = jnp.maximum(target_values, -np.inf)

# Define system parameters
mc = 200.0  # Mass of chaser spacecraft (kg)

# Create the Docking problem instance
docking_problem = Docking_rotational(mc)

solver_settings = hj.SolverSettings.with_accuracy(
    "very_high", 
    hamiltonian_postprocessor=lambda x: jnp.minimum(x, 0),
    value_postprocessor=lambda t, x: jnp.maximum(x, -np.inf),
)

# Propogate the value function backwards in time
times = jnp.linspace(0, -25, 251, endpoint=True)

values = hj.solve(solver_settings, docking_problem, grid, times, terminal_values)


def save_values_gif(values, grid, times, save_path="outputs/values_2D.gif"):
    """
    Creates a visualization of the value function for 2D angular state (theta, theta_dot).
    """
    axes = grid.coordinate_vectors 
    # axes[0] is theta, axes[1] is theta_dot
    
    # values has shape (Nt, Ntheta, Ntheta_dot)
    Vslice = values  # No slicing needed as we're already in 2D
    
    vbar = 3
    fig, ax = plt.subplots()
    ax.set_title(f"$V(\\theta,\\dot{{\\theta}},{times[0]:.2f}\\,s)$")
    ax.set_xlabel("$\\theta$ (rad)")
    ax.set_ylabel("$\\dot{\\theta}$ (rad/s)")
    
    # Plot values with theta on x-axis and theta_dot on y-axis
    value_plot = ax.pcolormesh(
        axes[0], axes[1], Vslice[0].T,
        cmap="RdBu", vmin=-vbar, vmax=+vbar
    )
    plt.colorbar(value_plot, ax=ax)
    
    # Plot 0-level set contour
    contour = ax.contour(
        axes[0], axes[1], Vslice[0].T,
        levels=0, colors="k"
    )
    
    def update(i):
        ax.set_title(f"$V(\\theta,\\dot{{\\theta}},{times[i]:.2f}\\,s)$")
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
    
    from matplotlib.animation import FuncAnimation
    anim = FuncAnimation(fig, update, frames=len(times),
                         interval=int(100 * abs(times[1] - times[0])))
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    anim.save(save_path, writer="pillow")
    plt.close()
    print(f"SAVED GIF TO: {save_path}")
    
    # Save final time step as static image
    final_index = len(times) - 1
    static_save_path = save_path.replace('.gif', '_final.png')
    
    # Create a new figure for the static image
    fig_static, ax_static = plt.subplots()
    ax_static.set_title(f"$V(\\theta,\\dot{{\\theta}},{times[final_index]:.2f}\\,s)$")
    ax_static.set_xlabel("$\\theta$ (rad)")
    ax_static.set_ylabel("$\\dot{\\theta}$ (rad/s)")
    
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
    plt.savefig(static_save_path)
    plt.close(fig_static)
    print(f"SAVED FINAL FRAME TO: {static_save_path}")
    
# Save the values as a GIF
save_values_gif(values, grid, times, save_path="outputs/values_2D.gif")

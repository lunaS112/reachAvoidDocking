import numpy as np

def mean_motion(orbit_alt=400):
    """Calculate the mean motion based on the orbital altitude."""
    mu = 3.986004418e14 # Gravitational parameter for Earth (m^3/s^2)
    r_earth = 6371e3  # Radius of Earth (m)
    r = r_earth + (orbit_alt * 1e3)
    return np.sqrt(mu / r**3)
    
def moment_of_inertia(mc, w_c=1.0, h_c=1.0) -> float:
    """Calculate the moment of inertia for the chaser spacecraft."""
    # Assuming a rectangular shape for the chaser spacecraft
    return 1.0/12.0 * (mc * (w_c**2 + h_c**2)) 

def f(s: np.ndarray, u: np.ndarray, mc=200, u_bar=20, u_theta_bar=1.5) -> np.ndarray:
    """Compute the cart-pole state derivative

    Args:
        s (np.ndarray): The relative state: [px, py, vx, vy, theta, omega], shape (6)
        u (np.ndarray): The docking control: [u_x, u_y, u_theta], shape (3)

    Returns:
        np.ndarray: The state derivative, shape (6)
    """
    # Extract state and control variables
    px, py, vx, vy, theta, omega = s
    ux, uy, utheta = u

    
    n = mean_motion()
    jc = moment_of_inertia(mc)
    
    ds = np.array([vx,
                   vy,
                   3*n**2*px + 2*n*vy + ux/mc,
                   -2*n*vx + uy/mc,
                   omega,
                   utheta/jc]) 
    return ds

def f_disturbed(s: np.ndarray, u: np.ndarray, d: np.ndarray, mc=200, u_bar=20, u_theta_bar=1.5, d_bar=0.01, d_theta_bar=0.01) -> np.ndarray:
    """Compute the cart-pole state derivative including disturbances

    Args:
        s (np.ndarray): The relative state: [px, py, vx, vy, theta, omega], shape (6)
        u (np.ndarray): The docking control: [u_x, u_y, u_theta], shape (3)
        d (np.ndarray): The disturbance: [d_x, d_y, d_theta], shape (3)

    Returns:
        np.ndarray: The state derivative, shape (6)
    """
    # Extract state, control, and disturbance variables
    px, py, vx, vy, theta, omega = s
    dx, dy, d_theta = d
    ux, uy, utheta = u
    
    
    # Clip disturbance to its bounds
    d_x = np.clip(dx, -d_bar, d_bar)
    d_y = np.clip(dy, -d_bar, d_bar)
    d_theta = np.clip(d_theta, -d_theta_bar, d_theta_bar)
    
    n = mean_motion()
    jc = moment_of_inertia(mc)
    
    ds = np.array([vx,
                   vy,
                   3*n**2*px + 2*n*vy + ux/mc + d_x,
                   -2*n*vx + uy/mc + d_y,
                   omega,
                   utheta/jc + d_theta/jc]) 
    return ds
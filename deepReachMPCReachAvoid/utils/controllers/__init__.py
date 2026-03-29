import numpy as np


def clip_state_for_execution(state, dynamics):
    """Wrap theta and clamp velocities/omega to the training domain.

    Matches the bounds used by dynamics.equivalent_wrapped_state (called
    inside MPC planning rollouts), ensuring execution states stay within
    the neural network's training distribution.

    Returns (state, was_clipped) where *was_clipped* is True if any
    velocity or omega component was outside bounds before clamping.
    """
    sr = dynamics.state_test_range()
    was_clipped = (
        state[2] < sr[2][0] or state[2] > sr[2][1]
        or state[3] < sr[3][0] or state[3] > sr[3][1]
        or state[5] < sr[5][0] or state[5] > sr[5][1]
    )
    state[4] = np.arctan2(np.sin(state[4]), np.cos(state[4]))
    state[2] = np.clip(state[2], sr[2][0], sr[2][1])
    state[3] = np.clip(state[3], sr[3][0], sr[3][1])
    state[5] = np.clip(state[5], sr[5][0], sr[5][1])
    return state, was_clipped


from utils.controllers.safety_filter import SafetyFilter
from utils.controllers.brat_controller import BRATController
from utils.controllers.mpc_controller import MPCController
from utils.controllers.mpc_terminal_controller import MPCTerminalController

# Grid-based HJ controller (requires jax, hj_reachability, cvxpy)
try:
    from utils.controllers.grid_based_controller import GridBasedController
except ImportError:
    GridBasedController = None

# 13D controllers
from utils.controllers.brt_controller_13d import BRTController13D
from utils.controllers.mpc_controller_13d import MPCController13D
from utils.controllers.mpc_terminal_controller_13d import MPCTerminalController13D

# 13D static plots
from utils.controllers.static_plots_13d import (
    plot_trajectory_13d, plot_states_13d, plot_controls_13d,
)

# 13D trajectory-only animation (MPC without BRT)
from utils.controllers.trajectory_only_animation_13d import TrajectoryAnimation13D

# Vanilla BRAT ablation baseline
from utils.controllers.vanilla_brat_controller import VanillaBRATController
from utils.controllers.vanilla_brt_controller_13d import VanillaBRTController13D

# RL controllers
try:
    from utils.controllers.rl_controller import RLController
    from utils.controllers.rl_controller_13d import RLController13D
except ImportError:
    RLController = None
    RLController13D = None

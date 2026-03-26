"""
Vanilla BRAT Controller -- ablation baseline.

Uses the exact same two-phase control framework as BRATController but
loads a value function trained with vanilla DeepReach (no MPC supervision,
no gradient refinement, no exact boundary alignment).

Comparing this against the full BRAT controller isolates the contribution
of training-pipeline improvements.
"""

from utils.controllers.brat_controller import BRATController


class VanillaBRATController(BRATController):
    """BRATController backed by a vanilla-DeepReach value function."""

    def simulate_docking(self, initial_state, max_sim_time, dynamics_fn=None):
        result = super().simulate_docking(initial_state, max_sim_time, dynamics_fn)
        result['controller_type'] = 'vanilla_brat'
        return result

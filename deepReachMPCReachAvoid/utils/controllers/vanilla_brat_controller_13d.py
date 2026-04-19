"""
Vanilla BRAT Controller for 13D -- ablation baseline.

Uses the exact same two-phase control framework as BRATController13D but
loads a value function trained with vanilla DeepReach (no MPC supervision,
no gradient refinement, no exact boundary alignment).

Comparing this against the full BRAT 13D controller isolates the contribution
of training-pipeline improvements.
"""

import torch
import numpy as np

from utils.controllers.brat_controller_13d import BRATController13D


class VanillaBRATController13D(BRATController13D):
    """BRATController13D backed by a vanilla-DeepReach value function.

    Overrides gradient/value queries to handle the vanilla DeepReach model
    whose ``io_to_dv`` does not add a boundary-function Jacobian term.
    With a single-sample input the base class ``output.squeeze()`` collapses
    the batch dimension, causing ``io_to_dv`` to return a 1-D tensor.
    Using ``squeeze(-1)`` instead preserves the batch dimension.
    """

    def get_value(self, state, time_query):
        if isinstance(state, np.ndarray):
            state = torch.tensor(state, dtype=torch.float32)
        state = state.to(self.device)
        if state.dim() == 1:
            state = state.unsqueeze(0)

        time_tensor = torch.tensor([[time_query]], dtype=torch.float32,
                                   device=self.device)
        coord = torch.cat([time_tensor, state], dim=-1)
        model_input = self.dynamics.coord_to_input(coord)

        with torch.no_grad():
            result = self.model({'coords': model_input})
            output = result['model_out'].squeeze(-1)

        return self.dynamics.io_to_value(model_input, output).item()

    def get_gradient(self, state, time_query):
        if isinstance(state, np.ndarray):
            state = torch.tensor(state, dtype=torch.float32)
        state = state.to(self.device)
        if state.dim() == 1:
            state = state.unsqueeze(0)

        time_tensor = torch.tensor([[time_query]], dtype=torch.float32,
                                   device=self.device)
        coord = torch.cat([time_tensor, state], dim=-1)
        model_input = self.dynamics.coord_to_input(coord)

        result = self.model({'coords': model_input})
        output = result['model_out'].squeeze(-1)
        model_in = result['model_in']

        dv = self.dynamics.io_to_dv(model_in, output)
        dvds = dv[0, 1:].detach().cpu().numpy()
        return dvds

    def simulate_docking(self, initial_state, max_sim_time, dynamics_fn=None):
        result = super().simulate_docking(initial_state, max_sim_time, dynamics_fn)
        result['controller_type'] = 'vanilla_brat_13d'
        return result

    def simulate_docking_batch(self, initial_states_np, max_sim_time):
        """Batch simulation — delegates to BRATController13D and fixes the label.

        The base-class batch methods already use squeeze(-1) throughout, so
        they are correct for the vanilla model even at batch-size 1.
        """
        results = super().simulate_docking_batch(initial_states_np, max_sim_time)
        for r in results:
            r['controller_type'] = 'vanilla_brat_13d'
        return results

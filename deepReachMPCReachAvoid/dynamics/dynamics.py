from abc import ABC, abstractmethod
from utils import diff_operators, quaternion

import math
import torch
import numpy as np
from multiprocessing import Pool
import torch.nn as nn
import scipy.io as spio
# during training, states will be sampled uniformly by each state dimension from the model-unit -1 to 1 range (for training stability),
# which may or may not correspond to proper test ranges
# note that coord refers to [time, *state], and input refers to whatever is fed directly to the model (often [time, *state, params])
# in the future, code will need to be fixed to correctly handle parametrized models


class Dynamics(ABC):
    def __init__(self,
                 name: str, loss_type: str, set_mode: str,
                 state_dim: int, input_dim: int,
                 control_dim: int, disturbance_dim: int,
                 state_mean: list, state_var: list,
                 value_mean: float, value_var: float, value_normto: float,
                 deepReach_model: bool):
        self.name = name
        self.loss_type = loss_type
        self.set_mode = set_mode
        self.state_dim = state_dim
        self.input_dim = input_dim
        self.control_dim = control_dim
        self.disturbance_dim = disturbance_dim
        self.state_mean = torch.tensor(state_mean)
        self.state_var = torch.tensor(state_var)
        self.value_mean = value_mean
        self.value_var = value_var
        self.value_normto = value_normto
        self.deepReach_model = deepReach_model

        assert self.loss_type in [
            'brt_hjivi', 'brat_hjivi'], f'loss type {self.loss_type} not recognized'
        if self.loss_type == 'brat_hjivi':
            assert callable(self.reach_fn) and callable(self.avoid_fn)
        assert self.set_mode in [
            'reach', 'avoid', 'reach_avoid'], f'set mode {self.set_mode} not recognized'
        for state_descriptor in [self.state_mean, self.state_var]:
            assert len(state_descriptor) == self.state_dim, 'state descriptor dimension does not equal state dimension, ' + \
                str(len(state_descriptor)) + ' != ' + str(self.state_dim)

    # ALL METHODS ARE BATCH COMPATIBLE

    # set deepreach model. choices: "vanilla" (vanilla DeepReach V=NN(x,t)), diff (diff model V=NN(x,t) + l(x)), exact ( V=NN(x,t) + l(x))
    def set_model(self, deepreach_model):
        self.deepReach_model = deepreach_model

    def override_state_range(self, state_range):
        """Override normalization parameters with a new state_range.

        This updates state_range_, state_mean, and state_var consistently.
        Must be called BEFORE any model queries so that coord_to_input /
        io_to_value use the correct normalization.

        Args:
            state_range: (state_dim, 2) list or tensor of [[lo, hi], ...].
        """
        if isinstance(state_range, list):
            state_range = torch.tensor(state_range, dtype=torch.float32)
        self.state_range_ = state_range.cuda()
        self.state_mean = ((self.state_range_[:, 0] + self.state_range_[:, 1]) / 2.0).cpu()
        self.state_var  = ((self.state_range_[:, 1] - self.state_range_[:, 0]) / 2.0).cpu()
        if hasattr(self, '_update_v_max'):
            self._update_v_max()

    # MODEL-UNIT CONVERSIONS
    # convert model input (normalized) to real coord
    def input_to_coord(self, input):
        coord = input.clone()
        coord[..., 1:] = (input[..., 1:] * self.state_var.to(device=input.device)
                          ) + self.state_mean.to(device=input.device)
        return coord

    # convert real coord to model input
    def coord_to_input(self, coord):
        input = coord*1.0
        input[..., 1:] = (coord[..., 1:] - self.state_mean.to(device=coord.device)
                          ) / self.state_var.to(device=coord.device)
        return input

    # convert model io to real value
    def io_to_value(self, input, output):
        if self.deepReach_model == 'diff':
            return (output * self.value_var / self.value_normto) + self.boundary_fn(self.input_to_coord(input)[..., 1:])
        elif self.deepReach_model == 'exact':
            return (output * input[..., 0] * self.value_var / self.value_normto) + self.boundary_fn(self.input_to_coord(input)[..., 1:])
        elif self.deepReach_model == 'exact_diff':
            # V(x,t)= l(x) + NN(x,t) - NN(x,0)
            # print(input.shape, output.shape)
            output0 = output[0].squeeze(dim=-1)
            output1 = output[1].squeeze(dim=-1)
            return (output0 - output1) * self.value_var / self.value_normto + self.boundary_fn(self.input_to_coord(input[0].detach())[..., 1:])
        else:
            return (output * self.value_var / self.value_normto) + self.value_mean

    # convert model io to real dv
    def io_to_dv(self, input, output):
        if self.deepReach_model == 'exact_diff':

            dodi1 = diff_operators.jacobian(
                output[0], input[0])[0].squeeze(dim=-2)
            dodi2 = diff_operators.jacobian(
                output[1], input[1])[0].squeeze(dim=-2)

            dvdt = (self.value_var / self.value_normto) * dodi1[..., 0]

            dvds_term1 = (self.value_var / self.value_normto /
                          self.state_var.to(device=dodi1.device)) * (dodi1[..., 1:]-dodi2[..., 1:])

            state = self.input_to_coord(input[0])[..., 1:]
            dvds_term2 = diff_operators.jacobian(self.boundary_fn(
                state).unsqueeze(dim=-1), state)[0].squeeze(dim=-2)
            dvds = dvds_term1 + dvds_term2
            return torch.cat((dvdt.unsqueeze(dim=-1), dvds), dim=-1)

        dodi = diff_operators.jacobian(
            output.unsqueeze(dim=-1), input)[0].squeeze(dim=-2)

        if self.deepReach_model == 'diff':
            dvdt = (self.value_var / self.value_normto) * dodi[..., 0]

            dvds_term1 = (self.value_var / self.value_normto /
                          self.state_var.to(device=dodi.device)) * dodi[..., 1:]
            state = self.input_to_coord(input)[..., 1:]
            dvds_term2 = diff_operators.jacobian(self.boundary_fn(
                state).unsqueeze(dim=-1), state)[0].squeeze(dim=-2)
            dvds = dvds_term1 + dvds_term2

        elif self.deepReach_model == 'exact':

            dvdt = (self.value_var / self.value_normto) * \
                (input[..., 0]*dodi[..., 0] + output)

            dvds_term1 = (self.value_var / self.value_normto /
                          self.state_var.to(device=dodi.device)) * dodi[..., 1:] * input[..., 0].unsqueeze(-1)
            state = self.input_to_coord(input)[..., 1:]
            dvds_term2 = diff_operators.jacobian(self.boundary_fn(
                state).unsqueeze(dim=-1), state)[0].squeeze(dim=-2)
            dvds = dvds_term1 + dvds_term2
        else:
            dvdt = (self.value_var / self.value_normto) * dodi[..., 0]
            dvds = (self.value_var / self.value_normto /
                    self.state_var.to(device=dodi.device)) * dodi[..., 1:]

        return torch.cat((dvdt.unsqueeze(dim=-1), dvds), dim=-1)

    # convert model io to real dv
    def io_to_2nd_derivative(self, input, output):
        hes = diff_operators.batchHessian(
            output.unsqueeze(dim=-1), input)[0].squeeze(dim=-2)

        if self.deepReach_model == 'diff':
            vis_term1 = (self.value_var / self.value_normto /
                         self.state_var.to(device=hes.device))**2 * hes[..., 1:]
            state = self.input_to_coord(input)[..., 1:]
            vis_term2 = diff_operators.batchHessian(self.boundary_fn(
                state).unsqueeze(dim=-1), state)[0].squeeze(dim=-2)
            hes = vis_term1 + vis_term2

        else:
            hes = (self.value_var / self.value_normto /
                   self.state_var.to(device=hes.device))**2 * hes[..., 1:]

        return hes

    def clamp_control(self, state, control):
        return control

    def clamp_state_input(self, state_input):
        return state_input

    def clamp_verification_state(self, state):
        return state
    # ALL FOLLOWING METHODS USE REAL UNITS

    @abstractmethod
    def periodic_transform_fn(self, input):
        raise NotImplementedError

    @abstractmethod
    def state_test_range(self):
        raise NotImplementedError

    @abstractmethod
    def equivalent_wrapped_state(self, state):
        raise NotImplementedError

    @abstractmethod
    def dsdt(self, state, control, disturbance):
        raise NotImplementedError

    @abstractmethod
    def boundary_fn(self, state):
        raise NotImplementedError

    @abstractmethod
    def sample_target_state(self, num_samples):
        raise NotImplementedError

    def sample_target_state_refined(self, num_samples):
        return self.sample_target_state(num_samples)

    @abstractmethod
    def cost_fn(self, state_traj):
        raise NotImplementedError

    @abstractmethod
    def hamiltonian(self, state, dvds):
        raise NotImplementedError

    @abstractmethod
    def optimal_control(self, state, dvds):
        raise NotImplementedError

    @abstractmethod
    def optimal_disturbance(self, state, dvds):
        raise NotImplementedError

    @abstractmethod
    def plot_config(self):
        raise NotImplementedError


class VertDrone2D(Dynamics):
    def __init__(self):
        self.gravity = 9.8                             # g
        self.input_multiplier = 12.0   # K
        self.input_magnitude_max = 1.0     # u_max
        self.state_range_ = torch.tensor(
            [[-4, 4], [-0.5, 3.5]]).cuda()  # v, z, k
        self.control_range_ = torch.tensor(
            [[-self.input_magnitude_max, self.input_magnitude_max]]).cuda()
        self.eps_var = torch.tensor([2]).cuda()
        self.control_init = torch.ones(
            1).cuda()*self.gravity/self.input_multiplier

        state_mean_ = (self.state_range_[:, 0]+self.state_range_[:, 1])/2.0
        state_var_ = (self.state_range_[:, 1]-self.state_range_[:, 0])/2.0

        super().__init__(
            name='VertDrone2D', loss_type='brt_hjivi', set_mode='avoid',
            # input_dim of the NN = state_dim + 1 (time dim)
            state_dim=2, input_dim=3,
            control_dim=1, disturbance_dim=0,
            state_mean=state_mean_.cpu().tolist(),
            state_var=state_var_.cpu().tolist(),
            # we estimate the ground-truth value function to be within [-0.5, 1.5] w.r.t. the state_range_ we used
            value_mean=0.5,
            # Then value_mean = 0.5*(-0.5 + 1.5) and value_max = 0.5*(1.5 - -0.5)
            value_var=1,
            value_normto=0.02,  # Don't need any changes
            deepReach_model='exact',  # chioce ['vanilla', 'exact'],
        )

    def control_range(self, state):
        return [[-self.input_magnitude_max, self.input_magnitude_max]]

    def state_test_range(self):
        return self.state_range_.cpu().tolist()

    def state_verification_range(self):
        return self.state_range_.cpu().tolist()
        # Here we verify the training results using the training range itself, we can verify on a smaller range for "stiff" systems

    def equivalent_wrapped_state(self, state):
        wrapped_state = torch.clone(state)
        return wrapped_state

    def periodic_transform_fn(self, input):
        return input.cuda()

    # ParameterizedVertDrone2D dynamics
    # \dot v = k*u - g
    # \dot z = v
    def dsdt(self, state, control, disturbance):
        dsdt = torch.zeros_like(state)
        dsdt[..., 0] = self.input_multiplier * control[..., 0] - self.gravity
        dsdt[..., 1] = state[..., 0]
        return dsdt

    def boundary_fn(self, state):
        # distance to ground (0m) and ceiling (3m)
        return -torch.abs(state[..., 1] - 1.5) + 1.5

    def sample_target_state(self, num_samples):
        raise NotImplementedError

    def cost_fn(self, state_traj):
        return torch.min(self.boundary_fn(state_traj), dim=-1).values

    def hamiltonian(self, state, dvds):
        return torch.abs(self.input_multiplier * dvds[..., 0]) * self.input_magnitude_max \
            - dvds[..., 0] * self.gravity \
            + dvds[..., 1] * state[..., 0]

    def optimal_control(self, state, dvds):
        return torch.sign(dvds[..., 0])[..., None]

    def optimal_disturbance(self, state, dvds):
        return torch.tensor([0])

    def plot_config(self):
        return {
            'state_slices': [0, 1.5],
            'state_labels': ['v', 'z'],
            'x_axis_idx': 0,  # which dim you want it to be the
            'y_axis_idx': 1,
            'z_axis_idx': -1,  # because there is only 2D
        }


class ParameterizedVertDrone2D(Dynamics):
    def __init__(self, gravity: float, input_multiplier: float, input_magnitude_max: float):
        self.gravity = gravity                             # g
        self.input_multiplier = input_multiplier   # k_max
        self.input_magnitude_max = input_magnitude_max     # u_max
        self.state_range_ = torch.tensor(
            [[-4, 4], [-0.5, 3.5], [0, self.input_multiplier]]).cuda()  # v, z, k
        self.control_range_ = torch.tensor(
            [[-self.input_magnitude_max, self.input_magnitude_max]]).cuda()
        self.eps_var = torch.tensor([2]).cuda()
        self.control_init = torch.ones(1).cuda()*gravity/input_multiplier

        state_mean_ = (self.state_range_[:, 0]+self.state_range_[:, 1])/2.0
        state_var_ = (self.state_range_[:, 1]-self.state_range_[:, 0])/2.0

        super().__init__(
            name='ParameterizedVertDrone2D', loss_type='brt_hjivi', set_mode='avoid',
            state_dim=3, input_dim=4, control_dim=1, disturbance_dim=0,
            state_mean=state_mean_.cpu().tolist(),
            state_var=state_var_.cpu().tolist(),
            value_mean=0.5,
            value_var=1,
            value_normto=0.02,
            deepReach_model='exact',  # chioce ['vanilla', 'exact'],
        )

    def control_range(self, state):
        return [[-self.input_magnitude_max, self.input_magnitude_max]]

    def state_test_range(self):
        return self.state_range_.cpu().tolist()

    def state_verification_range(self):
        return self.state_range_.cpu().tolist()

    def equivalent_wrapped_state(self, state):
        wrapped_state = torch.clone(state)
        return wrapped_state

    def periodic_transform_fn(self, input):
        return input.cuda()

    # ParameterizedVertDrone2D dynamics
    # \dot v = k*u - g
    # \dot z = v
    # \dot k = 0
    def dsdt(self, state, control, disturbance):
        dsdt = torch.zeros_like(state)
        dsdt[..., 0] = state[..., 2] * control[..., 0] - self.gravity
        dsdt[..., 1] = state[..., 0]
        dsdt[..., 2] = 0
        return dsdt

    def boundary_fn(self, state):
        return -torch.abs(state[..., 1] - 1.5) + 1.5

    def sample_target_state(self, num_samples):
        raise NotImplementedError

    def cost_fn(self, state_traj):
        return torch.min(self.boundary_fn(state_traj), dim=-1).values

    def hamiltonian(self, state, dvds):
        return torch.abs(state[..., 2] * dvds[..., 0]) * self.input_magnitude_max \
            - dvds[..., 0] * self.gravity \
            + dvds[..., 1] * state[..., 0]

    def optimal_control(self, state, dvds):
        return torch.sign(dvds[..., 0])[..., None]

    def optimal_disturbance(self, state, dvds):
        return torch.tensor([0])

    def plot_config(self):
        return {
            'state_slices': [0, 1.5, self.input_multiplier],
            'state_labels': ['v', 'z', 'k'],
            'x_axis_idx': 0,
            'y_axis_idx': 1,
            'z_axis_idx': 2,
        }


class Dubins3D(Dynamics):
    def __init__(self, set_mode: str):
        self.goalR = 0.5
        self.velocity = 1.
        self.omega_max = 1.2
        self.state_range_ = torch.tensor(
            [[-1, 1], [-1, 1], [-math.pi, math.pi]]).cuda()
        self.control_range_ = torch.tensor(
            [[-self.omega_max, self.omega_max]]).cuda()
        self.eps_var = torch.tensor([1]).cuda()
        self.control_init = torch.zeros(1).cuda()
        self.set_mode = set_mode
        self.avoid_fn_weight = 1.
        self.reach_fn_weight = 1.
        self.obstacles = [[0., 0.6, 0.1], [0.4, -0.8, 0.15] , [-0.6, -0.5, 0.2], [-0.7, 0 ,0.3]]
        if set_mode == 'reach_avoid':
            l_type = 'brat_hjivi'
        else:
            l_type = 'brt_hjivi'

        state_mean_ = (self.state_range_[:, 0]+self.state_range_[:, 1])/2.0
        state_var_ = (self.state_range_[:, 1]-self.state_range_[:, 0])/2.0
        super().__init__(
            name="Dubins3D", loss_type=l_type, set_mode=set_mode,
            state_dim=3, input_dim=5, control_dim=1, disturbance_dim=0,
            state_mean=state_mean_.cpu().tolist(),
            state_var=state_var_.cpu().tolist(),
            value_mean=0.5,
            value_var=1,
            value_normto=0.02,
            deepReach_model='exact'
        )

    def control_range(self, state):
        return [[-self.omega_max, self.omega_max]]

    def state_test_range(self):
        return self.state_range_.cpu().tolist()

    def state_verification_range(self):
        return self.state_range_.cpu().tolist()

    def equivalent_wrapped_state(self, state):
        wrapped_state = torch.clone(state)
        wrapped_state[..., 2] = (
            wrapped_state[..., 2] + math.pi) % (2 * math.pi) - math.pi
        return wrapped_state

    def periodic_transform_fn(self, input):
        output_shape = list(input.shape)
        output_shape[-1] = output_shape[-1]+1
        transformed_input = torch.zeros(output_shape)
        transformed_input[..., :3] = input[..., :3]
        transformed_input[..., 3] = torch.sin(input[..., 3]*self.state_var[-1])
        transformed_input[..., 4] = torch.cos(input[..., 3]*self.state_var[-1])
        return transformed_input.cuda()

    # Dubins3D dynamics
    # \dot x    = v \cos \theta
    # \dot y    = v \sin \theta
    # \dot \theta = u

    def dsdt(self, state, control, disturbance):
        dsdt = torch.zeros_like(state)
        dsdt[..., 0] = self.velocity * torch.cos(state[..., 2])
        dsdt[..., 1] = self.velocity * torch.sin(state[..., 2])
        dsdt[..., 2] = control[..., 0]
        return dsdt

    def reach_fn(self, state):
        state_ = state*1.0
        state_[..., 0] = state_[..., 0] - 0.
        state_[..., 1] = state_[..., 1]
        return (torch.norm(state[..., :2], dim=-1)-0.3)*self.reach_fn_weight

    def avoid_fn(self, state):
        # TODO: can speed it up by batch computation 
        avoid_val = torch.ones_like(state[...,-1])*torch.finfo().max
        for obstacle in self.obstacles:
            rel_state=state[...,:2]*1.0
            rel_state[...,0]-=obstacle[0]
            rel_state[...,1]-=obstacle[1]
            avoid_val=torch.minimum(avoid_val, (torch.norm(rel_state, dim=-1)-obstacle[2]))
        return self.avoid_fn_weight* avoid_val

    def boundary_fn(self, state):
        if self.set_mode in ['avoid', 'reach'] :
            return torch.norm(state[..., :2], dim=-1) - 0.5
        else:
            return torch.maximum(self.reach_fn(state), -self.avoid_fn(state))

    def sample_target_state(self, num_samples):
        raise NotImplementedError

    def cost_fn(self, state_traj):
        return torch.min(self.boundary_fn(state_traj), dim=-1).values

    def hamiltonian(self, state, dvds):
        if self.set_mode == "avoid":
            return self.velocity * (torch.cos(state[..., 2]) * dvds[..., 0] + torch.sin(state[..., 2]) * dvds[..., 1]) + self.omega_max * torch.abs(dvds[..., 2])
        elif self.set_mode in ['reach', 'reach_avoid']:
            return self.velocity * (torch.cos(state[..., 2]) * dvds[..., 0] + torch.sin(state[..., 2]) * dvds[..., 1]) - self.omega_max * torch.abs(dvds[..., 2])
        else:
            raise NotImplementedError

    def optimal_control(self, state, dvds):
        if self.set_mode == "avoid":
            return (self.omega_max * torch.sign(dvds[..., 2]))[..., None]
        elif self.set_mode in ['reach', 'reach_avoid']:
            return -(self.omega_max * torch.sign(dvds[..., 2]))[..., None]
        else:
            raise NotImplementedError

    def optimal_disturbance(self, state, dvds):
        return 0

    def plot_config(self):
        return {
            'state_slices': [0, 0, 0],
            'state_labels': ['x', 'y', r'$\theta$'],
            'x_axis_idx': 0,
            'y_axis_idx': 1,
            'z_axis_idx': 2,
        }


class Docking6D(Dynamics):
    # Fraction of state_range width used for Tier 4 broad uniform target sampling
    tier4_fraction = 0.15

    def __init__(self, set_mode: str, state_range=None, goal_state=None,
                 eps_p: float = None, eps_v: float = None,
                 eps_theta: float = None, eps_omega: float = None):
        """
        Args:
            set_mode:    'reach_avoid' (only supported mode).
            state_range: Optional (6,2) list overriding the default state bounds.
                         Used to restore training-time normalization when loading
                         a model trained with different bounds.
            goal_state:  Optional 6-element list overriding the default goal.
            eps_p:       Optional position tolerance override.
            eps_v:       Optional velocity tolerance override.
            eps_theta:   Optional angular position tolerance override.
            eps_omega:   Optional angular velocity tolerance override.
        """
        # Defining dynamic parameters
        self.orbit_alt = 400  # Orbital altitude (km)
        self.u_bar = 20.0  # Maximum control input
        self.u_theta_bar = 1.5  # Maximum control input for angular velocity

        # Define Chaser spacecraft (Planar)
        self.mc = 200    # Mass of chaser spacecraft (kg)
        self.w_c = 1.0  # width of chaser spacecraft (m) (along x-axis)
        self.h_c = 1.0  # height of chaser spacecraft (m) (along y-axis)
        self.chaser_buffer = np.sqrt(self.w_c**2 + self.h_c**2)/2 # Defining buffer to account for chaser spacecraft dimensions
        
        # Calculate derived parameters
        self.jc = self.moment_of_inertia()    # Moment of inertia of chaser spacecraft (kg*m^2)
        self.n = self.mean_motion()      # Mean motion (assuming circular orbit) (rad/s)

        # Docking tolerances (use overrides when provided, else defaults)
        self.eps_p     = eps_p     if eps_p     is not None else 0.1    # Position (m)
        self.eps_v     = eps_v     if eps_v     is not None else 0.1    # Velocity (m/s)
        self.eps_theta = eps_theta if eps_theta is not None else 0.04   # Angle (rad)
        self.eps_omega = eps_omega if eps_omega is not None else 0.05   # Angular vel (rad/s)

        # Values for 3 second inner controller
        #self.eps_p = 0.1 
        #self.eps_v = 0.05 
        #self.eps_theta = 0.05 
        #self.eps_omega = 0.0035 

        # Goal state (use override when provided, else default)
        if goal_state is not None:
            self.goal_state = torch.tensor(goal_state, dtype=torch.float32)
        else:
            self.goal_state = torch.tensor([0.0, 0.0, 0.0, 0.0, np.pi/2, 0.0])

        # Define target spacecraft (Planar)
        self.w_t = 6  # width of target spacecraft (m) (along x-axis)
        self.h_t = 3  # height of target spacecraft (m) (along y-axis)
        self.dock_rad = 1.5 # Radius of target spacecraft docking indentation (m)

        # BRAT parameters
        self.reach_fn_weight = 5.0
        self.avoid_fn_weight = 10.0
        self.set_mode = set_mode

        if set_mode == 'reach_avoid':
            l_type = 'brat_hjivi'
        else:
            raise NotImplementedError('Only reach_avoid mode is implemented for Docking6D')

        # look into what we want to make these
        #self.eps_var = torch.tensor([3]).cuda() 
        self.eps_var = torch.tensor([20, 20, 1.5]).cuda()
        self.control_init = torch.zeros(3).cuda()
        
        # Define state/control space (use override when provided, else default)
        self.state_dim = 6
        if state_range is not None:
            self.state_range_ = torch.tensor(state_range, dtype=torch.float32).cuda()
        else:
            #[[-4, 4], [-4, 4], [-1.0 , 1.0], [-1.0 , 1.0], [-math.pi, math.pi], [-0.75, 0.75]] used for 3 second inner controller
            self.state_range_ = torch.tensor(
                [[-15, 15], [-15, 15], [-1.5 , 1.5], [-1.5 , 1.5], [-math.pi, math.pi], [-1.0, 1.0]]).cuda()
        self.control_range_ = torch.tensor(
            [[-self.u_bar, self.u_bar], [-self.u_bar, self.u_bar], [-self.u_theta_bar, self.u_theta_bar]]).cuda()
        
        # Calculate state mean/var for normalization
        state_mean_ = (self.state_range_[:, 0]+self.state_range_[:, 1])/2.0
        state_var_ = (self.state_range_[:, 1]-self.state_range_[:, 0])/2.0
        self._update_v_max()  # Derive v_max from velocity bounds in state_range_

        # Define an MPC cost weight matrix for these dynamics
        # MPC cost: sum of stage costs + terminal cost
        # MPC weight matrix
        self.Q = torch.eye(self.state_dim)
        self.Q[0,0] = 3.0
        self.Q[1,1] = 3.0
        self.Q[2,2] = 10.0 
        self.Q[3,3] = 10.0 
        self.Q[4,4] = 5.0 
        self.Q[5,5] = 5.0 

        super().__init__(
            name="Docking6D",
            loss_type=l_type, 
            set_mode=set_mode,
            state_dim=6,
            input_dim=8,  # 6 states + 1 time + 1 for periodic transform
            control_dim=3, 
            disturbance_dim=0,
            state_mean=state_mean_.cpu().tolist(),
            state_var=state_var_.cpu().tolist(),
            value_mean=0.5, # Normalization (based off of problem) 
            value_var=1, # look into expected maximum V and V
            value_normto=0.02,
            deepReach_model='exact'
        )

    def mean_motion(self):
        """Calculate the mean motion based on the orbital altitude."""
        mu = torch.tensor(3.986004418e14)  # Gravitational parameter for Earth (m^3/s^2)
        r_earth = 6371e3  # Radius of Earth (m)
        r = r_earth + (self.orbit_alt * 1e3)
        return torch.sqrt(mu / r**3)

    def moment_of_inertia(self):
        """Calculate the moment of inertia for the chaser spacecraft."""
        # Assuming a rectangular shape for the chaser spacecraft
        return 1.0/12.0 * (self.mc * (self.w_c**2 + self.h_c**2)) 

    def control_range(self, state):
        return [[-self.u_bar, self.u_bar], [-self.u_bar, self.u_bar], [-self.u_theta_bar, self.u_theta_bar]]

    def state_test_range(self):
        return self.state_range_.cpu().tolist()

    def state_verification_range(self):
        return self.state_range_.cpu().tolist()

    def equivalent_wrapped_state(self, state):
        wrapped_state = torch.clone(state)
        # Wrap theta to [-π, π]
        wrapped_state[..., 4] = (
            wrapped_state[..., 4] + math.pi) % (2 * math.pi) - math.pi
        # Clamp velocities to their bounds to prevent out-of-distribution states
        wrapped_state[..., 2] = torch.clamp(wrapped_state[..., 2], 
                                            self.state_range_[2, 0].item(), 
                                            self.state_range_[2, 1].item())  # vx
        wrapped_state[..., 3] = torch.clamp(wrapped_state[..., 3], 
                                            self.state_range_[3, 0].item(), 
                                            self.state_range_[3, 1].item())  # vy
        wrapped_state[..., 5] = torch.clamp(wrapped_state[..., 5], 
                                            self.state_range_[5, 0].item(), 
                                            self.state_range_[5, 1].item())  # omega
        return wrapped_state

    def periodic_transform_fn(self, input):
        output_shape = list(input.shape)
        output_shape[-1] = output_shape[-1] + 1  # Add one more dimension for cos(theta)
        transformed_input = torch.zeros(output_shape)
        
        # Copy the first 5 elements: [time, x, y, vx, vy]
        transformed_input[..., :5] = input[..., :5]
        
        # Transform the periodic angle theta (at index 4 in state, index 5 in input)
        transformed_input[..., 5] = torch.sin(input[..., 5] * self.state_var[4])
        transformed_input[..., 6] = torch.cos(input[..., 5] * self.state_var[4])
        
        # Copy the remaining element: omega (at index 5 in state, index 6 in input)
        transformed_input[..., 7] = input[..., 6]
        
        return transformed_input.cuda()

    # Docking6D dynamics (CW equations + rotational dynamics)
    # \dot px = vx
    # \dot py = vy
    # \dot vx = 3*n^2*px + 2*n*vy + u_x/mc
    # \dot vy = -2*n*vx + u_y/mc
    # \dot \theta = \omega
    # \dot \omega = u_theta/jc

    def _update_v_max(self):
        """Derive v_max from the velocity bounds in state_range_ (indices 2, 3)."""
        self.v_max = float(self.state_range_[2:4, 1].max().item())

    def dsdt(self, state, control, disturbance):
        dsdt = torch.zeros_like(state)
        dsdt[..., 0] = state[..., 2]
        dsdt[..., 1] = state[..., 3]
        dsdt[..., 2] = 3 * self.n**2 * state[..., 0] + 2 * self.n * state[..., 3] + control[..., 0] / self.mc
        dsdt[..., 3] = -2 * self.n * state[..., 2] + control[..., 1] / self.mc

        current_vx = state[..., 2]
        current_vy = state[..., 3]

        # Clamp velocities to v_max
        dsdt[..., 2] = torch.where((current_vx >= self.v_max) & (dsdt[..., 2] > 0),
            torch.zeros_like(dsdt[..., 2]), dsdt[..., 2])
        dsdt[..., 2] = torch.where((current_vx <= -self.v_max) & (dsdt[..., 2] < 0),
            torch.zeros_like(dsdt[..., 2]), dsdt[..., 2])
        
        dsdt[..., 3] = torch.where((current_vy >= self.v_max) & (dsdt[..., 3] > 0),
            torch.zeros_like(dsdt[..., 3]), dsdt[..., 3])
        dsdt[..., 3] = torch.where((current_vy <= -self.v_max) & (dsdt[..., 3] < 0),
            torch.zeros_like(dsdt[..., 3]),dsdt[..., 3])

        dsdt[..., 4] = state[..., 5]
        dsdt[..., 5] = control[..., 2] / self.jc
        return dsdt

    # L2 Norm (exact)
    def reach_fn(self, state):
        """Signed distance <= 0 if within docking position/velocity tolerances using L2 norm."""
        px, py = state[..., 0], state[..., 1]
        vx, vy = state[..., 2], state[..., 3]
        theta, omega = state[..., 4], state[..., 5]
        
        goal_state = self.goal_state.to(state.device)

        # Extract goal state variables
        px_goal, py_goal = goal_state[0], goal_state[1]
        vx_goal, vy_goal = goal_state[2], goal_state[3]
        theta_goal, omega_goal = goal_state[4], goal_state[5]

        # L2 norm distances from goal for each component
        position_dist = torch.sqrt((px - px_goal)**2 + (py - py_goal)**2) - self.eps_p
        velocity_dist = torch.sqrt((vx - vx_goal)**2 + (vy - vy_goal)**2) - self.eps_v
        theta_dist = torch.abs(torch.atan2(torch.sin(theta - theta_goal), torch.cos(theta - theta_goal))) - self.eps_theta
        omega_dist = torch.abs(omega - omega_goal) - self.eps_omega

        position_dist = torch.where(position_dist < 0, position_dist * 15, position_dist * 0.2/2)
        velocity_dist = torch.where(velocity_dist < 0, velocity_dist * 15, velocity_dist * 2/2)
        theta_dist = torch.where(theta_dist < 0, theta_dist * 150, theta_dist * 0.2/2)
        omega_dist = torch.where(omega_dist < 0, omega_dist * 30, omega_dist * 2/2)

        # print("Position Max, Min:", position_dist.max(), ",", position_dist.min())
        # print("Velocity Max, Min:", velocity_dist.max(), ",", velocity_dist.min())
        # print("Theta    Max, Min:", theta_dist.max(), ",", theta_dist.min())
        # print("Omega    Max, Min:", omega_dist.max(), ",", omega_dist.min())

        # Maximum of all constraint violations (signed distance)
        goal = torch.stack([
            position_dist,
            velocity_dist,
            theta_dist,
            omega_dist], dim=-1)

        goal = torch.max(goal, axis=-1).values 

        #goal = torch.where(goal < 0, goal * 1000.0, goal * 0.02)
        return goal
    
    """ L inf norm for reach-avoid
    def reach_fn(self, state):
        # Signed distance <= 0 if within docking position/velocity tolerances.
        px = state[..., 0]
        py = state[..., 1]
        vx = state[..., 2]
        vy = state[..., 3]
        theta = state[..., 4]
        omega = state[..., 5]
        # Distance from allowable box in (px,py,vx,vy)
        goal = torch.stack([
            torch.abs(px) - self.eps_p,
            torch.abs(py) - self.eps_p,
            torch.abs(vx) - self.eps_v,
            torch.abs(vy) - self.eps_v,
            torch.abs(theta - np.pi/2) - self.eps_theta,
            torch.abs(omega) - self.eps_omega
        ], axis=-1)
        return (torch.max(goal, axis=-1))*self.reach_fn_weight """

    def avoid_fn(self, state):
        """Signed distance <= 0 if colliding with target body (rectangle) except bottom indentation."""
        px = state[..., 0]
        py = state[..., 1]

        # Target Spacecraft rectangular body (Including buffer for chaser dimensions)
        s_rect = torch.maximum(
            torch.abs(px) - (self.w_t/2 + self.chaser_buffer),
            torch.maximum(-(py + self.chaser_buffer), py - (self.h_t + self.chaser_buffer))
        )
        
        # Effective docking radius accounting for chaser buffer
        effective_rad = max(self.dock_rad - self.chaser_buffer, 1e-6)

        # Target Spacecraft docking indentation (semicircle)
        dist_semi = torch.sqrt(px**2 + py**2) - effective_rad
        s_dock = torch.maximum(-py, dist_semi)
        s_bubble = torch.maximum(s_rect, -s_dock)

        # Cutout to stop faliureset overlap
        s_cutout = torch.maximum(torch.abs(px) - effective_rad, 
                                 torch.maximum(-(py + self.chaser_buffer), py))

        s_fail = torch.maximum(s_bubble, -s_cutout + 0.02)

        s_fail[s_fail < 0] *= 0.750
        s_fail[s_fail > 0] *= 50.0

        return s_fail

    def check_collision_oriented(self, state):
        if isinstance(state, torch.Tensor):
            px = state[..., 0].item()
            py = state[..., 1].item()
            theta = state[..., 4].item()
        else:
            px, py = float(state[0]), float(state[1])
            theta = float(state[4])

        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        hw = self.w_c / 2.0   # half-width
        hh = self.h_c / 2.0   # half-height

        # 4 corners of the chaser in world frame
        corners = [
            (px + hw * cos_t - hh * sin_t, py + hw * sin_t + hh * cos_t),
            (px - hw * cos_t - hh * sin_t, py - hw * sin_t + hh * cos_t),
            (px - hw * cos_t + hh * sin_t, py - hw * sin_t - hh * cos_t),
            (px + hw * cos_t + hh * sin_t, py + hw * sin_t - hh * cos_t),
        ]

        for cx, cy in corners:
            if self._point_in_target_body(cx, cy):
                return True
        return False

    def _point_in_target_body(self, px, py):
        # Inside the target rectangle?
        if abs(px) > self.w_t / 2.0 or py < 0.0 or py > self.h_t:
            return False   # outside rectangle -> no collision

        # Inside the docking indentation (safe zone)?
        if px ** 2 + py ** 2 < self.dock_rad ** 2 and py >= 0.0:
            return False   # inside dock semicircle -> safe

        return True        # inside rectangle but outside dock -> collision

    def boundary_fn(self, state):
        if self.set_mode in ['reach_avoid']:
            return torch.maximum(self.reach_fn(state), -self.avoid_fn(state))
        else:
            raise NotImplementedError('Only reach_avoid mode is implemented for Docking6D')

    def sample_target_state(self, num_samples):
        """Multi-scale target sampling concentrated near the exact goal state.
        
        Tier 1 (5%):  Exact goal + tiny noise
        Tier 2 (35%): Gaussian around goal with tolerance-scale std 
        Tier 3 (20%): Boundary-focused sampling at 0.8-1.2x tolerance 
        Tier 4 (40%): Broader uniform sampling centered on goal 
        """
        samples = torch.zeros(num_samples, self.state_dim)
        idx = 0

        # Tier 1 (10%): Exact goal + tiny noise
        n_exact = int(num_samples * 0.1)
        noise_std = torch.tensor([
            self.eps_p * 0.1, self.eps_p * 0.1,     # position: 0.01m
            self.eps_v * 0.1, self.eps_v * 0.1,     # velocity: 0.01m/s
            self.eps_theta * 0.1,                     # theta: 0.001rad
            self.eps_omega * 0.1                      # omega: 0.005rad/s
        ])
        samples[idx:idx + n_exact] = self.goal_state.unsqueeze(0) + torch.randn(n_exact, self.state_dim) * noise_std
        idx += n_exact

        # Tier 2 (30%): Gaussian around goal with tolerance-scale std
        n_gaussian = int(num_samples * 0.3)
        goal_std = torch.tensor([
            self.eps_p * 2, self.eps_p * 2,
            self.eps_v * 2, self.eps_v * 2,
            self.eps_theta * 2,
            self.eps_omega * 2
        ])
        samples[idx:idx + n_gaussian] = self.goal_state.unsqueeze(0) + torch.randn(n_gaussian, self.state_dim) * goal_std
        idx += n_gaussian

        # Tier 3 (20%): Boundary-focused sampling at 0.8-1.2x tolerance
        # Samples on/near the reach set boundary where the value function transitions
        n_boundary = int(num_samples * 0.20)
        tolerances = torch.tensor([
            self.eps_p, self.eps_p,
            self.eps_v, self.eps_v,
            self.eps_theta,
            self.eps_omega
        ])
        # Each dimension independently at a random fraction of [0.8, 1.2]x its tolerance
        scale_factors = 0.8 + 0.4 * torch.rand(n_boundary, self.state_dim)
        # Random sign for each dimension (sample on both sides of goal)
        signs = torch.sign(torch.randn(n_boundary, self.state_dim))
        samples[idx:idx + n_boundary] = self.goal_state.unsqueeze(0) + signs * scale_factors * tolerances.unsqueeze(0)
        idx += n_boundary

        # Tier 4 (40%): Broader uniform sampling centered on goal_state
        # Half-width = fraction of state_range width, adapts when state_range is overridden
        n_broad = num_samples - idx
        sr = self.state_range_.cpu()
        state_widths = (sr[:, 1] - sr[:, 0])
        tier4_half_width = self.tier4_fraction * state_widths
        tier4_lo = torch.clamp(self.goal_state - tier4_half_width, sr[:, 0], sr[:, 1])
        tier4_hi = torch.clamp(self.goal_state + tier4_half_width, sr[:, 0], sr[:, 1])
        samples[idx:] = tier4_lo + torch.rand(n_broad, self.state_dim) * (tier4_hi - tier4_lo)

        return samples

    def sample_target_state_refined(self, num_samples):
        """Alias for sample_target_state (kept for backwards compatibility)."""
        return self.sample_target_state(num_samples)

    def cost_fn(self, state_traj):
        # Correct reach-avoid cost: min_t max{l(x(t)), max_{k<=t}{-g(x(k))}}
        # where l(x) is reach_fn (target set) and g(x) is avoid_fn (obstacle)
        reach_values = self.reach_fn(state_traj)
        avoid_values = self.avoid_fn(state_traj)
        return torch.min(torch.clamp(reach_values, min=torch.max(-avoid_values, dim=-1).values.unsqueeze(-1)), dim=-1).values
    
    def hamiltonian(self, state, dvds):
        if self.set_mode == 'reach_avoid':
            opt_control = self.optimal_control(state, dvds)
            dsdt_ = self.dsdt(state, opt_control, None)
            ham = torch.sum(dvds*dsdt_, dim=-1)

        else:
            raise NotImplementedError('Only reach_avoid mode is implemented for Docking6D')

        return ham
    
    """  Anylitical Hamiltonian for reach-avoid
    def hamiltonian(self, state, dvds):
        if self.set_mode == "reach_avoid":
            # Extract state variables
            px, py, vx, vy, theta, omega = state[..., 0], state[..., 1], state[..., 2], state[..., 3], state[..., 4], state[..., 5]
            
            # Extract costate variables
            dvds_px, dvds_py, dvds_vx, dvds_vy, dvds_theta, dvds_omega = dvds[..., 0], dvds[..., 1], dvds[..., 2], dvds[..., 3], dvds[..., 4], dvds[..., 5]
            
            # Drift terms (no control input)
            ham_drift = dvds_px * vx + dvds_py * vy + dvds_theta * omega
            ham_drift += dvds_vx * (3 * self.mean_motion()**2 * px + 2 * self.mean_motion() * vy)
            ham_drift += dvds_vy * (-2 * self.mean_motion() * vx)
            
            # Control terms (maximize for reach-avoid)
            ham_control = -torch.abs(dvds_vx / self.mc) * self.u_bar  # u_x control
            ham_control -= torch.abs(dvds_vy / self.mc) * self.u_bar  # u_y control  
            ham_control -= torch.abs(dvds_omega / self.moment_of_inertia()) * self.u_theta_bar  # u_theta control
            
            return ham_drift + ham_control
        else:
            raise NotImplementedError('Only reach_avoid mode is implemented for Docking6D') """
   
    def optimal_control(self, state, dvds):
        if self.set_mode == 'reach_avoid':
            # Extract the relevant costate variables for control
            dvds_vx = dvds[..., 2]    # ∂V/∂vx
            dvds_vy = dvds[..., 3]    # ∂V/∂vy  
            dvds_omega = dvds[..., 5] # ∂V/∂ω
            # Optimal bang-bang control 
            u_x = torch.where(dvds_vx > 0, -self.u_bar, self.u_bar)
            u_y = torch.where(dvds_vy > 0, -self.u_bar, self.u_bar)
            u_theta = torch.where(dvds_omega > 0, -self.u_theta_bar, self.u_theta_bar)
            
            return torch.stack([u_x, u_y, u_theta], dim=-1)
        else:
            raise NotImplementedError('Only reach_avoid mode is implemented for Docking6D')

    def optimal_disturbance(self, state, dvds):
        return 0

    def plot_config(self):
        return {
            'state_slices': [0, 0, 0, 0, np.pi/2, 0],
            'state_labels': ['x', 'y', 'vx', 'vy', r'$\theta$', r'$\omega$'],
            'x_axis_idx': 0,
            'y_axis_idx': 1,
            'z_axis_idx': 4,  # Use theta for the z-axis instead of vx
        }


class Docking13D(Dynamics):
    """
    13-state docking dynamics:
      - Relative translation in LVLH: (x,y,z,vx,vy,vz)
      - Chaser attitude: quaternion q (LVLH->Body), angular velocity omega (Body, relative to LVLH)
    Controls:
      - Body-frame force Fb (N): [Fx,Fy,Fz]

    # Fraction of state_range width used for Tier 4 broad uniform target sampling
    tier4_fraction = 0.15
      - Body-frame torque taub (N*m): [tx,ty,tz]
    """

    def __init__(self, set_mode: str):
        goal_state = None

        # Orbit / control bounds
        self.orbit_alt = 400  # km
        self.F_bar = 20.0     # max force per axis (N)  (ASSUMPTION: your old u_bar was in N)
        self.tau_bar = 1.5    # max torque per axis (N*m) (ASSUMPTION: your old u_theta_bar was torque)

        # Chaser spacecraft dimensions (now 3D)
        self.mc = 200.0
        self.w_c = 1.0  # x-dim (m)
        self.h_c = 1.0  # y-dim (m)
        self.d_c = 1.0  # z-dim (m)
        self.chaser_buffer_xy = math.sqrt(self.w_c**2 + self.h_c**2) / 2.0
        self.chaser_buffer_z = self.d_c / 2.0

        # Derived
        self.n = self.mean_motion()
        self.I = self.inertia_tensor_body()  # 3x3 torch tensor on CUDA later

        # Docking tolerances
        self.eps_p = 0.1
        self.eps_v = 0.1
        self.eps_q = 0.02       # radians, attitude error tolerance #TODO Validate this tolerance
        self.eps_omega = 0.05
        self.v_max = 2.5

        # Target geometry (still mainly planar in your code)
        self.w_t = 6.0
        self.h_t = 3.0
        self.d_t = 3.0          # target "thickness" in z (m) #TODO Validate this dimension
        self.dock_rad = 1.5

        # Goal: position/velocity zero, attitude = yaw(pi/2), omega=0 #TODO Validate goal attitude - currently 90 deg yaw and 0 roll/pitch
        self.q_goal = self.quat_from_axis_angle(
            torch.tensor([0.0, 0.0, 1.0]), math.pi / 2.0
        )

        if goal_state is None:
            # State: [x,y,z,vx,vy,vz,q0,q1,q2,q3,wx,wy,wz]
            self.goal_state = torch.tensor([
                0.0, 0.0, 0.0,
                0.0, 0.0, 0.0,
                float(self.q_goal[0]), float(self.q_goal[1]), float(self.q_goal[2]), float(self.q_goal[3]),
                0.0, 0.0, 0.0
            ])
        else:
            self.goal_state = torch.tensor(goal_state)

        # BRAT parameters
        self.reach_fn_weight = 5.0 #TODO Tune
        self.avoid_fn_weight = 10.0 #TODO Tune
        self.set_mode = set_mode

        if set_mode == 'reach_avoid':
            l_type = 'brat_hjivi'
        else:
            raise NotImplementedError('Only reach_avoid mode is implemented')

        # (HJ/DeepReach) ranges (ASSUMPTION: expand from planar; tune as needed)
        self.state_dim = 13
        self.control_dim = 6

        self.state_range_ = torch.tensor([
            [-15, 15],   # x
            [-15, 15],   # y
            [-15, 15],   # z
            [-2.0, 2.0],  # vx
            [-2.0, 2.0],  # vy
            [-2.0, 2.0],  # vz
            [-1, 1],   # q0
            [-1, 1],   # q1
            [-1, 1],   # q2
            [-1, 1],   # q3
            [-1.5, 1.5],  # wx
            [-1.5, 1.5],  # wy
            [-1.5, 1.5],  # wz
        ]).cuda()

        self.control_range_ = torch.tensor([
            [-self.F_bar, self.F_bar],   # Fx
            [-self.F_bar, self.F_bar],   # Fy
            [-self.F_bar, self.F_bar],   # Fz
            [-self.tau_bar, self.tau_bar],  # tx
            [-self.tau_bar, self.tau_bar],  # ty
            [-self.tau_bar, self.tau_bar],  # tz
        ]).cuda()

        # MPC initialization parameters
        self.eps_var = torch.tensor([self.F_bar, self.F_bar, self.F_bar, 
                                     self.tau_bar, self.tau_bar, self.tau_bar]).cuda()
        self.control_init = torch.zeros(6).cuda()  # Initial control guess for MPC

        # Normalization
        state_mean_ = (self.state_range_[:, 0] + self.state_range_[:, 1]) / 2.0
        state_var_  = (self.state_range_[:, 1] - self.state_range_[:, 0]) / 2.0

        # MPC weights (ASSUMPTION: quick extension; tune)
        self.Q = torch.eye(self.state_dim).cuda()
        # position
        self.Q[0,0] = 3.0; self.Q[1,1] = 3.0; self.Q[2,2] = 3.0
        # velocity
        self.Q[3,3] = 10.0; self.Q[4,4] = 10.0; self.Q[5,5] = 10.0
        # quaternion + omega (light penalties; docking precision via reach_fn)
        self.Q[6,6] = 1.0; self.Q[7,7] = 1.0; self.Q[8,8] = 1.0; self.Q[9,9] = 1.0
        self.Q[10,10] = 5.0; self.Q[11,11] = 5.0; self.Q[12,12] = 5.0

        super().__init__(
            name="Docking13D",
            loss_type=l_type,
            set_mode=set_mode,
            state_dim=self.state_dim,
            input_dim=self.state_dim + 1,   # states + time (removed periodic sin/cos)
            control_dim=self.control_dim,
            disturbance_dim=0,
            state_mean=state_mean_.cpu().tolist(),
            state_var=state_var_.cpu().tolist(),
            value_mean=0.5,
            value_var=1,
            value_normto=0.02,
            deepReach_model='exact'
        )

        # store for periodic transform usage
        self.state_mean = state_mean_
        self.state_var  = state_var_

        # Put inertia and q_goal on GPU
        self.I = self.I.cuda()
        self.q_goal = self.q_goal.cuda()

    # ---------- Orbital params ----------
    def mean_motion(self):
        mu = torch.tensor(3.986004418e14)  # m^3/s^2
        r_earth = 6371e3
        r = r_earth + (self.orbit_alt * 1e3)
        return torch.sqrt(mu / r**3)

    # ---------- Inertia ----------
    def inertia_tensor_body(self):
        # Rectangular box inertia about COM in body frame (diagonal)
        m = self.mc
        wx, hy, dz = self.w_c, self.h_c, self.d_c
        Ixx = (m / 12.0) * (hy**2 + dz**2)
        Iyy = (m / 12.0) * (wx**2 + dz**2)
        Izz = (m / 12.0) * (wx**2 + hy**2)
        return torch.diag(torch.tensor([Ixx, Iyy, Izz], dtype=torch.float32))

    # ---------- Quaternion utilities (scalar-first) ---------- #TODO: Move these into quaternion.py
    def quat_from_axis_angle(self, axis, angle):
        axis = axis / (torch.norm(axis) + 1e-12)
        half = 0.5 * angle
        q0 = torch.cos(torch.tensor(half))
        qv = axis * torch.sin(torch.tensor(half))
        return torch.tensor([q0, qv[0], qv[1], qv[2]], dtype=torch.float32)

    def quat_conj(self, q):
        return torch.stack([q[...,0], -q[...,1], -q[...,2], -q[...,3]], dim=-1)

    def quat_mul(self, q, p):
        # q ⊗ p
        q0,q1,q2,q3 = q[...,0],q[...,1],q[...,2],q[...,3]
        p0,p1,p2,p3 = p[...,0],p[...,1],p[...,2],p[...,3]
        return torch.stack([
            q0*p0 - q1*p1 - q2*p2 - q3*p3,
            q0*p1 + q1*p0 + q2*p3 - q3*p2,
            q0*p2 - q1*p3 + q2*p0 + q3*p1,
            q0*p3 + q1*p2 - q2*p1 + q3*p0
        ], dim=-1)

    def quat_to_R_LVLH_to_body(self, q):
        # q = [q0,q1,q2,q3] scalar-first
        q0,q1,q2,q3 = q[...,0],q[...,1],q[...,2],q[...,3]
        R = torch.zeros(q.shape[:-1] + (3,3), device=q.device, dtype=q.dtype)
        R[...,0,0] = 1 - 2*(q2*q2 + q3*q3)
        R[...,0,1] = 2*(q1*q2 + q0*q3)
        R[...,0,2] = 2*(q1*q3 - q0*q2)
        R[...,1,0] = 2*(q1*q2 - q0*q3)
        R[...,1,1] = 1 - 2*(q1*q1 + q3*q3)
        R[...,1,2] = 2*(q2*q3 + q0*q1)
        R[...,2,0] = 2*(q1*q3 + q0*q2)
        R[...,2,1] = 2*(q2*q3 - q0*q1)
        R[...,2,2] = 1 - 2*(q1*q1 + q2*q2)
        return R

    def quat_error_angle(self, q, q_goal):
        # angle of q_rel = q_goal^{-1} ⊗ q
        q_rel = self.quat_mul(self.quat_conj(q_goal), q)
        # shortest angle: 2*acos(|q0|)
        # Use atan2-based formula to avoid gradient explosion at q0=1
        # angle = 2*atan2(||q_vec||, |q0|) which is numerically stable
        q0 = torch.abs(q_rel[...,0])
        qv_norm = torch.sqrt(q_rel[...,1]**2 + q_rel[...,2]**2 + q_rel[...,3]**2 + 1e-8)
        return 2.0 * torch.atan2(qv_norm, q0)

    # ---------- Interface required by your base class ----------
    def control_range(self, state):
        return self.control_range_.cpu().tolist()

    def state_test_range(self):
        return self.state_range_.cpu().tolist()

    def state_verification_range(self):
        return self.state_range_.cpu().tolist()

    def equivalent_wrapped_state(self, state):
        # No periodic angle now; quaternion handles wrap inherently.
        # Optional: re-normalize quaternion here if you want.
        wrapped = torch.clone(state)
        q = wrapped[...,6:10]
        q = q / (torch.norm(q, dim=-1, keepdim=True) + 1e-12)
        wrapped[...,6:10] = q
        return wrapped

    def periodic_transform_fn(self, input):
        # Identity transform (no periodic theta anymore)
        return input.cuda()

    def sample_quat_near_goal(self, n, angle_std=0.15):
        """Sample quaternions near the goal quaternion with given angle std."""
        q_goal = self.q_goal.detach().cpu()
        axis = torch.randn(n, 3)
        axis = axis / (torch.norm(axis, dim=-1, keepdim=True) + 1e-12)
        angle = torch.randn(n) * angle_std
        half = 0.5 * angle
        q0 = torch.cos(half)
        qv = axis * torch.sin(half).unsqueeze(-1)
        q_delta = torch.cat([q0.unsqueeze(-1), qv], dim=-1)
        q_goal_rep = q_goal.unsqueeze(0).repeat(n, 1)
        q = self.quat_mul(q_goal_rep, q_delta)
        q = q / (torch.norm(q, dim=-1, keepdim=True) + 1e-12)
        return q

    def sample_target_state(self, num_samples):
        """Multi-scale target sampling concentrated near the exact goal state.
        
        Matches Docking6D approach:
        Tier 1 (10%): Exact goal + tiny noise (0.1x tolerance)
        Tier 2 (30%): Gaussian around goal with 2x tolerance std 
        Tier 3 (20%): Boundary-focused sampling at 0.8-1.2x tolerance 
        Tier 4 (40%): Broader uniform sampling centered on goal 
        """
        num_samples = int(num_samples)
        samples = torch.zeros(num_samples, self.state_dim)
        idx = 0

        # Tier 1 (10%): Exact goal + tiny noise
        n_exact = int(num_samples * 0.1)
        noise_std = torch.tensor([
            self.eps_p * 0.1, self.eps_p * 0.1, self.eps_p * 0.1,     # position: 0.01m
            self.eps_v * 0.1, self.eps_v * 0.1, self.eps_v * 0.1,     # velocity: 0.01m/s
            0.0, 0.0, 0.0, 0.0,                                        # quaternion: handled separately
            self.eps_omega * 0.1, self.eps_omega * 0.1, self.eps_omega * 0.1  # omega: 0.005rad/s
        ])
        samples[idx:idx + n_exact] = self.goal_state.unsqueeze(0) + torch.randn(n_exact, self.state_dim) * noise_std
        samples[idx:idx + n_exact, 6:10] = self.sample_quat_near_goal(n_exact, angle_std=self.eps_q * 0.1)
        idx += n_exact

        # Tier 2 (30%): Gaussian around goal with tolerance-scale std
        n_gaussian = int(num_samples * 0.3)
        goal_std = torch.tensor([
            self.eps_p * 2, self.eps_p * 2, self.eps_p * 2,
            self.eps_v * 2, self.eps_v * 2, self.eps_v * 2,
            0.0, 0.0, 0.0, 0.0,  # quaternion: handled separately
            self.eps_omega * 2, self.eps_omega * 2, self.eps_omega * 2
        ])
        samples[idx:idx + n_gaussian] = self.goal_state.unsqueeze(0) + torch.randn(n_gaussian, self.state_dim) * goal_std
        samples[idx:idx + n_gaussian, 6:10] = self.sample_quat_near_goal(n_gaussian, angle_std=self.eps_q * 2)
        idx += n_gaussian

        # Tier 3 (20%): Boundary-focused sampling at 0.8-1.2x tolerance
        # Samples on/near the reach set boundary where the value function transitions
        n_boundary = int(num_samples * 0.20)
        tolerances = torch.tensor([
            self.eps_p, self.eps_p, self.eps_p,
            self.eps_v, self.eps_v, self.eps_v,
            0.0, 0.0, 0.0, 0.0,  # quaternion: handled separately
            self.eps_omega, self.eps_omega, self.eps_omega
        ])
        # Each dimension independently at a random fraction of [0.8, 1.2]x its tolerance
        scale_factors = 0.8 + 0.4 * torch.rand(n_boundary, self.state_dim)
        # Random sign for each dimension (sample on both sides of goal)
        signs = torch.sign(torch.randn(n_boundary, self.state_dim))
        samples[idx:idx + n_boundary] = self.goal_state.unsqueeze(0) + signs * scale_factors * tolerances.unsqueeze(0)
        # Sample quaternions at boundary angle
        boundary_angles = 0.8 + 0.4 * torch.rand(n_boundary)  # [0.8, 1.2] * eps_q
        samples[idx:idx + n_boundary, 6:10] = self.sample_quat_near_goal(n_boundary, angle_std=self.eps_q)
        idx += n_boundary

        # Tier 4 (40%): Broader uniform sampling centered on goal_state
        # Half-width = fraction of state_range width, adapts when state_range is overridden
        n_broad = num_samples - idx
        sr = self.state_range_.cpu()
        state_widths = (sr[:, 1] - sr[:, 0])
        tier4_half_width = self.tier4_fraction * state_widths
        tier4_lo = torch.clamp(self.goal_state - tier4_half_width, sr[:, 0], sr[:, 1])
        tier4_hi = torch.clamp(self.goal_state + tier4_half_width, sr[:, 0], sr[:, 1])
        samples[idx:] = tier4_lo + torch.rand(n_broad, self.state_dim) * (tier4_hi - tier4_lo)
        # For quaternion dimensions, sample with broader angle std
        samples[idx:, 6:10] = self.sample_quat_near_goal(n_broad, angle_std=self.tier4_fraction * math.pi)

        return samples

    def sample_target_state_refined(self, num_samples):
        """Alias for sample_target_state (kept for backwards compatibility)."""
        return self.sample_target_state(num_samples)
    
    # ---------- 13D dynamics ----------
    def dsdt(self, state, control, disturbance):
        """
        state: [...,13]
          [x,y,z,vx,vy,vz,q0,q1,q2,q3,wx,wy,wz]
        control: [...,6]
          [Fx,Fy,Fz,tx,ty,tz] in BODY frame
        """
        dsdt = torch.zeros_like(state)

        # unpack
        x,y,z = state[...,0], state[...,1], state[...,2]
        vx,vy,vz = state[...,3], state[...,4], state[...,5]
        q = state[...,6:10]
        omega = state[...,10:13]

        # Normalize quaternion defensively (keeps simulation stable)
        q = q / (torch.norm(q, dim=-1, keepdim=True) + 1e-12)

        # Body controls
        Fb = control[...,0:3]       # N in body
        taub = control[...,3:6]     # N*m in body

        # Rotate Fb to LVLH acceleration: a_L = (1/m) R(q)^T Fb
        R_L2B = self.quat_to_R_LVLH_to_body(q)
        # Multiply: R^T * Fb
        aL = torch.matmul(R_L2B.transpose(-1, -2), Fb.unsqueeze(-1)).squeeze(-1) / self.mc

        n = self.n

        # Translation kinematics
        dsdt[...,0] = vx
        dsdt[...,1] = vy
        dsdt[...,2] = vz

        # 3D HCW dynamics
        ax = 2*n*vy + 3*(n**2)*x + aL[...,0]
        ay = -2*n*vx + aL[...,1]
        az = -(n**2)*z + aL[...,2]

        # velocity clamp like your original (applied to accel)
        ax = torch.where((vx >= self.v_max) & (ax > 0), torch.zeros_like(ax), ax)
        ax = torch.where((vx <= -self.v_max) & (ax < 0), torch.zeros_like(ax), ax)
        ay = torch.where((vy >= self.v_max) & (ay > 0), torch.zeros_like(ay), ay)
        ay = torch.where((vy <= -self.v_max) & (ay < 0), torch.zeros_like(ay), ay)
        az = torch.where((vz >= self.v_max) & (az > 0), torch.zeros_like(az), az)
        az = torch.where((vz <= -self.v_max) & (az < 0), torch.zeros_like(az), az)

        dsdt[...,3] = ax
        dsdt[...,4] = ay
        dsdt[...,5] = az

        # Quaternion kinematics: qdot = 0.5 * Omega(omega) * q
        wx,wy,wz = omega[...,0], omega[...,1], omega[...,2]
        Omega = torch.zeros(state.shape[:-1] + (4,4), device=state.device, dtype=state.dtype)
        Omega[...,0,1] = -wx; Omega[...,0,2] = -wy; Omega[...,0,3] = -wz
        Omega[...,1,0] =  wx; Omega[...,1,2] =  wz; Omega[...,1,3] = -wy
        Omega[...,2,0] =  wy; Omega[...,2,1] = -wz; Omega[...,2,3] =  wx
        Omega[...,3,0] =  wz; Omega[...,3,1] =  wy; Omega[...,3,2] = -wx

        qdot = 0.5 * torch.matmul(Omega, q.unsqueeze(-1)).squeeze(-1)
        dsdt[...,6:10] = qdot

        # Rigid-body rotational dynamics: omega_dot = I^{-1}(tau - omega x (I omega))
        I = self.I
        Iomega = torch.matmul(I, omega.unsqueeze(-1)).squeeze(-1)
        cross = torch.cross(omega, Iomega, dim=-1)
        omegadot = torch.linalg.solve(I, (taub - cross).unsqueeze(-1)).squeeze(-1)
        dsdt[...,10:13] = omegadot

        return dsdt

    # ---------- Reach set ----------
    def reach_fn(self, state):
        """
        Signed distance <= 0 if within docking tolerances in:
          - position (3D L2)
          - velocity (3D L2)
          - attitude (angle error)
          - omega (3D L2)
        """
        x,y,z = state[...,0], state[...,1], state[...,2]
        vx,vy,vz = state[...,3], state[...,4], state[...,5]
        q = state[...,6:10]
        omega = state[...,10:13]

        goal = self.goal_state.to(state.device)
        xg,yg,zg = goal[0], goal[1], goal[2]
        vxg,vyg,vzg = goal[3], goal[4], goal[5]
        qg = self.q_goal.to(state.device)
        omegag = goal[10:13]

        # Add epsilon inside sqrt to prevent gradient explosion at origin
        pos_dist = torch.sqrt((x-xg)**2 + (y-yg)**2 + (z-zg)**2 + 1e-8) - self.eps_p
        vel_dist = torch.sqrt((vx-vxg)**2 + (vy-vyg)**2 + (vz-vzg)**2 + 1e-8) - self.eps_v
        att_dist = self.quat_error_angle(q / (torch.norm(q, dim=-1, keepdim=True) + 1e-12), qg) - self.eps_q
        omg_dist = torch.sqrt(torch.sum((omega - omegag)**2, dim=-1) + 1e-8) - self.eps_omega

        # Keep your shaping idea (I won’t over-tune; same spirit)
        pos_dist = torch.where(pos_dist < 0, pos_dist * 15, pos_dist * 0.1)
        vel_dist = torch.where(vel_dist < 0, vel_dist * 15, vel_dist * 1.0)
        att_dist = torch.where(att_dist < 0, att_dist * 150, att_dist * 0.1)
        omg_dist = torch.where(omg_dist < 0, omg_dist * 30, omg_dist * 1.0)

        goal = torch.stack([pos_dist, vel_dist, att_dist, omg_dist], dim=-1)
        return torch.max(goal, dim=-1).values

    # ---------- Avoid set ----------
    def avoid_fn(self, state):
        """
        3D extension of your planar avoid_fn.
        Uncertainty: your original geometry is planar; I extrude in z.
        """
        x = state[...,0]
        y = state[...,1]
        z = state[...,2]

        # --- Rectangular prism SDF ---
        half_w = self.w_t / 2.0 + self.chaser_buffer_xy
        half_z = self.d_t / 2.0 + self.chaser_buffer_z
        s_box = torch.maximum(
            torch.abs(x) - half_w,
            torch.maximum(
                torch.maximum(-(y + self.chaser_buffer_xy), y - (self.h_t + self.chaser_buffer_xy)),
                torch.abs(z) - half_z
            )
        )

        # --- Hemispherical cutout (centered at origin, y >= 0) ---
        effective_rad = max(self.dock_rad - self.chaser_buffer_xy, 1e-6)
        # Add epsilon inside sqrt to prevent gradient explosion at origin
        dist_sphere = torch.sqrt(x**2 + y**2 + z**2 + 1e-8) - effective_rad
        s_hemi = torch.maximum(-y, dist_sphere)

        # Collision region: box minus hemispherical cutout
        s_bubble = torch.maximum(s_box, -s_hemi + 0.02)

        # --- Cylindrical cutout to prevent failure set from encapsulating target ---
        # Creates a safe approach corridor in negative y direction
        # The cylinder extends from y <= 0 with radius = effective_rad in the x-z plane
        # Add epsilon inside sqrt to prevent gradient explosion when x=z=0
        dist_cyl_xz = torch.sqrt(x**2 + z**2 + 1e-8) - effective_rad
        s_cutout = torch.maximum(dist_cyl_xz, 
                                 torch.maximum(-(y + self.chaser_buffer_xy), y))

        s_fail = torch.maximum(s_bubble, -s_cutout + 0.02)

        s_fail = torch.where(s_fail < 0, s_fail * 0.75, s_fail * 50.0)
        return s_fail

    def boundary_fn(self, state):
        if self.set_mode == 'reach_avoid':
            return torch.maximum(self.reach_fn(state), -self.avoid_fn(state))
        raise NotImplementedError

    def cost_fn(self, state_traj):
        return torch.min(self.boundary_fn(state_traj), dim=-1).values

    def hamiltonian(self, state, dvds):
        if self.set_mode != 'reach_avoid':
            raise NotImplementedError
        opt_control = self.optimal_control(state, dvds)
        dsdt_ = self.dsdt(state, opt_control, None)
        return torch.sum(dvds * dsdt_, dim=-1)

    def optimal_control(self, state, dvds):
        """
        Bang-bang control for reach-avoid, but:
          - Force optimal sign depends on attitude: dv/dv · (R^T F_b)/m = (R dv_v)/m · F_b
        """
        if self.set_mode != 'reach_avoid':
            raise NotImplementedError

        q = state[...,6:10]
        q = q / (torch.norm(q, dim=-1, keepdim=True) + 1e-12)
        R_L2B = self.quat_to_R_LVLH_to_body(q)

        # Costates for translational acceleration act through vx,vy,vz
        p_v = dvds[...,3:6]  # ∂V/∂v_LVLH

        # Hamiltonian force term: p_v · (1/m) R^T F_b = (1/m)(R p_v) · F_b
        # where (R p_v) gives body-frame coefficients.
        coeff_body = torch.matmul(R_L2B, p_v.unsqueeze(-1)).squeeze(-1) / self.mc

        Fx = torch.where(coeff_body[...,0] > 0, -self.F_bar, self.F_bar)
        Fy = torch.where(coeff_body[...,1] > 0, -self.F_bar, self.F_bar)
        Fz = torch.where(coeff_body[...,2] > 0, -self.F_bar, self.F_bar)

        # Costates for omega: omega_dot includes I^{-1} tau
        p_omega = dvds[...,10:13]  # ∂V/∂omega
        # maximize p_omega · (I^{-1} tau) => (I^{-T} p_omega) · tau
        coeff_tau = torch.linalg.solve(self.I.transpose(0,1), p_omega.unsqueeze(-1)).squeeze(-1)

        tx = torch.where(coeff_tau[...,0] > 0, -self.tau_bar, self.tau_bar)
        ty = torch.where(coeff_tau[...,1] > 0, -self.tau_bar, self.tau_bar)
        tz = torch.where(coeff_tau[...,2] > 0, -self.tau_bar, self.tau_bar)

        return torch.stack([Fx, Fy, Fz, tx, ty, tz], dim=-1)

    def optimal_disturbance(self, state, dvds):
        return 0

    def plot_config(self):
        # Use goal quaternion (90° yaw) so reach set shows up in position slices
        # q_goal = [cos(π/4), 0, 0, sin(π/4)] for 90° rotation about z-axis
        q0 = float(self.q_goal[0].cpu()) if self.q_goal[0].is_cuda else float(self.q_goal[0])
        q1 = float(self.q_goal[1].cpu()) if self.q_goal[1].is_cuda else float(self.q_goal[1])
        q2 = float(self.q_goal[2].cpu()) if self.q_goal[2].is_cuda else float(self.q_goal[2])
        q3 = float(self.q_goal[3].cpu()) if self.q_goal[3].is_cuda else float(self.q_goal[3])
        return {
            'state_slices': [0, 0, 0, 0, 0, 0, q0, q1, q2, q3, 0, 0, 0],
            'state_labels': ['x', 'y', 'z', 'vx', 'vy', 'vz', 'q0', 'q1', 'q2', 'q3', 'wx', 'wy', 'wz'],
            'x_axis_idx': 0,
            'y_axis_idx': 1,
            'z_axis_idx': 2,
        }


class Quadrotor(Dynamics):

    def __init__(self, collisionR: float, collective_thrust_max: float,  set_mode: str):  # simpler quadrotor
        self.collective_thrust_max = collective_thrust_max
        # self.body_rate_acc_max = body_rate_acc_max
        self.m = 1  # mass
        self.arm_l = 0.17
        self.CT = 1
        self.CM = 0.016
        self.Gz = -9.8

        self.dwx_max = 8
        self.dwy_max = 8
        self.dwz_max = 4
        self.dist_dwx_max = 0
        self.dist_dwy_max = 0
        self.dist_dwz_max = 0
        self.dist_f = 0

        self.collisionR = collisionR
        self.reach_fn_weight = 1.
        self.avoid_fn_weight = 0.3
        self.state_range_ = torch.tensor([
            [-3.0, 3.0],
            [-3.0, 3.0],
            [-3.0, 3.0],
            [-1.0, 1.0],
            [-1.0, 1.0],
            [-1.0, 1.0],
            [-1.0, 1.0],
            [-5.0, 5.0],
            [-5.0, 5.0],
            [-5.0, 5.0],
            [-5.0, 5.0],
            [-5.0, 5.0],
            [-5.0, 5.0],
        ]).cuda()
        self.control_range_ = torch.tensor([[-self.collective_thrust_max, self.collective_thrust_max],
                                           [-self.dwx_max, self.dwx_max],
                                            [-self.dwy_max, self.dwy_max],
                                            [-self.dwz_max, self.dwz_max]]).cuda()
        self.eps_var = torch.tensor([20, 8, 8, 4]).cuda()
        self.control_init = torch.tensor([-self.Gz*0.0, 0, 0, 0]).cuda()

        state_mean_ = (self.state_range_[:, 0]+self.state_range_[:, 1])/2.0
        state_var_ = (self.state_range_[:, 1]-self.state_range_[:, 0])/2.0
        if set_mode == 'reach_avoid':
            l_type = 'brat_hjivi'
        else:
            l_type = 'brt_hjivi'
        super().__init__(
            name='Quadrotor', loss_type=l_type, set_mode=set_mode,
            state_dim=13, input_dim=14, control_dim=4, disturbance_dim=0,
            state_mean=state_mean_.cpu().tolist(),
            state_var=state_var_.cpu().tolist(),
            value_mean=(math.sqrt(3.0**2 + 3.0**2) -
                        2 * self.collisionR) / 2,
            value_var=math.sqrt(3.0**2 + 3.0**2)/2,
            value_normto=0.02,
            deepReach_model='exact',
        )

    def normalize_q(self, x):
        # normalize quaternion
        normalized_x = x*1.0
        q_tensor = x[..., 3:7]
        q_tensor = torch.nn.functional.normalize(
            q_tensor, p=2, dim=-1)  # normalize quaternion
        normalized_x[..., 3:7] = q_tensor
        return normalized_x

    def clamp_state_input(self, state_input):
        return self.normalize_q(state_input)

    def control_range(self, state):
        return [[-self.collective_thrust_max, self.collective_thrust_max],
                [-self.dwx_max, self.dwx_max],
                [-self.dwy_max, self.dwy_max],
                [-self.dwz_max, self.dwz_max]]

    def state_test_range(self):
        return self.state_range_.cpu().tolist()

    def state_verification_range(self):
        return self.state_range_.cpu().tolist()

    def periodic_transform_fn(self, input):
        return input.cuda()

    def equivalent_wrapped_state(self, state):
        wrapped_state = torch.clone(state)
        # return wrapped_state
        return self.normalize_q(wrapped_state)

    def dsdt(self, state, control, disturbance):
        qw = state[..., 3] * 1.0
        qx = state[..., 4] * 1.0
        qy = state[..., 5] * 1.0
        qz = state[..., 6] * 1.0
        vx = state[..., 7] * 1.0
        vy = state[..., 8] * 1.0
        vz = state[..., 9] * 1.0
        wx = state[..., 10] * 1.0
        wy = state[..., 11] * 1.0
        wz = state[..., 12] * 1.0
        f = (control[..., 0]) * 1.0

        dsdt = torch.zeros_like(state)
        dsdt[..., 0] = vx
        dsdt[..., 1] = vy
        dsdt[..., 2] = vz
        dsdt[..., 3] = -(wx * qx + wy * qy + wz * qz) / 2.0
        dsdt[..., 4] = (wx * qw + wz * qy - wy * qz) / 2.0
        dsdt[..., 5] = (wy * qw - wz * qx + wx * qz) / 2.0
        dsdt[..., 6] = (wz * qw + wy * qx - wx * qy) / 2.0
        dsdt[..., 7] = 2 * (qw * qy + qx * qz) * self.CT / \
            self.m * f
        dsdt[..., 8] = 2 * (-qw * qx + qy * qz) * self.CT / \
            self.m * f
        dsdt[..., 9] = self.Gz + (1 - 2 * torch.pow(qx, 2) - 2 *
                                  torch.pow(qy, 2)) * self.CT / self.m * f
        dsdt[..., 10] = (control[..., 1]
                         ) * 1.0 - 5 * wy * wz / 9.0
        dsdt[..., 11] = (control[..., 2]
                         ) * 1.0 + 5 * wx * wz / 9.0
        dsdt[..., 12] = (control[..., 3]) * 1.0

        return dsdt

    def dist_to_cylinder(self, state, a, b):
        '''for cylinder with full body collision'''
        state_ = state*1.0
        state_[..., 0] = state_[..., 0] - a
        state_[..., 1] = state_[..., 1] - b

        # create normal vector
        v = torch.zeros_like(state_[..., 4:7])
        v[..., 2] = 1
        v = quaternion.quaternion_apply(state_[..., 3:7], v)
        vx = v[..., 0]
        vy = v[..., 1]
        vz = v[..., 2]
        # compute vector from center of quadrotor to the center of cylinder
        px = state_[..., 0]
        py = state_[..., 1]

        # get full body distance
        dist = torch.norm(state_[..., :2], dim=-1)
        # return dist- self.collisionR
        dist = dist - torch.sqrt((self.arm_l**2*px**2*vz**2)/(px**2*vx**2 + px**2*vz**2 + 2*px*py*vx*vy + py**2*vy**2 + py**2*vz**2)
                                 + (self.arm_l**2*py**2*vz**2)/(px**2*vx**2 + px**2*vz**2 + 2*px*py*vx*vy + py**2*vy**2 + py**2*vz**2))
        return torch.maximum(dist, torch.zeros_like(dist)) - self.collisionR

    def reach_fn(self, state):
        state_ = state*1.0
        state_[..., 0] = state_[..., 0] - 0.
        state_[..., 1] = state_[..., 1]
        return (torch.norm(state[..., :2], dim=-1)-0.3)*self.reach_fn_weight

    def avoid_fn(self, state):
        return self.avoid_fn_weight*torch.minimum(self.dist_to_cylinder(state, 0.0, 0.75), self.dist_to_cylinder(state, 0.0, -0.75))

    def boundary_fn(self, state):
        if self.set_mode == 'avoid':
            return self.dist_to_cylinder(state, 0.0, 0.0)
        else:
            return torch.maximum(self.reach_fn(state), -self.avoid_fn(state))

    def bc(self, state):
        return torch.norm(state[..., :2], dim=-1)-0.5

    def sample_target_state(self, num_samples):
        target_state_range = self.state_test_range()
        target_state_range[0] = [-1, 1]
        target_state_range[1] = [-0.25, 0.25]
        target_state_range = torch.tensor(target_state_range)
        return target_state_range[:, 0] + torch.rand(num_samples, self.state_dim)*(target_state_range[:, 1] - target_state_range[:, 0])

    def cost_fn(self, state_traj):
        if self.set_mode == 'avoid':
            return torch.min(self.boundary_fn(state_traj), dim=-1).values
        else:
            # return min_t max{l(x(t)), max_k_up_to_t{-g(x(k))}}, where l(x) is reach_fn, g(x) is avoid_fn
            reach_values = self.reach_fn(state_traj)
            avoid_values = self.avoid_fn(state_traj)
            return torch.min(torch.clamp(reach_values, min=torch.max(-avoid_values, dim=-1).values.unsqueeze(-1)), dim=-1).values

    def hamiltonian(self, state, dvds):
        if self.set_mode in ['reach', 'reach_avoid']:
            qw = state[..., 3] * 1.0
            qx = state[..., 4] * 1.0
            qy = state[..., 5] * 1.0
            qz = state[..., 6] * 1.0
            vx = state[..., 7] * 1.0
            vy = state[..., 8] * 1.0
            vz = state[..., 9] * 1.0
            wx = state[..., 10] * 1.0
            wy = state[..., 11] * 1.0
            wz = state[..., 12] * 1.0

            c1 = 2 * (qw * qy + qx * qz) * self.CT / self.m
            c2 = 2 * (-qw * qx + qy * qz) * self.CT / self.m
            c3 = (1 - 2 * torch.pow(qx, 2) - 2 *
                  torch.pow(qy, 2)) * self.CT / self.m

            # Compute the hamiltonian for the quadrotor
            ham = dvds[..., 0] * vx + dvds[..., 1] * vy + dvds[..., 2] * vz
            ham += -dvds[..., 3] * (wx * qx + wy * qy + wz * qz) / 2.0
            ham += dvds[..., 4] * (wx * qw + wz * qy - wy * qz) / 2.0
            ham += dvds[..., 5] * (wy * qw - wz * qx + wx * qz) / 2.0
            ham += dvds[..., 6] * (wz * qw + wy * qx - wx * qy) / 2.0
            ham += dvds[..., 9] * self.Gz
            ham += -dvds[..., 10] * 5 * wy * wz / \
                9.0 + dvds[..., 11] * 5 * wx * wz / 9.0

            ham -= torch.abs(dvds[..., 7] * c1 + dvds[..., 8] *
                             c2 + dvds[..., 9] * c3) * self.collective_thrust_max

            ham -= torch.abs(dvds[..., 10]) * self.dwx_max + torch.abs(
                dvds[..., 11]) * self.dwy_max + torch.abs(dvds[..., 12]) * self.dwz_max

        elif self.set_mode == 'avoid':
            qw = state[..., 3] * 1.0
            qx = state[..., 4] * 1.0
            qy = state[..., 5] * 1.0
            qz = state[..., 6] * 1.0
            vx = state[..., 7] * 1.0
            vy = state[..., 8] * 1.0
            vz = state[..., 9] * 1.0
            wx = state[..., 10] * 1.0
            wy = state[..., 11] * 1.0
            wz = state[..., 12] * 1.0

            c1 = 2 * (qw * qy + qx * qz) * self.CT / self.m
            c2 = 2 * (-qw * qx + qy * qz) * self.CT / self.m
            c3 = (1 - 2 * torch.pow(qx, 2) - 2 *
                  torch.pow(qy, 2)) * self.CT / self.m

            # Compute the hamiltonian for the quadrotor
            ham = dvds[..., 0] * vx + dvds[..., 1] * vy + dvds[..., 2] * vz
            ham += -dvds[..., 3] * (wx * qx + wy * qy + wz * qz) / 2.0
            ham += dvds[..., 4] * (wx * qw + wz * qy - wy * qz) / 2.0
            ham += dvds[..., 5] * (wy * qw - wz * qx + wx * qz) / 2.0
            ham += dvds[..., 6] * (wz * qw + wy * qx - wx * qy) / 2.0
            ham += dvds[..., 9] * self.Gz
            ham += -dvds[..., 10] * 5 * wy * wz / \
                9.0 + dvds[..., 11] * 5 * wx * wz / 9.0

            ham += torch.abs(dvds[..., 7] * c1 + dvds[..., 8] *
                             c2 + dvds[..., 9] * c3) * self.collective_thrust_max

            ham += torch.abs(dvds[..., 10]) * self.dwx_max + torch.abs(
                dvds[..., 11]) * self.dwy_max + torch.abs(dvds[..., 12]) * self.dwz_max

        else:
            raise NotImplementedError

        return ham

    def optimal_control(self, state, dvds):
        if self.set_mode in ['reach', 'reach_avoid']:
            qw = state[..., 3] * 1.0
            qx = state[..., 4] * 1.0
            qy = state[..., 5] * 1.0
            qz = state[..., 6] * 1.0

            c1 = 2 * (qw * qy + qx * qz) * self.CT / self.m
            c2 = 2 * (-qw * qx + qy * qz) * self.CT / self.m
            c3 = (1 - 2 * torch.pow(qx, 2) - 2 *
                  torch.pow(qy, 2)) * self.CT / self.m

            u1 = -self.collective_thrust_max * \
                torch.sign(dvds[..., 7] * c1 + dvds[..., 8] *
                           c2 + dvds[..., 9] * c3)
            u2 = -self.dwx_max * torch.sign(dvds[..., 10])
            u3 = -self.dwy_max * torch.sign(dvds[..., 11])
            u4 = -self.dwz_max * torch.sign(dvds[..., 12])
        elif self.set_mode == 'avoid':
            qw = state[..., 3] * 1.0
            qx = state[..., 4] * 1.0
            qy = state[..., 5] * 1.0
            qz = state[..., 6] * 1.0

            c1 = 2 * (qw * qy + qx * qz) * self.CT / self.m
            c2 = 2 * (-qw * qx + qy * qz) * self.CT / self.m
            c3 = (1 - 2 * torch.pow(qx, 2) - 2 *
                  torch.pow(qy, 2)) * self.CT / self.m

            u1 = self.collective_thrust_max * \
                torch.sign(dvds[..., 7] * c1 + dvds[..., 8] *
                           c2 + dvds[..., 9] * c3)
            u2 = self.dwx_max * torch.sign(dvds[..., 10])
            u3 = self.dwy_max * torch.sign(dvds[..., 11])
            u4 = self.dwz_max * torch.sign(dvds[..., 12])

        return torch.cat((u1[..., None], u2[..., None], u3[..., None], u4[..., None]), dim=-1)

    def optimal_disturbance(self, state, dvds):
        return torch.zeros(1)

    def plot_config(self):
        return {
            'state_slices': [0.96,  1.18,  0.54,  0.44, -0.45,  0.27, -0.73, -2.83, -1.07, -3.34, 3.19, -2.80,  3.43],
            'state_labels': ['x', 'y', 'z', 'qw', 'qx', 'qy', 'qz', 'vx', 'vy', 'vz', 'wx', 'wy', 'wz'],
            'x_axis_idx': 0,
            'y_axis_idx': 1,
            'z_axis_idx': 7,
        }


class F1tenth(Dynamics):
    def __init__(self):
        # variable for dynamics
        self.mu = 1.0489
        self.C_Sf = 4.718
        self.C_Sr = 5.4562
        self.lf = 0.15875
        self.lr = 0.17145
        self.h = 0.074
        self.m = 3.74
        self.I = 0.04712
        self.s_min = -0.4189
        self.s_max = 0.4189
        self.sv_min = -3.2
        self.sv_max = 3.2
        self.v_switch = 7.319
        self.a_max = 9.51
        self.v_min = 0.1
        self.v_max = 10.0
        self.omega_max = 6.0
        self.delta_t = 0.01
        self.g = 9.81
        self.lwb = self.lf + self.lr

        self.v_mean = (self.v_min + self.v_max) / 2
        self.v_var = (self.v_max - self.v_min) / 2

        # map info
        # self.dt = np.load(map_path)
        self.origin = [-78.21853769831466, -44.37590462453829]
        self.resolution = 0.062500
        self.width = 1600
        self.height = 1600

        # control constraints
        self.input_steering_v_max = self.sv_max
        self.input_acceleration_max = self.a_max

        self.xmean = 62.5/2
        self.xvar = 62.5/2
        self.ymean = 25
        self.yvar = 25

        self.x_min = self.xmean-self.xvar
        self.x_max = self.xmean+self.xvar
        self.y_min = self.ymean-self.yvar
        self.y_max = self.ymean+self.yvar

        self.state_range_ = torch.tensor([[self.x_min, self.x_max], [self.y_min, self.y_max], [-0.4189, 0.4189], [
                                         self.v_min, self.v_max], [-math.pi, math.pi], [-self.omega_max, self.omega_max], [-1, 1]]).cuda()
        self.control_range_ = torch.tensor(
            [[self.sv_min, self.sv_max], [-self.a_max, self.a_max]]).cuda()
        self.eps_var = torch.tensor([self.sv_max**2, self.a_max**2]).cuda()
        self.control_init = torch.tensor([0.0, 0.0]).cuda()

        # for the track
        self.obstaclemap_file = 'dynamics/F1_map_obstaclemap.mat'
        self.pixel2world = 0.0625
        self.obstacle_map = spio.loadmat(self.obstaclemap_file)
        self.obstacle_map = self.obstacle_map['obs_map']
        self.obstacle_map[self.obstacle_map == -0.] = 1
        self.obstacle_map = self.obstacle_map[int(self.y_min/self.pixel2world):int(self.y_max/self.pixel2world)+1,
                                              int(self.x_min/self.pixel2world):int(self.x_max/self.pixel2world)+1]
        self.obstacle_map = torch.tensor(self.obstacle_map)

        self.world_range = self.state_range_.cpu().numpy()
        self.x_rangearray = torch.arange(self.obstacle_map.shape[0])
        self.y_rangearray = torch.arange(self.obstacle_map.shape[1])

        state_mean_ = (self.state_range_[:, 0]+self.state_range_[:, 1])/2.0
        state_var_ = (self.state_range_[:, 1]-self.state_range_[:, 0])/2.0

        super().__init__(
            name='F1tenth', loss_type='brt_hjivi', set_mode='avoid',
            state_dim=7, input_dim=9, control_dim=2, disturbance_dim=0,

            state_mean=state_mean_.cpu().tolist(),
            state_var=state_var_.cpu().tolist(),
            value_mean=0.5,  # mean of expected value function
            value_var=1.5,  # (max - min)/2.0 of expected value function
            value_normto=0.02,
            deepReach_model='exact'
        )

    def state_test_range(self):
        return self.state_range_.cpu().tolist()

    def state_verification_range(self):
        return [
            [self.x_min, self.x_max],
            [self.y_min, self.y_max],                      # y
            [-0.4189, 0.4189],               # steering angle
            [0.1, 8.0],                   # velocity
            [-math.pi, math.pi],                  # pose theta
            [-4.5, 4.5],                        # pose theta rate
            [-0.8, 0.8],                       # slip angle
        ]

    def periodic_transform_fn(self, input):
        output_shape = list(input.shape)
        output_shape[-1] = output_shape[-1]+1
        transformed_input = torch.zeros(output_shape)
        transformed_input[..., :5] = input[..., :5]
        transformed_input[..., 5] = torch.sin(input[..., 5]*self.state_var[4])
        transformed_input[..., 6] = torch.cos(input[..., 5]*self.state_var[4])
        transformed_input[..., 7:] = input[..., 6:]
        return transformed_input.cuda()

    def dsdt(self, state, control, disturbance):
        # here the control is steering angle v and acceleration
        f = torch.zeros_like(state)
        current_vel = state[..., 3]  # [1, 65000]
        kinematic_mask = torch.abs(current_vel) < 0.5
        # switch to kinematic model for small velocities
        if torch.any(kinematic_mask):
            # print(f"kinematic_mask is {kinematic_mask.shape}")
            if len(kinematic_mask.shape) == 1:
                sample_idx = kinematic_mask.nonzero(as_tuple=True)[0]
                x_ks = state[kinematic_mask][..., 0:5]
                u_ks = control[kinematic_mask]
                f_ks = torch.zeros_like(x_ks)
                f_ks[..., 0] = x_ks[..., 3]*torch.cos(x_ks[..., 4])
                f_ks[..., 1] = x_ks[..., 3]*torch.sin(x_ks[..., 4])
                f_ks[..., 2] = u_ks[..., 0]
                f_ks[..., 3] = u_ks[..., 1]
                f_ks[..., 4] = x_ks[..., 3]/self.lwb*torch.tan(x_ks[..., 2])
                f[sample_idx, :5] = f_ks
                f[sample_idx, 5] = u_ks[..., 1]/self.lwb*torch.tan(state[kinematic_mask][..., 2])+state[kinematic_mask][..., 3]/(
                    self.lwb*torch.cos(state[kinematic_mask][..., 2])**2)*u_ks[..., 0]
                f[sample_idx, 6] = 0.
            else:
                batch_idx, sample_idx = kinematic_mask.nonzero(as_tuple=True)
                x_ks = state[kinematic_mask][..., 0:5]
                u_ks = control[kinematic_mask]
                f_ks = torch.zeros_like(x_ks)
                f_ks[..., 0] = x_ks[..., 3]*torch.cos(x_ks[..., 4])
                f_ks[..., 1] = x_ks[..., 3]*torch.sin(x_ks[..., 4])
                f_ks[..., 2] = u_ks[..., 0]
                f_ks[..., 3] = u_ks[..., 1]
                f_ks[..., 4] = x_ks[..., 3]/self.lwb*torch.tan(x_ks[..., 2])
                f[batch_idx, sample_idx, :5] = f_ks
                f[batch_idx, sample_idx, 5] = u_ks[..., 1]/self.lwb*torch.tan(state[kinematic_mask][..., 2])+state[kinematic_mask][..., 3]/(
                    self.lwb*torch.cos(state[kinematic_mask][..., 2])**2)*u_ks[..., 0]
                f[batch_idx, sample_idx, 6] = 0.

        dynamic_mask = ~kinematic_mask
        if torch.any(dynamic_mask):
            if len(kinematic_mask.shape) == 1:
                sample_idx = dynamic_mask.nonzero(as_tuple=True)[0]
                f[sample_idx, 0] = state[dynamic_mask][..., 3] * \
                    torch.cos(state[dynamic_mask][..., 6] +
                              state[dynamic_mask][..., 4])
                f[sample_idx, 1] = state[dynamic_mask][..., 3] * \
                    torch.sin(state[dynamic_mask][..., 6] +
                              state[dynamic_mask][..., 4])
                f[sample_idx, 2] = control[dynamic_mask][..., 0]
                f[sample_idx, 3] = control[dynamic_mask][..., 1]
                f[sample_idx, 4] = state[dynamic_mask][..., 5]
                f[sample_idx, 5] = -self.mu*self.m/(state[dynamic_mask][..., 3]*self.I*(self.lr+self.lf))*(self.lf**2*self.C_Sf*(self.g*self.lr-control[dynamic_mask][..., 1]*self.h) + self.lr**2*self.C_Sr*(self.g*self.lf + control[dynamic_mask][..., 1]*self.h))*state[dynamic_mask][..., 5] \
                    + self.mu*self.m/(self.I*(self.lr+self.lf))*(self.lr*self.C_Sr*(self.g*self.lf + control[dynamic_mask][..., 1]*self.h) - self.lf*self.C_Sf*(self.g*self.lr - control[dynamic_mask][..., 1]*self.h))*state[dynamic_mask][..., 6] \
                    + self.mu*self.m/(self.I*(self.lr+self.lf))*self.lf*self.C_Sf*(
                    self.g*self.lr - control[dynamic_mask][..., 1]*self.h)*state[dynamic_mask][..., 2]
                f[sample_idx, 6] = (self.mu/(state[dynamic_mask][..., 3]**2*(self.lr+self.lf))*(self.C_Sr*(self.g*self.lf + control[dynamic_mask][..., 1]*self.h)*self.lr - self.C_Sf*(self.g*self.lr - control[dynamic_mask][..., 1]*self.h)*self.lf)-1)*state[dynamic_mask][..., 5] \
                    - self.mu/(state[dynamic_mask][..., 3]*(self.lr+self.lf))*(self.C_Sr*(self.g*self.lf + control[dynamic_mask][..., 1]*self.h) + self.C_Sf*(self.g*self.lr-control[dynamic_mask][..., 1]*self.h))*state[dynamic_mask][..., 6] \
                    + self.mu/(state[dynamic_mask][..., 3]*(self.lr+self.lf))*(self.C_Sf*(
                        self.g*self.lr-control[dynamic_mask][..., 1]*self.h))*state[dynamic_mask][..., 2]
            else:
                batch_idx, sample_idx = dynamic_mask.nonzero(as_tuple=True)
                f[batch_idx, sample_idx, 0] = state[dynamic_mask][..., 3] * \
                    torch.cos(state[dynamic_mask][..., 6] +
                              state[dynamic_mask][..., 4])
                f[batch_idx, sample_idx, 1] = state[dynamic_mask][..., 3] * \
                    torch.sin(state[dynamic_mask][..., 6] +
                              state[dynamic_mask][..., 4])
                f[batch_idx, sample_idx, 2] = control[dynamic_mask][..., 0]
                f[batch_idx, sample_idx, 3] = control[dynamic_mask][..., 1]
                f[batch_idx, sample_idx, 4] = state[dynamic_mask][..., 5]
                f[batch_idx, sample_idx, 5] = -self.mu*self.m/(state[dynamic_mask][..., 3]*self.I*(self.lr+self.lf))*(self.lf**2*self.C_Sf*(self.g*self.lr-control[dynamic_mask][..., 1]*self.h) + self.lr**2*self.C_Sr*(self.g*self.lf + control[dynamic_mask][..., 1]*self.h))*state[dynamic_mask][..., 5] \
                    + self.mu*self.m/(self.I*(self.lr+self.lf))*(self.lr*self.C_Sr*(self.g*self.lf + control[dynamic_mask][..., 1]*self.h) - self.lf*self.C_Sf*(self.g*self.lr - control[dynamic_mask][..., 1]*self.h))*state[dynamic_mask][..., 6] \
                    + self.mu*self.m/(self.I*(self.lr+self.lf))*self.lf*self.C_Sf*(
                    self.g*self.lr - control[dynamic_mask][..., 1]*self.h)*state[dynamic_mask][..., 2]
                f[batch_idx, sample_idx, 6] = (self.mu/(state[dynamic_mask][..., 3]**2*(self.lr+self.lf))*(self.C_Sr*(self.g*self.lf + control[dynamic_mask][..., 1]*self.h)*self.lr - self.C_Sf*(self.g*self.lr - control[dynamic_mask][..., 1]*self.h)*self.lf)-1)*state[dynamic_mask][..., 5] \
                    - self.mu/(state[dynamic_mask][..., 3]*(self.lr+self.lf))*(self.C_Sr*(self.g*self.lf + control[dynamic_mask][..., 1]*self.h) + self.C_Sf*(self.g*self.lr-control[dynamic_mask][..., 1]*self.h))*state[dynamic_mask][..., 6] \
                    + self.mu/(state[dynamic_mask][..., 3]*(self.lr+self.lf))*(self.C_Sf*(
                        self.g*self.lr-control[dynamic_mask][..., 1]*self.h))*state[dynamic_mask][..., 2]
        # ------------------------------OPT CTRL--------------------------------

        return f

    def clamp_state_input(self, state_input):
        full_input = torch.cat(
            (torch.ones(state_input.shape[0], 1).to(state_input), state_input), dim=-1)
        state = self.input_to_coord(full_input)[..., 1:]
        lx = self.boundary_fn(state)
        return state_input[lx >= -1]

    def clamp_verification_state(self, state):
        lx = self.boundary_fn(state)
        return state[lx >= 0]

    def clamp_control(self, state, control):
        control_clamped = control*1.0

        smax_mask = state[..., 2] > self.s_max-0.01
        smin_mask = state[..., 2] < -self.s_max+0.01
        if len(smax_mask.shape) == 1:
            max_sample_idx = smax_mask.nonzero(as_tuple=True)[0]
            control_clamped[max_sample_idx, 0] = torch.clamp(
                control_clamped[max_sample_idx, 0], max=0)
            min_sample_idx = smin_mask.nonzero(as_tuple=True)[0]
            control_clamped[min_sample_idx, 0] = torch.clamp(
                control_clamped[min_sample_idx, 0], min=0)

        else:
            max_batch_idx, max_sample_idx = smax_mask.nonzero(as_tuple=True)
            control_clamped[max_batch_idx, max_sample_idx, 0] = torch.clamp(
                control_clamped[max_batch_idx, max_sample_idx, 0], max=0)
            min_batch_idx, min_sample_idx = smin_mask.nonzero(as_tuple=True)
            control_clamped[min_batch_idx, min_sample_idx, 0] = torch.clamp(
                control_clamped[min_batch_idx, min_sample_idx, 0], min=0)

        accelerate_upper = torch.ones(
            state.shape[:-1], device=state.device) * self.input_acceleration_max
        accelerate_upper[state[..., 3] > self.v_switch] = self.input_acceleration_max * \
            self.v_switch / state[state[..., 3] > self.v_switch][..., 3]

        acc_mask = control_clamped[..., 1] > accelerate_upper
        if len(acc_mask.shape) == 1:
            sample_idx = acc_mask.nonzero(as_tuple=True)[0]
            control_clamped[sample_idx, 1] = accelerate_upper[sample_idx]
        else:
            batch_idx, sample_idx = acc_mask.nonzero(as_tuple=True)
            control_clamped[batch_idx, sample_idx,
                            1] = accelerate_upper[batch_idx, sample_idx]

        assert ((accelerate_upper-control_clamped[..., 1]) >= 0.0).all()
        return control_clamped

    def interpolation(self, state_pixel_coords):
        self.obstacle_map = self.obstacle_map.to(state_pixel_coords)
        # Find the indices surrounding the query points
        x0 = torch.floor(state_pixel_coords[..., 0]).long()
        x1 = x0 + 1
        y0 = torch.floor(state_pixel_coords[..., 1]).long()
        y1 = y0 + 1
        # Ensure indices are within bounds
        x0 = torch.clamp(x0, 0, self.x_rangearray.size(0) - 1)
        x1 = torch.clamp(x1, 0, self.x_rangearray.size(0) - 1)
        y0 = torch.clamp(y0, 0, self.y_rangearray.size(0) - 1)
        y1 = torch.clamp(y1, 0, self.y_rangearray.size(0) - 1)

        # Gather the values at the corner points for each query point
        v00 = self.obstacle_map[x0, y0]
        v01 = self.obstacle_map[x0, y1]
        v10 = self.obstacle_map[x1, y0]
        v11 = self.obstacle_map[x1, y1]
        # Compute the fractional part for each query point
        x_frac = state_pixel_coords[..., 0] - x0.float()
        y_frac = state_pixel_coords[..., 1] - y0.float()
        # Bilinear interpolation for each query point
        v0 = v00 * (1 - x_frac) + v10 * x_frac
        v1 = v01 * (1 - x_frac) + v11 * x_frac
        # Interpolated value
        interp_values = v0 * (1 - y_frac) + v1 * y_frac
        return interp_values

    def boundary_fn(self, state):
        # MPC: state = B * N * H * 7
        # DeepReach: state = B * 7
        # Takes the cordinates in the real world and returns the lx for the obstacles at those coords
        # shift the origin so that the min is 0
        shiftedCoords = state - torch.tensor(self.world_range[..., 0].reshape(
            self.state_dim,), device=state.device)  # num states involve time as well
        # extract and flip the x and y pos for image space query
        if shiftedCoords.shape[0] == 1:
            shiftedCoords_pos_world = np.squeeze(shiftedCoords[..., 0:2])
        else:
            shiftedCoords_pos_world = shiftedCoords[..., 0:2]
        if len(shiftedCoords_pos_world.shape) == 2:
            # B*2 for deepreach, B*N*H*2 for MPC
            shiftedCoords_pos_image = torch.fliplr(shiftedCoords_pos_world)
        else:
            shiftedCoords_pos_image = torch.flip(shiftedCoords_pos_world, [-1])
        # convert the world coordinates to pixel coordinates
        # note this does not have to be integers due to the regularGridInterpolator
        shiftedCoords_pos_pixel = shiftedCoords_pos_image/self.pixel2world
        # query the generator
        # obstacle value only depends on pos
        obstacle_value = self.interpolation(shiftedCoords_pos_pixel)
        # obstacle_value = obstacle_value.reshape([obstacle_value.shape[0],1]) # this should be the lx
        return obstacle_value

    def cost_fn(self, state_traj):
        return torch.min(self.boundary_fn(state_traj), dim=-1).values

    def hamiltonian(self, state, dvds):
        if self.set_mode == 'reach':
            raise NotImplementedError

        elif self.set_mode == 'avoid':
            opt_control = self.optimal_control(state, dvds)
            dsdt_ = self.dsdt(state, opt_control, None)
            ham = torch.sum(dvds*dsdt_, dim=-1)
        return ham

    def optimal_control(self, state, dvds):

        if self.set_mode == 'reach':
            raise NotImplementedError
        elif self.set_mode == 'avoid':
            unsqueeze_u = False
            if state.shape[0] == 1:
                state = state.squeeze(0)
                dvds = dvds.squeeze(0)
                unsqueeze_u = True
            batch_dims = state.shape[:-1]
            u = torch.zeros(*batch_dims, 2, device=state.device)

            kinematic_mask = torch.abs(state[..., 3]) < 0.5
            # if the number of kinematic_mask's dimensional is greater than 1, print

            if torch.any(kinematic_mask):
                if len(kinematic_mask.shape) == 1:
                    sample_idx = kinematic_mask.nonzero(as_tuple=True)[0]
                    u[sample_idx, 0] = self.input_steering_v_max * torch.sign(
                        dvds[kinematic_mask][..., 2] + dvds[kinematic_mask][..., 5] * state[kinematic_mask][..., 3] / (self.lwb * torch.cos(state[kinematic_mask][..., 2])**2))
                    u[sample_idx, 1] = self.input_acceleration_max * torch.sign(
                        dvds[kinematic_mask][..., 3] + dvds[kinematic_mask][..., 5] / self.lwb * torch.tan(state[kinematic_mask][..., 2]))
                else:
                    batch_idx, sample_idx = kinematic_mask.nonzero(
                        as_tuple=True)
                    u[batch_idx, sample_idx, 0] = self.input_steering_v_max * torch.sign(
                        dvds[kinematic_mask][..., 2] + dvds[kinematic_mask][..., 5] * state[kinematic_mask][..., 3] / (self.lwb * torch.cos(state[kinematic_mask][..., 2])**2))
                    u[batch_idx, sample_idx, 1] = self.input_acceleration_max * torch.sign(
                        dvds[kinematic_mask][..., 3] + dvds[kinematic_mask][..., 5] / self.lwb * torch.tan(state[kinematic_mask][..., 2]))

            dynamic_mask = ~kinematic_mask
            if torch.any(dynamic_mask):
                if len(kinematic_mask.shape) == 1:
                    sample_idx = dynamic_mask.nonzero(as_tuple=True)[0]
                    u[sample_idx, 0] = self.input_steering_v_max * \
                        torch.sign(dvds[dynamic_mask][..., 2])

                    u[sample_idx, 1] = self.input_acceleration_max * torch.sign(
                        dvds[dynamic_mask][..., 3]
                        + dvds[dynamic_mask][..., 5] * ((-self.mu * self.m / (state[dynamic_mask][..., 3] * self.I * (self.lr + self.lf))*(-self.lf**2*self.C_Sf*self.h + self.lr**2*self.C_Sr*self.h))*state[dynamic_mask][..., 5]
                                                        + self.mu * self.m / (self.I * (self.lr + self.lf)) * (
                                                            self.lr*self.C_Sr*self.h + self.lf*self.C_Sf*self.h)*state[dynamic_mask][..., 6]
                                                        - self.mu * self.m / (self.I * (self.lr + self.lf)) * self.lf*self.C_Sf*self.h*state[dynamic_mask][..., 2])
                        + dvds[dynamic_mask][..., 6] * ((self.mu / (state[dynamic_mask][..., 3]**2 * (self.lr + self.lf)) * (self.C_Sr*self.h*self.lr + self.C_Sf*self.h*self.lf))*state[dynamic_mask][..., 5]
                                                        - self.mu / (state[dynamic_mask][..., 3] * (self.lr + self.lf)) * (
                                                            self.C_Sr*self.h - self.C_Sf*self.h)*state[dynamic_mask][..., 6]
                                                        - self.mu / (state[dynamic_mask][..., 3] * (self.lr + self.lf)) * self.C_Sf*self.h*state[dynamic_mask][..., 2])
                    )
                else:
                    batch_idx, sample_idx = dynamic_mask.nonzero(as_tuple=True)

                    u[batch_idx, sample_idx, 0] = self.input_steering_v_max * \
                        torch.sign(dvds[dynamic_mask][..., 2])

                    u[batch_idx, sample_idx, 1] = self.input_acceleration_max * torch.sign(
                        dvds[dynamic_mask][..., 3]
                        + dvds[dynamic_mask][..., 5] * ((-self.mu * self.m / (state[dynamic_mask][..., 3] * self.I * (self.lr + self.lf))*(-self.lf**2*self.C_Sf*self.h + self.lr**2*self.C_Sr*self.h))*state[dynamic_mask][..., 5]
                                                        + self.mu * self.m / (self.I * (self.lr + self.lf)) * (
                                                            self.lr*self.C_Sr*self.h + self.lf*self.C_Sf*self.h)*state[dynamic_mask][..., 6]
                                                        - self.mu * self.m / (self.I * (self.lr + self.lf)) * self.lf*self.C_Sf*self.h*state[dynamic_mask][..., 2])
                        + dvds[dynamic_mask][..., 6] * ((self.mu / (state[dynamic_mask][..., 3]**2 * (self.lr + self.lf)) * (self.C_Sr*self.h*self.lr + self.C_Sf*self.h*self.lf))*state[dynamic_mask][..., 5]
                                                        - self.mu / (state[dynamic_mask][..., 3] * (self.lr + self.lf)) * (
                                                            self.C_Sr*self.h - self.C_Sf*self.h)*state[dynamic_mask][..., 6]
                                                        - self.mu / (state[dynamic_mask][..., 3] * (self.lr + self.lf)) * self.C_Sf*self.h*state[dynamic_mask][..., 2])
                    )
            u = self.clamp_control(state, u)
            if unsqueeze_u:
                u = u[None, ...]
        return u

    def sample_target_state(self, num_samples):
        raise NotImplementedError

    def equivalent_wrapped_state(self, state):
        wrapped_state = torch.clone(state)
        wrapped_state[..., 4] = (
            wrapped_state[..., 4] + math.pi) % (2 * math.pi) - math.pi
        return wrapped_state

    def optimal_disturbance(self, state, dvds):
        return torch.tensor([0])

    def plot_config(self):
        return {
            'state_slices': [0, 0, 0, 8.0, 0, 0, 0],
            'state_labels': ['x', 'y', 'sangle', 'v', 'posetheta', 'poserate', 'slipangle'],
            'x_axis_idx': 0,
            'y_axis_idx': 1,
            'z_axis_idx': 4,
        }


class LessLinearND(Dynamics):
    def __init__(self, N: int, gamma: float, mu: float, alpha: float, goalR: float):
        u_max, set_mode = 0.5, "reach"  # TODO: unfix

        self.N = N
        self.u_max = u_max
        self.input_center = torch.zeros(N-1)
        self.input_shape = "box"
        self.game = set_mode

        self.A = (-0.5 * torch.eye(N) - torch.cat((torch.cat((torch.zeros(1, 1),
                  torch.ones(N-1, 1)), 0), torch.zeros(N, N-1)), 1)).cuda()
        self.B = torch.cat((torch.zeros(1, N-1), 0.4*torch.eye(N-1)), 0).cuda()
        self.Bumax = u_max * \
            torch.matmul(self.B, torch.ones(self.N-1).cuda()
                         ).unsqueeze(0).unsqueeze(0).cuda()
        self.C = torch.cat((torch.zeros(1, N-1), 0.1*torch.eye(N-1)), 0)
        self.gamma, self.mu, self.alpha = gamma, mu, alpha
        self.gamma_orig, self.mu_orig, self.alpha_orig = gamma, mu, alpha

        self.goalR_2d = goalR
        # accounts for N-dimensional combination
        self.goalR = ((N-1) ** 0.5) * self.goalR_2d
        # accounts for N-dimensional combination
        self.ellipse_params = torch.cat(
            (((N-1) ** 0.5) * torch.ones(1), torch.ones(N-1) / 1.), 0)

        self.state_range_ = torch.tensor(
            [[-1, 1] for _ in range(self.N)]).cuda()
        self.control_range_ = torch.tensor(
            [[-u_max, u_max] for _ in range(self.N-1)]).cuda()
        self.eps_var = torch.tensor([u_max for _ in range(self.N-1)]).cuda()
        self.control_init = torch.tensor([0.0 for _ in range(self.N-1)]).cuda()

        super().__init__(
            name='50D system', loss_type='brt_hjivi', set_mode=set_mode,
            state_dim=N, input_dim=N+1, control_dim=N-1, disturbance_dim=N-1,
            state_mean=[0 for _ in range(N)],
            state_var=[1 for _ in range(N)],
            value_mean=0.25,
            value_var=0.5,
            value_normto=0.02,
            deepReach_model="exact",
        )

    def vary_nonlinearity(self, epsilon):
        self.gamma = epsilon * self.gamma_orig
        self.mu = epsilon * self.mu_orig
        # self.alpha = epsilon * self.alpha_orig #shouldn't be varied since its not a scalar (1-\lambda) l(\cdot) +  \lambda f(\cdot)

    def state_test_range(self):
        return [[-1, 1] for _ in range(self.N)]

    def state_verification_range(self):
        return [[-1, 1] for _ in range(self.N)]

    def control_range(self, state):
        return self.control_range_.cpu().tolist()

    def equivalent_wrapped_state(self, state):
        wrapped_state = torch.clone(state)
        return wrapped_state

    # LessLinear dynamics
    # \dot xN    = (aN \cdot x) + (no ctrl or dist) + mu * sin(alpha * xN) * xN^2
    # \dot xi    = (ai \cdot x) + bi * ui + ci * di - gamma * xi * xN^2
    # i.e.
    # \dot x = Ax + Bu + Cd + NLterm(x, gamma, mu, alpha)
    # def dsdt(self, state, control, disturbance):
    #     dsdt = torch.zeros_like(state)
    #     nl_term_N = self.mu * torch.sin(self.alpha * state[..., 0]) * state[..., 0] * state[..., 0]
    #     nl_term_i = torch.multiply(-self.gamma * state[..., 0] * state[..., 0], state[..., 1:])
    #     dsdt[..., :] = torch.matmul(self.A, state[..., :]) + torch.matmul(self.B, control[..., :]) + torch.cat((nl_term_N, nl_term_i), 0)
    #     return dsdt
    def dsdt(self, state, control, disturbance):
        x0 = state[..., 0]  # shape: (...)
        x_rest = state[..., 1:]  # shape: (..., n-1)

        # Nonlinear terms
        nl_term_N = self.mu * \
            torch.sin(self.alpha * x0) * x0 * x0  # shape: (...)
        nl_term_N = nl_term_N.unsqueeze(-1)  # shape: (..., 1)

        x0_squared = (x0 ** 2).unsqueeze(-1)  # shape: (..., 1)
        nl_term_i = -self.gamma * x0_squared * \
            x_rest  # broadcasted: (..., n-1)

        nl_term = torch.cat([nl_term_N, nl_term_i], dim=-1)  # shape: (..., n)

        # Linear terms
        linear_term = torch.matmul(state, self.A.T) + \
            torch.matmul(control, self.B.T)

        return linear_term + nl_term

    def periodic_transform_fn(self, input):
        return input.cuda()

    def boundary_fn(self, state):
        if self.ellipse_params.device != state.device:  # FIXME: Patch to cover de/attached state bug
            if state.device.type == 'cuda':
                self.ellipse_params = self.ellipse_params.cuda()
            else:
                self.ellipse_params = self.ellipse_params.cpu()
        return 0.5 * (torch.square(torch.norm(self.ellipse_params * state[..., :], dim=-1)) - (self.goalR ** 2))
        # return 0.5 * (torch.square(torch.norm(torch.cat((((self.N-1)**0.5)*torch.ones(1),torch.ones(self.N-1)),0) * state[..., :], dim=-1)) - (self.goalR ** 2))

    def sample_target_state(self, num_samples):
        raise NotImplementedError

    def cost_fn(self, state_traj):
        return torch.min(self.boundary_fn(state_traj), dim=-1).values

    def hamiltonian(self, state, dvds):

        nl_term_N = (self.mu * torch.sin(self.alpha *
                     state[..., 0]) * state[..., 0] * state[..., 0]).unsqueeze(-1)
        nl_term_i = (-self.gamma * state[..., 0]
                     * state[..., 0]).t() * state[..., 1:]
        pAx = (dvds * (torch.matmul(state, self.A.t()) +
               torch.cat((nl_term_N, nl_term_i), 2))).sum(2)
        pBumax = (torch.abs(dvds) * self.Bumax).sum(2)

        if self.set_mode == 'reach':
            return pAx - pBumax
        elif self.set_mode == 'avoid':
            return pAx + pBumax

    def optimal_control(self, state, dvds):
        if self.set_mode == 'reach':
            return -self.u_max * torch.sign(dvds[..., 1:])
        elif self.set_mode == 'avoid':
            return self.u_max * torch.sign(dvds[..., 1:])

    def optimal_disturbance(self, state, dvds):
        return 0.0

    def plot_config(self):  # FIXME
        return {
            'state_slices': [0 for _ in range(self.N)],
            'state_labels': ['xN'] + ['x' + str(i) for i in range(1, self.N)],
            'x_axis_idx': 0,
            'y_axis_idx': 1,
            'z_axis_idx': 2,
        }

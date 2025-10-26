import numpy as np
import torch
from tqdm import tqdm
import torch.nn as nn
import math
import matplotlib.pyplot as plt
# mpl.use('Agg')
# torch.manual_seed(0)
# np.random.seed(0)

ROLLOUT_NUM = 100


class MPC:
    def __init__(self, dT, horizon, receding_horizon, num_samples, dynamics_, device, mode="MPC",
                  sample_mode="gaussian", lambda_=0.01, style= "direct",num_iterative_refinement=1):
        self.horizon = horizon
        self.num_samples = num_samples
        self.device = device
        self.receding_horizon = receding_horizon
        self.dynamics_=dynamics_

        self.dT = dT


        self.current_iter=0
        self.lambda_ = lambda_

        self.mode=mode
        self.sample_mode=sample_mode
        self.style = style # choice: receding, direct
        self.num_iterative_refinement=num_iterative_refinement


    def init_control_tensors(self):
        self.current_iter=0
        self.refine_iter_counter=0
        self.control_init =self.dynamics_.control_init.unsqueeze(0).repeat(self.batch_size,1)
        self.control_tensors = self.control_init.unsqueeze(1).repeat(1,self.horizon,1) # A * H * D_u
    

    def get_next_step_state(self, state, controls):
        current_dsdt = self.dynamics_.dsdt(
            state, controls, None)
        next_states= self.dynamics_.equivalent_wrapped_state(state + current_dsdt*self.dT)
        # next_states = torch.clamp(next_states, self.dynamics_.state_range_[..., 0], self.dynamics_.state_range_[..., 1])
        return next_states

    def rollout_dynamics(self, initial_state_tensor, rollout_horizon,eps_var_factor=1):
        # returns the state trajectory list and swith collision
        if self.sample_mode == "gaussian":
            epsilon_tensor = torch.randn(
                self.batch_size, self.num_samples, rollout_horizon, self.dynamics_.control_dim).to(self.device)*torch.sqrt(self.dynamics_.eps_var)*eps_var_factor # A * N * H * D_u

            epsilon_tensor[:, 0, ...] = 0.0  # always include the nominal trajectory
            # if self.num_iterative_refinement==0:
            #     epsilon_tensor*=0.0
            # receiding horizon with direct sampling
            epsilon_tensor[...,:self.receding_horizon*self.refine_iter_counter,:]=0.0

            permuted_controls = self.control_tensors[:,self.current_iter:self.current_iter+rollout_horizon,:].unsqueeze(1).repeat(1, 
                self.num_samples, 1, 1) + epsilon_tensor *1.0 # A * N * H * D_u
        elif self.sample_mode == "binary":
            permuted_controls = torch.sign(torch.empty(self.batch_size,self.num_samples, rollout_horizon, self.dynamics_.control_dim).uniform_(-1, 1)).to(self.device)
            permuted_controls [:, 0, ...] = self.control_tensors[:,self.current_iter:self.current_iter+rollout_horizon,:]*1.0 # always include the nominal trajectory

        # clamp control
        permuted_controls = torch.clamp(permuted_controls, self.dynamics_.control_range_[..., 0], self.dynamics_.control_range_[..., 1])

        # rollout trajs
        state_trajs = torch.zeros((self.batch_size, self.num_samples, rollout_horizon+1, self.dynamics_.state_dim)).to(self.device)  # A * N * H * D
        state_trajs[:, :, 0, :] = initial_state_tensor.unsqueeze(1).repeat(1, self.num_samples, 1) # A * N * D
        
        for k in range(rollout_horizon):
            permuted_controls[:, :, k, :]=self.dynamics_.clamp_control(state_trajs[:, :, k, :], permuted_controls[:, :, k, :])
            state_trajs[:, :, k+1,:]= self.get_next_step_state(
                state_trajs[:, :, k, :], permuted_controls[:, :, k, :])

        return state_trajs, permuted_controls
    

    def rollout_nominal_trajs(self,initial_state_tensor):
        # rollout trajs
        state_trajs = torch.zeros((self.batch_size, self.horizon+1, self.dynamics_.state_dim)).to(self.device)  # A * H * D
        state_trajs[:, 0, :] = initial_state_tensor*1.0 # A * D

        for k in range(self.horizon):
            self.control_tensors[:, k, :]=self.dynamics_.clamp_control(state_trajs[:, k, :], self.control_tensors[:, k, :])
            state_trajs[:, k+1,:]= self.get_next_step_state(
                state_trajs[:, k, :], self.control_tensors[:, k, :])
        return state_trajs
            
    def update_control_tensor(self, state_trajs, permuted_controls, receding=True):   
        costs = self.dynamics_.cost_fn(state_trajs) # A * N
        # if t_remaining>0.0:
        #     traj_times=torch.ones(self.batch_size,self.num_samples,1).to(self.device)*t_remaining
        #     state_trajs_clamped = torch.clamp(state_trajs[:, :, -1, :], torch.tensor(self.dynamics_.state_test_range(
        #                     )).to(self.device)[..., 0], torch.tensor(self.dynamics_.state_test_range()).to(self.device)[..., 1])
        #     # state_trajs_clamped = state_trajs[:, :, -1, :]*1.0
        #     traj_coords = torch.cat(
        #         (traj_times, state_trajs_clamped), dim=-1)
        #     traj_policy_results = policy(
        #         {'coords': self.dynamics_.coord_to_input(traj_coords.to(self.device))})
        #     terminal_values=self.dynamics_.io_to_value(traj_policy_results['model_in'].detach(
        #         ), traj_policy_results['model_out'].squeeze(dim=-1).detach())
        #     costs = torch.minimum(costs, terminal_values)

        # costs += self.dynamics_.boundary_fn(state_trajs[:,:,-1,:])*1e-6
        if self.mode=="MPC":
            # just use the best control
            if self.dynamics_.set_mode == 'avoid':
                best_costs, best_idx=costs.max(1)
            elif self.dynamics_.set_mode in ['reach', 'reach_avoid']:
                best_costs, best_idx=costs.min(1)
            else:
                raise NotImplementedError
            expanded_idx = best_idx[...,None, None, None].expand(-1, -1, permuted_controls.size(2), permuted_controls.size(3))  

            best_controls = torch.gather(permuted_controls, dim=1, index=expanded_idx).squeeze(1) # A * H * D_u

            control_tensors_ = best_controls*1.0
            expanded_idx_traj = best_idx[...,None, None, None].expand(-1, -1, state_trajs.size(2), state_trajs.size(3))  
            best_traj= torch.gather(state_trajs, dim=1, index=expanded_idx_traj).squeeze(1)
        elif self.mode=="MPPI":
            # use weighted average
            if self.dynamics_.set_mode == 'avoid':
                exp_terms = torch.exp((1/self.lambda_)*costs) # A * N
            elif self.dynamics_.set_mode in ['reach', 'reach_avoid']:
                exp_terms = torch.exp((1/self.lambda_)*-costs) # A * N
            else:
                raise NotImplementedError
            
            den = torch.sum(exp_terms, dim=-1) # A

            num = torch.sum(exp_terms[:, :, None, None].repeat(1,1,self.horizon, self.dynamics_.control_dim) * permuted_controls, dim=1) # A * H * D_u

            control_tensors_= num/den[:,None,None]
            control_tensors_= torch.clamp(
                    control_tensors_, self.dynamics_.control_range_[..., 0], self.dynamics_.control_range_[..., 1])
        else:
            raise NotImplementedError
        
        # Update controls
        current_controls = control_tensors_[:, :self.receding_horizon, :]
        if not receding:
            self.control_tensors=control_tensors_*1.0
        else:
            self.control_tensors[:, self.current_iter:,
                                :] = control_tensors_*1.0
            # self.control_tensors[:, self.horizon-self.receding_horizon:, :] = self.control_init.unsqueeze(1).repeat(1,self.receding_horizon,1) # A * H_r * D_u 


        return current_controls, best_traj, best_costs
    

    def rollout_with_policy(self, initial_condition_tensor, policy, policy_horizon, policy_start_iter=0):
        state_trajs = torch.zeros((self.batch_size, policy_horizon+1, self.dynamics_.state_dim)).to(self.device)  # A * H * D
        state_trajs[:, 0, :] = initial_condition_tensor*1.0
        state_trajs_clamped=state_trajs*1.0
        traj_times=torch.ones(self.batch_size,1).to(self.device)*policy_horizon*self.dT
        # update control from policy_start_iter to policy_start_iter+ policy horizon
        for k in range(policy_horizon):
            
            traj_coords = torch.cat(
                (traj_times, state_trajs_clamped[:, k, :]), dim=-1)
            traj_policy_results = policy(
                {'coords': self.dynamics_.coord_to_input(traj_coords.to(self.device))})
            traj_dvs = self.dynamics_.io_to_dv(
                traj_policy_results['model_in'], traj_policy_results['model_out'].squeeze(dim=-1)).detach()
        
            self.control_tensors[:, k+policy_start_iter, :] = self.dynamics_.optimal_control(
                traj_coords[:, 1:].to(self.device), traj_dvs[..., 1:].to(self.device))
            self.control_tensors[:, k+policy_start_iter, :]=self.dynamics_.clamp_control(state_trajs[:, k, :], self.control_tensors[:, k+policy_start_iter, :])
            state_trajs[:, k+1,:] = self.get_next_step_state(
                state_trajs[:, k, :], self.control_tensors[:, k+policy_start_iter, :])

            state_trajs_clamped[:, k+1,:] = torch.clamp(state_trajs[:, k+1,:], torch.tensor(self.dynamics_.state_test_range(
                    )).to(self.device)[..., 0], torch.tensor(self.dynamics_.state_test_range()).to(self.device)[..., 1])
            traj_times=traj_times-self.dT

        return state_trajs
    
    def warm_start_with_policy(self, initial_condition_tensor, policy=None, t_remaining=None):
        
        if self.incremental_horizon>0:
            # Rollout with the incremental horizon
            state_trajs_H, permuted_controls_H = self.rollout_dynamics(initial_condition_tensor, self.incremental_horizon)
        
            costs = self.dynamics_.cost_fn(state_trajs_H)  # A * N
            # Use the learned value function for terminal cost and compute the cost function
            if t_remaining>0.0:
                traj_times=torch.ones(self.batch_size,self.num_samples,1).to(self.device)*t_remaining
                state_trajs_clamped = torch.clamp(state_trajs_H[:, :, -1, :], torch.tensor(self.dynamics_.state_test_range(
                                )).to(self.device)[..., 0], torch.tensor(self.dynamics_.state_test_range()).to(self.device)[..., 1])

                traj_coords = torch.cat(
                    (traj_times, state_trajs_clamped), dim=-1)
                traj_policy_results = policy(
                    {'coords': self.dynamics_.coord_to_input(traj_coords.to(self.device))})
                terminal_values=self.dynamics_.io_to_value(traj_policy_results['model_in'].detach(
                    ), traj_policy_results['model_out'].squeeze(dim=-1).detach())
                if self.incremental_horizon>0:
                    costs = torch.minimum(costs, terminal_values)
                        
                    if self.dynamics_.set_mode == 'reach_avoid':
                        print(t_remaining,"B",state_trajs_H.shape,self.incremental_horizon)
                        avoid_value_max=torch.max(-self.dynamics_.avoid_fn(state_trajs_H), dim=-1).values
                        costs = torch.maximum(costs, avoid_value_max)
                else:
                    costs=terminal_values*1.0
            # Pick the best control and correponding traj
            if self.dynamics_.set_mode == 'avoid':
                best_costs, best_idx=costs.max(1)
            elif self.dynamics_.set_mode in ['reach', 'reach_avoid']:
                best_costs, best_idx=costs.min(1)
            else:
                raise NotImplementedError
            expanded_idx = best_idx[...,None, None, None].expand(-1, -1, permuted_controls_H.size(2), permuted_controls_H.size(3))  

            best_controls_H = torch.gather(permuted_controls_H, dim=1, index=expanded_idx).squeeze(1) # A * H * D_u
            expanded_idx_traj = best_idx[...,None, None, None].expand(-1, -1, state_trajs_H.size(2), state_trajs_H.size(3))  
            best_traj_H= torch.gather(state_trajs_H, dim=1, index=expanded_idx_traj).squeeze(1)

            # Rollout the remaining horizon with the learned policy and update the nominal control traj
            self.control_tensors[:,:self.incremental_horizon,:]=best_controls_H*1.0
            self.warm_start_traj = self.rollout_with_policy(best_traj_H[:,-1,:],policy,self.horizon-self.incremental_horizon,self.incremental_horizon)
            self.warm_start_traj = torch.cat([best_traj_H,self.warm_start_traj],dim=1)
        else:
            # Rollout using the learned policy and update the nominal control traj
            self.warm_start_traj = self.rollout_with_policy(initial_condition_tensor,policy,self.horizon)
    def get_control(self, initial_condition_tensor, num_iterative_refinement=1, policy=None, t_remaining=None):
        
        if self.style == 'direct':
            # last_best_costs=torch.ones(self.batch_size).to(self.device)*torch.finfo().max
            if num_iterative_refinement==-1: # rollout using the policy
                best_traj = self.rollout_with_policy(initial_condition_tensor,policy,self.horizon)
                # num_iterative_refinement=1
            for i in range(num_iterative_refinement+1):
                eps_var_factor=1
                self.refine_iter_counter=i
                # nominal_traj=self.rollout_nominal_trajs(initial_condition_tensor).cpu()
                state_trajs, permuted_controls = self.rollout_dynamics(initial_condition_tensor, self.horizon, eps_var_factor)
                self.all_state_trajs=state_trajs.detach().cpu()*1.0
                current_controls, best_traj, best_costs = self.update_control_tensor(
                    state_trajs, permuted_controls, receding=False) 
                
                
            return self.control_tensors, best_traj
        elif self.style == 'receding':
            # initial_condition_tensor: A*D
            state_trajs, permuted_controls = self.rollout_dynamics(initial_condition_tensor, self.horizon)

            current_controls, best_traj, best_costs = self.update_control_tensor(
                state_trajs, permuted_controls) 
        
            return current_controls, best_traj

    def get_opt_trajs(self,initial_condition_tensor, policy=None, t_remaining=0.0):
        
        num_iters = math.ceil((self.T)/self.dT)
        self.horizon = math.ceil((self.T)/self.dT)
        self.incremental_horizon =  math.ceil((self.T-t_remaining)/self.dT)
        if self.style == 'direct':
            
            self.init_control_tensors()
            if policy is not None:
                self.warm_start_with_policy(initial_condition_tensor, policy, t_remaining)
            best_controls, best_trajs = self.get_control(
                    initial_condition_tensor, self.num_iterative_refinement, policy, t_remaining=t_remaining)
            
            if self.dynamics_.set_mode in ['avoid', 'reach']:
                lxs = self.dynamics_.boundary_fn(best_trajs)   
                return best_trajs, lxs, num_iters, best_controls
            elif self.dynamics_.set_mode == 'reach_avoid':
                avoid_values=self.dynamics_.avoid_fn(best_trajs) 
                reach_values=self.dynamics_.reach_fn(best_trajs) 
                return best_trajs, avoid_values, reach_values, num_iters, best_controls
            else:
                raise NotImplementedError
            

        elif self.style == 'receding':
            # if self.dynamics_.set_mode =='reach_avoid':
            #     raise NotImplementedError

            state_trajs = torch.zeros(( self.batch_size, num_iters+1, self.dynamics_.state_dim)).to(self.device)  # A*H*D
            state_trajs[:, 0, :] = initial_condition_tensor

            self.init_control_tensors()
            if policy is not None:
                self.warm_start_with_policy(initial_condition_tensor, policy, t_remaining)
            lxs=torch.zeros(self.batch_size, num_iters+1).to(self.device)
            self.current_iter=0
            for i in tqdm(range(int(num_iters/self.receding_horizon))):
                best_controls,_ = self.get_control(
                        state_trajs[:,self.current_iter, :])
                for k in range(self.receding_horizon):
                    lxs[:,i*self.receding_horizon+k] = self.dynamics_.boundary_fn(
                                            state_trajs[:, i*self.receding_horizon+k, :]) 
                    best_controls[:, k, :]=self.dynamics_.clamp_control(state_trajs[:,i*self.receding_horizon+k,:], best_controls[:, k, :])
                    state_trajs[:,i*self.receding_horizon+1+k,:] = self.get_next_step_state(
                        state_trajs[:,i*self.receding_horizon+k,:], best_controls[:, k, :])
                self.current_iter+=self.receding_horizon
                self.horizon-=self.receding_horizon
            lxs[:,-1] = self.dynamics_.boundary_fn(state_trajs[:, -1, :]) 
            # return state_trajs, lxs, num_iters
            if self.dynamics_.set_mode in ['avoid', 'reach']:
                lxs = self.dynamics_.boundary_fn(state_trajs)   
                return state_trajs, lxs, num_iters, self.control_tensors
            elif self.dynamics_.set_mode == 'reach_avoid':
                avoid_values=self.dynamics_.avoid_fn(state_trajs) 
                reach_values=self.dynamics_.reach_fn(state_trajs) 
                return state_trajs, avoid_values, reach_values, num_iters, self.control_tensors
            else:
                raise NotImplementedError
        else:
            return NotImplementedError
        
    
    def get_batch_data(self, initial_condition_tensor, T, policy=None, t=0.0, style="random"):
        self.T=T*1.0
        self.batch_size=initial_condition_tensor.shape[0]
        if self.dynamics_.set_mode in ['avoid', 'reach']:
            state_trajs, lxs, num_iters, best_controls = self.get_opt_trajs(initial_condition_tensor, policy, t)
            costs,_=torch.min(lxs,dim=-1)
            
        elif self.dynamics_.set_mode == 'reach_avoid':
            state_trajs, avoid_values, reach_values, num_iters, best_controls = self.get_opt_trajs(initial_condition_tensor,policy, t)
            # costs=torch.min(torch.maximum(reach_values, torch.cummax(-avoid_values, dim=-1).values), dim=-1).values
            costs=torch.min(torch.clamp(reach_values, min=torch.max(-avoid_values, dim=-1).values.unsqueeze(-1)),dim=-1).values
        else:
            raise NotImplementedError
 
        if self.style == 'receding':
            # generating MPC dataset: {..., (t, x, J, u), ...}
            # if style=='terminal': # only generate terminal time samples
            #     num_iters=1
            coords=torch.zeros(self.batch_size* num_iters, self.dynamics_.state_dim+1).to(self.device)
            value_labels=torch.zeros(self.batch_size* num_iters).to(self.device)
            for i in range(num_iters):
                coords[i*self.batch_size: (i+1)*self.batch_size ,0] = self.T - i * self.dT
                coords[i*self.batch_size: (i+1)*self.batch_size,1:] = state_trajs[:, i, :]
                if self.dynamics_.set_mode in ['avoid', 'reach']:
                    value_labels[i*self.batch_size: (i+1)*self.batch_size],_ =  torch.min(lxs[..., i:],dim=-1) 
                elif self.dynamics_.set_mode == 'reach_avoid':
                    # value_labels[i*self.batch_size: (i+1)*self.batch_size] =  \
                    #             torch.min(torch.maximum(reach_values[..., i:], torch.cummax(-avoid_values[..., i:], dim=-1).values), dim=-1).values
                    value_labels[i*self.batch_size: (i+1)*self.batch_size] = \
                        torch.min(torch.clamp(reach_values[..., i:], min=torch.max(-avoid_values[..., i:], dim=-1).values.unsqueeze(-1)),dim=-1).values
                else:
                    raise NotImplementedError
            
                
            
            ##################### only use in range labels ###################################################
            output1 = torch.all(coords[...,1:] >= self.dynamics_.state_range_[
                                :, 0]-0.01, -1, keepdim=False)
            output2 = torch.all(coords[...,1:] <= self.dynamics_.state_range_[
                                :, 1]+0.01, -1, keepdim=False)
            in_range_index = torch.logical_and(torch.logical_and(output1, output2), ~torch.isnan(value_labels))


            coords=coords[in_range_index]
            value_labels=value_labels[in_range_index]
            ###################################################################################################

        elif self.style == 'direct':

            # generating MPC dataset: {..., (t, x, J, u), ...} NEW
            # if style=='terminal': # only generate terminal time samples
            #     num_iters=1
            coords=torch.empty(0, self.dynamics_.state_dim+1).to(self.device)
            value_labels=torch.empty(0).to(self.device)
            if self.dynamics_.set_mode in ['avoid', 'reach']: # bootstrapping will be accurate up until the min l(x) occur
                _,min_idx=torch.min(lxs,dim=-1) 
            elif self.dynamics_.set_mode == 'reach_avoid':
                # _,min_idx=torch.min(torch.maximum(reach_values, torch.cummax(-avoid_values, dim=-1).values), dim=-1)
                _,min_idx=torch.min(torch.clamp(reach_values, min=torch.max(-avoid_values, dim=-1).values.unsqueeze(-1)),dim=-1)
            for i in range(num_iters):
                coord_i=torch.zeros(self.batch_size,self.dynamics_.state_dim+1).to(self.device)
                coord_i[: ,0] = self.T - i * self.dT
                coord_i[:,1:] = state_trajs[:, i, :]
                if self.dynamics_.set_mode in ['avoid', 'reach']:
                    valid_idx=(min_idx>i).nonzero(as_tuple=True)
                    value_labels_i=  torch.min(lxs[valid_idx[0], i:],dim=-1).values 
                    coord_i=coord_i[valid_idx]
                elif self.dynamics_.set_mode == 'reach_avoid':
                    valid_idx=(min_idx>i).nonzero(as_tuple=True)
                    # value_labels_i =  \
                    #             torch.min(torch.maximum(reach_values[valid_idx[0], i:], torch.cummax(-avoid_values[valid_idx[0], i:], dim=-1).values), dim=-1).values
                    value_labels_i =  \
                                torch.min(torch.clamp(reach_values[valid_idx[0], i:], min=torch.max(-avoid_values[valid_idx[0], i:], dim=-1).values.unsqueeze(-1)),dim=-1).values
                    coord_i=coord_i[valid_idx]
                else:
                    raise NotImplementedError
                # add to data
                coords=torch.cat((coords,coord_i),dim=0)
                value_labels=torch.cat((value_labels,value_labels_i),dim=0)
                
            
            ##################### only use in range labels ###################################################
            output1 = torch.all(coords[...,1:] >= self.dynamics_.state_range_[
                                :, 0]-0.01, -1, keepdim=False)
            output2 = torch.all(coords[...,1:] <= self.dynamics_.state_range_[
                                :, 1]+0.01, -1, keepdim=False)
            in_range_index = torch.logical_and(torch.logical_and(output1, output2), ~torch.isnan(value_labels))


            coords=coords[in_range_index]
            value_labels=value_labels[in_range_index]

            ###################################################################################################
        else:
            raise NotImplementedError

        # coords=torch.empty(0, self.dynamics_.state_dim+1).to(self.device)
        # value_labels=torch.empty(0).to(self.device)
        # if self.dynamics_.set_mode in ['avoid', 'reach']: # bootstrapping will be accurate up until the min l(x) occur
        #     _,min_idx=torch.min(lxs,dim=-1) 
        # elif self.dynamics_.set_mode == 'reach_avoid':
        #     _,min_idx=torch.min(torch.maximum(reach_values, torch.cummax(-avoid_values, dim=-1).values), dim=-1)
        # for i in range(num_iters):
        #     coord_i=torch.zeros(self.batch_size,self.dynamics_.state_dim+1).to(self.device)
        #     coord_i[: ,0] = self.T - i * self.dT
        #     coord_i[:,1:] = state_trajs[:, i, :]
        #     if self.dynamics_.set_mode in ['avoid', 'reach']:
        #         valid_idx=(min_idx>i).nonzero(as_tuple=True)
        #         value_labels_i=  torch.min(lxs[valid_idx[0], i:],dim=-1).values 
        #         coord_i=coord_i[valid_idx]
        #     elif self.dynamics_.set_mode == 'reach_avoid':
        #         valid_idx=(min_idx>i).nonzero(as_tuple=True)
        #         value_labels_i =  \
        #                     torch.min(torch.maximum(reach_values[valid_idx[0], i:], torch.cummax(-avoid_values[valid_idx[0], i:], dim=-1).values), dim=-1).values
        #         coord_i=coord_i[valid_idx]
        #     else:
        #         raise NotImplementedError
        #     # add to data
        #     coords=torch.cat((coords,coord_i),dim=0)
        #     value_labels=torch.cat((value_labels,value_labels_i),dim=0)
            
        
        # ##################### only use in range labels ###################################################
        # output1 = torch.all(coords[...,1:] >= self.dynamics_.state_range_[
        #                     :, 0]-0.01, -1, keepdim=False)
        # output2 = torch.all(coords[...,1:] <= self.dynamics_.state_range_[
        #                     :, 1]+0.01, -1, keepdim=False)
        # in_range_index = torch.logical_and(torch.logical_and(output1, output2), ~torch.isnan(value_labels))


        # coords=coords[in_range_index]
        # value_labels=value_labels[in_range_index]
        
        coords=self.dynamics_.coord_to_input(coords)
        return costs, state_trajs, coords.detach().cpu().clone(), value_labels.detach().cpu().clone(), None
            

class MPCNet(nn.Module):
    def __init__(self, input_dim=3, num_hl=128):
        super(MPCNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, num_hl)
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(num_hl, num_hl)
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(num_hl, 1)

    def forward(self, coords):
        x = torch.relu(self.fc1(coords))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x.squeeze(-1)
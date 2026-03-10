"""
Visualization mixin for Experiment class.

This module provides visualization capabilities for MPC dataset analysis,
value function plotting, and recovery visualization.
"""
import wandb
import torch
import os
import math
import gc
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.io as spio
from utils import MPC


class ExperimentVizMixin:
    """Visualization methods for Experiment class.
    
    This mixin provides visualization capabilities for MPC dataset analysis,
    value function plotting, and recovery visualization.
    
    Expects the following attributes on self:
    - self.model: Neural network model
    - self.dataset: ReachabilityDataset instance
    - self.use_wandb: Boolean for wandb logging
    - self.experiment_dir: Path to experiment directory
    """

    def _quat_to_yaw(self, quat):
        # quat: [q0, q1, q2, q3] scalar-first
        q0, q1, q2, q3 = quat
        siny_cosp = 2.0 * (q0 * q3 + q1 * q2)
        cosy_cosp = 1.0 - 2.0 * (q2 * q2 + q3 * q3)
        return math.atan2(siny_cosp, cosy_cosp)

    def _annotate_quat(self, ax, quat_slice, theta_yaw):
        if quat_slice is None or theta_yaw is None:
            return
        q_text = f"q=[{quat_slice[0]:.3f},{quat_slice[1]:.3f},{quat_slice[2]:.3f},{quat_slice[3]:.3f}]"
        t_text = f"theta_yaw={theta_yaw:.3f} rad"
        ax.text(
            0.02, 0.98,
            f"{q_text}\n{t_text}",
            transform=ax.transAxes,
            ha='left', va='top',
            fontsize=8,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7, edgecolor='none')
        )

    def plotSingleFig(self, state_test_range, plot_config, x_resolution, y_resolution, times, delta_level=None,
                      quat_slice=None, theta_yaw=None, draw_target_set=False):
        x_min, x_max = state_test_range[plot_config['x_axis_idx']]
        y_min, y_max = state_test_range[plot_config['y_axis_idx']]

        xs = torch.linspace(x_min, x_max, x_resolution)
        ys = torch.linspace(y_min, y_max, y_resolution)
        xys = torch.cartesian_prod(xs, ys)
        fig = plt.figure(figsize=(6, 5*len(times)))
        X, Y = np.meshgrid(xs, ys)
        for i in range(len(times)):
            coords = torch.zeros(
                x_resolution*y_resolution, self.dataset.dynamics.state_dim + 1)
            coords[:, 0] = times[i]
            coords[:, 1:] = torch.tensor(plot_config['state_slices'])
            coords[:, 1 + plot_config['x_axis_idx']] = xys[:, 0]
            coords[:, 1 + plot_config['y_axis_idx']] = xys[:, 1]

            with torch.no_grad():
                model_results = self.model(
                    {'coords': self.dataset.dynamics.coord_to_input(coords.cuda())})

                values = self.dataset.dynamics.io_to_value(model_results['model_in'].detach(
                ), model_results['model_out'].squeeze(dim=-1).detach())

            ax = fig.add_subplot(len(times), 1, 1 + i)
            ax.set_title('t = %0.2f' % (times[i]))
            BRT_img = values.detach().cpu().numpy().reshape(x_resolution, y_resolution).T
            max_value = np.amax(BRT_img)
            min_value = np.amin(BRT_img)
            imshow_kwargs = {
                'vmax': max_value,
                'vmin': min_value,
                'cmap': 'coolwarm_r',
                'extent': (x_min, x_max, y_min, y_max),
                'origin': 'lower',
            }
            ax.imshow(BRT_img, **imshow_kwargs)
            lx = self.dataset.dynamics.boundary_fn(coords.cuda()[..., 1:]).detach(
            ).cpu().numpy().reshape(x_resolution, y_resolution).T
            zero_contour = ax.contour(X,
                                      Y,
                                      BRT_img,
                                      levels=[0.0],
                                      colors="black",
                                      linewidths=2,
                                      linestyles='--')

            failure_set_contour = ax.contour(X,
                                             Y,
                                             lx,
                                             levels=[0.0],
                                             colors="saddlebrown",
                                             linewidths=2,
                                             linestyles='-')
            if self.dataset.dynamics.name == 'Docking13D':
                self._annotate_quat(ax, quat_slice, theta_yaw)
            if draw_target_set and hasattr(self.dataset.dynamics, 'reach_fn'):
                reach_vals = self.dataset.dynamics.reach_fn(
                    coords.cuda()[..., 1:]).detach().cpu().numpy().reshape(x_resolution, y_resolution).T
                try:
                    ax.contour(X, Y, reach_vals, levels=[0.0], colors='limegreen',
                               linewidths=1.5, linestyles='--')
                except Exception:
                    pass
            if delta_level is not None:
                delta_contour = ax.contour(X,
                                           Y,
                                           BRT_img,
                                           levels=[delta_level],
                                           colors="black",
                                           linewidths=2,
                                           linestyles='-')
        return fig

    def plotMultipleFigs(self, state_test_range, plot_config, x_resolution, y_resolution, z_resolution, times,
                         delta_level=None, quat_slice=None, theta_yaw=None, z_values=None, draw_target_set=False):
        x_min, x_max = state_test_range[plot_config['x_axis_idx']]
        y_min, y_max = state_test_range[plot_config['y_axis_idx']]
        z_min, z_max = state_test_range[plot_config['z_axis_idx']]

        xs = torch.linspace(x_min, x_max, x_resolution)
        ys = torch.linspace(y_min, y_max, y_resolution)
        if z_values is not None:
            zs = torch.tensor(z_values, dtype=torch.float32)
        else:
            zs = torch.linspace(z_min, z_max, z_resolution)
        xys = torch.cartesian_prod(xs, ys)

        fig = plt.figure(figsize=(6*len(zs), 5*len(times)))
        X, Y = np.meshgrid(xs, ys)
        for i in range(len(times)):
            for j in range(len(zs)):
                coords = torch.zeros(
                    x_resolution*y_resolution, self.dataset.dynamics.state_dim + 1)
                coords[:, 0] = times[i]
                coords[:, 1:] = torch.tensor(plot_config['state_slices'])
                coords[:, 1 + plot_config['x_axis_idx']] = xys[:, 0]
                coords[:, 1 + plot_config['y_axis_idx']] = xys[:, 1]
                coords[:, 1 + plot_config['z_axis_idx']] = zs[j]

                lx = self.dataset.dynamics.boundary_fn(coords.cuda()[..., 1:]).detach(
                ).cpu().numpy().reshape(x_resolution, y_resolution).T
                with torch.no_grad():
                    model_results = self.model(
                        {'coords': self.dataset.dynamics.coord_to_input(coords.cuda())})
                    values = self.dataset.dynamics.io_to_value(model_results['model_in'].detach(
                    ), model_results['model_out'].squeeze(dim=-1).detach())

                ax = fig.add_subplot(len(times), len(zs), (j+1) + i*len(zs))
                ax.set_title('t = %0.2f, %s = %0.2f' % (
                    times[i], plot_config['state_labels'][plot_config['z_axis_idx']], zs[j]))

                BRT_img = values.detach().cpu().numpy().reshape(x_resolution, y_resolution).T

                max_value = np.amax(BRT_img)
                min_value = np.amin(BRT_img)
                imshow_kwargs = {
                    'vmax': max_value,
                    'vmin': min_value,
                    'cmap': 'coolwarm_r',
                    'extent': (x_min, x_max, y_min, y_max),
                    'origin': 'lower',
                }

                s1 = ax.imshow(BRT_img, **imshow_kwargs)
                fig.colorbar(s1)
                zero_contour = ax.contour(X,
                                          Y,
                                          BRT_img,
                                          levels=[0.0],
                                          colors="black",
                                          linewidths=1,
                                          linestyles='--')

                failure_set_contour = ax.contour(X,
                                                 Y,
                                                 lx,
                                                 levels=[0.0],
                                                 colors="brown",
                                                 linewidths=2,
                                                 linestyles='-')

                if self.dataset.dynamics.name == 'Docking13D':
                    self._annotate_quat(ax, quat_slice, theta_yaw)

                if draw_target_set and hasattr(self.dataset.dynamics, 'reach_fn'):
                    reach_vals = self.dataset.dynamics.reach_fn(
                        coords.cuda()[..., 1:]).detach().cpu().numpy().reshape(x_resolution, y_resolution).T
                    try:
                        ax.contour(X, Y, reach_vals, levels=[0.0], colors='limegreen',
                                   linewidths=1.5, linestyles='--')
                    except Exception:
                        pass

                if delta_level is not None:
                    delta_contour = ax.contour(X,
                                               Y,
                                               BRT_img,
                                               levels=[delta_level],
                                               colors="black",
                                               linewidths=1,
                                               linestyles='-')

        return fig

    def plotVelocitySlice(self, state_test_range, plot_config, x_resolution, y_resolution, z_resolution, times, delta_level=None):
        """
        Plot the learned value function over the velocity plane (vx vs vy) with theta slicing.
        
        - x_axis: vx (index 2)
        - y_axis: vy (index 3)
        - z_axis (slicing): theta (index 4)
        - Fixed dimensions: px, py, omega (indices 0, 1, 5)
        
        Args:
            state_test_range: State bounds from dynamics.state_test_range()
            plot_config: Configuration dict from dynamics.plot_config()
            x_resolution: Number of grid points along vx axis
            y_resolution: Number of grid points along vy axis
            z_resolution: Number of theta slices
            times: Time values to plot (creates rows in the figure)
            delta_level: Optional additional contour level to plot
            
        Returns:
            fig: matplotlib figure object
        """
        # Velocity plane axes (matching MPC velocity viz dimensions)
        x_axis_idx = 2  # vx
        y_axis_idx = 3  # vy
        z_axis_idx = 4  # theta (slicing dimension)
        
        # Get grid ranges
        x_min, x_max = state_test_range[x_axis_idx]  # vx range
        y_min, y_max = state_test_range[y_axis_idx]  # vy range
        z_min, z_max = state_test_range[z_axis_idx]  # theta range
        
        # Create grid points
        xs = torch.linspace(x_min, x_max, x_resolution)
        ys = torch.linspace(y_min, y_max, y_resolution)
        zs = torch.linspace(z_min, z_max, z_resolution)
        xys = torch.cartesian_prod(xs, ys)
        
        # Create figure: time rows x theta columns
        fig = plt.figure(figsize=(6 * len(zs), 5 * len(times)))
        X, Y = np.meshgrid(xs, ys)
        
        # State labels for velocity slice
        state_labels = ['px', 'py', 'vx', 'vy', r'$\theta$', r'$\omega$']
        
        for i in range(len(times)):
            for j in range(len(zs)):
                # Build coordinate tensor: [time, px, py, vx, vy, theta, omega]
                coords = torch.zeros(
                    x_resolution * y_resolution, self.dataset.dynamics.state_dim + 1)
                coords[:, 0] = times[i]
                coords[:, 1:] = torch.tensor(plot_config['state_slices'])
                # Set vx (x-axis) and vy (y-axis) from grid
                coords[:, 1 + x_axis_idx] = xys[:, 0]  # vx
                coords[:, 1 + y_axis_idx] = xys[:, 1]  # vy
                # Set theta (z-axis) from slice value
                coords[:, 1 + z_axis_idx] = zs[j]  # theta
                
                # Compute boundary function for contour
                lx = self.dataset.dynamics.boundary_fn(coords.cuda()[..., 1:]).detach(
                ).cpu().numpy().reshape(x_resolution, y_resolution).T
                
                # Evaluate model to get value function
                with torch.no_grad():
                    model_results = self.model(
                        {'coords': self.dataset.dynamics.coord_to_input(coords.cuda())})
                    values = self.dataset.dynamics.io_to_value(
                        model_results['model_in'].detach(),
                        model_results['model_out'].squeeze(dim=-1).detach())
                
                # Create subplot
                ax = fig.add_subplot(len(times), len(zs), (j + 1) + i * len(zs))
                ax.set_title('t = %0.2f, %s = %0.2f' % (
                    times[i], state_labels[z_axis_idx], zs[j]))
                
                # Reshape values for plotting
                BRT_img = values.detach().cpu().numpy().reshape(x_resolution, y_resolution).T
                
                max_value = np.amax(BRT_img)
                min_value = np.amin(BRT_img)
                imshow_kwargs = {
                    'vmax': max_value,
                    'vmin': min_value,
                    'cmap': 'coolwarm_r',
                    'extent': (x_min, x_max, y_min, y_max),
                    'origin': 'lower',
                }
                
                # Plot heatmap
                s1 = ax.imshow(BRT_img, **imshow_kwargs)
                fig.colorbar(s1)
                
                # Add zero-level contour (BRT boundary)
                zero_contour = ax.contour(X, Y, BRT_img,
                                          levels=[0.0],
                                          colors="black",
                                          linewidths=1,
                                          linestyles='--')
                
                # Add boundary function contour
                failure_set_contour = ax.contour(X, Y, lx,
                                                 levels=[0.0],
                                                 colors="brown",
                                                 linewidths=2,
                                                 linestyles='-')
                
                # Add optional delta level contour
                if delta_level is not None:
                    delta_contour = ax.contour(X, Y, BRT_img,
                                               levels=[delta_level],
                                               colors="black",
                                               linewidths=1,
                                               linestyles='-')
                
                # Set axis labels
                ax.set_xlabel(state_labels[x_axis_idx])
                ax.set_ylabel(state_labels[y_axis_idx])
        
        # Add figure title with fixed state information
        state_slices = plot_config['state_slices']
        fixed_str = f'Fixed: px={state_slices[0]:.2f}, py={state_slices[1]:.2f}, ω={state_slices[5]:.2f}'
        fig.suptitle(f'Value Function - Velocity Slice\n{fixed_str}', fontsize=14)
        
        plt.tight_layout()
        return fig

    def plotRotationSlice(self, state_test_range, plot_config, x_resolution, y_resolution, times, delta_level=None):
        """
        Plot the learned value function over the rotation plane (theta vs omega) without z-slicing.
        
        - x_axis: theta (index 4)
        - y_axis: omega (index 5)
        - No z-slicing (only time slicing)
        - Fixed dimensions: px, py, vx, vy (indices 0, 1, 2, 3)
        
        Args:
            state_test_range: State bounds from dynamics.state_test_range()
            plot_config: Configuration dict from dynamics.plot_config()
            x_resolution: Number of grid points along theta axis
            y_resolution: Number of grid points along omega axis
            times: Time values to plot (creates columns in the figure)
            delta_level: Optional additional contour level to plot
            
        Returns:
            fig: matplotlib figure object
        """
        # Rotation plane axes (matching MPC rotation viz dimensions)
        x_axis_idx = 4  # theta
        y_axis_idx = 5  # omega
        
        # Get grid ranges
        x_min, x_max = state_test_range[x_axis_idx]  # theta range
        y_min, y_max = state_test_range[y_axis_idx]  # omega range
        
        # Create grid points
        xs = torch.linspace(x_min, x_max, x_resolution)
        ys = torch.linspace(y_min, y_max, y_resolution)
        xys = torch.cartesian_prod(xs, ys)
        
        # Create figure: 1 row x time columns (no z-slicing)
        fig = plt.figure(figsize=(6 * len(times), 5))
        X, Y = np.meshgrid(xs, ys)
        
        # State labels for rotation slice
        state_labels = ['px', 'py', 'vx', 'vy', r'$\theta$', r'$\omega$']
        
        for i in range(len(times)):
            # Build coordinate tensor: [time, px, py, vx, vy, theta, omega]
            coords = torch.zeros(
                x_resolution * y_resolution, self.dataset.dynamics.state_dim + 1)
            coords[:, 0] = times[i]
            coords[:, 1:] = torch.tensor(plot_config['state_slices'])
            # Set theta (x-axis) and omega (y-axis) from grid
            coords[:, 1 + x_axis_idx] = xys[:, 0]  # theta
            coords[:, 1 + y_axis_idx] = xys[:, 1]  # omega
            
            # Compute boundary function for contour
            lx = self.dataset.dynamics.boundary_fn(coords.cuda()[..., 1:]).detach(
            ).cpu().numpy().reshape(x_resolution, y_resolution).T
            
            # Evaluate model to get value function
            with torch.no_grad():
                model_results = self.model(
                    {'coords': self.dataset.dynamics.coord_to_input(coords.cuda())})
                values = self.dataset.dynamics.io_to_value(
                    model_results['model_in'].detach(),
                    model_results['model_out'].squeeze(dim=-1).detach())
            
            # Create subplot (1 row)
            ax = fig.add_subplot(1, len(times), i + 1)
            ax.set_title('t = %0.2f' % (times[i]))
            
            # Reshape values for plotting
            BRT_img = values.detach().cpu().numpy().reshape(x_resolution, y_resolution).T
            
            max_value = np.amax(BRT_img)
            min_value = np.amin(BRT_img)
            imshow_kwargs = {
                'vmax': max_value,
                'vmin': min_value,
                'cmap': 'coolwarm_r',
                'extent': (x_min, x_max, y_min, y_max),
                'origin': 'lower',
            }
            
            # Plot heatmap
            s1 = ax.imshow(BRT_img, **imshow_kwargs)
            fig.colorbar(s1)
            
            # Add zero-level contour (BRT boundary)
            zero_contour = ax.contour(X, Y, BRT_img,
                                      levels=[0.0],
                                      colors="black",
                                      linewidths=1,
                                      linestyles='--')
            
            # Add boundary function contour
            failure_set_contour = ax.contour(X, Y, lx,
                                             levels=[0.0],
                                             colors="brown",
                                             linewidths=2,
                                             linestyles='-')
            
            # Add optional delta level contour
            if delta_level is not None:
                delta_contour = ax.contour(X, Y, BRT_img,
                                           levels=[delta_level],
                                           colors="black",
                                           linewidths=1,
                                           linestyles='-')
            
            # Set axis labels
            ax.set_xlabel(state_labels[x_axis_idx])
            if i == 0:
                ax.set_ylabel(state_labels[y_axis_idx])
        
        # Add figure title with fixed state information
        state_slices = plot_config['state_slices']
        fixed_str = f'Fixed: px={state_slices[0]:.2f}, py={state_slices[1]:.2f}, vx={state_slices[2]:.2f}, vy={state_slices[3]:.2f}'
        fig.suptitle(f'Value Function - Rotation Slice\n{fixed_str}', fontsize=14)
        
        plt.tight_layout()
        return fig

    def validate_mpc_ground_truth(self, epoch, x_resolution, y_resolution, z_resolution):
        if not self.dataset.use_MPC:
            return  # No MPC in this experiment
        
        # Get plot configuration from dynamics
        plot_config = self.dataset.dynamics.plot_config()
        state_test_range = self.dataset.dynamics.state_test_range()
        
        # Compute current curriculum time T
        if self.dataset.is_paused:
            T = self.dataset.paused_horizon
        else:
            # Not paused - use current curriculum time
            T = self.dataset.tMax * min(self.dataset.counter / self.dataset.counter_end, 1.0)
        
        if T <= 0:
            return  # No meaningful time horizon yet
        
        # Create MPC instance for ground truth computation
        device = next(self.model.parameters()).device
        mpc = MPC.MPC(
            horizon=None,
            receding_horizon=self.dataset.MPC_receding_horizon,
            dT=0.5,
            num_samples= 100,
            dynamics_=self.dataset.dynamics,
            device=device,
            mode=self.dataset.MPC_mode,
            sample_mode=self.dataset.MPC_sample_mode,
            lambda_=self.dataset.MPC_lambda_,
            style=self.dataset.MPC_style,
            num_iterative_refinement=self.dataset.num_iterative_refinement,
            cost_type=self.dataset.cost_type,
            mpc_percentage=self.dataset.mpc_percentage
        )
        
        # Get state slices and axis indices
        state_slices = list(plot_config['state_slices'])
        x_axis_idx = plot_config['x_axis_idx']
        y_axis_idx = plot_config['y_axis_idx']
        z_axis_idx = plot_config['z_axis_idx']

        # Get state ranges
        x_min, x_max = state_test_range[x_axis_idx]
        y_min, y_max = state_test_range[y_axis_idx]

        # Docking13D: vary vx (state index 3) instead of z-position, which is not a meaningful slice
        if self.dataset.dynamics.name == 'Docking13D':
            z_axis_idx = 3  # vx
            z_min, z_max = -1.5, 1.5
            state_slices[2] = 0.0  # fix z-position to 0
            z_label, z_units = 'vx', 'm/s'
        else:
            z_min, z_max = state_test_range[z_axis_idx]
            z_label, z_units = 'θ', 'rad'

        # Create slice values
        z_values = np.linspace(z_min, z_max, z_resolution)
        
        # Create figure: 1 row x z_resolution columns
        fig, axes = plt.subplots(1, z_resolution, figsize=(4 * z_resolution, 4))
        if z_resolution == 1:
            axes = [axes]
        
        # Create (x, y) grid
        xs = torch.linspace(x_min, x_max, x_resolution)
        ys = torch.linspace(y_min, y_max, y_resolution)
        xys = torch.cartesian_prod(xs, ys).to(device)
        
        # Coordinate arrays for contour plotting
        x_coords = np.linspace(x_min, x_max, x_resolution)
        y_coords = np.linspace(y_min, y_max, y_resolution)
        X, Y = np.meshgrid(x_coords, y_coords)
        
        # Global value range tracking
        all_costs = []
        
        print(f"Computing MPC ground truth at T={T:.2f}...")
        
        for idx, z_val in enumerate(z_values):
            # Create initial conditions with fixed z_val (theta)
            initial_conditions = torch.zeros(x_resolution * y_resolution, self.dataset.dynamics.state_dim).to(device)
            initial_conditions[:, :] = torch.tensor(state_slices).to(device)
            initial_conditions[:, x_axis_idx] = xys[:, 0]
            initial_conditions[:, y_axis_idx] = xys[:, 1]
            initial_conditions[:, z_axis_idx] = z_val
            
            # Compute MPC costs in batches to avoid memory issues
            costs_list = []
            batch_size = max(100, x_resolution * y_resolution // 4)
            num_batches = math.ceil(len(initial_conditions) / batch_size)
            
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, len(initial_conditions))
                batch_ics = initial_conditions[start_idx:end_idx]
                
                with torch.no_grad():
                    costs_batch, _, _, _ = mpc.get_batch_data(batch_ics, T)
                    costs_list.append(costs_batch.detach().cpu())
                
                # Clear GPU cache
                torch.cuda.empty_cache()
            
            costs = torch.cat(costs_list, dim=0)
            all_costs.append(costs)
        
        # Compute global color range
        all_costs_np = [c.numpy() for c in all_costs]
        valid_values = np.concatenate([c[~np.isnan(c)] for c in all_costs_np if np.any(~np.isnan(c))])
        if len(valid_values) > 0:
            vmin = np.percentile(valid_values, 2)  # Use percentiles to handle outliers
            vmax = np.percentile(valid_values, 98)
        else:
            vmin, vmax = -1, 1
        
        # Plot each theta slice
        for idx, (z_val, costs, ax) in enumerate(zip(z_values, all_costs, axes)):
            BRT_img = costs.numpy().reshape(x_resolution, y_resolution).T
            
            # Create grey background
            greys = np.full((*BRT_img.shape, 3), 180, dtype=np.uint8)
            ax.imshow(greys, extent=(x_min, x_max, y_min, y_max), origin='lower', aspect='equal')
            
            # Plot heatmap
            im = ax.imshow(BRT_img, cmap='RdYlBu', vmin=vmin, vmax=vmax,
                          extent=(x_min, x_max, y_min, y_max), origin='lower', aspect='equal')
            
            # Add 0-level contour (black) - the BRT boundary
            try:
                ax.contour(X, Y, BRT_img, levels=[0.0], colors='black', linewidths=1.5, linestyles='-')
            except:
                pass  # Contour may fail if all values are same sign
            
            # Add reach set contour (green)
            reach_ics = torch.zeros(x_resolution * y_resolution, self.dataset.dynamics.state_dim).to(device)
            reach_ics[:, :] = torch.tensor(state_slices).to(device)
            reach_ics[:, x_axis_idx] = xys[:, 0]
            reach_ics[:, y_axis_idx] = xys[:, 1]
            reach_ics[:, z_axis_idx] = z_val
            
            with torch.no_grad():
                reach_values = self.dataset.dynamics.reach_fn(reach_ics).detach().cpu().numpy()
            reach_img = reach_values.reshape(x_resolution, y_resolution).T
            
            try:
                ax.contour(X, Y, reach_img, levels=[0.0], colors='green', linewidths=1.5, linestyles='-')
            except:
                pass
            
            # Add avoid set contour (red) if available
            if hasattr(self.dataset.dynamics, 'avoid_fn'):
                try:
                    with torch.no_grad():
                        avoid_values = self.dataset.dynamics.avoid_fn(reach_ics).detach().cpu().numpy()
                    avoid_img = avoid_values.reshape(x_resolution, y_resolution).T
                    ax.contour(X, Y, avoid_img, levels=[0.0], colors='red', linewidths=1.5, linestyles='--')
                except:
                    pass
            
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_xlabel('px (m)')
            if idx == 0:
                ax.set_ylabel('py (m)')
            ax.set_title(f'{z_label}={z_val:.2f} {z_units}')
            
            # Add per-subplot colorbar
            fig.colorbar(im, ax=ax, shrink=0.8)
        
        # Add suptitle with time and fixed states
        if self.dataset.dynamics.name == 'Docking13D':
            fixed_str = f'Fixed: z={state_slices[2]:.2f}m, vy={state_slices[4]:.2f}, vz={state_slices[5]:.2f}, q=q_goal'
        else:
            fixed_str = f'Fixed: vx={state_slices[2]:.2f}, vy={state_slices[3]:.2f}, ω={state_slices[5]:.2f}'
        fig.suptitle(f'MPC Ground Truth (T={T:.2f}s, Epoch {epoch})\n{fixed_str}', fontsize=14)
        
        plt.tight_layout()
        
        if self.use_wandb:
            wandb.log({
                'step': epoch,
                'mpc_ground_truth': wandb.Image(fig),
            })
        
        plt.close(fig)
        
        # Clean up MPC instance
        del mpc
        gc.collect()
        torch.cuda.empty_cache()

    def plot_recovery_fig(self, dataset, dynamics, model, delta_level):
        # 1. for ground truth slices (if available), record (higher-res) grid of learned values
        # plot (with ground truth) learned BRTs, recovered BRTs
        z_res = 5
        plot_config = dataset.dynamics.plot_config()
        if os.path.exists(os.path.join(self.experiment_dir, 'ground_truth.mat')):
            ground_truth = spio.loadmat(os.path.join(
                self.experiment_dir, 'ground_truth.mat'))
            if 'gmat' in ground_truth:
                ground_truth_xs = ground_truth['gmat'][..., 0][:, 0, 0]
                ground_truth_ys = ground_truth['gmat'][..., 1][0, :, 0]
                ground_truth_zs = ground_truth['gmat'][..., 2][0, 0, :]
                ground_truth_values = ground_truth['data']
                ground_truth_ts = np.linspace(
                    0, 1, ground_truth_values.shape[3])

            elif 'g' in ground_truth:
                ground_truth_xs = ground_truth['g']['vs'][0, 0][0][0][:, 0]
                ground_truth_ys = ground_truth['g']['vs'][0, 0][1][0][:, 0]
                ground_truth_zs = ground_truth['g']['vs'][0, 0][2][0][:, 0]
                ground_truth_ts = ground_truth['tau'][0]
                ground_truth_values = ground_truth['data']

            # idxs to plot
            x_idxs = np.linspace(0, len(ground_truth_xs)-1,
                                 len(ground_truth_xs)).astype(dtype=int)
            y_idxs = np.linspace(0, len(ground_truth_ys)-1,
                                 len(ground_truth_ys)).astype(dtype=int)
            z_idxs = np.linspace(0, len(ground_truth_zs) -
                                 1, z_res).astype(dtype=int)
            t_idxs = np.array([len(ground_truth_ts)-1]).astype(dtype=int)

            # indexed ground truth to plot
            ground_truth_xs = ground_truth_xs[x_idxs]
            ground_truth_ys = ground_truth_ys[y_idxs]
            ground_truth_zs = ground_truth_zs[z_idxs]
            ground_truth_ts = ground_truth_ts[t_idxs]
            ground_truth_values = ground_truth_values[
                x_idxs[:, None, None, None],
                y_idxs[None, :, None, None],
                z_idxs[None, None, :, None],
                t_idxs[None, None, None, :]
            ]
            ground_truth_grids = ground_truth_values

            xs = ground_truth_xs
            ys = ground_truth_ys
            zs = ground_truth_zs
        else:
            ground_truth_grids = None
            resolution = 512
            xs = np.linspace(*dynamics.state_test_range()
                             [plot_config['x_axis_idx']], resolution)
            ys = np.linspace(*dynamics.state_test_range()
                             [plot_config['y_axis_idx']], resolution)
            zs = np.linspace(*dynamics.state_test_range()
                             [plot_config['z_axis_idx']], z_res)

        xys = torch.cartesian_prod(torch.tensor(xs), torch.tensor(ys))
        value_grids = np.zeros((len(zs), len(xs), len(ys)))
        for i in range(len(zs)):
            coords = torch.zeros(xys.shape[0], dataset.dynamics.state_dim + 1)
            coords[:, 0] = dataset.tMax
            coords[:, 1:] = torch.tensor(plot_config['state_slices'])
            coords[:, 1 + plot_config['x_axis_idx']] = xys[:, 0]
            coords[:, 1 + plot_config['y_axis_idx']] = xys[:, 1]
            if dataset.dynamics.state_dim > 2:
                coords[:, 1 + plot_config['z_axis_idx']] = zs[i]

            model_results = model(
                {'coords': dataset.dynamics.coord_to_input(coords.cuda())})
            values = dataset.dynamics.io_to_value(model_results['model_in'].detach(
            ), model_results['model_out'].detach().squeeze(dim=-1)).detach().cpu()
            value_grids[i] = values.reshape(len(xs), len(ys))

        fig = plt.figure()
        fig.suptitle(plot_config['state_slices'], fontsize=8)
        x_min, x_max = dataset.dynamics.state_test_range()[
            plot_config['x_axis_idx']]
        y_min, y_max = dataset.dynamics.state_test_range()[
            plot_config['y_axis_idx']]

        for i in range(len(zs)):
            values = value_grids[i]

            # learned BRT and recovered BRT
            ax = fig.add_subplot(1, len(zs), (i+1))
            ax.set_title('%s = %0.2f' % (
                plot_config['state_labels'][plot_config['z_axis_idx']], zs[i]), fontsize=8)

            image = np.full((*values.shape, 3), 255, dtype=int)
            BRT = values < 0
            recovered_BRT = values < delta_level

            if dynamics.set_mode in ['reach', 'reach_avoid']:
                image[BRT] = np.array([252, 227, 152])
                self.overlay_border(image, BRT, np.array([249, 188, 6]))
                image[recovered_BRT] = np.array([155, 241, 249])
                self.overlay_border(image, recovered_BRT,
                                    np.array([15, 223, 240]))
                if ground_truth_grids is not None:
                    self.overlay_ground_truth(image, i, ground_truth_grids)
            else:
                image[recovered_BRT] = np.array([155, 241, 249])
                image[BRT] = np.array([252, 227, 152])
                self.overlay_border(image, BRT, np.array([249, 188, 6]))
                # overlay recovered border over learned BRT
                self.overlay_border(image, recovered_BRT,
                                    np.array([15, 223, 240]))
                if ground_truth_grids is not None:
                    self.overlay_ground_truth(image, i, ground_truth_grids)

            ax.imshow(image.transpose(1, 0, 2), origin='lower',
                      extent=(x_min, x_max, y_min, y_max))

            ax.set_xlabel(plot_config['state_labels']
                          [plot_config['x_axis_idx']])
            ax.set_ylabel(plot_config['state_labels']
                          [plot_config['y_axis_idx']])
            ax.set_xticks([x_min, x_max])
            ax.set_yticks([y_min, y_max])
            ax.tick_params(labelsize=6)
            if i != 0:
                ax.set_yticks([])
        return fig, value_grids

    def overlay_ground_truth(self, image, z_idx, ground_truth_grids):
        thickness = max(0, image.shape[0] // 120 - 1)
        ground_truth_grid = ground_truth_grids[:, :, z_idx, 0]
        ground_truth_brts = ground_truth_grid < 0
        for x in range(ground_truth_brts.shape[0]):
            for y in range(ground_truth_brts.shape[1]):
                if not ground_truth_brts[x, y]:
                    continue
                neighbors = [
                    (x, y+1),
                    (x, y-1),
                    (x+1, y+1),
                    (x+1, y),
                    (x+1, y-1),
                    (x-1, y+1),
                    (x-1, y),
                    (x-1, y-1),
                ]
                for neighbor in neighbors:
                    if neighbor[0] >= 0 and neighbor[1] >= 0 and neighbor[0] < ground_truth_brts.shape[0] and neighbor[1] < ground_truth_brts.shape[1]:
                        if not ground_truth_brts[neighbor]:
                            image[x-thickness:x+1, y-thickness:y +
                                  1+thickness] = np.array([50, 50, 50])
                            break

    def overlay_border(self, image, set, color):
        thickness = max(0, image.shape[0] // 120 - 1)
        for x in range(set.shape[0]):
            for y in range(set.shape[1]):
                if not set[x, y]:
                    continue
                neighbors = [
                    (x, y+1),
                    (x, y-1),
                    (x+1, y+1),
                    (x+1, y),
                    (x+1, y-1),
                    (x-1, y+1),
                    (x-1, y),
                    (x-1, y-1),
                ]
                for neighbor in neighbors:
                    if neighbor[0] >= 0 and neighbor[1] >= 0 and neighbor[0] < set.shape[0] and neighbor[1] < set.shape[1]:
                        if not set[neighbor]:
                            image[x-thickness:x+1, y -
                                  thickness:y+1+thickness] = color
                            break


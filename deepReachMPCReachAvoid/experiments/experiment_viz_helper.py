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
                      quat_slice=None, theta_yaw=None):
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
                         delta_level=None, quat_slice=None, theta_yaw=None, z_values=None):
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
        
        This matches the dimension conventions from validate_mpc_dataset_velocity:
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
        
        This matches the dimension conventions from validate_mpc_dataset_rotation:
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

    def validate_mpc_dataset_position(self, epoch, x_resolution, y_resolution, z_resolution, time_resolution):
        """
        Visualize MPC training dataset values at the same time/theta slices as DeepReach validation.
        Creates two plots logged to wandb:
        1. Scatter only: just the MPC sample points
        2. Histogram: V-value distribution for each (time, theta) slice
        """
        if not self.dataset.use_MPC:
            return  # No MPC data to visualize
        
        if not hasattr(self.dataset, 'MPC_inputs') or self.dataset.MPC_inputs is None:
            return  # No MPC data available
        
        if len(self.dataset.MPC_inputs) < 10:
            return  # Not enough MPC data
        
        plot_config = self.dataset.dynamics.plot_config()
        state_test_range = self.dataset.dynamics.state_test_range()
        
        # Get axis indices from plot_config
        x_axis_idx = plot_config['x_axis_idx']  # 0 (px)
        y_axis_idx = plot_config['y_axis_idx']  # 1 (py)
        z_axis_idx = plot_config['z_axis_idx']  # 4 (theta)
        
        if z_axis_idx == -1:
            return  # Single fig case not implemented for MPC viz
        
        # Define grid ranges (same as plotMultipleFigs)
        x_min, x_max = state_test_range[x_axis_idx]
        y_min, y_max = state_test_range[y_axis_idx]
        z_min, z_max = state_test_range[z_axis_idx]
        
        # Define time and theta slices (same as validate)
        times = torch.linspace(0, self.dataset.tMax, time_resolution)
        zs = torch.linspace(z_min, z_max, z_resolution)  # theta slices
        
        # Convert MPC data from normalized to real coordinates
        mpc_inputs_real = self.dataset.dynamics.input_to_coord(
            self.dataset.MPC_inputs.clone()
        )  # Shape: [N, state_dim+1], columns: [time, px, py, vx, vy, theta, omega]
        mpc_values = self.dataset.MPC_values.clone()
        
        # Extract relevant columns
        mpc_times = mpc_inputs_real[:, 0].numpy()
        mpc_x = mpc_inputs_real[:, 1 + x_axis_idx].numpy()  # px
        mpc_y = mpc_inputs_real[:, 1 + y_axis_idx].numpy()  # py
        mpc_z = mpc_inputs_real[:, 1 + z_axis_idx].numpy()  # theta
        mpc_vals = mpc_values.numpy()
        
        # Get fixed state slices for non-plotted dimensions
        state_slices = plot_config['state_slices']
        
        # Define tolerances based on grid spacing
        if len(times) > 1:
            time_tol = (times[1] - times[0]).item() / 10.0
        else:
            time_tol = 0.01
                
        if len(zs) > 1:
            z_tol = (zs[1] - zs[0]).item() / 10.0
        else:
            z_tol = 0.1
        
        # Tolerance for other fixed dimensions (vx, vy, omega)
        other_dim_tols = {}
        for dim in range(self.dataset.dynamics.state_dim):
            if dim not in [x_axis_idx, y_axis_idx, z_axis_idx]:
                dim_range = state_test_range[dim][1] - state_test_range[dim][0]
                other_dim_tols[dim] = dim_range * 0.3
        
        # Grid for boundary function computation
        xs_grid = np.linspace(x_min, x_max, x_resolution)
        ys_grid = np.linspace(y_min, y_max, y_resolution)
        X, Y = np.meshgrid(xs_grid, ys_grid)
        
        # Create two figures
        fig_scatter = plt.figure(figsize=(6 * len(zs), 5 * len(times)))
        fig_hist = plt.figure(figsize=(6 * len(zs), 5 * len(times)))
        
        for i, t in enumerate(times):
            t_val = t.item()
            for j, z in enumerate(zs):
                z_val = z.item()
                
                # Filter MPC samples for this (time, theta) slice
                time_mask = np.abs(mpc_times - t_val) < time_tol
                z_mask = np.abs(mpc_z - z_val) < z_tol
                
                # Also filter by other fixed dimensions (vx, vy, omega)
                other_masks = np.ones(len(mpc_times), dtype=bool)
                for dim, tol in other_dim_tols.items():
                    dim_vals = mpc_inputs_real[:, 1 + dim].numpy()
                    slice_val = state_slices[dim]
                    other_masks &= np.abs(dim_vals - slice_val) < tol
                
                combined_mask = time_mask & z_mask & other_masks
                
                # Extract filtered data
                slice_x = mpc_x[combined_mask]
                slice_y = mpc_y[combined_mask]
                slice_vals = mpc_vals[combined_mask]
                
                num_samples = len(slice_x)
                subplot_idx = i * len(zs) + j + 1
                title_str = f't = {t_val:.2f}, {plot_config["state_labels"][z_axis_idx]} = {z_val:.2f}\n(n={num_samples})'
                
                # Create subplots for both figures
                ax_scatter = fig_scatter.add_subplot(len(times), len(zs), subplot_idx)
                ax_hist = fig_hist.add_subplot(len(times), len(zs), subplot_idx)
                
                ax_scatter.set_title(title_str, fontsize=10)
                ax_hist.set_title(title_str, fontsize=10)
                
                # Compute boundary function for all plots
                xys = torch.cartesian_prod(torch.tensor(xs_grid, dtype=torch.float32), 
                                           torch.tensor(ys_grid, dtype=torch.float32))
                coords = torch.zeros(x_resolution * y_resolution, self.dataset.dynamics.state_dim + 1)
                coords[:, 0] = t_val
                coords[:, 1:] = torch.tensor(state_slices, dtype=torch.float32)
                coords[:, 1 + x_axis_idx] = xys[:, 0]
                coords[:, 1 + y_axis_idx] = xys[:, 1]
                coords[:, 1 + z_axis_idx] = z_val
                lx = self.dataset.dynamics.boundary_fn(coords[..., 1:]).numpy().reshape(x_resolution, y_resolution).T
                
                if num_samples < 3:
                    # Not enough data - show gray background
                    ax_scatter.set_facecolor('lightgray')
                    ax_scatter.text(0.5, 0.5, 'Insufficient\nMPC data', 
                           transform=ax_scatter.transAxes, ha='center', va='center',
                           fontsize=12, color='darkgray')
                    if num_samples > 0:
                        scatter = ax_scatter.scatter(slice_x, slice_y, c=slice_vals, 
                                           cmap='coolwarm_r', s=30, edgecolors='black', 
                                           linewidth=0.5, alpha=0.9)
                        fig_scatter.colorbar(scatter, ax=ax_scatter)
                    ax_scatter.contour(X, Y, lx, levels=[0.0], colors='saddlebrown', 
                              linewidths=1.5, linestyles='-')
                else:
                    # Compute local min/max from scatter data
                    local_vmin = np.nanmin(slice_vals)
                    local_vmax = np.nanmax(slice_vals)
                    
                    # === SCATTER PLOT ===
                    scatter_sc = ax_scatter.scatter(slice_x, slice_y, c=slice_vals, 
                                       cmap='coolwarm_r', s=30, edgecolors='black',
                                       linewidth=0.5, alpha=0.9,
                                       vmin=local_vmin, vmax=local_vmax)
                    fig_scatter.colorbar(scatter_sc, ax=ax_scatter)
                    ax_scatter.contour(X, Y, lx, levels=[0.0], colors='saddlebrown', 
                                      linewidths=1.5, linestyles='-')
                
                # === HISTOGRAM PLOT ===
                if num_samples >= 1:
                    # Create histogram of V values
                    ax_hist.hist(slice_vals, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
                    
                    # Add vertical line at V=0
                    ax_hist.axvline(x=0, color='red', linestyle='--', linewidth=2, label='V=0')
                    
                    # Calculate percentages
                    pct_negative = np.sum(slice_vals < 0) / len(slice_vals) * 100
                    pct_positive = np.sum(slice_vals >= 0) / len(slice_vals) * 100
                    
                    # Add text annotation with percentages
                    ax_hist.text(0.02, 0.98, f'V<0: {pct_negative:.1f}%\nV≥0: {pct_positive:.1f}%',
                                transform=ax_hist.transAxes, fontsize=9,
                                verticalalignment='top', horizontalalignment='left',
                                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                    
                    ax_hist.set_xlabel('V value')
                    ax_hist.set_ylabel('Count')
                    ax_hist.legend(loc='upper right', fontsize=8)
                else:
                    ax_hist.set_facecolor('lightgray')
                    ax_hist.text(0.5, 0.5, 'No data', 
                                transform=ax_hist.transAxes, ha='center', va='center',
                                fontsize=12, color='darkgray')
                
                # Set limits and labels for scatter axis
                ax_scatter.set_xlim(x_min, x_max)
                ax_scatter.set_ylim(y_min, y_max)
                if i == len(times) - 1:
                    ax_scatter.set_xlabel(plot_config['state_labels'][x_axis_idx])
                if j == 0:
                    ax_scatter.set_ylabel(plot_config['state_labels'][y_axis_idx])
        
        # Add titles to figures
        fixed_str = f'Fixed: vx={state_slices[2]:.2f}, vy={state_slices[3]:.2f}, ω={state_slices[5]:.2f}'
        fig_scatter.suptitle(f'MPC Dataset Position - Scatter (Epoch {epoch})\n{fixed_str}', fontsize=14)
        fig_hist.suptitle(f'MPC Dataset Position - V Distribution (Epoch {epoch})\n{fixed_str}', fontsize=14)
        
        plt.tight_layout()
        
        if self.use_wandb:
            wandb.log({
                'step': epoch,
                'mpc_dataset_position_scatter': wandb.Image(fig_scatter),
                'mpc_dataset_position_histogram': wandb.Image(fig_hist),
            })
        
        plt.close(fig_scatter)
        plt.close(fig_hist)

    def validate_mpc_dataset_position_base_quat_scatter(self, epoch, x_resolution, y_resolution, z_resolution,
                                                        time_resolution, angle_tol=0.2):
        """Scatter plot of MPC samples for base quaternion (Docking13D only)."""
        if self.dataset.dynamics.name != 'Docking13D':
            return
        if not self.dataset.use_MPC:
            return
        if not hasattr(self.dataset, 'MPC_inputs') or self.dataset.MPC_inputs is None:
            return
        if len(self.dataset.MPC_inputs) < 10:
            return

        plot_config = self.dataset.dynamics.plot_config()
        state_test_range = self.dataset.dynamics.state_test_range()

        x_axis_idx = plot_config['x_axis_idx']
        y_axis_idx = plot_config['y_axis_idx']
        z_axis_idx = plot_config['z_axis_idx']
        if z_axis_idx == -1:
            return

        x_min, x_max = state_test_range[x_axis_idx]
        y_min, y_max = state_test_range[y_axis_idx]
        z_min, z_max = state_test_range[z_axis_idx]

        times = torch.linspace(0, self.dataset.tMax, time_resolution)
        zs = torch.linspace(z_min, z_max, z_resolution)

        mpc_inputs_real = self.dataset.dynamics.input_to_coord(
            self.dataset.MPC_inputs.clone()
        )
        mpc_values = self.dataset.MPC_values.clone()

        mpc_inputs_real_np = mpc_inputs_real.detach().cpu().numpy()
        mpc_values_np = mpc_values.detach().cpu().numpy()

        mpc_times = mpc_inputs_real_np[:, 0]
        mpc_x = mpc_inputs_real_np[:, 1 + x_axis_idx]
        mpc_y = mpc_inputs_real_np[:, 1 + y_axis_idx]
        mpc_z = mpc_inputs_real_np[:, 1 + z_axis_idx]

        q = mpc_inputs_real_np[:, 1 + 6:1 + 10]
        q0 = np.clip(np.abs(q[:, 0]), 0.0, 1.0)
        quat_angle = 2.0 * np.arccos(q0)
        quat_mask = quat_angle < angle_tol

        state_slices = plot_config['state_slices']

        if len(times) > 1:
            time_tol = (times[1] - times[0]).item() * 0.5
        else:
            time_tol = 0.05

        if len(zs) > 1:
            z_tol = (zs[1] - zs[0]).item() * 0.5
        else:
            z_tol = 0.25

        xs_grid = np.linspace(x_min, x_max, x_resolution)
        ys_grid = np.linspace(y_min, y_max, y_resolution)
        X, Y = np.meshgrid(xs_grid, ys_grid)

        fig_scatter = plt.figure(figsize=(6 * len(zs), 5 * len(times)))

        for i, t in enumerate(times):
            t_val = t.item()
            for j, z in enumerate(zs):
                z_val = z.item()

                time_mask = np.abs(mpc_times - t_val) < time_tol
                z_mask = np.abs(mpc_z - z_val) < z_tol

                combined_mask = time_mask & z_mask & quat_mask

                slice_x = mpc_x[combined_mask]
                slice_y = mpc_y[combined_mask]
                slice_vals = mpc_values_np[combined_mask]

                num_samples = len(slice_x)
                subplot_idx = i * len(zs) + j + 1
                title_str = f't = {t_val:.2f}, {plot_config["state_labels"][z_axis_idx]} = {z_val:.2f}\n(n={num_samples})'

                ax_scatter = fig_scatter.add_subplot(len(times), len(zs), subplot_idx)
                ax_scatter.set_title(title_str, fontsize=10)

                xys = torch.cartesian_prod(torch.tensor(xs_grid, dtype=torch.float32),
                                           torch.tensor(ys_grid, dtype=torch.float32))
                coords = torch.zeros(x_resolution * y_resolution, self.dataset.dynamics.state_dim + 1)
                coords[:, 0] = t_val
                coords[:, 1:] = torch.tensor(state_slices, dtype=torch.float32)
                coords[:, 1 + x_axis_idx] = xys[:, 0]
                coords[:, 1 + y_axis_idx] = xys[:, 1]
                coords[:, 1 + z_axis_idx] = z_val
                lx = self.dataset.dynamics.boundary_fn(coords[..., 1:]).numpy().reshape(x_resolution, y_resolution).T

                if num_samples < 3:
                    ax_scatter.set_facecolor('lightgray')
                    ax_scatter.text(0.5, 0.5, 'Insufficient\nMPC data',
                                    transform=ax_scatter.transAxes, ha='center', va='center',
                                    fontsize=12, color='darkgray')
                    if num_samples > 0:
                        scatter = ax_scatter.scatter(slice_x, slice_y, c=slice_vals,
                                                     cmap='coolwarm_r', s=30, edgecolors='black',
                                                     linewidth=0.5, alpha=0.9)
                        fig_scatter.colorbar(scatter, ax=ax_scatter)
                else:
                    local_vmin = np.nanmin(slice_vals)
                    local_vmax = np.nanmax(slice_vals)
                    scatter_sc = ax_scatter.scatter(slice_x, slice_y, c=slice_vals,
                                                    cmap='coolwarm_r', s=30, edgecolors='black',
                                                    linewidth=0.5, alpha=0.9,
                                                    vmin=local_vmin, vmax=local_vmax)
                    fig_scatter.colorbar(scatter_sc, ax=ax_scatter)

                ax_scatter.contour(X, Y, lx, levels=[0.0], colors='saddlebrown',
                                   linewidths=1.5, linestyles='-')

                ax_scatter.set_xlim(x_min, x_max)
                ax_scatter.set_ylim(y_min, y_max)
                if i == len(times) - 1:
                    ax_scatter.set_xlabel(plot_config['state_labels'][x_axis_idx])
                if j == 0:
                    ax_scatter.set_ylabel(plot_config['state_labels'][y_axis_idx])

        fixed_str = f'Base quaternion, fixed: vx={state_slices[3]:.2f}, vy={state_slices[4]:.2f}, vz={state_slices[5]:.2f}'
        fig_scatter.suptitle(f'MPC Dataset Position - Scatter (Epoch {epoch})\n{fixed_str}', fontsize=14)

        plt.tight_layout()

        if self.use_wandb:
            wandb.log({
                'step': epoch,
                'mpc_dataset_position_scatter_base_quat': wandb.Image(fig_scatter),
            })

        plt.close(fig_scatter)

    def validate_mpc_dataset_position_base_quat_hist(self, epoch, x_resolution, y_resolution, z_resolution, time_resolution,
                                                     angle_tol=0.2):
        """Sampling histogram for base quaternion (Docking13D only)."""
        if self.dataset.dynamics.name != 'Docking13D':
            return
        if not self.dataset.use_MPC:
            return
        if not hasattr(self.dataset, 'MPC_inputs') or self.dataset.MPC_inputs is None:
            return
        if len(self.dataset.MPC_inputs) < 10:
            return

        plot_config = self.dataset.dynamics.plot_config()
        state_test_range = self.dataset.dynamics.state_test_range()

        x_axis_idx = plot_config['x_axis_idx']
        y_axis_idx = plot_config['y_axis_idx']
        z_axis_idx = plot_config['z_axis_idx']
        if z_axis_idx == -1:
            return

        x_min, x_max = state_test_range[x_axis_idx]
        y_min, y_max = state_test_range[y_axis_idx]
        z_min, z_max = state_test_range[z_axis_idx]

        times = torch.linspace(0, self.dataset.tMax, time_resolution)
        zs = torch.linspace(z_min, z_max, z_resolution)

        mpc_inputs_real = self.dataset.dynamics.input_to_coord(
            self.dataset.MPC_inputs.clone()
        )

        mpc_times = mpc_inputs_real[:, 0].numpy()
        mpc_x = mpc_inputs_real[:, 1 + x_axis_idx].numpy()
        mpc_y = mpc_inputs_real[:, 1 + y_axis_idx].numpy()
        mpc_z = mpc_inputs_real[:, 1 + z_axis_idx].numpy()

        q = mpc_inputs_real[:, 1 + 6:1 + 10].numpy()
        q0 = np.clip(np.abs(q[:, 0]), 0.0, 1.0)
        quat_angle = 2.0 * np.arccos(q0)
        quat_mask = quat_angle < angle_tol

        state_slices = plot_config['state_slices']

        if len(times) > 1:
            time_tol = (times[1] - times[0]).item() / 10.0
        else:
            time_tol = 0.01

        if len(zs) > 1:
            z_tol = (zs[1] - zs[0]).item() / 10.0
        else:
            z_tol = 0.1

        other_dim_tols = {}
        for dim in range(self.dataset.dynamics.state_dim):
            if dim in [x_axis_idx, y_axis_idx, z_axis_idx, 6, 7, 8, 9]:
                continue
            dim_range = state_test_range[dim][1] - state_test_range[dim][0]
            other_dim_tols[dim] = dim_range * 0.3

        xs_grid = np.linspace(x_min, x_max, x_resolution)
        ys_grid = np.linspace(y_min, y_max, y_resolution)
        X, Y = np.meshgrid(xs_grid, ys_grid)

        fig_hist2d = plt.figure(figsize=(6 * len(zs), 5 * len(times)))

        for i, t in enumerate(times):
            t_val = t.item()
            for j, z in enumerate(zs):
                z_val = z.item()

                time_mask = np.abs(mpc_times - t_val) < time_tol
                z_mask = np.abs(mpc_z - z_val) < z_tol

                other_masks = np.ones(len(mpc_times), dtype=bool)
                for dim, tol in other_dim_tols.items():
                    dim_vals = mpc_inputs_real[:, 1 + dim].numpy()
                    slice_val = state_slices[dim]
                    other_masks &= np.abs(dim_vals - slice_val) < tol

                combined_mask = time_mask & z_mask & other_masks & quat_mask

                slice_x = mpc_x[combined_mask]
                slice_y = mpc_y[combined_mask]
                num_samples = len(slice_x)
                subplot_idx = i * len(zs) + j + 1

                ax = fig_hist2d.add_subplot(len(times), len(zs), subplot_idx)
                ax.set_title(f't = {t_val:.2f}, {plot_config["state_labels"][z_axis_idx]} = {z_val:.2f}\n(n={num_samples})', fontsize=10)

                if num_samples < 3:
                    ax.set_facecolor('lightgray')
                    ax.text(0.5, 0.5, 'Insufficient\nMPC data',
                            transform=ax.transAxes, ha='center', va='center',
                            fontsize=12, color='darkgray')
                else:
                    hist2d, x_edges, y_edges = np.histogram2d(
                        slice_x, slice_y,
                        bins=[x_resolution, y_resolution],
                        range=[[x_min, x_max], [y_min, y_max]]
                    )
                    im = ax.imshow(
                        hist2d.T,
                        extent=(x_min, x_max, y_min, y_max),
                        origin='lower',
                        cmap='viridis',
                        aspect='equal'
                    )
                    fig_hist2d.colorbar(im, ax=ax)

                xys = torch.cartesian_prod(torch.tensor(xs_grid, dtype=torch.float32),
                                           torch.tensor(ys_grid, dtype=torch.float32))
                coords = torch.zeros(x_resolution * y_resolution, self.dataset.dynamics.state_dim + 1)
                coords[:, 0] = t_val
                coords[:, 1:] = torch.tensor(state_slices, dtype=torch.float32)
                coords[:, 1 + x_axis_idx] = xys[:, 0]
                coords[:, 1 + y_axis_idx] = xys[:, 1]
                coords[:, 1 + z_axis_idx] = z_val
                lx = self.dataset.dynamics.boundary_fn(coords[..., 1:]).numpy().reshape(x_resolution, y_resolution).T
                ax.contour(X, Y, lx, levels=[0.0], colors='saddlebrown', linewidths=1.5, linestyles='-')

                ax.set_xlim(x_min, x_max)
                ax.set_ylim(y_min, y_max)
                if i == len(times) - 1:
                    ax.set_xlabel(plot_config['state_labels'][x_axis_idx])
                if j == 0:
                    ax.set_ylabel(plot_config['state_labels'][y_axis_idx])

        fixed_str = f'Fixed: vx={state_slices[3]:.2f}, vy={state_slices[4]:.2f}, vz={state_slices[5]:.2f}'
        fig_hist2d.suptitle(
            f'MPC Dataset Position - Sampling Histogram (Base Quaternion) (Epoch {epoch})\n{fixed_str}',
            fontsize=14
        )
        plt.tight_layout()

        if self.use_wandb:
            wandb.log({
                'step': epoch,
                'mpc_dataset_position_hist_base_quat': wandb.Image(fig_hist2d),
            })

        plt.close(fig_hist2d)

    def validate_mpc_dataset_velocity(self, epoch, x_resolution, y_resolution, z_resolution, time_resolution):
        """
        Visualize MPC training dataset values in velocity plane (vx, vy) with theta slicing.
        Creates two plots logged to wandb:
        1. Scatter only: just the MPC sample points
        2. Histogram: V-value distribution for each (time, theta) slice
        """
        if not self.dataset.use_MPC:
            return  # No MPC data to visualize
        
        if not hasattr(self.dataset, 'MPC_inputs') or self.dataset.MPC_inputs is None:
            return  # No MPC data available
        
        if len(self.dataset.MPC_inputs) < 10:
            return  # Not enough MPC data
        
        plot_config = self.dataset.dynamics.plot_config()
        state_test_range = self.dataset.dynamics.state_test_range()
        
        # Velocity plane axes
        x_axis_idx = 2  # vx
        y_axis_idx = 3  # vy
        z_axis_idx = 4  # theta (slicing dimension)
        
        # Define grid ranges
        x_min, x_max = state_test_range[x_axis_idx]  # vx range
        y_min, y_max = state_test_range[y_axis_idx]  # vy range
        z_min, z_max = state_test_range[z_axis_idx]  # theta range
        
        # Define time and theta slices (same as position)
        times = torch.linspace(0, self.dataset.tMax, time_resolution)
        zs = torch.linspace(z_min, z_max, z_resolution)  # theta slices
        
        # Convert MPC data from normalized to real coordinates
        mpc_inputs_real = self.dataset.dynamics.input_to_coord(
            self.dataset.MPC_inputs.clone()
        )  # Shape: [N, state_dim+1], columns: [time, px, py, vx, vy, theta, omega]
        mpc_values = self.dataset.MPC_values.clone()
        
        # Extract relevant columns
        mpc_times = mpc_inputs_real[:, 0].numpy()
        mpc_x = mpc_inputs_real[:, 1 + x_axis_idx].numpy()  # vx
        mpc_y = mpc_inputs_real[:, 1 + y_axis_idx].numpy()  # vy
        mpc_z = mpc_inputs_real[:, 1 + z_axis_idx].numpy()  # theta
        mpc_vals = mpc_values.numpy()
        
        # Get fixed state slices for non-plotted dimensions
        state_slices = plot_config['state_slices']
        
        # Define tolerances based on grid spacing
        if len(times) > 1:
            time_tol = (times[1] - times[0]).item() / 10.0
        else:
            time_tol = 0.01
                
        if len(zs) > 1:
            z_tol = (zs[1] - zs[0]).item() / 10.0
        else:
            z_tol = 0.1
        
        # Tolerance for other fixed dimensions (px, py, omega)
        other_dim_tols = {}
        for dim in range(self.dataset.dynamics.state_dim):
            if dim not in [x_axis_idx, y_axis_idx, z_axis_idx]:
                dim_range = state_test_range[dim][1] - state_test_range[dim][0]
                other_dim_tols[dim] = dim_range * 0.3
        
        # Create two figures
        fig_scatter = plt.figure(figsize=(6 * len(zs), 5 * len(times)))
        fig_hist = plt.figure(figsize=(6 * len(zs), 5 * len(times)))
        
        state_labels = ['px', 'py', 'vx', 'vy', r'$\theta$', r'$\omega$']
        
        for i, t in enumerate(times):
            t_val = t.item()
            for j, z in enumerate(zs):
                z_val = z.item()
                
                # Filter MPC samples for this (time, theta) slice
                time_mask = np.abs(mpc_times - t_val) < time_tol
                z_mask = np.abs(mpc_z - z_val) < z_tol
                
                # Also filter by other fixed dimensions (px, py, omega)
                other_masks = np.ones(len(mpc_times), dtype=bool)
                for dim, tol in other_dim_tols.items():
                    dim_vals = mpc_inputs_real[:, 1 + dim].numpy()
                    slice_val = state_slices[dim]
                    other_masks &= np.abs(dim_vals - slice_val) < tol
                
                combined_mask = time_mask & z_mask & other_masks
                
                # Extract filtered data
                slice_x = mpc_x[combined_mask]
                slice_y = mpc_y[combined_mask]
                slice_vals = mpc_vals[combined_mask]
                
                num_samples = len(slice_x)
                subplot_idx = i * len(zs) + j + 1
                title_str = f't = {t_val:.2f}, {state_labels[z_axis_idx]} = {z_val:.2f}\n(n={num_samples})'
                
                # Create subplots for both figures
                ax_scatter = fig_scatter.add_subplot(len(times), len(zs), subplot_idx)
                ax_hist = fig_hist.add_subplot(len(times), len(zs), subplot_idx)
                
                ax_scatter.set_title(title_str, fontsize=10)
                ax_hist.set_title(title_str, fontsize=10)
                
                if num_samples < 3:
                    # Not enough data - show gray background
                    ax_scatter.set_facecolor('lightgray')
                    ax_scatter.text(0.5, 0.5, 'Insufficient\nMPC data', 
                           transform=ax_scatter.transAxes, ha='center', va='center',
                           fontsize=12, color='darkgray')
                    if num_samples > 0:
                        scatter = ax_scatter.scatter(slice_x, slice_y, c=slice_vals, 
                                           cmap='coolwarm_r', s=30, edgecolors='black', 
                                           linewidth=0.5, alpha=0.9)
                        fig_scatter.colorbar(scatter, ax=ax_scatter)
                else:
                    # Compute local min/max from scatter data
                    local_vmin = np.nanmin(slice_vals)
                    local_vmax = np.nanmax(slice_vals)
                    
                    # === SCATTER PLOT ===
                    scatter_sc = ax_scatter.scatter(slice_x, slice_y, c=slice_vals, 
                                       cmap='coolwarm_r', s=30, edgecolors='black',
                                       linewidth=0.5, alpha=0.9,
                                       vmin=local_vmin, vmax=local_vmax)
                    fig_scatter.colorbar(scatter_sc, ax=ax_scatter)
                
                # === HISTOGRAM PLOT ===
                if num_samples >= 1:
                    # Create histogram of V values
                    ax_hist.hist(slice_vals, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
                    
                    # Add vertical line at V=0
                    ax_hist.axvline(x=0, color='red', linestyle='--', linewidth=2, label='V=0')
                    
                    # Calculate percentages
                    pct_negative = np.sum(slice_vals < 0) / len(slice_vals) * 100
                    pct_positive = np.sum(slice_vals >= 0) / len(slice_vals) * 100
                    
                    # Add text annotation with percentages
                    ax_hist.text(0.02, 0.98, f'V<0: {pct_negative:.1f}%\nV≥0: {pct_positive:.1f}%',
                                transform=ax_hist.transAxes, fontsize=9,
                                verticalalignment='top', horizontalalignment='left',
                                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                    
                    ax_hist.set_xlabel('V value')
                    ax_hist.set_ylabel('Count')
                    ax_hist.legend(loc='upper right', fontsize=8)
                else:
                    ax_hist.set_facecolor('lightgray')
                    ax_hist.text(0.5, 0.5, 'No data', 
                                transform=ax_hist.transAxes, ha='center', va='center',
                                fontsize=12, color='darkgray')
                
                # Set limits and labels for scatter axis
                ax_scatter.set_xlim(x_min, x_max)
                ax_scatter.set_ylim(y_min, y_max)
                if i == len(times) - 1:
                    ax_scatter.set_xlabel(state_labels[x_axis_idx])
                if j == 0:
                    ax_scatter.set_ylabel(state_labels[y_axis_idx])
        
        # Add titles to figures
        fixed_str = f'Fixed: px={state_slices[0]:.2f}, py={state_slices[1]:.2f}, ω={state_slices[5]:.2f}'
        fig_scatter.suptitle(f'MPC Dataset Velocity - Scatter (Epoch {epoch})\n{fixed_str}', fontsize=14)
        fig_hist.suptitle(f'MPC Dataset Velocity - V Distribution (Epoch {epoch})\n{fixed_str}', fontsize=14)
        
        plt.tight_layout()
        
        if self.use_wandb:
            wandb.log({
                'step': epoch,
                'mpc_dataset_velocity_scatter': wandb.Image(fig_scatter),
                'mpc_dataset_velocity_histogram': wandb.Image(fig_hist),
            })
        
        plt.close(fig_scatter)
        plt.close(fig_hist)

    def validate_mpc_dataset_rotation(self, epoch, x_resolution, y_resolution, time_resolution):
        """
        Visualize MPC training dataset values in rotation plane (theta, omega).
        No z-slicing since position and rotation are decoupled.
        Creates two plots logged to wandb:
        1. Scatter only: just the MPC sample points
        2. Histogram: V-value distribution for each time slice
        """
        if not self.dataset.use_MPC:
            return  # No MPC data to visualize
        
        if not hasattr(self.dataset, 'MPC_inputs') or self.dataset.MPC_inputs is None:
            return  # No MPC data available
        
        if len(self.dataset.MPC_inputs) < 10:
            return  # Not enough MPC data
        
        plot_config = self.dataset.dynamics.plot_config()
        state_test_range = self.dataset.dynamics.state_test_range()
        
        # Rotation plane axes (no z-slicing)
        x_axis_idx = 4  # theta
        y_axis_idx = 5  # omega
        
        # Define grid ranges
        x_min, x_max = state_test_range[x_axis_idx]  # theta range
        y_min, y_max = state_test_range[y_axis_idx]  # omega range
        
        # Define time slices only (no z-slicing)
        times = torch.linspace(0, self.dataset.tMax, time_resolution)
        
        # Convert MPC data from normalized to real coordinates
        mpc_inputs_real = self.dataset.dynamics.input_to_coord(
            self.dataset.MPC_inputs.clone()
        )  # Shape: [N, state_dim+1], columns: [time, px, py, vx, vy, theta, omega]
        mpc_values = self.dataset.MPC_values.clone()
        
        # Extract relevant columns
        mpc_times = mpc_inputs_real[:, 0].numpy()
        mpc_x = mpc_inputs_real[:, 1 + x_axis_idx].numpy()  # theta
        mpc_y = mpc_inputs_real[:, 1 + y_axis_idx].numpy()  # omega
        mpc_vals = mpc_values.numpy()
        
        # Get fixed state slices for non-plotted dimensions
        state_slices = plot_config['state_slices']
        
        # Define tolerances based on grid spacing
        if len(times) > 1:
            time_tol = (times[1] - times[0]).item() / 10.0
        else:
            time_tol = 0.01
        
        # Tolerance for other fixed dimensions (px, py, vx, vy)
        other_dim_tols = {}
        for dim in range(self.dataset.dynamics.state_dim):
            if dim not in [x_axis_idx, y_axis_idx]:
                dim_range = state_test_range[dim][1] - state_test_range[dim][0]
                other_dim_tols[dim] = dim_range * 0.3
        
        # Create two figures (1 row x time_resolution columns - no z-slicing)
        fig_scatter = plt.figure(figsize=(6 * len(times), 5))
        fig_hist = plt.figure(figsize=(6 * len(times), 5))
        
        state_labels = ['px', 'py', 'vx', 'vy', r'$\theta$', r'$\omega$']
        
        for i, t in enumerate(times):
            t_val = t.item()
            
            # Filter MPC samples for this time slice
            time_mask = np.abs(mpc_times - t_val) < time_tol
            
            # Also filter by other fixed dimensions (px, py, vx, vy)
            other_masks = np.ones(len(mpc_times), dtype=bool)
            for dim, tol in other_dim_tols.items():
                dim_vals = mpc_inputs_real[:, 1 + dim].numpy()
                slice_val = state_slices[dim]
                other_masks &= np.abs(dim_vals - slice_val) < tol
            
            combined_mask = time_mask & other_masks
            
            # Extract filtered data
            slice_x = mpc_x[combined_mask]
            slice_y = mpc_y[combined_mask]
            slice_vals = mpc_vals[combined_mask]
            
            num_samples = len(slice_x)
            subplot_idx = i + 1
            title_str = f't = {t_val:.2f}\n(n={num_samples})'
            
            # Create subplots for both figures (1 row)
            ax_scatter = fig_scatter.add_subplot(1, len(times), subplot_idx)
            ax_hist = fig_hist.add_subplot(1, len(times), subplot_idx)
            
            ax_scatter.set_title(title_str, fontsize=10)
            ax_hist.set_title(title_str, fontsize=10)
            
            if num_samples < 3:
                # Not enough data - show gray background
                ax_scatter.set_facecolor('lightgray')
                ax_scatter.text(0.5, 0.5, 'Insufficient\nMPC data', 
                       transform=ax_scatter.transAxes, ha='center', va='center',
                       fontsize=12, color='darkgray')
                if num_samples > 0:
                    scatter = ax_scatter.scatter(slice_x, slice_y, c=slice_vals, 
                                       cmap='coolwarm_r', s=30, edgecolors='black', 
                                       linewidth=0.5, alpha=0.9)
                    fig_scatter.colorbar(scatter, ax=ax_scatter)
            else:
                # Compute local min/max from scatter data
                local_vmin = np.nanmin(slice_vals)
                local_vmax = np.nanmax(slice_vals)
                
                # === SCATTER PLOT ===
                scatter_sc = ax_scatter.scatter(slice_x, slice_y, c=slice_vals, 
                                   cmap='coolwarm_r', s=30, edgecolors='black',
                                   linewidth=0.5, alpha=0.9,
                                   vmin=local_vmin, vmax=local_vmax)
                fig_scatter.colorbar(scatter_sc, ax=ax_scatter)
            
            # === HISTOGRAM PLOT ===
            if num_samples >= 1:
                # Create histogram of V values
                ax_hist.hist(slice_vals, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
                
                # Add vertical line at V=0
                ax_hist.axvline(x=0, color='red', linestyle='--', linewidth=2, label='V=0')
                
                # Calculate percentages
                pct_negative = np.sum(slice_vals < 0) / len(slice_vals) * 100
                pct_positive = np.sum(slice_vals >= 0) / len(slice_vals) * 100
                
                # Add text annotation with percentages
                ax_hist.text(0.02, 0.98, f'V<0: {pct_negative:.1f}%\nV≥0: {pct_positive:.1f}%',
                            transform=ax_hist.transAxes, fontsize=9,
                            verticalalignment='top', horizontalalignment='left',
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                
                ax_hist.set_xlabel('V value')
                ax_hist.set_ylabel('Count')
                ax_hist.legend(loc='upper right', fontsize=8)
            else:
                ax_hist.set_facecolor('lightgray')
                ax_hist.text(0.5, 0.5, 'No data', 
                            transform=ax_hist.transAxes, ha='center', va='center',
                            fontsize=12, color='darkgray')
            
            # Set limits and labels for scatter axis
            ax_scatter.set_xlim(x_min, x_max)
            ax_scatter.set_ylim(y_min, y_max)
            ax_scatter.set_xlabel(state_labels[x_axis_idx])
            if i == 0:
                ax_scatter.set_ylabel(state_labels[y_axis_idx])
        
        # Add titles to figures
        fixed_str = f'Fixed: px={state_slices[0]:.2f}, py={state_slices[1]:.2f}, vx={state_slices[2]:.2f}, vy={state_slices[3]:.2f}'
        fig_scatter.suptitle(f'MPC Dataset Rotation - Scatter (Epoch {epoch})\n{fixed_str}', fontsize=14)
        fig_hist.suptitle(f'MPC Dataset Rotation - V Distribution (Epoch {epoch})\n{fixed_str}', fontsize=14)
        
        plt.tight_layout()
        
        if self.use_wandb:
            wandb.log({
                'step': epoch,
                'mpc_dataset_rotation_scatter': wandb.Image(fig_scatter),
                'mpc_dataset_rotation_histogram': wandb.Image(fig_hist),
            })
        
        plt.close(fig_scatter)
        plt.close(fig_hist)

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
        z_min, z_max = state_test_range[z_axis_idx]
        
        # Create z values (theta angles)
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
            ax.set_title(f'θ={z_val:.2f} rad')
            
            # Add per-subplot colorbar
            fig.colorbar(im, ax=ax, shrink=0.8)
        
        # Add suptitle with time and fixed states
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


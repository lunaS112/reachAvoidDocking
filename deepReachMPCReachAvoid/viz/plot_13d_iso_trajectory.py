"""
13D Docking Trajectory - Iso View with Chaser Snapshots

Generates a Plotly interactive HTML showing:
- Target spacecraft with SIA Lab color scheme
- Full 3D trajectory trace from BRAT controller (v8 trained value function)
- 8 chaser satellite snapshots along the trajectory (motion blur effect)
"""

import argparse
import os
import sys
import numpy as np
import torch
import math
import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dynamics import dynamics
from utils.controllers.brat_controller_13d import BRATController13D
from utils.controllers.safety_filter import SafetyFilter
from utils.brat_visualization_13d import _indent_box_mesh

torch.manual_seed(1)
np.random.seed(1)

# SIA Lab color scheme (from figureGuide.md)
ORANGE = '#ff9500'       # Your method (highlight)
TEAL = '#0d948f'         # Secondary contribution
GREEN = '#2ca02c'        # Goal set
RED = '#d62728'          # Failure set
MEDIUM_GRAY = '#b3b3b3'  # Baselines
DARK_GRAY = '#6b6b6b'    # Secondary baselines
LIGHT_PURPLE = '#cdb3ff' # Confidence bands

# Pale versions for target/grid
PALE_GRAY = '#e0e0e0'    # Target body
PALE_TEAL = '#b2dfdb'    # Docking post
PALE_GRID = '#f0f0f0'    # Background grid


def quat_to_R(q):
    """Quaternion (scalar-first) to rotation matrix."""
    q0, q1, q2, q3 = q
    return np.array([
        [1 - 2*(q2**2 + q3**2), 2*(q1*q2 - q0*q3), 2*(q1*q3 + q0*q2)],
        [2*(q1*q2 + q0*q3), 1 - 2*(q1**2 + q3**2), 2*(q2*q3 - q0*q1)],
        [2*(q1*q3 - q0*q2), 2*(q2*q3 + q0*q1), 1 - 2*(q1**2 + q2**2)]
    ])


def add_chaser_plotly(fig, center, quaternion, dims, alpha=0.7, color='royalblue',
                      name='Chaser', edge_color='#e68600'):
    """Add a chaser satellite box as a Plotly Mesh3d trace."""
    import plotly.graph_objects as go

    hw, hh, hd = dims[0]/2, dims[1]/2, dims[2]/2
    local = np.array([
        [-hw, -hh, -hd], [ hw, -hh, -hd], [ hw,  hh, -hd], [-hw,  hh, -hd],
        [-hw, -hh,  hd], [ hw, -hh,  hd], [ hw,  hh,  hd], [-hw,  hh,  hd],
    ])

    qn = quaternion / np.linalg.norm(quaternion)
    R = quat_to_R(qn)
    verts = (R @ local.T).T + np.asarray(center)

    quad_idx = [
        [0,1,2,3], [4,5,6,7],
        [0,1,5,4], [2,3,7,6],
        [0,3,7,4], [1,2,6,5],
    ]
    i_list, j_list, k_list = [], [], []
    for qi in quad_idx:
        i_list += [qi[0], qi[0]]
        j_list += [qi[1], qi[2]]
        k_list += [qi[2], qi[3]]

    fig.add_trace(go.Mesh3d(
        x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
        i=i_list, j=j_list, k=k_list,
        color=color, opacity=alpha,
        flatshading=True,
        name=name,
        showlegend=False,
    ))

    edges = [
        (0,1),(1,2),(2,3),(3,0),
        (4,5),(5,6),(6,7),(7,4),
        (0,4),(1,5),(2,6),(3,7),
    ]
    xe, ye, ze = [], [], []
    for a, b in edges:
        xe += [verts[a,0], verts[b,0], None]
        ye += [verts[a,1], verts[b,1], None]
        ze += [verts[a,2], verts[b,2], None]
    fig.add_trace(go.Scatter3d(
        x=xe, y=ye, z=ze, mode='lines',
        line=dict(color=edge_color, width=2),
        showlegend=False,
    ))


def add_target_styled(fig, dynamics_):
    """Add target spacecraft with pale SIA Lab colors."""
    import plotly.graph_objects as go

    d = dynamics_
    hw, hh, hd = d.w_t / 2, d.h_t, d.d_t / 2

    cx = [-hw, hw, hw, -hw, -hw, hw, hw, -hw]
    cy = [0, 0, hh, hh, 0, 0, hh, hh]
    cz = [-hd, -hd, -hd, -hd, hd, hd, hd, hd]

    i_list, j_list, k_list = [], [], []
    quad_idx = [
        [0,1,2,3], [4,5,6,7],
        [0,1,5,4], [2,3,7,6],
        [0,3,7,4], [1,2,6,5],
    ]
    for qi in quad_idx:
        i_list += [qi[0], qi[0]]
        j_list += [qi[1], qi[2]]
        k_list += [qi[2], qi[3]]

    fig.add_trace(go.Mesh3d(
        x=cx, y=cy, z=cz,
        i=i_list, j=j_list, k=k_list,
        color=PALE_GRAY, opacity=0.45,
        flatshading=True,
        name='Target body',
        showlegend=False,
    ))

    edges = [
        (0,1),(1,2),(2,3),(3,0),
        (4,5),(5,6),(6,7),(7,4),
        (0,4),(1,5),(2,6),(3,7),
    ]
    xe, ye, ze = [], [], []
    for a, b in edges:
        xe += [cx[a], cx[b], None]
        ye += [cy[a], cy[b], None]
        ze += [cz[a], cz[b], None]
    fig.add_trace(go.Scatter3d(
        x=xe, y=ye, z=ze, mode='lines',
        line=dict(color=MEDIUM_GRAY, width=2),
        showlegend=False,
    ))

    pts = _indent_box_mesh(d.post_hw_x, d.post_hw_z, d.post_length)
    fig.add_trace(go.Mesh3d(
        x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
        opacity=0.40, color=PALE_TEAL,
        alphahull=0,
        name='Docking post',
        showlegend=False,
    ))


if __name__ == "__main__":
    import plotly.graph_objects as go

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['fig1', 'figlast', 'anim'], default='fig1',
                        help='fig1 = interactive HTML, figlast = high-res PNG, anim = MP4 animation')
    parser.add_argument('--trail', action='store_true',
                        help='anim only: leave a fading trail of past chaser snapshots behind the moving chaser')
    parser.add_argument('--fps', type=int, default=20,
                        help='anim only: output video framerate (default 20)')
    parser.add_argument('--stride', type=int, default=1,
                        help='anim only: render every Nth simulation step (default 1)')
    parser.add_argument('--out-dir', type=str, default=None,
                        help='anim only: write MP4 into this existing directory instead of a new timestamped one')
    parser.add_argument('--frame-width', type=int, default=1280)
    parser.add_argument('--frame-height', type=int, default=960)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load BRAT controller with v8 trained value function + safety filter
    checkpoint = './runs/Docking13D_RA_v8_TightOmega/training/checkpoints/model_final.pth'
    avoid_checkpoint = './runs/Docking13D_Avoid'
    safety_filter = SafetyFilter(
        mode=1,
        checkpoint_path=avoid_checkpoint,
        device=device,
    )
    controller = BRATController13D(
        checkpoint_path=checkpoint,
        tMax=10.0,
        dt=0.1,
        device=device,
        safety_filter=safety_filter,
    )

    dynamics_ = controller.dynamics

    # Initial condition: (2,9,1) with large roll + pitch offset from goal
    # Goal quaternion is 90° yaw about z-axis: [cos(45°), 0, 0, sin(45°)]
    # Build IC quaternion by composing: goal * roll(150°) * pitch(120°)
    q_goal_np = dynamics_.q_goal.cpu().numpy()

    def _axis_angle_quat(axis, deg):
        a = np.asarray(axis, dtype=float)
        a = a / np.linalg.norm(a)
        r = deg * math.pi / 180.0
        return np.array([math.cos(r/2), *(a * math.sin(r/2))])

    def _qmul(p, q):
        return np.array([
            p[0]*q[0] - p[1]*q[1] - p[2]*q[2] - p[3]*q[3],
            p[0]*q[1] + p[1]*q[0] + p[2]*q[3] - p[3]*q[2],
            p[0]*q[2] - p[1]*q[3] + p[2]*q[0] + p[3]*q[1],
            p[0]*q[3] + p[1]*q[2] - p[2]*q[1] + p[3]*q[0],
        ])

    q_roll  = _axis_angle_quat([1, 0, 0], 150.0)   # 150° roll
    q_pitch = _axis_angle_quat([0, 1, 0], 120.0)    # 120° pitch
    q_far = _qmul(_qmul(q_goal_np, q_roll), q_pitch)
    q_far = q_far / np.linalg.norm(q_far)

    ic = np.array([2.0, 9.0, 1.0, 0, 0, 0,
                    q_far[0], q_far[1], q_far[2], q_far[3],
                    0, 0, 0])

    # Print angular distance from goal
    q_goal_np = dynamics_.q_goal.cpu().numpy()
    dot = abs(np.dot(q_far, q_goal_np))
    ang_dist = 2 * math.acos(min(dot, 1.0)) * 180 / math.pi
    print(f"Goal quaternion: {q_goal_np}")
    print(f"IC quaternion:   {q_far}")
    print(f"Angular distance from goal: {ang_dist:.1f}°")

    # Simulate with BRAT controller (up to 120s)
    print("Computing BRAT trajectory (v8 model, max 120s)...")
    result = controller.simulate_docking(ic, max_sim_time=120.0)

    traj = result['trajectory']  # (N, 13)
    print(f"Trajectory: {traj.shape[0]} steps, {result['times'][-1]:.1f}s")
    print(f"Success: {result['success']}, Collision: {result['collision']}")

    chaser_dims = np.array([dynamics_.w_c, dynamics_.h_c, dynamics_.d_c])
    n_steps = traj.shape[0]

    # Layout ranges (shared across all modes) — iso view with grid tightly around trajectory + target
    all_pts = traj[:, :3]
    margin = 1.0
    # Trajectory extents
    traj_min = all_pts.min(axis=0)
    traj_max = all_pts.max(axis=0)
    # Target craft extents
    target_min = np.array([-dynamics_.w_t/2, 0, -dynamics_.d_t/2])
    target_max = np.array([ dynamics_.w_t/2, dynamics_.h_t,  dynamics_.d_t/2])
    # Combined extents + margin
    x_range = [min(traj_min[0], target_min[0]) - margin,
               max(traj_max[0], target_max[0]) + margin]
    y_range = [min(traj_min[1], target_min[1]) - margin,
               max(traj_max[1], target_max[1]) + margin]
    z_range = [min(traj_min[2], target_min[2]) - margin,
               max(traj_max[2], target_max[2]) + margin]

    # Make equal aspect ratio
    max_span = max(x_range[1]-x_range[0], y_range[1]-y_range[0], z_range[1]-z_range[0])
    for r in [x_range, y_range, z_range]:
        mid = (r[0] + r[1]) / 2
        r[0] = mid - max_span/2
        r[1] = mid + max_span/2

    # Round ranges to clean integers for tick labels
    import math as _m
    x_ticks = [_m.ceil(x_range[0]), _m.floor(x_range[1])]
    y_ticks = [_m.ceil(y_range[0]), _m.floor(y_range[1])]
    z_ticks = [_m.ceil(z_range[0]), _m.floor(z_range[1])]

    tick_font = dict(size=13, family='Arial, Helvetica, sans-serif')
    title_font = dict(size=15, family='Arial, Helvetica, sans-serif')

    if args.mode in ('figlast', 'anim'):
        pale_axis = dict(
            tickfont=tick_font,
        )
    else:
        pale_axis = dict(
            showticklabels=False,
            title='',
            showgrid=False,
            showbackground=False,
            zeroline=True,
        )

    def apply_layout(fig, width, height):
        fig.update_layout(
            title=None,
            showlegend=False,
            margin=dict(l=0, r=0, t=0, b=0),
            scene=dict(
                xaxis=dict(range=x_range, tickvals=x_ticks,
                           title=dict(text='x (m)', font=title_font), **pale_axis),
                yaxis=dict(range=y_range, tickvals=y_ticks,
                           title=dict(text='y (m)', font=title_font), **pale_axis),
                zaxis=dict(range=z_range, tickvals=z_ticks,
                           title=dict(text='z (m)', font=title_font), **pale_axis),
                aspectmode='cube',
                camera=dict(
                    eye=dict(x=1.5, y=-1.5, z=1.0),
                    up=dict(x=0, y=0, z=1),
                ),
            ),
            width=width,
            height=height,
            paper_bgcolor='white',
            plot_bgcolor='white',
        )

    # Save
    save_root = Path(__file__).parent
    daytime = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if args.mode == 'anim' and args.out_dir is not None:
        save_dir = Path(args.out_dir)
    else:
        save_dir = save_root / f"data/{daytime}__13D_iso_trajectory_BRAT"
    save_dir.mkdir(parents=True, exist_ok=True)

    if args.mode in ('fig1', 'figlast'):
        fig = go.Figure()
        add_target_styled(fig, dynamics_)

        # Full trajectory trace
        fig.add_trace(go.Scatter3d(
            x=traj[:, 0], y=traj[:, 1], z=traj[:, 2],
            mode='lines', line=dict(color=ORANGE, width=4),
            showlegend=False,
        ))
        # Start / end markers
        fig.add_trace(go.Scatter3d(
            x=[traj[0, 0]], y=[traj[0, 1]], z=[traj[0, 2]],
            mode='markers', marker=dict(size=6, color=GREEN, symbol='circle'),
            showlegend=False,
        ))
        fig.add_trace(go.Scatter3d(
            x=[traj[-1, 0]], y=[traj[-1, 1]], z=[traj[-1, 2]],
            mode='markers', marker=dict(size=6, color=RED, symbol='diamond'),
            showlegend=False,
        ))

        # 8 chaser snapshots
        n_snapshots = 8
        snapshot_indices = np.linspace(0, n_steps - 1, n_snapshots, dtype=int)
        if args.mode == 'figlast':
            alphas = np.full(n_snapshots, 0.75)
            colors = [ORANGE] * n_snapshots
        else:
            alphas = np.linspace(0.35, 0.90, n_snapshots)
            colors = ['#fff3e0', '#ffe0b2', '#ffcc80', '#ffb74d',
                      '#ffa726', '#ff9500', '#fb8c00', '#e68600']
        for i, idx in enumerate(snapshot_indices):
            state = traj[idx]
            add_chaser_plotly(
                fig, state[:3], state[6:10], chaser_dims,
                alpha=float(alphas[i]), color=colors[i],
                name=f'Chaser t={result["times"][idx]:.1f}s',
                edge_color=colors[i],
            )

        apply_layout(fig, width=1000, height=800)

        if args.mode == 'fig1':
            html_path = save_dir / "trajectory_iso_view.html"
            fig.write_html(str(html_path))
            print(f"Saved HTML to: {html_path}")
        else:
            png_path = save_dir / "trajectory_iso_view.png"
            fig.write_image(str(png_path), scale=6, width=1600, height=1200)
            print(f"Saved PNG to: {png_path}")

    else:
        # anim mode: render one frame per subsampled step, encode with ffmpeg
        import shutil, subprocess, tempfile

        frame_indices = list(range(0, n_steps, max(1, args.stride)))
        if frame_indices[-1] != n_steps - 1:
            frame_indices.append(n_steps - 1)

        # Trail snapshot cadence — match the 8-snapshot look of the PNG
        trail_stride = max(1, n_steps // 8)

        frames_dir = Path(tempfile.mkdtemp(prefix="traj_anim_"))
        print(f"Rendering {len(frame_indices)} frames to {frames_dir} ...")

        try:
            for fi, idx in enumerate(frame_indices):
                fig = go.Figure()
                add_target_styled(fig, dynamics_)

                # Trace up to current step
                fig.add_trace(go.Scatter3d(
                    x=traj[:idx + 1, 0], y=traj[:idx + 1, 1], z=traj[:idx + 1, 2],
                    mode='lines', line=dict(color=ORANGE, width=4),
                    showlegend=False,
                ))
                # Start marker always visible
                fig.add_trace(go.Scatter3d(
                    x=[traj[0, 0]], y=[traj[0, 1]], z=[traj[0, 2]],
                    mode='markers', marker=dict(size=6, color=GREEN, symbol='circle'),
                    showlegend=False,
                ))

                if args.trail:
                    # Fading past snapshots at fixed stride, then current chaser in front
                    past = [j for j in range(0, idx, trail_stride)]
                    for rank, j in enumerate(past):
                        age = (len(past) - rank)  # 1 = most recent past, larger = older
                        alpha = max(0.15, 0.7 * (0.75 ** (age - 1)))
                        st = traj[j]
                        add_chaser_plotly(
                            fig, st[:3], st[6:10], chaser_dims,
                            alpha=float(alpha), color=ORANGE,
                            name=f'trail_{j}', edge_color=ORANGE,
                        )

                st = traj[idx]
                add_chaser_plotly(
                    fig, st[:3], st[6:10], chaser_dims,
                    alpha=0.9, color=ORANGE,
                    name='chaser', edge_color='#e68600',
                )

                apply_layout(fig, width=args.frame_width, height=args.frame_height)

                frame_path = frames_dir / f"frame_{fi:05d}.png"
                fig.write_image(str(frame_path),
                                width=args.frame_width, height=args.frame_height, scale=1)

                if (fi + 1) % 25 == 0 or fi == len(frame_indices) - 1:
                    print(f"  frame {fi + 1}/{len(frame_indices)}")

            mp4_path = save_dir / "trajectory_iso_view.mp4"
            cmd = [
                'ffmpeg', '-y', '-framerate', str(args.fps),
                '-i', str(frames_dir / 'frame_%05d.png'),
                '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                '-vf', f'pad=ceil(iw/2)*2:ceil(ih/2)*2',
                '-crf', '18',
                str(mp4_path),
            ]
            print(f"Encoding MP4: {' '.join(cmd)}")
            subprocess.run(cmd, check=True)
            print(f"Saved MP4 to: {mp4_path}")
        finally:
            shutil.rmtree(frames_dir, ignore_errors=True)

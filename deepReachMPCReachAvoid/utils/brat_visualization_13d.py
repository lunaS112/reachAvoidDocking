"""
3D BRAT Visualization for Docking13D

Renders the backward reachable tube as a translucent 3D blob in (x, y, z)
position space, with the remaining 10 states fixed to user-supplied values
(typically the chaser's current state).  Ships with two interchangeable
back-ends:

  • **plotly**      – Interactive HTML with rotation/zoom.  Isosurface for
                      the V=0 boundary, optional volume trace for the value
                      gradient inside the BRAT.
  • **matplotlib**  – Static / animated figures consistent with the existing
                      ``trajectory_animation_13D.py``.  Uses marching cubes
                      to extract the V=0 surface and colour-maps it by value.

Both back-ends produce:
  • Translucent BRAT shell (V = 0 surface)
  • Internal value gradient visualisation (heat-map colouring)
  • Target spacecraft geometry (box + hemispherical dock cutout)
  • Reach tolerance sphere at the origin
  • Oriented chaser spacecraft cube
  • Trajectory trace (optional)
"""

from __future__ import annotations

import numpy as np
import torch

# --------------------------------------------------------------------------- #
#  Play / Pause JS injection for HTML animations
# --------------------------------------------------------------------------- #

_PLAY_PAUSE_JS = r"""
<div id="anim-controls" style="text-align:center;margin:8px 0;">
  <button id="play-btn"  style="font-size:16px;padding:6px 18px;cursor:pointer;">&#9654; Play</button>
  <button id="pause-btn" style="font-size:16px;padding:6px 18px;cursor:pointer;">&#9646;&#9646; Pause</button>
  <label style="margin-left:12px;">Speed:
    <input id="speed-range" type="range" min="100" max="2000" value="{interval}" step="50" style="vertical-align:middle;">
    <span id="speed-label">{interval}ms</span>
  </label>
</div>
<script>
(function() {
  var timer = null;
  var interval = {interval};
  var gd = document.querySelectorAll('.plotly-graph-div')[0];
  function getSlider() {
    try { return gd._fullLayout.sliders[0]; } catch(e) { return null; }
  }
  function stepForward() {
    var s = getSlider(); if (!s) return;
    var nSteps = s.steps.length;
    var cur = s.active || 0;
    var next = (cur + 1) % nSteps;
    Plotly.relayout(gd, {'sliders[0].active': next});
    var step = s.steps[next];
    Plotly.update(gd, step.args[0], step.args[1]);
  }
  document.getElementById('play-btn').onclick = function() {
    if (timer) return;
    timer = setInterval(stepForward, interval);
  };
  document.getElementById('pause-btn').onclick = function() {
    clearInterval(timer); timer = null;
  };
  document.getElementById('speed-range').oninput = function() {
    interval = parseInt(this.value);
    document.getElementById('speed-label').textContent = interval + 'ms';
    if (timer) { clearInterval(timer); timer = setInterval(stepForward, interval); }
  };
})();
</script>
"""


def _inject_play_pause_js(html_path: str, interval_ms: int = 500):
    """Append Play/Pause controls + JS to an existing Plotly HTML file."""
    snippet = _PLAY_PAUSE_JS.replace('{interval}', str(interval_ms))
    with open(html_path, 'r') as f:
        html = f.read()
    # Insert just before </body>
    html = html.replace('</body>', snippet + '\n</body>')
    with open(html_path, 'w') as f:
        f.write(html)


# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #

def _quat_to_R(q):
    """Scalar-first quaternion -> LVLH-to-body rotation matrix (3×3)."""
    q0, q1, q2, q3 = q
    return np.array([
        [1 - 2*(q2*q2 + q3*q3), 2*(q1*q2 + q0*q3),     2*(q1*q3 - q0*q2)],
        [2*(q1*q2 - q0*q3),     1 - 2*(q1*q1 + q3*q3),  2*(q2*q3 + q0*q1)],
        [2*(q1*q3 + q0*q2),     2*(q2*q3 - q0*q1),      1 - 2*(q1*q1 + q2*q2)],
    ])


def _oriented_box_mesh(center, half_dims, quaternion=None):
    """Return (vertices, faces) for an axis-aligned or oriented box.

    ``half_dims`` = (hw, hh, hd).
    ``quaternion`` (scalar-first) rotates body→LVLH; pass *None* for AA.

    Returns:
        verts: (8, 3) ndarray.
        faces: list of 6 quads, each quad = list of 4 (x,y,z).
    """
    hw, hh, hd = half_dims
    local = np.array([
        [-hw, -hh, -hd], [ hw, -hh, -hd], [ hw,  hh, -hd], [-hw,  hh, -hd],
        [-hw, -hh,  hd], [ hw, -hh,  hd], [ hw,  hh,  hd], [-hw,  hh,  hd],
    ])
    if quaternion is not None:
        R = _quat_to_R(quaternion)
        local = (R.T @ local.T).T  # body→LVLH
    verts = local + np.asarray(center)
    idx = [
        [0,1,2,3], [4,5,6,7],
        [0,1,5,4], [2,3,7,6],
        [0,3,7,4], [1,2,6,5],
    ]
    faces = [verts[i].tolist() for i in idx]
    return verts, faces


def _indent_box_mesh(hw_x, hw_z, depth, n_face=5):
    """Return surface points of a rectangular box for Mesh3d.
    (Legacy name kept for import compat; now also used for docking post.)
    Box: x ∈ [-hw_x, hw_x], z ∈ [-hw_z, hw_z], y ∈ [-depth, 0]
    """
    pts = []
    xs = np.linspace(-hw_x, hw_x, n_face)
    zs = np.linspace(-hw_z, hw_z, n_face)
    ys = np.linspace(-depth, 0, n_face)
    # Top (y = 0)
    X, Z = np.meshgrid(xs, zs)
    pts.append(np.column_stack([X.ravel(), np.zeros(X.size), Z.ravel()]))
    # Bottom (y = -depth)
    pts.append(np.column_stack([X.ravel(), np.full(X.size, -depth), Z.ravel()]))
    # Side walls
    Y, Z2 = np.meshgrid(ys, zs)
    pts.append(np.column_stack([np.full(Y.size, -hw_x), Y.ravel(), Z2.ravel()]))
    pts.append(np.column_stack([np.full(Y.size, hw_x), Y.ravel(), Z2.ravel()]))
    Y2, X2 = np.meshgrid(ys, xs)
    pts.append(np.column_stack([X2.ravel(), Y2.ravel(), np.full(Y2.size, -hw_z)]))
    pts.append(np.column_stack([X2.ravel(), Y2.ravel(), np.full(Y2.size, hw_z)]))
    return np.vstack(pts)


# --------------------------------------------------------------------------- #
#  Standalone geometry helpers (importable by TrajectoryAnimation13D etc.)
# --------------------------------------------------------------------------- #

def add_target_plotly(fig, dynamics):
    """Add target body (solid Mesh3d + wireframe + dock) as 3 Plotly traces."""
    import plotly.graph_objects as go

    d = dynamics
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
        color='silver', opacity=0.55,
        flatshading=True,
        name='Target body',
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
        line=dict(color='gray', width=2),
        name='Target edges', showlegend=False,
    ))

    pts = _indent_box_mesh(d.post_hw_x, d.post_hw_z, d.post_length)
    fig.add_trace(go.Mesh3d(
        x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
        opacity=0.45, color='limegreen',
        alphahull=0,
        name='Docking post',
    ))


def add_reach_sphere_plotly(fig, dynamics):
    """Add reach tolerance sphere as a Plotly Surface trace."""
    import plotly.graph_objects as go

    r = dynamics.eps_p
    u = np.linspace(0, 2 * np.pi, 20)
    v = np.linspace(0, np.pi, 20)
    xs = r * np.outer(np.cos(u), np.sin(v))
    ys = r * np.outer(np.sin(u), np.sin(v))
    zs = r * np.outer(np.ones_like(u), np.cos(v))
    fig.add_trace(go.Surface(
        x=xs, y=ys, z=zs,
        opacity=0.20, colorscale=[[0, 'lime'], [1, 'lime']],
        showscale=False, name='Reach tol.',
    ))


def add_chaser_plotly(fig, dynamics, state):
    """Add oriented chaser cube as a Plotly Mesh3d trace."""
    import plotly.graph_objects as go

    d = dynamics
    pos = state[:3]
    q = state[6:10] / (np.linalg.norm(state[6:10]) + 1e-12)
    verts, _ = _oriented_box_mesh(
        pos, (d.w_c / 2, d.h_c / 2, d.d_c / 2), quaternion=q)

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
    # Per-face colouring: docking face (body +x, quad 5 → tri 10-11) = blue,
    # top face (body +z, quad 1 → tri 2-3) = green, rest = orange.
    facecolor = ['orange'] * 12
    facecolor[10] = 'dodgerblue'   # body +x (dock face) tri A
    facecolor[11] = 'dodgerblue'   # body +x (dock face) tri B
    facecolor[2]  = 'limegreen'    # body +z (top face) tri A
    facecolor[3]  = 'limegreen'    # body +z (top face) tri B
    fig.add_trace(go.Mesh3d(
        x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
        i=i_list, j=j_list, k=k_list,
        opacity=1.0, facecolor=facecolor,
        flatshading=True,
        name='Chaser',
    ))


def add_chaser_wireframe_plotly(fig, dynamics, state):
    """Add chaser wireframe edges as a Scatter3d trace."""
    import plotly.graph_objects as go

    d = dynamics
    pos = state[:3]
    q = state[6:10] / (np.linalg.norm(state[6:10]) + 1e-12)
    verts, _ = _oriented_box_mesh(
        pos, (d.w_c / 2, d.h_c / 2, d.d_c / 2), quaternion=q)

    edges = [
        (0,1),(1,2),(2,3),(3,0),
        (4,5),(5,6),(6,7),(7,4),
        (0,4),(1,5),(2,6),(3,7),
    ]
    xe, ye, ze = [], [], []
    for a, b in edges:
        xe += [verts[a, 0], verts[b, 0], None]
        ye += [verts[a, 1], verts[b, 1], None]
        ze += [verts[a, 2], verts[b, 2], None]
    fig.add_trace(go.Scatter3d(
        x=xe, y=ye, z=ze, mode='lines',
        line=dict(color='black', width=2.5),
        name='Chaser edges', showlegend=False,
    ))


def add_chaser_axes_plotly(fig, dynamics, state):
    """Add orientation cone arrows anchored to chaser face centres."""
    import plotly.graph_objects as go

    d = dynamics
    pos = np.asarray(state[:3], dtype=np.float64)
    q = state[6:10] / (np.linalg.norm(state[6:10]) + 1e-12)
    R = _quat_to_R(q)

    # Body +x → nose (docking direction), body +z → "up"
    nose_dir = R.T @ np.array([1.0, 0.0, 0.0])  # body +x in LVLH
    up_dir   = R.T @ np.array([0.0, 0.0, 1.0])  # body +z in LVLH

    # Anchor cones at the face centre (outside the body)
    nose_origin = pos + nose_dir * (d.w_c / 2)
    up_origin   = pos + up_dir * (d.d_c / 2)

    arrow_scale = max(d.w_c, d.h_c, d.d_c) * 1.2
    fig.add_trace(go.Cone(
        x=[nose_origin[0]], y=[nose_origin[1]], z=[nose_origin[2]],
        u=[nose_dir[0] * arrow_scale],
        v=[nose_dir[1] * arrow_scale],
        w=[nose_dir[2] * arrow_scale],
        sizemode='absolute', sizeref=0.6,
        colorscale=[[0, 'dodgerblue'], [1, 'dodgerblue']],
        showscale=False, name='Nose (+x)',
    ))
    fig.add_trace(go.Cone(
        x=[up_origin[0]], y=[up_origin[1]], z=[up_origin[2]],
        u=[up_dir[0] * arrow_scale * 0.5],
        v=[up_dir[1] * arrow_scale * 0.5],
        w=[up_dir[2] * arrow_scale * 0.5],
        sizemode='absolute', sizeref=0.4,
        colorscale=[[0, 'limegreen'], [1, 'limegreen']],
        showscale=False, name='Up (+z)',
    ))


def add_dock_rim_plotly(fig, dynamics):
    """Add dock port rim rectangle as a Scatter3d trace at y=0."""
    import plotly.graph_objects as go

    hw_x = dynamics.post_hw_x
    hw_z = dynamics.post_hw_z
    # Rectangle at y=0 (base of post)
    rim_x = [-hw_x, hw_x, hw_x, -hw_x, -hw_x]
    rim_y = [0, 0, 0, 0, 0]
    rim_z = [-hw_z, -hw_z, hw_z, hw_z, -hw_z]
    fig.add_trace(go.Scatter3d(
        x=rim_x, y=rim_y, z=rim_z,
        mode='lines',
        line=dict(color='limegreen', width=4),
        name='Dock rim',
    ))


def add_goal_ghost_plotly(fig, dynamics):
    """Add a wireframe ghost cube at the goal orientation (origin)."""
    import plotly.graph_objects as go

    d = dynamics
    q_goal = np.array([
        np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)
    ])  # 90 deg yaw
    verts, _ = _oriented_box_mesh(
        [0, 0, 0], (d.w_c / 2, d.h_c / 2, d.d_c / 2), quaternion=q_goal)

    edges = [
        (0,1),(1,2),(2,3),(3,0),
        (4,5),(5,6),(6,7),(7,4),
        (0,4),(1,5),(2,6),(3,7),
    ]
    xe, ye, ze = [], [], []
    for a, b in edges:
        xe += [verts[a, 0], verts[b, 0], None]
        ye += [verts[a, 1], verts[b, 1], None]
        ze += [verts[a, 2], verts[b, 2], None]
    fig.add_trace(go.Scatter3d(
        x=xe, y=ye, z=ze, mode='lines',
        line=dict(color='cyan', width=2, dash='dash'),
        name='Goal pose',
    ))


def add_trajectory_plotly(fig, trajectory, safety_active=None):
    """Add trajectory Scatter3d trace (always emits exactly 1 trace).

    Parameters
    ----------
    fig : plotly Figure
    trajectory : (T, 13) array or None
    safety_active : (T,) bool array or None
        Per-step flag indicating safety filter was active.  When provided
        the trajectory is coloured cyan normally and red where the filter
        intervened.
    """
    import plotly.graph_objects as go

    if trajectory is not None and len(trajectory) > 1:
        if safety_active is not None and len(safety_active) >= len(trajectory):
            # Per-point colour: 0 = nominal (cyan), 1 = filter active (red)
            color_vals = safety_active[:len(trajectory)].astype(float)
            line_kwargs = dict(
                color=color_vals,
                colorscale=[[0, 'cyan'], [1, 'red']],
                cmin=0, cmax=1,
                width=4,
            )
        else:
            line_kwargs = dict(color='cyan', width=4)
        fig.add_trace(go.Scatter3d(
            x=trajectory[:, 0], y=trajectory[:, 1], z=trajectory[:, 2],
            mode='lines',
            line=line_kwargs,
            name='Trajectory',
        ))
    else:
        fig.add_trace(go.Scatter3d(
            x=[], y=[], z=[],
            mode='lines',
            line=dict(color='cyan', width=4),
            name='Trajectory',
            visible=False,
        ))


def draw_target_mpl(ax, dynamics):
    """Draw target spacecraft on a matplotlib 3D axis."""
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    d = dynamics
    hw, hd = d.w_t / 2, d.d_t / 2
    _, faces = _oriented_box_mesh(
        [0, d.h_t / 2, 0], (hw, d.h_t / 2, hd))
    mesh = Poly3DCollection(faces, alpha=0.35, linewidths=0.5)
    mesh.set_facecolor('silver')
    mesh.set_edgecolor('gray')
    ax.add_collection3d(mesh)

    # Docking post wireframe
    hw_x = d.post_hw_x
    hw_z = d.post_hw_z
    length = d.post_length
    # Draw wireframe lines at several y-slices through the post
    for y_frac in [0.0, 0.5, 1.0]:
        yy = -length * y_frac
        rim_x = [-hw_x, hw_x, hw_x, -hw_x, -hw_x]
        rim_z = [-hw_z, -hw_z, hw_z, hw_z, -hw_z]
        ax.plot(rim_x, [yy]*5, rim_z, color='limegreen', alpha=0.5, lw=1.0)
    # Vertical edges of the post
    for px, pz in [(-hw_x, -hw_z), (hw_x, -hw_z), (hw_x, hw_z), (-hw_x, hw_z)]:
        ax.plot([px, px], [0, -length], [pz, pz], color='limegreen', alpha=0.5, lw=1.0)


def draw_reach_sphere_mpl(ax, dynamics):
    """Draw reach tolerance sphere on a matplotlib 3D axis."""
    r = dynamics.eps_p
    u = np.linspace(0, 2 * np.pi, 15)
    v = np.linspace(0, np.pi, 15)
    xs = r * np.outer(np.cos(u), np.sin(v))
    ys = r * np.outer(np.sin(u), np.sin(v))
    zs = r * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(xs, ys, zs, color='lime', alpha=0.15)


def draw_chaser_mpl(ax, dynamics, state):
    """Draw oriented chaser cube + nose arrow on a matplotlib 3D axis."""
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    d = dynamics
    pos = state[:3]
    q = state[6:10] / (np.linalg.norm(state[6:10]) + 1e-12)
    _, faces = _oriented_box_mesh(
        pos, (d.w_c / 2, d.h_c / 2, d.d_c / 2), quaternion=q)
    mesh = Poly3DCollection(faces, alpha=0.6, linewidths=0.5)
    mesh.set_facecolor('orange')
    mesh.set_edgecolor('darkorange')
    ax.add_collection3d(mesh)

    R = _quat_to_R(q)
    nose_dir = R.T @ np.array([1.0, 0.0, 0.0])
    arrow_len = 1.5
    ax.quiver(pos[0], pos[1], pos[2],
              nose_dir[0] * arrow_len,
              nose_dir[1] * arrow_len,
              nose_dir[2] * arrow_len,
              color='blue', linewidth=2.0, arrow_length_ratio=0.25)


def compute_scene_limits(traj, dynamics, padding=1.5, max_range=None):
    """Compute target-centred axis limits encompassing the trajectory.

    Parameters
    ----------
    traj      : (T, >=3) array of states.
    dynamics  : Docking13D instance (provides target geometry).
    padding   : extra padding around the extent (metres).
    max_range : optional clamp; *None* → no clamping (full trajectory extent).

    Returns
    -------
    dict with 'xlim', 'ylim', 'zlim' keys, each a (lo, hi) tuple.
    """
    d = dynamics
    pad = padding
    x_lo = min(float(traj[:, 0].min()), -d.w_t / 2) - pad
    x_hi = max(float(traj[:, 0].max()),  d.w_t / 2) + pad
    y_lo = min(float(traj[:, 1].min()), 0.0) - pad
    y_hi = max(float(traj[:, 1].max()), d.h_t) + pad
    z_lo = min(float(traj[:, 2].min()), -d.d_t / 2) - pad
    z_hi = max(float(traj[:, 2].max()),  d.d_t / 2) + pad
    # Clamp to max_range if specified
    if max_range is not None:
        x_lo = max(x_lo, -max_range)
        x_hi = min(x_hi,  max_range)
        y_lo = max(y_lo, -max_range)
        y_hi = min(y_hi,  max_range)
        z_lo = max(z_lo, -max_range)
        z_hi = min(z_hi,  max_range)
    # Equalise ranges to form a cube → prevents aspect warping
    spans = [x_hi - x_lo, y_hi - y_lo, z_hi - z_lo]
    max_span = max(spans)
    x_mid = (x_lo + x_hi) / 2
    y_mid = (y_lo + y_hi) / 2
    z_mid = (z_lo + z_hi) / 2
    x_lo, x_hi = x_mid - max_span / 2, x_mid + max_span / 2
    y_lo, y_hi = y_mid - max_span / 2, y_mid + max_span / 2
    z_lo, z_hi = z_mid - max_span / 2, z_mid + max_span / 2
    return dict(xlim=(x_lo, x_hi), ylim=(y_lo, y_hi), zlim=(z_lo, z_hi))


def state_range_limits(dynamics):
    """Return axis limits from the dynamics class ``state_range_`` tensor.

    Uses positions dims 0-2 of ``dynamics.state_range_`` so the view
    always matches the training domain (respects ``override_state_range``).

    Returns
    -------
    dict with 'xlim', 'ylim', 'zlim' keys, each a (lo, hi) tuple.
    """
    sr = dynamics.state_range_.cpu().tolist()
    return dict(
        xlim=(sr[0][0], sr[0][1]),
        ylim=(sr[1][0], sr[1][1]),
        zlim=(sr[2][0], sr[2][1]),
    )


# --------------------------------------------------------------------------- #
#  Main class
# --------------------------------------------------------------------------- #

class BRATVisualizer13D:
    """Dual-backend 3D BRAT blob renderer for Docking13D.

    Parameters
    ----------
    controller : BRATController13D | MPCTerminalController13D
        Must expose ``self.model``, ``self.dynamics``, ``self.device``.
    backend : ``'plotly'`` | ``'matplotlib'``
        Rendering back-end.
    """

    BACKENDS = ('plotly', 'matplotlib')

    def __init__(self, controller, backend: str = 'plotly'):
        self.ctrl = controller
        self.model = controller.model
        self.dynamics = controller.dynamics
        self.device = controller.device
        self.set_backend(backend)

    def set_backend(self, backend: str):
        if backend not in self.BACKENDS:
            raise ValueError(f"backend must be one of {self.BACKENDS}")
        self._backend = backend

    # ------------------------------------------------------------------ #
    #  Value-field evaluation
    # ------------------------------------------------------------------ #

    def evaluate_brat_grid(self, current_state: np.ndarray,
                          time_query: float,
                          grid_bounds: list | None = None,
                          resolution: int = 50) -> tuple:
        """Evaluate V over a (x, y, z) grid.

        Non-spatial states are taken from *current_state* so that the blob
        reflects the BRAT conditioned on the chaser's current velocity,
        attitude and angular velocity.

        Parameters
        ----------
        current_state : (13,) array
        time_query    : scalar time for V query.
        grid_bounds   : [(xmin,xmax), (ymin,ymax), (zmin,zmax)] or *None*
                        (defaults to state_test_range for position).
        resolution    : grid points per axis.

        Returns
        -------
        X, Y, Z : (res, res, res) meshgrid arrays.
        V       : (res, res, res) value field.
        """
        if grid_bounds is None:
            # Auto-centre on chaser position ±8 m for better resolution
            half = 8.0
            clamp = 15.0
            pos = np.asarray(current_state[:3], dtype=np.float64)
            lo = np.clip(pos - half, -clamp, clamp)
            hi = np.clip(pos + half, -clamp, clamp)
            grid_bounds = [(lo[0], hi[0]), (lo[1], hi[1]), (lo[2], hi[2])]

        xs = np.linspace(grid_bounds[0][0], grid_bounds[0][1], resolution)
        ys = np.linspace(grid_bounds[1][0], grid_bounds[1][1], resolution)
        zs = np.linspace(grid_bounds[2][0], grid_bounds[2][1], resolution)
        X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')

        N = resolution ** 3
        states = np.tile(np.asarray(current_state, dtype=np.float32), (N, 1))
        states[:, 0] = X.ravel()
        states[:, 1] = Y.ravel()
        states[:, 2] = Z.ravel()

        states_t = torch.tensor(states, dtype=torch.float32,
                                device=self.device)
        time_col = torch.full((N, 1), time_query, dtype=torch.float32,
                              device=self.device)
        coords = torch.cat([time_col, states_t], dim=-1)
        model_input = self.dynamics.coord_to_input(coords)

        with torch.no_grad():
            result = self.model({'coords': model_input})
            output = result['model_out'].squeeze(-1)

        V_flat = self.dynamics.io_to_value(model_input, output).cpu().numpy()
        V = V_flat.reshape(resolution, resolution, resolution)
        return X, Y, Z, V

    # ------------------------------------------------------------------ #
    #  2D slice evaluation (for val-plot GIFs)
    # ------------------------------------------------------------------ #

    def evaluate_brat_grid_2d(self, current_state: np.ndarray,
                             time_query: float,
                             plane: str = 'xy',
                             grid_bounds_2d: list | None = None,
                             resolution: int = 100) -> tuple:
        """Evaluate V over a 2D position slice.

        Parameters
        ----------
        current_state : (13,) array — non-varied states taken from here.
        time_query    : scalar time for V query.
        plane         : ``'xy'``, ``'xz'``, or ``'yz'``.
        grid_bounds_2d: [(a_min,a_max), (b_min,b_max)] or *None*.
        resolution    : grid points per axis.

        Returns
        -------
        A, B : (res, res) meshgrid arrays for the two in-plane axes.
        V    : (res, res) value field.
        dVda, dVdb : (res, res) spatial gradients in the slice plane.
        """
        axis_map = {'xy': (0, 1, 2), 'xz': (0, 2, 1), 'yz': (1, 2, 0)}
        if plane not in axis_map:
            raise ValueError(f"plane must be one of {list(axis_map.keys())}")
        idx_a, idx_b, idx_fixed = axis_map[plane]

        if grid_bounds_2d is None:
            half, clamp = 8.0, 15.0
            pos = np.asarray(current_state[:3], dtype=np.float64)
            lo_a = np.clip(pos[idx_a] - half, -clamp, clamp)
            hi_a = np.clip(pos[idx_a] + half, -clamp, clamp)
            lo_b = np.clip(pos[idx_b] - half, -clamp, clamp)
            hi_b = np.clip(pos[idx_b] + half, -clamp, clamp)
            grid_bounds_2d = [(lo_a, hi_a), (lo_b, hi_b)]

        a_vals = np.linspace(grid_bounds_2d[0][0], grid_bounds_2d[0][1], resolution)
        b_vals = np.linspace(grid_bounds_2d[1][0], grid_bounds_2d[1][1], resolution)
        A, B = np.meshgrid(a_vals, b_vals, indexing='ij')

        N = resolution * resolution
        states = np.tile(np.asarray(current_state, dtype=np.float32), (N, 1))
        states[:, idx_a] = A.ravel()
        states[:, idx_b] = B.ravel()

        states_t = torch.tensor(states, dtype=torch.float32,
                                device=self.device).requires_grad_(True)
        time_col = torch.full((N, 1), time_query, dtype=torch.float32,
                              device=self.device)
        coords = torch.cat([time_col, states_t], dim=-1)
        model_input = self.dynamics.coord_to_input(coords)
        model_input.requires_grad_(True)

        result = self.model({'coords': model_input})
        output = result['model_out'].squeeze(-1)
        V_raw = self.dynamics.io_to_value(model_input, output)

        # Compute gradients w.r.t. the spatial state columns
        grad_outputs = torch.ones_like(V_raw)
        grads = torch.autograd.grad(V_raw, states_t, grad_outputs=grad_outputs,
                                    create_graph=False)[0]
        dVda = grads[:, idx_a].detach().cpu().numpy().reshape(resolution, resolution)
        dVdb = grads[:, idx_b].detach().cpu().numpy().reshape(resolution, resolution)

        V = V_raw.detach().cpu().numpy().reshape(resolution, resolution)
        return A, B, V, dVda, dVdb

    # ------------------------------------------------------------------ #
    #  Generic 2D slice evaluation (arbitrary state indices)
    # ------------------------------------------------------------------ #

    def evaluate_brat_grid_2d_generic(
        self,
        base_state: np.ndarray,
        time_query: float,
        idx_a: int,
        idx_b: int,
        grid_bounds_2d: list,
        resolution: int = 100,
    ) -> tuple:
        """Evaluate V over a 2D slice at arbitrary state indices.

        Parameters
        ----------
        base_state    : (13,) array — all states start from here.
        time_query    : scalar time for V query.
        idx_a, idx_b  : state indices to sweep (0-12).
        grid_bounds_2d: [(a_min, a_max), (b_min, b_max)].
        resolution    : grid points per axis.

        Returns
        -------
        A, B  : (res, res) meshgrid arrays.
        V     : (res, res) value field.
        dVda, dVdb : (res, res) gradients w.r.t. the swept axes.
        """
        a_vals = np.linspace(grid_bounds_2d[0][0], grid_bounds_2d[0][1],
                             resolution)
        b_vals = np.linspace(grid_bounds_2d[1][0], grid_bounds_2d[1][1],
                             resolution)
        A, B = np.meshgrid(a_vals, b_vals, indexing='ij')

        N = resolution * resolution
        states = np.tile(np.asarray(base_state, dtype=np.float32), (N, 1))
        states[:, idx_a] = A.ravel()
        states[:, idx_b] = B.ravel()

        states_t = torch.tensor(states, dtype=torch.float32,
                                device=self.device).requires_grad_(True)
        time_col = torch.full((N, 1), time_query, dtype=torch.float32,
                              device=self.device)
        coords = torch.cat([time_col, states_t], dim=-1)
        model_input = self.dynamics.coord_to_input(coords)
        model_input.requires_grad_(True)

        result = self.model({'coords': model_input})
        output = result['model_out'].squeeze(-1)
        V_raw = self.dynamics.io_to_value(model_input, output)

        grad_outputs = torch.ones_like(V_raw)
        grads = torch.autograd.grad(V_raw, states_t,
                                    grad_outputs=grad_outputs,
                                    create_graph=False)[0]
        dVda = grads[:, idx_a].detach().cpu().numpy().reshape(resolution,
                                                               resolution)
        dVdb = grads[:, idx_b].detach().cpu().numpy().reshape(resolution,
                                                               resolution)
        V = V_raw.detach().cpu().numpy().reshape(resolution, resolution)
        return A, B, V, dVda, dVdb

    # ------------------------------------------------------------------ #
    #  3D gradient evaluation (for Cone arrows in HTML animation)
    # ------------------------------------------------------------------ #

    def evaluate_brat_gradients_3d(self, current_state: np.ndarray,
                                  time_query: float,
                                  grid_bounds: list | None = None,
                                  resolution: int = 12) -> tuple:
        """Evaluate V and dV/d(x,y,z) on a coarse 3D grid for arrow overlay.

        Parameters
        ----------
        current_state : (13,) array.
        time_query    : scalar time for V query.
        grid_bounds   : [(xmin,xmax), (ymin,ymax), (zmin,zmax)].
        resolution    : grid points per axis (keep low, e.g. 8-12).

        Returns
        -------
        X, Y, Z : (res, res, res) meshgrid arrays.
        V       : (res, res, res) value field.
        dVdx, dVdy, dVdz : (res, res, res) spatial gradients.
        """
        if grid_bounds is None:
            half, clamp = 8.0, 15.0
            pos = np.asarray(current_state[:3], dtype=np.float64)
            lo = np.clip(pos - half, -clamp, clamp)
            hi = np.clip(pos + half, -clamp, clamp)
            grid_bounds = [(lo[0], hi[0]), (lo[1], hi[1]), (lo[2], hi[2])]

        xs = np.linspace(grid_bounds[0][0], grid_bounds[0][1], resolution)
        ys = np.linspace(grid_bounds[1][0], grid_bounds[1][1], resolution)
        zs = np.linspace(grid_bounds[2][0], grid_bounds[2][1], resolution)
        X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')

        N = resolution ** 3
        states = np.tile(np.asarray(current_state, dtype=np.float32), (N, 1))
        states[:, 0] = X.ravel()
        states[:, 1] = Y.ravel()
        states[:, 2] = Z.ravel()

        states_t = torch.tensor(states, dtype=torch.float32,
                                device=self.device).requires_grad_(True)
        time_col = torch.full((N, 1), time_query, dtype=torch.float32,
                              device=self.device)
        coords = torch.cat([time_col, states_t], dim=-1)
        model_input = self.dynamics.coord_to_input(coords)
        model_input.requires_grad_(True)

        result = self.model({'coords': model_input})
        output = result['model_out'].squeeze(-1)
        V_raw = self.dynamics.io_to_value(model_input, output)

        grad_outputs = torch.ones_like(V_raw)
        grads = torch.autograd.grad(V_raw, states_t, grad_outputs=grad_outputs,
                                    create_graph=False)[0]

        shape = (resolution, resolution, resolution)
        dVdx = grads[:, 0].detach().cpu().numpy().reshape(shape)
        dVdy = grads[:, 1].detach().cpu().numpy().reshape(shape)
        dVdz = grads[:, 2].detach().cpu().numpy().reshape(shape)
        V = V_raw.detach().cpu().numpy().reshape(shape)
        return X, Y, Z, V, dVdx, dVdy, dVdz

    # ------------------------------------------------------------------ #
    #  Plotly rendering
    # ------------------------------------------------------------------ #

    def render_plotly(self, X, Y, Z, V, chaser_state,
                      trajectory=None, title: str = '',
                      axis_range: dict | None = None,
                      safety_active=None):
        """Build a Plotly ``Figure`` with BRAT blob + spacecraft geometry.

        Always emits a **fixed** number and order of traces so that the
        visibility-toggle animation in :class:`ControllerAnimation13D`
        works correctly:

             0. Isosurface  (BRAT V=0) — or scatter fallback
             1. Scatter3d   (value gradient point cloud)
             2. Mesh3d      (target solid body)
             3. Scatter3d   (target wireframe edges)
             4. Mesh3d      (dock hemisphere)
             5. Scatter3d   (dock rim circle)
             6. Scatter3d   (goal ghost wireframe)
             7. Surface     (reach tolerance sphere)
             8. Mesh3d      (chaser cube — face coloured)
             9. Scatter3d   (chaser wireframe edges)
            10. Cone        (chaser nose arrow)
            11. Cone        (chaser up arrow)
            12. Scatter3d   (trajectory)

        Parameters
        ----------
        X, Y, Z, V : arrays from :meth:`evaluate_brat_grid`.
        chaser_state : (13,) current chaser state.
        trajectory   : (T, 13) or *None* — path to draw.
        title        : optional figure title.

        Returns
        -------
        plotly.graph_objects.Figure
        """
        import plotly.graph_objects as go

        fig = go.Figure()

        has_brat = bool(V.min() <= 0)

        # --- Trace 0: BRAT isosurface (V=0) or invisible placeholder --- #
        if has_brat:
            fig.add_trace(go.Isosurface(
                x=X.ravel(), y=Y.ravel(), z=Z.ravel(),
                value=V.ravel(),
                isomin=-0.02, isomax=0.02,
                surface_count=2,
                opacity=0.30,
                colorscale=[[0, 'rgba(50,120,255,0.35)'],
                            [1, 'rgba(50,120,255,0.35)']],
                showscale=False,
                caps=dict(x_show=False, y_show=False, z_show=False),
                name='BRAT (V=0)',
            ))
        else:
            # No BRAT region — emit an invisible placeholder so trace
            # indices stay consistent (trace 1 already shows the gradient).
            fig.add_trace(go.Scatter3d(
                x=[], y=[], z=[],
                mode='markers',
                marker=dict(size=1),
                name='BRAT (V=0)',
                visible=False,
            ))

        # --- Trace 1: Value gradient scatter cloud ---------------------- #
        flat_v = V.ravel()
        xf, yf, zf = X.ravel(), Y.ravel(), Z.ravel()
        # Subsample to ~2500 points for performance
        n_pts = len(flat_v)
        max_pts = 2500
        if n_pts > max_pts:
            idx_sub = np.linspace(0, n_pts - 1, max_pts, dtype=int)
            xf, yf, zf, flat_v = xf[idx_sub], yf[idx_sub], zf[idx_sub], flat_v[idx_sub]
        fig.add_trace(go.Scatter3d(
            x=xf, y=yf, z=zf,
            mode='markers',
            marker=dict(
                size=3,
                color=flat_v,
                colorscale='RdYlGn_r',
                opacity=0.50,
                cmin=float(V.min()),
                cmax=float(V.max()),
                colorbar=dict(title='V', x=1.05),
            ),
            name='Value field',
        ))

        # --- Traces 2-4: Target spacecraft ----------------------------- #
        self._add_target_plotly(fig)

        # --- Trace 5: Dock rim circle --------------------------------- #
        add_dock_rim_plotly(fig, self.dynamics)

        # --- Trace 6: Goal ghost wireframe ---------------------------- #
        add_goal_ghost_plotly(fig, self.dynamics)

        # --- Trace 7: Reach tolerance sphere -------------------------- #
        self._add_reach_sphere_plotly(fig)

        # --- Trace 8: Chaser cube (oriented, face-coloured) ----------- #
        sf_now = False
        if safety_active is not None and trajectory is not None and len(trajectory) > 0:
            sf_now = bool(safety_active[len(trajectory) - 1])
        self._add_chaser_plotly(fig, chaser_state, safety_active=sf_now)

        # --- Trace 9: Chaser wireframe edges -------------------------- #
        add_chaser_wireframe_plotly(fig, self.dynamics, chaser_state)

        # --- Traces 10-11: Chaser orientation arrows ------------------- #
        add_chaser_axes_plotly(fig, self.dynamics, chaser_state)

        # --- Trace 12: Trajectory (always present, may be empty) ------- #
        if trajectory is not None and len(trajectory) > 1:
            if safety_active is not None and len(safety_active) >= len(trajectory):
                color_vals = safety_active[:len(trajectory)].astype(float)
                line_kwargs = dict(
                    color=color_vals,
                    colorscale=[[0, 'cyan'], [1, 'red']],
                    cmin=0, cmax=1,
                    width=4,
                )
            else:
                line_kwargs = dict(color='cyan', width=4)
            fig.add_trace(go.Scatter3d(
                x=trajectory[:, 0], y=trajectory[:, 1], z=trajectory[:, 2],
                mode='lines',
                line=line_kwargs,
                name='Trajectory',
            ))
        else:
            fig.add_trace(go.Scatter3d(
                x=[], y=[], z=[],
                mode='lines',
                line=dict(color='cyan', width=4),
                name='Trajectory',
                visible=False,
            ))

        scene_kwargs = dict(
            xaxis_title='x (m)', yaxis_title='y (m)', zaxis_title='z (m)',
            aspectmode='data',
        )
        if axis_range is not None:
            if 'xlim' in axis_range:
                scene_kwargs['xaxis'] = dict(range=list(axis_range['xlim']))
            if 'ylim' in axis_range:
                scene_kwargs['yaxis'] = dict(range=list(axis_range['ylim']))
            if 'zlim' in axis_range:
                scene_kwargs['zaxis'] = dict(range=list(axis_range['zlim']))
        fig.update_layout(
            title=title,
            scene=scene_kwargs,
            legend=dict(x=0.01, y=0.99),
            margin=dict(l=0, r=0, t=40, b=0),
        )
        return fig

    # --- Plotly helper geometry ---------------------------------------- #

    def _add_target_plotly(self, fig):
        """Add target body (solid Mesh3d + wireframe + dock) as 3 traces."""
        import plotly.graph_objects as go

        d = self.dynamics
        hw, hh, hd = d.w_t / 2, d.h_t, d.d_t / 2

        # 8 corners of the box (y goes from 0 to h_t)
        cx = [-hw, hw, hw, -hw, -hw, hw, hw, -hw]
        cy = [0, 0, hh, hh, 0, 0, hh, hh]
        cz = [-hd, -hd, -hd, -hd, hd, hd, hd, hd]

        # Trace 2: Solid Mesh3d target body
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
            color='silver', opacity=0.55,
            flatshading=True,
            name='Target body',
        ))

        # Trace 3: Thin wireframe overlay for edge visibility
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
            line=dict(color='gray', width=2),
            name='Target edges', showlegend=False,
        ))

        # Trace 4: Docking post
        pts = _indent_box_mesh(d.post_hw_x, d.post_hw_z,
                               d.post_length)
        fig.add_trace(go.Mesh3d(
            x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
            opacity=0.45, color='limegreen',
            alphahull=0,
            name='Docking post',
        ))

    def _add_reach_sphere_plotly(self, fig):
        import plotly.graph_objects as go

        r = self.dynamics.eps_p
        u = np.linspace(0, 2*np.pi, 20)
        v = np.linspace(0, np.pi, 20)
        xs = r * np.outer(np.cos(u), np.sin(v))
        ys = r * np.outer(np.sin(u), np.sin(v))
        zs = r * np.outer(np.ones_like(u), np.cos(v))
        fig.add_trace(go.Surface(
            x=xs, y=ys, z=zs,
            opacity=0.20, colorscale=[[0, 'lime'], [1, 'lime']],
            showscale=False, name='Reach tol.',
        ))

    def _add_chaser_plotly(self, fig, state, safety_active=False):
        import plotly.graph_objects as go

        d = self.dynamics
        pos = state[:3]
        q = state[6:10] / (np.linalg.norm(state[6:10]) + 1e-12)
        verts, _ = _oriented_box_mesh(
            pos, (d.w_c / 2, d.h_c / 2, d.d_c / 2), quaternion=q)

        # Triangulate quads for Mesh3d (each quad → 2 triangles)
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

        # Per-face coloring: dock face (body +x, quad 5 → tri 10-11) = blue,
        # top face (body +z, quad 1 → tri 2-3) = green,
        # rest = orange (nominal) or red (safety filter active).
        body_color = 'red' if safety_active else 'orange'
        facecolor = [body_color] * 12
        facecolor[10] = 'dodgerblue'   # body +x (dock face) tri A
        facecolor[11] = 'dodgerblue'   # body +x (dock face) tri B
        facecolor[2]  = 'limegreen'    # body +z (top face) tri A
        facecolor[3]  = 'limegreen'    # body +z (top face) tri B

        fig.add_trace(go.Mesh3d(
            x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
            i=i_list, j=j_list, k=k_list,
            opacity=1.0, facecolor=facecolor,
            flatshading=True,
            name='Chaser',
        ))

    # ------------------------------------------------------------------ #
    #  Matplotlib rendering
    # ------------------------------------------------------------------ #

    def render_matplotlib(self, X, Y, Z, V, chaser_state,
                          trajectory=None, title: str = '', ax=None,
                          axis_limits: dict | None = None,
                          safety_active=None):
        """Render BRAT blob with matplotlib 3D.

        Uses marching cubes to extract the V=0 isosurface and colour-maps
        the faces by interpolated value for a gradient visualisation.
        Falls back to a scatter plot of V<0 points when the isosurface
        cannot be extracted.

        Parameters
        ----------
        ax : optional pre-existing Axes3D.  Created if *None*.
        axis_limits : dict with optional keys ``xlim``, ``ylim``, ``zlim``
            (each a (lo, hi) tuple).  Applied after all drawing to keep
            axes fixed across animation frames.

        Returns
        -------
        matplotlib.figure.Figure  (the *parent* figure of ax).
        """
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        if ax is None:
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')
        else:
            fig = ax.figure

        # --- Extract V=0 isosurface ----------------------------------- #
        brat_drawn = False
        try:
            from skimage.measure import marching_cubes
            verts_mc, faces_mc, _, values_mc = marching_cubes(
                V, level=0.0)
            # Scale verts to real coordinates
            spacing = np.array([
                X[1,0,0] - X[0,0,0],
                Y[0,1,0] - Y[0,0,0],
                Z[0,0,1] - Z[0,0,0],
            ])
            origin = np.array([X[0,0,0], Y[0,0,0], Z[0,0,0]])
            verts_real = verts_mc * spacing + origin

            # Colour faces by mean vertex value (depth inside BRAT)
            import matplotlib.cm as cm
            v_normed = np.clip(
                values_mc / (np.abs(values_mc).max() + 1e-8), -1, 1)
            face_vals = v_normed[faces_mc].mean(axis=1)
            face_colors = cm.RdYlGn_r((face_vals + 1) / 2)
            face_colors[:, 3] = 0.30  # alpha
            mesh = Poly3DCollection(
                verts_real[faces_mc], linewidths=0.1)
            mesh.set_facecolor(face_colors)
            mesh.set_edgecolor('steelblue')
            ax.add_collection3d(mesh)
            brat_drawn = True
        except (ImportError, ValueError):
            pass

        # Scatter fallback when marching cubes fails
        if not brat_drawn:
            mask = V.ravel() < 0
            if np.any(mask):
                import matplotlib.cm as cm
                pts_x = X.ravel()[mask]
                pts_y = Y.ravel()[mask]
                pts_z = Z.ravel()[mask]
                v_neg = V.ravel()[mask]
                c = cm.RdYlGn_r(
                    np.clip(v_neg / (np.abs(v_neg).min() + 1e-8), -1, 0) * -0.5 + 0.5)
                ax.scatter(pts_x, pts_y, pts_z, c=c, s=12, alpha=0.35,
                           label='BRAT (V<0)')
            elif V.min() > 0:
                print("[BRAT Viz] V > 0 everywhere — no BRAT region in this grid.")

        # --- Target body ---------------------------------------------- #
        self._draw_target_mpl(ax)

        # --- Reach tolerance ------------------------------------------ #
        self._draw_reach_sphere_mpl(ax)

        # --- Chaser box + orientation arrow --------------------------- #
        sf_now = False
        if safety_active is not None and trajectory is not None and len(trajectory) > 0:
            sf_now = bool(safety_active[len(trajectory) - 1])
        self._draw_chaser_mpl(ax, chaser_state, safety_active=sf_now)

        # --- Trajectory ----------------------------------------------- #
        if trajectory is not None and len(trajectory) > 1:
            if safety_active is not None and len(safety_active) >= len(trajectory):
                from utils.controllers.trajectory_only_animation_13d import _contiguous_ranges
                sa = safety_active[:len(trajectory)]
                nominal = ~sa
                for seg_s, seg_e in _contiguous_ranges(nominal):
                    s = max(seg_s - 1, 0)
                    ax.plot(trajectory[s:seg_e, 0], trajectory[s:seg_e, 1],
                            trajectory[s:seg_e, 2], 'c-', lw=2)
                for seg_s, seg_e in _contiguous_ranges(sa):
                    s = max(seg_s - 1, 0)
                    ax.plot(trajectory[s:seg_e, 0], trajectory[s:seg_e, 1],
                            trajectory[s:seg_e, 2], 'r-', lw=3)
                ax.plot([], [], 'c-', lw=2, label='Trajectory')
                if sa.any():
                    ax.plot([], [], 'r-', lw=3, label='Safety Filter')
            else:
                ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2],
                        'c-', lw=2, label='Trajectory')

        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.set_zlabel('z (m)')
        if title:
            ax.set_title(title)

        # --- Fixed axis limits ---------------------------------------- #
        if axis_limits is not None:
            if 'xlim' in axis_limits:
                ax.set_xlim3d(*axis_limits['xlim'])
            if 'ylim' in axis_limits:
                ax.set_ylim3d(*axis_limits['ylim'])
            if 'zlim' in axis_limits:
                ax.set_zlim3d(*axis_limits['zlim'])
        ax.set_box_aspect([1, 1, 1])

        ax.legend(loc='upper left', fontsize=8)
        return fig

    # --- Matplotlib helper geometry ----------------------------------- #

    def _draw_target_mpl(self, ax):
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        d = self.dynamics
        hw, hd = d.w_t / 2, d.d_t / 2
        # Target body as solid silver box
        _, faces = _oriented_box_mesh(
            [0, d.h_t / 2, 0], (hw, d.h_t / 2, hd))
        mesh = Poly3DCollection(faces, alpha=0.35, linewidths=0.5)
        mesh.set_facecolor('silver')
        mesh.set_edgecolor('gray')
        ax.add_collection3d(mesh)

        # Docking post wireframe
        hw_x = d.post_hw_x
        hw_z = d.post_hw_z
        length = d.post_length
        for y_frac in [0.0, 0.5, 1.0]:
            yy = -length * y_frac
            rim_x = [-hw_x, hw_x, hw_x, -hw_x, -hw_x]
            rim_z = [-hw_z, -hw_z, hw_z, hw_z, -hw_z]
            ax.plot(rim_x, [yy]*5, rim_z, color='limegreen', alpha=0.5, lw=1.0)
        for px, pz in [(-hw_x, -hw_z), (hw_x, -hw_z), (hw_x, hw_z), (-hw_x, hw_z)]:
            ax.plot([px, px], [0, -length], [pz, pz], color='limegreen', alpha=0.5, lw=1.0)

    def _draw_reach_sphere_mpl(self, ax):
        r = self.dynamics.eps_p
        u = np.linspace(0, 2*np.pi, 15)
        v = np.linspace(0, np.pi, 15)
        xs = r * np.outer(np.cos(u), np.sin(v))
        ys = r * np.outer(np.sin(u), np.sin(v))
        zs = r * np.outer(np.ones_like(u), np.cos(v))
        ax.plot_surface(xs, ys, zs, color='lime', alpha=0.15)

    def _draw_chaser_mpl(self, ax, state, safety_active=False):
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        d = self.dynamics
        pos = state[:3]
        q = state[6:10] / (np.linalg.norm(state[6:10]) + 1e-12)
        _, faces = _oriented_box_mesh(
            pos, (d.w_c / 2, d.h_c / 2, d.d_c / 2), quaternion=q)
        mesh = Poly3DCollection(faces, alpha=0.6, linewidths=0.5)
        mesh.set_facecolor('red' if safety_active else 'orange')
        mesh.set_edgecolor('darkred' if safety_active else 'darkorange')
        ax.add_collection3d(mesh)

        # Orientation arrow: body-frame +x direction ("nose")
        R = _quat_to_R(q)
        nose_dir = R.T @ np.array([1.0, 0.0, 0.0])  # body +x in LVLH
        arrow_len = 1.5
        ax.quiver(pos[0], pos[1], pos[2],
                  nose_dir[0] * arrow_len,
                  nose_dir[1] * arrow_len,
                  nose_dir[2] * arrow_len,
                  color='blue', linewidth=2.0, arrow_length_ratio=0.25)


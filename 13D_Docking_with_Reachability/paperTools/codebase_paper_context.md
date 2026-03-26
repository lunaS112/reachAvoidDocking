# Codebase Paper Context — Reach-Avoid Docking

> Extracted from the `reachAvoidDocking` repository. Every claim includes a source file and line reference. Intended for use by an AI agent drafting the research paper.

---

## 1. Problem Statement

The codebase implements a **learning-based Backward Reach-Avoid Tube (BRAT) framework** for autonomous spacecraft proximity docking. The paper title (from `13D_Docking_with_Reachability/template.tex`) is:

> *"Learning-Based Backward Reach-Avoid Tubes for Safe Spacecraft Proximity Docking in 13 Dimensions"*

**Core problem:** Spacecraft docking demands control policies that jointly guarantee target reachability and collision avoidance under coupled translational–rotational dynamics. The full problem requires a 13-dimensional state space, far exceeding the reach of classical grid-based Hamilton-Jacobi (HJ) methods.

**Approach:** A neural network approximates the HJ value function, trained via a combination of PDE-based (HJI) residual loss and curriculum-aware MPC-generated value labels in a semi-supervised scheme. The resulting controller provides continuous safety guarantees over the planning horizon.

**Validation strategy:** The method is first validated on a 6D planar docking model where a grid-based numerical HJ solution exists as ground truth, then scaled to the full 13D problem which is inaccessible to grid-based methods.

**Baselines compared:** Four controllers in 6D (BRAT, MPC, Terminal MPC, Grid-Based) and three in 13D (BRAT, MPC, Terminal MPC). The grid-based controller serves as numerical ground truth in 6D only.

*(Sources: `13D_Docking_with_Reachability/template.tex` abstract; `README.md`; `run_controller.sh` lines 60–137)*

---

## 2. System Dynamics

### 2.1 Orbital Mechanics Constants

All constants are defined in `dynamics/dynamics.py`.

| Parameter | Symbol | Value | Source |
|-----------|--------|-------|--------|
| Earth gravitational parameter | μ | 3.986004418 × 10¹⁴ m³/s² | `dynamics.py:667` |
| Earth radius | r_E | 6,371 km | `dynamics.py:668` |
| Orbital altitude | h | 400 km | `dynamics.py:559` (6D), `:1058` (13D) |
| Orbital radius | r | r_E + h = 6,771 km | `dynamics.py:669` |
| Mean motion | n | √(μ/r³) ≈ 1.082 × 10⁻³ rad/s | `dynamics.py:665–670` |
| Chaser mass | m_c | 200 kg | `dynamics.py:564` (6D), `:1063` (13D) |
| Chaser dimensions | w_c × h_c × d_c | 1.0 × 1.0 × 1.0 m | `dynamics.py:565–566` (6D), `:1064–1066` (13D) |
| Max force per axis | F̄ (u_bar) | 20 N | `dynamics.py:560` (6D), `:1059` (13D) |
| Max torque per axis | τ̄ (u_theta_bar) | 1.5 N·m | `dynamics.py:561` (6D), `:1060` (13D) |

### 2.2 6D Planar Dynamics (Docking6D)

**State vector** (6 dimensions):

| Index | Symbol | Description | Range | Units |
|-------|--------|-------------|-------|-------|
| 0 | p_x | Relative x-position (LVLH radial) | [−15, 15] | m |
| 1 | p_y | Relative y-position (LVLH along-track) | [−15, 15] | m |
| 2 | v_x | Relative x-velocity | [−1.5, 1.5] | m/s |
| 3 | v_y | Relative y-velocity | [−2.0, 2.0] | m/s |
| 4 | θ | Chaser yaw angle | [−π, π] | rad |
| 5 | ω | Chaser angular velocity | [−1.0, 1.0] | rad/s |

*(Source: `dynamics.py:628–629`)*

**Control vector** (3 dimensions): u = [u_x, u_y, u_θ] where u_x, u_y ∈ [−20, 20] N and u_θ ∈ [−1.5, 1.5] N·m.

*(Source: `dynamics.py:630–631`)*

**CW + rotational equations** (exact code at `dynamics.py:753–776`):

```
ṗ_x = v_x
ṗ_y = v_y
v̇_x = 3n²p_x + 2nv_y + u_x/m_c
v̇_y = −2nv_x + u_y/m_c
θ̇   = ω
ω̇   = u_θ/J_c
```

where J_c = (1/12) × m_c × (w_c² + h_c²) ≈ 33.33 kg·m² (`dynamics.py:672–675`).

**Velocity clamping:** Accelerations are zeroed when the corresponding velocity is at its state-range bound and the acceleration would push it further out. This prevents out-of-distribution states during simulation (`dynamics.py:760–772`).

**Angle wrapping:** θ is wrapped to [−π, π] via modulo arithmetic (`dynamics.py:688–690`). For differentiable integration, atan2(sin θ, cos θ) is used instead (`dynamics.py:713`).

**Periodic transform:** The angle θ is encoded as (sin θ, cos θ) before being fed to the neural network, expanding the 7-dimensional input (time + 6 states) to 8 dimensions (`dynamics.py:717–732`, super().__init__ `input_dim=8` at `:654`).

### 2.3 13D 3D Dynamics (Docking13D)

**State vector** (13 dimensions):

| Index | Symbol | Description | Range | Units |
|-------|--------|-------------|-------|-------|
| 0 | x | Relative x-position (LVLH) | [−10, 10] | m |
| 1 | y | Relative y-position (LVLH) | [−10, 10] | m |
| 2 | z | Relative z-position (LVLH) | [−10, 10] | m |
| 3 | v_x | Relative x-velocity | [−2.0, 2.0] | m/s |
| 4 | v_y | Relative y-velocity | [−2.0, 2.0] | m/s |
| 5 | v_z | Relative z-velocity | [−2.0, 2.0] | m/s |
| 6 | q_0 | Quaternion scalar (LVLH→Body) | [−1, 1] | — |
| 7 | q_1 | Quaternion i-component | [−1, 1] | — |
| 8 | q_2 | Quaternion j-component | [−1, 1] | — |
| 9 | q_3 | Quaternion k-component | [−1, 1] | — |
| 10 | ω_x | Body-frame roll rate (rel. LVLH) | [−1.5, 1.5] | rad/s |
| 11 | ω_y | Body-frame pitch rate | [−1.5, 1.5] | rad/s |
| 12 | ω_z | Body-frame yaw rate | [−1.5, 1.5] | rad/s |

*(Source: `dynamics.py:1156–1170`)*

**Control vector** (6 dimensions): u = [F_x, F_y, F_z, τ_x, τ_y, τ_z] where forces F_i ∈ [−20, 20] N (body-frame) and torques τ_i ∈ [−1.5, 1.5] N·m (body-frame).

*(Source: `dynamics.py:1172–1179`)*

**Extended HCW 3D dynamics** (exact code at `dynamics.py:1480–1561`):

*Translation kinematics:*
```
ẋ = v_x,  ẏ = v_y,  ż = v_z
```

*HCW translational acceleration (LVLH frame):*
```
a_L = (1/m_c) R(q)ᵀ F_b        (body force → LVLH acceleration)
v̇_x = 2nv_y + 3n²x + a_L,x
v̇_y = −2nv_x + a_L,y
v̇_z = −n²z + a_L,z
```

where R(q) = quat_to_R_LVLH_to_body(q) is the 3×3 rotation matrix from the scalar-first quaternion (`dynamics.py:1268–1281`, `:1503–1505`).

*Quaternion kinematics:*
```
q̇ = ½ Ω(ω) q
```

where Ω(ω) is the 4×4 skew-symmetric matrix (`dynamics.py:1531–1539`):
```
Ω = [  0   −ω_x  −ω_y  −ω_z ]
    [ ω_x    0    ω_z  −ω_y ]
    [ ω_y  −ω_z    0    ω_x ]
    [ ω_z   ω_y  −ω_x    0  ]
```

*Rotational dynamics (Euler's equation):*
```
ω̇ = I⁻¹(τ_b − ω × Iω)
```

where I = diag(I_xx, I_yy, I_zz) with I_xx = I_yy = I_zz = (m_c/12)(1² + 1²) ≈ 33.33 kg·m² (`dynamics.py:1237–1244`, `:1542–1546`).

**Quaternion canonicalization:** To resolve the q ↔ −q ambiguity, q₀ ≥ 0 is enforced in the periodic transform via sign multiplication (`dynamics.py:1335–1348`, labeled "Fix #3").

**No periodic transform expansion for 13D:** Unlike 6D, the 13D model feeds raw (canonicalized) state directly to the network, so `input_dim = state_dim + 1 = 14` (`dynamics.py:1207`).

### 2.4 Inertia Tensor

**6D (planar):** J_c = (1/12) m_c (w_c² + h_c²) ≈ 33.33 kg·m² (scalar moment of inertia). *Source: `dynamics.py:672–675`.*

**13D (3D):** Diagonal tensor I = diag(I_xx, I_yy, I_zz) where each = (m/12)(a² + b²) for the relevant pair of box dimensions. Since w_c = h_c = d_c = 1.0 m, all three diagonal elements are identical: ≈ 33.33 kg·m². *Source: `dynamics.py:1237–1244`.*

### 2.5 Chaser Collision Buffer

**6D:** chaser_buffer = √(w_c² + h_c²)/2 = √2/2 ≈ 0.707 m (diagonal half-extent of the 2D square). *Source: `dynamics.py:567`.*

**13D:** chaser_buffer = √(w_c² + h_c² + d_c²)/2 = √3/2 ≈ 0.866 m (3D bounding-sphere radius). *Source: `dynamics.py:1068`.*

---

## 3. Goal Set & Failure Set

### 3.1 Target Geometry

| Component | Parameter | 6D Value | 13D Value | Source |
|-----------|-----------|----------|-----------|--------|
| Target body width (x) | w_t | 6 m | 6 m | `dynamics.py:586`, `:1096` |
| Target body height (y) | h_t | 3 m | 3 m | `dynamics.py:587`, `:1097` |
| Target body depth (z) | d_t | N/A | 3 m | `dynamics.py:1098` |
| Docking post half-width (x) | post_hw_x | 0.6 m | 0.6 m | `dynamics.py:591`, `:1104` |
| Docking post half-width (z) | post_hw_z | N/A | 0.6 m | `dynamics.py:1105` |
| Post length (−y protrusion) | post_length | 0.2 m | 0.2 m | `dynamics.py:592`, `:1106` |

The target body occupies y ∈ [0, h_t] with |x| ≤ w_t/2 (and |z| ≤ d_t/2 in 3D). The docking post is a rectangular peg protruding from the target's −y face at y ∈ [−post_length, 0], with half-widths post_hw_x (and post_hw_z in 3D).

### 3.2 Goal Set (Reach Function ℓ(x))

The reach function ℓ(x) is a shaped signed-distance function: **negative inside the goal, positive outside**.

#### 6D Goal Set

**Docking tolerances** (Source: `dynamics.py:574–577`):

| Tolerance | Symbol | Value | Description |
|-----------|--------|-------|-------------|
| Position | ε_p | 0.1 m | |x| ≤ ε_p required |
| Velocity | ε_v | 0.1 m/s | √(v_x² + v_y²) ≤ ε_v |
| Angle | ε_θ | 0.04 rad ≈ 2.3° | |θ − θ_goal| ≤ ε_θ |
| Angular velocity | ε_ω | 0.05 rad/s ≈ 2.9°/s | |ω − ω_goal| ≤ ε_ω |

**Goal state:** [0, y_goal_center, 0, 0, π/2, 0] where y_goal_center ≈ −1.05 m (midpoint of goal band). *Source: `dynamics.py:605`.*

**Goal band in y:** y ∈ [goal_y_min, goal_y_max], computed as:
- goal_y_max = −(post_length + chaser_buffer + goal_clearance) where goal_clearance = 0.143 m
- goal_y_min = goal_y_max − goal_band_height where goal_band_height = 0.2 m

*(Source: `dynamics.py:595–599`)*

**Reach function implementation** (`dynamics.py:779–819`):

```python
x_dist  = |p_x| − ε_p
y_dist  = max(goal_y_min − p_y, p_y − goal_y_max)
pos_dist = max(x_dist, y_dist)
vel_dist = √(v_x² + v_y² + 1e-8) − ε_v
θ_dist   = |atan2(sin(θ − θ_goal), cos(θ − θ_goal))| − ε_θ
ω_dist   = |ω − ω_goal| − ε_ω
```

Each component undergoes **piecewise shaping**: linear scaling inside the goal (to ensure gradient visibility), tanh saturation outside:

| Component | Inner scale (×) | Outer tanh scale (×) |
|-----------|-----------------|---------------------|
| pos_dist | 20 | 0.5 |
| vel_dist | 20 | 1.0 |
| θ_dist | 150 | 1.0 |
| ω_dist | 30 | 1.0 |

Final: ℓ(x) = 1.2 × max(pos_dist, vel_dist, θ_dist, ω_dist)

#### 13D Goal Set

**Docking tolerances** (Source: `dynamics.py:1074–1090`):

| Tolerance | Symbol | Value | Description |
|-----------|--------|-------|-------------|
| Lateral position (XZ) | ε_p | 0.10 m | √(x² + z²) ≤ ε_p |
| Lateral velocity (XZ) | ε_v_lat | 0.02 m/s | √(v_x² + v_z²) ≤ ε_v_lat |
| Axial velocity (Y) low | ε_v_ax_lo | 0.03 m/s | Minimum closing speed |
| Axial velocity (Y) high | ε_v_ax_hi | 0.10 m/s | Maximum closing speed |
| Quaternion angle error | ε_q | 0.151 rad ≈ 8.7° | 2·atan2(‖q_vec‖, |q₀|) ≤ ε_q |
| Pitch/yaw rate (ω_y, ω_z) | ε_ω_py | 0.00698 rad/s ≈ 0.4°/s | √(ω_y² + ω_z²) ≤ ε_ω_py |
| Roll rate (ω_x) | ε_ω_r | 0.00698 rad/s ≈ 0.4°/s | |ω_x| ≤ ε_ω_r |

**Goal quaternion:** q_goal = [cos(π/4), 0, 0, sin(π/4)] ≈ [0.7071, 0, 0, 0.7071] (90° yaw so chaser body −Y faces target +Y). *Source: `dynamics.py:1124–1126`.*

**Goal band in y:** Uses orientation-dependent worst-case chaser extent:
- cb_y_worst = 0.5 × (cos(ε_q) + √2 × sin(ε_q)) ≈ 0.601 m (`dynamics.py:1114–1115`)
- goal_y_max = −(post_length + cb_y_worst + goal_clearance) where goal_clearance = 0.07 m
- goal_y_min = goal_y_max − 0.4 m
- goal_y_center ≈ −1.071 m

*(Source: `dynamics.py:1116–1120`)*

**Axial velocity band:** The chaser must be approaching (positive v_y in LVLH) at a controlled speed: v_y ∈ [ε_v_ax_lo, ε_v_ax_hi] = [0.03, 0.10] m/s. *Source: `dynamics.py:1596–1600`.*

**Reach function implementation** (`dynamics.py:1564–1625`):

Each component's inner scale is chosen so the goal center reaches depth ≈ −3 (comment at `:1614`):

| Component | Inner scale (×) | Outer tanh scale (×) |
|-----------|-----------------|---------------------|
| pos_dist | 30 | 0.5 |
| vlat_dist | 150 | 1.0 |
| vax_dist | 86 | 1.0 |
| att_dist | 20 | 1.0 |
| ω_py_dist | 430 | 1.0 |
| ω_r_dist | 430 | 1.0 |

Final: ℓ(x) = reach_fn_weight × max(pos, vlat, vax, att, ω_py, ω_r) where reach_fn_weight = 1.0 (`dynamics.py:1141`).

### 3.3 Failure Set (Avoid Function g(x))

The avoid function g(x) is a shaped signed-distance function: **negative inside the obstacle (unsafe), positive outside (safe)**.

#### 6D Avoid Set

(`dynamics.py:821–852`)

Two inflated rectangles form the obstacle:

1. **Target body:** y ∈ [0, h_t], |x| ≤ w_t/2, inflated by chaser_buffer on all sides.
2. **Docking post:** y ∈ [−post_length, 0], |x| ≤ post_hw_x, inflated by chaser_buffer.

```python
s_body = max(|p_x| − (w_t/2 + cb), max(−(p_y + cb), p_y − (h_t + cb)))
s_post = max(|p_x| − (post_hw_x + cb), max(−(p_y + post_length + cb), p_y − cb))
s_fail = min(s_body, s_post)           # union: deeper one wins
s_fail = s_fail < 0 ? s_fail × 1.5 : s_fail   # asymmetric shaping
g(x) = s_fail
```

#### 13D Avoid Set

(`dynamics.py:1627–1702`)

Extended to 3D rectangular prisms with **orientation-dependent Y-inflation**:

**cb_y(q)** — the support function of the rotated chaser cube projected onto LVLH-Y (`dynamics.py:1628–1644`):
```
R = quat_to_R_LVLH_to_body(q)
cb_y = 0.5 × (|R[0,1]| + |R[1,1]| + |R[2,1]|)
```
Range: 0.5 m (perfectly aligned) to 0.866 m (worst-case full bounding sphere).

This allows the goal set to sit closer to the post tip when the chaser is properly aligned for docking.

```python
s_body = max(|x| − (w_t/2 + cb), max(max(−(y + cb_y), y − (h_t + cb)), |z| − (d_t/2 + cb)))
s_post = max(|x| − (post_hw_x + cb), max(max(−(y + post_length + cb_y), y − cb_y), |z| − (post_hw_z + cb)))
s_fail = min(s_body, s_post)
```

Piecewise shaping for reach_avoid mode: `s_fail < 0 → s_fail × 5.0; else → tanh(s_fail × 5.0)`.

Final: g(x) = avoid_fn_weight × s_fail where avoid_fn_weight = 0.5 (`dynamics.py:1142`).

### 3.4 Boundary Function

For reach_avoid mode (`dynamics.py:890–892`, `:1704–1706`):

```
boundary_fn(x) = max(ℓ(x), −g(x))
```

- Inside goal and outside obstacle: boundary_fn < 0 (safe, reached)
- Inside obstacle: boundary_fn > 0 (unsafe, −g(x) > 0 dominates)
- Outside both: boundary_fn > 0 (not yet reached, ℓ(x) > 0 dominates)

### 3.5 Oriented Collision Check

At runtime, the SDF-based avoid function is supplemented by an explicit **8-corner oriented box collision check** for binary collision detection:

- **6D:** 4 corners of the 2D chaser rectangle, rotated by θ, checked against target body + post geometry (`dynamics.py:854–888`).
- **13D:** 8 corners of the 3D chaser cube, rotated by quaternion q via R(q)ᵀ, checked against 3D target geometry (`dynamics.py:1850–1883`).

---

## 4. Reachability Formulation

### 4.1 Value Function Definition

The value function V(x, t) encodes whether state x can reach the goal while avoiding obstacles within time horizon t:

- V(x, t) ≤ 0 ⟹ state x is inside the BRAT (can safely dock within time t)
- V(x, t) > 0 ⟹ state x is outside the BRAT

### 4.2 HJI PDE

The value function satisfies the Hamilton-Jacobi-Isaacs PDE:

```
∂V/∂t + H(x, ∇_x V) = 0
```

where the Hamiltonian H is computed as (`dynamics.py:984–991` for 6D, `:1788–1793` for 13D):

```
H(x, ∇V) = ∇V · f(x, u*(x, ∇V))
```

with optimal control u* chosen to minimize V (reach-avoid) or maximize V (avoid-only).

### 4.3 Cost Function (Reach-Avoid)

(`dynamics.py:974–982` for 6D, `:1712–1722` for 13D)

For a trajectory x(·), the reach-avoid cost is:

```
J(x(·)) = min_t max{ℓ(x(t)), max_{k≤t}{−g(x(k))}}
```

In code:
```python
reach_values = reach_fn(state_traj)       # ℓ(x(t)) for all t
avoid_values = avoid_fn(state_traj)       # g(x(t)) for all t
cost = min_t[ clamp(reach_values[t], min=max_k≤t{−avoid_values[k]}) ]
```

### 4.4 Optimal Control (Bang-Bang)

**6D** (`dynamics.py:993–1009`):
```
u_x = −F̄ · sign(∂V/∂v_x)       (minimize V)
u_y = −F̄ · sign(∂V/∂v_y)
u_θ = −τ̄ · sign(∂V/∂ω)
```

**13D** (`dynamics.py:1796–1835`):

The body-frame force allocation accounts for the chaser's orientation:
```
R = quat_to_R_LVLH_to_body(q)
coeff_body = (1/m_c) R · (∂V/∂v_LVLH)
coeff_tau  = I⁻ᵀ · (∂V/∂ω)

F_i = −F̄ · sign(coeff_body_i)    for i ∈ {x,y,z}
τ_i = −τ̄ · sign(coeff_tau_i)     for i ∈ {x,y,z}
```

The sign convention reverses for avoid-only mode (maximize V instead of minimize).

### 4.5 BRAT Loss Functions

Two loss function variants are implemented in `utils/losses.py`:

**BRT loss** (`init_brt_hjivi_loss`, `losses.py:7–70`) — for avoid-only problems.

**BRAT loss** (`init_brat_hjivi_loss`, `losses.py:73–125`) — for reach-avoid problems.

Both share the same three-component structure:

```
L_total = w_dirichlet · L_dirichlet + w_pde · L_pde + w_mpc · L_mpc
```

**1. Dirichlet boundary loss** (`losses.py:14–15`, `:80–81`):
```
L_dirichlet = Σ |V(x, t_min) − boundary_fn(x)| / dirichlet_loss_divisor
```
For the `exact` model variant, the Dirichlet loss is zero since the boundary condition is exactly encoded by construction (`losses.py:11–12`).

**2. PDE constraint loss:**

For BRT (`losses.py:55–62`):
```
diff_constraint = ∂V/∂t − H(x, ∇V)
```
With `minWith='target'`: `diff_constraint = max(∂V/∂t − H, V − boundary_fn(x))`

For BRAT (`losses.py:115–118`):
```
diff_constraint = min(max(∂V/∂t − H, V − ℓ(x)), V + g(x))
```
This enforces both the reach constraint (V ≤ ℓ) and the avoid constraint (V ≥ −g).

**3. MPC supervision loss** (`losses.py:16–27` for L1, `:30–42` for L2):
```
L_mpc = Σ |V_learned(x,t) − V_mpc(x,t)|    (L1 norm)
```
With **critical region penalty**: False positives (MPC says unsafe but network says safe) receive an additional penalty scaled by MPC_finetune_lambda:
```
penalty = λ · |V_learned − V_mpc| · |V_learned|    on misclassified points
```
*(Source: `losses.py:24–27`, `:86–89`)*

**Pretraining phase** (`losses.py:47–53`, `:103–109`): When all samples have Dirichlet mask (pretraining), PDE loss is zeroed and MPC loss is added at 0.3× weight.

---

## 5. DeepReach + MPC Integration

### 5.1 Neural Network Architecture

**File:** `utils/modules.py`

**Network class:** `SingleBVPNet` (`modules.py:105–128`)

| Parameter | Value | Source |
|-----------|-------|--------|
| Architecture | Fully connected (MLP) | `modules.py:113–114` |
| Activation | SIREN (sin(30x)) | `modules.py:26–32`, `:56` |
| Hidden layers | 3 (default, configurable) | `modules.py:109` |
| Hidden units | 512 (for docking; default 256) | `run_experiment.sh:63` |
| Output | 1 (scalar value function) | `modules.py:108` |
| Outermost layer | Linear (no activation) | `modules.py:114` |
| Input dim (6D) | 8 (time + 6 states + sin/cos θ expansion) | `dynamics.py:654` |
| Input dim (13D) | 14 (time + 13 states, no expansion) | `dynamics.py:1207` |

**SIREN initialization** (`modules.py:56`):
- First layer: uniform(−1/n_in, 1/n_in) (`modules.py:245–248`)
- Hidden layers: uniform(−√(6/n_in)/30, √(6/n_in)/30) (`modules.py:239–240`)

**Forward pass** (`modules.py:119–128`):
1. Clone input coordinates with `requires_grad_(True)` for autograd
2. Apply `periodic_transform_fn` (sin/cos expansion for 6D angles, quaternion canonicalization for 13D)
3. Pass through FCBlock
4. Return dict with `model_in` (original coords) and `model_out` (network scalar output)

### 5.2 Value Function Parameterization

Three model variants are implemented (`dynamics.py:88–100`):

| Variant | Formula | Description |
|---------|---------|-------------|
| `vanilla` | V = NN(x,t) · σ_v/σ_n + μ_v | Pure NN output |
| `diff` | V = NN(x,t) · σ_v/σ_n + boundary_fn(x) | Learns residual from boundary |
| **`exact`** | V = NN(x,t) · **t** · σ_v/σ_n + boundary_fn(x) | **Used for docking** |

The **`exact` parameterization** is the key contribution ("exact-initial-condition value parameterization" from the abstract). By multiplying the NN output by t, the value function exactly satisfies V(x, 0) = boundary_fn(x) regardless of the network weights. This eliminates the Dirichlet loss entirely.

**Normalization constants:**

| Parameter | 6D | 13D | Source |
|-----------|-----|------|--------|
| value_mean | 0.5 | 0.5 | `dynamics.py:659`, `:1212` |
| value_var | 1.0 | 1.0 | `dynamics.py:660`, `:1213` |
| value_normto | 0.02 | 0.05 | `dynamics.py:661`, `:1217` |
| Effective multiplier | 1/0.02 = 50× | 1/0.05 = 20× | — |

The 13D value_normto was increased from 0.02 to 0.05 ("Fix #2") to reduce the effective output multiplier, making the network less sensitive to individual MPC samples (`dynamics.py:1214–1217`).

### 5.3 Gradient Computation

**File:** `utils/diff_operators.py`

Value function gradients ∂V/∂t and ∂V/∂x are computed via PyTorch autograd through the network (`diff_operators.py:8–21`).

For the `exact` model (`dynamics.py:135–145`):
```
∂V/∂t = (σ_v/σ_n) × (t · ∂NN/∂t + NN)
∂V/∂x = (σ_v/σ_n / σ_state) × ∂NN/∂x · t + ∂boundary_fn/∂x
```

### 5.4 MPC Value Label Generation

**File:** `utils/MPC.py`

The MPC class generates training labels by optimizing control trajectories via perturbation-based shooting:

**Key parameters** (from `run_experiment.sh` and `run_experiment.py`):

| Parameter | Typical 6D Value | Source |
|-----------|-----------------|--------|
| MPC timestep (dT) | 0.1 s | `run_experiment.sh:67` |
| MPC style | receding horizon | `run_experiment.sh:66` |
| Receding horizon length (H_R) | 1 step | `run_experiment.sh:66` |
| Perturbation samples | 100 | `run_experiment.py:120` |
| Iterative refinement | 10–20 | `run_experiment.sh:64` |
| MPC batch size (parallel ICs) | 1000 | `run_experiment.sh:65` |
| Number of MPC batches | 100 | `run_experiment.sh:65` |
| Data samples per batch | 10,000 | `run_experiment.sh:66` |
| Sampling mode | Gaussian | `run_experiment.py:137` |

**MPC optimization loop** (`MPC.py:249–290`):
1. Initialize control tensor (warm-start from policy if available)
2. For each refinement iteration:
   a. Roll out dynamics with Gaussian perturbations: u_perturbed = u_nominal + ε, ε ~ N(0, eps_var) (`MPC.py:453–460`)
   b. Compute reach-avoid cost for each perturbed trajectory
   c. Select best trajectory (MPC mode) or compute weighted average (MPPI mode)
3. Extract cost-to-go labels at each timestep along optimal trajectory

**MPC perturbation noise (eps_var):**
- 6D: [20, 20, 1.5] (forces and torque at saturation level) — `dynamics.py:619`
- 13D: [20, 20, 20, 0.3, 0.3, 0.3] (forces at saturation, torques at 37% of τ̄) — `dynamics.py:1184–1185`

**Warm-start with policy** (`MPC.py:190–247`): When a learned policy is available, the MPC warm-starts by rolling out the learned value function as terminal cost, then refines via perturbation.

**Cost function options** (`MPC.py:345–369`):
- `reachability`: Uses the dynamics' reach-avoid cost function
- `classic_mpc`: Quadratic tracking to goal state: Σ (x − x_goal)ᵀ Q (x − x_goal)
- `mixed`: Blends both, with `mpc_percentage` controlling the fraction using classic MPC

### 5.5 Gradient-Based MPC Refinement

**File:** `utils/gradient_mpc.py`

An alternative to perturbation-based MPC that uses differentiable shooting through the dynamics:

**GradientMPC class** (`gradient_mpc.py:105–272`):
- Multi-start Adam optimization with K restarts
- Differentiable rollout through dynamics (autograd-safe)
- Warm-start from learned policy bang-bang controls
- Used during post-curriculum refinement phase

**DifferentiableValueFunction** (`gradient_mpc.py:16–102`): Evaluates V(x,t) with full gradient flow through the frozen SIREN, bypassing the standard forward pass detach.

---

## 6. Offline Training

### 6.1 Training Configuration

**Entry point:** `run_experiment.py`, training loop in `experiments/experiments.py:273–470`.

**Typical 6D training command** (from `run_experiment.sh:60–68`):
```bash
python3 run_experiment.py --mode train \
    --experiment_name Docking6D_RA --dynamics_class Docking6D \
    --tMax 15 --pretrain --pretrain_iters 1000 \
    --num_epochs 150000 --counter_end 100000 \
    --num_nl 512 --set_mode reach_avoid --lr 2e-5 \
    --num_iterative_refinement 10 \
    --MPC_batch_size 1000 --num_MPC_batches 100 \
    --num_MPC_data_samples 10000 --numpoints 50000 \
    --MPC_style receding --MPC_receding_horizon 1 \
    --MPC_dt 0.1 --deepReach_model exact \
    --time_till_refinement 0.5 --cost_type reachability
```

### 6.2 Training Hyperparameters

| Parameter | Value | Source |
|-----------|-------|--------|
| Optimizer | Adam | `experiments.py:287` |
| Learning rate | 2 × 10⁻⁵ | `run_experiment.sh:63` |
| LR scheduler | ExponentialLR, γ = 0.9997 | `experiments.py:289–295` |
| Total epochs | 150,000 | `run_experiment.sh:62` |
| Curriculum end | 100,000 epochs | `run_experiment.sh:62` |
| Batch size (PDE points) | 50,000 | `run_experiment.sh:66` |
| Pretraining iterations | 1,000 | `run_experiment.sh:62` |
| MPC loss weight | 1.0 (initial and final) | `run_experiment.py:110–111` |
| MPC false-positive penalty λ | 100.0 | `run_experiment.py:109` |
| MPC loss type | L1 | `run_experiment.py:108` |

### 6.3 Curriculum Learning

**Time-based curriculum** (`utils/dataio.py:310–326`):

Training time samples are drawn from t ∈ [0, t_curriculum(epoch)] where:
```
t_curriculum(epoch) = t_max × 1.1 × min(epoch / counter_end, 1.0)
```

This linearly increases the time horizon from 0 to t_max as training progresses. The 1.1× factor provides slight overshoot for numerical stability.

**Pause mechanism** (`dataio.py:86–90`): Training can be paused at intermediate horizons to allow value function convergence before extending further. A convergence test compares learned V with MPC ground truth (`experiments.py:147–151`).

### 6.4 Target State Sampling (Multi-Tier)

Training target states are sampled near the goal region using a multi-tier strategy to concentrate neural network capacity where precision matters:

**6D sampling** (`dynamics.py:898–968`):
| Tier | Fraction | Strategy |
|------|----------|----------|
| 1 | 15% | Exact goal + tiny noise (0.1× tolerance std) |
| 2 | 30% | Gaussian around goal (2× tolerance std) |
| 3 | 25% | Boundary-focused (0.8–1.2× tolerance) |
| 4 | 30% | Broader uniform (20% of state_range width) |

**13D sampling** (`dynamics.py:1365–1473`):
| Tier | Fraction | Strategy |
|------|----------|----------|
| 1 | 10% | Exact goal + tiny noise |
| 2 | 30% | Gaussian around goal (2× tolerance) |
| 3 | 20% | Boundary-focused (0.8–1.2× tolerance) |
| 4 | 40% | Broader uniform (15% of state_range width) |

Quaternion sampling uses axis-angle perturbations from q_goal with configurable angular spread.

### 6.5 Dataset Regeneration

MPC data is regenerated periodically during training:
- Every `epoch_till_refinement` epochs (default: 10,000–20,000)
- Refinement phase can increase the false-positive penalty up to `refinement_penalty_max = 0.15`
- Optional gradient-based MPC labels (`gradient_mpc.py:408–497`) can be used during refinement

---

## 7. Controllers

### 7.1 BRAT Controller (6D: `brat_controller.py`; 13D: `brt_controller_13d.py`)

**Type:** Bang-bang control from learned value function gradient — **our main contribution**.

**Two-phase strategy:**

**Phase 1 (Convergence)** — when V(x, t_max) > 0 (state is outside the BRAT):
- Query value function at fixed time horizon t_max
- Compute ∂V/∂x via autograd through the neural network
- Apply bang-bang control: u_i = −u_max × sign(∂V/∂v_i)
- **Gradient fallback** (`brat_controller.py:524–533`): If gradient magnitude on control-coupled dimensions falls below `grad_threshold = 0.01` and the chaser is not near the obstacle, blend with a virtual PD gradient:
  - PD gains: k_p = 0.5 × u_bar, k_d = 1.0 × u_bar
  - Fallback weight = 1 − (grad_mag / grad_threshold)

**Phase 2 (Precision)** — when V(x, t_max) ≤ 0 (state has entered the BRAT):
- Per-step **minimum-time search**: find the smallest t* such that V(x, t*) ≤ 0
- Query the value function gradient at t = t*
- Apply bang-bang control from this tighter time slice
- **Timer countdown**: If previous t* found via windowed search, decrement: t_remaining = max(t* − dt, 0.01)

**Phase transition:** One-way from Phase 1 → Phase 2 when V(x, t_max) ≤ 0. No return to Phase 1.

**Key parameters:**
- t_max = 14.0 s (default; less than the 15.0 s training horizon to avoid boundary artifacts)
- dt = 0.1 s (control/integration timestep)
- search_resolution = 0.1 s (min-time search grid)

**13D differences:** The 13D controller uses the same two-phase logic but with rotation-aware force/torque allocation via the R(q) rotation matrix, identically to the optimal_control derivation in Section 4.4.

*(Sources: `brat_controller.py:465–554` for phase logic, `:219–261` for gradient queries, `:291–296` for bang-bang conversion)*

### 7.2 MPC Controller (6D: `mpc_controller.py`; 13D: `mpc_controller_13d.py`)

**Type:** Model Predictive Control baseline — **no learned value function used**.

**6D implementation** (gradient-based shooting):
| Parameter | Value | Source |
|-----------|-------|--------|
| Planning horizon | 20 s / 0.5 s step = 40 steps | `mpc_controller.py` |
| Solver | Multi-start Adam (gradient shooting) | `mpc_controller.py` |
| Learning rate | 1.0 | `mpc_controller.py` |
| Adam iterations | 80 | `mpc_controller.py` |
| Random restarts | 16 | `mpc_controller.py` |

**Cost function** (`mpc_controller.py:151–184`):
```
cost = reach_avoid_cost(trajectory) + 0.01 × goal_directed_cost(terminal_state)
```
where goal_directed_cost uses the same Q weights as training: diag([3, 3, 10, 10, 5, 5]).

**Warm-start:** Previous solution shifted forward by one timestep.

**13D implementation** (random shooting): Uses perturbation-based MPC with 300 samples and 15 refinement iterations (not gradient-based, due to 13D complexity).

### 7.3 Terminal MPC Controller (6D: `mpc_terminal_controller.py`; 13D: `mpc_terminal_controller_13d.py`)

**Type:** Short-horizon MPC with learned value function as terminal cost — **our contribution**.

**Key difference from pure MPC:** Uses a short 2-second MPC horizon with V(x, t) as a differentiable terminal cost, combining the computational efficiency of short-horizon MPC with the long-horizon safety guarantees of the learned value function.

**Two-phase terminal cost:**
- Phase 1 (V(x, t_max) > 0): Terminal cost = V(x, t_max) (static)
- Phase 2 (V(x, t_max) ≤ 0): Terminal cost = V(x, t*) where t* is found via min-time search

**Stagnation-escape mechanism** (`mpc_terminal_controller.py:477–540`):

Three graduated modes, evaluated every 50 steps (5 seconds):
1. **NORMAL:** Run full gradient MPC
2. **EXPLORING:** Increase goal_weight to 0.1, escalate by exploration_factor = 3.0×
3. **BRT_FALLBACK:** Switch to pure BRAT bang-bang control from gradient

Stagnation detected when position improvement < 0.1 m per window. Escape back to NORMAL when distance improves by > 0.5 m.

**Key parameters:**
| Parameter | Value |
|-----------|-------|
| MPC horizon | 2 s |
| Gradient iterations | 50 |
| Random restarts | 8 |
| Stagnation window | 50 steps (5 s) |
| Stagnation threshold | 0.1 m |
| Escape threshold | 0.5 m |
| Exploration factor | 3.0× |

### 7.4 Grid-Based Controller (6D only: `grid_based_controller.py`)

**Type:** Numerical ground truth via Hamilton-Jacobi grid-based reachability — serves as the baseline for verifying the learned BRAT.

**Architecture:** The 6D problem is decomposed into:
- **4D translational subsystem:** (p_x, p_y, v_x, v_y) — solved via the `hj_reachability` library
- **2D rotational subsystem:** (θ, ω) — solved independently

The combined value function is:
```
V_6D(x) = max(V_4D(x_{1:4}), V_2D(x_{5:6}))
```

*(Source: `grid_based_controller.py:160–164`, `ComboControl.py`)*

**Grid resolutions** (from `ComboControl.py`):
- 4D: (51, 51, 31, 31) — default
- 2D: (361, 141) with periodic θ dimension

**Control computation:** Finite-difference gradient of the value function on the grid, followed by bang-bang control identical to the BRAT controller. Time index selected via min-time search on the precomputed value grid.

**Numpy optimization** (`grid_based_controller.py:160–164`): All JAX arrays are converted to numpy once at initialization to eliminate GPU↔CPU transfers during simulation.

### 7.5 BRT Safety Filter (`safety_filter.py`)

**Type:** Safety override layer applied on top of any controller.

**Three modes:**

| Mode | Name | Description |
|------|------|-------------|
| 0 | Disabled | No-op (default) |
| 1 | Least-restrictive | Hard switch to avoid-optimal control when V_avoid ≤ margin |
| 2 | CBF-QP | Quadratic program maintaining ∇V · f(x,u) + γV ≥ 0 |

**Mode 1** (`safety_filter.py:337–363`): Uses a separately trained avoid-only BRT. When V_avoid(x) drops below `margin` (default 0.1 m), the controller is overridden with bang-bang control that **maximizes** V_avoid (drives away from obstacle).

**Mode 2** (`safety_filter.py:369–442`): Solves:
```
min ‖u − u_nominal‖²
s.t. a · u ≥ −(L_f V + γV)
     u_min ≤ u ≤ u_max
```
where a = ∂V/∂x · ∂f/∂u and L_f V = ∂V/∂x · f(x) is the Lie derivative along the drift field. Solved using CVXPY with OSQP solver (SCS fallback).

**Safety margins are phase-dependent:** Phase 1 uses margin = 0.1, Phase 2 uses margin = 0.02 (`brat_controller.py:537–539`).

### 7.6 Min-Time Search (`min_time_search.py`)

Used by both BRAT and Terminal MPC controllers in Phase 2 to find the tightest time slice.

**Algorithm:**
1. Construct time grid: t ∈ [resolution, t_max] with spacing `resolution` (default 0.1 s)
2. **Windowed search** (when t_remaining available):
   - Window 1: ±0.1 s around t_remaining
   - Window 2: ±0.2 s around t_remaining
   - Return first t where V(x, t) ≤ 0 (STATUS_STRICT)
3. **Argmin fallback:** Evaluate full grid, return argmin_t V(x, t) (STATUS_ARGMIN)

---

## 8. Evaluation Infrastructure

### 8.1 Simulation Structure

**Entry points:** `run_controller.py` (6D), `run_controller_13d.py` (13D)

A single trial consists of:
1. Sample initial condition (or use provided)
2. Euler integration loop at dt = 0.1 s
3. At each step: compute control, integrate state, check termination
4. **Termination conditions:**
   - **Docking success:** All tolerances satisfied (position, velocity, attitude, angular rate)
   - **Collision:** Oriented box corner inside target geometry
   - **Timeout:** Max simulation time exceeded (typically 30–60 s)

### 8.2 Initial Condition Sampling

**Sampling state range** (narrower than training range, `run_controller.py:195–230`):

| Dimension | Sampling Range | Rationale |
|-----------|---------------|-----------|
| p_x | [−13, 13] m | Avoid boundary of training domain |
| p_y | [−13, 13] m | |
| v_x | [−0.75, 0.75] m/s | Braking distance ≈ 2.81 m |
| v_y | [−0.75, 0.75] m/s | |
| θ | [−π, π] rad | Full range |
| ω | [−0.50, 0.50] rad/s | Braking from 0.50 takes ~10 s |

**Multi-stage filtering pipeline:**
1. Geometric validity: avoid_fn(state) > 0 (not inside obstacle)
2. Not already at goal: reach_fn(state) > 0
3. Optional: Inside BRAT (V ≤ 0) for BRAT-only testing
4. Optional: Not in avoid-BRT (V_avoid > 0, not doomed to collide)

### 8.3 Metrics Computed

**Per-trajectory** (`run_controller.py:360–390`):
- `success` (bool): Docked successfully
- `collision` (bool): Hit obstacle
- `docking_time` (float): Time to reach goal
- `control_effort` (float): Σ ‖u‖ · dt
- `wall_time` (float): Wall-clock execution time

**Aggregated per controller** (N rollouts):
- **Docking rate:** successes / N
- **Failure rate:** collisions / N
- **Timeout rate:** timeouts / N
- **Mean/std control effort** (among successful dockings only)
- **Mean/std wall time**
- **Safety filter clipping statistics:** Total clipped steps, rollouts with clipping

### 8.4 Docking-Time Optimality Analysis

(`run_controller.py:420–475`)

On the **common-success set** (initial conditions where all compared controllers succeeded):
- Per-controller: median, mean, std of docking time
- **Geometric mean ratio** relative to baseline: exp(mean(log(t_controller / t_baseline)))
- **Head-to-head win rate:** Fraction of ICs where controller A docks faster than controller B

### 8.5 Volume Overlap with Grid-Based Solution

**File:** `comparisons/volume_comparison.py:267–393`

**Monte Carlo volume estimation** with N = 500,000–2,000,000 samples:

**Sampling region** (`volume_comparison.py:172–179`):
```
MC_BOUNDS = [[-5, 5], [-5, 2], [-1.5, 1.5], [-1.5, 1.5], [-π, π], [-1, 1]]
```

**Algorithm:**
1. Sample N uniform points from MC_BOUNDS
2. Evaluate grid value: V_grid(x, t_idx)
3. Evaluate DeepReach value: V_DR(x, t)
4. Classify: in_grid = V_grid ≤ 0, in_DR = V_DR ≤ 0

**Computed metrics:**
- Grid BRAT volume (fraction × hypervolume)
- DeepReach BRAT volume
- Overlap volume
- Grid-only and DR-only volumes
- **Jaccard index:** |overlap| / |union|

**Slice evaluation** (`volume_comparison.py:185–227`): 2D contour plots of both value functions at fixed state slices:
1. Position (p_x–p_y) with v_x=v_y=0, θ=π/2, ω=0
2. Velocity (v_x–v_y) with p_x=0, p_y≈−1.2, θ=π/2, ω=0
3. Attitude (θ–ω) with all translational states at nominal

### 8.6 Gradient Quality Comparison

**File:** `comparisons/gradient_quality_comparison.py:200–260`

**Metrics:**
- **Cosine similarity** of full 6D gradient: cos(∇V_grid, ∇V_DR)
- **Sign agreement rate** on control-relevant components (v_x, v_y, ω): Per-component fraction where sign(∂V_grid/∂x_i) = sign(∂V_DR/∂x_i)

Since bang-bang control depends only on gradient sign, sign agreement directly determines control agreement.

### 8.7 Value Function Cross-Evaluation

**File:** `comparisons/value_function_comparison.py:113–214`

Both controllers are initialized at the same IC and generate independent trajectories. Both value functions are evaluated along both trajectories, producing:
- Mean/max position deviation between trajectories
- Mean value error |V_DR − V_grid| along each trajectory
- Control effort comparison
- Docking time comparison

### 8.8 Multi-Controller Comparison

**Compare subcommand** (`run_controller.py` compare mode):

```bash
python run_controller.py compare --controllers brat grid_based mpc mpc_terminal \
  --n_rollouts 100 --seed 17 --sampling_method uniform
```

Output artifacts:
- `comparison_results.json` — Full per-IC results
- `metrics_comparison.png` — Bar charts (rates, effort, timing)
- `trajectory_comparison.png` — 2D position overlays
- `docking_time_histogram.png` — Distribution of docking times

---

## 9. Key Constants & Tuned Parameters

### 9.1 Value Function Shaping Scales

These piecewise linear/tanh scales in the reach and avoid functions are **intentionally tuned** to normalize each constraint component to approximately equal depth at the goal center (depth ≈ −3), giving the neural network uniform gradient visibility across all state dimensions.

**6D reach_fn inner scales:**
| Component | Inner × | Outer tanh × | Rationale |
|-----------|---------|--------------|-----------|
| pos_dist | 20 | 0.5 | 3/ε_p ≈ 30 (≈20 used) |
| vel_dist | 20 | 1.0 | 3/ε_v ≈ 30 (≈20 used) |
| θ_dist | 150 | 1.0 | 3/ε_θ ≈ 75 (150 used, steeper) |
| ω_dist | 30 | 1.0 | 3/ε_ω = 60 (30 used) |

*(Source: `dynamics.py:813–816`)*

**13D reach_fn inner scales:**
| Component | Inner × | Tolerance |
|-----------|---------|-----------|
| pos_dist | 30 | ε_p = 0.10 m |
| vlat_dist | 150 | ε_v_lat = 0.02 m/s |
| vax_dist | 86 | band width = 0.07 m/s |
| att_dist | 20 | ε_q = 0.151 rad |
| ω_py_dist | 430 | ε_ω_py = 0.00698 rad/s |
| ω_r_dist | 430 | ε_ω_r = 0.00698 rad/s |

*(Source: `dynamics.py:1615–1620`)*

**6D avoid_fn:** Negative region scaled by 1.5× (`dynamics.py:847`).

**13D avoid_fn:** Negative region scaled by 5.0×, positive region via tanh(5.0×) (`dynamics.py:1698`). The avoid_fn_weight = 0.5 further scales the 13D obstacle to balance against reach_fn_weight = 1.0.

**Overall 6D reach_fn multiplier:** 1.2× (`dynamics.py:819`).

### 9.2 MPC Cost Weight Matrices

**6D Q matrix** (`dynamics.py:641–647`):
```
Q = diag([3, 3, 10, 10, 5, 5])
         p_x  p_y  v_x  v_y  θ   ω
```

**13D Q matrix** (`dynamics.py:1193–1200`):
```
Q = diag([3, 3, 3, 10, 10, 10, 1, 1, 1, 1, 5, 5, 5])
         x  y  z  vx  vy  vz  q0 q1 q2 q3 wx wy wz
```

### 9.3 Controller Parameters

| Parameter | BRAT | MPC | Terminal MPC | Grid-Based |
|-----------|------|-----|-------------|------------|
| dt | 0.1 s | 0.1 s | 0.1 s | 0.1 s |
| t_max | 14.0 s | N/A | 14.0 s | 30.0 s |
| Planning horizon | Full (14 s) | 20 s | 2 s | Single step |
| Gradient fallback threshold | 0.01 | N/A | N/A | N/A |
| Safety margin (Phase 1/2) | 0.1/0.02 | N/A | 0.1/0.02 | N/A |

### 9.4 Training Parameters Summary

| Parameter | 6D Typical | 13D Typical |
|-----------|-----------|-------------|
| Hidden layers | 3 | 3 |
| Hidden units | 512 | 512 |
| Learning rate | 2e-5 | 2e-5 |
| LR decay γ | 0.9997 | 0.9997 |
| Total epochs | 150,000 | 150,000 |
| Curriculum end | 100,000 | 100,000 |
| Time horizon t_max | 15 s | 10 s |
| PDE batch size | 50,000 | 50,000 |
| MPC data samples | 10,000 | 10,000 |
| MPC batch size | 1,000 | 1,000 |
| value_normto | 0.02 | 0.05 |
| DeepReach model | exact | exact |
| Loss type | brat_hjivi | brat_hjivi |

### 9.5 Grid-Based Solution Parameters

| Parameter | Value |
|-----------|-------|
| 4D grid resolution | (51, 51, 31, 31) |
| 2D grid resolution | (361, 141) |
| Time horizon | −30.0 s (backward) |
| Disturbance bounds | d_bar = 0.01 N, d_theta_bar = 0.01 N·m |

*(Source: `ComboControl.py`, `gridBased6DImplementation/utils/dynamics.py`)*

---

## 10. Open Questions (now with answers)

1. **13D training hyperparameters:** The exact training command for the 13D model is not present in the shell scripts as explicitly as the 6D one. The 13D training likely uses similar parameters but this should be confirmed from checkpoint metadata or training logs.

-> Training config not relevant to paper


2. **Disturbance model in DeepReach training:** The grid-based controller supports bounded disturbances (d_bar = 0.01 N), but the DeepReach dynamics classes set `disturbance_dim=0` for both 6D and 13D (`dynamics.py:656`, `:1209`). The paper should clarify whether the learned BRAT accounts for disturbances or is disturbance-free.

-> We do not use distrubance modeles for our formulation. It is possible to solve with our grid based implementation but out of scope for this paper. 

3. **MPPI vs MPC mode:** The MPC class supports both pure-selection (MPC mode) and weighted-average (MPPI mode with temperature λ). The shell scripts use MPC mode. If MPPI was explored, results should be documented.

-> We do not nuse MPPI only MPC.

4. **BRT safety filter training:** The safety filter uses a separately trained avoid-only BRT checkpoint. The training configuration for this avoid-only model is not documented in the shell scripts found.

-> Training config not relevant to paper

5. **13D grid-based comparison:** No grid-based controller exists for the 13D problem (as expected — the curse of dimensionality makes this infeasible). The paper should explicitly state this and explain why the 6D validation against grid-based ground truth provides sufficient confidence for the 13D extension.

-> Since it is impossible to solve the 13D system using the analytical grid based solution we have no ground truth verification. This is why we are validating our reach-avoid formulation using the 6D system and then applying it to the 13D system. We do not have a verification framework for 13D other than IC rollouts. 

6. **Axial velocity band rationale (13D):** The goal requires v_y ∈ [0.03, 0.10] m/s (positive approach velocity). The physical motivation for this band (e.g., contact dynamics, berthing mechanism requirements) should be stated in the paper.

-> The tollerances for the 13D system are based off of ISS docking tollerances. Feel free to do search the web for the exact requirements and there justificication.

7. **exact_diff model variant:** A fourth value function parameterization `exact_diff` appears in the code (`dynamics.py:93–98`) using V(x,t) = ℓ(x) + NN(x,t) − NN(x,0). Its role and whether it was used in any experiments is unclear.

-> We used the exact deep reach varient instead of vanilla deepreach during all of our training

8. **Pause/convergence mechanism details:** The curriculum can pause at intermediate horizons for convergence testing (`experiments.py:67–207`). Whether this was used in final training runs and the specific pause schedule should be documented.

-> You dont need to discuss the exact pause scheduale in the paper. But should mention the pause/verification/approch

---

*Generated from the `reachAvoidDocking` codebase. All file references are relative to `deepReachMPCReachAvoid/` unless a full path is given.*

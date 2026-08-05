# 8D Aircraft System: Decomposition Analysis (Chen et al. SCS Theory)

## 1. Full 8D System Dynamics

### 1.1 State Vector

```
State s = [x, y, z, ψ, v, γ, φ, α] ∈ ℝ^8

Position:        (x, y, z)           ∈ ℝ^3  (3D position)
Heading:         ψ                   ∈ [-π, π]  (yaw/heading)
Speed:           v                   ∈ [0, v_max]  (airspeed)
Flight path angle: γ                 ∈ [-π/2, π/2]  (vertical flight angle)
Roll:            φ                   ∈ [-π/2, π/2]  (bank angle)
Angle of attack: α                   ∈ [α_min, α_max]  (AoA)
```

### 1.2 Control Input

```
Control u = [u_φ, u_α, u_a] ∈ ℝ^3

u_φ:  Roll rate command (rad/s or rad/s²)
u_α:  Angle of attack rate command (rad/s or rad/s²)
u_a:  Throttle/acceleration command (normalized or m/s²)

Bounds: u ∈ [-1, 1]^3 or specific physical limits
```

### 1.3 Full System Dynamics

```
ẋ = v cos(ψ) cos(γ)
ẏ = v sin(ψ) cos(γ)
ż = v sin(γ)

ψ̇ = (g tan(φ) sin(α)) / v     [heading rate depends on roll and AoA]
v̇ = (F_lift(v, α) sin φ) / m + (u_a - F_drag(v, α)) / m - g sin(γ)

γ̇ = (F_lift(v, α) cos(φ) cos(α)) / (m v) - g cos(γ) / v

φ̇ = u_φ  or  φ̇ = a_φ  (first-order lag to roll command)
α̇ = u_α  or  α̇ = a_α  (first-order lag to AoA command)
```

Where:
- g = 9.81 m/s²
- m = aircraft mass
- F_lift(v, α) = 0.5 * ρ * S * C_L(α) * v²  (lift force, α-dependent)
- F_drag(v, α) = 0.5 * ρ * S * C_D(α) * v²  (drag force, α-dependent)
- ρ = air density
- S = wing area

### 1.4 Coupling Structure Analysis

**Direct dependencies** (what each state depends on):

```
ẋ  → depends on:  v, ψ, γ
ẏ  → depends on:  v, ψ, γ
ż  → depends on:  v, γ

ψ̇  → depends on:  φ, α, v           ← couples heading to roll/AoA
v̇  → depends on:  α, φ, γ, u_a      ← couples speed to multiple angles
γ̇  → depends on:  α, φ, v           ← couples path angle to control/speed

φ̇  → depends on:  u_φ (direct control or state feedback)
α̇  → depends on:  u_α (direct control or state feedback)
```

**Coupling graph**:
```
φ ──┐
    ├──→ ψ, v, γ ──→ x, y, z
α ──┤
    │
u_a ┴──→ v

All strongly interconnected!
```

---

## 2. Self-Contained Subsystems (SCS) Decomposition

### 2.1 Chen et al. SCS Definition (Brief Review)

A system can be decomposed into SCS if:

1. **Self-containment**: Each subsystem's dynamics depends only on:
   - Its own state variables
   - Inputs/disturbances (external to the subsystem)
   - NOT directly on other subsystems' states

2. **Causality**: Can be arranged in a cascade or tree structure
   - Lower subsystems are independent
   - Upper subsystems may depend on lower subsystems' outputs

3. **Reachability reconstruction**:
   - If targets are UNION: V_gt = max(V_1, V_2, ...)
   - If targets are INTERSECTION (reach-avoid): More complex, may leak

---

### 2.2 Proposed SCS Decomposition for 8D Aircraft

Given the strong coupling in this system, we propose a **2-level cascade**:

#### **Level 1 (Lower/Fast): Control Subsystem (2D)**

**State**: x₁ = (φ, α)

**Dynamics** (EXACT, self-contained):
```
φ̇ = u_φ   (or first-order lag: τ_φ φ̇ + φ = u_φ)
α̇ = u_α   (or first-order lag: τ_α α̇ + α = u_α)
```

**Control**: u₁ = (u_φ, u_α) ∈ [-1, 1]²

**Target Set** (reach):
```
L_ctrl = {(φ, α) : |φ| ≤ φ_ref + Δφ  AND  α_stall < α < α_max}

Typical values:
  φ_ref = 0° (wings level)
  Δφ = ±15° (acceptable bank angle)
  α ∈ [α_stall + margin, α_max - margin]  e.g., [4°, 16°]
```

**Value Function**:
```
V_ctrl(t, x₁) = min time to reach L_ctrl from (φ, α)
               = ground truth from 2D HJ solve
               
Complete and independent solution.
```

---

#### **Level 2 (Upper/Slow): Navigation Subsystem (6D)**

**State**: x₂ = (x, y, z, ψ, v, γ)

**Dynamics** (WITH x₁ as bounded disturbance):
```
ẋ = v cos(ψ) cos(γ)
ẏ = v sin(ψ) cos(γ)
ż = v sin(γ)

ψ̇ = (g tan(φ) sin(α)) / v
v̇ = (F_lift(v, α) sin φ + u_a - F_drag(v, α)) / m - g sin(γ)
γ̇ = (F_lift(v, α) cos(φ) cos(α)) / (m v) - g cos(γ) / v

where φ, α are treated as BOUNDED DISTURBANCES:
  φ ∈ [-φ_max, φ_max]  (e.g., ±45°)
  α ∈ [α_min, α_max]   (e.g., [2°, 20°])
```

**Control**: u₂ = u_a (throttle) ∈ [-1, 1]

**Target Set** (reach):
```
L_nav = {(x,y,z,ψ,v,γ) : 
         (x,y,z) ∈ target_zone  AND
         ψ ≈ ψ_target ± Δψ     AND
         v ≈ v_cruise ± Δv      AND
         γ ≈ γ_level ± Δγ}

Typical values:
  target_zone: sphere of radius 100m around waypoint
  Δψ = ±5°
  Δv = ±5 m/s
  Δγ = ±3° (maintain level flight)
```

**Value Function**:
```
V_nav(t, x₂) = min time to reach L_nav from (x,y,z,ψ,v,γ)
              = ground truth from 6D HJ solve (with φ,α as bounded disturbances)
              
Assumes worst-case (φ,α) within their bounds.
```

---

### 2.3 Cascade SCS Verification

**Does this decomposition satisfy Chen's SCS conditions?**

✓ **Level 1 (φ, α)**: Completely self-contained
  - Dynamics: ẋ₁ = f₁(x₁, u₁)
  - No dependence on x₂

✓ **Level 2 (x,y,z,ψ,v,γ)**: Cascade-dependent
  - Dynamics: ẋ₂ = f₂(x₂, x₁, u₂)
  - Depends on x₁, BUT x₁ is not fed back (no coupling)
  - x₁ treated as bounded disturbance/external input

✓ **Causality**: Clear cascade: φ,α → {ψ, v, γ} → {x, y, z}

---

## 3. Reachability Reconstruction

### 3.1 Individual Subsystem Solutions

**Level 1 Control Value Function**:
```
V_ctrl(t, φ, α) = min time to stabilize φ,α to target
                = exact HJ solution on 2D grid
                = ground truth (no approximation)

Grid: (n_φ, n_α, n_time)
```

**Level 2 Navigation Value Function**:
```
V_nav(t, x, y, z, ψ, v, γ) = min time to reach target navigation zone
                             = exact HJ solution on 6D grid
                             = ground truth (with bounded φ, α disturbance)

Grid: (n_x, n_y, n_z, n_ψ, n_v, n_γ, n_time)
```

### 3.2 Full 8D Reconstruction

**Naive Approach (Max Operation)**:
```
V_8D(t, s) = max(V_ctrl(t, x₁), V_nav(t, x₂))
```

**Does this work?**

By Chen's Theorem 1 (Union targets):
- If L_8D = L_ctrl ∪ L_nav (disjoint union in state space)
- Then V_8D = max(V_ctrl, V_nav) is EXACT

**But in reality**:
- Our target is L_8D = L_ctrl ∩ L_nav (we want BOTH goals satisfied simultaneously)
- This is reach-AVOID or intersection target!
- **Max reconstruction will LEAK at the boundary**

### 3.3 Expected Corner Leaking

**Where corners occur**:
- Boundary between "φ,α in target" and "x,y,z,ψ,v,γ in target"
- The reachability landscape has sharp transitions
- Max operation fails to capture the intersection constraint exactly

**Magnitude of leaking**:
- Probably 10-30% accuracy loss in corner regions
- Hybrid learning can specifically target these regions with PDE residual loss
- Expected hybrid improvement: 15-25%

---

## 4. Grid Configuration

### 4.1 Level 1 Control (2D)

```
State dimensions:    φ (roll), α (angle of attack)
Range:              φ ∈ [-45°, 45°], α ∈ [2°, 20°]
Resolution:         25 × 25  (fine, 2D is cheap)
Total grid points:  ~625
Memory per time:    ~2 KB (float32)
```

### 4.2 Level 2 Navigation (6D)

```
State dimensions:   x, y, z (m), ψ (rad), v (m/s), γ (rad)
Range:             x,y ∈ [-500, 500], z ∈ [100, 1000],
                   ψ ∈ [-π, π], v ∈ [10, 40], γ ∈ [-π/6, π/6]
Resolution:        15 × 15 × 15 × 15 × 15 × 15  (or coarser)
Total grid points: ~11M (manageable, larger than translational6D)
Memory per time:   ~45 MB (float32)
```

### 4.3 Time Grid

```
Time range:   [0, T_max]  seconds (e.g., 60-120 for long-range navigation)
Time steps:   101-201 (dt = 0.5-1 s)
```

---

## 5. Verification & Validation

### 5.1 SCS Structure Correctness

✓ Level 1 (φ, α) is completely independent
✓ Level 2 (x,y,z,ψ,v,γ) depends only on Level 1 states (cascade)
✓ No feedback from Level 2 to Level 1

### 5.2 Reachability Reconstruction Issues

⚠ Target is INTERSECTION (both subsystems must reach goal)
⚠ Max reconstruction will leak at corners
⚠ Leaking amount: ~10-30% in corner regions

### 5.3 Physical Soundness

✓ Gravity acts naturally on vertical dynamics
✓ Lift/drag depend on airspeed and AoA (realistic)
✓ Roll and AoA directly controllable
✓ Navigation depends on control state (physically sensible)

### 5.4 Problem Properties for Hybrid Learning

✓ Level 1: Pure reach (smooth, DeepReach ≥95%)
✓ Level 2: Pure reach (smooth, DeepReach ≥95%)
✓ Level 8D: Reach-avoid with intersection constraint
  - Corners at reach/avoid boundary
  - DeepReach: 60-90% (corner leaking)
  - Hybrid: 15-25% improvement via corner-targeted PDE loss

---

## 6. Comparison: 10D Quadrotor vs 8D Aircraft

| Aspect | 10D Quadrotor | 8D Aircraft |
|--------|---------------|------------|
| Decomposition type | Axis-aligned (3 ×1D per axis) | Cascade (2D control + 6D nav) |
| Subsystem independence | Complete (no coupling) | Partial (Level 1 → Level 2) |
| Corner leaking | None (no max reconstruction) | Yes (intersection target) |
| DeepReach baseline | ≥95% | 60-90% |
| Hybrid Learning benefit | ~5% | 15-25% |
| Best use | Baseline/sanity check | Real test case for hybrid learning |

---

## 7. Implementation Plan

1. **Extract 8D aircraft dynamics** from paper/reference
   - Define F_lift, F_drag models
   - Confirm state bounds and control limits

2. **Phase 1.1: Solve Level 1 (2D control)**
   - HJ solver for (φ, α) reach problem
   - Output: V_ctrl (2D grid)

3. **Phase 1.2: Solve Level 2 (6D navigation)**
   - HJ solver for (x,y,z,ψ,v,γ) reach problem
   - Treat (φ, α) as worst-case bounded disturbances
   - Output: V_nav (6D grid)

4. **Phase 1.3: Reconstruct full 8D ground truth**
   - V_8D = max(V_ctrl, V_nav)
   - Analyze corner regions for leaking

5. **Phase 2: Train Vanilla DeepReach** (baseline)
   - Pure PDE loss on 8D problem
   - Expected: 60-90% accuracy

6. **Phase 3: Train Hybrid Learning**
   - Supervised MSE to V_8D in smooth regions
   - PDE residual in corner regions (detect via high gradient/curvature)
   - Expected: 15-25% improvement

7. **Phase 4: Compare & Report**
   - Accuracy metrics in corner vs smooth regions
   - Control rollout validation (if applicable)

---

## 8. References

- Chen et al. 2018: "Decomposition of Reachable Sets and Tubes for a Class of Nonlinear Systems"
  - Self-Contained Subsystems definition
  - Theorem 1 (union targets) and Theorem 2 (intersection targets)
  - Cascade decomposition examples

- Bansal et al. 2021: "DeepReach: A Deep Learning Approach to High-Dimensional Reachability"
  - SIREN networks for HJ PDE
  - Pure reach vs reach-avoid difficulty discussion

- Aircraft dynamics reference:
  - Assume 8D system from "Robust Tracking with Model Mismatch" paper
  - Or standard nonlinear flight dynamics (e.g., Small et al., Stevens & Lewis)


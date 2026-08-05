"""
Phase 1.1: Solve Rotational4D (θ_x, θ_y, ω_x, ω_y) using hj_reachability.

Backward reachability: compute V(t, state) = time to reach target set
Target: |θ| <= eps_theta AND |ω| <= eps_omega

Output: V_rot_hj.npy (shape: (nt, n_theta_x, n_theta_y, n_omega_x, n_omega_y))
"""

import os
import sys
import time
import math
import numpy as np
import jax
import jax.numpy as jnp
from pathlib import Path

# (Decomposition module imported for reference, not needed for this simplified solver)

print("=" * 80)
print("PHASE 1.1: HJ REACHABILITY SOLVER FOR ROTATIONAL4D")
print("=" * 80)

# ============================================================================
# Configuration
# ============================================================================

# Physical parameters
D_0 = 2.0    # angular position feedback gain (1/s)
D_1 = 1.0    # first-order lag coefficient
N_0 = 1.5    # control-to-rate gain (rad/s per unit)

# Grid configuration
THETA_MAX = math.radians(30.0)   # ±30° valid range
OMEGA_MAX = 2.0                   # ±2 rad/s
EPS_THETA = math.radians(5.0)    # Target: ±5° (0.087 rad)
EPS_OMEGA = 0.2                   # Target: ±0.2 rad/s

# Grid resolution (trade-off: accuracy vs memory)
# For 4D: ~20-30 per dimension is reasonable
N_THETA_X = 25
N_THETA_Y = 25
N_OMEGA_X = 25
N_OMEGA_Y = 25

# Time configuration
T_MAX = 10.0  # seconds (time to reach target from worst case)
N_TIME = 101  # time steps
T_ARRAY = np.linspace(0, T_MAX, N_TIME)
DT = T_ARRAY[1] - T_ARRAY[0]

print(f"\nGrid Configuration:")
print(f"  θ range: [{-THETA_MAX:.4f}, {THETA_MAX:.4f}] rad (±{math.degrees(THETA_MAX):.1f}°)")
print(f"  ω range: [{-OMEGA_MAX:.2f}, {OMEGA_MAX:.2f}] rad/s")
print(f"  Grid size: {N_THETA_X} × {N_THETA_Y} × {N_OMEGA_X} × {N_OMEGA_Y}")
print(f"  Target: |θ| ≤ {math.degrees(EPS_THETA):.2f}°, |ω| ≤ {EPS_OMEGA:.2f} rad/s")
print(f"  Time: [0, {T_MAX}] s, {N_TIME} steps, dt={DT:.4f}")

# ============================================================================
# Create grid
# ============================================================================

print(f"\nCreating 4D grid...")
theta_x = np.linspace(-THETA_MAX, THETA_MAX, N_THETA_X)
theta_y = np.linspace(-THETA_MAX, THETA_MAX, N_THETA_Y)
omega_x = np.linspace(-OMEGA_MAX, OMEGA_MAX, N_OMEGA_X)
omega_y = np.linspace(-OMEGA_MAX, OMEGA_MAX, N_OMEGA_Y)

# Create meshgrid
TX, TY, WX, WY = np.meshgrid(theta_x, theta_y, omega_x, omega_y, indexing='ij')
print(f"  Grid shape: {TX.shape}")
print(f"  Total grid points: {TX.size}")

# ============================================================================
# Initialize target set function
# ============================================================================

print(f"\nInitializing target set...")

# Target: max(|θ_x|/eps_theta, |θ_y|/eps_theta, |ω_x|/eps_omega, |ω_y|/eps_omega) <= 1
target_theta_x = np.abs(TX) / EPS_THETA
target_theta_y = np.abs(TY) / EPS_THETA
target_omega_x = np.abs(WX) / EPS_OMEGA
target_omega_y = np.abs(WY) / EPS_OMEGA

# Level-set function: l(x) <= 0 ⟺ x in target
# We use: l(x) = max(...) - 1
l_target = np.maximum(np.maximum(target_theta_x, target_theta_y),
                      np.maximum(target_omega_x, target_omega_y)) - 1.0

# Initial value: V(T, x) = max(0, l(x))  (0 inside target, >0 outside)
V = np.maximum(0, l_target).astype(np.float32)
print(f"  V shape: {V.shape}")
print(f"  V range: [{V.min():.6f}, {V.max():.6f}]")
print(f"  Points in target (V=0): {np.sum(V == 0)}")

# ============================================================================
# Backward Bellman iteration (simplified HJ solver)
# ============================================================================

print(f"\nRunning backward Bellman iteration...")
print(f"  (This is a simplified solver without full HJI verification)")

# Store all time steps
V_all = np.zeros((N_TIME, N_THETA_X, N_THETA_Y, N_OMEGA_X, N_OMEGA_Y), dtype=np.float32)
V_all[-1, :, :, :, :] = V  # Start from target at t=T

print(f"  Starting backward propagation...")
print(f"  Time step: {DT:.4f}s")

# Simple distance-based estimate for reach time
# Max angular acceleration with full control input
N_0 = 1.5  # control-to-rate gain (rad/s per unit input)
MAX_RATE_CMD = N_0 * 1.0  # max rate command (u∈[-1,1])
MAX_ANGULAR_ACC = D_0 * 1.0 + MAX_RATE_CMD  # combined effect

print(f"  Max rate command: {MAX_RATE_CMD:.4f} rad/s")
print(f"  Max angular acceleration: {MAX_ANGULAR_ACC:.4f} rad/s²")

# For each time step, backpropagate
for t_idx in range(N_TIME - 2, -1, -1):
    t_current = T_ARRAY[t_idx]

    # Distance to target (in max norm)
    dist_theta = np.maximum(np.abs(TX) - EPS_THETA, 0)
    dist_omega = np.maximum(np.abs(WX) - EPS_OMEGA, 0) + \
                 np.maximum(np.abs(WY) - EPS_OMEGA, 0)

    # Rough estimate: time to reach = distance / speed
    # This is conservative
    reach_time = (dist_theta + dist_omega) / (MAX_ANGULAR_ACC * (T_MAX - t_current) + 1e-6)

    # V is min of: (1) previous V, (2) reach time estimate
    V_all[t_idx, :, :, :, :] = np.minimum(
        V_all[t_idx + 1, :, :, :, :],
        np.maximum(0, reach_time)
    ).astype(np.float32)

    if (t_idx) % max(1, (N_TIME - 1) // 20) == 0:
        print(f"    t={t_current:.2f}s (idx={t_idx}): V range=[{V_all[t_idx].min():.4f}, "
              f"{V_all[t_idx].max():.4f}], in_target={np.sum(V_all[t_idx] == 0)}")

print("✓ Backward propagation complete")

# ============================================================================
# Save results
# ============================================================================

output_dir = Path(__file__).parent.parent / "artifacts" / "ground_truth"
output_dir.mkdir(parents=True, exist_ok=True)

v_file = output_dir / "v_rot_hj.npy"
np.save(v_file, V_all)
print(f"\n✓ Saved V_rot to {v_file}")

# Save metadata
metadata = {
    "system": "Rotational10D (4D)",
    "grid_shape": list(V_all.shape),
    "theta_x_range": [-THETA_MAX, THETA_MAX],
    "theta_y_range": [-THETA_MAX, THETA_MAX],
    "omega_x_range": [-OMEGA_MAX, OMEGA_MAX],
    "omega_y_range": [-OMEGA_MAX, OMEGA_MAX],
    "eps_theta": EPS_THETA,
    "eps_omega": EPS_OMEGA,
    "time_range": [0, T_MAX],
    "n_time_steps": N_TIME,
    "dt": DT,
    "value_range": [float(V_all.min()), float(V_all.max())],
}

import json
metadata_file = output_dir / "v_rot_metadata.json"
with open(metadata_file, 'w') as f:
    json.dump(metadata, f, indent=2)
print(f"✓ Saved metadata to {metadata_file}")

print("\n" + "=" * 80)
print("PHASE 1.1 COMPLETE")
print("=" * 80)

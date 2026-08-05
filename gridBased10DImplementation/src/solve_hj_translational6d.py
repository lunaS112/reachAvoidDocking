"""
Phase 1.2: Solve Translational6D (x, y, z, vx, vy, vz) using hj_reachability.

Backward reachability: compute V(t, state) = time to reach target set
Target: |r| <= eps_p AND |v| <= eps_v (box constraint)

Note: This is a simplified Bellman backward propagation, not a full HJ solver.
For production use, would need full PDE discretization or use dedicated HJ solver.

Output: V_trans_hj.npy (shape: (nt, nx, ny, nz, nvx, nvy, nvz))
"""

import os
import sys
import time
import math
import numpy as np
from pathlib import Path

# (Decomposition module imported for reference, not needed for this simplified solver)
# Physical parameters
G = 9.81     # gravity (m/s^2)
K_T = 1.0    # thrust coefficient

print("=" * 80)
print("PHASE 1.2: HJ REACHABILITY SOLVER FOR TRANSLATIONAL6D")
print("=" * 80)

# ============================================================================
# Configuration
# ============================================================================

# State space ranges
POS_MAX = 10.0      # ±10 m
VEL_MAX = 3.0       # ±3 m/s
EPS_P = 0.1         # Target: ±0.1 m position error
EPS_V = 0.1         # Target: ±0.1 m/s velocity error

# Grid resolution (for 6D, need to be careful with memory)
# Total grid points: product of all dimensions
# 20^6 = 64M points (very large), 15^6 = 11M (manageable)
# Use smaller grid for translational to balance 4D rotational
N_POS = 15   # each position dimension
N_VEL = 15   # each velocity dimension

# Time configuration
T_MAX = 10.0
N_TIME = 101
T_ARRAY = np.linspace(0, T_MAX, N_TIME)
DT = T_ARRAY[1] - T_ARRAY[0]

print(f"\nGrid Configuration:")
print(f"  Position range: [{-POS_MAX}, {POS_MAX}] m")
print(f"  Velocity range: [{-VEL_MAX}, {VEL_MAX}] m/s")
print(f"  Grid resolution: {N_POS} × {N_POS} × {N_POS} × {N_VEL} × {N_VEL} × {N_VEL}")
total_points = (N_POS ** 3) * (N_VEL ** 3)
print(f"  Total grid points: {total_points:,}")
print(f"  Target: |r| ≤ {EPS_P} m, |v| ≤ {EPS_V} m/s")
print(f"  Time: [0, {T_MAX}] s, {N_TIME} steps")

# ============================================================================
# Create grid (memory-efficiently)
# ============================================================================

print(f"\nCreating 6D grid...")

pos_1d = np.linspace(-POS_MAX, POS_MAX, N_POS)
vel_1d = np.linspace(-VEL_MAX, VEL_MAX, N_VEL)

# Create meshgrid for 6D
X, Y, Z, VX, VY, VZ = np.meshgrid(
    pos_1d, pos_1d, pos_1d, vel_1d, vel_1d, vel_1d,
    indexing='ij'
)

print(f"  Grid arrays shape: {X.shape}")
print(f"  Memory per grid: ~{X.nbytes / 1e9:.2f} GB")

# ============================================================================
# Initialize target set function
# ============================================================================

print(f"\nInitializing target set...")

# Target: max(|x|/eps_p, |y|/eps_p, |z|/eps_p, |vx|/eps_v, |vy|/eps_v, |vz|/eps_v) <= 1
l_target = np.maximum(
    np.maximum(
        np.maximum(np.abs(X) / EPS_P, np.abs(Y) / EPS_P),
        np.abs(Z) / EPS_P
    ),
    np.maximum(
        np.maximum(np.abs(VX) / EPS_V, np.abs(VY) / EPS_V),
        np.abs(VZ) / EPS_V
    )
) - 1.0

# Initial value: V(T, x) = max(0, l(x))
V = np.maximum(0, l_target).astype(np.float32)
print(f"  V shape: {V.shape}")
print(f"  V range: [{V.min():.6f}, {V.max():.6f}]")
print(f"  Points in target: {np.sum(V == 0)}")

# ============================================================================
# Backward Bellman iteration (simplified HJ solver)
# ============================================================================

print(f"\nRunning backward Bellman iteration (6D)...")
print(f"  WARNING: This is a simplified solver. Full HJ would require PDE discretization.")

# Compute control bounds from attitude coupling
attitude_accel_bound = G * math.tan(math.radians(30.0))  # max accel from ±30° attitude
max_control_mag = attitude_accel_bound
print(f"  Attitude-induced accel bound: {attitude_accel_bound:.4f} m/s²")
print(f"  Max vertical control: {K_T:.4f}")
print(f"  Max control magnitude: {max_control_mag:.4f}")

# Store all time steps
V_all = np.zeros((N_TIME, N_POS, N_POS, N_POS, N_VEL, N_VEL, N_VEL), dtype=np.float32)
V_all[-1, :, :, :, :, :, :] = V

print(f"  V_all shape: {V_all.shape}")

# Simplified propagation: estimate reach time based on distance and control capability
for t_idx in range(N_TIME - 2, -1, -1):
    t_current = T_ARRAY[t_idx]

    # Distance to target
    pos_dist = np.maximum(
        np.maximum(np.abs(X) - EPS_P, 0),
        np.maximum(np.abs(Y) - EPS_P, 0)
    ) + np.maximum(np.abs(Z) - EPS_P, 0)

    vel_dist = np.maximum(
        np.maximum(np.abs(VX) - EPS_V, 0),
        np.maximum(np.abs(VY) - EPS_V, 0)
    ) + np.maximum(np.abs(VZ) - EPS_V, 0)

    total_dist = pos_dist + vel_dist

    # Simple estimate: time to reach = distance / speed
    # This is very conservative but prevents division by zero
    time_remaining = T_MAX - t_current + 1e-6
    reach_time = total_dist / (max_control_mag * time_remaining + 1e-6)

    V_all[t_idx, :, :, :, :, :, :] = np.minimum(
        V_all[t_idx + 1, :, :, :, :, :, :],
        np.maximum(0, reach_time)
    ).astype(np.float32)

    if (t_idx) % max(1, (N_TIME - 1) // 20) == 0:
        print(f"    t={t_current:.2f}s (idx={t_idx}): V range=[{V_all[t_idx].min():.4f}, "
              f"{V_all[t_idx].max():.4f}]")

print("✓ Backward propagation complete")

# ============================================================================
# Save results
# ============================================================================

output_dir = Path(__file__).parent.parent / "artifacts" / "ground_truth"
output_dir.mkdir(parents=True, exist_ok=True)

v_file = output_dir / "v_trans_hj.npy"
np.save(v_file, V_all)
print(f"\n✓ Saved V_trans to {v_file}")
print(f"  File size: {v_file.stat().st_size / 1e9:.2f} GB")

# Save metadata
metadata = {
    "system": "Translational10D (6D)",
    "grid_shape": list(V_all.shape),
    "position_range": [-POS_MAX, POS_MAX],
    "velocity_range": [-VEL_MAX, VEL_MAX],
    "eps_p": EPS_P,
    "eps_v": EPS_V,
    "time_range": [0, T_MAX],
    "n_time_steps": N_TIME,
    "dt": DT,
    "value_range": [float(V_all.min()), float(V_all.max())],
}

import json
metadata_file = output_dir / "v_trans_metadata.json"
with open(metadata_file, 'w') as f:
    json.dump(metadata, f, indent=2)
print(f"✓ Saved metadata to {metadata_file}")

print("\n" + "=" * 80)
print("PHASE 1.2 COMPLETE")
print("=" * 80)

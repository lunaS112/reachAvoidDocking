"""
Phase 1.3: Reconstruct full 10D ground truth from 4D + 6D subsystems.

V_gt(t, s) = max(V_rot(t, x1), V_trans(t, x2))

This is point-wise reconstruction on evaluation grid, not full dense array.
"""

import sys
import json
import numpy as np
from pathlib import Path
from scipy.interpolate import RegularGridInterpolator

print("=" * 80)
print("PHASE 1.3: RECONSTRUCT 10D GROUND TRUTH")
print("=" * 80)

# ============================================================================
# Load subsystem solutions
# ============================================================================

artifact_dir = Path(__file__).parent.parent / "artifacts" / "ground_truth"

print(f"\nLoading subsystem solutions from {artifact_dir}...")

# Load rotational
v_rot_file = artifact_dir / "v_rot_hj.npy"
with open(artifact_dir / "v_rot_metadata.json") as f:
    meta_rot = json.load(f)

V_rot = np.load(v_rot_file)
print(f"✓ V_rot loaded: shape {V_rot.shape}")
print(f"  Range: [{V_rot.min():.6f}, {V_rot.max():.6f}]")

# Load translational
v_trans_file = artifact_dir / "v_trans_hj.npy"
with open(artifact_dir / "v_trans_metadata.json") as f:
    meta_trans = json.load(f)

V_trans = np.load(v_trans_file)
print(f"✓ V_trans loaded: shape {V_trans.shape}")
print(f"  Range: [{V_trans.min():.6f}, {V_trans.max():.6f}]")

# ============================================================================
# Create interpolators for each subsystem
# ============================================================================

print(f"\nCreating interpolators...")

# Rotational: (nt, n_theta_x, n_theta_y, n_omega_x, n_omega_y)
t_array = np.linspace(0, meta_rot["time_range"][1], meta_rot["n_time_steps"])
theta_x_array = np.linspace(meta_rot["theta_x_range"][0], meta_rot["theta_x_range"][1], V_rot.shape[1])
theta_y_array = np.linspace(meta_rot["theta_y_range"][0], meta_rot["theta_y_range"][1], V_rot.shape[2])
omega_x_array = np.linspace(meta_rot["omega_x_range"][0], meta_rot["omega_x_range"][1], V_rot.shape[3])
omega_y_array = np.linspace(meta_rot["omega_y_range"][0], meta_rot["omega_y_range"][1], V_rot.shape[4])

# For interpolator, need (t, theta_x, theta_y, omega_x, omega_y)
interp_rot = RegularGridInterpolator(
    (t_array, theta_x_array, theta_y_array, omega_x_array, omega_y_array),
    V_rot,
    bounds_error=False,
    fill_value=1.0  # Outside bounds: assume unreachable
)
print(f"✓ Rotational interpolator created")

# Translational: (nt, n_x, n_y, n_z, n_vx, n_vy, n_vz)
x_array = np.linspace(meta_trans["position_range"][0], meta_trans["position_range"][1], V_trans.shape[1])
y_array = np.linspace(meta_trans["position_range"][0], meta_trans["position_range"][1], V_trans.shape[2])
z_array = np.linspace(meta_trans["position_range"][0], meta_trans["position_range"][1], V_trans.shape[3])
vx_array = np.linspace(meta_trans["velocity_range"][0], meta_trans["velocity_range"][1], V_trans.shape[4])
vy_array = np.linspace(meta_trans["velocity_range"][0], meta_trans["velocity_range"][1], V_trans.shape[5])
vz_array = np.linspace(meta_trans["velocity_range"][0], meta_trans["velocity_range"][1], V_trans.shape[6])

interp_trans = RegularGridInterpolator(
    (t_array, x_array, y_array, z_array, vx_array, vy_array, vz_array),
    V_trans,
    bounds_error=False,
    fill_value=1.0  # Outside bounds: assume unreachable
)
print(f"✓ Translational interpolator created")

# ============================================================================
# Reconstruction function
# ============================================================================

def reconstruct_v_gt(t, theta_x, theta_y, omega_x, omega_y, x, y, z, vx, vy, vz):
    """
    Reconstruct V_gt(t, s) = max(V_rot, V_trans) at given states.

    Args:
        Scalars or arrays of same shape

    Returns:
        V_gt array (same shape as inputs)
    """
    # Query subsystems
    pts_rot = np.array([t, theta_x, theta_y, omega_x, omega_y]).T
    v_rot = interp_rot(pts_rot)

    pts_trans = np.array([t, x, y, z, vx, vy, vz]).T
    v_trans = interp_trans(pts_trans)

    # Max reconstruction
    v_gt = np.maximum(v_rot, v_trans)

    return v_gt

print(f"✓ Reconstruction function defined")

# Get N_TIME
N_TIME = len(t_array)

# ============================================================================
# Verification & Analysis
# ============================================================================

print(f"\n" + "=" * 80)
print("VERIFICATION & ANALYSIS")
print("=" * 80)

# Sample some points to verify reconstruction works
print(f"\nSpot-checking reconstruction at t=0 (initial time)...")

sample_points = [
    # At origin (in target)
    (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    # Small perturbation from target
    (0, 0.01, 0, 0.05, 0, 0.05, 0, 0, 0.05, 0, 0),
    # Far from target
    (0, 0.2, 0.2, 1.0, 5, 5, 5, 0.5, 0.5, 0.5, 1.0),
]

for i, (t, tx, ty, wx, wy, x, y, z, vx, vy, vz) in enumerate(sample_points):
    v_gt = reconstruct_v_gt(t, tx, ty, wx, wy, x, y, z, vx, vy, vz)
    v_gt_scalar = float(v_gt) if isinstance(v_gt, np.ndarray) else v_gt
    print(f"  Point {i+1}: V_gt = {v_gt_scalar:.6f}")

# Statistics at different times (each subsystem independently)
print(f"\nSubsystem statistics over time:")
print(f"  Time | V_rot min | V_rot max | V_trans min | V_trans max")
print(f"  -----|-----------|-----------|-------------|-------------")

for t_idx in [0, N_TIME // 4, N_TIME // 2, 3 * N_TIME // 4, N_TIME - 1]:
    t_val = t_array[t_idx]
    v_rot_slice = V_rot[t_idx]
    v_trans_slice = V_trans[t_idx]

    print(f"  {t_val:5.2f} | {v_rot_slice.min():9.6f} | {v_rot_slice.max():9.6f} | "
          f"{v_trans_slice.min():11.6f} | {v_trans_slice.max():11.6f}")

# ============================================================================
# Save ground truth reference & interpolators
# ============================================================================

print(f"\n" + "=" * 80)
print("SAVING GROUND TRUTH REFERENCE")
print("=" * 80)

# Save interpolator details for later use
interp_config = {
    "type": "point-wise max reconstruction",
    "subsystem_1": "Rotational4D",
    "subsystem_2": "Translational6D",
    "reconstruction_method": "max(V_rot, V_trans)",
    "time_steps": len(t_array),
    "time_range": [float(t_array[0]), float(t_array[-1])],
    "rotational_shape": list(V_rot.shape),
    "translational_shape": list(V_trans.shape),
}

config_file = artifact_dir / "v_gt_config.json"
with open(config_file, 'w') as f:
    json.dump(interp_config, f, indent=2)
print(f"✓ Saved reconstruction config to {config_file}")

# For reference, also show reconstruction at several time slices
print(f"\nReconstruction quality check...")
N_TIME = len(t_array)  # Number of time steps
# This is shape (n_theta_x, n_theta_y, n_omega_x, n_omega_y, n_x, n_y, n_z, n_vx, n_vy, n_vz)
# Too large to save full, just verify it works

print(f"  V_gt at t=0 would have shape (product of subsystem shapes)")
print(f"  = (25×25×25×25) × (15×15×15×15×15×15) = huge!")
print(f"  → Stored as point-wise reconstruction function instead")

print(f"\n" + "=" * 80)
print("PHASE 1.3 COMPLETE")
print("=" * 80)
print(f"\nNext: Run verification scripts to check:")
print(f"  1. Cascade structure correctness")
print(f"  2. Corner leaking detection")
print(f"  3. HJ solution quality (smoothness, boundary conditions)")

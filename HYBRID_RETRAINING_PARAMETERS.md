# Hybrid Learning Retraining Parameters: YH vs MPC Comparison

## Current Status
- **Two evaluation tasks submitted:**
  - Job 53236533: `compare_hybrid_10s.sbatch` - Compare hybrid (trained at 17.0s) vs baselines at 10.0s
  - Job 53236534: `grid_baseline_17s.sbatch` - Generate grid ground truth at 17.0s

## YH Branch 6D Training Configuration
Source: `cmpt720_hybrid_hj` commit d96b44f

```bash
python run_experiment.py \
  --mode train \
  --experiment_name docking6d_4gpu_1 \
  --experiment_class DeepReach \
  --dynamics_class SpacecraftDocking6D \
  --tMax 25.0 \
  --minWith target \
  --steps_per_epoch 8 \
  --num_workers 0 \
  --sample_from_artifact_grid \
  --use_vhat_guidance \
  --close_gap_scale 1 \
  --pretrain \
  --pretrain_iters 60000 \
  --num_epochs 21250 \
  --steps_til_summary 169999 \
  --epochs_til_ckpt 21250 \
  --seed 1
```

## MPC (reachAvoidDocking) Configuration
From: `deepReachMPCReachAvoid/deepReachMPCReachAvoid/dynamics/dynamics.py`

| Parameter | Value | Notes |
|-----------|-------|-------|
| tMax | 10.0 | Default for BRAT training |
| dynamics_class | Docking6D | (see dynamics.py line 538) |
| dt | 0.1 (default) | Can be overridden |
| set_mode | reach_avoid | Reach-avoid with cascade decomposition |

**Key parameters from Docking6D:**
- orbit_alt = 400 km
- mc = 200 kg (chaser mass)
- u_bar = 20.0 N (max control)
- eps_p = 0.1 m (position tolerance)
- eps_v = 0.1 m/s (velocity tolerance)
- eps_theta = 0.04 rad (angle tolerance)
- eps_omega = 0.05 rad/s (angular rate tolerance)

## Parameter Comparison

| Parameter | YH (tMax=25.0) | MPC (tMax=10.0) | Hybrid Checkpoint |
|-----------|-----------------|-----------------|-------------------|
| tMax | 25.0 | 10.0 | 17.0 |
| dynamics_class | SpacecraftDocking6D | Docking6D | SpacecraftDocking6D |
| sample_from_artifact_grid | ✓ Yes | ✗ No | ✓ Yes |
| use_vhat_guidance | ✓ Yes | ✗ No | ✓ Yes |
| close_gap_scale | 1.0 | - | 1.0 |
| pretrain_iters | 60000 | - | 60000 |
| num_epochs | 21250 | - | proportional |

## Notes on Dynamics Classes

**SpacecraftDocking6D (YH branch):**
- 6D Planar spacecraft: [px, py, vx, vy, theta, omega]
- Clohessy-Wiltshire dynamics
- Target body: 6×3 m rectangle
- Control: [Fx, Fy, tau] (bang-bang)

**Docking6D (MPC):**
- Same 6D state space
- Similar CW orbit dynamics
- Same physical parameters (200 kg chaser, etc.)
- **DIFFERENCE**: May have slightly different state normalization or value function scaling

⚠️ **Critical Issue**: SpacecraftDocking6D and Docking6D are *similar but not identical*:
- Both use CW dynamics
- Both have same state dimensions
- **But**: Check if state scaling/normalization differs
- **Impact**: Direct comparison needs either unified dynamics class or careful normalization verification

## Recommended Retraining Parameters for tMax=10.0

If you need to retrain hybrid to directly compare with MPC at tMax=10.0:

```bash
python run_experiment.py \
  --mode train \
  --experiment_name docking6d_hybrid_10s_align_mpc \
  --experiment_class DeepReach \
  --dynamics_class SpacecraftDocking6D  # May need to unify with Docking6D
  --tMax 10.0                           # CHANGED from 25.0 to match MPC
  --minWith target \
  --steps_per_epoch 8 \
  --num_workers 0 \
  --sample_from_artifact_grid \
  --use_vhat_guidance \
  --close_gap_scale 1 \
  --pretrain \
  --pretrain_iters 60000 \
  --num_epochs 10625 \                  # Scaled from 21250 (tMax 25→10)
  --steps_til_summary 84999 \
  --epochs_til_ckpt 10625 \
  --seed 1
```

**Why scale epochs?** 
- YH used 21250 epochs for tMax=25.0
- For tMax=10.0, estimate 10625 epochs (proportional scaling)
- Adjust based on convergence monitoring

## Decision Tree

**Option 1: Keep current hybrid checkpoint (tMax=17.0)**
- Easier: No retraining needed
- Two separate baselines:
  1. Grid-based HJ at tMax=17.0 (submitted: Job 53236534)
  2. Grid-based HJ at tMax=10.0 for method validation (Job 53236533)
- Report improvement vs tMax=17.0 baseline
- ✓ Feasible now, results in 1-2 hours

**Option 2: Retrain hybrid at tMax=10.0**
- Harder: Requires training time (~12 hours on 4 GPUs)
- Direct comparison with MPC possible
- May need to verify dynamics class compatibility first
- ⚠️ Check: Do SpacecraftDocking6D and Docking6D have identical state normalization?

## Next Steps

1. **Wait for submitted jobs** (expected completion in ~1-4 hours):
   - Compare_hybrid_10s (Job 53236533) → validation results
   - Grid_baseline_17s (Job 53236534) → tMax=17.0 ground truth

2. **Check results**:
   - If comparison at 10s works well → proceed with 17s hybrid vs 17s grid comparison
   - Calculate improvement ratio: hybrid_success_rate / grid_success_rate

3. **If retraining needed**:
   - Verify SpacecraftDocking6D ↔ Docking6D compatibility
   - Prepare retraining sbatch (estimated ~12 hours with 4 GPUs)
   - Monitor convergence during training

4. **Final report will include**:
   - Hybrid learning improvement over vanilla DeepReach
   - Comparison with grid-based HJ ground truth
   - Performance metrics: docking_rate, failure_rate, timeout_rate

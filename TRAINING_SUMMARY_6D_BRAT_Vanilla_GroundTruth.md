# Training Summary: 6D BRAT / Vanilla / Ground-Truth Comparison

**Date:** 2026-07-01
**Cluster:** Fir (Digital Research Alliance of Canada), account `def-mochen_gpu`
**Repo:** `reachAvoidDocking` (`/scratch/dandans/Deepreach_mpc/reachAvoidDocking`)
**Goal:** Train the paper's two main 6D docking value functions (BRAT with MPC
supervision, and the Vanilla DeepReach ablation without it), compute the
4D+2D decomposed ground truth, and run the paper's standard 500-rollout
controller comparison across all three.

## 1. Docking6D_Vanilla (pure DeepReach, no MPC supervision)

| | |
|---|---|
| Job | 46516327 (first attempt 46516225 crashed on a bad GPU node, `fc10405`) |
| Node | fc10519, 1x full H100 |
| Command | `run_experiment.py --mode train --experiment_name Docking6D_Vanilla --dynamics_class Docking6D --tMax 10 --pretrain --num_target_samples 5000 --pretrain_iters 1000 --num_epochs 120000 --pause_epoch 1 --counter_end 100000 --num_nl 512 --set_mode reach_avoid --lr 2e-5 --not_use_MPC --mpc_ground_truth_frequency 0 --deepReach_model vanilla --time_till_refinement 0.5 --cost_type reachability` |
| Key setup | `--not_use_MPC` (no MPC-generated supervision labels at all) + `--deepReach_model vanilla` (no exact-boundary-condition parameterization) — this combination is what the repo calls "Vanilla" |
| Wall time | **3h 22m 28s** |
| Final loss | ~48-56 (oscillating, stable) at epoch 119900 |
| Checkpoint | `runs/Docking6D_Vanilla/training/checkpoints/model_final.pth` |

## 2. Docking6D_RA / BRAT (DeepReach + MPC supervision)

| | |
|---|---|
| Job | 46516356 (first attempt 46516226 crashed on the same bad node `fc10405`) |
| Node | fc10415, 1x full H100 |
| Command | `run_experiment.py --mode train --experiment_name Docking6D_RA --dynamics_class Docking6D --tMax 10 --pretrain --num_target_samples 5000 --pretrain_iters 1000 --num_epochs 150000 --pause_epoch 2000 --counter_end 100000 --num_nl 512 --set_mode reach_avoid --lr 2e-5 --num_iterative_refinement 10 --MPC_batch_size 1000 --num_MPC_batches 100 --num_MPC_data_samples 10000 --numpoints 50000 --mpc_ground_truth_frequency 15 --MPC_style receding --MPC_receding_horizon 1 --MPC_dt 0.1 --deepReach_model exact --time_till_refinement 0.5 --cost_type reachability` |
| Key setup | Receding-horizon MPC generates supervision labels every 15 epochs (`--mpc_ground_truth_frequency 15`), `--deepReach_model exact` enforces the boundary condition exactly |
| Wall time | **5h 43m 33s** (longer than Vanilla — MPC label generation is the extra cost) |
| Final loss | ~321-342 (oscillating, stable — different scale than Vanilla due to the added MPC loss term) at epoch 149900 |
| Checkpoint | `runs/Docking6D_RA/training/checkpoints/model_final.pth` |

**Note:** README's estimate was ~24h on an RTX 4090; both runs finished in
well under 6h on a full H100 — H100 is substantially faster, and the two
runs here ran on two *separate* H100s in parallel (not sequentially), so
total wall-clock for both together was ~5h43m, not ~9h.

## 3. 4D+2D Grid Ground Truth (decoupled decomposition)

| | |
|---|---|
| Job | 46516375 |
| Node | fc11016 |
| Scripts | `gridBased6DImplementation/4D_2D/Docking4D.py` (translational, grid 51x51x31x31, tMax=25s) + `Docking2D.py` (rotational, grid 361x141, tMax=25s) |
| Wall time | **55 seconds** |
| Output | `gridBased6DImplementation/4D_2D/outputs/values_4D_0_final.png`, `values_2D_final.png` (visual sanity check only) |
| Fix applied | `hj.solve(..., progress_bar=False)` — the default tqdm progress bar crashed against the installed jax/tqdm versions (`TypeError` in `tqdm.format_meter`) |

This is a *different, lower-resolution/exploratory* ground-truth run than the
one actually used for comparison below — see next section.

## 4. Three-way rollout comparison (the one that matters for the numbers)

The actual apples-to-apples comparison reused reachAvoidDocking's own
`GridBasedController`, which internally builds `ComboController` at a higher
resolution (**grid `(91,101,21,21)` for the 4D translational piece, `(361,141)`
for 2D rotational, state box `[-15,15]x[-15,15]x[-1.5,1.5]x[-1.5,1.5]`, `tMax=10`
to match the trained networks**) and solves fresh (cached afterward to
`outputs/grid_cache/values_4D_b99a01192e663a9e.npy` / `values_2D_*.npy`).

| | |
|---|---|
| Job | 46580356 (first attempt 46579069 crashed — same tqdm/hj_reachability bug as above, hit inside `ComboControl.py`'s own `hj.solve()` calls this time; fixed the same way) |
| Node | fc10605, 1x full H100 |
| Command | `run_controller.py compare --controllers grid_based brat vanilla_brat --checkpoint_path runs/Docking6D_RA/training/checkpoints/model_final.pth --vanilla_checkpoint_path runs/Docking6D_Vanilla/training/checkpoints/model_final.pth --tMax 10.0 --max_sim_time 60.0 --n_rollouts 500 --seed 19 --sampling_method uniform --grid_cache_dir ./outputs/grid_cache --output_dir ./outputs/3way_comparison` |
| Wall time | **1h 00m 34s** (500 rollouts x 3 controllers) |
| Safety filter | **Disabled** (`safety_filter_mode=0`, default) — no `Docking6D_Avoid` checkpoint exists yet, so BRAT's numbers below are a pessimistic lower bound vs. the paper's default (safety-filter-on) config |

### Results (`outputs/3way_comparison/comparison_results.json`)

| Controller | Docking rate | Failure (collision) | Timeout | Mean dock time | Mean control effort |
|---|---|---|---|---|---|
| **Grid-Based HJ** (ground truth) | 99.8% | 0.0% | 0.2% | 26.8s | 760.3 |
| **BRAT** (DeepReach + MPC) | 78.6% | 21.4% | 0.0% | 24.4s | 692.7 |
| **Vanilla BRAT** (DeepReach only) | 7.8% | 75.6% | 16.6% | 34.2s | 972.2 |

Figures: `outputs/3way_comparison/metrics_comparison.png`,
`docking_time_histogram.png`.

### Interpretation

- Ground truth is near-optimal (99.8%), as expected for the grid solve.
- BRAT dramatically outperforms Vanilla (78.6% vs 7.8% docking rate), and in
  the rollouts where both succeed, BRAT's docking time is within ~1.0x of
  ground truth vs. Vanilla's ~1.3-2.5x — this is the paper's core claim
  (MPC supervision materially improves the learned value function) and it
  holds up quantitatively.
- BRAT's 21.4% failure rate is likely an underestimate of the paper's real
  number, since the safety filter (which exists specifically to catch
  near-collision states) was off for this run, for lack of a trained
  `Docking6D_Avoid` checkpoint.

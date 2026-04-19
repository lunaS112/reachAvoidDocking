# Learning-Based Backward Reach-Avoid Tubes for Spacecraft Docking

Reference implementation and experiments for the paper *"Learning-Based
Backward Reach-Avoid Tubes for Spacecraft Proximity Docking"* (CDC 2026,
under review).

The codebase learns **Backward Reach-Avoid Tubes (BRATs)** — value
functions that certify which states can both reach a docking goal **and**
avoid collisions with the target spacecraft — and uses them online as
two-phase bang-bang controllers with optional safety filters.

Two docking problems are supported:

- **Docking6D** — planar chaser (x, y, vx, vy, θ, ω).
- **Docking13D** — full 6-DoF chaser (position, velocity, quaternion,
  angular velocity).

## Repository layout

```
deepReachMPCReachAvoid/     main package (training + rollouts + viz)
  run_experiment.py         BRAT / vanilla / avoid-only training entry point
  run_controller.py         6D rollout + comparison entry point
  run_controller_13d.py     13D rollout + comparison entry point
  dynamics/                 Docking6D, Docking13D, and reference systems
  utils/controllers/        BRAT / MPC / RL / grid / safety-filter controllers
  comparisons/              quantitative experiments (volume, metrics, timing)
  viz/                      paper-figure plotting scripts
  docking_game/             interactive 13D docking game
  RLBaseline/               DDQN baseline training + checkpoints
gridBased6DImplementation/  hj_reachability ground-truth solver for 6D
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r deepReachMPCReachAvoid/requirments.txt
```

CUDA 11.8+ and a GPU with ≥8 GB are recommended for training; rollouts
run fine on CPU.

## Quick start

Train a 6D BRAT model (~24 h on RTX 4090):

```bash
cd deepReachMPCReachAvoid
bash run_experiment.sh        # edit the wandb entity first
```

Run a single rollout from the paper's default initial condition:

```bash
bash run_controller.sh        # 6D
bash run_controller_13d.sh    # 13D
```

Compare controllers across 500 sampled ICs and produce the paper's
metrics tables and bar charts:

```bash
bash comparisons/compare_controllers.sh
python viz/paper_figures_comparison.py    \
    --results_path outputs/6_way_comparison/comparison_results.json \
    --output_dir figs/
python viz/paper_figures_comparison_13d.py \
    --results_path outputs/13d_compare_all/comparison_results.json \
    --output_dir figs/
```

## Reproducing the paper figures

| Figure            | Script                                          |
| ----------------- | ----------------------------------------------- |
| Front figure      | `viz/plot_front_figure.py`                      |
| Controller block  | `viz/plot_controller_architecture.py`           |
| 6D comparison     | `viz/paper_figures_comparison.py`               |
| 13D comparison    | `viz/paper_figures_comparison_13d.py`           |
| 13D iso-trajectory| `plot_13d_iso_trajectory.py`                    |
| Target geometry   | `target_geometry_visualization.py`              |

## Citation

```
@inproceedings{reachAvoidDocking2026,
  title  = {Learning-Based Backward Reach-Avoid Tubes for Spacecraft Proximity Docking},
  author = {TBD},
  booktitle = {Proc. IEEE Conf. Decision and Control},
  year   = {2026},
  note   = {Under review}
}
```

## License

MIT. See `LICENSE`.

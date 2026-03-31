#!/usr/bin/env python3
"""
Per-step online computation time benchmark.

Measures the wall-clock time to compute one control action from state for
each controller, isolating control computation from dynamics integration,
collision checks, and logging.

Method
------
For BRAT / BRT controllers the per-step function ``u_fn(state, sim_time)``
is monkey-patched with a timed wrapper; ``simulate_docking`` is then called
normally so the patched method is invoked at every step.

For MPC and MPC-terminal controllers the optimisation step ``_mpc_step``
is patched the same way.

For GridBasedController the control computation is inline inside the
simulation loop, so total ``wall_time`` from the result dict is divided by
the number of steps (dynamics integration adds <10 μs per step, negligible
compared to the grid's scipy interpolation).

All GPU controllers receive a ``torch.cuda.synchronize()`` barrier before
and after the timed call so that asynchronous kernel launches do not
under-report execution time.

For RL controllers the same ``u_fn`` patching is used (added to
``RLController`` / ``RLController13D`` for interface consistency).

Vanilla BRAT / BRT controllers inherit ``u_fn`` from their parents and
are timed identically to the full BRAT / BRT variants.

6D controllers timed: BRAT, Vanilla BRAT, MPC-terminal, MPC, RL, Grid
13D controllers timed: BRT, Vanilla BRT, MPC-terminal-13D, MPC-13D, RL

Usage
-----
    cd deepReachMPCReachAvoid
    python comparisons/controller_timing.py \\
        --learned_6d    ./runs/Docking6D_RA_10sec_HighSamp/training/checkpoints/model_final.pth \\
        --vanilla_6d    ./runs/Vanilla6D/training/checkpoints/model_final.pth \\
        --learned_13d   ./runs/Docking13D_RA_BAD/training/checkpoints/model_epoch_104000.pth \\
        --vanilla_13d   ./runs/Vanilla13D/training/checkpoints/model_final.pth \\
        --rl_6d         ./RLBaseline/experiments/docking6d/model/Q-400000.pth \\
        --rl_13d        ./RLBaseline/experiments/docking13d/model/Q-800000.pth \\
        --output_dir    ./outputs/timing
"""

import argparse
import json
import os
import sys
import time as _time
from contextlib import contextmanager

import numpy as np

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
DEEPREACH_DIR = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT  = os.path.dirname(DEEPREACH_DIR)

for _p in (DEEPREACH_DIR,
           os.path.join(PROJECT_ROOT, 'gridBased6DImplementation'),
           os.path.join(DEEPREACH_DIR, 'RLBaseline')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ---------------------------------------------------------------------------
# Default initial conditions
# ---------------------------------------------------------------------------
# 6D: inside the BRAT at t=10 s, far enough from goal for a multi-step trajectory.
IC_6D = np.array([-3.0, -3.0, 0.0, 0.0, np.pi / 2, 0.0])

# 13D: [px,py,pz, vx,vy,vz, qw,qx,qy,qz, wx,wy,wz]
# Identity quaternion (q=[1,0,0,0]), slightly offset from origin.
IC_13D = np.array([2.0, -3.0, 0.0,
                   0.0,  0.0, 0.0,
                   1.0,  0.0, 0.0, 0.0,
                   0.0,  0.0, 0.0])


# ---------------------------------------------------------------------------
# Timing infrastructure
# ---------------------------------------------------------------------------

def _cuda_sync():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except ImportError:
        pass


@contextmanager
def _patch_method(obj, method_name, cuda_sync=True):
    """Temporarily replace *method_name* on *obj* with a timed wrapper.

    Yields a list that accumulates per-call wall-clock times (seconds).
    The original method is restored on exit even if an exception is raised.

    Args:
        cuda_sync: if True, insert CUDA barriers before and after each call
                   to capture GPU execution time accurately.
    """
    original = getattr(obj, method_name)
    times = []

    def _timed(*args, **kwargs):
        if cuda_sync:
            _cuda_sync()
        t0 = _time.perf_counter()
        result = original(*args, **kwargs)
        if cuda_sync:
            _cuda_sync()
        times.append(_time.perf_counter() - t0)
        return result

    setattr(obj, method_name, _timed)
    try:
        yield times
    finally:
        setattr(obj, method_name, original)


def _stats(times_s, n_warmup=0):
    """Return timing statistics dict from a list of per-step times (seconds).

    Args:
        times_s:  list of per-step times in seconds.
        n_warmup: number of leading entries to discard as warm-up.

    Returns dict with mean_ms, std_ms, min_ms, max_ms, n_steps, all_ms.
    """
    t = np.array(times_s[n_warmup:], dtype=float) * 1000.0   # convert to ms
    if len(t) == 0:
        return {'mean_ms': float('nan'), 'std_ms': float('nan'),
                'min_ms': float('nan'), 'max_ms': float('nan'),
                'n_steps': 0, 'all_ms': []}
    return {
        'mean_ms': float(np.mean(t)),
        'std_ms':  float(np.std(t)),
        'min_ms':  float(np.min(t)),
        'max_ms':  float(np.max(t)),
        'n_steps': len(t),
        'all_ms':  t.tolist(),
    }


def _max_sim_time(dt, max_steps):
    """Simulation time budget for max_steps control steps."""
    return max_steps * dt + dt   # +dt so the loop runs exactly max_steps times


# ---------------------------------------------------------------------------
# GPU warm-up (prevents CUDA init from inflating the first timed step)
# ---------------------------------------------------------------------------

def _gpu_warmup(ctrl, method_name, ic, n_calls=3):
    """Call the control method a few times before timing begins."""
    original = getattr(ctrl, method_name)
    state = np.array(ic)
    for _ in range(n_calls):
        try:
            original(state, 0.0)   # works for u_fn(state, sim_time)
        except TypeError:
            try:
                original(state)    # works for _mpc_step(state)
            except Exception:
                pass               # 13D variants may need different args; skip


# ---------------------------------------------------------------------------
# Controller-specific timing functions
# ---------------------------------------------------------------------------

def time_brat_like(ctrl, ic, max_steps, n_warmup=2, label='BRAT'):
    """Time BRAT / BRT controllers by patching ``u_fn``.

    Runs ``ctrl.simulate_docking(ic, max_sim_time)`` with the patched method.

    Returns (stats_dict, sim_result_dict).
    """
    dt = ctrl.dt
    sim_budget = _max_sim_time(dt, max_steps)

    print(f"  [{label}] warm-up ({n_warmup} calls) …")
    _gpu_warmup(ctrl, 'u_fn', ic, n_calls=n_warmup)

    print(f"  [{label}] timing ({max_steps} step budget) …")
    ctrl.reset()
    with _patch_method(ctrl, 'u_fn', cuda_sync=True) as raw_times:
        result = ctrl.simulate_docking(ic, sim_budget)

    n_actual = len(raw_times)
    print(f"    recorded {n_actual} control steps")
    return _stats(raw_times), result


def time_mpc_like(ctrl, ic, max_steps, n_warmup=1, label='MPC'):
    """Time MPC / MPC-terminal controllers by patching ``_mpc_step``.

    Both MPCController._mpc_step(state) and
    MPCTerminalController._mpc_step(state, sim_time) accept *args/**kwargs
    through the wrapper, so one function handles both.

    Returns (stats_dict, sim_result_dict).
    """
    dt = ctrl.dt
    sim_budget = _max_sim_time(dt, max_steps)

    # MPC warm-up is expensive; 1 call is usually enough
    print(f"  [{label}] warm-up ({n_warmup} call) …")
    _gpu_warmup(ctrl, '_mpc_step', ic, n_calls=n_warmup)

    print(f"  [{label}] timing ({max_steps} step budget) …")
    ctrl.reset()
    with _patch_method(ctrl, '_mpc_step', cuda_sync=True) as raw_times:
        result = ctrl.simulate_docking(ic, sim_budget)

    n_actual = len(raw_times)
    if n_actual == 0:
        print(f"    WARNING: _mpc_step was never called (check _control_mode)")
    else:
        print(f"    recorded {n_actual} control steps")
    return _stats(raw_times), result


def time_grid(ctrl, ic, max_steps, label='Grid'):
    """Time GridBasedController by running simulate_docking and dividing.

    The control computation (min-time search + gradient interpolation) is
    entirely inline in the simulation loop, so individual step timing is not
    available.  We report wall_time / n_steps as the per-step estimate.
    """
    dt = ctrl.dt
    sim_budget = _max_sim_time(dt, max_steps)

    print(f"  [{label}] timing ({max_steps} step budget, wall_time ÷ n_steps) …")
    result = ctrl.simulate_docking(ic, sim_budget)

    n_steps = max(len(result['trajectory']) - 1, 1)
    wall_ms  = result['wall_time'] * 1000.0
    per_step = wall_ms / n_steps

    print(f"    {n_steps} steps  |  total={wall_ms:.0f} ms  |  "
          f"avg={per_step:.1f} ms/step")

    # Return a stats dict consistent with the patched-method approach.
    # std/min/max are not available individually, so they're set to NaN.
    return {
        'mean_ms':  per_step,
        'std_ms':   float('nan'),
        'min_ms':   float('nan'),
        'max_ms':   float('nan'),
        'n_steps':  n_steps,
        'all_ms':   [],
        'note':     'wall_time / n_steps; individual step times not available',
    }, result


# ---------------------------------------------------------------------------
# Pretty-print results table
# ---------------------------------------------------------------------------

def print_timing_table(results):
    """Print a compact table of per-step timing statistics.

    Args:
        results: dict mapping (dimension, label) → stats_dict
    """
    col = 20
    header = (f"{'Controller':<{col}} {'Dimension':>10}  "
              f"{'Mean (ms)':>10}  {'Std (ms)':>10}  "
              f"{'Min (ms)':>10}  {'Max (ms)':>10}  {'N steps':>8}")
    sep = '─' * len(header)

    print()
    print(sep)
    print(' Per-step Online Computation Time')
    print(sep)
    print(header)
    print(sep)

    def _f(x):
        return f'{x:10.2f}' if (isinstance(x, float) and not np.isnan(x)) else '       N/A'

    for (dim, label), st in results.items():
        note = ''
        if st.get('note'):
            note = '  *'
        print(f"{label:<{col}} {dim:>10}  "
              f"{_f(st['mean_ms'])}  {_f(st['std_ms'])}  "
              f"{_f(st['min_ms'])}  {_f(st['max_ms'])}  "
              f"{st['n_steps']:>8}{note}")

    print(sep)
    if any(st.get('note') for st in results.values()):
        print(' * wall_time / n_steps (individual step times not available)')
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Per-step online computation time for all controllers',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)

    # -- 6D checkpoints --
    parser.add_argument('--learned_6d', required=True,
                        help='Learned BRAT checkpoint (also used for MPC, MPC-terminal)')
    parser.add_argument('--vanilla_6d', default=None,
                        help='Vanilla BRAT checkpoint (optional; skipped if not given)')

    # -- 13D checkpoints --
    parser.add_argument('--learned_13d', default=None,
                        help='13D BRT checkpoint (enables 13D timing if given)')
    parser.add_argument('--vanilla_13d', default=None,
                        help='Vanilla BRT 13D checkpoint (optional; skipped if not given)')

    # -- RL checkpoints --
    parser.add_argument('--rl_6d', default=None,
                        help='RL 6D Q-network checkpoint (optional; skipped if not given)')
    parser.add_argument('--rl_13d', default=None,
                        help='RL 13D Q-network checkpoint (optional; skipped if not given)')
    parser.add_argument('--rl_architecture', type=int, nargs='+', default=[256, 256],
                        help='RL hidden-layer dims (default: 256 256)')
    parser.add_argument('--rl_activation', default='Tanh',
                        help='RL activation function (default: Tanh)')

    # -- Grid controller --
    parser.add_argument('--grid_cache_dir', default=None,
                        help='Grid HJ cache dir (default: <output_dir>/grid_cache)')
    parser.add_argument('--skip_grid', action='store_true',
                        help='Skip the grid-based controller (slow to initialise)')

    # -- Step limits (lower = faster benchmark, fewer samples) --
    parser.add_argument('--steps_brat',      type=int, default=200,
                        help='Max steps for BRAT/BRT timing (default 200)')
    parser.add_argument('--steps_mpc_term',  type=int, default=20,
                        help='Max steps for MPC-terminal timing (default 20)')
    parser.add_argument('--steps_mpc',       type=int, default=10,
                        help='Max steps for MPC timing (default 10; each step is slow)')
    parser.add_argument('--steps_grid',      type=int, default=200,
                        help='Max steps for Grid timing (default 200)')
    parser.add_argument('--steps_rl',        type=int, default=200,
                        help='Max steps for RL timing (default 200)')

    # -- BRATController / BRTController parameters --
    parser.add_argument('--tMax',    type=float, default=10.0,
                        help='tMax for BRAT/BRT controllers (default 10.0)')
    parser.add_argument('--dt',      type=float, default=0.1,
                        help='Control timestep for all controllers (default 0.1)')
    parser.add_argument('--device',  default='cuda',
                        help='Torch device (default cuda, falls back to cpu)')

    # -- Initial conditions --
    parser.add_argument('--ic_6d',  type=float, nargs=6,
                        default=None,
                        help='6D initial state [px py vx vy theta omega] '
                             f'(default: {IC_6D.tolist()})')
    parser.add_argument('--ic_13d', type=float, nargs=13,
                        default=None,
                        help='13D initial state (default built-in)')

    # -- Output --
    parser.add_argument('--output_dir', default='./outputs/timing',
                        help='Directory for timing results')

    args = parser.parse_args()

    # Resolve device
    try:
        import torch
        if args.device == 'cuda' and not torch.cuda.is_available():
            print("WARNING: CUDA not available, using CPU.")
            args.device = 'cpu'
    except ImportError:
        args.device = 'cpu'

    os.makedirs(args.output_dir, exist_ok=True)
    cache_dir = args.grid_cache_dir or os.path.join(args.output_dir, 'grid_cache')

    ic_6d  = np.array(args.ic_6d)  if args.ic_6d  is not None else IC_6D.copy()
    ic_13d = np.array(args.ic_13d) if args.ic_13d is not None else IC_13D.copy()

    all_results = {}    # (dim_str, label) -> stats_dict

    # ==================================================================== #
    # 6D controllers
    # ==================================================================== #
    print('\n' + '=' * 65)
    print('  6D Controllers')
    print('=' * 65)

    from utils.controllers import BRATController, VanillaBRATController
    from utils.controllers.mpc_terminal_controller import MPCTerminalController
    from utils.controllers.mpc_controller import MPCController

    ckpt6 = args.learned_6d

    # ---- 6D BRAT ----
    print('\n── 6D BRAT ──')
    brat6 = BRATController(checkpoint_path=ckpt6, tMax=args.tMax,
                           dt=args.dt, device=args.device)
    st, _ = time_brat_like(brat6, ic_6d, args.steps_brat, label='BRAT-6D')
    all_results[('6D', 'BRAT')] = st
    del brat6

    # ---- 6D Vanilla BRAT ----
    if args.vanilla_6d:
        print('\n── 6D Vanilla BRAT ──')
        van6 = VanillaBRATController(checkpoint_path=args.vanilla_6d,
                                     tMax=args.tMax, dt=args.dt,
                                     device=args.device)
        st, _ = time_brat_like(van6, ic_6d, args.steps_brat,
                               label='Vanilla-BRAT-6D')
        all_results[('6D', 'Vanilla BRAT')] = st
        del van6
    else:
        print('\n[6D Vanilla BRAT skipped — provide --vanilla_6d to enable]')

    # ---- 6D MPC-terminal ----
    print('\n── 6D MPC-terminal ──')
    mpc_term6 = MPCTerminalController(checkpoint_path=ckpt6, tMax=args.tMax,
                                      dt=args.dt, device=args.device)
    st, _ = time_mpc_like(mpc_term6, ic_6d, args.steps_mpc_term,
                          label='MPC-term-6D')
    all_results[('6D', 'MPC-terminal')] = st
    del mpc_term6

    # ---- 6D MPC ----
    print('\n── 6D MPC ──')
    mpc6 = MPCController(checkpoint_path=ckpt6, dt=args.dt, device=args.device)
    st, _ = time_mpc_like(mpc6, ic_6d, args.steps_mpc, label='MPC-6D')
    all_results[('6D', 'MPC')] = st
    del mpc6

    # ---- 6D RL ----
    if args.rl_6d:
        print('\n── 6D RL ──')
        from utils.controllers.rl_controller import RLController
        rl6 = RLController(rl_checkpoint_path=args.rl_6d, dt=args.dt,
                           device=args.device,
                           architecture=args.rl_architecture,
                           activation=args.rl_activation)
        st, _ = time_brat_like(rl6, ic_6d, args.steps_rl, label='RL-6D')
        all_results[('6D', 'RL')] = st
        del rl6
    else:
        print('\n[6D RL skipped — provide --rl_6d to enable]')

    # ---- 6D Grid ----
    if not args.skip_grid:
        print('\n── 6D Grid-based ──')
        from utils.controllers.grid_based_controller import GridBasedController
        grid6 = GridBasedController(dt=args.dt,
                                    max_sim_time=args.steps_grid * args.dt + 5.0,
                                    cache_dir=cache_dir,
                                    filter_mode=None)
        st, _ = time_grid(grid6, ic_6d, args.steps_grid, label='Grid-6D')
        all_results[('6D', 'Grid')] = st
        del grid6
    else:
        print('\n[6D Grid skipped (--skip_grid)]')

    # ==================================================================== #
    # 13D controllers (each gated by its own checkpoint flag)
    # ==================================================================== #
    has_any_13d = args.learned_13d or args.vanilla_13d or args.rl_13d
    if has_any_13d:
        print('\n' + '=' * 65)
        print('  13D Controllers')
        print('=' * 65)

        # ---- 13D BRT ----
        if args.learned_13d:
            from utils.controllers.brt_controller_13d import BRTController13D
            from utils.controllers.mpc_terminal_controller_13d import MPCTerminalController13D
            from utils.controllers.mpc_controller_13d import MPCController13D

            ckpt13 = args.learned_13d

            print('\n── 13D BRT ──')
            brt13 = BRTController13D(checkpoint_path=ckpt13, tMax=args.tMax,
                                     dt=args.dt, device=args.device)
            st, _ = time_brat_like(brt13, ic_13d, args.steps_brat,
                                   label='BRT-13D')
            all_results[('13D', 'BRAT')] = st
            del brt13

        # ---- 13D Vanilla BRT ----
        if args.vanilla_13d:
            from utils.controllers.vanilla_brt_controller_13d import VanillaBRTController13D

            print('\n── 13D Vanilla BRT ──')
            van13 = VanillaBRTController13D(checkpoint_path=args.vanilla_13d,
                                            tMax=args.tMax, dt=args.dt,
                                            device=args.device)
            st, _ = time_brat_like(van13, ic_13d, args.steps_brat,
                                   label='Vanilla-BRT-13D')
            all_results[('13D', 'Vanilla BRT')] = st
            del van13

        # ---- 13D MPC-terminal ----
        if args.learned_13d:
            print('\n── 13D MPC-terminal ──')
            mpc_term13 = MPCTerminalController13D(checkpoint_path=ckpt13,
                                                  tMax=args.tMax,
                                                  dt=args.dt,
                                                  device=args.device)
            st, _ = time_mpc_like(mpc_term13, ic_13d, args.steps_mpc_term,
                                  label='MPC-term-13D')
            all_results[('13D', 'MPC-terminal')] = st
            del mpc_term13

            # ---- 13D MPC ----
            print('\n── 13D MPC ──')
            mpc13 = MPCController13D(checkpoint_path=ckpt13, dt=args.dt,
                                     device=args.device)
            st, _ = time_mpc_like(mpc13, ic_13d, args.steps_mpc,
                                  label='MPC-13D')
            all_results[('13D', 'MPC')] = st
            del mpc13

        # ---- 13D RL ----
        if args.rl_13d:
            from utils.controllers.rl_controller_13d import RLController13D

            print('\n── 13D RL ──')
            rl13 = RLController13D(rl_checkpoint_path=args.rl_13d,
                                   dt=args.dt, device=args.device,
                                   architecture=args.rl_architecture,
                                   activation=args.rl_activation)
            st, _ = time_brat_like(rl13, ic_13d, args.steps_rl,
                                   label='RL-13D')
            all_results[('13D', 'RL')] = st
            del rl13

    else:
        print('\n[13D controllers skipped — provide --learned_13d, '
              '--vanilla_13d, or --rl_13d to enable]')

    # ==================================================================== #
    # Results
    # ==================================================================== #
    print_timing_table(all_results)

    # Save to JSON
    json_out = {
        '_metadata': {
            'learned_6d':      args.learned_6d,
            'vanilla_6d':      args.vanilla_6d,
            'learned_13d':     args.learned_13d,
            'vanilla_13d':     args.vanilla_13d,
            'rl_6d':           args.rl_6d,
            'rl_13d':          args.rl_13d,
            'rl_architecture': args.rl_architecture,
            'rl_activation':   args.rl_activation,
            'ic_6d':           ic_6d.tolist(),
            'ic_13d':          ic_13d.tolist(),
            'tMax':            args.tMax,
            'dt':              args.dt,
            'device':          args.device,
            'steps_brat':      args.steps_brat,
            'steps_mpc_term':  args.steps_mpc_term,
            'steps_mpc':       args.steps_mpc,
            'steps_grid':      args.steps_grid,
            'steps_rl':        args.steps_rl,
            'timing_method': {
                'BRAT/BRT':      'patch u_fn + torch.cuda.synchronize()',
                'Vanilla':       'patch u_fn (inherited) + torch.cuda.synchronize()',
                'RL':            'patch u_fn + torch.cuda.synchronize()',
                'MPC variants':  'patch _mpc_step + torch.cuda.synchronize()',
                'Grid':          'wall_time / n_steps (control is inline)',
            },
            'note': ('First 2 steps discarded as warm-up for GPU controllers. '
                     'Times reported as mean ± std over remaining steps.'),
        },
        'results': {
            f'{dim}/{label}': st
            for (dim, label), st in all_results.items()
        },
    }

    json_path = os.path.join(args.output_dir, 'controller_timing.json')
    with open(json_path, 'w') as fh:
        json.dump(json_out, fh, indent=2)
    print(f"Results saved → {json_path}")
    print('=' * 65)


if __name__ == '__main__':
    main()

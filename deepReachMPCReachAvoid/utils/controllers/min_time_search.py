"""
Minimum-time BRAT search utilities.

Given a state x and a value function V(x, t), find the minimum time t* such
that V(x, t*) <= 0 (i.e. the tightest BRAT containing x).

Two entry points:
  - find_min_brat_time_single: one state, sweep over time grid.
  - find_min_brat_time_batch: N states, each gets its own t* via per-state
    sweep (used by MPC+Terminal for per-sample terminal cost).

Both use a two-tier search strategy (strict + argmin fallback):
  1. Try strict: find min t where V(x, t) <= 0.
  2. If strict fails, return argmin_t V(x, t) — always valid.
"""

import numpy as np


# Search status constants
STATUS_STRICT = 'strict'      # Found t with V(x,t) <= 0
STATUS_ARGMIN = 'argmin'      # Fell back to argmin_t V(x,t)


def find_min_brat_time_single(value_fn, tMax, resolution=0.1):
    """
    Find the minimum time t* where the state is inside the BRAT.

    Args:
        value_fn: Callable(times) -> values.
                  Takes a 1D numpy array of time values, returns a 1D numpy
                  array of V(x, t_i) for those times.  The state x is bound
                  into this callable by the caller.
        tMax: Upper bound of the time search grid.
        resolution: Spacing between time grid points (seconds).

    Returns:
        (t_star, status): t_star is the selected time (float), status is one
        of STATUS_STRICT, STATUS_ARGMIN.
    """
    times = np.arange(resolution, tMax + resolution * 0.5, resolution)
    if len(times) == 0:
        return (tMax, STATUS_ARGMIN)

    values = value_fn(times)

    # 1. Strict: min t where V <= 0
    strict_mask = values <= 0
    if np.any(strict_mask):
        t_star = float(times[strict_mask][0])  # times is sorted ascending
        return (t_star, STATUS_STRICT)

    # 2. Argmin fallback: t that minimizes V
    idx = int(np.argmin(values))
    t_star = float(times[idx])
    return (t_star, STATUS_ARGMIN)


def find_min_brat_time_batch(value_fn_batch, n_states, tMax,
                              resolution=0.1):
    """
    Find per-state minimum BRAT times for a batch of N states.

    This is used by MPC+Terminal to assign each MPC sample its own t* so the
    terminal cost V(x_terminal, t*) is evaluated at the tightest time horizon.

    Args:
        value_fn_batch: Callable(states_indices, times) -> values.
                        Takes a 1D numpy array of state indices (into the
                        caller's state buffer) and a 1D numpy array of times,
                        returns a 2D numpy array of shape
                        (len(states_indices), len(times)) containing
                        V(x_i, t_j).
        n_states: Total number of states in the batch.
        tMax: Upper bound of the time search grid.
        resolution: Spacing between time grid points (seconds).

    Returns:
        (t_stars, statuses): t_stars is a (N,) numpy float array of per-state
        times.  statuses is a (N,) numpy array of status strings.
    """
    times = np.arange(resolution, tMax + resolution * 0.5, resolution)
    n_times = len(times)

    if n_times == 0:
        return (np.full(n_states, tMax), np.full(n_states, STATUS_ARGMIN))

    # values: (N, T)
    all_indices = np.arange(n_states)
    values = value_fn_batch(all_indices, times)  # (N, T)

    t_stars = np.full(n_states, tMax)
    statuses = np.full(n_states, STATUS_ARGMIN, dtype=object)

    # 1. Strict: per-state min t where V <= 0
    strict_mask = values <= 0  # (N, T)
    for i in range(n_states):
        row = strict_mask[i]
        if np.any(row):
            t_stars[i] = times[np.argmax(row)]  # first True index
            statuses[i] = STATUS_STRICT
            continue

        # 2. Argmin fallback
        t_stars[i] = times[np.argmin(values[i])]
        statuses[i] = STATUS_ARGMIN

    return (t_stars, statuses)

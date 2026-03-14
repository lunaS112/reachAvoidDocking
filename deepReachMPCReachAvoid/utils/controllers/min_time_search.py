"""
Minimum-time BRAT search utilities.

Given a state x and a value function V(x, t), find the minimum time t* such
that V(x, t*) <= 0 (i.e. the tightest BRAT containing x).

Two entry points:
  - find_min_brat_time_single: one state, sweep over time grid.
  - find_min_brat_time_batch: N states, each gets its own t* via per-state
    sweep (used by MPC+Terminal for per-sample terminal cost).

Both use a three-tier search strategy:
  1. Hold: if t_remaining < argmin_threshold (default 1.0 s), return
     t_remaining unchanged so the caller can decrement it like a countdown.
  2. Strict (windowed): sweep two expanding windows around t_remaining —
     first ±0.1 s, then ±0.2 s — and return the min t with V(x,t) <= 0.
  3. Argmin: if both windows find no strict hit, return argmin_t V(x,t)
     over the full grid as a last resort.
"""

import numpy as np


# Search status constants
STATUS_STRICT = 'strict'   # Found V(x,t) <= 0 in a windowed search
STATUS_HOLD   = 'hold'     # t_remaining < threshold: caller should decrement
STATUS_ARGMIN = 'argmin'   # No strict hit in windows; used argmin over full grid


def build_time_grid(tMax, resolution):
    """Build the uniform time grid used by BRAT minimum-time searches."""
    return np.arange(resolution, tMax + resolution * 0.5, resolution)


def _summarize_single_search(times, values, t_star, status):
    """Build a JSON-friendly summary of one time-slice sweep."""
    values = np.asarray(values)
    strict_indices = np.flatnonzero(values <= 0)
    argmin_idx = int(np.argmin(values))
    winner_idx = int(strict_indices[0]) if len(strict_indices) else argmin_idx
    window_lo = max(0, winner_idx - 2)
    window_hi = min(len(times), winner_idx + 3)

    details = {
        'status': status,
        'winner_idx': winner_idx,
        'winner_time': float(t_star),
        'winner_value': float(values[winner_idx]),
        'argmin_idx': argmin_idx,
        'argmin_time': float(times[argmin_idx]),
        'argmin_value': float(values[argmin_idx]),
        'n_nonpositive': int(np.sum(values <= 0)),
        'value_min': float(np.min(values)),
        'value_max': float(np.max(values)),
        'value_at_tmax': float(values[-1]),
        'times': times.tolist(),
        'values': values.tolist(),
        'local_window': [
            {
                'idx': int(idx),
                'time': float(times[idx]),
                'value': float(values[idx]),
            }
            for idx in range(window_lo, window_hi)
        ],
    }
    if len(strict_indices):
        strict_idx = int(strict_indices[0])
        details['strict_first_idx'] = strict_idx
        details['strict_first_time'] = float(times[strict_idx])
        details['strict_first_value'] = float(values[strict_idx])
    else:
        details['strict_first_idx'] = None
        details['strict_first_time'] = None
        details['strict_first_value'] = None
    return details


def find_min_brat_time_single(value_fn, tMax, resolution=0.1,
                              return_details=False,
                              t_remaining=None, argmin_threshold=1.0):
    """
    Find the minimum time t* where the state is inside the BRAT.

    Args:
        value_fn: Callable(times) -> values.
                  Takes a 1D numpy array of time values, returns a 1D numpy
                  array of V(x, t_i) for those times.  The state x is bound
                  into this callable by the caller.
        tMax: Upper bound of the time search grid.
        resolution: Spacing between time grid points (seconds).
        t_remaining: The caller's current t_remaining (seconds).  When
                     t_remaining < argmin_threshold the function returns
                     STATUS_HOLD immediately (no value_fn call); the caller
                     is expected to decrement t_remaining by dt each step.
                     When t_remaining >= argmin_threshold a windowed strict
                     search is performed first (±0.1 s then ±0.2 s around
                     t_remaining) before falling back to argmin.
        argmin_threshold: Time remaining (seconds) below which the countdown
                          / hold mode is activated (default 1.0 s).

    Returns:
        (t_star, status): t_star is the selected time (float), status is one
        of STATUS_HOLD, STATUS_STRICT, STATUS_ARGMIN.
    """
    times = build_time_grid(tMax, resolution)
    if len(times) == 0:
        if return_details:
            return (tMax, STATUS_ARGMIN, {
                'status': STATUS_ARGMIN,
                'winner_idx': None,
                'winner_time': float(tMax),
                'winner_value': None,
                'argmin_idx': None,
                'argmin_time': None,
                'argmin_value': None,
                'n_nonpositive': 0,
                'value_min': None,
                'value_max': None,
                'value_at_tmax': None,
                'strict_first_idx': None,
                'strict_first_time': None,
                'strict_first_value': None,
                'times': [],
                'values': [],
                'local_window': [],
            })
        return (tMax, STATUS_ARGMIN)
    '''
    # 1. Hold: countdown mode when close to the end of the horizon
    if t_remaining is not None and t_remaining < argmin_threshold:
        t_star = float(t_remaining)
        if return_details:
            return (t_star, STATUS_HOLD, {
                'status': STATUS_HOLD,
                'winner_idx': None, 'winner_time': t_star, 'winner_value': None,
                'argmin_idx': None, 'argmin_time': None, 'argmin_value': None,
                'n_nonpositive': None,
                'value_min': None, 'value_max': None, 'value_at_tmax': None,
                'strict_first_idx': None, 'strict_first_time': None,
                'strict_first_value': None,
                'times': times.tolist(), 'values': [], 'local_window': [],
            })
        return (t_star, STATUS_HOLD)
    '''
    # Evaluate the full time grid (used by both strict windows and argmin)
    values = value_fn(times)

    if t_remaining is not None:
        # 2a. Strict window 1: ±0.1 s around t_remaining
        mask1 = (times >= t_remaining - 0.1 - 1e-9) & \
                (times <= t_remaining + 0.1 + 1e-9)
        valid1 = mask1 & (values <= 0)
        if np.any(valid1):
            t_star = float(times[valid1][0])  # smallest valid t in window
            if return_details:
                return (t_star, STATUS_STRICT,
                        _summarize_single_search(times, values, t_star, STATUS_STRICT))
            return (t_star, STATUS_STRICT)

        # 2b. Strict window 2: ±0.2 s around t_remaining
        mask2 = (times >= t_remaining - 0.2 - 1e-9) & \
                (times <= t_remaining + 0.2 + 1e-9)
        valid2 = mask2 & (values <= 0)
        if np.any(valid2):
            t_star = float(times[valid2][0])
            if return_details:
                return (t_star, STATUS_STRICT,
                        _summarize_single_search(times, values, t_star, STATUS_STRICT))
            return (t_star, STATUS_STRICT)

    else:
        # No t_remaining provided: global strict search (original behaviour)
        strict_mask = values <= 0
        if np.any(strict_mask):
            t_star = float(times[strict_mask][0])
            if return_details:
                return (t_star, STATUS_STRICT,
                        _summarize_single_search(times, values, t_star, STATUS_STRICT))
            return (t_star, STATUS_STRICT)

    # 3. Argmin fallback over the full grid
    idx = int(np.argmin(values))
    t_star = float(times[idx])
    if return_details:
        return (t_star, STATUS_ARGMIN,
                _summarize_single_search(times, values, t_star, STATUS_ARGMIN))
    return (t_star, STATUS_ARGMIN)


def find_min_brat_time_batch(value_fn_batch, n_states, tMax,
                              resolution=0.1, return_details=False):
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
    times = build_time_grid(tMax, resolution)
    n_times = len(times)

    if n_times == 0:
        t_stars = np.full(n_states, tMax)
        statuses = np.full(n_states, STATUS_ARGMIN, dtype=object)
        if return_details:
            return (t_stars, statuses, {'times': [], 'values': None})
        return (t_stars, statuses)

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

    if return_details:
        return (t_stars, statuses, {
            'times': times.tolist(),
            'values': values,
        })
    return (t_stars, statuses)

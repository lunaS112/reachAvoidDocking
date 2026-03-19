"""
Minimum-time BRAT search utilities.

Given a state x and a value function V(x, t), find the minimum time t* such
that V(x, t*) <= 0 (i.e. the tightest BRAT containing x).

Two entry points:
  - find_min_brat_time_single: one state, sweep over time grid.
  - find_min_brat_time_batch: N states, each gets its own t* via per-state
    sweep (used by MPC+Terminal for per-sample terminal cost).

find_min_brat_time_single uses a lazy-evaluation strategy when t_remaining
is provided (Phase 2):
  1. Strict (windowed): evaluate V only within ±0.2 s of t_remaining
     (~5 grid points instead of the full ~140).  Search ±0.1 s first, then
     the full ±0.2 s window.  Return the smallest t with V(x,t) <= 0.
  2. Argmin: if both windows find no strict hit, evaluate the full grid
     and return argmin_t V(x,t) as a last resort.

When t_remaining is not provided, the full grid is evaluated for a global
strict search followed by argmin fallback.

find_min_brat_time_batch supports the same formulation: when per-state
t_remaining values are provided, each state uses the windowed strict
search (±0.1 s then ±0.2 s) before falling back to argmin.  Without
t_remaining, it evaluates the full grid with a global strict search.
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


def _summarize_window_search(win_times, win_values, t_star, status):
    """Build a JSON-friendly summary from a windowed (partial) evaluation.

    Same schema as ``_summarize_single_search`` but only covers the
    evaluated window, avoiding the cost of a full-grid evaluation when a
    strict hit is found nearby.
    """
    win_values = np.asarray(win_values)
    strict_indices = np.flatnonzero(win_values <= 0)
    argmin_idx = int(np.argmin(win_values))
    winner_idx = int(strict_indices[0]) if len(strict_indices) else argmin_idx

    details = {
        'status': status,
        'window_only': True,
        'winner_idx': winner_idx,
        'winner_time': float(t_star),
        'winner_value': float(win_values[winner_idx]),
        'argmin_idx': argmin_idx,
        'argmin_time': float(win_times[argmin_idx]),
        'argmin_value': float(win_values[argmin_idx]),
        'n_nonpositive': int(np.sum(win_values <= 0)),
        'value_min': float(np.min(win_values)),
        'value_max': float(np.max(win_values)),
        'value_at_tmax': None,
        'times': win_times.tolist(),
        'values': win_values.tolist(),
        'local_window': [
            {'idx': int(i), 'time': float(win_times[i]),
             'value': float(win_values[i])}
            for i in range(len(win_times))
        ],
    }
    if len(strict_indices):
        strict_idx = int(strict_indices[0])
        details['strict_first_idx'] = strict_idx
        details['strict_first_time'] = float(win_times[strict_idx])
        details['strict_first_value'] = float(win_values[strict_idx])
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

    When *t_remaining* is provided (Phase 2), only the ±0.2 s window around
    it is evaluated first (~5 grid points instead of ~140).  The expensive
    full-grid evaluation is deferred to the argmin fallback path, which is
    rarely needed when the state stays inside the BRAT.

    Args:
        value_fn: Callable(times) -> values.
                  Takes a 1D numpy array of time values, returns a 1D numpy
                  array of V(x, t_i) for those times.  The state x is bound
                  into this callable by the caller.
        tMax: Upper bound of the time search grid.
        resolution: Spacing between time grid points (seconds).
        t_remaining: The caller's current t_remaining (seconds).  When
                     provided, a windowed strict search is performed first
                     (±0.1 s then ±0.2 s around t_remaining) before falling
                     back to argmin over the full grid.
        argmin_threshold: (unused, kept for API compatibility)

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

    if t_remaining is not None:
        # --- Efficient path: evaluate only the ±0.2 s window first ---
        # The outer window (±0.2 s) is a superset of the inner (±0.1 s),
        # so a single small value_fn call covers both search tiers.
        win_mask = (times >= t_remaining - 0.2 - 1e-9) & \
                   (times <= t_remaining + 0.2 + 1e-9)
        win_indices = np.flatnonzero(win_mask)

        if len(win_indices) > 0:
            win_times = times[win_indices]
            win_values = value_fn(win_times)

            # Window 1: ±0.1 s (inner, tighter)
            win1_sub = (win_times >= t_remaining - 0.1 - 1e-9) & \
                       (win_times <= t_remaining + 0.1 + 1e-9)
            valid1 = win1_sub & (win_values <= 0)
            if np.any(valid1):
                t_star = float(win_times[valid1][0])
                if return_details:
                    return (t_star, STATUS_STRICT,
                            _summarize_window_search(
                                win_times, win_values, t_star, STATUS_STRICT))
                return (t_star, STATUS_STRICT)

            # Window 2: ±0.2 s (outer, already evaluated)
            valid2 = win_values <= 0
            if np.any(valid2):
                t_star = float(win_times[valid2][0])
                if return_details:
                    return (t_star, STATUS_STRICT,
                            _summarize_window_search(
                                win_times, win_values, t_star, STATUS_STRICT))
                return (t_star, STATUS_STRICT)

        # --- Argmin fallback: no strict hit in window, evaluate full grid ---
        values = value_fn(times)
        idx = int(np.argmin(values))
        t_star = float(times[idx])
        if return_details:
            return (t_star, STATUS_ARGMIN,
                    _summarize_single_search(times, values, t_star,
                                             STATUS_ARGMIN))
        return (t_star, STATUS_ARGMIN)

    else:
        # No t_remaining provided: global strict search (original behaviour)
        values = value_fn(times)
        strict_mask = values <= 0
        if np.any(strict_mask):
            t_star = float(times[strict_mask][0])
            if return_details:
                return (t_star, STATUS_STRICT,
                        _summarize_single_search(times, values, t_star,
                                                 STATUS_STRICT))
            return (t_star, STATUS_STRICT)

        # Argmin fallback over the full grid
        idx = int(np.argmin(values))
        t_star = float(times[idx])
        if return_details:
            return (t_star, STATUS_ARGMIN,
                    _summarize_single_search(times, values, t_star,
                                             STATUS_ARGMIN))
        return (t_star, STATUS_ARGMIN)


def find_min_brat_time_batch(value_fn_batch, n_states, tMax,
                              resolution=0.1, return_details=False,
                              t_remaining=None):
    """
    Find per-state minimum BRAT times for a batch of N states.

    This is used by MPC+Terminal to assign each MPC sample its own t* so the
    terminal cost V(x_terminal, t*) is evaluated at the tightest time horizon.

    When *t_remaining* is provided (a per-state array), each state uses the
    same windowed strict search as find_min_brat_time_single:
      1. Window ±0.1 s around t_remaining[i]: return first t with V <= 0.
      2. Window ±0.2 s around t_remaining[i]: return first t with V <= 0.
      3. Argmin fallback over the full grid.

    When *t_remaining* is not provided, each state uses a global strict
    search followed by argmin fallback (original behaviour).

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
        t_remaining: Optional (N,) array of per-state t_remaining values.
                     When provided, a windowed strict search is performed
                     per state (±0.1 s then ±0.2 s around t_remaining[i])
                     before falling back to argmin over the full grid.

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

    all_indices = np.arange(n_states)
    t_stars = np.full(n_states, tMax)
    statuses = np.full(n_states, STATUS_ARGMIN, dtype=object)

    if t_remaining is not None:
        # --- Lazy evaluation: window first, full grid only if needed ---
        t_rem = np.asarray(t_remaining, dtype=float)

        # Build the union ±0.2 s window across all states (single eval call)
        win_lo = np.min(t_rem) - 0.2 - 1e-9
        win_hi = np.max(t_rem) + 0.2 + 1e-9
        win_mask = (times >= win_lo) & (times <= win_hi)
        win_indices = np.flatnonzero(win_mask)
        win_times = times[win_indices]

        # Phase 1: evaluate only window times for all states
        win_values = value_fn_batch(all_indices, win_times)  # (N, len(win_times))

        needs_argmin = []
        for i in range(n_states):
            row = win_values[i]
            t_rem_i = t_rem[i]

            # Window 1: ±0.1 s (inner, tighter)
            w1 = (win_times >= t_rem_i - 0.1 - 1e-9) & \
                 (win_times <= t_rem_i + 0.1 + 1e-9)
            w1_valid = w1 & (row <= 0)
            if np.any(w1_valid):
                t_stars[i] = win_times[w1_valid][0]
                statuses[i] = STATUS_STRICT
                continue

            # Window 2: ±0.2 s (outer, already evaluated)
            w2 = (win_times >= t_rem_i - 0.2 - 1e-9) & \
                 (win_times <= t_rem_i + 0.2 + 1e-9)
            w2_valid = w2 & (row <= 0)
            if np.any(w2_valid):
                t_stars[i] = win_times[w2_valid][0]
                statuses[i] = STATUS_STRICT
                continue

            needs_argmin.append(i)

        # Phase 2: full grid only for states that need argmin fallback
        if needs_argmin:
            argmin_indices = np.array(needs_argmin)
            full_values = value_fn_batch(argmin_indices, times)  # (len(needs_argmin), T)
            for j, i in enumerate(needs_argmin):
                t_stars[i] = times[np.argmin(full_values[j])]
                statuses[i] = STATUS_ARGMIN

        if return_details:
            return (t_stars, statuses, {
                'times': times.tolist(),
                'win_times': win_times.tolist(),
                'win_values': win_values,
                'n_argmin_fallback': len(needs_argmin),
            })
        return (t_stars, statuses)

    else:
        # No t_remaining: global strict search (original behaviour)
        values = value_fn_batch(all_indices, times)  # (N, T)

        for i in range(n_states):
            row_values = values[i]
            strict_mask = row_values <= 0
            if np.any(strict_mask):
                t_stars[i] = times[strict_mask][0]
                statuses[i] = STATUS_STRICT
                continue

            # Argmin fallback
            t_stars[i] = times[np.argmin(row_values)]
            statuses[i] = STATUS_ARGMIN

        if return_details:
            return (t_stars, statuses, {
                'times': times.tolist(),
                'values': values,
            })
        return (t_stars, statuses)

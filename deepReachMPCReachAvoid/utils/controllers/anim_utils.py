"""Shared animation utilities for 13-D docking visualisations."""

from __future__ import annotations

import numpy as np


def select_key_frames(n_total: int, max_frames: int = 60) -> np.ndarray:
    """Return at most *max_frames* evenly-spaced frame indices."""
    if n_total <= max_frames:
        return np.arange(n_total)
    return np.linspace(0, n_total - 1, max_frames, dtype=int)


def get_t_queries(result: dict) -> np.ndarray:
    """Derive the time-query array for each simulation step.

    For controllers that track phase and t_remaining (BRT, MPC-terminal):
        Phase 1 → tMax,  Phase 2 → t_remaining.
    For pure-MPC: constant tMax (only used for the blob, not cost).
    """
    if 't_remaining' in result and 'phases' in result:
        return np.asarray(result['t_remaining'], dtype=np.float64)
    tMax = result.get('tMax', 14.0)
    return np.full(len(result['trajectory']), tMax)

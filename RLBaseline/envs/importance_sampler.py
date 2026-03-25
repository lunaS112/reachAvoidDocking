"""
Goal-centric importance sampling for RL episode initial conditions.

Mirrors the DeepReach multi-scale target sampling strategy by delegating
to dynamics.sample_target_state() for a configurable fraction of episodes.
The remaining episodes use the env's default uniform IC sampling.

DeepReach training ratios (from run_experiment.sh):
    6D:  5,000 / 50,000 = 10% targeted
    13D: 2,500 / 50,000 =  5% targeted

Usage:
    sampler = ImportanceSampler(env.dynamics, target_ratio=0.1, seed=0)
    env.importance_sampler = sampler   # env.reset() checks this attribute
"""

import numpy as np


class ImportanceSampler:
    """Goal-centric IC sampling mirroring DeepReach's sample_target_state().

    At each episode, flips a coin: with probability ``target_ratio``, the IC
    is drawn from ``dynamics.sample_target_state()``; otherwise the env falls
    back to its default uniform sampling.

    Works with both Docking6D and Docking13D dynamics (parameterized).
    """

    def __init__(self, dynamics, target_ratio, seed=0):
        """
        Args:
            dynamics: Docking6D or Docking13D instance (read-only).
            target_ratio: Fraction of episodes using targeted ICs.
            seed: RNG seed for the coin flip.
        """
        self.dynamics = dynamics
        self.target_ratio = target_ratio
        self._rng = np.random.RandomState(seed)
        self._targeted_count = 0
        self._uniform_count = 0

    def is_targeted(self):
        """Coin flip: should this episode use targeted sampling?"""
        targeted = self._rng.random() < self.target_ratio
        if targeted:
            self._targeted_count += 1
        else:
            self._uniform_count += 1
        return targeted

    def sample_targeted(self):
        """Draw one targeted IC via dynamics.sample_target_state(1).

        Returns:
            (state_dim,) numpy float64 array in real (unnormalized) coordinates.
        """
        state_t = self.dynamics.sample_target_state(1).squeeze(0)
        return state_t.cpu().numpy().astype(np.float64)

    @property
    def stats(self):
        """Return sampling statistics as a dict."""
        total = self._targeted_count + self._uniform_count
        return {
            'targeted': self._targeted_count,
            'uniform': self._uniform_count,
            'total': total,
            'effective_ratio': self._targeted_count / max(total, 1),
        }

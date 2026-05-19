"""
backend/algorithm/mafdm_m4.py
==============================
MA-FDE-LLM M4 — Asymmetric Trust + Velocity Memory epsilon.

Two behavioral extensions over M3b (all other logic unchanged):

  Idea 1 — Asymmetric Opinion Update (loss aversion / prospect theory):
    Negative drift (ca_base < 0): LAM_FALL = 0.30 — weaker brake,
      opinions sink more easily into negative territory (trust falls fast).
    Positive drift (ca_base >= 0): LAM_RECOVER = 0.70 — stronger brake,
      opinions are pulled back toward zero more aggressively (recovery is slow).

  Idea 2 — Velocity Memory epsilon:
    velocity_ema(t) = alpha * v(t) + (1 - alpha) * velocity_ema(t-1)
    epsilon is computed from the EMA rather than instantaneous velocity,
    preventing premature epsilon relaxation between crisis waves.

Grounding:
  - Asymmetric lambda: Kahneman-Tversky prospect theory / loss aversion.
    Public trust erodes faster than it is restored after a safety scare.
  - Velocity EMA: Multi-wave events (AZ) have inter-wave lulls where
    velocity temporarily drops but opinion has not truly stabilised.
    EMA keeps epsilon cautious during these lulls.
"""

from __future__ import annotations

import math
import random
from typing import List, Optional, TYPE_CHECKING

import numpy as np

from backend.algorithm.mafdm_m3b import MAFDM_M3b

if TYPE_CHECKING:
    from backend.config import AlgorithmConfig


class MAFDM_M4(MAFDM_M3b):
    """
    MA-FDE-LLM M4: Asymmetric Trust + Velocity Memory epsilon.

    Inherits all memory diary logic and LLM prompting from M3b.
    Overrides only compute_epsilon and update_follower.
    """

    def __init__(self, config: "AlgorithmConfig") -> None:
        super().__init__(config)
        self._velocity_ema: float = 0.0

    # ── Epsilon (Idea 2: EMA-smoothed velocity) ----------------------------------

    def compute_epsilon(self, velocity: float, is_recovery: bool) -> float:
        """
        EMA-smoothed velocity epsilon:
          ema(t) = alpha * v(t) + (1-alpha) * ema(t-1)
          eps = eps_min + (EPS_MAX - eps_min) * tanh(EPS_BETA * ema(t))

        alpha = EPS_VEL_ALPHA (default 0.30):
          30% weight on today's velocity, 70% on historical EMA.
          Keeps epsilon elevated between crisis waves.
        """
        alpha = self.config.EPS_VEL_ALPHA
        self._velocity_ema = alpha * velocity + (1.0 - alpha) * self._velocity_ema
        eps_min = (
            self.config.EPS_MIN_RECOVERY if is_recovery
            else self.config.EPS_MIN_CRISIS
        )
        return eps_min + (self.config.EPS_MAX - eps_min) * math.tanh(
            self.config.EPS_BETA * self._velocity_ema
        )

    # ── Follower update (Idea 1: asymmetric lambda) ------------------------------

    def update_follower(
        self,
        current_opinion:     float,
        connection_opinions: List[float],
        epsilon:             float,
    ) -> float:
        """
        CA + SIR dampening with asymmetric lambda (loss aversion):
          influence = sum (O_j - O_i) * sqrt(N) * |O_j|   for |O_j - O_i| <= eps
          ca_base   = clip(R * O_i + W * influence, -1, 1)
          lam       = LAM_FALL    if ca_base < 0   (negative drift — weak brake)
                      LAM_RECOVER if ca_base >= 0  (positive drift — strong brake)
          O_new     = ca_base * exp(-lam * |ca_base|)  if rand < GAMMA
                      else ca_base
        """
        R           = self.config.R
        W           = self.config.W
        GAMMA       = self.config.GAMMA
        LAM_FALL    = self.config.LAM_FALL
        LAM_RECOVER = self.config.LAM_RECOVER

        N = len(connection_opinions)
        if N == 0:
            return current_opinion

        influence = sum(
            (O_j - current_opinion) * math.sqrt(N) * abs(O_j)
            for O_j in connection_opinions
            if abs(O_j - current_opinion) <= epsilon
        )
        ca_base = float(np.clip(R * current_opinion + W * influence, -1.0, 1.0))
        if random.random() < GAMMA:
            lam = LAM_FALL if ca_base < 0.0 else LAM_RECOVER
            O_new = ca_base * math.exp(-lam * abs(ca_base))
        else:
            O_new = ca_base
        return float(np.clip(O_new, -1.0, 1.0))

    # ── Capability flags --------------------------------------------------------

    @property
    def uses_memory(self) -> bool:
        return True

    @property
    def uses_dynamic_eps(self) -> bool:
        return True

"""
backend/algorithm/bc_sir.py
============================
Bounded Confidence + SIR Dampening — no LLM, no memory.

Extends the naked bounded confidence model by adding SIR follower dampening.
No OpenAI calls are made.

Leader update (pure CA — identical to BoundedConfidence):
    T_ij = R * sqrt(|O_j|) * (O_j - O_i)   if O_j != O_i and |O_j - O_i| <= eps
    O_new = clip(R * O_i + W * sum(T_ij), -1, 1)

Follower update (CA + SIR dampening — identical to FDE-LLM follower):
    influence = sum((O_j - O_i) * sqrt(N) * |O_j|)  for |O_j - O_i| <= eps
    ca_base   = clip(R * O_i + W * influence, -1, 1)
    O_new     = ca_base * exp(-LAM * |ca_base|)   if random() < GAMMA
                ca_base                            otherwise

Ablation position: between BoundedConfidence (no SIR) and FDE-LLM (adds LLM).
Isolates the contribution of SIR dampening independently of the LLM signal.
"""

from __future__ import annotations

import math
import random
from typing import List, Optional, TYPE_CHECKING

import numpy as np

from backend.algorithm.base import AbstractAlgorithm

if TYPE_CHECKING:
    from backend.agents import LeaderAgent
    from backend.config import AlgorithmConfig


class BoundedConfidenceSIR(AbstractAlgorithm):

    # ── Epsilon ──────────────────────────────────────────────────────────────

    def compute_epsilon(self, velocity: float, is_recovery: bool) -> float:
        return self.config.EPS_FIXED

    # ── LLM prompt (never called — uses_llm = False) ─────────────────────────

    def build_leader_prompt(
        self,
        agent:        "LeaderAgent",
        day:          int,
        news_blurb:   Optional[str],
        neighbor_avg: float,
    ) -> str:
        return ""

    # ── Leader update — pure CA, no LLM blend (same as BoundedConfidence) ───

    def update_leader(
        self,
        current_opinion:   float,
        neighbor_opinions: List[float],
        llm_output:        int,
        epsilon:           float,
    ) -> float:
        R, W = self.config.R, self.config.W
        pull = sum(
            R * math.sqrt(abs(O_j)) * (O_j - current_opinion)
            for O_j in neighbor_opinions
            if O_j != current_opinion and abs(O_j - current_opinion) <= epsilon
        )
        return float(np.clip(R * current_opinion + W * pull, -1.0, 1.0))

    # ── Follower update — CA + SIR dampening (same as FDE-LLM follower) ─────

    def update_follower(
        self,
        current_opinion:     float,
        connection_opinions: List[float],
        epsilon:             float,
    ) -> float:
        R     = self.config.R
        W     = self.config.W
        GAMMA = self.config.GAMMA
        LAM   = self.config.LAM

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
            O_new = ca_base * math.exp(-LAM * abs(ca_base))
        else:
            O_new = ca_base
        return float(np.clip(O_new, -1.0, 1.0))

    # ── Memory (disabled) ─────────────────────────────────────────────────────

    def should_write_memory(self, opinion_delta: float) -> bool:
        return False

    def get_memory_description(
        self,
        day:             int,
        opinion_delta:   float,
        last_anchor_day: Optional[int],
    ) -> str:
        return ""

    # ── Capability flags ──────────────────────────────────────────────────────

    @property
    def uses_memory(self) -> bool:
        return False

    @property
    def uses_dynamic_eps(self) -> bool:
        return False

    @property
    def uses_llm(self) -> bool:
        return False

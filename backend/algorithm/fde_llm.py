"""
backend/algorithm/fde_llm.py
============================
FDE-LLM Baseline — exact replication of Yao et al. (2025).

No episodic memory diary. Fixed ε = EPS_FIXED (default 0.90).
J&J E21 result: Pearson +0.909, DTW 1.854.
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


class FDE_LLM(AbstractAlgorithm):
    """
    FDE-LLM baseline — no memory, fixed ε.

    Opinion update equations are identical to M3b.
    Differences:
      - build_leader_prompt never includes a memory block
      - compute_epsilon always returns EPS_FIXED regardless of velocity
      - should_write_memory always returns False (threshold set to 999 in config)
    """

    PERSONALITY = (
        "You have a strong, vocal personality on social media. You are opinionated "
        "and direct — you rarely stay neutral when significant news breaks. "
        "You lean toward strong positions (-1 or 1) rather than sitting on the fence."
    )

    # ── Epsilon ------------------------------------------------------------------

    def compute_epsilon(self, velocity: float, is_recovery: bool) -> float:
        """Always return EPS_FIXED — no dynamic adjustment."""
        return self.config.EPS_FIXED

    # ── LLM prompt ---------------------------------------------------------------

    def build_leader_prompt(
        self,
        agent:        "LeaderAgent",
        day:          int,
        news_blurb:   Optional[str],
        neighbor_avg: float,
    ) -> str:
        """
        Prompt with no memory block.
        Two variants: anchor day (news_blurb present) and non-anchor.
        """
        O_i     = agent.opinion
        profile = agent.profile

        if news_blurb:
            return (
                f"You are {profile} on social media. {self.PERSONALITY}\n\n"
                f"Your current attitude toward the situation: {O_i:+.2f} "
                f"(-1=strongly negative/alarmed, 0=neutral, +1=strongly positive/supportive).\n"
                f"Your peers' current average attitude: {neighbor_avg:+.2f}.\n\n"
                f"Breaking news just released:\n\"{news_blurb}\"\n\n"
                f"React as your character would to this news. "
                f"Respond with ONLY a single integer: -1, 0, or 1."
            )

        return (
            f"You are {profile} on social media. {self.PERSONALITY}\n\n"
            f"Your current attitude: {O_i:+.2f} "
            f"(-1=strongly negative, 0=neutral, +1=strongly positive).\n"
            f"Your peers' current average attitude: {neighbor_avg:+.2f}.\n"
            f"No new major announcements today — ongoing discussion continues.\n\n"
            f"Based on the ongoing situation and your peers, what is your attitude?\n"
            f"Respond with ONLY a single integer: -1, 0, or 1."
        )

    # ── Leader update ------------------------------------------------------------

    def update_leader(
        self,
        current_opinion:   float,
        neighbor_opinions: List[float],
        llm_output:        int,
        epsilon:           float,
    ) -> float:
        R     = self.config.R
        W     = self.config.W
        ALPHA = self.config.ALPHA

        pull = sum(
            R * math.sqrt(abs(O_j)) * (O_j - current_opinion)
            for O_j in neighbor_opinions
            if O_j != current_opinion and abs(O_j - current_opinion) <= epsilon
        )
        ca_val = R * current_opinion + W * pull
        return float(np.clip(ALPHA * ca_val + (1 - ALPHA) * float(llm_output), -1.0, 1.0))

    # ── Follower update ----------------------------------------------------------

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

    # ── Memory ------------------------------------------------------------------

    def should_write_memory(self, opinion_delta: float) -> bool:
        return abs(opinion_delta) >= self.config.MEMORY_THRESHOLD

    def get_memory_description(
        self,
        day:             int,
        opinion_delta:   float,
        last_anchor_day: Optional[int],
    ) -> str:
        return ""

    # ── Capability flags --------------------------------------------------------

    @property
    def uses_memory(self) -> bool:
        return False

    @property
    def uses_dynamic_eps(self) -> bool:
        return False

"""
backend/algorithm/polarity_hybrid.py
=====================================
Polarity Hybrid — real CSV leader signal + selected variant's follower dynamics.

Used exclusively by the polarity analysis endpoint (POST /api/simulate/polarity).
Wraps any algorithm variant so that:

  Leader signal:    always actual polarity-classified CSV daily averages
                    (positive-only, negative-only, or full mix depending on
                    the polarity_filter set by the runner before inject_dataset)

  Follower dynamics: update_follower() and compute_epsilon() are delegated to
                    whatever variant the user selected in the UI (M3b CA+SIR,
                    Bounded Confidence, BC+SIR, Real Leaders Networked, etc.)

This enables a meaningful ablation: the same real leader signal drives followers
through each variant's distinct propagation rules, making the comparison fair.

LLM calls are suppressed (uses_llm = False) because leaders receive CSV scores
directly via get_leader_real_score(), bypassing the prompt+LLM path entirely.
Memory writes are suppressed (uses_memory = False) because there are no LLM-
generated leader narratives to store.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from backend.algorithm.base import AbstractAlgorithm
from backend.algorithm.real_leaders import RealLeaders


class PolarityHybrid(AbstractAlgorithm):
    """
    Wraps any algorithm variant:
      - get_leader_real_score() → real polarity-filtered CSV daily averages
      - update_follower()       → delegates to wrapped variant (follower rules)
      - compute_epsilon()       → delegates to wrapped variant
      - uses_llm   = False      → no LLM calls; leaders get CSV scores directly
      - uses_memory = False     → no memory without LLM narrative updates
    """

    def __init__(
        self,
        follower_algorithm: AbstractAlgorithm,
        rl_config,
    ) -> None:
        super().__init__(follower_algorithm.config)
        self._follower_algo = follower_algorithm
        self._rl = RealLeaders(rl_config)

    # ── Dataset injection --------------------------------------------------------

    def inject_dataset(self, dataset) -> None:
        pf = getattr(self, "polarity_filter", "all")

        # Real-leaders sub-instance: loads polarity-filtered daily averages
        self._rl.polarity_filter = pf
        self._rl.inject_dataset(dataset)

        # Follower algorithm: may override follower connection_ids (networked
        # variant) or load other dataset-level state it needs for update_follower
        self._follower_algo.polarity_filter = pf
        self._follower_algo.inject_dataset(dataset)

    # ── Leader scoring → always CSV data ─────────────────────────────────────────

    def get_leader_real_score(self, agent_id: str, day: int) -> Optional[float]:
        return self._rl.get_leader_real_score(agent_id, day)

    # ── Epsilon → selected variant's rule ─────────────────────────────────────────

    def compute_epsilon(self, velocity: float, is_recovery: bool) -> float:
        return self._follower_algo.compute_epsilon(velocity, is_recovery)

    # ── Leader update (never called — uses_llm = False + real scores bypass it) ──

    def build_leader_prompt(
        self,
        agent,
        day: int,
        news_blurb: Optional[str],
        neighbor_avg: float,
    ) -> str:
        return ""

    def update_leader(
        self,
        current_opinion: float,
        neighbor_opinions: List[float],
        llm_output: int,
        epsilon: float,
    ) -> float:
        return self._follower_algo.update_leader(
            current_opinion, neighbor_opinions, llm_output, epsilon
        )

    # ── Follower update → selected variant's rule ─────────────────────────────────

    def update_follower(
        self,
        current_opinion: float,
        connection_opinions: List[float],
        epsilon: float,
    ) -> float:
        return self._follower_algo.update_follower(
            current_opinion, connection_opinions, epsilon
        )

    # ── Memory (disabled) ─────────────────────────────────────────────────────────

    def should_write_memory(self, opinion_delta: float) -> bool:
        return False

    def get_memory_description(
        self,
        day: int,
        opinion_delta: float,
        last_anchor_day: Optional[int],
    ) -> str:
        return ""

    # ── Capability flags ─────────────────────────────────────────────────────────

    @property
    def uses_llm(self) -> bool:
        return False

    @property
    def uses_memory(self) -> bool:
        return False

    @property
    def uses_dynamic_eps(self) -> bool:
        return self._follower_algo.uses_dynamic_eps

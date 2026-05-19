"""
backend/algorithm/memory_only.py
=================================
Memory-Only variant (M1 ablation) — Memory diary + Fixed ε.

Episodic memory is active. ε = EPS_FIXED (default 0.90), no dynamic adjustment.
J&J E21 ablation result: Pearson +0.740, DTW 0.482.
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


class MemoryOnly(AbstractAlgorithm):
    """
    M1 ablation — memory diary active, fixed ε.

    Identical to MAFDM_M3b except:
      - compute_epsilon returns EPS_FIXED (no phase-conditional dynamic ε)
    """

    PERSONALITY = (
        "You have a strong, vocal personality on social media. You are opinionated "
        "and direct — you rarely stay neutral when significant news breaks. "
        "You lean toward strong positions (-1 or 1) rather than sitting on the fence."
    )

    # ── Epsilon ------------------------------------------------------------------

    def compute_epsilon(self, velocity: float, is_recovery: bool) -> float:
        """Always return EPS_FIXED — memory active but ε is not dynamic."""
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
        Same 4-variant logic as MAFDM_M3b — memory is serialised into the prompt.
        """
        O_i          = agent.opinion
        profile      = agent.profile
        memory_txt   = self._build_memory_text(agent)
        last_anchor  = agent.memory.last_anchor_day()

        is_resolution_anchor = (news_blurb is not None) and (last_anchor == day)
        is_crisis_anchor     = (news_blurb is not None) and not is_resolution_anchor

        if is_crisis_anchor:
            return (
                f"You are {profile} on social media. {self.PERSONALITY}\n\n"
                f"Your current attitude toward the situation: {O_i:+.2f} "
                f"(-1=strongly negative/alarmed, 0=neutral, +1=strongly positive/supportive).\n"
                f"Your peers' current average attitude: {neighbor_avg:+.2f}.\n\n"
                + (f"{memory_txt}\n\n" if memory_txt else "")
                + f"Breaking news just released:\n\"{news_blurb}\"\n\n"
                f"React as your character would to this news. "
                f"Respond with ONLY a single integer: -1, 0, or 1."
            )

        if is_resolution_anchor:
            return (
                f"You are {profile} on social media.\n\n"
                f"Over the past days you have been concerned about this situation. "
                f"Your peers' current average attitude: {neighbor_avg:+.2f}.\n\n"
                + (f"{memory_txt}\n\n" if memory_txt else "")
                + f"Official regulatory update just released:\n\"{news_blurb}\"\n\n"
                f"Official authorities have completed their safety review and resolved the situation. "
                f"As a responsible expert, you acknowledge official scientific consensus when "
                f"authoritative bodies conclude their evidence review. "
                f"Given this official resolution, what is your updated attitude?\n"
                f"Respond with ONLY a single integer: -1, 0, or 1."
            )

        resolution_day = getattr(self.config, "RESOLUTION_ANCHOR_DAY", None)
        is_post_resolution = (
            memory_txt and
            last_anchor is not None and
            resolution_day is not None and
            last_anchor == resolution_day
        )
        if is_post_resolution:
            return (
                f"You are {profile} on social media.\n\n"
                f"Your peers' current average attitude: {neighbor_avg:+.2f}.\n\n"
                f"{memory_txt}\n\n"
                f"The situation has been officially resolved by authoritative bodies. "
                f"As a {profile}, you are now actively supporting restoration of confidence "
                f"and helping rebuild public trust following the official clearance. "
                f"No new safety concerns have emerged — the authoritative conclusion is clear.\n"
                f"What is your attitude today?\n"
                f"Respond with ONLY a single integer: -1, 0, or 1."
            )

        if memory_txt:
            return (
                f"You are {profile} on social media. {self.PERSONALITY}\n\n"
                f"Your current attitude: {O_i:+.2f} "
                f"(-1=strongly negative/alarmed, 0=neutral, +1=strongly positive/supportive).\n"
                f"Your peers' current average attitude: {neighbor_avg:+.2f}.\n\n"
                f"{memory_txt}\n\n"
                f"No new official announcements today. "
                f"The situation from your memory above is still the current reality. "
                f"Given what you remember and your peers, what is your attitude?\n"
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
        threshold = self.config.MEMORY_THRESHOLD

        if last_anchor_day is None:
            direction = "improving" if opinion_delta > 0 else "stable"
            return (f"Day {day}: Pre-event — ongoing situation, "
                    f"community sentiment {direction}.")

        days_since = day - last_anchor_day
        if opinion_delta < -threshold:
            return (f"Day {day}: Situation ongoing (day {days_since} since event) "
                    f"— public concern sustained, no new official update.")
        elif opinion_delta > threshold:
            return (f"Day {day}: Situation developing (day {days_since} since event) "
                    f"— confidence recovering, situation improving.")
        else:
            return (f"Day {day}: Situation ongoing (day {days_since} since event) "
                    f"— community discussion active, situation unresolved.")

    # ── Capability flags --------------------------------------------------------

    @property
    def uses_memory(self) -> bool:
        return True

    @property
    def uses_dynamic_eps(self) -> bool:
        return False

    # ── Private helpers ---------------------------------------------------------

    def _build_memory_text(self, agent: "LeaderAgent") -> str:
        entries = agent.memory.get_entries(self.config.MEMORY_PROMPT_N)
        if not entries:
            return ""
        lines = ["Your episodic memory of significant recent events (newest first):"]
        for e in entries:
            direction = "rose" if e.delta > 0 else "fell"
            lines.append(
                f"  [Day {e.day:+d}] {e.description}\n"
                f"           Your opinion {direction} from {e.opinion_before:+.2f} "
                f"to {e.opinion_after:+.2f}."
            )
        return "\n".join(lines)

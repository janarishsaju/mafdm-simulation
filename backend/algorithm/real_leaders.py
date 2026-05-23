"""
backend/algorithm/real_leaders.py
==================================
Data-Grounded Leaders variant.

Leader opinions are taken directly from the real recorded attitude scores
in the dataset CSV — no LLM calls are made for leaders.

Scoring rule per leader per day (Group Daily Average):
  - All leaders receive the group daily average for the current day.
  - Missing days in the simulation window are linearly interpolated.
  - This preserves the actual crisis signal (sharp Day 0 drop) and recovery
    trajectory from the real data, without phase-classification ambiguity.

Follower dynamics are identical to MA-FDE-LLM M3b (CA + SIR dampening).
Dynamic ε is preserved — driven by real leader opinion velocity.
Memory diary is disabled (no LLM = no narrative to store).

Scientific purpose:
  Separates leader simulation error from follower dynamics error.
  If Pearson improves vs M3b → LLM leader simulation was the bottleneck.
  If Pearson stays similar → follower CA/SIR dynamics are the bottleneck.

No OpenAI API calls — runs instantly (data-only, no cache needed).
"""

from __future__ import annotations

from typing import Dict, Optional

from backend.algorithm.mafdm_m3b import MAFDM_M3b


class RealLeaders(MAFDM_M3b):
    """
    Data-grounded leaders: real CSV daily group averages drive all leaders,
    M3b CA+SIR drives followers.
    """

    def __init__(self, config) -> None:
        super().__init__(config)
        # Populated by inject_dataset() before the day loop starts
        self._daily_avgs:     Dict[int, float]            = {}  # {day: group_avg}
        self._initial_opinion: Dict[str, float]           = {}

    # ── Dataset injection -------------------------------------------------------

    def inject_dataset(self, dataset) -> None:
        """Receive real leader daily averages and initial opinions from the dataset."""
        pf = getattr(self, "polarity_filter", "all")
        if pf == "positive":
            self._daily_avgs = getattr(dataset, "_leader_daily_avgs_pos", {}) or getattr(dataset, "_leader_daily_avgs", {})
        elif pf == "negative":
            self._daily_avgs = getattr(dataset, "_leader_daily_avgs_neg", {}) or getattr(dataset, "_leader_daily_avgs", {})
        else:
            self._daily_avgs = getattr(dataset, "_leader_daily_avgs", {})
        self._initial_opinion = {ldr.agent_id: ldr.opinion for ldr in dataset.leaders}

    # ── Real score lookup -------------------------------------------------------

    def get_leader_real_score(self, agent_id: str, day: int) -> Optional[float]:
        """
        Return the group daily average for this day.

        All leaders receive the same group-level signal — the real community
        opinion average for that day (interpolated where no posts exist).
        Fallback to initial opinion only if day is outside the sim window.
        """
        if day in self._daily_avgs:
            return float(self._daily_avgs[day])
        return self._initial_opinion.get(agent_id)

    # ── Capability flags --------------------------------------------------------

    @property
    def uses_llm(self) -> bool:
        return False

    @property
    def uses_memory(self) -> bool:
        return False

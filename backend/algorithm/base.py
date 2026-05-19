"""
backend/algorithm/base.py
=========================
Abstract base class for all MA-FDE-LLM algorithm variants.

Every variant must implement this interface. The simulation runner
calls only these methods — it has no knowledge of the variant's
internal logic.

Contract:
  compute_epsilon     — given velocity and phase, return today's ε
  build_leader_prompt — given agent state, return the LLM prompt string
  update_leader       — given CA inputs + LLM output, return new opinion
  update_follower     — given connection opinions + ε, return new opinion
  should_write_memory — given opinion delta, return True if diary entry warranted
  get_memory_description — return human-readable description for a diary entry
  uses_memory         — True if this variant maintains a memory diary
  uses_dynamic_eps    — True if this variant recalculates ε each day
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from backend.agents import LeaderAgent, EpisodicMemory
    from backend.config import AlgorithmConfig


class AbstractAlgorithm(ABC):
    """
    Base class for all simulation algorithm variants.
    Subclasses receive the full config at construction time.
    """

    def __init__(self, config: "AlgorithmConfig") -> None:
        self.config = config

    # ── Epsilon ------------------------------------------------------------------

    @abstractmethod
    def compute_epsilon(self, velocity: float, is_recovery: bool) -> float:
        """
        Return today's interaction threshold ε.

        Args:
            velocity:    |mean_leader(t) − mean_leader(t−1)|
            is_recovery: True when the simulation is in the recovery phase

        Returns:
            ε ∈ [0, EPS_MAX]
        """

    # ── LLM prompt ---------------------------------------------------------------

    @abstractmethod
    def build_leader_prompt(
        self,
        agent:        "LeaderAgent",
        day:          int,
        news_blurb:   Optional[str],
        neighbor_avg: float,
    ) -> str:
        """
        Build the full LLM prompt for one leader on one day.

        Args:
            agent:        the leader agent (opinion + memory + profile)
            day:          current simulation day (relative to crisis Day 0)
            news_blurb:   anchor blurb text if today is an anchor day, else None
            neighbor_avg: mean opinion of this leader's grid neighbours

        Returns:
            Complete prompt string ready for the LLM.
        """

    # ── Leader update ------------------------------------------------------------

    @abstractmethod
    def update_leader(
        self,
        current_opinion:    float,
        neighbor_opinions:  List[float],
        llm_output:         int,
        epsilon:            float,
    ) -> float:
        """
        Compute a leader's new opinion.

        Args:
            current_opinion:   O_i at start of today
            neighbor_opinions: opinions of grid neighbours at start of today
            llm_output:        LLM response ∈ {-1, 0, +1}
            epsilon:           today's interaction threshold

        Returns:
            New opinion ∈ [-1.0, +1.0]
        """

    # ── Follower update ----------------------------------------------------------

    @abstractmethod
    def update_follower(
        self,
        current_opinion:    float,
        connection_opinions: List[float],
        epsilon:            float,
    ) -> float:
        """
        Compute a follower's new opinion.

        Args:
            current_opinion:     O_i at start of today
            connection_opinions: opinions of 5 connections (leaders + friends)
            epsilon:             today's interaction threshold

        Returns:
            New opinion ∈ [-1.0, +1.0]
        """

    # ── Memory ------------------------------------------------------------------

    @abstractmethod
    def should_write_memory(self, opinion_delta: float) -> bool:
        """
        Return True if the opinion change is significant enough to record.

        Args:
            opinion_delta: new_opinion − old_opinion

        Returns:
            True → write diary entry; False → skip
        """

    @abstractmethod
    def get_memory_description(
        self,
        day:             int,
        opinion_delta:   float,
        last_anchor_day: Optional[int],
    ) -> str:
        """
        Return a human-readable description for a non-anchor diary entry.

        Args:
            day:             current simulation day
            opinion_delta:   new_opinion − old_opinion
            last_anchor_day: day of the most recent anchor (None if pre-event)

        Returns:
            Description string stored in the memory diary entry.
        """

    # ── Capability flags --------------------------------------------------------

    @property
    @abstractmethod
    def uses_memory(self) -> bool:
        """True if this variant reads/writes the episodic memory diary."""

    @property
    @abstractmethod
    def uses_dynamic_eps(self) -> bool:
        """True if this variant recalculates ε each day from velocity."""

    @property
    def uses_llm(self) -> bool:
        """True if this variant makes LLM calls during the day loop.
        Defaults to True — override to False for pure CA variants."""
        return True

    # ── Anchor memory enrichment hook -------------------------------------------

    def enrich_anchor_memory(
        self,
        description: str,
        llm_outputs: Dict[str, int],
    ) -> str:
        """
        Optionally append community response context to an anchor memory description.

        Called by the simulation runner in step G after all anchor-day LLM calls
        complete. The default returns the description unchanged — only memory-aware
        variants that implement response stamping need to override this.

        Args:
            description:  the raw anchor memory string from the dataset meta
            llm_outputs:  {agent_id: vote} for all leaders on this anchor day

        Returns:
            Enriched description string (or the original if no enrichment needed).
        """
        return description

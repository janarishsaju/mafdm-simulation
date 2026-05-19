"""
backend/algorithm/mafdm_m5.py
==============================
MA-FDE-LLM M5 — Response-Stamped Anchors + Velocity-Phase Detection.

Two architectural improvements over M3b designed to generalise across
events with different crisis structures:

  Idea 1 — Response-Stamped Anchor Memory:
    After all leaders vote on an anchor day, the vote consensus is computed
    and appended to the anchor memory description stored in every leader's
    episodic diary. On subsequent days, agents read not just WHAT the news
    said, but HOW the simulated community reacted to it.

    Example:
      Day 3 blurb: "EMA says benefits outweigh risks, age restrictions applied."
      Consensus of leaders = −0.12 (mixed/skeptical)
      Memory written:
        "ANCHOR Day 3: EMA says benefits outweigh risks...
         Community response: mixed — significant uncertainty and skepticism persist."

    For J&J Day 0: consensus ≈ −0.95 → "strongly alarmed" stamp
    For J&J Day 10: consensus ≈ +0.90 → "broadly supportive" stamp  (recovery)
    For AZ Day 3:   consensus ≈ −0.10 → "mixed" stamp              (no false recovery)
    For AZ Day 23:  consensus ≈ −0.70 → "strongly alarmed" stamp

    This prevents the model from drifting positive between AZ waves because
    agents remember "the EMA update produced a mixed/skeptical community
    reaction," not just "EMA said benefits outweigh risks."

  Idea 2 — Velocity-Threshold Phase Detection:
    For events with RESOLUTION_ANCHOR_DAY = −999 (no clean resolution),
    the binary is_recovery flag is always False, so epsilon stays at the
    crisis floor (0.30) for the entire simulation — including inter-wave
    lulls when almost no change is happening. This over-restricts network
    connectivity during lulls.

    M5 replaces the binary flag with a velocity threshold:
      velocity < 0.02  →  lull (quiet period)  →  use recovery eps floor (0.50)
      velocity ≥ 0.02  →  active wave           →  use crisis eps floor (0.30)

    This keeps network connectivity healthy during inter-wave periods so
    agents can spread the sustained negative signal more efficiently.
    For events WITH a resolution day (J&J), is_recovery continues to be
    used as-is — no change in J&J behaviour.

Both changes are event-agnostic: no AZ-specific tuning, no new parameters.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, TYPE_CHECKING

from backend.algorithm.mafdm_m3b import MAFDM_M3b

if TYPE_CHECKING:
    from backend.config import AlgorithmConfig


# Velocity below this threshold is treated as "lull" for Idea 2.
_LULL_VELOCITY_THRESHOLD = 0.02


class MAFDM_M5(MAFDM_M3b):
    """
    MA-FDE-LLM M5: Response-Stamped Anchors + Velocity-Phase Detection.

    Inherits all memory diary logic, LLM prompting, leader update,
    and follower update from M3b. Overrides only compute_epsilon
    and enrich_anchor_memory.
    """

    # ── Idea 1: Response-Stamped Anchor Memory ----------------------------------

    def enrich_anchor_memory(
        self,
        description: str,
        llm_outputs: Dict[str, int],
    ) -> str:
        """
        Append community consensus stamp to anchor memory description.

        Stamp thresholds (mean LLM vote ∈ [−1, +1]):
          ≥ +0.40 → broadly supportive
          ≥ +0.10 → cautiously positive
          ≥ −0.10 → mixed / skeptical
          ≥ −0.40 → skeptical
          < −0.40 → strongly alarmed
        """
        if not llm_outputs:
            return description
        consensus = sum(llm_outputs.values()) / len(llm_outputs)
        if consensus >= 0.40:
            stamp = (
                "Community response: broadly supportive — "
                "sentiment shifting strongly positive."
            )
        elif consensus >= 0.10:
            stamp = (
                "Community response: cautiously positive — "
                "partial acceptance, uncertainty remains."
            )
        elif consensus >= -0.10:
            stamp = (
                "Community response: mixed — "
                "significant uncertainty and skepticism persist."
            )
        elif consensus >= -0.40:
            stamp = (
                "Community response: skeptical — "
                "widespread concern remains despite the update."
            )
        else:
            stamp = (
                "Community response: strongly alarmed — "
                "crisis signal dominant, trust sharply negative."
            )
        return f"{description}\n{stamp}"

    # ── Idea 2: Velocity-Threshold Phase Detection ------------------------------

    def compute_epsilon(self, velocity: float, is_recovery: bool) -> float:
        """
        ε(t) = eps_min + (EPS_MAX − eps_min) × tanh(EPS_BETA × velocity)

        eps_min selection:
          Events WITH resolution day:
            Use is_recovery exactly as M3b (no change for J&J).
          Events WITHOUT resolution (RESOLUTION_ANCHOR_DAY = -999):
            velocity < 0.02  → lull period  → EPS_MIN_RECOVERY (0.50)
            velocity ≥ 0.02  → active wave  → EPS_MIN_CRISIS   (0.30)
        """
        resolution_day = getattr(self.config, 'RESOLUTION_ANCHOR_DAY', None)
        no_resolution  = (resolution_day is None or resolution_day <= -999)

        if no_resolution:
            effective_recovery = velocity < _LULL_VELOCITY_THRESHOLD
        else:
            effective_recovery = is_recovery

        eps_min = (
            self.config.EPS_MIN_RECOVERY if effective_recovery
            else self.config.EPS_MIN_CRISIS
        )
        return eps_min + (self.config.EPS_MAX - eps_min) * math.tanh(
            self.config.EPS_BETA * velocity
        )

    # ── Capability flags --------------------------------------------------------

    @property
    def uses_memory(self) -> bool:
        return True

    @property
    def uses_dynamic_eps(self) -> bool:
        return True

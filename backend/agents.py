"""
backend/agents.py
=================
Agent data structures for the MA-FDE-LLM simulation.

Three classes:
  EpisodicMemory  — two-tier diary (pinned anchors + rolling entries)
  LeaderAgent     — opinion leader with grid position and memory
  FollowerAgent   — opinion follower with fixed connection list

These are pure data containers. All update logic lives in algorithm/.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from collections import deque
from typing import List, Optional, Tuple, Dict, Any


# ---------------------------------------------------------------------------
# Episodic Memory
# ---------------------------------------------------------------------------

@dataclass
class MemoryEntry:
    day:            int
    description:    str
    opinion_before: float
    opinion_after:  float
    delta:          float
    is_anchor:      bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "day":            self.day,
            "description":    self.description,
            "opinion_before": round(self.opinion_before, 3),
            "opinion_after":  round(self.opinion_after,  3),
            "delta":          round(self.delta, 3),
            "is_anchor":      self.is_anchor,
        }


class EpisodicMemory:
    """
    Two-tier episodic memory buffer.

    Tier 1 — pinned:   Anchor-day entries. Written once, never evicted.
    Tier 2 — rolling:  Significant non-anchor shifts. FIFO, capacity = MEMORY_ROLL_K.

    get_entries(n) returns the n most recent entries across both tiers,
    newest-first, for serialisation into the LLM prompt.
    """

    def __init__(self, roll_k: int = 6) -> None:
        self.pinned:  List[MemoryEntry]          = []
        self.rolling: deque[MemoryEntry]         = deque(maxlen=roll_k)
        self.roll_k   = roll_k

    def add(
        self,
        day:            int,
        description:    str,
        opinion_before: float,
        opinion_after:  float,
        is_anchor:      bool = False,
    ) -> None:
        entry = MemoryEntry(
            day            = day,
            description    = description,
            opinion_before = opinion_before,
            opinion_after  = opinion_after,
            delta          = opinion_after - opinion_before,
            is_anchor      = is_anchor,
        )
        if is_anchor:
            self.pinned.append(entry)
        else:
            self.rolling.append(entry)

    def get_entries(self, n: int = 4) -> List[MemoryEntry]:
        """Return up to n entries, newest-first, pinned + rolling combined."""
        combined = self.pinned + list(self.rolling)
        combined.sort(key=lambda e: e.day, reverse=True)
        return combined[:n]

    def has_entries(self) -> bool:
        return bool(self.pinned or self.rolling)

    def last_anchor_day(self) -> Optional[int]:
        if not self.pinned:
            return None
        return max(e.day for e in self.pinned)

    def clear(self) -> None:
        self.pinned.clear()
        self.rolling.clear()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pinned":  [e.to_dict() for e in self.pinned],
            "rolling": [e.to_dict() for e in self.rolling],
        }


# ---------------------------------------------------------------------------
# Leader Agent
# ---------------------------------------------------------------------------

@dataclass
class LeaderAgent:
    """
    An opinion leader placed on the CA grid.

    Fields set at initialisation:
      agent_id    — unique identifier (from CSV)
      opinion     — current opinion on [-1.0, +1.0]
      profile     — natural-language persona string (e.g. "a medical expert")
      grid_pos    — (row, col) on the CA grid

    Fields managed by the simulation runner:
      memory      — EpisodicMemory instance (created by runner, passed to algorithm)
    """

    agent_id:  str
    opinion:   float
    profile:   str
    grid_pos:  Tuple[int, int]
    memory:    EpisodicMemory = field(default_factory=EpisodicMemory)

    def __post_init__(self) -> None:
        self.opinion = float(max(-1.0, min(1.0, self.opinion)))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "opinion":  round(self.opinion, 4),
            "profile":  self.profile,
            "grid_pos": list(self.grid_pos),
        }


# ---------------------------------------------------------------------------
# Follower Agent
# ---------------------------------------------------------------------------

@dataclass
class FollowerAgent:
    """
    An opinion follower with a fixed set of connections.

    Connections are assigned once during setup (3 leaders + 2 followers)
    and never change throughout the simulation.

    connection_ids contains agent_id strings for both leader and follower
    connections. The runner resolves current opinions by looking them up
    in the shared opinion dictionaries.
    """

    agent_id:       str
    opinion:        float
    connection_ids: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.opinion = float(max(-1.0, min(1.0, self.opinion)))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id":       self.agent_id,
            "opinion":        round(self.opinion, 4),
            "connection_ids": self.connection_ids,
        }

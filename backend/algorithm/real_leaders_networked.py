"""
backend/algorithm/real_leaders_networked.py
============================================
Data-Grounded Leaders + Actual Connections variant.

Extends RealLeaders with one change: follower connection_ids are replaced
with actual derived connections from {dataset_id}_connections.csv instead
of random assignment.

Connection derivation (built offline, stored in datasets/):
  subreddit      — follower in same subreddit as leader group (272 followers)
  content_theme  — follower theme matches leader theme, no subreddit overlap (71)
  theme_fallback — nearest-theme match for 3 unrepresented themes (16)
  thread_scraped — real comment-tree co-participation from Arctic Shift (3)

All other simulation logic (leader scores, follower CA+SIR, dynamic ε)
is identical to the base RealLeaders variant.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import pandas as pd

from backend.algorithm.real_leaders import RealLeaders

_DATASETS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "datasets",
)


class RealLeadersNetworked(RealLeaders):
    """
    Real leader scores + actual derived connections from *_connections.csv.
    Follower connection_ids are replaced before the day loop starts.
    """

    def inject_dataset(self, dataset) -> None:
        super().inject_dataset(dataset)
        conn_file = self._find_connections_file()
        if conn_file is None:
            return
        self._override_follower_connections(dataset, conn_file)

    def _find_connections_file(self) -> Optional[str]:
        for fname in os.listdir(_DATASETS_DIR):
            if fname.endswith("_connections.csv"):
                return os.path.join(_DATASETS_DIR, fname)
        return None

    def _override_follower_connections(self, dataset, conn_file: str) -> None:
        conn_df = pd.read_csv(conn_file)

        conn_map: Dict[str, List[str]] = {}
        for _, row in conn_df.iterrows():
            leaders_raw  = str(row.get("leader_connection_ids",  "") or "")
            followers_raw = str(row.get("follower_connection_ids", "") or "")
            conn_map[row["agent_id"]] = (
                [x for x in leaders_raw.split(",")  if x] +
                [x for x in followers_raw.split(",") if x]
            )

        for follower in dataset.followers:
            if follower.agent_id in conn_map:
                follower.connection_ids = conn_map[follower.agent_id]

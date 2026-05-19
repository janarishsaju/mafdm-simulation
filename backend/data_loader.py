"""
backend/data_loader.py
======================
Generic CSV loader and dataset validator for the MA-FDE-LLM simulation.

Responsibilities:
  - Load a CSV of agent posts and the accompanying JSON metadata file
  - Validate required columns are present
  - Split agents into leaders and followers
  - Compute pre-event baselines and initial opinions
  - Build the real follower curve (ground truth for evaluation)
  - Return a LoadedDataset ready for the simulation runner

Dataset metadata (JSON) specifies:
  - Event anchor days and blurbs
  - Profile map (content_type → natural-language persona)
  - Column name mappings (supports different CSV schemas)

Adding a new dataset:
  1. Put the CSV in phase_9/datasets/
  2. Create a companion _meta.json (see jj_vaccine_meta.json for schema)
  3. Register the dataset id in list_datasets()
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

from backend.agents import EpisodicMemory, FollowerAgent, LeaderAgent

# Resolve datasets directory relative to this file
_DATASETS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "datasets",
)

KNOWN_DATASETS = {
    "jj_vaccine": "jj_vaccine_meta.json",
}


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class LoadedDataset:
    """
    Everything the simulation runner needs to start a run.

    leaders         — list of LeaderAgent, opinions set to pre-event values
    followers       — list of FollowerAgent, opinions set to pre-event values,
                      connection_ids already assigned
    real_curve      — dict mapping day → smoothed real follower average
                      (ground truth for Pearson / DTW evaluation)
    anchor_days     — dict mapping day (int) → news blurb string
    anchor_memories — dict mapping day (int) → memory description string
    resolution_anchor_day — the anchor day that marks the end of the crisis
    sim_start       — first simulation day (relative to Day 0)
    sim_end         — last simulation day
    event_label     — human-readable event description
    """

    leaders:               List[LeaderAgent]
    followers:             List[FollowerAgent]
    real_curve:            Dict[int, float]
    anchor_days:           Dict[int, str]
    anchor_memories:       Dict[int, str]
    resolution_anchor_day: Optional[int]
    sim_start:             int
    sim_end:               int
    event_label:           str

    # Convenience look-ups set by loader (not used by runner directly)
    n_leaders:   int = field(init=False)
    n_followers: int = field(init=False)

    def __post_init__(self) -> None:
        self.n_leaders   = len(self.leaders)
        self.n_followers = len(self.followers)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def list_datasets() -> List[Dict]:
    """Return summary info for all known datasets (for the frontend dropdown)."""
    result = []
    for dataset_id, meta_file in KNOWN_DATASETS.items():
        meta_path = os.path.join(_DATASETS_DIR, meta_file)
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            # Convert anchor_days keys to ints for consistent typing
            raw_anchors = meta.get("anchor_days", {})
            anchor_days = {int(k): v for k, v in raw_anchors.items()}
            result.append({
                "id":         dataset_id,
                "name":       meta.get("name", dataset_id),
                "label":      meta.get("event_date_label", ""),
                "sim_start":  meta.get("sim_start", -5),
                "sim_end":    meta.get("sim_end",   25),
                "anchor_days": anchor_days,
            })
    return result


def load_dataset(
    dataset_id: str,
    grid_rows:  int,
    grid_cols:  int,
    seed:       int,
    n_leader_connections:   int = 3,
    n_follower_connections: int = 2,
) -> LoadedDataset:
    """
    Load a dataset by id and build all agent structures.

    Args:
        dataset_id:              key from KNOWN_DATASETS
        grid_rows / grid_cols:   CA grid dimensions (from AlgorithmConfig)
        seed:                    random seed for grid placement and connections
        n_leader_connections:    how many leaders each follower is connected to
        n_follower_connections:  how many followers each follower is connected to

    Returns:
        LoadedDataset ready to pass to the simulation runner.
    """
    if dataset_id not in KNOWN_DATASETS:
        raise ValueError(
            f"Unknown dataset '{dataset_id}'. Available: {list(KNOWN_DATASETS.keys())}"
        )

    meta_path = os.path.join(_DATASETS_DIR, KNOWN_DATASETS[dataset_id])
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    csv_path = os.path.join(_DATASETS_DIR, meta["csv_file"])
    df = pd.read_csv(csv_path)

    _validate_columns(df, meta)

    rng = random.Random(seed)
    np.random.seed(seed)

    # ── Column aliases ─────────────────────────────────────────────────────
    col_type    = meta.get("type_column",    "agent_type")
    col_phase   = meta.get("phase_column",   "phase")
    col_profile = meta.get("profile_column", "content_type")
    col_day     = meta.get("day_column",     "days_from_incident_start")
    col_score   = meta.get("score_column",   "attitude_score")
    pre_phase   = meta.get("pre_event_phase","neutral_pre")
    leader_type = meta.get("leader_filter",  {}).get("agent_type", "opinion_leader")
    follower_type = meta.get("follower_filter", {}).get("agent_type", "opinion_follower")

    profile_map     = meta.get("profile_map", {})
    default_profile = meta.get("default_profile", "a social media commentator")

    sim_start = int(meta.get("sim_start", -5))
    sim_end   = int(meta.get("sim_end",    25))

    anchor_days     = {int(k): v for k, v in meta.get("anchor_days",     {}).items()}
    anchor_memories = {int(k): v for k, v in meta.get("anchor_memories", {}).items()}
    resolution_day  = meta.get("resolution_anchor_day", None)
    if resolution_day is not None:
        resolution_day = int(resolution_day)

    # ── Split ──────────────────────────────────────────────────────────────
    leaders_df   = df[df[col_type] == leader_type].copy()
    followers_df = df[df[col_type] == follower_type].copy()

    leader_ids   = leaders_df["agent_id"].unique().tolist()
    follower_ids = followers_df["agent_id"].unique().tolist()

    leader_baseline   = _compute_baseline(leaders_df,   col_phase, col_score, pre_phase)
    follower_baseline = _compute_baseline(followers_df, col_phase, col_score, pre_phase)

    # ── Leader CA grid ─────────────────────────────────────────────────────
    grid_size  = grid_rows * grid_cols
    n_leaders  = len(leader_ids)
    if n_leaders > grid_size:
        raise ValueError(
            f"Grid {grid_rows}×{grid_cols} = {grid_size} cells but {n_leaders} leaders."
        )

    flat_cells   = rng.sample(range(grid_size), n_leaders)
    grid_pos_map: Dict[str, Tuple[int, int]] = {}
    cell_to_id:   Dict[Tuple[int, int], str] = {}
    for aid, flat in zip(leader_ids, flat_cells):
        pos = (flat // grid_cols, flat % grid_cols)
        grid_pos_map[aid] = pos
        cell_to_id[pos]   = aid

    # ── Build leader agents ────────────────────────────────────────────────
    leader_profile_map = {
        row["agent_id"]: profile_map.get(
            row.get(col_profile, ""), default_profile
        )
        for _, row in leaders_df.drop_duplicates("agent_id").iterrows()
    }

    leaders = [
        LeaderAgent(
            agent_id = aid,
            opinion  = _get_initial_opinion(aid, leaders_df, col_phase, col_score,
                                             pre_phase, leader_baseline),
            profile  = leader_profile_map.get(aid, default_profile),
            grid_pos = grid_pos_map[aid],
            memory   = EpisodicMemory(roll_k=6),
        )
        for aid in leader_ids
    ]

    # ── Moore-neighbour look-up (built from grid) ──────────────────────────
    # Stored on the LoadedDataset so the runner can use it without recomputing
    def get_moore_neighbors(aid: str) -> List[str]:
        row, col = grid_pos_map[aid]
        return [
            cell_to_id[(row + dr, col + dc)]
            for dr in (-1, 0, 1) for dc in (-1, 0, 1)
            if not (dr == 0 and dc == 0)
            and (row + dr, col + dc) in cell_to_id
        ]

    # attach neighbour list to each LeaderAgent via a side dict (runner uses it)
    # We store it as an extra attribute on the dataset object below.

    # ── Build follower agents ──────────────────────────────────────────────
    n_lc = min(n_leader_connections,   n_leaders)
    n_fc = min(n_follower_connections, len(follower_ids) - 1)

    followers = [
        FollowerAgent(
            agent_id       = fid,
            opinion        = _get_initial_opinion(fid, followers_df, col_phase, col_score,
                                                  pre_phase, follower_baseline),
            connection_ids = (
                rng.sample(leader_ids, n_lc) +
                rng.sample([x for x in follower_ids if x != fid], n_fc)
            ),
        )
        for fid in follower_ids
    ]

    # ── Real follower curve (ground truth) ─────────────────────────────────
    real_curve = _build_real_curve(followers_df, col_day, col_score, sim_start, sim_end)

    dataset = LoadedDataset(
        leaders               = leaders,
        followers             = followers,
        real_curve            = real_curve,
        anchor_days           = anchor_days,
        anchor_memories       = anchor_memories,
        resolution_anchor_day = resolution_day,
        sim_start             = sim_start,
        sim_end               = sim_end,
        event_label           = meta.get("event_date_label", ""),
    )

    # Attach helper that the runner needs for Moore neighbours
    dataset._leader_grid_neighbors = {aid: get_moore_neighbors(aid) for aid in leader_ids}

    return dataset


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _validate_columns(df: pd.DataFrame, meta: dict) -> None:
    required = [
        meta.get("type_column",    "agent_type"),
        meta.get("phase_column",   "phase"),
        meta.get("day_column",     "days_from_incident_start"),
        meta.get("score_column",   "attitude_score"),
        "agent_id",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")


def _compute_baseline(
    df: pd.DataFrame,
    col_phase: str,
    col_score: str,
    pre_phase: str,
) -> float:
    pre = df[df[col_phase] == pre_phase][col_score]
    if len(pre) == 0:
        return float(df[col_score].mean())
    return float(pre.mean())


def _get_initial_opinion(
    agent_id:  str,
    df:        pd.DataFrame,
    col_phase: str,
    col_score: str,
    pre_phase: str,
    baseline:  float,
) -> float:
    rows = df[df["agent_id"] == agent_id]
    pre  = rows[rows[col_phase] == pre_phase]
    if len(pre) > 0:
        return float(pre[col_score].iloc[0])
    return baseline


def _build_real_curve(
    df:        pd.DataFrame,
    col_day:   str,
    col_score: str,
    sim_start: int,
    sim_end:   int,
) -> Dict[int, float]:
    """
    Compute the smoothed real follower average for each simulation day.
    Missing days are linearly interpolated; a Gaussian σ=1.5 is applied.
    Returns a dict day → smoothed_value.
    """
    days = list(range(sim_start, sim_end + 1))
    all_days = pd.DataFrame({"day": days})

    daily = (
        df[df[col_day].between(sim_start, sim_end)]
        .groupby(col_day)[col_score].mean()
        .reset_index()
        .rename(columns={col_day: "day", col_score: "real_avg"})
    )
    merged = all_days.merge(daily, on="day", how="left")
    merged["interp"] = merged["real_avg"].interpolate(method="linear", limit_direction="both")
    smoothed = gaussian_filter1d(merged["interp"].values, sigma=1.5)
    return {int(d): float(s) for d, s in zip(days, smoothed)}

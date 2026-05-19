"""
backend/results_cache.py
========================
File-based CSV cache for simulation day results.

Cache key  : dataset_id + variant_id
Cache file : phase_9/results/{dataset_id}__{variant_id}.csv

One CSV file per dataset-variant combination.
Columns saved are the exact DayResult fields — same structure as the
phase_8 simulation output CSVs so files are human-readable and consistent.

Workflow:
  has_cache()   → check before running
  load_cache()  → read rows → List[DayResult]  (instant, no LLM calls)
  save_cache()  → write rows after a live run
  clear_cache() → delete file so next Run triggers fresh LLM calls
  list_cache()  → show what is already cached (for UI status display)
"""

from __future__ import annotations

import os
from typing import List, Optional

import pandas as pd

from backend.simulation_runner import DayResult


# ---------------------------------------------------------------------------
# Directory
# ---------------------------------------------------------------------------

_RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results",
)

# Canonical column order — matches phase_8 output + DayResult fields
CACHE_COLUMNS = [
    "day",
    "leader_avg",
    "follower_avg",
    "epsilon",
    "velocity",
    "llm_pos",
    "llm_neu",
    "llm_neg",
    "is_anchor",
    "phase",
    "avg_memory_size",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _path(dataset_id: str, variant_id: str) -> str:
    os.makedirs(_RESULTS_DIR, exist_ok=True)
    return os.path.join(_RESULTS_DIR, f"{dataset_id}__{variant_id}.csv")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def has_cache(dataset_id: str, variant_id: str) -> bool:
    return os.path.isfile(_path(dataset_id, variant_id))


def load_cache(dataset_id: str, variant_id: str) -> Optional[List[DayResult]]:
    """
    Read a cached results CSV and return as List[DayResult].
    Returns None if no cache file exists.
    """
    p = _path(dataset_id, variant_id)
    if not os.path.isfile(p):
        return None

    df = pd.read_csv(p)

    # Normalise column names in case file came from phase_8 (eps / avg_mem_size)
    df = df.rename(columns={
        "eps":          "epsilon",
        "avg_mem_size": "avg_memory_size",
    })

    results: List[DayResult] = []
    for _, row in df.iterrows():
        results.append(DayResult(
            day             = int(row["day"]),
            leader_avg      = float(row["leader_avg"]),
            follower_avg    = float(row["follower_avg"]),
            epsilon         = float(row["epsilon"]),
            velocity        = float(row["velocity"]),
            llm_pos         = int(row["llm_pos"]),
            llm_neu         = int(row["llm_neu"]),
            llm_neg         = int(row["llm_neg"]),
            is_anchor       = bool(str(row["is_anchor"]).strip().lower() in ("true", "1", "yes")),
            phase           = str(row["phase"]),
            avg_memory_size = float(row["avg_memory_size"]),
        ))
    return results


def save_cache(
    dataset_id: str,
    variant_id: str,
    results:    List[DayResult],
) -> str:
    """
    Write simulation results to a CSV cache file.
    Returns the path of the saved file.
    """
    p = _path(dataset_id, variant_id)
    rows = [r.to_dict() for r in results]
    df = pd.DataFrame(rows)[CACHE_COLUMNS]
    df.to_csv(p, index=False)
    return p


def clear_cache(dataset_id: str, variant_id: str) -> bool:
    """Delete the cache file. Returns True if a file was deleted."""
    p = _path(dataset_id, variant_id)
    if os.path.isfile(p):
        os.remove(p)
        return True
    return False


def list_cache() -> List[dict]:
    """
    Return all cached dataset-variant combinations.
    Used by the API to tell the frontend which results are already available.
    """
    if not os.path.isdir(_RESULTS_DIR):
        return []
    items = []
    for fname in sorted(os.listdir(_RESULTS_DIR)):
        if not fname.endswith(".csv"):
            continue
        stem = fname[:-4]          # strip .csv
        parts = stem.split("__")
        if len(parts) == 2:
            items.append({
                "dataset_id": parts[0],
                "variant_id": parts[1],
                "filename":   fname,
            })
    return items

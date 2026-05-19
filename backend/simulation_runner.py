"""
backend/simulation_runner.py
=============================
Day-loop orchestrator for the MA-FDE-LLM simulation.

Executes the complete simulation across all days for any algorithm variant.
Calls only the AbstractAlgorithm interface — no variant-specific logic here.

Day loop order (per day):
  A. Compute ε from velocity + phase flag
  B. Resolution anchor reset (if day == resolution_anchor_day):
       clear all leader memories, write preliminary anchor entry
  C. Crisis anchor flag: set last_anchor BEFORE LLM call
  D. Build prompts for all leaders
  E. Call LLM in parallel (via llm_client)
  F. Update all leaders (CA + LLM fusion)
  G. Write memory entries (anchor or rolling, based on algorithm policy)
  H. Update all followers (CA + SIR)
  I. Record DayResult, call on_progress callback

This replicates the exact execution order in phase_8/simulate_jj_e21_m3b.py.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np

from backend.agents import LeaderAgent, FollowerAgent, EpisodicMemory
from backend.algorithm.base import AbstractAlgorithm
from backend.config import AlgorithmConfig
from backend.data_loader import LoadedDataset
from backend import llm_client


# ---------------------------------------------------------------------------
# Per-day result record
# ---------------------------------------------------------------------------

@dataclass
class DayResult:
    """Snapshot of simulation state after each day. Serialisable to JSON."""
    day:             int
    leader_avg:      float
    follower_avg:    float
    epsilon:         float
    velocity:        float
    llm_pos:         int
    llm_neu:         int
    llm_neg:         int
    is_anchor:       bool
    phase:           str   # "crisis" or "recovery"
    avg_memory_size: float

    def to_dict(self) -> dict:
        return {
            "day":             self.day,
            "leader_avg":      round(self.leader_avg,      4),
            "follower_avg":    round(self.follower_avg,    4),
            "epsilon":         round(self.epsilon,          4),
            "velocity":        round(self.velocity,         4),
            "llm_pos":         self.llm_pos,
            "llm_neu":         self.llm_neu,
            "llm_neg":         self.llm_neg,
            "is_anchor":       self.is_anchor,
            "phase":           self.phase,
            "avg_memory_size": round(self.avg_memory_size, 2),
        }


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

class SimulationRunner:
    """
    Stateless runner — create a fresh instance per simulation run.
    Reads state from dataset and config; writes nothing to disk.
    """

    def run(
        self,
        dataset:     LoadedDataset,
        algorithm:   AbstractAlgorithm,
        config:      AlgorithmConfig,
        on_progress: Optional[Callable[[DayResult], None]] = None,
    ) -> List[DayResult]:
        """
        Execute the full simulation and return a list of DayResult (one per day).

        Args:
            dataset:     pre-loaded dataset (agents, anchor blurbs, real curve)
            algorithm:   concrete algorithm instance
            config:      AlgorithmConfig (parameters)
            on_progress: optional callback — called after each day completes
                         (used by the API to stream SSE events to the frontend)

        Returns:
            List of DayResult, one per simulation day.
        """
        # ── Unpack dataset ────────────────────────────────────────────────
        leaders:   List[LeaderAgent]   = dataset.leaders
        followers: List[FollowerAgent] = dataset.followers
        leader_ids   = [a.agent_id for a in leaders]
        follower_ids = [f.agent_id for f in followers]

        # Moore-neighbour look-up (set by data_loader on LoadedDataset)
        grid_neighbors: Dict[str, List[str]] = getattr(
            dataset, "_leader_grid_neighbors", {a.agent_id: [] for a in leaders}
        )

        # Follower connection look-up
        follower_connections: Dict[str, List[str]] = {
            f.agent_id: f.connection_ids for f in followers
        }

        # Anchor config
        anchor_days:      Dict[int, str] = dataset.anchor_days
        anchor_memories:  Dict[int, str] = dataset.anchor_memories
        resolution_day:   Optional[int]  = dataset.resolution_anchor_day

        sim_start = dataset.sim_start
        sim_end   = dataset.sim_end
        days      = list(range(sim_start, sim_end + 1))

        # ── Mutable opinion tables ─────────────────────────────────────────
        leader_opinion:   Dict[str, float] = {a.agent_id: a.opinion for a in leaders}
        follower_opinion: Dict[str, float] = {f.agent_id: f.opinion for f in followers}

        # ── Runner state ──────────────────────────────────────────────────
        last_anchor:     Optional[int] = None
        prev_leader_avg: Optional[float] = None
        current_eps:     float = config.EPS_INIT

        results: List[DayResult] = []

        # ── Day loop ──────────────────────────────────────────────────────
        for day in days:
            is_anchor = day in anchor_days

            # ── A. Compute ε ───────────────────────────────────────────────
            # is_recovery: True only AFTER resolution anchor has fired.
            # At the top of the resolution day itself, last_anchor is still
            # the previous crisis anchor, so the crisis floor is used correctly.
            is_recovery = (last_anchor == resolution_day) if resolution_day is not None else False
            current_leader_avg = float(np.mean(list(leader_opinion.values())))

            if prev_leader_avg is not None:
                velocity    = abs(current_leader_avg - prev_leader_avg)
                current_eps = algorithm.compute_epsilon(velocity, is_recovery)
            else:
                velocity    = 0.0
                current_eps = config.EPS_INIT
            prev_leader_avg = current_leader_avg

            # ── B. Resolution anchor memory reset ──────────────────────────
            # Fires only on the resolution anchor day (e.g. Day 10 for J&J).
            # Clear all leader memories and write a preliminary anchor entry
            # (opinion_after will be patched in step G after the LLM update).
            if resolution_day is not None and day == resolution_day and algorithm.uses_memory:
                mem_desc = anchor_memories.get(day, f"ANCHOR Day {day}: situation resolved.")
                for leader in leaders:
                    leader.memory.clear()
                    leader.memory.add(
                        day            = day,
                        description    = mem_desc,
                        opinion_before = leader_opinion[leader.agent_id],
                        opinion_after  = leader_opinion[leader.agent_id],
                        is_anchor      = True,
                    )

            # ── C. Crisis anchor: set last_anchor before LLM ─────────────
            # Must happen before step D so the build_leader_prompt can detect
            # that this is an ongoing crisis (not a resolution) for days after.
            # Does NOT apply to the resolution anchor day.
            if is_anchor and day != resolution_day:
                last_anchor = day

            # ── D. Build prompts for all leaders ──────────────────────────
            old_leader_opinion = dict(leader_opinion)

            if algorithm.uses_llm:
                prompts: Dict[str, str] = {}
                for leader in leaders:
                    neighbors      = grid_neighbors.get(leader.agent_id, [])
                    neighbor_avg   = (
                        float(np.mean([leader_opinion[n] for n in neighbors]))
                        if neighbors else leader_opinion[leader.agent_id]
                    )
                    news_blurb = anchor_days.get(day, None)
                    prompts[leader.agent_id] = algorithm.build_leader_prompt(
                        agent        = leader,
                        day          = day,
                        news_blurb   = news_blurb,
                        neighbor_avg = neighbor_avg,
                    )

                # ── E. LLM calls (parallel) ────────────────────────────────
                llm_outputs = llm_client.call_llm_batch(
                    prompts     = prompts,
                    day         = day,
                    model       = config.LLM_MODEL,
                    temperature = config.LLM_TEMPERATURE,
                    max_tokens  = config.LLM_MAX_TOKENS,
                    max_workers = config.LLM_MAX_WORKERS,
                )
            else:
                # No LLM — pass neutral zeros (multiplied by 0 in pure CA update)
                llm_outputs = {a.agent_id: 0 for a in leaders}

            # ── F. Update leaders ──────────────────────────────────────────
            new_leader_opinion: Dict[str, float] = {}
            for leader in leaders:
                neighbors  = grid_neighbors.get(leader.agent_id, [])
                neigh_ops  = [leader_opinion[n] for n in neighbors]
                new_leader_opinion[leader.agent_id] = algorithm.update_leader(
                    current_opinion   = leader_opinion[leader.agent_id],
                    neighbor_opinions = neigh_ops,
                    llm_output        = llm_outputs.get(leader.agent_id, 0),
                    epsilon           = current_eps,
                )

            # ── G. Write memory entries ────────────────────────────────────
            if algorithm.uses_memory:
                if is_anchor and day != resolution_day:
                    # Crisis anchor — write pinned entry after LLM update.
                    # enrich_anchor_memory appends community consensus stamp (M5+);
                    # default implementation returns description unchanged (M3b, others).
                    mem_desc = anchor_memories.get(day, f"ANCHOR Day {day}: major event.")
                    mem_desc = algorithm.enrich_anchor_memory(mem_desc, llm_outputs)
                    for leader in leaders:
                        leader.memory.add(
                            day            = day,
                            description    = mem_desc,
                            opinion_before = old_leader_opinion[leader.agent_id],
                            opinion_after  = new_leader_opinion[leader.agent_id],
                            is_anchor      = True,
                        )
                elif not is_anchor:
                    # Non-anchor — write rolling entry if delta >= threshold
                    for leader in leaders:
                        O_before = old_leader_opinion[leader.agent_id]
                        O_after  = new_leader_opinion[leader.agent_id]
                        delta    = O_after - O_before
                        if algorithm.should_write_memory(delta):
                            desc = algorithm.get_memory_description(
                                day             = day,
                                opinion_delta   = delta,
                                last_anchor_day = last_anchor,
                            )
                            leader.memory.add(
                                day            = day,
                                description    = desc,
                                opinion_before = O_before,
                                opinion_after  = O_after,
                                is_anchor      = False,
                            )
                elif day == resolution_day:
                    # Patch opinion_after in the preliminary resolution anchor entry
                    for leader in leaders:
                        if leader.memory.pinned:
                            entry = leader.memory.pinned[-1]
                            entry.opinion_after = round(new_leader_opinion[leader.agent_id], 3)
                            entry.delta = round(
                                new_leader_opinion[leader.agent_id]
                                - old_leader_opinion[leader.agent_id], 3
                            )
                    last_anchor = resolution_day

            leader_opinion = new_leader_opinion
            leader_avg = float(np.mean(list(leader_opinion.values())))

            # ── H. Update followers ────────────────────────────────────────
            new_follower_opinion: Dict[str, float] = {}
            for follower in followers:
                conn_ops = [
                    leader_opinion.get(cid, follower_opinion.get(cid, 0.0))
                    for cid in follower_connections[follower.agent_id]
                ]
                new_follower_opinion[follower.agent_id] = algorithm.update_follower(
                    current_opinion     = follower_opinion[follower.agent_id],
                    connection_opinions = conn_ops,
                    epsilon             = current_eps,
                )
            follower_opinion = new_follower_opinion
            follower_avg = float(np.mean(list(follower_opinion.values())))

            # ── I. Record result ───────────────────────────────────────────
            llm_vals = list(llm_outputs.values())
            total_mem = sum(
                len(ldr.memory.pinned) + len(ldr.memory.rolling)
                for ldr in leaders
            ) if algorithm.uses_memory else 0

            result = DayResult(
                day             = day,
                leader_avg      = leader_avg,
                follower_avg    = follower_avg,
                epsilon         = current_eps,
                velocity        = velocity,
                llm_pos         = llm_vals.count(1),
                llm_neu         = llm_vals.count(0),
                llm_neg         = llm_vals.count(-1),
                is_anchor       = is_anchor,
                phase           = "recovery" if is_recovery else "crisis",
                avg_memory_size = total_mem / max(len(leaders), 1),
            )
            results.append(result)

            if on_progress is not None:
                on_progress(result)

        # Sync final opinions back to agent objects
        for leader in leaders:
            leader.opinion = leader_opinion[leader.agent_id]
        for follower in followers:
            follower.opinion = follower_opinion[follower.agent_id]

        return results

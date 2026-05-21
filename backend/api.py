"""
backend/api.py
==============
FastAPI application for the MA-FDE-LLM simulation platform.

Endpoints:
  GET  /api/health                           — liveness check
  GET  /api/datasets                         — list available datasets
  GET  /api/variants                         — list algorithm variants with default configs
  GET  /api/cache                            — list all cached results
  DELETE /api/cache/{dataset_id}/{variant_id} — clear one cache entry
  POST /api/simulate                         — run (or load from cache) and stream via SSE

SSE event format (text/event-stream):
  data: {"type": "start",      "data": {"source": "cache"|"live", ...}}
  data: {"type": "day",        "data": {...DayResult dict}}
  data: {"type": "evaluation", "data": {...EvaluationResult dict}}
  data: {"type": "error",      "data": {"message": "..."}}
  data: {"type": "done"}
"""

from __future__ import annotations

import asyncio
import json
import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from backend.algorithm import get_algorithm
from backend.config import AlgorithmConfig, get_variant_meta, list_variants
from backend.data_loader import LoadedDataset, list_datasets, load_dataset
from backend.evaluator import evaluate
from backend.results_cache import (
    clear_cache, has_cache, list_cache, load_cache, save_cache,
)
from backend.simulation_runner import DayResult, SimulationRunner


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="MA-FDE-LLM Simulation Platform",
    version="1.0.0",
    description="Modular simulation platform for the Memory-Augmented FDE-LLM algorithm.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_executor = ThreadPoolExecutor(max_workers=4)


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class SimulateRequest(BaseModel):
    dataset_id:       str                        = "jj_vaccine"
    variant_id:       str                        = "mafdm_m3b"
    config:           Optional[Dict[str, Any]]   = None
    force_rerun:      bool                       = False   # True → ignore cache, call LLM
    anchor_overrides: Optional[Dict[str, str]]   = None   # {day_str: blurb} user-edited anchors


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/health")
def health() -> dict:
    import os as _os
    return {
        "status": "ok",
        "service": "MA-FDE-LLM API",
        "live_runs_enabled": bool(_os.environ.get("OPENAI_API_KEY")),
    }


@app.get("/api/datasets")
def get_datasets() -> list:
    return list_datasets()


@app.get("/api/variants")
def get_variants() -> list:
    return list_variants()


@app.get("/api/cache")
def get_cache() -> list:
    """Return all cached dataset-variant combinations."""
    return list_cache()


@app.delete("/api/cache/{dataset_id}/{variant_id}")
def delete_cache(dataset_id: str, variant_id: str) -> dict:
    """Clear the cache for one dataset-variant pair."""
    deleted = clear_cache(dataset_id, variant_id)
    return {
        "deleted": deleted,
        "message": (
            f"Cache cleared for {dataset_id} / {variant_id}."
            if deleted else
            f"No cache found for {dataset_id} / {variant_id}."
        ),
    }


@app.post("/api/simulate")
async def simulate(request: SimulateRequest) -> StreamingResponse:
    """
    Run or load the simulation and stream progress as Server-Sent Events.

    If a cache file exists for this dataset+variant and force_rerun is False,
    results are loaded from the CSV instantly (no LLM calls).
    After a live run, results are saved to CSV for future requests.
    """
    # Validate variant
    try:
        variant_meta = get_variant_meta(request.variant_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    # Merge request config with variant defaults
    base_config = variant_meta.default_config.model_dump()
    if request.config:
        base_config.update({k: v for k, v in request.config.items() if v is not None})
    try:
        config = AlgorithmConfig(**base_config)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Invalid config: {exc}")

    loop = asyncio.get_event_loop()

    def _sse(type_: str, data: Any) -> str:
        payload = json.dumps({"type": type_, "data": data}, ensure_ascii=False)
        return f"data: {payload}\n\n"

    async def event_stream():
        # ── Determine source ───────────────────────────────────────────────
        use_cache = (
            not request.force_rerun
            and has_cache(request.dataset_id, request.variant_id)
        )
        source = "cache" if use_cache else "live"

        yield _sse("start", {
            "variant_id":   request.variant_id,
            "variant_name": variant_meta.name,
            "dataset_id":   request.dataset_id,
            "source":       source,
        })

        # ── Load dataset (always needed for real_curve + anchor metadata) ──
        try:
            dataset: LoadedDataset = await loop.run_in_executor(
                _executor,
                lambda: load_dataset(
                    dataset_id = request.dataset_id,
                    grid_rows  = config.GRID_ROWS,
                    grid_cols  = config.GRID_COLS,
                    seed       = config.SEED,
                ),
            )
        except Exception as exc:
            yield _sse("error", {"message": f"Dataset load failed: {exc}"})
            return

        # ── Apply user anchor overrides (if any) ──────────────────────────
        if request.anchor_overrides:
            for day_str, blurb in request.anchor_overrides.items():
                day = int(day_str)
                if blurb and blurb.strip():
                    dataset.anchor_days[day] = blurb.strip()

        # ── Cache hit: stream from CSV, skip LLM ──────────────────────────
        if use_cache:
            cached = load_cache(request.dataset_id, request.variant_id)
            if cached:
                for result in cached:
                    yield _sse("day", result.to_dict())
                    await asyncio.sleep(0)   # yield control so events flush

                try:
                    eval_result = await loop.run_in_executor(
                        _executor,
                        lambda: evaluate(
                            results      = cached,
                            real_curve   = dataset.real_curve,
                            variant_name = variant_meta.name,
                            event_label  = dataset.event_label,
                            anchor_days  = sorted(dataset.anchor_days.keys()),
                            variant_id   = request.variant_id,
                        ),
                    )
                    yield _sse("evaluation", eval_result.to_dict())
                except Exception as exc:
                    yield _sse("error", {"message": f"Evaluation failed: {exc}"})
                    return

                yield "data: {\"type\": \"done\"}\n\n"
                return

        # ── Live run ───────────────────────────────────────────────────────
        algorithm = get_algorithm(request.variant_id, config)
        runner    = SimulationRunner()
        queue: asyncio.Queue = asyncio.Queue()
        SENTINEL = object()

        def on_progress(result: DayResult) -> None:
            asyncio.run_coroutine_threadsafe(queue.put(result), loop)

        def run_simulation():
            try:
                all_results = runner.run(
                    dataset     = dataset,
                    algorithm   = algorithm,
                    config      = config,
                    on_progress = on_progress,
                )
                asyncio.run_coroutine_threadsafe(queue.put(all_results), loop)
            except Exception as exc:
                asyncio.run_coroutine_threadsafe(
                    queue.put({"__error__": str(exc), "__tb__": traceback.format_exc()}),
                    loop,
                )
            finally:
                asyncio.run_coroutine_threadsafe(queue.put(SENTINEL), loop)

        future = loop.run_in_executor(_executor, run_simulation)

        all_results = None

        while True:
            item = await queue.get()

            if item is SENTINEL:
                break

            if isinstance(item, dict) and "__error__" in item:
                yield _sse("error", {"message": item["__error__"]})
                return

            if isinstance(item, DayResult):
                yield _sse("day", item.to_dict())
            elif isinstance(item, list):
                all_results = item

        await future

        if all_results is None:
            yield _sse("error", {"message": "Simulation produced no results."})
            return

        # ── Save to cache ──────────────────────────────────────────────────
        try:
            await loop.run_in_executor(
                _executor,
                lambda: save_cache(request.dataset_id, request.variant_id, all_results),
            )
        except Exception as exc:
            # Non-fatal — results still returned to frontend
            print(f"  [CACHE SAVE ERROR]: {exc}")

        # ── Evaluation + chart ─────────────────────────────────────────────
        try:
            eval_result = await loop.run_in_executor(
                _executor,
                lambda: evaluate(
                    results      = all_results,
                    real_curve   = dataset.real_curve,
                    variant_name = variant_meta.name,
                    event_label  = dataset.event_label,
                    anchor_days  = sorted(dataset.anchor_days.keys()),
                    variant_id   = request.variant_id,
                ),
            )
            yield _sse("evaluation", eval_result.to_dict())
        except Exception as exc:
            yield _sse("error", {"message": f"Evaluation failed: {exc}"})
            return

        yield "data: {\"type\": \"done\"}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")

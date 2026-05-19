"""
backend/llm_client.py
======================
OpenAI LLM interface for the MA-FDE-LLM simulation.

Responsibilities:
  - Read the API key from the OPENAI_API_KEY environment variable
  - Execute parallel LLM calls for all leaders on a given day
  - Parse and validate LLM outputs (must be -1, 0, or +1)
  - Retry on transient errors with exponential back-off (max 3 attempts)
  - Return a dict {agent_id: int} for every leader
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Dict, List, Optional

from openai import OpenAI


# ---------------------------------------------------------------------------
# Module-level client — initialised lazily on first call
# ---------------------------------------------------------------------------

# Key is read from environment variable first; falls back to the project key
# used in phase_8 simulations.
_FALLBACK_KEY = (
    "sk-proj-d3PDvS0wUVZBwqDu6gvlTB5ytyfcmW5B7k9dTTyWNaWaIwvJarOZYab"
    "h1EZl1erUEq2_s3EwWzT3BlbkFJFm4pnLhWG9ePx82dW3Osdkx8Zs7Z6taFMNoK"
    "VD6l5zejjKrd2v4wcqiQOFo3z7IoCCUdDAsVQA"
)

_client: Optional[OpenAI] = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        api_key = os.environ.get("OPENAI_API_KEY", "") or _FALLBACK_KEY
        _client = OpenAI(api_key=api_key)
    return _client


# ---------------------------------------------------------------------------
# Single-agent LLM call
# ---------------------------------------------------------------------------

def call_llm_single(
    prompt:      str,
    agent_id:    str,
    day:         int,
    model:       str,
    temperature: float,
    max_tokens:  int,
) -> int:
    """
    Call the LLM with the given prompt and return an integer in {-1, 0, +1}.
    Retries up to 3 times on transient errors with 3-second back-off.
    Returns 0 (neutral) on persistent failure.
    """
    client = _get_client()

    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model    = model,
                messages = [{"role": "user", "content": prompt}],
                max_tokens  = max_tokens,
                temperature = temperature,
            )
            text = response.choices[0].message.content.strip().replace("+", "")
            val  = int(text)
            if val in (-1, 0, 1):
                return val
            return 0
        except (ValueError, TypeError):
            return 0
        except Exception as exc:
            if attempt < 2:
                time.sleep(3 * (attempt + 1))
            else:
                print(f"  [LLM ERROR agent={agent_id} day={day:+d}]: {exc}")
                return 0
    return 0


# ---------------------------------------------------------------------------
# Parallel batch call for all leaders on one day
# ---------------------------------------------------------------------------

def call_llm_batch(
    prompts:     Dict[str, str],
    day:         int,
    model:       str,
    temperature: float,
    max_tokens:  int,
    max_workers: int,
    on_result:   Optional[Callable[[str, int], None]] = None,
) -> Dict[str, int]:
    """
    Call the LLM for every agent in parallel.

    Args:
        prompts:     {agent_id: prompt_string}
        day:         simulation day (for error messages)
        model:       OpenAI model name
        temperature: sampling temperature
        max_tokens:  max response tokens
        max_workers: thread pool size
        on_result:   optional callback invoked as each result arrives
                     (used for streaming progress to the frontend)

    Returns:
        {agent_id: llm_output}  — all agents present, value ∈ {-1, 0, 1}
    """
    results: Dict[str, int] = {}

    def _call(aid: str) -> tuple[str, int]:
        val = call_llm_single(
            prompt      = prompts[aid],
            agent_id    = aid,
            day         = day,
            model       = model,
            temperature = temperature,
            max_tokens  = max_tokens,
        )
        return aid, val

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_call, aid): aid for aid in prompts}
        for future in as_completed(futures):
            aid, val = future.result()
            results[aid] = val
            if on_result is not None:
                on_result(aid, val)

    return results

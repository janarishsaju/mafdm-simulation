"""
backend/config.py
=================
Algorithm configuration definitions and variant registry.

Each algorithm variant is described by:
  - A human-readable name and description
  - A full AlgorithmConfig holding all tuneable parameters
  - A flag set indicating which modules are active

Adding a new variant:
  1. Create its class in backend/algorithm/
  2. Add a default AlgorithmConfig entry in VARIANT_DEFAULTS
  3. Register it in VARIANT_REGISTRY pointing to that class

No other file needs to change.
"""

from __future__ import annotations
from typing import Dict, Any
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Parameter model
# ---------------------------------------------------------------------------

class AlgorithmConfig(BaseModel):
    """
    Complete set of tuneable parameters for any MA-FDE-LLM variant.
    All fields have defaults matching the canonical J&J M3b gold standard.
    Frontend sends a JSON body matching this schema.
    """

    # ── CA / opinion dynamics ────────────────────────────────────────────────
    R: float = Field(0.99,  ge=0.0, le=1.0,  description="Opinion self-persistence weight")
    W: float = Field(0.30,  ge=0.0, le=2.0,  description="Neighbourhood influence weight")
    ALPHA: float = Field(0.40, ge=0.0, le=1.0, description="CA/LLM blend for leaders (ALPHA=CA, 1-ALPHA=LLM)")

    # ── SIR attenuation (follower dampening) ─────────────────────────────────
    GAMMA:       float = Field(0.90, ge=0.0, le=1.0, description="Probability of applying dampening to a follower")
    LAM:         float = Field(0.50, ge=0.0, le=5.0, description="Dampening brake strength (symmetric, used by M3b)")
    LAM_FALL:    float = Field(0.30, ge=0.0, le=5.0, description="M4: brake strength for negative drift (loss aversion — trust falls easily)")
    LAM_RECOVER: float = Field(0.70, ge=0.0, le=5.0, description="M4: brake strength for positive drift (loss aversion — recovery is slow)")

    # ── Velocity memory epsilon (M4) ─────────────────────────────────────────
    EPS_VEL_ALPHA: float = Field(0.30, ge=0.0, le=1.0, description="M4: EMA smoothing factor for velocity in epsilon calculation (0=full history, 1=instantaneous)")

    # ── Dynamic epsilon (interaction threshold) ───────────────────────────────
    EPS_INIT:         float = Field(0.90, ge=0.0, le=2.0, description="Starting epsilon (Day -5, no prior velocity)")
    EPS_MIN_CRISIS:   float = Field(0.30, ge=0.0, le=1.0, description="Epsilon floor — crisis phase (Days -5 to +9)")
    EPS_MIN_RECOVERY: float = Field(0.50, ge=0.0, le=1.0, description="Epsilon floor — recovery phase (Days +10 to +25)")
    EPS_MAX:          float = Field(1.50, ge=0.5, le=2.0, description="Epsilon ceiling (maximum at high velocity)")
    EPS_BETA:         float = Field(3.00, ge=0.1, le=10.0, description="Velocity sensitivity (tanh sharpness)")
    EPS_FIXED:        float = Field(0.90, ge=0.0, le=2.0, description="Fixed epsilon used by non-dynamic variants")

    # ── Episodic memory ───────────────────────────────────────────────────────
    MEMORY_THRESHOLD: float = Field(0.08, ge=0.0, le=1.0, description="Min opinion shift to write a rolling diary entry")
    MEMORY_ROLL_K:    int   = Field(6,    ge=1,   le=20,  description="Rolling notebook capacity (entries)")
    MEMORY_PROMPT_N:  int   = Field(4,    ge=1,   le=10,  description="Entries serialised into each LLM prompt")

    # ── Simulation window ─────────────────────────────────────────────────────
    SIM_START: int = Field(-5,  description="First simulation day relative to crisis (Day 0)")
    SIM_END:   int = Field(25,  description="Last simulation day relative to crisis (Day 0)")

    # ── CA grid ───────────────────────────────────────────────────────────────
    GRID_ROWS: int = Field(13, ge=5, le=50, description="Leader CA grid rows")
    GRID_COLS: int = Field(12, ge=5, le=50, description="Leader CA grid columns")

    # ── Event structure ───────────────────────────────────────────────────────
    RESOLUTION_ANCHOR_DAY: int = Field(
        10,
        description=(
            "Day number of the resolution/clearance anchor (e.g. Day 10 = J&J lift). "
            "Used by memory-enabled variants to switch from crisis to recovery prompt. "
            "Set to -999 if there is no resolution anchor in the dataset."
        ),
    )

    # ── Random seed ───────────────────────────────────────────────────────────
    SEED: int = Field(42, description="Random seed for reproducibility")

    # ── LLM ───────────────────────────────────────────────────────────────────
    LLM_MODEL:       str = Field("gpt-4o",  description="OpenAI model name")
    LLM_TEMPERATURE: float = Field(0.0, ge=0.0, le=2.0, description="LLM temperature (0 = deterministic)")
    LLM_MAX_TOKENS:  int   = Field(5,   ge=1,   le=50,  description="Max tokens in LLM response")
    LLM_MAX_WORKERS: int   = Field(15,  ge=1,   le=50,  description="Parallel LLM call workers")


# ---------------------------------------------------------------------------
# Variant metadata
# ---------------------------------------------------------------------------

class VariantMeta(BaseModel):
    """Human-readable metadata shown in the frontend dropdown."""
    id:          str
    name:        str
    description: str
    has_memory:  bool
    has_dynamic_eps: bool
    default_config: AlgorithmConfig


# ---------------------------------------------------------------------------
# Variant registry
# ---------------------------------------------------------------------------
# Maps variant_id → VariantMeta.
# The algorithm class reference is resolved lazily in algorithm/__init__.py
# to avoid circular imports.

VARIANT_DEFAULTS: Dict[str, VariantMeta] = {

    "mafdm_m3b": VariantMeta(
        id="mafdm_m3b",
        name="MA-FDE-LLM — Memory + Dynamic ε",
        description=(
            "Memory-Augmented FDE-LLM with phase-conditional dynamic ε. "
            "Gold standard: Pearson +0.944, DTW 0.496. "
            "Uses episodic memory diary + phase-conditional ε floor (0.30 crisis / 0.50 recovery)."
        ),
        has_memory=True,
        has_dynamic_eps=True,
        default_config=AlgorithmConfig(),
    ),

    "memory_only": VariantMeta(
        id="memory_only",
        name="FDE-LLM + Memory (Fixed ε)",
        description=(
            "FDE-LLM baseline with episodic memory diary added. Fixed ε = 0.90 (no dynamic adjustment). "
            "J&J ablation result: Pearson +0.740, DTW 0.482."
        ),
        has_memory=True,
        has_dynamic_eps=False,
        default_config=AlgorithmConfig(
            EPS_FIXED=0.90,
        ),
    ),

    "fde_llm": VariantMeta(
        id="fde_llm",
        name="FDE-LLM Baseline — No Memory, Fixed ε",
        description=(
            "Exact replication of Yao et al. (2025) FDE-LLM. "
            "No memory diary. Fixed ε = 0.90. "
            "J&J result: Pearson +0.909, DTW 1.854."
        ),
        has_memory=False,
        has_dynamic_eps=False,
        default_config=AlgorithmConfig(
            EPS_FIXED=0.90,
        ),
    ),

    "real_leaders": VariantMeta(
        id="real_leaders",
        name="Real Leader Scores → Simulated Followers",
        description=(
            "Leaders use actual recorded attitude scores from the dataset CSV (no LLM). "
            "Follower curve is fully simulated using M3b CA + SIR dynamics driven by real leader signals. "
            "Shows what follower dynamics alone can achieve when leader input is perfect. Runs instantly."
        ),
        has_memory=False,
        has_dynamic_eps=True,
        default_config=AlgorithmConfig(),
    ),

    "real_leaders_networked": VariantMeta(
        id="real_leaders_networked",
        name="Real Leader Scores → Actual Connections",
        description=(
            "Same real leader scores as the base variant, but follower connections "
            "are derived from actual data: subreddit co-participation (272), "
            "content-theme matching (71), theme fallback (16), and Reddit thread "
            "co-commenting via Arctic Shift scraping (3). "
            "Isolates the effect of network structure on follower dynamics. Runs instantly."
        ),
        has_memory=False,
        has_dynamic_eps=True,
        default_config=AlgorithmConfig(),
    ),

    "bc_sir": VariantMeta(
        id="bc_sir",
        name="Bounded Confidence + SIR (No LLM)",
        description=(
            "Bounded confidence CA model with SIR follower dampening added. No LLM, no memory. "
            "Isolates the contribution of SIR dampening independently of the LLM signal. "
            "Leaders: pure CA. Followers: CA + exp(−λ|ca_base|) brake (γ=0.90, λ=0.50)."
        ),
        has_memory=False,
        has_dynamic_eps=False,
        default_config=AlgorithmConfig(
            EPS_FIXED=0.90,
        ),
    ),

    "bounded_confidence": VariantMeta(
        id="bounded_confidence",
        name="Bounded Confidence Only (No LLM)",
        description=(
            "Pure CA bounded confidence model. No LLM calls, no memory, no SIR dampening. "
            "Agents interact only when |O_i − O_j| ≤ ε (fixed). "
            "Mathematical baseline — shows what network diffusion alone produces."
        ),
        has_memory=False,
        has_dynamic_eps=False,
        default_config=AlgorithmConfig(
            EPS_FIXED=0.90,
        ),
    ),

    "custom": VariantMeta(
        id="custom",
        name="Custom — Editable Variant",
        description=(
            "Fully editable variant. Start from M3b defaults and modify any parameter. "
            "Use this to test new ideas without affecting the canonical variants."
        ),
        has_memory=True,
        has_dynamic_eps=True,
        default_config=AlgorithmConfig(),
    ),
}


def get_variant_meta(variant_id: str) -> VariantMeta:
    if variant_id not in VARIANT_DEFAULTS:
        raise ValueError(
            f"Unknown variant '{variant_id}'. "
            f"Available: {list(VARIANT_DEFAULTS.keys())}"
        )
    return VARIANT_DEFAULTS[variant_id]


def list_variants() -> list[Dict[str, Any]]:
    """Return variant list safe for JSON serialisation."""
    return [
        {
            "id":              v.id,
            "name":            v.name,
            "description":     v.description,
            "has_memory":      v.has_memory,
            "has_dynamic_eps": v.has_dynamic_eps,
            "default_config":  v.default_config.model_dump(),
        }
        for v in VARIANT_DEFAULTS.values()
    ]

"""
backend/algorithm/__init__.py
==============================
Variant registry — maps variant_id to its concrete algorithm class.

Adding a new variant:
  1. Create the class in backend/algorithm/<name>.py
  2. Import it here
  3. Add an entry to VARIANT_REGISTRY
  4. Add default config in backend/config.py VARIANT_DEFAULTS

No other file needs to change.
"""

from backend.algorithm.mafdm_m3b         import MAFDM_M3b
from backend.algorithm.fde_llm           import FDE_LLM
from backend.algorithm.memory_only       import MemoryOnly
from backend.algorithm.custom            import CustomVariant
from backend.algorithm.bounded_confidence import BoundedConfidence
from backend.algorithm.bc_sir            import BoundedConfidenceSIR
from backend.algorithm.base              import AbstractAlgorithm

VARIANT_REGISTRY: dict[str, type[AbstractAlgorithm]] = {
    "mafdm_m3b":          MAFDM_M3b,
    "memory_only":        MemoryOnly,
    "fde_llm":            FDE_LLM,
    "bounded_confidence": BoundedConfidence,
    "bc_sir":             BoundedConfidenceSIR,
    "custom":             CustomVariant,
}


def get_algorithm(variant_id: str, config) -> AbstractAlgorithm:
    """
    Instantiate and return the algorithm for a given variant_id.

    Args:
        variant_id: one of the keys in VARIANT_REGISTRY
        config:     AlgorithmConfig instance

    Returns:
        Concrete AbstractAlgorithm subclass instance
    """
    if variant_id not in VARIANT_REGISTRY:
        raise ValueError(
            f"Unknown variant '{variant_id}'. "
            f"Available: {list(VARIANT_REGISTRY.keys())}"
        )
    return VARIANT_REGISTRY[variant_id](config)

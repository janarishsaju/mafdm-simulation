"""
backend/algorithm/custom.py
============================
Custom variant — fully editable, starts from M3b defaults.

Identical to MAFDM_M3b at construction time.
Use this variant to test new ideas without modifying the canonical implementations.
Both memory and dynamic ε are active unless the frontend config overrides them.
"""

from backend.algorithm.mafdm_m3b import MAFDM_M3b


class CustomVariant(MAFDM_M3b):
    """
    Inherits the full M3b implementation.
    Override any method here when experimenting with new algorithmic ideas.
    All config parameters are settable from the frontend.
    """

    @property
    def uses_memory(self) -> bool:
        return True

    @property
    def uses_dynamic_eps(self) -> bool:
        return True

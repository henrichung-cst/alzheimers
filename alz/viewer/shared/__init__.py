"""Shared viewer slice schema and composer primitives.

``cohort_slice`` — CohortViewerSlice dataclass (the unit one cohort adapter
  returns).
``compose`` — compose_viewer_slices() merges a list of slices into the unified
  payload dict.

These modules are SCHEMA ONLY in Phase 5B.  They are not wired into any
builder yet.  Phase 5C–5E create per-cohort adapters; Phase 5F wires the
composer into build_unified_viewer.build_payload().
"""
from .cohort_slice import CohortViewerSlice, EdgeSliceContribution
from .compose import compose_viewer_slices

__all__ = [
    "CohortViewerSlice",
    "EdgeSliceContribution",
    "compose_viewer_slices",
]

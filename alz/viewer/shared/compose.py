"""``compose_viewer_slices`` — reduce per-cohort slices into the unified payload.

The composer is the inverse of the cohort adapters: each adapter returns a
:class:`~alz.viewer.shared.cohort_slice.CohortViewerSlice`; the composer places
the owned sections, merges the shared-key contributions, and emits the top-level
keys in the EXACT order the current ``build_unified_viewer.build_payload()``
emits them, so the assembled payload is structurally identical to today's.

Phase 5B defines and unit-exercises this reducer; it is NOT yet called from
``build_unified_viewer``. Phase 5C wires the first real adapter (mukesh) through
it and verifies payload parity; 5D/5E add tcell/fivexfad; 5F folds song and
retires the inline ``build_payload`` assembly.

Cross-cohort note (R2 retired in wave 5E): the composer reaches across no
cohort and imposes no inter-cohort ordering constraint. Each adapter receives
only ``data: UnifiedData`` (the shared data object every cohort reads) and
builds its slice independently. The fivexfad adapter consumes its OWN bulk MEA
rows from ``UnifiedData``; there is no song→fivexfad data seam. The composer's
role is solely to place owned sections and merge shared-key contributions from
already-built slices.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

from .cohort_slice import SHARED_TOP_LEVEL_KEYS, CohortViewerSlice

# The canonical top-level payload key order emitted by
# build_unified_viewer.build_payload(). Optional cohort blocks ("human",
# "supporting_5xfad") are emitted only when present. Keeping this list in one
# place is what guarantees the composed payload's key order matches today's.
TOP_LEVEL_ORDER: tuple[str, ...] = (
    "meta",
    "kinases",
    "kinase_motifs",
    "celltypes",
    "kinase_celltype_evidence",
    "motif_peer_narrowing",
    "attribution_index",
    "specificity_units",
    "mechanism_attribution",
    "decomposition_index",
    "agreement_index",
    "subclass_breakdown",
    "audit_tables",
    "edge_slice_ref",
    "incytr_pathways",
    "human",
    "supporting_5xfad",
    "substrate_compare",
)

def merge_capabilities(
    base: dict[str, Any], slices: Iterable[CohortViewerSlice]
) -> dict[str, Any]:
    """Fold each slice's capability flags onto ``base`` (a copy of
    ``meta.capabilities``). A flag already ``True`` is never demoted; later
    slices can only promote ``False`` → ``True`` (the two-pass behavior of the
    current builder, R4)."""
    merged = dict(base)
    for sl in slices:
        for flag, value in sl.capabilities.items():
            merged[flag] = bool(merged.get(flag, False) or value)
    return merged


def merge_audit_tables(
    base_manifest: dict[str, Any], slices: Iterable[CohortViewerSlice]
) -> dict[str, Any]:
    """Merge per-cohort ``audit_entries`` into a copy of the base audit manifest
    (R1). ``base_manifest`` is the manifest shell (``preview_rows``, ``tables``,
    ``measurement_trace``); each slice's entries are added under ``tables`` by
    key. A duplicate table key across cohorts is a contract collision, raised."""
    merged = dict(base_manifest)
    tables: dict[str, Any] = dict(merged.get("tables", {}))
    for sl in slices:
        for key, entry in sl.audit_entries.items():
            if key in tables:
                raise ValueError(
                    f"audit_tables key {key!r} contributed by more than one "
                    f"cohort (latest: {sl.cohort_id!r})"
                )
            tables[key] = entry
    merged["tables"] = tables
    return merged


def merge_edge_slice_ref(
    base_ref: dict[str, Any], slices: Iterable[CohortViewerSlice]
) -> dict[str, Any]:
    """Aggregate every cohort's lazy-shard pointers into a copy of ``base_ref``
    (R8). Keys are the exact payload key names.

    ``base_ref`` supplies defaults (e.g. ``present_human_perdonor_kinase_ids: []``
    for the mouse-only case). A cohort slice may override a base-provided key
    (e.g. mukesh overrides the empty default with its real list). Two different
    slices contributing the same key is a collision, raised rather than silently
    overwriting."""
    merged = dict(base_ref)
    seen_from_slice: dict[str, str] = {}  # key -> cohort_id that contributed it
    for sl in slices:
        for key, value in sl.edge_slice_ref_entries().items():
            if key in seen_from_slice:
                raise ValueError(
                    f"edge_slice_ref key {key!r} contributed by more than one "
                    f"cohort ({seen_from_slice[key]!r} and {sl.cohort_id!r})"
                )
            merged[key] = value
            seen_from_slice[key] = sl.cohort_id
    return merged


def union_kinase_names(slices: Iterable[CohortViewerSlice]) -> list[str]:
    """Sorted union of every cohort's kinase names — the input to the motif
    builder (R3). Matches the current ``sorted(motif_names)`` union over song,
    human, and 5xFAD kinase names."""
    names: set[str] = set()
    for sl in slices:
        names.update(sl.kinase_names)
    return sorted(names)


def compose_viewer_slices(
    slices: list[CohortViewerSlice],
    *,
    meta: dict[str, Any],
    audit_manifest_base: dict[str, Any],
    edge_slice_ref_base: dict[str, Any],
    kinase_motifs_builder: Callable[[list[str]], dict[str, Any]],
) -> dict[str, Any]:
    """Assemble the unified payload dict from per-cohort slices.

    Parameters
    ----------
    slices:
        Built cohort slices. Owned-section keys must be disjoint across slices
        (a top-level key is owned by exactly one cohort); a collision is raised.
    meta:
        The composer-assembled ``meta`` block. Its ``capabilities`` sub-dict is
        the base onto which each slice's capability flags are merged.
    audit_manifest_base:
        The audit-manifest shell (``preview_rows``/``tables``/``measurement_trace``)
        before per-cohort ``audit_entries`` are merged in.
    edge_slice_ref_base:
        The ``edge_slice_ref`` shell (e.g. ``schema_version``) before per-cohort
        shard pointers are merged in.
    kinase_motifs_builder:
        ``names -> {name: {kin_type, positions, amino_acids, matrix, st_fav}}``.
        Injected so the composer never reaches the PSSM data source itself.

    Returns
    -------
    dict
        Payload with top-level keys in :data:`TOP_LEVEL_ORDER` (optional cohort
        blocks present only when a slice owns them).
    """
    assembled: dict[str, Any] = {}

    # Owned sections — exactly one cohort per top-level key.
    for sl in slices:
        for key, value in sl.owned_sections.items():
            if key in SHARED_TOP_LEVEL_KEYS:
                # Defense-in-depth; CohortViewerSlice already rejects this.
                raise ValueError(
                    f"cohort {sl.cohort_id!r} owned_sections key {key!r} is "
                    "composer-owned"
                )
            if key in assembled:
                raise ValueError(
                    f"top-level key {key!r} owned by more than one cohort "
                    f"(latest: {sl.cohort_id!r})"
                )
            assembled[key] = value

    # Shared keys — merged from contributions.
    assembled["meta"] = {**meta, "capabilities": merge_capabilities(
        meta.get("capabilities", {}), slices
    )}
    assembled["kinase_motifs"] = kinase_motifs_builder(union_kinase_names(slices))
    assembled["audit_tables"] = merge_audit_tables(audit_manifest_base, slices)
    assembled["edge_slice_ref"] = merge_edge_slice_ref(edge_slice_ref_base, slices)

    # Emit in canonical order; optional blocks only when present.
    payload = {key: assembled[key] for key in TOP_LEVEL_ORDER if key in assembled}

    # Anything a cohort owned that is not in the canonical order is a contract
    # gap — surface it rather than dropping it silently.
    leftover = set(assembled) - set(payload)
    if leftover:
        raise ValueError(
            f"composed keys not in TOP_LEVEL_ORDER (contract drift): "
            f"{sorted(leftover)}"
        )
    return payload

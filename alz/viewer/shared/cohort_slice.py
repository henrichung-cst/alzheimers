"""``CohortViewerSlice`` — the unit one cohort adapter returns.

A slice carries everything ONE cohort contributes to the unified viewer payload,
split into two disjoint parts:

* **owned sections** (``owned_sections``) — top-level payload keys this cohort
  alone produces. The composer places each at the payload top level verbatim
  (mukesh owns ``"human"``; fivexfad owns ``"supporting_5xfad"``; song owns
  ``"kinases"``, ``"celltypes"``, ``"attribution_index"``, … ). No re-mapping.

* **merge contributions** — values that several cohorts feed into a SHARED
  top-level key, which the composer reduces across all slices:
    - ``capabilities``    → ``meta.capabilities``
    - ``audit_entries``   → ``audit_tables.tables``
    - ``edge_slice_ref``  → ``edge_slice_ref`` (flat per-shard-family pointers)
    - ``kinase_names``    → ``kinase_motifs`` (name union; the PSSM matrices are
      built by the composer's injected motif builder, not here)

This module defines the contract only. The emitted payload must stay
structurally identical to today's (no new/renamed keys, no version bump); the
slice is an internal representation, verified at first wiring (Phase 5C). See
``alz/viewer/shared/compose.py`` for the reducer and the canonical key order.
See ``docs/audits/cohort_abstraction_refactor/phase_5A_payload_inventory.md`` for
the full field map this schema mirrors.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Top-level keys the composer assembles by merging contributions from every
# slice. A cohort must NOT place any of these in ``owned_sections`` — they are
# never owned by a single cohort.
SHARED_TOP_LEVEL_KEYS: frozenset[str] = frozenset(
    {"meta", "kinase_motifs", "audit_tables", "edge_slice_ref"}
)


@dataclass(frozen=True)
class EdgeSliceContribution:
    """One lazy-shard family's entries in the flat ``edge_slice_ref`` map.

    ``entries`` is merged verbatim into ``edge_slice_ref`` by the composer; its
    keys are the EXACT payload key names so nothing is re-mapped downstream — e.g.
    the mukesh ``human_perdonor`` family contributes
    ``{"human_perdonor_url": ..., "human_perdonor_index": ...,
       "present_human_perdonor_kinase_ids": [...]}``.

    ``family`` is a human-readable tag (``"human_perdonor"``, ``"decomp_ols"``,
    ``"song_concordance"``, ``"incytr_pathways"``) used only for diagnostics and
    collision reporting; it is not emitted into the payload.
    """

    family: str
    entries: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.family:
            raise ValueError("EdgeSliceContribution.family must be non-empty")


@dataclass(frozen=True)
class CohortViewerSlice:
    """Everything one cohort contributes to the unified viewer payload.

    Parameters
    ----------
    cohort_id:
        ``"song"`` | ``"mukesh"`` | ``"tcell"`` | ``"fivexfad"``.
    context_ids:
        The context ids this cohort populates (``("song_ad",)`` for song,
        ``("donor1", "donor2")`` for tcell). Used by the composer to validate
        ``meta.contexts`` coverage.
    owned_sections:
        Top-level payload keys owned solely by this cohort, value-verbatim.
        Must not intersect :data:`SHARED_TOP_LEVEL_KEYS`.
    capabilities:
        This cohort's ``meta.capabilities`` flag contributions (e.g.
        ``{"human_reference": True}``).
    audit_entries:
        This cohort's ``audit_tables.tables`` entries, keyed by table key.
    edge_slice_ref:
        This cohort's lazy-shard pointer contributions.
    kinase_names:
        Kinase names this cohort contributes to the ``kinase_motifs`` union.
    provenance:
        Free-form build provenance (source paths, row counts, schema versions);
        not emitted into the payload, recorded for the monitoring report.
    """

    cohort_id: str
    context_ids: tuple[str, ...]
    owned_sections: dict[str, Any] = field(default_factory=dict)
    capabilities: dict[str, bool] = field(default_factory=dict)
    audit_entries: dict[str, Any] = field(default_factory=dict)
    edge_slice_ref: tuple[EdgeSliceContribution, ...] = ()
    kinase_names: tuple[str, ...] = ()
    provenance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.cohort_id:
            raise ValueError("CohortViewerSlice.cohort_id must be non-empty")
        collisions = SHARED_TOP_LEVEL_KEYS.intersection(self.owned_sections)
        if collisions:
            raise ValueError(
                f"cohort {self.cohort_id!r} claims shared composer-owned keys as "
                f"owned_sections: {sorted(collisions)}. These are assembled by the "
                "composer from per-cohort contributions, not owned by one cohort."
            )

    def edge_slice_ref_entries(self) -> dict[str, Any]:
        """Flatten this cohort's shard-family pointers into one dict.

        Raises if two families declare the same payload key — that is a real
        contract collision, not a silent last-writer-wins.
        """
        merged: dict[str, Any] = {}
        for contribution in self.edge_slice_ref:
            for key, value in contribution.entries.items():
                if key in merged:
                    raise ValueError(
                        f"cohort {self.cohort_id!r} edge_slice_ref key {key!r} "
                        f"declared by two shard families (second: "
                        f"{contribution.family!r})"
                    )
                merged[key] = value
        return merged

#!/usr/bin/env python3
"""Canonical T-cell marker-set definitions (the single source of truth).

Used by the per-cell AUROC validation and mirrored into the evidence report's
glossary. These are canonical marker biology; the panels named *_lineage carry
cell-type (CD4/CD8) info, tcell_core is pan-T, and the rest are functional
sub-state panels.
"""
from __future__ import annotations

SIGNATURES = {
    "tcell_core": ["CD3D", "CD3E", "TRAC", "ZAP70"],
    "cd4_lineage": ["CD4", "IL7R", "CCR7", "LTB"],
    "cd8_lineage": ["CD8A", "CD8B", "NKG7"],
    "exhaustion": ["PDCD1", "CTLA4", "LAG3", "HAVCR2", "TIGIT", "TOX", "ENTPD1"],
    "progenitor_exhaustion": ["TCF7", "LEF1", "SELL", "CCR7", "IL7R"],
    "cytotoxic": ["GZMB", "GZMH", "GNLY", "NKG7", "PRF1", "EOMES"],
    "th17": ["RORC", "CCR6", "IL17A", "IL17F", "IL23R", "KLRB1"],
    "tfh": ["CXCR5", "BCL6", "ICOS", "PDCD1", "IL21"],
    "treg": ["FOXP3", "IL2RA", "CTLA4", "IKZF2"],
    "naive_memory": ["CCR7", "SELL", "TCF7", "LEF1", "IL7R"],
}
LINEAGE_PANELS = {"cd4_lineage", "cd8_lineage"}
CORE_PANELS = {"tcell_core"}


def _marker_class(panel: str) -> str:
    if panel in LINEAGE_PANELS:
        return "type"
    if panel in CORE_PANELS:
        return "core"
    return "state"

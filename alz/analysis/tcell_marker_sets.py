#!/usr/bin/env python3
"""Canonical T-cell marker-set definitions (the single source of truth).

Used by Matt's historical AUROC reproduction and the current native-cluster
annotation. These are canonical marker biology; the panels named *_lineage carry
cell-type (CD4/CD8) info, tcell_core is pan-T, and the rest are functional
sub-state panels.

``SIGNATURES`` is frozen to the panels used in Matt's historical report. The
additional tuples are non-cycle programs used as positive and negative evidence
for the current per-cell classifier. Internal signed module values are not
exported as analysis scores.
"""
from __future__ import annotations

from dataclasses import dataclass

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


EXHAUSTION_MARKERS = tuple(SIGNATURES["exhaustion"])
TERMINAL_EXHAUSTION_MARKERS = ("LAG3", "HAVCR2", "TOX", "ENTPD1")
CHECKPOINT_CONTEXT_MARKERS = ("PDCD1", "TIGIT")
PROGENITOR_MARKERS = tuple(SIGNATURES["progenitor_exhaustion"])
CYTOTOXIC_MARKERS = tuple(SIGNATURES["cytotoxic"])
TERMINAL_EFFECTOR_MARKERS = ("GZMB", "GNLY", "PRF1")
ACTIVATION_MARKERS = (
    "CD69",
    "IL2RA",
    "TNFRSF4",
    "ICOS",
    "CD40LG",
    "IFNG",
    "TNF",
    "MIR155HG",
    "IRF4",
)


@dataclass(frozen=True)
class MarkerModule:
    """A coherent non-cycle RNA program."""

    name: str
    genes: tuple[str, ...]


@dataclass(frozen=True)
class PerCellStateDefinition:
    """Expected high and low programs for one lineage-specific state."""

    lineage: str
    type_name: str
    positive_modules: tuple[MarkerModule, ...]
    negative_modules: tuple[MarkerModule, ...]


PROGENITOR_MODULE = MarkerModule("progenitor/stem-like", PROGENITOR_MARKERS)
CHECKPOINT_CONTEXT_MODULE = MarkerModule(
    "checkpoint context", CHECKPOINT_CONTEXT_MARKERS
)
TERMINAL_EXHAUSTION_MODULE = MarkerModule(
    "terminal exhaustion", TERMINAL_EXHAUSTION_MARKERS
)
CYTOTOXIC_MODULE = MarkerModule("cytotoxic machinery", CYTOTOXIC_MARKERS)
TERMINAL_EFFECTOR_MODULE = MarkerModule(
    "terminal effector", TERMINAL_EFFECTOR_MARKERS
)
ACTIVATION_MODULE = MarkerModule("activation", ACTIVATION_MARKERS)


PER_CELL_STATE_DEFINITIONS = {
    "CD8 precursor exhausted (TPEX)": PerCellStateDefinition(
        "CD8",
        "CD8PrecursorExhausted",
        (PROGENITOR_MODULE, CHECKPOINT_CONTEXT_MODULE),
        (TERMINAL_EXHAUSTION_MODULE, TERMINAL_EFFECTOR_MODULE),
    ),
    "CD8 exhausted (TEX)": PerCellStateDefinition(
        "CD8",
        "CD8Exhausted",
        (TERMINAL_EXHAUSTION_MODULE,),
        (PROGENITOR_MODULE,),
    ),
    "CD8 cytotoxic effector": PerCellStateDefinition(
        "CD8",
        "CD8CytotoxicEffector",
        (CYTOTOXIC_MODULE,),
        (TERMINAL_EXHAUSTION_MODULE,),
    ),
    "CD8 activated": PerCellStateDefinition(
        "CD8",
        "CD8Activated",
        (ACTIVATION_MODULE,),
        (TERMINAL_EXHAUSTION_MODULE,),
    ),
    "CD8 naive/memory": PerCellStateDefinition(
        "CD8",
        "CD8NaiveMemory",
        (PROGENITOR_MODULE,),
        (
            CHECKPOINT_CONTEXT_MODULE,
            TERMINAL_EXHAUSTION_MODULE,
            CYTOTOXIC_MODULE,
            ACTIVATION_MODULE,
        ),
    ),
    "CD4 exhaustion-associated": PerCellStateDefinition(
        "CD4",
        "CD4ExhaustionAssociated",
        (TERMINAL_EXHAUSTION_MODULE,),
        (PROGENITOR_MODULE,),
    ),
    "CD4 cytotoxic": PerCellStateDefinition(
        "CD4",
        "CD4Cytotoxic",
        (CYTOTOXIC_MODULE,),
        (PROGENITOR_MODULE,),
    ),
    "CD4 activated": PerCellStateDefinition(
        "CD4",
        "CD4Activated",
        (ACTIVATION_MODULE,),
        (TERMINAL_EXHAUSTION_MODULE,),
    ),
    "CD4 naive/memory": PerCellStateDefinition(
        "CD4",
        "CD4NaiveMemory",
        (PROGENITOR_MODULE,),
        (
            CHECKPOINT_CONTEXT_MODULE,
            TERMINAL_EXHAUSTION_MODULE,
            CYTOTOXIC_MODULE,
            ACTIVATION_MODULE,
        ),
    ),
}

COLLAPSED_STATE_LABELS = {
    "CD8 precursor exhausted (TPEX)": "CD8",
}

STATE_COLORS = {
    "CD4": "#80cdc1",
    "CD4 activated": "#fdb863",
    "CD4 cytotoxic": "#c2a5cf",
    "CD4 exhaustion-associated": "#d6604d",
    "CD4 naive/memory": "#4393c3",
    "CD8": "#5e3c99",
    "CD8 activated": "#f46d43",
    "CD8 cytotoxic effector": "#7b3294",
    "CD8 exhausted (TEX)": "#b2182b",
    "CD8 naive/memory": "#2166ac",
    "contaminant": "#404040",
}


def per_cell_marker_genes() -> tuple[str, ...]:
    """Return the ordered union of non-cycle per-cell classifier markers."""
    return tuple(
        dict.fromkeys(
            gene
            for module in (
                TERMINAL_EXHAUSTION_MARKERS,
                CHECKPOINT_CONTEXT_MARKERS,
                PROGENITOR_MARKERS,
                CYTOTOXIC_MARKERS,
                TERMINAL_EFFECTOR_MARKERS,
                ACTIVATION_MARKERS,
            )
            for gene in module
        )
    )


def _marker_class(panel: str) -> str:
    if panel in LINEAGE_PANELS:
        return "type"
    if panel in CORE_PANELS:
        return "core"
    return "state"

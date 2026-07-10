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
NAIVE_STEMNESS_MARKERS = ("TCF7", "LEF1")
NAIVE_HOMING_MARKERS = ("CCR7", "SELL")
RESTING_MEMORY_MARKERS = ("IL7R", "CD27")
INHIBITORY_RECEPTOR_MARKERS = (
    "PDCD1",
    "HAVCR2",
    "LAG3",
    "CTLA4",
    "TIGIT",
    "ENTPD1",
)
CHECKPOINT_RECEPTOR_MARKERS = ("PDCD1", "CTLA4", "TIGIT")
TERMINAL_INHIBITORY_RECEPTOR_MARKERS = ("HAVCR2", "LAG3", "ENTPD1")
EXHAUSTION_TF_MARKERS = ("TOX", "NR4A1")
LATE_EXHAUSTION_SIGNATURE_MARKERS = ("HAVCR2", "LAG3", "ENTPD1", "TOX", "NR4A1")
ACUTE_ACTIVATION_MARKERS = ("CD69", "IL2RA", "TNFRSF4", "ICOS", "CD40LG")
EFFECTOR_FUNCTION_MARKERS = ("GZMB", "PRF1", "IFNG", "TNF")
GRANZYME_MARKERS = ("GZMB", "GZMH", "GNLY")
PERFORIN_MARKERS = ("PRF1",)
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


CYTOTOXIC_MODULE = MarkerModule("cytotoxic machinery", CYTOTOXIC_MARKERS)
NAIVE_STEMNESS_MODULE = MarkerModule("naive stemness", NAIVE_STEMNESS_MARKERS)
NAIVE_HOMING_MODULE = MarkerModule("naive homing", NAIVE_HOMING_MARKERS)
RESTING_MEMORY_MODULE = MarkerModule(
    "resting/memory identity", RESTING_MEMORY_MARKERS
)
INHIBITORY_RECEPTOR_MODULE = MarkerModule(
    "inhibitory receptors", INHIBITORY_RECEPTOR_MARKERS
)
EXHAUSTION_TF_MODULE = MarkerModule(
    "exhaustion transcription factors", EXHAUSTION_TF_MARKERS
)
LATE_EXHAUSTION_SIGNATURE_MODULE = MarkerModule(
    "late exhaustion signature", LATE_EXHAUSTION_SIGNATURE_MARKERS
)
ACUTE_ACTIVATION_MODULE = MarkerModule(
    "acute activation", ACUTE_ACTIVATION_MARKERS
)
EFFECTOR_FUNCTION_MODULE = MarkerModule(
    "effector function", EFFECTOR_FUNCTION_MARKERS
)
GRANZYME_MODULE = MarkerModule("granzyme program", GRANZYME_MARKERS)
PERFORIN_MODULE = MarkerModule("perforin", PERFORIN_MARKERS)


PER_CELL_STATE_DEFINITIONS = {
    "CD8 exhausted (TEX)": PerCellStateDefinition(
        "CD8",
        "CD8Exhausted",
        (LATE_EXHAUSTION_SIGNATURE_MODULE,),
        (ACUTE_ACTIVATION_MODULE, EFFECTOR_FUNCTION_MODULE),
    ),
    "CD8 cytotoxic effector": PerCellStateDefinition(
        "CD8",
        "CD8CytotoxicEffector",
        (GRANZYME_MODULE, PERFORIN_MODULE),
        (INHIBITORY_RECEPTOR_MODULE, EXHAUSTION_TF_MODULE),
    ),
    "CD8 activated/effector": PerCellStateDefinition(
        "CD8",
        "CD8ActivatedEffector",
        (ACUTE_ACTIVATION_MODULE, EFFECTOR_FUNCTION_MODULE),
        (INHIBITORY_RECEPTOR_MODULE, EXHAUSTION_TF_MODULE),
    ),
    "CD8 naive-like": PerCellStateDefinition(
        "CD8",
        "CD8NaiveLike",
        (NAIVE_STEMNESS_MODULE, NAIVE_HOMING_MODULE),
        (
            INHIBITORY_RECEPTOR_MODULE,
            EXHAUSTION_TF_MODULE,
            CYTOTOXIC_MODULE,
            ACUTE_ACTIVATION_MODULE,
        ),
    ),
    "CD8 resting/memory": PerCellStateDefinition(
        "CD8",
        "CD8RestingMemory",
        (RESTING_MEMORY_MODULE,),
        (INHIBITORY_RECEPTOR_MODULE, EXHAUSTION_TF_MODULE, ACUTE_ACTIVATION_MODULE),
    ),
    "CD4 exhaustion-associated": PerCellStateDefinition(
        "CD4",
        "CD4ExhaustionAssociated",
        (LATE_EXHAUSTION_SIGNATURE_MODULE,),
        (ACUTE_ACTIVATION_MODULE, EFFECTOR_FUNCTION_MODULE),
    ),
    "CD4 cytotoxic": PerCellStateDefinition(
        "CD4",
        "CD4Cytotoxic",
        (GRANZYME_MODULE, PERFORIN_MODULE),
        (INHIBITORY_RECEPTOR_MODULE, EXHAUSTION_TF_MODULE),
    ),
    "CD4 activated/effector": PerCellStateDefinition(
        "CD4",
        "CD4ActivatedEffector",
        (ACUTE_ACTIVATION_MODULE, EFFECTOR_FUNCTION_MODULE),
        (INHIBITORY_RECEPTOR_MODULE, EXHAUSTION_TF_MODULE),
    ),
    "CD4 naive-like": PerCellStateDefinition(
        "CD4",
        "CD4NaiveLike",
        (NAIVE_STEMNESS_MODULE, NAIVE_HOMING_MODULE),
        (
            INHIBITORY_RECEPTOR_MODULE,
            EXHAUSTION_TF_MODULE,
            CYTOTOXIC_MODULE,
            ACUTE_ACTIVATION_MODULE,
        ),
    ),
    "CD4 resting/memory": PerCellStateDefinition(
        "CD4",
        "CD4RestingMemory",
        (RESTING_MEMORY_MODULE,),
        (INHIBITORY_RECEPTOR_MODULE, EXHAUSTION_TF_MODULE, ACUTE_ACTIVATION_MODULE),
    ),
}

COLLAPSED_STATE_LABELS = {
    "CD8 precursor exhausted (TPEX)": "CD8",
}

STATE_COLORS = {
    "CD4": "#80cdc1",
    "CD4 activated/effector": "#fdb863",
    "CD4 cytotoxic": "#c2a5cf",
    "CD4 exhaustion-associated": "#d6604d",
    "CD4 naive-like": "#4393c3",
    "CD4 resting/memory": "#92c5de",
    "CD8": "#5e3c99",
    "CD8 activated/effector": "#f46d43",
    "CD8 cytotoxic effector": "#7b3294",
    "CD8 exhausted (TEX)": "#b2182b",
    "CD8 naive-like": "#2166ac",
    "CD8 resting/memory": "#67a9cf",
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
                NAIVE_STEMNESS_MARKERS,
                NAIVE_HOMING_MARKERS,
                RESTING_MEMORY_MARKERS,
                INHIBITORY_RECEPTOR_MARKERS,
                CHECKPOINT_RECEPTOR_MARKERS,
                TERMINAL_INHIBITORY_RECEPTOR_MARKERS,
                EXHAUSTION_TF_MARKERS,
                LATE_EXHAUSTION_SIGNATURE_MARKERS,
                ACUTE_ACTIVATION_MARKERS,
                EFFECTOR_FUNCTION_MARKERS,
                GRANZYME_MARKERS,
                PERFORIN_MARKERS,
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

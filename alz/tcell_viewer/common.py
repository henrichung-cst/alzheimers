"""Shared constants and cross-cutting helpers for the T-cell viewer.

No intra-package imports (only stdlib / external / alz.shared.config /
alz.viewer.shared.*). Imported by both the builder and all four slice modules.

Vocabulary note — T-cell specificity vs enrichment
---------------------------------------------------
The T-cell cohort uses two metrics that **diverge** from the AD cohorts:

- **Enrichment** (within-cohort): concentration of expression in a ProjecTILs
  T-state, computed as fold over the kinase's median state (the same
  computational method the AD cohorts call "specificity" / WMB-tier share, but
  named enrichment here to reflect the T-state activation-continuum reading).

- **Specificity** (NSCLC reference): N-of-7 coarse-lineage prevalence count —
  how many of the 7 TME cell-type lineages express the kinase at ≥10% of cells.
  This is a DIFFERENT kind of metric than human-AD "specificity"
  (log2(celltype_mean/brain_mean) ratio with no detection gate).

Do NOT harmonize these two names with the AD cohorts — the collision is
intentional to fit each cohort's data structure, and is documented here to
prevent future agents from re-introducing it.
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# T-cell cohort constants
# ---------------------------------------------------------------------------

DONORS = ("donor1", "donor2")

# Per-donor MEA presence — donor2 had no stoichiometry matrix (no IMAC) and no
# pY motif → all four MEA variants skipped per the donor2 mea_manifest.json.
DONOR_WITH_MEA = ("donor1",)

# Honest caveat surfaced wherever concordance is shown. Interrogation 2026-06-03
# (docs/tcell_exhaustion_analysis_summary.md): a kinase's catalytic
# activity is post-translationally decoupled from its own mRNA, so concordance
# (sign of bulk NES × transcript Δ) carries no directional information — sign-
# agreement runs at chance (OR≈1; identical in the published mouse Song method).
# It is therefore SHOWN as a label and never used to filter/rank.
TCELL_ATTRIBUTION_CAVEAT = (
    "Concordance is directional co-evidence only and is never used to filter: a "
    "kinase's activity (substrate phosphorylation) is post-translationally "
    "decoupled from its own transcript, so sign-agreement runs at chance "
    "(OR≈1, same in the mouse Song reference). Read it alongside specificity.")

# Sequential viridis progression replaces the mouse 3-disease palette.
# Sampled from the matplotlib viridis colormap at evenly spaced points.
TIMEPOINT_COLOR_MAP = {
    "d2":  "#440154",
    "d5":  "#482878",
    "d7":  "#3e4a89",
    "d9":  "#31688e",
    "d11": "#26828e",
    "d13": "#1f9e89",
    "d15": "#35b779",
    "d17": "#6dcd59",
    "d19": "#b4de2c",
    "d20": "#fde725",
}

# Metric vocabulary constants — use these in payload keys and UI labels so
# changes are made in one place.
ENRICHMENT = "enrichment"   # within-cohort ProjecTILs T-state concentration
SPECIFICITY = "specificity"  # NSCLC N-of-7 coarse-lineage breadth

PROJECTILS_LABEL_MAP = {
    "CD8.CM": "CD8CM",
    "CD8.EM": "CD8EM",
    "CD8.MAIT": "CD8MAIT",
    "CD8.NaiveLike": "CD8Naive",
    "CD8.TEMRA": "CD8TEMRA",
    "CD8.TEX": "CD8Tex",
    "CD8.TPEX": "CD8Tpex",
    "CD4.CTL_EOMES": "CD4CTLeomes",
    "CD4.CTL_Exh": "CD4CTLexh",
    "CD4.CTL_GNLY": "CD4CTLgnly",
    "CD4.NaiveLike": "CD4Naive",
    "CD4.Tfh": "CD4Tfh",
    "CD4.Th17": "CD4Th17",
    "CD4.Treg": "Treg",
}


def _incytr_sanitize(name: str) -> str:
    """Match the upstream sanitize in alz/integration/load.R:sanitize_celltype."""
    return name.replace("/", "-").replace(" ", "_").replace(".", "")


_TCELL_CONTRAST_RE = re.compile(r"^D\d+_(d\d+)_vs_d2$")


def _short_contrast(s: object) -> str:
    """Rewrite `D1_d13_vs_d2` -> `d13` (the viewer's CONTRASTS token).

    Pass-through for already-short tokens. Strict regex so a typo in the
    pipeline output surfaces as an unchanged value, not a silent shim.
    """
    if s is None:
        return ""
    m = _TCELL_CONTRAST_RE.match(str(s))
    return m.group(1) if m else str(s)

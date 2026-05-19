"""Shared path/config constants for the deconvolution proportion-proxy pipeline."""
from __future__ import annotations

import os

import config  # noqa: E402 — repo-level config, available because callers run from repo root

REPO_ROOT = config.REPO_ROOT

# ---------------------------------------------------------------------------
# Spine path resolution — data/incytr_frozen/v2_46clusters/spines/<name>/
# ---------------------------------------------------------------------------
SPINES_ROOT = os.path.join(REPO_ROOT, "data", "incytr_frozen", "v2_46clusters", "spines")


def spine_dir(name: str) -> str:
    """Return the spines/<name>/ directory (does not check existence)."""
    return os.path.join(SPINES_ROOT, name)


def resolve_spine_csv(name: str) -> str:
    """Resolve the cluster_spine.csv path for `name`."""
    return os.path.join(spine_dir(name), "cluster_spine.csv")


def load_spine_clusters(name: str) -> list[str]:
    """Return ordered list of in-spine cluster names for spine `name`."""
    import pandas as pd  # local import to keep this module cheap to import
    df = pd.read_csv(resolve_spine_csv(name))
    return df.loc[df["in_spine"] == True, "cluster_name"].tolist()  # noqa: E712

DECON_INPUT_DIR = os.path.join(
    REPO_ROOT, "outputs", "reports", "deconvolution", "wmb_decomposition",
)
PS_DECONVOLUTED_FILE = os.path.join(DECON_INPUT_DIR, "ps_wmb_decomposition.csv")
PY_DECONVOLUTED_FILE = os.path.join(DECON_INPUT_DIR, "py_wmb_decomposition.csv")
PR_DECONVOLUTED_FILE = os.path.join(DECON_INPUT_DIR, "pr_wmb_decomposition.csv")
WMB_CLASS_SIZE_FILE = os.path.join(DECON_INPUT_DIR, "wmb_class_size.csv")

OUTPUT_DIR = os.path.join(REPO_ROOT, "outputs", "reports", "deconvolution")
SITE_OLS_FILE = os.path.join(OUTPUT_DIR, "site_level_ols.parquet")
MEA_FILE = os.path.join(OUTPUT_DIR, "kinase_enrichment_raw.csv")
PRIMARY_TABLE = os.path.join(OUTPUT_DIR, "kinase_enrichment_wmb.csv")
SUMMARY_JSON = os.path.join(OUTPUT_DIR, "summary.json")

# Power floor: any sample group with fewer cells than this in a given
# (wmb_class, contrast) pair is reported as the row's `n_cells_min`; readers
# typically drop rows below this threshold.
MIN_CELLS_PER_GROUP = 20

# Direction-match thresholds for the per-row "match"/"opposite"/"flat"
# annotation produced by snrna_concordance.py — describes whether snRNA
# evidence agrees in sign with the bulk NES at this row. Not a gate.
SNRNA_FDR_HIGH = 0.10
SNRNA_LFC_FLAT = 0.05

# Cohort-concordance gate (one binomial test per (wmb_class, contrast)
# stratum, BH across strata). 0.25 was chosen because the calibration
# audit found 0/43 strata passed at FDR<0.10. See
# outputs/reports/deconvolution/per_animal/cohort_concordance_calibration.md
COHORT_FDR_THRESH = 0.25

# Hard expression-presence floor (log2(CPM+1) in snRNA pseudobulk).
# Chosen as ~1.7% of bulk-sig rows fall below 0.5; the 0.10 5th-percentile
# auto-pick was effectively no filter.
EXPR_PRESENCE_FLOOR = 0.5

# Deconvolution-side significance threshold (matches live MEA gate).
DECON_FDR_THRESH = 0.25

# Genotype × time factorial coding — same as live config.
GENOTYPE_CODING = {
    "WTyp": {"App": 0, "Tau": 0, "Int": 0},
    "AppP": {"App": 1, "Tau": 0, "Int": 0},
    "Ttau": {"App": 0, "Tau": 1, "Int": 0},
    "ApTt": {"App": 1, "Tau": 1, "Int": 1},
}

CONTRASTS = {
    "App_2mo":  {"App": 1},
    "App_4mo":  {"App": 1, "App_x_time4": 1},
    "App_6mo":  {"App": 1, "App_x_time6": 1},
    "Tau_2mo":  {"Tau": 1},
    "Tau_4mo":  {"Tau": 1, "Tau_x_time4": 1},
    "Tau_6mo":  {"Tau": 1, "Tau_x_time6": 1},
    "ApTt_2mo": {"App": 1, "Tau": 1, "Int": 1},
    "ApTt_4mo": {"App": 1, "Tau": 1, "Int": 1, "App_x_time4": 1, "Tau_x_time4": 1},
    "ApTt_6mo": {"App": 1, "Tau": 1, "Int": 1, "App_x_time6": 1, "Tau_x_time6": 1},
}


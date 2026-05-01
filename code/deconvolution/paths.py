"""Shared path/config constants for the deconvolution proportion-proxy pipeline."""
from __future__ import annotations

import os

import config  # noqa: E402 — repo-level config, available because callers run from repo root

REPO_ROOT = config.REPO_ROOT

DECON_INPUT_DIR = os.path.join(
    REPO_ROOT, "data", "raw", "external", "gdrive_shared",
    "integrations", "yuyu01", "documentation", "incytr",
    "deconvolution", "deconvolution_with_new_clusters_20250721",
)
PS_DECONVOLUTED_FILE = os.path.join(DECON_INPUT_DIR, "ps_yuyu_deconvoluted.csv")
PY_DECONVOLUTED_FILE = os.path.join(DECON_INPUT_DIR, "py_yuyu_deconvoluted.csv")
CLUSTER_SIZE_FILE = os.path.join(DECON_INPUT_DIR, "yuyu_clustersize.csv")
SAMPLE_KEY_FILE = os.path.join(
    REPO_ROOT, "data", "raw", "external", "gdrive_shared",
    "integrations", "yuyu01", "documentation", "incytr",
    "deconvolution", "yuyu_samplekey.csv",
)

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
CLUSTER_MAPPING_FILE = os.path.join(CODE_DIR, "yuyu_46_to_wmb_class.csv")

OUTPUT_DIR = os.path.join(REPO_ROOT, "outputs", "reports", "deconvolution")
SITE_OLS_FILE = os.path.join(OUTPUT_DIR, "site_level_ols.parquet")
MEA_FILE = os.path.join(OUTPUT_DIR, "kinase_enrichment_raw.csv")
PRIMARY_TABLE = os.path.join(OUTPUT_DIR, "kinase_enrichment_46clusters.csv")
ROLLUP_TABLE = os.path.join(OUTPUT_DIR, "kinase_enrichment_wmb_rollup.csv")
SUMMARY_JSON = os.path.join(OUTPUT_DIR, "summary.json")

# Power floor: any sample group with fewer cells than this in a given
# cluster marks that (cluster, contrast) pair as "Insufficient" confidence.
MIN_CELLS_PER_GROUP = 20

# snRNA cross-check thresholds (kinase gene LFC concordance).
SNRNA_FDR_HIGH = 0.10     # FDR ceiling for direction-match → High confidence
SNRNA_LFC_FLAT = 0.05     # |LFC| below this is treated as "flat"

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

# Map contrast name → snRNA `pathway` value in song_concordance.csv.
CONTRAST_TO_PATHWAY = {
    "App_2mo": "App", "App_4mo": "App", "App_6mo": "App",
    "Tau_2mo": "Tau", "Tau_4mo": "Tau", "Tau_6mo": "Tau",
    "ApTt_2mo": "ApTt", "ApTt_4mo": "ApTt", "ApTt_6mo": "ApTt",
}

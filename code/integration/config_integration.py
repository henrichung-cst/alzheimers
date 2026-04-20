"""Integration-specific configuration for Incytr Phase 1 proof of concept."""

import os
import sys

# ---------------------------------------------------------------------------
# Repo root and path setup
# ---------------------------------------------------------------------------
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO_ROOT, "code"))

import config as main_config  # noqa: E402

# ---------------------------------------------------------------------------
# Phase 1 parameters
# ---------------------------------------------------------------------------
CONTRAST = "App_4mo"
CONDITION_WT = "WTyp"          # mutant column value in h5ad
CONDITION_DISEASE = "AppP"     # mutant column value in h5ad
TIMEPOINT = "4mo"
SEX_FILTER = "ma"              # sex column value in h5ad
SENDER = "Microglia-PVM"
RECEIVER = "L5 IT"
INCYTR_CONDITIONS = ("WT", "App")  # labels used in Incytr metadata

# Map h5ad mutant values to Incytr condition labels
MUTANT_TO_CONDITION = {
    CONDITION_WT: INCYTR_CONDITIONS[0],
    CONDITION_DISEASE: INCYTR_CONDITIONS[1],
}

# ---------------------------------------------------------------------------
# Factorial mode parameters (all genotypes x timepoints, males only)
# ---------------------------------------------------------------------------
# h5ad mutant values for each genotype
FACTORIAL_GENOTYPES = ["WTyp", "AppP", "Ttau", "ApTt"]
FACTORIAL_TIMEPOINTS = ["2mo", "4mo", "6mo"]
FACTORIAL_SEX = "ma"

# Design matrix columns (matches kinase_attribution.py OLS)
DESIGN_COLUMNS = [
    "const", "App", "Tau", "Int",
    "time_4mo", "time_6mo",
    "App_x_time4", "App_x_time6",
    "Tau_x_time4", "Tau_x_time6",
]

# Encoding: h5ad mutant value -> genotype indicator columns
MUTANT_TO_DESIGN = {
    "WTyp": {"App": 0, "Tau": 0, "Int": 0},
    "AppP": {"App": 1, "Tau": 0, "Int": 0},
    "Ttau": {"App": 0, "Tau": 1, "Int": 0},
    "ApTt": {"App": 1, "Tau": 1, "Int": 1},
}

# Encoding: timepoint -> time indicator columns
TIMEPOINT_TO_DESIGN = {
    "2mo": {"time_4mo": 0, "time_6mo": 0},
    "4mo": {"time_4mo": 1, "time_6mo": 0},
    "6mo": {"time_4mo": 0, "time_6mo": 1},
}

# 9 contrast coefficient vectors (from CONTRAST_COEFS in kinase_attribution.py)
# Each vector is over DESIGN_COLUMNS in order.
FACTORIAL_CONTRASTS = {
    "App_2mo": [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    "App_4mo": [0, 1, 0, 0, 0, 0, 1, 0, 0, 0],
    "App_6mo": [0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
    "Tau_2mo": [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    "Tau_4mo": [0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
    "Tau_6mo": [0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
    "ApTt_2mo": [0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
    "ApTt_4mo": [0, 1, 1, 1, 0, 0, 1, 0, 1, 0],
    "ApTt_6mo": [0, 1, 1, 1, 0, 0, 0, 1, 0, 1],
}

# ---------------------------------------------------------------------------
# Tiered integration thresholds
# ---------------------------------------------------------------------------
PHOSPHO_FDR_GATE = 0.25        # MEA FDR for discordance and filter tier
DISCORDANCE_RANK_QUARTILE = 0.25

# Expression detection threshold (fraction of cells with nonzero UMI).
# NOT READ BY PYTHON — the R wrappers (run_incytr.R, run_incytr_all_pairs.R,
# run_incytr_factorial_all_pairs.R) read EXPR_DETECTION_THRESHOLD from the
# environment with a default of 0.10. This constant mirrors that default for
# documentation purposes only. To change the operative threshold, set the
# env var before invoking the runners.

EXPRESSION_DETECTION_THRESHOLD = 0.10

# Kinase-imputed pathway expansion
# When enabled, receiver genes that are substrates of MEA-significant kinases
# are added to the receiver gene list even if they fail the expression threshold.
# These pathways are labeled "kinase-imputed" vs "expression-confirmed".
ENABLE_KINASE_IMPUTATION = True
# Tighter than PHOSPHO_FDR_GATE (0.25) because imputation multiplies search
# scope: one false-positive kinase drags its full motif-predicted substrate
# set into every receiver. See docs/integrations/kinase_incytr_integration.md.
KINASE_IMPUTATION_FDR = 0.10

# Per-receiver attribution gate (R1): require (receiver, kinase) combined_score
# in unified_attribution.csv to exceed TAU before that kinase's substrates are
# imputed into that receiver. None -> use the median combined_score across
# attributed (receiver, kinase) pairs in the current contrast. Set to 0.0 to
# disable gating (legacy cell-type-agnostic behavior).
KINASE_IMPUTATION_ATTRIBUTION_TAU = None

# Per-receiver expression floor (R3): imputed substrates must have at least this
# fraction of cells in the receiver with nonzero UMI. Excludes genes with zero
# RNA in the receiver even when bulk protein evidence exists elsewhere. Applied
# in the R wrapper using det_rates. Set to 0.0 to disable.
EXPR_IMPUTATION_FLOOR = 0.05

# Legacy-compatibility knob: when True the adapter emits one flat
# kinase_imputed_genes.csv as before (no per-receiver gating, no best_fdr-based
# weighting) for baseline regression checks. Default False (refined behavior).
KINASE_IMPUTATION_LEGACY = False

# Substrate-based reranking parameters
LAMBDA_VALUES = [0.1, 0.25, 0.5, 1.0, 2.0]
N_PERMUTATIONS = 10_000
SENDER_ATTRIBUTION_DISCOUNT = 0.25

# Sensitivity analysis parameters
DETECTION_THRESHOLD_SENSITIVITY = 0.20   # alternative threshold for comparison run
N_BOOTSTRAP_ITERATIONS = 500             # L5 IT bootstrap resampling iterations

# ---------------------------------------------------------------------------
# Paths: pipeline outputs (read-only)
# ---------------------------------------------------------------------------
KINASE_ATTR_DIR = os.path.join(
    REPO_ROOT, "outputs", "reports", "kinase_attribution"
)
ATTR_RECOVERY_DIR = os.path.join(
    REPO_ROOT, "outputs", "reports", "attribution_recovery"
)
SNRNA_DIR = os.path.join(
    REPO_ROOT, "outputs", "reports", "snrna_integration"
)
DATA_INGEST_DIR = os.path.join(
    REPO_ROOT, "outputs", "reports", "data_ingest"
)

MEA_STOICHIOMETRY_CSV = os.path.join(KINASE_ATTR_DIR, "mea_stoichiometry.csv")
UNIFIED_ATTRIBUTION_CSV = os.path.join(KINASE_ATTR_DIR, "unified_attribution.csv")
SITE_LEVEL_OLS_CSV = os.path.join(KINASE_ATTR_DIR, "site_level_ols.csv")
STOICHIOMETRY_MATRIX_CSV = os.path.join(KINASE_ATTR_DIR, "stoichiometry_matrix.csv")
SAMPLE_MAPPING_CSV = os.path.join(DATA_INGEST_DIR, "sample_mapping.csv")

H5AD_PATH = os.path.join(
    main_config.SONG_TRANSCRIPTOMICS_DIR, "170_gex_celltypes_00.h5ad"
)
KINASE_TO_GENE_CSV = os.path.join(
    main_config.SONG_ANALYSIS_CACHE_DIR, "kinase_to_gene_mapping.csv"
)

# ---------------------------------------------------------------------------
# Paths: intermediates (written by adapters, read by R wrappers)
# ---------------------------------------------------------------------------
INTERMEDIATES_DIR = os.path.join(
    REPO_ROOT, "code", "integration", "intermediates"
)
ALL_PAIRS_DIR = os.path.join(INTERMEDIATES_DIR, "all_pairs")

# Factorial mode intermediates (separate from legacy single-contrast)
FACTORIAL_DIR = os.path.join(INTERMEDIATES_DIR, "factorial")
FACTORIAL_ALL_PAIRS_DIR = os.path.join(FACTORIAL_DIR, "all_pairs")

# Phase 3: Cross-pair aggregation
PDS_SIGNIFICANCE_THRESHOLD = 0.1   # |PDS| threshold for "disease-altered"
AGGREGATION_DIR = os.path.join(ALL_PAIRS_DIR, "aggregation")

# Backbone-level permutation tests (aggregate_cross_pair.py --permutations)
N_PERMUTATIONS_AGGREGATE = 10_000

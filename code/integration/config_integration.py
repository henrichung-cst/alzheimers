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
# Tiered integration thresholds
# ---------------------------------------------------------------------------
PHOSPHO_FDR_GATE = 0.25        # MEA FDR for discordance and filter tier
DISCORDANCE_RANK_QUARTILE = 0.25

# Expression detection threshold (fraction of cells with nonzero UMI)
# Used by run_incytr.R for gene filtering before pathway_inference().
# Currently 0.50; lower values include more genes but explode pathway count.
EXPRESSION_DETECTION_THRESHOLD = 0.50

# Kinase-imputed pathway expansion
# When enabled, receiver genes that are substrates of MEA-significant kinases
# are added to the receiver gene list even if they fail the expression threshold.
# These pathways are labeled "kinase-imputed" vs "expression-confirmed".
ENABLE_KINASE_IMPUTATION = True
KINASE_IMPUTATION_FDR = 0.25   # MEA FDR threshold for "significant" kinases

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

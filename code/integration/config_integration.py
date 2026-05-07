"""Integration-specific configuration for Incytr Phase 1 proof of concept."""

import os
import sys
import hashlib
import json

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

# Design matrix columns (matches kinase_enrich.py OLS)
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

# 9 contrast coefficient vectors (from CONTRAST_COEFS in kinase_enrich.py)
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

# Explicit (cond_ref, cond_alt) condition labels for each contrast.
# Used by the factorial R wrapper to look up per-condition SiK_score_<cond>
# and apply the directional KPDS combination in PDS_<c>. Replaces the
# package's contrast_to_conditions heuristic, which silently misroutes
# multi-coefficient interaction contrasts (7 of 9 here). Conditions are
# {genotype}_{timepoint} as keyed in animal_metadata.csv.
FACTORIAL_CONTRAST_CONDITIONS = {
    "App_2mo":  ("WTyp_2mo", "AppP_2mo"),
    "App_4mo":  ("WTyp_4mo", "AppP_4mo"),
    "App_6mo":  ("WTyp_6mo", "AppP_6mo"),
    "Tau_2mo":  ("WTyp_2mo", "Ttau_2mo"),
    "Tau_4mo":  ("WTyp_4mo", "Ttau_4mo"),
    "Tau_6mo":  ("WTyp_6mo", "Ttau_6mo"),
    "ApTt_2mo": ("WTyp_2mo", "ApTt_2mo"),
    "ApTt_4mo": ("WTyp_4mo", "ApTt_4mo"),
    "ApTt_6mo": ("WTyp_6mo", "ApTt_6mo"),
}
assert set(FACTORIAL_CONTRAST_CONDITIONS) == set(FACTORIAL_CONTRASTS), (
    "FACTORIAL_CONTRAST_CONDITIONS must cover every contrast in FACTORIAL_CONTRASTS"
)

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

# Kinase-imputed pathway expansion (kinase pack — gated by
# INCYTR_LAYER_KINASE_PACK env var; see incytr_runtime.sh / .R registries and
# docs/integrations/incytr_layer_inventory.md).
# When the pack is enabled, receiver genes that are substrates of
# MEA-significant kinases are added to the receiver gene list even if they
# fail the expression threshold. These pathways are labeled "kinase-imputed"
# vs "expression-confirmed".
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
MEA_STOICHIOMETRY_PY_CSV = os.path.join(KINASE_ATTR_DIR, "mea_stoichiometry_pY.csv")
SITE_LEVEL_OLS_PY_CSV = os.path.join(KINASE_ATTR_DIR, "site_level_ols_pY.csv")
STOICHIOMETRY_MATRIX_PY_CSV = os.path.join(KINASE_ATTR_DIR, "stoichiometry_matrix_pY.csv")
SAMPLE_MAPPING_CSV = os.path.join(DATA_INGEST_DIR, "sample_mapping.csv")

WMB_DECOMPOSITION_DIR = os.path.join(
    REPO_ROOT, "outputs", "reports", "deconvolution", "wmb_decomposition"
)
PR_WMB_DECOMPOSITION_CSV = os.path.join(WMB_DECOMPOSITION_DIR, "pr_wmb_decomposition.csv")
PS_WMB_DECOMPOSITION_CSV = os.path.join(WMB_DECOMPOSITION_DIR, "ps_wmb_decomposition.csv")
PY_WMB_DECOMPOSITION_CSV = os.path.join(WMB_DECOMPOSITION_DIR, "py_wmb_decomposition.csv")
SEAAD_TO_WMB_CLASS_CSV = main_config.SEAAD_TO_WMB_CLASS_FILE

H5AD_PATH = os.path.join(
    main_config.SONG_TRANSCRIPTOMICS_DIR, "170_gex_celltypes_00.h5ad"
)
KINASE_TO_GENE_CSV = os.path.join(
    main_config.SONG_ANALYSIS_CACHE_DIR, "kinase_to_gene_mapping.csv"
)

# ---------------------------------------------------------------------------
# Paths: intermediates (written by adapters, read by R wrappers)
# ---------------------------------------------------------------------------
INCYTR_BASE = os.path.join(REPO_ROOT, "outputs", "incytr")
UNIVERSE_BASE = os.path.join(INCYTR_BASE, "universes")
SCORING_BASE = os.path.join(INCYTR_BASE, "scoring")
CONFIG_BASE = os.path.join(INCYTR_BASE, "configs")
NORMALIZED_SCHEMA_VERSION = 2


def _stable_digest(payload, n=16):
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:n]


def resolve_universe_id(config=None):
    """Return a stable universe id for the active input universe.

    The id is intentionally based on input path identities and universe-defining
    filters, not output timestamps, so independent runs over the same universe
    share normalized dimension tables.
    """
    payload = {
        "schema_version": NORMALIZED_SCHEMA_VERSION,
        "h5ad": H5AD_PATH,
        "kinase_to_gene": KINASE_TO_GENE_CSV,
        "factorial_genotypes": FACTORIAL_GENOTYPES,
        "factorial_timepoints": FACTORIAL_TIMEPOINTS,
        "factorial_sex": FACTORIAL_SEX,
        "expression_detection_threshold": EXPRESSION_DETECTION_THRESHOLD,
        "kinase_imputation_fdr": KINASE_IMPUTATION_FDR,
        "kinase_imputation_attribution_tau": KINASE_IMPUTATION_ATTRIBUTION_TAU,
        "expr_imputation_floor": EXPR_IMPUTATION_FLOOR,
    }
    if config:
        payload.update(config)
    return "u_" + _stable_digest(payload)


def resolve_scoring_id(config=None):
    """Return a stable scoring id for knobs that affect pathway scores."""
    payload = {
        "schema_version": NORMALIZED_SCHEMA_VERSION,
        "universe_id": resolve_universe_id(),
        "phospho_fdr_gate": PHOSPHO_FDR_GATE,
        "discordance_rank_quartile": DISCORDANCE_RANK_QUARTILE,
        "sender_attribution_discount": SENDER_ATTRIBUTION_DISCOUNT,
        "lambda_values": LAMBDA_VALUES,
    }
    if config:
        payload.update(config)
    return "s_" + _stable_digest(payload)


def resolve_config_id(config=None):
    """Return a stable aggregation config id."""
    payload = {
        "schema_version": NORMALIZED_SCHEMA_VERSION,
        "universe_id": resolve_universe_id(),
        "scoring_id": resolve_scoring_id(),
        "pds_significance_threshold": PDS_SIGNIFICANCE_THRESHOLD,
        "n_permutations_aggregate": N_PERMUTATIONS_AGGREGATE,
    }
    if config:
        payload.update(config)
    return "c_" + _stable_digest(payload)


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

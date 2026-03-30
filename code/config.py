import os

# --- Site classification ---
# Percentile-based classification (primary method)
PERCENT_RANK = "logFC"
PERCENT_THRESH = 5  # top/bottom 5% of sites by LFC
# Foreground quality gate: the 95th percentile of |LFC| must exceed this value
# for a comparison to be eligible for enrichment. When the deconvoluted signal
# is too low (e.g. Excitatory-Pyramidal, mean intensity ~0.05), the top 5% of
# sites have |LFC| < 0.001 and the percentile method runs Fisher's test on
# noise, producing spurious kinase hits.  A threshold of 0.1 requires that
# at least the extreme 5% of sites show a ~7% fold change.
MIN_FOREGROUND_LFC = 0.1

# --- Enrichment significance (tiered) ---
# LFF: minimum |log2 frequency factor| for a kinase to be called significant.
LFF_THRESH = .01
# PVAL_SIG: adjusted p-value threshold for statistical significance.
# Used for: thick borders on bubble maps, "significant" tier in tables,
# narrative claims in reports.
PVAL_SIG = .1
# PVAL_DISPLAY: lenient p-value ceiling for the "display" tier.
# A kinase cell qualifies as "display" if it is in the top/bottom
# BUBBLE_PERCENTILE% of LFF AND adj_pval < PVAL_DISPLAY.
# Kinases that are neither "significant" nor "display" in any cell
# are excluded from visualizations.  Tables still contain all kinases.
PVAL_DISPLAY = .5
KL_METHOD = "percentile_rank"
KL_THRESH = 15
KIN_TYPE = "ser_thr"
MAX_COMPARISONS = None

# --- Bubble map display ---
# --- Multiple testing correction method ---
# "bh"          — original BH correction from kinase-library (backward compatible)
# "permutation" — permutation-based empirical p-values on FET
# "meff_bh"     — BH correction using m_eff (effective number of independent tests)
CORRECTION_METHOD = "permutation"
N_PERMUTATIONS = 1000
PERMUTATION_SEED = 42
N_WORKERS = 12

BUBBLE_PERCENTILE = 5  # show kinases in top/bottom N% of LFF per comparison

# --- Substrate evidence tiers ---
SUBSTRATE_TIER_BOUNDARIES = (0.5, 0.75)  # (moderate, strong) median |LFC| cutoffs

# --- File paths ---
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SONG_WORKSPACE_DIR = os.path.join(REPO_ROOT, "data", "incytr_collections", "song")
SONG_PROTEOMICS_DIR = os.path.join(SONG_WORKSPACE_DIR, "proteomics")
SONG_ANALYSIS_SUPPORT_DIR = os.path.join(SONG_WORKSPACE_DIR, "analysis_support")
SONG_ANALYSIS_CACHE_DIR = os.path.join(SONG_WORKSPACE_DIR, "analysis_cache")

INPUT_FILES = {
    "ser_thr": os.path.join(SONG_PROTEOMICS_DIR, "ps_yuyu_deconvoluted.csv"),
    "tyrosine": os.path.join(SONG_PROTEOMICS_DIR, "py_yuyu_deconvoluted.csv"),
}
MEDIAN_CLUSTER_SIZES_FILE = os.path.join(SONG_ANALYSIS_SUPPORT_DIR, "median_cluster_sizes.csv")
MAPPING_CACHE_FILE = os.path.join(SONG_ANALYSIS_CACHE_DIR, "kinase_to_gene_mapping.csv")
ALLEN_EXPRESSION_CACHE_FILE = os.path.join(SONG_ANALYSIS_CACHE_DIR, "allen_expression_cache.csv")
ORGANISM = "mouse"
KEPT_RANKS = None # Number of top kinases to keep in summary. None for all.

CONDITION_COLORS = {"Ttau": "#1f77b4", "AppP": "#ff7f0e", "ApTt": "#d62728"}
GENDER_MAP = {"ma": "M"}

# Bulk data settings
BULK_INPUT_FILES = {
    "ser_thr": os.path.join(SONG_PROTEOMICS_DIR, "source", "imac_median.csv"),
    "tyrosine": os.path.join(SONG_PROTEOMICS_DIR, "source", "py_median.csv"),
}
BULK_GENDER_MAP = {"M": "M"}
BULK_CONDITION_MAP = {"Ttau": "T22", "AppP": "APP", "ApTt": "T22/APP"}  # canonical → column suffix


def get_input_file(kin_type=None):
    if kin_type is None:
        kin_type = KIN_TYPE
    return INPUT_FILES[kin_type]


def get_bulk_input_file(kin_type=None):
    if kin_type is None:
        kin_type = KIN_TYPE
    return BULK_INPUT_FILES[kin_type]


INPUT_FILE = get_input_file()
BULK_INPUT_FILE = get_bulk_input_file()


# =============================================================================
# SAP Model Configuration (Hurdle-Tweedie condition-specific deconvolution)
# =============================================================================

# --- Data paths ---
A_OBS_FILE = os.path.join(
    SONG_WORKSPACE_DIR, "method_records", "aobs_desp_standardized",
    "inputs", "A_obs_fractions.tsv",
)
AGGEXP_FILE = os.path.join(
    SONG_WORKSPACE_DIR, "method_records", "legacy_deconvolution_20250721",
    "inputs", "aggexp.csv",
)
DESP_BASELINE_FILE = INPUT_FILES["ser_thr"]  # ps_yuyu_deconvoluted.csv
BULK_PHOSPHO_FILE = BULK_INPUT_FILES["ser_thr"]  # imac_median.csv
SAMPLEKEY_FILE = os.path.join(SONG_WORKSPACE_DIR, "source", "metadata", "yuyu_samplekey.csv")
CLUSTERSIZE_FILE = os.path.join(
    SONG_WORKSPACE_DIR, "method_records", "legacy_deconvolution_20250721",
    "inputs", "yuyu_clustersize.csv",
)
KLDATA_FILE = os.path.join(SONG_WORKSPACE_DIR, "kinase", "kldata.csv")

# --- 5+1 cell-type pooling (SAP §2.1) ---
# Maps each of the 10 A_obs cell types to a resolved SAP type.
AOBS_POOL_MAP = {
    "Excitatory neurons": "Excitatory_neurons",
    "Oligodendrocytes":   "Oligodendrocytes",
    "Interneurons":       "GABAergic_neurons",
    "Astrocytes":         "Astrocytes",
    "Microglia":          "Microglia",
    "Endothelial cells":  "Other",
    "High MT":            "Other",
    "Medium spiny neurons": "Other",
    "OPCs":               "Other",
    "Other":              "Other",
}

# Maps DESP column cell-type suffixes to resolved SAP types.
# DESP uses broad labels: Glut (glutamatergic/excitatory), Gaba (GABAergic/inhibitory).
DESP_POOL_MAP = {
    "Glut":               "Excitatory_neurons",
    "Oligodendrocytes":   "Oligodendrocytes",
    "Gaba":               "GABAergic_neurons",
    "Astrocytes":         "Astrocytes",
    "Microglia":          "Microglia",
    "Endothelial cells":  "Other",
    "Medium spiny neurons": "Other",
    "OPCs":               "Other",
}

# Maps aggexp fine-grained cluster name prefixes to resolved SAP types.
# Prefixes are matched against the row index after stripping the trailing
# sample number (e.g. "Astrocytes14" → prefix "Astrocytes").
AGGEXP_POOL_MAP = {
    # Excitatory neurons
    "Excitatory-Rorb":        "Excitatory_neurons",
    "Excitatory-Pyramidal":   "Excitatory_neurons",
    "Excitatory-Pyramidal-Satb2-Cux2": "Excitatory_neurons",
    "Foxp2-Excitatory-Neurons-layers-6-and-2-3": "Excitatory_neurons",
    "glutamatergic-excitatory-neurons": "Excitatory_neurons",
    "Glutamatergic-excitatory-neurons-Cortical-layer-2-4-pyramidal-neurons": "Excitatory_neurons",
    "Excitatory-neurons":     "Excitatory_neurons",
    "Excitatory principal neurons in the hippocampal dentate gyrus": "Excitatory_neurons",
    "Excitatory-neurons-Cajal-Retzius-cells-layer-I-Reelin": "Excitatory_neurons",
    # Interneurons (GABAergic / inhibitory)
    "Inhibitory-Neurons":     "GABAergic_neurons",
    "Erbb4-VIP-inhibitory-neurons": "GABAergic_neurons",
    "Erbb4-inhibitory-neurons": "GABAergic_neurons",
    "VIP-positive-interneuron": "GABAergic_neurons",
    "GABAergic inhibitory interneurons": "GABAergic_neurons",
    "GABAergic-inhibitory-interneurons-Dlx6os1-Erbb4": "GABAergic_neurons",
    "GABAergic-inhibitory-interneurons-VIP-positive": "GABAergic_neurons",
    "Ndnf-positive-neurogliaform-inhibitory-interneurons-GABAergic": "GABAergic_neurons",
    "Reln-neurons":           "GABAergic_neurons",
    "Basal-Ganglia-GABAergic-Neurons": "GABAergic_neurons",
    # Astrocytes
    "Astrocytes":             "Astrocytes",
    "Ptprz1-protoplasmic-astrocytes": "Astrocytes",
    # Oligodendrocytes
    "Oligodendrocytes":       "Oligodendrocytes",
    # Microglia
    "Microglia":              "Microglia",
    # Other (pooled)
    "OPC":                    "Other",
    "Striatal-medium-spiny-neuron": "Other",
    "Endothelial-cell":       "Other",
    "Pericyte":               "Other",
    "Vascular-Leptomeningeal-Cells": "Other",
    "Choroid-Plexus-Epithelial-Cells": "Other",
    "Ependymal-cell":         "Other",
    "Cholinergic-Neurons":    "Other",
}

# Canonical order for the 6 resolved cell types (5 estimated + 1 pooled).
SAP_CELLTYPES = [
    "Excitatory_neurons",
    "Oligodendrocytes",
    "GABAergic_neurons",
    "Astrocytes",
    "Microglia",
    "Other",
]
# Only the first 5 receive condition-effect estimates (Delta).
SAP_ESTIMATED_CELLTYPES = SAP_CELLTYPES[:5]

# --- Factorial design (SAP §3.4) ---
SAP_CONDITIONS = ["WTyp", "AppP", "Ttau", "ApTt"]
SAP_TIMEPOINTS = ["2mo", "4mo", "6mo"]
SAP_GENDERS = ["ma", "fe"]

# 2×2 Amyloid × Tau factorial indicator matrix (SAP §3.4)
# Keys: condition → (App indicator, Tau indicator, App×Tau indicator)
SAP_FACTORIAL = {
    "WTyp": (0, 0, 0),
    "AppP": (1, 0, 0),
    "Ttau": (0, 1, 0),
    "ApTt": (1, 1, 1),
}

# --- Pre-fit diagnostic thresholds (SAP §6.0) ---
CONDITION_NUMBER_MAX = 5e3
COMPOSITION_MIN_RANK = 5
COMPOSITION_MIN_SV = 0.01
MAX_EFFECTIVE_DOF = 20
MIN_RESIDUAL_DOF = 4

# --- Feature filtering (SAP §2.5) ---
MIN_SAMPLE_DETECTION = 6  # site must be nonzero in >= 6 of 24 samples

# --- RNA preprocessing (SAP §2.2) ---
MIN_GENE_DETECTION = 12  # gene must be nonzero in >= 12 of 24 samples

# Phosphatase gene families for §3.2.1 covariate construction.
# No site-specific phosphatase-substrate database is available, so these are
# used as a global opposing term in the kinase-phosphatase balance covariate.
PHOSPHATASE_GENE_PREFIXES = [
    "Ppp", "Ptpn", "Ptpr", "Dusp", "Ppm", "Ssh", "Ctdsp", "Ctds",
]
PHOSPHATASE_GENES_EXTRA = [
    "Pten", "Cdc25a", "Cdc25b", "Cdc25c", "Inpp5d", "Inpp5e",
    "Inpp4a", "Inpp4b", "Synj1", "Synj2", "Mtmr1", "Mtmr2",
]

# Maps clustersize.csv row names to resolved SAP types for sample fingerprinting.
# =============================================================================
# Phase 2: Hurdle-Tweedie Model Hyperparameters (§3–§4)
# =============================================================================

# --- Tweedie power parameter (§3.1) ---
TWEEDIE_P_GRID = [1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8]

# --- Group Lasso / LOCO-CV grids (§4) ---
GAMMA_GRID = [0, 0.5, 1.0, 1.5]          # CVS adaptive weight exponent (§4.3)
ETA_GRID = [2.0, 2.5, 3.0, 4.0, 5.0]     # interaction penalty multiplier (§4.4)
ETA_MIN = 2.0
LAMBDA_RHO_GRID = [0.01, 0.1, 1.0, 10.0]  # ridge penalty on rho_j (§3.2.1)
N_INTENSITY_STRATA = 4                     # quartile-based intensity strata (§4.2)
N_LOCO_FOLDS = 4                           # one per condition (§4.2)
N_BOOTSTRAP = 500                          # residual block bootstrap (§7.2)

# --- Two-stage LOCO-CV (§4.2 implementation) ---
# Stage 1: search (λ, λ_ρ) with fixed η, γ defaults
LOCO_STAGE1_ETA_DEFAULT = 2.0
LOCO_STAGE1_GAMMA_DEFAULT = 0.5
# Stage 2: search (η, γ) with best (λ, λ_ρ) from stage 1
# (uses existing ETA_GRID and GAMMA_GRID)
LAMBDA_GRID_FAST_N = 5                     # 5-point log grid for --fit-fast
LAMBDA_GRID_FULL_N = 8                     # 8-point log grid for --fit

# --- IRLS convergence (§3.5) ---
IRLS_MAX_ITER = 50
IRLS_TOL = 1e-6
OUTER_MAX_ITER = 40
OUTER_TOL = 1e-4

# --- VIF diagnostic (§6.0) ---
VIF_THRESHOLD = 10
VIF_PASS_FRACTION = 0.90

CLUSTERSIZE_POOL_MAP = {
    "Excitatory_neurons": [
        "Excitatory-Rorb", "Excitatory-Pyramidal", "Excitatory-Pyramidal-Satb2-Cux2",
        "Foxp2-Excitatory-Neurons-layers-6-and-2-3", "glutamatergic-excitatory-neurons",
        "Glutamatergic-excitatory-neurons-Cortical-layer-2-4-pyramidal-neurons",
        "Excitatory principal neurons in the hippocampal dentate gyrus", "Excitatory-neurons",
    ],
    "Oligodendrocytes": ["Oligodendrocytes"],
    "GABAergic_neurons": [
        "Inhibitory-Neurons", "Erbb4-VIP-inhibitory-neurons", "Erbb4-inhibitory-neurons",
        "VIP-positive-interneuron", "GABAergic inhibitory interneurons",
        "GABAergic-inhibitory-interneurons-Dlx6os1-Erbb4",
        "GABAergic-inhibitory-interneurons-VIP-positive",
        "Ndnf-positive-neurogliaform-inhibitory-interneurons-GABAergic",
        "Reln-neurons", "Basal-Ganglia-GABAergic-Neurons",
    ],
    "Astrocytes": ["Astrocytes", "Ptprz1-protoplasmic-astrocytes"],
    "Microglia": ["Microglia"],
}

# =============================================================================
# Phase 3: Validation Suite (§5, §6.1–§6.5)
# =============================================================================

# --- Output paths ---
SAP_VALIDATION_DIR = os.path.join(SONG_ANALYSIS_CACHE_DIR, "sap_validation")
SAP_MODEL_FILE = os.path.join(SONG_ANALYSIS_CACHE_DIR, "sap_model_fit.npz")

# --- §5: bMIND benchmark ---
BMIND_CONCORDANCE_R_THRESH = 0.3      # site-level Pearson r
BMIND_JACCARD_THRESH = 0.25           # kinase enrichment Jaccard
BMIND_TOP_K_KINASES = 20              # for Jaccard calculation

# --- §6.1: Synthetic phospho-validation ---
SYNTH_PEARSON_OVERALL = 0.60
SYNTH_PEARSON_PER_CELLTYPE = {
    "Excitatory_neurons": 0.70,
    "Oligodendrocytes":   0.70,
    "GABAergic_neurons":  0.65,
    "Astrocytes":         0.60,
    "Microglia":          0.50,
}
SYNTH_SLOPE_RANGE = (0.8, 1.2)
SYNTH_SCENARIOS = ["mdes", "sparse", "dense", "de_novo", "rna_discordant"]
SYNTH_SPARSE_FRAC = 0.05
SYNTH_SPARSE_NTYPES = 2
SYNTH_DENSE_FRAC = 0.25
SYNTH_RNA_RHO_GRID = [0.0, 0.2, 0.4, 0.6]

# --- §6.3: Perturbation audit ---
PERTURB_SIGMA_GRID = [0.03, 0.05, 0.07]
PERTURB_N_ITER = 200
PERTURB_COLLAPSE_THRESH = 0.10        # >10% collapse → flag

# --- §6.4: Permutation null ---
PERM_NULL_N = 500
PERM_NULL_SPARSITY_TOLERANCE = 0.10   # 10% relative inflation allowed

# --- §6.5: Residual orthogonality ---
RESIDUAL_ORTH_ALPHA = 0.05

import os

# =============================================================================
# Core paths
# =============================================================================

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SONG_WORKSPACE_DIR = os.path.join(REPO_ROOT, "data", "incytr_collections", "song")
SONG_PRIMARY_DIR = os.path.join(SONG_WORKSPACE_DIR, "primary")
SONG_PRIMARY_PROTEOMICS_DIR = os.path.join(SONG_PRIMARY_DIR, "proteomics")
SONG_TRANSCRIPTOMICS_DIR = os.path.join(SONG_WORKSPACE_DIR, "transcriptomics")
SONG_SNRNA_SAMPLE_MANIFEST = os.path.join(
    SONG_TRANSCRIPTOMICS_DIR, "snrna_sample_manifest.csv"
)
SONG_PROTEOMICS_DIR = os.path.join(SONG_WORKSPACE_DIR, "proteomics")
SONG_ANALYSIS_SUPPORT_DIR = os.path.join(SONG_WORKSPACE_DIR, "analysis_support")
SONG_ANALYSIS_CACHE_DIR = os.path.join(SONG_WORKSPACE_DIR, "analysis_cache")

EXTERNAL_DATA_DIR = os.path.join(REPO_ROOT, "data", "external")

# =============================================================================
# Live pipeline: enrichment parameters
# =============================================================================

# MEA enrichment (GSEA-based, replaces DiffPhos Fisher)
KL_METHOD = "percentile_rank"
KL_THRESH = 15
MEA_FDR_THRESH = 0.25           # standard GSEA FDR threshold
MEA_PERMUTATION_NUM = 1000      # GSEApy prerank permutations
MEA_SEED = 112123               # GSEApy default seed

# Unified attribution
SEA_AD_LFC_MIN = 0.1            # minimum |sea_ad_lfc| for moderate confidence
SPECIFICITY_HIGH = 0.4          # WMB specificity threshold for high confidence
SPECIFICITY_LOW = 0.2           # WMB specificity threshold for moderate confidence

# Shared permutation parameters (attribution recovery)
N_PERMS = 1000
PERM_SEED = 42

# Deprecated: DiffPhos parameters (retained for archived code only)
PERCENT_RANK = "logFC"
PERCENT_THRESH = 5
LFF_THRESH = .01
PVAL_SIG = .1

# =============================================================================
# Live pipeline: output directories
# =============================================================================

DATA_INGEST_OUTPUT_DIR = os.path.join("outputs", "reports", "data_ingest")
KINASE_ATTRIBUTION_OUTPUT_DIR = os.path.join("outputs", "reports", "kinase_attribution")
ATTRIBUTION_RECOVERY_OUTPUT_DIR = os.path.join("outputs", "reports", "attribution_recovery")

# =============================================================================
# Supporting: external atlas acquisition (atlas_reference.py)
# =============================================================================

ALLEN_ABC_CACHE_DIR = os.path.join(EXTERNAL_DATA_DIR, "allen_abc")
SEA_AD_DIR = os.path.join(EXTERNAL_DATA_DIR, "sea_ad")
ALLEN_AGING_DIR = os.path.join(EXTERNAL_DATA_DIR, "allen_aging")
ATLAS_REFERENCE_OUTPUT_DIR = os.path.join("outputs", "reports", "atlas_reference")

# Spot-check kinases for WMB expression validation
WMB_SPOT_CHECK_KINASES = ["Gsk3b", "Cdk5", "Camk2a", "Mapk1", "Lrrk2"]

# ABC Atlas dataset keys
WMB_DATASET_KEY = "WMB-10Xv3"
WMB_REPR_REGION = "WMB-10Xv3-HPF"
AGING_DATASET_KEY = "Zeng-Aging-Mouse-10Xv3"

# All 13 WMB-10Xv3 region keys (log2 variants only)
WMB_ALL_REGION_KEYS = [
    "WMB-10Xv3-CB/log2",
    "WMB-10Xv3-CTXsp/log2",
    "WMB-10Xv3-HPF/log2",
    "WMB-10Xv3-HY/log2",
    "WMB-10Xv3-Isocortex-1/log2",
    "WMB-10Xv3-Isocortex-2/log2",
    "WMB-10Xv3-MB/log2",
    "WMB-10Xv3-MY/log2",
    "WMB-10Xv3-OLF/log2",
    "WMB-10Xv3-P/log2",
    "WMB-10Xv3-PAL/log2",
    "WMB-10Xv3-STR/log2",
    "WMB-10Xv3-TH/log2",
]

# SEA-AD access
SEA_AD_S3_BUCKET = "sea-ad-single-cell-profiling"

# =============================================================================
# Supporting: WMB expression export for Track B
# =============================================================================

WMB_EXPRESSION_OUTPUT_DIR = os.path.join("outputs", "reports", "wmb_expression")
WMB_EXPRESSION_FILE = os.path.join(WMB_EXPRESSION_OUTPUT_DIR, "wmb_kinase_expression.csv")
WMB_REGIONAL_EXPRESSION_FILE = os.path.join(WMB_EXPRESSION_OUTPUT_DIR, "wmb_regional_kinase_expression.csv")
WMB_PROTEOME_EXPRESSION_FILE = os.path.join(WMB_EXPRESSION_OUTPUT_DIR, "wmb_proteome_expression.csv")
PROTEOME_GENE_LIST_FILE = os.path.join(DATA_INGEST_OUTPUT_DIR, "total_proteome_genes.txt")
TIER1_CONDITION_TO_CONTRAST = {"AppP": "App", "Ttau": "Tau", "ApTt": "Int"}

# =============================================================================
# Supporting: shared cell-type definitions and data paths
#
# The 5+1 cell-type pooling, factorial design, and SAP data paths are shared
# infrastructure used by the live pipeline,
# supporting scripts (atlas_reference, wmb_expression), and archived SAP code.
# =============================================================================

# --- 5+1 cell-type pooling ---
# Maps each of the 10 A_obs cell types to a resolved type.
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

# Maps DESP column cell-type suffixes to resolved types.
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

# Maps aggexp fine-grained cluster name prefixes to resolved types.
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

# --- Factorial design ---
SAP_CONDITIONS = ["WTyp", "AppP", "Ttau", "ApTt"]
SAP_TIMEPOINTS = ["2mo", "4mo", "6mo"]
SAP_GENDERS = ["ma", "fe"]

# 2×2 Amyloid × Tau factorial indicator matrix
# Keys: condition → (App indicator, Tau indicator, App×Tau indicator)
SAP_FACTORIAL = {
    "WTyp": (0, 0, 0),
    "AppP": (1, 0, 0),
    "Ttau": (0, 1, 0),
    "ApTt": (1, 1, 1),
}

# --- Data paths ---
A_OBS_FILE = os.path.join(
    SONG_WORKSPACE_DIR, "method_records", "aobs_desp_standardized",
    "inputs", "A_obs_fractions.tsv",
)
AGGEXP_FILE = os.path.join(
    SONG_WORKSPACE_DIR, "method_records", "legacy_deconvolution_20250721",
    "inputs", "aggexp.csv",
)
DESP_BASELINE_FILE = os.path.join(SONG_PROTEOMICS_DIR, "ps_yuyu_deconvoluted.csv")
BULK_PHOSPHO_FILE = os.path.join(SONG_PROTEOMICS_DIR, "source", "imac_median.csv")
SAMPLEKEY_FILE = os.path.join(SONG_WORKSPACE_DIR, "source", "metadata", "yuyu_samplekey.csv")
CLUSTERSIZE_FILE = os.path.join(
    SONG_WORKSPACE_DIR, "method_records", "legacy_deconvolution_20250721",
    "inputs", "yuyu_clustersize.csv",
)
KLDATA_FILE = os.path.join(SONG_WORKSPACE_DIR, "kinase", "kldata.csv")

# Maps clustersize.csv row names to resolved types for sample fingerprinting.
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

# --- Pre-fit diagnostic thresholds ---
CONDITION_NUMBER_MAX = 5e3
COMPOSITION_MIN_RANK = 5
COMPOSITION_MIN_SV = 0.01
MAX_EFFECTIVE_DOF = 20
MIN_RESIDUAL_DOF = 4

# --- Feature filtering ---
MIN_SAMPLE_DETECTION = 6  # site must be nonzero in >= 6 of 24 samples

# --- RNA preprocessing ---
MIN_GENE_DETECTION = 12  # gene must be nonzero in >= 12 of 24 samples

# Phosphatase gene families for kinase-phosphatase balance covariate.
PHOSPHATASE_GENE_PREFIXES = [
    "Ppp", "Ptpn", "Ptpr", "Dusp", "Ppm", "Ssh", "Ctdsp", "Ctds",
]
PHOSPHATASE_GENES_EXTRA = [
    "Pten", "Cdc25a", "Cdc25b", "Cdc25c", "Inpp5d", "Inpp5e",
    "Inpp4a", "Inpp4b", "Synj1", "Synj2", "Mtmr1", "Mtmr2",
]

# =============================================================================
# Archived: deconvoluted cluster analysis (kl_analysis_clusters.py)
#
# Settings below this line are used only by archived or legacy code.
# They are retained for reproducibility but are not part of the live pipeline.
# =============================================================================

# Foreground quality gate for deconvoluted enrichment
MIN_FOREGROUND_LFC = 0.1

# Display tier (lenient threshold for bubble map inclusion)
PVAL_DISPLAY = .5
BUBBLE_PERCENTILE = 5  # show kinases in top/bottom N% of LFF per comparison

# Multiple testing correction method
CORRECTION_METHOD = "permutation"
N_PERMUTATIONS = 1000
PERMUTATION_SEED = 42
N_WORKERS = 12

KIN_TYPE = "ser_thr"
MAX_COMPARISONS = None

INPUT_FILES = {
    "ser_thr": os.path.join(SONG_PROTEOMICS_DIR, "ps_yuyu_deconvoluted.csv"),
    "tyrosine": os.path.join(SONG_PROTEOMICS_DIR, "py_yuyu_deconvoluted.csv"),
}
MEDIAN_CLUSTER_SIZES_FILE = os.path.join(SONG_ANALYSIS_SUPPORT_DIR, "median_cluster_sizes.csv")
MAPPING_CACHE_FILE = os.path.join(SONG_ANALYSIS_CACHE_DIR, "kinase_to_gene_mapping.csv")
ALLEN_EXPRESSION_CACHE_FILE = os.path.join(SONG_ANALYSIS_CACHE_DIR, "allen_expression_cache.csv")
ORGANISM = "mouse"
KEPT_RANKS = None  # Number of top kinases to keep in summary. None for all.

CONDITION_COLORS = {"Ttau": "#1f77b4", "AppP": "#ff7f0e", "ApTt": "#d62728"}
GENDER_MAP = {"ma": "M"}

# Substrate evidence tiers (sap_tier_annotation.py)
SUBSTRATE_TIER_BOUNDARIES = (0.5, 0.75)

# =============================================================================
# Archived: bulk analysis (kl_analysis_bulk.py)
# =============================================================================

BULK_INPUT_FILES = {
    "ser_thr": os.path.join(SONG_PROTEOMICS_DIR, "source", "imac_median.csv"),
    "tyrosine": os.path.join(SONG_PROTEOMICS_DIR, "source", "py_median.csv"),
}
BULK_GENDER_MAP = {"M": "M"}
BULK_CONDITION_MAP = {"Ttau": "T22", "AppP": "APP", "ApTt": "T22/APP"}


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
# Archived: SAP model hyperparameters
# =============================================================================

# Tweedie power parameter
TWEEDIE_P_GRID = [1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8]

# Group Lasso / LOCO-CV grids
GAMMA_GRID = [0, 0.5, 1.0, 1.5, 2.0]
ETA_GRID = [2.0, 2.5, 3.0, 4.0, 5.0]
ETA_MIN = 2.0
LAMBDA_RHO_GRID = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
N_INTENSITY_STRATA = 4
N_LOCO_FOLDS = 3
N_BOOTSTRAP = 500

# Two-stage LOCO-CV
LOCO_STAGE1_ETA_DEFAULT = 2.0
LOCO_STAGE1_GAMMA_DEFAULT = 0.5
LAMBDA_GRID_FAST_N = 5
LAMBDA_GRID_FULL_N = 8

# IRLS convergence
IRLS_MAX_ITER = 50
IRLS_TOL = 1e-6
OUTER_MAX_ITER = 80
OUTER_TOL = 1e-4

# VIF diagnostic
VIF_THRESHOLD = 10
VIF_PASS_FRACTION = 0.90

# =============================================================================
# Archived: SAP validation suite
# =============================================================================

SAP_VALIDATION_DIR = os.path.join(SONG_ANALYSIS_CACHE_DIR, "sap_validation")
SAP_MODEL_FILE = os.path.join(SONG_ANALYSIS_CACHE_DIR, "sap_model_fit.npz")

# bMIND benchmark
BMIND_CONCORDANCE_R_THRESH = 0.3
BMIND_JACCARD_THRESH = 0.25
BMIND_TOP_K_KINASES = 20

# Synthetic phospho-validation
SYNTH_PEARSON_OVERALL = 0.60
SYNTH_PEARSON_PER_CELLTYPE = {
    "Excitatory_neurons": 0.70,
    "Oligodendrocytes":   0.70,
    "GABAergic_neurons":  0.65,
    "Astrocytes":         0.60,
    "Microglia":          0.50,
}
SYNTH_SLOPE_RANGE = (0.8, 1.2)
SYNTH_SCENARIOS = ["mdes", "sparse", "dense", "de_novo", "rna_discordant",
                    "kinase_program", "low_rank"]
SYNTH_SPARSE_FRAC = 0.05
SYNTH_SPARSE_NTYPES = 2
SYNTH_DENSE_FRAC = 0.25
SYNTH_RNA_RHO_GRID = [0.0, 0.2, 0.4, 0.6]
SYNTH_MIN_KINASE_SUBSTRATES = 5
SYNTH_FM_RIDGE_FRAC = 0.1

# Perturbation audit
PERTURB_SIGMA_GRID = [0.03, 0.05, 0.07]
PERTURB_N_ITER = 200
PERTURB_COLLAPSE_THRESH = 0.10

# Permutation null
PERM_NULL_N = 500
PERM_NULL_SPARSITY_TOLERANCE = 0.10

# Residual orthogonality
RESIDUAL_ORTH_ALPHA = 0.05

# =============================================================================
# Archived: correlation-based cell-type matching
# =============================================================================

MODULE5C_OUTPUT_DIR = os.path.join("outputs", "reports", "module5c")

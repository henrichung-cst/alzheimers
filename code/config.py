import os

import numpy as np

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
MEA_WINSORIZE_PERCENTILE = 1.0  # winsorize site LFCs at this percentile before MEA

# Sample filtering: outlier exclusion and sex-based subsetting
OUTLIER_ZSCORE_THRESH = 3.0     # within-group z-score threshold for outlier exclusion
ANALYSIS_MODE = os.environ.get("ANALYSIS_MODE", "males_only")  # "males_only" or "full_cohort"

# Unified attribution: 24 SEA-AD subclasses used for cell-type attribution
SEA_AD_SUBCLASSES = [
    # GABAergic
    "Chandelier", "Lamp5", "Lamp5 Lhx6", "Pax6", "Pvalb",
    "Sncg", "Sst", "Sst Chodl", "Vip",
    # Glutamatergic
    "L2/3 IT", "L4 IT", "L5 ET", "L5 IT", "L5/6 NP",
    "L6 CT", "L6 IT", "L6 IT Car3", "L6b",
    # Non-neuronal
    "Astrocyte", "Endothelial", "Microglia-PVM", "OPC",
    "Oligodendrocyte", "VLMC",
]
SEA_AD_LFC_MIN = 0.1            # minimum |sea_ad_lfc| for moderate confidence

# Map mouse-model pathway prefixes to the most biologically appropriate
# SEA-AD CPS stratification (amyloid cascade model: plaques precede tangles).
# Pearson r(early, late) ≈ −0.12 with ~48% sign flips across kinase genes,
# confirming early and late AD drive distinct transcriptomic programs.
SEA_AD_PATHWAY_MAP = {
    "App":  "early",   # amyloid-driven → early/low-CPS human donors
    "Tau":  "late",    # tau-driven → late/high-CPS human donors
    "ApTt": "full",    # combined pathology → full CPS range
}
N_CELL_TYPES = len(SEA_AD_SUBCLASSES)
# Thresholds are multiples of uniform (1/N): 2× for high, 1× for moderate
SPECIFICITY_HIGH = 2.0 / N_CELL_TYPES   # ~0.083: ≥2× more specific than uniform
SPECIFICITY_LOW = 1.0 / N_CELL_TYPES    # ~0.042: ≥1× uniform (above-average specificity)

# =============================================================================
# snRNA cluster → SEA-AD subclass mapping (single authoritative edit point)
# =============================================================================
#
# Maps each snRNA cluster label (row names in yuyu_clustersize.csv, as annotated
# in kr_cluster_id_key.csv) to its SEA-AD subclass assignment.
#
# This is the single authoritative mapping from the Song/InCyTr snRNA-seq
# cluster annotations to the 24 SEA-AD subclasses used throughout the pipeline.
# Any change to the cluster → subclass relationship should be made HERE.
#
# Fields per entry:
#   sea_ad_subclass  One of SEA_AD_SUBCLASSES, or a researcher category
#                    ("Generic_excitatory", "Generic_inhibitory",
#                     "Medium spiny neurons", "Other").
#   confidence       "unambiguous" — clear 1:1 correspondence (marker genes / naming)
#                    "close"       — strong but imperfect match; best available
#                    "ambiguous"   — class-level match only; cannot resolve subclass
#                    "unmapped"    — no cortical equivalent or unidentified cluster
#   note             Free-text rationale for the mapping decision.
#
# Unnamed clusters (cluster-27, cluster-64, …) are handled programmatically:
# any label starting with "cluster-" is assigned to "Other" / "unmapped".
SNRNA_CLUSTER_TAXONOMY = {
    # ── Unambiguous mappings ─────────────────────────────────────────────
    "Astrocytes":              {"sea_ad_subclass": "Astrocyte",       "confidence": "unambiguous", "note": "Direct match"},
    "Ptprz1-protoplasmic-astrocytes": {"sea_ad_subclass": "Astrocyte", "confidence": "unambiguous", "note": "Protoplasmic astrocyte subtype"},
    "Excitatory-Rorb":         {"sea_ad_subclass": "L4 IT",           "confidence": "unambiguous", "note": "Rorb is the canonical L4 marker"},
    "Excitatory-Pyramidal-Satb2-Cux2": {"sea_ad_subclass": "L2/3 IT", "confidence": "unambiguous", "note": "Satb2+Cux2 mark upper-layer IT"},
    "Erbb4-VIP-inhibitory-neurons": {"sea_ad_subclass": "Vip",        "confidence": "unambiguous", "note": "VIP+ inhibitory"},
    "VIP-positive-interneuron": {"sea_ad_subclass": "Vip",            "confidence": "unambiguous", "note": "VIP+ interneuron"},
    "GABAergic-inhibitory-interneurons-VIP-positive": {"sea_ad_subclass": "Vip", "confidence": "unambiguous", "note": "VIP+ GABAergic"},
    "Ndnf-positive-neurogliaform-inhibitory-interneurons-GABAergic": {"sea_ad_subclass": "Lamp5", "confidence": "unambiguous", "note": "Ndnf neurogliaform → Lamp5 subclass"},
    "Oligodendrocytes":        {"sea_ad_subclass": "Oligodendrocyte", "confidence": "unambiguous", "note": "Direct match"},
    "OPC":                     {"sea_ad_subclass": "OPC",             "confidence": "unambiguous", "note": "Direct match"},
    "Microglia":               {"sea_ad_subclass": "Microglia-PVM",   "confidence": "unambiguous", "note": "Direct match"},
    "Endothelial-cell":        {"sea_ad_subclass": "Endothelial",     "confidence": "unambiguous", "note": "Direct match"},
    "Vascular-Leptomeningeal-Cells": {"sea_ad_subclass": "VLMC",      "confidence": "unambiguous", "note": "Direct match"},
    # ── Close mappings ───────────────────────────────────────────────────
    "Excitatory-Pyramidal":    {"sea_ad_subclass": "L5 IT",           "confidence": "close", "note": "Deep pyramidal without upper-layer markers → L5 IT"},
    "Foxp2-Excitatory-Neurons-layers-6-and-2-3": {"sea_ad_subclass": "L6 CT", "confidence": "close", "note": "Foxp2 is a strong L6 marker"},
    "Glutamatergic-excitatory-neurons-Cortical-layer-2-4-pyramidal-neurons": {"sea_ad_subclass": "L2/3 IT", "confidence": "close", "note": "Layer 2-4 pyramidal, closest to L2/3 IT"},
    "Pericyte":                {"sea_ad_subclass": "VLMC",            "confidence": "close", "note": "Mural/perivascular, closest SEA-AD type is VLMC"},
    # ── Ambiguous: class-level only ──────────────────────────────────────
    "glutamatergic-excitatory-neurons": {"sea_ad_subclass": "Generic_excitatory", "confidence": "ambiguous", "note": "No layer/marker info to resolve subclass"},
    "Excitatory-neurons":      {"sea_ad_subclass": "Generic_excitatory", "confidence": "ambiguous", "note": "No layer/marker info to resolve subclass"},
    "Inhibitory-Neurons":      {"sea_ad_subclass": "Generic_inhibitory", "confidence": "ambiguous", "note": "No marker info to resolve subclass"},
    "GABAergic inhibitory interneurons": {"sea_ad_subclass": "Generic_inhibitory", "confidence": "ambiguous", "note": "No marker info to resolve subclass"},
    "Erbb4-inhibitory-neurons": {"sea_ad_subclass": "Generic_inhibitory", "confidence": "ambiguous", "note": "MGE-derived (Erbb4) but could be Pvalb or Sst"},
    "GABAergic-inhibitory-interneurons-Dlx6os1-Erbb4": {"sea_ad_subclass": "Generic_inhibitory", "confidence": "ambiguous", "note": "MGE-derived but cannot distinguish Pvalb/Sst"},
    "Reln-neurons":            {"sea_ad_subclass": "Generic_inhibitory", "confidence": "ambiguous", "note": "Reelin expressed in multiple types"},
    # ── No cortical equivalent / unmapped ────────────────────────────────
    "Striatal-medium-spiny-neuron": {"sea_ad_subclass": "Medium spiny neurons", "confidence": "unmapped", "note": "Subcortical; preserved as researcher category"},
    "Basal-Ganglia-GABAergic-Neurons": {"sea_ad_subclass": "Other",   "confidence": "unmapped", "note": "Subcortical"},
    "Excitatory principal neurons in the hippocampal dentate gyrus": {"sea_ad_subclass": "Other", "confidence": "unmapped", "note": "Hippocampal, no cortical equivalent"},
    "Excitatory-neurons-Cajal-Retzius-cells-layer-I-Reelin": {"sea_ad_subclass": "Other", "confidence": "unmapped", "note": "Cajal-Retzius, transient L1 cell type"},
    "Choroid-Plexus-Epithelial-Cells": {"sea_ad_subclass": "Other",   "confidence": "unmapped", "note": "Non-parenchymal"},
    "Ependymal-cell":          {"sea_ad_subclass": "Other",           "confidence": "unmapped", "note": "Non-parenchymal"},
    "Cholinergic-Neurons":     {"sea_ad_subclass": "Other",           "confidence": "unmapped", "note": "No SEA-AD subclass"},
}

# Reporting-time collapse: SEA-AD subclasses → researcher tissue categories.
# Recovers the original A_obs-style groupings for interpretability.
SUBCLASS_TO_TISSUE_CATEGORY = {
    # GABAergic → Interneurons
    "Chandelier": "Interneurons", "Lamp5": "Interneurons",
    "Lamp5 Lhx6": "Interneurons", "Pax6": "Interneurons",
    "Pvalb": "Interneurons", "Sncg": "Interneurons",
    "Sst": "Interneurons", "Sst Chodl": "Interneurons",
    "Vip": "Interneurons",
    # Glutamatergic → Excitatory neurons
    "L2/3 IT": "Excitatory neurons", "L4 IT": "Excitatory neurons",
    "L5 ET": "Excitatory neurons", "L5 IT": "Excitatory neurons",
    "L5/6 NP": "Excitatory neurons", "L6 CT": "Excitatory neurons",
    "L6 IT": "Excitatory neurons", "L6 IT Car3": "Excitatory neurons",
    "L6b": "Excitatory neurons",
    # Non-neuronal
    "Astrocyte": "Astrocytes", "Oligodendrocyte": "Oligodendrocytes",
    "Microglia-PVM": "Microglia", "OPC": "OPCs",
    "Endothelial": "Endothelial cells", "VLMC": "Endothelial cells",
    # Researcher categories (from ambiguous / unmapped clusters)
    "Generic_excitatory": "Excitatory neurons",
    "Generic_inhibitory": "Interneurons",
    "Medium spiny neurons": "Medium spiny neurons",
    "Other": "Other",
}

# Shared permutation parameters (attribution recovery)
N_PERMS = 1000
PERM_SEED = 42

# Deprecated: DiffPhos parameters (retained for archived code only)
PERCENT_RANK = "logFC"
PERCENT_THRESH = 5
LFF_THRESH = .01
PVAL_SIG = .1

# =============================================================================
# Live pipeline: missing-value imputation (shared across PCA steps)
# =============================================================================


def minprob_impute(mat, q=0.01, rng=None):
    """MinProb imputation for MNAR proteomics data.

    For each protein (row), draws missing values from N(q-th percentile,
    per-protein SD) of the observed log2 intensities. This models the
    assumption that missing values are predominantly low-abundance (MNAR).

    Parameters
    ----------
    mat : np.ndarray
        Log2-transformed matrix (proteins × samples). NaN marks missing.
    q : float
        Quantile of observed values to use as the imputation center (default 0.01).
    rng : np.random.Generator or None
        Random number generator for reproducibility. If None, uses default.

    Returns
    -------
    np.ndarray
        Matrix with NaN replaced by draws from N(quantile, sigma_obs).
    """
    if rng is None:
        rng = np.random.default_rng(42)
    out = mat.copy()
    for i in range(mat.shape[0]):
        row = mat[i]
        observed = row[np.isfinite(row)]
        missing = ~np.isfinite(row)
        if missing.sum() == 0 or len(observed) < 2:
            continue
        mu = np.quantile(observed, q)
        sigma = np.std(observed, ddof=1)
        out[i, missing] = rng.normal(mu, sigma, size=missing.sum())
    return out


# =============================================================================
# Live pipeline: output directories
# =============================================================================

DATA_INGEST_OUTPUT_DIR = os.path.join("outputs", "reports", "data_ingest")
KINASE_ATTRIBUTION_OUTPUT_DIR = os.path.join("outputs", "reports", "kinase_attribution")
ATTRIBUTION_RECOVERY_OUTPUT_DIR = os.path.join("outputs", "reports", "attribution_recovery")
SUPPLEMENTARY_OUTPUT_DIR = os.path.join("outputs", "reports", "supplementary")

# =============================================================================
# Supporting: external atlas acquisition (atlas_reference.py)
# =============================================================================

ALLEN_ABC_CACHE_DIR = os.path.join(EXTERNAL_DATA_DIR, "allen_abc")
SEA_AD_DIR = os.path.join(EXTERNAL_DATA_DIR, "sea_ad")
# Pathway-matched effect size files (keyed by SEA_AD_PATHWAY_MAP values)
SEA_AD_EFFECT_SIZES = {
    "full":  os.path.join(SEA_AD_DIR, "effect_sizes.h5ad"),
    "early": os.path.join(SEA_AD_DIR, "effect_sizes_early.h5ad"),
    "late":  os.path.join(SEA_AD_DIR, "effect_sizes_late.h5ad"),
}
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

# Pre-extracted gene-subset h5ad files (proteome + kinase genes only).
# Much smaller than full regional h5ad files, enabling fast expression queries
# without decompressing the full atlas.
WMB_SUBSET_DIR = os.path.join(
    ALLEN_ABC_CACHE_DIR, "expression_matrices", "WMB-10Xv3-subset"
)
WMB_SUBSET_FILENAME_FMT = "{region}-log2-subset.h5ad"

# SEA-AD access
SEA_AD_S3_BUCKET = "sea-ad-single-cell-profiling"

# =============================================================================
# Supporting: WMB expression export for unified attribution
# =============================================================================

WMB_EXPRESSION_OUTPUT_DIR = os.path.join("outputs", "reports", "wmb_expression")
WMB_EXPRESSION_FILE = os.path.join(WMB_EXPRESSION_OUTPUT_DIR, "wmb_kinase_expression.csv")
WMB_REGIONAL_EXPRESSION_FILE = os.path.join(WMB_EXPRESSION_OUTPUT_DIR, "wmb_regional_kinase_expression.csv")
WMB_PROTEOME_EXPRESSION_FILE = os.path.join(WMB_EXPRESSION_OUTPUT_DIR, "wmb_proteome_expression.csv")
PROTEOME_GENE_LIST_FILE = os.path.join(DATA_INGEST_OUTPUT_DIR, "total_proteome_genes.txt")
TIER1_CONDITION_TO_CONTRAST = {"AppP": "App", "Ttau": "Tau", "ApTt": "Int"}

# =============================================================================
# Supporting: Song snRNA-seq integration (paired within-cohort evidence)
# =============================================================================
#
# The 170_gex_celltypes_00.h5ad contains Allen Cell Type Mapper annotations
# (210 subclass_name labels, per-nucleus confidence scores) for 63,695 nuclei
# across 28 paired animals. This maps directly to 22/24 SEA-AD subclasses,
# bypassing the lossy 46-cluster taxonomy (which only covers 12/24).

SONG_H5AD_FILE = os.path.join(SONG_TRANSCRIPTOMICS_DIR, "170_gex_celltypes_00.h5ad")

# Output paths
SNRNA_INTEGRATION_OUTPUT_DIR = os.path.join("outputs", "reports", "snrna_integration")
SONG_PSEUDOBULK_FILE = os.path.join(SNRNA_INTEGRATION_OUTPUT_DIR, "pseudobulk_cpm.csv")
SONG_CELL_COUNTS_FILE = os.path.join(SNRNA_INTEGRATION_OUTPUT_DIR, "pseudobulk_cell_counts.csv")
SONG_EXPRESSION_FILE = os.path.join(SNRNA_INTEGRATION_OUTPUT_DIR, "song_expression_specificity.csv")
SONG_CONCORDANCE_FILE = os.path.join(SNRNA_INTEGRATION_OUTPUT_DIR, "song_concordance.csv")

# Thresholds
SONG_LFC_MIN = 0.1       # minimum |song_lfc| for concordance (same as SEA_AD_LFC_MIN)
# Weighted concordance: Song is within-cohort same-species (3×), SEA-AD is
# cross-species human proxy (1×). When both are available, effective
# concordance = (3 × song_cs + 1 × sea_ad_cs) / 4.
SONG_CONCORDANCE_WEIGHT = 3.0
SEA_AD_CONCORDANCE_WEIGHT = 1.0
SONG_MIN_CELLS = 10       # minimum cells per animal×subclass for pseudobulk
SONG_MIN_ANIMALS = 10     # minimum animals per subclass for concordance DE
SONG_MIN_SUBCLASS_PROB = 0.9  # minimum subclass_prob for nucleus inclusion

# 210 Allen Cell Type Mapper subclass_name → 22 SEA-AD subclass mapping.
# Derived from inspecting the 170_gex_celltypes_00.h5ad annotations.
# Only cortical and non-neuronal types are mapped; subcortical/olfactory
# types (STR, OB, HY, MB, etc.) are excluded as they lack SEA-AD equivalents.
SONG_SUBCLASS_MAP = {
    # GABAergic interneurons
    "Pvalb Gaba": "Pvalb",
    "Pvalb chandelier Gaba": "Chandelier",
    "Sst Gaba": "Sst",
    "Sst Chodl Gaba": "Sst Chodl",
    "Vip Gaba": "Vip",
    "Lamp5 Gaba": "Lamp5",
    "Lamp5 Lhx6 Gaba": "Lamp5 Lhx6",
    "Sncg Gaba": "Sncg",
    # Glutamatergic excitatory neurons
    "L2/3 IT CTX Glut": "L2/3 IT",
    "L4/5 IT CTX Glut": "L4 IT",
    "L5 IT CTX Glut": "L5 IT",
    "L5 ET CTX Glut": "L5 ET",
    "L5 NP CTX Glut": "L5/6 NP",
    "L6 CT CTX Glut": "L6 CT",
    "L6 IT CTX Glut": "L6 IT",
    "L6b CTX Glut": "L6b",
    # Non-neuronal
    "Oligo NN": "Oligodendrocyte",
    "OPC NN": "OPC",
    "Astro-TE NN": "Astrocyte",
    "Microglia NN": "Microglia-PVM",
    "Endo NN": "Endothelial",
    "VLMC NN": "VLMC",
    "Peri NN": "VLMC",  # pericytes grouped with VLMC (mural/perivascular)
}

# Pathway map for Song concordance: contrast prefix → factorial term
SONG_PATHWAY_MAP = {"App": "App", "Tau": "Tau", "ApTt": "ApTt"}

# =============================================================================
# Supporting: shared cell-type definitions and data paths
#
# The 5+1 cell-type pooling, factorial design, and SAP data paths are shared
# infrastructure used by the live pipeline,
# supporting scripts (atlas_reference, wmb_expression), and archived SAP code.
# =============================================================================

# --- 5+1 cell-type pooling ---
# Legacy: Maps each of the 10 A_obs cell types to a resolved type.
# Superseded by SNRNA_CLUSTER_TAXONOMY + SUBCLASS_TO_TISSUE_CATEGORY for the
# live pipeline.  Retained for archived code compatibility.
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

# Legacy: coarse 5+1 cell-type labels from the archived deconvolution pipeline.
# Used only by archived code under archive/code/. Live pipeline uses
# SEA_AD_SUBCLASSES (24 data-driven subclasses) instead.
SAP_CELLTYPES = [
    "Excitatory_neurons",
    "Oligodendrocytes",
    "GABAergic_neurons",
    "Astrocytes",
    "Microglia",
    "Other",
]
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
# Legacy: A_obs 10-category fractions.  Live pipeline now uses
# SNRNA_CLUSTER_TAXONOMY + CLUSTERSIZE_FILE for subclass-resolution fractions.
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

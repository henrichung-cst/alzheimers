import os

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SONG_WORKSPACE_DIR = os.path.join(REPO_ROOT, "data", "incytr_collections", "song")
SONG_PRIMARY_DIR = os.path.join(SONG_WORKSPACE_DIR, "primary")
SONG_PRIMARY_PROTEOMICS_DIR = os.path.join(SONG_PRIMARY_DIR, "proteomics")
SONG_TRANSCRIPTOMICS_DIR = os.path.join(SONG_WORKSPACE_DIR, "transcriptomics")
SONG_SNRNA_SAMPLE_MANIFEST = os.path.join(
    SONG_TRANSCRIPTOMICS_DIR, "snrna_sample_manifest.csv"
)
SONG_PROTEOMICS_DIR = os.path.join(SONG_WORKSPACE_DIR, "proteomics")
SONG_ANALYSIS_CACHE_DIR = os.path.join(SONG_WORKSPACE_DIR, "analysis_cache")

EXTERNAL_DATA_DIR = os.path.join(REPO_ROOT, "data", "external")

KL_METHOD = "percentile_rank"
KL_THRESH = 15
MEA_FDR_THRESH = 0.25           # standard GSEA FDR threshold
MEA_PERMUTATION_NUM = 1000      # GSEApy prerank permutations
MEA_SEED = 112123               # GSEApy default seed
MEA_WINSORIZE_PERCENTILE = 1.0  # winsorize site LFCs at this percentile before MEA

OUTLIER_ZSCORE_THRESH = 3.0     # within-group z-score threshold for outlier exclusion
ANALYSIS_MODE = os.environ.get("ANALYSIS_MODE", "males_only")  # "males_only" or "full_cohort"

SEA_AD_SUBCLASSES = [
    "Chandelier", "Lamp5", "Lamp5 Lhx6", "Pax6", "Pvalb",
    "Sncg", "Sst", "Sst Chodl", "Vip",
    "L2/3 IT", "L4 IT", "L5 ET", "L5 IT", "L5/6 NP",
    "L6 CT", "L6 IT", "L6 IT Car3", "L6b",
    "Astrocyte", "Endothelial", "Microglia-PVM", "OPC",
    "Oligodendrocyte", "VLMC",
]
SEA_AD_LFC_MIN = 0.1            # minimum |sea_ad_lfc| for moderate confidence

SEA_AD_PATHWAY_MAP = {
    "App":  "early",   # amyloid-driven → early/low-CPS human donors
    "Tau":  "late",    # tau-driven → late/high-CPS human donors
    "ApTt": "full",    # combined pathology → full CPS range
}
N_CELL_TYPES = len(SEA_AD_SUBCLASSES)
SPECIFICITY_HIGH = 2.0 / N_CELL_TYPES   # ~0.083: ≥2× more specific than uniform
SPECIFICITY_LOW = 1.0 / N_CELL_TYPES    # ~0.042: ≥1× uniform (above-average specificity)

SNRNA_CLUSTER_TAXONOMY = {
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
    "Excitatory-Pyramidal":    {"sea_ad_subclass": "L5 IT",           "confidence": "close", "note": "Deep pyramidal without upper-layer markers → L5 IT"},
    "Foxp2-Excitatory-Neurons-layers-6-and-2-3": {"sea_ad_subclass": "L6 CT", "confidence": "close", "note": "Foxp2 is a strong L6 marker"},
    "Glutamatergic-excitatory-neurons-Cortical-layer-2-4-pyramidal-neurons": {"sea_ad_subclass": "L2/3 IT", "confidence": "close", "note": "Layer 2-4 pyramidal, closest to L2/3 IT"},
    "Pericyte":                {"sea_ad_subclass": "VLMC",            "confidence": "close", "note": "Mural/perivascular, closest SEA-AD type is VLMC"},
    "glutamatergic-excitatory-neurons": {"sea_ad_subclass": "Generic_excitatory", "confidence": "ambiguous", "note": "No layer/marker info to resolve subclass"},
    "Excitatory-neurons":      {"sea_ad_subclass": "Generic_excitatory", "confidence": "ambiguous", "note": "No layer/marker info to resolve subclass"},
    "Inhibitory-Neurons":      {"sea_ad_subclass": "Generic_inhibitory", "confidence": "ambiguous", "note": "No marker info to resolve subclass"},
    "GABAergic inhibitory interneurons": {"sea_ad_subclass": "Generic_inhibitory", "confidence": "ambiguous", "note": "No marker info to resolve subclass"},
    "Erbb4-inhibitory-neurons": {"sea_ad_subclass": "Generic_inhibitory", "confidence": "ambiguous", "note": "MGE-derived (Erbb4) but could be Pvalb or Sst"},
    "GABAergic-inhibitory-interneurons-Dlx6os1-Erbb4": {"sea_ad_subclass": "Generic_inhibitory", "confidence": "ambiguous", "note": "MGE-derived but cannot distinguish Pvalb/Sst"},
    "Reln-neurons":            {"sea_ad_subclass": "Generic_inhibitory", "confidence": "ambiguous", "note": "Reelin expressed in multiple types"},
    "Striatal-medium-spiny-neuron": {"sea_ad_subclass": "Medium spiny neurons", "confidence": "unmapped", "note": "Subcortical; preserved as researcher category"},
    "Basal-Ganglia-GABAergic-Neurons": {"sea_ad_subclass": "Other",   "confidence": "unmapped", "note": "Subcortical"},
    "Excitatory principal neurons in the hippocampal dentate gyrus": {"sea_ad_subclass": "Other", "confidence": "unmapped", "note": "Hippocampal, no cortical equivalent"},
    "Excitatory-neurons-Cajal-Retzius-cells-layer-I-Reelin": {"sea_ad_subclass": "Other", "confidence": "unmapped", "note": "Cajal-Retzius, transient L1 cell type"},
    "Choroid-Plexus-Epithelial-Cells": {"sea_ad_subclass": "Other",   "confidence": "unmapped", "note": "Non-parenchymal"},
    "Ependymal-cell":          {"sea_ad_subclass": "Other",           "confidence": "unmapped", "note": "Non-parenchymal"},
    "Cholinergic-Neurons":     {"sea_ad_subclass": "Other",           "confidence": "unmapped", "note": "No SEA-AD subclass"},
}

SUBCLASS_TO_TISSUE_CATEGORY = {
    "Chandelier": "Interneurons", "Lamp5": "Interneurons",
    "Lamp5 Lhx6": "Interneurons", "Pax6": "Interneurons",
    "Pvalb": "Interneurons", "Sncg": "Interneurons",
    "Sst": "Interneurons", "Sst Chodl": "Interneurons",
    "Vip": "Interneurons",
    "L2/3 IT": "Excitatory neurons", "L4 IT": "Excitatory neurons",
    "L5 ET": "Excitatory neurons", "L5 IT": "Excitatory neurons",
    "L5/6 NP": "Excitatory neurons", "L6 CT": "Excitatory neurons",
    "L6 IT": "Excitatory neurons", "L6 IT Car3": "Excitatory neurons",
    "L6b": "Excitatory neurons",
    "Astrocyte": "Astrocytes", "Oligodendrocyte": "Oligodendrocytes",
    "Microglia-PVM": "Microglia", "OPC": "OPCs",
    "Endothelial": "Endothelial cells", "VLMC": "Endothelial cells",
    "Generic_excitatory": "Excitatory neurons",
    "Generic_inhibitory": "Interneurons",
    "Medium spiny neurons": "Medium spiny neurons",
    "Other": "Other",
}

DISEASE_GROUPS = ["App", "Tau", "ApTt"]
TIMEPOINTS = ["2mo", "4mo", "6mo"]
DISEASE_COLORS = {"App": "#c62828", "Tau": "#1565c0", "ApTt": "#6a1b9a"}
TISSUE_ORDER = [
    "Excitatory neurons", "Interneurons", "Astrocytes",
    "Oligodendrocytes", "OPCs", "Microglia", "Endothelial cells",
]

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

DATA_INGEST_OUTPUT_DIR = os.path.join("outputs", "reports", "data_ingest")
KINASE_ATTRIBUTION_OUTPUT_DIR = os.path.join("outputs", "reports", "kinase_attribution")
ATTRIBUTION_RECOVERY_OUTPUT_DIR = os.path.join("outputs", "reports", "attribution_recovery")
SUPPLEMENTARY_OUTPUT_DIR = os.path.join("outputs", "reports", "supplementary")

ALLEN_ABC_CACHE_DIR = os.path.join(EXTERNAL_DATA_DIR, "allen_abc")
SEA_AD_DIR = os.path.join(EXTERNAL_DATA_DIR, "sea_ad")
SEA_AD_EFFECT_SIZES = {
    "full":  os.path.join(SEA_AD_DIR, "effect_sizes.h5ad"),
    "early": os.path.join(SEA_AD_DIR, "effect_sizes_early.h5ad"),
    "late":  os.path.join(SEA_AD_DIR, "effect_sizes_late.h5ad"),
}
ALLEN_AGING_DIR = os.path.join(EXTERNAL_DATA_DIR, "allen_aging")
ATLAS_REFERENCE_OUTPUT_DIR = os.path.join("outputs", "reports", "atlas_reference")

WMB_SPOT_CHECK_KINASES = ["Gsk3b", "Cdk5", "Camk2a", "Mapk1", "Lrrk2"]

WMB_DATASET_KEY = "WMB-10Xv3"
AGING_DATASET_KEY = "Zeng-Aging-Mouse-10Xv3"

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

WMB_SUBSET_DIR = os.path.join(
    ALLEN_ABC_CACHE_DIR, "expression_matrices", "WMB-10Xv3-subset"
)
WMB_SUBSET_FILENAME_FMT = "{region}-log2-subset.h5ad"

WMB_METADATA_VERSION = "20241115"
WMB_METADATA_CSV = os.path.join(
    ALLEN_ABC_CACHE_DIR, "metadata", "WMB-10X", WMB_METADATA_VERSION, "views",
    "cell_metadata_with_cluster_annotation.csv",
)

SEA_AD_S3_BUCKET = "sea-ad-single-cell-profiling"

WMB_EXPRESSION_OUTPUT_DIR = os.path.join("outputs", "reports", "wmb_expression")
WMB_EXPRESSION_FILE = os.path.join(WMB_EXPRESSION_OUTPUT_DIR, "wmb_kinase_expression.csv")
WMB_REGIONAL_EXPRESSION_FILE = os.path.join(WMB_EXPRESSION_OUTPUT_DIR, "wmb_regional_kinase_expression.csv")
WMB_PROTEOME_EXPRESSION_FILE = os.path.join(WMB_EXPRESSION_OUTPUT_DIR, "wmb_proteome_expression.csv")
PROTEOME_GENE_LIST_FILE = os.path.join(DATA_INGEST_OUTPUT_DIR, "total_proteome_genes.txt")

SONG_H5AD_FILE = os.path.join(SONG_TRANSCRIPTOMICS_DIR, "170_gex_celltypes_00.h5ad")

SNRNA_INTEGRATION_OUTPUT_DIR = os.path.join("outputs", "reports", "snrna_integration")
SONG_PSEUDOBULK_FILE = os.path.join(SNRNA_INTEGRATION_OUTPUT_DIR, "pseudobulk_cpm.csv")
SONG_CELL_COUNTS_FILE = os.path.join(SNRNA_INTEGRATION_OUTPUT_DIR, "pseudobulk_cell_counts.csv")
SONG_EXPRESSION_FILE = os.path.join(SNRNA_INTEGRATION_OUTPUT_DIR, "song_expression_specificity.csv")
SONG_CONCORDANCE_FILE = os.path.join(SNRNA_INTEGRATION_OUTPUT_DIR, "song_concordance.csv")

SONG_LFC_MIN = 0.1       # minimum |song_lfc| for concordance (same as SEA_AD_LFC_MIN)
SONG_CONCORDANCE_WEIGHT = 3.0
SEA_AD_CONCORDANCE_WEIGHT = 1.0
SONG_MIN_CELLS = 10       # minimum cells per animal×subclass for pseudobulk
SONG_MIN_ANIMALS = 10     # minimum animals per subclass for concordance DE
SONG_MIN_SUBCLASS_PROB = 0.9  # minimum subclass_prob for nucleus inclusion

SONG_SUBCLASS_MAP = {
    "Pvalb Gaba": "Pvalb",
    "Pvalb chandelier Gaba": "Chandelier",
    "Sst Gaba": "Sst",
    "Sst Chodl Gaba": "Sst Chodl",
    "Vip Gaba": "Vip",
    "Lamp5 Gaba": "Lamp5",
    "Lamp5 Lhx6 Gaba": "Lamp5 Lhx6",
    "Sncg Gaba": "Sncg",
    "L2/3 IT CTX Glut": "L2/3 IT",
    "L4/5 IT CTX Glut": "L4 IT",
    "L5 IT CTX Glut": "L5 IT",
    "L5 ET CTX Glut": "L5 ET",
    "L5 NP CTX Glut": "L5/6 NP",
    "L6 CT CTX Glut": "L6 CT",
    "L6 IT CTX Glut": "L6 IT",
    "L6b CTX Glut": "L6b",
    "Oligo NN": "Oligodendrocyte",
    "OPC NN": "OPC",
    "Astro-TE NN": "Astrocyte",
    "Microglia NN": "Microglia-PVM",
    "Endo NN": "Endothelial",
    "VLMC NN": "VLMC",
    "Peri NN": "VLMC",  # pericytes grouped with VLMC (mural/perivascular)
}

SONG_PATHWAY_MAP = {"App": "App", "Tau": "Tau", "ApTt": "ApTt"}

SAP_FACTORIAL = {
    "WTyp": (0, 0, 0),
    "AppP": (1, 0, 0),
    "Ttau": (0, 1, 0),
    "ApTt": (1, 1, 1),
}

A_OBS_FILE = os.path.join(
    SONG_WORKSPACE_DIR, "method_records", "aobs_desp_standardized",
    "inputs", "A_obs_fractions.tsv",
)
CLUSTERSIZE_FILE = os.path.join(
    SONG_WORKSPACE_DIR, "method_records", "legacy_deconvolution_20250721",
    "inputs", "yuyu_clustersize.csv",
)
KLDATA_FILE = os.path.join(SONG_WORKSPACE_DIR, "kinase", "kldata.csv")

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

PHOSPHATASE_GENE_PREFIXES = [
    "Ppp", "Ptpn", "Ptpr", "Dusp", "Ppm", "Ssh", "Ctdsp", "Ctds",
]
PHOSPHATASE_GENES_EXTRA = [
    "Pten", "Cdc25a", "Cdc25b", "Cdc25c", "Inpp5d", "Inpp5e",
    "Inpp4a", "Inpp4b", "Synj1", "Synj2", "Mtmr1", "Mtmr2",
]

MAPPING_CACHE_FILE = os.path.join(SONG_ANALYSIS_CACHE_DIR, "kinase_to_gene_mapping.csv")

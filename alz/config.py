import os

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SONG_WORKSPACE_DIR = os.path.join(REPO_ROOT, "data", "datasets", "song")
SONG_PRIMARY_DIR = os.path.join(SONG_WORKSPACE_DIR, "primary")
SONG_PRIMARY_PROTEOMICS_DIR = os.path.join(SONG_PRIMARY_DIR, "proteomics")
SONG_PRIMARY_PHOSPHO_DIR = os.path.join(SONG_PRIMARY_DIR, "phospho")
SONG_PRIMARY_METADATA_DIR = os.path.join(SONG_PRIMARY_DIR, "metadata")

# Phospho track input files (S/T via IMAC; Y via dedicated pY enrichment).
SONG_IMAC_SITEQUANT_FILE = os.path.join(
    SONG_PRIMARY_PHOSPHO_DIR,
    "song_IMAC_sitequant_merged_labeled (2).xlsx",
)
SONG_IMAC_COMPOSITE_FILE = os.path.join(
    SONG_PRIMARY_PHOSPHO_DIR,
    "song_IMAC_compositeSites_merged_labeled (2).xlsx",
)
SONG_PY_SITEQUANT_FILE = os.path.join(
    SONG_PRIMARY_PHOSPHO_DIR,
    "song_pY_sitequant_merged_labeled (2).xlsx",
)
SONG_PY_COMPOSITE_FILE = os.path.join(
    SONG_PRIMARY_PHOSPHO_DIR,
    "song_pY_compositeSites_merged_labeled (2).xlsx",
)

# Phospho-track config: each track maps to an input file, column-rename rule,
# residue type for the motif sanity check, output suffix, and kinase-library
# kin_type for MEA. Track "st" is the legacy IMAC Ser/Thr track; "py" is the
# tyrosine-enriched track added in 2026-04.
PHOSPHO_TRACKS = {
    "st": {
        "name": "st",
        "label": "Ser/Thr",
        "input_file": SONG_IMAC_SITEQUANT_FILE,
        "composite_file": SONG_IMAC_COMPOSITE_FILE,
        "output_suffix": "",          # legacy filenames unchanged
        "kin_type": "ser_thr",
        "residue": "ST",              # central motif residue must be S or T
        "column_prefix": "p",         # p1_126c_sn_sum
        "site_id_source": "column",   # site_id column already present
        "kl_thresh": 15,              # kinase-library default; within-family
                                       # substrate Jaccard ~0.034 on this prerank
    },
    "py": {
        "name": "py",
        "label": "Tyr",
        "input_file": SONG_PY_SITEQUANT_FILE,
        "composite_file": SONG_PY_COMPOSITE_FILE,
        "output_suffix": "_pY",
        "kin_type": "tyrosine",
        "residue": "Y",
        "column_prefix": "plex",      # plex1_126_sn_sum  → renamed to p1_126_sn_sum on load
        "site_id_source": "synthesize",  # site_id := f"{protein_id}_{site_position}"
        "kl_thresh": 7,               # tightened from default 15 to bring
                                       # within-family Jaccard from 0.244 to
                                       # 0.122; controls Tyr-family co-firing
                                       # (see docs/foundation/multiple_testing.md)
    },
}
SONG_TRANSCRIPTOMICS_DIR = os.path.join(SONG_WORKSPACE_DIR, "transcriptomics")
SONG_SNRNA_SAMPLE_MANIFEST = os.path.join(
    SONG_TRANSCRIPTOMICS_DIR, "snrna_sample_manifest.csv"
)
SONG_ANALYSIS_CACHE_DIR = os.path.join(SONG_WORKSPACE_DIR, "analysis_cache")

EXTERNAL_DATA_DIR = os.path.join(REPO_ROOT, "data", "external")

KL_METHOD = "percentile_rank"
KL_THRESH = 15  # legacy ST default; tracks read kl_thresh from PHOSPHO_TRACKS
MEA_FDR_THRESH = 0.25           # standard GSEA FDR threshold
SITE_FDR_DIAGNOSTIC_THRESH = 0.05  # per-site OLS FDR cutoff for log-only diagnostic counts

TMT_REF_CHANNEL = "126"  # Ref_Pool TMT channel ID present in every plex
MEA_MIN_SITES = 100             # min ranked sites per contrast to attempt MEA
MEA_PERMUTATION_NUM = 1000      # GSEApy prerank permutations
MEA_SEED = 112123               # GSEApy default seed
MEA_WINSORIZE_PERCENTILE = 1.0  # winsorize site LFCs at this percentile before MEA

OUTLIER_ZSCORE_THRESH = 3.0     # within-group z-score threshold for outlier exclusion
ANALYSIS_MODE = os.environ.get("ANALYSIS_MODE", "males_only")  # "males_only" or "full_cohort"

WMB_CLASSES = [
    "01 IT-ET Glut", "02 NP-CT-L6b Glut", "03 OB-CR Glut", "04 DG-IMN Glut",
    "05 OB-IMN GABA", "06 CTX-CGE GABA", "07 CTX-MGE GABA", "08 CNU-MGE GABA",
    "09 CNU-LGE GABA", "10 LSX GABA", "11 CNU-HYa GABA", "12 HY GABA",
    "13 CNU-HYa Glut", "14 HY Glut", "15 HY Gnrh1 Glut", "16 HY MM Glut",
    "17 MH-LH Glut", "18 TH Glut", "19 MB Glut", "20 MB GABA",
    "21 MB Dopa", "22 MB-HB Sero", "23 P Glut", "24 MY Glut",
    "25 Pineal Glut", "26 P GABA", "27 MY GABA", "28 CB GABA",
    "29 CB Glut", "30 Astro-Epen", "31 OPC-Oligo", "32 OEC",
    "33 Vascular", "34 Immune",
]
WMB_CLASS_MANIFEST_FILE = os.path.join(
    EXTERNAL_DATA_DIR, "allen_abc", "wmb_class_manifest.csv"
)
SEAAD_TO_WMB_CLASS_FILE = os.path.join(
    EXTERNAL_DATA_DIR, "sea_ad", "seaad_subclass_to_wmb_class.csv"
)

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
N_CELL_TYPES = len(WMB_CLASSES)         # 34 — WMB-class spine
SPECIFICITY_HIGH = 2.0 / N_CELL_TYPES   # ~0.059: ≥2× more specific than uniform across 34 classes
SPECIFICITY_LOW = 1.0 / N_CELL_TYPES    # ~0.029: ≥1× uniform across 34 classes
COMBINED_SCORE_SPECIFICITY_BASE = 0.5   # baseline weight on concordance even at zero specificity

CLASS_TO_TISSUE_CATEGORY = {
    "01 IT-ET Glut": "Excitatory neurons",
    "02 NP-CT-L6b Glut": "Excitatory neurons",
    "03 OB-CR Glut": "Excitatory neurons",
    "04 DG-IMN Glut": "Excitatory neurons",
    "05 OB-IMN GABA": "Interneurons",
    "06 CTX-CGE GABA": "Interneurons",
    "07 CTX-MGE GABA": "Interneurons",
    "08 CNU-MGE GABA": "Interneurons",
    "09 CNU-LGE GABA": "Interneurons",
    "10 LSX GABA": "Subcortical neurons",
    "11 CNU-HYa GABA": "Subcortical neurons",
    "12 HY GABA": "Subcortical neurons",
    "13 CNU-HYa Glut": "Subcortical neurons",
    "14 HY Glut": "Subcortical neurons",
    "15 HY Gnrh1 Glut": "Subcortical neurons",
    "16 HY MM Glut": "Subcortical neurons",
    "17 MH-LH Glut": "Subcortical neurons",
    "18 TH Glut": "Subcortical neurons",
    "19 MB Glut": "Brainstem neurons",
    "20 MB GABA": "Brainstem neurons",
    "21 MB Dopa": "Brainstem neurons",
    "22 MB-HB Sero": "Brainstem neurons",
    "23 P Glut": "Brainstem neurons",
    "24 MY Glut": "Brainstem neurons",
    "25 Pineal Glut": "Brainstem neurons",
    "26 P GABA": "Brainstem neurons",
    "27 MY GABA": "Brainstem neurons",
    "28 CB GABA": "Cerebellum",
    "29 CB Glut": "Cerebellum",
    "30 Astro-Epen": "Astrocytes",
    "31 OPC-Oligo": "Oligodendrocytes",
    "32 OEC": "Oligodendrocytes",
    "33 Vascular": "Endothelial cells",
    "34 Immune": "Microglia",
    "Other": "Other",
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
    "Excitatory neurons", "Interneurons", "Subcortical neurons",
    "Brainstem neurons", "Cerebellum",
    "Astrocytes", "Oligodendrocytes", "OPCs", "Microglia",
    "Endothelial cells", "Other",
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

WMB_DATASET_KEY = "WMB-10Xv3"

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
WMB_EXPRESSION_SUBCLASS_FILE = os.path.join(WMB_EXPRESSION_OUTPUT_DIR, "wmb_kinase_expression_subclass.csv")
WMB_SUBCLASS_TO_CLASS_FILE = os.path.join(EXTERNAL_DATA_DIR, "allen_abc", "wmb_subclass_to_class.csv")
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

KLDATA_FILE = os.path.join(SONG_WORKSPACE_DIR, "kinase", "kldata.csv")

PHOSPHATASE_GENE_PREFIXES = [
    "Ppp", "Ptpn", "Ptpr", "Dusp", "Ppm", "Ssh", "Ctdsp", "Ctds",
]
PHOSPHATASE_GENES_EXTRA = [
    "Pten", "Cdc25a", "Cdc25b", "Cdc25c", "Inpp5d", "Inpp5e",
    "Inpp4a", "Inpp4b", "Synj1", "Synj2", "Mtmr1", "Mtmr2",
]

MAPPING_CACHE_FILE = os.path.join(SONG_ANALYSIS_CACHE_DIR, "kinase_to_gene_mapping.csv")

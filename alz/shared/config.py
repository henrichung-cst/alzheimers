import os
import re
from typing import TYPE_CHECKING

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SONG_WORKSPACE_DIR = os.path.join(REPO_ROOT, "data", "datasets", "song")
SONG_PRIMARY_DIR = os.path.join(SONG_WORKSPACE_DIR, "primary")
SONG_PRIMARY_PROTEOMICS_DIR = os.path.join(SONG_PRIMARY_DIR, "proteomics")
SONG_PRIMARY_PHOSPHO_DIR = os.path.join(SONG_PRIMARY_DIR, "phospho")
SONG_PRIMARY_METADATA_DIR = os.path.join(SONG_PRIMARY_DIR, "metadata")

# Phospho track input files (S/T via IMAC; Y via dedicated pY enrichment).
SONG_PROTEIN_QUANT_FILE = os.path.join(
    SONG_PRIMARY_PROTEOMICS_DIR,
    "song2024_tmttotal_protein_quant_merged_labeled (2).xlsx",
)
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
DERIVED_DATA_DIR = os.path.join(REPO_ROOT, "data", "derived")
DERIVED_BRIDGES_DIR = os.path.join(DERIVED_DATA_DIR, "bridges")
DERIVED_AGGREGATES_DIR = os.path.join(DERIVED_DATA_DIR, "aggregates")
DERIVED_CACHES_DIR = os.path.join(DERIVED_DATA_DIR, "caches")

KL_METHOD = "percentile_rank"
MEA_FDR_THRESH = 0.25           # standard GSEA FDR threshold
SITE_FDR_DIAGNOSTIC_THRESH = 0.05  # per-site OLS FDR cutoff for log-only diagnostic counts

TMT_REF_CHANNEL = "126"  # Ref_Pool TMT channel ID present in every plex
MEA_MIN_SITES = 100             # min ranked sites per contrast to attempt MEA
MEA_PERMUTATION_NUM = 1000      # GSEApy prerank permutations
MEA_SEED = 112123               # GSEApy default seed
MEA_THREADS = int(os.environ.get("ALZ_MEA_THREADS", "8"))  # GSEApy prerank threads
MEA_WINSORIZE_PERCENTILE = 1.0  # winsorize site LFCs at this percentile before MEA

OUTLIER_ZSCORE_THRESH = 3.0     # within-group z-score threshold for outlier exclusion


def _load_analysis_mode() -> str:
    # Cohort selection: source of truth is `conf/base/parameters.yml`.
    # KEDRO_ENV=full_cohort overlays `conf/full_cohort/parameters.yml`
    # (legacy env-var name retained until callers migrate to --cohort flags).
    from pathlib import Path

    import yaml

    base = Path(REPO_ROOT) / "conf" / "base" / "parameters.yml"
    with open(base) as f:
        params = yaml.safe_load(f) or {}
    env = os.environ.get("KEDRO_ENV")
    if env:
        overlay_path = Path(REPO_ROOT) / "conf" / env / "parameters.yml"
        if overlay_path.exists():
            with open(overlay_path) as f:
                params.update(yaml.safe_load(f) or {})
    return params["analysis_mode"]


if TYPE_CHECKING:
    ANALYSIS_MODE: str

# Allen WMB published taxonomy (34 classes). Used by atlas helpers
# (wmb_expression.py) that emit per-class expression for the WMB specificity
# branch of attribution. Analysis spine is CLUSTER_SPINE (Levy-t5) below;
# clusters crosswalk to WMB classes via CLUSTER_TO_WMB_CLASS_FILE.
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

# Levy 31-cluster spine (levy_t5): min_cells=5, no rank gate. Built once by
# alz/integration/build_cluster_spine.py; downstream stages read CLUSTER_SPINE.
CLUSTER_SPINE_NAME = "levy_t5"


def provenance_stamp(**extras) -> dict:
    """`{analysis_mode, spine, produced_at}` for summary/audit JSON sidecars.

    Stamped on every artifact the viewer cross-checks so a mismatched analysis
    mode or spine causes a hardfail at `build_unified_viewer` time instead of
    silent vocabulary drift.
    """
    import datetime as _dt

    stamp = {
        "analysis_mode": _lazy("ANALYSIS_MODE"),
        "spine": CLUSTER_SPINE_NAME,
        "produced_at": _dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }
    stamp.update(extras)
    return stamp
CLUSTER_SPINE_DIR = os.path.join(REPO_ROOT, "data", "incytr_frozen", "v2_46clusters")
CLUSTER_SPINE_FILE = os.path.join(
    CLUSTER_SPINE_DIR, "spines", CLUSTER_SPINE_NAME, "cluster_spine.csv"
)
BARCODE_TO_CLUSTER_FILE = os.path.join(CLUSTER_SPINE_DIR, "barcode_to_cluster.csv")
CELL_METADATA_FILE = os.path.join(CLUSTER_SPINE_DIR, "cell_metadata.csv")
KR_CLUSTER_ID_KEY_FILE = os.path.join(
    CLUSTER_SPINE_DIR, "provenance", "kr_cluster_id_key.csv"
)
CLUSTER_TO_WMB_CLASS_FILE = os.path.join(DERIVED_BRIDGES_DIR, "cluster_to_wmb_class.csv")
CLUSTER_TO_SEAAD_SUPERTYPE_FILE = os.path.join(DERIVED_BRIDGES_DIR, "cluster_to_seaad_supertype.csv")
CLUSTER_TO_HBCA_SUPERCLUSTER_FILE = os.path.join(DERIVED_BRIDGES_DIR, "cluster_to_hbca_supercluster.csv")


def _load_cluster_spine() -> list[str]:
    import csv
    out: list[str] = []
    with open(CLUSTER_SPINE_FILE, newline="") as f:
        for row in csv.DictReader(f):
            if row["in_spine"].strip().lower() == "true":
                out.append(row["cluster_name"])
    return sorted(out)


if TYPE_CHECKING:
    CLUSTER_SPINE: "list[str]"

# Coarse-level (cell-level) labels as carried by Seurat `obj@meta.data`.
# Does NOT nest deterministically under CLUSTER_SPINE (per-cell labels, not
# rollups). Exposed for forward flexibility — analyses default to the
# subclass spine (CLUSTER_SPINE); coarse can be opted-in by reading
# `cluster_coarse` from barcode_to_cluster.csv.
CLUSTER_COARSE_LEVELS = [
    "Astrocytes", "Endothelial cells", "Excitatory neurons", "High MT",
    "Interneurons", "Medium spiny neurons", "Microglia", "Oligodendrocytes",
    "OPCs", "Other",
]
# Subset of CLUSTER_COARSE_LEVELS with QC/junk classes ("High MT", "Other")
# removed — useful for biological-only views.
CLUSTER_COARSE_BIOLOGICAL = [c for c in CLUSTER_COARSE_LEVELS
                             if c not in ("High MT", "Other")]

SEA_AD_LFC_MIN = 0.1            # minimum |sea_ad_lfc| for moderate confidence

SEA_AD_PATHWAY_MAP = {
    "App":  "early",   # amyloid-driven → early/low-CPS human donors
    "Tau":  "late",    # tau-driven → late/high-CPS human donors
    "ApTt": "full",    # combined pathology → full CPS range
}
if TYPE_CHECKING:
    N_CELL_TYPES: int

_LAZY = {
    "ANALYSIS_MODE": _load_analysis_mode,
    "CLUSTER_SPINE": _load_cluster_spine,
    "N_CELL_TYPES": lambda: len(_lazy("CLUSTER_SPINE")),
}
_LAZY_CACHE: dict[str, object] = {}


def _lazy(name):
    if name not in _LAZY_CACHE:
        _LAZY_CACHE[name] = _LAZY[name]()
    return _LAZY_CACHE[name]


def __getattr__(name):                 # PEP 562
    if name in _LAZY:
        return _lazy(name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

# WMB even-split baseline (1/N) — see wmb_specificity_uniform(). N is the retained
# WMB classes the spine maps onto that actually carry atlas cells (N≈9), so the
# "above even-split" threshold is 1/N, NOT 1/N_CELL_TYPES. The viewer WMB badge
# tooltip uses it to render the concentration tier's threshold (tier × uniform).

def wmb_specificity_uniform():
    """Even-split baseline (1/N) for the WMB concentration tier.

    The detection-gated WMB metric (``wmb_expression.py``) reports
    ``wmb_concentration_tier`` as multiples of an even split over the retained WMB
    classes that actually carry atlas cells — empirically **9**, not the 11
    declared by the crosswalk: two retained classes ('07 CTX-MGE GABA',
    '13 CNU-HYa Glut') have no cells in the whole-brain scope, so they never enter
    any gene's denominator. The even-split threshold is therefore ``1/N`` with
    N = present retained classes — NOT ``1/N_CELL_TYPES`` (the 31-cluster count,
    which read ~3x more enriched than it is). The sole consumer is the viewer
    badge tooltip (``meta.wmb_uniform``), which shows ``tier × uniform`` as the
    concentration threshold each tier represents; falls back to the crosswalk's
    distinct-class count before the expression artifact is built.
    """
    retained = set(load_cluster_to_wmb_class_map().values())
    try:
        import pandas as pd
        present = set(
            pd.read_csv(WMB_EXPRESSION_FILE, usecols=["cell_type"])["cell_type"]
        ) & retained
        n = len(present) or len(retained)
    except (FileNotFoundError, OSError, ValueError, KeyError):
        n = len(retained)
    return 1.0 / n

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

# --- Specificity unit grouping (confidence-pill resolution) ---------------
# The confidence pill measures how exclusively a kinase is expressed in one cell
# type. The native 31 Song clusters over-split some cell types (e.g. excitatory
# neurons → 6 pyramidal subtypes), which dilutes a pan-class kinase's signal and
# makes it look non-specific. For those WMB classes we collapse the Song clusters
# into the WMB-class parent and compute specificity there. For WMB classes whose
# Song clusters are genuinely *distinct* cell types (e.g. vascular = endothelial
# + pericyte + choroid + leptomeningeal), we do NOT collapse — the native cluster
# is the correct unit. 1:1 classes keep their single cluster with a combined
# label. The collapse never happens silently: the viewer shows a collapsed unit
# as an expandable parent revealing its child clusters.
#
# Listed = collapse (member Song clusters are subtypes of ONE cell type).
# Unlisted multi-cluster classes stay split; 1:1 classes get a combined label.
SPECIFICITY_COLLAPSE_WMB_CLASSES = {
    "01 IT-ET Glut",      # cortical excitatory pyramidal subtypes
    "06 CTX-CGE GABA",    # CGE-derived interneuron subtypes
    "09 CNU-LGE GABA",    # striatal GABAergic (basal-ganglia GABA + MSN)
    "02 NP-CT-L6b Glut",  # deep/transient excitatory (Cajal-Retzius + Foxp2 L6)
}

# Friendly display labels for collapsed WMB-class parents.
WMB_CLASS_DISPLAY = {
    "01 IT-ET Glut":     "Excitatory neurons (IT/ET)",
    "06 CTX-CGE GABA":   "Interneurons (CGE)",
    "09 CNU-LGE GABA":   "Striatal GABAergic",
    "02 NP-CT-L6b Glut": "Deep excitatory (NP/CT/L6b)",
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


# ---------------------------------------------------------------------------
# Shared bulk-MEA helpers (centralized; formerly duplicated across
# normalize.py / enrich.py / attribute.py / recover.py).
# ---------------------------------------------------------------------------

def load_params() -> dict:
    """Load conf/base/parameters.yml with optional KEDRO_ENV overlay.

    Same path/overlay logic as `_load_analysis_mode`, but returns the whole
    parameter dict. Used by the bulk-MEA stage CLIs.
    """
    from pathlib import Path

    import yaml

    base = Path(REPO_ROOT) / "conf" / "base" / "parameters.yml"
    with open(base) as f:
        params = yaml.safe_load(f) or {}
    env = os.environ.get("KEDRO_ENV")
    if env:
        overlay_path = Path(REPO_ROOT) / "conf" / env / "parameters.yml"
        if overlay_path.exists():
            with open(overlay_path) as f:
                params.update(yaml.safe_load(f) or {})
    return params


def resolve_track(track):
    """Look up a phospho-track config by name; return the dict from PHOSPHO_TRACKS.

    Accepts either a track name (``"st"``/``"py"``) or an already-resolved
    track-config dict (returned unchanged).
    """
    if isinstance(track, dict):
        return track
    if track not in PHOSPHO_TRACKS:
        raise ValueError(
            f"Unknown phospho track {track!r}; valid: {list(PHOSPHO_TRACKS)}"
        )
    return PHOSPHO_TRACKS[track]


def track_output(filename, track_cfg):
    """Compose a KINASE_ATTRIBUTION output path with the track suffix appended.

    The suffix goes before the extension, e.g.
    ``stoichiometry_matrix.csv`` + track ``py`` → ``stoichiometry_matrix_pY.csv``.
    """
    cfg = resolve_track(track_cfg)
    suffix = cfg["output_suffix"]
    if not suffix:
        return os.path.join(KINASE_ATTRIBUTION_OUTPUT_DIR, filename)
    base, ext = os.path.splitext(filename)
    return os.path.join(KINASE_ATTRIBUTION_OUTPUT_DIR, f"{base}{suffix}{ext}")


def load_sample_mapping():
    """Load the sample mapping produced by the data-ingest stage."""
    import pandas as pd

    path = os.path.join(DATA_INGEST_OUTPUT_DIR, "sample_mapping.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Sample mapping not found at {path}. Run song.py --mapping first."
        )
    return pd.read_csv(path)

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

# Cortex + hippocampal-formation subset of WMB-10Xv3. Same Allen taxonomy as the
# standalone "Mouse Whole Cortex and Hippocampus 10x" dataset — those cells are
# the cortical slice of WMB-10Xv3 under a different release name, so restricting
# the region stream is equivalent to swapping in that dataset.
WMB_CORTEX_REGION_KEYS = [
    "WMB-10Xv3-CTXsp/log2",
    "WMB-10Xv3-HPF/log2",
    "WMB-10Xv3-Isocortex-1/log2",
    "WMB-10Xv3-Isocortex-2/log2",
]

# Region scope is env-switchable. Default is whole_brain because the
# specificity score is a ratio whose denominator is the brain-wide reference:
# restricting the cell pool to cortex+HPF shrinks both numerator and denominator
# and only damages classes whose cells mostly live outside the cortical mask
# (e.g. 09 CNU-LGE GABA, which the Striatal-MSN cluster legitimately targets).
# cortex_hpf remains available as a sensitivity-check toggle.
WMB_REGION_SCOPE = os.environ.get("WMB_REGION_SCOPE", "whole_brain").lower()
if WMB_REGION_SCOPE not in {"cortex_hpf", "whole_brain"}:
    raise ValueError(
        f"WMB_REGION_SCOPE must be 'cortex_hpf' or 'whole_brain', got {WMB_REGION_SCOPE!r}"
    )


def wmb_region_keys() -> list:
    """Return the active WMB region-key list per WMB_REGION_SCOPE."""
    return (WMB_CORTEX_REGION_KEYS if WMB_REGION_SCOPE == "cortex_hpf"
            else WMB_ALL_REGION_KEYS)

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

# ---------------------------------------------------------------------------
# Human reference data — SEA-AD MTG expression + Allen HBCA (CR03)
# ---------------------------------------------------------------------------

# SEA-AD MTG per-supertype mean expression (genes × 139 supertypes).
# Phase-2 download: atlas_reference.py --sea-ad-expression
# Note: the raw MTG h5ad (~50 GB) is downloaded, chunked by supertype, and
# the per-supertype mean written here. The donor-level h5ad object key on
# s3://sea-ad-single-cell-profiling/MTG/RNAseq/ is confirmed during phase 2;
# the canonical object name is expected to match "SEAAD_MTG_RNAseq_final-nuclei.2024-02-13.h5ad"
# (the same file targeted by _sea_ad_download_main_h5ad). Confirm with:
#   aws s3 ls s3://sea-ad-single-cell-profiling/MTG/RNAseq/ --no-sign-request
SEA_AD_EXPRESSION_FILE = os.path.join(DERIVED_AGGREGATES_DIR, "seaad", "expression_by_supertype.csv")

# Allen Human Brain Cell Atlas (HBCA) via abc_atlas_access.
# Phase-2 download: atlas_reference.py --hbca-download
# Dataset key within abc_atlas_access is expected to be "WHB-10Xv3" or similar
# human equivalent of WMB-10Xv3; verify with cache.list_directories before download.
# Top-level taxonomy field: HBCA uses "class" (analogous to WMB "class") — confirm
# during phase 2 by inspecting the cell_metadata CSV.
HBCA_CACHE_DIR = os.path.join(EXTERNAL_DATA_DIR, "allen_hbca")
HBCA_EXPRESSION_FILE = os.path.join(DERIVED_AGGREGATES_DIR, "hbca", "expression_by_class.csv")

# Human reference output files consumed by human_celltype_attribution.py
HUMAN_REFERENCE_OUTPUT_DIR = os.path.join("outputs", "reports", "human_reference_expression")
SEAAD_KINASE_SPECIFICITY_FILE = os.path.join(
    HUMAN_REFERENCE_OUTPUT_DIR, "seaad_kinase_specificity.csv"
)
HBCA_KINASE_SPECIFICITY_FILE = os.path.join(
    HUMAN_REFERENCE_OUTPUT_DIR, "hbca_kinase_specificity.csv"
)
# Raw mean log2 expression matrices (kinase × celltype), emitted alongside the
# specificity matrices. Consumed by the human Attribution sub-tab to surface
# absolute expression as a sanity check (mirrors mouse wmb_mean_log2_expression).
SEAAD_KINASE_EXPRESSION_FILE = os.path.join(
    HUMAN_REFERENCE_OUTPUT_DIR, "seaad_kinase_expression.csv"
)
HBCA_KINASE_EXPRESSION_FILE = os.path.join(
    HUMAN_REFERENCE_OUTPUT_DIR, "hbca_kinase_expression.csv"
)

# Final per-kinase cell-type attribution output
HUMAN_CELLTYPE_ATTRIBUTION_OUTPUT_DIR = os.path.join(
    "outputs", "reports", "kinase_attribution_human"
)
CELLTYPE_SPECIFICITY_FILE = os.path.join(
    HUMAN_CELLTYPE_ATTRIBUTION_OUTPUT_DIR, "celltype_specificity.csv"
)

# Top-N cell types per kinase per reference (for payload convenience tables)
HUMAN_CELLTYPE_TOP_N = 8

# SEA-AD MTG supertype list (139 supertypes; populated from expression file at runtime).
# This is a runtime constant — populated by human_reference_expression.py.
# The static fallback here is empty; callers that need the full list should load
# the expression CSV directly.
SEAAD_MTG_SUPERTYPES: list[str] = []

# Human reference taxonomy label for HBCA (WHB-10Xv3, Siletti 2023).
# Confirmed via WHB-taxonomy/cluster_annotation_term_set: the top-level taxon
# is "supercluster" (term_set label CCN202210140_SUPC) — ~31 superclusters.
# HBCA cell_metadata only carries cluster_alias; superculuster is derived by
# joining cluster_to_cluster_annotation_membership filtered to supercluster.
HBCA_CLASS_FIELD = "supercluster"
HBCA_TAXONOMY_TERM_SET = "CCN202210140_SUPC"

WMB_EXPRESSION_OUTPUT_DIR = os.path.join("outputs", "reports", "wmb_expression")
WMB_EXPRESSION_FILE = os.path.join(WMB_EXPRESSION_OUTPUT_DIR, "wmb_kinase_expression.csv")
# Per-kinase standard attribution breadth summary (effective number of cell
# types / top class) over the retained WMB classes. Detection-gated; the share
# `specificity_score` it replaces is removed from wmb_kinase_expression.csv.
WMB_KINASE_SPECIFICITY_FILE = os.path.join(WMB_EXPRESSION_OUTPUT_DIR, "wmb_kinase_specificity.csv")
WMB_EXPRESSION_SUBCLASS_FILE = os.path.join(WMB_EXPRESSION_OUTPUT_DIR, "wmb_kinase_expression_subclass.csv")
WMB_SUBCLASS_TO_CLASS_FILE = os.path.join(DERIVED_BRIDGES_DIR, "wmb_subclass_to_class.csv")
WMB_REGIONAL_EXPRESSION_FILE = os.path.join(WMB_EXPRESSION_OUTPUT_DIR, "wmb_regional_kinase_expression.csv")
WMB_PROTEOME_EXPRESSION_FILE = os.path.join(WMB_EXPRESSION_OUTPUT_DIR, "wmb_proteome_expression.csv")
PROTEOME_GENE_LIST_FILE = os.path.join(DATA_INGEST_OUTPUT_DIR, "total_proteome_genes.txt")

# --- TODO #2: 10x NSCLC reference (T-cell cohort cell-type specificity) ------
# Public 10x Flex dataset "Aggregate of 900k human NSCLC + normal-adjacent
# cells" (Cell Ranger multi 7.1.0, slug 16plex_900k_32_NSCLC_multiplex).
# Cell types are derived (no labels ship): coarse graphclust marker labeling +
# ProjecTILs refinement of the T/NK compartment. See nsclc_expression.py.
NSCLC_10X_CACHE_DIR = os.path.join(EXTERNAL_DATA_DIR, "nsclc_10x")
NSCLC_10X_H5_FILE = os.path.join(NSCLC_10X_CACHE_DIR, "sample_feature_bc_matrix.h5")
NSCLC_10X_ANALYSIS_TGZ = os.path.join(NSCLC_10X_CACHE_DIR, "analysis.tar.gz")
NSCLC_10X_GRAPHCLUST_FILE = os.path.join(
    NSCLC_10X_CACHE_DIR, "analysis", "clustering",
    "gene_expression_graphclust", "clusters.csv")
NSCLC_10X_DIFFEXP_FILE = os.path.join(
    NSCLC_10X_CACHE_DIR, "analysis", "diffexp",
    "gene_expression_graphclust", "differential_expression.csv")
# ProjecTILs full-cohort projection (all cells gated by scGate -> 14 states).
NSCLC_PROJECTILS_PREDICTIONS_FILE = os.path.join(
    NSCLC_10X_CACHE_DIR, "projectils_predictions.csv")
# Derived per-cell label tables.
NSCLC_CLUSTER_LABELS_FILE = os.path.join(NSCLC_10X_CACHE_DIR, "nsclc_cluster_labels.csv")
NSCLC_CELL_LABELS_FILE = os.path.join(NSCLC_10X_CACHE_DIR, "nsclc_cell_labels.csv")
# Outputs.
NSCLC_REFERENCE_OUTPUT_DIR = os.path.join("outputs", "reports", "nsclc_reference")
NSCLC_KINASE_EXPRESSION_FILE = os.path.join(NSCLC_REFERENCE_OUTPUT_DIR, "nsclc_kinase_expression.csv")
NSCLC_KINASE_AUDIT_FILE = os.path.join(NSCLC_REFERENCE_OUTPUT_DIR, "nsclc_kinase_audit.csv")

SONG_H5AD_FILE = os.path.join(SONG_TRANSCRIPTOMICS_DIR, "170_gex_celltypes_00.h5ad")

SNRNA_INTEGRATION_OUTPUT_DIR = os.path.join("outputs", "reports", "snrna_integration")
SONG_PSEUDOBULK_FILE = os.path.join(SNRNA_INTEGRATION_OUTPUT_DIR, "pseudobulk_cpm.csv")
SONG_CELL_COUNTS_FILE = os.path.join(SNRNA_INTEGRATION_OUTPUT_DIR, "pseudobulk_cell_counts.csv")
SONG_EXPRESSION_FILE = os.path.join(SNRNA_INTEGRATION_OUTPUT_DIR, "song_expression_specificity.csv")
# Per-(gene, cluster) detection: fraction of spine nuclei with a non-zero raw
# count, pooled across animals. Count-based → normalization-free; the single
# presence gate of the standard attribution metric. Written by --pseudobulk
# (needs per-cell counts), consumed by --specificity.
SONG_DETECTION_FILE = os.path.join(SNRNA_INTEGRATION_OUTPUT_DIR, "song_detection.csv")
SONG_CONCORDANCE_FILE = os.path.join(SNRNA_INTEGRATION_OUTPUT_DIR, "song_concordance.csv")

SONG_LFC_MIN = 0.1       # minimum |song_lfc| for concordance (same as SEA_AD_LFC_MIN)
SONG_CONCORDANCE_WEIGHT = 3.0
SEA_AD_CONCORDANCE_WEIGHT = 1.0
SONG_MIN_CELLS = int(os.environ.get("SONG_MIN_CELLS", "5"))   # min cells per animal×cluster (Levy-t5 spine gate); overridable via env for bench sweeps
SONG_MIN_ANIMALS = 10     # minimum animals per cluster for concordance DE


def load_barcode_labels(granularity: str = "subclass") -> dict[str, str]:
    """Per-cell label map at the requested granularity.

    granularity:
      - "subclass" (default): 46 Levy names (the basis for CLUSTER_SPINE)
      - "coarse":   10 Cluster_coarse levels (per-cell, not a rollup)
      - "fine":     40 Cluster_fine levels (legacy, retained for traceability)

    Callers filter to a spine / level subset themselves.
    """
    import csv
    col = {
        "subclass": "cluster_subclass",
        "coarse": "cluster_coarse",
        "fine": "cluster_fine",
    }.get(granularity)
    if col is None:
        raise ValueError(f"unknown granularity: {granularity!r}")
    out: dict[str, str] = {}
    with open(BARCODE_TO_CLUSTER_FILE, newline="") as f:
        for row in csv.DictReader(f):
            out[row["barcode"]] = row[col]
    return out


def load_barcode_to_cluster_map() -> dict[str, str]:
    """Backwards-compatible alias for `load_barcode_labels('subclass')`."""
    return load_barcode_labels("subclass")


def load_cluster_to_wmb_class_map() -> dict[str, str]:
    """Spine-cluster → WMB-class (1:1, hand-curated). 31 entries."""
    import csv
    out: dict[str, str] = {}
    with open(CLUSTER_TO_WMB_CLASS_FILE, newline="") as f:
        for row in csv.DictReader(f):
            out[row["cluster_name"]] = row["wmb_class_label"].strip()
    return out


def load_specificity_unit_map() -> dict[str, dict]:
    """Song cluster → its specificity unit (the resolution the pill is scored at).

    One curated mechanism for the three cases (see SPECIFICITY_COLLAPSE_WMB_CLASSES):

    * collapse  — WMB class is in the collapse set: unit = the WMB-class parent,
                  ``collapsed=True`` (member clusters are subtypes of one cell type).
    * combined  — WMB class maps 1:1 to a single Song cluster: unit = that cluster,
                  with a combined "cluster · class" label, ``collapsed=False``.
    * split     — WMB class has several genuinely distinct clusters but is not in
                  the collapse set: unit = the bare Song cluster, ``collapsed=False``.

    Returns ``{cluster: {"unit": id, "label": str, "collapsed": bool,
    "wmb_class": str, "children": [clusters in the same unit]}}``.
    """
    from collections import Counter
    cluster_to_wmb = load_cluster_to_wmb_class_map()
    n_per_class = Counter(cluster_to_wmb.values())

    # First pass: assign each cluster a unit id.
    unit_of: dict[str, str] = {}
    for cluster, wmb in cluster_to_wmb.items():
        unit_of[cluster] = wmb if wmb in SPECIFICITY_COLLAPSE_WMB_CLASSES else cluster

    children: dict[str, list[str]] = {}
    for cluster, unit in unit_of.items():
        children.setdefault(unit, []).append(cluster)
    for unit in children:
        children[unit].sort()

    out: dict[str, dict] = {}
    for cluster, wmb in cluster_to_wmb.items():
        unit = unit_of[cluster]
        if wmb in SPECIFICITY_COLLAPSE_WMB_CLASSES:
            label, collapsed = WMB_CLASS_DISPLAY.get(wmb, wmb), True
        elif n_per_class[wmb] == 1:
            label, collapsed = f"{cluster} · {wmb}", False
        else:
            label, collapsed = cluster, False
        out[cluster] = {
            "unit": unit, "label": label, "collapsed": collapsed,
            "wmb_class": wmb, "children": children[unit],
        }
    return out


def load_cluster_to_seaad_supertype_map() -> dict[str, list[tuple[str, float]]]:
    """Spine-cluster → list of (SEA-AD supertype, weight). 31 entries.

    `n/a` rows return an empty list so callers can drop those clusters from
    SEA-AD evidence cleanly. Weights within a mapped cluster sum to 1.0.
    """
    import csv
    out: dict[str, list[tuple[str, float]]] = {}
    with open(CLUSTER_TO_SEAAD_SUPERTYPE_FILE, newline="") as f:
        for row in csv.DictReader(f):
            cluster = row["cluster_name"]
            supertype = row["seaad_supertype"]
            entries = out.setdefault(cluster, [])
            if supertype != "n/a":
                entries.append((supertype, float(row["weight"])))
    return out

def load_cluster_to_hbca_supercluster_map() -> dict[str, list[tuple[str, float]]]:
    """Spine-cluster → list of (HBCA supercluster, weight).

    Weights are 1.0 per edge; `_rollup_matrix_to_levy_t5` normalizes by the
    sum at rollup time, so equal-weight links average the source columns.
    """
    import csv
    out: dict[str, list[tuple[str, float]]] = {}
    with open(CLUSTER_TO_HBCA_SUPERCLUSTER_FILE, newline="") as f:
        for row in csv.DictReader(f):
            cluster = row["cluster_name"]
            supercluster = row["hbca_supercluster"]
            if supercluster == "n/a":
                out.setdefault(cluster, [])
                continue
            out.setdefault(cluster, []).append((supercluster, float(row["weight"])))
    return out


SONG_PATHWAY_MAP = {"App": "App", "Tau": "Tau", "ApTt": "ApTt"}

SAP_FACTORIAL = {
    "WTyp": (0, 0, 0),
    "AppP": (1, 0, 0),
    "Ttau": (0, 1, 0),
    "ApTt": (1, 1, 1),
}

# Raw-TMT-label genotype tokens → SAP canonical condition names.
# Boundary normalizer used at ingest time (data_ingest) and inside
# `parse_animal_id` (anywhere an animal_id string is parsed downstream).
GENOTYPE_TO_SAP = {
    "WT": "WTyp",
    "APP": "AppP",
    "T22": "Ttau",
    "T22/APP": "ApTt",
}

# Disease × time contrast linear combinations on the factorial OLS coefficients
# (const, App, Tau, Int, time_4mo, time_6mo, App_x_time4, App_x_time6,
# Tau_x_time4, Tau_x_time6). 9 contrasts total.
CONTRAST_COEFS = {
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

# Animal-ID regex. Two flavors of input occur:
#   - Full TMT label:        "1_C198(L)_M_2mo_WT"
#   - Decomposition variant: "1_M_2mo_WT"  (no mouse_id segment)
# `parse_animal_id` accepts either, returns long-form genotype.
_ANIMAL_ID_FULL_RE = re.compile(
    r"^(?P<sample_num>\d+)_(?P<mouse_id_raw>.+?)_(?P<sex>M|F)_"
    r"(?P<timepoint>2mo|4mo|6mo)_(?P<genotype>WT|T22|APP|T22/APP)$"
)
_ANIMAL_ID_SHORT_RE = re.compile(
    r"^(?P<sample_num>\d+)_(?P<sex>M|F)_"
    r"(?P<timepoint>\dmo)_(?P<genotype>WT|T22|APP|T22/APP)$"
)


def parse_animal_id(animal_str: str) -> dict | None:
    """Parse a TMT animal ID string, returning long-form genotype.

    Returns a dict with keys sample_num, mouse_id_raw (None for short form),
    sex, timepoint, genotype (long form: WTyp/AppP/Ttau/ApTt). Returns None
    if the string doesn't match either expected format.
    """
    m = _ANIMAL_ID_FULL_RE.match(animal_str)
    if m is not None:
        d = m.groupdict()
        d["sample_num"] = int(d["sample_num"])
    else:
        m = _ANIMAL_ID_SHORT_RE.match(animal_str)
        if m is None:
            return None
        d = m.groupdict()
        d["sample_num"] = int(d["sample_num"])
        d["mouse_id_raw"] = None
    d["genotype"] = GENOTYPE_TO_SAP[d["genotype"]]
    return d

PHOSPHATASE_GENE_PREFIXES = [
    "Ppp", "Ptpn", "Ptpr", "Dusp", "Ppm", "Ssh", "Ctdsp", "Ctds",
]
PHOSPHATASE_GENES_EXTRA = [
    "Pten", "Cdc25a", "Cdc25b", "Cdc25c", "Inpp5d", "Inpp5e",
    "Inpp4a", "Inpp4b", "Synj1", "Synj2", "Mtmr1", "Mtmr2",
]

MAPPING_CACHE_FILE = os.path.join(DERIVED_CACHES_DIR, "kinase_to_gene_mapping.csv")

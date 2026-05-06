"""Path and numeric constants for the unified viewer build.

Single source of truth shared by `code/build_unified_viewer.py` and
`code/viewer/pathway_payload.py`. No logic — just paths and tunables.

Sections:
- Top-level inputs (factorial aggregation outputs)
- Viewer output dirs + payload paths
- Kinase-side artifacts (kept; Kinase Explorer reads these)
- Pathway-side artifacts (candidates for sunset)
- Numeric tunables
"""

from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # code/
sys.path.insert(0, HERE)

import config  # noqa: E402
import config_integration as icfg  # noqa: E402

# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
AGGREGATION_DIR = os.path.join(icfg.FACTORIAL_ALL_PAIRS_DIR, "aggregation")
EDGES_PARQUET = os.path.join(AGGREGATION_DIR, "kinase_backbone_edges.parquet")
EDGE_META_JSON = os.path.join(AGGREGATION_DIR, "edge_index_metadata.json")

# Pathway-side inputs (sunset candidates — only pathway tabs read these)
BACKBONE_PERM_CSV = os.path.join(
    AGGREGATION_DIR, "backbone_permutation_pvalues_by_contrast.csv"
)
BACKBONE_REC_CSV = os.path.join(
    AGGREGATION_DIR, "backbone_recurrence_by_contrast.csv"
)
BACKBONE_VOCAB_CACHE = os.path.join(AGGREGATION_DIR, "backbone_vocab.parquet")

# ---------------------------------------------------------------------------
# Viewer outputs
# ---------------------------------------------------------------------------
UNIFIED_VIEWER_OUTPUT_DIR = os.path.join(config.REPO_ROOT, "outputs", "reports")
UNIFIED_VIEWER_DIR = os.path.join(UNIFIED_VIEWER_OUTPUT_DIR, "unified_viewer")
PAYLOAD_JSON = os.path.join(UNIFIED_VIEWER_DIR, "unified_viewer.payload.json")
PAYLOAD_JSON_GZ = PAYLOAD_JSON + ".gz"
UNIFIED_VIEWER_HTML = os.path.join(UNIFIED_VIEWER_DIR, "index.html")

# Kinase-side artifacts (kept)
PER_KINASE_SUMMARY = os.path.join(
    UNIFIED_VIEWER_DIR, "edge_summaries", "per_kinase_summary.parquet"
)
EDGE_SLICES_KINASE_DIR = os.path.join(UNIFIED_VIEWER_DIR, "edge_slices", "kinase")
EDGE_SLICES_DECOMP_OLS_DIR = os.path.join(
    UNIFIED_VIEWER_DIR, "edge_slices", "decomp_ols"
)
DECOMP_OLS_PARQUET = os.path.join(
    config.REPO_ROOT, "outputs", "reports", "deconvolution",
    "per_animal", "site_level_ols.parquet",
)

# Pathway-side artifacts (sunset candidates)
SIDECAR_PARQUET = os.path.join(
    UNIFIED_VIEWER_OUTPUT_DIR, "kinase_backbone_edges_sig.parquet"
)
PER_BACKBONE_SUMMARY = os.path.join(
    UNIFIED_VIEWER_DIR, "edge_summaries", "per_backbone_summary.parquet"
)
EDGE_SLICES_BACKBONE_DIR = os.path.join(
    UNIFIED_VIEWER_DIR, "edge_slices", "backbone"
)

# ---------------------------------------------------------------------------
# Audit / measurement-trace (kinase-side)
# ---------------------------------------------------------------------------
AUDIT_SOURCES_DIR = os.path.join(UNIFIED_VIEWER_DIR, "audit_sources")
MEASUREMENT_TRACE_DIR = os.path.join(AUDIT_SOURCES_DIR, "measurement_trace")
MEASUREMENT_TRACE_INDEX = os.path.join(MEASUREMENT_TRACE_DIR, "index.json")
PIPELINE_OVERVIEW_SRC = os.path.join(config.REPO_ROOT, "docs", "pipeline_overview.html")
PIPELINE_OVERVIEW_DEST = os.path.join(UNIFIED_VIEWER_DIR, "pipeline_overview.html")
REPORT_MD = os.path.join(config.REPO_ROOT, "pipeline_notes", "phase2_payload_report.md")

# ---------------------------------------------------------------------------
# Numeric tunables
# ---------------------------------------------------------------------------
SCHEMA_VERSION = 1
TOP_N_KINASES = 5                  # per (backbone, contrast) preview in JSON
EDGE_STREAM_BATCH = 1_000_000      # rows per pyarrow batch; caps RAM
AUDIT_PREVIEW_ROWS = 25
MEASUREMENT_TRACE_SCHEMA_VERSION = 3

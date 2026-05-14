"""Path and numeric constants for the unified viewer build."""

from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # alz/
sys.path.insert(0, HERE)

import config  # noqa: E402

# ---------------------------------------------------------------------------
# Viewer outputs
# ---------------------------------------------------------------------------
UNIFIED_VIEWER_OUTPUT_DIR = os.path.join(config.REPO_ROOT, "outputs", "reports")
UNIFIED_VIEWER_DIR = os.path.join(UNIFIED_VIEWER_OUTPUT_DIR, "unified_viewer")
PAYLOAD_JSON = os.path.join(UNIFIED_VIEWER_DIR, "unified_viewer.payload.json")
PAYLOAD_JSON_GZ = PAYLOAD_JSON + ".gz"
UNIFIED_VIEWER_HTML = os.path.join(UNIFIED_VIEWER_DIR, "index.html")

# Decomp-OLS shards (lazy-loaded by the kinase audit drawer)
EDGE_SLICES_DECOMP_OLS_DIR = os.path.join(
    UNIFIED_VIEWER_DIR, "edge_slices", "decomp_ols"
)
DECOMP_OLS_PARQUET = os.path.join(
    config.REPO_ROOT, "outputs", "reports", "deconvolution",
    "per_animal", "site_level_ols.parquet",
)

# Incytr-pathway shards (lazy-loaded by the pathway-table tab)
EDGE_SLICES_INCYTR_PATHWAYS_DIR = os.path.join(
    UNIFIED_VIEWER_DIR, "edge_slices", "incytr_pathways"
)
INCYTR_FACTORIAL_OUTPUTS_DIR = os.path.join(
    config.REPO_ROOT, "outputs", "reports", "incytr_factorial"
)
INCYTR_FACTORIAL_INPUTS_DIR = os.path.join(
    config.REPO_ROOT, "data", "incytr_factorial_inputs"
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
TOP_N_KINASES = 5                  # per-kinase preview rows in JSON
AUDIT_PREVIEW_ROWS = 25
MEASUREMENT_TRACE_SCHEMA_VERSION = 3

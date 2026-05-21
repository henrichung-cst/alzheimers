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

# Song concordance shards — lazy-loaded per gene.
EDGE_SLICES_SONG_CONCORDANCE_DIR = os.path.join(
    UNIFIED_VIEWER_DIR, "edge_slices", "song_concordance"
)

# Incytr-pathway shards (lazy-loaded by the pathway-table tab)
EDGE_SLICES_INCYTR_PATHWAYS_DIR = os.path.join(
    UNIFIED_VIEWER_DIR, "edge_slices", "incytr_pathways"
)

# Per-kinase shards for the human Audit drawer's leading-edge + substrate-motif
# fields. These two columns dominated PAYLOAD.human.perdonor_index (~50 MB);
# fetched on demand from the Trace + Running Enrichment sub-tabs.
EDGE_SLICES_HUMAN_PERDONOR_DIR = os.path.join(
    UNIFIED_VIEWER_DIR, "edge_slices", "human_perdonor"
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

# Per-cluster transcript pseudobulk shards backing the Incytr Pathways
# "Measurement Trace" panel. Substrate is the per-(cluster, Group) mean of
# `Data.input@assays$originalexp@data` emitted by
# bench/incytr_pair_levy_t5/emit_expr_bygroup.R — bit-for-bit the same matrix
# Incytr's `Cal_scFC` consumes, so the panel agrees numerically with the FC
# tab's `*_sclog2FC` columns.
TRANSCRIPT_TRACE_DIR = os.path.join(AUDIT_SOURCES_DIR, "transcript_trace")
TRANSCRIPT_TRACE_INDEX = os.path.join(TRANSCRIPT_TRACE_DIR, "index.json")
TRANSCRIPT_TRACE_PSEUDOBULK = os.path.join(
    config.REPO_ROOT, "outputs", "reports", "decomposition", "levy_t5",
    "transcript_per_cluster.parquet",
)
TRANSCRIPT_TRACE_SAMPLEKEY = os.path.join(
    config.REPO_ROOT, "data", "incytr_frozen", "v2_46clusters",
    "provenance", "yuyu_samplekey.csv",
)
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
TRANSCRIPT_TRACE_SCHEMA_VERSION = 3

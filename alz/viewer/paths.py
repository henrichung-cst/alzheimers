"""Path and numeric constants for the unified viewer build."""

from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from alz.shared import config  # noqa: E402

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
    config.REPO_ROOT, "outputs", "reports", "decomposition",
    config.CLUSTER_SPINE_NAME, "per_animal", "site_level_ols.parquet",
)

# Song concordance shards — lazy-loaded per gene.
EDGE_SLICES_SONG_CONCORDANCE_DIR = os.path.join(
    UNIFIED_VIEWER_DIR, "edge_slices", "song_concordance"
)

# Incytr-pathway shards (lazy-loaded by the pathway-table tab)
EDGE_SLICES_INCYTR_PATHWAYS_DIR = os.path.join(
    UNIFIED_VIEWER_DIR, "edge_slices", "incytr_pathways"
)
# Incytr backbone-grain shards — one subdir per grain (R-EM, L-R-EM, R-EM-T).
# Only the R-EM-T subdir gets per-(sender,receiver) parquet shards; R-EM and
# L-R-EM ship their global binary index as the only file in their subdir.
EDGE_SLICES_INCYTR_BACKBONE_DIR = os.path.join(
    UNIFIED_VIEWER_DIR, "edge_slices", "incytr_backbone"
)
# Source backbone parquets emitted by the pair-mode driver (B-2, parallel to wide/).
INCYTR_BACKBONE_PAIR_MODE_DIR = os.path.join(
    config.REPO_ROOT, "outputs", "reports", "incytr_pair_mode", "backbone"
)
# 5xFAD incytr shards — separate dirs per tissue so Song's rmtree never clobbers them.
EDGE_SLICES_INCYTR_PATHWAYS_5XFAD_CORTEX_DIR = os.path.join(
    UNIFIED_VIEWER_DIR, "edge_slices", "incytr_pathways_fivexfad_cortex"
)
EDGE_SLICES_INCYTR_PATHWAYS_5XFAD_HIPPO_DIR = os.path.join(
    UNIFIED_VIEWER_DIR, "edge_slices", "incytr_pathways_fivexfad_hippocampus"
)

# Per-kinase shards for the human Audit drawer's leading-edge + substrate-motif
# fields. These two columns dominated PAYLOAD.human.perdonor_index (~50 MB);
# fetched on demand from the Trace + Running Enrichment sub-tabs.
EDGE_SLICES_HUMAN_PERDONOR_DIR = os.path.join(
    UNIFIED_VIEWER_DIR, "edge_slices", "human_perdonor"
)
INCYTR_PAIR_MODE_OUTPUTS_DIR = os.path.join(
    config.REPO_ROOT, "outputs", "reports", "incytr_pair_mode"
)

# B4 kinase↔Incytr bridge outputs (per-cohort subdirs: song / fivexfad_cortex /
# fivexfad_hippocampus). Carries kinase_participation.csv (n_backbones, n_paths).
KINASE_INCYTR_BRIDGE_DIR = os.path.join(
    config.REPO_ROOT, "outputs", "reports", "kinase_incytr_bridge"
)

# 5xFAD supporting-cohort kinase attribution inputs (also referenced by the
# shared audit-manifest composer in build_unified_viewer._audit_specs).
FIVEXFAD_KINASE_DIR = os.path.join(
    config.REPO_ROOT, "outputs", "reports", "kinase_attribution_5xfad"
)

# ---------------------------------------------------------------------------
# Audit / measurement-trace (kinase-side)
# ---------------------------------------------------------------------------
AUDIT_SOURCES_DIR = os.path.join(UNIFIED_VIEWER_DIR, "audit_sources")
MEASUREMENT_TRACE_DIR = os.path.join(AUDIT_SOURCES_DIR, "measurement_trace")
MEASUREMENT_TRACE_INDEX = os.path.join(MEASUREMENT_TRACE_DIR, "index.json")

# Per-cluster transcript pseudobulk shards backing the Incytr Pathways
# Evidence tab's transcript sub-row. Substrate is the per-(cluster, Group)
# mean of `Data.input@assays$originalexp@data` emitted by
# alz/incytr_pair/emit_expr_bygroup.R — bit-for-bit the same matrix
# Incytr's `Cal_scFC` consumes, so the JS-side LFC recomputation in
# evidence_row.js agrees numerically with the stored `*_sclog2FC` columns.
TRANSCRIPT_TRACE_DIR = os.path.join(AUDIT_SOURCES_DIR, "transcript_trace")
TRANSCRIPT_TRACE_INDEX = os.path.join(TRANSCRIPT_TRACE_DIR, "index.json")
TRANSCRIPT_TRACE_PSEUDOBULK = os.path.join(
    config.REPO_ROOT, "outputs", "reports", "decomposition",
    config.CLUSTER_SPINE_NAME, "transcript_per_cluster.parquet",
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

# Per-cluster protein + phospho (pS/pT + pY) raw-value shards backing the
# Incytr Pathways "Evidence" tab.  Substrate is
# outputs/reports/decomposition/levy_t5/{protein,phospho,phospho_pY}_per_cluster.parquet.
OMICS_TRACE_DIR = os.path.join(AUDIT_SOURCES_DIR, "omics_trace")
OMICS_TRACE_INDEX = os.path.join(OMICS_TRACE_DIR, "index.json")
OMICS_TRACE_SCHEMA_VERSION = 3

# Companion limma-normalized condition means (per cluster × contrast × layer)
# backing the Evidence tab's right-edge LFC recomputation. Mirrors Incytr's
# `normalizeBetweenArrays` step pre-`Cal_foldchange` so a JS-side
# `log2((D + 1e-3) / (W + 1e-3))` (ε = 1e-3 matching the driver's
# `correction = 0.001`) agrees with stored `*_pr/_ps/_py_log2FC` to <= 1e-4.
# Built by alz/integration/build_normalized_substrate.py.
OMICS_TRACE_NORMALIZED_DIR = os.path.join(
    AUDIT_SOURCES_DIR, "omics_trace_normalized"
)
OMICS_TRACE_NORMALIZED_INDEX = os.path.join(
    OMICS_TRACE_NORMALIZED_DIR, "index.json"
)
OMICS_TRACE_NORMALIZED_SCHEMA_VERSION = 1

# 5xFAD evidence-panel shards (Incytr Pathways Evidence tab, fivexfad contexts).
# Per-tissue dirs so Song's rmtree never clobbers them. Built read-only from the
# already-on-disk per-sample matrices by build_{omics,transcript}_trace_fivexfad.py;
# wired into PAYLOAD.meta.{omics,transcript}_trace.by_context.
OMICS_TRACE_FIVEXFAD_SCHEMA_VERSION = 1
OMICS_TRACE_FIVEXFAD_CORTEX_DIR = os.path.join(
    AUDIT_SOURCES_DIR, "omics_trace_fivexfad_cortex"
)
OMICS_TRACE_FIVEXFAD_CORTEX_INDEX = os.path.join(
    OMICS_TRACE_FIVEXFAD_CORTEX_DIR, "index.json"
)
OMICS_TRACE_FIVEXFAD_HIPPO_DIR = os.path.join(
    AUDIT_SOURCES_DIR, "omics_trace_fivexfad_hippocampus"
)
OMICS_TRACE_FIVEXFAD_HIPPO_INDEX = os.path.join(
    OMICS_TRACE_FIVEXFAD_HIPPO_DIR, "index.json"
)

TRANSCRIPT_TRACE_FIVEXFAD_SCHEMA_VERSION = 1
TRANSCRIPT_TRACE_FIVEXFAD_CORTEX_DIR = os.path.join(
    AUDIT_SOURCES_DIR, "transcript_trace_fivexfad_cortex"
)
TRANSCRIPT_TRACE_FIVEXFAD_CORTEX_INDEX = os.path.join(
    TRANSCRIPT_TRACE_FIVEXFAD_CORTEX_DIR, "index.json"
)
TRANSCRIPT_TRACE_FIVEXFAD_HIPPO_DIR = os.path.join(
    AUDIT_SOURCES_DIR, "transcript_trace_fivexfad_hippocampus"
)
TRANSCRIPT_TRACE_FIVEXFAD_HIPPO_INDEX = os.path.join(
    TRANSCRIPT_TRACE_FIVEXFAD_HIPPO_DIR, "index.json"
)

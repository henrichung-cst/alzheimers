"""Path and numeric constants for the T-cell viewer build."""

from __future__ import annotations

from collections.abc import Mapping
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
UNIFIED_VIEWER_DIR = os.path.join(UNIFIED_VIEWER_OUTPUT_DIR, "tcell_viewer")
PAYLOAD_JSON = os.path.join(UNIFIED_VIEWER_DIR, "tcell_viewer.payload.json")
PAYLOAD_JSON_GZ = PAYLOAD_JSON + ".gz"
UNIFIED_VIEWER_HTML = os.path.join(UNIFIED_VIEWER_DIR, "index.html")

# Incytr-pathway shards (lazy-loaded by the pathway-table tab).
# Shards keyed `{donor}__{sender}__{receiver}.parquet`.
EDGE_SLICES_INCYTR_PATHWAYS_DIR = os.path.join(
    UNIFIED_VIEWER_DIR, "edge_slices", "incytr_pathways"
)

# Backbone-grain shards (lazy-loaded by the pathway-table tab's grain selector).
# Shard filenames are `{sender}__{receiver}.parquet` with no context in the
# name, so each donor gets its own subdir joined at call time.
EDGE_SLICES_INCYTR_BACKBONE_DIR = os.path.join(
    UNIFIED_VIEWER_DIR, "edge_slices", "incytr_backbone"
)

# T-cell pair-mode parquet root (per-donor wide outputs). The production default
# is the completed positive/negative-marker per-cell labeling rerun. Its state
# ontology remains report-backed; ProjecTILs is supporting evidence only.
def resolve_incytr_pair_mode_tcells_dir(
    environ: Mapping[str, str] | None = None,
) -> str:
    """Resolve the explicit diagnostic override or the production default."""
    env = os.environ if environ is None else environ
    return env.get(
        "TCELL_INCYTR_PAIR_MODE_DIR",
        os.path.join(
            config.REPO_ROOT, "outputs", "reports",
            "incytr_pair_mode_tcells",
        ),
    )


INCYTR_PAIR_MODE_TCELLS_DIR = resolve_incytr_pair_mode_tcells_dir()

# T-cell kinase MEA (bulk; donor1 only — donor2 has no IMAC).
KINASE_ATTRIBUTION_TCELLS_DIR = os.path.join(
    config.REPO_ROOT, "outputs", "reports", "kinase_attribution_tcells"
)

# T-cell scRNA-derived cell-type predictions per donor.
TCELLS_INCYTR_INPUTS_DIR = os.path.join(
    config.REPO_ROOT, "data", "derived", "tcells_incytr_inputs"
)

# ---------------------------------------------------------------------------
# Audit (small T-cell manifest — full kinase audit drawer deferred until
# per-cluster substrate trace is wired)
# ---------------------------------------------------------------------------
AUDIT_SOURCES_DIR = os.path.join(UNIFIED_VIEWER_DIR, "audit_sources")
REPORT_MD = os.path.join(config.REPO_ROOT, "pipeline_notes", "tcell_payload_report.md")

# ---------------------------------------------------------------------------
# Numeric tunables
# ---------------------------------------------------------------------------
SCHEMA_VERSION = 1
TOP_N_KINASES = 5
AUDIT_PREVIEW_ROWS = 25

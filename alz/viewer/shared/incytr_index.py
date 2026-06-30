"""Incytr binary filter-index format: vocab constants + the pure per-pair encoders shared by
both viewer builders' shard writers (phase 5d-2). One source of truth for the `.bin.gz` layout
decoded by incytr_global_index.js."""
import numpy as np
import pandas as pd
from alz.viewer.shared.payload_helpers import _INCYTR_FC_NODES

_INCYTR_LABEL_NODES = _INCYTR_FC_NODES
_INCYTR_LABEL_COLS = tuple(f"{n}_label" for n in _INCYTR_LABEL_NODES)
_INCYTR_LABEL_VOCAB = ("DEG", "prG")

# Base score columns — always emitted by every cohort.
_INCYTR_SCORE_COLS_BASE = ("TPDS", "PPDS", "PhPDS_ps", "PhPDS_py", "SiK_score")
# Optional PTM-track score columns — only emitted by cohorts that ran the
# corresponding assay (5xFAD acetylation / ubiquitination).  Surfaced in
# score_columns and the binary index ONLY when the source parquets have at
# least one non-zero value; all-zero columns are never shipped (honesty rule).
_INCYTR_SCORE_COLS_OPTIONAL = ("Ack_score", "KGG_score", "Rme1_score")
# Backward-compatible alias: base-only tuple.  New code that needs optional-col
# gating calls _active_optional_score_cols() and combines manually.
_INCYTR_SCORE_COLS = _INCYTR_SCORE_COLS_BASE

# ---------------------------------------------------------------------------
# Backbone grain definitions (shared by all cohort builders).
# ---------------------------------------------------------------------------
# Surviving node columns per grain.  "Full" (the existing pathway grain)
# uses all four nodes and is handled by the existing pathway builder.
BACKBONE_GRAIN_NODES: dict[str, tuple[str, ...]] = {
    "R-EM":   ("Receptor", "EM"),
    "L-R-EM": ("Ligand", "Receptor", "EM"),
    "R-EM-T": ("Receptor", "EM", "Target"),
}
# inline  = global binary index only (all rows ship as one file; no per-pair shards).
# sharded = global binary index (Top mode) + per-(sender,receiver) parquet shards
#           (Cell Type mode).  R-EM-T is sharded because 2.78M rows at full scale
#           cannot be loaded in a single fetch for a per-pair drill-down.
BACKBONE_GRAIN_MODE: dict[str, str] = {
    "R-EM":   "inline",
    "L-R-EM": "inline",
    "R-EM-T": "sharded",
}
# Backbone binary index sidecar filename (parallel to _INCYTR_INDEX_FILENAME
# for Full pathways; kept distinct to avoid collision in the same output dir).
_BACKBONE_INDEX_FILENAME = "incytr_backbone_index.bin.gz"

# B-6: spine index filename (per grain, on-demand sidecar for widen mode).
# Maps spine-key → present (sender,receiver) pairs within that grain.
_BACKBONE_SPINE_INDEX_FILENAME = "backbone_spine_index.json.gz"

# DuckDB SQL expressions for the per-grain spine key (surviving node values
# joined with '|').  Used both by the builder and as the canonical key format
# the JS side uses when computing _ipSpineKey().
BACKBONE_GRAIN_SPINE_EXPR: dict[str, str] = {
    "R-EM":   "Receptor || '|' || EM",
    "L-R-EM": "Ligand || '|' || Receptor || '|' || EM",
    "R-EM-T": "Receptor || '|' || EM || '|' || Target",
}

# FC metric suffixes emitted by the incytr driver (one per node × assay channel).
_INCYTR_FC_METRICS = ("sclog2FC", "pr_log2FC", "ps_log2FC", "py_log2FC")
_INCYTR_FC_COLS = tuple(
    f"{node}_{metric}" for node in _INCYTR_FC_NODES for metric in _INCYTR_FC_METRICS
)
# Label source column → canonical label column rename (raw driver → viewer payload).
_INCYTR_LABEL_SRC = tuple(f"{n}.label" for n in _INCYTR_LABEL_NODES)

# Pre-aggregation threshold grids — user input is snapped to the nearest entry.
_INCYTR_PATHWAY_PVALUES = (0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0)
_INCYTR_PATHWAY_ABS_PDS = (0.0, 0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0)

# On-disk sidecar filenames (same for every cohort).
_INCYTR_INDEX_FILENAME = "incytr_index.bin.gz"
_INCYTR_GENE_NODE_INDEX_FILENAME = "gene_node_index.json.gz"

_SIGN_VEC_LABELS = (
    "always-up",        # uuu — every PDS > 0
    "always-down",      # ddd — every PDS < 0
    "monotonic-up",     # PDS[2] < PDS[4] < PDS[6] (strictly)
    "monotonic-down",   # PDS[2] > PDS[4] > PDS[6] (strictly)
    "mixed",            # sign changes across timepoints
)


def _active_optional_score_cols(
    src_cols: set[str],
    con,
    view_name: str = "src",
) -> tuple[str, ...]:
    """Return optional score columns that are present in src AND have at least
    one non-zero row.  All-zero columns are excluded (honesty rule — never ship
    an empty score channel for a cohort that didn't run the assay)."""
    active: list[str] = []
    for c in _INCYTR_SCORE_COLS_OPTIONAL:
        if c not in src_cols:
            continue
        n = con.execute(
            f'SELECT COUNT(*) FROM {view_name} WHERE "{c}" IS NOT NULL AND "{c}" != 0'
        ).fetchone()[0]
        if n > 0:
            active.append(c)
    return tuple(active)


def _idx_label_bits(frame) -> np.ndarray:
    # 2 bits per node (Ligand/Receptor/EM/Target): 0=none, 1=DEG, 2=prG.
    bits = np.zeros(len(frame), dtype="<u1")
    for shift, col in zip((0, 2, 4, 6), _INCYTR_LABEL_COLS):
        if col not in frame.columns:
            continue
        c = frame[col]
        codes = (c.cat.codes.to_numpy()
                 if isinstance(c.dtype, pd.CategoricalDtype)
                 else pd.Categorical(c, categories=_INCYTR_LABEL_VOCAB).codes)
        # cat.codes: -1 NaN, 0 DEG, 1 prG → +1 → 0/1/2
        bits |= ((codes + 1).astype("<u1") << shift).astype("<u1")
    return bits


def _idx_traj_bits(series) -> np.ndarray:
    s = series.fillna("").astype(str)
    bits = np.zeros(len(s), dtype="<u1")
    for i, label in enumerate(_SIGN_VEC_LABELS):  # exact tokens, no collisions
        bits |= (s.str.contains(label, regex=False).to_numpy().astype("<u1") << i)
    return bits

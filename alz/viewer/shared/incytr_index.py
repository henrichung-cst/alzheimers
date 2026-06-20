"""Incytr binary filter-index format: vocab constants + the pure per-pair encoders shared by
both viewer builders' shard writers (phase 5d-2). One source of truth for the `.bin.gz` layout
decoded by incytr_global_index.js."""
import numpy as np
import pandas as pd
from alz.viewer.shared.payload_helpers import _INCYTR_FC_NODES

_INCYTR_LABEL_NODES = _INCYTR_FC_NODES
_INCYTR_LABEL_COLS = tuple(f"{n}_label" for n in _INCYTR_LABEL_NODES)
_INCYTR_LABEL_VOCAB = ("DEG", "prG")
_INCYTR_SCORE_COLS = ("TPDS", "PPDS", "PhPDS_ps", "PhPDS_py", "SiK_score")

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

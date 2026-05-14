"""Nodes for the normalize pipeline.

Thin wrappers over `alz.kinase_normalize.step_normalize` so the legacy CLI
shim and the Kedro pipeline run the same code paths. Per-track phospho
sitequant Excels arrive as raw DataFrames from the catalog; the in-node
post-processor canonicalises them to the IMAC schema before handing off to
``step_normalize`` (which returns the five catalog-managed outputs and
emits PCA PNGs as a side effect).
"""

from __future__ import annotations

import pandas as pd

from alz.kinase_normalize import (
    _postprocess_phospho_df,
    step_normalize,
)


def normalize_track(sample_mapping: pd.DataFrame,
                    total_proteome_xlsx: pd.DataFrame,
                    phospho_sitequant_xlsx: pd.DataFrame,
                    track: str):
    """Run Stage-1 normalization for a single track.

    Returns ``(stoichiometry_matrix, raw_phospho_normalized,
    total_proteome_normalized, stoichiometry_qc, normalization_summary)``.
    """
    sq = _postprocess_phospho_df(phospho_sitequant_xlsx, track)
    return step_normalize(track, sample_mapping, total_proteome_xlsx, sq)

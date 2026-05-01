"""Stage 5: per-row confidence calibration.

| Confidence  | Condition                                                                  |
|-------------|----------------------------------------------------------------------------|
| High        | deconv FDR < 0.25 + snRNA FDR < 0.10 + direction_match = "match"           |
| Moderate    | deconv FDR < 0.25 + snRNA flat or n/a                                      |
| Low         | deconv FDR < 0.25 + snRNA significant in opposite direction                |
| Insufficient| any sample group has < MIN_CELLS_PER_GROUP cells in this cluster/contrast  |

The Insufficient gate is computed from yuyu_clustersize.csv: for each
(cluster, contrast) we compute the minimum cell count across the four
sample groups participating in that contrast (the contrast's WT
references plus its disease arm at the relevant timepoint).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from deconvolution import paths

# Each contrast (e.g. App_4mo) compares WT vs disease at one timepoint.
# The cell-count floor must hold across the male WT references AND the
# disease arm of the contrast. For factorial OLS the relevant samples
# are all male animals at any timepoint contributing to the design,
# but the disease-specific cell count is the binding constraint, so we
# check the disease arm at the contrast timepoint.

CONTRAST_GROUPS = {
    "App_2mo":  ["ma_2mo_AppP", "ma_2mo_WTyp"],
    "App_4mo":  ["ma_4mo_AppP", "ma_4mo_WTyp"],
    "App_6mo":  ["ma_6mo_AppP", "ma_6mo_WTyp"],
    "Tau_2mo":  ["ma_2mo_Ttau", "ma_2mo_WTyp"],
    "Tau_4mo":  ["ma_4mo_Ttau", "ma_4mo_WTyp"],
    "Tau_6mo":  ["ma_6mo_Ttau", "ma_6mo_WTyp"],
    "ApTt_2mo": ["ma_2mo_ApTt", "ma_2mo_WTyp"],
    "ApTt_4mo": ["ma_4mo_ApTt", "ma_4mo_WTyp"],
    "ApTt_6mo": ["ma_6mo_ApTt", "ma_6mo_WTyp"],
}


def compute_min_cells(cluster_size_df: pd.DataFrame) -> pd.DataFrame:
    """Return long DataFrame: cluster × contrast → n_cells_min."""
    rows = []
    for cluster in cluster_size_df.index:
        for contrast, groups in CONTRAST_GROUPS.items():
            counts = []
            for g in groups:
                if g in cluster_size_df.columns:
                    counts.append(int(cluster_size_df.loc[cluster, g]))
            n_min = min(counts) if counts else 0
            rows.append({
                "cluster": cluster, "contrast": contrast, "n_cells_min": n_min,
            })
    return pd.DataFrame(rows)


def label(merged: pd.DataFrame, cluster_size_df: pd.DataFrame) -> pd.DataFrame:
    """Add `confidence` and `n_cells_min` columns to the Stage 4 table."""
    floor = compute_min_cells(cluster_size_df)
    df = merged.merge(floor, how="left", on=["cluster", "contrast"])

    deconv_sig = df["FDR"] < paths.DECON_FDR_THRESH
    snrna_sig = df["kinase_gene_FDR_snRNA"] < paths.SNRNA_FDR_HIGH
    insufficient = df["n_cells_min"].fillna(0) < paths.MIN_CELLS_PER_GROUP

    conf = np.full(len(df), "Low", dtype=object)
    conf[deconv_sig & snrna_sig & (df["direction_match"] == "match")] = "High"
    conf[deconv_sig & df["direction_match"].isin(["flat", "n/a"])] = "Moderate"
    # "opposite" → Low (already default for deconv_sig rows)
    conf[~deconv_sig] = "NotSig"
    conf[insufficient] = "Insufficient"

    df["confidence"] = conf
    return df

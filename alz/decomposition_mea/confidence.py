"""Stage 5: attach per-row evidence columns to the Stage 4 MEA table.

Joins the numeric/boolean columns a downstream reader needs to decide
whether to act on a row — but does **not** assign a categorical label.
Readers gate on the underlying columns directly:

- ``FDR`` (bulk MEA)
- ``n_cells_min`` (per-row floor across the contrast's two groups)
- ``cohort_concordant`` / ``frac_match`` / ``cohort_fdr`` (stratum-level
  binomial against snRNA kinase-gene LFC)
- ``expressed`` (kinase mRNA above ``EXPR_PRESENCE_FLOOR`` in this WMB class)
- ``kinase_gene_LFC_snRNA`` / ``direction_match`` (per-row sign agreement)

Per-row snRNA FDR is not used as a gate: the n≈15-male snRNA cohort
produces saturated per-row FDRs. Cohort-level binomial concordance and
expression presence are the surviving signals.
"""
from __future__ import annotations

import pandas as pd

from alz.decomposition_mea import paths

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


def compute_min_cells(group_class_counts: pd.DataFrame) -> pd.DataFrame:
    """Return long DataFrame: wmb_class × contrast → n_cells_min.

    ``group_class_counts`` rows are WMB classes, columns are group sample
    IDs (e.g. ``ma_2mo_AppP``). Each contrast's floor is the minimum
    nucleus count across its two groups.
    """
    rows = []
    for wmb_class in group_class_counts.index:
        for contrast, groups in CONTRAST_GROUPS.items():
            counts = [
                int(group_class_counts.loc[wmb_class, g])
                for g in groups if g in group_class_counts.columns
            ]
            rows.append({
                "wmb_class": wmb_class, "contrast": contrast,
                "n_cells_min": min(counts) if counts else 0,
            })
    return pd.DataFrame(rows)


def attach_evidence(merged: pd.DataFrame, group_class_counts: pd.DataFrame,
                    cohort_df: pd.DataFrame,
                    expressed: pd.Series) -> pd.DataFrame:
    """Add ``n_cells_min``, ``cohort_concordant``, ``frac_match``,
    ``cohort_fdr``, and ``expressed`` to the Stage 4 table. No categorical
    label is assigned; downstream readers gate on these columns directly.
    """
    floor = compute_min_cells(group_class_counts)
    df = merged.merge(floor, how="left", on=["wmb_class", "contrast"])

    cohort_keep = cohort_df.reindex(
        columns=["wmb_class", "contrast", "frac_match",
                 "cohort_fdr", "cohort_concordant"])
    df = df.merge(cohort_keep, how="left", on=["wmb_class", "contrast"])
    df["cohort_concordant"] = df["cohort_concordant"].fillna(False).astype(bool)
    df["expressed"] = expressed.reindex(df.index).fillna(False).astype(bool)
    return df

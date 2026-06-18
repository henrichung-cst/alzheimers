"""Shared MEA post-processing helpers for per-entity (donor / timepoint) runs.

Public API
----------
KIND_SPEC : dict
    Shared ``lfc_key`` + ``infix`` for ``stoich`` and ``raw`` kinds.
    Cohort-specific extra keys (``matrix_kind`` for mukesh, ``base`` for
    tcells) are merged in by each producer locally.

build_nes_fdr_matrices(mea_df, *, entity_col_name, contrast_suffix,
                       entity_order) -> (nes_wide, fdr_wide)
    Pivot a long MEA DataFrame into (kinase × entity) NES and FDR matrices.
    ``contrast_suffix`` is stripped from the ``contrast`` column to form
    entity labels; ``entity_order`` is an explicit list used to reindex
    columns (caller is responsible for computing the order: mukesh = ad_ids
    + ctrl_ids, tcell = numeric-day-sorted labels).

build_recurrence_summary(nes_wide, fdr_wide, *, subset_ids, axis_noun,
                         fdr_thresh) -> pd.DataFrame
    Compute a per-kinase recurrence summary over a column subset of the
    wide matrices.  ``axis_noun`` controls the count-column names
    (``"donors"`` → ``n_donors_sig``, ``"timepoints"`` → ``n_timepoints_sig``,
    etc.).  An empty ``subset_ids`` returns a correctly-shaped empty DataFrame
    (preserving mukesh's empty-CTRL guard).

mea_output_path(out_dir, stem, infix, suffix) -> str
    Return the canonical ``{out_dir}/{stem}{infix}{suffix}.csv`` path.
"""

from __future__ import annotations

import os
from typing import Sequence

import pandas as pd

# ---------------------------------------------------------------------------
# Shared KIND_SPEC (lfc_key + infix only).
# Each producer merges in its cohort-specific key (matrix_kind / base).
# ---------------------------------------------------------------------------

KIND_SPEC: dict[str, dict[str, str]] = {
    "stoich": {"lfc_key": "stoich_lfc", "infix": ""},
    "raw":    {"lfc_key": "raw_lfc",    "infix": "_raw"},
}


# ---------------------------------------------------------------------------
# NES / FDR wide pivot
# ---------------------------------------------------------------------------

def build_nes_fdr_matrices(
    mea_df: pd.DataFrame,
    *,
    entity_col_name: str,
    contrast_suffix: str,
    entity_order: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pivot long MEA result into (kinase × entity) NES and FDR DataFrames.

    Parameters
    ----------
    mea_df:
        Long-form MEA result with at least ``contrast``, ``kinase``, ``NES``,
        ``FDR`` columns (as written by ``kinase_enrich._run_mea``).
    entity_col_name:
        Name of the entity axis column to create (e.g. ``"donor"`` or
        ``"timepoint"``).  The column is **not** added to ``mea_df`` in-place;
        a working copy is used.
    contrast_suffix:
        Suffix to strip from ``contrast`` to produce entity labels (e.g.
        ``"_vs_CTRLmean"`` for mukesh, ``"_vs_d2"`` for tcells).
    entity_order:
        Explicit ordered list of entity labels used to reindex wide columns.
        Caller computes the order (mukesh: ``ad_ids + ctrl_ids``; tcells:
        numeric-day-sorted labels found in the long table).

    Returns
    -------
    (nes_wide, fdr_wide) : tuple[pd.DataFrame, pd.DataFrame]
        kinase-indexed wide matrices with entity labels as columns.
    """
    df = mea_df.copy()
    df[entity_col_name] = df["contrast"].str.replace(
        contrast_suffix, "", regex=False
    )
    nes_wide = df.pivot_table(
        index="kinase", columns=entity_col_name, values="NES", aggfunc="first"
    ).reindex(columns=list(entity_order))
    fdr_wide = df.pivot_table(
        index="kinase", columns=entity_col_name, values="FDR", aggfunc="first"
    ).reindex(columns=list(entity_order))
    return nes_wide, fdr_wide


# ---------------------------------------------------------------------------
# Recurrence summary
# ---------------------------------------------------------------------------

def build_recurrence_summary(
    nes_wide: pd.DataFrame,
    fdr_wide: pd.DataFrame,
    *,
    subset_ids: Sequence[str],
    axis_noun: str,
    fdr_thresh: float,
) -> pd.DataFrame:
    """Compute a per-kinase recurrence summary over a column subset.

    Parameters
    ----------
    nes_wide, fdr_wide:
        kinase-indexed wide matrices returned by ``build_nes_fdr_matrices``.
    subset_ids:
        Column subset to compute recurrence over.  For mukesh this is either
        ``ad_ids`` or ``ctrl_ids``; for tcells it is all timepoint labels.
        An **empty** list returns a correctly-shaped empty DataFrame (with
        the right column names) so downstream writers get a zero-row file
        instead of a crash.
    axis_noun:
        Noun that parameterises count-column names: ``"donors"`` produces
        ``n_donors_sig``/``n_donors_up``/``n_donors_down``/``n_donors_tested``;
        ``"timepoints"`` produces the ``n_timepoints_*`` equivalents.
    fdr_thresh:
        FDR threshold for significance (use ``alz.shared.config.MEA_FDR_THRESH``).

    Returns
    -------
    pd.DataFrame
        Columns: ``kinase``, ``n_{axis_noun}_sig``, ``n_{axis_noun}_up``,
        ``n_{axis_noun}_down``, ``n_{axis_noun}_tested``, ``median_nes``,
        ``median_nes_sig_only``.  Sorted descending by sig, then tested.
    """
    sig_col = f"n_{axis_noun}_sig"
    up_col  = f"n_{axis_noun}_up"
    dn_col  = f"n_{axis_noun}_down"
    tst_col = f"n_{axis_noun}_tested"
    _COLS = ["kinase", sig_col, up_col, dn_col, tst_col, "median_nes", "median_nes_sig_only"]

    if not subset_ids:
        return pd.DataFrame(columns=_COLS)

    nes_sub = nes_wide.reindex(columns=list(subset_ids))
    fdr_sub = fdr_wide.reindex(columns=list(subset_ids))
    sig = fdr_sub < fdr_thresh
    up  = sig & (nes_sub > 0)
    dn  = sig & (nes_sub < 0)
    return pd.DataFrame({
        "kinase":            fdr_sub.index,
        sig_col:             sig.sum(axis=1).values,
        up_col:              up.sum(axis=1).values,
        dn_col:              dn.sum(axis=1).values,
        tst_col:             fdr_sub.notna().sum(axis=1).values,
        "median_nes":        nes_sub.median(axis=1, skipna=True).values,
        "median_nes_sig_only": nes_sub.where(sig).median(axis=1, skipna=True).values,
    }).sort_values([sig_col, tst_col], ascending=[False, False])


# ---------------------------------------------------------------------------
# Path helper
# ---------------------------------------------------------------------------

def mea_output_path(out_dir: str, stem: str, infix: str, suffix: str) -> str:
    """Return ``{out_dir}/{stem}{infix}{suffix}.csv``."""
    return os.path.join(out_dir, f"{stem}{infix}{suffix}.csv")

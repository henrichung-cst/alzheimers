"""Shared trajectory annotation helper for viewer cohort builders.

Both Song/AD and 5xFAD pivot their long-form shard data by (path, disease) ×
timepoint to classify each path-disease combination as always-up, always-down,
monotonic-up, monotonic-down, or mixed.  The logic is identical; only the
timepoint set and valid-disease set differ between cohorts.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from alz.viewer.shared.incytr_index import _SIGN_VEC_LABELS

if TYPE_CHECKING:
    pass


def annotate_trajectory_columns(
    df: "pd.DataFrame",
    timepoints: "tuple[str, ...]",
    valid_diseases: "set[str]",
    source_label: str = "pair_mode",
) -> "tuple[pd.DataFrame, dict, dict]":
    """Add ``traj_labels`` and ``sign_vec`` columns to a long-form shard DataFrame.

    Fully vectorised (no Python-level row loops) — handles 10M+ rows in a
    few seconds via pandas pivot + string ops.

    Parameters
    ----------
    df:
        Long-form shard DataFrame.  Must have columns: sender, receiver, Path,
        contrast, PDS.
    timepoints:
        Ordered tuple of timepoint labels that appear after the ``_`` separator
        in the ``contrast`` column (e.g. ``("2mo", "4mo", "6mo")`` for Song,
        ``("3mo", "6mo", "9mo", "12mo")`` for 5xFAD).
    valid_diseases:
        Set of disease labels that appear before the ``_`` separator in
        ``contrast`` (e.g. ``{"App", "Tau", "ApTt"}`` for Song, ``{"TG"}`` for
        5xFAD).
    source_label:
        Human-readable tag used in progress prints only.

    Returns
    -------
    df:
        Annotated copy with ``traj_labels`` and ``sign_vec`` columns added.
    recur_index:
        ``{ path_string → [disease, …] }`` — diseases with ≥1 non-flat
        timepoint. Small enough to inline in the payload.
    traj_summary:
        ``{ label → count }`` aggregate across all (path, disease) pairs.
    """
    df = df.copy()
    df["_path_str"] = (
        df["sender"].astype(str) + "||"
        + df["receiver"].astype(str) + "||"
        + df["Path"].astype(str)
    )
    split = df["contrast"].str.split("_", n=1, expand=True)
    df["_disease"] = split[0].fillna("")
    df["_timepoint"] = split[1].fillna("")

    if df.empty:
        df["traj_labels"] = ""
        df["sign_vec"] = ""
        return df, {}, {}

    # ---- 1. Per-row sign char + raw PDS (no flat threshold) -----------------
    pds_col = df["PDS"].astype(float)
    sign_ser = pd.Series("", index=df.index, dtype="str")
    sign_ser.loc[pds_col > 0] = "u"
    sign_ser.loc[pds_col < 0] = "d"
    df["_sign"] = sign_ser
    df["_pds"] = pds_col

    # ---- 2. Pivot: (path, disease) × timepoint → sign & PDS ----------------
    pivot_mask = (
        df["_disease"].isin(valid_diseases)
        & df["_timepoint"].isin(set(timepoints))
        & df["_sign"].isin(["u", "d"])
    )
    sub = df.loc[pivot_mask, ["_path_str", "_disease", "_timepoint", "_sign", "_pds"]]

    if sub.empty:
        print(f"  trajectory ({source_label}): no canonical contrasts; skipping",
              flush=True)
        df["traj_labels"] = ""
        df["sign_vec"] = ""
        return df, {}, {}

    sign_pivot = sub.pivot_table(
        index=["_path_str", "_disease"],
        columns="_timepoint",
        values="_sign",
        aggfunc="first",
    )
    pds_pivot = sub.pivot_table(
        index=["_path_str", "_disease"],
        columns="_timepoint",
        values="_pds",
        aggfunc="first",
    )
    for tp in timepoints:
        if tp not in sign_pivot.columns:
            sign_pivot[tp] = pd.NA
        if tp not in pds_pivot.columns:
            pds_pivot[tp] = pd.NA
    sign_pivot = sign_pivot[list(timepoints)]
    pds_pivot = pds_pivot[list(timepoints)]
    complete_mask = sign_pivot.notna().all(axis=1) & pds_pivot.notna().all(axis=1)
    sign_pivot = sign_pivot.loc[complete_mask]
    pds_pivot = pds_pivot.loc[complete_mask]

    if sign_pivot.empty:
        df["traj_labels"] = ""
        df["sign_vec"] = ""
        return df, {}, {}

    # ---- 3. Vectorised label derivation (non-exclusive) ---------------------
    out = pd.DataFrame(index=sign_pivot.index)
    # sign_vec: concatenate sign chars across timepoints in order.
    out["sign_vec"] = sign_pivot[list(timepoints)].apply(
        lambda row: "".join(row.astype(str)), axis=1
    )
    out["always_up"] = (sign_pivot == "u").all(axis=1)
    out["always_down"] = (sign_pivot == "d").all(axis=1)
    # Monotonic: each consecutive pair of PDS values must be strictly ordered.
    pds_arr = pds_pivot[list(timepoints)]
    out["monotonic_up"] = (
        pds_arr.diff(axis=1).iloc[:, 1:] > 0
    ).all(axis=1)
    out["monotonic_down"] = (
        pds_arr.diff(axis=1).iloc[:, 1:] < 0
    ).all(axis=1)
    # "mixed" = sign changes (sign_vec uses both 'u' and 'd').
    out["mixed"] = ~(out["always_up"] | out["always_down"])

    def _join_labels(row):
        names = []
        if row["always_up"]:
            names.append("always-up")
        if row["always_down"]:
            names.append("always-down")
        if row["monotonic_up"]:
            names.append("monotonic-up")
        if row["monotonic_down"]:
            names.append("monotonic-down")
        if row["mixed"]:
            names.append("mixed")
        return ";".join(names)

    out["traj_labels"] = out.apply(_join_labels, axis=1)

    # ---- 4. Back-join onto every shard row ----------------------------------
    traj_map = out[["sign_vec", "traj_labels"]].reset_index()
    df = df.merge(traj_map, on=["_path_str", "_disease"], how="left")
    df["traj_labels"] = df["traj_labels"].fillna("")
    df["sign_vec"] = df["sign_vec"].fillna("")

    # ---- 5. recur_index — path → list of diseases with complete trajectory --
    sig_pivot = out.reset_index()[["_path_str", "_disease"]]
    recur_index: dict = {}
    if len(sig_pivot):
        recur_series = sig_pivot.groupby("_path_str", sort=False)["_disease"].agg(list)
        recur_index = {str(pid): dis for pid, dis in recur_series.items()}

    # ---- 6. Trajectory summary (per-label counts across (path, disease)) ----
    traj_summary: dict = {lbl: int(out[lbl.replace("-", "_")].sum())
                          for lbl in _SIGN_VEC_LABELS}

    n_paths = len(out.index.get_level_values("_path_str").unique())
    print(f"  trajectory ({source_label}): {n_paths:,} unique paths annotated; "
          f"{len(recur_index):,} recur in ≥1 disease; "
          f"label dist = {dict(sorted(traj_summary.items()))}", flush=True)

    df.drop(columns=["_path_str", "_disease", "_timepoint", "_sign", "_pds"],
            inplace=True, errors="ignore")
    return df, recur_index, traj_summary

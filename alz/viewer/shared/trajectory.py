"""Shared trajectory annotation helper for viewer cohort builders.

Viewer cohorts classify a trajectory by pivoting a path/series combination
over an ordered axis.  The classifier is cohort-agnostic; each builder supplies
the series and axis extractors for its contrast vocabulary.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import pandas as pd

from alz.viewer.shared.incytr_index import _SIGN_VEC_LABELS

if TYPE_CHECKING:
    pass


def annotate_trajectory_columns(
    df: "pd.DataFrame",
    timepoints: "tuple[str, ...] | None" = None,
    valid_diseases: "set[str] | None" = None,
    source_label: str = "pair_mode",
    *,
    series_key: "Callable[[Any], Any]",
    axis_value: "Callable[[Any], Any]",
    ordered_axis: "tuple[str, ...] | None" = None,
    valid_series: "set[str] | None" = None,
) -> "tuple[pd.DataFrame, dict, dict]":
    """Add ``traj_labels`` and ``sign_vec`` columns to a long-form shard DataFrame.

    Fully vectorised (no Python-level row loops) — handles 10M+ rows in a
    few seconds via pandas pivot + string ops.

    Parameters
    ----------
    df:
        Long-form shard DataFrame.  Must have columns: sender, receiver, Path,
        contrast, PDS.
    timepoints / valid_diseases:
        Backwards-compatible names for ``ordered_axis`` / ``valid_series``.
        They preserve the existing cohort-wrapper call shape; the extractors
        define how those values are obtained from the input rows.
    series_key:
        Extracts the trajectory series for each row.  The callable may accept
        the complete DataFrame and return a same-index Series, or accept one
        row and return a scalar.
    axis_value:
        Extracts the ordered-axis value for each row, with the same callable
        conventions as ``series_key``.
    ordered_axis:
        Complete ordered axis.  A trajectory is labelled only when it has a
        real PDS at every value in this tuple.
    valid_series:
        Series values admitted to the trajectory pivot.
    source_label:
        Human-readable tag used in progress prints only.

    Returns
    -------
    df:
        Annotated copy with ``traj_labels`` and ``sign_vec`` columns added.
    recur_index:
        ``{ path_string → [series, …] }`` — series with ≥1 non-flat
        timepoint. Small enough to inline in the payload.
    traj_summary:
        ``{ label → count }`` aggregate across all (path, series) pairs.
    """
    if ordered_axis is None:
        ordered_axis = tuple(timepoints or ())
    if valid_series is None:
        valid_series = set(valid_diseases or ())

    def _extract(mapping: "Callable[[Any], Any]", name: str) -> "pd.Series":
        """Evaluate a vectorized extractor, or a row extractor for callers."""
        try:
            values = mapping(df)
        except (AttributeError, IndexError, KeyError, TypeError):
            values = df.apply(mapping, axis=1)
        if isinstance(values, pd.Series):
            return values.reindex(df.index)
        if pd.api.types.is_scalar(values):
            return pd.Series(values, index=df.index)
        try:
            return pd.Series(values, index=df.index)
        except ValueError as exc:
            raise ValueError(
                f"trajectory {name} extractor must return a scalar or one value per row"
            ) from exc

    df = df.copy()
    df["_path_str"] = (
        df["sender"].astype(str) + "||"
        + df["receiver"].astype(str) + "||"
        + df["Path"].astype(str)
    )
    df["_series"] = _extract(series_key, "series_key").fillna("").astype(str)
    df["_axis"] = _extract(axis_value, "axis_value").fillna("").astype(str)

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

    # ---- 2. Pivot: (path, series) × axis → sign & PDS -----------------------
    pivot_mask = (
        df["_series"].isin(valid_series)
        & df["_axis"].isin(set(ordered_axis))
        & df["_sign"].isin(["u", "d"])
    )
    sub = df.loc[pivot_mask, ["_path_str", "_series", "_axis", "_sign", "_pds"]]

    if sub.empty:
        print(f"  trajectory ({source_label}): no canonical contrasts; skipping",
              flush=True)
        df["traj_labels"] = ""
        df["sign_vec"] = ""
        return df, {}, {}

    sign_pivot = sub.pivot_table(
        index=["_path_str", "_series"],
        columns="_axis",
        values="_sign",
        aggfunc="first",
    )
    pds_pivot = sub.pivot_table(
        index=["_path_str", "_series"],
        columns="_axis",
        values="_pds",
        aggfunc="first",
    )
    for axis in ordered_axis:
        if axis not in sign_pivot.columns:
            sign_pivot[axis] = pd.NA
        if axis not in pds_pivot.columns:
            pds_pivot[axis] = pd.NA
    sign_pivot = sign_pivot[list(ordered_axis)]
    pds_pivot = pds_pivot[list(ordered_axis)]
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
    out["sign_vec"] = sign_pivot[list(ordered_axis)].apply(
        lambda row: "".join(row.astype(str)), axis=1
    )
    out["always_up"] = (sign_pivot == "u").all(axis=1)
    out["always_down"] = (sign_pivot == "d").all(axis=1)
    # Monotonic: each consecutive pair of PDS values must be strictly ordered.
    pds_arr = pds_pivot[list(ordered_axis)]
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
    df = df.merge(traj_map, on=["_path_str", "_series"], how="left")
    df["traj_labels"] = df["traj_labels"].fillna("")
    df["sign_vec"] = df["sign_vec"].fillna("")

    # ---- 5. recur_index — path → list of series with complete trajectory ---
    sig_pivot = out.reset_index()[["_path_str", "_series"]]
    recur_index: dict = {}
    if len(sig_pivot):
        recur_series = sig_pivot.groupby("_path_str", sort=False)["_series"].agg(list)
        recur_index = {str(pid): series for pid, series in recur_series.items()}

    # ---- 6. Trajectory summary (per-label counts across (path, series)) ----
    traj_summary: dict = {lbl: int(out[lbl.replace("-", "_")].sum())
                          for lbl in _SIGN_VEC_LABELS}

    n_paths = len(out.index.get_level_values("_path_str").unique())
    print(f"  trajectory ({source_label}): {n_paths:,} unique paths annotated; "
          f"{len(recur_index):,} recur in ≥1 series; "
          f"label dist = {dict(sorted(traj_summary.items()))}", flush=True)

    df.drop(columns=["_path_str", "_series", "_axis", "_sign", "_pds"],
            inplace=True, errors="ignore")
    return df, recur_index, traj_summary

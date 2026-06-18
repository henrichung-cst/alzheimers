from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from alz.shared import config


def _normalize_context_cols(context_cols: list[str]) -> list[str]:
    """Preserve order while removing duplicate context columns."""
    return list(dict.fromkeys(context_cols))


def _add_constant_cols(df: pd.DataFrame, constants: dict[str, str]) -> pd.DataFrame:
    """Add constant-valued columns to every row."""
    out = df.copy()
    for name, value in constants.items():
        out[name] = value
    return out


def _prepare_side(df: pd.DataFrame, context_cols: list[str], *, prefix: str) -> pd.DataFrame:
    """Return only pairing keys and renamed evidence columns for one modality."""
    required_cols = context_cols + ["kinase", "NES", "FDR"]
    selected = df[required_cols].copy()
    return selected.rename(columns={"NES": f"{prefix}_NES", "FDR": f"{prefix}_FDR"})


def _merge_pairing_keys(
    stoich_df: pd.DataFrame,
    raw_df: pd.DataFrame,
    *,
    key_cols: list[str],
) -> pd.DataFrame:
    """Return one row per unique pairing key from both inputs."""
    return (
        pd.concat([stoich_df[key_cols], raw_df[key_cols]], ignore_index=True)
        .drop_duplicates(ignore_index=True)
    )


def _evaluate_mechanism(
    *,
    stoich_nes: float | np.float64 | np.integer | None,
    raw_nes: float | np.float64 | np.integer | None,
    stoich_significant: bool,
    raw_significant: bool,
) -> tuple[str, str]:
    """Return `(sign_relation, mechanism_call)` for one evaluable pair."""
    if stoich_significant and raw_significant:
        if np.sign(stoich_nes) == np.sign(raw_nes):
            return "same", "both"
        return "opposite", "discordant"
    if stoich_significant:
        return "stoich_only", "activity_driven"
    if raw_significant:
        return "raw_only", "abundance_driven"
    return "none", "not_significant"


def classify_mechanisms(
    stoich_df: pd.DataFrame,
    raw_df: pd.DataFrame,
    *,
    context_cols: list[str],
    fdr_thresh: float = config.MEA_FDR_THRESH,
) -> pd.DataFrame:
    """Classify stoich-vs-raw mechanisms for matched `(context, kinase)` pairs."""

    key_cols = list(context_cols) + ["kinase"]
    stoich_prepared = _prepare_side(stoich_df, context_cols=context_cols, prefix="stoich")
    raw_prepared = _prepare_side(raw_df, context_cols=context_cols, prefix="raw")

    stoich_count = (
        stoich_prepared.groupby(key_cols, dropna=False)
        .size()
        .rename("stoich_count")
        .reset_index()
    )
    raw_count = (
        raw_prepared.groupby(key_cols, dropna=False)
        .size()
        .rename("raw_count")
        .reset_index()
    )

    stoich_uniques = stoich_prepared.merge(stoich_count, on=key_cols, how="left")
    stoich_uniques = stoich_uniques[stoich_uniques["stoich_count"] == 1].drop(
        columns=["stoich_count"]
    )

    raw_uniques = raw_prepared.merge(raw_count, on=key_cols, how="left")
    raw_uniques = raw_uniques[raw_uniques["raw_count"] == 1].drop(columns=["raw_count"])

    key_rows = _merge_pairing_keys(stoich_prepared, raw_prepared, key_cols=key_cols)
    out = key_rows.merge(stoich_uniques, on=key_cols, how="left")
    out = out.merge(raw_uniques, on=key_cols, how="left")
    out = out.merge(stoich_count, on=key_cols, how="left")
    out = out.merge(raw_count, on=key_cols, how="left")

    out["stoich_count"] = out["stoich_count"].fillna(0).astype(int)
    out["raw_count"] = out["raw_count"].fillna(0).astype(int)

    stoich_present = out["stoich_count"] > 0
    raw_present = out["raw_count"] > 0

    out["stoich_NES_num"] = pd.to_numeric(out["stoich_NES"], errors="coerce")
    out["stoich_FDR_num"] = pd.to_numeric(out["stoich_FDR"], errors="coerce")
    out["raw_NES_num"] = pd.to_numeric(out["raw_NES"], errors="coerce")
    out["raw_FDR_num"] = pd.to_numeric(out["raw_FDR"], errors="coerce")

    stoich_evaluable = stoich_present & out["stoich_NES_num"].notna() & out["stoich_FDR_num"].notna()
    raw_evaluable = raw_present & out["raw_NES_num"].notna() & out["raw_FDR_num"].notna()

    out["stoich_significant"] = False
    out["raw_significant"] = False
    out.loc[stoich_evaluable, "stoich_significant"] = (
        out.loc[stoich_evaluable, "stoich_FDR_num"] < fdr_thresh
    )
    out.loc[raw_evaluable, "raw_significant"] = (
        out.loc[raw_evaluable, "raw_FDR_num"] < fdr_thresh
    )

    duplicate_pair_rows = (out["stoich_count"] > 1) | (out["raw_count"] > 1)
    out["skip_reason"] = None
    out.loc[duplicate_pair_rows, "skip_reason"] = "duplicate_pair_rows"
    out.loc[(~duplicate_pair_rows) & (~stoich_present), "skip_reason"] = "missing_stoich_row"
    out.loc[(~duplicate_pair_rows) & (stoich_present) & (~raw_present), "skip_reason"] = "missing_raw_row"
    malformed = (stoich_present & ~stoich_evaluable) | (raw_present & ~raw_evaluable)
    out.loc[
        (~duplicate_pair_rows) & stoich_present & raw_present & malformed,
        "skip_reason",
    ] = "invalid_numeric_values"

    evaluable = out["skip_reason"].isna()
    out["sign_relation"] = "not_evaluable"
    out["mechanism_call"] = "not_evaluable"

    relation_calls = [
        _evaluate_mechanism(
            stoich_nes=row["stoich_NES_num"],
            raw_nes=row["raw_NES_num"],
            stoich_significant=row["stoich_significant"],
            raw_significant=row["raw_significant"],
        )
        for _, row in out[evaluable].iterrows()
    ]

    if relation_calls:
        sign_values = [r[0] for r in relation_calls]
        call_values = [r[1] for r in relation_calls]
        out.loc[evaluable, "sign_relation"] = sign_values
        out.loc[evaluable, "mechanism_call"] = call_values

    columns = key_cols + [
        "stoich_NES",
        "stoich_FDR",
        "raw_NES",
        "raw_FDR",
        "stoich_significant",
        "raw_significant",
        "sign_relation",
        "mechanism_call",
        "skip_reason",
    ]
    return out[columns].sort_values(by=key_cols).reset_index(drop=True).copy()


def classify_mechanism_files(
    stoich_path: str | Path,
    raw_path: str | Path,
    out_path: str | Path,
    *,
    context_cols: list[str],
    cohort: str | None = None,
    extra_constant_cols: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Read MEA CSV inputs, classify by helper, and write one CSV output."""
    stoich_df = pd.read_csv(stoich_path)
    raw_df = pd.read_csv(raw_path)

    extra_constant_cols = extra_constant_cols or {}
    effective_context_cols = _normalize_context_cols(
        context_cols + list(extra_constant_cols.keys())
    )
    if cohort is not None and "cohort" not in effective_context_cols:
        effective_context_cols.append("cohort")

    if extra_constant_cols:
        stoich_df = _add_constant_cols(stoich_df, extra_constant_cols)
        raw_df = _add_constant_cols(raw_df, extra_constant_cols)
    if cohort is not None:
        cohort_constant = {"cohort": cohort}
        stoich_df = _add_constant_cols(stoich_df, cohort_constant)
        raw_df = _add_constant_cols(raw_df, cohort_constant)

    out = classify_mechanisms(stoich_df, raw_df, context_cols=effective_context_cols)
    out.to_csv(out_path, index=False)
    return out


def _parse_constant_arg(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            f"Invalid --constant value {value!r}; expected KEY=VALUE."
        )
    key, val = value.split("=", 1)
    if not key:
        raise argparse.ArgumentTypeError("Constant key cannot be empty.")
    return key, val


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for file-level mechanism attribution."""
    parser = argparse.ArgumentParser(description="Classify paired stoich/raw MEA CSV files.")
    parser.add_argument("--stoich", required=True, help="Input stoichiometry MEA CSV.")
    parser.add_argument("--raw", required=True, help="Input raw phospho MEA CSV.")
    parser.add_argument("--out", required=True, help="Output classified MEA CSV.")
    parser.add_argument(
        "--context",
        action="append",
        default=[],
        metavar="COL",
        help="Context column to join on. Repeatable.",
    )
    parser.add_argument("--cohort", default=None, help="Optional cohort label.")
    parser.add_argument(
        "--constant",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        type=_parse_constant_arg,
        help="Additional constant context in KEY=VALUE form. Repeatable.",
    )
    args = parser.parse_args(argv)

    context_cols = _normalize_context_cols(args.context)
    extra_constant_cols = dict(args.constant) if args.constant else None

    classify_mechanism_files(
        stoich_path=args.stoich,
        raw_path=args.raw,
        out_path=args.out,
        context_cols=context_cols,
        cohort=args.cohort,
        extra_constant_cols=extra_constant_cols,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

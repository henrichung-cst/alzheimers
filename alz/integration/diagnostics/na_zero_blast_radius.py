"""Quantify blast radius of the score_factorial_paths NA->0 bug
(see ~/Projects/work/incytr/docs/incytr_proposals/score_factorial_paths_na_to_zero_bug.md).

For the current 5xfad_kldata output, count:

  1. (animal, receiver) cells with 0 cells in the input metadata —
     these are the only cells that can produce per-animal SigProb=NA
     at min_cells=1 (the streaming engine's default).
  2. For each receiver, how many paths route through a sender whose
     (sender, animal) cell has < 1 cell. Those are the paths whose
     per-animal SigProb gets NA'd at that animal — and then zeroed by
     R/factorial.R:1900.
  3. Per-condition columns in the parquet that ended up exactly 0.0
     or NaN — the visible footprint of the bug.
  4. Of the current "high-|TPDS|" paths (the analysis-relevant set),
     what fraction touch at least one zeroed/NaN per-animal cell.

This runs against the existing parquet output + the upstream expression
metadata; no rerun. With min_cells=1 (the default in
score_factorial_paths), only animal x cell-type combos with literally
zero barcodes get NA-then-zeroed. If min_cells were raised, more cells
would be affected.

Outputs:
  outputs/reports/incytr_factorial_5xfad_kldata/diagnostics/
    na_zero_blast_radius.csv
    na_zero_blast_radius_per_receiver.csv
"""

from __future__ import annotations

import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

REPO_ROOT = "/home/hchung/Projects/work/alzheimers"
INPUT_DIR = f"{REPO_ROOT}/data/incytr_factorial_inputs"
OUTPUT_ROOT = f"{REPO_ROOT}/outputs/reports/incytr_factorial_5xfad_kldata"
RECEIVER_CACHE = f"{OUTPUT_ROOT}/receiver_cache"
CLUSTER_MAP = f"{REPO_ROOT}/data/incytr/v2_46clusters/barcode_to_cluster.csv"
OUT_DIR = f"{OUTPUT_ROOT}/diagnostics"
MIN_CELLS = 1  # score_factorial_paths default
TPDS_HI = 0.5


def build_animal_cluster_counts() -> pd.DataFrame:
    """Return (animal_id, cluster) -> n_cells frame."""
    meta = pd.read_csv(f"{INPUT_DIR}/expression_metadata.csv")
    cm = pd.read_csv(CLUSTER_MAP)
    if "barcode" in meta.columns:
        meta = meta.merge(cm[["barcode", "cluster_subclass"]],
                          on="barcode", how="left")
    else:
        bcs = pd.read_csv(f"{INPUT_DIR}/expression_barcodes.csv",
                          header=None)[0].tolist()
        if bcs and str(bcs[0]).lower() in ("barcode", "0"):
            bcs = bcs[1:]
        meta["barcode"] = bcs
        meta = meta.merge(cm[["barcode", "cluster_subclass"]],
                          on="barcode", how="left")
    counts = (meta.groupby(["animal_id", "cluster_subclass"])
                  .size().rename("n_cells").reset_index())
    return counts


def load_pair_parquet(p: str) -> pd.DataFrame:
    return pq.ParquetFile(p).read().to_pandas()


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    counts = build_animal_cluster_counts()
    animals = sorted(counts["animal_id"].unique())
    clusters = sorted(counts["cluster_subclass"].dropna().unique())
    grid = pd.MultiIndex.from_product([animals, clusters],
                                       names=["animal_id", "cluster_subclass"]).to_frame(index=False)
    grid = grid.merge(counts, on=["animal_id", "cluster_subclass"], how="left")
    grid["n_cells"] = grid["n_cells"].fillna(0).astype(int)
    grid["below_min"] = grid["n_cells"] < MIN_CELLS
    print(f"=== (animal, cluster) grid: {len(grid):,} cells; "
          f"{int(grid['below_min'].sum()):,} below min_cells={MIN_CELLS} ===")
    print("Animal x cluster combos with 0 cells:")
    zero = grid[grid["below_min"]]
    if len(zero) == 0:
        print("  (none — every animal × cluster has at least 1 cell)")
    else:
        print(zero.to_string(index=False))
    grid.to_csv(f"{OUT_DIR}/na_zero_blast_radius_animal_cluster.csv", index=False)

    anim_meta = pd.read_csv(f"{INPUT_DIR}/animal_metadata.csv")
    anim_to_cond = dict(zip(anim_meta["animal_id"],
                            anim_meta["genotype"] + "_" + anim_meta["timepoint"]))
    cond_counts = anim_meta.groupby(["genotype", "timepoint"]).size().rename("n").reset_index()
    cond_counts["condition"] = cond_counts["genotype"] + "_" + cond_counts["timepoint"]
    n1_conds = set(cond_counts.loc[cond_counts["n"] == 1, "condition"])
    print(f"\n=== Conditions with n_animals==1 (most affected by NA->0): {sorted(n1_conds)}")

    receivers = sorted(d.name.removeprefix("receiver=")
                       for d in Path(RECEIVER_CACHE).iterdir()
                       if d.is_dir() and d.name.startswith("receiver="))
    print(f"\n=== Sweeping {len(receivers)} receivers ===")

    rows = []
    for rcv in receivers:
        rcv_dir = f"{RECEIVER_CACHE}/receiver={rcv}"
        parts = sorted(glob.glob(f"{rcv_dir}/*.parquet"))
        if not parts:
            continue
        frames = []
        for p in parts:
            d = load_pair_parquet(p)
            if "receiver" in d.columns:
                d = d.drop(columns=["receiver"])
            frames.append(d)
        df = pd.concat(frames, ignore_index=True)

        sigprob_cols = [c for c in df.columns if c.startswith("SigProb_")
                        and c not in ("SigProb_ref", "SigProb_alt")
                        and "_ref" not in c and "_alt" not in c]
        sigprob_cols = [c for c in sigprob_cols
                        if c.removeprefix("SigProb_") in set(anim_meta["genotype"]
                                                              .str.cat(anim_meta["timepoint"], sep="_"))]
        for col in sigprob_cols:
            cond = col.removeprefix("SigProb_")
            sub = df[df.get("contrast", cond) == df["contrast"].iloc[0]]
            vals = df[col]
            n_total = len(vals)
            n_zero = int((vals == 0.0).sum())
            n_nan = int(vals.isna().sum())
            rows.append({
                "receiver": rcv,
                "condition": cond,
                "n_in_n1_cond": cond in n1_conds,
                "n_paths": n_total,
                "n_zero": n_zero,
                "n_nan": n_nan,
                "frac_zero_or_nan": (n_zero + n_nan) / max(n_total, 1),
            })

        hi = df[df["TPDS"].abs() >= TPDS_HI]
        for cname, hi_sub in hi.groupby("contrast"):
            n = len(hi_sub)
            ref_col = f"SigProb_{cname.split('_')[0]}_{cname.split('_')[1]}"  # noop
            zero_or_nan_any = 0
            for col in sigprob_cols:
                col_vals = hi_sub[col]
                zero_or_nan_any += int(((col_vals == 0.0) | col_vals.isna()).sum())
            rows.append({
                "receiver": rcv,
                "condition": f"HI_TPDS@{cname}",
                "n_in_n1_cond": False,
                "n_paths": n,
                "n_zero": -1,
                "n_nan": -1,
                "frac_zero_or_nan": zero_or_nan_any / max(n * len(sigprob_cols), 1),
            })

        print(f"  {rcv:48s} n_paths={len(df):>7,}  hi_tpds_any={len(hi):>6,}")

    out = pd.DataFrame(rows)
    out.to_csv(f"{OUT_DIR}/na_zero_blast_radius_per_receiver.csv", index=False)

    print("\n=== Summary: fraction of per-condition columns that are exactly 0 or NaN ===")
    summary = (out[out["n_zero"] >= 0]
               .groupby(["condition", "n_in_n1_cond"])
               .agg(mean_frac_zero_or_nan=("frac_zero_or_nan", "mean"),
                    total_paths=("n_paths", "sum"),
                    total_zero=("n_zero", "sum"),
                    total_nan=("n_nan", "sum"))
               .reset_index()
               .sort_values(["n_in_n1_cond", "mean_frac_zero_or_nan"],
                            ascending=[False, False]))
    print(summary.to_string(index=False))
    summary.to_csv(f"{OUT_DIR}/na_zero_blast_radius_summary.csv", index=False)

    hi_summary = (out[out["condition"].str.startswith("HI_TPDS@")]
                  .groupby("condition")
                  .agg(mean_frac_zero_or_nan=("frac_zero_or_nan", "mean"),
                       total_paths=("n_paths", "sum"))
                  .reset_index()
                  .sort_values("mean_frac_zero_or_nan", ascending=False))
    print("\n=== High-|TPDS| (>=0.5) paths: avg fraction of their per-cond cells = 0/NaN ===")
    print(hi_summary.to_string(index=False))
    hi_summary.to_csv(f"{OUT_DIR}/na_zero_blast_radius_hi_tpds_summary.csv", index=False)

    print(f"\nWrote diagnostics to {OUT_DIR}/")


if __name__ == "__main__":
    main()

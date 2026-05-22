"""Export levy_t5 per-cluster decomposition matrices to the pair-mode wide-CSV
format consumed by the Incytr R driver.

Inputs (long-form, per-animal):
  outputs/reports/decomposition/levy_t5/protein_per_cluster.parquet
  outputs/reports/decomposition/levy_t5/phospho_per_cluster.parquet     (IMAC pS/pT)
  outputs/reports/decomposition/levy_t5/phospho_per_cluster_pY.parquet  (pY)

Outputs (wide, males-only, group-medianed):
  data/derived/incytr_inputs/pr_yuyu_deconvoluted.csv
  data/derived/incytr_inputs/ps_yuyu_deconvoluted.csv
  data/derived/incytr_inputs/py_yuyu_deconvoluted.csv

Aggregation:
- Filter animal_id to males only (sex token `M` in `<n>_<lab>_<sex>_<age>_<geno>`).
- Map genotype tokens to legacy condition vocab:
    WT -> WTyp, APP -> AppP, T22 -> Ttau, T22/APP -> ApTt
- Median across the 3 male animals per (age, geno) group.
- Wide schema: row keys + columns `<condition>_<cluster>` where
  condition = `ma_<age>_<geno>` (12 conditions x 31 clusters = 372 value columns).

Schema matches what alz/incytr_pair/incytr_commandline.R expects:
- pr: row key `Gene Symbol` (driver hard-codes the spaced form). We also
      keep `gene_symbol` for debuggability.
- ps: row keys `site_id`, `gene_symbol`.
- py: row keys `site_id`, `gene_symbol`.

The driver later collapses multiple rows per `gene_symbol` via mean — so we
do not have to dedupe here.
"""
from __future__ import annotations

import argparse
import os
import sys

import pandas as pd
import pyarrow.parquet as pq

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO_ROOT, "alz"))
import config  # noqa: E402

DEC_DIR = os.path.join(REPO_ROOT, "outputs", "reports", "decomposition", "levy_t5")
OUT_DIR = os.path.join(REPO_ROOT, "data", "derived", "incytr_inputs")


def _parse_animal(animal_id: str) -> tuple[str, str, str] | None:
    parsed = config.parse_animal_id(animal_id)
    if parsed is None:
        return None
    return parsed["sex"], parsed["timepoint"], parsed["genotype"]


def _animal_to_condition(animal_id: str) -> str | None:
    parsed = _parse_animal(animal_id)
    if parsed is None or parsed[0] != "M":
        return None
    _, age, geno = parsed
    return f"ma_{age}_{geno}"


def _pivot(df: pd.DataFrame, key_cols: list[str], value_col: str = "value") -> pd.DataFrame:
    """Pivot long → wide on (key_cols, condition × cluster) using median across animals."""
    df = df.copy()
    df["condition"] = df["animal_id"].map(_animal_to_condition)
    df = df[df["condition"].notna()]
    df["col"] = df["condition"] + "_" + df["cluster"].astype(str)
    # Median across the 3 males per (age, geno) group.
    agg = (
        df.groupby(key_cols + ["col"], sort=False, observed=True)[value_col]
        .median()
        .reset_index()
    )
    wide = agg.pivot(index=key_cols, columns="col", values=value_col).reset_index()
    wide.columns.name = None
    return wide


def export_protein() -> str:
    src = os.path.join(DEC_DIR, "protein_per_cluster.parquet")
    print(f"[export-pair] reading {src}")
    df = pq.ParquetFile(src).read(
        columns=["gene_symbol", "animal_id", "cluster", "value"]
    ).to_pandas()
    print(f"[export-pair]   rows={len(df):,}, animals={df['animal_id'].nunique()}")
    wide = _pivot(df, key_cols=["gene_symbol"])
    # Driver hard-codes `pr$\`Gene Symbol\`` (with space + capital).
    wide.insert(1, "Gene Symbol", wide["gene_symbol"])
    out = os.path.join(OUT_DIR, "pr_yuyu_deconvoluted.csv")
    print(f"[export-pair]   writing {out}  shape={wide.shape}")
    wide.to_csv(out, index=False)
    return out


def export_phospho(src_name: str, out_name: str) -> str:
    src = os.path.join(DEC_DIR, src_name)
    print(f"[export-pair] reading {src}")
    df = pq.ParquetFile(src).read(
        columns=["site_id", "gene_symbol", "animal_id", "cluster", "value"]
    ).to_pandas()
    print(f"[export-pair]   rows={len(df):,}, animals={df['animal_id'].nunique()}, "
          f"sites={df['site_id'].nunique():,}")
    wide = _pivot(df, key_cols=["site_id", "gene_symbol"])
    out = os.path.join(OUT_DIR, out_name)
    print(f"[export-pair]   writing {out}  shape={wide.shape}")
    wide.to_csv(out, index=False)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--track", choices=["pr", "ps", "py", "all"], default="all")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    if args.track in ("pr", "all"):
        export_protein()
    if args.track in ("ps", "all"):
        export_phospho("phospho_per_cluster.parquet", "ps_yuyu_deconvoluted.csv")
    if args.track in ("py", "all"):
        export_phospho("phospho_per_cluster_pY.parquet", "py_yuyu_deconvoluted.csv")

    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Audit Song/AD cell-count sparsity versus Incytr pathway burden.

This is a small interpretability diagnostic. It does not alter canonical
Incytr outputs; it writes a CSV and scatterplot under
outputs/reports/incytr_pair_mode/cell_count_qc/.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_COUNTS = REPO_ROOT / "outputs/reports/snrna_integration/pseudobulk_cell_counts.csv"
DEFAULT_INCYTR_DIR = REPO_ROOT / "outputs/reports/incytr_pair_mode/wide"
DEFAULT_OUTDIR = REPO_ROOT / "outputs/reports/incytr_pair_mode/cell_count_qc"
LOW_SIGNAL_MEDIAN_N = 3


def _sql_path(path: Path) -> str:
    return str(path).replace("'", "''")


def _read_cell_counts(path: Path) -> pd.DataFrame:
    counts = pd.read_csv(path)
    required = {"sample", "cell_type", "n_cells"}
    missing = required - set(counts.columns)
    if missing:
        raise ValueError(f"{path} missing required columns: {sorted(missing)}")
    counts = counts[counts["sample"].astype(str).str.contains("_ma_", regex=False)].copy()
    counts["n_cells"] = pd.to_numeric(counts["n_cells"], errors="coerce")
    counts = counts.dropna(subset=["cell_type", "n_cells"])
    stats = counts.groupby("cell_type", sort=False)["n_cells"].agg(
        median_n="median",
        mean_n="mean",
        min_n="min",
        total_n="sum",
        n_samples="count",
    )
    return stats.reset_index()


def _read_pathway_counts(incytr_dir: Path) -> pd.DataFrame:
    parquet_glob = incytr_dir / "*_incytr_output.parquet"
    if not list(incytr_dir.glob("*_incytr_output.parquet")):
        raise FileNotFoundError(f"no Incytr parquets found under {incytr_dir}")
    con = duckdb.connect()
    con.execute("PRAGMA threads=8")
    con.execute("PRAGMA memory_limit='12GB'")
    src = f"read_parquet('{_sql_path(parquet_glob)}')"
    receiver = con.execute(f"""
        SELECT
          "Receiver.group" AS cell_type,
          COUNT(*)::UBIGINT AS receiver_paths_all,
          SUM(CASE WHEN ABS(PDS) > 1 THEN 1 ELSE 0 END)::UBIGINT
            AS receiver_paths_abs_pds_gt1
        FROM {src}
        GROUP BY 1
    """).fetchdf()
    sender = con.execute(f"""
        SELECT
          "Sender.group" AS cell_type,
          COUNT(*)::UBIGINT AS sender_paths_all,
          SUM(CASE WHEN ABS(PDS) > 1 THEN 1 ELSE 0 END)::UBIGINT
            AS sender_paths_abs_pds_gt1
        FROM {src}
        GROUP BY 1
    """).fetchdf()
    con.close()
    out = receiver.merge(sender, on="cell_type", how="outer").fillna(0)
    out["endpoint_paths_abs_pds_gt1"] = (
        out["receiver_paths_abs_pds_gt1"] + out["sender_paths_abs_pds_gt1"]
    )
    return out


def _write_scatter(df: pd.DataFrame, outpath: Path) -> None:
    plot_df = df[df["median_n"].notna()].copy()
    plot_df["is_low_signal"] = plot_df["median_n"] <= LOW_SIGNAL_MEDIAN_N
    fig, ax = plt.subplots(figsize=(8, 5.8))
    colors = plot_df["is_low_signal"].map({True: "#b42318", False: "#2f5f9f"})
    ax.scatter(
        plot_df["median_n"],
        plot_df["receiver_paths_abs_pds_gt1"] + 1,
        c=colors,
        alpha=0.82,
        edgecolor="#222",
        linewidth=0.35,
    )
    ax.axvline(
        LOW_SIGNAL_MEDIAN_N,
        color="#b42318",
        linestyle="--",
        linewidth=1,
        label=f"median n <= {LOW_SIGNAL_MEDIAN_N}",
    )
    ax.set_yscale("log")
    ax.set_xlabel("Median male pseudobulk n_cells")
    ax.set_ylabel("Receiver Incytr pathways with |PDS| > 1 (+1, log scale)")
    ax.set_title("Song/AD Incytr pathway burden is concentrated in sparse receiver cell types")
    label_df = pd.concat([
        plot_df[plot_df["is_low_signal"]],
        plot_df.nlargest(5, "receiver_paths_abs_pds_gt1"),
    ]).drop_duplicates("cell_type")
    for _, row in label_df.iterrows():
        ax.annotate(
            row["cell_type"],
            (row["median_n"], row["receiver_paths_abs_pds_gt1"] + 1),
            xytext=(4, 3),
            textcoords="offset points",
            fontsize=7,
        )
    ax.legend(frameon=False, loc="best")
    ax.grid(True, which="major", color="#d0d7de", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--counts", type=Path, default=DEFAULT_COUNTS)
    ap.add_argument("--incytr-dir", type=Path, default=DEFAULT_INCYTR_DIR)
    ap.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    cell_counts = _read_cell_counts(args.counts)
    pathway_counts = _read_pathway_counts(args.incytr_dir)
    df = cell_counts.merge(pathway_counts, on="cell_type", how="outer")
    count_cols = [c for c in df.columns if c.endswith("_all") or c.endswith("_gt1")]
    for col in count_cols:
        df[col] = df[col].fillna(0).astype("int64")
    df["low_signal_median_le_3"] = df["median_n"] <= LOW_SIGNAL_MEDIAN_N
    df = df.sort_values(
        ["receiver_paths_abs_pds_gt1", "median_n"],
        ascending=[False, True],
        na_position="last",
    )

    csv_path = args.outdir / "cell_count_incytr_pathway_qc.csv"
    png_path = args.outdir / "median_cells_vs_receiver_paths.png"
    df.to_csv(csv_path, index=False)
    _write_scatter(df, png_path)

    corr = df[["median_n", "receiver_paths_abs_pds_gt1"]].dropna()
    spearman = corr["median_n"].corr(corr["receiver_paths_abs_pds_gt1"], method="spearman")
    print(f"wrote {csv_path}")
    print(f"wrote {png_path}")
    print(f"low-signal celltypes: {df['low_signal_median_le_3'].sum()}")
    print(f"spearman(median_n, receiver_paths_abs_pds_gt1) = {spearman:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

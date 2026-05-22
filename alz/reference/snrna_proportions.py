"""Stage 5: per-(gene, cluster, animal) cell-type proportion weights.

Forward-projection prior for Stage 6 (`build_celltype_decomposition.py`).
Computes per-cell-rate weights

    f_percell(gene, cluster, animal) = (expr_c / Σ_c' expr_c') × (N_total / N_c)

on the active spine (default Levy-t5; pass --spine to switch). Per-cell-rate
units are used because Incytr's pathway scoring operates on per-cell
ligand/receptor levels, not cluster mass shares. See
`docs/incytr_deconvolution_pivot.md` for the full decision rationale.

Per-animal where Song snRNA observed (28 animals); group-pooled by
(genotype × timepoint × sex) for the 44 TMT-only animals, with fallback
to (genotype × timepoint) if a stratum has no Song donor.

Outputs under `outputs/reports/decomposition/{spine}/`:
  proportions.parquet               (animal_id, cluster, gene, f_percell)
  proportions_provenance.csv        per-animal source + group_key + n_donors
  coverage_report.csv               per-cluster nonzero-gene counts, drops

Usage:
  pixi run python alz/snrna_proportions.py --run
  pixi run python alz/snrna_proportions.py --run --spine levy_t5
  pixi run python alz/snrna_proportions.py --summary
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from alz.shared import config
from alz.integration.config_integration import load_cluster_spine


def _resolve_spine(name: str) -> list[str]:
    return load_cluster_spine(name)

# Song snRNA sample-ID encoding → TMT metadata vocab.
# Sample IDs look like `{id}_{sex}_{timepoint}_{genotype}`. Genotype is
# already long-form (matches sample_mapping after data_ingest normalizes
# at write time); only sex needs translation (ma/fe → M/F).
SEX_DECODE = {"ma": "M", "fe": "F"}


def _decode_snrna_sample(sample_id: str) -> tuple[str, str, str]:
    """Returns (sex, timepoint, genotype) matching sample_mapping vocab.

    Raises if the sample ID doesn't parse — better to fail loud than to
    silently mis-stratify donors.
    """
    parts = sample_id.split("_")
    if len(parts) < 4:
        raise ValueError(f"Unparseable snRNA sample id: {sample_id!r}")
    sex_raw, timepoint, genotype = parts[-3], parts[-2], parts[-1]
    if sex_raw not in SEX_DECODE or genotype not in config.SAP_FACTORIAL:
        raise ValueError(f"Unknown sex/genotype tokens in {sample_id!r}")
    return SEX_DECODE[sex_raw], timepoint, genotype


def _output_dir(spine: str) -> str:
    return os.path.join("outputs", "reports", "decomposition", spine)


def _load_pseudobulk(spine_clusters: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (expr_df, counts_df).

    expr_df: MultiIndex (sample, cell_type) × gene columns, CPM units.
    counts_df: MultiIndex (sample, cell_type) × column n_cells.
    Filtered to spine_clusters only. Hardfails if any spine cluster is
    absent from the pseudobulk — that means the pseudobulk file predates
    the active spine and needs regenerating.
    """
    expr = pd.read_csv(config.SONG_PSEUDOBULK_FILE)
    pb_classes = set(expr["cell_type"].unique())
    missing = sorted(set(spine_clusters) - pb_classes)
    if missing:
        raise RuntimeError(
            f"{len(missing)} spine clusters missing from pseudobulk: "
            f"{missing}. Re-run `snrna_integration.py --pseudobulk` "
            f"against the active Levy-t5 spine."
        )
    expr = expr[expr["cell_type"].isin(spine_clusters)]
    expr = expr.set_index(["sample", "cell_type"])

    counts = pd.read_csv(config.SONG_CELL_COUNTS_FILE)
    counts = counts[counts["cell_type"].isin(spine_clusters)]
    counts = counts.set_index(["sample", "cell_type"])
    return expr, counts


def _load_sample_mapping() -> pd.DataFrame:
    """Returns per-animal metadata, deduplicated to one row per animal_id.

    Columns retained: animal_id, sex, timepoint, genotype, has_snrna_seq,
    snrna_sample_id.
    """
    df = pd.read_csv(os.path.join(config.DATA_INGEST_OUTPUT_DIR, "sample_mapping.csv"))
    keep = ["animal_id", "sex", "timepoint", "genotype",
            "has_snrna_seq", "snrna_sample_id"]
    df = df[keep].drop_duplicates(subset=["animal_id"]).reset_index(drop=True)
    # boolean coerce — CSV may carry strings
    df["has_snrna_seq"] = df["has_snrna_seq"].astype(str).str.lower() == "true"
    return df


def _compute_animal_weights(
    expr_sub: pd.DataFrame,
    counts_sub: pd.DataFrame,
    spine_clusters: list[str],
) -> pd.DataFrame:
    """Compute f_percell for one animal's pseudobulk.

    Inputs are already subset to a single sample. expr_sub indexed by
    cell_type × gene columns; counts_sub indexed by cell_type, n_cells.

    Returns long DataFrame (cluster, gene, f_percell). Genes with
    Σ_c expr_c == 0 are dropped (no signal anywhere).
    """
    expr_sub = expr_sub.reindex(spine_clusters)
    counts_sub = counts_sub.reindex(spine_clusters)
    if expr_sub.isna().all(axis=1).any() or counts_sub["n_cells"].isna().any():
        # Some clusters absent from this animal's pseudobulk → set to 0
        expr_sub = expr_sub.fillna(0.0)
        counts_sub = counts_sub.fillna(0)

    # Mass-fraction f_c = expr_c / Σ_c' expr_c', per gene
    col_sums = expr_sub.sum(axis=0)  # one value per gene
    keep_genes = col_sums > 0
    expr_sub = expr_sub.loc[:, keep_genes]
    col_sums = col_sums.loc[keep_genes]
    f = expr_sub.div(col_sums, axis=1)  # (cluster × gene)

    # Per-cell-rate weight = f × (N_total / N_c)
    n_c = counts_sub["n_cells"].astype(float)
    n_total = float(n_c.sum())
    if n_total <= 0:
        raise RuntimeError("Animal has zero total cells across spine clusters")
    # Guard: cluster with 0 cells contributes 0 expr anyway, so f row = 0
    size_factor = np.where(n_c > 0, n_total / n_c.replace(0, np.nan), 0.0)
    size_factor = pd.Series(size_factor, index=n_c.index).fillna(0.0)
    f_percell = f.mul(size_factor, axis=0)  # broadcast over genes

    # Internal sanity: column sums of f must be ≈ 1 (excluding the dropped
    # zero-sum genes). Tolerance is tight because this is exact arithmetic.
    f_col_sums = f.sum(axis=0).values
    if not np.allclose(f_col_sums, 1.0, atol=1e-9):
        max_dev = float(np.max(np.abs(f_col_sums - 1.0)))
        raise RuntimeError(f"Mass-fraction identity violated; max deviation {max_dev}")

    # Long form
    long = f_percell.stack().reset_index()
    long.columns = ["cluster", "gene", "f_percell"]
    return long


def _pool_donors_for_stratum(
    donor_sample_ids: list[str],
    expr_df: pd.DataFrame,
    counts_df: pd.DataFrame,
    spine_clusters: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Mean across donor animals of pseudobulk (per cluster × gene) and
    cell counts (per cluster). Returns (mean_expr, mean_counts) indexed
    by cluster for a single virtual donor.
    """
    expr_stack = []
    counts_stack = []
    for sid in donor_sample_ids:
        if sid not in expr_df.index.get_level_values("sample"):
            continue
        e = expr_df.xs(sid, level="sample").reindex(spine_clusters).fillna(0.0)
        c = counts_df.xs(sid, level="sample").reindex(spine_clusters).fillna(0)
        expr_stack.append(e)
        counts_stack.append(c)
    if not expr_stack:
        raise RuntimeError(f"No donor pseudobulks for {donor_sample_ids}")
    mean_expr = sum(expr_stack) / len(expr_stack)
    mean_counts = sum(counts_stack) / len(counts_stack)
    return mean_expr, mean_counts


def step_proportions(spine: str) -> None:
    spine_clusters = _resolve_spine(spine)
    print(f"[proportions] spine={spine} ({len(spine_clusters)} clusters)")

    out_dir = _output_dir(spine)
    os.makedirs(out_dir, exist_ok=True)

    expr_df, counts_df = _load_pseudobulk(spine_clusters)
    sample_meta = _load_sample_mapping()
    print(f"[proportions] pseudobulk: {len(expr_df.index.get_level_values('sample').unique())} "
          f"snRNA samples × {len(spine_clusters)} clusters × {expr_df.shape[1]} genes")
    print(f"[proportions] animals: {len(sample_meta)} total, "
          f"{int(sample_meta['has_snrna_seq'].sum())} Song-observed, "
          f"{int((~sample_meta['has_snrna_seq']).sum())} TMT-only")

    # Donor index built directly from the snRNA pseudobulk's sample roster,
    # NOT from sample_mapping's has_snrna_seq flag. The latter only flags
    # animals present in BOTH TMT and snRNA (28 of 72); the snRNA dataset
    # has additional samples not in the TMT cohort (e.g., E137 covers the
    # M 4mo T22/APP stratum which has no paired TMT animal). Using the
    # broader pool gives every stratum at least one donor.
    snrna_samples = expr_df.index.get_level_values("sample").unique().tolist()
    donors_by_stratum: dict[tuple[str, str, str], list[str]] = {}
    for sid in snrna_samples:
        key = _decode_snrna_sample(sid)
        donors_by_stratum.setdefault(key, []).append(sid)
    missing_strata = [
        (row.sex, row.timepoint, row.genotype)
        for row in sample_meta.itertuples(index=False)
        if not row.has_snrna_seq
        and (row.sex, row.timepoint, row.genotype) not in donors_by_stratum
    ]
    if missing_strata:
        raise RuntimeError(
            f"Strata with no Song donor: {sorted(set(missing_strata))}"
        )

    chunks: list[pd.DataFrame] = []
    provenance_rows: list[dict] = []

    for row in sample_meta.itertuples(index=False):
        if row.has_snrna_seq:
            try:
                long = _compute_animal_weights(
                    expr_df.xs(row.snrna_sample_id, level="sample"),
                    counts_df.xs(row.snrna_sample_id, level="sample"),
                    spine_clusters,
                )
            except KeyError:
                raise RuntimeError(
                    f"Animal {row.animal_id} flagged has_snrna_seq=True but "
                    f"snrna_sample_id {row.snrna_sample_id!r} missing from pseudobulk"
                )
            source, group_key, n_donors = "observed", row.snrna_sample_id, 1
        else:
            stratum = (row.sex, row.timepoint, row.genotype)
            donor_ids = donors_by_stratum[stratum]
            group_key = f"{row.sex}_{row.timepoint}_{row.genotype}"
            source = "imputed"
            mean_expr, mean_counts = _pool_donors_for_stratum(
                donor_ids, expr_df, counts_df, spine_clusters
            )
            long = _compute_animal_weights(mean_expr, mean_counts, spine_clusters)
            n_donors = len(donor_ids)

        long.insert(0, "animal_id", row.animal_id)
        chunks.append(long)
        provenance_rows.append({
            "animal_id": row.animal_id,
            "source": source,
            "group_key": group_key,
            "n_donors_in_group": n_donors,
            "sex": row.sex,
            "timepoint": row.timepoint,
            "genotype": row.genotype,
        })

    proportions = pd.concat(chunks, ignore_index=True)
    proportions["f_percell"] = proportions["f_percell"].astype("float32")

    # Coverage report — per cluster, count nonzero genes and median weight
    coverage = (
        proportions.assign(nonzero=lambda d: d["f_percell"] > 0)
        .groupby("cluster")
        .agg(
            n_nonzero_rows=("nonzero", "sum"),
            n_total_rows=("f_percell", "size"),
            median_f_percell=("f_percell", "median"),
        )
        .reset_index()
    )

    prov_df = pd.DataFrame(provenance_rows)

    out_parquet = os.path.join(out_dir, "proportions.parquet")
    out_prov = os.path.join(out_dir, "proportions_provenance.csv")
    out_cov = os.path.join(out_dir, "coverage_report.csv")
    proportions.to_parquet(out_parquet, index=False)
    prov_df.to_csv(out_prov, index=False)
    coverage.to_csv(out_cov, index=False)

    print(f"[proportions] wrote {out_parquet}  rows={len(proportions):,}")
    print(f"[proportions]   observed: {(prov_df['source']=='observed').sum()} animals")
    print(f"[proportions]   imputed:  {(prov_df['source']=='imputed').sum()} animals")


def summary(spine: str) -> None:
    out_dir = _output_dir(spine)
    parquet = os.path.join(out_dir, "proportions.parquet")
    prov = os.path.join(out_dir, "proportions_provenance.csv")
    if not os.path.exists(parquet):
        print(f"[summary] {parquet} not found — run --run first")
        return
    df = pd.read_parquet(parquet)
    p = pd.read_csv(prov)
    print(f"[summary] proportions rows: {len(df):,}")
    print(f"[summary] animals: {df['animal_id'].nunique()}, "
          f"clusters: {df['cluster'].nunique()}, "
          f"genes: {df['gene'].nunique()}")
    print(f"[summary] provenance breakdown:")
    print(p["source"].value_counts().to_string())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", action="store_true", help="compute proportions")
    ap.add_argument("--summary", action="store_true", help="print cached summary")
    ap.add_argument("--spine", default=config.CLUSTER_SPINE_NAME,
                    help=f"cell-type spine name (default: {config.CLUSTER_SPINE_NAME})")
    args = ap.parse_args()
    if not (args.run or args.summary):
        ap.print_help()
        return 1
    if args.run:
        step_proportions(args.spine)
    if args.summary:
        summary(args.spine)
    return 0


if __name__ == "__main__":
    sys.exit(main())

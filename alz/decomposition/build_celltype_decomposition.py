"""Stage 6 — proportional decomposition of bulk phospho + protein onto the
per-cluster proportions from Stage 5.

Forward projection (linear space):

    P_c(gene, A)    = f_c(gene, A) × bulk_protein(gene, A)
    Phos_c(site, A) = f_c(parent_gene(site), A) × bulk_phospho(site, A)

`f_c` here is the per-cell-rate weight produced by `alz/snrna_proportions.py`
(`(expr_c / Σ expr) × (N_total / N_c)`), so the decomposition is **not**
mass-preserving in the literal Σ_c P_c sense. The verifiable mass identity is

    Σ_c [ P_c × (N_c / N_total) ] = bulk_value

(because `f_c × N_c / N_total = share_c`, and Σ_c share_c = 1 per (gene, A)).

Sites whose parent gene is absent from the snRNA pseudobulk are dropped and
reported in the audit JSON — there is no cell-type prior to project them onto.

Outputs (under `outputs/reports/decomposition/{spine}/`):
  - `protein_per_cluster.parquet` : long-form (gene_symbol, animal_id,
    cluster, value, log2_value)
  - `phospho_per_cluster.parquet` : long-form (site_id, gene_symbol,
    animal_id, cluster, value, log2_value)
  - `decomposition_audit.json`    : coverage + identity-check stats
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402

REPO = Path(config.REPO_ROOT)
BULK_DIR = REPO / "outputs/reports/kinase_attribution"
TOTAL_PROTEOME_FILE = BULK_DIR / "total_proteome_normalized.csv"
SAMPLE_MAPPING_FILE = REPO / "outputs/reports/data_ingest/sample_mapping.csv"
CELL_COUNTS_FILE = REPO / "outputs/reports/snrna_integration/pseudobulk_cell_counts.csv"


def _phospho_paths(track: str) -> tuple[Path, Path]:
    """Return (raw_phospho_normalized.csv, phospho_per_cluster.parquet) for track."""
    if track == "st":
        return BULK_DIR / "raw_phospho_normalized.csv", Path("phospho_per_cluster.parquet")
    if track == "py":
        return BULK_DIR / "raw_phospho_normalized_pY.csv", Path("phospho_per_cluster_pY.parquet")
    raise ValueError(f"Unknown phospho track: {track!r} (expected 'st' or 'py')")


def _spine_dir(spine: str) -> Path:
    return REPO / "outputs/reports/decomposition" / spine


def _load_sample_mapping() -> pd.DataFrame:
    mp = pd.read_csv(SAMPLE_MAPPING_FILE)
    if not {"column_name", "animal_id"}.issubset(mp.columns):
        raise KeyError(f"{SAMPLE_MAPPING_FILE}: missing column_name / animal_id")
    return mp[["column_name", "animal_id"]]


def _bulk_to_long(
    df: pd.DataFrame, value_cols: list[str], id_cols: list[str],
    col_to_animal: dict[str, str], value_name: str,
) -> pd.DataFrame:
    """Wide bulk → long (id_cols..., animal_id, value)."""
    long = df.melt(
        id_vars=id_cols,
        value_vars=value_cols,
        var_name="column_name",
        value_name=value_name,
    )
    long["animal_id"] = long["column_name"].map(col_to_animal)
    if long["animal_id"].isna().any():
        miss = sorted(long.loc[long["animal_id"].isna(), "column_name"].unique())
        raise KeyError(f"Sample mapping missing for columns: {miss[:5]} ...")
    long = long.drop(columns=["column_name"]).dropna(subset=[value_name])
    return long


def _project(
    bulk_long: pd.DataFrame, proportions: pd.DataFrame, gene_key: str,
    value_col: str,
) -> pd.DataFrame:
    """Merge bulk × proportions on (gene, animal_id); multiply."""
    merged = bulk_long.merge(
        proportions, left_on=[gene_key, "animal_id"], right_on=["gene", "animal_id"],
        how="inner",
    )
    if "gene" in merged.columns and gene_key != "gene":
        merged = merged.drop(columns=["gene"])
    merged["value"] = merged[value_col].astype(np.float64) * merged["f_percell"].astype(np.float64)
    merged = merged.drop(columns=[value_col, "f_percell"])
    with np.errstate(divide="ignore", invalid="ignore"):
        merged["log2_value"] = np.where(
            merged["value"] > 0, np.log2(merged["value"]), np.nan
        )
    return merged


def _identity_audit(
    decomp: pd.DataFrame, bulk_long: pd.DataFrame, gene_key: str,
    n_cells_long: pd.DataFrame,
) -> dict:
    """Σ_c [P_c × (N_c / N_total)] should equal bulk_value per (gene, animal).

    `n_cells_long` is per-(animal_id, cluster, n_cells); we collapse to
    per-cluster fractions per snRNA-observed animal. Identity is checked only
    where the proportion source is `observed` to avoid imputed-fraction noise.
    """
    weights = n_cells_long.copy()
    totals = weights.groupby("animal_id")["n_cells"].transform("sum")
    weights["w"] = weights["n_cells"] / totals
    weights = weights[["animal_id", "cluster", "w"]]

    j = decomp.merge(weights, on=["animal_id", "cluster"], how="inner")
    j["weighted"] = j["value"] * j["w"]
    re_bulk = (
        j.groupby([gene_key, "animal_id"], as_index=False)["weighted"]
        .sum()
        .rename(columns={"weighted": "reconstructed"})
    )
    cmp = re_bulk.merge(
        bulk_long.rename(columns={bulk_long.columns[-1]: "bulk"}),
        on=[gene_key, "animal_id"], how="inner",
    )
    cmp["abs_err"] = (cmp["reconstructed"] - cmp["bulk"]).abs()
    cmp["rel_err"] = cmp["abs_err"] / cmp["bulk"].abs().clip(lower=1e-12)
    return {
        "n_compared": int(len(cmp)),
        "max_rel_err": float(cmp["rel_err"].max()) if len(cmp) else None,
        "median_rel_err": float(cmp["rel_err"].median()) if len(cmp) else None,
        "frac_rel_err_below_1e-3": float((cmp["rel_err"] < 1e-3).mean()) if len(cmp) else None,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--spine", default="levy_t5",
                    help="cluster-spine subdirectory under outputs/reports/decomposition/ (default: levy_t5)")
    ap.add_argument("--track", default="both", choices=["st", "py", "both"],
                    help="phospho track(s) to project: st (IMAC pS/pT), py, or both")
    args = ap.parse_args()
    tracks = ["st", "py"] if args.track == "both" else [args.track]

    out_dir = _spine_dir(args.spine)
    if not out_dir.exists():
        raise FileNotFoundError(f"Spine directory not found: {out_dir} "
                                f"(run alz/snrna_proportions.py first)")
    prop_path = out_dir / "proportions.parquet"
    if not prop_path.exists():
        raise FileNotFoundError(f"{prop_path} missing")

    print(f"Stage 6 — decomposition on spine: {args.spine}")
    print(f"  Proportions: {prop_path}")
    proportions = pd.read_parquet(prop_path)
    print(f"    {len(proportions):,} (animal, cluster, gene) rows; "
          f"{proportions['cluster'].nunique()} clusters; "
          f"{proportions['animal_id'].nunique()} animals")

    print(f"  Sample mapping: {SAMPLE_MAPPING_FILE}")
    mp = _load_sample_mapping()
    col_to_animal = dict(zip(mp["column_name"], mp["animal_id"]))

    # --- Phospho (per track) ---
    snrna_genes = set(proportions["gene"].unique())
    phospho_stats: dict[str, dict] = {}
    for track in tracks:
        in_path, out_name = _phospho_paths(track)
        print(f"  [{track}] Bulk phospho: {in_path}")
        if not in_path.exists():
            if track == "py":
                print(f"    [{track}] missing — skipping pY track (Stage 1 did not emit it)")
                phospho_stats[track] = {"status": "missing", "input": str(in_path)}
                continue
            raise FileNotFoundError(in_path)
        phospho = pd.read_csv(in_path)
        meta_cols_ph = ["site_id", "gene_symbol", "motif"]
        val_cols_ph = [c for c in phospho.columns if c not in meta_cols_ph]
        phospho = phospho[["site_id", "gene_symbol"] + val_cols_ph]
        phospho_long = _bulk_to_long(
            phospho, val_cols_ph, ["site_id", "gene_symbol"], col_to_animal, "bulk_value",
        )
        n_sites_total = phospho_long["site_id"].nunique()
        parent_in_snrna = phospho_long["gene_symbol"].isin(snrna_genes)
        dropped_sites = phospho_long.loc[~parent_in_snrna, "site_id"].nunique()
        print(f"    [{track}] Bulk phospho rows: {len(phospho_long):,}; "
              f"sites total={n_sites_total:,}; "
              f"sites dropped (parent gene absent from snRNA)={dropped_sites:,}")
        phospho_long = phospho_long.loc[parent_in_snrna].reset_index(drop=True)

        print(f"  [{track}] Projecting phospho × proportions ...")
        phospho_dec = _project(phospho_long, proportions, "gene_symbol", "bulk_value")
        phospho_out = out_dir / out_name
        phospho_dec[["site_id", "gene_symbol", "animal_id", "cluster", "value", "log2_value"]]\
            .to_parquet(phospho_out, index=False)
        print(f"    [{track}] Wrote {phospho_out} ({len(phospho_dec):,} rows)")
        phospho_stats[track] = {
            "status": "ok",
            "input": str(in_path),
            "output": str(phospho_out),
            "n_sites_input": int(n_sites_total),
            "n_sites_dropped_parent_absent": int(dropped_sites),
            "n_rows_output": int(len(phospho_dec)),
        }

    # --- Total proteome ---
    print(f"  Bulk protein: {TOTAL_PROTEOME_FILE}")
    if not TOTAL_PROTEOME_FILE.exists():
        raise FileNotFoundError(
            f"{TOTAL_PROTEOME_FILE} missing — re-run alz/bulk_mea/normalize.py "
            f"(Stage 1 now emits total_proteome_normalized.csv)."
        )
    protein = pd.read_csv(TOTAL_PROTEOME_FILE)
    meta_cols_pr = [c for c in ("gene_symbol", "protein_id") if c in protein.columns]
    val_cols_pr = [c for c in protein.columns if c not in meta_cols_pr]
    protein = protein.dropna(subset=["gene_symbol"]).copy()
    protein["gene_symbol"] = protein["gene_symbol"].astype(str)
    protein = protein[protein["gene_symbol"].str.match(r"^[A-Za-z][\w\-\.]*$")]
    # one row per gene (first occurrence; total proteome ≈ unique)
    protein = protein.drop_duplicates(subset=["gene_symbol"], keep="first")
    protein_long = _bulk_to_long(
        protein, val_cols_pr, ["gene_symbol"], col_to_animal, "bulk_value",
    )
    n_genes_total = protein_long["gene_symbol"].nunique()
    parent_in_snrna_pr = protein_long["gene_symbol"].isin(snrna_genes)
    dropped_genes = protein_long.loc[~parent_in_snrna_pr, "gene_symbol"].nunique()
    print(f"    Bulk protein rows: {len(protein_long):,}; "
          f"genes total={n_genes_total:,}; "
          f"genes dropped (absent from snRNA)={dropped_genes:,}")
    protein_long = protein_long.loc[parent_in_snrna_pr].reset_index(drop=True)

    print("  Projecting protein × proportions ...")
    protein_dec = _project(protein_long, proportions, "gene_symbol", "bulk_value")
    protein_out = out_dir / "protein_per_cluster.parquet"
    protein_dec[["gene_symbol", "animal_id", "cluster", "value", "log2_value"]]\
        .to_parquet(protein_out, index=False)
    print(f"    Wrote {protein_out} ({len(protein_dec):,} rows)")

    # --- Identity audit (observed-only, protein layer) ---
    print("  Identity audit (protein, snRNA-observed animals only) ...")
    counts = pd.read_csv(CELL_COUNTS_FILE)
    rename = {"sample": "snrna_sample_id"}
    counts = counts.rename(columns=rename)
    mp_full = pd.read_csv(SAMPLE_MAPPING_FILE)
    mp_full = mp_full.loc[mp_full["has_snrna_seq"] == True,  # noqa: E712
                          ["animal_id", "snrna_sample_id"]]
    n_cells_long = counts.merge(mp_full, on="snrna_sample_id", how="inner")
    n_cells_long = n_cells_long.rename(columns={"cell_type": "cluster"})
    n_cells_long = n_cells_long[["animal_id", "cluster", "n_cells"]]
    n_cells_long = n_cells_long[
        n_cells_long["cluster"].isin(proportions["cluster"].unique())
    ]
    obs_animals = set(n_cells_long["animal_id"].unique())
    protein_obs = protein_dec.loc[protein_dec["animal_id"].isin(obs_animals)]
    bulk_obs = protein_long.loc[protein_long["animal_id"].isin(obs_animals),
                                ["gene_symbol", "animal_id", "bulk_value"]]
    audit = _identity_audit(protein_obs, bulk_obs, "gene_symbol", n_cells_long)
    print(f"    {audit}")

    audit_full = {
        "spine": args.spine,
        "n_clusters": int(proportions["cluster"].nunique()),
        "n_animals": int(proportions["animal_id"].nunique()),
        "phospho": phospho_stats,
        "protein": {
            "n_genes_input": int(n_genes_total),
            "n_genes_dropped_absent_from_snrna": int(dropped_genes),
            "n_rows_output": int(len(protein_dec)),
        },
        "mass_identity_protein_observed": audit,
    }
    audit_path = out_dir / "decomposition_audit.json"
    with open(audit_path, "w") as fh:
        json.dump(audit_full, fh, indent=2)
    print(f"  Wrote {audit_path}")


if __name__ == "__main__":
    main()

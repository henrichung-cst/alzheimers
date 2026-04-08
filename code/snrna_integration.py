#!/usr/bin/env python3
"""Song snRNA-seq integration: within-cohort evidence for kinase attribution.

Computes pseudobulk expression, within-cohort expression specificity, and
within-cohort transcriptomic concordance from paired snRNA-seq data
(170_gex_celltypes_00.h5ad, 63,695 nuclei × 30,567 genes, 28 animals).

Uses Allen Cell Type Mapper annotations (210 subclass_name labels) to map
nuclei directly to 22/24 SEA-AD subclasses, bypassing the lossy 46-cluster
taxonomy that only covers 12/24.

Inputs:
  data/incytr_collections/song/transcriptomics/170_gex_celltypes_00.h5ad

Outputs:
  outputs/reports/snrna_integration/pseudobulk_cpm.csv
  outputs/reports/snrna_integration/pseudobulk_cell_counts.csv
  outputs/reports/snrna_integration/song_expression_specificity.csv
  outputs/reports/snrna_integration/song_concordance.csv

Usage:
    python code/snrna_integration.py --pseudobulk    # S1: pseudobulk from h5ad
    python code/snrna_integration.py --specificity   # S2: within-cohort specificity
    python code/snrna_integration.py --concordance   # S3: within-cohort concordance
    python code/snrna_integration.py --run           # All stages in order
    python code/snrna_integration.py --summary       # Print cached results
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Tuple

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from statsmodels.stats.multitest import multipletests

import config

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OUTPUT_DIR = config.SNRNA_INTEGRATION_OUTPUT_DIR
PSEUDOBULK_FILE = config.SONG_PSEUDOBULK_FILE
CELL_COUNTS_FILE = config.SONG_CELL_COUNTS_FILE
SPECIFICITY_FILE = config.SONG_EXPRESSION_FILE
CONCORDANCE_FILE = config.SONG_CONCORDANCE_FILE

# Factorial genotype coding — derived from config.SAP_FACTORIAL
# (canonical source, keys match h5ad `mutant` column values).
GENOTYPE_FACTORIAL = {
    k: {"App": v[0], "Tau": v[1], "Int": v[2]}
    for k, v in config.SAP_FACTORIAL.items()
}


# ---------------------------------------------------------------------------
# S1: Pseudobulk computation
# ---------------------------------------------------------------------------


def step_pseudobulk() -> None:
    """Compute pseudobulk expression from 170_gex_celltypes_00.h5ad.

    For each (animal, SEA-AD subclass) pair, sums raw counts across nuclei
    passing confidence and subclass filters, then applies CPM + log2
    normalization.
    """
    import anndata as ad

    print("=" * 72)
    print("S1  Pseudobulk computation from paired snRNA-seq")
    print("=" * 72)

    h5ad_path = config.SONG_H5AD_FILE
    if not os.path.exists(h5ad_path):
        print(f"  ERROR: h5ad file not found: {h5ad_path}")
        sys.exit(1)

    print(f"  Loading {h5ad_path} ...")
    adata = ad.read_h5ad(h5ad_path)
    print(f"  Loaded: {adata.shape[0]:,} nuclei × {adata.shape[1]:,} genes")

    # Filter by subclass confidence and mappability
    mask_prob = adata.obs["subclass_prob"] >= config.SONG_MIN_SUBCLASS_PROB
    mask_mapped = adata.obs["subclass_name"].isin(config.SONG_SUBCLASS_MAP)
    mask = mask_prob & mask_mapped
    n_pass = mask.sum()
    print(f"  After filtering (prob >= {config.SONG_MIN_SUBCLASS_PROB}, "
          f"mapped subclass): {n_pass:,} nuclei "
          f"({n_pass / len(mask) * 100:.1f}%)")

    # Extract sparse matrix and metadata (avoid full dense copy)
    X = adata.X[mask.values]  # sparse CSR slice — no dense materialization
    obs = adata.obs[mask].copy()
    obs["sea_ad_subclass"] = obs["subclass_name"].map(config.SONG_SUBCLASS_MAP)
    genes = adata.var_names.tolist()
    n_genes = len(genes)
    del adata  # free memory

    # Aggregate per (animal, subclass)
    print("  Aggregating pseudobulk per (animal, subclass) ...")
    groups = obs.groupby(["sample", "sea_ad_subclass"], observed=True)

    cell_counts_rows = []
    pb_meta = []       # (sample, cell_type) per passing group
    pb_data = []       # pre-allocated log2(CPM+1) arrays

    for (sample, subclass), grp_idx in groups.groups.items():
        n_cells = len(grp_idx)
        cell_counts_rows.append({
            "sample": sample, "cell_type": subclass, "n_cells": n_cells,
        })

        if n_cells < config.SONG_MIN_CELLS:
            continue

        # Sum raw counts across nuclei (sparse → dense 1-D)
        row_indices = obs.index.get_indexer(grp_idx)
        raw_sum = np.asarray(X[row_indices].sum(axis=0)).flatten()

        # CPM normalization: counts per million
        total = raw_sum.sum()
        if total > 0:
            cpm = raw_sum / total * 1e6
            log2_cpm = np.log2(cpm + 1)
        else:
            log2_cpm = np.zeros(n_genes)

        pb_meta.append({"sample": sample, "cell_type": subclass})
        pb_data.append(log2_cpm)

    # Build output DataFrames
    cell_counts = pd.DataFrame(cell_counts_rows)
    if pb_data:
        pb_array = np.vstack(pb_data)  # (n_groups, n_genes)
        pseudobulk = pd.DataFrame(pb_array, columns=genes)
        pseudobulk.insert(0, "cell_type",
                          [m["cell_type"] for m in pb_meta])
        pseudobulk.insert(0, "sample", [m["sample"] for m in pb_meta])
    else:
        pseudobulk = pd.DataFrame(columns=["sample", "cell_type"] + genes)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    cell_counts.to_csv(CELL_COUNTS_FILE, index=False)
    print(f"  Cell counts: {len(cell_counts)} (animal × subclass) pairs → "
          f"{CELL_COUNTS_FILE}")

    pseudobulk.to_csv(PSEUDOBULK_FILE, index=False)
    n_samples = pseudobulk["sample"].nunique()
    n_subtypes = pseudobulk["cell_type"].nunique()
    print(f"  Pseudobulk: {len(pseudobulk)} rows "
          f"({n_samples} animals × {n_subtypes} subclasses, "
          f"min {config.SONG_MIN_CELLS} cells gate) → {PSEUDOBULK_FILE}")

    # Summary stats
    gated = cell_counts[cell_counts["n_cells"] >= config.SONG_MIN_CELLS]
    print(f"\n  Coverage after {config.SONG_MIN_CELLS}-cell gate:")
    for ct in sorted(gated["cell_type"].unique()):
        ct_rows = gated[gated["cell_type"] == ct]
        n_animals = len(ct_rows)
        total_cells = ct_rows["n_cells"].sum()
        min_cells = ct_rows["n_cells"].min()
        print(f"    {ct:<20s}  {n_animals:>2d} animals  "
              f"total={total_cells:>5d}  min={min_cells:>3d}")


# ---------------------------------------------------------------------------
# S2: Within-cohort expression specificity
# ---------------------------------------------------------------------------


def step_specificity() -> None:
    """Compute within-cohort expression specificity per gene per subclass.

    Pools all animals (males + females) to maximize power for a static
    property. Mirrors the WMB specificity formula:
        specificity = mean_in_subclass / sum(means_across_all_subclasses)
    """
    print("=" * 72)
    print("S2  Within-cohort expression specificity")
    print("=" * 72)

    if not os.path.exists(PSEUDOBULK_FILE):
        print(f"  ERROR: pseudobulk file not found: {PSEUDOBULK_FILE}")
        print("  Run --pseudobulk first.")
        sys.exit(1)

    pb = pd.read_csv(PSEUDOBULK_FILE)
    print(f"  Loaded pseudobulk: {len(pb)} rows")

    gene_cols = [c for c in pb.columns if c not in ("sample", "cell_type")]
    subclasses = sorted(pb["cell_type"].unique())
    print(f"  Subclasses: {len(subclasses)}, genes: {len(gene_cols)}")

    # Mean expression per subclass (pool across all animals)
    mean_by_ct = pb.groupby("cell_type")[gene_cols].mean()  # (n_subclass, n_genes)

    # Specificity: fraction of total expression in each subclass
    total_expr = mean_by_ct.sum(axis=0)  # (n_genes,)
    total_expr = total_expr.replace(0, np.nan)  # avoid division by zero
    specificity = mean_by_ct.div(total_expr, axis=1)  # (n_subclass, n_genes)

    # Reshape to long format matching WMB output schema (vectorized)
    spec_long = specificity.stack().reset_index()
    spec_long.columns = ["cell_type", "gene_symbol", "specificity_score"]
    mean_long = mean_by_ct.stack().reset_index()
    mean_long.columns = ["cell_type", "gene_symbol", "mean_expression"]
    df = spec_long.merge(mean_long, on=["cell_type", "gene_symbol"])
    df = df[np.isfinite(df["specificity_score"]) & (df["mean_expression"] > 0)].copy()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df.to_csv(SPECIFICITY_FILE, index=False)
    print(f"  Output: {len(df)} (gene × subclass) pairs → {SPECIFICITY_FILE}")

    # Summary: top specific genes per subclass
    print(f"\n  Top 3 most specific genes per subclass:")
    for ct in sorted(df["cell_type"].unique()):
        ct_df = df[df["cell_type"] == ct].nlargest(3, "specificity_score")
        genes_str = ", ".join(
            f"{r['gene_symbol']} ({r['specificity_score']:.3f})"
            for _, r in ct_df.iterrows()
        )
        print(f"    {ct:<20s}  {genes_str}")


# ---------------------------------------------------------------------------
# S3: Within-cohort transcriptomic concordance
# ---------------------------------------------------------------------------


def _fit_ols_batch(
    Y: np.ndarray, X: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized OLS for multiple genes.

    Adapted from archive/code/sap_module1_de.py.

    Args:
        Y: (n_genes, n_samples) response matrix (log2(CPM+1))
        X: (n_samples, p) design matrix

    Returns:
        beta_hat: (n_genes, p)
        se: (n_genes, p)
        t_stats: (n_genes, p)
        p_values: (n_genes, p)  two-sided
    """
    n, p = X.shape
    df_resid = n - p

    XtX_inv = np.linalg.inv(X.T @ X)
    beta_hat = (XtX_inv @ X.T @ Y.T).T

    residuals = Y - beta_hat @ X.T
    sigma2 = np.sum(residuals**2, axis=1) / df_resid

    diag_XtXinv = np.diag(XtX_inv)
    se = np.sqrt(np.outer(sigma2, diag_XtXinv))

    t_stats = beta_hat / np.where(se > 0, se, np.inf)
    p_values = 2.0 * sp_stats.t.sf(np.abs(t_stats), df=df_resid)

    return beta_hat, se, t_stats, p_values


def step_concordance() -> None:
    """Compute within-cohort transcriptomic concordance via factorial OLS.

    Males-only analysis for consistency with the primary kinase enrichment.
    Design matrix: intercept + App + Tau + Int (4 params).
    Pools across timepoints for statistical power (df_resid ≈ 11).

    Output pathways:
        App_lfc = β_App  (applies to App_2mo, App_4mo, App_6mo)
        Tau_lfc = β_Tau  (applies to Tau_2mo, Tau_4mo, Tau_6mo)
        ApTt_lfc = β_App + β_Tau + β_Int  (applies to ApTt_2mo, ApTt_4mo, ApTt_6mo)
    """
    print("=" * 72)
    print("S3  Within-cohort transcriptomic concordance (males-only OLS)")
    print("=" * 72)

    if not os.path.exists(PSEUDOBULK_FILE):
        print(f"  ERROR: pseudobulk file not found: {PSEUDOBULK_FILE}")
        print("  Run --pseudobulk first.")
        sys.exit(1)

    pb = pd.read_csv(PSEUDOBULK_FILE)
    gene_cols = [c for c in pb.columns if c not in ("sample", "cell_type")]

    # Parse sample metadata
    pb["sex"] = pb["sample"].str.split("_").str[1]
    pb["timepoint"] = pb["sample"].str.split("_").str[2]
    pb["genotype"] = pb["sample"].str.split("_").str[3]

    # Males only
    pb_male = pb[pb["sex"] == "ma"].copy()
    n_males = pb_male["sample"].nunique()
    print(f"  Males-only pseudobulk: {len(pb_male)} rows from {n_males} animals")

    subclasses = sorted(pb_male["cell_type"].unique())
    all_rows = []

    for ct in subclasses:
        ct_df = pb_male[pb_male["cell_type"] == ct].copy()
        n_animals = ct_df["sample"].nunique()

        if n_animals < config.SONG_MIN_ANIMALS:
            print(f"    {ct:<20s}  SKIP ({n_animals} animals < "
                  f"{config.SONG_MIN_ANIMALS} min)")
            continue

        # Build design matrix: intercept + App + Tau + Int
        X_rows = []
        for _, row in ct_df.iterrows():
            geno = row["genotype"]
            coding = GENOTYPE_FACTORIAL[geno]
            X_rows.append([1.0, coding["App"], coding["Tau"], coding["Int"]])
        X = np.array(X_rows)  # (n_animals, 4)
        # Expression matrix: (n_genes, n_animals)
        Y = ct_df[gene_cols].values.T  # already log2(CPM+1)

        # Filter genes with nonzero expression in at least half the animals
        min_detect = max(3, n_animals // 2)
        gene_detect = (Y > 0).sum(axis=1)
        gene_mask = gene_detect >= min_detect
        Y_filt = Y[gene_mask]
        gene_names_filt = [gene_cols[i] for i in range(len(gene_cols)) if gene_mask[i]]

        if len(gene_names_filt) == 0:
            print(f"    {ct:<20s}  SKIP (no genes pass detection filter)")
            continue

        # Vectorized OLS
        beta_hat, se, _, p_values = _fit_ols_batch(Y_filt, X)

        # Extract factorial coefficients (indices: 1=App, 2=Tau, 3=Int)
        idx_app, idx_tau, idx_int = 1, 2, 3

        # Derive pathway contrasts
        app_lfc = beta_hat[:, idx_app]
        tau_lfc = beta_hat[:, idx_tau]
        aptt_lfc = beta_hat[:, idx_app] + beta_hat[:, idx_tau] + beta_hat[:, idx_int]

        app_se = se[:, idx_app]
        tau_se = se[:, idx_tau]
        # ApTt SE via variance propagation: Var(β_A + β_T + β_I)
        cov_mat = np.linalg.inv(X.T @ X)
        var_aptt = (cov_mat[idx_app, idx_app] + cov_mat[idx_tau, idx_tau]
                    + cov_mat[idx_int, idx_int]
                    + 2 * cov_mat[idx_app, idx_tau]
                    + 2 * cov_mat[idx_app, idx_int]
                    + 2 * cov_mat[idx_tau, idx_int])
        sigma2 = np.sum((Y_filt - beta_hat @ X.T)**2, axis=1) / (X.shape[0] - X.shape[1])
        aptt_se = np.sqrt(sigma2 * var_aptt)

        app_pval = p_values[:, idx_app]
        tau_pval = p_values[:, idx_tau]
        # ApTt p-value from t-test
        df_resid = X.shape[0] - X.shape[1]
        aptt_t = aptt_lfc / np.where(aptt_se > 0, aptt_se, np.inf)
        aptt_pval = 2.0 * sp_stats.t.sf(np.abs(aptt_t), df=df_resid)

        for pathway, lfc_arr, se_arr, pval_arr in [
            ("App", app_lfc, app_se, app_pval),
            ("Tau", tau_lfc, tau_se, tau_pval),
            ("ApTt", aptt_lfc, aptt_se, aptt_pval),
        ]:
            for gi, gene in enumerate(gene_names_filt):
                all_rows.append({
                    "gene_symbol": gene,
                    "cell_type": ct,
                    "pathway": pathway,
                    "song_lfc": float(lfc_arr[gi]),
                    "song_se": float(se_arr[gi]),
                    "song_pval": float(pval_arr[gi]),
                    "n_animals": n_animals,
                })

        n_sig_app = (app_pval < 0.05).sum()
        n_sig_tau = (tau_pval < 0.05).sum()
        n_sig_aptt = (aptt_pval < 0.05).sum()
        print(f"    {ct:<20s}  {n_animals:>2d} animals  "
              f"{len(gene_names_filt):>5d} genes  "
              f"sig(p<.05): App={n_sig_app}, Tau={n_sig_tau}, ApTt={n_sig_aptt}")

    df = pd.DataFrame(all_rows)

    # BH FDR correction per (cell_type, pathway)
    print("\n  Applying BH FDR correction per (subclass, pathway) ...")
    df["song_fdr"] = np.nan
    for (ct, _pw), idx in df.groupby(["cell_type", "pathway"]).groups.items():
        if len(idx) > 0:
            _, fdr_vals, _, _ = multipletests(
                df.loc[idx, "song_pval"].values, alpha=0.05, method="fdr_bh",
            )
            df.loc[idx, "song_fdr"] = fdr_vals

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df.to_csv(CONCORDANCE_FILE, index=False)
    print(f"\n  Output: {len(df)} (gene × subclass × pathway) rows → "
          f"{CONCORDANCE_FILE}")

    # Summary
    n_sig = (df["song_fdr"] < 0.05).sum()
    n_total = len(df)
    print(f"  Significant at FDR < 0.05: {n_sig}/{n_total} "
          f"({n_sig / n_total * 100:.1f}%)")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def print_summary() -> None:
    """Print summary of cached snRNA integration outputs."""
    print("=" * 72)
    print("Song snRNA Integration — Summary")
    print("=" * 72)

    for label, path in [
        ("Pseudobulk CPM", PSEUDOBULK_FILE),
        ("Cell counts", CELL_COUNTS_FILE),
        ("Expression specificity", SPECIFICITY_FILE),
        ("Concordance", CONCORDANCE_FILE),
    ]:
        if os.path.exists(path):
            df = pd.read_csv(path)
            print(f"\n  {label}: {path}")
            print(f"    {len(df)} rows, {len(df.columns)} columns")
            if "cell_type" in df.columns:
                print(f"    Cell types: {df['cell_type'].nunique()}")
            if "sample" in df.columns:
                print(f"    Samples: {df['sample'].nunique()}")
            if "gene_symbol" in df.columns:
                print(f"    Genes: {df['gene_symbol'].nunique()}")
            if "pathway" in df.columns:
                print(f"    Pathways: {sorted(df['pathway'].unique())}")
            if "song_fdr" in df.columns:
                n_sig = (df["song_fdr"] < 0.05).sum()
                print(f"    Significant (FDR < 0.05): {n_sig}/{len(df)}")
        else:
            print(f"\n  {label}: NOT FOUND ({path})")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Song snRNA-seq integration for kinase attribution"
    )
    parser.add_argument("--pseudobulk", action="store_true",
                        help="S1: Compute pseudobulk from h5ad")
    parser.add_argument("--specificity", action="store_true",
                        help="S2: Within-cohort expression specificity")
    parser.add_argument("--concordance", action="store_true",
                        help="S3: Within-cohort transcriptomic concordance")
    parser.add_argument("--run", action="store_true",
                        help="Run all stages in order")
    parser.add_argument("--summary", action="store_true",
                        help="Print cached results summary")
    args = parser.parse_args()

    if not any([args.pseudobulk, args.specificity, args.concordance,
                args.run, args.summary]):
        parser.print_help()
        sys.exit(1)

    if args.summary:
        print_summary()
        return

    if args.run or args.pseudobulk:
        step_pseudobulk()
    if args.run or args.specificity:
        step_specificity()
    if args.run or args.concordance:
        step_concordance()

    print("\nDone.")


if __name__ == "__main__":
    main()

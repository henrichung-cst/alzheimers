#!/usr/bin/env python3
"""Module 1: snRNA-seq pseudobulk differential expression for kinases/phosphatases.

Per-cell-type OLS on log2(CPM+1) with factorial parameterization
(App, Tau, App x Tau). Produces cell-type-resolved DE estimates that
the phosphoproteomics deconvolution cannot provide.

See sap_rescue.md Module 1 for specification.

Usage:
    python code/sap_module1_de.py --run        # full DE pipeline
    python code/sap_module1_de.py --validate   # positive control checks only
    python code/sap_module1_de.py --summary    # print cached results
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from statsmodels.stats.multitest import multipletests

import config
import sap_data

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OUTPUT_DIR = os.path.join("outputs", "reports", "module1_kinase_de")
DE_OUTPUT_FILE = os.path.join(OUTPUT_DIR, "kinase_phosphatase_de.csv")
CONDITION_CONTRASTS_FILE = os.path.join(OUTPUT_DIR, "condition_contrasts.csv")
VALIDATION_FILE = os.path.join(OUTPUT_DIR, "validation_checks.json")
SUMMARY_FILE = os.path.join(OUTPUT_DIR, "module1_summary.json")

# Factorial contrast column indices in the design matrix
# Design: intercept(0) + female(1) + time_4mo(2) + time_6mo(3) + App(4) + Tau(5) + Int(6)
IDX_APP = 4
IDX_TAU = 5
IDX_INT = 6
CONTRAST_NAMES = ["App", "Tau", "Int"]
CONTRAST_INDICES = [IDX_APP, IDX_TAU, IDX_INT]

# ---------------------------------------------------------------------------
# Design matrix
# ---------------------------------------------------------------------------


def build_design_matrix(sample_meta: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    """Build the 24x7 OLS design matrix.

    Columns: intercept, female, time_4mo, time_6mo, App, Tau, Int.
    Uses the same factorial coding as the SAP model (config.SAP_FACTORIAL).

    Returns:
        X: (24, 7) design matrix
        col_names: column labels
    """
    n = len(sample_meta)
    female = (sample_meta["gender"].values == "fe").astype(float)
    time_4mo = (sample_meta["timepoint"].values == "4mo").astype(float)
    time_6mo = (sample_meta["timepoint"].values == "6mo").astype(float)
    fact = np.array(
        [config.SAP_FACTORIAL[c] for c in sample_meta["condition"].values],
        dtype=float,
    )  # (n, 3)

    X = np.column_stack([np.ones(n), female, time_4mo, time_6mo, fact])
    col_names = ["intercept", "female", "time_4mo", "time_6mo", "App", "Tau", "Int"]
    return X, col_names


# ---------------------------------------------------------------------------
# Vectorized OLS
# ---------------------------------------------------------------------------


def fit_ols_batch(
    Y: np.ndarray, X: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized OLS for multiple genes.

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

    # OLS: beta = (X'X)^{-1} X'Y'
    XtX_inv = np.linalg.inv(X.T @ X)  # (p, p)
    beta_hat = (XtX_inv @ X.T @ Y.T).T  # (n_genes, p)

    # Residuals and sigma^2
    residuals = Y - beta_hat @ X.T  # (n_genes, n_samples)
    sigma2 = np.sum(residuals**2, axis=1) / df_resid  # (n_genes,)

    # Standard errors
    diag_XtXinv = np.diag(XtX_inv)  # (p,)
    se = np.sqrt(np.outer(sigma2, diag_XtXinv))  # (n_genes, p)

    # t-statistics and p-values
    t_stats = beta_hat / np.where(se > 0, se, np.inf)
    p_values = 2.0 * sp_stats.t.sf(np.abs(t_stats), df=df_resid)

    return beta_hat, se, t_stats, p_values


# ---------------------------------------------------------------------------
# Per-cell-type DE
# ---------------------------------------------------------------------------


def run_de_per_celltype(
    cpm_kp: pd.DataFrame,
    sample_meta: pd.DataFrame,
) -> pd.DataFrame:
    """Run OLS DE for each cell type and collect results.

    Args:
        cpm_kp: MultiIndex (cell_type, sample_id) x genes. CPM values (not z-scored).
        sample_meta: (24,) with gender, timepoint, condition columns.

    Returns:
        DataFrame with columns: gene, cell_type, contrast, log2fc, se, pval_raw,
        pval_adj, significant_005, significant_010
    """
    X, col_names = build_design_matrix(sample_meta)
    sample_order = sample_meta.index.tolist()
    est_types = config.SAP_ESTIMATED_CELLTYPES

    all_rows = []

    for ct in est_types:
        print(f"  Fitting {ct}...")
        ct_df = cpm_kp.loc[ct]  # (24 samples, n_genes)
        ct_df = ct_df.loc[sample_order]  # align to sample_meta order
        genes = ct_df.columns.tolist()

        # log2(CPM + 1) transform
        Y = np.log2(ct_df.values.T + 1)  # (n_genes, 24)

        beta_hat, se, t_stats, p_values = fit_ols_batch(Y, X)

        # Extract factorial contrasts
        for ci, cname in zip(CONTRAST_INDICES, CONTRAST_NAMES):
            for gi, gene in enumerate(genes):
                all_rows.append({
                    "gene": gene,
                    "cell_type": ct,
                    "contrast": cname,
                    "log2fc": float(beta_hat[gi, ci]),
                    "se": float(se[gi, ci]),
                    "pval_raw": float(p_values[gi, ci]),
                })

    df = pd.DataFrame(all_rows)

    # BH FDR per cell type
    print("  Applying BH FDR correction per cell type...")
    df["pval_adj"] = np.nan
    for ct in est_types:
        mask = df["cell_type"] == ct
        _, pvals_adj, _, _ = multipletests(
            df.loc[mask, "pval_raw"].values,
            alpha=0.05,
            method="fdr_bh",
        )
        df.loc[mask, "pval_adj"] = pvals_adj

    df["significant_005"] = df["pval_adj"] <= 0.05
    df["significant_010"] = df["pval_adj"] <= 0.10

    return df


# ---------------------------------------------------------------------------
# Condition-level contrasts (for Module 2 joining)
# ---------------------------------------------------------------------------


def compute_condition_contrasts(
    cpm_kp: pd.DataFrame,
    sample_meta: pd.DataFrame,
) -> pd.DataFrame:
    """Derive condition-vs-WT contrasts from the factorial model.

    AppP vs WT = beta_App
    Ttau vs WT = beta_Tau
    ApTt vs WT = beta_App + beta_Tau + beta_Int

    Computes proper SE and p-value for the ApTt linear combination.
    """
    X, col_names = build_design_matrix(sample_meta)
    sample_order = sample_meta.index.tolist()
    est_types = config.SAP_ESTIMATED_CELLTYPES
    n, p = X.shape
    df_resid = n - p
    XtX_inv = np.linalg.inv(X.T @ X)

    # Contrast vectors (length p)
    # AppP vs WT: just beta_App
    c_app = np.zeros(p)
    c_app[IDX_APP] = 1.0
    # Ttau vs WT: just beta_Tau
    c_tau = np.zeros(p)
    c_tau[IDX_TAU] = 1.0
    # ApTt vs WT: beta_App + beta_Tau + beta_Int
    c_aptt = np.zeros(p)
    c_aptt[IDX_APP] = 1.0
    c_aptt[IDX_TAU] = 1.0
    c_aptt[IDX_INT] = 1.0

    contrasts = [
        ("AppP", c_app),
        ("Ttau", c_tau),
        ("ApTt", c_aptt),
    ]

    all_rows = []

    for ct in est_types:
        ct_df = cpm_kp.loc[ct].loc[sample_order]
        genes = ct_df.columns.tolist()
        Y = np.log2(ct_df.values.T + 1)  # (n_genes, 24)

        beta_hat = (XtX_inv @ X.T @ Y.T).T  # (n_genes, p)
        residuals = Y - beta_hat @ X.T
        sigma2 = np.sum(residuals**2, axis=1) / df_resid  # (n_genes,)

        for cond_name, c_vec in contrasts:
            # c'beta for each gene
            lfc = beta_hat @ c_vec  # (n_genes,)
            # var(c'beta) = sigma^2 * c' (X'X)^{-1} c
            var_factor = float(c_vec @ XtX_inv @ c_vec)
            se_cond = np.sqrt(sigma2 * var_factor)  # (n_genes,)
            t_cond = lfc / np.where(se_cond > 0, se_cond, np.inf)
            pval_cond = 2.0 * sp_stats.t.sf(np.abs(t_cond), df=df_resid)

            for gi, gene in enumerate(genes):
                all_rows.append({
                    "gene": gene,
                    "cell_type": ct,
                    "condition": cond_name,
                    "log2fc": float(lfc[gi]),
                    "se": float(se_cond[gi]),
                    "pval_raw": float(pval_cond[gi]),
                })

    df = pd.DataFrame(all_rows)

    # BH FDR per cell type
    df["pval_adj"] = np.nan
    for ct in est_types:
        mask = df["cell_type"] == ct
        _, pvals_adj, _, _ = multipletests(
            df.loc[mask, "pval_raw"].values, alpha=0.05, method="fdr_bh",
        )
        df.loc[mask, "pval_adj"] = pvals_adj

    df["significant_005"] = df["pval_adj"] <= 0.05
    df["significant_010"] = df["pval_adj"] <= 0.10

    return df


# ---------------------------------------------------------------------------
# Validation checks
# ---------------------------------------------------------------------------


def run_validation_checks(
    cpm_full: pd.DataFrame,
    sample_meta: pd.DataFrame,
    de_results: pd.DataFrame,
) -> Dict:
    """Run positive control checks per sap_rescue.md Module 1.6.

    1. Transgene confirmation: App up in AppP/ApTt, Mapt up in Ttau/ApTt
    2. Sex gene confirmation: Xist sex effect, Ddx3y opposite
    3. Ceiling check: no CT with >50% kinases significant at FDR 0.05

    Args:
        cpm_full: Full CPM-normalized aggexp (not just KP subset), for
                  transgene/sex gene checks. MultiIndex (cell_type, sample_id).
    """
    checks: Dict = {}
    X, _ = build_design_matrix(sample_meta)
    sample_order = sample_meta.index.tolist()

    # --- 1. Transgene checks ---
    # App and Mapt are the mouse endogenous genes. Under transgenic conditions,
    # the human transgene expression adds to pseudobulk counts at these loci.
    transgene_genes = {
        "App": ["App"],
        "Mapt": ["Mapt"],
    }
    transgene_results = {}
    for tg_label, candidates in transgene_genes.items():
        for gene in candidates:
            if gene in cpm_full.columns:
                # Run OLS on this gene across all 5 CTs, check condition effects
                gene_effects = {}
                for ct in config.SAP_ESTIMATED_CELLTYPES:
                    ct_df = cpm_full.loc[ct].loc[sample_order]
                    y = np.log2(ct_df[gene].values + 1)
                    beta = np.linalg.lstsq(X, y, rcond=None)[0]
                    resid = y - X @ beta
                    sigma2 = np.sum(resid**2) / (len(y) - X.shape[1])
                    XtXinv = np.linalg.inv(X.T @ X)
                    se = np.sqrt(sigma2 * np.diag(XtXinv))
                    gene_effects[ct] = {
                        "App_lfc": float(beta[IDX_APP]),
                        "Tau_lfc": float(beta[IDX_TAU]),
                        "Int_lfc": float(beta[IDX_INT]),
                        "App_pval": float(2 * sp_stats.t.sf(abs(beta[IDX_APP] / se[IDX_APP]), df=len(y) - X.shape[1])),
                        "Tau_pval": float(2 * sp_stats.t.sf(abs(beta[IDX_TAU] / se[IDX_TAU]), df=len(y) - X.shape[1])),
                    }
                transgene_results[f"{tg_label}({gene})"] = gene_effects
    checks["transgene"] = transgene_results

    # --- 2. Sex gene checks ---
    sex_genes = {
        "Xist": "female_up",     # Should have positive female coefficient
        "Ddx3y": "male_up",      # Should have negative female coefficient
        "Uty": "male_up",
        "Kdm5d": "male_up",
    }
    sex_results = {}
    for gene, expected_dir in sex_genes.items():
        if gene in cpm_full.columns:
            gene_sex = {}
            for ct in config.SAP_ESTIMATED_CELLTYPES:
                ct_df = cpm_full.loc[ct].loc[sample_order]
                y = np.log2(ct_df[gene].values + 1)
                beta = np.linalg.lstsq(X, y, rcond=None)[0]
                resid = y - X @ beta
                sigma2 = np.sum(resid**2) / (len(y) - X.shape[1])
                XtXinv = np.linalg.inv(X.T @ X)
                se_fem = np.sqrt(sigma2 * XtXinv[1, 1])
                pval = float(2 * sp_stats.t.sf(abs(beta[1] / se_fem), df=len(y) - X.shape[1]))
                correct = (expected_dir == "female_up" and beta[1] > 0) or \
                          (expected_dir == "male_up" and beta[1] < 0)
                gene_sex[ct] = {
                    "female_lfc": float(beta[1]),
                    "pval": pval,
                    "correct_direction": bool(correct),
                }
            sex_results[gene] = gene_sex
    checks["sex_genes"] = sex_results

    # --- 3. Ceiling check ---
    ceiling = {}
    for ct in config.SAP_ESTIMATED_CELLTYPES:
        ct_mask = de_results["cell_type"] == ct
        for contrast in CONTRAST_NAMES:
            c_mask = ct_mask & (de_results["contrast"] == contrast)
            n_total = c_mask.sum()
            n_sig = (de_results.loc[c_mask, "significant_005"]).sum()
            frac = float(n_sig / n_total) if n_total > 0 else 0.0
            ceiling[f"{ct}_{contrast}"] = {
                "n_significant": int(n_sig),
                "n_total": int(n_total),
                "fraction": frac,
                "PASS": frac <= 0.50,
            }
    checks["ceiling"] = ceiling
    checks["ceiling_all_pass"] = all(v["PASS"] for v in ceiling.values())

    return checks


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def compute_summary(de_results: pd.DataFrame, validation: Dict) -> Dict:
    """Compute summary statistics for the DE analysis."""
    n_genes = de_results["gene"].nunique()
    n_cts = de_results["cell_type"].nunique()
    n_tests_per_ct = n_genes * len(CONTRAST_NAMES)

    summary = {
        "n_genes": n_genes,
        "n_cell_types": n_cts,
        "n_models": n_genes * n_cts,
        "n_tests_per_ct": n_tests_per_ct,
        "contrasts": CONTRAST_NAMES,
    }

    # Per-cell-type significance counts
    sig_counts = {}
    for ct in config.SAP_ESTIMATED_CELLTYPES:
        ct_df = de_results[de_results["cell_type"] == ct]
        sig_counts[ct] = {
            "FDR_005": int(ct_df["significant_005"].sum()),
            "FDR_010": int(ct_df["significant_010"].sum()),
            "total_tests": len(ct_df),
        }
        # Per-contrast breakdown
        for c in CONTRAST_NAMES:
            c_df = ct_df[ct_df["contrast"] == c]
            sig_counts[ct][f"{c}_FDR005"] = int(c_df["significant_005"].sum())
    summary["per_celltype"] = sig_counts

    # Top hits (most significant per contrast)
    top_hits = {}
    for c in CONTRAST_NAMES:
        c_df = de_results[de_results["contrast"] == c].copy()
        c_sig = c_df[c_df["significant_005"]].sort_values("pval_adj")
        top_hits[c] = {
            "n_significant_005": len(c_sig),
            "top_5": c_sig.head(5)[["gene", "cell_type", "log2fc", "pval_adj"]].to_dict("records"),
        }
    summary["top_hits"] = top_hits
    summary["validation_passed"] = validation.get("ceiling_all_pass", False)

    return summary


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str], List[str]]:
    """Load CPM-normalized expression matrices and sample metadata.

    Returns:
        cpm_kp: KP-subset CPM matrix (MultiIndex, ~541 genes)
        cpm_full: Full CPM matrix (MultiIndex, ~24K genes) for validation
        sample_meta: (24,) with gender, timepoint, condition
        kinase_genes, phosphatase_genes: gene lists
    """
    print("Loading aggexp and building CPM matrix...")
    aggexp = sap_data.load_aggexp_pooled()
    sample_map = sap_data.build_aggexp_sample_map(aggexp)

    # Get KP subset with CPM values
    gkp, kinase_genes, phosphatase_genes, cpm_kp = sap_data.preprocess_aggexp(
        aggexp, sample_map, return_cpm=True,
    )

    # Build full CPM matrix for validation (transgene/sex gene checks)
    print("  Building full CPM matrix for validation...")
    df_full = sap_data._remap_aggexp_samples(aggexp, sample_map)
    est_types = config.SAP_ESTIMATED_CELLTYPES
    df_full = df_full.loc[df_full.index.get_level_values("cell_type").isin(est_types)]
    cpm_full = sap_data._cpm_normalize(df_full)
    cpm_full = sap_data._filter_genes(cpm_full)

    # Build sample_meta from the sample IDs
    rows = []
    for _, sid in sorted(sample_map.items()):
        parts = sid.split("_")
        rows.append({
            "sample_id": sid,
            "gender": parts[0],
            "timepoint": parts[1],
            "condition": parts[2],
        })
    sample_meta = pd.DataFrame(rows).set_index("sample_id")

    return cpm_kp, cpm_full, sample_meta, kinase_genes, phosphatase_genes


def run_pipeline() -> None:
    """Run the full Module 1 DE pipeline."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    cpm_kp, cpm_full, sample_meta, kinase_genes, phosphatase_genes = load_data()

    print(f"\nDataset: {cpm_kp.shape[1]} KP genes, "
          f"{len(kinase_genes)} kinases, {len(phosphatase_genes)} phosphatases")
    print(f"Full aggexp: {cpm_full.shape[1]} genes (for validation)")
    print(f"Cell types: {config.SAP_ESTIMATED_CELLTYPES}")
    print(f"Samples: {len(sample_meta)}")
    print(f"Design: 7 parameters, {len(sample_meta) - 7} residual df\n")

    # --- Step 1: Factorial DE ---
    print("=" * 60)
    print("Step 1: Per-cell-type OLS DE (factorial contrasts)")
    print("=" * 60)
    de_results = run_de_per_celltype(cpm_kp, sample_meta)
    de_results.to_csv(DE_OUTPUT_FILE, index=False)
    print(f"\n  Saved: {DE_OUTPUT_FILE} ({len(de_results)} rows)")

    # --- Step 2: Condition-level contrasts for Module 2 ---
    print("\n" + "=" * 60)
    print("Step 2: Condition-vs-WT contrasts (for Module 2)")
    print("=" * 60)
    cond_contrasts = compute_condition_contrasts(cpm_kp, sample_meta)
    cond_contrasts.to_csv(CONDITION_CONTRASTS_FILE, index=False)
    print(f"  Saved: {CONDITION_CONTRASTS_FILE} ({len(cond_contrasts)} rows)")

    # --- Step 3: Validation checks ---
    print("\n" + "=" * 60)
    print("Step 3: Validation checks")
    print("=" * 60)
    validation = run_validation_checks(cpm_full, sample_meta, de_results)

    # Print validation results
    print("\n  Transgene checks:")
    for tg, ct_data in validation.get("transgene", {}).items():
        for ct, effects in ct_data.items():
            ct_short = ct.split("_")[0][:3]
            print(f"    {tg} in {ct_short}: App={effects['App_lfc']:+.3f} "
                  f"(p={effects['App_pval']:.3g}), "
                  f"Tau={effects['Tau_lfc']:+.3f} (p={effects['Tau_pval']:.3g})")

    print("\n  Sex gene checks:")
    for gene, ct_data in validation.get("sex_genes", {}).items():
        dirs = [f"{ct.split('_')[0][:3]}={'OK' if v['correct_direction'] else 'FAIL'}"
                for ct, v in ct_data.items()]
        print(f"    {gene}: {', '.join(dirs)}")

    print(f"\n  Ceiling check: {'PASS' if validation['ceiling_all_pass'] else 'FAIL'}")
    if not validation["ceiling_all_pass"]:
        for key, val in validation["ceiling"].items():
            if not val["PASS"]:
                print(f"    FAIL: {key} ({val['fraction']:.1%} significant)")

    with open(VALIDATION_FILE, "w") as f:
        json.dump(validation, f, indent=2)

    # --- Step 4: Summary ---
    summary = compute_summary(de_results, validation)
    with open(SUMMARY_FILE, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Summary saved: {SUMMARY_FILE}")

    # Print key results
    print("\n" + "=" * 60)
    print("Results Summary")
    print("=" * 60)
    for ct, counts in summary["per_celltype"].items():
        ct_short = ct[:15].ljust(15)
        print(f"  {ct_short}  FDR<0.05: {counts['FDR_005']:4d}/{counts['total_tests']}  "
              f"FDR<0.10: {counts['FDR_010']:4d}/{counts['total_tests']}")

    for c in CONTRAST_NAMES:
        hits = summary["top_hits"][c]
        print(f"\n  {c} contrast: {hits['n_significant_005']} significant (FDR<0.05)")
        for h in hits["top_5"]:
            print(f"    {h['gene']:12s} in {h['cell_type'][:12]:12s} "
                  f"log2FC={h['log2fc']:+.3f}  q={h['pval_adj']:.2e}")


def print_summary() -> None:
    """Print cached summary."""
    if not os.path.exists(SUMMARY_FILE):
        print("No cached results. Run --run first.")
        return
    with open(SUMMARY_FILE) as f:
        summary = json.load(f)
    print(json.dumps(summary, indent=2))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Module 1: snRNA-seq pseudobulk DE")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run", action="store_true", help="Run full DE pipeline")
    group.add_argument("--validate", action="store_true", help="Run validation checks only")
    group.add_argument("--summary", action="store_true", help="Print cached results")
    args = parser.parse_args()

    if args.summary:
        print_summary()
    elif args.validate:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        cpm_kp, cpm_full, sample_meta, _, _ = load_data()
        if not os.path.exists(DE_OUTPUT_FILE):
            print("No DE results found. Running DE first...")
            de_results = run_de_per_celltype(cpm_kp, sample_meta)
            de_results.to_csv(DE_OUTPUT_FILE, index=False)
        else:
            de_results = pd.read_csv(DE_OUTPUT_FILE)
        validation = run_validation_checks(cpm_full, sample_meta, de_results)
        with open(VALIDATION_FILE, "w") as f:
            json.dump(validation, f, indent=2)
        print(json.dumps(validation, indent=2))
    elif args.run:
        run_pipeline()


if __name__ == "__main__":
    main()

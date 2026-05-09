"""Nodes for the enrich pipeline.

Thin wrappers over helpers in `alz.kinase_enrich` so the legacy CLI shim
and the Kedro pipeline run the same code paths.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from alz import config
from alz.kinase_enrich import (
    CONTRAST_COEFS,
    _bh_fdr,
    _build_design_matrix,
    _filter_samples,
    _resolve_track,
    _run_mea,
    _run_ols_all_sites,
    load_sample_exclusions,
)


def filter_samples(sample_mapping: pd.DataFrame, analysis_mode: str,
                   sample_exclusions_path: str) -> pd.DataFrame:
    """Apply outlier exclusion + sex filter. Optional exclusions file is
    handled in-node — Kedro's catalog cannot natively express 'may not exist'."""
    if sample_exclusions_path and os.path.exists(sample_exclusions_path):
        excl_df = pd.read_csv(sample_exclusions_path)
        excluded = set(excl_df.loc[excl_df["excluded"], "mouse_id"])
    else:
        excluded = set()
    n0 = len(sample_mapping)
    filt = sample_mapping[~sample_mapping["mouse_id"].isin(excluded)].copy()
    n_excl = n0 - len(filt)
    if analysis_mode == "males_only":
        filt = filt[filt["sex"] == "M"].copy()
    n_final = len(filt)
    print(f"  Sample filter ({analysis_mode}): {n0} -> {n_final} "
          f"({n_excl} outliers excluded, {n0 - n_excl - n_final} sex-filtered)")
    return filt


def fit_and_contrast(stoichiometry_matrix: pd.DataFrame,
                     raw_phospho_normalized: pd.DataFrame,
                     filtered_mapping: pd.DataFrame,
                     analysis_mode: str):
    """Build design matrix, run OLS on stoich + raw, compute per-contrast LFC/SE/FDR.

    Returns ``(site_level_ols_df, results_by_contrast)``. The dict is consumed
    in-memory by the MEA node (no on-disk roundtrip)."""
    bio_cols = filtered_mapping["column_name"].tolist()
    X = _build_design_matrix(filtered_mapping, bio_cols, analysis_mode=analysis_mode)
    X_np = X.values
    param_names = list(X.columns)
    print(f"  Design matrix: {X_np.shape} (samples x params); cols={param_names}")

    print("  --- OLS on stoichiometry ---")
    Y_stoich = stoichiometry_matrix[bio_cols].values
    betas_s, pvals_s, nobs_s, xtxinv_s = _run_ols_all_sites(Y_stoich, X_np)

    print("  --- OLS on raw phospho (log2-transformed) ---")
    raw_vals = raw_phospho_normalized[bio_cols].values.copy()
    raw_vals[raw_vals <= 0] = np.nan
    with np.errstate(divide="ignore"):
        Y_raw = np.log2(raw_vals)
    betas_r, pvals_r, nobs_r, xtxinv_r = _run_ols_all_sites(Y_raw, X_np)

    print("  --- Computing per-contrast LFC/SE/FDR ---")
    results_by_contrast = {}
    for contrast_name, coefs in CONTRAST_COEFS.items():
        c_vec = np.zeros(len(param_names))
        for param, weight in coefs.items():
            c_vec[param_names.index(param)] = weight

        lfc_s = betas_s @ c_vec
        var_c_s = np.einsum("p,ipq,q->i", c_vec, xtxinv_s, c_vec)
        residuals_s = Y_stoich - (X_np @ betas_s.T).T
        dof_s = nobs_s - len(param_names)
        dof_s[dof_s <= 0] = 1
        sigma2_s = np.nansum(residuals_s ** 2, axis=1) / dof_s
        se_contrast_s = np.sqrt(var_c_s * sigma2_s)
        t_contrast_s = lfc_s / se_contrast_s
        p_contrast_s = 2 * sp_stats.t.sf(np.abs(t_contrast_s), df=dof_s)
        fdr_s = _bh_fdr(p_contrast_s)

        lfc_r = betas_r @ c_vec
        var_c_r = np.einsum("p,ipq,q->i", c_vec, xtxinv_r, c_vec)
        residuals_r = Y_raw - (X_np @ betas_r.T).T
        dof_r = nobs_r - len(param_names)
        dof_r[dof_r <= 0] = 1
        sigma2_r = np.nansum(residuals_r ** 2, axis=1) / dof_r
        se_contrast_r = np.sqrt(var_c_r * sigma2_r)
        t_contrast_r = lfc_r / se_contrast_r
        p_contrast_r = 2 * sp_stats.t.sf(np.abs(t_contrast_r), df=dof_r)
        fdr_r = _bh_fdr(p_contrast_r)

        results_by_contrast[contrast_name] = {
            "stoich_lfc": lfc_s, "stoich_pval": p_contrast_s, "stoich_fdr": fdr_s,
            "raw_lfc": lfc_r, "raw_pval": p_contrast_r, "raw_fdr": fdr_r,
        }

        thresh = config.SITE_FDR_DIAGNOSTIC_THRESH
        n_sig_s = int(np.sum(fdr_s < thresh))
        n_sig_r = int(np.sum(fdr_r < thresh))
        print(f"  {contrast_name}: stoich {n_sig_s} sig sites (FDR<{thresh}), "
              f"raw {n_sig_r} sig sites")

    site_results = pd.DataFrame({
        "site_id": stoichiometry_matrix["site_id"].values,
        "gene_symbol": stoichiometry_matrix["gene_symbol"].values,
        "matched_protein": stoichiometry_matrix["matched_protein"].values,
        "n_obs_stoich": nobs_s,
    })
    for cn in CONTRAST_COEFS:
        res = results_by_contrast[cn]
        site_results[f"stoich_lfc_{cn}"] = res["stoich_lfc"]
        site_results[f"stoich_pval_{cn}"] = res["stoich_pval"]
        site_results[f"stoich_fdr_{cn}"] = res["stoich_fdr"]
        site_results[f"raw_lfc_{cn}"] = res["raw_lfc"]
        site_results[f"raw_pval_{cn}"] = res["raw_pval"]
        site_results[f"raw_fdr_{cn}"] = res["raw_fdr"]

    return site_results, results_by_contrast


def run_mea(stoichiometry_matrix: pd.DataFrame, results_by_contrast: dict,
            track: str):
    """MEA wrapper. Returns (mea_df, shift_df, winsorized_df, substrate_df)."""
    print(f"  --- MEA kinase enrichment (track={track}) ---")
    return _run_mea(
        stoichiometry_matrix["motif"], results_by_contrast, "stoich_lfc",
        site_ids=stoichiometry_matrix["site_id"].values,
        gene_symbols=stoichiometry_matrix["gene_symbol"].values,
        track=track,
    )

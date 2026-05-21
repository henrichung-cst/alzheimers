"""Stage 2: males-only factorial OLS per cluster, per track.

Mirrors the live pipeline's design (const, App, Tau, Int, time_4mo,
time_6mo, App×time4, App×time6, Tau×time4, Tau×time6) and contrast
linear combinations. Operates on log2(deconvoluted intensity).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

import config
from deconvolution import paths
from deconvolution.load_deconvoluted import (
    DeconvoluatedTrack, males_only, parse_sample_metadata, safe_log2,
)

PARAM_NAMES = [
    "const", "App", "Tau", "Int",
    "time_4mo", "time_6mo",
    "App_x_time4", "App_x_time6", "Tau_x_time4", "Tau_x_time6",
]


def build_design(samples: list[str]) -> tuple[np.ndarray, list[str]]:
    """Build the males-only design matrix for the given sample order."""
    meta = parse_sample_metadata(samples).set_index("sample").loc[samples]

    X = pd.DataFrame(index=range(len(samples)))
    X["const"] = 1.0
    for idx, factor in enumerate(("App", "Tau", "Int")):
        X[factor] = meta["genotype"].map(
            lambda g, i=idx: config.SAP_FACTORIAL[g][i]
        ).astype(float).values
    X["time_4mo"] = (meta["timepoint"].values == "4mo").astype(float)
    X["time_6mo"] = (meta["timepoint"].values == "6mo").astype(float)
    X["App_x_time4"] = X["App"] * X["time_4mo"]
    X["App_x_time6"] = X["App"] * X["time_6mo"]
    X["Tau_x_time4"] = X["Tau"] * X["time_4mo"]
    X["Tau_x_time6"] = X["Tau"] * X["time_6mo"]
    return X[PARAM_NAMES].values, PARAM_NAMES


def _ols_batch(Y: np.ndarray, X: np.ndarray, XtX_inv: np.ndarray):
    """Vectorized OLS. Y: (n_sites, n_samples). Returns betas, sigma2, dof."""
    n_samples, n_params = X.shape
    dof = n_samples - n_params

    finite = np.all(np.isfinite(Y), axis=1)
    n_sites = Y.shape[0]
    betas = np.full((n_sites, n_params), np.nan)
    sigma2 = np.full(n_sites, np.nan)

    if finite.sum() == 0:
        return betas, sigma2, dof

    Y_c = Y[finite]
    B = (XtX_inv @ X.T @ Y_c.T).T
    resid = Y_c - B @ X.T
    s2 = np.sum(resid ** 2, axis=1) / dof

    betas[finite] = B
    sigma2[finite] = s2
    return betas, sigma2, dof


def _contrast_lfc_se(betas: np.ndarray, sigma2: np.ndarray,
                     XtX_inv: np.ndarray, contrast_vec: np.ndarray, dof: int):
    var_scale = float(contrast_vec @ XtX_inv @ contrast_vec)
    se_contrast = np.sqrt(sigma2 * var_scale)
    lfc = betas @ contrast_vec
    t = lfc / np.where(se_contrast > 0, se_contrast, np.inf)
    p = 2 * sp_stats.t.sf(np.abs(t), df=dof)
    return lfc, se_contrast, p


def _contrast_vector(contrast_name: str) -> np.ndarray:
    spec = paths.CONTRASTS[contrast_name]
    v = np.zeros(len(PARAM_NAMES))
    for k, val in spec.items():
        v[PARAM_NAMES.index(k)] = float(val)
    return v


def run_track(track: DeconvoluatedTrack,
              clusters: list[str] | None = None) -> pd.DataFrame:
    """Run males-only factorial OLS for each cluster across all sites.

    Returns a long DataFrame: site × cluster × contrast → lfc/se/pval (raw
    pre-FDR; FDR is applied in the MEA stage at site-level if needed, but
    MEA pre-rank uses LFC directly so per-cluster FDR is reported only
    for the kinase summary).
    """
    if clusters is None:
        clusters = track.clusters

    male_samples = males_only(track.samples)
    X, _ = build_design(male_samples)
    XtX_inv = np.linalg.inv(X.T @ X)
    contrast_vecs = {c: _contrast_vector(c) for c in paths.CONTRASTS}

    site_id = track.site_id().values
    motif = track.meta["motif"].astype(str).values
    gene_symbol = track.meta["gene_symbol"].astype(str).values
    n_sites = len(site_id)

    available = set(track.values.columns)
    out_frames = []
    for ci, cluster in enumerate(clusters, 1):
        cols_present = [(s, cluster) for s in male_samples
                        if (s, cluster) in available]
        if len(cols_present) < len(male_samples):
            continue
        sub = track.values.reindex(columns=cols_present)
        Y = safe_log2(sub.values.astype(float))
        betas, sigma2, dof = _ols_batch(Y, X, XtX_inv)

        n_contr = len(contrast_vecs)
        lfc_arr = np.empty(n_sites * n_contr)
        se_arr = np.empty(n_sites * n_contr)
        p_arr = np.empty(n_sites * n_contr)
        contrast_arr = np.empty(n_sites * n_contr, dtype=object)
        for j, (contrast_name, cvec) in enumerate(contrast_vecs.items()):
            lfc, se_c, pval = _contrast_lfc_se(betas, sigma2, XtX_inv, cvec, dof)
            sl = slice(j * n_sites, (j + 1) * n_sites)
            lfc_arr[sl] = lfc
            se_arr[sl] = se_c
            p_arr[sl] = pval
            contrast_arr[sl] = contrast_name

        out_frames.append(pd.DataFrame({
            "site_id": np.tile(site_id, n_contr),
            "gene_symbol": np.tile(gene_symbol, n_contr),
            "motif": np.tile(motif, n_contr),
            "cluster": cluster,
            "contrast": contrast_arr,
            "lfc": lfc_arr,
            "se": se_arr,
            "pval": p_arr,
            "track": track.track,
        }))
        if ci % 5 == 0 or ci == len(clusters):
            print(f"    [{track.track}] OLS done for {ci}/{len(clusters)} clusters")

    if not out_frames:
        return pd.DataFrame()
    return pd.concat(out_frames, ignore_index=True)

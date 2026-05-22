"""Stage 2 of kinase attribution: OLS site-level models + MEA kinase enrichment.

Inputs (under outputs/reports/kinase_attribution/, written by Stage 1):
  stoichiometry_matrix.csv (track-suffixed)
  raw_phospho_normalized.csv (track-suffixed)
  outputs/reports/data_ingest/sample_mapping.csv
  outputs/reports/data_ingest/sample_exclusions.csv (optional)
  alz/config.py

Outputs (under outputs/reports/kinase_attribution/, track-suffixed):
  mea_stoichiometry.csv
  mea_global_shift.csv
  winsorized_sites.csv
  mea_substrate_sets.csv
  site_level_ols.csv
"""

import os
import sys
from pathlib import Path

# Bootstrap project root onto sys.path so direct invocation
# (`python alz/kinase_enrich.py`) can resolve `from alz import config`.
# Phase 2's bridge adds `alz/` to sys.path; this adds the parent.
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import pandas as pd
import yaml
from scipy import stats as sp_stats
from statsmodels.stats.multitest import multipletests

from alz import config

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OUTPUT_DIR = config.KINASE_ATTRIBUTION_OUTPUT_DIR
DATA_INGEST_DIR = config.DATA_INGEST_OUTPUT_DIR

# Canonical factorial coding + contrasts live in config. Re-export
# CONTRAST_COEFS so existing import sites (e.g. decomposition/enrich_celltype)
# keep working.
CONTRAST_COEFS = config.CONTRAST_COEFS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_sample_mapping():
    """Load sample mapping from data ingestion stage."""
    path = os.path.join(DATA_INGEST_DIR, "sample_mapping.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Sample mapping not found at {path}. Run data_ingest.py --mapping first."
        )
    return pd.read_csv(path)


def load_sample_exclusions():
    """Load excluded mouse IDs from outlier analysis."""
    path = os.path.join(DATA_INGEST_DIR, "sample_exclusions.csv")
    if not os.path.exists(path):
        return set()
    df = pd.read_csv(path)
    return set(df.loc[df["excluded"], "mouse_id"])


def _filter_samples(mapping, analysis_mode=None):
    """Apply outlier exclusion + sex filter. analysis_mode falls back to config.ANALYSIS_MODE."""
    if analysis_mode is None:
        analysis_mode = config.ANALYSIS_MODE
    excluded = load_sample_exclusions()
    n0 = len(mapping)
    filt = mapping[~mapping["mouse_id"].isin(excluded)].copy()
    n_excl = n0 - len(filt)
    if analysis_mode == "males_only":
        filt = filt[filt["sex"] == "M"].copy()
    n_final = len(filt)
    print(f"  Sample filter ({analysis_mode}): {n0} -> {n_final} "
          f"({n_excl} outliers excluded, {n0 - n_excl - n_final} sex-filtered)")
    return filt


def _resolve_track(track):
    """Look up a phospho-track config by name; return the dict from config."""
    if isinstance(track, dict):
        return track
    if track not in config.PHOSPHO_TRACKS:
        raise ValueError(
            f"Unknown phospho track {track!r}; "
            f"valid: {list(config.PHOSPHO_TRACKS)}"
        )
    return config.PHOSPHO_TRACKS[track]


def _track_output(filename, track_cfg):
    """Compose an output path with the track's suffix appended before the extension."""
    cfg = _resolve_track(track_cfg)
    suffix = cfg["output_suffix"]
    if not suffix:
        return os.path.join(OUTPUT_DIR, filename)
    base, ext = os.path.splitext(filename)
    return os.path.join(OUTPUT_DIR, f"{base}{suffix}{ext}")


def _bh_fdr(pvals):
    """Benjamini-Hochberg FDR correction, NaN-safe."""
    valid = ~np.isnan(pvals)
    result = np.full_like(pvals, np.nan)
    if valid.sum() == 0:
        return result
    _, adj, _, _ = multipletests(pvals[valid], method="fdr_bh")
    result[valid] = adj
    return result


def _build_design_matrix(mapping, bio_cols, analysis_mode=None):
    """Build the factorial OLS design matrix.

    males_only mode:  N x 10 (const, App, Tau, Int, time_4mo, time_6mo,
                               App_x_time4, App_x_time6, Tau_x_time4, Tau_x_time6)
    full_cohort mode: N x 11 (adds 'female' column)
    """
    if analysis_mode is None:
        analysis_mode = config.ANALYSIS_MODE
    meta = mapping.set_index("column_name").loc[bio_cols].reset_index()

    X = pd.DataFrame(index=range(len(bio_cols)))
    X["const"] = 1.0
    for idx, factor in enumerate(("App", "Tau", "Int")):
        X[factor] = meta["genotype"].map(
            lambda g, i=idx: config.SAP_FACTORIAL[g][i]).astype(float)

    if analysis_mode != "males_only":
        X["female"] = (meta["sex"] == "F").astype(float)

    X["time_4mo"] = (meta["timepoint"] == "4mo").astype(float)
    X["time_6mo"] = (meta["timepoint"] == "6mo").astype(float)

    X["App_x_time4"] = X["App"] * X["time_4mo"]
    X["App_x_time6"] = X["App"] * X["time_6mo"]
    X["Tau_x_time4"] = X["Tau"] * X["time_4mo"]
    X["Tau_x_time6"] = X["Tau"] * X["time_6mo"]

    return X


def _run_ols_all_sites(Y, X):
    """Vectorized OLS for all sites.

    Returns
    -------
    betas, pvals, n_obs, xtxinv_per_site
        ``xtxinv_per_site`` has shape (n_sites, n_params, n_params): the
        global ``(X'X)^{-1}`` broadcast for complete-data sites and the
        per-site ``(X_i' X_i)^{-1}`` for partial-data sites. Contrast SE
        for partial sites needs the site's own design covariance — using
        the global one understates SE and inflates t-stats.
    """
    n_sites, n_samples = Y.shape
    n_params = X.shape[1]
    betas = np.full((n_sites, n_params), np.nan)
    pvals = np.full((n_sites, n_params), np.nan)
    n_obs = np.zeros(n_sites, dtype=int)
    xtxinv_per_site = np.full((n_sites, n_params, n_params), np.nan)

    complete_mask = np.all(np.isfinite(Y), axis=1)
    n_complete = complete_mask.sum()

    XtX_inv = np.linalg.inv(X.T @ X)

    if n_complete > 0:
        Y_c = Y[complete_mask]
        B_c = (XtX_inv @ X.T @ Y_c.T).T
        residuals = Y_c - (X @ B_c.T).T
        dof = n_samples - n_params
        sigma2 = np.sum(residuals ** 2, axis=1) / dof
        cov_diag = np.diag(XtX_inv)
        se = np.sqrt(np.outer(sigma2, cov_diag))
        t_stats = B_c / se
        p_c = 2 * sp_stats.t.sf(np.abs(t_stats), df=dof)

        betas[complete_mask] = B_c
        pvals[complete_mask] = p_c
        n_obs[complete_mask] = n_samples
        xtxinv_per_site[complete_mask] = XtX_inv

    partial_idx = np.where(~complete_mask)[0]
    for i in partial_idx:
        valid = np.isfinite(Y[i])
        n_valid = valid.sum()
        if n_valid < n_params + 2:
            continue
        Xi = X[valid]
        yi = Y[i, valid]
        try:
            XtX_inv_i = np.linalg.inv(Xi.T @ Xi)
        except np.linalg.LinAlgError:
            continue
        bi = XtX_inv_i @ Xi.T @ yi
        resid = yi - Xi @ bi
        dof = n_valid - n_params
        s2 = np.sum(resid ** 2) / dof
        se_i = np.sqrt(np.diag(XtX_inv_i) * s2)
        t_i = bi / se_i
        p_i = 2 * sp_stats.t.sf(np.abs(t_i), df=dof)
        betas[i] = bi
        pvals[i] = p_i
        n_obs[i] = n_valid
        xtxinv_per_site[i] = XtX_inv_i

    print(f"  OLS: {n_complete} complete-data sites (fast), "
          f"{len(partial_idx)} partial-data sites")
    print(f"  Sites with valid fits: {np.sum(np.isfinite(betas[:, 0]))}")

    return betas, pvals, n_obs, xtxinv_per_site


def _winsorize_lfc(lfc_array, pct=None):
    """Winsorize LFC values at the given percentile to limit outlier influence."""
    if pct is None:
        pct = config.MEA_WINSORIZE_PERCENTILE
    lower = np.nanpercentile(lfc_array, pct)
    upper = np.nanpercentile(lfc_array, 100 - pct)
    outlier_mask = (lfc_array < lower) | (lfc_array > upper)
    clipped = np.clip(lfc_array, lower, upper)
    return clipped, outlier_mask, lower, upper


def _run_mea(motif_series, results_by_contrast, lfc_key,
             site_ids=None, gene_symbols=None, track="st"):
    """Run MEA (GSEA-based) kinase enrichment across contrasts.

    Median-centers and winsorizes (1st/99th pct) per contrast before GSEA
    pre-rank. Returns a 4-tuple ``(mea_df, shift_df, winsorized_df,
    substrate_df)``; callers (CLI shim or Kedro nodes) decide where to
    persist them.
    """
    from kinase_library import RankedPhosData, create_kin_sub_sets
    from kinase_library.utils._global_vars import kl_method_comp_direction_dict

    track_cfg = _resolve_track(track)
    enrichment_results = {}
    outlier_records = []
    shift_records = []
    substrate_records = []
    kl_comp_direction = kl_method_comp_direction_dict[config.KL_METHOD]
    for contrast_name, res in results_by_contrast.items():
        raw_lfc = res[lfc_key].copy()

        median_shift = float(np.nanmedian(raw_lfc))
        centered_lfc = raw_lfc - median_shift
        shift_records.append({
            "contrast": contrast_name,
            "median_shift": median_shift,
            "mean_before": float(np.nanmean(raw_lfc)),
            "pct_pos_before": float(np.nanmean(raw_lfc > 0) * 100),
            "pct_pos_after": float(np.nanmean(centered_lfc > 0) * 100),
        })
        if abs(median_shift) > 0.001:
            print(f"  {contrast_name}: centered by {median_shift:+.4f} "
                  f"(pos sites: {shift_records[-1]['pct_pos_before']:.0f}% "
                  f"-> {shift_records[-1]['pct_pos_after']:.0f}%)")

        clipped_lfc, outlier_mask, lo, hi = _winsorize_lfc(centered_lfc)
        n_clipped = np.nansum(outlier_mask)
        if n_clipped > 0:
            print(f"  {contrast_name}: winsorized {int(n_clipped)} sites "
                  f"to [{lo:.3f}, {hi:.3f}]")
            idxs = np.where(outlier_mask)[0]
            for idx in idxs:
                rec = {"contrast": contrast_name,
                       "original_lfc": raw_lfc[idx],
                       "clipped_lfc": clipped_lfc[idx],
                       "lower_bound": lo, "upper_bound": hi}
                if site_ids is not None:
                    rec["site_id"] = site_ids[idx]
                if gene_symbols is not None:
                    rec["gene_symbol"] = gene_symbols[idx]
                outlier_records.append(rec)

        enrich_df = pd.DataFrame({
            "motif": motif_series.values,
            "log2_fold_change": clipped_lfc,
        })
        enrich_df = enrich_df.dropna(subset=["log2_fold_change"])
        enrich_df = enrich_df[enrich_df["motif"].notna() &
                              (enrich_df["motif"] != "")]
        if len(enrich_df) < config.MEA_MIN_SITES:
            print(f"  WARNING: only {len(enrich_df)} sites for "
                  f"{contrast_name} (< {config.MEA_MIN_SITES}), skipping")
            continue
        rpd = RankedPhosData(
            dp_data=enrich_df,
            rank_col="log2_fold_change",
            seq_col="motif",
        )
        result = rpd.mea(
            kin_type=track_cfg["kin_type"],
            kl_method=config.KL_METHOD,
            kl_thresh=track_cfg["kl_thresh"],
            permutation_num=config.MEA_PERMUTATION_NUM,
            seed=config.MEA_SEED,
        )
        er = result.enrichment_results.copy()
        er.index.name = "kinase"
        er = er.reset_index()
        er["contrast"] = contrast_name
        er["residue_type"] = track_cfg["residue"]
        er["track"] = track_cfg["name"]
        enrichment_results[contrast_name] = er
        n_sig = (er["FDR"] < config.MEA_FDR_THRESH).sum()
        print(f"  {contrast_name}: {len(er)} kinases tested, "
              f"{n_sig} significant (FDR<{config.MEA_FDR_THRESH})")

        kin_sub_sets = create_kin_sub_sets(
            rpd.data_kl_values, threshold=track_cfg["kl_thresh"],
            comp_direction=kl_comp_direction,
        )
        # `data_kl_values` carries `*_percentile_ranks` (1-K substrate ranks);
        # we want the underlying `*_percentiles` (0-100 substrate-vs-kinase
        # match strength), which is set in parallel on dp_data_pps.
        pct_attr = f"{track_cfg['kin_type']}_percentiles"
        pct_values = getattr(rpd.dp_data_pps, pct_attr)
        # Same motif can appear at multiple sites; all duplicates carry the
        # same percentile (deterministic from the PSSM), so take the first.
        pct_by_motif = pct_values[~pct_values.index.duplicated(keep="first")]
        for kinase_name, motifs in kin_sub_sets.items():
            for motif in motifs:
                substrate_records.append({
                    "kinase": kinase_name,
                    "contrast": contrast_name,
                    "motif": motif,
                    "residue_type": track_cfg["residue"],
                    "track": track_cfg["name"],
                    "kl_percentile": float(pct_by_motif.at[motif, kinase_name]),
                })

    outlier_df = pd.DataFrame(outlier_records)
    shift_df = pd.DataFrame(shift_records)
    substrate_df = pd.DataFrame(substrate_records)
    if enrichment_results:
        mea_df = pd.concat(enrichment_results.values(), ignore_index=True)
    else:
        mea_df = pd.DataFrame()
    return mea_df, shift_df, outlier_df, substrate_df


def _prepare_raw_ols(mapping, bio_cols, raw_df):
    """Shared OLS preparation for raw phospho data: log2-transform + run OLS."""
    X = _build_design_matrix(mapping, bio_cols)
    X_np = X.values
    param_names = list(X.columns)
    raw_vals = raw_df[bio_cols].values.copy()
    raw_vals[raw_vals <= 0] = np.nan
    with np.errstate(divide="ignore"):
        Y_raw = np.log2(raw_vals)
    betas_r, pvals_r, nobs_r, xtxinv_r = _run_ols_all_sites(Y_raw, X_np)
    return X, X_np, param_names, Y_raw, betas_r, pvals_r, nobs_r, xtxinv_r


# ===========================================================================
# CLI
# ===========================================================================

def _load_params():
    """Load parameters from conf/base/parameters.yml with optional KEDRO_ENV overlay."""
    project_root = Path(__file__).resolve().parent.parent
    params_path = project_root / "conf" / "base" / "parameters.yml"
    with open(params_path) as f:
        params = yaml.safe_load(f)
    env = os.environ.get("KEDRO_ENV")
    if env:
        overlay_path = project_root / "conf" / env / "parameters.yml"
        if overlay_path.exists():
            with open(overlay_path) as f:
                params.update(yaml.safe_load(f))
    return params


def _fit_and_contrast(stoich_df, raw_df, mapping, analysis_mode, contrast_coefs):
    """Build design matrix, run OLS on stoich + raw, compute per-contrast LFC/SE/FDR."""
    from scipy import stats as sp_stats

    bio_cols = mapping["column_name"].tolist()
    X = _build_design_matrix(mapping, bio_cols, analysis_mode=analysis_mode)
    X_np = X.values
    param_names = list(X.columns)
    print(f"  Design matrix: {X_np.shape} (samples x params); cols={param_names}")

    print("  --- OLS on stoichiometry ---")
    Y_stoich = stoich_df[bio_cols].values
    betas_s, pvals_s, nobs_s, xtxinv_s = _run_ols_all_sites(Y_stoich, X_np)

    print("  --- OLS on raw phospho (log2-transformed) ---")
    raw_vals = raw_df[bio_cols].values.copy()
    raw_vals[raw_vals <= 0] = np.nan
    with np.errstate(divide="ignore"):
        Y_raw = np.log2(raw_vals)
    betas_r, pvals_r, nobs_r, xtxinv_r = _run_ols_all_sites(Y_raw, X_np)

    print("  --- Computing per-contrast LFC/SE/FDR ---")
    results_by_contrast = {}
    for contrast_name, coefs in contrast_coefs.items():
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
        "site_id": stoich_df["site_id"].values,
        "gene_symbol": stoich_df["gene_symbol"].values,
        "matched_protein": stoich_df["matched_protein"].values,
        "n_obs_stoich": nobs_s,
    })
    for cn in contrast_coefs:
        res = results_by_contrast[cn]
        site_results[f"stoich_lfc_{cn}"] = res["stoich_lfc"]
        site_results[f"stoich_pval_{cn}"] = res["stoich_pval"]
        site_results[f"stoich_fdr_{cn}"] = res["stoich_fdr"]
        site_results[f"raw_lfc_{cn}"] = res["raw_lfc"]
        site_results[f"raw_pval_{cn}"] = res["raw_pval"]
        site_results[f"raw_fdr_{cn}"] = res["raw_fdr"]

    return site_results, results_by_contrast


def main():
    """Run OLS + MEA enrichment directly for both tracks."""
    _ensure_output_dir()
    params = _load_params()
    analysis_mode = params.get("analysis_mode", config.ANALYSIS_MODE)

    print(f"\n=== Stage 2: OLS + MEA Kinase Enrichment ({analysis_mode}) ===\n")

    mapping_full = load_sample_mapping()
    filtered_mapping = _filter_samples(mapping_full, analysis_mode=analysis_mode)

    for track_name in config.PHOSPHO_TRACKS:
        track_cfg = config.PHOSPHO_TRACKS[track_name]
        print(f"\n--- Track: {track_name} ({track_cfg['label']}) ---")

        stoich_path = _track_output("stoichiometry_matrix.csv", track_cfg)
        raw_path = _track_output("raw_phospho_normalized.csv", track_cfg)

        if not os.path.exists(stoich_path):
            print(f"  {stoich_path} not found; skipping track.")
            continue
        if not os.path.exists(raw_path):
            print(f"  {raw_path} not found; skipping track.")
            continue

        stoich_df = pd.read_csv(stoich_path)
        raw_df = pd.read_csv(raw_path)

        site_ols, results_by_contrast = _fit_and_contrast(
            stoich_df, raw_df, filtered_mapping,
            analysis_mode, config.CONTRAST_COEFS)

        print(f"  --- MEA kinase enrichment (track={track_name}) ---")
        mea_df, shift_df, winsorized_df, substrate_df = _run_mea(
            stoich_df["motif"], results_by_contrast, "stoich_lfc",
            site_ids=stoich_df["site_id"].values,
            gene_symbols=stoich_df["gene_symbol"].values,
            track=track_name,
        )

        site_ols.to_csv(_track_output("site_level_ols.csv", track_cfg), index=False)
        if mea_df is not None and len(mea_df) > 0:
            mea_df.to_csv(
                _track_output("mea_stoichiometry.csv", track_cfg), index=False)
        shift_df.to_csv(
            _track_output("mea_global_shift.csv", track_cfg), index=False)
        winsorized_df.to_csv(
            _track_output("winsorized_sites.csv", track_cfg), index=False)
        substrate_df.to_csv(
            _track_output("mea_substrate_sets.csv", track_cfg), index=False)

        print(f"  [{track_name}] Saved site_level_ols, mea_stoichiometry, "
              "mea_global_shift, winsorized_sites, mea_substrate_sets")

    print("\nStage 2 complete.")


if __name__ == "__main__":
    main()

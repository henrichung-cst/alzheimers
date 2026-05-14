"""Stage 7 — per-cluster factorial OLS + MEA on the projected phospho cube.

Consumes Stage 6's `phospho_per_cluster.parquet` (one row per
(site, cluster, animal)) and runs the bulk-pipeline OLS + MEA for each
cluster on the same 9 factorial contrasts, reusing helpers from
`alz.kinase_enrich`.

For each cluster:
  1. pivot to (site × column_name) on `log2_value`
  2. filter samples (outlier exclusion + analysis_mode sex filter)
  3. skip if design loses rank or no contrasts are estimable
  4. factorial OLS → 9 contrasts (App/Tau/ApTt × 2/4/6mo)
  5. median-center + winsorize each contrast LFC and run GSEA MEA
     (kinase-library RankedPhosData.mea)

Outputs (under `outputs/reports/decomposition/{spine}/`, track-suffixed):
  - `mea_per_cluster.parquet`            — long: cluster, contrast, kinase, NES, pval, FDR, ...
  - `site_level_ols_per_cluster.parquet` — long: cluster, site_id, contrast, lfc, se, pval, fdr
  - `mea_global_shift_per_cluster.csv`   — per (cluster, contrast) median shift / winsorization summary
  - `winsorized_sites_per_cluster.csv`   — clipped sites with original/clipped LFCs
  - `mea_substrate_sets_per_cluster.csv` — substrate sets used per (cluster, contrast, kinase)
  - `enrich_audit.json`                  — per-cluster status (ok / skipped + reason)

Single-track (default `st`); pass `--track py` for the tyrosine track.
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
from kinase_enrich import (  # noqa: E402
    CONTRAST_COEFS,
    _bh_fdr,
    _build_design_matrix,
    _filter_samples,
    _run_mea,
    _run_ols_all_sites,
    _resolve_track,
)
from scipy import stats as sp_stats  # noqa: E402

REPO = Path(config.REPO_ROOT)
DEC_ROOT = REPO / "outputs/reports/decomposition"
BULK_DIR = REPO / "outputs/reports/kinase_attribution"


def _spine_dir(spine: str) -> Path:
    return DEC_ROOT / spine


def _track_phospho_csv(track_cfg: dict) -> Path:
    suffix = track_cfg["output_suffix"]
    base = "raw_phospho_normalized"
    name = f"{base}{suffix}.csv" if suffix else f"{base}.csv"
    return BULK_DIR / name


def _site_motif_map(track_cfg: dict) -> pd.DataFrame:
    path = _track_phospho_csv(track_cfg)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing — Stage 1 must emit raw phospho before Stage 7."
        )
    df = pd.read_csv(path, usecols=["site_id", "gene_symbol", "motif"])
    return df.drop_duplicates(subset=["site_id"])


def _contrast_lfc_se(
    betas: np.ndarray,
    xtxinv: np.ndarray,
    sigma2: np.ndarray,
    param_names: list[str],
    coef_map: dict[str, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Effective LFC + SE for one contrast across all sites."""
    n_sites, n_params = betas.shape
    c = np.zeros(n_params)
    for name, w in coef_map.items():
        if name not in param_names:
            return np.full(n_sites, np.nan), np.full(n_sites, np.nan)
        c[param_names.index(name)] = w
    lfc = betas @ c
    # var_i = c' XtXinv_i c * sigma2_i  ; xtxinv has shape (n_sites, p, p)
    var = np.einsum("i,sij,j->s", c, xtxinv, c) * sigma2
    se = np.sqrt(np.where(np.isfinite(var) & (var >= 0), var, np.nan))
    return lfc, se


def _ols_for_cluster(
    cluster: str,
    phospho_long: pd.DataFrame,
    mapping: pd.DataFrame,
    filt_cols: set[str],
    motif_map: pd.DataFrame,
    analysis_mode: str,
):
    """Run OLS + per-contrast LFC/SE for one cluster.

    On success returns
        (audit, mea_input, site_ols_long, motif_series)
    where `mea_input` is dict[contrast → DataFrame({'lfc': ...})] suitable
    for `_run_mea(..., lfc_key='lfc')`. On skip returns 3-tuple
    (audit, {}, empty DataFrame).
    """
    sub = phospho_long[phospho_long["cluster"] == cluster]
    if sub.empty:
        return {"status": "skipped", "reason": "no rows for cluster"}, {}, pd.DataFrame()

    a2c = dict(zip(mapping["animal_id"], mapping["column_name"]))
    sub = sub.assign(column_name=sub["animal_id"].map(a2c))
    sub = sub.dropna(subset=["column_name"])
    bio_cols_all = sub["column_name"].unique().tolist()
    keep_cols = [c for c in bio_cols_all if c in filt_cols]
    if len(keep_cols) < 12:
        return ({"status": "skipped",
                 "reason": f"only {len(keep_cols)} samples post-filter"},
                {}, pd.DataFrame())

    wide = (
        sub[sub["column_name"].isin(keep_cols)]
        .pivot_table(index=["site_id", "gene_symbol"],
                     columns="column_name", values="log2_value",
                     aggfunc="first")
        .reindex(columns=keep_cols)
    )
    wide = wide.dropna(how="all")
    if wide.empty:
        return ({"status": "skipped", "reason": "no sites with any data"},
                {}, pd.DataFrame())

    site_meta = wide.index.to_frame(index=False)
    Y = wide.values.astype(float)

    X = _build_design_matrix(mapping, keep_cols, analysis_mode=analysis_mode)
    X_np = X.values
    param_names = list(X.columns)
    rank = int(np.linalg.matrix_rank(X_np))
    if rank < X_np.shape[1]:
        return ({"status": "skipped",
                 "reason": f"design rank {rank} < {X_np.shape[1]}"},
                {}, pd.DataFrame())

    betas, _, n_obs, xtxinv = _run_ols_all_sites(Y, X_np)

    # site-level residual variance for per-contrast SE
    n_sites = Y.shape[0]
    sigma2 = np.full(n_sites, np.nan)
    fit_idx = np.where(np.isfinite(betas[:, 0]))[0]
    for i in fit_idx:
        mask = np.isfinite(Y[i])
        if mask.sum() <= X_np.shape[1]:
            continue
        Xi = X_np[mask]
        yi = Y[i, mask]
        resid = yi - Xi @ betas[i]
        dof = mask.sum() - X_np.shape[1]
        sigma2[i] = (resid @ resid) / dof

    motif_lookup = dict(zip(motif_map["site_id"], motif_map["motif"]))
    motif_series = pd.Series(
        [motif_lookup.get(s) for s in site_meta["site_id"].values]
    )

    site_rows = []
    mea_input = {}
    for contrast, coefs in CONTRAST_COEFS.items():
        lfc, se = _contrast_lfc_se(betas, xtxinv, sigma2, param_names, coefs)
        with np.errstate(divide="ignore", invalid="ignore"):
            t = lfc / se
        dof = np.maximum(n_obs - X_np.shape[1], 1)
        p = 2 * sp_stats.t.sf(np.abs(t), df=dof)
        fdr = _bh_fdr(p)
        mea_input[contrast] = pd.DataFrame({"lfc": lfc})
        site_rows.append(pd.DataFrame({
            "cluster": cluster,
            "contrast": contrast,
            "site_id": site_meta["site_id"].values,
            "gene_symbol": site_meta["gene_symbol"].values,
            "lfc": lfc,
            "se": se,
            "t": t,
            "pval": p,
            "fdr": fdr,
            "n_obs": n_obs,
        }))

    site_ols_long = pd.concat(site_rows, ignore_index=True)
    audit = {
        "status": "ok",
        "n_samples_kept": len(keep_cols),
        "n_sites_fit": int(len(fit_idx)),
        "design_rank": rank,
        "design_n_params": int(X_np.shape[1]),
    }
    return audit, mea_input, site_ols_long, motif_series, site_meta


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--spine", default="levy19")
    ap.add_argument("--track", default="st", choices=list(config.PHOSPHO_TRACKS))
    ap.add_argument("--analysis-mode", default=None,
                    help="override config.ANALYSIS_MODE (males_only / full_cohort)")
    args = ap.parse_args()

    track_cfg = _resolve_track(args.track)
    analysis_mode = args.analysis_mode or config.ANALYSIS_MODE
    out_dir = _spine_dir(args.spine)
    phospho_path = out_dir / "phospho_per_cluster.parquet"
    if not phospho_path.exists():
        raise FileNotFoundError(
            f"{phospho_path} missing — run Stage 6 "
            f"(alz/decomposition/build_celltype_decomposition.py) first."
        )

    print(f"Stage 7 — per-cluster MEA on spine: {args.spine} "
          f"(track={args.track}, analysis_mode={analysis_mode})")
    phospho_long = pd.read_parquet(phospho_path)
    print(f"  Loaded {len(phospho_long):,} rows; "
          f"{phospho_long['cluster'].nunique()} clusters; "
          f"{phospho_long['site_id'].nunique():,} sites")

    mapping = pd.read_csv(REPO / "outputs/reports/data_ingest/sample_mapping.csv")
    motif_map = _site_motif_map(track_cfg)
    # pre-compute the sex/outlier-filtered column set once
    filt_cols = set(
        _filter_samples(mapping, analysis_mode=analysis_mode)["column_name"]
    )

    clusters = sorted(phospho_long["cluster"].unique())
    print(f"  Iterating {len(clusters)} clusters ...\n")

    all_mea, all_shift, all_wins, all_substrate, all_site_ols = [], [], [], [], []
    audit_per_cluster = {}

    for cluster in clusters:
        print(f"--- cluster: {cluster} ---")
        result = _ols_for_cluster(
            cluster, phospho_long, mapping, filt_cols, motif_map, analysis_mode,
        )
        if len(result) == 3:
            audit, _, _ = result
            audit_per_cluster[cluster] = audit
            print(f"  SKIP: {audit.get('reason')}")
            continue
        audit, mea_input, site_ols, motif_series, _ = result
        audit_per_cluster[cluster] = audit
        all_site_ols.append(site_ols)

        mea_df, shift_df, wins_df, subs_df = _run_mea(
            motif_series=motif_series,
            results_by_contrast=mea_input,
            lfc_key="lfc",
            site_ids=None,
            gene_symbols=None,
            track=track_cfg,
        )
        for d in (mea_df, shift_df, wins_df, subs_df):
            if isinstance(d, pd.DataFrame) and not d.empty:
                d.insert(0, "cluster", cluster)
        if isinstance(mea_df, pd.DataFrame) and not mea_df.empty:
            all_mea.append(mea_df)
        if isinstance(shift_df, pd.DataFrame) and not shift_df.empty:
            all_shift.append(shift_df)
        if isinstance(wins_df, pd.DataFrame) and not wins_df.empty:
            all_wins.append(wins_df)
        if isinstance(subs_df, pd.DataFrame) and not subs_df.empty:
            all_substrate.append(subs_df)

    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = track_cfg["output_suffix"]

    mea_out = out_dir / f"mea_per_cluster{suffix}.parquet"
    site_out = out_dir / f"site_level_ols_per_cluster{suffix}.parquet"
    shift_out = out_dir / f"mea_global_shift_per_cluster{suffix}.csv"
    wins_out = out_dir / f"winsorized_sites_per_cluster{suffix}.csv"
    subs_out = out_dir / f"mea_substrate_sets_per_cluster{suffix}.csv"
    audit_path = out_dir / f"enrich_audit{suffix}.json"

    if all_mea:
        pd.concat(all_mea, ignore_index=True).to_parquet(mea_out, index=False)
        print(f"\nWrote {mea_out}")
    else:
        print("\nNo MEA results produced.")
    if all_site_ols:
        pd.concat(all_site_ols, ignore_index=True).to_parquet(site_out, index=False)
        print(f"Wrote {site_out}")
    if all_shift:
        pd.concat(all_shift, ignore_index=True).to_csv(shift_out, index=False)
    if all_wins:
        pd.concat(all_wins, ignore_index=True).to_csv(wins_out, index=False)
    if all_substrate:
        pd.concat(all_substrate, ignore_index=True).to_csv(subs_out, index=False)

    with open(audit_path, "w") as fh:
        json.dump({
            "spine": args.spine,
            "track": args.track,
            "analysis_mode": analysis_mode,
            "n_clusters": len(clusters),
            "n_processed": sum(1 for v in audit_per_cluster.values()
                               if v.get("status") == "ok"),
            "n_skipped": sum(1 for v in audit_per_cluster.values()
                             if v.get("status") == "skipped"),
            "per_cluster": audit_per_cluster,
        }, fh, indent=2)
    print(f"Wrote {audit_path}")


if __name__ == "__main__":
    main()

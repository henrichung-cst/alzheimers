"""Stage 3: two-track kinase MEA per cell type per contrast.

For each (cluster, contrast, track) triple, prerank phosphosites by
median-centered + winsorized LFC and run the kinase library MEA against
the track's substrate set (ser/thr or tyrosine).

Reuses the same preprocessing (median-center, winsorize) and library
parameters as the live pipeline so NES is comparable in distribution.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config
from deconvolution import paths


_TRACK_CFG = {
    "st": {"kin_type": "ser_thr", "label": "Ser/Thr"},
    "py": {"kin_type": "tyrosine", "label": "Tyr"},
}


def _winsorize(lfc: np.ndarray, pct: float):
    lo = np.nanpercentile(lfc, pct)
    hi = np.nanpercentile(lfc, 100 - pct)
    return np.clip(lfc, lo, hi)


def run_mea(site_ols: pd.DataFrame, track: str,
            permutation_num: int | None = None) -> pd.DataFrame:
    """Run MEA per (cluster, contrast) for a single track.

    Parameters
    ----------
    site_ols : DataFrame from factorial_ols.run_track (single track)
    track : "st" or "py"
    permutation_num : override config.MEA_PERMUTATION_NUM if given

    Returns DataFrame: kinase × wmb_class × contrast → NES, FDR, pval, track.
    """
    from kinase_library import RankedPhosData

    track_cfg = _TRACK_CFG[track]
    n_perm = permutation_num if permutation_num is not None else config.MEA_PERMUTATION_NUM

    site_ols = site_ols[site_ols["track"] == track]
    if site_ols.empty:
        return pd.DataFrame()

    out_rows = []
    groups = list(site_ols.groupby(["wmb_class", "contrast"]))
    for i, ((wmb_class, contrast), sub) in enumerate(groups, 1):
        if sub.empty:
            continue

        lfc = sub["lfc"].values.astype(float)
        if not np.any(np.isfinite(lfc)):
            continue

        lfc_centered = lfc - np.nanmedian(lfc)
        lfc_clipped = _winsorize(lfc_centered, config.MEA_WINSORIZE_PERCENTILE)

        prerank = pd.DataFrame({
            "motif": sub["motif"].values,
            "log2_fold_change": lfc_clipped,
        })
        prerank = prerank.dropna(subset=["log2_fold_change"])
        prerank = prerank[prerank["motif"].notna() & (prerank["motif"] != "")]
        if len(prerank) < 100:
            continue

        try:
            rpd = RankedPhosData(
                dp_data=prerank,
                rank_col="log2_fold_change",
                seq_col="motif",
            )
            result = rpd.mea(
                kin_type=track_cfg["kin_type"],
                kl_method=config.KL_METHOD,
                kl_thresh=config.KL_THRESH,
                permutation_num=n_perm,
                seed=config.MEA_SEED,
            )
        except Exception as e:
            print(f"    MEA FAILED [{track}][{wmb_class}][{contrast}]: {e}")
            continue

        er = result.enrichment_results.copy()
        er.index.name = "kinase"
        er = er.reset_index()
        er["wmb_class"] = wmb_class
        er["contrast"] = contrast
        er["track"] = track
        out_rows.append(er)

        if i % 25 == 0 or i == len(groups):
            print(f"    [{track}] MEA {i}/{len(groups)} (wmb_class, contrast) pairs")

    if not out_rows:
        return pd.DataFrame()
    return pd.concat(out_rows, ignore_index=True)

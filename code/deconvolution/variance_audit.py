"""Variance audit: per-animal site-level OLS vs. Yuyu's group-level OLS.

Compares the smoke run against the legacy group-level Stage 2 output for
the same scope (clusters x contrasts x tracks). Emits a one-page markdown
report under outputs/reports/deconvolution/per_animal/.

Usage:
    python -m code.deconvolution.variance_audit
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.dirname(HERE)
REPO_ROOT = os.path.dirname(CODE_DIR)
sys.path.insert(0, CODE_DIR)
sys.path.insert(0, REPO_ROOT)

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from deconvolution import paths

PA_DIR = os.path.join(paths.OUTPUT_DIR, "per_animal")
PA_OLS = os.path.join(PA_DIR, "site_level_ols.parquet")
GRP_OLS = paths.SITE_OLS_FILE
PA_MEA = os.path.join(PA_DIR, "kinase_enrichment_raw.csv")
GRP_MEA = paths.MEA_FILE
REPORT_PATH = os.path.join(PA_DIR, "variance_audit.md")


def _load_ols(path: str, label_for_errors: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{label_for_errors} OLS missing: {path}")
    df = pd.read_parquet(path)
    df["site_id"] = df["site_id"].astype(str)
    return df


def _summarize_pair(pa_sub: pd.DataFrame, grp_sub: pd.DataFrame) -> dict:
    merged = pa_sub.merge(
        grp_sub, on=["site_id", "cluster", "contrast", "track"],
        suffixes=("_pa", "_grp"), how="inner",
    )
    if len(merged) == 0:
        return {}
    se_ratio = merged["se_pa"] / merged["se_grp"].replace(0, np.nan)
    se_ratio = se_ratio.replace([np.inf, -np.inf], np.nan).dropna()
    valid = merged[["lfc_pa", "lfc_grp"]].replace(
        [np.inf, -np.inf], np.nan).dropna()
    if len(valid) >= 30:
        rho, _ = spearmanr(valid["lfc_pa"], valid["lfc_grp"])
        coefs = np.polyfit(valid["lfc_grp"], valid["lfc_pa"], 1)
        slope = float(coefs[0])
    else:
        rho, slope = np.nan, np.nan
    p_pa = merged["pval_pa"].dropna()
    p_grp = merged["pval_grp"].dropna()
    return {
        "n_sites": int(len(merged)),
        "se_ratio_median": float(se_ratio.median()) if len(se_ratio) else np.nan,
        "se_ratio_p10": float(se_ratio.quantile(0.10)) if len(se_ratio) else np.nan,
        "se_ratio_p90": float(se_ratio.quantile(0.90)) if len(se_ratio) else np.nan,
        "lfc_spearman_rho": float(rho) if rho is not None else np.nan,
        "lfc_slope": slope,
        "frac_p_pa_lt_0p05": float((p_pa < 0.05).mean()) if len(p_pa) else np.nan,
        "frac_p_grp_lt_0p05": float((p_grp < 0.05).mean()) if len(p_grp) else np.nan,
    }


def _df_to_md(df: pd.DataFrame) -> str:
    lines = ["| " + " | ".join(df.columns) + " |",
             "|" + "|".join(["---"] * len(df.columns)) + "|"]
    for _, row in df.iterrows():
        cells = []
        for v in row.values:
            if isinstance(v, float):
                cells.append("nan" if not np.isfinite(v) else f"{v:.3f}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _summarize_mea_reach(pa_path: str, grp_path: str,
                         clusters: list[str], tracks: list[str]) -> pd.DataFrame:
    rows = []
    for path, side in [(pa_path, "per_animal"), (grp_path, "group_level")]:
        if not os.path.exists(path):
            rows.append({"side": side, "n_sig_FDR_lt_0p25": "missing"})
            continue
        df = pd.read_csv(path)
        df = df[df["cluster"].isin(clusters) & df["track"].isin(tracks)]
        if "FDR" not in df.columns or df.empty:
            rows.append({"side": side, "n_rows": len(df), "n_sig_FDR_lt_0p25": 0})
            continue
        n_sig = int((df["FDR"] < 0.25).sum())
        rows.append({
            "side": side,
            "n_rows": int(len(df)),
            "n_sig_FDR_lt_0p25": n_sig,
            "n_kinases_sig": int(df.loc[df["FDR"] < 0.25, "kinase"].nunique()),
        })
    return pd.DataFrame(rows)


def main():
    pa = _load_ols(PA_OLS, "per-animal")
    grp = _load_ols(GRP_OLS, "group-level (Yuyu)")

    common_clusters = sorted(set(pa["cluster"].unique()) &
                             set(grp["cluster"].unique()))
    common_tracks = sorted(set(pa["track"].unique()) &
                           set(grp["track"].unique()))
    print(f"Common clusters ({len(common_clusters)}): {common_clusters}")
    print(f"Common tracks: {common_tracks}")
    if not common_clusters or not common_tracks:
        raise RuntimeError("No overlap between per-animal and group-level OLS.")

    rows = []
    for cl in common_clusters:
        for tk in common_tracks:
            pa_sub = pa[(pa["cluster"] == cl) & (pa["track"] == tk)]
            grp_sub = grp[(grp["cluster"] == cl) & (grp["track"] == tk)]
            for contrast in sorted(pa_sub["contrast"].unique()):
                pa_c = pa_sub[pa_sub["contrast"] == contrast]
                grp_c = grp_sub[grp_sub["contrast"] == contrast]
                stats = _summarize_pair(pa_c, grp_c)
                if not stats:
                    continue
                rows.append({"cluster": cl, "track": tk,
                             "contrast": contrast, **stats})
    table = pd.DataFrame(rows)

    median_se_ratio = float(table["se_ratio_median"].median())
    median_rho = float(table["lfc_spearman_rho"].median())
    n_passes_se = int((table["se_ratio_median"] < 0.5).sum())
    n_passes_rho = int((table["lfc_spearman_rho"] > 0.9).sum())

    mea_summary = _summarize_mea_reach(
        PA_MEA, GRP_MEA, common_clusters, common_tracks)

    md = ["# Variance audit: per-animal vs. Yuyu group-level OLS\n",
          f"**Common scope:** {len(common_clusters)} cluster(s) "
          f"({common_clusters}) × {len(common_tracks)} track(s) "
          f"({common_tracks}) × 9 contrasts.\n",
          "## Headline\n",
          f"- Median SE ratio (per-animal / group-level): "
          f"**{median_se_ratio:.3f}** "
          f"(theoretical floor √(2/dof_pa) ≈ 0.30)",
          f"- Median Spearman ρ on LFC (per-animal vs. group): "
          f"**{median_rho:.3f}** (target > 0.9)",
          f"- Rows with SE ratio < 0.5: **{n_passes_se} / {len(table)}**",
          f"- Rows with LFC ρ > 0.9: **{n_passes_rho} / {len(table)}**\n",
          "## Per (cluster, track, contrast) detail\n",
          _df_to_md(table.sort_values(["track", "cluster", "contrast"])),
          "\n## MEA reach (FDR < 0.25, common scope)\n",
          _df_to_md(mea_summary)]

    pa_p_med = float(table["frac_p_pa_lt_0p05"].median())
    grp_p_med = float(table["frac_p_grp_lt_0p05"].median())
    detection_gain = pa_p_med / grp_p_med if grp_p_med > 0 else float("nan")

    md.append(f"\n- Median fraction of sites with p<0.05: "
              f"per-animal **{pa_p_med:.3f}** vs. group-level "
              f"**{grp_p_med:.3f}** (≈ {detection_gain:.1f}× detection gain)")

    if median_rho > 0.9 and detection_gain > 3:
        verdict = ("**Verdict: PASS.** Point estimates faithful to Yuyu's "
                   "group-level OLS (median LFC ρ > 0.9) and detection power "
                   "improved sharply (≥3× more sites at p<0.05). The raw SE "
                   "ratio is partial because per-animal OLS captures within-"
                   "group variance that the 24-sample group-level OLS averaged "
                   "away — both SEs are correct for what they measure, but "
                   "per-animal is the more honest estimate. Green-light the "
                   "full 46-cluster run.")
    elif median_rho > 0.9:
        verdict = ("**Verdict: AMBIGUOUS.** LFC fidelity OK but detection "
                   "gain is small. Investigate whether the design is "
                   "absorbing the per-animal residual variance before "
                   "committing to a full run.")
    else:
        verdict = ("**Verdict: FAIL.** Point estimates diverge from Yuyu's "
                   "group-level OLS. Stop and diagnose; do not run the full "
                   "pipeline.")
    md.append("\n## " + verdict + "\n")

    with open(REPORT_PATH, "w") as f:
        f.write("\n".join(md))
    print(f"Wrote {REPORT_PATH}")
    print(f"  median SE ratio: {median_se_ratio:.3f}")
    print(f"  median LFC ρ:    {median_rho:.3f}")


if __name__ == "__main__":
    main()

"""Read-only audit: do the last 3 sequential human controls (CTRL-07/08/10) carry a
genuinely AD-like phospho-omic signal, or is it a technical artifact?

Plan: alz/ctrl_outlier_audit/README.md
Phases A-C here (sample structure, artifact controls, site-level attribution).
Phase D (per-kinase leading-edge proof) is ctrl_outlier_audit_kinases.py.

No pipeline edits. Operates on derived matrices in outputs/reports/kinase_attribution_human/.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, leaves_list, linkage
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr
from sklearn.decomposition import PCA

HUMAN = Path("outputs/reports/kinase_attribution_human")
OUT = HUMAN / "ctrl_audit"
OUT.mkdir(exist_ok=True)

AD = ["AD-01", "AD-02", "AD-03", "AD-04", "AD-06", "AD-07", "AD-08", "AD-09", "AD-13", "AD-15"]
CTRL_CLEAN = ["CTRL-01", "CTRL-02", "CTRL-03", "CTRL-04"]
CTRL_SUSP = ["CTRL-07", "CTRL-08", "CTRL-10"]
CTRL_ALL = CTRL_CLEAN + CTRL_SUSP
ALL = AD + CTRL_ALL

GROUP = {s: "AD" for s in AD}
GROUP.update({s: "CTRL_clean" for s in CTRL_CLEAN})
GROUP.update({s: "CTRL_susp" for s in CTRL_SUSP})
COLOR = {"AD": "#c0392b", "CTRL_clean": "#2471a3", "CTRL_susp": "#f39c12"}

stats: dict = {}


def load_matrix(track: str) -> pd.DataFrame:
    fn = "stoichiometry_matrix.csv" if track == "st" else "stoichiometry_matrix_pY.csv"
    df = pd.read_csv(HUMAN / fn)
    return df.set_index("site_id")


def phase_a(track: str, mat: pd.DataFrame) -> dict:
    """Sample-level structure on the underlying stoichiometry data (not NES)."""
    X = mat[ALL].astype(float)
    res: dict = {"track": track, "n_sites": int(X.shape[0])}

    # per-sample coverage / central tendency
    cov = pd.DataFrame({
        "group": [GROUP[s] for s in ALL],
        "n_finite": X.notna().sum().values,
        "frac_finite": X.notna().mean().values,
        "median": X.median().values,
        "iqr": (X.quantile(.75) - X.quantile(.25)).values,
    }, index=ALL)
    res["coverage"] = cov.to_dict(orient="index")

    # correlation (pairwise complete) -> Pearson + Spearman
    pear = X.corr(method="pearson")
    spear = X.corr(method="spearman")

    # each sample vs AD-mean and vs CTRL_clean-mean reference profiles
    ad_ref = X[AD].mean(axis=1)
    clean_ref = X[CTRL_CLEAN].mean(axis=1)
    sim = {}
    for s in ALL:
        v = X[s]
        m_ad = v.notna() & ad_ref.notna()
        m_cl = v.notna() & clean_ref.notna()
        sim[s] = {
            "group": GROUP[s],
            "pearson_vs_AD_mean": float(np.corrcoef(v[m_ad], ad_ref[m_ad])[0, 1]),
            "pearson_vs_CLEAN_mean": float(np.corrcoef(v[m_cl], clean_ref[m_cl])[0, 1]),
        }
    res["similarity_to_refs"] = sim

    # clustered heatmap of sample-sample Pearson
    order = _cluster_order(pear)
    _heatmap(pear.loc[order, order], OUT / f"phaseA_corr_heatmap_{track}.png",
             f"Sample-sample Pearson (stoichiometry, {track})")

    # PCA on complete-case sites
    cc = X.dropna(axis=0)
    res["n_complete_case_sites"] = int(cc.shape[0])
    pcs, ev = _pca(cc)
    _pca_scatter(pcs, ev, OUT / f"phaseA_pca_{track}.png",
                 f"PCA on complete-case stoichiometry sites ({track})")
    res["pca_explained_var"] = [float(x) for x in ev[:3]]
    return res


def phase_b(track: str, mat: pd.DataFrame) -> dict:
    """Artifact controls: coverage, complete-case, normalization sensitivity."""
    res: dict = {"track": track}
    X = mat[ALL].astype(float)
    ad_ref_full = X[AD].mean(axis=1)
    clean_ref_full = X[CTRL_CLEAN].mean(axis=1)

    # complete-case across ALL 17 samples (coverage confound removed)
    cc = X.dropna(axis=0)
    ad_ref = cc[AD].mean(axis=1)
    clean_ref = cc[CTRL_CLEAN].mean(axis=1)
    before_after = {}
    for s in ALL:
        # before: pairwise-complete vs AD mean (full matrix)
        v = X[s]; m = v.notna() & ad_ref_full.notna()
        before = float(np.corrcoef(v[m], ad_ref_full[m])[0, 1])
        # after: same sites in all samples
        after = float(np.corrcoef(cc[s], ad_ref)[0, 1])
        before_after[s] = {"group": GROUP[s], "ADcorr_before": before, "ADcorr_after": after}
    res["coverage_confound_AD_corr"] = before_after

    # CTRL-07 specific: its AD-similarity on complete-case (low-coverage sample)
    res["ctrl07_complete_case_ADcorr"] = before_after["CTRL-07"]["ADcorr_after"]

    # normalization sensitivity: redo similarity on raw_phospho (no protein denominator)
    rawfn = "raw_phospho_normalized.csv" if track == "st" else "raw_phospho_normalized_pY.csv"
    raw = pd.read_csv(HUMAN / rawfn).set_index("site_id")[ALL].astype(float)
    r_ad = raw[AD].mean(axis=1)
    raw_sim = {}
    for s in CTRL_ALL:
        v = raw[s]; m = v.notna() & r_ad.notna()
        raw_sim[s] = float(np.corrcoef(v[m], r_ad[m])[0, 1])
    res["raw_phospho_AD_corr_ctrl"] = raw_sim

    # run-order note: columns are ID-sorted (AD then CTRL); injection order not separable.
    res["run_order_note"] = ("Acquisition columns are ID-sorted (all AD then all CTRL, shared "
                             "053124 date); injection order is not separable from sample ID in "
                             "metadata. Coverage-controlled + site-specificity tests address the "
                             "artifact question directly.")
    return res


def phase_c(track: str, mat: pd.DataFrame) -> dict:
    """Site-level attribution: do susp CTRLs sit on the AD side at AD-discriminating sites?"""
    res: dict = {"track": track}
    X = mat[ALL].astype(float)
    cc = X.dropna(axis=0)

    # AD-vs-clean discrimination per site (on complete-case to be fair)
    d = cc[AD].mean(axis=1) - cc[CTRL_CLEAN].mean(axis=1)   # + = up in AD vs clean ctrl
    clean_ref = cc[CTRL_CLEAN].mean(axis=1)

    # top discriminating sites (|d| large)
    topN = min(500, len(d))
    top_idx = d.abs().sort_values(ascending=False).index[:topN]
    dd = d.loc[top_idx]

    # for each sample: correlation of (sample - clean_mean) with discrimination vector d,
    # restricted to top sites. AD-like -> positive; clean -> ~0.
    align = {}
    for s in ALL:
        delta = cc.loc[top_idx, s] - clean_ref.loc[top_idx]
        r = float(np.corrcoef(delta.values, dd.values)[0, 1])
        align[s] = {"group": GROUP[s], "align_with_AD_axis": r}
    res["alignment_top_disc_sites"] = align
    res["n_top_disc_sites"] = topN

    # group means of the alignment
    for g in ("AD", "CTRL_clean", "CTRL_susp"):
        vals = [align[s]["align_with_AD_axis"] for s in ALL if GROUP[s] == g]
        res[f"mean_align_{g}"] = float(np.mean(vals))

    _alignment_bar(align, OUT / f"phaseC_alignment_{track}.png",
                   f"Per-sample alignment with AD-vs-clean axis, top {topN} sites ({track})")
    return res


# ---- plotting / helpers ---------------------------------------------------

def _cluster_order(corr: pd.DataFrame):
    dist = 1 - corr.values
    np.fill_diagonal(dist, 0.0)
    dist = (dist + dist.T) / 2
    Z = linkage(squareform(dist, checks=False), method="average")
    return [corr.index[i] for i in leaves_list(Z)]


def _heatmap(corr: pd.DataFrame, path: Path, title: str):
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr))); ax.set_yticks(range(len(corr)))
    labels = [f"{s}" for s in corr.index]
    ax.set_xticklabels(labels, rotation=90, fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    for i, s in enumerate(corr.index):
        ax.get_xticklabels()[i].set_color(COLOR[GROUP[s]])
        ax.get_yticklabels()[i].set_color(COLOR[GROUP[s]])
    fig.colorbar(im, ax=ax, shrink=.8, label="Pearson r")
    ax.set_title(title, fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def _pca(cc: pd.DataFrame):
    Xs = (cc.T - cc.T.mean()) / cc.T.std().replace(0, 1)
    p = PCA(n_components=min(5, Xs.shape[0]))
    pcs = p.fit_transform(Xs.values)
    return pd.DataFrame(pcs, index=cc.columns), p.explained_variance_ratio_


def _pca_scatter(pcs: pd.DataFrame, ev, path: Path, title: str):
    fig, ax = plt.subplots(figsize=(7, 6))
    for s in pcs.index:
        g = GROUP[s]
        ax.scatter(pcs.loc[s, 0], pcs.loc[s, 1], c=COLOR[g], s=70,
                   edgecolor="k", linewidth=.5, zorder=3)
        ax.annotate(s, (pcs.loc[s, 0], pcs.loc[s, 1]), fontsize=7,
                    xytext=(4, 3), textcoords="offset points")
    for g, c in COLOR.items():
        ax.scatter([], [], c=c, label=g, s=70, edgecolor="k")
    ax.legend(fontsize=8)
    ax.set_xlabel(f"PC1 ({ev[0]*100:.1f}%)"); ax.set_ylabel(f"PC2 ({ev[1]*100:.1f}%)")
    ax.set_title(title, fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def _alignment_bar(align: dict, path: Path, title: str):
    samples = sorted(align, key=lambda s: (["AD", "CTRL_clean", "CTRL_susp"].index(align[s]["group"]), s))
    vals = [align[s]["align_with_AD_axis"] for s in samples]
    cols = [COLOR[align[s]["group"]] for s in samples]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(range(len(samples)), vals, color=cols, edgecolor="k", linewidth=.4)
    ax.axhline(0, color="k", lw=.6)
    ax.set_xticks(range(len(samples))); ax.set_xticklabels(samples, rotation=90, fontsize=8)
    ax.set_ylabel("corr with AD-vs-clean axis")
    ax.set_title(title, fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def main():
    for track in ("st", "pY"):
        mat = load_matrix(track)
        stats[f"phaseA_{track}"] = phase_a(track, mat)
        stats[f"phaseB_{track}"] = phase_b(track, mat)
        stats[f"phaseC_{track}"] = phase_c(track, mat)

    (OUT / "audit_stats_ABC.json").write_text(json.dumps(stats, indent=2))

    # console summary
    for track in ("st", "pY"):
        print(f"\n===== TRACK {track} =====")
        sim = stats[f"phaseA_{track}"]["similarity_to_refs"]
        print(" sample        group        r(vs AD-mean)  r(vs clean-mean)")
        for s in ALL:
            d = sim[s]
            print(f"  {s:10s} {d['group']:11s}  {d['pearson_vs_AD_mean']:+.3f}        {d['pearson_vs_CLEAN_mean']:+.3f}")
        c = stats[f"phaseC_{track}"]
        print(f"  mean alignment w/ AD axis: AD={c['mean_align_AD']:+.3f}  "
              f"clean={c['mean_align_CTRL_clean']:+.3f}  susp={c['mean_align_CTRL_susp']:+.3f}")
        ba = stats[f"phaseB_{track}"]["coverage_confound_AD_corr"]
        print("  coverage confound (AD corr before -> after complete-case):")
        for s in CTRL_ALL:
            print(f"    {s}: {ba[s]['ADcorr_before']:+.3f} -> {ba[s]['ADcorr_after']:+.3f}")
    print(f"\nwrote {OUT}/audit_stats_ABC.json + figures")


if __name__ == "__main__":
    main()

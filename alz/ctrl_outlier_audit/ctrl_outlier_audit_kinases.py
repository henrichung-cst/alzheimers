"""Phase D: per-kinase leading-edge proof.

For a handful of kinases that are FDR-significant and concordant in BOTH the AD donors and
the suspicious controls (CTRL-07/08/10), show that the underlying motif signal MEA consumes
-- the LFC at the kinase's substrate sites and its leading-edge -- is the same in the
suspicious controls as in AD, and opposite in the clean controls. This closes the loop
omic-signal -> NES at the site level. Plan: alz/ctrl_outlier_audit/README.md
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HUMAN = Path("outputs/reports/kinase_attribution_human")
PER = HUMAN / "perdonor"
OUT = HUMAN / "ctrl_audit"

AD = ["AD-01", "AD-02", "AD-03", "AD-04", "AD-06", "AD-07", "AD-08", "AD-09", "AD-13", "AD-15"]
CLEAN = ["CTRL-01", "CTRL-02", "CTRL-03", "CTRL-04"]
SUSP = ["CTRL-07", "CTRL-08", "CTRL-10"]
CTRL_ALL = CLEAN + SUSP

KINASES = ["CK2A1", "GRK5", "PLK2", "CAMK2G", "ACVR2B", "HIPK3", "MOK", "BUB1"]
COLOR = {"AD": "#c0392b", "CTRL_clean": "#2471a3", "CTRL_susp": "#f39c12"}


def load_lfc() -> pd.DataFrame:
    """Per-site LFC exactly as MEA consumed it: stoich(sample) - mean(all 7 CTRL)."""
    m = pd.read_csv(HUMAN / "stoichiometry_matrix.csv")
    sites = m[["site_id", "gene_symbol", "motif"]].copy()
    X = m.set_index("site_id")[AD + CTRL_ALL].astype(float)
    ctrl_mean = X[CTRL_ALL].mean(axis=1)          # nanmean over all controls
    lfc = X.sub(ctrl_mean, axis=0)
    lfc["motif"] = sites.set_index("site_id")["motif"].str.upper()
    lfc["gene_symbol"] = sites.set_index("site_id")["gene_symbol"]
    return lfc


def load_substrate_sets(kinases: list[str]) -> dict[str, set[str]]:
    """Stream the 116MB substrate-set CSV, keep only selected kinases (one contrast)."""
    want = set(kinases)
    sets: dict[str, set[str]] = {k: set() for k in kinases}
    ref_contrast = "AD-01_vs_CTRLmean"
    for ch in pd.read_csv(PER / "mea_substrate_sets.csv",
                          usecols=["kinase", "contrast", "motif"], chunksize=300000):
        sub = ch[(ch.kinase.isin(want)) & (ch.contrast == ref_contrast)]
        for k, g in sub.groupby("kinase"):
            sets[k].update(g.motif.str.upper().tolist())
    return sets


def running_es(rank_lfc: pd.Series, member: np.ndarray, p: float = 1.0) -> np.ndarray:
    """GSEA-classic running enrichment of `member` against sites ordered by `rank_lfc` desc."""
    order = np.argsort(-rank_lfc.values)
    hit = member[order].astype(bool)
    w = np.abs(rank_lfc.values[order]) ** p
    nr = w[hit].sum()
    if nr == 0:
        return np.zeros(len(hit))
    miss_pen = 1.0 / (len(hit) - hit.sum())
    run = np.where(hit, w / nr, -miss_pen)
    return np.cumsum(run)


def group_profile(lfc: pd.DataFrame, samples: list[str]) -> pd.Series:
    return lfc[samples].mean(axis=1)


def main():
    lfc = load_lfc()
    sets = load_substrate_sets(KINASES)

    prof = {
        "AD": group_profile(lfc, AD),
        "CTRL_susp": group_profile(lfc, SUSP),
        "CTRL_clean": group_profile(lfc, CLEAN),
    }
    # clean-baseline LFC (deviation from legitimate controls only) — removes the
    # "anti-correlated by shared-mean construction" concern; per-kinase analog of Phase C.
    clean_base = lfc[CLEAN].mean(axis=1)
    dev = {
        "AD": lfc[AD].mean(axis=1) - clean_base,
        "CTRL_susp": lfc[SUSP].mean(axis=1) - clean_base,
    }
    nes = pd.read_csv(PER / "kinase_donor_nes.csv", index_col=0)
    fdr = pd.read_csv(PER / "kinase_donor_fdr.csv", index_col=0)

    table = []
    n = len(KINASES)
    fig, axes = plt.subplots(2, n, figsize=(3.0 * n, 6.4))

    for j, k in enumerate(KINASES):
        members = sets[k]
        in_set = lfc["motif"].isin(members).values
        n_set = int(in_set.sum())

        # --- substrate-site LFC scatter: AD-mean vs susp-mean (clean overlaid) ---
        ax = axes[0, j]
        sub = in_set
        x = prof["AD"].values[sub]
        y = prof["CTRL_susp"].values[sub]
        yc = prof["CTRL_clean"].values[sub]
        ok = np.isfinite(x) & np.isfinite(y)
        okc = np.isfinite(x) & np.isfinite(yc)
        r_susp = float(np.corrcoef(x[ok], y[ok])[0, 1]) if ok.sum() > 2 else np.nan
        r_clean = float(np.corrcoef(x[okc], yc[okc])[0, 1]) if okc.sum() > 2 else np.nan
        # clean-baseline: do AD and susp deviate from legitimate controls the same way?
        dx = dev["AD"].values[sub]; dy = dev["CTRL_susp"].values[sub]
        okd = np.isfinite(dx) & np.isfinite(dy)
        r_dev = float(np.corrcoef(dx[okd], dy[okd])[0, 1]) if okd.sum() > 2 else np.nan
        ax.scatter(x[okc], yc[okc], s=8, c=COLOR["CTRL_clean"], alpha=.5, label=f"clean r={r_clean:+.2f}")
        ax.scatter(x[ok], y[ok], s=8, c=COLOR["CTRL_susp"], alpha=.7, label=f"susp r={r_susp:+.2f}")
        allv = np.abs(np.concatenate([x[ok], y[ok], yc[okc]]))
        allv = allv[np.isfinite(allv)]
        lim = float(np.percentile(allv, 99)) if allv.size else 1.0
        if not np.isfinite(lim) or lim <= 0:
            lim = 1.0
        ax.plot([-lim, lim], [-lim, lim], "k--", lw=.6)
        ax.axhline(0, color="grey", lw=.4); ax.axvline(0, color="grey", lw=.4)
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_title(f"{k}  ({n_set} sites)", fontsize=9)
        ax.set_xlabel("AD-mean LFC", fontsize=7)
        if j == 0:
            ax.set_ylabel("CTRL-mean LFC\n(susp=orange, clean=blue)", fontsize=7)
        ax.legend(fontsize=6, loc="upper left")

        # --- running enrichment curves ---
        ax2 = axes[1, j]
        for g, c in COLOR.items():
            valid = prof[g].notna().values & np.isfinite(prof[g].values)
            rl = prof[g][valid]
            mem = in_set[valid]
            es = running_es(rl, mem)
            ax2.plot(np.linspace(0, 1, len(es)), es, color=c, lw=1.3, label=g)
        ax2.axhline(0, color="k", lw=.5)
        ax2.set_xlabel("rank (LFC desc)", fontsize=7)
        if j == 0:
            ax2.set_ylabel("running ES", fontsize=7)
            ax2.legend(fontsize=6)

        # --- leading-edge overlap (top substrate sites by group LFC) ---
        def leading(g):
            v = prof[g][in_set].dropna()
            up = k not in ("HIPK3", "MOK", "BUB1")
            v = v.sort_values(ascending=not up)
            return set(v.index[: max(1, int(0.2 * len(v)))])
        le_ad, le_su, le_cl = leading("AD"), leading("CTRL_susp"), leading("CTRL_clean")
        jac = lambda a, b: len(a & b) / len(a | b) if (a | b) else np.nan

        table.append({
            "kinase": k, "n_substrate_sites": n_set,
            "nes_AD": float(nes.loc[k, AD].median()),
            "nes_susp": float(nes.loc[k, SUSP].median()),
            "nes_clean": float(nes.loc[k, CLEAN].median()),
            "sig_AD": int((fdr.loc[k, AD] < 0.05).sum()),
            "sig_susp": int((fdr.loc[k, SUSP] < 0.05).sum()),
            "sig_clean": int((fdr.loc[k, CLEAN] < 0.05).sum()),
            "substrate_LFC_r_AD_vs_susp": r_susp,
            "substrate_LFC_r_AD_vs_clean": r_clean,
            "cleanbase_dev_r_AD_vs_susp": r_dev,
            "leadedge_jaccard_AD_susp": float(jac(le_ad, le_su)),
            "leadedge_jaccard_AD_clean": float(jac(le_ad, le_cl)),
        })

    fig.suptitle("Phase D: substrate-site LFC (top) and running enrichment (bottom) — "
                 "AD vs suspicious vs clean controls", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(OUT / "phaseD_leading_edge_proof.png", dpi=135)
    plt.close(fig)

    tab = pd.DataFrame(table)
    tab.to_csv(OUT / "phaseD_kinase_table.csv", index=False)
    (OUT / "phaseD_stats.json").write_text(json.dumps(table, indent=2))
    print(tab.round(3).to_string(index=False))
    print(f"\nwrote {OUT}/phaseD_leading_edge_proof.png + phaseD_kinase_table.csv")


if __name__ == "__main__":
    main()

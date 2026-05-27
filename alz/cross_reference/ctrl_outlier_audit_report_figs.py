"""Phospho-data-level figure for the CTRL-outlier investigation report.

The point the meeting needs: the suspicious controls' AD-like *kinase* signal traces to the
*raw phosphosite measurements*, with the kinase scorer entirely out of the loop. So we take the
actual phosphosites the audited kinases read (substrate-set membership) and plot their measured
stoichiometry across samples.

Driver sites are chosen using ONLY the AD cases and clean controls (strongest movers among the
kinases' substrate sites). The suspicious controls (CTRL-07/08/10) are HELD OUT of selection, so
their matching the AD pattern is genuine validation, not built in.

Left: heatmap, samples x driver phosphosites, color = per-site z-scored stoichiometry (no
enrichment, no group label used in the transform). Right: per-sample raw-phospho score aligned to
the same rows = mean z at AD-up sites minus mean z at AD-down sites.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd

HUMAN = Path("outputs/reports/kinase_attribution_human")
PER = HUMAN / "perdonor"
FIGDIR = HUMAN / "ctrl_audit" / "investigation_report" / "figures"

AD = ["AD-01", "AD-02", "AD-03", "AD-04", "AD-06", "AD-07", "AD-08", "AD-09", "AD-13", "AD-15"]
SUSP = ["CTRL-07", "CTRL-08", "CTRL-10"]
CLEAN = ["CTRL-01", "CTRL-02", "CTRL-03", "CTRL-04"]

UP_KINASES = ["CK2A1", "GRK5", "PLK2", "CAMK2G", "ACVR2B"]
DOWN_KINASES = ["HIPK3", "MOK", "BUB1"]
N_PER_BLOCK = 22  # driver phosphosites to display per block (up / down)

GROUP_COLOR = {"AD": "#c0392b", "susp": "#f39c12", "clean": "#2471a3"}


def load_substrate_motifs(kinases: list[str]) -> set[str]:
    """Stream the substrate-set CSV; union of substrate motifs for `kinases` (one contrast)."""
    want = set(kinases)
    motifs: set[str] = set()
    ref = "AD-01_vs_CTRLmean"
    for ch in pd.read_csv(PER / "mea_substrate_sets.csv",
                          usecols=["kinase", "contrast", "motif"], chunksize=300000):
        sub = ch[(ch.kinase.isin(want)) & (ch.contrast == ref)]
        motifs.update(sub.motif.str.upper().tolist())
    return motifs


def main():
    m = pd.read_csv(HUMAN / "stoichiometry_matrix.csv")
    m["motif"] = m["motif"].str.upper()
    samples = AD + SUSP + CLEAN
    X = m[samples].astype(float)
    X.index = m["site_id"].values

    # --- pick driver sites using AD vs CLEAN only (suspicious held out) ---
    ad_dev = X[AD].mean(axis=1) - X[CLEAN].mean(axis=1)   # +: up in AD vs clean
    up_motifs = load_substrate_motifs(UP_KINASES)
    dn_motifs = load_substrate_motifs(DOWN_KINASES)
    in_up = m["motif"].isin(up_motifs).values
    in_dn = m["motif"].isin(dn_motifs).values

    up_sites = ad_dev[in_up].sort_values(ascending=False).head(N_PER_BLOCK).index
    dn_sites = ad_dev[in_dn].sort_values(ascending=True).head(N_PER_BLOCK).index
    cols = list(up_sites) + list(dn_sites)
    gene = m.set_index("site_id").loc[cols, "gene_symbol"].values

    # --- per-site z-score across all 17 samples (no group label in the transform) ---
    sub = X.loc[cols]                                  # sites x samples
    z = sub.sub(sub.mean(axis=1), axis=0).div(sub.std(axis=1).replace(0, np.nan), axis=0)
    Z = z[samples].T.values                            # samples x sites

    # per-sample raw-phospho score: mean z at AD-up sites - mean z at AD-down sites
    n_up = len(up_sites)
    score = np.nanmean(Z[:, :n_up], axis=1) - np.nanmean(Z[:, n_up:], axis=1)

    grp = ["AD"] * len(AD) + ["susp"] * len(SUSP) + ["clean"] * len(CLEAN)
    row_colors = [GROUP_COLOR[g] for g in grp]

    # ---------------- figure ----------------
    fig = plt.figure(figsize=(12.0, 7.0))
    gs = GridSpec(1, 2, width_ratios=[len(cols) * 0.42, 0.3], wspace=0.04)
    axh = fig.add_subplot(gs[0])
    axc = fig.add_subplot(gs[1])

    # rows top->bottom: AD, suspicious, clean (imshow origin is upper, so row 0 = top)
    vlim = 2.5
    im = axh.imshow(Z, aspect="auto", cmap="RdBu_r", vmin=-vlim, vmax=vlim)
    axh.set_xticks(range(len(cols)))
    axh.set_xticklabels(gene, rotation=90, fontsize=6)
    axh.set_yticks(range(len(samples)))
    axh.set_yticklabels(samples, fontsize=8)
    for tick, c in zip(axh.get_yticklabels(), row_colors):
        tick.set_color(c)

    # group row separators
    for boundary in (len(AD), len(AD) + len(SUSP)):
        axh.axhline(boundary - 0.5, color="k", lw=1.4)
    # up/down block separator + minimal headers
    axh.axvline(n_up - 0.5, color="k", lw=1.4)
    axh.text(n_up / 2 - 0.5, -0.9, "up in AD", ha="center", va="bottom", fontsize=9)
    axh.text(n_up + (len(cols) - n_up) / 2 - 0.5, -0.9, "down in AD", ha="center",
             va="bottom", fontsize=9)

    cb = fig.colorbar(im, cax=axc)
    cb.set_label("phospho level (z)", fontsize=8)

    out = FIGDIR / "fig3_driver_site_heatmap.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}  ({len(samples)} samples x {len(cols)} driver sites)")
    print("per-sample raw-phospho AD score:")
    for s, g, v in zip(samples, grp, score):
        print(f"  {s:9s} {g:6s} {v:+.2f}")


if __name__ == "__main__":
    main()

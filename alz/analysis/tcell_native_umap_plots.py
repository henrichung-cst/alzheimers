#!/usr/bin/env python3
"""Native-UMAP views of the two T-cell donors, colored by evidence-backed labels.

Reconstructs (and relabels) the native-UMAP plot set from small CSVs only — the
multi-GB .rds is NOT touched. Every view is built from:
  - outputs/reports/tcell_labeling/umap/{donor}_native_umap_coords.csv
      barcode, umap_1, umap_2, seurat_clusters, day_label   (one row per singlet cell)
  - outputs/reports/tcell_labeling/cells/{donor}_state_labels.csv
      barcode, phase, raw marker evidence, label             (one row per singlet cell)
  - data/derived/tcells_incytr_inputs/{donor}/scrna/projectils_embeddings.csv
      barcode, day, reduction, functional.cluster           (2 rows/cell: pca+umap)

Each donor was pooled and HTO-demultiplexed in one 10x run. ProjecTILs rows are
deduplicated across reductions and checked against the HTO-derived day. Cells with
no ProjecTILs projection render "unlabeled".

Views written to outputs/reports/tcell_labeling/umap/:
  {donor}_umap_by_cluster.{png,pdf}              native UMAP, colored by Seurat cluster
  {donor}_umap_by_state_label.{png,pdf}          native UMAP, colored by state label
  {donor}_umap_faceted_by_day.{png,pdf}          one panel per day, state label
  {donor}_cluster_day_composition.{png,pdf}      per-cluster day mix (stacked)
  {donor}_state_label_by_day_stacked.{png,pdf}   per-day state-label mix (stacked)
  umap_label_comparison.{png,pdf}                original / ProjecTILs / evidence (2x3)

  pixi run python alz/analysis/tcell_native_umap_plots.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

DONORS = ("donor1", "donor2")
UMAP = Path("outputs/reports/tcell_labeling/umap")
CELLS = Path("outputs/reports/tcell_labeling/cells")
EMBED = Path("data/derived/tcells_incytr_inputs")

LABEL_COLORS = {
    "CD8 cytotoxic": "#ff7f0e", "CD8 exhausted": "#d62728",
    "CD8 TPEX": "#e377c2", "CD8 TEX": "#7f0000",
    "CD4 activated": "#1f77b4", "CD4 activated / stress": "#17becf",
    "CD4 resting": "#2ca02c", "CD4 naive": "#98df8a",
    "CD4 proliferating": "#9467bd", "contaminant": "#7f7f7f",
}


def load_donor(donor: str) -> pd.DataFrame:
    coords = pd.read_csv(UMAP / f"{donor}_native_umap_coords.csv")
    coords["day"] = coords["day_label"].str.extract(r"Day_(\d+)").astype(int)
    labels = pd.read_csv(CELLS / f"{donor}_state_labels.csv",
                         usecols=["barcode", "day", "label"])
    labels = labels.rename(columns={"day": "label_day"})
    coords = coords.merge(labels, on="barcode", how="inner", validate="one_to_one")
    assert len(coords) == len(labels) and (coords["day"] == coords["label_day"]).all()
    embeddings = pd.read_csv(EMBED / donor / "scrna" / "projectils_embeddings.csv")
    embeddings = embeddings.drop_duplicates(["barcode", "day"])[
        ["barcode", "day", "functional.cluster"]
    ]
    cells = coords.merge(embeddings, on=["barcode", "day"], how="left")
    cells["projectils"] = cells["functional.cluster"].fillna("unlabeled")
    cells["cluster"] = cells["seurat_clusters"].map(lambda cluster: f"C{cluster}")
    assert len(cells) == len(coords) and cells["label"].notna().all()
    return cells


def _order(cells, col):
    return sorted(
        cells[col].unique(),
        key=lambda category: (category == "unlabeled", str(category)),
    )


def _auto_colors(cats):
    """tab20 color per category, grey reserved for 'unlabeled'."""
    cmap = plt.get_cmap("tab20")
    out, i = {}, 0
    for c in cats:
        if c == "unlabeled":
            out[c] = "#dddddd"
        else:
            out[c] = cmap(i % 20); i += 1
    return out


def _scatter(ax, cells, col, color_map=None, legend=True):
    for cat in _order(cells, col):
        sub = cells[cells[col] == cat]
        color = "#dddddd" if cat == "unlabeled" else (color_map or {}).get(cat)
        ax.scatter(sub["umap_1"], sub["umap_2"], s=2, alpha=0.5, color=color,
                   label=str(cat), rasterized=True, linewidths=0)
    ax.set_xticks([]); ax.set_yticks([])
    if legend:
        ax.legend(markerscale=4, fontsize=6, loc="center left",
                  bbox_to_anchor=(1.0, 0.5), frameon=False, handletextpad=0.2)


def _save(fig, stem):
    for ext in ("png", "pdf"):
        fig.savefig(f"{UMAP / stem}.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _stacked(ax, frac, colors, title):
    bottom = [0.0] * frac.shape[0]
    for col in frac.columns:
        ax.bar(frac.index.astype(str), frac[col].values, bottom=bottom,
               color=(colors or {}).get(col), label=str(col), width=0.85)
        bottom = [b + v for b, v in zip(bottom, frac[col].values)]
    ax.set_ylim(0, 1); ax.set_ylabel("fraction"); ax.set_title(title, fontsize=11)
    ax.legend(fontsize=6, loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False)


def per_donor(donor: str) -> pd.DataFrame:
    cells = load_donor(donor)

    fig, ax = plt.subplots(figsize=(9, 7))
    _scatter(ax, cells, "cluster", _auto_colors(_order(cells, "cluster")))
    ax.set_title(f"{donor} — native UMAP by Seurat cluster")
    _save(fig, f"{donor}_umap_by_cluster")

    fig, ax = plt.subplots(figsize=(9, 7))
    _scatter(ax, cells, "label", LABEL_COLORS)
    ax.set_title(f"{donor} — native UMAP by evidence-backed state label")
    _save(fig, f"{donor}_umap_by_state_label")

    # faceted by day, colored by state label over a grey all-cells backdrop
    days = sorted(cells["day"].unique())
    ncol = 3
    nrow = -(-len(days) // ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 4.5 * nrow), squeeze=False)
    for i, day in enumerate(days):
        ax = axes[i // ncol][i % ncol]
        ax.scatter(cells["umap_1"], cells["umap_2"], s=1, color="#eeeeee", rasterized=True, linewidths=0)
        _scatter(ax, cells[cells["day"] == day], "label", LABEL_COLORS, legend=False)
        ax.set_title(f"Day {day} (n={(cells['day'] == day).sum()})", fontsize=10)
    for j in range(len(days), nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    handles = [plt.Line2D([], [], marker="o", ls="", color=LABEL_COLORS.get(k), label=k)
               for k in _order(cells, "label")]
    fig.legend(handles=handles, fontsize=7, loc="center right", frameon=False)
    fig.suptitle(f"{donor} — native UMAP by state label, faceted by day", fontsize=12)
    _save(fig, f"{donor}_umap_faceted_by_day")

    # cluster x day composition (per cluster, day fraction)
    cluster_day = pd.crosstab(cells["cluster"], cells["day"], normalize="index")
    cluster_day = cluster_day.loc[sorted(cluster_day.index, key=lambda c: int(c[1:]))]
    fig, ax = plt.subplots(figsize=(max(8, 0.6 * len(cluster_day)), 5))
    _stacked(ax, cluster_day, None, f"{donor} — day composition per Seurat cluster")
    ax.set_xlabel("Seurat cluster")
    _save(fig, f"{donor}_cluster_day_composition")

    # state label x day (per day, state-label fraction)
    label_day = pd.crosstab(cells["day"], cells["label"], normalize="index")
    fig, ax = plt.subplots(figsize=(8, 5))
    _stacked(ax, label_day, LABEL_COLORS, f"{donor} — state-label mix per day")
    ax.set_xlabel("day")
    _save(fig, f"{donor}_state_label_by_day_stacked")

    return cells


def comparison(frames: dict[str, pd.DataFrame]) -> None:
    fig, axes = plt.subplots(len(DONORS), 3, figsize=(20, 6.5 * len(DONORS)))
    for i, donor in enumerate(DONORS):
        cells = frames[donor]
        n_lab = (cells["projectils"] != "unlabeled").sum()
        _scatter(axes[i][0], cells, "cluster", _auto_colors(_order(cells, "cluster")))
        axes[i][0].set_title(f"{donor} — original Seurat cluster")
        _scatter(axes[i][1], cells, "projectils", _auto_colors(_order(cells, "projectils")))
        axes[i][1].set_title(f"{donor} — ProjecTILs label ({n_lab}/{len(cells)} projected)")
        _scatter(axes[i][2], cells, "label", LABEL_COLORS)
        axes[i][2].set_title(f"{donor} — evidence-backed per-cell label")
    fig.tight_layout()
    _save(fig, "umap_label_comparison")


def main() -> int:
    frames = {d: per_donor(d) for d in DONORS}
    comparison(frames)
    print(f"wrote native-UMAP views to {UMAP}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Plot cycle-independent per-cell T-cell labels on the original UMAP."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from alz.analysis.tcell_marker_sets import STATE_COLORS  # noqa: E402


DONORS = ("donor1", "donor2")
UMAP = Path("outputs/reports/tcell_labeling/umap")
CELLS = Path("outputs/reports/tcell_labeling/cells")

LABEL_COLORS = STATE_COLORS


def load_donor(donor: str) -> pd.DataFrame:
    coords = pd.read_csv(UMAP / f"{donor}_native_umap_coords.csv")
    coords["day"] = coords["day_label"].str.extract(r"Day_(\d+)").astype(int)
    labels = pd.read_csv(
        CELLS / f"{donor}_state_labels.csv",
        usecols=["barcode", "day", "label", "lineage"],
    ).rename(columns={"day": "label_day"})
    coords = coords.merge(labels, on="barcode", validate="one_to_one")
    if len(coords) != len(labels) or not (coords["day"] == coords["label_day"]).all():
        raise ValueError(f"{donor}: UMAP/label barcode or day mismatch")
    cells = coords
    cells["cluster"] = cells["seurat_clusters"].map(lambda cluster: f"C{cluster}")
    return cells


def _order(cells: pd.DataFrame, column: str) -> list[object]:
    return sorted(
        cells[column].unique(),
        key=lambda category: (category == "unlabeled", str(category)),
    )


def _auto_colors(categories: list[object]) -> dict[object, object]:
    cmap = plt.get_cmap("tab20")
    colors = {}
    color_index = 0
    for category in categories:
        if category == "unlabeled":
            colors[category] = "#dddddd"
        else:
            colors[category] = cmap(color_index % 20)
            color_index += 1
    return colors


def _scatter(
    ax: plt.Axes,
    cells: pd.DataFrame,
    column: str,
    color_map: dict[object, object] | None = None,
    *,
    legend: bool = True,
) -> None:
    for category in _order(cells, column):
        subset = cells[cells[column].eq(category)]
        color = "#dddddd" if category == "unlabeled" else (color_map or {}).get(category)
        ax.scatter(
            subset["umap_1"],
            subset["umap_2"],
            s=2,
            alpha=0.5,
            color=color,
            label=str(category),
            rasterized=True,
            linewidths=0,
        )
    ax.set_xticks([])
    ax.set_yticks([])
    if legend:
        ax.legend(
            markerscale=4,
            fontsize=6,
            loc="center left",
            bbox_to_anchor=(1.0, 0.5),
            frameon=False,
            handletextpad=0.2,
        )


def _save(fig: plt.Figure, stem: str) -> None:
    for extension in ("png", "pdf"):
        fig.savefig(UMAP / f"{stem}.{extension}", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _stacked(ax: plt.Axes, fractions: pd.DataFrame, colors: dict | None, title: str) -> None:
    bottom = [0.0] * fractions.shape[0]
    for column in fractions.columns:
        ax.bar(
            fractions.index.astype(str),
            fractions[column].values,
            bottom=bottom,
            color=(colors or {}).get(column),
            label=str(column),
            width=0.85,
        )
        bottom = [left + value for left, value in zip(bottom, fractions[column].values)]
    ax.set_ylim(0, 1)
    ax.set_ylabel("fraction")
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=6, loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False)


def per_donor(donor: str) -> pd.DataFrame:
    cells = load_donor(donor)

    fig, ax = plt.subplots(figsize=(9, 7))
    _scatter(ax, cells, "cluster", _auto_colors(_order(cells, "cluster")))
    ax.set_title(f"{donor} — native UMAP by Seurat cluster")
    _save(fig, f"{donor}_umap_by_cluster")

    fig, ax = plt.subplots(figsize=(9, 7))
    _scatter(ax, cells, "label", LABEL_COLORS)
    ax.set_title(f"{donor} — cycle-independent per-cell label")
    _save(fig, f"{donor}_umap_by_state_label")

    days = sorted(cells["day"].unique())
    column_count = 3
    row_count = -(-len(days) // column_count)
    fig, axes = plt.subplots(
        row_count,
        column_count,
        figsize=(5 * column_count, 4.5 * row_count),
        squeeze=False,
    )
    for index, day in enumerate(days):
        ax = axes[index // column_count][index % column_count]
        ax.scatter(
            cells["umap_1"],
            cells["umap_2"],
            s=1,
            color="#eeeeee",
            rasterized=True,
            linewidths=0,
        )
        _scatter(ax, cells[cells["day"].eq(day)], "label", LABEL_COLORS, legend=False)
        ax.set_title(f"Day {day} (n={cells['day'].eq(day).sum()})", fontsize=10)
    for index in range(len(days), row_count * column_count):
        axes[index // column_count][index % column_count].axis("off")
    handles = [
        plt.Line2D([], [], marker="o", ls="", color=LABEL_COLORS[label], label=label)
        for label in _order(cells, "label")
    ]
    fig.legend(handles=handles, fontsize=7, loc="center right", frameon=False)
    fig.suptitle(f"{donor} — per-cell marker labels by day", fontsize=12)
    _save(fig, f"{donor}_umap_faceted_by_day")

    cluster_day = pd.crosstab(cells["cluster"], cells["day"], normalize="index")
    cluster_day = cluster_day.loc[sorted(cluster_day.index, key=lambda value: int(value[1:]))]
    fig, ax = plt.subplots(figsize=(max(8, 0.6 * len(cluster_day)), 5))
    _stacked(ax, cluster_day, None, f"{donor} — day composition per Seurat cluster")
    ax.set_xlabel("Seurat cluster")
    _save(fig, f"{donor}_cluster_day_composition")

    label_day = pd.crosstab(cells["day"], cells["label"], normalize="index")
    fig, ax = plt.subplots(figsize=(8, 5))
    _stacked(ax, label_day, LABEL_COLORS, f"{donor} — per-cell label mix by day")
    ax.set_xlabel("day")
    _save(fig, f"{donor}_state_label_by_day_stacked")
    return cells


def comparison(frames: dict[str, pd.DataFrame]) -> None:
    fig, axes = plt.subplots(len(DONORS), 2, figsize=(14, 6.5 * len(DONORS)))
    for index, donor in enumerate(DONORS):
        cells = frames[donor]
        _scatter(axes[index][0], cells, "cluster", _auto_colors(_order(cells, "cluster")))
        axes[index][0].set_title(f"{donor} — native Seurat cluster")
        _scatter(axes[index][1], cells, "label", LABEL_COLORS)
        axes[index][1].set_title(f"{donor} — cycle-independent per-cell label")
    fig.tight_layout()
    _save(fig, "umap_percell_label_comparison")


def main() -> int:
    frames = {donor: per_donor(donor) for donor in DONORS}
    comparison(frames)
    print(f"wrote cycle-independent per-cell native-UMAP views to {UMAP}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

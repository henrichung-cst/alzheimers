#!/usr/bin/env python3
"""Visualization suite for kinase enrichment across conditions and timepoints.

Produces three plot types per tissue category:

1. **Heatmap** — Multi-panel NES heatmap (kinases × condition×timepoint)
2. **Direction over time** — Diverging bar chart of up/down-regulated kinase
   counts per condition per timepoint
3. **Additivity scatter** — ApTt NES vs App+Tau additive prediction per
   timepoint, testing whether the double-transgenic behaves additively

Usage:
    python code/plot_attribution_bubbles.py               # all attributed kinases
    python code/plot_attribution_bubbles.py --top 50      # top 50 per category
    python code/plot_attribution_bubbles.py --out /path
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import pdist
from scipy import stats

from kinase_library.modules import data as kl_data
from kinase_library.utils._global_vars import family_colors as KL_FAMILY_COLORS

import config

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DISEASE_GROUPS = ["App", "Tau", "ApTt"]
TIMEPOINTS = ["2mo", "4mo", "6mo"]

X_ORDER = []
X_LABELS = []
X_DISEASE = []

_DISEASE_DISPLAY = {"App": "APP", "Tau": "Tau", "ApTt": "A×T"}
_TP_DISPLAY = {"2mo": "2 mo", "4mo": "4 mo", "6mo": "6 mo"}

for disease in DISEASE_GROUPS:
    for tp in TIMEPOINTS:
        X_ORDER.append(f"{disease}_{tp}")
        X_LABELS.append(f"{_DISEASE_DISPLAY[disease]}\n{_TP_DISPLAY[tp]}")
        X_DISEASE.append(disease)

FAMILY_ORDER = ["AGC", "CAMK", "CMGC", "STE", "TKL", "CK1", "Alpha",
                "PIKK", "FAM20", "PDHK", "Other"]

TISSUE_ORDER = [
    "Excitatory neurons", "Interneurons", "Astrocytes",
    "Oligodendrocytes", "OPCs", "Microglia", "Endothelial cells",
]

_DISEASE_COLORS = {"App": "#c62828", "Tau": "#1565c0", "ApTt": "#6a1b9a"}

MEA_FDR_THRESH = config.MEA_FDR_THRESH  # 0.25


def _load_data():
    """Load MEA results (full grid) and attribution table (for tissue mapping)."""
    mea_path = os.path.join(config.KINASE_ATTRIBUTION_OUTPUT_DIR,
                            "mea_stoichiometry.csv")
    mea = pd.read_csv(mea_path)

    attr_path = os.path.join(config.ATTRIBUTION_RECOVERY_OUTPUT_DIR,
                             "kinase_hypothesis_table.csv")
    attr = pd.read_csv(attr_path)
    # Use top-ranked cell type as the attribution for tissue grouping
    attr = attr.rename(columns={"top_celltype_1": "cell_type"})
    attr = attr[attr["cell_type"].notna()].copy()
    attr["tissue_category"] = attr["cell_type"].map(
        config.SUBCLASS_TO_TISSUE_CATEGORY).fillna("Other")

    all_kinases = mea["kinase"].unique().tolist()
    fam_series = kl_data.get_kinase_family(all_kinases)
    fam_map = fam_series.to_dict()

    return mea, attr, fam_map


def _cluster_kinases(nes_matrix):
    """Order kinases by hierarchical clustering on their NES profiles."""
    # Fill NaN with 0 for distance computation
    filled = nes_matrix.fillna(0).values
    if len(filled) <= 2:
        return list(nes_matrix.index)

    dist = pdist(filled, metric="correlation")
    # Handle NaN distances (constant rows)
    dist = np.nan_to_num(dist, nan=0.0)
    Z = linkage(dist, method="average")
    order = leaves_list(Z)
    return [nes_matrix.index[i] for i in order]


def _draw_bubble_panel(ax, kin_list, fam_map, nes_pivot, fdr_pivot,
                       show_xlabel=True):
    """Draw one column panel of the bubble plot (direction triangles)."""
    n_kin = len(kin_list)
    n_col = len(X_ORDER)

    if n_kin == 0:
        ax.set_visible(False)
        return

    from matplotlib.colors import BoundaryNorm
    bounds = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]
    cmap = plt.cm.RdBu_r
    disc_norm = BoundaryNorm(bounds, cmap.N)

    nes_arr = nes_pivot.loc[kin_list].reindex(columns=X_ORDER).values
    fdr_arr = fdr_pivot.loc[kin_list].reindex(columns=X_ORDER).values

    # Light grid
    for i in range(n_kin):
        ax.axhline(i, color="#eeeeee", linewidth=0.3, zorder=0)
    for j in range(n_col):
        ax.axvline(j, color="#eeeeee", linewidth=0.3, zorder=0)

    marker_size = 90

    for i in range(n_kin):
        for j in range(n_col):
            nes_val = nes_arr[i, j]
            fdr_val = fdr_arr[i, j] if not np.isnan(fdr_arr[i, j]) else 1.0

            if np.isnan(nes_val):
                continue

            is_sig = fdr_val < MEA_FDR_THRESH
            color = cmap(disc_norm(np.clip(nes_val, -2.0, 2.0)))
            marker = "^" if nes_val > 0 else "v"
            edge_color = "black" if is_sig else "none"
            edge_width = 1.2 if is_sig else 0.0

            ax.scatter(j, i, marker=marker, s=marker_size, c=[color],
                       edgecolors=edge_color, linewidths=edge_width,
                       zorder=3)

    # Y-axis: kinase labels colored by family
    ax.set_yticks(range(n_kin))
    ylabels = ax.set_yticklabels(kin_list, fontsize=6.5)
    for lbl, kin in zip(ylabels, kin_list):
        fam = fam_map.get(kin, "Other")
        lbl.set_color(KL_FAMILY_COLORS.get(fam, "#333333"))
        lbl.set_fontweight("bold")

    # X-axis
    ax.set_xticks(range(n_col))
    if show_xlabel:
        xlabels = ax.set_xticklabels(X_LABELS, fontsize=8, fontweight="bold")
        for lbl, disease in zip(xlabels, X_DISEASE):
            lbl.set_color(_DISEASE_COLORS.get(disease, "#333333"))
    else:
        ax.set_xticklabels([])

    ax.set_xlim(-0.5, n_col - 0.5)
    ax.set_ylim(-0.5, n_kin - 0.5)
    ax.invert_yaxis()

    # Condition-group separators
    for i in range(n_col - 1):
        if X_DISEASE[i] != X_DISEASE[i + 1]:
            ax.axvline(i + 0.5, color="black", linewidth=1.5,
                       linestyle="-", zorder=4)

    ax.set_aspect("auto")
    ax.set_facecolor("white")


def _draw_panel(ax, kin_list, fam_map, nes_pivot, fdr_pivot,
                show_xlabel=True):
    """Draw one column panel of the heatmap."""
    n_kin = len(kin_list)
    n_col = len(X_ORDER)

    if n_kin == 0:
        ax.set_visible(False)
        return

    from matplotlib.colors import BoundaryNorm
    bounds = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]
    cmap = plt.cm.RdBu_r
    disc_norm = BoundaryNorm(bounds, cmap.N)

    nes_arr = nes_pivot.loc[kin_list].reindex(columns=X_ORDER).values
    fdr_arr = fdr_pivot.loc[kin_list].reindex(columns=X_ORDER).values

    for i in range(n_kin):
        for j in range(n_col):
            nes_val = nes_arr[i, j]
            fdr_val = fdr_arr[i, j] if not np.isnan(fdr_arr[i, j]) else 1.0

            if np.isnan(nes_val):
                color = "#f5f5f5"
                text = ""
            else:
                color = cmap(disc_norm(np.clip(nes_val, -2.0, 2.0)))
                is_sig = fdr_val < MEA_FDR_THRESH
                text = f"{nes_val:.2f}{'*' if is_sig else ''}"

            rect = mpatches.FancyBboxPatch(
                (j - 0.47, i - 0.47), 0.94, 0.94,
                boxstyle="square,pad=0",
                facecolor=color,
                edgecolor="#e0e0e0",
                linewidth=0.3,
                zorder=2,
            )
            ax.add_patch(rect)

            if text:
                text_color = "white" if abs(nes_val) > 1.2 else "black"
                ax.text(j, i, text, ha="center", va="center",
                        fontsize=5, color=text_color, zorder=3)

    # Y-axis: kinase labels colored by family
    ax.set_yticks(range(n_kin))
    ylabels = ax.set_yticklabels(kin_list, fontsize=6.5)
    for lbl, kin in zip(ylabels, kin_list):
        fam = fam_map.get(kin, "Other")
        lbl.set_color(KL_FAMILY_COLORS.get(fam, "#333333"))
        lbl.set_fontweight("bold")

    # X-axis
    ax.set_xticks(range(n_col))
    if show_xlabel:
        xlabels = ax.set_xticklabels(X_LABELS, fontsize=8, fontweight="bold")
        for lbl, disease in zip(xlabels, X_DISEASE):
            lbl.set_color(_DISEASE_COLORS.get(disease, "#333333"))
    else:
        ax.set_xticklabels([])

    ax.set_xlim(-0.5, n_col - 0.5)
    ax.set_ylim(-0.5, n_kin - 0.5)
    ax.invert_yaxis()

    # Condition-group separators
    for i in range(n_col - 1):
        if X_DISEASE[i] != X_DISEASE[i + 1]:
            ax.axvline(i + 0.5, color="black", linewidth=1.5,
                       linestyle="-", zorder=4)

    ax.set_aspect("auto")


def _make_tissue_heatmap(mea, attr, fam_map, tissue_name, output_dir,
                         top_n=None):
    """Create and save a multi-panel heatmap for one tissue category."""
    tissue_attr = attr[attr["tissue_category"] == tissue_name]
    attributed_kinases = set(tissue_attr["kinase"].unique())

    if not attributed_kinases:
        print(f"  Skipping {tissue_name}: no attributed kinases")
        return

    mea_tissue = mea[mea["kinase"].isin(attributed_kinases)].copy()
    mea_tissue = mea_tissue[mea_tissue["contrast"].isin(X_ORDER)]

    # Optionally limit to top N by max |NES| among significant hits
    if top_n is not None:
        sig_mea = mea_tissue[mea_tissue["FDR"] < MEA_FDR_THRESH]
        if len(sig_mea) > 0:
            top_kins = (sig_mea.groupby("kinase")["NES"]
                        .apply(lambda x: x.abs().max())
                        .nlargest(top_n).index)
        else:
            top_kins = (mea_tissue.groupby("kinase")["NES"]
                        .apply(lambda x: x.abs().max())
                        .nlargest(top_n).index)
        mea_tissue = mea_tissue[mea_tissue["kinase"].isin(top_kins)].copy()

    # Build NES and FDR pivot matrices
    nes_pivot = mea_tissue.pivot_table(
        index="kinase", columns="contrast", values="NES", aggfunc="first")
    fdr_pivot = mea_tissue.pivot_table(
        index="kinase", columns="contrast", values="FDR", aggfunc="first")
    nes_pivot = nes_pivot.reindex(columns=X_ORDER)
    fdr_pivot = fdr_pivot.reindex(columns=X_ORDER)

    # Cluster kinases by NES profile similarity
    kin_order = _cluster_kinases(nes_pivot)
    n_kin = len(kin_order)

    if n_kin == 0:
        return

    # Split into side-by-side panels
    max_per_panel = 40
    num_panels = max(1, (n_kin + max_per_panel - 1) // max_per_panel)
    # Keep panels roughly equal
    kin_chunks = np.array_split(kin_order, num_panels)

    # Figure layout
    n_col = len(X_ORDER)
    panel_width = n_col * 0.7 + 1.5
    panel_heights = [len(chunk) * 0.28 + 0.5 for chunk in kin_chunks]
    max_panel_h = max(panel_heights)

    fig_width = panel_width * num_panels + 2.5
    fig_height = max_panel_h + 2.5

    fig, axes = plt.subplots(
        nrows=1, ncols=num_panels,
        figsize=(fig_width, fig_height),
        gridspec_kw={"wspace": 0.35},
        squeeze=False,
    )
    axes = axes.flatten()

    for i, (ax, chunk) in enumerate(zip(axes, kin_chunks)):
        _draw_panel(ax, list(chunk), fam_map, nes_pivot, fdr_pivot,
                    show_xlabel=True)

    # Title
    subclasses = sorted(tissue_attr["cell_type"].unique())
    subtitle = ", ".join(subclasses)
    fig.suptitle(tissue_name, fontsize=14, fontweight="bold", y=0.99)
    fig.text(0.5, 0.965, f"({subtitle})",
             fontsize=7, color="#666666", ha="center")

    # Discrete colorbar — place to the right of the last panel
    from matplotlib.colors import BoundaryNorm
    bounds = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]
    disc_norm = BoundaryNorm(bounds, plt.cm.RdBu_r.N)
    sm = plt.cm.ScalarMappable(cmap="RdBu_r", norm=disc_norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes.tolist(), shrink=0.5, pad=0.02,
                        aspect=20, ticks=bounds)
    cbar.set_label("NES", fontsize=9)

    # Legend text
    fig.text(0.99, 0.02, "* = FDR < 0.25    Labels colored by kinase family",
             fontsize=7, fontstyle="italic", color="#555555", ha="right")

    # Family color legend
    families_present = sorted(
        set(fam_map.get(k, "Other") for k in kin_order),
        key=lambda f: FAMILY_ORDER.index(f) if f in FAMILY_ORDER else 99)
    fam_handles = [
        plt.Line2D([], [], marker="s",
                   color=KL_FAMILY_COLORS.get(f, "#333"),
                   markersize=6, linestyle="None", label=f)
        for f in families_present
    ]
    fig.legend(handles=fam_handles, loc="lower left",
               fontsize=6, ncol=len(families_present),
               framealpha=0.9, title="Kinase family", title_fontsize=7,
               bbox_to_anchor=(0.01, 0.0))

    # Save
    safe_name = tissue_name.replace(" ", "_").replace("/", "_").lower()
    suffix = f"_top{top_n}" if top_n else ""

    png_path = os.path.join(output_dir, f"heatmap_{safe_name}{suffix}.png")
    fig.savefig(png_path, dpi=200, bbox_inches="tight")

    pdf_path = os.path.join(output_dir, f"heatmap_{safe_name}{suffix}.pdf")
    fig.savefig(pdf_path, bbox_inches="tight")

    plt.close(fig)
    print(f"  {tissue_name}: {n_kin} kinases, "
          f"{num_panels} panels -> {png_path}")


def _make_tissue_bubble(mea, attr, fam_map, tissue_name, output_dir,
                        top_n=None):
    """Create and save a multi-panel bubble plot for one tissue category.

    Same clustering and panel layout as heatmaps, but uses direction triangles:
    - Triangle up/down = NES sign (upregulated/downregulated)
    - Color = NES value (RdBu_r diverging)
    - Black border = FDR < 0.25 (significant)
    - No marker = missing data
    """
    tissue_attr = attr[attr["tissue_category"] == tissue_name]
    attributed_kinases = set(tissue_attr["kinase"].unique())

    if not attributed_kinases:
        return

    mea_tissue = mea[mea["kinase"].isin(attributed_kinases)].copy()
    mea_tissue = mea_tissue[mea_tissue["contrast"].isin(X_ORDER)]

    if top_n is not None:
        sig_mea = mea_tissue[mea_tissue["FDR"] < MEA_FDR_THRESH]
        if len(sig_mea) > 0:
            top_kins = (sig_mea.groupby("kinase")["NES"]
                        .apply(lambda x: x.abs().max())
                        .nlargest(top_n).index)
        else:
            top_kins = (mea_tissue.groupby("kinase")["NES"]
                        .apply(lambda x: x.abs().max())
                        .nlargest(top_n).index)
        mea_tissue = mea_tissue[mea_tissue["kinase"].isin(top_kins)].copy()

    nes_pivot = mea_tissue.pivot_table(
        index="kinase", columns="contrast", values="NES", aggfunc="first")
    fdr_pivot = mea_tissue.pivot_table(
        index="kinase", columns="contrast", values="FDR", aggfunc="first")
    nes_pivot = nes_pivot.reindex(columns=X_ORDER)
    fdr_pivot = fdr_pivot.reindex(columns=X_ORDER)

    kin_order = _cluster_kinases(nes_pivot)
    n_kin = len(kin_order)

    if n_kin == 0:
        return

    # Split into side-by-side panels (same as heatmap)
    max_per_panel = 40
    num_panels = max(1, (n_kin + max_per_panel - 1) // max_per_panel)
    kin_chunks = np.array_split(kin_order, num_panels)

    n_col = len(X_ORDER)
    panel_width = n_col * 0.7 + 1.5
    panel_heights = [len(chunk) * 0.28 + 0.5 for chunk in kin_chunks]
    max_panel_h = max(panel_heights)

    fig_width = panel_width * num_panels + 2.5
    fig_height = max_panel_h + 2.5

    fig, axes = plt.subplots(
        nrows=1, ncols=num_panels,
        figsize=(fig_width, fig_height),
        gridspec_kw={"wspace": 0.35},
        squeeze=False,
    )
    axes = axes.flatten()

    for i, (ax, chunk) in enumerate(zip(axes, kin_chunks)):
        _draw_bubble_panel(ax, list(chunk), fam_map, nes_pivot, fdr_pivot,
                           show_xlabel=True)

    # Title
    subclasses = sorted(tissue_attr["cell_type"].unique())
    subtitle = ", ".join(subclasses)
    fig.suptitle(tissue_name, fontsize=14, fontweight="bold", y=0.99)
    fig.text(0.5, 0.965, f"({subtitle})",
             fontsize=7, color="#666666", ha="center")

    # Discrete colorbar
    from matplotlib.colors import BoundaryNorm
    bounds = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]
    disc_norm = BoundaryNorm(bounds, plt.cm.RdBu_r.N)
    sm = plt.cm.ScalarMappable(cmap="RdBu_r", norm=disc_norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes.tolist(), shrink=0.5, pad=0.02,
                        aspect=20, ticks=bounds)
    cbar.set_label("NES", fontsize=9)

    # Legend: significance + direction
    sig_handle = plt.Line2D([], [], marker="^", color="gray", markersize=8,
                            markeredgecolor="black", markeredgewidth=1.2,
                            linestyle="None", label="FDR < 0.25")
    nonsig_handle = plt.Line2D([], [], marker="^", color="gray", markersize=8,
                               markeredgecolor="none", linestyle="None",
                               label="FDR \u2265 0.25")
    up_handle = plt.Line2D([], [], marker="^", color="#c62828", markersize=8,
                           linestyle="None", label="\u2191 Upregulated")
    down_handle = plt.Line2D([], [], marker="v", color="#1565c0", markersize=8,
                             linestyle="None", label="\u2193 Downregulated")
    fig.legend(handles=[sig_handle, nonsig_handle, up_handle, down_handle],
               loc="lower right", fontsize=7, ncol=2,
               framealpha=0.9, bbox_to_anchor=(0.99, 0.0))

    # Family color legend
    families_present = sorted(
        set(fam_map.get(k, "Other") for k in kin_order),
        key=lambda f: FAMILY_ORDER.index(f) if f in FAMILY_ORDER else 99)
    fam_handles = [
        plt.Line2D([], [], marker="s",
                   color=KL_FAMILY_COLORS.get(f, "#333"),
                   markersize=6, linestyle="None", label=f)
        for f in families_present
    ]
    fig.legend(handles=fam_handles, loc="lower left",
               fontsize=6, ncol=len(families_present),
               framealpha=0.9, title="Kinase family", title_fontsize=7,
               bbox_to_anchor=(0.01, 0.0))

    # Save
    safe_name = tissue_name.replace(" ", "_").replace("/", "_").lower()
    suffix = f"_top{top_n}" if top_n else ""

    png_path = os.path.join(output_dir, f"bubble_{safe_name}{suffix}.png")
    fig.savefig(png_path, dpi=200, bbox_inches="tight")

    pdf_path = os.path.join(output_dir, f"bubble_{safe_name}{suffix}.pdf")
    fig.savefig(pdf_path, bbox_inches="tight")

    plt.close(fig)
    print(f"  {tissue_name} (bubble): {n_kin} kinases, "
          f"{num_panels} panels -> {png_path}")


def _plot_winsorization_diagnostic(output_dir):
    """Diagnostic figure showing the effect of winsorization on site LFC distributions.

    Panel layout: 3 rows (diseases) x 3 cols (timepoints).
    Each panel shows a histogram of raw LFCs with winsorization bounds marked
    and clipped sites highlighted.
    """
    ols_path = os.path.join(config.KINASE_ATTRIBUTION_OUTPUT_DIR,
                            "site_level_ols.csv")
    winsor_path = os.path.join(config.KINASE_ATTRIBUTION_OUTPUT_DIR,
                               "winsorized_sites.csv")
    if not os.path.exists(winsor_path):
        print("  Winsorization diagnostic: no winsorized_sites.csv found")
        return

    ols = pd.read_csv(ols_path)
    winsor = pd.read_csv(winsor_path)

    fig, axes = plt.subplots(3, 3, figsize=(14, 10))

    for i, disease in enumerate(DISEASE_GROUPS):
        for j, tp in enumerate(TIMEPOINTS):
            ax = axes[i, j]
            contrast = f"{disease}_{tp}"
            col = f"stoich_lfc_{contrast}"

            vals = ols[col].dropna()
            ws = winsor[winsor["contrast"] == contrast]

            # Histogram
            bins = np.linspace(vals.quantile(0.001), vals.quantile(0.999), 80)
            ax.hist(vals, bins=bins, color="#b0bec5", edgecolor="none",
                    alpha=0.8, label="All sites")

            if len(ws) > 0:
                lo = ws["lower_bound"].iloc[0]
                hi = ws["upper_bound"].iloc[0]
                ax.axvline(lo, color="#c62828", linewidth=1.5, linestyle="--",
                           label=f"Bounds [{lo:.2f}, {hi:.2f}]")
                ax.axvline(hi, color="#c62828", linewidth=1.5, linestyle="--")

                # Highlight clipped sites on the x-axis
                ax.scatter(ws["original_lfc"], np.zeros(len(ws)),
                           c="#c62828", s=15, zorder=5, marker="|",
                           label=f"{len(ws)} clipped")

                # Annotate extreme outliers (top 3 by |original_lfc|)
                ws_sorted = ws.reindex(
                    ws["original_lfc"].abs().sort_values(ascending=False).index)
                for _, row in ws_sorted.head(5).iterrows():
                    if "gene_symbol" in row and pd.notna(row["gene_symbol"]):
                        ax.annotate(
                            row["gene_symbol"],
                            xy=(row["original_lfc"], 0),
                            xytext=(0, 15), textcoords="offset points",
                            fontsize=5, color="#c62828", ha="center",
                            arrowprops=dict(arrowstyle="-", color="#c62828",
                                            lw=0.5))

            ax.set_title(f"{_DISEASE_DISPLAY[disease]} {_TP_DISPLAY[tp]}",
                         fontsize=9, fontweight="bold",
                         color=_DISEASE_COLORS[disease])
            if j == 0:
                ax.set_ylabel("# Sites")
            if i == 2:
                ax.set_xlabel("Stoichiometry LFC")
            if i == 0 and j == 2:
                ax.legend(fontsize=6, loc="upper right")

    fig.suptitle("Winsorization Diagnostic: Site-Level Stoichiometry LFC Distributions\n"
                 f"Clipped at {config.MEA_WINSORIZE_PERCENTILE}th / "
                 f"{100 - config.MEA_WINSORIZE_PERCENTILE}th percentile",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    for ext in ("png", "pdf"):
        path = os.path.join(output_dir, f"winsorization_diagnostic.{ext}")
        fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Winsorization diagnostic -> {output_dir}/winsorization_diagnostic.png")


def _plot_direction_over_time(mea, attr, output_dir):
    """Diverging bar chart: up/down significant kinase counts per condition×timepoint.

    One panel per tissue category.  Bars above zero = upregulated (NES > 0),
    bars below = downregulated (NES < 0).  Only attributed kinases are counted.
    """
    # Merge tissue category onto MEA via attribution table
    # Join on both kinase AND contrast so each tissue only counts kinases
    # that were actually attributed in that contrast
    attr_kins = attr[["kinase", "tissue_category"]].drop_duplicates()
    sig = mea[mea["FDR"] < MEA_FDR_THRESH].copy()
    sig = sig.merge(attr_kins, on="kinase", how="inner")

    sig["direction"] = np.where(sig["NES"] > 0, "up", "down")

    # Parse disease / timepoint from contrast (e.g. "App_2mo")
    sig["disease"] = sig["contrast"].str.split("_").str[0]
    sig["timepoint"] = sig["contrast"].str.split("_").str[1]

    tissues = [t for t in TISSUE_ORDER if t in sig["tissue_category"].unique()]
    if not tissues:
        print("  Direction plot: no significant attributed kinases")
        return

    ncols = min(len(tissues), 4)
    nrows = (len(tissues) + ncols - 1) // ncols
    fig, axs = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows),
                            squeeze=False)
    axs_flat = axs.ravel()

    bar_width = 0.25
    x_base = np.arange(len(TIMEPOINTS))

    y_max = 0
    for i, tissue in enumerate(tissues):
        ax = axs_flat[i]
        ts = sig[sig["tissue_category"] == tissue]

        for j, disease in enumerate(DISEASE_GROUPS):
            up_vals, down_vals = [], []
            for tp in TIMEPOINTS:
                mask = (ts["disease"] == disease) & (ts["timepoint"] == tp)
                n_up = ts[mask & (ts["direction"] == "up")]["kinase"].nunique()
                n_down = ts[mask & (ts["direction"] == "down")]["kinase"].nunique()
                up_vals.append(n_up)
                down_vals.append(-n_down)

            x = x_base + j * bar_width
            color = _DISEASE_COLORS[disease]
            ax.bar(x, up_vals, bar_width, color=color, edgecolor="white",
                   linewidth=0.5, label=_DISEASE_DISPLAY[disease] if i == 0 else None)
            ax.bar(x, down_vals, bar_width, color=color, edgecolor="white",
                   linewidth=0.5, alpha=0.55)

            y_max = max(y_max, max(abs(v) for v in up_vals + down_vals))

        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(x_base + bar_width)
        ax.set_xticklabels(TIMEPOINTS)
        ax.set_title(tissue, fontsize=10, fontweight="bold")
        if i % ncols == 0:
            ax.set_ylabel("# Significant Kinases")

    # Consistent y-axis
    y_pad = max(y_max * 0.15, 2)
    for i in range(len(tissues)):
        axs_flat[i].set_ylim(-y_max - y_pad, y_max + y_pad)
    for j in range(len(tissues), nrows * ncols):
        axs_flat[j].axis("off")

    # Legend
    legend_handles = [
        mpatches.Patch(facecolor=_DISEASE_COLORS[d], edgecolor="white",
                       label=_DISEASE_DISPLAY[d])
        for d in DISEASE_GROUPS
    ]
    legend_handles.append(mpatches.Patch(facecolor="gray", alpha=1.0,
                                         label="\u2191 Upregulated"))
    legend_handles.append(mpatches.Patch(facecolor="gray", alpha=0.55,
                                         label="\u2193 Downregulated"))
    fig.legend(handles=legend_handles, loc="lower center",
               ncol=len(legend_handles), fontsize=9, frameon=True,
               bbox_to_anchor=(0.5, -0.02))

    fig.suptitle("Kinase Dysregulation Direction Over Time",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0.04, 1, 0.96])

    for ext in ("png", "pdf"):
        path = os.path.join(output_dir, f"direction_over_time.{ext}")
        fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Direction plot -> {output_dir}/direction_over_time.png")


def _plot_additivity_scatter(mea, attr, output_dir):
    """ApTt NES vs (App + Tau) additive prediction, one panel per timepoint.

    Tests whether the double-transgenic kinase enrichment is explained by
    simple addition of the single-transgenic effects, or shows synergy /
    antagonism (deviation from the diagonal).
    """
    # Get attributed kinases with tissue info
    attr_kins = attr[["kinase", "tissue_category"]].drop_duplicates()
    sig_kinases = attr_kins["kinase"].unique()

    # Build NES lookup: kinase × contrast
    mea_sub = mea[mea["kinase"].isin(sig_kinases)].copy()
    nes_pivot = mea_sub.pivot_table(index="kinase", columns="contrast",
                                     values="NES", aggfunc="first")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)

    cat_colors = {
        "App only": "#c62828",
        "Tau only": "#1565c0",
        "ApTt emergent": "#6a1b9a",
        "App+Tau (no ApTt)": "#9467bd",
        "App+ApTt": "#2ca02c",
        "Tau+ApTt": "#8c564b",
        "All three": "#e377c2",
        "None sig": "#cccccc",
    }

    for j, tp in enumerate(TIMEPOINTS):
        ax = axes[j]
        app_col = f"App_{tp}"
        tau_col = f"Tau_{tp}"
        aptt_col = f"ApTt_{tp}"

        # Need all three contrasts
        cols_needed = [app_col, tau_col, aptt_col]
        if not all(c in nes_pivot.columns for c in cols_needed):
            ax.set_visible(False)
            continue

        df = nes_pivot[cols_needed].dropna().copy()
        df["additive"] = df[app_col] + df[tau_col]
        df["aptt"] = df[aptt_col]

        if len(df) == 0:
            ax.set_visible(False)
            continue

        # Classify by significance
        fdr_pivot = mea_sub.pivot_table(index="kinase", columns="contrast",
                                         values="FDR", aggfunc="first")
        sig_app = fdr_pivot.reindex(df.index).get(app_col, pd.Series(dtype=float)) < MEA_FDR_THRESH
        sig_tau = fdr_pivot.reindex(df.index).get(tau_col, pd.Series(dtype=float)) < MEA_FDR_THRESH
        sig_aptt = fdr_pivot.reindex(df.index).get(aptt_col, pd.Series(dtype=float)) < MEA_FDR_THRESH

        cats = pd.Series("None sig", index=df.index)
        cats[sig_app & ~sig_tau & ~sig_aptt] = "App only"
        cats[~sig_app & sig_tau & ~sig_aptt] = "Tau only"
        cats[~sig_app & ~sig_tau & sig_aptt] = "ApTt emergent"
        cats[sig_app & sig_tau & ~sig_aptt] = "App+Tau (no ApTt)"
        cats[sig_app & ~sig_tau & sig_aptt] = "App+ApTt"
        cats[~sig_app & sig_tau & sig_aptt] = "Tau+ApTt"
        cats[sig_app & sig_tau & sig_aptt] = "All three"

        # Plot in order so interesting categories appear on top
        plot_order = ["None sig", "App only", "Tau only", "App+Tau (no ApTt)",
                      "App+ApTt", "Tau+ApTt", "All three", "ApTt emergent"]
        for cat in plot_order:
            mask = cats == cat
            if mask.sum() == 0:
                continue
            ax.scatter(df.loc[mask, "additive"], df.loc[mask, "aptt"],
                       c=cat_colors[cat], s=14, alpha=0.6, label=cat,
                       rasterized=True, edgecolors="none")

        # Diagonal
        lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]),
                max(ax.get_xlim()[1], ax.get_ylim()[1])]
        ax.plot(lims, lims, "k--", alpha=0.3, lw=1)
        ax.set_xlim(lims)
        ax.set_ylim(lims)

        # Pearson r
        mask_finite = np.isfinite(df["additive"]) & np.isfinite(df["aptt"])
        if mask_finite.sum() > 2:
            r, p = stats.pearsonr(df.loc[mask_finite, "additive"],
                                  df.loc[mask_finite, "aptt"])
            ax.text(0.05, 0.95, f"r = {r:.3f}", transform=ax.transAxes,
                    va="top", fontsize=10, fontweight="bold")

        ax.set_title(f"{_TP_DISPLAY[tp]}", fontsize=12, fontweight="bold")
        if j == 0:
            ax.set_ylabel("A\u00d7T NES")
        ax.set_xlabel("APP + Tau NES (additive prediction)")

    # Legend
    handles = [mpatches.Patch(color=cat_colors[c], label=c)
               for c in cat_colors if c != "None sig"]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8,
               bbox_to_anchor=(0.5, -0.04))

    fig.suptitle("A\u00d7T NES vs Additive Prediction (APP + Tau)\n"
                 "Deviation from diagonal indicates synergy or antagonism",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0.06, 1, 0.93))

    for ext in ("png", "pdf"):
        path = os.path.join(output_dir, f"additivity_scatter.{ext}")
        fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Additivity scatter -> {output_dir}/additivity_scatter.png")


def plot_attribution_heatmaps(output_dir=None, top_n=None):
    """Main entry: produce per-tissue-category heatmaps."""
    if output_dir is None:
        output_dir = os.path.join(config.ATTRIBUTION_RECOVERY_OUTPUT_DIR,
                                  "bubble_plots")
    os.makedirs(output_dir, exist_ok=True)

    mea, attr, fam_map = _load_data()

    print(f"Generating per-tissue heatmaps + bubble plots "
          f"({attr['kinase'].nunique()} attributed kinases, "
          f"{len(TISSUE_ORDER)} tissue categories)")

    for tissue in TISSUE_ORDER:
        _make_tissue_heatmap(mea, attr, fam_map, tissue, output_dir,
                             top_n=top_n)
        _make_tissue_bubble(mea, attr, fam_map, tissue, output_dir,
                            top_n=top_n)

    other_tissues = set(attr["tissue_category"].unique()) - set(TISSUE_ORDER)
    for tissue in sorted(other_tissues):
        _make_tissue_heatmap(mea, attr, fam_map, tissue, output_dir,
                             top_n=top_n)
        _make_tissue_bubble(mea, attr, fam_map, tissue, output_dir,
                            top_n=top_n)

    # Direction-over-time diverging bar chart
    _plot_direction_over_time(mea, attr, output_dir)

    # Additivity scatter (ApTt vs App+Tau)
    _plot_additivity_scatter(mea, attr, output_dir)

    # Winsorization diagnostic (if winsorized_sites.csv exists)
    _plot_winsorization_diagnostic(output_dir)

    print(f"\nAll plots saved to {output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Attribution heatmaps")
    parser.add_argument("--out", default=None, help="Output directory")
    parser.add_argument("--top", type=int, default=None,
                        help="Show only top N kinases per tissue category")
    args = parser.parse_args()
    plot_attribution_heatmaps(output_dir=args.out, top_n=args.top)

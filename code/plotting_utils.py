import os

import config
import kinase_library as kl
import matplotlib.colors as mcol
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from kinase_library.modules import data
from kinase_library.utils import _global_vars, exceptions
from natsort import natsort_keygen, natsorted
from scipy.cluster import hierarchy

# Fixed colorblind-friendly family palette (high-contrast on white, no yellow).
COLORBLIND_FAMILY_PALETTE = [
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # bluish green
    "#CC79A7",  # reddish purple
    "#56B4E9",  # sky blue
    "#E66100",  # dark orange
    "#5D3A9B",  # indigo
    "#1B9E77",  # teal green
    "#A6761D",  # brown
    "#666666",  # dark gray
]


def plot_bubblemap(
    lff_data,
    pval_data,
    brain_expressed_data=None,
    sig_lff=config.LFF_THRESH,
    sig_pval=config.PVAL_SIG,
    lff_percentile=None,
    kinases=None,
    sort_kins_by="family",
    cond_order=None,
    only_sig_kins=False,
    only_sig_conds=False,
    kin_clust=False,
    condition_clust=False,
    cluster_by=None,
    cluster_by_matrix=None,
    cluster_method="average",
    color_kins_by="family",
    kin_categories_colors=None,
    cond_colors=None,
    title=None,
    family_legend=True,
    lff_cbar=True,
    save_fig=False,
    figsize=(10, 5),
    subplot_width_ratios=(6, 1),
    lff_clim=(-2, 2),
    bubblesize_range=(20, 120),
    num_panels=4,
    vertical=True,
    xaxis_label="Condition",
    yaxis_label="Kinase",
    xlabel=True,
    xlabels_size=10,
    ylabel=True,
    ylabels_size=10,
    legend_position="right",
    cond_separator_fn=None,
    cond_minor_separator_fn=None,
    lff_color_bins=5,
):
    """
    Bubble map of kinase enrichment results.

    Encoding channels:
    - Size: |LFF| magnitude (larger = stronger effect)
    - Color: LFF (coolwarm diverging, blue=down, red=up)
    - Shape: direction triangles (up=upregulated, down=downregulated)
    - Border: significance (thick black=adj_pval <= sig_pval, no border otherwise)
    - Label marker: "*" on kinase axis labels when Allen Brain Atlas confirmed

    Row inclusion: when lff_percentile is set, includes kinases in the top/bottom N%
    of LFF for any comparison OR any kinase that meets significance
    (|LFF| >= sig_lff AND adj_pval <= sig_pval).  Callers should pre-filter rows
    with adj_pval >= PVAL_DISPLAY before passing data to this function.
    """

    if not (
        (lff_data.index.equals(pval_data.index))
        and (lff_data.columns.equals(pval_data.columns))
    ):
        raise ValueError(
            "lff_data and pval_data must have the same columns and indices."
        )

    _brain_expressed_orig = None
    if brain_expressed_data is not None:
        if not (
            (lff_data.index.equals(brain_expressed_data.index))
            and (lff_data.columns.equals(brain_expressed_data.columns))
        ):
            brain_expressed_data = brain_expressed_data.reindex(
                index=lff_data.index, columns=lff_data.columns
            ).fillna(False).astype(bool)
        _brain_expressed_orig = brain_expressed_data

    if kinases is None:
        kinases = lff_data.index.to_list()
    else:
        if len(kinases) != len(set(kinases)):
            kinases = list(dict.fromkeys(kinases))
    lff_data = lff_data.loc[kinases]
    pval_data = pval_data.loc[kinases]

    if not sort_kins_by:
        kins_order = kinases
    elif sort_kins_by == "family":
        kinase_info = (
            data.get_kinase_family(kinases)
            .reset_index()
            .sort_values(by=["FAMILY", "MATRIX_NAME"], key=natsort_keygen())
            .set_index("MATRIX_NAME")
        )
        kins_order = kinase_info.index
    elif sort_kins_by == "name":
        kins_order = natsorted(kinases)
    else:
        raise ValueError("sort_kins_by must be either 'family', 'name', or False.")

    if cond_order is None:
        cond_order = lff_data.columns

    label_info_col = None
    if color_kins_by:
        exceptions.check_color_kins_method(color_kins_by)
        label_info_col = color_kins_by.upper()
    kin_colors = {}
    use_family_coloring = bool(color_kins_by)

    sorted_lff_data = lff_data.loc[kins_order, cond_order]
    sorted_pval_data = pval_data.loc[kins_order, cond_order]
    if brain_expressed_data is not None:
        sorted_brain_data = _brain_expressed_orig.loc[kins_order, cond_order]
    else:
        sorted_brain_data = None

    # --- Row inclusion ---
    if lff_percentile is not None:
        # Percentile-based cell-level filter: show only cells in top/bottom N% per comparison
        notable_cell = pd.DataFrame(False, index=sorted_lff_data.index, columns=sorted_lff_data.columns)
        for col in sorted_lff_data.columns:
            col_vals = sorted_lff_data[col].dropna()
            if len(col_vals) == 0:
                continue
            upper = np.percentile(col_vals, 100 - lff_percentile)
            lower = np.percentile(col_vals, lff_percentile)
            notable_cell[col] = (sorted_lff_data[col] >= upper) | (sorted_lff_data[col] <= lower)
        # Always keep statistically significant cells, even if they are not in
        # top/bottom percentile bins, so true hits are never dropped from plots.
        sig_cell = (abs(sorted_lff_data) >= sig_lff) & (sorted_pval_data <= sig_pval)
        notable_cell = notable_cell | sig_cell
        # Mask non-notable cells, then keep only rows with at least one notable cell
        display_lff = sorted_lff_data.mask(~notable_cell)
        display_pval = sorted_pval_data.mask(~notable_cell)
        sig_kins = list(notable_cell.loc[notable_cell.any(axis=1)].index)
        if not sig_kins:
            print("  No notable kinases found — skipping plot.")
            return
        display_lff = display_lff.loc[sig_kins]
        display_pval = display_pval.loc[sig_kins]
        if sorted_brain_data is not None:
            sorted_brain_data = sorted_brain_data.loc[sig_kins]
    else:
        # Legacy: dual threshold cell-level filter
        sig_data = (abs(sorted_lff_data) >= sig_lff) & (sorted_pval_data <= sig_pval)
        display_lff = sorted_lff_data.mask(~sig_data)
        display_pval = sorted_pval_data.mask(~sig_data)
        if only_sig_kins:
            sig_kins = list(sig_data.loc[sig_data.any(axis=1)].index)
            if not sig_kins:
                print("  No significant kinases found — skipping plot.")
                return
            display_lff = display_lff.loc[sig_kins]
            display_pval = display_pval.loc[sig_kins]
            if sorted_brain_data is not None:
                sorted_brain_data = sorted_brain_data.loc[sig_kins]
        if only_sig_conds:
            sig_conds = list(sig_data.loc[:, sig_data.any()].columns)
            display_lff = display_lff[sig_conds]
            display_pval = display_pval[sig_conds]
            if sorted_brain_data is not None:
                sorted_brain_data = sorted_brain_data[sig_conds]

    if display_lff.empty or display_lff.shape[0] == 0:
        print("  No kinases to display — skipping plot.")
        return

    # Auto-disable family-based coloring when too many families are present to remain legible.
    if color_kins_by == "family":
        displayed_families = data.get_kinase_info(display_lff.index)["FAMILY"].dropna().unique()
        if len(displayed_families) > 10:
            use_family_coloring = False
            print(
                f"  Family coloring disabled: {len(displayed_families)} families "
                "in plotted kinases (>10)."
            )

    if use_family_coloring:
        kinome_data = data.get_kinase_info(display_lff.index)
        kin_categories_list = natsorted(kinome_data[label_info_col].dropna().unique())
        if kin_categories_colors is None:
            if color_kins_by == "family":
                palette = COLORBLIND_FAMILY_PALETTE
                kin_categories_colors = {
                    k: palette[i % len(palette)]
                    for i, k in enumerate(kin_categories_list)
                }
            else:
                kin_categories_colors = {
                    k: getattr(_global_vars, color_kins_by + "_colors")[k]
                    for k in kin_categories_list
                }
        else:
            if set(list(kin_categories_colors.keys())) < set(kin_categories_list):
                raise ValueError(
                    "Some families are missing from kin_categories_colors dictionary ({})".format(
                        list(
                            set(kin_categories_list)
                            - set(list(kin_categories_colors.keys()))
                        )
                    )
                )
        for kin, label_type in zip(kinome_data.index, kinome_data[label_info_col]):
            kin_colors[kin] = kin_categories_colors[label_type]

    if lff_clim is None:
        lff_clim = (min(display_lff.min().min(), 0), max(display_lff.max().max(), 0))
    if lff_color_bins is None or lff_color_bins < 2:
        lff_color_bins = 5
    # Tiered LFF coloring for improved print legibility.
    bin_edges = np.linspace(lff_clim[0], lff_clim[1], int(lff_color_bins) + 1)

    # P-value significance mask at cell granularity (kinase x comparison)
    pval_sig_mask = display_pval <= sig_pval
    # Brain-atlas flag at kinase granularity (used for label asterisk)
    brain_sig_kinases = set()
    if sorted_brain_data is not None:
        brain_row_flag = sorted_brain_data.fillna(False).astype(bool).any(axis=1)
        brain_sig_kinases = set(brain_row_flag.index[brain_row_flag])

    if kin_clust:
        if display_lff.shape[0] < 2:
            kin_clust = False
        if kin_clust and cluster_by == "pattern":
            # Cluster by comparison-profile similarity:
            # 1) tiered LFF (same tiering used for color),
            # 2) significance-aware direction,
            # 3) presence pattern across conditions/timepoints.
            lff_vals = display_lff.fillna(0).values
            present = (~display_lff.isna()).astype(float).values
            sig_mask_vals = ((display_pval <= sig_pval) & (~display_lff.isna())).astype(float).values
            direction_vals = np.sign(lff_vals)

            # Convert LFF to centered tier scores in [-1, 1].
            tier_idx = np.digitize(lff_vals, bin_edges[1:-1], right=False).astype(float)
            tier_center = (float(lff_color_bins) - 1.0) / 2.0
            tier_scale = max(tier_center, 1.0)
            tier_scores = ((tier_idx - tier_center) / tier_scale) * present

            # Significant direction channel (non-significant -> 0).
            sig_direction = direction_vals * sig_mask_vals

            cluster_matrix = np.concatenate([tier_scores, sig_direction, present], axis=1)
            Z = hierarchy.linkage(cluster_matrix, method=cluster_method)
        elif kin_clust:
            if cluster_by is None:
                raise ValueError("If kin_clust is True, cluster_by must be specified.")
            exceptions.check_cluster_method(cluster_by)
            if cluster_by == "lff":
                Z = hierarchy.linkage(
                    display_lff.fillna(0).values, method=cluster_method
                )
            elif cluster_by == "pval":
                Z = hierarchy.linkage(
                    (-np.log10(display_pval.fillna(1))).fillna(0).values, method=cluster_method
                )
            elif cluster_by == "both":
                Z = hierarchy.linkage(
                    (
                        np.sign(display_lff.fillna(0))
                        * (-np.log10(display_pval.fillna(1))).fillna(0)
                    ).values,
                    method=cluster_method,
                )
            elif cluster_by == "custom":
                if cluster_by_matrix is None:
                    raise ValueError("cluster_by_matrix must be specified.")
                Z = hierarchy.linkage(cluster_by_matrix.values, method=cluster_method)
            else:
                raise ValueError("Unsupported cluster_by mode.")

        if kin_clust:
            leaves = hierarchy.leaves_list(Z)
            display_lff = display_lff.iloc[leaves]
            display_pval = display_pval.iloc[leaves]
            pval_sig_mask = pval_sig_mask.iloc[leaves]
            if sorted_brain_data is not None:
                sorted_brain_data = sorted_brain_data.iloc[leaves]

    if condition_clust:
        if cluster_by is None:
            raise ValueError(
                "If condition_clust is True, cluster_by must be specified."
            )
        exceptions.check_cluster_method(cluster_by)
        if cluster_by == "lff":
            Z = hierarchy.linkage(
                display_lff.fillna(0).transpose().values, method=cluster_method
            )
        elif cluster_by == "pval":
            Z = hierarchy.linkage(
                (-np.log10(display_pval.fillna(1))).fillna(0).transpose().values,
                method=cluster_method,
            )
        elif cluster_by == "both":
            Z = hierarchy.linkage(
                (np.sign(display_lff.fillna(0)) * (-np.log10(display_pval.fillna(1))).fillna(0))
                .transpose()
                .values,
                method=cluster_method,
            )
        elif cluster_by == "custom":
            if cluster_by_matrix is None:
                raise ValueError("cluster_by_matrix must be specified.")
            Z = hierarchy.linkage(
                cluster_by_matrix.transpose().values, method=cluster_method
            )
        leaves = hierarchy.leaves_list(Z)
        display_lff = display_lff.iloc[:, leaves]
        display_pval = display_pval.iloc[:, leaves]
        pval_sig_mask = pval_sig_mask.iloc[:, leaves]
        if sorted_brain_data is not None:
            sorted_brain_data = sorted_brain_data.iloc[:, leaves]

    # Split into panels
    lff_matrices = np.array_split(display_lff, num_panels)
    pval_sig_matrices = np.array_split(pval_sig_mask, num_panels)
    if sorted_brain_data is not None:
        brain_matrices = np.array_split(sorted_brain_data, num_panels)
    else:
        brain_matrices = [None] * num_panels

    cnorm = mcol.BoundaryNorm(bin_edges, ncolors=plt.cm.coolwarm.N, clip=True)

    if vertical:
        lff_matrices = [mat.transpose() for mat in lff_matrices]
        pval_sig_matrices = [mat.transpose() for mat in pval_sig_matrices]
        brain_matrices = [mat.transpose() if mat is not None else None for mat in brain_matrices]

    fig = plt.figure(layout="constrained", figsize=figsize)
    if legend_position == "bottom":
        subfigs = fig.subfigures(nrows=2, ncols=1, height_ratios=(4, 1))
    else:
        subfigs = fig.subfigures(nrows=1, ncols=2, width_ratios=subplot_width_ratios)
    chart_subfig = subfigs[0]
    legend_subfig = subfigs[1]

    if not vertical:
        axes = chart_subfig.subplots(nrows=num_panels, ncols=1, squeeze=False).ravel()
    else:
        axes = chart_subfig.subplots(nrows=1, ncols=num_panels, squeeze=False).ravel()

    # Matplotlib scatter `s` is marker area (pt^2).  Scale the linear
    # multiplier down as panels increase so triangles stay readable.
    _triangle_scale = {1: 2.5, 2: 2.5, 3: 1.8, 4: 1.5}
    linear_mult = _triangle_scale.get(num_panels, 1.5)
    marker_size = bubblesize_range[1] * (linear_mult ** 2)
    sig_edge_color = "black"
    nonsig_edge_color = "none"
    sig_linewidth = 2.0
    nonsig_linewidth = 0.0

    for idx in range(num_panels):
        lff_matrix = lff_matrices[idx].loc[:, ::-1]
        pval_sig_matrix = pval_sig_matrices[idx].loc[:, ::-1]
        ax = axes[idx]

        # Melt to long format
        melt_lff = pd.melt(
            lff_matrix, ignore_index=False, var_name="condition", value_name="lff"
        ).set_index("condition", append=True)[::-1]
        melt_lff.index.names = ["kinase", "condition"]

        melt_pval_sig = pd.melt(
            pval_sig_matrix, ignore_index=False, var_name="condition", value_name="pval_sig"
        ).set_index("condition", append=True)[::-1]
        melt_pval_sig.index.names = ["kinase", "condition"]

        dfs_to_concat = [melt_lff, melt_pval_sig]

        all_data = pd.concat(dfs_to_concat, axis=1).reset_index()
        # Points with actual LFF values to plot
        plot_data = all_data[~all_data["lff"].isna()].copy()

        # Layer 1: invisible base layer for axis tick setup
        sns.scatterplot(
            x="kinase", y="condition", data=all_data,
            legend=False, ax=ax, alpha=0, size=0,
        )

        if not plot_data.empty:
            # Build numeric positions for categorical axes
            kinase_cats = all_data["kinase"].unique()
            condition_cats = all_data["condition"].unique()
            kin_to_num = {k: i for i, k in enumerate(kinase_cats)}
            cond_to_num = {c: i for i, c in enumerate(condition_cats)}

            # Render direction shapes: triangle-up for positive, triangle-down for negative
            for direction, marker_shape in [("+", "^"), ("-", "v")]:
                if direction == "+":
                    subset = plot_data[plot_data["lff"] > 0]
                else:
                    subset = plot_data[plot_data["lff"] <= 0]

                if subset.empty:
                    continue

                x_pos = subset["kinase"].map(kin_to_num).values
                y_pos = subset["condition"].map(cond_to_num).values
                colors = plt.cm.coolwarm(cnorm(subset["lff"].values))

                # Border encodes per-cell significance
                sig = subset["pval_sig"].fillna(False).values
                linewidths = np.where(sig, sig_linewidth, nonsig_linewidth)
                edgecolors = [sig_edge_color if s else nonsig_edge_color for s in sig]

                ax.scatter(
                    x_pos, y_pos, c=colors, s=marker_size, marker=marker_shape,
                    edgecolors=edgecolors, linewidths=linewidths, zorder=3,
                )

        minx = ax.get_xticks()[0]
        maxx = ax.get_xticks()[-1]
        if len(ax.get_xticks()) == 1:
            eps = 0.5
        else:
            eps = ((maxx - minx) / (len(ax.get_xticks()) - 1)) / 2
        ax.set_xlim(maxx + eps, minx - eps)
        miny = ax.get_yticks()[0]
        maxy = ax.get_yticks()[-1]
        if len(ax.get_yticks()) == 1:
            eps = 0.5
        else:
            eps = ((maxy - miny) / (len(ax.get_yticks()) - 1)) / 2
        ax.set_ylim(maxy + eps, miny - eps)

        ax.grid(which="major", color="k", linestyle=":")
        ax.set_axisbelow(True)
        ax.set_aspect("equal", "box")
        ax.tick_params(
            axis="x", which="major", labelsize=xlabels_size, labelrotation=90
        )
        ax.tick_params(axis="y", which="major", labelsize=ylabels_size)

    fig.canvas.draw()
    axes = chart_subfig.axes
    for ax in axes:
        if vertical:
            ax.set_xlabel(xaxis_label, fontsize=14, fontweight="bold")
            ax.set_ylabel(yaxis_label, fontsize=14, fontweight="bold")
        else:
            ax.set_xlabel(yaxis_label, fontsize=14, fontweight="bold")
            ax.set_ylabel(xaxis_label, fontsize=14, fontweight="bold")
        if xlabel:
            ax.xaxis.set_major_locator(mticker.FixedLocator(ax.get_xticks()))
            ax.set_xticklabels(ax.get_xticklabels(), fontweight="bold")
        else:
            ax.set(xlabel=None)
        if ylabel:
            ax.yaxis.set_major_locator(mticker.FixedLocator(ax.get_yticks()))
            ax.set_yticklabels(ax.get_yticklabels(), fontweight="bold")
        else:
            ax.set(ylabel=None)

        # Append * to kinase labels that are brain-atlas confirmed
        if vertical:
            old_labels = [lbl.get_text() for lbl in ax.get_yticklabels()]
            new_labels = [f"{k}*" if k in brain_sig_kinases else k for k in old_labels]
            ax.set_yticklabels(new_labels, fontweight="bold", fontsize=ylabels_size)
        else:
            old_labels = [lbl.get_text() for lbl in ax.get_xticklabels()]
            new_labels = [f"{k}*" if k in brain_sig_kinases else k for k in old_labels]
            ax.set_xticklabels(new_labels, fontweight="bold", fontsize=xlabels_size, rotation=90)

        if use_family_coloring:
            kin_axis_labels = ax.get_yticklabels() if vertical else ax.get_xticklabels()
            for lbl in kin_axis_labels:
                kin_name = lbl.get_text().rstrip("*")
                if kin_name in kin_colors:
                    lbl.set_color(kin_colors[kin_name])
        if cond_colors:
            if vertical:
                for lbl in ax.get_xticklabels():
                    label_color = cond_colors[lbl.get_text()]
                    lbl.set_color(label_color)
            else:
                for lbl in ax.get_yticklabels():
                    label_color = cond_colors[lbl.get_text()]
                    lbl.set_color(label_color)

    # Draw condition group separators
    if cond_separator_fn or cond_minor_separator_fn:
        for ax in chart_subfig.axes:
            if vertical:
                tick_labels = [lbl.get_text() for lbl in ax.get_xticklabels()]
                tick_positions = ax.get_xticks()
            else:
                tick_labels = [lbl.get_text() for lbl in ax.get_yticklabels()]
                tick_positions = ax.get_yticks()
            for i in range(len(tick_labels) - 1):
                if not tick_labels[i] or not tick_labels[i + 1]:
                    continue
                sep_pos = (tick_positions[i] + tick_positions[i + 1]) / 2
                if cond_separator_fn and cond_separator_fn(tick_labels[i]) != cond_separator_fn(tick_labels[i + 1]):
                    if vertical:
                        ax.axvline(x=sep_pos, color="black", linewidth=2, linestyle="-", zorder=2)
                    else:
                        ax.axhline(y=sep_pos, color="black", linewidth=2, linestyle="-", zorder=2)
                elif (cond_minor_separator_fn
                      and cond_minor_separator_fn(tick_labels[i]) != cond_minor_separator_fn(tick_labels[i + 1])):
                    if vertical:
                        ax.axvline(x=sep_pos, color="gray", linewidth=1, linestyle="--", zorder=2)
                    else:
                        ax.axhline(y=sep_pos, color="gray", linewidth=1, linestyle="--", zorder=2)

    # --- Legend rendering ---
    # Legend items: family, colorbar, shape/border/asterisk symbols
    num_legend_items = sum([
        bool(family_legend and use_family_coloring),
        bool(lff_cbar),
        True,   # shape + border + asterisk combined
    ])

    if legend_position == "bottom":
        legend_axes = legend_subfig.subplots(nrows=1, ncols=max(num_legend_items, 1), squeeze=False).ravel()
        legend_idx = 0

        if family_legend and use_family_coloring:
            patches = [
                mpatches.Patch(color=x[1], label=x[0])
                for x in kin_categories_colors.items()
            ]
            ncol = max(1, len(patches) // 4)
            legend_axes[legend_idx].legend(
                handles=patches, loc="center", title="Family",
                facecolor="white", ncol=ncol, fontsize=8,
            )
            legend_axes[legend_idx].axis("off")
            legend_idx += 1

        if lff_cbar:
            sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=cnorm)
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=legend_axes[legend_idx], orientation="horizontal",
                                fraction=0.8, pad=0.1)
            cbar.ax.set_xlabel("log\u2082(FF)")
            cbar.set_ticks(bin_edges)
            legend_axes[legend_idx].axis("off")
            legend_idx += 1

        # Shape, border legend (combined)
        symbol_handles = [
            plt.Line2D([0], [0], marker="^", color="w", markerfacecolor="salmon",
                       markeredgecolor="black", markersize=8, label="Upregulated"),
            plt.Line2D([0], [0], marker="v", color="w", markerfacecolor="cornflowerblue",
                       markeredgecolor="black", markersize=8, label="Downregulated"),
            plt.Line2D([0], [0], marker="^", color="w", markerfacecolor="lightgray",
                       markeredgecolor=sig_edge_color, markeredgewidth=sig_linewidth, markersize=8,
                       label=f"Significant (adj p \u2264 {sig_pval})"),
            plt.Line2D([0], [0], marker="^", color="w", markerfacecolor="lightgray",
                       markeredgecolor="none", markeredgewidth=0, markersize=8,
                       label="Not significant"),
            plt.Line2D([0], [0], linestyle="None", marker="None",
                       label="* = Brain atlas confirmed"),
        ]
        legend_axes[legend_idx].legend(
            handles=symbol_handles, loc="center",
            title="Symbols", facecolor="white", fontsize=8, ncol=2,
        )
        legend_axes[legend_idx].axis("off")
        legend_idx += 1

        for i in range(legend_idx, len(legend_axes)):
            legend_axes[i].axis("off")

    else:
        # Right-side vertical legend layout
        legend_axes = legend_subfig.subplots(nrows=3, ncols=1, squeeze=False).ravel()

        if family_legend and use_family_coloring:
            patches = [
                mpatches.Patch(color=x[1], label=x[0])
                for x in kin_categories_colors.items()
            ]
            legend_axes[0].legend(handles=patches, loc="center", title="Family", facecolor="white")
        legend_axes[0].axis("off")

        if lff_cbar:
            legend_axes[1].set(aspect=10)
            sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=cnorm)
            sm.set_array([])
            cbar = fig.colorbar(sm, cax=legend_axes[1])
            cbar.ax.set_title("log\u2082(FF)")
            cbar.set_ticks(bin_edges)
        else:
            legend_axes[1].axis("off")

        # Shape, border combined
        symbol_handles = [
            plt.Line2D([0], [0], marker="^", color="w", markerfacecolor="salmon",
                       markeredgecolor="black", markersize=8, label="Upregulated"),
            plt.Line2D([0], [0], marker="v", color="w", markerfacecolor="cornflowerblue",
                       markeredgecolor="black", markersize=8, label="Downregulated"),
            plt.Line2D([0], [0], marker="^", color="w", markerfacecolor="lightgray",
                       markeredgecolor=sig_edge_color, markeredgewidth=sig_linewidth, markersize=8,
                       label=f"Significant (adj p \u2264 {sig_pval})"),
            plt.Line2D([0], [0], marker="^", color="w", markerfacecolor="lightgray",
                       markeredgecolor="none", markeredgewidth=0, markersize=8,
                       label="Not significant"),
            plt.Line2D([0], [0], linestyle="None", marker="None",
                       label="* = Brain atlas confirmed"),
        ]
        legend_axes[2].legend(
            handles=symbol_handles, loc="center",
            title="Symbols", facecolor="white",
        )
        legend_axes[2].axis("off")

    chart_subfig.suptitle(title, fontsize=16)

    if save_fig:
        fig.savefig(save_fig, dpi=300, bbox_inches="tight")

def generate_bubble_map(
    input_filepath,
    seq_col,
    bg_col,
    fg_cols,
    lfc_threshold,
    title,
    output_dir="bubble_maps",
    sig_lff=config.LFF_THRESH,
    sig_pval=config.PVAL_SIG,
    kin_type=config.KIN_TYPE,
    kl_method=config.KL_METHOD,
    kl_thresh=config.KL_THRESH
):
    """
    Generates a bubble map visualization for kinase enrichment data.
    """
    df = pd.read_csv(input_filepath)[[seq_col, bg_col] + fg_cols]

    pval_data = pd.DataFrame()
    lff_data = pd.DataFrame()

    for fg_col in fg_cols:
        df[f"{fg_col}"] = np.log2((df[fg_col] + 1) / (df[bg_col] + 1))

        dpd = kl.DiffPhosData(dp_data=df, lfc_col=f"{fg_col}", lfc_thresh=lfc_threshold, seq_col=seq_col)
        enrich_data = dpd.kinase_enrichment(
                kin_type=kin_type,
                kl_method=kl_method,
                kl_thresh=kl_thresh,
            )
        enrich_data_df = enrich_data.combined_enrichment_results[
            ["most_sig_log2_freq_factor", "most_sig_fisher_adj_pval"]
        ]

        pval_col = enrich_data_df[["most_sig_fisher_adj_pval"]].rename(
            columns={"most_sig_fisher_adj_pval": fg_col})
        lff_col = enrich_data_df[["most_sig_log2_freq_factor"]].rename(
            columns={"most_sig_log2_freq_factor": fg_col})
        if pval_data.empty:
            pval_data = pval_col
            lff_data = lff_col
        else:
            pval_data = pval_data.join(pval_col, how="outer")
            lff_data = lff_data.join(lff_col, how="outer")

    plot_filename = os.path.join(output_dir, f"{title}_bubble_map.png")
    plot_bubblemap(
        lff_data,
        pval_data,
        sig_lff=sig_lff,
        sig_pval=sig_pval,
        title=title,
        only_sig_kins=False,
        kin_clust=False,
        condition_clust=False,
        lff_clim=(-2, 2),
        figsize=(15, 10),
        num_panels=6,
        xlabels_size=8,
        ylabels_size=8,
        save_fig=plot_filename,
        )


def plot_summary_heatmap(summary_df, save_fig=None, figsize=(20, 10)):
    """
    Bird's-eye overview heatmap: count of significant kinases per cluster/condition/timepoint/gender.

    Layout: 2×5 grid of subplots (one per cluster). Each subplot has:
    - Rows: conditions (TTau, AppP, ApTt)
    - Columns: timepoints (2mo, 4mo, 6mo) × gender (M, F)
    - Color intensity = count of significant kinases
    """
    conditions = ["Ttau", "AppP", "ApTt"]
    timepoints = ["2mo", "4mo", "6mo"]
    genders = ["M"]
    clusters = sorted(summary_df["cluster"].unique())

    col_labels = [f"{g} {t}" for t in timepoints for g in genders]

    nrows, ncols = 2, 5
    if len(clusters) <= 5:
        nrows = 1

    fig, axs = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    axs_flat = axs.ravel()

    vmax = summary_df["num_sig_kinases"].max() if not summary_df.empty else 1

    for i, cluster in enumerate(clusters):
        if i >= nrows * ncols:
            break
        ax = axs_flat[i]
        cluster_df = summary_df[summary_df["cluster"] == cluster]

        heatmap_data = np.zeros((len(conditions), len(col_labels)))
        for ri, cond in enumerate(conditions):
            for ci, col_label in enumerate(col_labels):
                gender, tp = col_label.split(" ")
                match = cluster_df[
                    (cluster_df["condition"] == cond)
                    & (cluster_df["gender"] == gender)
                    & (cluster_df["timepoint"] == tp)
                ]
                if not match.empty:
                    heatmap_data[ri, ci] = match["num_sig_kinases"].values[0]

        sns.heatmap(
            pd.DataFrame(heatmap_data, index=conditions, columns=col_labels),
            ax=ax, cmap="YlOrRd", vmin=0, vmax=vmax,
            annot=True, fmt=".0f", cbar=i == 0,
            linewidths=0.5, linecolor="white",
        )
        ax.set_title(cluster, fontsize=10, fontweight="bold")
        ax.set_ylabel("" if i % ncols != 0 else "Condition")
        ax.set_xlabel("")

    # Turn off unused axes
    for j in range(len(clusters), nrows * ncols):
        axs_flat[j].axis("off")

    fig.suptitle("Significant Kinase Counts by Cluster", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    if save_fig:
        fig.savefig(save_fig, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_direction_over_time(
    sig_df,
    condition_colors=None,
    lfc_threshold=None,
    title_prefix=None,
    panel_col="cluster",
    panel_order=None,
    save_fig=None,
    figsize=None,
):
    """
    Diverging bar chart: count of significant up/down-regulated kinases per
    timepoint × condition, collapsed across gender.

    One panel per value of `panel_col` (default: cluster). X-axis = timepoint,
    grouped bars per condition. Bars above zero = upregulated, below = downregulated.
    """
    if sig_df.empty:
        return

    df = sig_df.copy()
    if lfc_threshold is not None:
        df = df[df["lfc_threshold"] == lfc_threshold]
    if df.empty:
        return

    # Collapse gender: count unique kinases per (panel_col, condition, timepoint, direction)
    counts = (
        df.groupby([panel_col, "condition", "timepoint", "direction"])["kinase"]
        .nunique()
        .reset_index(name="count")
    )
    # Make downregulated counts negative for diverging display
    counts.loc[counts["direction"] == "-", "count"] *= -1

    up = counts[counts["direction"] == "+"].copy()
    down = counts[counts["direction"] == "-"].copy()

    panels = panel_order if panel_order is not None else sorted(df[panel_col].unique())
    conditions = ["Ttau", "AppP", "ApTt"]
    timepoints = ["2mo", "4mo", "6mo"]

    if condition_colors is None:
        condition_colors = {"Ttau": "#1f77b4", "AppP": "#ff7f0e", "ApTt": "#d62728"}

    ncols = min(len(panels), 4)
    nrows = (len(panels) + ncols - 1) // ncols
    if figsize is None:
        figsize = (5 * ncols, 4 * nrows)

    fig, axs = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    axs_flat = axs.ravel()

    bar_width = 0.25
    x_base = np.arange(len(timepoints))

    # Track global max for consistent y-axis
    y_max = 0

    for i, panel in enumerate(panels):
        ax = axs_flat[i]

        for j, cond in enumerate(conditions):
            up_vals = []
            down_vals = []
            for tp in timepoints:
                u = up[(up[panel_col] == panel) & (up["condition"] == cond) & (up["timepoint"] == tp)]
                d = down[(down[panel_col] == panel) & (down["condition"] == cond) & (down["timepoint"] == tp)]
                up_vals.append(u["count"].values[0] if len(u) else 0)
                down_vals.append(d["count"].values[0] if len(d) else 0)

            x = x_base + j * bar_width
            color = condition_colors.get(cond, "gray")
            ax.bar(x, up_vals, bar_width, color=color, edgecolor="white",
                   linewidth=0.5, label=cond if i == 0 else None)
            ax.bar(x, down_vals, bar_width, color=color, edgecolor="white",
                   linewidth=0.5, alpha=0.55)

            y_max = max(y_max, max(abs(v) for v in up_vals + down_vals))

        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(x_base + bar_width)
        ax.set_xticklabels(timepoints)
        ax.set_title(panel, fontsize=10, fontweight="bold")
        if i % ncols == 0:
            ax.set_ylabel("# Significant Kinases")

    # Apply consistent y-axis limits
    y_pad = max(y_max * 0.15, 2)
    for i in range(len(panels)):
        axs_flat[i].set_ylim(-y_max - y_pad, y_max + y_pad)

    # Turn off unused axes
    for j in range(len(panels), nrows * ncols):
        axs_flat[j].axis("off")

    # Build legend with direction annotations
    legend_handles = [
        mpatches.Patch(facecolor=condition_colors.get(c, "gray"), edgecolor="white", label=c)
        for c in conditions
    ]
    legend_handles.append(mpatches.Patch(facecolor="gray", alpha=1.0, label="↑ Upregulated"))
    legend_handles.append(mpatches.Patch(facecolor="gray", alpha=0.55, label="↓ Downregulated"))

    fig.legend(handles=legend_handles, loc="lower center", ncol=len(legend_handles),
               fontsize=9, frameon=True, bbox_to_anchor=(0.5, -0.02))

    lfc_label = f" (LFC≥{lfc_threshold})" if lfc_threshold is not None else ""
    prefix = title_prefix if title_prefix else "Kinase Dysregulation Direction Over Time"
    fig.suptitle(f"{prefix}{lfc_label}",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0.04, 1, 0.96])

    if save_fig:
        fig.savefig(save_fig, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_direction_by_family(
    sig_df,
    kinase_families,
    condition_colors=None,
    lfc_threshold=None,
    major_families=None,
    save_fig=None,
    figsize=None,
):
    """
    Faceted diverging bar chart: rows = kinase families, columns = clusters.
    Each panel shows up/down kinase counts per timepoint × condition, collapsed
    across gender. Minor families are grouped into 'Other'.

    Parameters
    ----------
    sig_df : DataFrame
        kinase_results table with columns: kinase, cluster, condition,
        timepoint, direction, lfc_threshold.
    kinase_families : dict
        Mapping of kinase name → family string (e.g. {"AKT1": "AGC"}).
    major_families : list, optional
        Families to show as individual rows. All others grouped as 'Other'.
        Default: ["AGC", "CAMK", "CMGC", "STE", "TKL"].
    """
    if sig_df.empty:
        return

    df = sig_df.copy()
    if lfc_threshold is not None:
        df = df[df["lfc_threshold"] == lfc_threshold]
    if df.empty:
        return

    if major_families is None:
        major_families = ["AGC", "CAMK", "CMGC", "STE", "TKL"]

    # Map kinases to families, collapse minor families to 'Other'
    df["family"] = df["kinase"].map(kinase_families).fillna("Other")
    df.loc[~df["family"].isin(major_families), "family"] = "Other"
    family_order = major_families + ["Other"]

    # Collapse gender: count unique kinases per (family, cluster, condition, timepoint, direction)
    counts = (
        df.groupby(["family", "cluster", "condition", "timepoint", "direction"])["kinase"]
        .nunique()
        .reset_index(name="count")
    )
    counts.loc[counts["direction"] == "-", "count"] *= -1

    up = counts[counts["direction"] == "+"].copy()
    down = counts[counts["direction"] == "-"].copy()

    clusters = sorted(df["cluster"].unique())
    conditions = ["Ttau", "AppP", "ApTt"]
    timepoints = ["2mo", "4mo", "6mo"]

    if condition_colors is None:
        condition_colors = {"Ttau": "#1f77b4", "AppP": "#ff7f0e", "ApTt": "#d62728"}

    nrows = len(family_order)
    ncols = len(clusters)
    if figsize is None:
        figsize = (3.2 * ncols, 2.5 * nrows)

    fig, axs = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)

    bar_width = 0.25
    x_base = np.arange(len(timepoints))

    # First pass: compute per-family y_max for consistent scaling within each row
    family_ymax = {}
    for family in family_order:
        ymax = 0
        for cluster in clusters:
            for cond in conditions:
                for tp in timepoints:
                    for src in [up, down]:
                        m = src[
                            (src["family"] == family)
                            & (src["cluster"] == cluster)
                            & (src["condition"] == cond)
                            & (src["timepoint"] == tp)
                        ]
                        if len(m):
                            ymax = max(ymax, abs(m["count"].values[0]))
        family_ymax[family] = ymax

    # Second pass: plot
    for ri, family in enumerate(family_order):
        y_max = family_ymax[family]
        y_pad = max(y_max * 0.2, 1)

        for ci, cluster in enumerate(clusters):
            ax = axs[ri, ci]

            for j, cond in enumerate(conditions):
                up_vals = []
                down_vals = []
                for tp in timepoints:
                    u = up[
                        (up["family"] == family)
                        & (up["cluster"] == cluster)
                        & (up["condition"] == cond)
                        & (up["timepoint"] == tp)
                    ]
                    d = down[
                        (down["family"] == family)
                        & (down["cluster"] == cluster)
                        & (down["condition"] == cond)
                        & (down["timepoint"] == tp)
                    ]
                    up_vals.append(u["count"].values[0] if len(u) else 0)
                    down_vals.append(d["count"].values[0] if len(d) else 0)

                x = x_base + j * bar_width
                color = condition_colors.get(cond, "gray")
                ax.bar(x, up_vals, bar_width, color=color, edgecolor="white", linewidth=0.5)
                ax.bar(x, down_vals, bar_width, color=color, edgecolor="white",
                       linewidth=0.5, alpha=0.55)

            ax.axhline(0, color="black", linewidth=0.6)
            ax.set_ylim(-y_max - y_pad, y_max + y_pad)
            ax.set_xticks(x_base + bar_width)
            ax.set_xticklabels(timepoints, fontsize=7)
            ax.tick_params(axis="y", labelsize=7)

            # Row labels (family) on the left column
            if ci == 0:
                ax.set_ylabel(family, fontsize=10, fontweight="bold")
            # Column labels (cluster) on the top row
            if ri == 0:
                ax.set_title(cluster, fontsize=9, fontweight="bold")

    # Legend
    legend_handles = [
        mpatches.Patch(facecolor=condition_colors.get(c, "gray"), edgecolor="white", label=c)
        for c in conditions
    ]
    legend_handles.append(mpatches.Patch(facecolor="gray", alpha=1.0, label="↑ Upregulated"))
    legend_handles.append(mpatches.Patch(facecolor="gray", alpha=0.55, label="↓ Downregulated"))

    fig.legend(handles=legend_handles, loc="lower center", ncol=len(legend_handles),
               fontsize=9, frameon=True, bbox_to_anchor=(0.5, -0.01))

    lfc_label = f" (LFC≥{lfc_threshold})" if lfc_threshold is not None else ""
    fig.suptitle(f"Kinase Dysregulation by Family Over Time{lfc_label}",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])

    if save_fig:
        fig.savefig(save_fig, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_kinase_activity_heatmap(
    enrichment_dir,
    comparisons,
    cluster,
    lfc_threshold,
    lff_thresh=config.LFF_THRESH,
    pval_thresh=config.PVAL_SIG,
    lff_percentile=config.BUBBLE_PERCENTILE,
    condition_colors=None,
    kinase_expression_info=None,
    save_fig=None,
):
    """
    Per-cluster kinase activity heatmap condensing all sex × condition × timepoint
    comparisons into a single figure.

    Rows = kinases (grouped by family), columns = comparisons ordered as
    Male(Ttau 2m 4m 6m, AppP 2m 4m 6m, ApTt 2m 4m 6m) | Female(…).
    Color = LFF (diverging RdBu_r), dots = significant hits.

    Parameters
    ----------
    enrichment_dir : str
        Path to directory containing per-comparison enrichment CSVs.
    comparisons : list[dict]
        Pre-filtered to one cluster and one LFC threshold.
    cluster, lfc_threshold : str/float
        Used for title and filename only (comparisons already filtered).
    lff_thresh, pval_thresh : float
        Significance thresholds for overlay markers.
    lff_percentile : float
        Top/bottom percentile of LFF that defines row inclusion (default 5).
    condition_colors : dict, optional
        Condition → color mapping for column color bar.
    kinase_expression_info : dict or None
        {kinase: {"brain_expressed": 0|1, ...}} for brain expression sidebar.
    save_fig : str, optional
        Output path for the figure.
    """
    # --- Build data matrices ---
    sex_order = ["M"]
    sex_map = {"ma": "M"}
    condition_order = ["Ttau", "AppP", "ApTt"]
    timepoint_order = ["2mo", "4mo", "6mo"]

    lff_series = {}
    pval_series = {}

    for comp in comparisons:
        bg, fg = comp["bg"], comp["fg"]
        # Support both percentile (new) and LFC (legacy) title formats
        pct = comp.get("percent_thresh", config.PERCENT_THRESH)
        title = f"{bg}_vs_{fg}_pct{pct}_lff{config.LFF_THRESH}".replace("/", "_")
        path = os.path.join(enrichment_dir, f"{title}_enrichment_results.csv")
        if not os.path.exists(path):
            continue

        res = pd.read_csv(path, index_col=0)

        sex = sex_map.get(comp["gender"], comp["gender"])
        col_name = f"{sex}_{comp['condition']}_{comp['timepoint']}"
        lff_series[col_name] = res["most_sig_log2_freq_factor"]
        pval_series[col_name] = res["most_sig_fisher_adj_pval"]

    if not lff_series:
        print(f"  No enrichment data for {cluster} {lfc_threshold} — skipping heatmap")
        return

    lff_df = pd.DataFrame(lff_series)
    pval_df = pd.DataFrame(pval_series)

    # --- Column ordering: Sex → Condition → Timepoint ---
    col_order = []
    for sex in sex_order:
        for cond in condition_order:
            for tp in timepoint_order:
                col = f"{sex}_{cond}_{tp}"
                if col in lff_df.columns:
                    col_order.append(col)

    lff_df = lff_df.reindex(columns=col_order)
    pval_df = pval_df.reindex(columns=col_order)

    # --- Row filtering: percentile-based (top/bottom N% of LFF per column) ---
    notable_mask = pd.DataFrame(False, index=lff_df.index, columns=lff_df.columns)
    for col in lff_df.columns:
        vals = lff_df[col].dropna()
        if len(vals) == 0:
            continue
        upper = np.percentile(vals, 100 - lff_percentile)
        lower = np.percentile(vals, lff_percentile)
        notable_mask[col] = (lff_df[col] >= upper) | (lff_df[col] <= lower)

    notable_rows = notable_mask.any(axis=1)
    if notable_rows.sum() == 0:
        print(f"  No notable kinases for {cluster} {lfc_threshold} — skipping heatmap")
        return

    lff_df = lff_df.loc[notable_rows]
    pval_df = pval_df.loc[notable_rows]

    # --- Annotate kinase families ---
    kinase_families = {}
    for kinase in lff_df.index:
        try:
            info = kl.get_kinase_info(kinase)
            kinase_families[kinase] = info["FAMILY"]
        except Exception:
            kinase_families[kinase] = "Other"

    family_series = pd.Series(kinase_families).reindex(lff_df.index).fillna("Other")

    # Sort rows: by family, then by mean |LFF| within family (descending)
    mean_abs_lff = lff_df.abs().mean(axis=1)
    sort_df = pd.DataFrame({"family": family_series, "mean_abs_lff": mean_abs_lff})
    sort_df = sort_df.sort_values(["family", "mean_abs_lff"], ascending=[True, False])
    row_order = sort_df.index.tolist()

    lff_df = lff_df.loc[row_order]
    pval_df = pval_df.loc[row_order]
    family_series = family_series.loc[row_order]

    # --- Family color sidebar ---
    all_families = sorted(family_series.unique())
    family_palette = sns.color_palette("Set2", len(all_families))
    family_color_map = dict(zip(all_families, family_palette))
    row_colors = family_series.map(family_color_map).rename("Family")

    # --- Column color bar (condition) ---
    if condition_colors is None:
        condition_colors = {"Ttau": "#1f77b4", "AppP": "#ff7f0e", "ApTt": "#d62728"}

    col_cond = [c.split("_")[1] for c in col_order]
    col_colors = pd.Series(
        [condition_colors.get(c, "gray") for c in col_cond],
        index=col_order,
        name="Condition",
    )

    # --- Pretty column labels (keep sex prefix for uniqueness) ---
    display_cols = [c.replace("_", " ") for c in col_order]
    lff_plot = lff_df.copy()
    lff_plot.columns = display_cols
    pval_plot = pval_df.copy()
    pval_plot.columns = display_cols
    row_colors.index = lff_plot.index
    col_colors.index = display_cols

    # --- Determine symmetric color limits ---
    finite_vals = lff_plot.values[np.isfinite(lff_plot.values)]
    vmax = max(2.0, np.nanpercentile(finite_vals, 97)) if len(finite_vals) else 2.0
    vmax = np.ceil(vmax * 2) / 2  # round up to nearest 0.5

    # --- Plot ---
    n_rows = len(lff_plot)
    n_cols = len(display_cols)
    fig_width = max(10, n_cols * 0.55 + 3)
    fig_height = max(6, n_rows * 0.3 + 3)

    # --- Build figure manually (avoids clustermap dendrogram whitespace) ---
    # Layout: [ylabel_margin | row_colors | heatmap | colorbar + legends]

    # Measure longest kinase name for left margin
    max_kin_len = max(len(str(k)) for k in lff_plot.index)
    ylabel_margin = max(1.0, max_kin_len * 0.07)  # approx inches for y-labels

    row_colors_width = 0.2  # family sidebar
    brain_col_width = 0.15 if kinase_expression_info else 0  # brain expression sidebar
    label_gap = 0.15        # gap between y-labels and sidebar
    gap = 0.05              # gap between sidebar and heatmap
    heatmap_width = max(6, n_cols * 0.5)
    right_margin = 2.5      # colorbar + legends
    fig_width = ylabel_margin + label_gap + row_colors_width + brain_col_width + gap + heatmap_width + right_margin

    col_colors_height = 0.2
    heatmap_height = max(3, n_rows * 0.25)
    xlabel_height = 1.8     # rotated x-labels
    top_margin = 1.0        # title + sex labels + col_colors
    fig_height = top_margin + col_colors_height + heatmap_height + xlabel_height

    fig = plt.figure(figsize=(fig_width, fig_height))

    # Fractional positions
    left = (ylabel_margin + label_gap + row_colors_width + brain_col_width + gap) / fig_width
    bottom = xlabel_height / fig_height
    w = heatmap_width / fig_width
    h = heatmap_height / fig_height
    col_h = col_colors_height / fig_height
    rc_w = row_colors_width / fig_width
    rc_left = (ylabel_margin + label_gap) / fig_width
    bc_w = brain_col_width / fig_width if brain_col_width else 0
    bc_left = rc_left + rc_w

    ax = fig.add_axes([left, bottom, w, h])
    ax_col_colors = fig.add_axes([left, bottom + h, w, col_h])
    ax_row_colors = fig.add_axes([rc_left, bottom, rc_w, h])

    # Draw main heatmap
    lff_arr = lff_plot.fillna(0).values
    im = ax.imshow(lff_arr, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                   interpolation="nearest")
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(display_cols, rotation=90, fontsize=7, ha="center")
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels([])  # labels go on row_colors axes
    ax.tick_params(axis="x", length=2, pad=2)
    ax.tick_params(axis="y", length=0)

    # Grid lines between cells
    for edge in range(n_rows + 1):
        ax.axhline(edge - 0.5, color="white", linewidth=0.3)
    for edge in range(n_cols + 1):
        ax.axvline(edge - 0.5, color="white", linewidth=0.3)

    # --- Significance overlay (dots) ---
    pval_arr = pval_plot.values
    lff_arr_raw = lff_plot.values
    for i in range(n_rows):
        for j in range(n_cols):
            p = pval_arr[i, j]
            lv = lff_arr_raw[i, j]
            if np.isfinite(p) and np.isfinite(lv) and p <= pval_thresh and abs(lv) >= lff_thresh:
                ax.plot(j, i, marker="o", color="black",
                        markersize=2.5, markeredgewidth=0)

    # Vertical separators between condition blocks
    col_idx = 0
    for sex in sex_order:
        for cond in condition_order:
            nc = sum(1 for c in col_order if c.startswith(f"{sex}_{cond}_"))
            col_idx += nc
            if col_idx < n_cols:
                ax.axvline(x=col_idx - 0.5, color="gray", linewidth=0.8, linestyle="--")

    # --- Column color bar (condition) ---
    col_color_arr = np.array([[mcol.to_rgba(col_colors.iloc[j]) for j in range(n_cols)]])
    ax_col_colors.imshow(col_color_arr, aspect="auto", interpolation="nearest")
    ax_col_colors.set_xticks([])
    ax_col_colors.set_yticks([])
    ax_col_colors.set_xlim(ax.get_xlim())
    for spine in ax_col_colors.spines.values():
        spine.set_visible(False)

    # --- Row color bar (family) ---
    row_color_arr = np.array([[mcol.to_rgba(family_color_map[family_series.iloc[i]])]
                              for i in range(n_rows)])
    ax_row_colors.imshow(row_color_arr, aspect="auto", interpolation="nearest")
    ax_row_colors.set_xticks([])
    ax_row_colors.set_ylim(ax.get_ylim())
    # Re-apply kinase y-labels after imshow
    ax_row_colors.set_yticks(range(n_rows))
    ax_row_colors.set_yticklabels(lff_plot.index, fontsize=7)
    ax_row_colors.tick_params(axis="y", length=0, pad=2)
    for spine in ax_row_colors.spines.values():
        spine.set_visible(False)

    # --- Brain expression sidebar ---
    if kinase_expression_info and bc_w > 0:
        ax_brain = fig.add_axes([bc_left, bottom, bc_w, h])
        brain_colors = []
        for kinase in lff_plot.index:
            expressed = kinase_expression_info.get(kinase, {}).get("brain_expressed", 0)
            brain_colors.append("#2ca02c" if expressed else "#d3d3d3")
        brain_arr = np.array([[mcol.to_rgba(c)] for c in brain_colors])
        ax_brain.imshow(brain_arr, aspect="auto", interpolation="nearest")
        ax_brain.set_xticks([])
        ax_brain.set_yticks([])
        ax_brain.set_ylim(ax.get_ylim())
        for spine in ax_brain.spines.values():
            spine.set_visible(False)

    # --- Colorbar ---
    cbar_left = left + w + 0.02
    cbar_ax = fig.add_axes([cbar_left, bottom + h * 0.2, 0.015, h * 0.6])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("Log\u2082 Frequency Factor", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    # --- Legends (to the right of colorbar) ---
    legend_left = cbar_left + 0.06
    # Family legend
    family_handles = [
        mpatches.Patch(facecolor=family_color_map[f], label=f) for f in all_families
    ]
    fig.legend(
        handles=family_handles, title="Family",
        loc="upper left", bbox_to_anchor=(legend_left, bottom + h),
        fontsize=7, title_fontsize=8, frameon=True, borderpad=0.5,
        bbox_transform=fig.transFigure,
    )

    # Significance dot legend
    sig_handle = plt.Line2D(
        [0], [0], marker="o", color="black", linestyle="None",
        markersize=3, label=f"adj p \u2264 {pval_thresh} & |LFF| \u2265 {lff_thresh}",
    )
    fig.legend(
        handles=[sig_handle],
        loc="upper left", bbox_to_anchor=(legend_left, bottom + h * 0.35),
        fontsize=7, frameon=True, borderpad=0.5,
        bbox_transform=fig.transFigure,
    )

    # Brain expression legend
    if kinase_expression_info:
        brain_handles = [
            mpatches.Patch(facecolor="#2ca02c", label="Brain expressed"),
            mpatches.Patch(facecolor="#d3d3d3", label="Not confirmed"),
        ]
        fig.legend(
            handles=brain_handles, title="Allen Brain Atlas",
            loc="upper left", bbox_to_anchor=(legend_left, bottom + h * 0.2),
            fontsize=7, title_fontsize=8, frameon=True, borderpad=0.5,
            bbox_transform=fig.transFigure,
        )

    fig.suptitle(
        f"{cluster} \u2014 Kinase Activity ({lfc_threshold})",
        fontsize=13, fontweight="bold",
        x=(left + w / 2), y=1 - 0.1 / fig_height,
    )

    if save_fig:
        fig.savefig(save_fig, dpi=300, bbox_inches="tight", pad_inches=0.3)
    plt.close("all")
    print(f"  Heatmap: {save_fig} ({n_rows} kinases \u00d7 {n_cols} comparisons)")


# ---------------------------------------------------------------------------
# Kinase Ranking Visualization (Section B of redesign plan)
# ---------------------------------------------------------------------------

def compute_kinase_rankings(
    enrichment_dir,
    comparisons,
    pval_thresh=config.PVAL_SIG,
    lff_thresh=config.LFF_THRESH,
    kinase_expression_info=None,
    cluster_filter=None,
):
    """Rank ALL kinases across comparisons with significance tier annotation.

    Every kinase that appears in any enrichment CSV is included.  Each kinase
    is annotated with a significance_tier based on its best cell:
      - significant: |LFF| >= lff_thresh AND adj_pval <= pval_thresh
      - display: in top/bottom BUBBLE_PERCENTILE% of LFF for that comparison
                 AND adj_pval < PVAL_DISPLAY (but not significant)
      - non_significant: everything else
    Statistics are computed across all comparisons where it appears.

    Parameters
    ----------
    enrichment_dir : str
        Path to the directory containing per-comparison enrichment CSVs.
    comparisons : list[dict]
        Comparison dicts (must have keys: bg, fg, condition, timepoint, cluster, gender).
    pval_thresh : float
        Adjusted p-value threshold for "significant" tier (default config.PVAL_SIG).
    lff_thresh : float
        |LFF| threshold for "significant" tier (default config.LFF_THRESH).
    kinase_expression_info : dict or None
        {kinase: {"brain_expressed": 0|1, ...}} from the prepare step.
    cluster_filter : str or None
        If set, restrict to comparisons for this cluster only.

    Returns
    -------
    pd.DataFrame with columns:
        rank, kinase, significance_tier, n_sig_comparisons, n_total_comparisons,
        clusters_sig, clusters_total, cluster_list, timepoints, conditions,
        mean_abs_lff, best_adj_pval, lff_Ttau, lff_AppP, lff_ApTt, brain_expressed
    Sorted by n_sig_comparisons desc, n_total_comparisons desc, mean |LFF| desc.
    """
    # Collect ALL appearances and flag significance per kinase×comparison
    kinase_all = {}   # kinase -> list of {comp metadata + lff + adj_pval + is_sig}

    for comp in comparisons:
        if cluster_filter and comp["cluster"] != cluster_filter:
            continue

        bg, fg = comp["bg"], comp["fg"]
        pct = comp.get("percent_thresh", 5)
        title = f"{bg}_vs_{fg}_pct{pct}_lff{lff_thresh}".replace("/", "_")

        results_path = os.path.join(enrichment_dir, f"{title}_enrichment_results.csv")
        if not os.path.exists(results_path):
            continue

        res = pd.read_csv(results_path, index_col=0)

        # Compute percentile thresholds for this comparison
        lff_series = res["most_sig_log2_freq_factor"].dropna()
        if len(lff_series) > 0:
            pct_upper = np.percentile(lff_series, 100 - config.BUBBLE_PERCENTILE)
            pct_lower = np.percentile(lff_series, config.BUBBLE_PERCENTILE)
        else:
            pct_upper, pct_lower = np.inf, -np.inf

        for kinase in res.index:
            lff_val = res.loc[kinase, "most_sig_log2_freq_factor"]
            adj_pval = res.loc[kinase, "most_sig_fisher_adj_pval"]
            is_sig = (abs(lff_val) >= lff_thresh) and (adj_pval <= pval_thresh)
            in_pct_tail = (lff_val >= pct_upper) or (lff_val <= pct_lower)
            is_display = (not is_sig) and in_pct_tail and (adj_pval < config.PVAL_DISPLAY)

            if kinase not in kinase_all:
                kinase_all[kinase] = []
            kinase_all[kinase].append({
                "condition": comp["condition"],
                "timepoint": comp["timepoint"],
                "cluster": comp["cluster"],
                "lff": lff_val,
                "adj_pval": adj_pval,
                "is_sig": is_sig,
                "is_display": is_display,
            })

    if not kinase_all:
        return pd.DataFrame()

    clusters = sorted(set(
        c["cluster"] for c in comparisons
        if not cluster_filter or c["cluster"] == cluster_filter
    ))

    records = []
    for kinase, appearances in kinase_all.items():
        sig_hits = [a for a in appearances if a["is_sig"]]
        clusters_sig = sorted(set(h["cluster"] for h in sig_hits))
        clusters_total = sorted(set(a["cluster"] for a in appearances))
        timepoints_present = sorted(set(a["timepoint"] for a in appearances))
        conditions_present = sorted(set(a["condition"] for a in appearances))
        mean_abs_lff = round(np.mean([abs(a["lff"]) for a in appearances]), 3)
        best_adj_pval = round(min(a["adj_pval"] for a in appearances), 4)

        # Tier based on best comparison
        display_hits = [a for a in appearances if a["is_display"]]
        if sig_hits:
            tier = "significant"
        elif display_hits:
            tier = "display"
        else:
            tier = "non_significant"

        condition_lff = {}
        for cond in ["Ttau", "AppP", "ApTt"]:
            cond_vals = [a["lff"] for a in appearances if a["condition"] == cond]
            condition_lff[cond] = round(np.mean(cond_vals), 3) if cond_vals else None

        brain_expressed = "Yes"
        if kinase_expression_info:
            brain_expressed = (
                "Yes" if kinase_expression_info.get(kinase, {}).get("brain_expressed", 0) else "No"
            )

        records.append({
            "kinase": kinase,
            "significance_tier": tier,
            "n_sig_comparisons": len(sig_hits),
            "n_total_comparisons": len(appearances),
            "clusters_sig": f"{len(clusters_sig)}/{len(clusters)}",
            "clusters_total": f"{len(clusters_total)}/{len(clusters)}",
            "cluster_list": ", ".join(clusters_total),
            "timepoints": ", ".join(timepoints_present),
            "conditions": ", ".join(conditions_present),
            "mean_abs_lff": mean_abs_lff,
            "best_adj_pval": best_adj_pval,
            "lff_Ttau": condition_lff["Ttau"],
            "lff_AppP": condition_lff["AppP"],
            "lff_ApTt": condition_lff["ApTt"],
            "brain_expressed": brain_expressed,
        })

    df = pd.DataFrame(records)

    # Sort: most significant comparisons first, then total appearances, then effect size
    df = df.sort_values(
        ["n_sig_comparisons", "n_total_comparisons", "mean_abs_lff"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    df.insert(0, "rank", range(1, len(df) + 1))
    return df

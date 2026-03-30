"""Temporal trajectory analysis of kinase dysregulation in AD models.

Visualizes direction of dysregulation over time (overall and per cell type)
and provides a cluster × condition × timepoint heatmap for burden patterns.

Usage:
    python code/analyze_temporal_trajectories.py               # all 4 modes
    python code/analyze_temporal_trajectories.py --mode deconv  # single mode
"""

import argparse
import os
import sys

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
import config
import downstream_utils as du
from plotting_utils import plot_direction_over_time as _plot_direction_panels


def build_lff_matrix(mode):
    """Build a (kinase, condition, cluster, timepoint) -> LFF lookup from raw enrichment CSVs.

    Returns a DataFrame with columns: kinase, condition, timepoint, cluster, lff, direction.
    """
    all_results = du.load_all_enrichment_results(mode)
    if du.graceful_empty(all_results, f"{mode} enrichment results"):
        return pd.DataFrame()

    rows = []
    for comp_name, df in all_results.items():
        parsed = du.parse_comparison_name(comp_name, mode)
        if parsed is None:
            continue
        for kinase, row in df.iterrows():
            rows.append({
                "kinase": kinase,
                "condition": parsed["condition"],
                "timepoint": parsed["timepoint"],
                "cluster": parsed["cluster"],
                "lff": row.get("most_sig_log2_freq_factor", np.nan),
                "direction": row.get("most_sig_direction", ""),
                "adj_pval": row.get("most_sig_fisher_adj_pval", np.nan),
            })
    return pd.DataFrame(rows)


def build_temporal_table(lff_matrix, sig_thresh=config.PVAL_SIG, lff_thresh=config.LFF_THRESH):
    """Build a per-(kinase, condition, cluster) table with LFF/direction/significance at each timepoint.

    Returns a DataFrame with one row per (kinase, condition, cluster) that is
    significant at any timepoint. Columns include the raw LFF, direction, and
    a boolean significance flag at each timepoint — no subjective pattern labels.
    """
    tp_order = ["2mo", "4mo", "6mo"]
    records = []
    for (kinase, condition, cluster), group in lff_matrix.groupby(["kinase", "condition", "cluster"]):
        group_indexed = group.set_index("timepoint").reindex(tp_order)

        row_data = {"kinase": kinase, "condition": condition, "cluster": cluster}
        any_sig = False
        for tp in tp_order:
            if tp not in group_indexed.index or pd.isna(group_indexed.loc[tp, "adj_pval"]):
                row_data[f"sig_{tp}"] = False
                row_data[f"direction_{tp}"] = ""
                row_data[f"lff_{tp}"] = np.nan
            else:
                r = group_indexed.loc[tp]
                is_sig = (r["adj_pval"] <= sig_thresh) and (abs(r["lff"]) >= lff_thresh)
                row_data[f"sig_{tp}"] = is_sig
                row_data[f"direction_{tp}"] = r["direction"]
                row_data[f"lff_{tp}"] = round(r["lff"], 4)
                if is_sig:
                    any_sig = True

        if any_sig:
            records.append(row_data)

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_direction_overall(sig_kinases, out_dir, mode):
    """Diverging bar: up vs down kinase counts per condition x timepoint, aggregated across clusters."""
    if du.graceful_empty(sig_kinases, "sig kinases for direction plot"):
        return

    conditions = [c for c in du.CONDITIONS if c in sig_kinases["condition"].unique()]
    timepoints = [t for t in du.TIMEPOINTS if t in sig_kinases["timepoint"].unique()]
    if not conditions or not timepoints:
        return

    fig, axes = plt.subplots(1, len(conditions), figsize=(3.5 * len(conditions), 4),
                             sharey=True)
    if len(conditions) == 1:
        axes = [axes]

    for ax, cond in zip(axes, conditions):
        up_counts = []
        dn_counts = []
        for tp in timepoints:
            sub = sig_kinases[(sig_kinases["condition"] == cond) & (sig_kinases["timepoint"] == tp)]
            n_up = (sub["direction"] == "+").sum()
            n_dn = (sub["direction"] == "-").sum()
            up_counts.append(n_up)
            dn_counts.append(-n_dn)

        x = np.arange(len(timepoints))
        ax.bar(x, up_counts, color="#d62728", alpha=0.8, label="Upregulated" if cond == conditions[0] else "")
        ax.bar(x, dn_counts, color="#1f77b4", alpha=0.8, label="Downregulated" if cond == conditions[0] else "")
        ax.axhline(0, color="black", linewidth=0.5)

        for i, (u, d) in enumerate(zip(up_counts, dn_counts)):
            if u > 0:
                ax.text(i, u + 1, str(u), ha="center", fontsize=8, color="#d62728")
            if d < 0:
                ax.text(i, d - 3, str(-d), ha="center", fontsize=8, color="#1f77b4")

        ax.set_xticks(x)
        ax.set_xticklabels(timepoints)
        ax.set_title(cond, fontsize=11, fontweight="bold",
                     color=config.CONDITION_COLORS.get(cond, "black"))
        if ax == axes[0]:
            ax.set_ylabel("Significant kinase hits")

    axes[0].legend(fontsize=8, loc="lower left")
    fig.suptitle(f"Direction of kinase dysregulation over time — {mode}", fontsize=11)
    fig.tight_layout()
    du.save_fig(fig, os.path.join(out_dir, "direction_over_time.png"))


def plot_direction_by_cluster(sig_kinases, out_dir, mode):
    """Diverging bar: up vs down per timepoint x condition, one panel per cluster.

    Uses the pipeline's plotting_utils.plot_direction_over_time for consistency.
    """
    if du.graceful_empty(sig_kinases, "sig kinases for cluster direction plot"):
        return

    clusters = sorted(sig_kinases["cluster"].unique())
    if len(clusters) <= 1:
        print("  [SKIP] direction_by_cluster: only 1 cluster (bulk mode)")
        return

    save_path = os.path.join(out_dir, "direction_by_cluster.png")
    _plot_direction_panels(
        sig_kinases,
        condition_colors=config.CONDITION_COLORS,
        title_prefix="Kinase Dysregulation by Cell Type",
        panel_col="cluster",
        panel_order=clusters,
        save_fig=save_path,
    )
    print(f"  Saved: {save_path}")


def plot_cluster_condition_heatmap(sig_kinases, out_dir, mode):
    """Heatmap: clusters × (condition × timepoint), cell=sig kinase count.

    Companion to direction_by_cluster — same data collapsed to counts,
    easier to scan for burden patterns at the cost of losing up/down distinction.
    """
    if du.graceful_empty(sig_kinases, "sig kinases for cluster×condition heatmap"):
        return

    clusters = sorted(sig_kinases["cluster"].unique())
    if len(clusters) <= 1:
        print("  [SKIP] cluster_condition_heatmap: only 1 cluster (bulk mode)")
        return

    conditions = [c for c in du.CONDITIONS if c in sig_kinases["condition"].unique()]
    timepoints = [t for t in du.TIMEPOINTS if t in sig_kinases["timepoint"].unique()]

    # Build pivot: rows=clusters, columns=condition_timepoint
    col_order = [f"{c}_{tp}" for c in conditions for tp in timepoints]
    counts = sig_kinases.groupby(["cluster", "condition", "timepoint"]).size().reset_index(name="count")
    counts["col"] = counts["condition"] + "_" + counts["timepoint"]
    pivot = counts.pivot_table(index="cluster", columns="col", values="count", fill_value=0)
    pivot = pivot.reindex(columns=[c for c in col_order if c in pivot.columns], fill_value=0)
    pivot = pivot.reindex(clusters, fill_value=0)

    fig, ax = plt.subplots(figsize=(max(8, len(pivot.columns) * 0.9), max(4, len(clusters) * 0.45)))
    im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([c.replace("_", "\n") for c in pivot.columns], fontsize=8)
    ax.set_yticks(range(len(pivot)))
    ax.set_yticklabels(pivot.index, fontsize=8)

    for i in range(len(pivot)):
        for j in range(len(pivot.columns)):
            val = int(pivot.iloc[i, j])
            if val > 0:
                ax.text(j, i, str(val), ha="center", va="center", fontsize=8)

    fig.colorbar(im, ax=ax, label="Sig kinase count", shrink=0.6)
    ax.set_title(f"Significant kinases: cluster × condition × timepoint — {mode}")
    fig.tight_layout()
    du.save_fig(fig, os.path.join(out_dir, "cluster_condition_heatmap.png"))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_analysis(mode):
    """Run full temporal trajectory analysis for one mode."""
    print(f"\n{'='*60}")
    print(f"Temporal trajectory analysis: {mode}")
    print(f"{'='*60}")

    # Build LFF matrix from raw enrichment results
    print("Loading enrichment results...")
    lff_matrix = build_lff_matrix(mode)
    if du.graceful_empty(lff_matrix, f"{mode} LFF matrix"):
        return

    out_dir = du.setup_analysis_dir("temporal", mode)

    print(f"  {len(lff_matrix)} kinase x comparison entries loaded")

    # Load significant kinases
    sig_kinases = du.load_significant_kinases(mode)

    # 1. Temporal table (raw LFF/direction/significance per timepoint, no pattern labels)
    print("Building temporal table...")
    temporal_table = build_temporal_table(lff_matrix)
    temporal_table.to_csv(os.path.join(out_dir, "kinase_temporal_classification.csv"), index=False)
    print(f"  {len(temporal_table)} kinase x condition x cluster entries with any significance")

    # 2. Visualizations
    print("Generating plots...")
    plot_direction_overall(sig_kinases, out_dir, mode)
    plot_direction_by_cluster(sig_kinases, out_dir, mode)
    plot_cluster_condition_heatmap(sig_kinases, out_dir, mode)


def main():
    parser = argparse.ArgumentParser(description="Temporal trajectory analysis of AD kinase enrichment")
    parser.add_argument("--mode", choices=du.ALL_MODES, default=None,
                        help="Analysis mode (default: run all)")
    args = parser.parse_args()

    modes = [args.mode] if args.mode else du.ALL_MODES
    for mode in modes:
        run_analysis(mode)

    print("\nDone.")


if __name__ == "__main__":
    main()

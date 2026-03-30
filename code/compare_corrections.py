"""Compare BH vs permutation p-value correction methods.

Reads enrichment CSVs that contain both `most_sig_fisher_bh_pval` and
`most_sig_fisher_perm_pval` columns (generated when CORRECTION_METHOD
is set to "permutation"). Produces lecture-ready figures showing the
problem (BH over-correction with correlated tests) and the fix
(permutation-based empirical p-values).

Usage:
    python code/compare_corrections.py [--mode deconv|bulk] [--pval-thresh 0.1]
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats

sys.path.insert(0, os.path.dirname(__file__))
import config


def load_enrichment_results(enrichment_dir):
    """Load all enrichment CSVs that have both BH and permutation columns."""
    results = []
    for fname in sorted(os.listdir(enrichment_dir)):
        if not fname.endswith("_enrichment_results.csv"):
            continue
        fpath = os.path.join(enrichment_dir, fname)
        df = pd.read_csv(fpath, index_col=0)
        if "most_sig_fisher_bh_pval" not in df.columns:
            continue
        if "most_sig_fisher_perm_pval" not in df.columns:
            continue

        # Parse comparison metadata from filename
        parts = fname.replace("_enrichment_results.csv", "").split("_vs_")
        if len(parts) != 2:
            continue
        fg_parts = parts[1].split("_")
        # Format: gender_timepoint_condition_cluster_pctN_lffN
        if len(fg_parts) < 4:
            continue
        gender = fg_parts[0]
        timepoint = fg_parts[1]
        condition = fg_parts[2]
        # Cluster name may contain hyphens, and pct/lff suffix is at the end
        remaining = "_".join(fg_parts[3:])
        # Split off pct and lff suffixes
        pct_idx = remaining.rfind("_pct")
        if pct_idx >= 0:
            cluster = remaining[:pct_idx]
        else:
            cluster = remaining

        df["comparison"] = fname.replace("_enrichment_results.csv", "")
        df["gender"] = gender
        df["timepoint"] = timepoint
        df["condition"] = condition
        df["cluster"] = cluster
        df["kinase"] = df.index
        results.append(df)

    if not results:
        return pd.DataFrame()
    return pd.concat(results, ignore_index=True)


def plot_pvalue_null_diagnostic(all_df, output_dir):
    """Panel 1: P-value histograms showing non-uniform null.

    Shows raw Fisher p-value distributions for representative comparisons,
    highlighting the right-tail depletion caused by kinase-kinase dependence.
    """
    # Pick 4 representative comparisons: 2 strong signal, 2 loss contexts
    target_contexts = [
        ("4mo", "Astrocytes", "AppP", "Strong signal (AppP)"),
        ("4mo", "Astrocytes", "ApTt", "Signal loss (ApTt)"),
        ("6mo", "Astrocytes", "Ttau", "Strong signal (Ttau)"),
        ("6mo", "Astrocytes", "ApTt", "Signal loss (ApTt)"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.ravel()

    for i, (tp, cluster, cond, label) in enumerate(target_contexts):
        ax = axes[i]
        mask = ((all_df["timepoint"] == tp) &
                (all_df["cluster"] == cluster) &
                (all_df["condition"] == cond))
        sub = all_df[mask]
        if sub.empty:
            ax.set_title(f"{tp} {cluster} {cond}\n(no data)")
            continue

        raw_p = sub["most_sig_fisher_pval"].values
        n_kinases = len(raw_p)

        # Histogram
        bins = np.linspace(0, 1, 21)
        counts, _ = np.histogram(raw_p, bins=bins)
        expected = n_kinases / 20  # uniform expectation per bin

        ax.bar(bins[:-1], counts, width=0.05, align="edge",
               color="steelblue", alpha=0.7, edgecolor="white")
        ax.axhline(y=expected, color="red", ls="--", lw=1.5,
                   alpha=0.7, label=f"Uniform null ({expected:.0f}/bin)")

        ax.set_xlabel("Raw Fisher p-value")
        ax.set_ylabel("Count")
        ax.set_title(f"{tp} {cluster} — {label}\n(n={n_kinases} kinases)")
        ax.legend(fontsize=8)

    fig.suptitle("P-Value Null Distribution Diagnostic\n"
                 "Right-tail depletion reveals kinase-kinase dependence",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    path = os.path.join(output_dir, "null_diagnostic_histograms.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_qq_diagnostic(all_df, output_dir):
    """Panel 2: QQ-plot of raw p-values against Uniform(0,1)."""
    target_contexts = [
        ("4mo", "Astrocytes", "AppP"),
        ("4mo", "Astrocytes", "ApTt"),
        ("6mo", "Astrocytes", "ApTt"),
        ("4mo", "Foxp2-Excitatory-Neurons-layers-6-and-2-3", "ApTt"),
    ]
    cond_colors = config.CONDITION_COLORS

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot([0, 5], [0, 5], "k-", alpha=0.3, lw=1, label="Uniform(0,1)")

    for tp, cluster, cond in target_contexts:
        mask = ((all_df["timepoint"] == tp) &
                (all_df["cluster"] == cluster) &
                (all_df["condition"] == cond))
        sub = all_df[mask]
        if sub.empty:
            continue

        raw_p = np.sort(sub["most_sig_fisher_pval"].values)
        n = len(raw_p)
        expected = -np.log10(np.arange(1, n + 1) / (n + 1))
        observed = -np.log10(np.clip(raw_p, 1e-300, 1))

        ax.scatter(np.sort(expected)[::-1], np.sort(observed)[::-1],
                   s=8, alpha=0.6, color=cond_colors.get(cond, "gray"),
                   label=f"{tp} {cluster[:15]}.. {cond}")

    ax.set_xlabel("Expected -log10(p) [Uniform]")
    ax.set_ylabel("Observed -log10(p)")
    ax.set_title("QQ Plot: Raw Fisher P-Values vs Uniform Null\n"
                 "Deviation above diagonal = excess small p-values")
    ax.legend(fontsize=8, loc="upper left")

    path = os.path.join(output_dir, "qq_diagnostic.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_bh_vs_perm_scatter(all_df, pval_thresh, output_dir):
    """Panel 3: Scatter of BH-adjusted vs permutation-adjusted p-values."""
    cond_colors = config.CONDITION_COLORS

    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot in reverse order so ApTt (most important) is on top
    for cond in ["Ttau", "AppP", "ApTt"]:
        sub = all_df[all_df["condition"] == cond]
        bh = -np.log10(sub["most_sig_fisher_bh_pval"].clip(lower=1e-300))
        perm = -np.log10(sub["most_sig_fisher_perm_pval"].clip(lower=1e-300))
        ax.scatter(bh, perm, s=6, alpha=0.25,
                   color=cond_colors.get(cond, "gray"), label=cond,
                   edgecolors="none")

    # Reference line
    lim = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([0, lim], [0, lim], "k-", alpha=0.3, lw=1)

    # Significance thresholds — shade the "gained" quadrant
    thresh_line = -np.log10(pval_thresh)
    ax.axhline(y=thresh_line, color="red", ls="--", lw=0.8, alpha=0.5)
    ax.axvline(x=thresh_line, color="red", ls="--", lw=0.8, alpha=0.5)

    # Annotate quadrants
    ax.text(0.3, thresh_line + 0.3, "Recovered by\npermutation",
            fontsize=9, color="green", fontstyle="italic", ha="center")
    ax.text(thresh_line + 0.3, 0.3, "Lost by\npermutation",
            fontsize=9, color="red", fontstyle="italic", ha="center")

    ax.set_xlabel("-log10(BH adjusted p-value)", fontsize=11)
    ax.set_ylabel("-log10(Permutation empirical p-value)", fontsize=11)
    ax.set_title("BH vs Permutation Correction\n"
                 f"Red lines = significance threshold (p < {pval_thresh})",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, markerscale=4, loc="upper left")

    path = os.path.join(output_dir, "bh_vs_permutation_scatter.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_recovery_bar_chart(all_df, pval_thresh, output_dir):
    """Panel 4: Mean significant kinases per comparison — BH vs permutation."""
    cond_colors = config.CONDITION_COLORS
    lff_thresh = config.LFF_THRESH
    tp_order = ["2mo", "4mo", "6mo"]

    # Per-comparison counts, then average across clusters
    comp_records = []
    for comp in all_df["comparison"].unique():
        sub = all_df[all_df["comparison"] == comp]
        if sub.empty:
            continue
        lff_mask = sub["most_sig_log2_freq_factor"].abs() >= lff_thresh
        comp_records.append({
            "condition": sub["condition"].iloc[0],
            "timepoint": sub["timepoint"].iloc[0],
            "cluster": sub["cluster"].iloc[0],
            "bh": ((sub["most_sig_fisher_bh_pval"] <= pval_thresh) & lff_mask).sum(),
            "perm": ((sub["most_sig_fisher_perm_pval"] <= pval_thresh) & lff_mask).sum(),
        })
    comp_df = pd.DataFrame(comp_records)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5), sharey=True)
    for i, cond in enumerate(["Ttau", "AppP", "ApTt"]):
        ax = axes[i]
        x = np.arange(len(tp_order))
        w = 0.35

        bh_means, bh_sems, perm_means, perm_sems = [], [], [], []
        for tp in tp_order:
            tp_sub = comp_df[(comp_df["condition"] == cond) & (comp_df["timepoint"] == tp)]
            bh_means.append(tp_sub["bh"].mean())
            bh_sems.append(tp_sub["bh"].sem())
            perm_means.append(tp_sub["perm"].mean())
            perm_sems.append(tp_sub["perm"].sem())

        bars_bh = ax.bar(x - w/2, bh_means, w, yerr=bh_sems, capsize=3,
                         label="BH", color=cond_colors[cond], alpha=0.4)
        bars_pm = ax.bar(x + w/2, perm_means, w, yerr=perm_sems, capsize=3,
                         label="Permutation", color=cond_colors[cond], alpha=0.9)

        # Add value labels on bars
        for bar, val in zip(bars_bh, bh_means):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                        f"{val:.0f}", ha="center", va="bottom", fontsize=7)
        for bar, val in zip(bars_pm, perm_means):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f"{val:.0f}", ha="center", va="bottom", fontsize=7)

        ax.set_xticks(x)
        ax.set_xticklabels(tp_order)
        ax.set_title(cond, fontsize=12, fontweight="bold")
        ax.legend(fontsize=8)
        if i == 0:
            ax.set_ylabel("Mean significant kinases\nper comparison")

    fig.suptitle(f"Signal Recovery: BH vs Permutation Correction\n"
                 f"(mean per comparison; adj_p < {pval_thresh}, |LFF| >= {lff_thresh})",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    path = os.path.join(output_dir, "recovery_bar_chart.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def write_recovery_table(all_df, pval_thresh, output_dir):
    """Summary table: kinases gained/lost per comparison under permutation."""
    lff_thresh = config.LFF_THRESH
    records = []

    for comp in all_df["comparison"].unique():
        sub = all_df[all_df["comparison"] == comp]
        if sub.empty:
            continue
        lff_mask = sub["most_sig_log2_freq_factor"].abs() >= lff_thresh
        bh_sig = set(sub[lff_mask & (sub["most_sig_fisher_bh_pval"] <= pval_thresh)]["kinase"])
        perm_sig = set(sub[lff_mask & (sub["most_sig_fisher_perm_pval"] <= pval_thresh)]["kinase"])

        gained = perm_sig - bh_sig
        lost = bh_sig - perm_sig

        records.append({
            "comparison": comp,
            "condition": sub["condition"].iloc[0],
            "timepoint": sub["timepoint"].iloc[0],
            "cluster": sub["cluster"].iloc[0],
            "n_bh_sig": len(bh_sig),
            "n_perm_sig": len(perm_sig),
            "n_gained": len(gained),
            "n_lost": len(lost),
            "net_change": len(gained) - len(lost),
            "gained_kinases": ", ".join(sorted(gained)[:10]),
            "lost_kinases": ", ".join(sorted(lost)[:10]),
        })

    table = pd.DataFrame(records).sort_values("net_change", ascending=False)
    path = os.path.join(output_dir, "correction_comparison_table.csv")
    table.to_csv(path, index=False)
    print(f"  Saved: {path}")

    # Print summary
    print(f"\n  Recovery summary ({len(table)} comparisons):")
    for cond in ["Ttau", "AppP", "ApTt"]:
        sub = table[table["condition"] == cond]
        print(f"    {cond}: BH avg={sub['n_bh_sig'].mean():.1f}, "
              f"Perm avg={sub['n_perm_sig'].mean():.1f}, "
              f"net change={sub['net_change'].mean():+.1f}")

    return table


def write_top_recovered_kinases(all_df, pval_thresh, output_dir):
    """Biological validation: kinases recovered by permutation but invisible under BH.

    For each kinase, counts how many comparisons it was significant under
    permutation but NOT under BH. Highlights the most frequently recovered
    kinases with their brain expression, mean effect size, and which
    conditions they appear in.
    """
    lff_thresh = config.LFF_THRESH
    lff_mask = all_df["most_sig_log2_freq_factor"].abs() >= lff_thresh
    bh_sig = (all_df["most_sig_fisher_bh_pval"] <= pval_thresh) & lff_mask
    perm_sig = (all_df["most_sig_fisher_perm_pval"] <= pval_thresh) & lff_mask
    gained = perm_sig & ~bh_sig  # significant under perm but not BH

    gained_df = all_df[gained].copy()
    if gained_df.empty:
        print("  No recovered kinases found.")
        return

    # Aggregate per kinase
    records = []
    for kinase, grp in gained_df.groupby("kinase"):
        records.append({
            "kinase": kinase,
            "n_comparisons_recovered": len(grp),
            "conditions_recovered_in": ", ".join(sorted(grp["condition"].unique())),
            "timepoints_recovered_in": ", ".join(sorted(grp["timepoint"].unique())),
            "clusters_recovered_in": len(grp["cluster"].unique()),
            "mean_lff": grp["most_sig_log2_freq_factor"].mean(),
            "mean_abs_lff": grp["most_sig_log2_freq_factor"].abs().mean(),
            "mean_raw_pval": grp["most_sig_fisher_pval"].mean(),
            "mean_bh_pval": grp["most_sig_fisher_bh_pval"].mean(),
            "mean_perm_pval": grp["most_sig_fisher_perm_pval"].mean(),
            "brain_expressed": int(grp["brain_expressed"].iloc[0]) if "brain_expressed" in grp.columns else -1,
        })

    top = pd.DataFrame(records).sort_values("n_comparisons_recovered", ascending=False)
    path = os.path.join(output_dir, "top_recovered_kinases.csv")
    top.to_csv(path, index=False)

    print(f"\n  Top 20 recovered kinases (significant under permutation, invisible under BH):")
    print(f"  {'Kinase':<10} {'Recovered':>8} {'Brain':>5} {'mean|LFF|':>9} {'mean raw p':>10} {'mean BH p':>10} {'mean perm p':>10}  Conditions")
    for _, row in top.head(20).iterrows():
        brain = "Y" if row["brain_expressed"] == 1 else "N"
        print(f"  {row['kinase']:<10} {int(row['n_comparisons_recovered']):>8} {brain:>5} "
              f"{row['mean_abs_lff']:>9.3f} {row['mean_raw_pval']:>10.4f} "
              f"{row['mean_bh_pval']:>10.4f} {row['mean_perm_pval']:>10.4f}  "
              f"{row['conditions_recovered_in']}")

    print(f"\n  Full table: {path}")
    return top


def main():
    parser = argparse.ArgumentParser(
        description="Compare BH vs permutation p-value correction")
    parser.add_argument("--mode", choices=["deconv", "bulk"], default="deconv")
    parser.add_argument("--pval-thresh", type=float, default=None,
                        help=f"P-value threshold (default: config.PVAL_SIG={config.PVAL_SIG})")
    args = parser.parse_args()

    pval_thresh = args.pval_thresh or config.PVAL_SIG
    enrichment_dir = os.path.join(f"outputs/{args.mode}", "enrichment_results")
    output_dir = os.path.join(f"outputs/{args.mode}", "correction_comparison")
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading enrichment results from {enrichment_dir}...")
    all_df = load_enrichment_results(enrichment_dir)
    if all_df.empty:
        print("No enrichment CSVs with both BH and permutation columns found.")
        print("Run the pipeline with --correction permutation first.")
        return

    n_comparisons = all_df["comparison"].nunique()
    print(f"  Loaded {len(all_df)} kinase records across {n_comparisons} comparisons")

    print("\nGenerating comparison figures...")
    plot_pvalue_null_diagnostic(all_df, output_dir)
    plot_qq_diagnostic(all_df, output_dir)
    plot_bh_vs_perm_scatter(all_df, pval_thresh, output_dir)
    plot_recovery_bar_chart(all_df, pval_thresh, output_dir)
    write_recovery_table(all_df, pval_thresh, output_dir)
    write_top_recovered_kinases(all_df, pval_thresh, output_dir)

    print(f"\nAll figures saved to {output_dir}/")


if __name__ == "__main__":
    main()

"""Examine factorial Incytr results and kinase support analysis.

Reads the aggregation CSV outputs from aggregate_factorial.py and produces
interpretable biological findings: additivity analysis (ApTt vs App+Tau),
temporal trajectories, cell-type specificity, and kinase validation.

All inputs are small CSVs that fit in memory — no DuckDB needed.

Usage:
  python examine_factorial.py --run          # all sections
  python examine_factorial.py --summary      # quick overview
  python examine_factorial.py --additivity   # core finding
  python examine_factorial.py --temporal     # trajectory classification
  python examine_factorial.py --celltype     # cell-type centrality
  python examine_factorial.py --kinase       # kinase concordance/coverage
  python examine_factorial.py --figures      # publication composites
"""

import argparse
import os
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, leaves_list

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config_integration as icfg

# Import tissue categories from main config (via icfg's sys.path setup)
from config import (SUBCLASS_TO_TISSUE_CATEGORY, DISEASE_COLORS,
                    TISSUE_ORDER)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
GENOTYPES = ["App", "Tau", "ApTt"]
TIMEPOINTS = ["2mo", "4mo", "6mo"]
CONTRASTS = [f"{g}_{t}" for g in GENOTYPES for t in TIMEPOINTS]
TP_ORDER = {"2mo": 0, "4mo": 1, "6mo": 2}
GENO_DISPLAY = {"App": "APP", "Tau": "Tau", "ApTt": u"A\u00d7T"}

TISSUE_COLORS = {
    "Excitatory neurons": "#e74c3c",
    "Interneurons": "#e67e22",
    "Astrocytes": "#27ae60",
    "Oligodendrocytes": "#3498db",
    "OPCs": "#9b59b6",
    "Microglia": "#1abc9c",
    "Endothelial cells": "#95a5a6",
}

ADDITIVITY_THRESH = 0.005  # |interaction| below this = "additive"

AGG_DIR = os.path.join(icfg.FACTORIAL_ALL_PAIRS_DIR, "aggregation")

# Publication-quality matplotlib defaults
PUB_RCPARAMS = {
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_all_data():
    """Load all aggregation CSVs into a dict of DataFrames."""
    data = {}

    data["hub"] = pd.read_csv(os.path.join(AGG_DIR, "hub_matrix_by_contrast.csv"))
    data["comparison"] = pd.read_csv(
        os.path.join(AGG_DIR, "contrast_comparison.csv"))
    data["temporal"] = pd.read_csv(
        os.path.join(AGG_DIR, "temporal_dynamics.csv"))
    data["kinase"] = pd.read_csv(
        os.path.join(AGG_DIR, "kinase_tpds_integration.csv"))

    # Large CSVs — load only when needed, but preload headers
    bb_path = os.path.join(AGG_DIR, "backbone_recurrence_by_contrast.csv")
    tgt_path = os.path.join(AGG_DIR, "target_convergence_by_contrast.csv")
    data["_bb_path"] = bb_path
    data["_tgt_path"] = tgt_path

    print(f"Loaded: hub={len(data['hub'])}, temporal={len(data['temporal'])}, "
          f"kinase={len(data['kinase'])} rows")
    return data


def _load_backbone(data):
    """Lazy-load backbone recurrence CSV."""
    if "backbone" not in data:
        print("  Loading backbone recurrence (269 MB)...")
        data["backbone"] = pd.read_csv(
            data["_bb_path"],
            dtype={"contrast": "category", "receiver": "category",
                   "Receptor": "category", "EM": "category",
                   "Target": "category"})
        print(f"  Loaded {len(data['backbone'])} backbone rows")
    return data["backbone"]


def _load_target_convergence(data):
    """Lazy-load target convergence CSV."""
    if "target" not in data:
        print("  Loading target convergence (31 MB)...")
        data["target"] = pd.read_csv(data["_tgt_path"])
        print(f"  Loaded {len(data['target'])} target rows")
    return data["target"]


# ---------------------------------------------------------------------------
# S1: Summary Statistics
# ---------------------------------------------------------------------------

def section_summary(data, out_dir):
    """Print and write summary statistics."""
    hub = data["hub"]
    kinase = data["kinase"]

    lines = []
    lines.append("=" * 70)
    lines.append("FACTORIAL INCYTR RESULTS — SUMMARY STATISTICS")
    lines.append("=" * 70)

    # Per-contrast totals
    lines.append("\n--- Per-contrast totals ---")
    lines.append(f"{'Contrast':<12} {'n_sig':>10} {'pct_sig':>8} "
                 f"{'mean|TPDS|':>11} {'mean_TPDS':>10}")
    for c in CONTRASTS:
        ch = hub[hub["contrast"] == c]
        n_sig = int(ch["n_significant"].sum())
        n_total = int(ch["n_pathways"].sum())
        pct = 100 * n_sig / max(n_total, 1)
        mat = ch["mean_abs_tpds"].mean()
        mt = ch["mean_tpds"].mean()
        lines.append(f"  {c:<10} {n_sig:>10,} {pct:>7.2f}% {mat:>11.6f} {mt:>+10.6f}")

    # Rank contrasts by total significant
    lines.append("\n--- Contrast ranking by total significant pathways ---")
    ranking = []
    for c in CONTRASTS:
        ch = hub[hub["contrast"] == c]
        ranking.append((c, int(ch["n_significant"].sum())))
    ranking.sort(key=lambda x: -x[1])
    for i, (c, n) in enumerate(ranking, 1):
        lines.append(f"  {i}. {c:<12} {n:>10,}")

    # Top 5 sender-receiver pairs overall
    lines.append("\n--- Top 5 sender-receiver pairs by mean |TPDS| ---")
    pair_avg = hub.groupby(["sender", "receiver"])["mean_abs_tpds"].mean()
    top5 = pair_avg.nlargest(5)
    for (s, r), val in top5.items():
        lines.append(f"  {s} -> {r}: {val:.6f}")

    # Backbone recurrence
    bb = _load_backbone(data)
    lines.append("\n--- Backbone recurrence ---")
    for thresh in [2, 5, 10, 15, 21]:
        n = int((bb["n_senders_significant"] >= thresh).sum())
        lines.append(f"  n_senders_sig >= {thresh:>2}: {n:>10,} backbones")

    # Target convergence
    tgt = _load_target_convergence(data)
    lines.append("\n--- Target convergence ---")
    for thresh in [3, 5, 10, 15]:
        n = int((tgt["n_senders_significant"] >= thresh).sum())
        lines.append(f"  n_senders_sig >= {thresh:>2}: {n:>10,} targets")

    # Kinase integration summary
    lines.append("\n--- Kinase integration ---")
    lines.append(f"{'Contrast':<12} {'coverage':>9} {'concordance':>12}")
    for _, row in kinase.iterrows():
        cov = 100 * row["n_sig_with_kinase"] / max(row["n_sig_tpds"], 1)
        conc_total = row["n_concordant"] + row["n_discordant"]
        conc_rate = 100 * row["n_concordant"] / max(conc_total, 1)
        lines.append(f"  {row['contrast']:<10} {cov:>8.1f}% {conc_rate:>11.1f}%")

    lines.append("\n" + "=" * 70)

    text = "\n".join(lines)
    print(text)

    path = os.path.join(out_dir, "summary_statistics.txt")
    with open(path, "w") as f:
        f.write(text + "\n")
    print(f"\n  Wrote {path}")


# ---------------------------------------------------------------------------
# S2: Additivity Analysis
# ---------------------------------------------------------------------------

def _classify_additivity(row):
    """Classify interaction between ApTt observed and App+Tau predicted."""
    predicted = row["predicted_additive"]
    observed = row["observed_aptt"]
    interaction = row["interaction"]

    if abs(interaction) < ADDITIVITY_THRESH:
        return "additive"
    if abs(predicted) < 1e-9:
        # Predicted is ~0, any nonzero observed is hard to classify
        return "emergent"
    if np.sign(observed) != np.sign(predicted):
        return "antagonistic"
    if abs(observed) < abs(predicted):
        return "sub-additive"
    return "super-additive"


def section_additivity(data, out_dir):
    """Additivity analysis: is ApTt = App + Tau?"""
    hub = data["hub"]

    rows = []
    for tp in TIMEPOINTS:
        app_c = f"App_{tp}"
        tau_c = f"Tau_{tp}"
        aptt_c = f"ApTt_{tp}"

        app_df = hub[hub["contrast"] == app_c][
            ["sender", "receiver", "mean_tpds"]].rename(
            columns={"mean_tpds": "app_tpds"})
        tau_df = hub[hub["contrast"] == tau_c][
            ["sender", "receiver", "mean_tpds"]].rename(
            columns={"mean_tpds": "tau_tpds"})
        aptt_df = hub[hub["contrast"] == aptt_c][
            ["sender", "receiver", "mean_tpds"]].rename(
            columns={"mean_tpds": "observed_aptt"})

        merged = app_df.merge(tau_df, on=["sender", "receiver"])
        merged = merged.merge(aptt_df, on=["sender", "receiver"])
        merged["timepoint"] = tp
        merged["predicted_additive"] = merged["app_tpds"] + merged["tau_tpds"]
        merged["interaction"] = (
            merged["observed_aptt"] - merged["predicted_additive"])
        merged["additivity_class"] = merged.apply(_classify_additivity, axis=1)
        rows.append(merged)

    add_df = pd.concat(rows, ignore_index=True)

    # Add tissue category for the receiver
    add_df["receiver_tissue"] = add_df["receiver"].map(
        SUBCLASS_TO_TISSUE_CATEGORY).fillna("Other")

    # Write per-pair results
    add_path = os.path.join(out_dir, "additivity_by_pair_timepoint.csv")
    add_df.to_csv(add_path, index=False)
    print(f"  Wrote {add_path} ({len(add_df)} rows)")

    # Summary table
    summary = add_df.groupby(["timepoint", "additivity_class"]).size().unstack(
        fill_value=0).reindex(TIMEPOINTS)
    summary_path = os.path.join(out_dir, "additivity_summary.csv")
    summary.to_csv(summary_path)
    print(f"  Wrote {summary_path}")
    print(f"\n  Additivity summary:\n{summary}\n")

    # Print key finding
    # NOTE: The interaction term is constant across timepoints because the
    # OLS model has no Int×time interactions — the difference ApTt-(App+Tau)
    # always equals exactly the Int coefficient (beta_3). The scatter plots
    # still differ because App and Tau contributions change with timepoint.
    print("\n  NOTE: Interaction term is constant across timepoints (model has")
    print("  no Int x time interactions — only App x time and Tau x time).")
    print("  The Int coefficient captures the time-invariant non-additivity.\n")

    for tp in TIMEPOINTS:
        tp_data = add_df[add_df["timepoint"] == tp]
        n_sub = (tp_data["additivity_class"] == "sub-additive").sum()
        n_sup = (tp_data["additivity_class"] == "super-additive").sum()
        n_ant = (tp_data["additivity_class"] == "antagonistic").sum()
        med_int = tp_data["interaction"].median()
        print(f"  {tp}: sub-additive={n_sub}, super-additive={n_sup}, "
              f"antagonistic={n_ant}, median interaction={med_int:+.6f}")

    # --- Plot: additivity scatter ---
    plt.rcParams.update(PUB_RCPARAMS)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for i, tp in enumerate(TIMEPOINTS):
        ax = axes[i]
        tp_data = add_df[add_df["timepoint"] == tp]

        for tissue in TISSUE_ORDER:
            mask = tp_data["receiver_tissue"] == tissue
            if mask.sum() == 0:
                continue
            ax.scatter(tp_data.loc[mask, "predicted_additive"],
                       tp_data.loc[mask, "observed_aptt"],
                       c=TISSUE_COLORS.get(tissue, "#999999"),
                       label=tissue, alpha=0.6, s=20, edgecolors="none")

        # Diagonal line
        lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]),
                max(ax.get_xlim()[1], ax.get_ylim()[1])]
        ax.plot(lims, lims, "k--", alpha=0.3, linewidth=1)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel("Predicted (App + Tau)")
        ax.set_ylabel(f"Observed ({GENO_DISPLAY['ApTt']})")
        ax.set_title(f"{tp}")
        ax.axhline(0, color="grey", linewidth=0.5, alpha=0.3)
        ax.axvline(0, color="grey", linewidth=0.5, alpha=0.3)
        ax.set_aspect("equal")

    axes[0].legend(fontsize=6, loc="upper left", framealpha=0.8)
    fig.suptitle("Additivity: observed vs predicted pathway effects",
                 fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(out_dir, "additivity_scatter_by_timepoint.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Wrote {path}")

    # --- Plot: interaction distribution + top deviators ---
    # Since the interaction is constant across timepoints, show it once
    # with the distribution and the top deviators as a horizontal bar chart.
    int_2mo = add_df[add_df["timepoint"] == "2mo"].copy()
    int_2mo = int_2mo.sort_values("interaction")

    fig, (ax_hist, ax_bar) = plt.subplots(1, 2, figsize=(14, 6),
                                          gridspec_kw={"width_ratios": [1, 1]})

    # Left: histogram of interaction terms
    ax_hist.hist(int_2mo["interaction"].values, bins=40,
                 color=DISEASE_COLORS["ApTt"], alpha=0.6, edgecolor="k",
                 linewidth=0.5)
    ax_hist.axvline(0, color="black", linewidth=1.5, linestyle="--")
    med = int_2mo["interaction"].median()
    ax_hist.axvline(med, color="red", linewidth=1, linestyle="-",
                    label=f"median={med:+.4f}")
    ax_hist.set_xlabel("Interaction (observed - predicted)")
    ax_hist.set_ylabel("Number of sender-receiver pairs")
    ax_hist.set_title(f"{GENO_DISPLAY['ApTt']} interaction term\n"
                      f"(constant across timepoints, = OLS Int coefficient)")
    ax_hist.legend(fontsize=8)

    # Right: top 10 most sub-additive and top 10 most super-additive
    n_show = 10
    top_sub = int_2mo.head(n_show)
    top_sup = int_2mo.tail(n_show)
    show = pd.concat([top_sub, top_sup])
    labels_bar = [f"{row['sender']} -> {row['receiver']}"
                  for _, row in show.iterrows()]
    colors_bar = [TISSUE_COLORS.get(
        SUBCLASS_TO_TISSUE_CATEGORY.get(r, "Other"), "#999")
        for r in show["receiver"]]
    ax_bar.barh(range(len(show)), show["interaction"].values,
                color=colors_bar, alpha=0.7, edgecolor="k", linewidth=0.5)
    ax_bar.set_yticks(range(len(show)))
    ax_bar.set_yticklabels(labels_bar, fontsize=6)
    ax_bar.axvline(0, color="black", linewidth=1)
    ax_bar.set_xlabel("Interaction term")
    ax_bar.set_title("Top 10 sub-additive / super-additive pairs")
    ax_bar.invert_yaxis()

    plt.tight_layout()
    path = os.path.join(out_dir, "sub_additivity_magnitude.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Wrote {path}")


# ---------------------------------------------------------------------------
# S3: Temporal Trajectory
# ---------------------------------------------------------------------------

def _classify_trajectory(tpds_2, tpds_4, tpds_6, sig_thresh=0.001):
    """Classify temporal trajectory of mean_abs_tpds across 3 timepoints."""
    vals = [tpds_2, tpds_4, tpds_6]
    max_val = max(vals)
    min_val = min(vals)

    if max_val < sig_thresh:
        return "none"

    if tpds_2 < tpds_4 < tpds_6:
        return "progressive"
    if tpds_2 > tpds_4 > tpds_6:
        return "declining"
    if tpds_4 > tpds_2 and tpds_4 > tpds_6:
        return "peaked"
    if max_val > 0 and min_val / max_val > 0.67:
        return "sustained"
    if tpds_2 > tpds_4 and tpds_2 > tpds_6 and tpds_6 < sig_thresh:
        return "early"
    if tpds_6 > tpds_2 and tpds_6 > tpds_4 and tpds_2 < sig_thresh:
        return "late"
    return "mixed"


def section_temporal(data, out_dir):
    """Temporal trajectory classification."""
    temporal = data["temporal"]

    # Pivot to get mean_abs_tpds per sender-receiver-genotype-timepoint
    rows = []
    for geno in GENOTYPES:
        gdata = temporal[temporal["genotype"] == geno]
        for (s, r), grp in gdata.groupby(["sender", "receiver"]):
            tpds_by_tp = {}
            for _, row in grp.iterrows():
                tpds_by_tp[row["timepoint"]] = row["mean_abs_tpds"]

            t2 = tpds_by_tp.get("2mo", 0)
            t4 = tpds_by_tp.get("4mo", 0)
            t6 = tpds_by_tp.get("6mo", 0)
            label = _classify_trajectory(t2, t4, t6)
            rows.append({
                "sender": s, "receiver": r, "genotype": geno,
                "tpds_2mo": t2, "tpds_4mo": t4, "tpds_6mo": t6,
                "trajectory_label": label,
            })

    traj_df = pd.DataFrame(rows)
    traj_path = os.path.join(out_dir, "trajectory_classification.csv")
    traj_df.to_csv(traj_path, index=False)
    print(f"  Wrote {traj_path} ({len(traj_df)} rows)")

    # Print distribution
    for geno in GENOTYPES:
        g_data = traj_df[traj_df["genotype"] == geno]
        dist = g_data["trajectory_label"].value_counts()
        print(f"\n  {GENO_DISPLAY[geno]} trajectory distribution:")
        for label, count in dist.items():
            pct = 100 * count / len(g_data)
            print(f"    {label:<15} {count:>4} ({pct:.1f}%)")

    # --- Plot: trajectory distribution ---
    plt.rcParams.update(PUB_RCPARAMS)
    label_order = ["progressive", "peaked", "sustained", "declining",
                   "early", "late", "mixed", "none"]
    label_colors = {
        "progressive": "#c62828", "peaked": "#e67e22",
        "sustained": "#27ae60", "declining": "#3498db",
        "early": "#9b59b6", "late": "#1abc9c",
        "mixed": "#95a5a6", "none": "#ecf0f1",
    }

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(GENOTYPES))
    width = 0.7
    bottom = np.zeros(len(GENOTYPES))

    for label in label_order:
        fracs = []
        for geno in GENOTYPES:
            g_data = traj_df[traj_df["genotype"] == geno]
            n = (g_data["trajectory_label"] == label).sum()
            fracs.append(100 * n / len(g_data))
        bars = ax.bar(x, fracs, width, bottom=bottom,
                      label=label, color=label_colors.get(label, "#999"))
        bottom += fracs

    ax.set_xticks(x)
    ax.set_xticklabels([GENO_DISPLAY[g] for g in GENOTYPES])
    ax.set_ylabel("% of sender-receiver pairs")
    ax.set_title("Temporal trajectory distribution by genotype")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    plt.tight_layout()
    path = os.path.join(out_dir, "trajectory_distribution.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Wrote {path}")

    # --- Plot: top pairs temporal line plots ---
    # Find top 10 pairs by max mean_abs_tpds across any contrast
    pair_max = traj_df.groupby(["sender", "receiver"])[
        ["tpds_2mo", "tpds_4mo", "tpds_6mo"]].max().max(axis=1)
    top10 = pair_max.nlargest(10).index

    fig, axes = plt.subplots(2, 5, figsize=(20, 8), sharey=True)
    axes_flat = axes.flatten()

    for idx, (s, r) in enumerate(top10):
        ax = axes_flat[idx]
        for geno in GENOTYPES:
            g_row = traj_df[
                (traj_df["sender"] == s) & (traj_df["receiver"] == r)
                & (traj_df["genotype"] == geno)]
            if g_row.empty:
                continue
            vals = [g_row.iloc[0][f"tpds_{tp}"] for tp in TIMEPOINTS]
            ax.plot(TIMEPOINTS, vals, marker="o",
                    color=DISEASE_COLORS[geno],
                    label=GENO_DISPLAY[geno], linewidth=1.5, markersize=4)
        ax.set_title(f"{s}\n-> {r}", fontsize=7)
        if idx == 0:
            ax.legend(fontsize=6)

    fig.suptitle("Top 10 pairs — temporal trajectories by genotype", fontsize=13)
    fig.supylabel("mean |TPDS|", fontsize=10)
    plt.tight_layout(rect=[0.02, 0, 1, 0.95])
    path = os.path.join(out_dir, "temporal_top_pairs.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Wrote {path}")


# ---------------------------------------------------------------------------
# S4: Cell-Type Analysis
# ---------------------------------------------------------------------------

def section_celltype(data, out_dir):
    """Cell-type centrality and ranking analysis."""
    hub = data["hub"]

    # Receiver centrality: sum of mean_abs_tpds from all senders
    recv_cent = hub.groupby(["contrast", "receiver"]).agg(
        total_tpds=("mean_abs_tpds", "sum"),
        total_sig=("n_significant", "sum"),
    ).reset_index()
    recv_cent["role"] = "receiver"
    recv_cent = recv_cent.rename(columns={"receiver": "celltype"})

    # Sender influence: sum of mean_abs_tpds to all receivers
    send_inf = hub.groupby(["contrast", "sender"]).agg(
        total_tpds=("mean_abs_tpds", "sum"),
        total_sig=("n_significant", "sum"),
    ).reset_index()
    send_inf["role"] = "sender"
    send_inf = send_inf.rename(columns={"sender": "celltype"})

    cent_df = pd.concat([recv_cent, send_inf], ignore_index=True)
    cent_df["tissue_category"] = cent_df["celltype"].map(
        SUBCLASS_TO_TISSUE_CATEGORY).fillna("Other")

    cent_path = os.path.join(out_dir, "celltype_centrality.csv")
    cent_df.to_csv(cent_path, index=False)
    print(f"  Wrote {cent_path} ({len(cent_df)} rows)")

    # --- Plot: receiver ranking heatmap ---
    plt.rcParams.update(PUB_RCPARAMS)

    # Pivot: rows = cell types, columns = contrasts, values = total_tpds received
    recv_pivot = recv_cent.pivot(
        index="celltype", columns="contrast", values="total_tpds")
    recv_pivot = recv_pivot.reindex(columns=CONTRASTS).fillna(0)

    # Cluster rows
    if len(recv_pivot) > 2:
        Z = linkage(recv_pivot.values, method="ward")
        order = leaves_list(Z)
        recv_pivot = recv_pivot.iloc[order]

    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(recv_pivot.values, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(CONTRASTS)))
    ax.set_xticklabels(CONTRASTS, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(recv_pivot)))
    ax.set_yticklabels(recv_pivot.index, fontsize=8)
    plt.colorbar(im, ax=ax, shrink=0.7, label="Total mean |TPDS| received")
    ax.set_title("Receiver cell-type centrality by contrast")
    plt.tight_layout()
    path = os.path.join(out_dir, "receiver_ranking_heatmap.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Wrote {path}")

    # --- Plot: sender vs receiver bubble ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for i, geno in enumerate(GENOTYPES):
        ax = axes[i]
        geno_contrasts = [f"{geno}_{tp}" for tp in TIMEPOINTS]

        # Average over the 3 timepoints for this genotype
        recv_g = recv_cent[recv_cent["contrast"].isin(geno_contrasts)].groupby(
            "celltype").agg(recv_tpds=("total_tpds", "mean"),
                           recv_sig=("total_sig", "mean")).reset_index()
        send_g = send_inf[send_inf["contrast"].isin(geno_contrasts)].groupby(
            "celltype").agg(send_tpds=("total_tpds", "mean"),
                           send_sig=("total_sig", "mean")).reset_index()
        merged = recv_g.merge(send_g, on="celltype")
        merged["tissue"] = merged["celltype"].map(
            SUBCLASS_TO_TISSUE_CATEGORY).fillna("Other")

        for tissue in TISSUE_ORDER:
            mask = merged["tissue"] == tissue
            if mask.sum() == 0:
                continue
            ax.scatter(merged.loc[mask, "send_tpds"],
                       merged.loc[mask, "recv_tpds"],
                       c=TISSUE_COLORS.get(tissue, "#999"),
                       s=merged.loc[mask, "recv_sig"] / 20 + 20,
                       label=tissue, alpha=0.7, edgecolors="k", linewidth=0.5)

        # Label each point
        for _, row in merged.iterrows():
            ax.annotate(row["celltype"], (row["send_tpds"], row["recv_tpds"]),
                        fontsize=5, alpha=0.7,
                        textcoords="offset points", xytext=(3, 3))

        ax.set_xlabel("Sender influence (mean |TPDS| sent)")
        ax.set_ylabel("Receiver centrality (mean |TPDS| received)")
        ax.set_title(f"{GENO_DISPLAY[geno]} vs WT")

    axes[0].legend(fontsize=6, loc="upper left")
    fig.suptitle("Cell-type sender influence vs receiver centrality", fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(out_dir, "celltype_sender_receiver_bubble.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Wrote {path}")


# ---------------------------------------------------------------------------
# S5: Kinase Validation
# ---------------------------------------------------------------------------

def section_kinase(data, out_dir):
    """Kinase concordance and coverage validation."""
    kinase = data["kinase"].copy()

    # Compute rates
    kinase["coverage_pct"] = (
        100 * kinase["n_sig_with_kinase"] / kinase["n_sig_tpds"].clip(lower=1))
    conc_total = kinase["n_concordant"] + kinase["n_discordant"]
    kinase["concordance_pct"] = (
        100 * kinase["n_concordant"] / conc_total.clip(lower=1))
    kinase["genotype"] = kinase["contrast"].str.split("_").str[0]
    kinase["timepoint"] = kinase["contrast"].str.split("_").str[1]

    # Print summary
    lines = []
    lines.append("--- Kinase Validation Summary ---")
    lines.append(f"{'Contrast':<12} {'Coverage':>9} {'Concordance':>12} "
                 f"{'n_conc':>8} {'n_disc':>8} {'n_mixed':>8}")
    for _, row in kinase.iterrows():
        lines.append(
            f"  {row['contrast']:<10} {row['coverage_pct']:>8.1f}% "
            f"{row['concordance_pct']:>11.1f}% "
            f"{int(row['n_concordant']):>8,} {int(row['n_discordant']):>8,} "
            f"{int(row['n_mixed']):>8,}")
    lines.append("")

    # Key findings
    mean_cov = kinase["coverage_pct"].mean()
    mean_conc = kinase["concordance_pct"].mean()
    lines.append(f"  Mean coverage: {mean_cov:.1f}%")
    lines.append(f"  Mean concordance: {mean_conc:.1f}% "
                 f"(chance = 50.0%)")

    text = "\n".join(lines)
    print(text)
    path = os.path.join(out_dir, "kinase_validation_summary.txt")
    with open(path, "w") as f:
        f.write(text + "\n")
    print(f"  Wrote {path}")

    plt.rcParams.update(PUB_RCPARAMS)

    # --- Plot: concordance by genotype ---
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(GENOTYPES))
    width = 0.25
    for j, tp in enumerate(TIMEPOINTS):
        vals = []
        for geno in GENOTYPES:
            row = kinase[(kinase["genotype"] == geno)
                         & (kinase["timepoint"] == tp)]
            vals.append(row["concordance_pct"].values[0] if len(row) else 0)
        bars = ax.bar(x + j * width, vals, width,
                      label=tp, alpha=0.8,
                      color=plt.cm.Blues(0.3 + 0.25 * j))
        # Annotate counts
        for k, v in enumerate(vals):
            ax.annotate(f"{v:.0f}%", (x[k] + j * width, v),
                        textcoords="offset points", xytext=(0, 4),
                        ha="center", fontsize=7)

    ax.axhline(50, color="red", linestyle="--", linewidth=1, alpha=0.5,
               label="chance (50%)")
    ax.set_xticks(x + width)
    ax.set_xticklabels([GENO_DISPLAY[g] for g in GENOTYPES])
    ax.set_ylabel("Concordance rate (%)")
    ax.set_title("Kinase-pathway concordance by genotype and timepoint")
    ax.legend()
    ax.set_ylim(0, 100)
    plt.tight_layout()
    path = os.path.join(out_dir, "kinase_concordance_by_genotype.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Wrote {path}")

    # --- Plot: coverage temporal ---
    fig, ax = plt.subplots(figsize=(8, 5))
    for geno in GENOTYPES:
        g_data = kinase[kinase["genotype"] == geno].sort_values("timepoint")
        ax.plot(g_data["timepoint"], g_data["coverage_pct"],
                marker="o", color=DISEASE_COLORS[geno],
                label=GENO_DISPLAY[geno], linewidth=2, markersize=6)

    ax.set_ylabel("Coverage rate (% sig pathways with kinase support)")
    ax.set_title("Kinase support coverage across timepoints")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(out_dir, "kinase_coverage_temporal.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Wrote {path}")


# ---------------------------------------------------------------------------
# S6: Publication Figures
# ---------------------------------------------------------------------------

def section_figures(data, out_dir):
    """Composite publication-quality figures."""
    hub = data["hub"]
    kinase = data["kinase"].copy()
    plt.rcParams.update(PUB_RCPARAMS)

    # Load pre-computed results if available
    add_path = os.path.join(out_dir, "additivity_by_pair_timepoint.csv")
    traj_path = os.path.join(out_dir, "trajectory_classification.csv")

    if not os.path.exists(add_path) or not os.path.exists(traj_path):
        print("  Run --additivity and --temporal first before --figures")
        return

    add_df = pd.read_csv(add_path)
    traj_df = pd.read_csv(traj_path)

    # ---- Figure 1: Overview (2x2) ----
    fig = plt.figure(figsize=(14, 11))
    gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

    # (A) Temporal dynamics: total significant per genotype x timepoint
    ax_a = fig.add_subplot(gs[0, 0])
    temporal = data["temporal"]
    for geno in GENOTYPES:
        g = temporal[temporal["genotype"] == geno]
        tp_sig = g.groupby("timepoint")["n_significant"].sum()
        tp_sig = tp_sig.reindex(TIMEPOINTS)
        ax_a.plot(TIMEPOINTS, tp_sig.values, marker="o",
                  color=DISEASE_COLORS[geno], label=GENO_DISPLAY[geno],
                  linewidth=2, markersize=6)
    ax_a.set_ylabel("Total significant pathways")
    ax_a.set_title("(A) Temporal dynamics")
    ax_a.legend(fontsize=8)

    # (B) Additivity scatter at 6mo
    ax_b = fig.add_subplot(gs[0, 1])
    tp_data = add_df[add_df["timepoint"] == "6mo"]
    for tissue in TISSUE_ORDER:
        mask = tp_data["receiver_tissue"] == tissue
        if mask.sum() == 0:
            continue
        ax_b.scatter(tp_data.loc[mask, "predicted_additive"],
                     tp_data.loc[mask, "observed_aptt"],
                     c=TISSUE_COLORS.get(tissue, "#999"),
                     label=tissue, alpha=0.6, s=15, edgecolors="none")
    lims = [min(ax_b.get_xlim()[0], ax_b.get_ylim()[0]),
            max(ax_b.get_xlim()[1], ax_b.get_ylim()[1])]
    ax_b.plot(lims, lims, "k--", alpha=0.3, linewidth=1)
    ax_b.set_xlim(lims)
    ax_b.set_ylim(lims)
    ax_b.set_aspect("equal")
    ax_b.set_xlabel("Predicted (App + Tau)")
    ax_b.set_ylabel(f"Observed ({GENO_DISPLAY['ApTt']})")
    ax_b.set_title("(B) Additivity at 6 months")
    ax_b.legend(fontsize=5, loc="upper left")

    # (C) Receiver ranking heatmap (top 10)
    ax_c = fig.add_subplot(gs[1, 0])
    recv_cent = hub.groupby(["contrast", "receiver"])["mean_abs_tpds"].sum()
    recv_pivot = recv_cent.unstack(level="contrast").reindex(columns=CONTRASTS)
    recv_pivot = recv_pivot.fillna(0)
    # Top 10 by mean across contrasts
    top10 = recv_pivot.mean(axis=1).nlargest(10).index
    recv_top = recv_pivot.loc[top10]
    im = ax_c.imshow(recv_top.values, cmap="YlOrRd", aspect="auto")
    ax_c.set_xticks(range(len(CONTRASTS)))
    ax_c.set_xticklabels(CONTRASTS, rotation=45, ha="right", fontsize=6)
    ax_c.set_yticks(range(len(recv_top)))
    ax_c.set_yticklabels(recv_top.index, fontsize=7)
    plt.colorbar(im, ax=ax_c, shrink=0.7)
    ax_c.set_title("(C) Top 10 receiver cell types")

    # (D) Kinase concordance
    ax_d = fig.add_subplot(gs[1, 1])
    conc_total = kinase["n_concordant"] + kinase["n_discordant"]
    kinase["concordance_pct"] = (
        100 * kinase["n_concordant"] / conc_total.clip(lower=1))
    kinase["genotype"] = kinase["contrast"].str.split("_").str[0]
    kinase["timepoint"] = kinase["contrast"].str.split("_").str[1]

    x = np.arange(len(GENOTYPES))
    width = 0.25
    for j, tp in enumerate(TIMEPOINTS):
        vals = []
        for geno in GENOTYPES:
            row = kinase[(kinase["genotype"] == geno)
                         & (kinase["timepoint"] == tp)]
            vals.append(row["concordance_pct"].values[0] if len(row) else 0)
        ax_d.bar(x + j * width, vals, width, label=tp, alpha=0.8,
                 color=plt.cm.Blues(0.3 + 0.25 * j))
    ax_d.axhline(50, color="red", linestyle="--", linewidth=1, alpha=0.5)
    ax_d.set_xticks(x + width)
    ax_d.set_xticklabels([GENO_DISPLAY[g] for g in GENOTYPES])
    ax_d.set_ylabel("Concordance (%)")
    ax_d.set_title("(D) Kinase concordance")
    ax_d.legend(fontsize=7)
    ax_d.set_ylim(0, 70)

    for fmt in ["png", "pdf"]:
        path = os.path.join(out_dir, f"figure_overview.{fmt}")
        fig.savefig(path)
    plt.close(fig)
    print(f"  Wrote figure_overview.png/.pdf")

    # ---- Figure 2: Sub-additivity detail ----
    fig = plt.figure(figsize=(14, 6))
    gs = GridSpec(1, 2, figure=fig, wspace=0.3)

    # (A) Interaction distribution (constant across timepoints)
    ax_a = fig.add_subplot(gs[0, 0])
    # Use any single timepoint — interaction is identical at all timepoints
    int_vals = add_df[add_df["timepoint"] == "2mo"]["interaction"].values
    ax_a.hist(int_vals, bins=40, color=DISEASE_COLORS["ApTt"], alpha=0.6,
              edgecolor="k", linewidth=0.5)
    ax_a.axvline(0, color="black", linewidth=1.5, linestyle="--")
    med = np.median(int_vals)
    ax_a.axvline(med, color="red", linewidth=1, label=f"median={med:+.4f}")
    ax_a.set_xlabel("Interaction (observed - predicted)")
    ax_a.set_ylabel("Number of pairs")
    ax_a.set_title("(A) Interaction term distribution\n"
                   "(= OLS Int coefficient, time-invariant)")
    ax_a.legend(fontsize=8)

    # (B) Top 10 most sub-additive pairs
    ax_b = fig.add_subplot(gs[0, 1])
    int_sorted = add_df[add_df["timepoint"] == "2mo"].sort_values("interaction")
    top_sub = int_sorted.head(10)

    colors_bar = [TISSUE_COLORS.get(
        SUBCLASS_TO_TISSUE_CATEGORY.get(r, "Other"), "#999")
        for r in top_sub["receiver"]]
    ax_b.barh(range(len(top_sub)), top_sub["interaction"].values,
              color=colors_bar, alpha=0.7, edgecolor="k", linewidth=0.5)
    labels = [f"{row['sender']} -> {row['receiver']}"
              for _, row in top_sub.iterrows()]
    ax_b.set_yticks(range(len(top_sub)))
    ax_b.set_yticklabels(labels, fontsize=7)
    ax_b.axvline(0, color="black", linewidth=1)
    ax_b.set_xlabel("Interaction (observed - predicted)")
    ax_b.set_title("(B) Most sub-additive pairs")
    ax_b.invert_yaxis()

    for fmt in ["png", "pdf"]:
        path = os.path.join(out_dir, f"figure_sub_additivity.{fmt}")
        fig.savefig(path)
    plt.close(fig)
    print(f"  Wrote figure_sub_additivity.png/.pdf")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Examine factorial Incytr results")
    parser.add_argument("--summary", action="store_true",
                        help="Summary statistics")
    parser.add_argument("--additivity", action="store_true",
                        help="Additivity analysis (ApTt vs App+Tau)")
    parser.add_argument("--temporal", action="store_true",
                        help="Temporal trajectory classification")
    parser.add_argument("--celltype", action="store_true",
                        help="Cell-type centrality analysis")
    parser.add_argument("--kinase", action="store_true",
                        help="Kinase concordance and coverage validation")
    parser.add_argument("--figures", action="store_true",
                        help="Publication-quality composite figures")
    parser.add_argument("--run", action="store_true",
                        help="Run all sections")
    args = parser.parse_args()

    if args.run:
        args.summary = args.additivity = args.temporal = True
        args.celltype = args.kinase = args.figures = True

    if not any([args.summary, args.additivity, args.temporal,
                args.celltype, args.kinase, args.figures]):
        parser.print_help()
        return

    data = load_all_data()
    out_dir = os.path.join(AGG_DIR, "examination")
    os.makedirs(out_dir, exist_ok=True)
    print(f"Output: {out_dir}\n")

    if args.summary:
        print("\n=== S1: Summary Statistics ===")
        section_summary(data, out_dir)

    if args.additivity:
        print("\n=== S2: Additivity Analysis ===")
        section_additivity(data, out_dir)

    if args.temporal:
        print("\n=== S3: Temporal Trajectory ===")
        section_temporal(data, out_dir)

    if args.celltype:
        print("\n=== S4: Cell-Type Analysis ===")
        section_celltype(data, out_dir)

    if args.kinase:
        print("\n=== S5: Kinase Validation ===")
        section_kinase(data, out_dir)

    if args.figures:
        print("\n=== S6: Publication Figures ===")
        section_figures(data, out_dir)

    print("\nExamination complete.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
ApTt Additivity Analysis

Tests whether the combined-pathology (ApTt) kinase signature is the sum of
single-pathology signatures (TTau + AppP), or whether amyloid-tau interaction
produces emergent signaling.

Includes confound diagnostics:
  - Percentile ceiling effect: does the pct cutoff artificially equalize
    the number of significant kinases across conditions?
  - Raw data size: do conditions differ in available phosphosites,
    which could mask or inflate synergistic signal?

Reads existing enrichment CSVs — no re-running of the pipeline required.
"""

import argparse
import os
import sys
from itertools import product

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from scipy import stats

# Add code/ directory to path for imports (matches other scripts in this repo)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

# ── Constants ──────────────────────────────────────────────────────────────────

PVAL_THRESH = config.PVAL_SIG
CONDITIONS = ["Ttau", "AppP", "ApTt"]

# Bulk condition mapping (canonical → filename token)
BULK_COND_MAP = {"Ttau": "T22", "AppP": "APP", "ApTt": "T22_APP"}


# ── Data Loading ──────────────────────────────────────────────────────────────

def _enrichment_filename(gender, timepoint, condition, cluster, pct, bulk=False):
    """Build the enrichment CSV filename for one comparison."""
    pct_str = str(pct) if pct != int(pct) else str(int(pct))
    if bulk:
        g = config.BULK_GENDER_MAP.get(gender, gender)
        cond = BULK_COND_MAP[condition]
        return (f"{g}_{timepoint}_WT_vs_{g}_{timepoint}_{cond}"
                f"_pct{pct_str}_lff{config.LFF_THRESH}_enrichment_results.csv")
    else:
        return (f"{gender}_{timepoint}_WTyp_{cluster}_vs_"
                f"{gender}_{timepoint}_{condition}_{cluster}_"
                f"pct{pct_str}_lff{config.LFF_THRESH}_enrichment_results.csv")


def load_condition_triplet(gender, timepoint, cluster, pct, enrichment_dir, bulk=False):
    """Load and merge the 3 matched enrichment CSVs (TTau, AppP, ApTt).

    Returns a DataFrame indexed by kinase with columns per condition:
      lff, pval, dir, n_substrates, median_substrate_lfc
    Plus derived: lff_sum, lff_residual, sig flags.
    Returns None if any file is missing.
    """
    dfs = {}
    for cond in CONDITIONS:
        fname = _enrichment_filename(gender, timepoint, cond, cluster, pct, bulk)
        fpath = os.path.join(enrichment_dir, fname)
        if not os.path.exists(fpath):
            return None
        df = pd.read_csv(fpath, index_col=0)
        dfs[cond] = df

    merged = pd.DataFrame(index=dfs["Ttau"].index)
    for cond in CONDITIONS:
        df = dfs[cond]
        merged[f"lff_{cond}"] = df["most_sig_log2_freq_factor"]
        merged[f"pval_{cond}"] = df["most_sig_fisher_adj_pval"]
        merged[f"dir_{cond}"] = df["most_sig_direction"]
        merged[f"n_subs_{cond}"] = df["n_substrates"]
        merged[f"median_lfc_{cond}"] = df["median_substrate_lfc"]

    merged["lff_sum"] = merged["lff_Ttau"] + merged["lff_AppP"]
    merged["lff_residual"] = merged["lff_ApTt"] - merged["lff_sum"]

    for cond in CONDITIONS:
        merged[f"sig_{cond}"] = merged[f"pval_{cond}"] < PVAL_THRESH

    return merged


def get_contexts(bulk=False):
    """Generate all (gender, timepoint, cluster) context tuples."""
    genders = ["ma"] if not bulk else ["M"]
    timepoints = ["2mo", "4mo", "6mo"]

    if bulk:
        return [{"gender": g, "timepoint": tp, "cluster": None}
                for g, tp in product(genders, timepoints)]
    else:
        clusters_df = pd.read_csv(config.MEDIAN_CLUSTER_SIZES_FILE, index_col=0)
        top_clusters = (clusters_df.sort_values("median_cluster_size", ascending=False)
                        .head(10).index.tolist())
        return [{"gender": g, "timepoint": tp, "cluster": c}
                for g, tp, c in product(genders, timepoints, top_clusters)]


def load_all_triplets(pct, enrichment_dir, bulk=False):
    """Load all available triplets. Returns list of (context_dict, merged_df)."""
    contexts = get_contexts(bulk)
    results = []
    for ctx in contexts:
        df = load_condition_triplet(
            ctx["gender"], ctx["timepoint"], ctx["cluster"], pct, enrichment_dir, bulk)
        if df is not None:
            results.append((ctx, df))
    return results


# ── Analysis 1: Set Overlap ──────────────────────────────────────────────────

def classify_kinase_sets(df):
    """Classify each kinase into one of 7 mutually exclusive overlap categories."""
    t = df["sig_Ttau"]
    a = df["sig_AppP"]
    x = df["sig_ApTt"]

    categories = pd.Series("none", index=df.index)
    categories[t & ~a & ~x] = "Ttau_only"
    categories[~t & a & ~x] = "AppP_only"
    categories[~t & ~a & x] = "ApTt_only_emergent"
    categories[t & a & ~x] = "Ttau+AppP_missing_ApTt"
    categories[t & ~a & x] = "Ttau+ApTt"
    categories[~t & a & x] = "AppP+ApTt"
    categories[t & a & x] = "all_three"
    return categories


def compute_set_overlap(triplets, output_dir):
    """Compute set overlap summary for all contexts."""
    rows = []
    for ctx, df in triplets:
        cats = classify_kinase_sets(df)
        counts = cats.value_counts()
        row = {
            "gender": config.GENDER_MAP.get(ctx["gender"], ctx["gender"]),
            "timepoint": ctx["timepoint"],
            "cluster": ctx["cluster"] or "bulk",
        }
        for cat in ["Ttau_only", "AppP_only", "ApTt_only_emergent",
                     "Ttau+AppP_missing_ApTt", "Ttau+ApTt", "AppP+ApTt",
                     "all_three", "none"]:
            row[cat] = counts.get(cat, 0)
        rows.append(row)

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(output_dir, "set_overlap_summary.csv"), index=False)
    print(f"  Set overlap summary: {len(out)} contexts written.")
    return out


# ── Analysis 2: LFF Additivity Correlation ───────────────────────────────────

def compute_additivity_correlations(triplets, output_dir):
    """Compute per-context correlation between ApTt LFF and TTau+AppP LFF."""
    corr_rows = []
    detail_rows = []

    for ctx, df in triplets:
        x = df["lff_sum"]
        y = df["lff_ApTt"]
        mask = np.isfinite(x) & np.isfinite(y)
        x_clean, y_clean = x[mask], y[mask]

        pearson_r, pearson_p = stats.pearsonr(x_clean, y_clean)
        spearman_r, spearman_p = stats.spearmanr(x_clean, y_clean)

        ctx_label = (f"{config.GENDER_MAP.get(ctx['gender'], ctx['gender'])}_"
                     f"{ctx['timepoint']}_{ctx['cluster'] or 'bulk'}")

        corr_rows.append({
            "gender": config.GENDER_MAP.get(ctx["gender"], ctx["gender"]),
            "timepoint": ctx["timepoint"],
            "cluster": ctx["cluster"] or "bulk",
            "pearson_r": pearson_r,
            "pearson_p": pearson_p,
            "spearman_r": spearman_r,
            "spearman_p": spearman_p,
            "mean_residual": df["lff_residual"].mean(),
            "std_residual": df["lff_residual"].std(),
        })

        # Top outliers by |residual|
        top_outliers = df.nlargest(5, "lff_residual")
        bot_outliers = df.nsmallest(5, "lff_residual")
        for kinase, row in pd.concat([top_outliers, bot_outliers]).iterrows():
            detail_rows.append({
                "context": ctx_label,
                "gender": config.GENDER_MAP.get(ctx["gender"], ctx["gender"]),
                "timepoint": ctx["timepoint"],
                "cluster": ctx["cluster"] or "bulk",
                "kinase": kinase,
                "lff_ApTt": row["lff_ApTt"],
                "lff_sum": row["lff_sum"],
                "lff_residual": row["lff_residual"],
                "type": "synergistic" if row["lff_residual"] > 0 else "antagonistic",
            })

    corr_df = pd.DataFrame(corr_rows)
    corr_df.to_csv(os.path.join(output_dir, "additivity_correlations.csv"), index=False)

    detail_df = pd.DataFrame(detail_rows)
    detail_df.to_csv(os.path.join(output_dir, "additivity_kinase_detail.csv"), index=False)

    print(f"  Additivity correlations: {len(corr_df)} contexts, "
          f"{len(detail_df)} outlier entries.")
    return corr_df, detail_df


# ── Analysis 3: Emergent Kinase Identification ───────────────────────────────

def compute_emergent_kinases(triplets, output_dir):
    """Kinases significant in ApTt but NOT in either TTau or AppP."""
    records = []
    for ctx, df in triplets:
        emergent = df[df["sig_ApTt"] & ~df["sig_Ttau"] & ~df["sig_AppP"]]
        for kinase, row in emergent.iterrows():
            records.append({
                "kinase": kinase,
                "gender": config.GENDER_MAP.get(ctx["gender"], ctx["gender"]),
                "timepoint": ctx["timepoint"],
                "cluster": ctx["cluster"] or "bulk",
                "dir_ApTt": row["dir_ApTt"],
                "lff_ApTt": row["lff_ApTt"],
                "pval_ApTt": row["pval_ApTt"],
                "lff_Ttau": row["lff_Ttau"],
                "pval_Ttau": row["pval_Ttau"],
                "lff_AppP": row["lff_AppP"],
                "pval_AppP": row["pval_AppP"],
            })

    if not records:
        print("  No emergent kinases found.")
        pd.DataFrame().to_csv(os.path.join(output_dir, "emergent_kinases.csv"), index=False)
        return pd.DataFrame()

    detail = pd.DataFrame(records)

    # Aggregate across contexts
    agg = (detail.groupby("kinase")
           .agg(
               n_contexts=("kinase", "size"),
               predominant_dir=("dir_ApTt", lambda x: x.mode().iloc[0] if len(x) > 0 else "?"),
               mean_lff_ApTt=("lff_ApTt", "mean"),
               mean_lff_Ttau=("lff_Ttau", "mean"),
               mean_lff_AppP=("lff_AppP", "mean"),
               mean_pval_Ttau=("pval_Ttau", "mean"),
               mean_pval_AppP=("pval_AppP", "mean"),
           )
           .sort_values(by="n_contexts", ascending=False))

    agg.to_csv(os.path.join(output_dir, "emergent_kinases.csv"))
    detail.to_csv(os.path.join(output_dir, "emergent_kinases_detail.csv"), index=False)
    print(f"  Emergent kinases: {len(agg)} unique kinases across {len(detail)} instances.")
    return agg


# ── Analysis 4: Direction Concordance ─────────────────────────────────────────

def compute_direction_concordance(triplets, output_dir):
    """Among kinases significant in ApTt AND (TTau or AppP): does direction match?"""
    records = []
    for ctx, df in triplets:
        mask = df["sig_ApTt"] & (df["sig_Ttau"] | df["sig_AppP"])
        subset = df[mask]
        for kinase, row in subset.iterrows():
            for cond in ["Ttau", "AppP"]:
                if row[f"sig_{cond}"]:
                    concordant = row["dir_ApTt"] == row[f"dir_{cond}"]
                    records.append({
                        "kinase": kinase,
                        "gender": config.GENDER_MAP.get(ctx["gender"], ctx["gender"]),
                        "timepoint": ctx["timepoint"],
                        "cluster": ctx["cluster"] or "bulk",
                        "compared_condition": cond,
                        "dir_ApTt": row["dir_ApTt"],
                        f"dir_{cond}": row[f"dir_{cond}"],
                        "concordant": concordant,
                    })

    out = pd.DataFrame(records)
    out.to_csv(os.path.join(output_dir, "direction_concordance.csv"), index=False)
    if len(out) > 0:
        n_conc = out["concordant"].sum()
        n_total = len(out)
        print(f"  Direction concordance: {n_conc}/{n_total} "
              f"({100*n_conc/n_total:.1f}%) concordant.")
    else:
        print("  Direction concordance: no overlapping significant kinases.")
    return out


# ── Analysis 5: Signal Concentration (Diffuse Signal Hypothesis) ──────────────

def _gini(values):
    """Gini coefficient (0 = perfectly equal, 1 = maximally concentrated)."""
    v = np.sort(np.asarray(values, dtype=float))
    n = len(v)
    if n == 0 or v.sum() == 0:
        return np.nan
    index = np.arange(1, n + 1)
    return (2 * np.sum(index * v) - (n + 1) * np.sum(v)) / (n * np.sum(v))


def _top_k_concentration(values, k=10):
    """Fraction of total signal in top-k entries."""
    v = np.sort(np.asarray(values, dtype=float))[::-1]
    total = v.sum()
    if total == 0:
        return np.nan
    return v[:k].sum() / total


def _lorenz_curve(values):
    """Return (x, y) arrays for a Lorenz curve of sorted values."""
    v = np.sort(np.asarray(values, dtype=float))
    cumsum = np.cumsum(v)
    total = cumsum[-1]
    if total == 0:
        return np.linspace(0, 1, len(v)), np.linspace(0, 1, len(v))
    x = np.arange(1, len(v) + 1) / len(v)
    y = cumsum / total
    return np.concatenate([[0], x]), np.concatenate([[0], y])


def compute_signal_concentration(pct, enrichment_dir, output_dir, bulk=False):
    """Measure whether ApTt distributes kinase evidence more diffusely than
    the single-pathology conditions.

    For each context × condition, computes from raw Fisher p-values:
      - Gini coefficient of -log10(p): higher = concentrated in few kinases
      - Top-10 concentration: fraction of total evidence in 10 strongest kinases
      - Raw p-value thresholds: counts at p<0.001, p<0.01, p<0.05
      - BH survival: how many survive multiple-testing correction

    Tests the diffuse-signal hypothesis: if ApTt activates many kinases weakly
    rather than a few strongly, its Gini will be lower and BH correction will
    disproportionately penalise it.

    Returns (summary_df, per_kinase_df).
    """
    contexts = get_contexts(bulk)
    pct_str = str(pct) if pct != int(pct) else str(int(pct))
    rows = []
    lorenz_data = []

    for ctx in contexts:
        cluster = ctx["cluster"]
        tp = ctx["timepoint"]
        gender = ctx["gender"]
        g_label = config.GENDER_MAP.get(gender, gender)

        for cond in CONDITIONS:
            fname = _enrichment_filename(gender, tp, cond, cluster, pct, bulk)
            fpath = os.path.join(enrichment_dir, fname)
            if not os.path.exists(fpath):
                continue
            df = pd.read_csv(fpath, index_col=0)

            raw_p = df["most_sig_fisher_pval"]
            adj_p = df["most_sig_fisher_adj_pval"]
            neg_log_p = -np.log10(raw_p.clip(lower=1e-300))

            g = _gini(neg_log_p)
            t10 = _top_k_concentration(neg_log_p, 10)
            t20 = _top_k_concentration(neg_log_p, 20)

            # Higher Criticism: formal test for diffuse signal
            from permutation_correction import compute_higher_criticism
            hc_star, hc_idx = compute_higher_criticism(raw_p.values)

            rows.append({
                "gender": g_label,
                "timepoint": tp,
                "cluster": cluster or "bulk",
                "condition": cond,
                "gini": g,
                "top10_conc": t10,
                "top20_conc": t20,
                "hc_star": hc_star,
                "hc_index": hc_idx,
                "n_raw_p001": int((raw_p < 0.001).sum()),
                "n_raw_p01": int((raw_p < 0.01).sum()),
                "n_raw_p05": int((raw_p < 0.05).sum()),
                "n_adj_p10": int((adj_p < 0.1).sum()),
                "total_evidence": neg_log_p.sum(),
                "max_evidence": neg_log_p.max(),
            })

            # Save Lorenz curve data for plotting
            lx, ly = _lorenz_curve(neg_log_p.values)
            lorenz_data.append({
                "gender": g_label, "timepoint": tp,
                "cluster": cluster or "bulk", "condition": cond,
                "lorenz_x": lx, "lorenz_y": ly,
            })

    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(output_dir, "signal_concentration.csv"), index=False)

    # Paired statistical test: ApTt Gini vs avg(Ttau, AppP)
    pivot_gini = summary.pivot_table(
        index=["timepoint", "cluster"], columns="condition", values="gini")
    common = pivot_gini.dropna()
    if len(common) > 2:
        aptt_g = common["ApTt"].values
        comp_g = (common["Ttau"].values + common["AppP"].values) / 2
        diff = aptt_g - comp_g
        t_stat, t_pval = stats.ttest_rel(aptt_g, comp_g)
        w_stat, w_pval = stats.wilcoxon(diff)
        n_diffuse = (diff < 0).sum()
        n_total = len(diff)

        print(f"  Signal concentration: {n_total} contexts")
        print(f"    Mean Gini — Ttau: {common['Ttau'].mean():.3f}, "
              f"AppP: {common['AppP'].mean():.3f}, ApTt: {common['ApTt'].mean():.3f}")
        print(f"    ApTt more diffuse in {n_diffuse}/{n_total} contexts")
        print(f"    Paired t-test (ApTt vs component avg): "
              f"t={t_stat:.3f}, p={t_pval:.4f}")
        print(f"    Wilcoxon signed-rank: p={w_pval:.4f}")

        # Higher Criticism summary
        pivot_hc = summary.pivot_table(
            index=["timepoint", "cluster"], columns="condition", values="hc_star")
        hc_common = pivot_hc.dropna()
        if len(hc_common) > 0:
            for cond in CONDITIONS:
                if cond in hc_common.columns:
                    vals = hc_common[cond]
                    n_sig = (vals > 2).sum()
                    print(f"    HC* — {cond}: mean={vals.mean():.2f}, "
                          f">{'>'}2 in {n_sig}/{len(vals)} contexts")

        # Stratified by timepoint
        for tp in ["2mo", "4mo", "6mo"]:
            tp_mask = common.index.get_level_values("timepoint") == tp
            if tp_mask.sum() < 3:
                continue
            tp_diff = diff[tp_mask]
            tp_t, tp_p = stats.ttest_1samp(tp_diff, 0)
            n_tp_diff = (tp_diff < 0).sum()
            print(f"    {tp}: ApTt more diffuse in {n_tp_diff}/{tp_mask.sum()}, "
                  f"mean diff={tp_diff.mean():.4f}, p={tp_p:.4f}")

    return summary, lorenz_data


def plot_signal_concentration(summary, lorenz_data, pct, output_dir):
    """4-panel figure visualizing the diffuse-signal hypothesis.

    A: Lorenz curves for representative loss context (ApTt below diagonal = diffuse)
    B: Gini coefficient by condition×timepoint (grouped bar)
    C: Heatmap of Gini difference (ApTt - component avg) per cluster×timepoint
    D: Raw p<0.001 counts vs adj_p<0.1 counts — shows BH attrition
    """
    import seaborn as sns

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    # ── Panel A: Lorenz curves for a representative context ──
    ax = axes[0, 0]
    # Pick a context where ApTt signal loss is visible
    target_contexts = [
        ("4mo", "Astrocytes"),
        ("6mo", "Astrocytes"),
        ("2mo", "Astrocytes"),  # control: ApTt works here
    ]
    cond_colors = config.CONDITION_COLORS
    linestyles = {"2mo": ":", "4mo": "-", "6mo": "--"}

    for tp, cluster in target_contexts:
        for cond in CONDITIONS:
            match = [d for d in lorenz_data
                     if d["timepoint"] == tp and d["cluster"] == cluster
                     and d["condition"] == cond]
            if not match:
                continue
            d = match[0]
            ax.plot(d["lorenz_x"], d["lorenz_y"],
                    color=cond_colors[cond], ls=linestyles[tp],
                    lw=1.5, alpha=0.8)
    ax.plot([0, 1], [0, 1], "k-", alpha=0.2, lw=0.5)

    # Build legend manually
    from matplotlib.lines import Line2D
    handles = []
    for cond in CONDITIONS:
        handles.append(Line2D([0], [0], color=cond_colors[cond], lw=2, label=cond))
    for tp, ls in linestyles.items():
        handles.append(Line2D([0], [0], color="gray", ls=ls, lw=1.5, label=tp))
    ax.legend(handles=handles, fontsize=7, loc="upper left")
    ax.set_xlabel("Fraction of kinases (ranked by evidence)")
    ax.set_ylabel("Cumulative fraction of total evidence")
    ax.set_title("A. Lorenz Curves — Astrocytes\n(below diagonal = more diffuse)")

    # ── Panel B: Gini by condition×timepoint ──
    ax = axes[0, 1]
    tp_order = ["2mo", "4mo", "6mo"]
    bar_width = 0.25
    for i, cond in enumerate(CONDITIONS):
        means = []
        sems = []
        for tp in tp_order:
            vals = summary[(summary["condition"] == cond) &
                           (summary["timepoint"] == tp)]["gini"]
            means.append(vals.mean())
            sems.append(vals.sem())
        x = np.arange(len(tp_order)) + i * bar_width
        ax.bar(x, means, bar_width, yerr=sems, capsize=3,
               color=cond_colors[cond], label=cond, alpha=0.85)
    ax.set_xticks(np.arange(len(tp_order)) + bar_width)
    ax.set_xticklabels(tp_order)
    ax.set_ylabel("Gini coefficient")
    ax.set_title("B. Signal Concentration by Condition\n(lower = more diffuse)")
    ax.legend(fontsize=8)

    # ── Panel C: Heatmap of Gini difference ──
    ax = axes[1, 0]
    pivot = summary.pivot_table(
        index=["timepoint", "cluster"], columns="condition", values="gini")
    pivot = pivot.dropna()
    pivot["diff"] = pivot["ApTt"] - (pivot["Ttau"] + pivot["AppP"]) / 2
    heat = pivot["diff"].reset_index().pivot_table(
        index="cluster", columns="timepoint", values="diff")
    heat = heat.reindex(columns=["2mo", "4mo", "6mo"])
    vmax = max(abs(heat.min().min()), abs(heat.max().max()))
    sns.heatmap(heat, annot=True, fmt=".3f", cmap="RdBu_r",
                center=0, vmin=-vmax, vmax=vmax, ax=ax,
                linewidths=0.5, cbar_kws={"label": "Gini diff (+ = concentrated, - = diffuse)"})
    ax.set_title("C. ApTt Gini - avg(Ttau,AppP)\n(blue = ApTt more diffuse)")
    ax.set_ylabel("")

    # ── Panel D: Higher Criticism statistic by condition×timepoint ──
    ax = axes[1, 1]
    if "hc_star" in summary.columns:
        for i, cond in enumerate(CONDITIONS):
            means = []
            sems = []
            for tp in tp_order:
                vals = summary[(summary["condition"] == cond) &
                               (summary["timepoint"] == tp)]["hc_star"]
                means.append(vals.mean())
                sems.append(vals.sem())
            x = np.arange(len(tp_order)) + i * bar_width
            ax.bar(x, means, bar_width, yerr=sems, capsize=3,
                   color=cond_colors[cond], label=cond, alpha=0.85)
        ax.axhline(y=2, color="red", ls="--", lw=1, alpha=0.5, label="HC*=2 (significance)")
        ax.set_xticks(np.arange(len(tp_order)) + bar_width)
        ax.set_xticklabels(tp_order)
        ax.set_ylabel("Higher Criticism (HC*)")
        ax.set_title("D. Global Signal Detection\n(HC*>2 = significant diffuse signal)")
        ax.legend(fontsize=8)
    else:
        # Fallback: BH attrition
        from matplotlib.patches import Patch
        for cond in CONDITIONS:
            sub = summary[summary["condition"] == cond]
            raw = sub["n_raw_p05"].values
            adj = sub["n_adj_p10"].values
            with np.errstate(divide="ignore", invalid="ignore"):
                survival = np.where(raw > 0, adj / raw, 0)
            for tp_idx, tp in enumerate(tp_order):
                tp_mask = sub["timepoint"] == tp
                tp_surv = survival[tp_mask.values]
                x_pos = tp_idx + CONDITIONS.index(cond) * 0.25
                ax.bar(x_pos, tp_surv.mean(), 0.2, color=cond_colors[cond],
                       alpha=0.85)
        ax.set_xticks([x + 0.25 for x in range(3)])
        ax.set_xticklabels(tp_order)
        ax.set_ylabel("BH survival rate")
        ax.set_title("D. Multiple-Testing Attrition")
        ax.legend(handles=[Patch(color=cond_colors[c], label=c) for c in CONDITIONS],
                  fontsize=8)

    pct_str = str(pct) if pct != int(pct) else str(int(pct))
    fig.suptitle("Signal Concentration Analysis: Diffuse-Signal Hypothesis\n"
                 "Does ApTt activate many kinases weakly rather than few strongly?",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    path = os.path.join(output_dir, f"signal_concentration_pct{pct_str}.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Signal concentration plot saved: {path}")


# ── Confound Diagnostic: Percentile Ceiling Effect ───────────────────────────

def diagnose_percentile_ceiling(triplets, output_dir):
    """Check whether the percentile cutoff artificially equalizes significant
    kinase counts across conditions.

    For each context, compares:
      - n_sig per condition (are they suspiciously equal?)
      - LFF magnitude distributions (is ApTt's LFF range compressed relative
        to what additivity would predict?)
    """
    rows = []
    for ctx, df in triplets:
        g_label = config.GENDER_MAP.get(ctx["gender"], ctx["gender"])
        base = {
            "gender": g_label,
            "timepoint": ctx["timepoint"],
            "cluster": ctx["cluster"] or "bulk",
        }
        for cond in CONDITIONS:
            sig_mask = df[f"sig_{cond}"]
            lff_col = f"lff_{cond}"
            nsubs_col = f"n_subs_{cond}"
            lff_vals = df[lff_col]
            nsubs_vals = df[nsubs_col]
            sig_lff = lff_vals[sig_mask]

            rows.append({
                **base,
                "condition": cond,
                "n_sig": sig_mask.sum(),
                "n_kinases_tested": len(df),
                "pct_sig": 100 * sig_mask.sum() / len(df) if len(df) > 0 else 0,
                "mean_lff_all": lff_vals.mean(),
                "std_lff_all": lff_vals.std(),
                "mean_lff_sig": sig_lff.mean() if len(sig_lff) > 0 else np.nan,
                "lff_range": lff_vals.max() - lff_vals.min(),
                "lff_iqr": lff_vals.quantile(0.75) - lff_vals.quantile(0.25),
                "mean_n_substrates": nsubs_vals.mean(),
                "median_n_substrates": nsubs_vals.median(),
                "total_substrates": nsubs_vals.sum(),
                "mean_n_subs_sig": nsubs_vals[sig_mask].mean() if sig_mask.sum() > 0 else np.nan,
            })

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(output_dir, "confound_percentile_ceiling.csv"), index=False)

    # Summary: are sig counts suspiciously equal across conditions?
    pivot = out.pivot_table(index=["gender", "timepoint", "cluster"],
                            columns="condition", values="n_sig")
    pivot["max_diff"] = pivot.max(axis=1) - pivot.min(axis=1)
    pivot["cv"] = pivot[CONDITIONS].std(axis=1) / pivot[CONDITIONS].mean(axis=1)

    n_equal = (pivot["max_diff"] == 0).sum()
    n_near_equal = (pivot["cv"] < 0.1).sum()
    n_total = len(pivot)

    print(f"  Percentile ceiling diagnostic: {n_total} contexts")
    print(f"    Exactly equal sig counts across conditions: {n_equal}/{n_total}")
    print(f"    Near-equal (CV < 0.1): {n_near_equal}/{n_total}")

    return out


# ── Confound Visualization: Condition Comparison Panel ───────────────────────

def plot_confound_panel(triplets, pct, output_dir):
    """3-panel figure comparing conditions across the two confound axes.

    Panel A: Significant kinase count per condition per timepoint (bar chart)
    Panel B: Total substrate count per condition (bar chart)
    Panel C: LFF spread per condition (box plot)
    """
    import seaborn as sns

    rows = []
    for ctx, df in triplets:
        g_label = config.GENDER_MAP.get(ctx["gender"], ctx["gender"])
        for cond in CONDITIONS:
            rows.append({
                "gender": g_label,
                "timepoint": ctx["timepoint"],
                "cluster": ctx["cluster"] or "bulk",
                "condition": cond,
                "n_sig": df[f"sig_{cond}"].sum(),
                "total_substrates": df[f"n_subs_{cond}"].sum(),
                "lff_iqr": (df[f"lff_{cond}"].quantile(0.75)
                            - df[f"lff_{cond}"].quantile(0.25)),
                "mean_abs_lff": df[f"lff_{cond}"].abs().mean(),
            })
    rdf = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    cond_colors = [config.CONDITION_COLORS[c] for c in CONDITIONS]

    # Panel A: sig kinase counts
    ax = axes[0]
    agg_a = rdf.groupby(["timepoint", "condition"])["n_sig"].mean().reset_index()
    for i, cond in enumerate(CONDITIONS):
        sub = agg_a[agg_a["condition"] == cond]
        ax.bar([x + i * 0.25 for x in range(len(sub))], sub["n_sig"],
               width=0.25, color=cond_colors[i], label=cond)
    ax.set_xticks([x + 0.25 for x in range(3)])
    ax.set_xticklabels(["2mo", "4mo", "6mo"])
    ax.set_ylabel("Mean sig kinases per context")
    ax.set_title("A. Significant Kinase Counts by Condition")
    ax.legend()

    # Panel B: total substrates
    ax = axes[1]
    agg_b = rdf.groupby(["timepoint", "condition"])["total_substrates"].mean().reset_index()
    for i, cond in enumerate(CONDITIONS):
        sub = agg_b[agg_b["condition"] == cond]
        ax.bar([x + i * 0.25 for x in range(len(sub))], sub["total_substrates"],
               width=0.25, color=cond_colors[i], label=cond)
    ax.set_xticks([x + 0.25 for x in range(3)])
    ax.set_xticklabels(["2mo", "4mo", "6mo"])
    ax.set_ylabel("Mean total substrates per context")
    ax.set_title("B. Substrate Pool Size by Condition")
    ax.legend()

    # Panel C: LFF spread
    ax = axes[2]
    sns.boxplot(data=rdf, x="condition", y="mean_abs_lff", hue="condition",
                order=CONDITIONS, palette=config.CONDITION_COLORS,
                legend=False, ax=ax)
    ax.set_ylabel("Mean |LFF| per context")
    ax.set_title("C. LFF Magnitude Distribution by Condition")

    pct_str = str(pct) if pct != int(pct) else str(int(pct))
    fig.suptitle(f"Confound Diagnostics: Condition Comparison (pct{pct_str})", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    path = os.path.join(output_dir, f"confound_panel_pct{pct_str}.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Confound panel saved: {path}")


# ── Visualization 1: Additivity Scatter (Hero Figure) ────────────────────────

def plot_additivity_scatter(triplets, pct, output_dir):
    """X = TTau+AppP LFF, Y = ApTt LFF, diagonal = perfect additivity.
    1 row (M) x 3 cols (2mo, 4mo, 6mo), clusters pooled."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
    tp_order = ["2mo", "4mo", "6mo"]

    pooled = {}
    for ctx, df in triplets:
        g_label = config.GENDER_MAP.get(ctx["gender"], ctx["gender"])
        key = (g_label, ctx["timepoint"])
        if key not in pooled:
            pooled[key] = []
        cats = classify_kinase_sets(df)
        pooled[key].append(df.assign(category=cats))

    cat_colors = {
        "none": "#cccccc",
        "Ttau_only": "#1f77b4",
        "AppP_only": "#ff7f0e",
        "ApTt_only_emergent": "#d62728",
        "Ttau+AppP_missing_ApTt": "#9467bd",
        "Ttau+ApTt": "#2ca02c",
        "AppP+ApTt": "#8c564b",
        "all_three": "#e377c2",
    }

    for j, tp in enumerate(tp_order):
        ax = axes[j]
        key = ("M", tp)
        if key not in pooled:
            ax.set_visible(False)
            continue

        combined = pd.concat(pooled[key], ignore_index=False)

        for cat in ["none", "Ttau_only", "AppP_only", "Ttau+AppP_missing_ApTt",
                    "Ttau+ApTt", "AppP+ApTt", "all_three", "ApTt_only_emergent"]:
            mask = combined["category"] == cat
            if mask.sum() == 0:
                continue
            ax.scatter(combined.loc[mask, "lff_sum"],
                       combined.loc[mask, "lff_ApTt"],
                       c=cat_colors[cat], s=8, alpha=0.4, label=cat, rasterized=True)

        lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]),
                max(ax.get_xlim()[1], ax.get_ylim()[1])]
        ax.plot(lims, lims, "k--", alpha=0.3, lw=1)
        ax.set_xlim(lims)
        ax.set_ylim(lims)

        x = combined["lff_sum"]
        y = combined["lff_ApTt"]
        mask_finite = np.isfinite(x) & np.isfinite(y)
        r, _ = stats.pearsonr(x[mask_finite], y[mask_finite])
        ax.text(0.05, 0.95, f"r = {r:.3f}", transform=ax.transAxes,
                va="top", fontsize=10, fontweight="bold")

        ax.set_title(f"M — {tp}", fontsize=12)
        if j == 0:
            ax.set_ylabel("ApTt LFF")
        ax.set_xlabel("TTau + AppP LFF (additive prediction)")

    handles = [mpatches.Patch(color=cat_colors[c], label=c.replace("_", " "))
               for c in cat_colors if c != "none"]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8,
               bbox_to_anchor=(0.5, -0.02))

    pct_str = str(pct) if pct != int(pct) else str(int(pct))
    fig.suptitle("ApTt LFF vs Additive Prediction (TTau + AppP)\n"
                 "Concordance with additive model — LFF is enrichment score, not raw signal",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))

    path = os.path.join(output_dir, f"scatter_pct{pct_str}.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Scatter plot saved: {path}")


# ── Visualization 2: UpSet Plot ──────────────────────────────────────────────

def plot_upset(triplets, pct, output_dir):
    """UpSet-style bar chart of TTau/AppP/ApTt significant kinase intersections."""
    for gender_label in ["M"]:
        gender_triplets = [(ctx, df) for ctx, df in triplets
                           if config.GENDER_MAP.get(ctx["gender"], ctx["gender"]) == gender_label]
        if not gender_triplets:
            continue

        cat_counts = pd.Series(dtype=int)
        for _, df in gender_triplets:
            cats = classify_kinase_sets(df)
            cat_counts = cat_counts.add(cats.value_counts(), fill_value=0)

        cat_counts = cat_counts.drop("none", errors="ignore").sort_values(ascending=False)

        if cat_counts.sum() == 0:
            continue

        fig, ax = plt.subplots(figsize=(10, 5))
        colors = {
            "Ttau_only": "#1f77b4", "AppP_only": "#ff7f0e",
            "ApTt_only_emergent": "#d62728", "Ttau+AppP_missing_ApTt": "#9467bd",
            "Ttau+ApTt": "#2ca02c", "AppP+ApTt": "#8c564b", "all_three": "#e377c2",
        }
        bar_colors = [colors.get(str(c), "#999999") for c in cat_counts.index]
        ax.bar(range(len(cat_counts)), cat_counts.values, color=bar_colors)
        ax.set_xticks(range(len(cat_counts)))
        ax.set_xticklabels([str(c).replace("_", " ") for c in cat_counts.index],
                           rotation=35, ha="right", fontsize=9)
        ax.set_ylabel("Count (kinase x context instances)")
        ax.set_title(f"Significant Kinase Set Intersections — {gender_label}")
        fig.tight_layout()

        pct_str = str(pct) if pct != int(pct) else str(int(pct))
        path = os.path.join(output_dir, f"upset_{gender_label}_pct{pct_str}.png")
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  UpSet plot saved: {path}")


# ── Visualization 3: Emergence Heatmap ───────────────────────────────────────

def plot_emergence_heatmap(triplets, pct, output_dir):
    """Heatmap: rows = clusters, cols = timepoint, cell = count of emergent kinases."""
    import seaborn as sns

    rows = []
    for ctx, df in triplets:
        n_emergent = (df["sig_ApTt"] & ~df["sig_Ttau"] & ~df["sig_AppP"]).sum()
        g_label = config.GENDER_MAP.get(ctx["gender"], ctx["gender"])
        rows.append({
            "cluster": ctx["cluster"] or "bulk",
            "col": f"{g_label}_{ctx['timepoint']}",
            "n_emergent": n_emergent,
        })

    if not rows:
        return

    pivot = pd.DataFrame(rows).pivot_table(
        index="cluster", columns="col", values="n_emergent", aggfunc="sum", fill_value=0)

    col_order = [f"{g}_{tp}" for g in ["M"] for tp in ["2mo", "4mo", "6mo"]]
    pivot = pivot.reindex(columns=[c for c in col_order if c in pivot.columns])

    fig, ax = plt.subplots(figsize=(10, max(4, len(pivot) * 0.5)))
    sns.heatmap(pivot, annot=True, fmt="d", cmap="YlOrRd", ax=ax,
                linewidths=0.5, cbar_kws={"label": "# emergent kinases"})
    ax.set_title("Emergent (ApTt-only) Kinases per Cluster and Condition")
    ax.set_ylabel("")
    fig.tight_layout()

    pct_str = str(pct) if pct != int(pct) else str(int(pct))
    path = os.path.join(output_dir, f"emergence_heatmap_pct{pct_str}.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Emergence heatmap saved: {path}")


# ── Visualization 4: Residual Distribution ──────────────────────────────────

def plot_residual_distribution(triplets, pct, output_dir):
    """Histogram of lff_residual across all kinases and contexts."""
    all_residuals = []
    for _, df in triplets:
        all_residuals.extend(df["lff_residual"].dropna().tolist())

    if not all_residuals:
        return

    residuals = np.array(all_residuals)

    fig, ax1 = plt.subplots(1, 1, figsize=(8, 5))

    ax1.hist(residuals, bins=80, color="#5b8fba", edgecolor="white", alpha=0.8)
    ax1.axvline(0, color="red", ls="--", lw=1.5, label="Additive (residual=0)")
    ax1.axvline(residuals.mean(), color="orange", ls="-", lw=1.5,
                label=f"Mean = {residuals.mean():.3f}")
    ax1.set_xlabel("LFF Residual (ApTt - [TTau + AppP])")
    ax1.set_ylabel("Count")
    ax1.set_title("Residual Distribution")
    ax1.legend(fontsize=9)

    t_stat, t_pval = stats.ttest_1samp(residuals, 0)
    ax1.text(0.95, 0.95,
             f"n = {len(residuals)}\nmean = {residuals.mean():.4f}\n"
             f"t = {t_stat:.2f}, p = {t_pval:.2e}",
             transform=ax1.transAxes, va="top", ha="right", fontsize=9,
             bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    fig.suptitle("LFF Additivity Residual: ApTt - (TTau + AppP)\n"
                 "Positive = synergistic, Negative = sub-additive", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.92))

    pct_str = str(pct) if pct != int(pct) else str(int(pct))
    path = os.path.join(output_dir, f"residual_distribution_pct{pct_str}.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Residual distribution saved: {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="ApTt additivity analysis")
    parser.add_argument("--pct", type=float, default=config.PERCENT_THRESH,
                        help=f"Percentile threshold (default: {config.PERCENT_THRESH})")
    parser.add_argument("--bulk", action="store_true",
                        help="Use bulk (non-deconvoluted) data")
    args = parser.parse_args()

    mode = "bulk" if args.bulk else "deconv"
    enrichment_dir = os.path.join("outputs", mode, "enrichment_results")
    output_dir = os.path.join("outputs", mode, "additivity")
    os.makedirs(output_dir, exist_ok=True)

    pct = args.pct

    print(f"=== ApTt Additivity Analysis ===")
    print(f"Mode: {mode}, Percentile: {pct}")
    print(f"Reading from: {enrichment_dir}")
    print(f"Writing to:   {output_dir}\n")

    # Load all triplets
    print("Loading condition triplets...")
    triplets = load_all_triplets(pct, enrichment_dir, bulk=args.bulk)
    print(f"  Loaded {len(triplets)} complete triplets.\n")

    if not triplets:
        print("ERROR: No complete triplets found. Check enrichment_dir and pct value.")
        sys.exit(1)

    # Core analyses
    print("1. Set overlap analysis...")
    compute_set_overlap(triplets, output_dir)

    print("2. LFF additivity correlations...")
    compute_additivity_correlations(triplets, output_dir)

    print("3. Emergent kinase identification...")
    compute_emergent_kinases(triplets, output_dir)

    print("4. Direction concordance...")
    compute_direction_concordance(triplets, output_dir)

    print("\n5. Signal concentration (diffuse-signal hypothesis)...")
    conc_summary, lorenz_data = compute_signal_concentration(
        pct, enrichment_dir, output_dir, bulk=args.bulk)

    # Confound diagnostic
    print("\n6. Confound: percentile ceiling effect...")
    diagnose_percentile_ceiling(triplets, output_dir)

    # Visualizations
    print("\n7. Additivity scatter plot...")
    plot_additivity_scatter(triplets, pct, output_dir)

    print("8. UpSet plots...")
    plot_upset(triplets, pct, output_dir)

    print("9. Emergence heatmap...")
    plot_emergence_heatmap(triplets, pct, output_dir)

    print("10. Residual distribution...")
    plot_residual_distribution(triplets, pct, output_dir)

    print("\n11. Signal concentration plot...")
    plot_signal_concentration(conc_summary, lorenz_data, pct, output_dir)

    print("\n12. Confound comparison panel...")
    plot_confound_panel(triplets, pct, output_dir)

    print(f"\nDone. All outputs in {output_dir}/")


if __name__ == "__main__":
    main()

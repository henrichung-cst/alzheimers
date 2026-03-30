"""
Orchestration script for kinase enrichment analysis on deconvoluted phosphoproteomics data.

Workflow:
1. Prepare: Get all available kinases, map them, and filter by brain expression.
2. Enrich: Run Kinase Library enrichment for all comparisons (TARGETED to expressed kinases).
3. Summarize: Aggregate enrichment results into summary tables.
4. Plot: Generate all visualizations (volcano, bubble, heatmap, direction, UpSet, scatter).
"""

import gc
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Monkeypatch pandas to ignore 'future.no_silent_downcasting' option
# which is missing in pandas < 2.2.0 but requested by kinase_library
try:
    pd.set_option('future.no_silent_downcasting', True)
except pd.errors.OptionError:
    orig_set_option = pd.set_option
    def patched_set_option(key, value):
        if key == 'future.no_silent_downcasting':
            return
        return orig_set_option(key, value)
    pd.set_option = patched_set_option

import argparse

import config
import kinase_library as kl
from analysis_utils import (
    annotate_kinase_expression,
    assign_evidence_tier,
    calculate_log2_fold_change,
    extract_substrate_lfc_stats,
    get_expression_cache,
    get_mapping_cache,
    resolve_kinase_symbol,
    save_expression_cache,
    save_mapping_cache,
)
from plotting_utils import (
    compute_kinase_rankings,
    plot_bubblemap,
    plot_direction_over_time,
    plot_summary_heatmap,
)


def short_label(gender, condition, timepoint, gender_map=None):
    """Return a compact label like 'F ApTt 2mo' from comparison fields."""
    if gender_map is None:
        gender_map = config.GENDER_MAP
    g = gender_map.get(gender, gender)
    return f"{g} {condition} {timepoint}"


def _comparison_title(bg, fg, comp):
    """Build a canonical filename-safe title for a comparison."""
    pct = comp.get("percent_thresh", config.PERCENT_THRESH)
    return f"{bg}_vs_{fg}_pct{pct}_lff{config.LFF_THRESH}".replace("/", "_")


def get_comparisons(test_mode=False, input_file=None):
    """Generates the list of comparisons to process.

    Uses percentile-based site classification (single threshold) instead
    of iterating over multiple fixed LFC thresholds.
    """
    top_clusters = (
        pd.read_csv(config.MEDIAN_CLUSTER_SIZES_FILE, index_col=0)
        .sort_values(by="median_cluster_size", ascending=False)
        .head(10).index.tolist()
    )

    # Exclude clusters with zero-variance source data (e.g. all zeros from deconvolution).
    # Read a small sample to check variance without loading the full 259MB file.
    if input_file is None:
        input_file = config.get_input_file()
    if os.path.exists(input_file):
        sample = pd.read_csv(input_file, nrows=500, low_memory=False)
        before = len(top_clusters)
        valid_clusters = []
        for c in top_clusters:
            cols = [col for col in sample.columns if col.endswith(f"_{c}")]
            if cols and sample[cols].std().sum() > 0:
                valid_clusters.append(c)
            elif not cols:
                valid_clusters.append(c)  # keep if columns not found (shouldn't happen)
        dropped = before - len(valid_clusters)
        if dropped:
            print(f"  Excluded {dropped} cluster(s) with zero-variance source data: "
                  f"{set(top_clusters) - set(valid_clusters)}")
        top_clusters = valid_clusters

    genders = ["ma"]
    conditions = ["Ttau", "AppP", "ApTt"]
    timepoints = ["2mo", "4mo", "6mo"]

    if test_mode:
        print("Running in TEST mode: limiting scope to 1 cluster and 1 condition.")
        top_clusters = [top_clusters[0]]
        conditions = ["AppP"]
        timepoints = ["2mo"]

    comparisons = []
    for gender in genders:
        for condition in conditions:
            for timepoint in timepoints:
                for cluster in top_clusters:
                    bg = f"{gender}_{timepoint}_WTyp_{cluster}"
                    fg = f"{gender}_{timepoint}_{condition}_{cluster}"
                    comparisons.append({
                        "bg": bg,
                        "fg": fg,
                        "percent_thresh": config.PERCENT_THRESH,
                        "cluster": cluster,
                        "gender": gender,
                        "condition": condition,
                        "timepoint": timepoint
                    })
    return comparisons

def run_prepare_step(kin_type=None):
    """Step 1: Get all available kinases, map them, and filter by expression."""
    if kin_type is None:
        kin_type = config.KIN_TYPE
    print("Step 1: Preparing Expressed Kinases List...")

    # Get all kinases from the library based on config
    all_kinases = kl.modules.data.get_kinase_list(kin_type=kin_type)
    print(f"Found {len(all_kinases)} total kinases in the library for kin_type='{kin_type}'")

    # Map to gene symbols
    mapping_cache = get_mapping_cache(config.MAPPING_CACHE_FILE)
    new_mappings = 0
    for kin in all_kinases:
        if kin not in mapping_cache:
            resolve_kinase_symbol(kin, mapping_cache)
            new_mappings += 1
    save_mapping_cache(mapping_cache, config.MAPPING_CACHE_FILE)
    print(f"Mapping complete. Total mappings: {len(mapping_cache)} ({new_mappings} new).")

    # Annotate expression (no filtering — all kinases retained)
    exp_cache = get_expression_cache(config.ALLEN_EXPRESSION_CACHE_FILE)
    all_kinases, kinase_expression_info = annotate_kinase_expression(
        all_kinases, mapping_cache, exp_cache, config.ORGANISM
    )
    save_expression_cache(exp_cache, config.ALLEN_EXPRESSION_CACHE_FILE)

    num_expressed = sum(1 for v in kinase_expression_info.values() if v["brain_expressed"] == 1)
    print(f"Expression annotation complete. {num_expressed} / {len(all_kinases)} "
          f"kinases confirmed expressed in {config.ORGANISM} brain.")
    return all_kinases, kinase_expression_info

def run_enrichment_step(comparisons, target_kinases=None, kinase_expression_info=None,
                        input_file=None, output_dir="outputs/deconv", kin_type=None):
    """Step 2: Run targeted kinase enrichment for all comparisons.

    Uses percentile-based site classification (percent_rank='logFC',
    percent_thresh from comparison dict) for adaptive thresholding.
    After enrichment, extracts per-kinase substrate LFC statistics
    for continuous evidence strength scoring.
    """
    if kin_type is None:
        kin_type = config.KIN_TYPE
    if input_file is None:
        input_file = config.get_input_file(kin_type)
    print("Step 2: Running Targeted Kinase Enrichment (percentile mode)...")
    os.makedirs(os.path.join(output_dir, "enrichment_results"), exist_ok=True)

    df = pd.read_csv(input_file)
    count = 0

    for comp in comparisons:
        if config.MAX_COMPARISONS is not None and count >= config.MAX_COMPARISONS:
            print(f"Reached MAX_COMPARISONS limit ({config.MAX_COMPARISONS}). Stopping.")
            break

        bg, fg = comp["bg"], comp["fg"]
        pct = comp.get("percent_thresh", config.PERCENT_THRESH)
        title = _comparison_title(bg, fg, comp)

        existing_results_path = os.path.join(output_dir, "enrichment_results", f"{title}_enrichment_results.csv")
        if os.path.exists(existing_results_path):
            print(f"Skipping {title} (results already exist).")
            continue

        print(f"Processing: {title}")
        try:
            work_df = df[[bg, fg, "motif"]].copy()
            work_df['log2_fold_change'] = calculate_log2_fold_change(work_df, fg, bg)

            # Foreground quality gate: skip comparisons where the extreme
            # sites have negligible fold change (percentile method on noise).
            lfc_vals = work_df['log2_fold_change'].replace([np.inf, -np.inf], np.nan).dropna()
            p95_abs_lfc = np.percentile(lfc_vals.abs(), 95)
            if p95_abs_lfc < config.MIN_FOREGROUND_LFC:
                print(f"  SKIPPED: 95th %ile |LFC| = {p95_abs_lfc:.4f} "
                      f"< {config.MIN_FOREGROUND_LFC} (foreground quality gate)")
                continue

            dpd = kl.DiffPhosData(
                work_df, seq_col='motif', lfc_col='log2_fold_change',
                percent_rank=config.PERCENT_RANK, percent_thresh=pct,
            )
            enrich_data = dpd.kinase_enrichment(
                kin_type=kin_type,
                kl_method=config.KL_METHOD,
                kl_thresh=config.KL_THRESH,
            )

            # Build results dataframe
            result_cols = [
                'most_sig_direction', 'most_sig_log2_freq_factor',
                'most_sig_fisher_pval', 'most_sig_fisher_adj_pval',
            ]
            results_df = enrich_data.combined_enrichment_results[result_cols].copy()

            # Always preserve the library's BH-corrected p-values
            results_df['most_sig_fisher_bh_pval'] = results_df['most_sig_fisher_adj_pval'].copy()

            # Apply configured correction method
            if config.CORRECTION_METHOD == "permutation":
                from permutation_correction import run_permutation_correction
                empirical_p = run_permutation_correction(
                    work_df, results_df['most_sig_fisher_pval'],
                    kin_type=kin_type, pct=pct,
                    n_perm=config.N_PERMUTATIONS,
                    seed=config.PERMUTATION_SEED,
                    n_workers=config.N_WORKERS,
                )
                results_df['most_sig_fisher_perm_pval'] = empirical_p
                results_df['most_sig_fisher_adj_pval'] = empirical_p

            elif config.CORRECTION_METHOD == "meff_bh":
                from permutation_correction import compute_meff, apply_meff_bh
                m_eff = compute_meff(kin_type, config.KL_METHOD, config.KL_THRESH)
                meff_adj = apply_meff_bh(results_df['most_sig_fisher_pval'], m_eff)
                results_df['most_sig_fisher_meff_pval'] = meff_adj
                results_df['most_sig_fisher_adj_pval'] = meff_adj

            # else "bh": keep kinase-library's BH as-is

            # Extract substrate LFC stats for significant kinases
            results_df = extract_substrate_lfc_stats(
                enrich_data, results_df,
                lff_thresh=config.LFF_THRESH, pval_thresh=config.PVAL_SIG,
            )

            # Assign evidence tiers
            results_df['evidence_tier'] = results_df['median_substrate_lfc'].apply(
                lambda x: assign_evidence_tier(x, config.SUBSTRATE_TIER_BOUNDARIES)
            )

            # Add expression annotations
            if kinase_expression_info is not None:
                results_df['brain_expressed'] = results_df.index.map(
                    lambda k: kinase_expression_info.get(k, {}).get("brain_expressed", 0)
                )
                results_df['num_allen_experiments'] = results_df.index.map(
                    lambda k: kinase_expression_info.get(k, {}).get("num_experiments", 0)
                )

            enrichment_results_path = os.path.join(output_dir, "enrichment_results", f"{title}_enrichment_results.csv")
            results_df.to_csv(enrichment_results_path)

            # Free memory
            del dpd, enrich_data, work_df, results_df
            if count % 20 == 0:
                gc.collect()

            count += 1
        except KeyError:
            print(f"Skipping {bg} vs {fg}: missing data columns.")
            continue
        except Exception as e:
            print(f"Error processing {bg} vs {fg}: {type(e).__name__}: {e}")
            continue

def run_summarize_step(comparisons, kinase_expression_info=None, output_dir="outputs/deconv",
                       gender_map=None):
    """Step 3: Aggregate enrichment results into summary tables."""
    if gender_map is None:
        gender_map = config.GENDER_MAP
    print("Step 3: Generating Summary Tables...")
    summaries = []
    all_kinase_records = []  # For kinase_results.csv (ALL kinases × comparison)

    for comp in comparisons:
        bg, fg = comp["bg"], comp["fg"]
        title = _comparison_title(bg, fg, comp)

        results_path = os.path.join(output_dir, "enrichment_results", f"{title}_enrichment_results.csv")
        if not os.path.exists(results_path):
            continue

        res = pd.read_csv(results_path, index_col=0)

        # Sort and take top N if configured
        res_sorted = res.sort_values(by="most_sig_log2_freq_factor", key=abs, ascending=False)
        if config.KEPT_RANKS is not None:
            res_kept = res_sorted.head(config.KEPT_RANKS)
        else:
            res_kept = res_sorted

        sig_mask = (abs(res_kept["most_sig_log2_freq_factor"]) >= config.LFF_THRESH) & \
                   (res_kept["most_sig_fisher_adj_pval"] <= config.PVAL_SIG)
        num_sig = sig_mask.sum()

        gender_short = gender_map.get(comp["gender"], comp["gender"])
        pct = comp.get("percent_thresh", config.PERCENT_THRESH)

        # Comparison-level summary (one row per comparison)
        summaries.append({
            "comparison": title,
            "gender": gender_short,
            "condition": comp["condition"],
            "timepoint": comp["timepoint"],
            "cluster": comp["cluster"],
            "percent_thresh": pct,
            "num_sig_kinases": num_sig,
            "num_expressed_kinases": len(res_kept),
            "mean_lff_magnitude": res_kept["most_sig_log2_freq_factor"].abs().mean(),
        })

        # ALL kinase × comparison records with significance tier
        # Compute percentile tails for display tier
        lff_series = res_kept["most_sig_log2_freq_factor"].dropna()
        if len(lff_series) > 0:
            pct_upper = np.percentile(lff_series, 100 - config.BUBBLE_PERCENTILE)
            pct_lower = np.percentile(lff_series, config.BUBBLE_PERCENTILE)
        else:
            pct_upper, pct_lower = np.inf, -np.inf

        for kinase in res_kept.index:
            row = res_kept.loc[kinase]
            lff_val = row["most_sig_log2_freq_factor"]
            adj_pval = row["most_sig_fisher_adj_pval"]

            # Tiered significance:
            #   "significant"     — |LFF| >= LFF_THRESH and adj_pval <= PVAL_SIG
            #   "display"         — in top/bottom BUBBLE_PERCENTILE% of LFF
            #                       AND adj_pval < PVAL_DISPLAY (shown in viz, not stat. sig.)
            #   "non_significant" — everything else (excluded from viz)
            in_percentile_tail = (lff_val >= pct_upper) or (lff_val <= pct_lower)
            if abs(lff_val) >= config.LFF_THRESH and adj_pval <= config.PVAL_SIG:
                tier = "significant"
            elif in_percentile_tail and adj_pval < config.PVAL_DISPLAY:
                tier = "display"
            else:
                tier = "non_significant"

            record = {
                "kinase": kinase,
                "gender": gender_short,
                "condition": comp["condition"],
                "timepoint": comp["timepoint"],
                "cluster": comp["cluster"],
                "percent_thresh": pct,
                "direction": row["most_sig_direction"],
                "lff": lff_val,
                "pval": row["most_sig_fisher_pval"],
                "adj_pval": adj_pval,
                "significance_tier": tier,
                "brain_expressed": int(row.get("brain_expressed", 0)),
                "num_allen_experiments": row.get("num_allen_experiments", None),
            }
            # Include substrate evidence columns if present
            for sub_col in ["n_substrates", "median_substrate_lfc", "q75_substrate_lfc",
                            "max_substrate_lfc", "evidence_tier"]:
                if sub_col in res_kept.columns:
                    record[sub_col] = row.get(sub_col, None)
            all_kinase_records.append(record)

    # Save comparison-level summary CSV
    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(os.path.join(output_dir, "enrichment_summary.csv"), index=False)
    print(f"Comparison summary: {output_dir}/enrichment_summary.csv ({len(summary_df)} comparisons)")

    # Save kinase results table (ALL kinases × comparison, with significance tier)
    results_df = None
    if all_kinase_records:
        results_df = pd.DataFrame(all_kinase_records)
        results_df = results_df.sort_values(["cluster", "kinase", "timepoint", "gender", "condition"])
        results_df.to_csv(os.path.join(output_dir, "kinase_results.csv"), index=False)
        n_sig = (results_df["significance_tier"] == "significant").sum()
        n_display = (results_df["significance_tier"] == "display").sum()
        n_ns = (results_df["significance_tier"] == "non_significant").sum()
        print(f"Kinase results: {output_dir}/kinase_results.csv "
              f"({len(results_df)} total: {n_sig} significant, {n_display} display, {n_ns} non-significant)")

        # Generate per-kinase aggregated summary CSV (based on significant tier only)
        sig_only = results_df[results_df["significance_tier"] == "significant"]
        if not sig_only.empty:
            agg_dict = {
                "num_comparisons_significant": ("kinase", "size"),
                "mean_lff_when_significant": ("lff", lambda x: x.abs().mean()),
                "predominant_direction": ("direction", lambda x: x.mode().iloc[0] if not x.mode().empty else "unknown"),
                "clusters_significant_in": ("cluster", lambda x: ", ".join(sorted(set(x)))),
            }
            if "median_substrate_lfc" in sig_only.columns:
                agg_dict["mean_median_substrate_lfc"] = ("median_substrate_lfc", lambda x: x.dropna().mean())
                agg_dict["predominant_tier"] = (
                    "evidence_tier",
                    lambda x: x.dropna().mode().iloc[0] if not x.dropna().mode().empty else "unknown",
                )

            kinase_summary = sig_only.groupby("kinase").agg(**agg_dict).reset_index()

            if kinase_expression_info:
                kinase_summary["brain_expressed"] = kinase_summary["kinase"].map(
                    lambda k: kinase_expression_info.get(k, {}).get("brain_expressed", 0)
                )
                kinase_summary["num_allen_experiments"] = kinase_summary["kinase"].map(
                    lambda k: kinase_expression_info.get(k, {}).get("num_experiments", 0)
                )
            kinase_summary = kinase_summary.sort_values("num_comparisons_significant", ascending=False)
            kinase_summary.to_csv(os.path.join(output_dir, "kinase_summary.csv"), index=False)
            print(f"Kinase summary: {output_dir}/kinase_summary.csv ({len(kinase_summary)} kinases)")

def _iter_enrichment_results(comparisons, output_dir="outputs/deconv"):
    """Yield (comp, res_kept) for each comparison with existing results.

    Loads each enrichment CSV, sorts by absolute LFF, and keeps top N if configured.
    """
    for comp in comparisons:
        bg, fg = comp["bg"], comp["fg"]
        title = _comparison_title(bg, fg, comp)

        results_path = os.path.join(output_dir, "enrichment_results", f"{title}_enrichment_results.csv")
        if not os.path.exists(results_path):
            continue

        res = pd.read_csv(results_path, index_col=0)

        # Sort and take top N if configured
        res_sorted = res.sort_values(by="most_sig_log2_freq_factor", key=abs, ascending=False)
        if config.KEPT_RANKS is not None:
            res_kept = res_sorted.head(config.KEPT_RANKS)
        else:
            res_kept = res_sorted

        yield comp, res_kept


def run_plot_step(comparisons, kinase_expression_info=None, volcano=False,
                  output_dir="outputs/deconv", condition_colors=None, gender_map=None):
    """Step 4: Generate all visualizations from enrichment results and summary CSVs."""
    if condition_colors is None:
        condition_colors = config.CONDITION_COLORS
    if gender_map is None:
        gender_map = config.GENDER_MAP
    print("Step 4: Generating Visualizations...")

    pct = config.PERCENT_THRESH
    clusters = sorted(set(c["cluster"] for c in comparisons))

    # --- Part A (optional): Volcano plots ---
    if volcano:
        from kinase_library.modules.enrichment import plot_volcano as kl_plot_volcano
        os.makedirs(os.path.join(output_dir, "volcano_plots"), exist_ok=True)
        print("\n--- Volcano plots ---")
        for comp, res_kept in _iter_enrichment_results(comparisons, output_dir=output_dir):
            bg, fg = comp["bg"], comp["fg"]
            title = _comparison_title(bg, fg, comp)
            volcano_path = os.path.join(output_dir, "volcano_plots", f"{title}_volcano.png")

            try:
                kl_plot_volcano(
                    res_kept,
                    lff_col='most_sig_log2_freq_factor',
                    pval_col='most_sig_fisher_adj_pval',
                    sig_lff=config.LFF_THRESH,
                    sig_pval=config.PVAL_SIG,
                    title=title,
                    save_fig=volcano_path,
                )
                plt.close('all')
            except Exception as e:
                print(f"  Warning: volcano plot failed for {title}: {type(e).__name__}: {e}")
                plt.close('all')
        print("Volcano plots complete.")

    # --- Part B: Bubble maps and other visualizations ---
    # Load summary CSVs from disk (file-based handoff from summarize step)
    summary_path = os.path.join(output_dir, "enrichment_summary.csv")
    results_path = os.path.join(output_dir, "kinase_results.csv")

    summary_df = pd.read_csv(summary_path) if os.path.exists(summary_path) else pd.DataFrame()
    results_df = pd.read_csv(results_path) if os.path.exists(results_path) else None
    # Direction charts use only statistically significant kinases
    sig_kins_df = (
        results_df[results_df["significance_tier"] == "significant"].copy()
        if results_df is not None and "significance_tier" in results_df.columns
        else None
    )

    # Build group_data for bubble maps
    group_data = {}  # key: (cluster, percent_thresh, gender) -> list of (comp, res_kept, label)
    for comp, res_kept in _iter_enrichment_results(comparisons, output_dir=output_dir):
        gender_short = gender_map.get(comp["gender"], comp["gender"])
        bubble_label = f"{comp['condition']} {comp['timepoint']}"
        group_key = (comp["cluster"], comp.get("percent_thresh", config.PERCENT_THRESH), gender_short)
        if group_key not in group_data:
            group_data[group_key] = []
        group_data[group_key].append((comp, res_kept, bubble_label))

    # Define condition ordering
    timepoint_order = ["2mo", "4mo", "6mo"]
    condition_order_list = list(condition_colors.keys())

    cond_colors_map = {}
    for cond in condition_order_list:
        for tp in timepoint_order:
            label = f"{cond} {tp}"
            cond_colors_map[label] = condition_colors.get(cond, "black")

    canonical_cond_order = []
    for cond in condition_order_list:
        for tp in timepoint_order:
            canonical_cond_order.append(f"{cond} {tp}")

    os.makedirs(os.path.join(output_dir, "bubble_maps"), exist_ok=True)

    def condition_group(label):
        """Group by condition (first token in label like 'AppP 2mo')."""
        return label.split()[0]

    def timepoint_group(label):
        """Group by condition+timepoint (the full label is already condition+timepoint)."""
        return label

    def _build_group_dataframes(group_items, kinase_expression_info):
        """Build lff, pval, and brain_expressed DataFrames from a list of (comp, res_kept, label) tuples."""
        group_lff = pd.DataFrame()
        group_pval = pd.DataFrame()
        group_brain = pd.DataFrame() if kinase_expression_info else None

        for comp, res_kept, label in group_items:
            lff_col = res_kept["most_sig_log2_freq_factor"].rename(label)
            pval_col = res_kept["most_sig_fisher_adj_pval"].rename(label)

            if group_lff.empty:
                group_lff = pd.DataFrame(lff_col)
                group_pval = pd.DataFrame(pval_col)
            else:
                group_lff = group_lff.join(lff_col, how="outer")
                group_pval = group_pval.join(pval_col, how="outer")

            if kinase_expression_info is not None:
                brain_col = res_kept.index.to_series().map(
                    lambda k: bool(kinase_expression_info.get(k, {}).get("brain_expressed", 0))
                ).rename(label)
                brain_col.index = res_kept.index
                if group_brain.empty:
                    group_brain = pd.DataFrame(brain_col)
                else:
                    group_brain = group_brain.join(brain_col, how="outer")

        return group_lff, group_pval, group_brain

    def _get_displayed_kinases(results_df, cluster, pct_thresh):
        """Return set of kinases with tier 'significant' or 'display' for a cluster."""
        if results_df is None or results_df.empty:
            return set()
        mask = (
            (results_df["cluster"] == cluster)
            & (results_df["percent_thresh"] == pct_thresh)
            & (results_df["significance_tier"].isin(["significant", "display"]))
        )
        return set(results_df.loc[mask, "kinase"])

    def _generate_bubble(group_lff, group_pval, group_brain, cond_order,
                         save_path, title_str, cond_colors_map,
                         displayed_kinase_set=None,
                         sep_fn=None, minor_sep_fn=None):
        """Generate a single bubble map with given data and save it."""
        if group_lff.empty or not cond_order:
            return

        group_lff = group_lff[cond_order]
        group_pval = group_pval[cond_order]
        if group_brain is not None and not group_brain.empty:
            group_brain = group_brain.reindex(
                index=group_lff.index, columns=cond_order
            ).fillna(False).astype(bool)
        else:
            group_brain = None

        # Use precomputed tier from kinase_results.csv:
        # show kinases that are "significant" or "display" in any comparison.
        if displayed_kinase_set is not None:
            notable_any = group_lff.index.isin(displayed_kinase_set)
        else:
            # Fallback: show all kinases (should not happen in normal flow)
            notable_any = pd.Series(True, index=group_lff.index)
        n_display_kins = notable_any.sum()
        if n_display_kins == 0:
            print(f"  No percentile-tail or significant kinases — skipping {title_str}")
            return

        n_conds = len(cond_order)
        row_cap_per_image = 60
        displayed_kinases = list(group_lff.index[notable_any])

        # If too dense, split into two pages for readability.
        if n_display_kins > row_cap_per_image:
            split_sets = [list(x) for x in np.array_split(displayed_kinases, 2)]
            targets = []
            for i, kin_set in enumerate(split_sets, start=1):
                if not kin_set:
                    continue
                part_path = save_path.replace(".png", f"_part{i}.png")
                part_title = f"{title_str} (Part {i}/2)"
                targets.append((part_path, part_title, kin_set))
        else:
            targets = [(save_path, title_str, displayed_kinases)]

        for target_path, target_title, kin_subset in targets:
            sub_lff = group_lff.loc[kin_subset]
            sub_pval = group_pval.loc[kin_subset]
            sub_brain = group_brain.loc[kin_subset] if group_brain is not None else None

            n_sub_kins = len(kin_subset)
            num_panels = max(1, min(4, n_sub_kins // 30 + 1))
            kins_per_panel = n_sub_kins / num_panels
            fig_width = max(14, n_conds * 0.8)
            fig_height = max(6, kins_per_panel * 0.35 + 4)

            print(f"  Generating bubble map: {target_path} ({n_sub_kins} kinases, {n_conds} conditions)")
            try:
                plot_bubblemap(
                    sub_lff,
                    sub_pval,
                    brain_expressed_data=sub_brain,
                    title=target_title,
                    save_fig=target_path,
                    figsize=(fig_width, fig_height),
                    sig_lff=config.LFF_THRESH,
                    sig_pval=config.PVAL_SIG,
                    lff_percentile=config.BUBBLE_PERCENTILE,
                    cond_order=cond_order,
                    cond_colors=cond_colors_map,
                    num_panels=num_panels,
                    xlabels_size=9,
                    ylabels_size=9,
                    legend_position="bottom",
                    kin_clust=True,
                    cluster_by="pattern",
                    lff_color_bins=5,
                    cond_separator_fn=sep_fn,
                    cond_minor_separator_fn=minor_sep_fn,
                )
                plt.close('all')
            except Exception as e:
                print(f"  Warning: bubble map failed for {target_title}: {type(e).__name__}: {e}")
                plt.close('all')

    # === Per-cluster bubble maps (9 conditions each) ===
    print("\n--- Per-cluster bubble maps ---")
    for (cluster, pct_thresh, _gender), group_items in group_data.items():
        group_lff, group_pval, group_brain = _build_group_dataframes(
            group_items, kinase_expression_info
        )
        if group_lff.empty:
            continue

        present_labels = set(group_lff.columns)
        cond_order = [c for c in canonical_cond_order if c in present_labels]

        safe_cluster = str(cluster).replace("/", "_")
        save_path = os.path.join(output_dir, "bubble_maps", f"{safe_cluster}_pct{pct_thresh}_bubble.png")
        title_str = (
            f"{cluster} — significant (adj p ≤ {config.PVAL_SIG}) "
            f"+ display (top/bottom {config.BUBBLE_PERCENTILE}% LFF, adj p < {config.PVAL_DISPLAY})"
        )

        display_set = _get_displayed_kinases(results_df, cluster, pct_thresh)
        _generate_bubble(
            group_lff, group_pval, group_brain, cond_order,
            save_path, title_str, cond_colors_map,
            displayed_kinase_set=display_set,
            sep_fn=condition_group, minor_sep_fn=timepoint_group,
        )

    # Generate overview summary heatmap
    if not summary_df.empty:
        print("\nGenerating overview heatmap...")
        heatmap_path = os.path.join(output_dir, "bubble_maps", "overview_sig_kinase_counts.png")
        plot_summary_heatmap(
            summary_df,
            save_fig=heatmap_path,
        )
        print(f"Overview heatmap: {heatmap_path}")

    # Generate sex-stratified direction charts
    if sig_kins_df is not None and not sig_kins_df.empty:
        print("\nBuilding kinase family mapping...")
        kinase_families = {}
        for kinase in sig_kins_df["kinase"].unique():
            try:
                info = kl.get_kinase_info(kinase)
                kinase_families[kinase] = info["FAMILY"]
            except Exception:
                kinase_families[kinase] = "Other"

        print(f"  Mapped {len(kinase_families)} kinases to families")

        major_families = ["AGC", "CAMK", "CMGC", "STE", "TKL"]
        family_panel_order = major_families + ["Other"]
        sig_with_family = sig_kins_df.copy()
        sig_with_family["family"] = sig_with_family["kinase"].map(kinase_families).fillna("Other")
        sig_with_family.loc[~sig_with_family["family"].isin(major_families), "family"] = "Other"

        print("\nGenerating direction charts...")
        save_path = os.path.join(output_dir, "bubble_maps", f"direction_over_time_pct{pct}.png")
        plot_direction_over_time(
            sig_with_family,
            condition_colors=condition_colors,
            title_prefix="Kinase Dysregulation",
            save_fig=save_path,
        )
        print(f"  Aggregate: {save_path}")

        os.makedirs(os.path.join(output_dir, "bubble_maps", "families"), exist_ok=True)
        for family in major_families + ["Other"]:
            family_df = sig_with_family[sig_with_family["family"] == family]
            if family_df.empty:
                continue
            safe_family = family.replace("/", "_")
            save_path = os.path.join(output_dir, "bubble_maps", "families", f"{safe_family}_pct{pct}.png")
            plot_direction_over_time(
                family_df,
                condition_colors=condition_colors,
                title_prefix=f"{family} Family",
                save_fig=save_path,
            )
            print(f"  {family}: {save_path}")

        os.makedirs(os.path.join(output_dir, "bubble_maps", "clusters"), exist_ok=True)
        for cluster in sorted(sig_with_family["cluster"].unique()):
            cluster_df = sig_with_family[sig_with_family["cluster"] == cluster]
            if cluster_df.empty:
                continue
            safe_cluster = str(cluster).replace("/", "_")
            save_path = os.path.join(output_dir, "bubble_maps", "clusters", f"{safe_cluster}_pct{pct}.png")
            plot_direction_over_time(
                cluster_df,
                condition_colors=condition_colors,
                title_prefix=f"{cluster}",
                panel_col="family",
                panel_order=family_panel_order,
                save_fig=save_path,
            )
            print(f"  {cluster}: {save_path}")

    # --- Part D: Kinase ranking tables ---
    print("\n--- Kinase ranking tables ---")
    enrichment_dir = os.path.join(output_dir, "enrichment_results")
    rankings_dir = os.path.join(output_dir, "rankings")
    os.makedirs(rankings_dir, exist_ok=True)

    # Global ranking (all clusters pooled)
    global_ranking = compute_kinase_rankings(
        enrichment_dir, comparisons,
        pval_thresh=config.PVAL_SIG,
        lff_thresh=config.LFF_THRESH,
        kinase_expression_info=kinase_expression_info,
    )
    if not global_ranking.empty:
        save_path = os.path.join(rankings_dir, "global_ranking.csv")
        global_ranking.to_csv(save_path, index=False)
        print(f"  Global ranking: {save_path} ({len(global_ranking)} kinases)")

    # Per-cluster rankings
    for cluster in clusters:
        cluster_ranking = compute_kinase_rankings(
            enrichment_dir, comparisons,
            pval_thresh=config.PVAL_SIG,
            lff_thresh=config.LFF_THRESH,
            kinase_expression_info=kinase_expression_info,
            cluster_filter=cluster,
        )
        if cluster_ranking.empty:
            continue
        safe_cluster = str(cluster).replace("/", "_")
        save_path = os.path.join(rankings_dir, f"{safe_cluster}_ranking.csv")
        cluster_ranking.to_csv(save_path, index=False)
        print(f"  {cluster}: {save_path} ({len(cluster_ranking)} kinases)")


def main():
    parser = argparse.ArgumentParser(description="Targeted Kinase Enrichment Orchestrator")
    parser.add_argument("--step", choices=["prepare", "enrich", "summarize", "plot", "all"],
                        default="all", help="Execution step to run (default: all)")
    parser.add_argument("--test", action="store_true", help="Run in test mode (limited scope)")
    parser.add_argument("--volcano", action="store_true",
                        help="Generate per-comparison volcano plots (slow, 360 plots)")
    parser.add_argument("--kin-type", choices=["ser_thr", "tyrosine"], default="ser_thr",
                        help="Kinase type to analyze (default: ser_thr)")
    parser.add_argument("--percent-thresh", type=float, default=None,
                        help="Override percentile threshold (default: config.PERCENT_THRESH)")
    parser.add_argument("--correction", choices=["bh", "permutation", "meff_bh"],
                        default=None,
                        help="Override multiple-testing correction method")
    args = parser.parse_args()

    input_file = config.get_input_file(args.kin_type)
    comparisons = get_comparisons(args.test, input_file=input_file)
    if args.percent_thresh is not None:
        for comp in comparisons:
            comp["percent_thresh"] = args.percent_thresh
    if args.correction is not None:
        config.CORRECTION_METHOD = args.correction

    # Dynamic output directory based on kinase type
    output_dir = "outputs/deconv" if args.kin_type == "ser_thr" else "outputs/deconv_tyrosine"

    all_kinases = None
    kinase_expression_info = None
    if args.step in ["prepare", "enrich", "summarize", "plot", "all"]:
        all_kinases, kinase_expression_info = run_prepare_step(kin_type=args.kin_type)

    if args.step in ["enrich", "all"]:
        run_enrichment_step(comparisons, kinase_expression_info=kinase_expression_info,
                           input_file=input_file, output_dir=output_dir,
                           kin_type=args.kin_type)

    if args.step in ["summarize", "all"]:
        run_summarize_step(comparisons, kinase_expression_info=kinase_expression_info,
                          output_dir=output_dir)

    if args.step in ["plot", "all"]:
        run_plot_step(comparisons, kinase_expression_info=kinase_expression_info,
                     volcano=args.volcano, output_dir=output_dir)

if __name__ == "__main__":
    main()

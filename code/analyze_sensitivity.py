"""Sensitivity analysis: test robustness of headline claims across threshold combinations.

Varies three axes:
  - Percentile threshold (site selection): pct2.5, pct5, pct10
  - Adjusted p-value: 0.05, 0.10
  - LFF floor: 0.01, 0.1, 0.5

Outputs to outputs/{mode}/sensitivity/:
  - threshold_sweep.csv     — hit counts across 18 threshold combos
  - core_claims.csv         — headline claim metrics at each combo
  - threshold_robustness.png — compact 2-panel summary figure
"""

import argparse
import os
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import downstream_utils as du

PCT_VALUES = [2.5, 5, 10]
PVAL_VALUES = [0.05, 0.10]
LFF_VALUES = [0.01, 0.1, 0.5]

PYTHON = sys.executable


def run_alt_enrichment(mode):
    """Run enrichment at pct2.5 and pct10 via subprocess."""
    if "deconv" in mode and "bulk" not in mode:
        script = "code/kl_analysis_clusters.py"
    else:
        script = "code/kl_analysis_bulk.py"

    kin_flag = []
    if "tyrosine" in mode:
        kin_flag = ["--kin-type", "tyrosine"]

    for pct in [2.5, 10]:
        cmd = [PYTHON, script, "--step", "enrich",
               "--percent-thresh", str(pct)] + kin_flag
        print(f"  Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)


def build_threshold_sweep(mode):
    """Build the threshold sweep table: hit counts at each threshold combo."""
    rows = []
    for pct in PCT_VALUES:
        for pval in PVAL_VALUES:
            for lff in LFF_VALUES:
                sig = du.refilter_enrichment(mode, pval, lff, pct_filter=pct)
                if sig.empty:
                    rows.append({
                        "pct": pct, "adj_p": pval, "lff_floor": lff,
                        "total_hits": 0, "unique_kinases": 0,
                        "comparisons_with_hits": 0,
                    })
                    continue
                n_comparisons = sig.groupby(
                    ["condition", "timepoint", "cluster"]
                ).ngroups
                rows.append({
                    "pct": pct, "adj_p": pval, "lff_floor": lff,
                    "total_hits": len(sig),
                    "unique_kinases": sig["kinase"].nunique(),
                    "comparisons_with_hits": n_comparisons,
                })
    return pd.DataFrame(rows)


def build_core_claims(mode):
    """Test headline claims at each threshold combo."""
    rows = []
    for pct in PCT_VALUES:
        for pval in PVAL_VALUES:
            for lff in LFF_VALUES:
                sig = du.refilter_enrichment(mode, pval, lff, pct_filter=pct)

                # Shared 4mo: kinases sig in all 3 conditions at 4mo
                shared_4mo = 0
                if not sig.empty:
                    sig_4mo = sig[sig["timepoint"] == "4mo"]
                    if not sig_4mo.empty:
                        kinase_conds = sig_4mo.groupby("kinase")["condition"].apply(set)
                        shared_4mo = (kinase_conds.apply(
                            lambda s: {"Ttau", "AppP", "ApTt"}.issubset(s)
                        )).sum()

                # AppP reversal: negative at 4mo AND positive at 6mo
                reversal = 0
                pct_down_4mo = None
                pct_up_6mo = None
                if not sig.empty:
                    appp = sig[sig["condition"] == "AppP"]
                    appp_4mo = appp[appp["timepoint"] == "4mo"]
                    appp_6mo = appp[appp["timepoint"] == "6mo"]
                    if not appp_4mo.empty:
                        pct_down_4mo = round(
                            (appp_4mo["direction"] == "-").sum() / len(appp_4mo) * 100, 1
                        )
                    if not appp_6mo.empty:
                        pct_up_6mo = round(
                            (appp_6mo["direction"] == "+").sum() / len(appp_6mo) * 100, 1
                        )
                    neg_4mo = set(appp_4mo[appp_4mo["direction"] == "-"]["kinase"])
                    pos_6mo = set(appp_6mo[appp_6mo["direction"] == "+"]["kinase"])
                    reversal = len(neg_4mo & pos_6mo)

                rows.append({
                    "pct": pct, "adj_p": pval, "lff_floor": lff,
                    "shared_4mo_count": shared_4mo,
                    "reversal_count": reversal,
                    "pct_down_4mo_AppP": pct_down_4mo,
                    "pct_up_6mo_AppP": pct_up_6mo,
                })
    return pd.DataFrame(rows)


def plot_robustness(sweep_df, claims_df, out_path):
    """Generate compact 2-panel robustness figure."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Panel A: total hits grouped bar chart
    # x-axis = pct, groups = adj_p x lff_floor combos
    combos = [(p, l) for p in PVAL_VALUES for l in LFF_VALUES]
    x = np.arange(len(PCT_VALUES))
    width = 0.12
    offset = -(len(combos) - 1) / 2 * width

    for i, (pval, lff) in enumerate(combos):
        vals = []
        for pct in PCT_VALUES:
            row = sweep_df[
                (sweep_df["pct"] == pct) &
                (sweep_df["adj_p"] == pval) &
                (sweep_df["lff_floor"] == lff)
            ]
            vals.append(row["total_hits"].values[0] if len(row) > 0 else 0)
        label = f"p≤{pval}, |LFF|≥{lff}"
        ax1.bar(x + offset + i * width, vals, width, label=label)

    ax1.set_xticks(x)
    ax1.set_xticklabels([f"pct{p}" for p in PCT_VALUES])
    ax1.set_ylabel("Total significant hits")
    ax1.set_title("A. Hit counts across thresholds")
    ax1.legend(fontsize=7, loc="upper right")

    # Panel B: shared 4mo count line plot
    tick_labels = []
    for pct in PCT_VALUES:
        sub = claims_df[claims_df["pct"] == pct].copy()
        sub["stringency"] = sub["adj_p"].astype(str) + " / " + sub["lff_floor"].astype(str)
        sub = sub.sort_values(["adj_p", "lff_floor"])
        ax2.plot(range(len(sub)), sub["shared_4mo_count"].values,
                 marker="o", label=f"pct{pct}")
        if pct == PCT_VALUES[0]:
            tick_labels = list(sub["stringency"].values)

    ax2.set_xticks(range(len(tick_labels)))
    ax2.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=8)
    ax2.set_xlabel("adj_p / |LFF| floor")
    ax2.set_ylabel("Shared 4mo kinases (all 3 conditions)")
    ax2.set_title("B. Core suppression claim stability")
    ax2.legend()

    fig.tight_layout()
    du.save_fig(fig, out_path)


def run_sensitivity(mode):
    """Run the full sensitivity analysis for one mode."""
    # Check if we have enrichment results at all
    results = du.load_all_enrichment_results(mode)
    if du.graceful_empty(results, f"sensitivity/{mode}"):
        return

    # Check which percentiles have CSVs
    available_pcts = set()
    for name in results:
        for pct in PCT_VALUES:
            if f"pct{pct}" in name:
                available_pcts.add(pct)
    missing = set(PCT_VALUES) - available_pcts
    if missing:
        print(f"  Warning: no enrichment CSVs for pct {missing} in {mode}. "
              f"Run with --run-enrichment to generate them.")

    out_dir = du.setup_analysis_dir("sensitivity", mode)

    print(f"\n  Building threshold sweep table...")
    sweep = build_threshold_sweep(mode)
    sweep.to_csv(os.path.join(out_dir, "threshold_sweep.csv"), index=False)
    print(f"    {os.path.join(out_dir, 'threshold_sweep.csv')} ({len(sweep)} rows)")

    print(f"  Building core claims table...")
    claims = build_core_claims(mode)
    claims.to_csv(os.path.join(out_dir, "core_claims.csv"), index=False)
    print(f"    {os.path.join(out_dir, 'core_claims.csv')} ({len(claims)} rows)")

    # Only plot if we have at least 2 pct values with data
    pcts_with_data = len(sweep[sweep["total_hits"] > 0]["pct"].unique())
    if pcts_with_data >= 2:
        print(f"  Generating robustness figure...")
        plot_robustness(sweep, claims, os.path.join(out_dir, "threshold_robustness.png"))
    else:
        print(f"  Skipping figure (only {pcts_with_data} percentile(s) with data)")


def main():
    parser = argparse.ArgumentParser(description="Sensitivity analysis across threshold combinations")
    parser.add_argument("--mode", choices=du.ALL_MODES, default=None,
                        help="Run for a single mode (default: deconv + bulk ser/thr only)")
    parser.add_argument("--run-enrichment", action="store_true",
                        help="Run enrichment at pct2.5 and pct10 before analysis")
    args = parser.parse_args()

    modes = [args.mode] if args.mode else ["deconv", "bulk"]

    if args.run_enrichment:
        print("=== Running enrichment at alternative percentiles ===")
        for mode in modes:
            print(f"\n--- {mode} ---")
            run_alt_enrichment(mode)

    print("\n=== Sensitivity analysis ===")
    for mode in modes:
        print(f"\n--- {mode} ---")
        run_sensitivity(mode)

    print("\nDone.")


if __name__ == "__main__":
    main()

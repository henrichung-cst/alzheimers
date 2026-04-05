#!/usr/bin/env python3
"""Q1: Threshold sensitivity analysis for attribution confidence tiers.

Sweeps WMB specificity and SEA-AD LFC thresholds to show how the confidence
tier distribution changes. Identifies near-miss rows that flip tier under
small threshold perturbations.

Usage:
    python code/supplementary/threshold_sensitivity.py --run
    python code/supplementary/threshold_sensitivity.py --summary
"""

import argparse
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

OUTPUT_DIR = os.path.join(config.SUPPLEMENTARY_OUTPUT_DIR, "threshold_sensitivity")

# Sweep ranges (multiples of 1/N for WMB, absolute for LFC)
WMB_MULTIPLIERS = [1.0, 1.5, 2.0, 2.5, 3.0]
LFC_THRESHOLDS = [0.05, 0.10, 0.15, 0.20, 0.25]


import numpy as np


def _assign_confidence_vectorized(concordance, wmb_spec, abs_lfc,
                                  spec_high, spec_low, lfc_min):
    """Vectorized confidence assignment across arrays."""
    result = np.full(len(concordance), "low", dtype=object)
    result[concordance <= 0] = "none"
    # Apply moderate/high in order so high overwrites moderate where both match
    moderate_mask = (concordance > 0) & ((wmb_spec >= spec_low) | (abs_lfc > lfc_min))
    result[moderate_mask] = "moderate"
    high_mask = (concordance > 0) & (wmb_spec >= spec_high) & (abs_lfc > lfc_min)
    result[high_mask] = "high"
    return result


def _ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def step_run():
    """Run threshold sensitivity sweep."""
    _ensure_output_dir()
    print("\n=== Threshold Sensitivity Analysis ===\n")

    # Load the full (unfiltered) attribution table
    full_path = os.path.join(config.KINASE_ATTRIBUTION_OUTPUT_DIR,
                             "unified_attribution_full.csv")
    if not os.path.exists(full_path):
        raise FileNotFoundError(
            f"{full_path} not found. Run kinase_attribution.py --attribute first.")
    df = pd.read_csv(full_path)
    print(f"  Loaded {len(df)} rows from unified_attribution_full.csv")

    n = config.N_CELL_TYPES  # 24

    # Current defaults
    default_spec_high = config.SPECIFICITY_HIGH  # 2.0/24
    default_spec_low = config.SPECIFICITY_LOW    # 1.0/24
    default_lfc = config.SEA_AD_LFC_MIN          # 0.1

    # Pre-extract arrays once for vectorized sweep
    concordance = df["concordance_score"].values
    wmb_spec = df["wmb_specificity"].fillna(0.0).values if "wmb_specificity" in df.columns else np.zeros(len(df))
    abs_lfc = df["sea_ad_lfc"].fillna(0.0).abs().values if "sea_ad_lfc" in df.columns else np.zeros(len(df))

    # Sweep grid
    grid_rows = []
    for wmb_mult in WMB_MULTIPLIERS:
        spec_high = wmb_mult / n
        spec_low = (wmb_mult / 2.0) / n
        for lfc_min in LFC_THRESHOLDS:
            tiers = _assign_confidence_vectorized(
                concordance, wmb_spec, abs_lfc, spec_high, spec_low, lfc_min)
            unique, counts_arr = np.unique(tiers, return_counts=True)
            counts = dict(zip(unique, counts_arr.astype(int)))
            grid_rows.append({
                "wmb_multiplier": wmb_mult,
                "wmb_spec_high": round(spec_high, 4),
                "wmb_spec_low": round(spec_low, 4),
                "lfc_min": lfc_min,
                "is_default": (wmb_mult == 2.0 and lfc_min == default_lfc),
                "n_high": counts.get("high", 0),
                "n_moderate": counts.get("moderate", 0),
                "n_low": counts.get("low", 0),
                "n_none": counts.get("none", 0),
                "n_attributed": (counts.get("high", 0) +
                                 counts.get("moderate", 0) +
                                 counts.get("low", 0)),
            })

    grid = pd.DataFrame(grid_rows)
    grid_path = os.path.join(OUTPUT_DIR, "sensitivity_grid.csv")
    grid.to_csv(grid_path, index=False)
    print(f"  Saved {grid_path} ({len(grid)} threshold combinations)")

    # Near-miss analysis: rows that flip tier under +/- 1 step perturbation
    # Use steps adjacent to defaults
    wmb_neighbors = [m for m in WMB_MULTIPLIERS
                     if abs(m - 2.0) <= 0.5 and m != 2.0]  # [1.5, 2.5]
    lfc_neighbors = [t for t in LFC_THRESHOLDS
                     if abs(t - 0.10) <= 0.05 and t != 0.10]  # [0.05, 0.15]

    # Assign default tiers
    default_tiers = _assign_confidence_vectorized(
        concordance, wmb_spec, abs_lfc,
        default_spec_high, default_spec_low, default_lfc)
    df["default_tier"] = default_tiers

    near_miss_flags = np.zeros(len(df), dtype=bool)
    perturbed_tiers = {}

    for wmb_mult in wmb_neighbors + [2.0]:
        for lfc_min in lfc_neighbors + [0.10]:
            if wmb_mult == 2.0 and lfc_min == 0.10:
                continue
            spec_high = wmb_mult / n
            spec_low = (wmb_mult / 2.0) / n
            col = f"tier_wmb{wmb_mult}_lfc{lfc_min}"
            perturbed = _assign_confidence_vectorized(
                concordance, wmb_spec, abs_lfc, spec_high, spec_low, lfc_min)
            perturbed_tiers[col] = perturbed
            near_miss_flags |= (perturbed != default_tiers)

    near_miss = df[near_miss_flags].copy()
    for col, tiers in perturbed_tiers.items():
        near_miss[col] = tiers[near_miss_flags]

    near_miss_path = os.path.join(OUTPUT_DIR, "near_miss_rows.csv")
    near_miss.to_csv(near_miss_path, index=False)
    print(f"  Near-miss rows (flip tier under 1-step perturbation): {len(near_miss)}")
    print(f"  Saved {near_miss_path}")

    # Heatmap
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        pivot = grid.pivot(index="wmb_multiplier", columns="lfc_min",
                           values="n_attributed")
        fig, ax = plt.subplots(figsize=(8, 5))
        im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([f"{v:.2f}" for v in pivot.columns])
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([f"{v:.1f}x" for v in pivot.index])
        ax.set_xlabel("SEA-AD |LFC| threshold")
        ax.set_ylabel("WMB specificity multiplier (× 1/N)")
        ax.set_title("Number of attributed rows by threshold")
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val = pivot.values[i, j]
                ax.text(j, i, str(int(val)), ha="center", va="center",
                        fontsize=9, color="white" if val > pivot.values.max() * 0.6 else "black")
        # Mark default
        default_row = list(pivot.index).index(2.0)
        default_col = list(pivot.columns).index(0.10)
        from matplotlib.patches import Rectangle
        ax.add_patch(Rectangle((default_col - 0.5, default_row - 0.5), 1, 1,
                                   fill=False, edgecolor="blue", linewidth=2))
        fig.colorbar(im, ax=ax, label="Attributed rows")
        fig.tight_layout()
        heatmap_path = os.path.join(OUTPUT_DIR, "sensitivity_heatmap.png")
        fig.savefig(heatmap_path, dpi=150)
        plt.close(fig)
        print(f"  Saved {heatmap_path}")
    except ImportError:
        print("  matplotlib not available, skipping heatmap")

    # Summary
    default_row = grid[grid["is_default"]].iloc[0]
    summary = {
        "n_combinations": len(grid),
        "n_near_miss": len(near_miss),
        "default_attributed": int(default_row["n_attributed"]),
        "min_attributed": int(grid["n_attributed"].min()),
        "max_attributed": int(grid["n_attributed"].max()),
        "default_high": int(default_row["n_high"]),
        "default_moderate": int(default_row["n_moderate"]),
        "default_low": int(default_row["n_low"]),
    }
    summary_path = os.path.join(OUTPUT_DIR, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Default: {summary['default_attributed']} attributed "
          f"(range: {summary['min_attributed']}-{summary['max_attributed']} "
          f"across {summary['n_combinations']} combinations)")


def step_summary():
    """Print cached summary."""
    summary_path = os.path.join(OUTPUT_DIR, "summary.json")
    if not os.path.exists(summary_path):
        print("No summary found. Run --run first.")
        return
    with open(summary_path) as f:
        s = json.load(f)
    print(f"\nThreshold Sensitivity Analysis:")
    print(f"  {s['n_combinations']} threshold combinations tested")
    print(f"  Default: {s['default_attributed']} attributed "
          f"(high={s['default_high']}, moderate={s['default_moderate']}, "
          f"low={s['default_low']})")
    print(f"  Range: {s['min_attributed']}-{s['max_attributed']} attributed")
    print(f"  Near-miss rows: {s['n_near_miss']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="Run analysis")
    parser.add_argument("--summary", action="store_true", help="Print cached summary")
    args = parser.parse_args()

    if not any([args.run, args.summary]):
        parser.print_help()
        sys.exit(1)
    if args.run:
        step_run()
    if args.summary:
        step_summary()

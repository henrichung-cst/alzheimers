#!/usr/bin/env python3
"""Final attribution-table assembly and cross-contrast analysis.

Reads the unified attribution table from kinase_attribution.py and produces:
  S1: Cross-contrast consistency analysis
  S2: Final attribution table with cross-contrast annotations

Inputs:
  outputs/reports/kinase_attribution/unified_attribution.csv
  outputs/reports/kinase_attribution/mea_stoichiometry.csv

Outputs (all under outputs/reports/attribution_recovery/):
  cross_contrast_matrix.csv, cross_contrast_heatmap.png
  final_attribution_table.csv
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OUTPUT_DIR = config.ATTRIBUTION_RECOVERY_OUTPUT_DIR
KINASE_ATTR_DIR = config.KINASE_ATTRIBUTION_OUTPUT_DIR


def _ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def _load_unified_attribution():
    """Load unified attribution table from kinase_attribution.py."""
    path = os.path.join(KINASE_ATTR_DIR, "unified_attribution.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run kinase_attribution.py --attribute first.")
    return pd.read_csv(path)


# ===========================================================================
# S1: Cross-contrast consistency
# ===========================================================================

def step_cross_contrast():
    """S1: Check whether same kinase-cell type pairs appear across contrasts."""
    _ensure_output_dir()
    print("\n=== S1: Cross-Contrast Consistency ===\n")

    attr = _load_unified_attribution()
    contrasts = sorted(attr["contrast"].unique())
    print(f"  {len(attr)} attributed rows across {len(contrasts)} contrasts")

    # Build matrix: (kinase, cell_type) x contrast, value = combined_score
    pivot = attr.pivot_table(
        index=["kinase", "gene_symbol", "cell_type"],
        columns="contrast",
        values="combined_score",
        aggfunc="first",
    ).reset_index()

    # Count how many contrasts each (kinase, cell_type) is attributed in
    score_cols = [c for c in pivot.columns if c in contrasts]
    pivot["n_contrasts"] = pivot[score_cols].notna().sum(axis=1)
    pivot["mean_score"] = pivot[score_cols].mean(axis=1)

    # Sort by consistency (n_contrasts desc, then mean_score desc)
    pivot = pivot.sort_values(
        ["n_contrasts", "mean_score"], ascending=[False, False])

    matrix_path = os.path.join(OUTPUT_DIR, "cross_contrast_matrix.csv")
    pivot.to_csv(matrix_path, index=False)
    print(f"  Saved {matrix_path} ({len(pivot)} rows)")

    # Summary
    for n in range(len(contrasts), 0, -1):
        count = (pivot["n_contrasts"] >= n).sum()
        if count > 0:
            print(f"  Attributed in {n}+ contrasts: {count} kinase-CT pairs")

    # Heatmap: top kinase-CT pairs by consistency
    top_pairs = pivot[pivot["n_contrasts"] >= 2].head(30)
    if len(top_pairs) > 0:
        labels = top_pairs.apply(
            lambda r: f"{r['kinase']} / {r['cell_type']}", axis=1)
        plot_data = top_pairs[score_cols].values

        fig, ax = plt.subplots(
            figsize=(max(6, len(score_cols) * 2), max(4, len(top_pairs) * 0.35)))
        im = ax.imshow(plot_data, aspect="auto", cmap="RdYlGn",
                       vmin=-0.5, vmax=1.0)

        ax.set_xticks(range(len(score_cols)))
        ax.set_xticklabels(score_cols, fontsize=10)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=8)

        # Mark NaN cells
        for i in range(plot_data.shape[0]):
            for j in range(plot_data.shape[1]):
                if np.isnan(plot_data[i, j]):
                    ax.text(j, i, "—", ha="center", va="center",
                            fontsize=8, color="gray")

        plt.colorbar(im, ax=ax, label="Combined Score", shrink=0.8)
        ax.set_title("Cross-Contrast Consistency (top kinase-CT pairs)")

        heatmap_path = os.path.join(OUTPUT_DIR, "cross_contrast_heatmap.png")
        fig.savefig(heatmap_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {heatmap_path}")
    else:
        print("  No kinase-CT pairs attributed in 2+ contrasts for heatmap.")


# ===========================================================================
# S2: Final Attribution Table
# ===========================================================================

def step_comprehensive():
    """S2: Build the final attribution table with cross-contrast annotations."""
    _ensure_output_dir()
    print("\n=== S2: Final Attribution Table ===\n")

    attr = _load_unified_attribution()

    # Load cross-contrast matrix if available
    cc_path = os.path.join(OUTPUT_DIR, "cross_contrast_matrix.csv")
    cc_df = pd.read_csv(cc_path) if os.path.exists(cc_path) else pd.DataFrame()

    # Build cross-contrast count lookup
    cc_lookup = {}
    if len(cc_df) > 0 and "n_contrasts" in cc_df.columns:
        for _, row in cc_df.iterrows():
            key = (row["kinase"], row["cell_type"])
            cc_lookup[key] = int(row["n_contrasts"])

    # Add cross-contrast count to attribution table
    attr["n_contrasts_attributed"] = attr.apply(
        lambda r: cc_lookup.get((r["kinase"], r["cell_type"]), 1),
        axis=1)

    # Load mechanism annotation if available
    mech_path = os.path.join(KINASE_ATTR_DIR, "mechanism_annotation.csv")
    mech_lookup = {}
    if os.path.exists(mech_path):
        mech_df = pd.read_csv(mech_path)
        for _, row in mech_df.iterrows():
            mech_lookup[(row["kinase"], row["contrast"])] = row["mechanism"]

    if mech_lookup:
        attr["mechanism_annotation"] = attr.apply(
            lambda r: mech_lookup.get((r["kinase"], r["contrast"]), ""),
            axis=1)

    # Sort by combined_score descending
    attr = attr.sort_values("combined_score", ascending=False)

    # Save final table
    final_path = os.path.join(OUTPUT_DIR, "final_attribution_table.csv")
    attr.to_csv(final_path, index=False)
    print(f"  Saved {final_path} ({len(attr)} rows)")

    # Summary
    n_unique_kinases = attr["kinase"].nunique()
    n_unique_pairs = attr.groupby(["kinase", "contrast"]).ngroups
    print(f"\n  {n_unique_kinases} unique kinases")
    print(f"  {n_unique_pairs} kinase-contrast pairs")
    print(f"  {len(attr)} total attributed (kinase, contrast, cell_type) rows")

    print("\n  By confidence:")
    for conf, cnt in attr["combined_confidence"].value_counts().items():
        print(f"    {conf}: {cnt}")

    print("\n  By cell type:")
    for ct, cnt in attr["cell_type"].value_counts().items():
        print(f"    {ct}: {cnt}")

    print("\n  By contrast:")
    for cn, cnt in attr["contrast"].value_counts().items():
        print(f"    {cn}: {cnt}")

    multi = attr[attr["n_contrasts_attributed"] >= 2]
    if len(multi) > 0:
        n_multi = multi.groupby(["kinase", "cell_type"]).ngroups
        print(f"\n  Kinase-CT pairs consistent across 2+ contrasts: {n_multi}")

    print("\n  S2 complete.")


# ===========================================================================
# Summary
# ===========================================================================

def print_summary():
    """Print cached results summary."""
    print("\n" + "=" * 70)
    print("Attribution Recovery — Summary")
    print("=" * 70)

    # S1: Cross-contrast
    cc_path = os.path.join(OUTPUT_DIR, "cross_contrast_matrix.csv")
    if os.path.exists(cc_path):
        cc_df = pd.read_csv(cc_path)
        print("\nS1: Cross-Contrast Consistency")
        print(f"  {len(cc_df)} kinase-cell type pairs")
        if "n_contrasts" in cc_df.columns:
            for n in sorted(cc_df["n_contrasts"].unique(), reverse=True):
                cnt = (cc_df["n_contrasts"] == n).sum()
                print(f"    In {n} contrasts: {cnt}")
    else:
        print("\nS1: Not yet computed")

    # S2: Final table
    final_path = os.path.join(OUTPUT_DIR, "final_attribution_table.csv")
    if os.path.exists(final_path):
        final_df = pd.read_csv(final_path)
        print(f"\nS2: Final Attribution Table")
        print(f"  {len(final_df)} total attributed rows")
        print(f"  {final_df['kinase'].nunique()} unique kinases")
        if "combined_confidence" in final_df.columns:
            print("  By confidence:")
            for conf, cnt in final_df["combined_confidence"].value_counts().items():
                print(f"    {conf}: {cnt}")
    else:
        print("\nS2: Not yet computed")

    print()


# ===========================================================================
# CLI
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Attribution Recovery: Cross-contrast analysis and final table",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--cross-contrast", action="store_true",
                       help="S1: Cross-contrast consistency analysis")
    group.add_argument("--comprehensive", action="store_true",
                       help="S2: Final attribution table")
    group.add_argument("--run", action="store_true",
                       help="Run all steps in order")
    group.add_argument("--summary", action="store_true",
                       help="Print cached results summary")

    args = parser.parse_args()

    if args.cross_contrast or args.run:
        step_cross_contrast()
    if args.comprehensive or args.run:
        step_comprehensive()
    if args.summary:
        print_summary()
    if args.run:
        print_summary()


if __name__ == "__main__":
    main()

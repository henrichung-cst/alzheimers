"""Quantify substrate overlap within kinase families.

Addresses reviewer concern about pseudo-replication: e.g., PAK1-6
sharing substrates means their "independent" enrichment signals may
reflect a single underlying substrate pool.

Runs enrichment on a representative comparison (AppP 4mo bulk),
extracts driving substrates per kinase, and computes pairwise
Jaccard similarity within families.
"""

import os
import sys
from itertools import combinations

import kinase_library as kl
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import config
from analysis_utils import calculate_log2_fold_change

# Families to analyze (kinase-library abbreviations)
FAMILIES = {
    "PAK": ["PAK1", "PAK2", "PAK3", "PAK4", "PAK5", "PAK6"],
    "PKC": ["PKCA", "PKCB", "PKCG", "PKCD", "PKCE", "PKCH", "PKCI", "PKCT", "PKCZ"],
    "AKT": ["AKT1", "AKT2", "AKT3"],
    "PKA": ["PKACA", "PKACB", "PKACG"],
}

# Representative comparison: AppP 4mo bulk (all families significant here)
BG_COL = "M_4mo_WT"
FG_COL = "M_4mo_APP"


def get_enrichment():
    """Run enrichment on the representative comparison."""
    df = pd.read_csv(config.get_bulk_input_file(config.KIN_TYPE))
    work_df = df[[BG_COL, FG_COL, "motif"]].copy()
    work_df["log2_fold_change"] = calculate_log2_fold_change(work_df, FG_COL, BG_COL)

    dpd = kl.DiffPhosData(
        work_df, seq_col="motif", lfc_col="log2_fold_change",
        percent_rank=config.PERCENT_RANK, percent_thresh=config.PERCENT_THRESH,
    )
    enrich_data = dpd.kinase_enrichment(
        kin_type=config.KIN_TYPE,
        kl_method=config.KL_METHOD,
        kl_thresh=config.KL_THRESH,
    )
    return enrich_data


def get_substrates(enrich_data, kinase, activity):
    """Extract the set of substrate motifs driving a kinase's enrichment."""
    try:
        subs = enrich_data.enriched_subs(
            [kinase], activity,
            data_columns=["motif"],
            as_dataframe=True,
        )
        if subs is not None and not subs.empty:
            return set(subs["motif"].tolist())
    except Exception as e:
        print(f"  Warning: could not get substrates for {kinase} ({activity}): {e}")
    return set()


def compute_family_overlap(enrich_data, family_name, members):
    """Compute pairwise Jaccard similarity for a kinase family."""
    results_df = enrich_data.combined_enrichment_results

    # Get direction for each kinase
    substrate_sets = {}
    for kinase in members:
        if kinase not in results_df.index:
            print(f"  {kinase} not in enrichment results, skipping")
            continue
        direction = results_df.loc[kinase, "most_sig_direction"]
        activity = "inhibited" if direction == "-" else "activated"
        subs = get_substrates(enrich_data, kinase, activity)
        if subs:
            substrate_sets[kinase] = subs
            print(f"  {kinase}: {len(subs)} substrates ({activity})")

    if len(substrate_sets) < 2:
        print(f"  Insufficient kinases with substrates for {family_name}")
        return None

    # Pairwise Jaccard
    rows = []
    for k1, k2 in combinations(sorted(substrate_sets.keys()), 2):
        s1, s2 = substrate_sets[k1], substrate_sets[k2]
        intersection = len(s1 & s2)
        union = len(s1 | s2)
        jaccard = intersection / union if union > 0 else 0
        rows.append({
            "family": family_name,
            "kinase_1": k1,
            "kinase_2": k2,
            "substrates_1": len(s1),
            "substrates_2": len(s2),
            "intersection": intersection,
            "union": union,
            "jaccard": round(jaccard, 3),
        })

    pair_df = pd.DataFrame(rows)

    # Family-level summary
    all_subs = set()
    for s in substrate_sets.values():
        all_subs.update(s)
    total_member_subs = sum(len(s) for s in substrate_sets.values())

    summary = {
        "family": family_name,
        "n_members": len(substrate_sets),
        "mean_jaccard": round(pair_df["jaccard"].mean(), 3),
        "max_jaccard": round(pair_df["jaccard"].max(), 3),
        "min_jaccard": round(pair_df["jaccard"].min(), 3),
        "unique_substrates": len(all_subs),
        "total_substrates": total_member_subs,
        "redundancy_ratio": round(1 - len(all_subs) / total_member_subs, 3) if total_member_subs > 0 else 0,
    }

    return pair_df, summary


def main():
    print("Running enrichment on AppP 4mo bulk...")
    enrich_data = get_enrichment()

    all_pairs = []
    summaries = []

    for family_name, members in FAMILIES.items():
        print(f"\n--- {family_name} family ---")
        result = compute_family_overlap(enrich_data, family_name, members)
        if result is None:
            continue
        pair_df, summary = result
        all_pairs.append(pair_df)
        summaries.append(summary)

        print(f"  Mean Jaccard: {summary['mean_jaccard']}")
        print(f"  Max Jaccard:  {summary['max_jaccard']}")
        print(f"  Unique/Total substrates: {summary['unique_substrates']}/{summary['total_substrates']} "
              f"(redundancy: {summary['redundancy_ratio']:.0%})")

    # Save outputs
    out_dir = os.path.join("outputs", "bulk")
    os.makedirs(out_dir, exist_ok=True)

    if all_pairs:
        pairs_out = os.path.join(out_dir, "substrate_overlap_pairs.csv")
        pd.concat(all_pairs, ignore_index=True).to_csv(pairs_out, index=False)
        print(f"\nSaved pairwise overlap: {pairs_out}")

    if summaries:
        summary_out = os.path.join(out_dir, "substrate_overlap_summary.csv")
        pd.DataFrame(summaries).to_csv(summary_out, index=False)
        print(f"Saved family summary: {summary_out}")

        # Print markdown table for easy report inclusion
        print("\n### Substrate overlap summary (for report)")
        print("| Family | Members | Mean Jaccard | Max Jaccard | Unique / Total substrates | Redundancy |")
        print("|--------|:-------:|:------------:|:-----------:|:-------------------------:|:----------:|")
        for s in summaries:
            print(f"| {s['family']} | {s['n_members']} | {s['mean_jaccard']} | {s['max_jaccard']} | "
                  f"{s['unique_substrates']} / {s['total_substrates']} | {s['redundancy_ratio']:.0%} |")


if __name__ == "__main__":
    main()

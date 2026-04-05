#!/usr/bin/env python3
"""Q5: Parent protein quality diagnostics for activity-driven kinases.

Checks whether the 12 activity-driven kinases (identified by mechanism
annotation) have reliable parent protein data, or whether noisy parent
protein estimates could create spurious stoichiometry signals.

Usage:
    python code/supplementary/parent_protein_qc.py --run
    python code/supplementary/parent_protein_qc.py --summary
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

OUTPUT_DIR = os.path.join(config.SUPPLEMENTARY_OUTPUT_DIR, "parent_protein_qc")

TOTAL_PROTEOME_FILE = os.path.join(
    config.SONG_PRIMARY_PROTEOMICS_DIR,
    "song2024_tmttotal_protein_quant_merged_labeled (2).xlsx",
)
KINASE_ATTR_DIR = config.KINASE_ATTRIBUTION_OUTPUT_DIR
DATA_INGEST_DIR = config.DATA_INGEST_OUTPUT_DIR


def _ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def _load_sample_mapping():
    """Load sample mapping from data_ingest output."""
    path = os.path.join(DATA_INGEST_DIR, "sample_mapping.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run data_ingest.py --mapping first.")
    return pd.read_csv(path)


def step_run():
    """Run parent protein QC diagnostics."""
    _ensure_output_dir()
    print("\n=== Parent Protein QC for Activity-Driven Kinases ===\n")

    # 1. Load mechanism annotation
    mech_path = os.path.join(KINASE_ATTR_DIR, "mechanism_annotation.csv")
    if not os.path.exists(mech_path):
        raise FileNotFoundError(
            f"{mech_path} not found. Run kinase_attribution.py --mechanism-annotation first.")
    mech = pd.read_csv(mech_path)

    activity_driven = mech[mech["mechanism"] == "activity_driven"].copy()
    all_kinases = mech[mech["mechanism"].isin(
        ["activity_driven", "abundance_driven", "both"])].copy()
    print(f"  Mechanism annotation: {len(mech)} total, "
          f"{len(activity_driven)} activity-driven")

    if len(activity_driven) == 0:
        print("  No activity-driven kinases found. Nothing to check.")
        return

    # 2. Map kinases to gene symbols
    k2g = pd.read_csv(config.MAPPING_CACHE_FILE)
    kinase_to_gene = dict(zip(k2g["kinase_abbreviation"], k2g["gene_symbol"]))
    activity_driven["gene_symbol"] = activity_driven["kinase"].map(
        lambda k: kinase_to_gene.get(k, k))
    all_kinases["gene_symbol"] = all_kinases["kinase"].map(
        lambda k: kinase_to_gene.get(k, k))

    # Get unique gene symbols for activity-driven kinases
    ad_genes = set(activity_driven["gene_symbol"].dropna().str.upper())
    all_genes = set(all_kinases["gene_symbol"].dropna().str.upper())
    print(f"  Activity-driven unique genes: {len(ad_genes)}")
    print(f"  All significant unique genes: {len(all_genes)}")

    # 3. Load total proteome to assess parent protein quality
    mapping = _load_sample_mapping()
    bio_cols = mapping["column_name"].tolist()

    print("  Loading total proteome...")
    # Read only needed columns to reduce I/O and memory
    needed_cols = ["Gene Symbol"] + bio_cols
    tp = pd.read_excel(TOTAL_PROTEOME_FILE, header=1, usecols=needed_cols)
    tp_genes = tp["Gene Symbol"].fillna("").astype(str).str.upper()

    # Extract bio columns
    tp_bio = tp[[c for c in bio_cols if c in tp.columns]].apply(
        pd.to_numeric, errors="coerce")

    # 4. Compute QC metrics per protein
    qc_rows = []
    for gene_upper in all_genes:
        mask = tp_genes == gene_upper
        if mask.sum() == 0:
            qc_rows.append({
                "gene_symbol": gene_upper,
                "found_in_proteome": False,
                "n_rows": 0,
                "detection_rate": 0.0,
                "median_abundance": np.nan,
                "cv": np.nan,
                "is_activity_driven": gene_upper in ad_genes,
            })
            continue

        # Take the row with highest median if multiple rows
        vals_all = tp_bio.loc[mask]
        medians = vals_all.median(axis=1, skipna=True)
        best_row_idx = medians.idxmax()
        vals = tp_bio.loc[best_row_idx].values.astype(float)

        n_total = len(vals)
        n_detected = int(np.sum(np.isfinite(vals) & (vals > 0)))
        detection_rate = n_detected / n_total if n_total > 0 else 0.0

        valid = vals[np.isfinite(vals) & (vals > 0)]
        median_abundance = float(np.median(valid)) if len(valid) > 0 else np.nan
        cv = float(np.std(valid) / np.mean(valid)) if len(valid) > 1 and np.mean(valid) > 0 else np.nan

        qc_rows.append({
            "gene_symbol": gene_upper,
            "found_in_proteome": True,
            "n_rows": int(mask.sum()),
            "detection_rate": round(detection_rate, 3),
            "median_abundance": round(median_abundance, 2) if np.isfinite(median_abundance) else np.nan,
            "cv": round(cv, 3) if np.isfinite(cv) else np.nan,
            "is_activity_driven": gene_upper in ad_genes,
        })

    qc_df = pd.DataFrame(qc_rows)

    # 5. Save per-kinase QC
    ad_qc = qc_df[qc_df["is_activity_driven"]].copy()
    ad_qc_path = os.path.join(OUTPUT_DIR, "activity_driven_parent_qc.csv")
    ad_qc.to_csv(ad_qc_path, index=False)
    print(f"\n  Activity-driven parent protein QC:")
    print(f"    Found in proteome: {ad_qc['found_in_proteome'].sum()} / {len(ad_qc)}")
    if len(ad_qc[ad_qc["found_in_proteome"]]) > 0:
        detected = ad_qc[ad_qc["found_in_proteome"]]
        print(f"    Detection rate: median={detected['detection_rate'].median():.3f}, "
              f"min={detected['detection_rate'].min():.3f}")
        print(f"    CV: median={detected['cv'].median():.3f}, "
              f"max={detected['cv'].max():.3f}")
    print(f"  Saved {ad_qc_path}")

    # 6. Comparison: activity-driven vs all others
    other_qc = qc_df[~qc_df["is_activity_driven"]].copy()
    comparison_rows = []
    for label, subset in [("activity_driven", ad_qc), ("other_significant", other_qc)]:
        detected = subset[subset["found_in_proteome"]]
        comparison_rows.append({
            "group": label,
            "n_kinases": len(subset),
            "n_found_in_proteome": int(subset["found_in_proteome"].sum()),
            "median_detection_rate": round(detected["detection_rate"].median(), 3) if len(detected) > 0 else np.nan,
            "min_detection_rate": round(detected["detection_rate"].min(), 3) if len(detected) > 0 else np.nan,
            "median_cv": round(detected["cv"].median(), 3) if len(detected) > 0 else np.nan,
            "max_cv": round(detected["cv"].max(), 3) if len(detected) > 0 else np.nan,
            "median_abundance": round(detected["median_abundance"].median(), 2) if len(detected) > 0 else np.nan,
        })

    comp_df = pd.DataFrame(comparison_rows)
    comp_path = os.path.join(OUTPUT_DIR, "parent_qc_comparison.csv")
    comp_df.to_csv(comp_path, index=False)
    print(f"\n  Comparison (activity-driven vs other significant kinases):")
    for _, row in comp_df.iterrows():
        print(f"    {row['group']}: n={row['n_kinases']}, "
              f"detection={row['median_detection_rate']}, "
              f"CV={row['median_cv']}, "
              f"abundance={row['median_abundance']}")
    print(f"  Saved {comp_path}")

    # 7. Summary
    ad_detected = ad_qc[ad_qc["found_in_proteome"]]
    flagged = ad_detected[
        (ad_detected["detection_rate"] < 0.5) | (ad_detected["cv"] > 1.0)
    ] if len(ad_detected) > 0 else pd.DataFrame()

    summary = {
        "n_activity_driven": len(ad_qc),
        "n_found_in_proteome": int(ad_qc["found_in_proteome"].sum()),
        "median_detection_rate": round(ad_detected["detection_rate"].median(), 3) if len(ad_detected) > 0 else None,
        "median_cv": round(ad_detected["cv"].median(), 3) if len(ad_detected) > 0 else None,
        "n_flagged": len(flagged),
        "flagged_genes": flagged["gene_symbol"].tolist() if len(flagged) > 0 else [],
    }
    summary_path = os.path.join(OUTPUT_DIR, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    if len(flagged) > 0:
        print(f"\n  WARNING: {len(flagged)} activity-driven kinases have "
              f"poor parent protein quality: {', '.join(flagged['gene_symbol'])}")
    else:
        print(f"\n  All activity-driven kinases have adequate parent protein quality.")


def step_summary():
    """Print cached summary."""
    summary_path = os.path.join(OUTPUT_DIR, "summary.json")
    if not os.path.exists(summary_path):
        print("No summary found. Run --run first.")
        return
    with open(summary_path) as f:
        s = json.load(f)
    print(f"\nParent Protein QC:")
    print(f"  Activity-driven kinases: {s['n_activity_driven']}")
    print(f"  Found in proteome: {s['n_found_in_proteome']}")
    print(f"  Median detection rate: {s['median_detection_rate']}")
    print(f"  Median CV: {s['median_cv']}")
    if s["n_flagged"] > 0:
        print(f"  FLAGGED (low detection or high CV): {', '.join(s['flagged_genes'])}")
    else:
        print(f"  No quality concerns flagged")


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

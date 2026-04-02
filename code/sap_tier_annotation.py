#!/usr/bin/env python3
"""Tier 2–3 annotation pipeline for SAP kinase enrichment.

Produces a per-kinase, per-cell-type annotation table that cross-references:
  - Tier 1:  Bulk-level kinase enrichment (from kl_analysis_bulk.py)
  - Tier 2a: DESP substrate cell-type concentration (from X^base)
  - Tier 2b: Kinase cell-type expression (from R_annotation.csv / snRNA-seq + Allen)
  - Tier 3:  Concordance between substrate localization and kinase expression

See sap_extension.md §2.3 and sap.md §8.1.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import config
from sap_data import load_x_base

# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------
TIER_OUTPUT_DIR = os.path.join(config.REPO_ROOT, "outputs", "reports", "tier_annotation")

TIER2A_FILE = os.path.join(TIER_OUTPUT_DIR, "tier2a_substrate_concentration.csv")
TIER2B_FILE = os.path.join(TIER_OUTPUT_DIR, "tier2b_kinase_expression.csv")
TIER3_FILE = os.path.join(TIER_OUTPUT_DIR, "tier3_cross_reference.csv")
SUMMARY_FILE = os.path.join(TIER_OUTPUT_DIR, "tier_annotation_summary.json")

R_ANNOTATION_FILE = os.path.join(
    config.REPO_ROOT, "outputs", "reports", "preflight", "R_annotation.csv"
)
BULK_KINASE_RESULTS_FILE = os.path.join(
    config.REPO_ROOT, "outputs", "bulk", "kinase_results.csv"
)

CELLTYPES = config.SAP_ESTIMATED_CELLTYPES  # 5 resolved cell types


# ---------------------------------------------------------------------------
# Tier 2a: DESP substrate cell-type concentration
# ---------------------------------------------------------------------------

def compute_tier2a() -> pd.DataFrame:
    """Compute per-kinase, per-cell-type substrate concentration scores.

    For each kinase K, find its substrates in kldata, match them to X^base,
    and compute substrate enrichment per cell type.

    Strategy: compute per-site cell-type fractions from X^base, then for each
    kinase average the per-site fractions across its substrates. The enrichment
    score is this kinase-specific mean divided by the global mean (across all
    sites). Enrichment > 1 means the kinase's substrates are concentrated in
    that cell type relative to the phosphoproteome average.

    Returns DataFrame with columns:
        kinase, n_substrates_matched, n_substrates_total,
        conc_{CT} (mean per-site fraction for kinase substrates),
        enrich_{CT} (enrichment vs global mean),
        dominant_ct (CT with highest enrichment),
        specificity (1 - normalized Shannon entropy of enrichment, 0=uniform)
    """
    print("Computing Tier 2a: DESP substrate concentration scores...")

    # Load X^base and site metadata
    x_base, site_meta = load_x_base()
    x_est = x_base[CELLTYPES].values  # (J, 5)
    site_ids = site_meta[["protein_id", "site_position"]].copy()

    # Compute per-site cell-type fractions
    row_sums = np.nansum(x_est, axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    site_fracs = x_est / row_sums  # (J, 5), NaN preserved

    # Global baseline: mean per-site fraction across all sites
    global_frac = np.nanmean(site_fracs, axis=0)  # (5,)
    print(f"  Global CT fractions: "
          + ", ".join(f"{ct}={global_frac[i]:.3f}" for i, ct in enumerate(CELLTYPES)))

    # Load kinase-substrate mapping
    kldata = pd.read_csv(config.KLDATA_FILE, usecols=["protein_id", "site_position", "KINASE"])
    kldata = kldata.dropna(subset=["KINASE"])
    kinases_all = sorted(kldata["KINASE"].unique())
    print(f"  kldata: {len(kldata)} assignments, {len(kinases_all)} kinases")

    # Build site index for fast lookup
    site_key = site_ids["protein_id"].astype(str) + ":" + site_ids["site_position"].astype(str)
    site_lookup = pd.Series(np.arange(len(site_key)), index=site_key)

    # Match kldata sites to X^base
    kl_key = kldata["protein_id"].astype(str) + ":" + kldata["site_position"].astype(str)
    kldata = kldata.copy()
    kldata["_key"] = kl_key
    kldata["_idx"] = kldata["_key"].map(site_lookup)

    matched = kldata.dropna(subset=["_idx"])
    matched_idx = matched["_idx"].astype(int).values
    print(f"  Matched {matched['_key'].nunique()} / {kldata['_key'].nunique()} "
          f"unique sites to X^base")

    # Compute per-kinase enrichment
    rows = []
    for kinase in kinases_all:
        mask = matched["KINASE"] == kinase
        sub_idx = matched_idx[mask.values]
        n_total = (kldata["KINASE"] == kinase).sum()
        sub_idx_unique = np.unique(sub_idx)
        n_matched = len(sub_idx_unique)

        if n_matched == 0:
            row = {"kinase": kinase, "n_substrates_matched": 0,
                   "n_substrates_total": n_total}
            for ct in CELLTYPES:
                row[f"conc_{ct}"] = np.nan
                row[f"enrich_{ct}"] = np.nan
            row["dominant_ct"] = None
            row["specificity"] = np.nan
            rows.append(row)
            continue

        # Mean per-site CT fraction for this kinase's substrates
        kinase_fracs = np.nanmean(site_fracs[sub_idx_unique], axis=0)  # (5,)

        # Enrichment: kinase mean / global mean
        with np.errstate(divide="ignore", invalid="ignore"):
            enrichment = kinase_fracs / global_frac  # (5,)
        enrichment = np.where(np.isfinite(enrichment), enrichment, np.nan)

        # Dominant CT by enrichment
        valid_enrich = np.where(np.isnan(enrichment), -np.inf, enrichment)
        dominant = CELLTYPES[np.argmax(valid_enrich)]

        # Specificity from enrichment profile (normalized Shannon entropy)
        e_pos = enrichment[np.isfinite(enrichment) & (enrichment > 0)]
        if len(e_pos) > 1:
            p = e_pos / e_pos.sum()
            H = -np.sum(p * np.log(p))
            H_max = np.log(len(e_pos))
            spec = 1.0 - H / H_max if H_max > 0 else 0.0
        elif len(e_pos) == 1:
            spec = 1.0
        else:
            spec = np.nan

        row = {"kinase": kinase, "n_substrates_matched": n_matched,
               "n_substrates_total": n_total}
        for i, ct in enumerate(CELLTYPES):
            row[f"conc_{ct}"] = kinase_fracs[i]
            row[f"enrich_{ct}"] = enrichment[i]
        row["dominant_ct"] = dominant
        row["specificity"] = spec
        rows.append(row)

    df = pd.DataFrame(rows)
    os.makedirs(TIER_OUTPUT_DIR, exist_ok=True)
    df.to_csv(TIER2A_FILE, index=False)
    print(f"  Saved {len(df)} kinases → {TIER2A_FILE}")

    # Summary stats
    matched_kinases = df[df.n_substrates_matched > 0]
    print(f"  Kinases with matched substrates: {len(matched_kinases)}/{len(df)}")
    print(f"  Median substrates matched: {matched_kinases.n_substrates_matched.median():.0f}")
    print(f"  Mean specificity: {matched_kinases.specificity.mean():.3f}")
    print(f"  Dominant CT distribution:")
    for ct, n in matched_kinases.dominant_ct.value_counts().items():
        print(f"    {ct}: {n}")

    return df


# ---------------------------------------------------------------------------
# Tier 2b: Kinase cell-type expression annotation
# ---------------------------------------------------------------------------

def compute_tier2b() -> pd.DataFrame:
    """Load and enhance the R_annotation from preflight with specificity scores.

    Adds:
        - n_celltypes_expressed: number of cell types where R=1
        - expression_specificity: 1 if expressed in exactly 1 CT, 0 if all 5, NaN if none
        - ct_specific: name of the single CT if expressed in exactly 1

    Returns DataFrame with all R_annotation columns plus the new ones.
    """
    print("\nComputing Tier 2b: Kinase expression annotation...")

    r_annot = pd.read_csv(R_ANNOTATION_FILE)
    print(f"  Loaded R_annotation: {len(r_annot)} kinases")

    r_cols = [f"R_{ct}" for ct in CELLTYPES]

    # Expression breadth
    r_annot["n_celltypes_expressed"] = r_annot[r_cols].sum(axis=1).astype(int)

    # Expression specificity: (K - n_expressed) / (K - 1) where K=5
    # 1.0 = single CT, 0.0 = all 5, NaN = none
    K = len(CELLTYPES)
    n_expr = r_annot["n_celltypes_expressed"]
    r_annot["expression_specificity"] = np.where(
        n_expr == 0, np.nan,
        (K - n_expr) / (K - 1)
    )

    # Identify CT-specific kinases (expressed in exactly 1 CT)
    r_annot["ct_specific"] = None
    for _, row in r_annot.iterrows():
        if row["n_celltypes_expressed"] == 1:
            for ct in CELLTYPES:
                if row[f"R_{ct}"]:
                    r_annot.loc[row.name, "ct_specific"] = ct
                    break

    os.makedirs(TIER_OUTPUT_DIR, exist_ok=True)
    r_annot.to_csv(TIER2B_FILE, index=False)
    print(f"  Saved {len(r_annot)} kinases → {TIER2B_FILE}")

    # Summary stats
    expressed = r_annot[r_annot.n_celltypes_expressed > 0]
    print(f"  Kinases expressed in ≥1 CT: {len(expressed)}/{len(r_annot)}")
    print(f"  Expression breadth distribution:")
    for n in range(0, K + 1):
        count = (r_annot.n_celltypes_expressed == n).sum()
        if count > 0:
            print(f"    {n} CTs: {count} kinases")
    ct_spec = r_annot[r_annot.ct_specific.notna()]
    if len(ct_spec) > 0:
        print(f"  CT-specific kinases ({len(ct_spec)}):")
        for ct, n in ct_spec.ct_specific.value_counts().items():
            print(f"    {ct}: {n}")

    return r_annot


# ---------------------------------------------------------------------------
# Tier 3: Cross-reference
# ---------------------------------------------------------------------------

def compute_tier3(tier2a: pd.DataFrame = None, tier2b: pd.DataFrame = None) -> pd.DataFrame:
    """Cross-reference Tier 1 bulk enrichment with Tier 2a/2b annotations.

    For each Tier-1-significant kinase × condition × timepoint, annotates:
    - Tier 2a: substrate concentration per CT, dominant CT
    - Tier 2b: expression per CT, expression specificity
    - Concordance: for each CT, whether substrate concentration AND kinase
      expression agree (both present → concordant, one missing → discordant)

    Concordance logic per cell type:
        - "concordant": kinase expressed in CT (R=1) AND substrates enriched
          in CT (enrichment > 1, i.e. above phosphoproteome average)
        - "expression_only": kinase expressed but substrates not enriched
        - "substrate_only": substrates enriched but kinase not expressed
        - "absent": neither expressed nor enriched
        - "no_data": kinase not in R_annotation or no matched substrates
    """
    print("\nComputing Tier 3: Cross-referenced kinase enrichment...")

    # Load Tier 1
    bulk = pd.read_csv(BULK_KINASE_RESULTS_FILE)
    sig = bulk[bulk.significance_tier == "significant"].copy()
    print(f"  Tier 1: {len(sig)} significant kinase×comparison entries "
          f"({sig.kinase.nunique()} unique kinases)")

    # Load Tier 2a/2b if not provided
    if tier2a is None:
        tier2a = pd.read_csv(TIER2A_FILE)
    if tier2b is None:
        tier2b = pd.read_csv(TIER2B_FILE)

    # Merge Tier 2a onto significant entries
    t2a_cols = ["kinase", "n_substrates_matched", "dominant_ct", "specificity"]
    t2a_cols += [f"conc_{ct}" for ct in CELLTYPES]
    t2a_cols += [f"enrich_{ct}" for ct in CELLTYPES]
    sig = sig.merge(tier2a[t2a_cols], on="kinase", how="left", suffixes=("", "_t2a"))

    # Merge Tier 2b
    r_cols = [f"R_{ct}" for ct in CELLTYPES]
    t2b_cols = ["kinase", "in_aggexp", "allen_expressed",
                "n_celltypes_expressed", "expression_specificity", "ct_specific"] + r_cols
    # Only keep columns that exist
    t2b_cols = [c for c in t2b_cols if c in tier2b.columns]
    sig = sig.merge(tier2b[t2b_cols], on="kinase", how="left")

    # Compute per-CT concordance
    # Enrichment > 1 means substrates are concentrated above the global average
    for ct in CELLTYPES:
        enrich_col = f"enrich_{ct}"
        r_col = f"R_{ct}"
        conc_status = f"concordance_{ct}"

        has_enrich = sig[enrich_col].notna()
        has_r = r_col in sig.columns and sig[r_col].notna().any()

        concentrated = has_enrich & (sig[enrich_col] > 1.0)
        expressed = sig[r_col].fillna(False).astype(bool) if has_r else pd.Series(False, index=sig.index)

        sig[conc_status] = "no_data"
        sig.loc[has_enrich, conc_status] = "absent"
        sig.loc[concentrated & ~expressed, conc_status] = "substrate_only"
        sig.loc[~concentrated & expressed, conc_status] = "expression_only"
        sig.loc[concentrated & expressed, conc_status] = "concordant"

    # Overall concordance: count CTs with concordant status
    conc_cols = [f"concordance_{ct}" for ct in CELLTYPES]
    sig["n_concordant_cts"] = (sig[conc_cols] == "concordant").sum(axis=1)
    sig["n_discordant_cts"] = (
        (sig[conc_cols] == "substrate_only").sum(axis=1) +
        (sig[conc_cols] == "expression_only").sum(axis=1)
    )

    # Best concordant CT: the CT with highest enrichment among concordant CTs
    best_cts = []
    for _, row in sig.iterrows():
        best = None
        best_enrich = -1
        for ct in CELLTYPES:
            if row[f"concordance_{ct}"] == "concordant":
                e = row.get(f"enrich_{ct}", 0)
                if e is not None and e > best_enrich:
                    best_enrich = e
                    best = ct
        best_cts.append(best)
    sig["best_concordant_ct"] = best_cts

    os.makedirs(TIER_OUTPUT_DIR, exist_ok=True)
    sig.to_csv(TIER3_FILE, index=False)
    print(f"  Saved {len(sig)} rows → {TIER3_FILE}")

    # Summary stats
    print(f"\n  Concordance summary (across {len(sig)} sig entries):")
    print(f"    Entries with ≥1 concordant CT: "
          f"{(sig.n_concordant_cts > 0).sum()} ({(sig.n_concordant_cts > 0).mean():.1%})")
    print(f"    Mean concordant CTs per entry: {sig.n_concordant_cts.mean():.2f}")

    # Per-CT concordance breakdown
    print(f"\n  Per-cell-type concordance breakdown:")
    for ct in CELLTYPES:
        col = f"concordance_{ct}"
        vc = sig[col].value_counts()
        conc = vc.get("concordant", 0)
        sub_only = vc.get("substrate_only", 0)
        expr_only = vc.get("expression_only", 0)
        absent = vc.get("absent", 0)
        print(f"    {ct:25s}  concordant={conc:3d}  "
              f"substrate_only={sub_only:3d}  expression_only={expr_only:3d}  "
              f"absent={absent:3d}")

    # Top concordant kinases
    top = sig[sig.n_concordant_cts > 0].sort_values("n_concordant_cts", ascending=False)
    if len(top) > 0:
        print(f"\n  Top concordant kinases (by # concordant CTs):")
        seen = set()
        for _, row in top.iterrows():
            k = row["kinase"]
            if k in seen:
                continue
            seen.add(k)
            print(f"    {k:10s}  {row.condition:5s} {row.timepoint}  "
                  f"best_ct={row.best_concordant_ct}  "
                  f"n_conc={row.n_concordant_cts}  lff={row.lff:+.3f}")
            if len(seen) >= 15:
                break

    return sig


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary():
    """Print cached results from all tiers."""
    print("=" * 60)
    print("Tier 2–3 Annotation Summary")
    print("=" * 60)

    if os.path.exists(SUMMARY_FILE):
        with open(SUMMARY_FILE) as f:
            summary = json.load(f)
        for k, v in summary.items():
            print(f"\n{k}:")
            if isinstance(v, dict):
                for k2, v2 in v.items():
                    print(f"  {k2}: {v2}")
            else:
                print(f"  {v}")
    else:
        print("No cached summary found. Run --all first.")
        return

    # Print Tier 3 highlights
    if os.path.exists(TIER3_FILE):
        t3 = pd.read_csv(TIER3_FILE)
        print(f"\nTier 3 highlights ({len(t3)} entries):")
        conc = t3[t3.n_concordant_cts > 0]
        print(f"  Entries with concordant CT attribution: {len(conc)} ({len(conc)/len(t3):.1%})")

        # Unique kinases with concordance
        if len(conc) > 0:
            kinase_best = conc.groupby("kinase").agg(
                n_sig_entries=("kinase", "size"),
                max_concordant_cts=("n_concordant_cts", "max"),
                best_ct=("best_concordant_ct", "first"),
            ).sort_values("max_concordant_cts", ascending=False)
            print(f"  Unique kinases with concordance: {len(kinase_best)}")
            print(f"\n  Top 20 kinases by concordance breadth:")
            for kinase, row in kinase_best.head(20).iterrows():
                print(f"    {kinase:10s}  max_conc_cts={row.max_concordant_cts}  "
                      f"n_sig={row.n_sig_entries}  best_ct={row.best_ct}")


def save_summary(tier2a: pd.DataFrame, tier2b: pd.DataFrame, tier3: pd.DataFrame):
    """Save a JSON summary of key statistics."""
    matched_2a = tier2a[tier2a.n_substrates_matched > 0]
    expressed_2b = tier2b[tier2b.n_celltypes_expressed > 0]

    summary = {
        "tier2a": {
            "total_kinases": len(tier2a),
            "kinases_with_substrates": len(matched_2a),
            "median_substrates_matched": float(matched_2a.n_substrates_matched.median()),
            "mean_specificity": float(matched_2a.specificity.mean()),
            "dominant_ct_counts": matched_2a.dominant_ct.value_counts().to_dict(),
        },
        "tier2b": {
            "total_kinases": len(tier2b),
            "kinases_expressed": len(expressed_2b),
            "expression_breadth": tier2b.n_celltypes_expressed.value_counts().sort_index().to_dict(),
            "ct_specific_count": int(tier2b.ct_specific.notna().sum()),
        },
        "tier3": {
            "total_sig_entries": len(tier3),
            "unique_sig_kinases": int(tier3.kinase.nunique()),
            "entries_with_concordance": int((tier3.n_concordant_cts > 0).sum()),
            "pct_with_concordance": float((tier3.n_concordant_cts > 0).mean()),
            "mean_concordant_cts": float(tier3.n_concordant_cts.mean()),
        },
    }

    with open(SUMMARY_FILE, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary → {SUMMARY_FILE}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Tier 2–3 annotation pipeline")
    parser.add_argument("--tier2a", action="store_true", help="Compute Tier 2a substrate concentration")
    parser.add_argument("--tier2b", action="store_true", help="Compute Tier 2b kinase expression")
    parser.add_argument("--tier3", action="store_true", help="Compute Tier 3 cross-reference")
    parser.add_argument("--all", action="store_true", help="Run full pipeline (2a + 2b + 3)")
    parser.add_argument("--summary", action="store_true", help="Print cached results")
    args = parser.parse_args()

    if args.summary:
        print_summary()
        return

    if not any([args.tier2a, args.tier2b, args.tier3, args.all]):
        parser.print_help()
        return

    tier2a = tier2b = tier3 = None

    if args.tier2a or args.all:
        tier2a = compute_tier2a()

    if args.tier2b or args.all:
        tier2b = compute_tier2b()

    if args.tier3 or args.all:
        tier3 = compute_tier3(tier2a, tier2b)

    if args.all and tier2a is not None and tier2b is not None and tier3 is not None:
        save_summary(tier2a, tier2b, tier3)


if __name__ == "__main__":
    main()

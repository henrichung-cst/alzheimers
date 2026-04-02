#!/usr/bin/env python3
"""Module 2: Cross-modal kinase triangulation.

Joins Tier 1 bulk phospho kinase enrichment with Module 1 cell-type-resolved
RNA differential expression to produce condition-specific concordance calls.

See sap_rescue.md Module 2 for specification.

Usage:
    python code/sap_module2_triangulation.py --run        # full pipeline
    python code/sap_module2_triangulation.py --permtest   # permutation test for concordance
    python code/sap_module2_triangulation.py --profiles   # detailed hit profiles
    python code/sap_module2_triangulation.py --summary    # print cached results
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

import config

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OUTPUT_DIR = os.path.join("outputs", "reports", "module2_triangulation")
FULL_JOIN_FILE = os.path.join(OUTPUT_DIR, "full_join.csv")
CONCORDANT_FILE = os.path.join(OUTPUT_DIR, "concordant.csv")
SUMMARY_FILE = os.path.join(OUTPUT_DIR, "summary.csv")
META_FILE = os.path.join(OUTPUT_DIR, "module2_summary.json")
PERMTEST_FILE = os.path.join(OUTPUT_DIR, "permutation_test.json")
PROFILES_FILE = os.path.join(OUTPUT_DIR, "hit_profiles.csv")

# Input files
TIER1_FILE = os.path.join("outputs", "bulk", "kinase_results.csv")
MODULE1_COND_FILE = os.path.join(
    "outputs", "reports", "module1_kinase_de", "condition_contrasts.csv",
)
TIER2A_FILE = os.path.join(
    "outputs", "reports", "tier_annotation", "tier2a_substrate_concentration.csv",
)
TIER2B_FILE = os.path.join(
    "outputs", "reports", "tier_annotation", "tier2b_kinase_expression.csv",
)

# Condition mapping: Tier 1 condition name → Module 1 condition contrast name
CONDITION_MAP = {
    "AppP": "AppP",
    "Ttau": "Ttau",
    "ApTt": "ApTt",
}

# Concordance thresholds
RNA_FDR_STRICT = 0.05
RNA_FDR_RELAXED = 0.10
RNA_PVAL_NOMINAL = 0.05


# ---------------------------------------------------------------------------
# Kinase-to-gene mapping
# ---------------------------------------------------------------------------


def load_kinase_gene_map() -> pd.DataFrame:
    """Load kinase abbreviation → mouse gene symbol mapping.

    Returns DataFrame with columns: kinase, gene_symbol (mouse title-case).
    """
    cache_path = config.MAPPING_CACHE_FILE
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"Kinase-gene mapping not found: {cache_path}")

    mapping = pd.read_csv(cache_path)
    # Cache has: kinase_abbreviation, gene_symbol (human uppercase)
    # Convert to mouse title-case
    mapping = mapping.rename(columns={"kinase_abbreviation": "kinase"})
    mapping["gene_mouse"] = mapping["gene_symbol"].apply(
        lambda g: g[0].upper() + g[1:].lower() if isinstance(g, str) and len(g) > 1
        else (g.upper() if isinstance(g, str) else None)
    )
    return mapping[["kinase", "gene_symbol", "gene_mouse"]].dropna()


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_tier1() -> pd.DataFrame:
    """Load Tier 1 bulk enrichment results (significant entries only)."""
    df = pd.read_csv(TIER1_FILE)
    sig = df[df["significance_tier"] == "significant"].copy()
    # Keep relevant columns
    sig = sig[["kinase", "condition", "timepoint", "direction", "lff", "adj_pval"]].copy()
    sig = sig.rename(columns={"lff": "phospho_lff", "adj_pval": "phospho_pval"})
    return sig


def load_module1_conditions() -> pd.DataFrame:
    """Load Module 1 condition-vs-WT contrasts."""
    df = pd.read_csv(MODULE1_COND_FILE)
    df = df.rename(columns={
        "log2fc": "rna_log2fc",
        "se": "rna_se",
        "pval_raw": "rna_pval_raw",
        "pval_adj": "rna_pval_adj",
    })
    return df


# ---------------------------------------------------------------------------
# Join and concordance
# ---------------------------------------------------------------------------


def build_full_join(
    tier1: pd.DataFrame,
    module1: pd.DataFrame,
    kinase_map: pd.DataFrame,
) -> pd.DataFrame:
    """Join Tier 1 phospho enrichment with Module 1 RNA DE.

    Each Tier 1 entry (kinase × condition × timepoint) is joined with
    Module 1 results (gene × cell_type × condition) via the kinase-gene mapping.
    This expands each Tier 1 entry to 5 rows (one per cell type).
    """
    # Map Tier 1 kinase names to mouse gene symbols
    tier1_with_gene = tier1.merge(
        kinase_map[["kinase", "gene_mouse"]],
        on="kinase",
        how="left",
    )
    # Drop entries with no gene mapping
    n_unmapped = tier1_with_gene["gene_mouse"].isna().sum()
    if n_unmapped > 0:
        print(f"  Warning: {n_unmapped}/{len(tier1_with_gene)} Tier 1 entries have no gene mapping")
    tier1_with_gene = tier1_with_gene.dropna(subset=["gene_mouse"])

    # Join on gene × condition
    joined = tier1_with_gene.merge(
        module1,
        left_on=["gene_mouse", "condition"],
        right_on=["gene", "condition"],
        how="left",
    )

    # Some kinases may not have Module 1 data (gene not in aggexp)
    n_no_rna = joined["rna_log2fc"].isna().sum()
    n_with_rna = joined["rna_log2fc"].notna().sum()
    print(f"  Joined: {n_with_rna} entries with RNA data, {n_no_rna} without")

    return joined


def compute_concordance(joined: pd.DataFrame) -> pd.DataFrame:
    """Compute concordance status for each joined entry.

    Adds columns:
    - direction_match: phospho and RNA effects have same sign
    - concordance_strict: Tier 1 significant AND RNA FDR < 0.05 AND direction match
    - concordance_relaxed: ... AND RNA FDR < 0.10
    - concordance_nominal: ... AND RNA raw p < 0.05
    - score: -log10(phospho_pval) * -log10(rna_pval_raw) for concordant entries
    """
    df = joined.copy()

    # Direction matching: positive phospho LFF with positive RNA log2FC, or both negative
    phospho_sign = np.sign(df["phospho_lff"].values)
    rna_sign = np.sign(df["rna_log2fc"].values)
    df["direction_match"] = (phospho_sign == rna_sign) & (phospho_sign != 0) & (rna_sign != 0)

    # Concordance at different RNA thresholds
    has_rna = df["rna_pval_adj"].notna()
    df["concordance_strict"] = (
        has_rna
        & (df["rna_pval_adj"] <= RNA_FDR_STRICT)
        & df["direction_match"]
    )
    df["concordance_relaxed"] = (
        has_rna
        & (df["rna_pval_adj"] <= RNA_FDR_RELAXED)
        & df["direction_match"]
    )
    df["concordance_nominal"] = (
        has_rna
        & (df["rna_pval_raw"] <= RNA_PVAL_NOMINAL)
        & df["direction_match"]
    )

    # Status label
    df["status"] = "neither"
    df.loc[has_rna & (df["rna_pval_raw"] <= RNA_PVAL_NOMINAL) & ~df["direction_match"], "status"] = "discordant"
    df.loc[has_rna & (df["rna_pval_raw"] > RNA_PVAL_NOMINAL), "status"] = "phospho_only"
    df.loc[df["concordance_nominal"], "status"] = "concordant_nominal"
    df.loc[df["concordance_relaxed"], "status"] = "concordant_relaxed"
    df.loc[df["concordance_strict"], "status"] = "concordant_strict"

    # Combined evidence score (for all entries with RNA data)
    df["score"] = np.nan
    valid = has_rna & (df["phospho_pval"] > 0) & (df["rna_pval_raw"] > 0)
    df.loc[valid, "score"] = (
        -np.log10(df.loc[valid, "phospho_pval"].values)
        * -np.log10(df.loc[valid, "rna_pval_raw"].values)
    )

    return df


# ---------------------------------------------------------------------------
# Tier 2a/2b annotation
# ---------------------------------------------------------------------------


def annotate_with_tiers(df: pd.DataFrame) -> pd.DataFrame:
    """Add Tier 2a substrate concentration and Tier 2b expression data."""
    out = df.copy()

    # Tier 2a: substrate concentration enrichment
    if os.path.exists(TIER2A_FILE):
        t2a = pd.read_csv(TIER2A_FILE)
        # t2a has columns: kinase, enrich_{CT}, dominant_ct, specificity, ...
        # Join enrichment for the matching cell type
        enrich_cols = [c for c in t2a.columns if c.startswith("enrich_")]
        ct_col_map = {}
        for col in enrich_cols:
            ct_name = col.replace("enrich_", "")
            ct_col_map[ct_name] = col

        # Wide-to-long for matching
        t2a_long = []
        for _, row in t2a.iterrows():
            for ct_name, col in ct_col_map.items():
                t2a_long.append({
                    "kinase": row["kinase"],
                    "cell_type_short": ct_name,
                    "substrate_enrichment": row[col],
                })
        t2a_long = pd.DataFrame(t2a_long)

        # Map cell type names to match Module 1 format
        ct_name_map = {ct.split("_")[0]: ct for ct in config.SAP_ESTIMATED_CELLTYPES}
        t2a_long["cell_type"] = t2a_long["cell_type_short"].map(ct_name_map)
        t2a_long = t2a_long.dropna(subset=["cell_type"])

        out = out.merge(
            t2a_long[["kinase", "cell_type", "substrate_enrichment"]],
            on=["kinase", "cell_type"],
            how="left",
        )
    else:
        out["substrate_enrichment"] = np.nan

    # Tier 2b: kinase expression
    if os.path.exists(TIER2B_FILE):
        t2b = pd.read_csv(TIER2B_FILE)
        r_cols = [c for c in t2b.columns if c.startswith("R_")]
        ct_col_map = {}
        for col in r_cols:
            ct_name = col.replace("R_", "")
            ct_col_map[ct_name] = col

        t2b_long = []
        for _, row in t2b.iterrows():
            for ct_name, col in ct_col_map.items():
                t2b_long.append({
                    "kinase": row["kinase"],
                    "cell_type_short": ct_name,
                    "kinase_expressed": bool(row[col]),
                })
        t2b_long = pd.DataFrame(t2b_long)

        ct_name_map = {ct.split("_")[0]: ct for ct in config.SAP_ESTIMATED_CELLTYPES}
        t2b_long["cell_type"] = t2b_long["cell_type_short"].map(ct_name_map)
        t2b_long = t2b_long.dropna(subset=["cell_type"])

        out = out.merge(
            t2b_long[["kinase", "cell_type", "kinase_expressed"]],
            on=["kinase", "cell_type"],
            how="left",
        )
    else:
        out["kinase_expressed"] = np.nan

    return out


# ---------------------------------------------------------------------------
# Permutation test for concordance rate
# ---------------------------------------------------------------------------


def _permute_condition_contrasts(
    cpm_kp: "pd.DataFrame",
    sample_meta: pd.DataFrame,
    rng: np.random.RandomState,
) -> pd.DataFrame:
    """Refit Module 1 condition contrasts with permuted condition labels.

    Restricted permutation: shuffle condition labels within each
    sex × timepoint stratum (preserving nuisance structure).
    """
    from sap_module1_de import (
        CONTRAST_NAMES,
        IDX_APP,
        IDX_INT,
        IDX_TAU,
        build_design_matrix,
    )
    from statsmodels.stats.multitest import multipletests

    # Permute conditions within strata
    sm_perm = sample_meta.copy()
    for (gender, tp), grp in sm_perm.groupby(["gender", "timepoint"]):
        perm_conds = rng.permutation(grp["condition"].values)
        sm_perm.loc[grp.index, "condition"] = perm_conds

    # Rebuild design matrix with permuted conditions
    X, _ = build_design_matrix(sm_perm)
    sample_order = sm_perm.index.tolist()
    n, p = X.shape
    df_resid = n - p
    XtX_inv = np.linalg.inv(X.T @ X)

    # Contrast vectors for condition-vs-WT
    c_app = np.zeros(p); c_app[IDX_APP] = 1.0
    c_tau = np.zeros(p); c_tau[IDX_TAU] = 1.0
    c_aptt = np.zeros(p); c_aptt[IDX_APP] = 1.0; c_aptt[IDX_TAU] = 1.0; c_aptt[IDX_INT] = 1.0
    contrasts = [("AppP", c_app), ("Ttau", c_tau), ("ApTt", c_aptt)]

    all_rows = []
    for ct in config.SAP_ESTIMATED_CELLTYPES:
        ct_df = cpm_kp.loc[ct].loc[sample_order]
        genes = ct_df.columns.tolist()
        Y = np.log2(ct_df.values.T + 1)  # (n_genes, 24)

        beta_hat = (XtX_inv @ X.T @ Y.T).T
        residuals = Y - beta_hat @ X.T
        sigma2 = np.sum(residuals**2, axis=1) / df_resid

        for cond_name, c_vec in contrasts:
            lfc = beta_hat @ c_vec
            var_factor = float(c_vec @ XtX_inv @ c_vec)
            se_cond = np.sqrt(sigma2 * var_factor)
            t_cond = lfc / np.where(se_cond > 0, se_cond, np.inf)
            pval_cond = 2.0 * sp_stats.t.sf(np.abs(t_cond), df=df_resid)

            for gi, gene in enumerate(genes):
                all_rows.append({
                    "gene": gene,
                    "cell_type": ct,
                    "condition": cond_name,
                    "log2fc": float(lfc[gi]),
                    "se": float(se_cond[gi]),
                    "pval_raw": float(pval_cond[gi]),
                })

    df = pd.DataFrame(all_rows)
    df["pval_adj"] = np.nan
    for ct in config.SAP_ESTIMATED_CELLTYPES:
        mask = df["cell_type"] == ct
        _, pvals_adj, _, _ = multipletests(df.loc[mask, "pval_raw"].values, alpha=0.05, method="fdr_bh")
        df.loc[mask, "pval_adj"] = pvals_adj

    # Rename to match Module 1 condition_contrasts format expected by build_full_join
    df = df.rename(columns={
        "log2fc": "rna_log2fc",
        "se": "rna_se",
        "pval_raw": "rna_pval_raw",
        "pval_adj": "rna_pval_adj",
    })
    return df


def _count_concordant(
    tier1: pd.DataFrame,
    module1_perm: pd.DataFrame,
    kinase_map: pd.DataFrame,
) -> int:
    """Count nominal concordant entries for a permuted Module 1 result."""
    joined = build_full_join(tier1, module1_perm, kinase_map)
    joined = compute_concordance(joined)
    return int(joined["concordance_nominal"].sum())


def run_permutation_test(n_perm: int = 1000, seed: int = 42) -> None:
    """Restricted permutation test for concordance rate.

    Permutes condition labels within sex × timepoint strata, refits
    Module 1 OLS, rejoins with Tier 1, and counts concordant entries.
    Compares observed count to null distribution.
    """
    import sap_data

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load data
    print("Loading data for permutation test...")
    kinase_map = load_kinase_gene_map()
    tier1 = load_tier1()

    # Load CPM data (same as Module 1)
    aggexp = sap_data.load_aggexp_pooled()
    sample_map = sap_data.build_aggexp_sample_map(aggexp)
    _, _, _, cpm_kp = sap_data.preprocess_aggexp(aggexp, sample_map, return_cpm=True)

    # Build sample_meta
    rows = []
    for _, sid in sorted(sample_map.items()):
        parts = sid.split("_")
        rows.append({"sample_id": sid, "gender": parts[0], "timepoint": parts[1], "condition": parts[2]})
    sample_meta = pd.DataFrame(rows).set_index("sample_id")

    # Observed concordance count
    module1_obs = load_module1_conditions()
    n_observed = _count_concordant(tier1, module1_obs, kinase_map)
    print(f"  Observed concordant entries (nominal): {n_observed}")

    # Permutation null
    print(f"\n  Running {n_perm} permutations (restricted within sex × timepoint strata)...")
    rng = np.random.RandomState(seed)
    null_counts = np.zeros(n_perm, dtype=int)

    for i in range(n_perm):
        if (i + 1) % 100 == 0:
            print(f"    Permutation {i + 1}/{n_perm}...")
        module1_perm = _permute_condition_contrasts(cpm_kp, sample_meta, rng)
        null_counts[i] = _count_concordant(tier1, module1_perm, kinase_map)

    # Empirical p-value (one-sided: how often does null >= observed?)
    n_ge = np.sum(null_counts >= n_observed)
    pval_emp = (n_ge + 1) / (n_perm + 1)  # conservative correction

    result = {
        "n_permutations": n_perm,
        "seed": seed,
        "observed_concordant": n_observed,
        "null_mean": float(np.mean(null_counts)),
        "null_median": float(np.median(null_counts)),
        "null_std": float(np.std(null_counts)),
        "null_95th": float(np.percentile(null_counts, 95)),
        "null_99th": float(np.percentile(null_counts, 99)),
        "null_max": int(np.max(null_counts)),
        "null_min": int(np.min(null_counts)),
        "n_null_ge_observed": int(n_ge),
        "empirical_pval": float(pval_emp),
        "enrichment_ratio": float(n_observed / np.mean(null_counts)) if np.mean(null_counts) > 0 else float("inf"),
        "null_distribution": null_counts.tolist(),
    }

    with open(PERMTEST_FILE, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n  {'=' * 60}")
    print(f"  Permutation test results")
    print(f"  {'=' * 60}")
    print(f"  Observed concordant:   {n_observed}")
    print(f"  Null mean ± std:       {result['null_mean']:.1f} ± {result['null_std']:.1f}")
    print(f"  Null 95th percentile:  {result['null_95th']:.0f}")
    print(f"  Null 99th percentile:  {result['null_99th']:.0f}")
    print(f"  Null max:              {result['null_max']}")
    print(f"  Enrichment ratio:      {result['enrichment_ratio']:.2f}x")
    print(f"  Empirical p-value:     {result['empirical_pval']:.4f}")
    print(f"  Saved: {PERMTEST_FILE}")


# ---------------------------------------------------------------------------
# Detailed hit profiles
# ---------------------------------------------------------------------------


def build_hit_profiles(
    full_join: pd.DataFrame,
    concordant: pd.DataFrame,
) -> pd.DataFrame:
    """Build detailed profiles for top concordant kinases.

    For each concordant kinase, reports per cell-type and condition:
    phospho p-value/LFF, RNA log2FC/p-values, substrate enrichment,
    kinase expression, concordance score.
    """
    if len(concordant) == 0:
        print("  No concordant entries to profile.")
        return pd.DataFrame()

    # Get per-kinase summary to rank
    kinase_ranks = concordant.groupby("kinase").agg(
        n_concordant=("kinase", "size"),
        max_score=("score", "max"),
    ).sort_values("max_score", ascending=False)

    # Build profile for all concordant entries, ranked
    profiles = concordant[[
        "kinase", "gene_mouse", "condition", "timepoint", "cell_type",
        "phospho_lff", "phospho_pval",
        "rna_log2fc", "rna_se", "rna_pval_raw", "rna_pval_adj",
        "direction_match", "score",
        "substrate_enrichment", "kinase_expressed",
    ]].copy()

    # Add rank
    profiles["kinase_rank"] = profiles["kinase"].map(
        {k: i + 1 for i, k in enumerate(kinase_ranks.index)}
    )
    profiles = profiles.sort_values(["kinase_rank", "score"], ascending=[True, False])

    return profiles


def print_hit_profiles(profiles: pd.DataFrame) -> None:
    """Print formatted hit profiles for top kinases."""
    if len(profiles) == 0:
        print("No concordant entries to profile.")
        return

    kinases_seen = set()
    for _, row in profiles.iterrows():
        k = row["kinase"]
        if k not in kinases_seen:
            if kinases_seen:
                print()
            kinases_seen.add(k)
            k_rows = profiles[profiles["kinase"] == k]
            print(f"  {'─' * 70}")
            print(f"  #{int(row['kinase_rank']):d}  {k} ({row['gene_mouse']})  "
                  f"— {len(k_rows)} concordant entries")
            print(f"  {'─' * 70}")

        expressed = "yes" if row.get("kinase_expressed") else ("no" if row.get("kinase_expressed") is False else "?")
        sub_enr = f"{row['substrate_enrichment']:.2f}" if pd.notna(row.get("substrate_enrichment")) else "N/A"
        print(f"    {row['condition']:5s} {row['timepoint']:4s} {row['cell_type'][:15]:15s}  "
              f"phospho: LFF={row['phospho_lff']:+.3f} p={row['phospho_pval']:.2e}  "
              f"RNA: log2FC={row['rna_log2fc']:+.4f} p_raw={row['rna_pval_raw']:.2e} "
              f"q={row['rna_pval_adj']:.2e}  "
              f"score={row['score']:.1f}  expr={expressed}  sub_enr={sub_enr}")


def run_profiles() -> None:
    """Generate and print detailed hit profiles from cached results."""
    if not os.path.exists(FULL_JOIN_FILE):
        print("No cached results. Run --run first.")
        return

    full_join = pd.read_csv(FULL_JOIN_FILE)
    concordant = full_join[full_join["concordance_nominal"]].copy()
    concordant = concordant.sort_values("score", ascending=False)

    profiles = build_hit_profiles(full_join, concordant)
    if len(profiles) > 0:
        profiles.to_csv(PROFILES_FILE, index=False)
        print(f"  Saved: {PROFILES_FILE} ({len(profiles)} entries)\n")
        print_hit_profiles(profiles)


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------


def build_summary(concordant: pd.DataFrame) -> pd.DataFrame:
    """Per-kinase summary: which cell types are concordant under which conditions."""
    if len(concordant) == 0:
        return pd.DataFrame(columns=[
            "kinase", "gene", "n_concordant_entries",
            "cell_types", "conditions", "mean_score",
        ])

    groups = concordant.groupby("kinase").agg(
        gene=("gene_mouse", "first"),
        n_concordant_entries=("kinase", "size"),
        cell_types=("cell_type", lambda x: ", ".join(sorted(x.unique()))),
        conditions=("condition", lambda x: ", ".join(sorted(x.unique()))),
        mean_score=("score", "mean"),
        max_score=("score", "max"),
    ).reset_index().sort_values("max_score", ascending=False)

    return groups


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run_pipeline() -> None:
    """Run the full Module 2 triangulation pipeline."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load data
    print("Loading kinase-gene mapping...")
    kinase_map = load_kinase_gene_map()
    print(f"  {len(kinase_map)} kinase-gene pairs")

    print("\nLoading Tier 1 bulk enrichment...")
    tier1 = load_tier1()
    print(f"  {len(tier1)} significant entries, {tier1['kinase'].nunique()} unique kinases")

    print("\nLoading Module 1 condition contrasts...")
    module1 = load_module1_conditions()
    print(f"  {len(module1)} entries across {module1['cell_type'].nunique()} cell types")

    # Step 1: Join
    print("\n" + "=" * 60)
    print("Step 1: Join Tier 1 with Module 1")
    print("=" * 60)
    full_join = build_full_join(tier1, module1, kinase_map)

    # Step 2: Concordance
    print("\n" + "=" * 60)
    print("Step 2: Compute concordance")
    print("=" * 60)
    full_join = compute_concordance(full_join)

    # Step 3: Annotate with Tier 2a/2b
    print("\n" + "=" * 60)
    print("Step 3: Annotate with Tier 2a/2b")
    print("=" * 60)
    full_join = annotate_with_tiers(full_join)

    # Save full join
    full_join.to_csv(FULL_JOIN_FILE, index=False)
    print(f"\n  Full join saved: {FULL_JOIN_FILE} ({len(full_join)} rows)")

    # Concordant table (nominal threshold — the most permissive concordance level)
    concordant = full_join[full_join["concordance_nominal"]].copy()
    concordant = concordant.sort_values("score", ascending=False)
    concordant.to_csv(CONCORDANT_FILE, index=False)
    print(f"  Concordant saved: {CONCORDANT_FILE} ({len(concordant)} rows)")

    # Summary
    summary_df = build_summary(concordant)
    summary_df.to_csv(SUMMARY_FILE, index=False)
    print(f"  Summary saved: {SUMMARY_FILE} ({len(summary_df)} rows)")

    # Meta/stats
    meta = {
        "tier1_entries": len(tier1),
        "tier1_kinases": int(tier1["kinase"].nunique()),
        "full_join_entries": len(full_join),
        "concordance_counts": {
            "strict_FDR005": int(full_join["concordance_strict"].sum()),
            "relaxed_FDR010": int(full_join["concordance_relaxed"].sum()),
            "nominal_pval005": int(full_join["concordance_nominal"].sum()),
        },
        "status_distribution": full_join["status"].value_counts().to_dict(),
        "concordant_kinases_nominal": int(concordant["kinase"].nunique()),
    }
    with open(META_FILE, "w") as f:
        json.dump(meta, f, indent=2)

    # Print results
    print("\n" + "=" * 60)
    print("Results Summary")
    print("=" * 60)
    print(f"  Full join: {len(full_join)} entries")
    print(f"  Status distribution:")
    for status, count in full_join["status"].value_counts().items():
        print(f"    {status}: {count}")
    print(f"\n  Concordance counts:")
    print(f"    Strict  (RNA FDR < 0.05): {meta['concordance_counts']['strict_FDR005']}")
    print(f"    Relaxed (RNA FDR < 0.10): {meta['concordance_counts']['relaxed_FDR010']}")
    print(f"    Nominal (RNA p < 0.05):   {meta['concordance_counts']['nominal_pval005']}")

    if len(concordant) > 0:
        print(f"\n  Top concordant kinases (nominal threshold):")
        for _, row in summary_df.head(10).iterrows():
            print(f"    {row['kinase']:12s} ({row['gene']:8s}) "
                  f"n={row['n_concordant_entries']:2d}  "
                  f"CTs=[{row['cell_types']}]  "
                  f"score={row['max_score']:.2f}")
    else:
        print("\n  No concordant entries at nominal threshold.")
        print("  The full join table contains all entries for exploratory analysis.")


def print_summary() -> None:
    """Print cached summary."""
    if not os.path.exists(META_FILE):
        print("No cached results. Run --run first.")
        return
    with open(META_FILE) as f:
        meta = json.load(f)
    print(json.dumps(meta, indent=2))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Module 2: Cross-modal kinase triangulation")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run", action="store_true", help="Run full triangulation pipeline")
    group.add_argument("--permtest", action="store_true", help="Permutation test for concordance rate")
    group.add_argument("--profiles", action="store_true", help="Detailed hit profiles")
    group.add_argument("--summary", action="store_true", help="Print cached results")
    parser.add_argument("--n-perm", type=int, default=1000, help="Number of permutations (default: 1000)")
    args = parser.parse_args()

    if args.summary:
        print_summary()
    elif args.run:
        run_pipeline()
    elif args.permtest:
        run_permutation_test(n_perm=args.n_perm)
    elif args.profiles:
        run_profiles()


if __name__ == "__main__":
    main()

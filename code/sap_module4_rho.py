#!/usr/bin/env python3
"""Module 4: RNA-Phospho coupling landscape.

Characterizes the distribution of rho_j (RNA covariate coefficients) across
the phosphoproteome and identifies biological structure in RNA-coupled vs
RNA-decoupled sites.

See sap_rescue.md Module 4 for specification.

Usage:
    python code/sap_module4_rho.py --run          # full analysis
    python code/sap_module4_rho.py --aptt-detail  # ApTt decoupling breakdown
    python code/sap_module4_rho.py --summary      # print cached results
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from statsmodels.stats.multitest import multipletests

import config
import sap_data
import sap_model

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OUTPUT_DIR = os.path.join("outputs", "reports", "module4_rho_landscape")
RHO_DIST_FILE = os.path.join(OUTPUT_DIR, "rho_distribution.csv")
KINASE_SUMMARY_FILE = os.path.join(OUTPUT_DIR, "kinase_rho_summary.csv")
FAMILY_ENRICH_FILE = os.path.join(OUTPUT_DIR, "kinase_family_enrichment.csv")
CONDITION_COUPLING_FILE = os.path.join(OUTPUT_DIR, "condition_coupling.csv")
APTT_DETAIL_FILE = os.path.join(OUTPUT_DIR, "aptt_decoupling_detail.csv")
APTT_FAMILY_FILE = os.path.join(OUTPUT_DIR, "aptt_family_breakdown.csv")
APTT_META_FILE = os.path.join(OUTPUT_DIR, "aptt_detail_summary.json")
META_FILE = os.path.join(OUTPUT_DIR, "module4_summary.json")

# Default threshold for RNA-coupled classification
DEFAULT_TAU = 1.0

# Manually curated kinase families for enrichment testing.
# Names match kldata KINASE abbreviations (uppercase).
# Activity-regulated: activated by second messengers, not transcription
ACTIVITY_REGULATED = {
    "CaMKII": ["CAMK2A", "CAMK2B", "CAMK2D", "CAMK2G"],
    "PKC": ["PKCA", "PKCB", "PKCD", "PKCE", "PKCG", "PKCH", "PKCI", "PKCT", "PKCZ"],
    "AMPK": ["AMPKA1", "AMPKA2"],
    "CaMK4": ["CAMK4"],
    "CaMK1": ["CAMK1A", "CAMK1D", "CAMK1G"],
    "PKA": ["PKACA", "PKACB"],
}

# Transcriptionally regulated: downstream of transcriptional programs
TRANSCRIPTIONALLY_REGULATED = {
    "CDK": ["CDK1", "CDK2", "CDK4", "CDK5", "CDK6", "CDK7", "CDK9"],
    "MAPK": ["ERK1", "ERK2", "JNK1", "JNK2", "JNK3", "P38A", "P38B", "P38D", "P38G"],
    "JAK": ["JAK1", "JAK2", "JAK3"],
}


# ---------------------------------------------------------------------------
# Rho extraction and classification
# ---------------------------------------------------------------------------


def extract_rho(model: sap_model.ModelFit, data: sap_data.SAPData) -> pd.DataFrame:
    """Extract per-site rho values with metadata."""
    n = len(model.site_params)
    rho = np.array([sp.rho for sp in model.site_params])
    converged = model.converged if model.converged is not None else np.zeros(n, dtype=bool)

    df = pd.DataFrame({
        "site_idx": range(n),
        "rho": rho,
        "converged": converged,
    })

    # Add site metadata
    site_meta = data.site_meta
    for col in ["site_id", "gene_symbol", "site_position"]:
        if col in site_meta.columns:
            df[col] = site_meta[col].values[:n]

    return df


def classify_sites(rho_df: pd.DataFrame, tau: float = DEFAULT_TAU) -> pd.DataFrame:
    """Classify sites as RNA-coupled positive, negative, or decoupled.

    Args:
        rho_df: DataFrame with 'rho' column.
        tau: threshold for classification. |rho| <= tau → decoupled.
    """
    df = rho_df.copy()
    df["category"] = "decoupled"
    df.loc[df["rho"] > tau, "category"] = "coupled_positive"
    df.loc[df["rho"] < -tau, "category"] = "coupled_negative"
    return df


# ---------------------------------------------------------------------------
# Kinase-substrate mapping (site → kinase abbreviations)
# ---------------------------------------------------------------------------


def build_site_to_kinase_map(site_meta: pd.DataFrame) -> Dict[int, List[str]]:
    """Map site indices to kinase abbreviations from kldata.csv.

    Returns KINASE abbreviations (uppercase, matching family definitions)
    for all kinases, not filtered to genes in gkp.
    """
    kldata = pd.read_csv(
        config.KLDATA_FILE,
        usecols=["gene_symbol", "site_position", "KINASE"],
        low_memory=False,
    )
    kldata = kldata.dropna(subset=["KINASE"])

    # Build lookup: (substrate_gene, site_pos) → set of kinase abbreviations
    substrate_to_kinases: Dict[Tuple[str, str], set] = {}
    for _, row in kldata.iterrows():
        key = (row["gene_symbol"], str(row["site_position"]))
        substrate_to_kinases.setdefault(key, set()).add(row["KINASE"])

    # Match against site_meta
    result: Dict[int, List[str]] = {}
    for idx in range(len(site_meta)):
        gene = site_meta.iloc[idx].get("gene_symbol", "")
        pos = str(site_meta.iloc[idx].get("site_position", ""))
        kinases = substrate_to_kinases.get((gene, pos), set())
        if kinases:
            result[idx] = sorted(kinases)

    return result


# ---------------------------------------------------------------------------
# Kinase coupling profiles
# ---------------------------------------------------------------------------


def compute_kinase_coupling(
    classified: pd.DataFrame,
    site_kinase_map: Dict[int, List[str]],
) -> pd.DataFrame:
    """Per-kinase coupling profile: fraction of substrates in each category."""
    # Collect per-kinase substrate categories
    kinase_data: Dict[str, Dict[str, int]] = {}
    for site_idx, kinases in site_kinase_map.items():
        if site_idx >= len(classified):
            continue
        cat = classified.iloc[site_idx]["category"]
        for k in kinases:
            if k not in kinase_data:
                kinase_data[k] = {"coupled_positive": 0, "coupled_negative": 0, "decoupled": 0, "total": 0}
            kinase_data[k][cat] += 1
            kinase_data[k]["total"] += 1

    rows = []
    for kinase, counts in sorted(kinase_data.items()):
        n = counts["total"]
        rows.append({
            "kinase_gene": kinase,
            "n_substrates": n,
            "frac_coupled_pos": counts["coupled_positive"] / n if n > 0 else 0,
            "frac_coupled_neg": counts["coupled_negative"] / n if n > 0 else 0,
            "frac_decoupled": counts["decoupled"] / n if n > 0 else 0,
            "frac_coupled_any": (counts["coupled_positive"] + counts["coupled_negative"]) / n if n > 0 else 0,
            "dominant_category": max(
                ["coupled_positive", "coupled_negative", "decoupled"],
                key=lambda c: counts[c],
            ),
        })

    return pd.DataFrame(rows).sort_values("n_substrates", ascending=False)


# ---------------------------------------------------------------------------
# Family enrichment tests
# ---------------------------------------------------------------------------


def test_family_enrichment(
    kinase_coupling: pd.DataFrame,
    classified: pd.DataFrame,
    site_kinase_map: Dict[int, List[str]],
) -> pd.DataFrame:
    """Fisher's exact test: are family substrates enriched in decoupled/coupled?

    Tests both activity-regulated (expected: more decoupled) and transcriptionally
    regulated (expected: more coupled) kinase families.
    """
    # Global category counts
    total_coupled = (classified["category"] != "decoupled").sum()
    total_decoupled = (classified["category"] == "decoupled").sum()

    results = []

    all_families = {
        **{f"activity_{name}": genes for name, genes in ACTIVITY_REGULATED.items()},
        **{f"transcriptional_{name}": genes for name, genes in TRANSCRIPTIONALLY_REGULATED.items()},
    }

    for family_label, gene_list in all_families.items():
        # Find sites that are substrates of any gene in this family
        family_sites = set()
        for site_idx, kinases in site_kinase_map.items():
            if site_idx < len(classified) and any(k in gene_list for k in kinases):
                family_sites.add(site_idx)

        if len(family_sites) < 5:
            continue

        family_cats = classified.iloc[list(family_sites)]["category"]
        fam_decoupled = (family_cats == "decoupled").sum()
        fam_coupled = (family_cats != "decoupled").sum()

        # 2x2 contingency: family vs rest × decoupled vs coupled
        # For activity-regulated: test enrichment in decoupled
        table = np.array([
            [fam_decoupled, fam_coupled],
            [total_decoupled - fam_decoupled, total_coupled - fam_coupled],
        ])

        expected_direction = "decoupled" if family_label.startswith("activity_") else "coupled"
        if expected_direction == "decoupled":
            _, pval = sp_stats.fisher_exact(table, alternative="greater")
        else:
            _, pval = sp_stats.fisher_exact(table, alternative="less")

        frac_decoupled = fam_decoupled / len(family_sites) if len(family_sites) > 0 else 0
        global_frac_decoupled = total_decoupled / (total_decoupled + total_coupled)

        results.append({
            "family": family_label,
            "expected_direction": expected_direction,
            "n_substrates": len(family_sites),
            "n_decoupled": int(fam_decoupled),
            "n_coupled": int(fam_coupled),
            "frac_decoupled": frac_decoupled,
            "global_frac_decoupled": global_frac_decoupled,
            "enrichment_ratio": frac_decoupled / global_frac_decoupled if global_frac_decoupled > 0 else 0,
            "pval_raw": pval,
        })

    df = pd.DataFrame(results)
    if len(df) > 0:
        _, pvals_adj, _, _ = multipletests(df["pval_raw"].values, alpha=0.05, method="fdr_bh")
        df["pval_adj"] = pvals_adj
        df["significant"] = df["pval_adj"] <= 0.05
    return df


# ---------------------------------------------------------------------------
# Condition-specificity of coupling
# ---------------------------------------------------------------------------


def compute_condition_specificity(
    model: sap_model.ModelFit,
    data: sap_data.SAPData,
) -> pd.DataFrame:
    """Test whether RNA-phospho coupling strength varies by condition.

    For each site, compute correlation between r_{k,i,j} and model residuals,
    stratified by condition. Reports per-site, per-condition correlation.
    """
    n_sites = len(model.site_params)
    n_samples = data.bulk_phospho.shape[1]

    # Precompute arrays
    sa = sap_model.SampleArrays.from_data(data.sample_meta, data.a_obs)
    conditions = data.sample_meta["condition"].values
    x_base_all = data.x_base.values  # (J, 6)
    y_all = data.bulk_phospho.values  # (J, 24)
    r_tensor = data.r_tensor  # (6, 24, J)

    # For each condition, get sample indices
    cond_indices = {}
    for c in config.SAP_CONDITIONS:
        cond_indices[c] = np.where(conditions == c)[0]

    rows = []
    n_assessed = min(n_sites, 2000)  # subsample for speed
    rng = np.random.RandomState(42)
    site_subset = rng.choice(n_sites, n_assessed, replace=False) if n_sites > n_assessed else np.arange(n_sites)

    for j in site_subset:
        sp = model.site_params[j]
        if r_tensor is None:
            continue

        # Compute model prediction
        mu_j = sap_model.compute_mu(
            sp, data.a_obs, data.sample_meta,
            x_base_all[j], r_tensor[:, :, j],
            cached_arrays=sa,
        )
        residuals = y_all[j] - mu_j

        # Per-condition correlation between r_tensor and residuals
        for c, idx in cond_indices.items():
            if len(idx) < 3:
                continue
            # Average r across cell types for this site
            r_mean = r_tensor[:5, idx, j].mean(axis=0)  # (n_cond_samples,)
            resid_c = residuals[idx]

            if np.std(r_mean) < 1e-10 or np.std(resid_c) < 1e-10:
                corr = 0.0
            else:
                corr = float(np.corrcoef(r_mean, resid_c)[0, 1])

            rows.append({
                "site_idx": int(j),
                "condition": c,
                "r_resid_corr": corr,
                "rho": sp.rho,
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# ApTt decoupling breakdown
# ---------------------------------------------------------------------------

# AD-relevant kinase families for targeted analysis
AD_RELEVANT_FAMILIES = {
    "GSK3": ["GSK3A", "GSK3B"],
    "CDK5": ["CDK5"],
    "CaMKII": ["CAMK2A", "CAMK2B", "CAMK2D", "CAMK2G"],
    "CK1": ["CK1A", "CK1D", "CK1E", "CK1G1", "CK1G2", "CK1G3"],
    "DYRK": ["DYRK1A", "DYRK1B", "DYRK2"],
    "MAPK_ERK": ["ERK1", "ERK2"],
    "MAPK_JNK": ["JNK1", "JNK2", "JNK3"],
    "MAPK_p38": ["P38A", "P38B", "P38D", "P38G"],
    "CDK_cell_cycle": ["CDK1", "CDK2", "CDK4", "CDK6"],
    "PKC": ["PKCA", "PKCB", "PKCD", "PKCE", "PKCG"],
    "AMPK": ["AMPKA1", "AMPKA2"],
    "PKA": ["PKACA", "PKACB"],
}


def run_aptt_detail() -> None:
    """Break down ApTt-specific decoupling by kinase family and motif.

    Reads cached condition_coupling.csv and rho_distribution.csv,
    then stratifies the ApTt negative coupling signal by:
    1. Kinase family — which families drive the negative coupling?
    2. Phosphosite motif context (proline-directed vs basophilic vs acidophilic)
    3. AD-relevant kinases specifically
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.exists(CONDITION_COUPLING_FILE) or not os.path.exists(RHO_DIST_FILE):
        print("Cached results not found. Run --run first.")
        return

    print("Loading cached results...")
    cond_df = pd.read_csv(CONDITION_COUPLING_FILE)
    rho_df = pd.read_csv(RHO_DIST_FILE)

    # Load model and data for site-to-kinase mapping
    print("Loading production model and data...")
    model = sap_model.load_model()
    data, _ = sap_data.load_all(include_rna=True)
    site_kinase_map = build_site_to_kinase_map(data.site_meta)

    # Merge rho_df site metadata into cond_df
    site_info = rho_df[["site_idx", "gene_symbol", "site_position", "category"]].copy()
    detail = cond_df.merge(site_info, on="site_idx", how="left")

    # Add kinase assignments
    detail["kinases"] = detail["site_idx"].map(
        lambda idx: ",".join(site_kinase_map.get(idx, []))
    )
    detail["has_kinase"] = detail["kinases"] != ""

    # --------------- 1. Per-family ApTt coupling ---------------
    print("\n" + "=" * 60)
    print("1. ApTt residual-RNA coupling by kinase family")
    print("=" * 60)

    aptt = detail[detail["condition"] == "ApTt"].copy()
    aptt_mapped = aptt[aptt["has_kinase"]].copy()

    # Expand: each site may map to multiple kinases
    family_rows = []
    for _, row in aptt_mapped.iterrows():
        for kinase in row["kinases"].split(","):
            for fam_name, fam_members in AD_RELEVANT_FAMILIES.items():
                if kinase in fam_members:
                    family_rows.append({
                        "site_idx": row["site_idx"],
                        "kinase": kinase,
                        "family": fam_name,
                        "r_resid_corr": row["r_resid_corr"],
                        "rho": row["rho"],
                        "gene_symbol": row.get("gene_symbol", ""),
                    })

    fam_df = pd.DataFrame(family_rows) if family_rows else pd.DataFrame()

    family_summary = []
    if len(fam_df) > 0:
        # Global ApTt mean for reference
        global_aptt_mean = float(aptt["r_resid_corr"].mean())
        global_aptt_std = float(aptt["r_resid_corr"].std())

        for fam_name in AD_RELEVANT_FAMILIES:
            fam_sites = fam_df[fam_df["family"] == fam_name]
            if len(fam_sites) < 3:
                continue
            vals = fam_sites["r_resid_corr"].dropna().values
            if len(vals) < 3:
                continue
            # One-sample t-test: is this family's ApTt coupling different from 0?
            t_stat, p_zero = sp_stats.ttest_1samp(vals, 0)
            # Compare to global mean
            t_vs_global, p_vs_global = sp_stats.ttest_1samp(vals, global_aptt_mean)

            family_summary.append({
                "family": fam_name,
                "n_sites": len(fam_sites),
                "n_unique_sites": int(fam_sites["site_idx"].nunique()),
                "mean_r_resid_corr": float(np.mean(vals)),
                "std_r_resid_corr": float(np.std(vals)),
                "median_r_resid_corr": float(np.median(vals)),
                "pval_vs_zero": float(p_zero),
                "pval_vs_global": float(p_vs_global),
                "more_negative_than_global": float(np.mean(vals)) < global_aptt_mean,
            })

        fam_summary_df = pd.DataFrame(family_summary).sort_values("mean_r_resid_corr")

        # BH correction on vs-zero p-values
        if len(fam_summary_df) > 0:
            _, padj, _, _ = multipletests(fam_summary_df["pval_vs_zero"].values, alpha=0.05, method="fdr_bh")
            fam_summary_df["pval_vs_zero_adj"] = padj

        fam_summary_df.to_csv(APTT_FAMILY_FILE, index=False)

        print(f"\n  Global ApTt mean r: {global_aptt_mean:+.4f} ± {global_aptt_std:.4f}")
        print(f"  {'Family':<18s}  {'n':>5s}  {'mean r':>8s}  {'p(≠0)':>10s}  {'q(≠0)':>10s}  {'vs global':>10s}")
        print(f"  {'─' * 75}")
        for _, row in fam_summary_df.iterrows():
            marker = "***" if row["pval_vs_zero_adj"] < 0.001 else ("**" if row["pval_vs_zero_adj"] < 0.01 else ("*" if row["pval_vs_zero_adj"] < 0.05 else ""))
            print(f"  {row['family']:<18s}  {row['n_sites']:5d}  {row['mean_r_resid_corr']:+8.4f}  "
                  f"{row['pval_vs_zero']:10.2e}  {row['pval_vs_zero_adj']:10.2e}  "
                  f"{row['pval_vs_global']:10.2e}  {marker}")
    else:
        print("  No kinase-mapped sites with ApTt coupling data.")
        fam_summary_df = pd.DataFrame()

    # --------------- 2. Motif context ---------------
    print("\n" + "=" * 60)
    print("2. ApTt coupling by phosphosite motif context")
    print("=" * 60)

    # Classify motifs using surrounding sequence context from kldata
    # We'll use a simpler proxy: kinase class implies motif preference
    # Pro-directed: CDK, MAPK (ERK, JNK, p38), DYRK, GSK3
    # Basophilic: CaMKII, PKC, PKA, AMPK
    # Acidophilic: CK1, CK2
    motif_classes = {
        "proline_directed": ["CDK5", "CDK_cell_cycle", "MAPK_ERK", "MAPK_JNK", "MAPK_p38", "DYRK", "GSK3"],
        "basophilic": ["CaMKII", "PKC", "PKA", "AMPK"],
        "acidophilic": ["CK1"],
    }

    if len(fam_df) > 0:
        fam_df_copy = fam_df.copy()
        fam_df_copy["motif_class"] = "other"
        for mclass, families in motif_classes.items():
            fam_df_copy.loc[fam_df_copy["family"].isin(families), "motif_class"] = mclass

        motif_summary = fam_df_copy.groupby("motif_class").agg(
            n_sites=("site_idx", "nunique"),
            mean_r=("r_resid_corr", "mean"),
            std_r=("r_resid_corr", "std"),
            median_r=("r_resid_corr", "median"),
        ).reset_index()

        print(f"\n  {'Motif class':<20s}  {'n sites':>8s}  {'mean r':>8s}  {'std':>8s}  {'median r':>8s}")
        print(f"  {'─' * 60}")
        for _, row in motif_summary.iterrows():
            print(f"  {row['motif_class']:<20s}  {row['n_sites']:8d}  {row['mean_r']:+8.4f}  "
                  f"{row['std_r']:8.4f}  {row['median_r']:+8.4f}")

        # Test: proline-directed vs basophilic
        pro = fam_df_copy[fam_df_copy["motif_class"] == "proline_directed"]["r_resid_corr"].dropna().values
        baso = fam_df_copy[fam_df_copy["motif_class"] == "basophilic"]["r_resid_corr"].dropna().values
        if len(pro) >= 3 and len(baso) >= 3:
            mw_stat, mw_pval = sp_stats.mannwhitneyu(pro, baso, alternative="two-sided")
            print(f"\n  Proline-directed vs basophilic (Mann-Whitney): U={mw_stat:.0f}, p={mw_pval:.3g}")
    else:
        motif_summary = pd.DataFrame()

    # --------------- 3. Concentration vs diffuse ---------------
    print("\n" + "=" * 60)
    print("3. Is ApTt decoupling concentrated or diffuse?")
    print("=" * 60)

    # Compare ApTt r distribution for kinase-mapped vs unmapped sites
    aptt_with = aptt[aptt["has_kinase"]]["r_resid_corr"].dropna().values
    aptt_without = aptt[~aptt["has_kinase"]]["r_resid_corr"].dropna().values
    print(f"\n  Kinase-mapped sites:   n={len(aptt_with):5d}  mean r={np.mean(aptt_with):+.4f}" if len(aptt_with) > 0 else "\n  Kinase-mapped sites:   n=0")
    print(f"  Unmapped sites:        n={len(aptt_without):5d}  mean r={np.mean(aptt_without):+.4f}" if len(aptt_without) > 0 else f"  Unmapped sites:        n=0")
    if len(aptt_with) >= 3 and len(aptt_without) >= 3:
        mw_stat, mw_pval = sp_stats.mannwhitneyu(aptt_with, aptt_without, alternative="two-sided")
        print(f"  Mann-Whitney: U={mw_stat:.0f}, p={mw_pval:.3g}")
    elif len(aptt_without) == 0:
        print("  (All sites have kinase mappings — comparison not applicable)")

    # Top/bottom 10% of ApTt coupling — which kinases are enriched?
    aptt_valid = aptt["r_resid_corr"].dropna().values
    q10 = np.percentile(aptt_valid, 10)
    q90 = np.percentile(aptt_valid, 90)
    most_negative = aptt[aptt["r_resid_corr"] <= q10]
    most_positive = aptt[aptt["r_resid_corr"] >= q90]

    def _top_kinases(subset_df: pd.DataFrame) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for _, row in subset_df.iterrows():
            for k in site_kinase_map.get(row["site_idx"], []):
                counts[k] = counts.get(k, 0) + 1
        return dict(sorted(counts.items(), key=lambda x: -x[1])[:15])

    neg_kinases = _top_kinases(most_negative)
    pos_kinases = _top_kinases(most_positive)

    print(f"\n  Most negative ApTt coupling (bottom 10%, r ≤ {q10:+.3f}):")
    print(f"    n={len(most_negative)} sites")
    for k, n in list(neg_kinases.items())[:10]:
        # Check if it's in any AD-relevant family
        ad_fam = next((f for f, ms in AD_RELEVANT_FAMILIES.items() if k in ms), "")
        tag = f"  [{ad_fam}]" if ad_fam else ""
        print(f"    {k:12s}  n={n:4d}{tag}")

    print(f"\n  Most positive ApTt coupling (top 10%, r ≥ {q90:+.3f}):")
    print(f"    n={len(most_positive)} sites")
    for k, n in list(pos_kinases.items())[:10]:
        ad_fam = next((f for f, ms in AD_RELEVANT_FAMILIES.items() if k in ms), "")
        tag = f"  [{ad_fam}]" if ad_fam else ""
        print(f"    {k:12s}  n={n:4d}{tag}")

    # --------------- 4. Condition contrast ---------------
    print("\n" + "=" * 60)
    print("4. Per-condition coupling for AD-relevant families")
    print("=" * 60)

    # For each family with enough sites, show coupling across all 4 conditions
    if len(fam_df) > 0:
        # Rebuild with all conditions, not just ApTt
        all_cond_fam = []
        mapped_detail = detail[detail["has_kinase"]].copy()
        for _, row in mapped_detail.iterrows():
            for kinase in row["kinases"].split(","):
                for fam_name, fam_members in AD_RELEVANT_FAMILIES.items():
                    if kinase in fam_members:
                        all_cond_fam.append({
                            "site_idx": row["site_idx"],
                            "condition": row["condition"],
                            "family": fam_name,
                            "r_resid_corr": row["r_resid_corr"],
                        })
        all_cond_df = pd.DataFrame(all_cond_fam)

        if len(all_cond_df) > 0:
            pivot = all_cond_df.groupby(["family", "condition"])["r_resid_corr"].agg(["mean", "count"]).reset_index()
            pivot_wide = pivot.pivot(index="family", columns="condition", values="mean")

            # Only show families with >= 10 substrate-sites
            fam_counts = all_cond_df.groupby("family")["site_idx"].nunique()
            show_fams = fam_counts[fam_counts >= 10].index

            print(f"\n  {'Family':<18s}  {'WTyp':>8s}  {'AppP':>8s}  {'Ttau':>8s}  {'ApTt':>8s}  {'n sites':>7s}")
            print(f"  {'─' * 65}")
            for fam in sorted(show_fams):
                if fam in pivot_wide.index:
                    row = pivot_wide.loc[fam]
                    ns = fam_counts[fam]
                    print(f"  {fam:<18s}  {row.get('WTyp', np.nan):+8.4f}  {row.get('AppP', np.nan):+8.4f}  "
                          f"{row.get('Ttau', np.nan):+8.4f}  {row.get('ApTt', np.nan):+8.4f}  {ns:7d}")

    # Save detail
    detail.to_csv(APTT_DETAIL_FILE, index=False)

    # Save meta
    meta: Dict = {
        "global_aptt_mean_r": float(aptt["r_resid_corr"].mean()),
        "global_aptt_std_r": float(aptt["r_resid_corr"].std()),
        "n_aptt_sites": len(aptt),
        "n_kinase_mapped": int(aptt["has_kinase"].sum()),
        "n_families_tested": len(fam_summary_df),
    }
    if len(fam_summary_df) > 0:
        sig_fams = fam_summary_df[fam_summary_df["pval_vs_zero_adj"] < 0.05]
        meta["significant_families"] = sig_fams["family"].tolist()
        meta["most_negative_family"] = fam_summary_df.iloc[0]["family"]
        meta["most_negative_family_mean_r"] = float(fam_summary_df.iloc[0]["mean_r_resid_corr"])

    with open(APTT_META_FILE, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\n  Saved: {APTT_DETAIL_FILE}")
    print(f"  Saved: {APTT_FAMILY_FILE}")
    print(f"  Saved: {APTT_META_FILE}")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run_pipeline(tau: float = DEFAULT_TAU) -> None:
    """Run the full Module 4 rho landscape analysis."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load model and data
    print("Loading production model...")
    model = sap_model.load_model()
    print(f"  {len(model.site_params)} sites, p={model.p}")

    print("Loading data...")
    data, _ = sap_data.load_all(include_rna=True)

    # Step 1: Extract and classify rho
    print("\n" + "=" * 60)
    print("Step 1: Extract and classify rho")
    print("=" * 60)
    rho_df = extract_rho(model, data)
    classified = classify_sites(rho_df, tau=tau)

    cat_counts = classified["category"].value_counts()
    print(f"  Sites: {len(classified)}")
    print(f"  tau = {tau}")
    for cat, count in cat_counts.items():
        print(f"    {cat}: {count} ({count / len(classified):.1%})")

    classified.to_csv(RHO_DIST_FILE, index=False)
    print(f"  Saved: {RHO_DIST_FILE}")

    # Step 2: Kinase coupling profiles
    print("\n" + "=" * 60)
    print("Step 2: Kinase coupling profiles")
    print("=" * 60)
    print("  Building site-to-kinase mapping...")
    site_kinase_map = build_site_to_kinase_map(data.site_meta)
    mapped_sites = len(site_kinase_map)
    print(f"  {mapped_sites}/{len(classified)} sites mapped to kinases")

    kinase_coupling = compute_kinase_coupling(classified, site_kinase_map)
    kinase_coupling.to_csv(KINASE_SUMMARY_FILE, index=False)
    print(f"  {len(kinase_coupling)} kinases with substrate data")
    print(f"  Saved: {KINASE_SUMMARY_FILE}")

    # Show extremes
    most_decoupled = kinase_coupling.nlargest(5, "frac_decoupled")
    most_coupled = kinase_coupling.nsmallest(5, "frac_decoupled")
    print("\n  Most decoupled (by fraction):")
    for _, row in most_decoupled.iterrows():
        print(f"    {row['kinase_gene']:12s}  n={row['n_substrates']:5d}  "
              f"decoupled={row['frac_decoupled']:.1%}")
    print("\n  Most coupled (by fraction):")
    for _, row in most_coupled.iterrows():
        print(f"    {row['kinase_gene']:12s}  n={row['n_substrates']:5d}  "
              f"decoupled={row['frac_decoupled']:.1%}")

    # Step 3: Family enrichment
    print("\n" + "=" * 60)
    print("Step 3: Kinase family enrichment tests")
    print("=" * 60)
    family_results = test_family_enrichment(kinase_coupling, classified, site_kinase_map)
    family_results.to_csv(FAMILY_ENRICH_FILE, index=False)
    print(f"  {len(family_results)} families tested")

    if len(family_results) > 0:
        for _, row in family_results.iterrows():
            sig_marker = "*" if row.get("significant", False) else " "
            print(f"  {sig_marker} {row['family']:30s}  n={row['n_substrates']:5d}  "
                  f"decoupled={row['frac_decoupled']:.1%} (global={row['global_frac_decoupled']:.1%})  "
                  f"ER={row['enrichment_ratio']:.2f}  p={row['pval_adj']:.3g}")

    # Step 4: Condition specificity
    print("\n" + "=" * 60)
    print("Step 4: Condition specificity of coupling")
    print("=" * 60)
    cond_df = compute_condition_specificity(model, data)
    cond_df.to_csv(CONDITION_COUPLING_FILE, index=False)
    print(f"  Assessed {cond_df['site_idx'].nunique()} sites")

    # Per-condition mean correlation
    if len(cond_df) > 0:
        cond_summary = cond_df.groupby("condition")["r_resid_corr"].agg(["mean", "std", "median"])
        print("\n  Per-condition residual-RNA correlation:")
        for cond, row in cond_summary.iterrows():
            print(f"    {cond}: mean={row['mean']:+.4f}  std={row['std']:.4f}  median={row['median']:+.4f}")

        # Test: is coupling different across conditions? (Kruskal-Wallis)
        groups = [g["r_resid_corr"].dropna().values for _, g in cond_df.groupby("condition")]
        groups = [g for g in groups if len(g) > 1]
        if len(groups) >= 2:
            kw_stat, kw_pval = sp_stats.kruskal(*groups)
            print(f"\n  Kruskal-Wallis test: H={kw_stat:.2f}, p={kw_pval:.3g}")

    # Save meta
    meta = {
        "n_sites": len(classified),
        "tau": tau,
        "category_counts": cat_counts.to_dict(),
        "rho_distribution": {
            "mean": float(classified["rho"].mean()),
            "median": float(classified["rho"].median()),
            "iqr_25": float(classified["rho"].quantile(0.25)),
            "iqr_75": float(classified["rho"].quantile(0.75)),
            "frac_abs_gt_1": float((classified["rho"].abs() > 1).mean()),
        },
        "n_kinases_profiled": len(kinase_coupling),
        "n_families_tested": len(family_results),
        "n_families_significant": int(family_results["significant"].sum()) if len(family_results) > 0 else 0,
        "condition_coupling_sites_assessed": int(cond_df["site_idx"].nunique()),
    }
    with open(META_FILE, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\n  Summary saved: {META_FILE}")


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
    parser = argparse.ArgumentParser(description="Module 4: RNA-Phospho coupling landscape")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run", action="store_true", help="Run full analysis")
    group.add_argument("--aptt-detail", action="store_true", help="ApTt decoupling breakdown")
    group.add_argument("--summary", action="store_true", help="Print cached results")
    parser.add_argument("--tau", type=float, default=DEFAULT_TAU,
                        help=f"Coupling threshold (default: {DEFAULT_TAU})")
    args = parser.parse_args()

    if args.summary:
        print_summary()
    elif args.run:
        run_pipeline(tau=args.tau)
    elif args.aptt_detail:
        run_aptt_detail()


if __name__ == "__main__":
    main()

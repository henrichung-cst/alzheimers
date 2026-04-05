#!/usr/bin/env python3
"""Q2: Aggregation robustness — compare median vs mean vs weighted supertype-to-subclass.

Re-runs the SEA-AD supertype-to-subclass aggregation using three methods and
reports which kinase-cell type attributions are sensitive to the aggregation
choice.

Usage:
    python code/supplementary/aggregation_robustness.py --run
    python code/supplementary/aggregation_robustness.py --summary
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

OUTPUT_DIR = os.path.join(config.SUPPLEMENTARY_OUTPUT_DIR, "aggregation_robustness")


def _assign_confidence(concordance_score, wmb_specificity, sea_ad_lfc):
    """Parameterized confidence assignment (duplicated from kinase_attribution)."""
    if concordance_score <= 0:
        return "none"
    if wmb_specificity >= config.SPECIFICITY_HIGH and abs(sea_ad_lfc) > config.SEA_AD_LFC_MIN:
        return "high"
    if wmb_specificity >= config.SPECIFICITY_LOW or abs(sea_ad_lfc) > config.SEA_AD_LFC_MIN:
        return "moderate"
    return "low"


def _aggregate_supertypes(effects_by_subclass, method, counts_by_subclass=None):
    """Aggregate supertype LFCs to subclass level using specified method.

    Parameters
    ----------
    effects_by_subclass : dict[str, list[float]]
    method : str, one of "median", "mean", "weighted_mean"
    counts_by_subclass : dict[str, list[int]] or None
        Per-supertype cell counts (only used for weighted_mean)
    """
    result = {}
    for subclass, vals in effects_by_subclass.items():
        if method == "median":
            result[subclass] = float(np.median(vals))
        elif method == "mean":
            result[subclass] = float(np.mean(vals))
        elif method == "weighted_mean":
            if counts_by_subclass and subclass in counts_by_subclass:
                weights = counts_by_subclass[subclass]
                if len(weights) == len(vals) and sum(weights) > 0:
                    result[subclass] = float(np.average(vals, weights=weights))
                else:
                    result[subclass] = float(np.mean(vals))
            else:
                result[subclass] = float(np.mean(vals))
    return result


def _ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def step_run():
    """Run aggregation robustness comparison."""
    _ensure_output_dir()
    print("\n=== Aggregation Robustness Analysis ===\n")

    # Load MEA results (significant kinases)
    mea_path = os.path.join(config.KINASE_ATTRIBUTION_OUTPUT_DIR, "mea_stoichiometry.csv")
    if not os.path.exists(mea_path):
        raise FileNotFoundError(f"{mea_path} not found. Run kinase_attribution.py --enrich first.")
    mea = pd.read_csv(mea_path)
    sig = mea[mea["FDR"] < config.MEA_FDR_THRESH].copy()
    print(f"  {len(sig)} significant kinase-contrast pairs")

    # Load kinase-to-gene mapping
    k2g = pd.read_csv(config.MAPPING_CACHE_FILE)
    kinase_to_gene = dict(zip(k2g["kinase_abbreviation"], k2g["gene_symbol"]))
    sig["gene_symbol"] = sig["kinase"].map(lambda k: kinase_to_gene.get(k, k))

    # Load SEA-AD
    try:
        import anndata as ad
    except ImportError:
        raise ImportError("anndata required. Install with: mamba install anndata")

    sea_ad_path = os.path.join(config.SEA_AD_DIR, "effect_sizes.h5ad")
    if not os.path.exists(sea_ad_path):
        raise FileNotFoundError(f"{sea_ad_path} not found.")
    print("  Loading SEA-AD effect sizes...")
    adata = ad.read_h5ad(sea_ad_path)
    sea_ad_genes_upper = {g.upper(): g for g in adata.obs_names}
    supertypes = list(adata.var_names)
    st_to_subclass = dict(zip(adata.var_names, adata.var["Subclass"]))

    # Check if cell counts are available for weighted aggregation
    has_cell_counts = "n_cells" in adata.var.columns
    st_cell_counts = {}
    if has_cell_counts:
        st_cell_counts = dict(zip(adata.var_names, adata.var["n_cells"]))
        print("  Cell counts available for weighted aggregation")
    else:
        print("  Cell counts not available — weighted_mean will fall back to mean")

    gene_to_idx = {g: i for i, g in enumerate(adata.obs_names)}

    # Load WMB specificity for confidence tier computation
    wmb_spec = {}
    if os.path.exists(config.WMB_EXPRESSION_FILE):
        wmb = pd.read_csv(config.WMB_EXPRESSION_FILE)
        wmb_grouped = wmb.groupby(
            [wmb["gene_symbol"].str.upper(), "cell_type"]
        )["specificity_score"].max()
        wmb_spec = wmb_grouped.to_dict()

    # Pre-compute per-gene subclass effects and aggregations (avoid N+1)
    methods = ["median", "mean", "weighted_mean"]
    gene_aggregations = {}  # gene_upper -> {method -> {subclass -> lfc}}

    unique_genes = set(sig["gene_symbol"].dropna().str.upper())
    for gene_upper in unique_genes:
        if gene_upper not in sea_ad_genes_upper:
            continue

        sea_ad_gene = sea_ad_genes_upper[gene_upper]
        gene_idx = gene_to_idx[sea_ad_gene]
        effects = adata.X[gene_idx, :]
        if hasattr(effects, "toarray"):
            effects = effects.toarray().flatten()
        else:
            effects = np.asarray(effects).flatten()

        sc_effects = {}
        sc_cell_counts_local = {}
        for i, st in enumerate(supertypes):
            subclass = st_to_subclass[st]
            val = effects[i]
            if not np.isfinite(val):
                continue
            sc_effects.setdefault(subclass, []).append(val)
            if has_cell_counts and st in st_cell_counts:
                sc_cell_counts_local.setdefault(subclass, []).append(
                    int(st_cell_counts[st]))

        gene_aggregations[gene_upper] = {
            method: _aggregate_supertypes(sc_effects, method, sc_cell_counts_local)
            for method in methods
        }

    print(f"  Pre-computed aggregations for {len(gene_aggregations)} genes")

    # Build results using pre-computed aggregations
    all_results = []
    for _, row in sig.iterrows():
        kinase = row["kinase"]
        contrast = row["contrast"]
        nes = row["NES"]
        gene = row["gene_symbol"]
        gene_upper = gene.upper() if isinstance(gene, str) else ""

        if gene_upper not in gene_aggregations:
            continue

        for method in methods:
            agg = gene_aggregations[gene_upper][method]
            for subclass, lfc in agg.items():
                concordance = np.sign(nes) * lfc
                spec = wmb_spec.get((gene_upper, subclass), 0.0)
                conf = _assign_confidence(concordance, spec, lfc)
                all_results.append({
                    "kinase": kinase,
                    "gene_symbol": gene,
                    "contrast": contrast,
                    "cell_type": subclass,
                    "method": method,
                    "aggregated_lfc": lfc,
                    "concordance_score": concordance,
                    "wmb_specificity": spec,
                    "confidence": conf,
                })

    results_df = pd.DataFrame(all_results)
    if len(results_df) == 0:
        print("  No results produced (no kinase genes found in SEA-AD).")
        return

    # Pivot to compare methods
    pivot_key = ["kinase", "gene_symbol", "contrast", "cell_type"]
    comparison_rows = []
    for key, grp in results_df.groupby(pivot_key):
        method_confs = dict(zip(grp["method"], grp["confidence"]))
        method_lfcs = dict(zip(grp["method"], grp["aggregated_lfc"]))
        unique_confs = set(method_confs.values())
        comparison_rows.append({
            "kinase": key[0],
            "gene_symbol": key[1],
            "contrast": key[2],
            "cell_type": key[3],
            "conf_median": method_confs.get("median", ""),
            "conf_mean": method_confs.get("mean", ""),
            "conf_weighted_mean": method_confs.get("weighted_mean", ""),
            "lfc_median": method_lfcs.get("median", np.nan),
            "lfc_mean": method_lfcs.get("mean", np.nan),
            "lfc_weighted_mean": method_lfcs.get("weighted_mean", np.nan),
            "stable": len(unique_confs) == 1,
        })

    comp_df = pd.DataFrame(comparison_rows)
    comp_path = os.path.join(OUTPUT_DIR, "aggregation_comparison.csv")
    comp_df.to_csv(comp_path, index=False)
    print(f"  Saved {comp_path} ({len(comp_df)} kinase-cell type pairs)")

    # Method-sensitive pairs
    sensitive = comp_df[~comp_df["stable"]].copy()
    sensitive_path = os.path.join(OUTPUT_DIR, "method_sensitive_pairs.csv")
    sensitive.to_csv(sensitive_path, index=False)
    print(f"  Method-sensitive pairs: {len(sensitive)} / {len(comp_df)} "
          f"({100*len(sensitive)/max(len(comp_df),1):.1f}%)")
    print(f"  Saved {sensitive_path}")

    # Summary
    summary = {
        "total_pairs": len(comp_df),
        "stable_pairs": int(comp_df["stable"].sum()),
        "sensitive_pairs": len(sensitive),
        "pct_stable": round(100 * comp_df["stable"].mean(), 1),
        "cell_counts_available": has_cell_counts,
    }
    summary_path = os.path.join(OUTPUT_DIR, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Stability: {summary['pct_stable']}% of pairs have same "
          f"confidence across all methods")


def step_summary():
    """Print cached summary."""
    summary_path = os.path.join(OUTPUT_DIR, "summary.json")
    if not os.path.exists(summary_path):
        print("No summary found. Run --run first.")
        return
    with open(summary_path) as f:
        s = json.load(f)
    print(f"\nAggregation Robustness Analysis:")
    print(f"  {s['total_pairs']} kinase-cell type pairs compared")
    print(f"  {s['stable_pairs']} stable ({s['pct_stable']}%)")
    print(f"  {s['sensitive_pairs']} method-sensitive")
    print(f"  Cell counts for weighting: {'yes' if s['cell_counts_available'] else 'no (fell back to mean)'}")


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

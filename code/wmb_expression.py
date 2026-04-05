#!/usr/bin/env python3
"""WMB Expression Export: supporting surface for unified cell-type attribution.

Computes per-cell-type kinase/phosphatase expression from the Allen Institute
Whole Mouse Brain (WMB) 10Xv3 HPF dataset.  The output file
(wmb_kinase_expression.csv) is consumed by kinase_attribution.py unified
attribution and attribution_recovery.py.

Extracted from the archived pre-stoichiometry concordance logic to isolate
the live supporting dependency.

Inputs:
  data/external/allen_abc/  (WMB 10Xv3 cell metadata + HPF h5ad, via ABC Atlas cache)
  data/incytr_collections/song/kinase/kldata.csv  (kinase substrate library)

Outputs:
  outputs/reports/wmb_expression/wmb_kinase_expression.csv

Optional dependencies:
    pip install git+https://github.com/alleninstitute/abc_atlas_access.git
    pip install anndata

Usage:
    python code/wmb_expression.py --run       # Compute WMB expression matrix
    python code/wmb_expression.py --summary   # Print cached results
"""

from __future__ import annotations

import argparse
import os
from typing import Dict

import numpy as np
import pandas as pd

import config
from atlas_reference import (
    match_subclass,
    get_all_kinase_genes,
    get_phosphatase_genes_from_genelist,
    get_abc_cache,
    _extract_gene_symbols,
    _get_expression_path,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OUTPUT_DIR = config.WMB_EXPRESSION_OUTPUT_DIR
WMB_EXPR_FILE = config.WMB_EXPRESSION_FILE

# The 5 cell types used for attribution (exclude "Other")
CT5 = [ct for ct in config.SAP_CELLTYPES if ct != "Other"]

# ---------------------------------------------------------------------------
# Subclass-to-5+1 mapping
# ---------------------------------------------------------------------------


def _match_subclass(label: str) -> str:
    """Map a subclass/supertype label to 5+1 pooling via keyword matching.

    Wraps atlas_reference.match_subclass with additional keywords for aging
    atlas cluster names (COP/NFOL/MFOL/MOL for oligodendrocyte lineage).
    """
    result = match_subclass(label)
    if result != "Other":
        return result
    # Additional keywords for aging atlas cluster names
    for kw in ["COP", "NFOL", "MFOL", "MOL"]:
        if kw in label:  # case-sensitive to avoid false matches
            return "Oligodendrocytes"
    return "Other"


# SEA-AD subclass keyword patterns for matching WMB subclass labels.
# Order matters: more specific patterns must come before general ones.
# Each entry: (sea_ad_subclass, [keywords_to_match_in_wmb_label])
_SEA_AD_SUBCLASS_PATTERNS = [
    # GABAergic — specific before general
    ("Chandelier", ["chandelier"]),
    ("Lamp5 Lhx6", ["lamp5 lhx6", "lamp5_lhx6"]),
    ("Lamp5", ["lamp5"]),
    ("Sst Chodl", ["sst chodl", "sst_chodl"]),
    ("Sst", ["sst"]),
    ("Pax6", ["pax6"]),
    ("Pvalb", ["pvalb"]),
    ("Sncg", ["sncg"]),
    ("Vip", ["vip"]),
    # Glutamatergic — specific before general
    ("L6 IT Car3", ["car3"]),
    ("L6 CT", ["l6 ct", "l6_ct"]),
    ("L6 IT", ["l6 it", "l6_it"]),
    ("L6b", ["l6b"]),
    ("L5 ET", ["l5 et", "l5_et"]),
    ("L5 IT", ["l5 it", "l5_it"]),
    ("L5/6 NP", ["l5/6 np", "np "]),
    ("L4 IT", ["l4 it", "l4_it", "l4/5 it"]),
    ("L2/3 IT", ["l2/3 it", "l2/3_it", "l2 it"]),
    # Non-neuronal
    ("Astrocyte", ["astro"]),
    ("Microglia-PVM", ["micro", "pvm"]),
    ("OPC", ["opc", "cop ", "nfol", "mfol"]),
    ("Oligodendrocyte", ["oligo", " mol "]),
    ("Endothelial", ["endo"]),
    ("VLMC", ["vlmc"]),
]


def _match_sea_ad_subclass(label: str) -> str:
    """Map a WMB subclass label to a SEA-AD subclass name.

    WMB has 338 region-specific subclasses (e.g. '052 Pvalb Gaba').
    SEA-AD has 24 subclasses (e.g. 'Pvalb').  This function maps WMB → SEA-AD
    via case-insensitive keyword matching, returning 'Other' for unmatched.
    """
    label_lower = label.lower()
    for sea_ad_name, keywords in _SEA_AD_SUBCLASS_PATTERNS:
        for kw in keywords:
            if kw in label_lower:
                return sea_ad_name
    return "Other"


def _human_to_mouse(symbol: str) -> str:
    """Convert human gene symbol (uppercase) to mouse (title-case)."""
    if len(symbol) > 1:
        return symbol[0].upper() + symbol[1:].lower()
    return symbol.upper()


def _mouse_to_human(symbol: str) -> str:
    """Convert mouse gene symbol to human (uppercase)."""
    return symbol.upper()


# ---------------------------------------------------------------------------
# WMB Expression Computation
# ---------------------------------------------------------------------------


def _stream_wmb_expression(
    gene_names: list,
    gene_indices: np.ndarray,
    label: str = "genes",
    chunk_size: int = 5000,
    cell_type_set: str = "subclass",
    skip_regional: bool = False,
) -> tuple:
    """Stream through all 13 WMB regions, accumulating per-cell-type expression.

    Args:
        cell_type_set: "subclass" for 24 SEA-AD subclasses (default),
                       "5plus1" for 5 broad classes (legacy).

    Returns (accum, regional_rows) where accum is {ct: {expr_sum, nonzero_count,
    n_cells}} and regional_rows is a list of per-region per-gene dicts.
    """
    import anndata as ad

    cache = get_abc_cache()
    n_genes = len(gene_indices)

    # Determine cell type categories and mapping function
    if cell_type_set == "subclass":
        ct_list = config.SEA_AD_SUBCLASSES
        map_fn = _match_sea_ad_subclass
        ct_col = "cell_type"
    else:
        ct_list = CT5
        map_fn = _match_subclass
        ct_col = "cell_type_5plus1"

    # Load WMB cell metadata (shared across all regions)
    print("  Loading WMB cell metadata ...")
    wmb_meta = cache.get_metadata_dataframe(
        directory="WMB-10X", file_name="cell_metadata_with_cluster_annotation"
    )
    print(f"  Total WMB cells: {len(wmb_meta):,}")

    wmb_meta["_ct_mapped"] = wmb_meta["subclass"].apply(
        lambda s: map_fn(s) if pd.notna(s) else "Other"
    )

    meta_index_set = set(wmb_meta.index.tolist())
    wmb_meta_by_label = None
    if "cell_label" in wmb_meta.columns:
        wmb_meta_by_label = wmb_meta.set_index("cell_label")

    # Verify gene panel from first region
    first_key = config.WMB_ALL_REGION_KEYS[0]
    first_path = _get_expression_path(cache, config.WMB_DATASET_KEY, first_key)
    adata_ref = ad.read_h5ad(first_path, backed="r")
    ref_var_names = list(adata_ref.var_names)
    if hasattr(adata_ref, "file") and adata_ref.file is not None:
        adata_ref.file.close()

    print(f"  Target {label}: {n_genes}")
    print(f"  Cell type set: {cell_type_set} ({len(ct_list)} categories)")

    # Per-cell-type accumulators
    accum: Dict[str, Dict] = {}
    for ct in ct_list:
        accum[ct] = {
            "expr_sum": np.zeros(n_genes, dtype=np.float64),
            "nonzero_count": np.zeros(n_genes, dtype=np.int64),
            "n_cells": 0,
        }

    regional_rows = []

    for ri, file_key in enumerate(config.WMB_ALL_REGION_KEYS, 1):
        region = file_key.split("/")[0].replace("WMB-10Xv3-", "")
        region_path = _get_expression_path(cache, config.WMB_DATASET_KEY, file_key)

        print(f"\n  [{ri}/{len(config.WMB_ALL_REGION_KEYS)}] Region: {region}")
        adata = ad.read_h5ad(region_path, backed="r")
        print(f"    Shape: {adata.shape}")

        assert list(adata.var_names) == ref_var_names, \
            f"Gene panel mismatch in {region}: expected {len(ref_var_names)} genes"

        # Match cells to metadata
        h5ad_cells = set(adata.obs.index.tolist())
        overlap_main = len(h5ad_cells & meta_index_set)

        if overlap_main >= 100:
            use_meta = wmb_meta
        elif wmb_meta_by_label is not None:
            use_meta = wmb_meta_by_label
        else:
            print(f"    WARNING: Cannot match cells for {region}, skipping")
            if hasattr(adata, "file") and adata.file is not None:
                adata.file.close()
            continue

        common = sorted(h5ad_cells & set(use_meta.index.tolist()))

        if len(common) < 100:
            if "subclass" in adata.obs.columns:
                print(f"    Low metadata overlap ({len(common)}) — using h5ad subclass")
                cell_ct_map = adata.obs["subclass"].apply(
                    lambda s: map_fn(s) if pd.notna(s) else "Other"
                ).to_dict()
                common = list(adata.obs.index)
            else:
                print(f"    WARNING: Insufficient cell overlap for {region}, skipping")
                if hasattr(adata, "file") and adata.file is not None:
                    adata.file.close()
                continue
        else:
            cell_ct_map = use_meta.loc[common, "_ct_mapped"].to_dict()

        print(f"    Matched cells: {len(common):,}")

        h5ad_obs_list = list(adata.obs.index)
        h5ad_cell_to_row = {c: i for i, c in enumerate(h5ad_obs_list)}

        row_ct_pairs = []
        for c in common:
            ct = cell_ct_map.get(c, "Other")
            if ct != "Other" and c in h5ad_cell_to_row:
                row_ct_pairs.append((h5ad_cell_to_row[c], ct))
        row_ct_pairs.sort(key=lambda x: x[0])

        if not row_ct_pairs:
            print("    No matched cells in target cell types")
            if hasattr(adata, "file") and adata.file is not None:
                adata.file.close()
            continue

        all_row_indices = np.array([p[0] for p in row_ct_pairs])
        all_row_cts = [p[1] for p in row_ct_pairs]

        region_accum: Dict[str, Dict] = {}
        for ct in ct_list:
            region_accum[ct] = {
                "expr_sum": np.zeros(n_genes, dtype=np.float64),
                "nonzero_count": np.zeros(n_genes, dtype=np.int64),
                "n_cells": 0,
            }

        for start in range(0, len(all_row_indices), chunk_size):
            chunk_rows = all_row_indices[start:start + chunk_size]
            chunk_cts = all_row_cts[start:start + chunk_size]
            chunk_data = adata.X[chunk_rows][:, gene_indices]
            if hasattr(chunk_data, "toarray"):
                chunk_data = chunk_data.toarray()

            # Vectorized: one np.unique pass instead of N per-type scans
            chunk_cts_arr = np.array(chunk_cts)
            unique_cts, inverse = np.unique(chunk_cts_arr, return_inverse=True)
            for ct_idx, ct in enumerate(unique_cts):
                if ct not in region_accum:
                    continue
                ct_mask = (inverse == ct_idx)
                ct_data = chunk_data[ct_mask]
                region_accum[ct]["expr_sum"] += ct_data.sum(axis=0)
                region_accum[ct]["nonzero_count"] += (ct_data > 0).sum(axis=0)
                region_accum[ct]["n_cells"] += int(ct_mask.sum())

        for ct in ct_list:
            n_cells = region_accum[ct]["n_cells"]
            if n_cells == 0:
                continue

            print(f"    {ct}: {n_cells:,} cells")

            accum[ct]["expr_sum"] += region_accum[ct]["expr_sum"]
            accum[ct]["nonzero_count"] += region_accum[ct]["nonzero_count"]
            accum[ct]["n_cells"] += n_cells

            if not skip_regional:
                region_mean = region_accum[ct]["expr_sum"] / n_cells
                region_frac = region_accum[ct]["nonzero_count"] / n_cells
                for i, gene in enumerate(gene_names):
                    regional_rows.append({
                        "region": region,
                        "gene_symbol": gene,
                        ct_col: ct,
                        "mean_log2_expression": round(float(region_mean[i]), 6),
                        "fraction_cells_expressing": round(float(region_frac[i]), 6),
                        "n_cells": n_cells,
                    })

        if hasattr(adata, "file") and adata.file is not None:
            adata.file.close()

    return accum, regional_rows


def _build_gene_index(cache):
    """Build gene symbol → column index mapping from the first WMB region.

    Returns (atlas_genes, gene_to_idx, gene_fmt).
    """
    import anndata as ad

    first_key = config.WMB_ALL_REGION_KEYS[0]
    first_path = _get_expression_path(cache, config.WMB_DATASET_KEY, first_key)
    adata_ref = ad.read_h5ad(first_path, backed="r")

    atlas_genes, gene_fmt = _extract_gene_symbols(adata_ref)

    if "gene_symbol" in adata_ref.var.columns:
        gene_to_idx = {}
        for i, sym in enumerate(adata_ref.var["gene_symbol"]):
            if pd.notna(sym):
                gene_to_idx[sym] = i
    else:
        gene_to_idx = {g: i for i, g in enumerate(adata_ref.var_names)}

    if hasattr(adata_ref, "file") and adata_ref.file is not None:
        adata_ref.file.close()

    return atlas_genes, gene_to_idx, gene_fmt


def compute_wmb_expression() -> pd.DataFrame:
    """Compute whole-brain kinase/phosphatase expression matrix.

    Streams through all 13 WMB-10Xv3 regional h5ad files, accumulating
    cell-weighted expression sums per cell type.  This correctly models
    whole-brain homogenate: larger regions contribute proportionally more.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("WMB Whole-Brain Kinase Expression Matrix (13 regions)")
    print("=" * 60)

    cache = get_abc_cache()
    atlas_genes, gene_to_idx, gene_fmt = _build_gene_index(cache)
    print(f"  Gene format: {gene_fmt}")

    mouse_kinases, _ = get_all_kinase_genes()
    phosphatases = get_phosphatase_genes_from_genelist(atlas_genes)
    all_kp = mouse_kinases | phosphatases

    kinase_genes = sorted(all_kp & set(gene_to_idx.keys()))
    kinase_idx = np.array([gene_to_idx[g] for g in kinase_genes])

    accum, regional_rows = _stream_wmb_expression(
        kinase_genes, kinase_idx, label="kinase/phosphatase",
        cell_type_set="subclass",
    )

    # Compute global whole-brain means
    print("\n  Computing whole-brain aggregates ...")
    rows = []
    for ct in config.SEA_AD_SUBCLASSES:
        n_total = accum[ct]["n_cells"]
        if n_total == 0:
            continue
        mean_expr = accum[ct]["expr_sum"] / n_total
        frac_expr = accum[ct]["nonzero_count"] / n_total

        print(f"    {ct}: {n_total:,} cells across all regions")

        for i, gene in enumerate(kinase_genes):
            rows.append({
                "kinase_id": _mouse_to_human(gene),
                "gene_symbol": gene,
                "cell_type": ct,
                "mean_log2_expression": round(float(mean_expr[i]), 6),
                "fraction_cells_expressing": round(float(frac_expr[i]), 6),
                "specificity_score": np.nan,
                "binary_expressed": bool(mean_expr[i] > 1 and frac_expr[i] > 0.10),
            })

    df = pd.DataFrame(rows)

    # Compute specificity scores (across all 24 subclasses per gene)
    for gene in kinase_genes:
        gene_mask = df["gene_symbol"] == gene
        gene_df = df.loc[gene_mask]
        total_expr = gene_df["mean_log2_expression"].sum()
        if total_expr > 0:
            spec = (gene_df["mean_log2_expression"] / total_expr).round(6)
            df.loc[gene_mask, "specificity_score"] = spec.values
        else:
            df.loc[gene_mask, "specificity_score"] = 0.0

    df.to_csv(WMB_EXPR_FILE, index=False)
    print(f"\n  Saved {len(df)} rows to {WMB_EXPR_FILE}")

    # Save per-region breakdown
    if regional_rows:
        regional_df = pd.DataFrame(regional_rows)
        regional_df["kinase_id"] = regional_df["gene_symbol"].apply(_mouse_to_human)
        regional_df.to_csv(config.WMB_REGIONAL_EXPRESSION_FILE, index=False)
        print(f"  Saved {len(regional_df)} regional rows to "
              f"{config.WMB_REGIONAL_EXPRESSION_FILE}")

    # Summary
    for ct in config.SEA_AD_SUBCLASSES:
        sub = df[df["cell_type"] == ct]
        if len(sub) == 0:
            continue
        n_expr = sub["binary_expressed"].sum()
        print(f"    {ct}: {n_expr}/{len(sub)} kinases expressed")

    return df


def compute_wmb_proteome_expression() -> pd.DataFrame:
    """Compute whole-brain expression for all genes in the total proteome.

    Uses the same streaming infrastructure as the kinase computation but
    extracts all ~6,444 proteome genes instead of ~400 kinases.  Output is
    consumed by data_ingest.py --markers for cell-type marker assessment.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("WMB Whole-Brain Proteome Expression Matrix (13 regions)")
    print("=" * 60)

    # Load proteome gene list (produced by data_ingest.py --phospho-match)
    gene_list_path = config.PROTEOME_GENE_LIST_FILE
    if not os.path.exists(gene_list_path):
        raise FileNotFoundError(
            f"Proteome gene list not found at {gene_list_path}. "
            "Run: python code/data_ingest.py --phospho-match"
        )
    with open(gene_list_path) as f:
        proteome_genes_upper = [line.strip() for line in f if line.strip()]
    print(f"  Proteome genes (human symbols): {len(proteome_genes_upper)}")

    # Convert to mouse case for atlas matching
    proteome_genes_mouse = [_human_to_mouse(g) for g in proteome_genes_upper]

    cache = get_abc_cache()
    _, gene_to_idx, gene_fmt = _build_gene_index(cache)
    print(f"  Gene format: {gene_fmt}")

    # Intersect with atlas
    matched_genes = []
    matched_indices = []
    for g in proteome_genes_mouse:
        if g in gene_to_idx:
            matched_genes.append(g)
            matched_indices.append(gene_to_idx[g])

    print(f"  Proteome genes found in atlas: {len(matched_genes)}/{len(proteome_genes_upper)}")
    gene_indices = np.array(matched_indices)

    accum, _ = _stream_wmb_expression(
        matched_genes, gene_indices, label="proteome", chunk_size=2000,
        cell_type_set="5plus1", skip_regional=True,
    )

    # Build output DataFrame
    print("\n  Computing whole-brain aggregates ...")
    rows = []
    for ct in CT5:
        n_total = accum[ct]["n_cells"]
        if n_total == 0:
            continue
        mean_expr = accum[ct]["expr_sum"] / n_total
        frac_expr = accum[ct]["nonzero_count"] / n_total

        print(f"    {ct}: {n_total:,} cells across all regions")

        for i, gene in enumerate(matched_genes):
            rows.append({
                "gene_symbol_mouse": gene,
                "gene_symbol_human": _mouse_to_human(gene),
                "cell_type_5plus1": ct,
                "mean_log2_expression": round(float(mean_expr[i]), 6),
                "fraction_cells_expressing": round(float(frac_expr[i]), 6),
                "specificity_score": np.nan,
                "binary_expressed": bool(mean_expr[i] > 1 and frac_expr[i] > 0.10),
                "n_cells": n_total,
            })

    df = pd.DataFrame(rows)

    # Compute specificity scores
    for gene in matched_genes:
        gene_mask = df["gene_symbol_mouse"] == gene
        gene_df = df.loc[gene_mask]
        total_expr = gene_df["mean_log2_expression"].sum()
        if total_expr > 0:
            spec = (gene_df["mean_log2_expression"] / total_expr).round(6)
            df.loc[gene_mask, "specificity_score"] = spec.values
        else:
            df.loc[gene_mask, "specificity_score"] = 0.0

    out_path = config.WMB_PROTEOME_EXPRESSION_FILE
    df.to_csv(out_path, index=False)
    print(f"\n  Saved {len(df)} rows to {out_path}")

    # Summary: top-5 most specific genes per cell type
    for ct in CT5:
        sub = df[df["cell_type_5plus1"] == ct].copy()
        n_expr = sub["binary_expressed"].sum()
        top5 = sub.sort_values("specificity_score", ascending=False).head(5)
        top_str = ", ".join(
            f"{r['gene_symbol_mouse']}({r['specificity_score']:.2f})"
            for _, r in top5.iterrows()
        )
        print(f"    {ct}: {n_expr}/{len(sub)} expressed; top-5: {top_str}")

    return df


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def print_summary() -> None:
    """Print cached WMB expression results."""
    if not os.path.exists(WMB_EXPR_FILE):
        print(f"  No cached results at {WMB_EXPR_FILE}")
        return

    df = pd.read_csv(WMB_EXPR_FILE)
    ct_col = "cell_type" if "cell_type" in df.columns else "cell_type_5plus1"
    print(f"  WMB expression: {len(df)} rows, {df[ct_col].nunique()} cell types")
    for ct in sorted(df[ct_col].unique()):
        sub = df[df[ct_col] == ct]
        n_expr = sub["binary_expressed"].sum()
        print(f"    {ct}: {n_expr}/{len(sub)} kinases expressed")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="WMB Expression Export: supporting surface for unified cell-type attribution"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run", action="store_true",
                       help="Compute WMB kinase expression matrix")
    group.add_argument("--proteome", action="store_true",
                       help="Compute proteome-wide WMB expression matrix")
    group.add_argument("--summary", action="store_true",
                       help="Print cached results")

    args = parser.parse_args()

    if args.run:
        compute_wmb_expression()
    elif args.proteome:
        compute_wmb_proteome_expression()
    elif args.summary:
        print_summary()


if __name__ == "__main__":
    main()

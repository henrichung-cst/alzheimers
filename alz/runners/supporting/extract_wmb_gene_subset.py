#!/usr/bin/env python3
"""Extract gene-subset h5ad files from the full WMB regional expression matrices.

Reads each of the 13 WMB-10Xv3 regional h5ad files and writes a subset
containing only the genes needed by the pipeline (proteome + kinase/phosphatase
union). The resulting files are much smaller and can be read without
decompressing the full atlas.

Prerequisites:
  - Full h5ad files must be decompressed (run decompress_atlas_cache.sh WMB)
  - data_ingest.py --phospho-match must have run (produces total_proteome_genes.txt)

Output:
  data/external/allen_abc/expression_matrices/WMB-10Xv3-subset/
    WMB-10Xv3-{region}-log2-subset.h5ad  (one per region)
    MANIFEST.json                          (provenance record)

Usage:
  python alz/runners/supporting/extract_wmb_gene_subset.py
  python alz/runners/supporting/extract_wmb_gene_subset.py --dry-run
"""

import argparse
import json
import os
import sys
from datetime import datetime

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

# Add repo root to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "code"))

import config
from atlas_reference import (
    get_abc_cache,
    get_all_kinase_genes,
    get_phosphatase_genes_from_genelist,
    _extract_gene_symbols,
    _get_expression_path,
)
from wmb_expression import _human_to_mouse


def _build_gene_union():
    """Build the union of proteome + kinase/phosphatase gene symbols (mouse case).

    Returns (gene_set, source_info) where gene_set is a set of mouse-case
    gene symbols and source_info describes what was included.
    """
    genes = set()
    sources = {}

    # 1. Proteome genes (from data_ingest --phospho-match)
    gene_list_path = config.PROTEOME_GENE_LIST_FILE
    if os.path.exists(gene_list_path):
        with open(gene_list_path) as f:
            proteome_upper = [line.strip() for line in f if line.strip()]
        proteome_mouse = {_human_to_mouse(g) for g in proteome_upper}
        genes |= proteome_mouse
        sources["proteome"] = len(proteome_mouse)
    else:
        print(f"  WARNING: Proteome gene list not found at {gene_list_path}")
        print("  Run: python alz/data_ingest.py --phospho-match")

    # 2. Kinase/phosphatase genes
    try:
        cache = get_abc_cache()
        first_key = config.WMB_ALL_REGION_KEYS[0]
        first_path = _get_expression_path(cache, config.WMB_DATASET_KEY, first_key)
        adata_ref = ad.read_h5ad(first_path, backed="r")
        atlas_genes, _ = _extract_gene_symbols(adata_ref)
        if hasattr(adata_ref, "file") and adata_ref.file is not None:
            adata_ref.file.close()

        mouse_kinases, _ = get_all_kinase_genes()
        phosphatases = get_phosphatase_genes_from_genelist(atlas_genes)
        kp_genes = mouse_kinases | phosphatases
        genes |= kp_genes
        sources["kinase_phosphatase"] = len(kp_genes)
    except Exception as e:
        print(f"  WARNING: Could not load kinase/phosphatase genes: {e}")

    sources["total_union"] = len(genes)
    return genes, sources


def extract_subsets(dry_run=False):
    """Extract gene-subset h5ad files for all 13 WMB regions."""
    print("=" * 60)
    print("WMB Gene-Subset Extraction")
    print("=" * 60)

    # Build gene union
    print("\n  Building gene union (proteome + kinase/phosphatase)...")
    gene_union, source_info = _build_gene_union()
    print(f"  Gene union: {len(gene_union)} genes")
    for src, n in source_info.items():
        print(f"    {src}: {n}")

    if len(gene_union) == 0:
        print("  ERROR: No genes to extract. Aborting.")
        return

    # Setup output directory
    out_dir = config.WMB_SUBSET_DIR
    os.makedirs(out_dir, exist_ok=True)

    cache = get_abc_cache()

    # Get gene index from first region
    first_key = config.WMB_ALL_REGION_KEYS[0]
    first_path = _get_expression_path(cache, config.WMB_DATASET_KEY, first_key)
    adata_ref = ad.read_h5ad(first_path, backed="r")

    # Build gene symbol → column index mapping
    if "gene_symbol" in adata_ref.var.columns:
        gene_to_idx = {}
        for i, sym in enumerate(adata_ref.var["gene_symbol"]):
            if pd.notna(sym):
                gene_to_idx[sym] = i
    else:
        gene_to_idx = {g: i for i, g in enumerate(adata_ref.var_names)}

    if hasattr(adata_ref, "file") and adata_ref.file is not None:
        adata_ref.file.close()

    # Find which genes from the union exist in the atlas
    matched_genes = sorted(gene_union & set(gene_to_idx.keys()))
    matched_indices = np.array([gene_to_idx[g] for g in matched_genes])
    print(f"\n  Genes found in atlas: {len(matched_genes)}/{len(gene_union)}")

    if dry_run:
        print("\n  DRY RUN — would extract subsets for these regions:")
        for file_key in config.WMB_ALL_REGION_KEYS:
            region = file_key.split("/")[0].replace("WMB-10Xv3-", "")
            print(f"    {region}")
        print(f"\n  Output directory: {out_dir}")
        return

    # Sort indices for contiguous read optimization
    sort_order = np.argsort(matched_indices)
    sorted_indices = matched_indices[sort_order]

    manifest = {
        "created": datetime.now().isoformat(),
        "source": "WMB-10Xv3 (Allen Brain Cell Atlas)",
        "gene_count": len(matched_genes),
        "gene_sources": source_info,
        "regions": {},
    }

    total_input_bytes = 0
    total_output_bytes = 0

    for ri, file_key in enumerate(config.WMB_ALL_REGION_KEYS, 1):
        region = file_key.split("/")[0]  # e.g. "WMB-10Xv3-CB"
        region_short = region.replace("WMB-10Xv3-", "")
        out_path = os.path.join(
            out_dir, config.WMB_SUBSET_FILENAME_FMT.format(region=region)
        )

        print(f"\n  [{ri}/{len(config.WMB_ALL_REGION_KEYS)}] {region_short}")

        if not dry_run and os.path.exists(out_path):
            print(f"    SKIP: subset already exists at {out_path}")
            output_size = os.path.getsize(out_path)
            total_output_bytes += output_size
            continue

        region_path = _get_expression_path(cache, config.WMB_DATASET_KEY, file_key)
        if not os.path.exists(region_path):
            print(f"    SKIP: Full h5ad not found at {region_path}")
            print(f"    Run: bash alz/runners/supporting/decompress_atlas_cache.sh WMB")
            continue

        input_size = os.path.getsize(region_path)
        total_input_bytes += input_size

        adata = ad.read_h5ad(str(region_path), backed="r")
        print(f"    Full shape: {adata.shape[0]:,} cells × {adata.shape[1]:,} genes")

        # Read subset columns in row chunks to limit memory usage.
        # The contiguous slice spans ~31K cols, so each dense chunk is
        # row_chunk × 31K × 4 bytes.  At 10K rows that's ~1.2 GB.
        min_idx = int(sorted_indices[0])
        max_idx = int(sorted_indices[-1])
        local_indices = sorted_indices - min_idx
        n_cells = adata.shape[0]
        slice_width = max_idx - min_idx + 1
        row_chunk = max(1000, min(50_000, int(2e9 / (slice_width * 4))))

        print(f"    Reading column slice [{min_idx}:{max_idx+1}] "
              f"({slice_width} cols, need {len(sorted_indices)}, "
              f"row_chunk={row_chunk:,})...")

        chunks = []
        for row_start in range(0, n_cells, row_chunk):
            row_end = min(row_start + row_chunk, n_cells)
            chunk = adata.X[row_start:row_end, min_idx:max_idx + 1]
            if hasattr(chunk, "toarray"):
                chunk = chunk.toarray()
            chunks.append(sparse.csr_matrix(chunk[:, local_indices]))

        X_subset = sparse.vstack(chunks, format="csr")

        print(f"    Subset shape: {X_subset.shape[0]:,} cells × {X_subset.shape[1]:,} genes")

        # Build subset var metadata
        var_subset = adata.var.iloc[sorted_indices].copy()

        # Create new anndata with subset
        adata_subset = ad.AnnData(
            X=X_subset,
            obs=adata.obs.copy(),
            var=var_subset,
        )

        if hasattr(adata, "file") and adata.file is not None:
            adata.file.close()

        adata_subset.write_h5ad(out_path)
        output_size = os.path.getsize(out_path)
        total_output_bytes += output_size

        ratio = output_size / input_size * 100
        print(f"    Saved: {out_path}")
        print(f"    Size: {output_size / 1e9:.2f} GB "
              f"({ratio:.1f}% of original {input_size / 1e9:.2f} GB)")

        manifest["regions"][region_short] = {
            "file": f"{region}-log2-subset.h5ad",
            "cells": adata_subset.shape[0],
            "genes": adata_subset.shape[1],
            "size_bytes": output_size,
            "source_size_bytes": input_size,
        }

        del adata_subset, X_subset, chunks

    # Write manifest
    manifest_path = os.path.join(out_dir, "MANIFEST.json")
    manifest["total_output_bytes"] = total_output_bytes
    manifest["total_input_bytes"] = total_input_bytes
    if total_input_bytes > 0:
        manifest["compression_ratio"] = round(
            total_output_bytes / total_input_bytes * 100, 1
        )
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n  {'=' * 50}")
    print(f"  Extraction complete.")
    print(f"  Total input:  {total_input_bytes / 1e9:.2f} GB")
    print(f"  Total output: {total_output_bytes / 1e9:.2f} GB")
    if total_input_bytes > 0:
        print(f"  Ratio: {total_output_bytes / total_input_bytes * 100:.1f}%")
    print(f"  Manifest: {manifest_path}")
    print(f"\n  The full h5ad files can now be re-compressed:")
    print(f"  bash alz/runners/supporting/compress_atlas_cache.sh")


def main():
    parser = argparse.ArgumentParser(
        description="Extract gene-subset h5ad files from full WMB atlas"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be extracted without writing files"
    )
    args = parser.parse_args()
    extract_subsets(dry_run=args.dry_run)


if __name__ == "__main__":
    main()

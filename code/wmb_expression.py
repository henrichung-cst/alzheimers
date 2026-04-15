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
import contextlib
import glob
import os
import shutil
import subprocess
from typing import Dict, Optional

import numpy as np
import pandas as pd

import config
from atlas_reference import (
    get_all_kinase_genes,
    get_phosphatase_genes_from_genelist,
    get_abc_cache,
    _extract_gene_symbols,

)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OUTPUT_DIR = config.WMB_EXPRESSION_OUTPUT_DIR
WMB_EXPR_FILE = config.WMB_EXPRESSION_FILE

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

# ---------------------------------------------------------------------------
# Auto-decompress / recompress for compressed subset files
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _ensure_subsets_decompressed():
    """Decompress WMB subset .zst files if needed; recompress on exit.

    On entry, if subset h5ad files are zstd-compressed, decompresses them.
    On exit (including exceptions), recompresses them.  If a previous run
    was killed (stale .h5ad files coexist with .h5ad.zst), recompresses
    the stale files before proceeding.
    """
    subset_dir = config.WMB_SUBSET_DIR

    if not shutil.which("zstd"):
        raise RuntimeError(
            "zstd not found — required for auto-decompression. "
            "Install with: sudo dnf install zstd"
        )

    # Crash recovery: if .h5ad files exist alongside .zst, a prior run
    # was interrupted — recompress the stale files before proceeding.
    stale = sorted(glob.glob(os.path.join(subset_dir, "*.h5ad")))
    zst_files = sorted(glob.glob(os.path.join(subset_dir, "*.h5ad.zst")))
    if stale and zst_files:
        print("  Detected stale decompressed subsets from a previous run, "
              "recompressing first...")
        for f in stale:
            subprocess.run(["zstd", "-3", "-T0", "--rm", "-q", f], check=True)
        zst_files = sorted(glob.glob(os.path.join(subset_dir, "*.h5ad.zst")))

    # One-time metadata CSV decompression (left uncompressed afterward)
    meta_csv = config.WMB_METADATA_CSV
    meta_zst = meta_csv + ".zst"
    if not os.path.exists(meta_csv) and os.path.exists(meta_zst):
        print("  Auto-decompressing WMB metadata CSV (one-time)...")
        subprocess.run(["zstd", "-d", "-f", "-T0", "--rm", "-q", meta_zst],
                       check=True)

    if not zst_files:
        yield
        return

    # Decompress subsets — track paths for recompression
    print(f"  Auto-decompressing {len(zst_files)} subset files "
          f"(~51 GB temporarily)...")
    decompressed = []
    for zst in zst_files:
        subprocess.run(["zstd", "-d", "-f", "-T0", "--rm", "-q", zst],
                       check=True)
        decompressed.append(zst[:-4])  # strip .zst suffix
    print("  Decompression complete.")

    try:
        yield
    finally:
        to_compress = [f for f in decompressed if os.path.exists(f)]
        if to_compress:
            print(f"  Re-compressing {len(to_compress)} subset files...")
            for f in to_compress:
                subprocess.run(["zstd", "-3", "-T0", "--rm", "-q", f],
                               check=True)
            print("  Subsets re-compressed.")


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
# Subset-aware path resolution
# ---------------------------------------------------------------------------


def _get_subset_path(file_key: str) -> Optional[str]:
    """Return subset h5ad path if it exists, else None.

    Subset files are pre-extracted gene-subset h5ad files that contain only
    the proteome + kinase/phosphatase genes (~6,800 of 32,285). They're much
    smaller and faster to read than the full regional files.
    """
    region = file_key.split("/")[0]  # e.g. "WMB-10Xv3-CB"
    subset_path = os.path.join(
        config.WMB_SUBSET_DIR,
        config.WMB_SUBSET_FILENAME_FMT.format(region=region),
    )
    if os.path.exists(subset_path):
        return subset_path
    return None


def _get_full_h5ad_path(file_key: str) -> str:
    """Construct the expected local h5ad path for a WMB region without
    triggering an S3 download via the ABC cache."""
    region = file_key.split("/")[0]  # e.g. "WMB-10Xv3-CB"
    return os.path.join(
        config.ALLEN_ABC_CACHE_DIR, "expression_matrices",
        config.WMB_DATASET_KEY, "20230630",
        f"{region}-log2.h5ad",
    )


def _validate_h5ad(path: str) -> bool:
    """Check that an h5ad file can be opened (not truncated or compressed)."""
    import h5py
    try:
        with h5py.File(path, "r"):
            pass
        return True
    except OSError:
        return False


def _build_gene_index():
    """Build gene symbol → column index mapping.

    Prefers subset h5ad files (smaller, faster) when available. Falls back
    to the full regional files. Returns (atlas_genes, gene_to_idx, gene_fmt).
    """
    import anndata as ad

    first_key = config.WMB_ALL_REGION_KEYS[0]

    # Prefer subset file for index building
    subset_path = _get_subset_path(first_key)
    if subset_path:
        ref_path = subset_path
        print(f"  Gene index: using subset file")
    else:
        ref_path = _get_full_h5ad_path(first_key)
        print(f"  Gene index: using full file")
        if not os.path.exists(ref_path) or not _validate_h5ad(ref_path):
            zst = ref_path + ".zst"
            hint = (" (.zst exists — decompress first, or run "
                    "extract_wmb_gene_subset.py)") if os.path.exists(zst) else ""
            raise OSError(
                f"Cannot open {ref_path}: file missing or truncated{hint}"
            )

    adata_ref = ad.read_h5ad(str(ref_path), backed="r")
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


# ---------------------------------------------------------------------------
# WMB Expression Computation
# ---------------------------------------------------------------------------


def _stream_wmb_expression(
    gene_names: list,
    gene_indices: np.ndarray,
    label: str = "genes",
    chunk_size: int = 5000,
    skip_regional: bool = False,
) -> tuple:
    """Stream through all 13 WMB regions, accumulating per-cell-type expression.

    Uses the 24 SEA-AD subclasses as cell-type categories.

    Returns (accum, regional_rows) where accum is {ct: {expr_sum, nonzero_count,
    n_cells}} and regional_rows is a list of per-region per-gene dicts.
    """
    import anndata as ad

    cache = get_abc_cache()
    n_genes = len(gene_indices)

    ct_list = config.SEA_AD_SUBCLASSES
    map_fn = _match_sea_ad_subclass

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

    # Detect whether subset files are available
    _first_subset = _get_subset_path(config.WMB_ALL_REGION_KEYS[0])
    using_subsets = _first_subset is not None
    if using_subsets:
        print("  Using pre-extracted gene-subset h5ad files (fast path)")
    else:
        print("  Using full regional h5ad files")
        # Pre-compute contiguous slice bounds for full-file fallback
        if len(gene_indices) > 0:
            _sorted_gi = np.sort(gene_indices)
            _min_col, _max_col = int(_sorted_gi[0]), int(_sorted_gi[-1])
            _local_indices = gene_indices - _min_col
            print(f"  Contiguous slice: cols [{_min_col}:{_max_col+1}] "
                  f"({_max_col - _min_col + 1} cols for {n_genes} genes)")

    print(f"  Target {label}: {n_genes}")
    print(f"  Cell type set: 24 SEA-AD subclasses ({len(ct_list)} categories)")

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

        # Prefer subset file, fall back to full file
        subset_path = _get_subset_path(file_key)
        if subset_path:
            region_path = subset_path
        else:
            region_path = _get_full_h5ad_path(file_key)
            if not os.path.exists(region_path) or not _validate_h5ad(region_path):
                zst = region_path + ".zst"
                reason = ("truncated" if os.path.exists(region_path)
                          else "missing (.zst only)" if os.path.exists(zst)
                          else "missing")
                print(f"\n  [{ri}/{len(config.WMB_ALL_REGION_KEYS)}] "
                      f"Region: {region} — SKIPPED ({reason}; "
                      f"decompress or run extract_wmb_gene_subset.py)")
                continue

        print(f"\n  [{ri}/{len(config.WMB_ALL_REGION_KEYS)}] Region: {region}"
              f"{' (subset)' if subset_path else ''}")
        adata = ad.read_h5ad(str(region_path), backed="r")
        print(f"    Shape: {adata.shape}")

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

            if using_subsets:
                # Subset files contain only target genes — read all columns
                chunk_data = adata.X[chunk_rows][:, gene_indices]
            else:
                # Full files: use contiguous column slice for HDF5 efficiency,
                # then pick needed columns in memory
                chunk_slice = adata.X[chunk_rows, _min_col:_max_col + 1]
                if hasattr(chunk_slice, "toarray"):
                    chunk_slice = chunk_slice.toarray()
                chunk_data = chunk_slice[:, _local_indices]

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
                        "cell_type": ct,
                        "mean_log2_expression": round(float(region_mean[i]), 6),
                        "fraction_cells_expressing": round(float(region_frac[i]), 6),
                        "n_cells": n_cells,
                    })

        if hasattr(adata, "file") and adata.file is not None:
            adata.file.close()

    return accum, regional_rows


def compute_wmb_expression(force: bool = False) -> pd.DataFrame:
    """Compute whole-brain kinase/phosphatase expression matrix.

    Streams through all 13 WMB-10Xv3 regional h5ad files, accumulating
    cell-weighted expression sums per cell type.  This correctly models
    whole-brain homogenate: larger regions contribute proportionally more.

    If the cached output exists and force=False, returns the cached result.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("WMB Whole-Brain Kinase Expression Matrix (13 regions)")
    print("=" * 60)

    # Cache staleness check: compare output mtime against kinase mapping cache
    if not force and os.path.exists(WMB_EXPR_FILE):
        mapping_cache = config.MAPPING_CACHE_FILE
        if (not os.path.exists(mapping_cache)
                or os.path.getmtime(WMB_EXPR_FILE) > os.path.getmtime(mapping_cache)):
            print("  Cached kinase expression is up-to-date, skipping recomputation")
            print(f"  (use --force to recompute)")
            return pd.read_csv(WMB_EXPR_FILE)

    with _ensure_subsets_decompressed():
        atlas_genes, gene_to_idx, gene_fmt = _build_gene_index()
        print(f"  Gene format: {gene_fmt}")

        mouse_kinases, _ = get_all_kinase_genes()
        phosphatases = get_phosphatase_genes_from_genelist(atlas_genes)
        all_kp = mouse_kinases | phosphatases

        kinase_genes = sorted(all_kp & set(gene_to_idx.keys()))
        kinase_idx = np.array([gene_to_idx[g] for g in kinase_genes])

        accum, regional_rows = _stream_wmb_expression(
            kinase_genes, kinase_idx, label="kinase/phosphatase",
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


def compute_wmb_proteome_expression(force: bool = False) -> pd.DataFrame:
    """Compute whole-brain expression for all genes in the total proteome.

    Uses the same streaming infrastructure as the kinase computation but
    extracts all ~6,444 proteome genes instead of ~400 kinases.  Output is
    consumed by data_ingest.py --markers for cell-type marker assessment.

    If the cached output is newer than the input gene list, returns the
    cached result immediately (unless force=True).
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("WMB Whole-Brain Proteome Expression Matrix (13 regions)")
    print("=" * 60)

    # Cache staleness check: skip if output is newer than input gene list
    gene_list_path = config.PROTEOME_GENE_LIST_FILE
    out_path = config.WMB_PROTEOME_EXPRESSION_FILE
    if not force and os.path.exists(out_path):
        if (not os.path.exists(gene_list_path)
                or os.path.getmtime(out_path) > os.path.getmtime(gene_list_path)):
            print("  Cached proteome expression is up-to-date, skipping recomputation")
            print(f"  (use --force to recompute)")
            return pd.read_csv(out_path)

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

    with _ensure_subsets_decompressed():
        _, gene_to_idx, gene_fmt = _build_gene_index()
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
            matched_genes, gene_indices, label="proteome", chunk_size=5000,
            skip_regional=True,
        )

        # Build output DataFrame
        print("\n  Computing whole-brain aggregates ...")
        rows = []
        for ct in config.SEA_AD_SUBCLASSES:
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
                    "cell_type": ct,
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
    for ct in config.SEA_AD_SUBCLASSES:
        sub = df[df["cell_type"] == ct].copy()
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
    print(f"  WMB expression: {len(df)} rows, {df['cell_type'].nunique()} cell types")
    for ct in sorted(df["cell_type"].unique()):
        sub = df[df["cell_type"] == ct]
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
    parser.add_argument("--force", action="store_true",
                        help="Force recomputation even if cached results exist")

    args = parser.parse_args()

    if args.run:
        compute_wmb_expression(force=args.force)
    elif args.proteome:
        compute_wmb_proteome_expression(force=args.force)
    elif args.summary:
        print_summary()


if __name__ == "__main__":
    main()

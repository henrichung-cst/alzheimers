#!/usr/bin/env python3
"""WMB Expression Export: supporting surface for unified cell-type attribution.

Computes per-cell-type kinase/phosphatase expression from the Allen Institute
Whole Mouse Brain (WMB) 10Xv3 dataset, pooled across all 13 anatomical
regions (cerebellum, cortical subplate, hippocampal formation, hypothalamus,
isocortex, midbrain, medulla, olfactory, pons, pallidum, striatum,
thalamus). The output file (wmb_kinase_expression.csv) is consumed by
alz/bulk_mea/attribute.py and alz/bulk_mea/recover.py.

Extracted from the archived pre-stoichiometry concordance logic to isolate
the live supporting dependency.

Inputs:
  data/external/allen_abc/  (WMB 10Xv3 cell metadata + per-region h5ad files
                             across all 13 regions, via ABC Atlas cache)
  Kinase gene universe is fetched via atlas_reference.get_all_kinase_genes,
  which itself reads data/derived/caches/kinase_to_gene_mapping.csv.

Outputs:
  outputs/reports/wmb_expression/wmb_kinase_expression.csv

Optional dependencies:
    pip install git+https://github.com/alleninstitute/abc_atlas_access.git
    pip install anndata

Usage:
    python alz/reference/wmb_expression.py --run       # Compute WMB expression matrix
    python alz/reference/wmb_expression.py --summary   # Print cached results
"""

from __future__ import annotations

import argparse
import contextlib
import glob
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from alz.shared import config
from alz.cross_reference import specificity
from alz.reference.atlas import (
    get_all_kinase_genes,
    get_phosphatase_genes_from_genelist,
    _extract_gene_symbols,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OUTPUT_DIR = config.WMB_EXPRESSION_OUTPUT_DIR
WMB_EXPR_FILE = config.WMB_EXPRESSION_FILE

# Cell-type vocabulary: only the WMB classes that any spine cluster maps to
# (lineage-level lookup; the 19 spine clusters fan into ~9 WMB classes via
# cluster_to_wmb_class.csv). Cells whose class is outside that set are
# accumulated into "Other" and dropped from the output. The full 34-class
# enumeration in config.WMB_CLASSES is retained as an atlas helper but is
# no longer the analysis spine.
RETAINED_WMB_CLASSES = sorted(set(config.load_cluster_to_wmb_class_map().values()))

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


def _wmb_class_label(value) -> str:
    """Return the WMB class label for a cell, or 'Other' if NaN/missing."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "Other"
    s = str(value).strip()
    return s if s else "Other"


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

    first_key = config.wmb_region_keys()[0]

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


def _scope_sidecar_path(csv_path: str) -> str:
    return csv_path.replace(".csv", ".scope.json")


def _cached_scope_matches(csv_path: str) -> bool:
    path = _scope_sidecar_path(csv_path)
    if not os.path.exists(path):
        return False
    try:
        with open(path) as f:
            return json.load(f).get("scope") == config.WMB_REGION_SCOPE
    except (OSError, ValueError):
        return False


def _write_scope_sidecar(csv_path: str) -> None:
    with open(_scope_sidecar_path(csv_path), "w") as f:
        json.dump(
            {"scope": config.WMB_REGION_SCOPE, "regions": config.wmb_region_keys()},
            f,
            indent=2,
        )


def _stream_wmb_expression(
    gene_names: list,
    gene_indices: np.ndarray,
    label: str = "genes",
    chunk_size: int = 5000,
    skip_regional: bool = False,
) -> tuple:
    """Stream through all 13 WMB regions, accumulating per-cell-type expression.

    Cell types are the 34 WMB classes from the published Allen taxonomy
    (wmb_meta["class"] column). A per-WMB-subclass accumulator is also
    maintained in parallel for the audit sidecar.

    Returns (accum, regional_rows, subclass_accum) where accum and
    subclass_accum are {ct: {expr_sum, nonzero_count, n_cells}} keyed on
    class label and subclass label respectively.
    """
    import anndata as ad

    n_genes = len(gene_indices)

    ct_list = list(RETAINED_WMB_CLASSES) + ["Other"]

    # Load WMB cell metadata directly from cached CSV — avoids hard dependency
    # on abc_atlas_access (the cache wrapper merely returns this CSV).
    print("  Loading WMB cell metadata ...")
    wmb_meta = pd.read_csv(config.WMB_METADATA_CSV, low_memory=False)
    if "cell_label" in wmb_meta.columns:
        wmb_meta = wmb_meta.set_index("cell_label", drop=False)
    print(f"  Total WMB cells: {len(wmb_meta):,}")

    wmb_meta["_ct_mapped"] = wmb_meta["class"].apply(_wmb_class_label)
    wmb_meta["_subclass_mapped"] = wmb_meta["subclass"].apply(_wmb_class_label)

    meta_index_set = set(wmb_meta.index.tolist())
    wmb_meta_by_label = None
    if "cell_label" in wmb_meta.columns:
        wmb_meta_by_label = wmb_meta.set_index("cell_label")

    # Detect whether subset files are available
    region_keys = config.wmb_region_keys()
    print(f"  WMB region scope: {config.WMB_REGION_SCOPE} "
          f"({len(region_keys)} region files)")
    _first_subset = _get_subset_path(region_keys[0])
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
    print(f"  Cell type set: {len(RETAINED_WMB_CLASSES)} retained WMB classes "
          f"(of {len(config.WMB_CLASSES)} total) + Other")

    # Per-class accumulators (primary)
    accum: Dict[str, Dict] = {}
    for ct in ct_list:
        accum[ct] = {
            "expr_sum": np.zeros(n_genes, dtype=np.float64),
            "nonzero_count": np.zeros(n_genes, dtype=np.int64),
            "n_cells": 0,
        }

    # Per-subclass accumulators (audit sidecar; lazy init as we encounter them)
    subclass_accum: Dict[str, Dict] = {}

    regional_rows = []

    for ri, file_key in enumerate(region_keys, 1):
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
                print(f"\n  [{ri}/{len(region_keys)}] "
                      f"Region: {region} — SKIPPED ({reason}; "
                      f"decompress or run extract_wmb_gene_subset.py)")
                continue

        print(f"\n  [{ri}/{len(region_keys)}] Region: {region}"
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
            if "class" in adata.obs.columns:
                print(f"    Low metadata overlap ({len(common)}) — using h5ad class")
                cell_ct_map = adata.obs["class"].apply(_wmb_class_label).to_dict()
                if "subclass" in adata.obs.columns:
                    cell_subclass_map = adata.obs["subclass"].apply(_wmb_class_label).to_dict()
                else:
                    cell_subclass_map = {c: "Other" for c in adata.obs.index}
                common = list(adata.obs.index)
            else:
                print(f"    WARNING: Insufficient cell overlap for {region}, skipping")
                if hasattr(adata, "file") and adata.file is not None:
                    adata.file.close()
                continue
        else:
            cell_ct_map = use_meta.loc[common, "_ct_mapped"].to_dict()
            cell_subclass_map = use_meta.loc[common, "_subclass_mapped"].to_dict()

        print(f"    Matched cells: {len(common):,}")

        h5ad_obs_list = list(adata.obs.index)
        h5ad_cell_to_row = {c: i for i, c in enumerate(h5ad_obs_list)}

        row_records = []
        for c in common:
            if c not in h5ad_cell_to_row:
                continue
            ct = cell_ct_map.get(c, "Other")
            sc = cell_subclass_map.get(c, "Other")
            row_records.append((h5ad_cell_to_row[c], ct, sc))
        row_records.sort(key=lambda x: x[0])

        if not row_records:
            print("    No matched cells")
            if hasattr(adata, "file") and adata.file is not None:
                adata.file.close()
            continue

        all_row_indices = np.array([r[0] for r in row_records])
        all_row_cts = [r[1] for r in row_records]
        all_row_subs = [r[2] for r in row_records]

        region_accum: Dict[str, Dict] = {}
        for ct in ct_list:
            region_accum[ct] = {
                "expr_sum": np.zeros(n_genes, dtype=np.float64),
                "nonzero_count": np.zeros(n_genes, dtype=np.int64),
                "n_cells": 0,
            }
        region_subclass_accum: Dict[str, Dict] = {}

        for start in range(0, len(all_row_indices), chunk_size):
            chunk_rows = all_row_indices[start:start + chunk_size]
            chunk_cts = all_row_cts[start:start + chunk_size]
            chunk_subs = all_row_subs[start:start + chunk_size]

            if using_subsets:
                chunk_data = adata.X[chunk_rows][:, gene_indices]
            else:
                chunk_slice = adata.X[chunk_rows, _min_col:_max_col + 1]
                if hasattr(chunk_slice, "toarray"):
                    chunk_slice = chunk_slice.toarray()
                chunk_data = chunk_slice[:, _local_indices]

            if hasattr(chunk_data, "toarray"):
                chunk_data = chunk_data.toarray()

            # Class-level accumulation (vectorized)
            chunk_cts_arr = np.array(chunk_cts)
            unique_cts, inverse = np.unique(chunk_cts_arr, return_inverse=True)
            for ct_idx, ct in enumerate(unique_cts):
                if ct not in region_accum:
                    region_accum[ct] = {
                        "expr_sum": np.zeros(n_genes, dtype=np.float64),
                        "nonzero_count": np.zeros(n_genes, dtype=np.int64),
                        "n_cells": 0,
                    }
                ct_mask = (inverse == ct_idx)
                ct_data = chunk_data[ct_mask]
                region_accum[ct]["expr_sum"] += ct_data.sum(axis=0)
                region_accum[ct]["nonzero_count"] += (ct_data > 0).sum(axis=0)
                region_accum[ct]["n_cells"] += int(ct_mask.sum())

            # Subclass-level accumulation (audit sidecar)
            chunk_subs_arr = np.array(chunk_subs)
            unique_subs, inv_sub = np.unique(chunk_subs_arr, return_inverse=True)
            for sub_idx, sub in enumerate(unique_subs):
                if sub not in region_subclass_accum:
                    region_subclass_accum[sub] = {
                        "expr_sum": np.zeros(n_genes, dtype=np.float64),
                        "nonzero_count": np.zeros(n_genes, dtype=np.int64),
                        "n_cells": 0,
                    }
                sub_mask = (inv_sub == sub_idx)
                sub_data = chunk_data[sub_mask]
                region_subclass_accum[sub]["expr_sum"] += sub_data.sum(axis=0)
                region_subclass_accum[sub]["nonzero_count"] += (sub_data > 0).sum(axis=0)
                region_subclass_accum[sub]["n_cells"] += int(sub_mask.sum())

        for ct, ra in region_accum.items():
            n_cells = ra["n_cells"]
            if n_cells == 0:
                continue

            print(f"    {ct}: {n_cells:,} cells")

            if ct not in accum:
                accum[ct] = {
                    "expr_sum": np.zeros(n_genes, dtype=np.float64),
                    "nonzero_count": np.zeros(n_genes, dtype=np.int64),
                    "n_cells": 0,
                }
            accum[ct]["expr_sum"] += ra["expr_sum"]
            accum[ct]["nonzero_count"] += ra["nonzero_count"]
            accum[ct]["n_cells"] += n_cells

            if not skip_regional:
                region_mean = ra["expr_sum"] / n_cells
                region_frac = ra["nonzero_count"] / n_cells
                for i, gene in enumerate(gene_names):
                    regional_rows.append({
                        "region": region,
                        "gene_symbol": gene,
                        "cell_type": ct,
                        "mean_log2_expression": round(float(region_mean[i]), 6),
                        "fraction_cells_expressing": round(float(region_frac[i]), 6),
                        "n_cells": n_cells,
                    })

        for sub, ra in region_subclass_accum.items():
            n_cells = ra["n_cells"]
            if n_cells == 0:
                continue
            if sub not in subclass_accum:
                subclass_accum[sub] = {
                    "expr_sum": np.zeros(n_genes, dtype=np.float64),
                    "nonzero_count": np.zeros(n_genes, dtype=np.int64),
                    "n_cells": 0,
                }
            subclass_accum[sub]["expr_sum"] += ra["expr_sum"]
            subclass_accum[sub]["nonzero_count"] += ra["nonzero_count"]
            subclass_accum[sub]["n_cells"] += n_cells

        if hasattr(adata, "file") and adata.file is not None:
            adata.file.close()

    return accum, regional_rows, subclass_accum


def _write_attribution_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the standard attribution metric and write the two WMB outputs.

    See alz/cross_reference/specificity.py and
    docs/plans/attribution/standard_attribution_metric.md for the definition. The 9 retained
    WMB classes are already broad lineages, so there is no coarse vs. native
    distinction — the metric is computed once over them (group_col=None).

      wmb_kinase_expression.csv  — per (kinase, class): raw facts (mean,
        fraction, binary_expressed, n_cells) + native metrics (detected,
        linear_expression, concentration, concentration_tier). The share
        `specificity_score` is removed; detection replaces it. "Other" catch-all
        rows are kept for visibility with detection only (excluded from breadth).
      wmb_kinase_specificity.csv — per kinase: breadth summary (n_celltypes
        detected, effective number of cell types, top class, top concentration).
    """
    retained_mask = df["cell_type"].isin(RETAINED_WMB_CLASSES)
    retained = df[retained_mask].copy()

    per_label, _, per_gene = specificity.compute(
        retained, gene_col="gene_symbol", label_col="cell_type",
        mean_log2_col="mean_log2_expression",
        frac_col="fraction_cells_expressing", ncells_col="n_cells")

    # "Other" rows: detection only (frac ≥ 0.10); not part of the breadth
    # denominator, so concentration/tier are undefined.
    other = df[~retained_mask].copy()
    if len(other):
        _oml = other["mean_log2_expression"].to_numpy(dtype=float)
        other["linear_expression"] = np.where(
            np.isfinite(_oml), np.maximum(np.exp2(_oml) - 1.0, 0.0), 0.0)
        other["detected"] = other["fraction_cells_expressing"].astype(float) >= specificity.DETECTION_FRAC_MIN
        other["concentration"] = np.nan
        other["concentration_of_total"] = np.nan
        other["concentration_tier"] = 0

    expr_cols = ["kinase_id", "gene_symbol", "cell_type",
                 "mean_log2_expression", "fraction_cells_expressing",
                 "binary_expressed", "n_cells",
                 "linear_expression", "detected", "concentration",
                 "concentration_of_total", "concentration_tier"]
    expr = pd.concat([per_label, other], ignore_index=True)[expr_cols]
    expr.to_csv(WMB_EXPR_FILE, index=False)
    print(f"\n  Saved {len(expr)} rows to {WMB_EXPR_FILE}")
    _write_scope_sidecar(WMB_EXPR_FILE)

    per_gene = per_gene.rename(columns={
        "n_detected_native": "n_celltypes_detected",
        "effective_n_native": "effective_n_celltypes",
        "top_celltype_native": "top_celltype",
        "top_concentration_native": "top_concentration",
    })
    kid = retained.drop_duplicates("gene_symbol")[["gene_symbol", "kinase_id"]]
    per_gene = per_gene.merge(kid, on="gene_symbol", how="left")
    for c in ("effective_n_celltypes", "top_concentration"):
        per_gene[c] = per_gene[c].astype(float).round(6)
    per_gene.to_csv(config.WMB_KINASE_SPECIFICITY_FILE, index=False)
    print(f"  Saved {len(per_gene)} per-kinase breadth rows to "
          f"{config.WMB_KINASE_SPECIFICITY_FILE}")

    n_det = int((per_gene["n_celltypes_detected"] > 0).sum())
    eff = per_gene.loc[per_gene["n_celltypes_detected"] > 0, "effective_n_celltypes"]
    print(f"  Kinases detected in ≥1 class: {n_det}/{len(per_gene)}; "
          f"median effective # cell types = "
          f"{eff.median():.2f}" if len(eff) else "  (no detected kinases)")
    return expr


def compute_metrics() -> pd.DataFrame:
    """Recompute the standard attribution metric from the existing WMB
    expression CSV — no h5ad re-stream (detection is already present)."""
    if not os.path.exists(WMB_EXPR_FILE):
        raise FileNotFoundError(f"{WMB_EXPR_FILE} missing — run --run first.")
    df = pd.read_csv(WMB_EXPR_FILE)
    needed = {"kinase_id", "gene_symbol", "cell_type", "mean_log2_expression",
              "fraction_cells_expressing", "binary_expressed", "n_cells"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"{WMB_EXPR_FILE} missing columns {missing}")
    return _write_attribution_metrics(df)


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

    if not force and os.path.exists(WMB_EXPR_FILE):
        mapping_cache = config.MAPPING_CACHE_FILE
        scope_match = _cached_scope_matches(WMB_EXPR_FILE)
        mapping_fresh = (not os.path.exists(mapping_cache)
                         or os.path.getmtime(WMB_EXPR_FILE) > os.path.getmtime(mapping_cache))
        if mapping_fresh and scope_match:
            print(f"  Cached kinase expression matches scope={config.WMB_REGION_SCOPE}, skipping")
            print(f"  (use --force to recompute)")
            return pd.read_csv(WMB_EXPR_FILE)
        if not scope_match:
            print(f"  Scope mismatch (cache vs requested {config.WMB_REGION_SCOPE}) — recomputing")

    with _ensure_subsets_decompressed():
        atlas_genes, gene_to_idx, gene_fmt = _build_gene_index()
        print(f"  Gene format: {gene_fmt}")

        mouse_kinases, _ = get_all_kinase_genes()
        phosphatases = get_phosphatase_genes_from_genelist(atlas_genes)
        all_kp = mouse_kinases | phosphatases

        kinase_genes = sorted(all_kp & set(gene_to_idx.keys()))
        kinase_idx = np.array([gene_to_idx[g] for g in kinase_genes])

        accum, regional_rows, subclass_accum = _stream_wmb_expression(
            kinase_genes, kinase_idx, label="kinase/phosphatase",
        )

        # Compute global whole-brain means at WMB class level
        print("\n  Computing whole-brain aggregates ...")
        rows = []
        ct_universe = list(RETAINED_WMB_CLASSES) + ["Other"]
        for ct in ct_universe:
            if ct not in accum:
                continue
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
                    "binary_expressed": bool(
                        mean_expr[i] > 1
                        and frac_expr[i] > specificity.DETECTION_FRAC_MIN),
                    "n_cells": int(n_total),
                })

        df = pd.DataFrame(rows)

        # Standard attribution metric (detection-gated concentration + effective
        # number of cell types) over the retained WMB classes. Replaces the
        # share `specificity_score`, which was shown inversely predictive of
        # presence. Writes wmb_kinase_expression.csv (augmented) +
        # wmb_kinase_specificity.csv (per-kinase breadth). See
        # alz/cross_reference/specificity.py.
        df = _write_attribution_metrics(df)

        # Audit sidecar: per-WMB-subclass expression
        sub_rows = []
        for sub_label, sa in subclass_accum.items():
            n_total = sa["n_cells"]
            if n_total == 0:
                continue
            mean_expr = sa["expr_sum"] / n_total
            frac_expr = sa["nonzero_count"] / n_total
            for i, gene in enumerate(kinase_genes):
                sub_rows.append({
                    "kinase_id": _mouse_to_human(gene),
                    "gene_symbol": gene,
                    "wmb_subclass": sub_label,
                    "mean_log2_expression": round(float(mean_expr[i]), 6),
                    "fraction_cells_expressing": round(float(frac_expr[i]), 6),
                    "n_cells": int(n_total),
                })
        if sub_rows:
            sub_df = pd.DataFrame(sub_rows)
            sub_path = os.path.join(OUTPUT_DIR, "wmb_kinase_expression_subclass.csv")
            sub_df.to_csv(sub_path, index=False)
            print(f"  Saved {len(sub_df)} subclass-level audit rows to {sub_path}")

        # Save per-region breakdown
        if regional_rows:
            regional_df = pd.DataFrame(regional_rows)
            regional_df["kinase_id"] = regional_df["gene_symbol"].apply(_mouse_to_human)
            regional_df.to_csv(config.WMB_REGIONAL_EXPRESSION_FILE, index=False)
            print(f"  Saved {len(regional_df)} regional rows to "
                  f"{config.WMB_REGIONAL_EXPRESSION_FILE}")

    # Summary
    for ct in ct_universe:
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

    gene_list_path = config.PROTEOME_GENE_LIST_FILE
    out_path = config.WMB_PROTEOME_EXPRESSION_FILE
    if not force and os.path.exists(out_path):
        scope_match = _cached_scope_matches(out_path)
        list_fresh = (not os.path.exists(gene_list_path)
                      or os.path.getmtime(out_path) > os.path.getmtime(gene_list_path))
        if list_fresh and scope_match:
            print(f"  Cached proteome expression matches scope={config.WMB_REGION_SCOPE}, skipping")
            print(f"  (use --force to recompute)")
            return pd.read_csv(out_path)
        if not scope_match:
            print(f"  Scope mismatch (cache vs requested {config.WMB_REGION_SCOPE}) — recomputing")

    if not os.path.exists(gene_list_path):
        raise FileNotFoundError(
            f"Proteome gene list not found at {gene_list_path}. "
            "Run: python alz/ingest/song.py --phospho-match"
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

        accum, _, _ = _stream_wmb_expression(
            matched_genes, gene_indices, label="proteome", chunk_size=5000,
            skip_regional=True,
        )

        # Build output DataFrame at WMB class level
        print("\n  Computing whole-brain aggregates ...")
        rows = []
        ct_universe = list(RETAINED_WMB_CLASSES) + ["Other"]
        for ct in ct_universe:
            if ct not in accum:
                continue
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
                    "binary_expressed": bool(
                        mean_expr[i] > 1
                        and frac_expr[i] > specificity.DETECTION_FRAC_MIN),
                    "n_cells": n_total,
                })

        df = pd.DataFrame(rows)

        # Specificity over named WMB classes only (Other excluded from denominator)
        named_mask = df["cell_type"].isin(RETAINED_WMB_CLASSES)
        for gene in matched_genes:
            gene_mask = (df["gene_symbol_mouse"] == gene) & named_mask
            gene_df = df.loc[gene_mask]
            total_expr = gene_df["mean_log2_expression"].sum()
            if total_expr > 0:
                spec = (gene_df["mean_log2_expression"] / total_expr).round(6)
                df.loc[gene_mask, "specificity_score"] = spec.values
            else:
                df.loc[gene_mask, "specificity_score"] = 0.0
        df.loc[~named_mask, "specificity_score"] = 0.0

        out_path = config.WMB_PROTEOME_EXPRESSION_FILE
        df.to_csv(out_path, index=False)
        print(f"\n  Saved {len(df)} rows to {out_path}")

        _write_scope_sidecar(out_path)

    # Summary: top-5 most specific genes per cell type
    for ct in ct_universe:
        sub = df[df["cell_type"] == ct].copy()
        if len(sub) == 0:
            continue
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
    group.add_argument("--metrics", action="store_true",
                       help="Recompute the standard attribution metric from the "
                            "existing expression CSV (no h5ad re-stream)")
    parser.add_argument("--force", action="store_true",
                        help="Force recomputation even if cached results exist")

    args = parser.parse_args()

    if args.run:
        compute_wmb_expression(force=args.force)
    elif args.proteome:
        compute_wmb_proteome_expression(force=args.force)
    elif args.metrics:
        compute_metrics()
    elif args.summary:
        print_summary()


if __name__ == "__main__":
    main()

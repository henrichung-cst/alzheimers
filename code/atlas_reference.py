#!/usr/bin/env python3
"""External Atlas Data Acquisition and Structure Characterization.

Acquires three Allen Institute transcriptomic datasets, characterizes their
structure, and produces mapping/coverage reports.  No analysis is performed --
the structure reports here inform downstream attribution analysis.

See sap_atlas.md for full specification.

Optional dependencies:
    pip install git+https://github.com/alleninstitute/abc_atlas_access.git
    pip install anndata
    pip install boto3   # for SEA-AD S3 access

Usage:
    python code/atlas_reference.py --aging     # Aging Mouse (priority 1)
    python code/atlas_reference.py --sea-ad    # SEA-AD MTG (priority 2)
    python code/atlas_reference.py --wmb       # WMB characterization (priority 3)
    python code/atlas_reference.py --mapping   # Cross-atlas taxonomy mapping
    python code/atlas_reference.py --coverage  # Kinase gene coverage report
    python code/atlas_reference.py --run       # All steps in priority order
    python code/atlas_reference.py --summary   # Print cached results
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

import config

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OUTPUT_DIR = config.ATLAS_REFERENCE_OUTPUT_DIR

WMB_REPORT_FILE = os.path.join(OUTPUT_DIR, "wmb_structure_report.json")
WMB_SUBCLASS_FILE = os.path.join(OUTPUT_DIR, "wmb_subclass_counts.csv")
AGING_REPORT_FILE = os.path.join(OUTPUT_DIR, "aging_mouse_structure_report.json")
SEA_AD_REPORT_FILE = os.path.join(OUTPUT_DIR, "sea_ad_structure_report.json")
TAXONOMY_MAPPING_FILE = os.path.join(OUTPUT_DIR, "taxonomy_mapping.csv")
COVERAGE_FILE = os.path.join(OUTPUT_DIR, "kinase_gene_coverage.csv")
SUMMARY_FILE = os.path.join(OUTPUT_DIR, "atlas_reference_summary.json")

# Keyword rules for mapping atlas subclasses to our 5+1 pooling.
# Keys are our resolved labels; values are case-insensitive substrings.
SUBCLASS_KEYWORDS: Dict[str, List[str]] = {
    "Excitatory_neurons": [
        "IT", "ET", "CT", "NP", "L2/3", "L4", "L5", "L6",
        "Glut", "excitat",
    ],
    "GABAergic_neurons": [
        "Lamp5", "Sncg", "Vip", "Sst", "Pvalb",
        "GABA", "inhibit", "Chandelier", "Pax6",
    ],
    "Oligodendrocytes": ["Oligo"],
    "Astrocytes": ["Astro"],
    "Microglia": ["Micro", "PVM"],
}

# Re-export from config (single source of truth)
SEA_AD_SUBCLASSES = config.SEA_AD_SUBCLASSES

# ---------------------------------------------------------------------------
# Lazy imports for optional dependencies
# ---------------------------------------------------------------------------


def _import_abc():
    try:
        from abc_atlas_access.abc_atlas_cache.abc_project_cache import (
            AbcProjectCache,
        )
        return AbcProjectCache
    except ImportError:
        raise ImportError(
            "abc_atlas_access not installed. Run:\n"
            "  pip install git+https://github.com/alleninstitute/abc_atlas_access.git"
        )


def _import_anndata():
    try:
        import anndata
        return anndata
    except ImportError:
        raise ImportError("anndata not installed. Run: pip install anndata")


def _import_boto3():
    try:
        import boto3
        return boto3
    except ImportError:
        raise ImportError("boto3 not installed. Run: pip install boto3")


# ---------------------------------------------------------------------------
# Kinase / phosphatase gene lists (standalone, no aggexp dependency)
# ---------------------------------------------------------------------------


def get_all_kinase_genes() -> Tuple[Set[str], Set[str]]:
    """Return (mouse_symbols, human_symbols) for all kinases in kldata.csv.

    Uses the same conversion logic as sap_data._get_kinase_genes but without
    intersecting with aggexp columns, so this returns the full universe.
    """
    kldata = pd.read_csv(config.KLDATA_FILE, usecols=["GENE_NAME"], low_memory=False)
    human_genes = set(kldata["GENE_NAME"].dropna().unique())
    mouse_genes: Set[str] = set()
    for g in human_genes:
        mouse = g[0].upper() + g[1:].lower() if len(g) > 1 else g.upper()
        mouse_genes.add(mouse)
    # Also include symbols from the mapping cache
    if os.path.exists(config.MAPPING_CACHE_FILE):
        cache = pd.read_csv(config.MAPPING_CACHE_FILE)
        if "gene_symbol" in cache.columns:
            for g in cache["gene_symbol"].dropna():
                mouse_genes.add(g)
    return mouse_genes, human_genes


def get_phosphatase_genes_from_genelist(gene_set: Set[str]) -> Set[str]:
    """Identify phosphatase genes within a given gene set.

    Matches config.PHOSPHATASE_GENE_PREFIXES as starts-with and
    config.PHOSPHATASE_GENES_EXTRA as exact matches (title-case).
    """
    found: Set[str] = set()
    for gene in gene_set:
        for prefix in config.PHOSPHATASE_GENE_PREFIXES:
            if gene.startswith(prefix):
                found.add(gene)
                break
    for g in config.PHOSPHATASE_GENES_EXTRA:
        if g in gene_set:
            found.add(g)
    return found


def get_all_kp_genes() -> Tuple[Set[str], Set[str], Set[str]]:
    """Return (mouse_kinases, mouse_phosphatases, human_kinases).

    For phosphatases, since we cannot enumerate all prefix-matched genes
    without a reference gene list, we return only the known extras here.
    The full phosphatase set is computed per-atlas via
    get_phosphatase_genes_from_genelist().
    """
    mouse_kin, human_kin = get_all_kinase_genes()
    # Known phosphatase extras (title-case mouse symbols)
    mouse_phos: Set[str] = set(config.PHOSPHATASE_GENES_EXTRA)
    return mouse_kin, mouse_phos, human_kin


# ---------------------------------------------------------------------------
# ABC Atlas cache helper
# ---------------------------------------------------------------------------


def get_abc_cache():
    """Get or create the ABC Atlas project cache."""
    AbcProjectCache = _import_abc()
    cache_dir = Path(config.ALLEN_ABC_CACHE_DIR)
    cache_dir.mkdir(parents=True, exist_ok=True)
    return AbcProjectCache.from_s3_cache(cache_dir)


def _extract_gene_symbols(adata) -> Tuple[Set[str], str]:
    """Extract gene symbols from an anndata object.

    If var_names are Ensembl IDs, looks for a 'gene_symbol' column in var.
    Returns (set_of_symbols, format_description).
    """
    sample = adata.var_names[0] if len(adata.var_names) > 0 else ""
    if sample.startswith("ENSMUS") or sample.startswith("ENSG"):
        # Ensembl IDs — look for gene_symbol column
        for col in ["gene_symbol", "gene_name", "symbol", "name"]:
            if col in adata.var.columns:
                syms = set(adata.var[col].dropna().tolist())
                return syms, f"ensembl_index+{col}"
        return set(adata.var_names.tolist()), "ensembl"
    return set(adata.var_names.tolist()), "gene_symbol"


def _find_dataset_key(cache, pattern: str) -> Optional[str]:
    """Search cache.list_directories for a key matching *pattern* (case-insensitive)."""
    dirs = cache.list_directories
    pat_lower = pattern.lower()
    for d in dirs:
        if pat_lower in d.lower():
            return d
    return None


def _get_expression_path(cache, dataset_key: str, file_key: str) -> Path:
    """Download (if needed) and return local path for an expression matrix.

    ``file_key`` is the value from ``cache.list_expression_matrix_files()``,
    e.g. ``"Zeng-Aging-Mouse-10Xv3/log2"`` or ``"WMB-10Xv3-HPF/log2"``.
    """
    result = cache.get_file_path(directory=dataset_key, file_name=file_key)
    # get_file_path may return a dict (S3 mode) or a Path (local mode)
    if isinstance(result, dict):
        return Path(result["local_path"])
    return Path(result)


# ---------------------------------------------------------------------------
# §1  WMB characterization
# ---------------------------------------------------------------------------


def characterize_wmb() -> dict:
    """Download and characterize WMB metadata and one expression matrix."""
    ad = _import_anndata()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("WMB: Allen Mouse Whole Brain Atlas characterization")
    print("=" * 60)

    cache = get_abc_cache()

    # -- List available directories --
    all_dirs = cache.list_directories
    print(f"  ABC Atlas directories ({len(all_dirs)} total):")
    for d in sorted(all_dirs):
        print(f"    {d}")

    # -- Find WMB dataset --
    wmb_key = config.WMB_DATASET_KEY
    if wmb_key not in all_dirs:
        wmb_key = _find_dataset_key(cache, "WMB-10Xv3") or wmb_key
    print(f"\n  Using WMB dataset key: {wmb_key}")

    # -- Cell metadata (in WMB-10X, not WMB-10Xv3) --
    meta_dir = "WMB-10X"
    print(f"  Downloading WMB cell metadata from {meta_dir} ...")
    cell_meta = cache.get_metadata_dataframe(
        directory=meta_dir, file_name="cell_metadata_with_cluster_annotation",
    )
    print(f"  Cell metadata: {cell_meta.shape[0]:,} cells x {cell_meta.shape[1]} columns")
    print(f"  Columns: {list(cell_meta.columns)}")

    # Taxonomy hierarchy
    taxonomy_cols = [c for c in cell_meta.columns
                     if any(kw in c.lower() for kw in
                            ["class", "subclass", "supertype", "cluster"])]
    print(f"  Taxonomy columns: {taxonomy_cols}")

    # Subclass counts
    subclass_col = next((c for c in taxonomy_cols if "subclass" in c.lower()), None)
    subclass_counts = {}
    if subclass_col:
        vc = cell_meta[subclass_col].value_counts()
        subclass_counts = vc.to_dict()
        vc_df = vc.reset_index()
        vc_df.columns = ["subclass", "cell_count"]
        vc_df.to_csv(WMB_SUBCLASS_FILE, index=False)
        print(f"  Unique subclasses: {len(vc)}")
        print(f"  Saved subclass counts to {WMB_SUBCLASS_FILE}")

    # Region, sex, age
    region_col = next((c for c in cell_meta.columns
                       if "region" in c.lower()), None)
    region_labels = sorted(cell_meta[region_col].dropna().unique().tolist()) if region_col else []
    sex_present = any("sex" in c.lower() for c in cell_meta.columns)
    age_present = any("age" in c.lower() for c in cell_meta.columns)

    # Aging taxonomy mapping
    aging_map_col = next(
        (c for c in cell_meta.columns if "aging" in c.lower() or "mapped" in c.lower()),
        None,
    )

    # -- Expression matrix --
    print(f"\n  Listing expression matrices for {wmb_key} ...")
    expr_files = cache.list_expression_matrix_files(wmb_key)
    print(f"  Available matrices ({len(expr_files)}):")
    for ef in sorted(expr_files):
        print(f"    {ef}")

    # Find a representative region (HPF preferred)
    repr_key = None
    for ef in expr_files:
        if "HPF" in ef and "log2" in ef:
            repr_key = ef
            break
    if repr_key is None:
        # Fallback: first log2 matrix
        for ef in expr_files:
            if "log2" in ef:
                repr_key = ef
                break
    if repr_key is None and expr_files:
        repr_key = expr_files[0]

    gene_info: Dict[str, Any] = {}
    spot_checks: Dict[str, Any] = {}
    kinase_coverage = {"n_kinases_found": 0, "n_kinases_total": 0,
                       "n_phosphatases_found": 0, "missing_kinases": []}

    if repr_key:
        print(f"\n  Downloading expression matrix: {repr_key} ...")
        adata = ad.read_h5ad(
            _get_expression_path(cache, wmb_key, repr_key),
            backed="r",
        )
        print(f"  Expression: {adata.shape[0]:,} cells x {adata.shape[1]:,} genes")

        gene_names, gene_fmt = _extract_gene_symbols(adata)
        gene_info = {
            "n_cells": adata.shape[0],
            "n_genes": adata.shape[1],
            "repr_region": repr_key,
            "gene_id_format": gene_fmt,
            "sample_genes": sorted(list(gene_names))[:10],
        }

        # Kinase/phosphatase coverage
        mouse_kin, human_kin = get_all_kinase_genes()
        found_kin = mouse_kin & gene_names
        phos_in_atlas = get_phosphatase_genes_from_genelist(gene_names)
        missing = sorted(mouse_kin - gene_names)
        kinase_coverage = {
            "n_kinases_found": len(found_kin),
            "n_kinases_total": len(mouse_kin),
            "n_phosphatases_found": len(phos_in_atlas),
            "missing_kinases": missing[:30],  # truncate for readability
            "n_missing": len(missing),
        }
        print(f"  Kinase coverage: {len(found_kin)}/{len(mouse_kin)}")
        print(f"  Phosphatase coverage: {len(phos_in_atlas)}")

        # Spot-check kinases
        # Build symbol→index lookup (handles Ensembl IDs via gene_symbol col)
        sym_to_idx: Dict[str, int] = {}
        if "gene_symbol" in adata.var.columns:
            for i, sym in enumerate(adata.var["gene_symbol"]):
                sym_to_idx[sym] = i
        else:
            for i, sym in enumerate(adata.var_names):
                sym_to_idx[sym] = i

        if subclass_col and subclass_col in adata.obs.columns:
            for kinase in config.WMB_SPOT_CHECK_KINASES:
                if kinase not in sym_to_idx:
                    spot_checks[kinase] = "NOT_FOUND"
                    print(f"  Spot-check {kinase}: NOT FOUND in gene list")
                    continue
                gene_idx = sym_to_idx[kinase]
                expr = adata[:, gene_idx].X
                if hasattr(expr, "toarray"):
                    expr = expr.toarray().ravel()
                else:
                    expr = np.asarray(expr).ravel()
                per_subclass = (
                    pd.Series(expr, index=adata.obs[subclass_col].values)
                    .groupby(level=0)
                    .mean()
                    .sort_values(ascending=False)  # type: ignore[call-overload]
                )
                spot_checks[kinase] = per_subclass.head(10).to_dict()
                print(f"  Spot-check {kinase}: top subclass = "
                      f"{per_subclass.index[0]} ({per_subclass.iloc[0]:.3f})")

        adata.file.close()
    else:
        print("  WARNING: No expression matrix found.")

    # -- Build report --
    report = {
        "dataset": "Allen Mouse Whole Brain Atlas (WMB)",
        "dataset_key": wmb_key,
        "status": "complete",
        "cell_metadata": {
            "n_cells": int(cell_meta.shape[0]),
            "n_columns": int(cell_meta.shape[1]),
            "columns": list(cell_meta.columns),
            "taxonomy_columns": taxonomy_cols,
            "subclass_column": subclass_col,
            "n_subclasses": len(subclass_counts),
            "subclass_counts": {k: int(v) for k, v in subclass_counts.items()},
            "region_column": region_col,
            "region_labels": region_labels,
            "sex_present": sex_present,
            "age_present": age_present,
            "aging_taxonomy_mapping_column": aging_map_col,
        },
        "expression_matrix": gene_info,
        "available_expression_files": sorted(expr_files),
        "kinase_coverage": kinase_coverage,
        "spot_checks": spot_checks,
    }

    with open(WMB_REPORT_FILE, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n  Saved report to {WMB_REPORT_FILE}")
    return report


def download_all_wmb_expression() -> list:
    """Download all 13 WMB-10Xv3 log2 expression matrices (~95 GB total).

    Each file is ~3-7 GB.  The ABC cache is idempotent: files already on
    disk are skipped, so this is safe to re-run after interruption.
    """
    cache = get_abc_cache()
    wmb_key = config.WMB_DATASET_KEY
    paths = []
    for i, file_key in enumerate(config.WMB_ALL_REGION_KEYS, 1):
        region = file_key.split("/")[0].replace("WMB-10Xv3-", "")
        print(f"  [{i}/{len(config.WMB_ALL_REGION_KEYS)}] Downloading {region} ...")
        p = _get_expression_path(cache, wmb_key, file_key)
        print(f"    -> {p}")
        paths.append(p)
    print(f"\n  Downloaded {len(paths)} region files.")
    return paths


# ---------------------------------------------------------------------------
# §2  Aging Mouse Brain Atlas characterization
# ---------------------------------------------------------------------------


def characterize_aging() -> dict:
    """Download and characterize the Allen Aging Mouse Brain Atlas."""
    ad = _import_anndata()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(config.ALLEN_AGING_DIR, exist_ok=True)

    print("=" * 60)
    print("Aging Mouse: Allen Aging Mouse Brain Atlas characterization")
    print("=" * 60)

    cache = get_abc_cache()
    all_dirs = cache.list_directories

    # -- Find aging dataset key --
    aging_key = config.AGING_DATASET_KEY
    if aging_key not in all_dirs:
        aging_key = _find_dataset_key(cache, "aging") or aging_key
        if aging_key not in all_dirs:
            aging_key = _find_dataset_key(cache, "Zeng") or aging_key
    print(f"  Using Aging dataset key: {aging_key}")

    if aging_key not in all_dirs:
        print(f"  WARNING: '{aging_key}' not found in ABC Atlas directories.")
        print(f"  Available directories: {sorted(all_dirs)}")
        report = {"dataset": "Allen Aging Mouse Brain Atlas", "status": "error",
                  "error": f"Dataset key '{aging_key}' not found",
                  "available_directories": sorted(all_dirs)}
        with open(AGING_REPORT_FILE, "w") as f:
            json.dump(report, f, indent=2)
        return report

    # -- Cell metadata --
    print("  Downloading Aging Mouse cell metadata ...")
    cell_meta = cache.get_metadata_dataframe(
        directory=aging_key, file_name="cell_metadata",
    )
    print(f"  Cell metadata: {cell_meta.shape[0]:,} cells x {cell_meta.shape[1]} columns")
    print(f"  Columns: {list(cell_meta.columns)}")

    # Taxonomy hierarchy
    taxonomy_cols = [c for c in cell_meta.columns
                     if any(kw in c.lower() for kw in
                            ["class", "subclass", "supertype", "cluster"])]
    subclass_col = next((c for c in taxonomy_cols if "subclass" in c.lower()), None)
    subclass_counts = {}
    if subclass_col:
        vc = cell_meta[subclass_col].value_counts()
        subclass_counts = vc.to_dict()

    # Age groups — prefer donor_age_category (aged/adult) over donor_age (P549 etc.)
    age_cat_col = next(
        (c for c in cell_meta.columns if "age_category" in c.lower()), None,
    )
    age_col = age_cat_col or next(
        (c for c in cell_meta.columns if "age" in c.lower()), None,
    )
    age_info: Dict[str, Any] = {}
    if age_col:
        age_values = cell_meta[age_col].value_counts()
        age_info = {
            "column": age_col,
            "values": {str(k): int(v) for k, v in age_values.items()},
            "n_groups": len(age_values),
        }
        print(f"  Age column: {age_col}")
        print(f"  Age groups: {dict(age_values)}")

    # Sex
    sex_col = next((c for c in cell_meta.columns if "sex" in c.lower()), None)
    sex_info: Dict[str, Any] = {}
    if sex_col:
        sex_values = cell_meta[sex_col].value_counts()
        sex_info = {
            "column": sex_col,
            "values": {str(k): int(v) for k, v in sex_values.items()},
        }

    # WMB taxonomy mapping
    wmb_map_col = next(
        (c for c in cell_meta.columns
         if any(kw in c.lower() for kw in ["wmb", "whole_brain", "mapped", "supertype"])),
        None,
    )

    # Brain regions
    region_col = next((c for c in cell_meta.columns if "region" in c.lower()), None)
    region_labels = sorted(cell_meta[region_col].dropna().unique().tolist()) if region_col else []

    # Cell counts per age x subclass (for MAST feasibility)
    age_subclass_counts: Dict[str, Dict[str, int]] = {}
    if age_col and subclass_col:
        for age_val in cell_meta[age_col].unique():
            mask = cell_meta[age_col] == age_val
            vc = cell_meta.loc[mask, subclass_col].value_counts()
            age_subclass_counts[str(age_val)] = {str(k): int(v) for k, v in vc.items()}

    # -- Search for pre-computed DE results --
    print("\n  Searching for pre-computed DE results ...")
    de_found = False
    de_info: Dict[str, Any] = {}

    # Check all metadata/supplementary files in the aging directory
    try:
        aging_files = cache.list_metadata_files(aging_key)
        print(f"  Metadata files in aging directory: {aging_files}")
        de_patterns = ["de", "differential", "mast", "age_effect", "supplementary"]
        for fname in aging_files:
            if any(pat in fname.lower() for pat in de_patterns):
                print(f"    Potential DE file: {fname}")
                de_found = True
                de_info["file_name"] = fname
                try:
                    de_path = cache.get_metadata_path(aging_key, fname)
                    de_df = pd.read_csv(de_path)
                    de_info.update({
                        "shape": list(de_df.shape),
                        "columns": list(de_df.columns),
                        "sample_rows": de_df.head(3).to_dict(orient="records"),
                    })
                    # Copy to expected location
                    aging_de_file = os.path.join(config.ALLEN_AGING_DIR,
                                                 "aging_de_results.csv")
                    de_df.to_csv(aging_de_file, index=False)
                    de_info["saved_to"] = aging_de_file
                    print(f"    Saved DE table to {aging_de_file}")
                except Exception as e:
                    de_info["read_error"] = str(e)
    except Exception as e:
        print(f"  Could not list metadata files: {e}")
        de_info["list_error"] = str(e)

    if not de_found:
        print("  No pre-computed DE table found in ABC Atlas data.")
        print("  Check Nature paper supplementary tables for MAST results.")
        de_info["status"] = "not_found_in_abc"

    # -- Cluster annotations (may contain cluster_age_bias) --
    cluster_age_bias_info: Dict[str, Any] = {}
    try:
        cluster_ann = cache.get_metadata_dataframe(
            directory=aging_key, file_name="cell_cluster_annotations",
        )
        print(f"\n  Cluster annotations: {cluster_ann.shape[0]:,} rows x "
              f"{cluster_ann.shape[1]} columns")
        print(f"  Columns: {list(cluster_ann.columns)}")
        if "cluster_age_bias" in cluster_ann.columns:
            bias_counts = cluster_ann["cluster_age_bias"].value_counts()
            cluster_age_bias_info = {
                "column": "cluster_age_bias",
                "values": {str(k): int(v) for k, v in bias_counts.items()},
                "note": "Per-cell cluster age bias label from Allen",
            }
            print(f"  cluster_age_bias: {dict(bias_counts)}")
    except Exception as e:
        print(f"  Could not load cluster annotations: {e}")

    # -- Expression matrix --
    print(f"\n  Listing expression matrices for {aging_key} ...")
    try:
        expr_files = cache.list_expression_matrix_files(aging_key)
        print(f"  Available matrices ({len(expr_files)}):")
        for ef in sorted(expr_files):
            print(f"    {ef}")
    except Exception as e:
        print(f"  Could not list expression files: {e}")
        expr_files = []

    gene_info: Dict[str, Any] = {}
    kinase_coverage: Dict[str, Any] = {}

    # Download one representative region for characterization
    repr_key = None
    for ef in expr_files:
        if any(kw in ef.upper() for kw in ["HPF", "CTX", "HIP"]) and "log2" in ef:
            repr_key = ef
            break
    if repr_key is None:
        for ef in expr_files:
            if "log2" in ef:
                repr_key = ef
                break
    if repr_key is None and expr_files:
        repr_key = expr_files[0]

    if repr_key:
        print(f"\n  Downloading expression matrix: {repr_key} ...")
        try:
            adata = ad.read_h5ad(
                _get_expression_path(cache, aging_key, repr_key),
                backed="r",
            )
            gene_names, gene_fmt = _extract_gene_symbols(adata)
            gene_info = {
                "n_cells": adata.shape[0],
                "n_genes": adata.shape[1],
                "repr_region": repr_key,
                "gene_id_format": gene_fmt,
                "sample_genes": sorted(list(gene_names))[:10],
            }
            print(f"  Expression: {adata.shape[0]:,} cells x {adata.shape[1]:,} genes")

            mouse_kin, _ = get_all_kinase_genes()
            found_kin = mouse_kin & gene_names
            phos_in_atlas = get_phosphatase_genes_from_genelist(gene_names)
            kinase_coverage = {
                "n_kinases_found": len(found_kin),
                "n_kinases_total": len(mouse_kin),
                "n_phosphatases_found": len(phos_in_atlas),
                "n_missing": len(mouse_kin - gene_names),
            }
            print(f"  Kinase coverage: {len(found_kin)}/{len(mouse_kin)}")

            adata.file.close()
        except Exception as e:
            print(f"  Error reading expression matrix: {e}")
            gene_info["error"] = str(e)

    # -- Build report --
    report = {
        "dataset": "Allen Aging Mouse Brain Atlas",
        "dataset_key": aging_key,
        "status": "complete",
        "cell_metadata": {
            "n_cells": int(cell_meta.shape[0]),
            "columns": list(cell_meta.columns),
            "taxonomy_columns": taxonomy_cols,
            "subclass_column": subclass_col,
            "n_subclasses": len(subclass_counts),
            "subclass_counts": {k: int(v) for k, v in subclass_counts.items()},
        },
        "age": age_info,
        "sex": sex_info,
        "wmb_taxonomy_mapping_column": wmb_map_col,
        "regions": {"column": region_col, "labels": region_labels},
        "precomputed_de": de_info,
        "cluster_age_bias": cluster_age_bias_info,
        "expression_matrix": gene_info,
        "available_expression_files": sorted(expr_files),
        "kinase_coverage": kinase_coverage,
        "age_subclass_counts": age_subclass_counts,
    }

    with open(AGING_REPORT_FILE, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n  Saved report to {AGING_REPORT_FILE}")
    return report


# ---------------------------------------------------------------------------
# §3  SEA-AD characterization
# ---------------------------------------------------------------------------


def _sea_ad_get_s3_client():
    """Return a boto3 S3 client configured for unsigned access."""
    boto3 = _import_boto3()
    from botocore import UNSIGNED
    from botocore.config import Config as BotoConfig
    return boto3.client("s3", config=BotoConfig(signature_version=UNSIGNED))


def _sea_ad_download_h5ad(target_dir: str) -> Optional[Path]:
    """Download SEA-AD MTG RNAseq h5ad from S3. Returns local path or None."""
    os.makedirs(target_dir, exist_ok=True)
    s3 = _sea_ad_get_s3_client()

    # Search only the MTG/RNAseq prefix for the main h5ad
    print("  Listing SEA-AD MTG/RNAseq files ...")
    paginator = s3.get_paginator("list_objects_v2")
    mtg_files: List[Tuple[str, int]] = []

    for page in paginator.paginate(
        Bucket=config.SEA_AD_S3_BUCKET, Prefix="MTG/RNAseq/",
    ):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            size_mb = obj["Size"] // (1024 * 1024)
            # Only top-level h5ad files (not per-donor objects)
            if key.endswith(".h5ad") and "donor_objects" not in key:
                mtg_files.append((key, size_mb))
                print(f"    {key} ({size_mb:,} MB)")

    if not mtg_files:
        print("  No MTG RNAseq h5ad files found.")
        return None

    # Prefer SEAAD (not Reference) final-nuclei RNAseq
    target_key = mtg_files[0][0]
    for f, _ in mtg_files:
        if ("SEAAD" in f and "final-nuclei" in f.lower()
                and "previous_objects" not in f and "Supplementary" not in f):
            target_key = f
            break

    local_path = Path(target_dir) / Path(target_key).name
    if local_path.exists():
        print(f"  Already downloaded: {local_path}")
        return local_path

    print(f"  Downloading {target_key} to {local_path} ...")
    print("  (This may take a while — file is several GB)")
    s3.download_file(config.SEA_AD_S3_BUCKET, target_key, str(local_path))
    print(f"  Download complete: {local_path}")
    return local_path


def _sea_ad_find_de_files(s3) -> List[str]:
    """Find pre-computed DE files in SEA-AD S3 bucket."""
    paginator = s3.get_paginator("list_objects_v2")
    de_files: List[str] = []
    for page in paginator.paginate(
        Bucket=config.SEA_AD_S3_BUCKET,
        Prefix="MTG/RNAseq/Supplementary Information/Nebula Results/",
    ):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith(".csv") and "Continuous_Pseudo-progression_Score" in key:
                de_files.append(key)
    return de_files


def _sea_ad_download_nebula(target_dir: str) -> Dict[str, Path]:
    """Download small Nebula result h5ad files from SEA-AD S3.

    Returns dict of {name: local_path} for effect_sizes, pvalues, etc.
    These are ~79MB each — much more tractable than the 34GB main h5ad.
    """
    s3 = _sea_ad_get_s3_client()
    os.makedirs(target_dir, exist_ok=True)

    nebula_prefix = "MTG/RNAseq/Supplementary Information/Nebula Results/"
    paginator = s3.get_paginator("list_objects_v2")
    downloaded: Dict[str, Path] = {}

    for page in paginator.paginate(
        Bucket=config.SEA_AD_S3_BUCKET, Prefix=nebula_prefix,
    ):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            # Only download top-level h5ad files (not per-donor CSVs)
            if key.endswith(".h5ad") and key.count("/") <= nebula_prefix.count("/"):
                fname = Path(key).name
                local = Path(target_dir) / fname
                if not local.exists():
                    size_mb = obj["Size"] // (1024 * 1024)
                    print(f"  Downloading {fname} ({size_mb} MB) ...")
                    s3.download_file(config.SEA_AD_S3_BUCKET, key, str(local))
                else:
                    print(f"  Already cached: {fname}")
                name = fname.replace(".h5ad", "")
                downloaded[name] = local
    return downloaded


def characterize_sea_ad() -> dict:
    """Download and characterize the SEA-AD MTG dataset.

    Strategy: download the small Nebula DE results (effect_sizes, pvalues,
    ~79MB each) for gene list and cell-type info. Use ABC Atlas SEAAD-taxonomy
    for metadata. Avoids downloading the 34GB raw h5ad.
    """
    ad = _import_anndata()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(config.SEA_AD_DIR, exist_ok=True)

    print("=" * 60)
    print("SEA-AD: Seattle Alzheimer's Disease MTG characterization")
    print("=" * 60)

    # -- Download Nebula result files (small, ~79MB each) --
    print("  Downloading Nebula DE results ...")
    nebula_files = _sea_ad_download_nebula(config.SEA_AD_DIR)
    print(f"  Downloaded {len(nebula_files)} Nebula files: {list(nebula_files.keys())}")

    # -- Read effect_sizes.h5ad for gene list and cell-type info --
    h5ad_path = nebula_files.get("effect_sizes")
    if h5ad_path is None:
        # Fallback: try downloading the main SEAAD h5ad
        print("  WARNING: No effect_sizes.h5ad found. Trying main h5ad ...")
        h5ad_path = _sea_ad_download_h5ad(config.SEA_AD_DIR)

    if h5ad_path is None:
        report = {"dataset": "SEA-AD MTG", "status": "error",
                  "error": "Could not find any SEA-AD data files"}
        with open(SEA_AD_REPORT_FILE, "w") as f:
            json.dump(report, f, indent=2)
        return report

    # -- Read h5ad --
    print(f"\n  Reading {h5ad_path.name} ...")
    adata = ad.read_h5ad(h5ad_path)
    print(f"  Shape: {adata.shape}")
    print(f"  obs columns: {list(adata.obs.columns)}")
    print(f"  var columns: {list(adata.var.columns)}")

    obs = adata.obs
    var = adata.var

    # -- Determine if this is a Nebula summary (genes×celltypes) or full cell h5ad --
    is_nebula = "effect_sizes" in str(h5ad_path)

    n_nuclei: Optional[int] = None
    n_donors: Optional[int] = None
    donor_meta_cols: List[str] = []
    taxonomy_cols: List[str] = []
    subclass_col: Optional[str] = None
    subclass_counts: Dict[str, int] = {}
    cps_info: Dict[str, Any] = {}
    donor_summary: Dict[str, Any] = {}

    if is_nebula:
        # Nebula structure: obs.index = genes, var.index = cell type supertypes
        # var has columns: Subclass, Class, Supertype
        n_genes_total = obs.shape[0]
        n_cell_types = var.shape[0]
        print(f"  Nebula summary: {n_genes_total:,} genes x {n_cell_types} cell types")

        # Taxonomy from var columns (cell types)
        subclass_col_var = "Subclass" if "Subclass" in var.columns else None
        if subclass_col_var:
            subclass_labels = sorted(var[subclass_col_var].unique().tolist())
            subclass_counts = {sc: int((var[subclass_col_var] == sc).sum())
                               for sc in subclass_labels}
            taxonomy_cols = [c for c in var.columns
                             if c in ("Class", "Subclass", "Supertype")]
            subclass_col = subclass_col_var
            print(f"  Subclasses: {len(subclass_labels)}")
            for sc in subclass_labels:
                print(f"    {sc}: {subclass_counts[sc]} supertypes")

        # Gene names are in obs.index (human symbols)
        gene_names_raw = set(obs.index.tolist())
    else:
        # Full cell-level h5ad
        n_nuclei = obs.shape[0]
        print(f"  Nuclei: {n_nuclei:,}")

        # Donor metadata
        donor_col = next((c for c in obs.columns if "donor" in c.lower()), None)
        n_donors = obs[donor_col].nunique() if donor_col else None
        donor_meta_cols = [c for c in obs.columns
                           if any(kw in c.lower() for kw in
                                  ["donor", "age", "sex", "cerad", "braak",
                                   "adnc", "cps", "cognit", "pmi"])]

        # Taxonomy
        taxonomy_cols = [c for c in obs.columns
                         if any(kw in c.lower() for kw in
                                ["class", "subclass", "supertype", "cluster"])]
        subclass_col = next((c for c in taxonomy_cols if "subclass" in c.lower()), None)
        if subclass_col:
            vc = obs[subclass_col].value_counts()
            subclass_counts = {str(k): int(v) for k, v in vc.items()}

        # CPS
        cps_col = next((c for c in obs.columns if "cps" in c.lower()), None)
        if cps_col:
            cps_vals = obs[cps_col].dropna()
            cps_info = {
                "column": cps_col,
                "n_non_null": int(len(cps_vals)),
                "range": [float(cps_vals.min()), float(cps_vals.max())],
                "mean": float(cps_vals.mean()),
            }

        # Donor summary
        donor_col = next((c for c in obs.columns if "donor" in c.lower()), None)
        if donor_col:
            for mc in donor_meta_cols:
                try:
                    unique = obs.groupby(donor_col)[mc].first().dropna()
                    donor_summary[mc] = {
                        "n_non_null": int(len(unique)),
                        "sample_values": [str(v) for v in unique.head(5).tolist()],
                    }
                except Exception:
                    pass

        # Gene names from var
        gene_names_raw = set(var.index.tolist())
        for col in var.columns:
            if "name" in col.lower() or "symbol" in col.lower():
                gene_names_raw |= set(var[col].dropna().tolist())

    gene_names_upper = {g.upper() for g in gene_names_raw}

    _, human_kin = get_all_kinase_genes()
    # Case-insensitive matching for human genes
    found_by_case = human_kin & gene_names_upper
    # Also check original casing
    found_exact = human_kin & gene_names_raw
    found_kin = found_by_case | found_exact
    n_missing_kin = len(human_kin) - len(found_kin)

    # Phosphatases via prefix matching on uppercase gene list
    phos_in_atlas = get_phosphatase_genes_from_genelist(gene_names_upper)

    gene_id_format = "gene_symbol"
    sample_genes = list(gene_names_raw)[:10]
    if sample_genes and sample_genes[0].startswith("ENS"):
        gene_id_format = "ensembl"

    kinase_coverage = {
        "n_kinases_found": len(found_kin),
        "n_kinases_total": len(human_kin),
        "n_phosphatases_found": len(phos_in_atlas),
        "n_missing_kinases": n_missing_kin,
        "gene_id_format": gene_id_format,
        "case_insensitive_match": len(found_by_case),
        "note": "Human gene symbols — most map to mouse by case change",
    }
    print(f"  Kinase coverage: {len(found_kin)}/{len(human_kin)} (case-insensitive)")
    print(f"  Phosphatase coverage: {len(phos_in_atlas)}")

    if hasattr(adata, "file") and adata.file is not None:
        adata.file.close()
    del adata

    # -- Search for pre-computed DE (Nebula CPS results) --
    print("\n  Searching for pre-computed DE results in S3 ...")
    de_info: Dict[str, Any] = {}
    try:
        s3 = _sea_ad_get_s3_client()
        de_files = _sea_ad_find_de_files(s3)
        if de_files:
            print(f"  Found {len(de_files)} CPS DE files (Nebula)")
            # Download one example to characterize structure
            example_key = de_files[0]
            example_local = Path(config.SEA_AD_DIR) / "example_de.csv"
            if not example_local.exists():
                s3.download_file(config.SEA_AD_S3_BUCKET, example_key,
                                 str(example_local))
            example_df = pd.read_csv(example_local)
            de_info = {
                "status": "found",
                "n_files": len(de_files),
                "example_file": example_key,
                "example_shape": list(example_df.shape),
                "example_columns": list(example_df.columns),
                "cell_type_in_filename": True,
                "note": "Per-cell-type Nebula DE across CPS",
            }
            print(f"  Example DE columns: {list(example_df.columns)}")
            # List unique cell types from filenames
            cell_types = set()
            for f in de_files:
                parts = f.split("/")[-1].split("_across_")[0]
                cell_types.add(parts)
            de_info["cell_types_in_de"] = sorted(cell_types)
            print(f"  Cell types with DE: {len(cell_types)}")
        else:
            print("  No CPS DE files found.")
            de_info = {"status": "not_found"}
    except Exception as e:
        print(f"  Error searching for DE files: {e}")
        de_info = {"status": "error", "error": str(e)}

    # -- Build report --
    report = {
        "dataset": "SEA-AD MTG",
        "status": "complete",
        "h5ad_path": str(h5ad_path),
        "is_nebula_summary": is_nebula,
        "nebula_files": {k: str(v) for k, v in nebula_files.items()},
        "n_nuclei": n_nuclei if not is_nebula else None,
        "n_obs_entries": n_nuclei,
        "n_donors": n_donors,
        "obs_columns": list(obs.columns),
        "var_columns": list(var.columns),
        "donor_metadata_columns": donor_meta_cols,
        "donor_summary": donor_summary,
        "taxonomy": {
            "columns": taxonomy_cols,
            "subclass_column": subclass_col,
            "n_subclasses": len(subclass_counts),
            "subclass_counts": subclass_counts,
        },
        "cps": cps_info,
        "gene_info": {
            "n_genes": len(gene_names_raw),
            "gene_id_format": gene_id_format,
            "sample_genes": sample_genes,
        },
        "kinase_coverage": kinase_coverage,
        "precomputed_de": de_info,
    }

    with open(SEA_AD_REPORT_FILE, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n  Saved report to {SEA_AD_REPORT_FILE}")
    return report


# ---------------------------------------------------------------------------
# §4  Taxonomy mapping
# ---------------------------------------------------------------------------


def match_subclass(label: str) -> str:
    """Map a subclass label to our 5+1 pooling via keyword matching."""
    label_lower = label.lower()
    for our_label, keywords in SUBCLASS_KEYWORDS.items():
        for kw in keywords:
            if kw.lower() in label_lower:
                return our_label
    return "Other"


# Derived: SEA-AD subclass → 5+1 parent mapping
SUBCLASS_TO_5PLUS1 = {sc: match_subclass(sc) for sc in SEA_AD_SUBCLASSES}


def build_taxonomy_mapping() -> pd.DataFrame:
    """Build cross-atlas cell-type mapping table from cached structure reports."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("Taxonomy Mapping: Cross-atlas cell-type alignment")
    print("=" * 60)

    # Load structure reports
    reports: Dict[str, dict] = {}
    for name, path in [("wmb", WMB_REPORT_FILE),
                       ("aging", AGING_REPORT_FILE),
                       ("sea_ad", SEA_AD_REPORT_FILE)]:
        if os.path.exists(path):
            with open(path) as f:
                reports[name] = json.load(f)
            print(f"  Loaded {name} report")
        else:
            print(f"  WARNING: {name} report not found at {path}")

    if not reports:
        print("  ERROR: No structure reports found. Run --aging, --sea-ad, --wmb first.")
        return pd.DataFrame()

    # Extract subclass labels and counts from each report
    def _get_subclasses(report: dict) -> Dict[str, int]:
        """Extract subclass → count from a structure report."""
        meta = report.get("cell_metadata", report.get("taxonomy", {}))
        return meta.get("subclass_counts", {})

    wmb_sc = _get_subclasses(reports.get("wmb", {}))
    aging_sc = _get_subclasses(reports.get("aging", {}))
    sea_ad_sc = _get_subclasses(reports.get("sea_ad", {}))

    print(f"  WMB subclasses: {len(wmb_sc)}")
    print(f"  Aging subclasses: {len(aging_sc)}")
    print(f"  SEA-AD subclasses: {len(sea_ad_sc)}")

    # Build mapping rows
    rows: List[Dict[str, Any]] = []

    # Map WMB subclasses
    for sc, count in sorted(wmb_sc.items()):
        our = match_subclass(sc)
        rows.append({
            "our_label": our,
            "wmb_subclass": sc,
            "aging_subclass": "",
            "sea_ad_subclass": "",
            "wmb_cell_count": count,
            "aging_cell_count": 0,
            "sea_ad_cell_count": 0,
            "mapping_confidence": "high" if our != "Other" else "low",
            "notes": "",
        })

    # Map Aging subclasses (may overlap with WMB via taxonomy mapping)
    for sc, count in sorted(aging_sc.items()):
        our = match_subclass(sc)
        # Try to find matching WMB row
        matched = False
        for row in rows:
            if row["our_label"] == our and row["wmb_subclass"].lower() == sc.lower():
                row["aging_subclass"] = sc
                row["aging_cell_count"] = count
                matched = True
                break
        if not matched:
            rows.append({
                "our_label": our,
                "wmb_subclass": "",
                "aging_subclass": sc,
                "sea_ad_subclass": "",
                "wmb_cell_count": 0,
                "aging_cell_count": count,
                "sea_ad_cell_count": 0,
                "mapping_confidence": "medium",
                "notes": "aging-only subclass",
            })

    # Map SEA-AD subclasses
    for sc, count in sorted(sea_ad_sc.items()):
        our = match_subclass(sc)
        matched = False
        for row in rows:
            if row["our_label"] == our and not row["sea_ad_subclass"]:
                row["sea_ad_subclass"] = sc
                row["sea_ad_cell_count"] = count
                matched = True
                break
        if not matched:
            rows.append({
                "our_label": our,
                "wmb_subclass": "",
                "aging_subclass": "",
                "sea_ad_subclass": sc,
                "wmb_cell_count": 0,
                "aging_cell_count": 0,
                "sea_ad_cell_count": count,
                "mapping_confidence": "medium",
                "notes": "sea-ad-only subclass",
            })

    df = pd.DataFrame(rows)
    # Sort by our label, then by descending total count
    df["_total"] = df["wmb_cell_count"] + df["aging_cell_count"] + df["sea_ad_cell_count"]
    df = df.sort_values(["our_label", "_total"], ascending=[True, False])
    df = df.drop(columns=["_total"])

    df.to_csv(TAXONOMY_MAPPING_FILE, index=False)
    print(f"\n  Saved taxonomy mapping ({len(df)} rows) to {TAXONOMY_MAPPING_FILE}")

    # Summary per our_label
    for label in config.SAP_CELLTYPES:
        subset = df[df["our_label"] == label]
        n_wmb = subset["wmb_subclass"].astype(bool).sum()
        n_aging = subset["aging_subclass"].astype(bool).sum()
        n_sea = subset["sea_ad_subclass"].astype(bool).sum()
        print(f"  {label}: WMB={n_wmb}, Aging={n_aging}, SEA-AD={n_sea} subclasses")

    return df


# ---------------------------------------------------------------------------
# §5  Kinase gene coverage
# ---------------------------------------------------------------------------


def build_kinase_coverage() -> pd.DataFrame:
    """Build per-gene coverage table across all atlases."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("Kinase Gene Coverage Report")
    print("=" * 60)

    mouse_kin, human_kin = get_all_kinase_genes()
    mouse_phos_extras = set(config.PHOSPHATASE_GENES_EXTRA)

    # Build human→mouse mapping
    human_to_mouse: Dict[str, str] = {}
    for g in human_kin:
        mouse = g[0].upper() + g[1:].lower() if len(g) > 1 else g.upper()
        human_to_mouse[g] = mouse

    # Collect gene lists from structure reports
    atlas_genes: Dict[str, Set[str]] = {}
    for name, path in [("wmb", WMB_REPORT_FILE),
                       ("aging", AGING_REPORT_FILE),
                       ("sea_ad", SEA_AD_REPORT_FILE)]:
        if not os.path.exists(path):
            print(f"  WARNING: {name} report not found, skipping")
            continue
        with open(path) as f:
            report = json.load(f)
        # Extract gene list from the expression matrix info
        expr = report.get("expression_matrix", report.get("gene_info", {}))
        sample_genes = expr.get("sample_genes", [])
        # We don't store the full gene list in the JSON — just the coverage stats.
        # For a precise per-gene table, we'd need to re-read the h5ad.
        atlas_genes[name] = set(sample_genes)  # placeholder

    # For a proper coverage table, we need the full gene lists.
    # Try to read them from cached h5ad files.
    ad = None
    try:
        ad = _import_anndata()
    except ImportError:
        pass

    # WMB genes
    wmb_genes: Set[str] = set()
    if ad:
        wmb_report = {}
        if os.path.exists(WMB_REPORT_FILE):
            with open(WMB_REPORT_FILE) as f:
                wmb_report = json.load(f)
        repr_region = wmb_report.get("expression_matrix", {}).get("repr_region")
        if repr_region:
            try:
                cache = get_abc_cache()
                path = _get_expression_path(
                    cache,
                    wmb_report.get("dataset_key", config.WMB_DATASET_KEY),
                    repr_region,
                )
                adata = ad.read_h5ad(path, backed="r")
                wmb_genes, _ = _extract_gene_symbols(adata)
                adata.file.close()
                print(f"  WMB: {len(wmb_genes):,} genes")
            except Exception as e:
                print(f"  Could not read WMB expression: {e}")

    # Aging genes
    aging_genes: Set[str] = set()
    if ad:
        aging_report = {}
        if os.path.exists(AGING_REPORT_FILE):
            with open(AGING_REPORT_FILE) as f:
                aging_report = json.load(f)
        repr_region = aging_report.get("expression_matrix", {}).get("repr_region")
        if repr_region:
            try:
                cache = get_abc_cache()
                path = _get_expression_path(
                    cache,
                    aging_report.get("dataset_key", config.AGING_DATASET_KEY),
                    repr_region,
                )
                adata = ad.read_h5ad(path, backed="r")
                aging_genes, _ = _extract_gene_symbols(adata)
                adata.file.close()
                print(f"  Aging: {len(aging_genes):,} genes")
            except Exception as e:
                print(f"  Could not read Aging expression: {e}")

    # SEA-AD genes — Nebula h5ad has genes in obs.index (not var)
    sea_ad_genes: Set[str] = set()
    if ad:
        sea_ad_report = {}
        if os.path.exists(SEA_AD_REPORT_FILE):
            with open(SEA_AD_REPORT_FILE) as f:
                sea_ad_report = json.load(f)
        h5ad_path = sea_ad_report.get("h5ad_path")
        if h5ad_path and os.path.exists(h5ad_path):
            try:
                adata = ad.read_h5ad(h5ad_path)
                is_nebula = sea_ad_report.get("is_nebula_summary", False)
                if is_nebula:
                    # Nebula: genes are obs.index, cell types are var.index
                    sea_ad_genes = set(adata.obs.index.tolist())
                else:
                    sea_ad_genes, _ = _extract_gene_symbols(adata)
                    if hasattr(adata, "file") and adata.file is not None:
                        adata.file.close()
                print(f"  SEA-AD: {len(sea_ad_genes):,} genes")
            except Exception as e:
                print(f"  Could not read SEA-AD expression: {e}")

    sea_ad_genes_upper = {g.upper() for g in sea_ad_genes}

    # Build per-gene table
    rows: List[Dict[str, Any]] = []

    # Kinase genes
    for human_sym in sorted(human_kin):
        mouse_sym = human_to_mouse.get(human_sym, human_sym)
        rows.append({
            "gene": mouse_sym,
            "type": "kinase",
            "in_kldata": True,
            "in_wmb": mouse_sym in wmb_genes if wmb_genes else None,
            "in_aging": mouse_sym in aging_genes if aging_genes else None,
            "in_sea_ad": human_sym in sea_ad_genes_upper if sea_ad_genes else None,
            "mouse_symbol": mouse_sym,
            "human_symbol": human_sym,
        })

    # Phosphatase extras
    for gene in sorted(mouse_phos_extras):
        human_sym = gene.upper()
        rows.append({
            "gene": gene,
            "type": "phosphatase",
            "in_kldata": False,
            "in_wmb": gene in wmb_genes if wmb_genes else None,
            "in_aging": gene in aging_genes if aging_genes else None,
            "in_sea_ad": human_sym in sea_ad_genes_upper if sea_ad_genes else None,
            "mouse_symbol": gene,
            "human_symbol": human_sym,
        })

    # Phosphatase prefix-matched genes (from WMB genes as reference)
    if wmb_genes:
        phos_from_wmb = get_phosphatase_genes_from_genelist(wmb_genes)
        for gene in sorted(phos_from_wmb - mouse_phos_extras):
            human_sym = gene.upper()
            rows.append({
                "gene": gene,
                "type": "phosphatase",
                "in_kldata": False,
                "in_wmb": True,
                "in_aging": gene in aging_genes if aging_genes else None,
                "in_sea_ad": human_sym in sea_ad_genes_upper if sea_ad_genes else None,
                "mouse_symbol": gene,
                "human_symbol": human_sym,
            })

    df = pd.DataFrame(rows)
    df.to_csv(COVERAGE_FILE, index=False)

    # Summary stats
    if len(df) > 0:
        for atlas, col in [("WMB", "in_wmb"), ("Aging", "in_aging"), ("SEA-AD", "in_sea_ad")]:
            valid = df[col].dropna()
            if len(valid) > 0:
                n_found = valid.sum()
                print(f"  {atlas}: {int(n_found)}/{len(valid)} genes found")

    print(f"\n  Saved coverage table ({len(df)} genes) to {COVERAGE_FILE}")
    return df


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_all() -> None:
    """Run all characterization steps in priority order."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("External Atlas Data Acquisition — Full Pipeline")
    print("=" * 60)

    # Priority 1: Aging Mouse
    print("\n--- Step 1/5: Aging Mouse Brain Atlas ---")
    aging_report = characterize_aging()

    # Priority 2: SEA-AD
    print("\n--- Step 2/5: SEA-AD MTG ---")
    sea_ad_report = characterize_sea_ad()

    # Priority 3: WMB
    print("\n--- Step 3/5: Allen Mouse Whole Brain Atlas ---")
    wmb_report = characterize_wmb()

    # Cross-atlas mapping
    print("\n--- Step 4/5: Taxonomy Mapping ---")
    build_taxonomy_mapping()

    # Coverage report
    print("\n--- Step 5/5: Kinase Gene Coverage ---")
    build_kinase_coverage()

    # Master summary
    summary: Dict[str, Any] = {
        "script": "atlas_reference",
        "description": "External Atlas Data Acquisition and Structure Characterization",
        "aging_mouse": {
            "status": aging_report.get("status", "unknown"),
            "report": AGING_REPORT_FILE,
        },
        "sea_ad": {
            "status": sea_ad_report.get("status", "unknown"),
            "report": SEA_AD_REPORT_FILE,
        },
        "wmb": {
            "status": wmb_report.get("status", "unknown"),
            "report": WMB_REPORT_FILE,
        },
        "taxonomy_mapping": TAXONOMY_MAPPING_FILE,
        "kinase_coverage": COVERAGE_FILE,
    }
    with open(SUMMARY_FILE, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Master summary saved to {SUMMARY_FILE}")


def print_summary() -> None:
    """Print cached summary from all reports."""
    if not os.path.exists(SUMMARY_FILE):
        # Try individual reports
        found_any = False
        for name, path in [("WMB", WMB_REPORT_FILE),
                           ("Aging Mouse", AGING_REPORT_FILE),
                           ("SEA-AD", SEA_AD_REPORT_FILE)]:
            if os.path.exists(path):
                found_any = True
                with open(path) as f:
                    report = json.load(f)
                print(f"\n{'=' * 60}")
                print(f"{name} Report")
                print(f"{'=' * 60}")
                print(json.dumps(report, indent=2))
        if not found_any:
            print("No cached results. Run --run or individual steps first.")
        return

    with open(SUMMARY_FILE) as f:
        summary = json.load(f)
    print(json.dumps(summary, indent=2))

    # Also print key stats from individual reports
    for name, path in [("WMB", WMB_REPORT_FILE),
                       ("Aging Mouse", AGING_REPORT_FILE),
                       ("SEA-AD", SEA_AD_REPORT_FILE)]:
        if os.path.exists(path):
            with open(path) as f:
                report = json.load(f)
            status = report.get("status", "unknown")
            n_cells = (report.get("cell_metadata", {}).get("n_cells")
                       or report.get("n_nuclei", "?"))
            cov = report.get("kinase_coverage", {})
            n_kin = cov.get("n_kinases_found", "?")
            n_total = cov.get("n_kinases_total", "?")
            print(f"\n  {name}: status={status}, cells={n_cells}, "
                  f"kinase coverage={n_kin}/{n_total}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="External Atlas Data Acquisition "
                    "and Structure Characterization",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run", action="store_true",
                       help="Run all steps (Aging > SEA-AD > WMB > mapping > coverage)")
    group.add_argument("--aging", action="store_true",
                       help="Aging Mouse Brain Atlas only")
    group.add_argument("--sea-ad", action="store_true",
                       help="SEA-AD MTG only")
    group.add_argument("--wmb", action="store_true",
                       help="WMB characterization only")
    group.add_argument("--wmb-download", action="store_true",
                       help="Download all 13 WMB-10Xv3 log2 expression matrices (~95 GB)")
    group.add_argument("--mapping", action="store_true",
                       help="Cross-atlas taxonomy mapping")
    group.add_argument("--coverage", action="store_true",
                       help="Kinase gene coverage report")
    group.add_argument("--summary", action="store_true",
                       help="Print cached results")
    args = parser.parse_args()

    if args.summary:
        print_summary()
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if args.run:
        run_all()
    elif args.aging:
        characterize_aging()
    elif args.sea_ad:
        characterize_sea_ad()
    elif args.wmb:
        characterize_wmb()
    elif args.wmb_download:
        download_all_wmb_expression()
    elif args.mapping:
        build_taxonomy_mapping()
    elif args.coverage:
        build_kinase_coverage()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""External Atlas Data Acquisition.

Acquires Allen Institute transcriptomic datasets used by the live pipeline:
  - WMB (Allen Mouse Whole Brain): per-region log2 expression matrices, downloaded
    on demand by the WMB cache.  Used by `wmb_expression.py`.
  - SEA-AD (Seattle Alzheimer's Disease MTG): pre-computed Nebula effect-size h5ads
    (`effect_sizes{,_early,_late}.h5ad`).  Used by `kinase_attribute.py` for
    transcriptomic concordance.
  - SEA-AD MTG expression: per-supertype mean expression from the full donor-level
    h5ad (~50 GB).  Used by `human_reference_expression.py`.  Phase-2 download.
  - Allen Human Brain Cell Atlas (HBCA): per-class expression via abc_atlas_access.
    Human analog of WMB.  Used by `human_reference_expression.py`.  Phase-2 download.

Also exports a small set of helpers consumed by `wmb_expression.py` and
`runners/supporting/extract_wmb_gene_subset.py`:
  get_all_kinase_genes, get_phosphatase_genes_from_genelist, get_all_kp_genes,
  get_abc_cache, _extract_gene_symbols, _find_dataset_key, _get_expression_path

Optional dependencies:
    pip install git+https://github.com/alleninstitute/abc_atlas_access.git
    pip install anndata
    pip install boto3   # for SEA-AD S3 access

Usage:
    python alz/atlas_reference.py --sea-ad            # Download SEA-AD effect-size h5ads
    python alz/atlas_reference.py --wmb-download      # Download all 13 WMB-10Xv3 log2 matrices (~95 GB)
    python alz/atlas_reference.py --run               # SEA-AD + WMB download
    python alz/atlas_reference.py --sea-ad-full       # Fallback: full SEA-AD MTG h5ad (~34 GB)
    python alz/atlas_reference.py --sea-ad-expression # Phase 2: compute per-supertype mean expression
    python alz/atlas_reference.py --hbca-download     # Phase 2: download Allen HBCA (~95 GB)
"""

from __future__ import annotations

import argparse
import os

from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd

import config

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


def _import_boto3():
    try:
        import boto3
        return boto3
    except ImportError:
        raise ImportError("boto3 not installed. Run: pip install boto3")


# ---------------------------------------------------------------------------
# Kinase / phosphatase gene lists
# ---------------------------------------------------------------------------


def get_all_kinase_genes() -> Tuple[Set[str], Set[str]]:
    """Return (mouse_symbols, human_symbols) for all kinases in kldata.csv."""
    kldata = pd.read_csv(config.KLDATA_FILE, usecols=["GENE_NAME"], low_memory=False)
    human_genes = set(kldata["GENE_NAME"].dropna().unique())
    mouse_genes: Set[str] = set()
    for g in human_genes:
        mouse = g[0].upper() + g[1:].lower() if len(g) > 1 else g.upper()
        mouse_genes.add(mouse)
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

    Phosphatases here are only the known extras; the full atlas-resolved set
    is computed per-atlas via get_phosphatase_genes_from_genelist().
    """
    mouse_kin, human_kin = get_all_kinase_genes()
    mouse_phos: Set[str] = set(config.PHOSPHATASE_GENES_EXTRA)
    return mouse_kin, mouse_phos, human_kin


# ---------------------------------------------------------------------------
# ABC Atlas cache helpers
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
    """Download (if needed) and return local path for an expression matrix."""
    result = cache.get_file_path(directory=dataset_key, file_name=file_key)
    if isinstance(result, dict):
        return Path(result["local_path"])
    return Path(result)


# ---------------------------------------------------------------------------
# WMB acquisition
# ---------------------------------------------------------------------------


def download_all_wmb_expression() -> List[Path]:
    """Download all 13 WMB-10Xv3 log2 expression matrices (~95 GB total).

    Each file is ~3-7 GB.  The ABC cache is idempotent: files already on
    disk are skipped, so this is safe to re-run after interruption.
    """
    cache = get_abc_cache()
    wmb_key = config.WMB_DATASET_KEY
    paths: List[Path] = []
    for i, file_key in enumerate(config.WMB_ALL_REGION_KEYS, 1):
        region = file_key.split("/")[0].replace("WMB-10Xv3-", "")
        print(f"  [{i}/{len(config.WMB_ALL_REGION_KEYS)}] Downloading {region} ...")
        p = _get_expression_path(cache, wmb_key, file_key)
        print(f"    -> {p}")
        paths.append(p)
    return paths


# ---------------------------------------------------------------------------
# SEA-AD acquisition
# ---------------------------------------------------------------------------


def _sea_ad_get_s3_client():
    """Return a boto3 S3 client configured for unsigned access."""
    boto3 = _import_boto3()
    from botocore import UNSIGNED
    from botocore.config import Config as BotoConfig
    return boto3.client("s3", config=BotoConfig(signature_version=UNSIGNED))


def _sea_ad_download_nebula(target_dir: str) -> Dict[str, Path]:
    """Download Nebula result h5ad files (effect_sizes{,_early,_late}) from SEA-AD S3.

    These are ~79MB each — much more tractable than the 34GB raw h5ad.
    Returns dict of {name: local_path}.
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
                downloaded[fname.replace(".h5ad", "")] = local
    return downloaded


def _sea_ad_download_main_h5ad(target_dir: str) -> Optional[Path]:
    """Fallback: download the full SEA-AD MTG RNAseq h5ad (~34 GB).

    Not invoked by --run; provided for one-off use if Nebula effect sizes
    are insufficient and the full single-cell object is needed.
    """
    os.makedirs(target_dir, exist_ok=True)
    s3 = _sea_ad_get_s3_client()

    print("  Listing SEA-AD MTG/RNAseq files ...")
    paginator = s3.get_paginator("list_objects_v2")
    mtg_files: List[Tuple[str, int]] = []
    for page in paginator.paginate(
        Bucket=config.SEA_AD_S3_BUCKET, Prefix="MTG/RNAseq/",
    ):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith(".h5ad") and "donor_objects" not in key:
                mtg_files.append((key, obj["Size"] // (1024 * 1024)))

    if not mtg_files:
        print("  No MTG RNAseq h5ad files found.")
        return None

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

    print(f"  Downloading {target_key} -> {local_path} (several GB) ...")
    s3.download_file(config.SEA_AD_S3_BUCKET, target_key, str(local_path))
    return local_path


def download_sea_ad() -> Dict[str, Path]:
    """Acquire SEA-AD effect-size h5ads required by the live pipeline."""
    print("=" * 60)
    print("SEA-AD: Nebula effect-size h5ad acquisition")
    print("=" * 60)
    os.makedirs(config.SEA_AD_DIR, exist_ok=True)
    files = _sea_ad_download_nebula(config.SEA_AD_DIR)
    print(f"\n  {len(files)} file(s) under {config.SEA_AD_DIR}")
    return files


# ---------------------------------------------------------------------------
# SEA-AD MTG per-supertype expression (CR03 Phase-2 download)
# ---------------------------------------------------------------------------


def download_sea_ad_expression(chunk_size: int = 2000, force: bool = False) -> Path:
    """Download the full SEA-AD MTG h5ad and compute per-supertype mean expression.

    The full donor-level h5ad (~50 GB) is streamed in chunks to compute
    log-mean expression per supertype without loading the full matrix into RAM.
    Output: ``data/external/sea_ad/expression_by_supertype.csv``
    (rows = gene HGNC symbols, columns = 139 supertypes).

    Phase-2 only: requires ~50 GB download + ~16 GB RAM headroom.
    Do NOT call during phase 1 (code-only) runs.
    """
    import anndata as ad

    out_path = Path(config.SEA_AD_EXPRESSION_FILE)
    if out_path.exists() and not force:
        print(f"  Cached: {out_path} (use --force to recompute)")
        return out_path

    os.makedirs(config.SEA_AD_DIR, exist_ok=True)

    # Identify and download (if needed) the full MTG h5ad.
    local_h5ad = _sea_ad_download_main_h5ad(config.SEA_AD_DIR)
    if local_h5ad is None or not local_h5ad.exists():
        raise RuntimeError(
            "SEA-AD MTG h5ad not found and could not be downloaded. "
            "Check S3 access and try again."
        )

    print(f"\n  Computing per-supertype mean expression from {local_h5ad}")
    print("  This requires substantial RAM (~8-16 GB); streaming in chunks ...")

    adata = ad.read_h5ad(str(local_h5ad), backed="r")
    print(f"  Shape: {adata.shape}")

    # Determine gene name vector (prefer HGNC symbol column).
    if "gene_symbol" in adata.var.columns:
        gene_names = adata.var["gene_symbol"].tolist()
    elif "feature_name" in adata.var.columns:
        gene_names = adata.var["feature_name"].tolist()
    else:
        gene_names = adata.var_names.tolist()

    # Determine supertype field in obs.
    supertype_field = None
    for cand in ("Supertype", "supertype", "cell_type_alias_label",
                 "supertype_label", "supertypes"):
        if cand in adata.obs.columns:
            supertype_field = cand
            break
    if supertype_field is None:
        raise RuntimeError(
            f"Cannot find supertype column in obs. Available columns: "
            f"{list(adata.obs.columns)}"
        )
    print(f"  Supertype field: '{supertype_field}'")

    supertypes = sorted(adata.obs[supertype_field].dropna().unique().tolist())
    st_to_idx: Dict[str, int] = {s: i for i, s in enumerate(supertypes)}
    n_genes = adata.shape[1]
    n_supertypes = len(supertypes)

    expr_sum = np.zeros((n_genes, n_supertypes), dtype=np.float64)
    cell_count = np.zeros(n_supertypes, dtype=np.int64)

    obs_st = adata.obs[supertype_field].tolist()
    n_cells = adata.shape[0]

    print(f"  Supertypes: {n_supertypes}, Genes: {n_genes}, Cells: {n_cells:,}")

    for start in range(0, n_cells, chunk_size):
        end = min(start + chunk_size, n_cells)
        chunk = adata.X[start:end]
        if hasattr(chunk, "toarray"):
            chunk = chunk.toarray()

        for local_i, global_i in enumerate(range(start, end)):
            st = obs_st[global_i]
            si = st_to_idx.get(st)
            if si is None:
                continue
            expr_sum[:, si] += chunk[local_i].astype(np.float64)
            cell_count[si] += 1

        if (start // chunk_size) % 10 == 0:
            print(f"    ... {end:,}/{n_cells:,} cells processed", flush=True)

    if hasattr(adata, "file") and adata.file is not None:
        adata.file.close()

    # Compute per-supertype mean and write CSV.
    with np.errstate(divide="ignore", invalid="ignore"):
        mean_expr = np.where(
            cell_count[None, :] > 0,
            expr_sum / cell_count[None, :],
            0.0,
        )

    df = pd.DataFrame(mean_expr, index=gene_names, columns=supertypes)
    df.index.name = "gene"
    df.to_csv(out_path)
    print(f"\n  Saved {df.shape} matrix to {out_path}")
    print(f"  Supertypes: {supertypes[:5]} ... (first 5 of {len(supertypes)})")
    return out_path


# ---------------------------------------------------------------------------
# Allen HBCA acquisition (CR03 Phase-2 download)
# ---------------------------------------------------------------------------


def download_hbca() -> Optional[Path]:
    """Download Allen Human Brain Cell Atlas (HBCA) expression matrices.

    Mirrors the WMB download approach: uses abc_atlas_access to stream
    per-region log-expression files.  The HBCA dataset key within
    abc_atlas_access is expected to be a human equivalent of WMB-10Xv3
    (e.g. "WHB-10Xv3" or "HBCA-10Xv3"); the exact key is discovered by
    inspecting ``cache.list_directories``.

    Phase-2 only — this is a multi-hour, ~95 GB download.
    Do NOT call during phase 1 (code-only) runs.

    Returns the local HBCA cache directory path, or None if no matching
    dataset was found.
    """
    cache = get_abc_cache()

    # Discover HBCA dataset key — look for human whole-brain keys.
    candidate_patterns = ["WHB", "HBCA", "human", "Human"]
    hbca_key = None
    print("  Scanning abc_atlas_access for human brain datasets ...")
    dirs = cache.list_directories
    for pat in candidate_patterns:
        found = _find_dataset_key(cache, pat)
        if found:
            hbca_key = found
            print(f"  Found HBCA dataset key: '{hbca_key}' (matched pattern '{pat}')")
            break

    if hbca_key is None:
        print(
            "  WARNING: No HBCA dataset found via abc_atlas_access. "
            "Available directories:\n    " + "\n    ".join(str(d) for d in dirs[:20])
        )
        return None

    os.makedirs(config.HBCA_CACHE_DIR, exist_ok=True)
    print(f"  HBCA cache dir: {config.HBCA_CACHE_DIR}")
    print(f"  Downloading HBCA dataset '{hbca_key}' (multi-GB, multi-hour) ...")

    # List files for this dataset key and download expression matrices.
    try:
        # abc_atlas_access caches automatically; re-running is idempotent.
        downloaded_path = _get_expression_path(cache, hbca_key, "")
    except Exception as exc:
        print(f"  WARNING: Could not enumerate HBCA files: {exc}")
        print("  Use abc_atlas_access directly to identify file keys, then re-run.")
        return None

    print(f"  HBCA download complete → {config.HBCA_CACHE_DIR}")
    return Path(config.HBCA_CACHE_DIR)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def run_all() -> None:
    """SEA-AD effect sizes + WMB log2 expression matrices."""
    print("\n--- Step 1/2: SEA-AD effect sizes ---")
    download_sea_ad()
    print("\n--- Step 2/2: WMB-10Xv3 log2 expression (13 regions) ---")
    download_all_wmb_expression()


def main():
    parser = argparse.ArgumentParser(
        description="External Atlas Data Acquisition (Allen WMB + SEA-AD)",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run", action="store_true",
                       help="Download SEA-AD effect sizes + all WMB-10Xv3 matrices")
    group.add_argument("--sea-ad", action="store_true",
                       help="Download SEA-AD Nebula effect-size h5ads only")
    group.add_argument("--wmb-download", action="store_true",
                       help="Download all 13 WMB-10Xv3 log2 expression matrices (~95 GB)")
    group.add_argument("--sea-ad-full", action="store_true",
                       help="Fallback: download the full SEA-AD MTG h5ad (~34 GB)")
    group.add_argument("--sea-ad-expression", action="store_true",
                       help="Phase 2: download SEA-AD MTG h5ad and compute "
                            "per-supertype mean expression → "
                            "data/external/sea_ad/expression_by_supertype.csv")
    group.add_argument("--hbca-download", action="store_true",
                       help="Phase 2: download Allen Human Brain Cell Atlas (HBCA) "
                            "expression matrices (~95 GB) via abc_atlas_access")
    parser.add_argument("--force", action="store_true",
                        help="Force recompute even if output already exists "
                             "(applies to --sea-ad-expression)")
    args = parser.parse_args()

    if args.run:
        run_all()
    elif args.sea_ad:
        download_sea_ad()
    elif args.wmb_download:
        download_all_wmb_expression()
    elif args.sea_ad_full:
        _sea_ad_download_main_h5ad(config.SEA_AD_DIR)
    elif args.sea_ad_expression:
        download_sea_ad_expression(force=getattr(args, "force", False))
    elif args.hbca_download:
        download_hbca()


if __name__ == "__main__":
    main()

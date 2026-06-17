#!/usr/bin/env python3
"""External Atlas Data Acquisition.

Acquires Allen Institute transcriptomic datasets used by the live pipeline:
  - WMB (Allen Mouse Whole Brain): per-region log2 expression matrices, downloaded
    on demand by the WMB cache.  Used by `wmb_expression.py`.
  - SEA-AD (Seattle Alzheimer's Disease MTG): pre-computed Nebula effect-size h5ads
    (`effect_sizes{,_early,_late}.h5ad`).  Used by `alz/bulk_mea/attribute.py` for
    transcriptomic concordance.
  - SEA-AD MTG expression: per-supertype mean expression from the full donor-level
    h5ad (~50 GB).  Used by `human_expression.py`.  Phase-2 download.
  - Allen Human Brain Cell Atlas (HBCA): per-class expression via abc_atlas_access.
    Human analog of WMB.  Used by `human_expression.py`.  Phase-2 download.

Also exports a small set of helpers consumed by `wmb_expression.py` and
`runners/supporting/extract_wmb_gene_subset.py`:
  get_all_kinase_genes, get_phosphatase_genes_from_genelist, get_all_kp_genes,
  get_abc_cache, _extract_gene_symbols, _find_dataset_key, _get_expression_path

Optional dependencies:
    pip install git+https://github.com/alleninstitute/abc_atlas_access.git
    pip install anndata
    pip install boto3   # for SEA-AD S3 access

Usage:
    python alz/reference/atlas.py --sea-ad            # Download SEA-AD effect-size h5ads
    python alz/reference/atlas.py --wmb-download      # Download all 13 WMB-10Xv3 log2 matrices (~95 GB)
    python alz/reference/atlas.py --run               # SEA-AD + WMB download
    python alz/reference/atlas.py --sea-ad-full       # Fallback: full SEA-AD MTG h5ad (~34 GB)
    python alz/reference/atlas.py --sea-ad-expression # Phase 2: compute per-supertype mean expression
    python alz/reference/atlas.py --hbca-download     # Phase 2: download Allen HBCA (~95 GB)
"""

from __future__ import annotations

import argparse
import os
import sys

from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from alz.shared import config

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
    """Return (mouse_symbols, human_symbols) for all kinases in the kinome.

    Source of truth is `kinase_to_gene_mapping.csv` (kinase-abbreviation →
    HGNC gene symbol), built once by `alz/shared/map_kinases_to_genes.py` from
    `kinase_library.get_kinome_info()`. Mouse symbol is the title-case of
    the human symbol (canonical for mouse orthologs of phospho-active
    kinases; explicit exceptions live in `kinase_to_gene_overrides.csv`).
    """
    if not os.path.exists(config.MAPPING_CACHE_FILE):
        raise RuntimeError(
            f"{config.MAPPING_CACHE_FILE} missing — run "
            f"`pixi run python alz/shared/map_kinases_to_genes.py` first."
        )
    cache = pd.read_csv(config.MAPPING_CACHE_FILE)
    human_genes = {str(g) for g in cache["gene_symbol"].dropna()}
    mouse_genes = {
        g[0].upper() + g[1:].lower() if len(g) > 1 else g.upper()
        for g in human_genes
    }
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


def download_sea_ad_expression(chunk_size: int = 5000, force: bool = False,
                               mem_cap_gb: float = 22.0) -> Path:
    """Download the full SEA-AD MTG h5ad and compute per-supertype mean expression.

    Streams the CSR-sparse X matrix via h5py directly (no anndata backed-mode
    handle) so memory stays bounded across slices. anndata's backed mode leaks
    h5py chunk-cache state across X[start:end] reads and OOMs at ~26 GB on this
    dataset (1.378M cells × 36,601 genes, 7.6B nonzeros).

    A process-level RLIMIT_AS cap (default 22 GB) keeps any blow-up confined to
    this process — kernel global OOM-killer doesn't fire.

    Output: ``data/derived/aggregates/seaad/expression_by_supertype.csv``
    (rows = gene HGNC symbols, columns = 139 supertypes).
    """
    out_path = Path(config.SEA_AD_EXPRESSION_FILE)
    if out_path.exists() and not force:
        print(f"  Cached: {out_path} (use --force to recompute)")
        return out_path

    os.makedirs(config.SEA_AD_DIR, exist_ok=True)

    local_h5ad = _sea_ad_download_main_h5ad(config.SEA_AD_DIR)
    if local_h5ad is None or not local_h5ad.exists():
        raise RuntimeError(
            "SEA-AD MTG h5ad not found and could not be downloaded. "
            "Check S3 access and try again."
        )

    # Cap process VM so MemoryError lands here, not in the OOM-killer.
    if mem_cap_gb and mem_cap_gb > 0:
        import resource
        cap = int(mem_cap_gb * (1024 ** 3))
        soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        resource.setrlimit(resource.RLIMIT_AS,
                           (cap, hard if hard != resource.RLIM_INFINITY else cap))
        print(f"  RLIMIT_AS capped at {mem_cap_gb:g} GB for this process")

    import h5py
    import gc

    print(f"\n  Computing per-supertype mean expression from {local_h5ad}")
    print(f"  chunk_size={chunk_size:,} cells/slice (raw h5py, no anndata)")

    # 32 MB h5py chunk cache per dataset — big enough to amortize sequential reads,
    # bounded enough that it doesn't grow indefinitely.
    with h5py.File(str(local_h5ad), "r", rdcc_nbytes=32 * 1024 * 1024) as f:
        # Gene symbols
        var_index = f["var/_index"][:]
        gene_names = [g.decode() if isinstance(g, bytes) else g for g in var_index]
        n_genes = len(gene_names)

        # Supertype categories + integer codes per cell
        if "Supertype" not in f["obs"]:
            raise RuntimeError(
                f"obs/Supertype not found. Available: {list(f['obs'].keys())[:10]}"
            )
        cats = f["obs/__categories/Supertype"][:]
        supertypes = [c.decode() if isinstance(c, bytes) else c for c in cats]
        n_supertypes = len(supertypes)
        st_codes = f["obs/Supertype"][:]  # int16 codes, ~2.6 MB

        # CSR layout
        if "X" not in f or "indptr" not in f["X"]:
            raise RuntimeError("X is not a CSR sparse group in this h5ad")
        indptr = f["X/indptr"][:]  # 1.378M+1 × int64 ≈ 11 MB
        n_cells = int(indptr.shape[0] - 1)
        data_ds = f["X/data"]
        indices_ds = f["X/indices"]

        print(f"  Supertypes: {n_supertypes}, Genes: {n_genes}, Cells: {n_cells:,}")
        print(f"  Total nonzeros: {int(indptr[-1]):,}")

        expr_sum = np.zeros((n_genes, n_supertypes), dtype=np.float64)
        cell_count = np.zeros(n_supertypes, dtype=np.int64)

        for start in range(0, n_cells, chunk_size):
            end = min(start + chunk_size, n_cells)
            d_start = int(indptr[start])
            d_end = int(indptr[end])

            # Per-chunk reads: contiguous slabs of data + column indices.
            data_chunk = data_ds[d_start:d_end]
            idx_chunk = indices_ds[d_start:d_end]
            row_lens = np.diff(indptr[start:end + 1])

            # Map each nonzero entry to its supertype code via row repetition.
            row_idx_per_entry = np.repeat(np.arange(end - start, dtype=np.int64),
                                          row_lens)
            codes_per_entry = st_codes[start:end][row_idx_per_entry]

            # Scatter-add into expr_sum (no duplicates within a row in CSR, but
            # multiple cells of the same supertype touch the same (gene, code)).
            np.add.at(expr_sum,
                      (idx_chunk.astype(np.intp, copy=False),
                       codes_per_entry.astype(np.intp, copy=False)),
                      data_chunk.astype(np.float64, copy=False))

            # Cell counts via bincount.
            chunk_codes = st_codes[start:end]
            valid = chunk_codes[chunk_codes >= 0]
            if valid.size:
                cell_count += np.bincount(valid.astype(np.int64),
                                          minlength=n_supertypes)

            del data_chunk, idx_chunk, row_lens, row_idx_per_entry, codes_per_entry
            if (start // chunk_size) % 5 == 0:
                gc.collect()
                print(f"    ... {end:,}/{n_cells:,} cells processed", flush=True)

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


def download_hbca(force: bool = False, mem_cap_gb: float = 22.0,
                  chunk_size: int = 5000) -> Optional[Path]:
    """Download Allen HBCA (WHB-10Xv3) log2 expression and aggregate per supercluster.

    Output (the contract consumed by ``alz/reference/human_expression.py``):
        ``data/derived/aggregates/hbca/expression_by_class.csv``
        (rows = gene HGNC symbols, columns = HBCA superclusters).

    Sources (Siletti 2023, ~3.37M nuclei, 31 superclusters):
      * ``WHB-10Xv3-Neurons-log2.h5ad``       (~33 GB)
      * ``WHB-10Xv3-Nonneurons-log2.h5ad``    (~17 GB)
      * ``WHB-10Xv3/cell_metadata.csv``       (cluster_alias per cell)
      * ``WHB-taxonomy/cluster_to_cluster_annotation_membership.csv``
        (cluster_alias → supercluster, filtered to term_set
        ``config.HBCA_TAXONOMY_TERM_SET``).

    Memory: streams CSR-sparse X via h5py (no anndata backed mode), capped at
    ``mem_cap_gb`` GB via RLIMIT_AS so a blow-up lands as MemoryError in this
    process instead of triggering the global OOM-killer.
    """
    out_path = Path(config.HBCA_EXPRESSION_FILE)
    if out_path.exists() and not force:
        print(f"  Cached: {out_path} (use --force to recompute)")
        return out_path

    os.makedirs(config.HBCA_CACHE_DIR, exist_ok=True)
    cache = get_abc_cache()
    hbca_key = _find_dataset_key(cache, "WHB-10Xv3") or _find_dataset_key(cache, "WHB")
    if hbca_key is None:
        print("  ERROR: WHB-10Xv3 not found in abc_atlas_access.")
        return None
    taxonomy_key = _find_dataset_key(cache, "WHB-taxonomy")
    if taxonomy_key is None:
        print("  ERROR: WHB-taxonomy not found in abc_atlas_access.")
        return None
    print(f"  HBCA dataset key: {hbca_key} | taxonomy: {taxonomy_key}")

    # Cap process VM so MemoryError lands here, not in the OOM-killer.
    if mem_cap_gb and mem_cap_gb > 0:
        import resource
        cap = int(mem_cap_gb * (1024 ** 3))
        soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        resource.setrlimit(resource.RLIMIT_AS,
                           (cap, hard if hard != resource.RLIM_INFINITY else cap))
        print(f"  RLIMIT_AS capped at {mem_cap_gb:g} GB for this process")

    # 1. Cluster_alias → supercluster (~31 classes).
    print("\n  Building cluster_alias → supercluster map ...")
    membership = cache.get_metadata_dataframe(
        directory=taxonomy_key,
        file_name="cluster_to_cluster_annotation_membership",
    )
    supc = membership[
        membership["cluster_annotation_term_set_label"] == config.HBCA_TAXONOMY_TERM_SET
    ][["cluster_alias", "cluster_annotation_term_name"]].rename(
        columns={"cluster_annotation_term_name": "supercluster"}
    )
    supc = supc.drop_duplicates("cluster_alias").set_index("cluster_alias")["supercluster"]
    print(f"    {len(supc):,} cluster_alias → {supc.nunique()} superclusters")

    # 2. cell_label → cluster_alias for the whole HBCA (3.37M cells).
    print("\n  Loading cell_metadata (cell_label, cluster_alias) ...")
    cell_meta = cache.get_metadata_dataframe(
        directory=hbca_key, file_name="cell_metadata",
    )[["cell_label", "cluster_alias"]]
    cell_meta["supercluster"] = cell_meta["cluster_alias"].map(supc)
    n_unmapped = cell_meta["supercluster"].isna().sum()
    if n_unmapped:
        print(f"    WARNING: {n_unmapped:,} cells with cluster_alias not in supercluster map")
    cell_to_supc = (
        cell_meta.dropna(subset=["supercluster"])
        .set_index("cell_label")["supercluster"]
    )
    superclusters = sorted(cell_to_supc.unique())
    supc_to_code = {s: i for i, s in enumerate(superclusters)}
    print(f"    {len(superclusters)} superclusters: {superclusters}")

    # 3. Stream each log2 h5ad and accumulate.
    matrices = [
        ("WHB-10Xv3-Neurons-log2.h5ad", "WHB-10Xv3-Neurons"),
        ("WHB-10Xv3-Nonneurons-log2.h5ad", "WHB-10Xv3-Nonneurons"),
    ]

    expr_sum = None
    cell_count = np.zeros(len(superclusters), dtype=np.int64)
    gene_names_ref: Optional[List[str]] = None

    import h5py

    for file_name, subset_label in matrices:
        print(f"\n  Downloading {file_name} (resumable via abc cache) ...")
        local = _get_expression_path(cache, hbca_key, f"{subset_label}/log2")
        # _get_expression_path returns a directory or file depending on the
        # abc cache impl; the actual file should be reachable by name.
        local = Path(local)
        if local.is_dir():
            local = local / file_name
        if not local.exists():
            # Fallback: enumerate the directory.
            cand = list(local.parent.glob(file_name)) if local.parent.exists() else []
            if not cand:
                raise FileNotFoundError(f"Could not locate {file_name} after download (tried {local})")
            local = cand[0]
        print(f"    -> {local} ({local.stat().st_size / 1e9:.1f} GB)")

        with h5py.File(str(local), "r", rdcc_nbytes=32 * 1024 * 1024) as f:
            # var/gene_symbol is a categorical group (cats + codes) carrying HGNC.
            # Use HGNC symbols (not Ensembl gene_identifier) as the row label so
            # the output matches the human kinase gene namespace.
            gs = f["var/gene_symbol"]
            if isinstance(gs, h5py.Group):
                cats = gs["categories"][:]
                code_arr = gs["codes"][:]
                gene_arr = cats[code_arr]
            else:
                gene_arr = gs[:]
            gene_names = [g.decode() if isinstance(g, bytes) else g for g in gene_arr]
            n_genes = len(gene_names)
            if gene_names_ref is None:
                gene_names_ref = gene_names
                expr_sum = np.zeros((n_genes, len(superclusters)), dtype=np.float64)
            elif gene_names != gene_names_ref:
                raise RuntimeError("Gene order mismatch between Neurons and Nonneurons matrices")

            # obs/cell_label — string-array dataset of cell barcodes.
            ol = f["obs/cell_label"]
            if isinstance(ol, h5py.Group):
                ol_cats = ol["categories"][:]
                ol_codes = ol["codes"][:]
                obs_labels = ol_cats[ol_codes]
            else:
                obs_labels = ol[:]
            n_cells = len(obs_labels)
            # Vectorized mapping cell_label -> supercluster code via pandas reindex.
            codes_series = pd.Series(
                [l.decode() if isinstance(l, bytes) else l for l in obs_labels]
            ).map(cell_to_supc).map(supc_to_code)
            codes = codes_series.fillna(-1).to_numpy(dtype=np.int32)
            n_kept = int((codes >= 0).sum())
            print(f"    Cells: {n_cells:,} | mapped to supercluster: {n_kept:,}")

            # CSR streaming
            if "X/indptr" not in f:
                raise RuntimeError(f"{file_name}: X is not CSR-sparse")
            indptr = f["X/indptr"][:]
            data_ds = f["X/data"]
            indices_ds = f["X/indices"]
            print(f"    Genes: {n_genes:,}  Nonzeros: {int(indptr[-1]):,}")

            for start in range(0, n_cells, chunk_size):
                end = min(start + chunk_size, n_cells)
                d_start = int(indptr[start])
                d_end = int(indptr[end])
                data_chunk = data_ds[d_start:d_end]
                idx_chunk = indices_ds[d_start:d_end]
                row_lens = np.diff(indptr[start:end + 1])
                row_idx_per_entry = np.repeat(
                    np.arange(end - start, dtype=np.int64), row_lens
                )
                code_per_entry = codes[start:end][row_idx_per_entry]
                keep = code_per_entry >= 0
                if not keep.any():
                    continue
                np.add.at(
                    expr_sum,
                    (idx_chunk[keep].astype(np.intp, copy=False),
                     code_per_entry[keep].astype(np.intp, copy=False)),
                    data_chunk[keep].astype(np.float64, copy=False),
                )
                # Per-supercluster cell count
                vals, cnts = np.unique(codes[start:end][codes[start:end] >= 0],
                                       return_counts=True)
                cell_count[vals] += cnts
                if (start // chunk_size) % 20 == 0:
                    print(f"      slice {start:,}/{n_cells:,}")

    # 4. Mean = sum / count, then collapse duplicate HGNC symbols by mean
    # (Ensembl→symbol is many-to-one in WHB-10Xv3 var).
    cc = cell_count.astype(np.float64)
    cc[cc == 0] = 1.0  # avoid /0; columns with 0 cells will remain 0
    mean = expr_sum / cc[None, :]
    df = pd.DataFrame(mean, index=gene_names_ref, columns=superclusters)
    df.index.name = "gene"
    n_raw = len(df)
    df = df.groupby(level=0).mean()
    print(f"  collapsed duplicate HGNC symbols: {n_raw:,} -> {len(df):,}")
    df.to_csv(out_path)
    print(f"\n  Wrote {out_path}  ({df.shape[0]:,} genes × {df.shape[1]} superclusters)")
    print(f"    Cells per supercluster: min={int(cell_count.min()):,} "
          f"max={int(cell_count.max()):,} total={int(cell_count.sum()):,}")
    return out_path


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
                            "data/derived/aggregates/seaad/expression_by_supertype.csv")
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

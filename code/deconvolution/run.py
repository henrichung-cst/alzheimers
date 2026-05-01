#!/usr/bin/env python3
"""Orchestrator: Song 46-cluster proportion-proxy → per-cell-type kinase enrichment.

Stages 1-6 from docs/song_deconvolution_plan.md.

This pipeline is a chief-scientist deliverable. It is NOT wired into
`pixi run live` or `pixi run dual` and does NOT reopen the direct
cell-type deconvolution path closed by docs/foundation/analysis_charter.md.

Usage:
    python -m code.deconvolution.run --run               # full pipeline
    python -m code.deconvolution.run --clusters Astrocytes Microglia  # subset
    python -m code.deconvolution.run --tracks st         # ser/thr only
    python -m code.deconvolution.run --permutations 200  # quick MEA
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

# Make `code/` importable so `import config` / `from code.deconvolution …` work
HERE = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.dirname(HERE)
REPO_ROOT = os.path.dirname(CODE_DIR)
sys.path.insert(0, CODE_DIR)
sys.path.insert(0, REPO_ROOT)

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

from deconvolution import paths
from deconvolution.load_deconvoluted import (
    load_track, load_cluster_sizes, load_cluster_mapping,
)
from deconvolution.factorial_ols import run_track
from deconvolution.mea_per_celltype import run_mea
from deconvolution.snrna_concordance import annotate
from deconvolution.confidence import label
from deconvolution.rollup_wmb import aggregate


def _print_header(msg: str) -> None:
    print("=" * 72)
    print(msg)
    print("=" * 72)


def main():
    parser = argparse.ArgumentParser(
        description="Song 46-cluster proportion-proxy → kinase enrichment",
    )
    parser.add_argument("--run", action="store_true",
                        help="Run all stages (1-6) end-to-end")
    parser.add_argument("--clusters", nargs="*", default=None,
                        help="Subset of cluster names (default: all 46)")
    parser.add_argument("--tracks", nargs="*", default=["st", "py"],
                        choices=["st", "py"],
                        help="Phospho tracks to process")
    parser.add_argument("--permutations", type=int, default=None,
                        help="Override MEA permutations (default: config.MEA_PERMUTATION_NUM)")
    parser.add_argument("--ols-only", action="store_true",
                        help="Stop after Stage 2 (OLS); useful for smoke testing")
    parser.add_argument("--summary", action="store_true",
                        help="Print summary of cached outputs")
    args = parser.parse_args()

    if args.summary:
        for name, path in [
            ("OLS site-level", paths.SITE_OLS_FILE),
            ("MEA raw", paths.MEA_FILE),
            ("Primary 46-cluster table", paths.PRIMARY_TABLE),
            ("WMB rollup table", paths.ROLLUP_TABLE),
        ]:
            if os.path.exists(path):
                df = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)
                print(f"  {name}: {path}  ({len(df):,} rows, {len(df.columns)} cols)")
            else:
                print(f"  {name}: NOT FOUND ({path})")
        return

    if not args.run and not args.ols_only:
        parser.print_help()
        sys.exit(1)

    os.makedirs(paths.OUTPUT_DIR, exist_ok=True)
    t0 = time.time()

    # Stage 1 — load
    _print_header("Stage 1  Load pre-computed deconvolution outputs")
    tracks = {}
    for t in args.tracks:
        print(f"  Loading {t} track …")
        tracks[t] = load_track(t)
        print(f"    {len(tracks[t].meta):,} sites × "
              f"{len(tracks[t].samples)} samples × "
              f"{len(tracks[t].clusters)} clusters")

    cluster_size_df = load_cluster_sizes()
    mapping_df = load_cluster_mapping()
    print(f"  Cluster size matrix: {cluster_size_df.shape}")
    print(f"  Cluster→WMB mapping: {len(mapping_df)} rows")

    cluster_subset = args.clusters

    # Stage 2 — males-only factorial OLS per cluster per track
    _print_header("Stage 2  Males-only factorial OLS per cluster per track")
    ols_frames = []
    for t, track in tracks.items():
        print(f"  [{t}] running OLS …")
        df = run_track(track, clusters=cluster_subset)
        if not df.empty:
            ols_frames.append(df)
    if not ols_frames:
        print("  No OLS results produced; exiting.")
        sys.exit(1)
    site_ols = pd.concat(ols_frames, ignore_index=True)
    site_ols.to_parquet(paths.SITE_OLS_FILE, index=False)
    print(f"  Saved {paths.SITE_OLS_FILE} ({len(site_ols):,} rows)")

    if args.ols_only:
        print(f"\n--ols-only set; stopping after Stage 2. Elapsed {time.time()-t0:.1f}s")
        return

    # Stage 3 — two-track MEA per cluster per contrast
    _print_header("Stage 3  Two-track kinase MEA per cluster per contrast")
    mea_frames = []
    for t in args.tracks:
        print(f"  [{t}] running MEA …")
        m = run_mea(site_ols, t, permutation_num=args.permutations)
        if not m.empty:
            mea_frames.append(m)
    if not mea_frames:
        print("  No MEA results produced; exiting.")
        sys.exit(1)
    mea = pd.concat(mea_frames, ignore_index=True)
    mea.to_csv(paths.MEA_FILE, index=False)
    print(f"  Saved {paths.MEA_FILE} ({len(mea):,} rows)")

    # Stage 4 — snRNA cross-check (kinase gene LFC concordance)
    _print_header("Stage 4  snRNA kinase-gene LFC concordance")
    annotated = annotate(mea, mapping_df)
    print(f"  Annotated {len(annotated):,} MEA rows; "
          f"{annotated['kinase_gene_LFC_snRNA'].notna().sum():,} have snRNA match")

    # Stage 5 — confidence calibration
    _print_header("Stage 5  Per-row confidence calibration")
    primary = label(annotated, cluster_size_df)

    primary_cols = [
        "kinase", "cluster", "wmb_class", "contrast", "track",
        "NES", "FDR", "kinase_gene_LFC_snRNA", "kinase_gene_FDR_snRNA",
        "direction_match", "confidence", "n_cells_min",
    ]
    keep = [c for c in primary_cols if c in primary.columns]
    # "Leading substrates" can balloon the CSV by an order of magnitude;
    # the raw MEA table preserves it for downstream inspection.
    drop_cols = {"Leading substrates"}
    extras = [c for c in primary.columns if c not in keep and c not in drop_cols]
    primary = primary[keep + extras]

    rolled = aggregate(primary)

    # Rename to cell_type only at write time; rollup expects 'cluster'.
    primary.rename(columns={"cluster": "cell_type"}).to_csv(paths.PRIMARY_TABLE, index=False)
    print(f"  Saved {paths.PRIMARY_TABLE} ({len(primary):,} rows)")

    conf_counts = primary["confidence"].value_counts().to_dict()
    print(f"  Confidence breakdown: {conf_counts}")

    _print_header("Stage 6  WMB-class rollup (secondary view)")
    rolled.to_csv(paths.ROLLUP_TABLE, index=False)
    print(f"  Saved {paths.ROLLUP_TABLE} ({len(rolled):,} rows)")

    # Summary JSON
    summary = {
        "n_clusters_processed": int(primary["cluster"].nunique()),
        "n_kinases": int(primary["kinase"].nunique()),
        "n_contrasts": int(primary["contrast"].nunique()),
        "n_tracks": int(primary["track"].nunique()),
        "n_rows_primary": int(len(primary)),
        "n_rows_rollup": int(len(rolled)),
        "confidence_breakdown": conf_counts,
        "deconv_sig_rows": int((primary["FDR"] < paths.DECON_FDR_THRESH).sum()),
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    with open(paths.SUMMARY_JSON, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Saved {paths.SUMMARY_JSON}")
    print(f"  Total elapsed: {summary['elapsed_seconds']}s")


if __name__ == "__main__":
    main()

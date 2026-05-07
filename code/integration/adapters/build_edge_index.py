"""Build the normalized kinase <-> backbone universe layer.

Streams per-pair kinase_routes.parquet across all pairs and emits the
``outputs/incytr/universes/{universe_id}/`` dimension tables (cells, genes,
contrasts, kinases, backbones, pathways, pair_dim) plus partitioned
``routes/{pair_id}.parquet`` files. Downstream queries reconstruct the wide
edge join via the DuckDB view declared in ``normalization.open_edges_view``.

The pre-cutover materialized ``kinase_backbone_edges.parquet`` writer is
preserved at the ``legacy-incytr-storage-cutover`` git tag.

Usage:
  python build_edge_index.py
  python build_edge_index.py --pair-filter 'Astrocyte__Endothelial'  # smoke
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd  # noqa: F401  # used transitively by helpers

ADAPTERS_DIR = os.path.dirname(os.path.abspath(__file__))
INTEGRATION_DIR = os.path.dirname(ADAPTERS_DIR)
REPO_ROOT = os.path.abspath(os.path.join(INTEGRATION_DIR, "..", ".."))
CODE_DIR = os.path.join(REPO_ROOT, "code")
for p in (INTEGRATION_DIR, CODE_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)
import config_integration as icfg  # noqa: E402
from config import SEA_AD_SUBCLASSES  # noqa: E402
from normalization import (  # noqa: E402
    build_universe_tables_from_legacy,
    resolve_paths,
    write_factorial_pathway_scores_for_pair,
    write_routes_from_legacy,
)

CONTRASTS = list(icfg.FACTORIAL_CONTRASTS.keys())


def discover_pair_dirs(pair_filter=None):
    import fnmatch
    entries = []
    for name in sorted(os.listdir(icfg.FACTORIAL_ALL_PAIRS_DIR)):
        if "__" not in name:
            continue
        d = os.path.join(icfg.FACTORIAL_ALL_PAIRS_DIR, name)
        if not os.path.isdir(d):
            continue
        if pair_filter and not fnmatch.fnmatch(name, pair_filter):
            continue
        entries.append((name, d))
    return entries


def unsanitize(name):
    lookup = {s.replace("/", "-").replace(" ", "_"): s for s in SEA_AD_SUBCLASSES}
    return lookup.get(name, name.replace("_", " "))


def split_pair(dir_name):
    sender_san, receiver_san = dir_name.split("__", 1)
    return unsanitize(sender_san), unsanitize(receiver_san)


def _scores_cols(path):
    return pd.read_csv(
        path,
        usecols=["Path", "EM", "Target", "Receptor"],
        dtype="string",
    )


def _collect_vocab(pairs_with_routes):
    """Pass 1: scan every pair, collect kinase set + unique backbone keys.
    Retains only small string sets; per-pair frames are released each iter."""
    kinase_set = set()
    bb_frames = []
    ct_set = set()
    t0 = time.monotonic()
    for i, (pair_name, d) in enumerate(pairs_with_routes, 1):
        sender, receiver = split_pair(pair_name)
        ct_set.add(sender)
        ct_set.add(receiver)

        routes_k = pd.read_parquet(
            os.path.join(d, "kinase_routes.parquet"), columns=["kinase"]
        )
        kinase_set.update(routes_k["kinase"].unique())
        del routes_k

        scores = _scores_cols(os.path.join(d, "kinase_support_scores.csv"))
        bb = scores[["Receptor", "EM", "Target"]].drop_duplicates().copy()
        bb["receiver"] = receiver
        bb_frames.append(bb[["receiver", "Receptor", "EM", "Target"]])

        if i % 50 == 0 or i == len(pairs_with_routes):
            print(f"  vocab [{i}/{len(pairs_with_routes)}] "
                  f"kinases={len(kinase_set)} bb_frames={len(bb_frames)}")

    backbones = (
        pd.concat(bb_frames, ignore_index=True)
        .drop_duplicates()
        .sort_values(["receiver", "Receptor", "EM", "Target"])
        .reset_index(drop=True)
    )
    backbones["backbone_id"] = np.arange(len(backbones), dtype=np.uint32)
    print(f"  vocab pass: {time.monotonic() - t0:.1f}s; "
          f"kinases={len(kinase_set)} backbones={len(backbones)} "
          f"celltypes={len(ct_set)}")
    return sorted(kinase_set), backbones, sorted(ct_set)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair-filter", metavar="PATTERN", default=None)
    args = parser.parse_args()

    t0 = time.monotonic()
    pair_dirs = discover_pair_dirs(args.pair_filter)
    print(f"Discovered {len(pair_dirs)} pair dirs")

    pairs_with_routes = []
    pairs_missing_routes = []
    for name, d in pair_dirs:
        if os.path.exists(os.path.join(d, "kinase_routes.parquet")):
            pairs_with_routes.append((name, d))
        else:
            pairs_missing_routes.append(name)

    if not pairs_with_routes:
        print("No pair directories contain kinase_routes.parquet. "
              "Run sidecar/kinase_pack/compute_kinase_support_factorial.py "
              "--emit-kinase-routes first (requires INCYTR_LAYER_KINASE_PACK=1).")
        sys.exit(1)

    print(f"  {len(pairs_with_routes)} with routes; "
          f"{len(pairs_missing_routes)} missing routes")

    kinases_sorted, backbones, celltypes_present = _collect_vocab(pairs_with_routes)

    npaths = resolve_paths()
    print(f"Writing normalized universe layer: {npaths.universe_dir}")
    universe_tables = build_universe_tables_from_legacy(
        pair_dirs=pairs_with_routes,
        backbones=backbones,
        kinases=kinases_sorted,
        celltypes=celltypes_present,
        contrasts=CONTRASTS,
        split_pair=split_pair,
        output_dir=npaths.universe_dir,
    )
    print("  normalized universe: "
          f"pairs={len(universe_tables['pair_dim'])} "
          f"backbones={len(universe_tables['backbones'])} "
          f"pathways={len(universe_tables['pathways'])}")
    n_route_rows = write_routes_from_legacy(pairs_with_routes, npaths.universe_dir)
    print(f"  normalized routes: {n_route_rows:,} rows")

    n_score_rows = 0
    for pair_name, pair_dir in pairs_with_routes:
        scores_csv = os.path.join(pair_dir, "kinase_support_scores.csv")
        if not os.path.exists(scores_csv):
            continue
        scores_df = pd.read_csv(scores_csv)
        write_factorial_pathway_scores_for_pair(
            scores_df,
            universe_dir=npaths.universe_dir,
            scoring_dir=npaths.scoring_dir,
            pair_name=pair_name,
            contrasts=CONTRASTS,
        )
        n_score_rows += len(scores_df) * len(CONTRASTS)
    print(f"  normalized scoring: {n_score_rows:,} rows "
          f"({len(pairs_with_routes)} pairs x {len(CONTRASTS)} contrasts)")
    print(f"\n=== done in {time.monotonic() - t0:.1f}s ===")


if __name__ == "__main__":
    main()

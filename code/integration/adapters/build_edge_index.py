"""Build the canonical kinase <-> backbone edge index.

Concatenates per-pair kinase_routes.parquet across all 460 pairs, assigns
stable integer ids (kinase_id, backbone_id, celltype_id, contrast_id), and
writes kinase_backbone_edges.parquet + edge_index_metadata.json.

Contract: see pipeline_notes/phase1_edge_schema.md.

Usage:
  python build_edge_index.py
  python build_edge_index.py --pair-filter 'Astrocyte__Endothelial'  # smoke
  python build_edge_index.py --skip-verify                           # skip assertions
"""

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np
import pandas as pd

ADAPTERS_DIR = os.path.dirname(os.path.abspath(__file__))
INTEGRATION_DIR = os.path.dirname(ADAPTERS_DIR)
REPO_ROOT = os.path.abspath(os.path.join(INTEGRATION_DIR, "..", ".."))
CODE_DIR = os.path.join(REPO_ROOT, "code")
for p in (INTEGRATION_DIR, CODE_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)
import config_integration as icfg  # noqa: E402
from config import SEA_AD_SUBCLASSES  # noqa: E402

CONTRASTS = list(icfg.FACTORIAL_CONTRASTS.keys())
OUT_DIR = os.path.join(icfg.FACTORIAL_ALL_PAIRS_DIR, "aggregation")
OUT_EDGES = os.path.join(OUT_DIR, "kinase_backbone_edges.parquet")
OUT_META = os.path.join(OUT_DIR, "edge_index_metadata.json")


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
    """Reverse sanitize_celltype_name: restore '/', ' ' in subclass strings."""
    lookup = {s.replace("/", "-").replace(" ", "_"): s for s in SEA_AD_SUBCLASSES}
    return lookup.get(name, name.replace("_", " "))


def split_pair(dir_name):
    sender_san, receiver_san = dir_name.split("__", 1)
    return unsanitize(sender_san), unsanitize(receiver_san)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair-filter", metavar="PATTERN", default=None)
    parser.add_argument("--skip-verify", action="store_true")
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
              "Run compute_kinase_support_factorial.py --emit-kinase-routes first.")
        sys.exit(1)

    print(f"  {len(pairs_with_routes)} with routes; "
          f"{len(pairs_missing_routes)} missing routes")

    # Load routes + per-pair validation data
    t_load = time.monotonic()
    kinases = set()
    backbone_key_set = set()
    verify_failures = []
    dfs = []
    for i, (pair_name, d) in enumerate(pairs_with_routes, 1):
        sender, receiver = split_pair(pair_name)
        routes = pd.read_parquet(os.path.join(d, "kinase_routes.parquet"))
        # Pull (EM, Target, Receptor) from kinase_support_scores.csv once via
        # left-join on Path (backbone key on disk is Path string).
        scores = pd.read_csv(
            os.path.join(d, "kinase_support_scores.csv"),
            usecols=["Path", "EM", "Target", "Receptor", "Ligand"],
        )
        routes = routes.merge(scores, on="Path", how="left", validate="many_to_one")
        routes["sender"] = sender
        routes["receiver"] = receiver
        dfs.append(routes)
        kinases.update(routes["kinase"].unique())
        for em, tg, rc in zip(routes["EM"], routes["Target"], routes["Receptor"]):
            backbone_key_set.add((receiver, rc, em, tg))

        if not args.skip_verify:
            vf = _verify_pair(routes, d, pair_name)
            if vf:
                verify_failures.append(vf)

        if i % 50 == 0 or i == len(pairs_with_routes):
            print(f"  loaded [{i}/{len(pairs_with_routes)}] "
                  f"{pair_name} rows={len(routes)}")

    print(f"  load+verify: {time.monotonic() - t_load:.1f}s")

    if verify_failures:
        print(f"\nVERIFY FAILED for {len(verify_failures)} pairs. "
              "First 3:")
        for f in verify_failures[:3]:
            print("  -", f)
        sys.exit(2)

    # Side tables
    kinases_sorted = sorted(kinases)
    kin_to_id = {k: i for i, k in enumerate(kinases_sorted)}

    backbones_sorted = sorted(backbone_key_set)
    bb_to_id = {k: i for i, k in enumerate(backbones_sorted)}

    ct_set = set()
    for name, _ in pairs_with_routes:
        s, r = split_pair(name)
        ct_set.add(s)
        ct_set.add(r)
    celltypes_present = sorted(ct_set)
    ct_to_id = {c: i for i, c in enumerate(celltypes_present)}

    cn_to_id = {c: i for i, c in enumerate(CONTRASTS)}

    # Concatenate + join ids
    t_join = time.monotonic()
    routes_all = pd.concat(dfs, ignore_index=True)
    del dfs
    routes_all["kinase_id"] = routes_all["kinase"].map(kin_to_id).astype("uint16")
    routes_all["contrast_id"] = routes_all["contrast"].map(cn_to_id).astype("uint8")
    routes_all["sender_id"] = routes_all["sender"].map(ct_to_id).astype("uint8")
    routes_all["receiver_id"] = routes_all["receiver"].map(ct_to_id).astype("uint8")
    bb_tuples = list(zip(routes_all["receiver"], routes_all["Receptor"],
                         routes_all["EM"], routes_all["Target"]))
    routes_all["backbone_id"] = np.fromiter(
        (bb_to_id[t] for t in bb_tuples), dtype=np.uint32, count=len(bb_tuples))
    routes_all["support_magnitude"] = (
        routes_all["support_contribution"].abs().astype("float32")
    )
    # concordance: sign(kinase_nes) * sign(TPDS). TPDS lookup deferred —
    # we need it from scores; compute from support_contribution sign for now.
    # For a first cut, concordance = nes_sign (TPDS-adjusted at viewer time).
    routes_all["concordance"] = routes_all["nes_sign"].astype("int8")
    routes_all["lambda_bin"] = np.uint8(0)
    print(f"  id assignment: {time.monotonic() - t_join:.1f}s")

    edges = routes_all[[
        "kinase_id", "backbone_id", "contrast_id",
        "sender_id", "receiver_id",
        "support_contribution", "support_magnitude",
        "concordance", "lambda_bin",
    ]]

    os.makedirs(OUT_DIR, exist_ok=True)
    t_write = time.monotonic()
    edges.to_parquet(OUT_EDGES, index=False, compression="zstd")
    print(f"  wrote {OUT_EDGES} ({len(edges):,} rows) in "
          f"{time.monotonic() - t_write:.1f}s")

    meta = {
        "schema_version": 1,
        "source": "pipeline Unit 1.3 build_edge_index.py",
        "n_edges": int(len(edges)),
        "n_pairs_with_routes": len(pairs_with_routes),
        "n_pairs_missing_routes": len(pairs_missing_routes),
        "pairs_missing_routes": pairs_missing_routes[:20],
        "kinases": kinases_sorted,
        "backbones_n": len(backbones_sorted),
        "celltypes": celltypes_present,
        "contrasts": CONTRASTS,
        "backbone_key_cols": ["receiver", "Receptor", "EM", "Target"],
        "verified_sum_contribution": not args.skip_verify,
        "total_seconds": round(time.monotonic() - t0, 1),
    }
    with open(OUT_META, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  wrote {OUT_META}")

    print(f"\n=== done in {time.monotonic() - t0:.1f}s ===")


def _verify_pair(routes, pair_dir, pair_name, tol=1e-4):
    """Per-pair verify: sum(support_contribution) over kinases per (Path,
    contrast) matches kinase_support_score_sum_{contrast} in the
    pair's kinase_support_scores.csv, and count matches n_distinct_kinases.
    """
    scores = pd.read_csv(os.path.join(pair_dir, "kinase_support_scores.csv"))

    agg = (
        routes.groupby(["Path", "contrast"], sort=False)
        .agg(route_sum=("support_contribution", "sum"),
             route_n=("kinase", "nunique"))
        .reset_index()
    )

    for contrast in CONTRASTS:
        rc = agg[agg["contrast"] == contrast][["Path", "route_sum", "route_n"]]
        sc_col = f"kinase_support_score_sum_{contrast}"
        nk_col = f"n_distinct_kinases_{contrast}"
        merged = rc.merge(scores[["Path", sc_col, nk_col]], on="Path",
                          how="left")
        diff = (merged["route_sum"].astype("float64")
                - merged[sc_col].astype("float64"))
        max_diff = diff.abs().max() if len(diff) else 0.0
        if max_diff > tol:
            return (f"{pair_name}/{contrast}: "
                    f"max |route_sum - score_sum| = {max_diff:.6g} (tol {tol})")
        count_mismatch = (merged["route_n"] != merged[nk_col]).sum()
        if count_mismatch:
            return (f"{pair_name}/{contrast}: "
                    f"{count_mismatch} paths with route_n != n_distinct_kinases")
    return None


if __name__ == "__main__":
    main()

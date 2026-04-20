"""Build the canonical kinase <-> backbone edge index.

Streams per-pair kinase_routes.parquet across all 460 pairs in two passes:

  Pass 1 (vocab): collect the set of kinases and backbone (receiver, Receptor,
          EM, Target) keys. No route data retained.
  Pass 2 (emit):  for each pair, load routes, assign integer ids via vector
          merges/maps, append a RecordBatch to a streaming ParquetWriter.

This avoids concatenating ~48 GB of per-pair frames (460 pairs x ~105 MB each)
and the Python-tuple materialization that caused OOMs in an earlier one-shot
implementation.

Contract: see pipeline_notes/phase1_edge_schema.md.

Usage:
  python build_edge_index.py
  python build_edge_index.py --pair-filter 'Astrocyte__Endothelial'  # smoke
  python build_edge_index.py --skip-verify                           # skip assertions
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

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

EDGE_SCHEMA = pa.schema([
    ("kinase_id", pa.uint16()),
    ("backbone_id", pa.uint32()),
    ("contrast_id", pa.uint8()),
    ("sender_id", pa.uint8()),
    ("receiver_id", pa.uint8()),
    ("support_contribution", pa.float32()),
    ("support_magnitude", pa.float32()),
    ("concordance", pa.int8()),
    ("lambda_bin", pa.uint8()),
])


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
            os.path.join(d, "kinase_routes.parquet"), columns=["kinase", "Path"]
        )
        kinase_set.update(routes_k["kinase"].unique())
        used_paths = routes_k["Path"].unique()
        del routes_k

        scores = _scores_cols(os.path.join(d, "kinase_support_scores.csv"))
        scores = scores[scores["Path"].isin(used_paths)]
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


def _emit_pair(pair_name, d, kin_to_id, backbones, ct_to_id, cn_to_id,
               writer, verify):
    """Load one pair, build ids, append a RecordBatch. Return verify failure
    string or None."""
    sender, receiver = split_pair(pair_name)

    routes = pd.read_parquet(os.path.join(d, "kinase_routes.parquet"))
    scores = _scores_cols(os.path.join(d, "kinase_support_scores.csv"))
    routes = routes.merge(scores, on="Path", how="left", validate="many_to_one")

    if verify:
        vf = _verify_pair(routes, d, pair_name)
        if vf:
            return vf

    # Assign ids vectorized. backbones is shared, so filter to this receiver
    # before merging to keep the join small.
    bb_r = backbones[backbones["receiver"] == receiver][
        ["Receptor", "EM", "Target", "backbone_id"]
    ]
    routes = routes.merge(
        bb_r, on=["Receptor", "EM", "Target"], how="left", validate="many_to_one"
    )
    if routes["backbone_id"].isna().any():
        return f"{pair_name}: unresolved backbone_id after merge"

    kinase_id = routes["kinase"].map(kin_to_id).astype("uint16")
    contrast_id = routes["contrast"].map(cn_to_id).astype("uint8")
    sender_id = np.full(len(routes), ct_to_id[sender], dtype="uint8")
    receiver_id = np.full(len(routes), ct_to_id[receiver], dtype="uint8")
    backbone_id = routes["backbone_id"].astype("uint32")
    support = routes["support_contribution"].astype("float32")
    magnitude = support.abs()
    concordance = routes["nes_sign"].astype("int8")
    lambda_bin = np.zeros(len(routes), dtype="uint8")

    batch = pa.RecordBatch.from_arrays(
        [
            pa.array(kinase_id.values, type=pa.uint16()),
            pa.array(backbone_id.values, type=pa.uint32()),
            pa.array(contrast_id.values, type=pa.uint8()),
            pa.array(sender_id, type=pa.uint8()),
            pa.array(receiver_id, type=pa.uint8()),
            pa.array(support.values, type=pa.float32()),
            pa.array(magnitude.values, type=pa.float32()),
            pa.array(concordance.values, type=pa.int8()),
            pa.array(lambda_bin, type=pa.uint8()),
        ],
        schema=EDGE_SCHEMA,
    )
    writer.write_batch(batch)
    return None


def _verify_pair(routes, pair_dir, pair_name, tol=1e-4):
    verify_cols = ["Path"] + [f"kinase_support_score_sum_{c}" for c in CONTRASTS] \
        + [f"n_distinct_kinases_{c}" for c in CONTRASTS]
    scores = pd.read_csv(
        os.path.join(pair_dir, "kinase_support_scores.csv"), usecols=verify_cols
    )
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

    kinases_sorted, backbones, celltypes_present = _collect_vocab(pairs_with_routes)
    kin_to_id = {k: i for i, k in enumerate(kinases_sorted)}
    ct_to_id = {c: i for i, c in enumerate(celltypes_present)}
    cn_to_id = {c: i for i, c in enumerate(CONTRASTS)}

    os.makedirs(OUT_DIR, exist_ok=True)
    t_emit = time.monotonic()
    verify_failures = []
    with pq.ParquetWriter(OUT_EDGES, EDGE_SCHEMA, compression="zstd") as writer:
        for i, (pair_name, d) in enumerate(pairs_with_routes, 1):
            vf = _emit_pair(pair_name, d, kin_to_id, backbones, ct_to_id,
                            cn_to_id, writer, verify=not args.skip_verify)
            if vf:
                verify_failures.append(vf)
                if len(verify_failures) >= 3:
                    break
                continue
            if i % 25 == 0 or i == len(pairs_with_routes):
                print(f"  emit [{i}/{len(pairs_with_routes)}] {pair_name}")

    if verify_failures:
        print(f"\nVERIFY FAILED for {len(verify_failures)} pairs:")
        for f in verify_failures:
            print("  -", f)
        sys.exit(2)

    # Read footer for authoritative row count.
    pf = pq.ParquetFile(OUT_EDGES)
    n_edges = pf.metadata.num_rows
    print(f"  emit pass: {time.monotonic() - t_emit:.1f}s; "
          f"edges={n_edges:,}")

    meta = {
        "schema_version": 1,
        "source": "pipeline Unit 1.3 build_edge_index.py",
        "n_edges": int(n_edges),
        "n_pairs_with_routes": len(pairs_with_routes),
        "n_pairs_missing_routes": len(pairs_missing_routes),
        "pairs_missing_routes": pairs_missing_routes[:20],
        "kinases": kinases_sorted,
        "backbones_n": int(len(backbones)),
        "celltypes": celltypes_present,
        "contrasts": CONTRASTS,
        "backbone_key_cols": ["receiver", "Receptor", "EM", "Target"],
        "verified_sum_contribution": not args.skip_verify,
        "total_seconds": round(time.monotonic() - t0, 1),
    }
    with open(OUT_META, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  wrote {OUT_EDGES}")
    print(f"  wrote {OUT_META}")
    print(f"\n=== done in {time.monotonic() - t0:.1f}s ===")


if __name__ == "__main__":
    main()

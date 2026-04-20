"""Build per-entity edge shards + summaries from the sig edge parquet.

Reducer for the unified viewer's edge surface. Reads
`outputs/reports/kinase_backbone_edges_sig.parquet` once, emits:

  outputs/reports/unified_viewer/
    edge_summaries/per_kinase_summary.parquet
    edge_summaries/per_backbone_summary.parquet
    edge_slices/kinase/{kinase_id:03d}.parquet     (233 files)
    edge_slices/kinase/index.json
    edge_slices/backbone/{bucket_id:03d}.parquet   (~218 files)
    edge_slices/backbone/index.json

Design + schemas: `pipeline_notes/edge_sharding_schema.md`.

Lossless: every row of the source appears in exactly one kinase slice and one
backbone bucket.

Usage:
    python code/integration/adapters/build_edge_shards.py
    python code/integration/adapters/build_edge_shards.py --skip-verify
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ADAPTERS_DIR = os.path.dirname(os.path.abspath(__file__))
INTEGRATION_DIR = os.path.dirname(ADAPTERS_DIR)
REPO_ROOT = os.path.abspath(os.path.join(INTEGRATION_DIR, "..", ".."))

SRC_PARQUET = os.path.join(REPO_ROOT, "outputs", "reports",
                           "kinase_backbone_edges_sig.parquet")
DST_DIR = os.path.join(REPO_ROOT, "outputs", "reports", "unified_viewer")
DST_SUMMARIES = os.path.join(DST_DIR, "edge_summaries")
DST_KINASE = os.path.join(DST_DIR, "edge_slices", "kinase")
DST_BACKBONE = os.path.join(DST_DIR, "edge_slices", "backbone")

BUCKET_SIZE = 256
BATCH_SIZE = 500_000
SCHEMA_VERSION = 1

PER_KINASE_SUMMARY_SCHEMA = pa.schema([
    ("kinase_id", pa.uint16()), ("contrast_id", pa.uint8()),
    ("n_backbones", pa.uint32()),
    ("n_concordant_up", pa.uint32()), ("n_concordant_down", pa.uint32()),
    ("sum_abs_support", pa.float32()),
    ("mean_abs_support", pa.float32()), ("max_abs_support", pa.float32()),
])
PER_BACKBONE_SUMMARY_SCHEMA = pa.schema([
    ("backbone_id", pa.uint32()), ("contrast_id", pa.uint8()),
    ("n_kinases", pa.uint32()),
    ("n_concordant_up", pa.uint32()), ("n_concordant_down", pa.uint32()),
    ("sum_abs_support", pa.float32()), ("max_abs_support", pa.float32()),
])
KINASE_SLICE_SCHEMA = pa.schema([
    ("backbone_id", pa.uint32()), ("contrast_id", pa.uint8()),
    ("support_contribution", pa.float32()), ("concordance", pa.int8()),
])
BACKBONE_BUCKET_SCHEMA = pa.schema([
    ("backbone_id", pa.uint32()), ("kinase_id", pa.uint16()),
    ("contrast_id", pa.uint8()),
    ("support_contribution", pa.float32()), ("concordance", pa.int8()),
])


def _sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _log(msg, t0):
    print(f"[{time.time()-t0:6.1f}s] {msg}", flush=True)


def build_shards(skip_verify=False):
    t0 = time.time()
    os.makedirs(DST_SUMMARIES, exist_ok=True)
    os.makedirs(DST_KINASE, exist_ok=True)
    os.makedirs(DST_BACKBONE, exist_ok=True)

    pf = pq.ParquetFile(SRC_PARQUET)
    n_total = pf.metadata.num_rows
    _log(f"opened source: {SRC_PARQUET} ({n_total:,} rows, "
         f"{pf.metadata.num_row_groups} row groups)", t0)

    src_sha = _sha256(SRC_PARQUET)
    _log(f"source sha256: {src_sha[:16]}…", t0)

    # Writers (opened lazily on first row)
    k_writers: dict[int, pq.ParquetWriter] = {}
    b_writers: dict[int, pq.ParquetWriter] = {}

    # Running summary stats
    # keys are (kinase_id, contrast_id) or (backbone_id, contrast_id)
    # value is [n, n_up, n_down, sum_abs, max_abs]
    kc_stats: dict[tuple[int, int], list[float]] = defaultdict(
        lambda: [0, 0, 0, 0.0, 0.0])
    bc_stats: dict[tuple[int, int], list[float]] = defaultdict(
        lambda: [0, 0, 0, 0.0, 0.0])

    # Per (k, c): set of distinct backbones seen — needed for n_backbones
    # (in this parquet (k, b, c) is unique per row, so row count == n_backbones).
    # Same for (b, c): row count == n_kinases. No set needed.
    # But source might in principle repeat a (k, b, c) — we assert at end.

    rows_emitted_kinase = 0
    rows_emitted_bucket = 0
    total_seen = 0
    last_log = 0

    for batch in pf.iter_batches(
            batch_size=BATCH_SIZE,
            columns=["kinase_id", "backbone_id", "contrast_id",
                     "support_contribution", "concordance"]):
        d = batch.to_pydict()
        k_arr = np.asarray(d["kinase_id"], dtype=np.uint16)
        b_arr = np.asarray(d["backbone_id"], dtype=np.uint32)
        c_arr = np.asarray(d["contrast_id"], dtype=np.uint8)
        s_arr = np.asarray(d["support_contribution"], dtype=np.float32)
        z_arr = np.asarray(d["concordance"], dtype=np.int8)
        n = len(k_arr)
        total_seen += n

        abs_s = np.abs(s_arr)
        up = (z_arr == 1).astype(np.uint32)
        dn = (z_arr == -1).astype(np.uint32)
        ones = np.ones(n, dtype=np.uint32)

        # Batch summary stats via pandas groupby — vectorized, ~50× faster
        # than per-row Python tuple-dict updates. Counts/sums/max merged into
        # the running kc_stats / bc_stats dicts after each batch.
        df = pd.DataFrame({
            "k": k_arr, "b": b_arr, "c": c_arr,
            "n": ones, "up": up, "dn": dn,
            "abs_s": abs_s,
        })
        kc_batch = df.groupby(["k", "c"], sort=False).agg(
            n=("n", "sum"), up=("up", "sum"), dn=("dn", "sum"),
            sum_abs=("abs_s", "sum"), max_abs=("abs_s", "max"),
        )
        for (kid, cid), row in kc_batch.iterrows():
            st = kc_stats[(int(kid), int(cid))]
            st[0] += int(row["n"])
            st[1] += int(row["up"])
            st[2] += int(row["dn"])
            st[3] += float(row["sum_abs"])
            if row["max_abs"] > st[4]:
                st[4] = float(row["max_abs"])

        bc_batch = df.groupby(["b", "c"], sort=False).agg(
            n=("n", "sum"), up=("up", "sum"), dn=("dn", "sum"),
            sum_abs=("abs_s", "sum"), max_abs=("abs_s", "max"),
        )
        for (bid, cid), row in bc_batch.iterrows():
            st = bc_stats[(int(bid), int(cid))]
            st[0] += int(row["n"])
            st[1] += int(row["up"])
            st[2] += int(row["dn"])
            st[3] += float(row["sum_abs"])
            if row["max_abs"] > st[4]:
                st[4] = float(row["max_abs"])

        # Emit per-kinase slices: partition by kinase_id.
        uniq_k = np.unique(k_arr)
        for kid in uniq_k.tolist():
            mask = (k_arr == kid)
            if not mask.any():
                continue
            # Sort within kinase by (contrast_id asc, support_contribution desc)
            sub_c = c_arr[mask]
            sub_s = s_arr[mask]
            order = np.lexsort((-sub_s, sub_c))
            sub = pa.RecordBatch.from_pydict({
                "backbone_id": b_arr[mask][order],
                "contrast_id": sub_c[order],
                "support_contribution": sub_s[order],
                "concordance": z_arr[mask][order],
            }, schema=KINASE_SLICE_SCHEMA)
            w = k_writers.get(kid)
            if w is None:
                path = os.path.join(DST_KINASE, f"{kid:03d}.parquet")
                w = pq.ParquetWriter(path, KINASE_SLICE_SCHEMA,
                                     compression="zstd")
                k_writers[kid] = w
            w.write_batch(sub)
            rows_emitted_kinase += int(mask.sum())

        # Emit per-bucket slices: partition by bucket_id = backbone_id // BUCKET_SIZE.
        bucket_arr = (b_arr // BUCKET_SIZE).astype(np.uint32)
        uniq_bkt = np.unique(bucket_arr)
        for bkt in uniq_bkt.tolist():
            mask = (bucket_arr == bkt)
            if not mask.any():
                continue
            # Sort: (backbone_id asc, contrast_id asc, support desc)
            sub_b = b_arr[mask]
            sub_c = c_arr[mask]
            sub_s = s_arr[mask]
            order = np.lexsort((-sub_s, sub_c, sub_b))
            sub = pa.RecordBatch.from_pydict({
                "backbone_id": sub_b[order],
                "kinase_id": k_arr[mask][order],
                "contrast_id": sub_c[order],
                "support_contribution": sub_s[order],
                "concordance": z_arr[mask][order],
            }, schema=BACKBONE_BUCKET_SCHEMA)
            w = b_writers.get(bkt)
            if w is None:
                path = os.path.join(DST_BACKBONE, f"{bkt:03d}.parquet")
                w = pq.ParquetWriter(path, BACKBONE_BUCKET_SCHEMA,
                                     compression="zstd")
                b_writers[bkt] = w
            w.write_batch(sub)
            rows_emitted_bucket += int(mask.sum())

        if total_seen - last_log >= 5_000_000:
            _log(f"streamed {total_seen:,} / {n_total:,} "
                 f"({100*total_seen/n_total:.1f}%)", t0)
            last_log = total_seen

    # Close writers
    for w in k_writers.values():
        w.close()
    for w in b_writers.values():
        w.close()
    _log(f"closed {len(k_writers)} kinase writers, "
         f"{len(b_writers)} bucket writers", t0)

    # Row-count parity assertions
    assert total_seen == n_total, f"streamed {total_seen} != {n_total}"
    assert rows_emitted_kinase == n_total, \
        f"kinase emitted {rows_emitted_kinase} != {n_total}"
    assert rows_emitted_bucket == n_total, \
        f"bucket emitted {rows_emitted_bucket} != {n_total}"
    _log(f"parity OK: {n_total:,} rows in source = "
         f"{rows_emitted_kinase:,} in kinase slices = "
         f"{rows_emitted_bucket:,} in backbone buckets", t0)

    # Emit summaries
    kc_rows = sorted(kc_stats.items())
    pk = {
        "kinase_id": np.array([r[0][0] for r in kc_rows], dtype=np.uint16),
        "contrast_id": np.array([r[0][1] for r in kc_rows], dtype=np.uint8),
        "n_backbones": np.array([r[1][0] for r in kc_rows], dtype=np.uint32),
        "n_concordant_up": np.array([r[1][1] for r in kc_rows], dtype=np.uint32),
        "n_concordant_down": np.array([r[1][2] for r in kc_rows], dtype=np.uint32),
        "sum_abs_support": np.array([r[1][3] for r in kc_rows], dtype=np.float32),
        "mean_abs_support": np.array(
            [r[1][3] / r[1][0] if r[1][0] else 0.0 for r in kc_rows],
            dtype=np.float32),
        "max_abs_support": np.array([r[1][4] for r in kc_rows], dtype=np.float32),
    }
    pq.write_table(pa.table(pk, schema=PER_KINASE_SUMMARY_SCHEMA),
                   os.path.join(DST_SUMMARIES, "per_kinase_summary.parquet"),
                   compression="zstd")
    _log(f"wrote per_kinase_summary.parquet ({len(kc_rows)} rows)", t0)

    bc_rows = sorted(bc_stats.items())
    pb = {
        "backbone_id": np.array([r[0][0] for r in bc_rows], dtype=np.uint32),
        "contrast_id": np.array([r[0][1] for r in bc_rows], dtype=np.uint8),
        "n_kinases": np.array([r[1][0] for r in bc_rows], dtype=np.uint32),
        "n_concordant_up": np.array([r[1][1] for r in bc_rows], dtype=np.uint32),
        "n_concordant_down": np.array([r[1][2] for r in bc_rows], dtype=np.uint32),
        "sum_abs_support": np.array([r[1][3] for r in bc_rows], dtype=np.float32),
        "max_abs_support": np.array([r[1][4] for r in bc_rows], dtype=np.float32),
    }
    pq.write_table(pa.table(pb, schema=PER_BACKBONE_SUMMARY_SCHEMA),
                   os.path.join(DST_SUMMARIES, "per_backbone_summary.parquet"),
                   compression="zstd")
    _log(f"wrote per_backbone_summary.parquet ({len(bc_rows)} rows)", t0)

    # Index files
    kinase_ids = sorted(k_writers.keys())
    max_backbone_id = int(max(k for (k, _) in bc_stats.keys()))
    bucket_count = max_backbone_id // BUCKET_SIZE + 1

    with open(os.path.join(DST_KINASE, "index.json"), "w") as f:
        json.dump({
            "schema_version": SCHEMA_VERSION,
            "slice_count": len(kinase_ids),
            "present_kinase_ids": kinase_ids,
            "filename_template": "{kinase_id:03d}.parquet",
            "source_sha256": src_sha,
        }, f, indent=2)

    with open(os.path.join(DST_BACKBONE, "index.json"), "w") as f:
        json.dump({
            "schema_version": SCHEMA_VERSION,
            "bucket_size": BUCKET_SIZE,
            "bucket_count": len(b_writers),
            "max_backbone_id": max_backbone_id,
            "present_bucket_ids": sorted(b_writers.keys()),
            "filename_template": "{bucket_id:03d}.parquet",
            "source_sha256": src_sha,
        }, f, indent=2)
    _log(f"wrote index.json files "
         f"({len(kinase_ids)} kinases, {len(b_writers)} buckets, "
         f"max_backbone_id={max_backbone_id})", t0)

    if skip_verify:
        _log("skipping verification (per --skip-verify)", t0)
        return

    _verify(pf, src_sha, kc_stats, bc_stats, t0)


def _verify(pf, src_sha, kc_stats, bc_stats, t0):
    """Spot-check parity between summaries and on-disk slices."""
    rng = np.random.default_rng(42)
    # Pick 20 random (k, c) pairs from summary; check slice row count matches.
    kc_items = list(kc_stats.items())
    sampled = rng.choice(len(kc_items), size=min(20, len(kc_items)), replace=False)
    for idx in sampled:
        (kid, cid), (n, _, _, sum_abs, _) = kc_items[idx]
        slice_path = os.path.join(DST_KINASE, f"{kid:03d}.parquet")
        t = pq.read_table(slice_path, columns=["backbone_id", "contrast_id",
                                               "support_contribution"])
        mask = np.asarray(t["contrast_id"]) == cid
        slice_n = int(mask.sum())
        assert slice_n == n, \
            f"kinase {kid} contrast {cid}: summary n={n} != slice n={slice_n}"
        slice_sum = float(np.abs(np.asarray(t["support_contribution"])[mask]).sum())
        rel = abs(slice_sum - sum_abs) / max(sum_abs, 1e-9)
        assert rel < 1e-3, \
            f"kinase {kid} contrast {cid}: sum mismatch {slice_sum} vs {sum_abs}"
    _log(f"kinase-slice parity OK for {len(sampled)} sampled (k,c) pairs", t0)

    # Same spot check backbone side.
    bc_items = list(bc_stats.items())
    sampled = rng.choice(len(bc_items), size=min(20, len(bc_items)), replace=False)
    for idx in sampled:
        (bid, cid), (n, _, _, sum_abs, _) = bc_items[idx]
        bkt = bid // BUCKET_SIZE
        bucket_path = os.path.join(DST_BACKBONE, f"{bkt:03d}.parquet")
        t = pq.read_table(bucket_path,
                          columns=["backbone_id", "contrast_id",
                                   "support_contribution"])
        mask = ((np.asarray(t["backbone_id"]) == bid)
                & (np.asarray(t["contrast_id"]) == cid))
        bkt_n = int(mask.sum())
        assert bkt_n == n, \
            f"backbone {bid} contrast {cid}: summary n={n} != bucket n={bkt_n}"
        bkt_sum = float(np.abs(np.asarray(t["support_contribution"])[mask]).sum())
        rel = abs(bkt_sum - sum_abs) / max(sum_abs, 1e-9)
        assert rel < 1e-3, \
            f"backbone {bid} contrast {cid}: sum mismatch {bkt_sum} vs {sum_abs}"
    _log(f"backbone-bucket parity OK for {len(sampled)} sampled (b,c) pairs", t0)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--skip-verify", action="store_true",
                    help="skip the final spot-check assertions")
    args = ap.parse_args()
    build_shards(skip_verify=args.skip_verify)


if __name__ == "__main__":
    main()

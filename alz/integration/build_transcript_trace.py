#!/usr/bin/env python3
"""Build per-cluster transcript pseudobulk shards for the Incytr Pathways
"Measurement Trace" panel.

Substrate is the long-form parquet at
``outputs/reports/decomposition/levy_t5/transcript_per_cluster.parquet``,
emitted by ``bench/incytr_pair_levy_t5/emit_expr_bygroup.R``:
per-(cluster, Group) mean of ``Data.input@assays$originalexp@data`` — bit-for-
bit the matrix Incytr's ``Cal_scFC`` consumes via
``Expr_bygroup(..., mean_method = "mean")``. The panel therefore matches the
FC tab's ``*_sclog2FC`` values: any user can recover
``log2((WT_arm + 1e-5) / (disease_arm + 1e-5))`` by hand from the shard,
modulo the sign flip applied in ``pair_to_receiver_cache.py`` so the viewer's
"positive = up in disease" tooltip stays correct.

Pathway-side cluster discovery reads the existing
``edge_slices/incytr_pathways/index.json`` (unsanitized sender/receiver names
already there). Each pathway cluster must be present in the parquet's
``cluster`` column or the build hard-fails.

Filename slugging uses ``sanitize_celltype`` from
``alz.integration.pair_to_receiver_cache`` — imported, not re-implemented.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))  # alz/

import config  # noqa: E402

from integration.pair_to_receiver_cache import _sanitize_celltype  # noqa: E402
from viewer.paths import (  # noqa: E402
    EDGE_SLICES_INCYTR_PATHWAYS_DIR,
    TRANSCRIPT_TRACE_DIR,
    TRANSCRIPT_TRACE_INDEX,
    TRANSCRIPT_TRACE_PSEUDOBULK,
    TRANSCRIPT_TRACE_SAMPLEKEY,
    TRANSCRIPT_TRACE_SCHEMA_VERSION,
    UNIFIED_VIEWER_DIR,
)

# Pair-mode sample-key column → (sex, timepoint, genotype). SCRNA_ID looks
# like "ma_4mo_AppP" / "fe_2mo_WTyp". Sex prefix `ma` → M, `fe` → F.
_SEX_MAP = {"ma": "M", "fe": "F"}


def _load_pathway_clusters(index_path: str) -> set[str]:
    """Read the existing incytr_pathways index.json and return the union of
    sender/receiver unsanitized strings.
    """
    if not os.path.exists(index_path):
        raise FileNotFoundError(
            f"incytr_pathways index missing at {index_path}; build the "
            f"pathway shards before the transcript-trace step."
        )
    with open(index_path) as f:
        idx = json.load(f)
    present = idx.get("present") or []
    clusters: set[str] = set()
    for pair in present:
        if len(pair) >= 2:
            clusters.add(pair[0])
            clusters.add(pair[1])
    return clusters


def _decode_group(group: str) -> tuple[str, str, str]:
    """`ma_4mo_AppP` → (M, 4mo, AppP)."""
    parts = group.split("_")
    if len(parts) != 3:
        raise ValueError(f"unrecognized group string: {group!r}")
    sex_code, tp, geno = parts
    sex = _SEX_MAP.get(sex_code)
    if sex is None:
        raise ValueError(f"unrecognized sex prefix in group {group!r}")
    return sex, tp, geno


def _load_samplekey(path: str) -> dict[str, dict]:
    """Return {SCRNA_ID -> {ms_id, sex, timepoint, genotype}}."""
    sk = pd.read_csv(path)
    needed = {"MS_ID", "SCRNA_ID", "Group"}
    if not needed.issubset(sk.columns):
        raise ValueError(
            f"sample-key {path} missing columns {needed - set(sk.columns)}"
        )
    out: dict[str, dict] = {}
    for _, row in sk.iterrows():
        scrna = str(row["SCRNA_ID"])
        sex, tp, geno = _decode_group(scrna)
        out[scrna] = {
            "ms_id": str(row["MS_ID"]),
            "sex": sex,
            "timepoint": tp,
            "genotype": geno,
        }
    return out


def build(force: bool = False) -> dict:
    """Build transcript_trace shards. Returns the index dict (also written
    to TRANSCRIPT_TRACE_INDEX).
    """
    if not force and os.path.exists(TRANSCRIPT_TRACE_INDEX):
        with open(TRANSCRIPT_TRACE_INDEX) as f:
            existing = json.load(f)
        if existing.get("trace_schema_version") == TRANSCRIPT_TRACE_SCHEMA_VERSION:
            return existing

    if os.path.exists(TRANSCRIPT_TRACE_DIR):
        shutil.rmtree(TRANSCRIPT_TRACE_DIR)
    os.makedirs(TRANSCRIPT_TRACE_DIR, exist_ok=True)

    t0 = time.time()
    if not os.path.exists(TRANSCRIPT_TRACE_PSEUDOBULK):
        raise FileNotFoundError(
            f"transcript-trace substrate missing: {TRANSCRIPT_TRACE_PSEUDOBULK}. "
            f"Run bench/incytr_pair_levy_t5/emit_expr_bygroup.R (step E3 in "
            f"alz/runners/main/run_pair_mode_pipeline.sh) from any directory "
            f"before rebuilding the viewer."
        )

    print(f"  transcript_trace: reading {TRANSCRIPT_TRACE_PSEUDOBULK}", flush=True)
    df = pd.read_parquet(TRANSCRIPT_TRACE_PSEUDOBULK)
    needed = {"cluster", "group", "gene", "value"}
    if not needed.issubset(df.columns):
        raise ValueError(
            f"expr_bygroup parquet missing required columns "
            f"{needed - set(df.columns)} (found {list(df.columns)})"
        )
    df["cluster"] = df["cluster"].astype(str)
    df["group"] = df["group"].astype(str)
    df["gene"] = df["gene"].astype(str)
    df["value"] = df["value"].astype("float32")

    unique_clusters = sorted(df["cluster"].unique().tolist())
    unique_groups = sorted(df["group"].unique().tolist())
    print(f"  transcript_trace: substrate has {len(unique_clusters)} clusters × "
          f"{len(unique_groups)} groups × {df['gene'].nunique()} genes",
          flush=True)

    # Sample-key validation.
    samplekey = _load_samplekey(TRANSCRIPT_TRACE_SAMPLEKEY)
    sk_groups = set(samplekey.keys())
    sub_groups = set(unique_groups)
    missing_in_sk = sub_groups - sk_groups
    if missing_in_sk:
        raise ValueError(
            f"transcript_trace: substrate Group values {sorted(missing_in_sk)[:5]} "
            f"absent from sample-key SCRNA_ID column at "
            f"{TRANSCRIPT_TRACE_SAMPLEKEY}"
        )

    # Pathway-side cluster discovery + coverage check.
    incytr_idx = os.path.join(EDGE_SLICES_INCYTR_PATHWAYS_DIR, "index.json")
    pathway_clusters = _load_pathway_clusters(incytr_idx)
    cluster_set = set(unique_clusters)
    missing = pathway_clusters - cluster_set
    if missing:
        raise ValueError(
            f"transcript_trace: pathway output references cluster(s) "
            f"{sorted(missing)} that are absent from the pseudobulk at "
            f"{TRANSCRIPT_TRACE_PSEUDOBULK}. Substrate drift — regenerate "
            f"expr_bygroup.parquet with the active cluster vocabulary."
        )
    print(f"  transcript_trace: {len(pathway_clusters)} pathway clusters "
          f"(all present in pseudobulk)", flush=True)

    # Per-cluster shard write. The long-form parquet already groups by cluster;
    # we just project the per-row metadata (sex/timepoint/genotype) and emit.
    shards_written: dict[str, str] = {}
    for cluster, sub in df.groupby("cluster", sort=True):
        if cluster not in pathway_clusters:
            continue
        sub = sub.copy()
        meta = sub["group"].map(samplekey)
        sub["sex"] = [m["sex"] for m in meta]
        sub["timepoint"] = [m["timepoint"] for m in meta]
        sub["genotype"] = [m["genotype"] for m in meta]
        out = sub[["gene", "group", "sex", "timepoint", "genotype", "value"]]
        slug = _sanitize_celltype(cluster)
        out_path = os.path.join(TRANSCRIPT_TRACE_DIR, f"{slug}.parquet")
        out.to_parquet(out_path, index=False, compression="zstd")
        shards_written[cluster] = os.path.relpath(out_path, UNIFIED_VIEWER_DIR)

    # Index.
    index = {
        "trace_schema_version": TRANSCRIPT_TRACE_SCHEMA_VERSION,
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "label": "Transcript pseudobulk trace",
        "source_pseudobulk": os.path.relpath(
            TRANSCRIPT_TRACE_PSEUDOBULK, config.REPO_ROOT
        ),
        "source_samplekey": os.path.relpath(
            TRANSCRIPT_TRACE_SAMPLEKEY, config.REPO_ROOT
        ),
        "substrate_note": (
            "Per-(cluster, Group) mean of Data.input@assays$originalexp@data "
            "(LogNormalize log1p-CP10K) — the same matrix Incytr's Cal_scFC "
            "consumes via Expr_bygroup(mean_method='mean')."
        ),
        "sanitize_rule": "replace('/', '-'); replace(' ', '_')",
        "filename_template": "{cluster}.parquet",
        "relative_path": os.path.relpath(TRANSCRIPT_TRACE_DIR, UNIFIED_VIEWER_DIR),
        "clusters": sorted(shards_written.keys()),
        "shard_files": shards_written,
        "groups": sorted(unique_groups),
        "n_libraries_per_arm": 1,
        "note": "Transcript pseudobulk · 1 library per arm · males-only by default",
    }
    with open(TRANSCRIPT_TRACE_INDEX, "w") as f:
        json.dump(index, f)
    print(f"  transcript_trace: wrote {len(shards_written)} cluster shards in "
          f"{time.time() - t0:.1f}s", flush=True)
    return index


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true",
                    help="Rebuild even if index is current.")
    args = ap.parse_args()
    build(force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

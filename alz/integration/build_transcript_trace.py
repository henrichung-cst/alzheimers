#!/usr/bin/env python3
"""Build per-cluster transcript pseudobulk shards for the Incytr Pathways
"Measurement Trace" panel.

Substrate is the wide aggexp.csv produced by aggregate_expression.R:
``do.call(rbind, datalist)`` over 24 ``AggregateExpression`` frames (one per
sex × timepoint × genotype group). Each frame contributes the per-cluster row
for that Group; R deduplicates colliding row names by appending integer
suffixes with no separator (``Astrocytes`` → ``Astrocytes1`` → ``Astrocytes2``
…).  Some cluster names already end in digits (``cluster-27``,
``Excitatory-Pyramidal-Satb2-Cux2``, ``Foxp2-Excitatory-Neurons-layers-6-and-2-3``),
so trailing-integer parsing is ambiguous. The reliable approach used here:

  1. Treat rows whose ``Group`` matches the file's first row as the *canonical*
     vocabulary (those rows are guaranteed to carry unsuffixed names — they
     were the first occurrence in the rbind).
  2. For every subsequent row, resolve to a canonical cluster by *longest
     canonical-prefix match*: the row's name equals ``C`` or ``C + <digits>``
     where ``C`` is a canonical name. Prefer the longest ``C`` so
     ``cluster-271`` resolves to canonical ``cluster-27`` rather than failing
     because no ``cluster-2`` exists.

Pathway-side cluster discovery reads the existing
``edge_slices/incytr_pathways/index.json`` (unsanitized sender/receiver names
already there). Each pathway cluster must be present in the canonical set or
the build hard-fails.

Filename slugging uses ``sanitize_celltype`` from
``alz.integration.pair_to_receiver_cache`` — imported, not re-implemented.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time

import numpy as np
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
_DIGIT_SUFFIX = re.compile(r"(\d+)$")


def _resolve_canonical(name: str, canonical: set[str], sorted_canonical: list[str]) -> str | None:
    """Map a (possibly dedup-suffixed) row name to its canonical cluster.

    Strategy: exact match first; otherwise longest canonical prefix where the
    trailing remainder is all-digits.
    """
    if name in canonical:
        return name
    # sorted_canonical is pre-sorted by descending length so the first match
    # is the longest valid prefix.
    for c in sorted_canonical:
        if len(c) >= len(name):
            continue
        if name.startswith(c) and name[len(c):].isdigit():
            return c
    return None


def _load_aggexp_long(pseudobulk_path: str) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Return (long_df, canonical_clusters, groups) where long_df has columns
    ``cluster, group, gene, value``.
    """
    if not os.path.exists(pseudobulk_path):
        raise FileNotFoundError(
            f"transcript-trace substrate missing: {pseudobulk_path}. "
            f"Pull data/incytr_frozen/v2_46clusters/provenance/aggexp.csv "
            f"before rebuilding the viewer."
        )
    print(f"  transcript_trace: reading {pseudobulk_path}", flush=True)
    df = pd.read_csv(pseudobulk_path, index_col=0)
    if "Group" not in df.columns:
        raise ValueError(
            f"aggexp substrate missing required `Group` column "
            f"(found: {list(df.columns)[:5]}...)"
        )
    row_names = list(df.index.astype(str))
    groups_col = df["Group"].astype(str).tolist()

    # Canonical = rows whose Group equals the FIRST row's Group. Those are
    # guaranteed to carry unsuffixed names (rbind dedup hadn't kicked in yet).
    first_group = groups_col[0]
    canonical_names: list[str] = [
        row_names[i] for i, g in enumerate(groups_col) if g == first_group
    ]
    if len(set(canonical_names)) != len(canonical_names):
        raise ValueError(
            "aggexp first-Group block has duplicate row names; cannot derive "
            "canonical cluster vocabulary."
        )
    canonical_set = set(canonical_names)
    print(f"  transcript_trace: canonical cluster vocabulary = "
          f"{len(canonical_names)} names (from Group={first_group!r})",
          flush=True)

    # Pre-sort canonical names by descending length for longest-match lookup.
    sorted_canonical = sorted(canonical_set, key=len, reverse=True)

    # Resolve every row's cluster.
    resolved: list[str] = []
    for i, raw in enumerate(row_names):
        c = _resolve_canonical(raw, canonical_set, sorted_canonical)
        if c is None:
            raise ValueError(
                f"aggexp row {i} name {raw!r} (Group={groups_col[i]!r}) does "
                f"not resolve to any canonical cluster. Canonical vocab size "
                f"= {len(canonical_set)}."
            )
        resolved.append(c)

    # Per-(cluster, group) uniqueness check: each cluster should appear at
    # most once per group (rbind dedup means a literal-name collision would
    # produce a different suffixed name — same canonical + same group is the
    # genuine error).
    df_key = pd.DataFrame({"cluster": resolved, "group": groups_col})
    dup_mask = df_key.duplicated(keep=False)
    if dup_mask.any():
        sample = df_key[dup_mask].head(5).to_dict("records")
        raise ValueError(
            f"transcript_trace: duplicate (cluster, group) keys after "
            f"resolution: e.g. {sample}"
        )

    # Pivot to long form: gene columns are everything except Group.
    gene_cols = [c for c in df.columns if c != "Group"]
    print(f"  transcript_trace: melting {len(df):,} rows × {len(gene_cols):,} "
          f"genes → long form", flush=True)
    values = df[gene_cols].to_numpy(dtype="float32")
    # Build the long table column-wise to avoid pandas' expensive melt for
    # 1078 × 25k.
    n_rows, n_genes = values.shape
    long_cluster = np.repeat(resolved, n_genes)
    long_group = np.repeat(groups_col, n_genes)
    long_gene = np.tile(gene_cols, n_rows)
    long_value = values.ravel()
    # Drop NaN/zero-only at write time (gene-by-gene); keep all here.
    long_df = pd.DataFrame({
        "cluster": pd.Categorical(long_cluster, categories=canonical_names),
        "group": pd.Categorical(long_group),
        "gene": long_gene,
        "value": long_value,
    })
    unique_groups = sorted(set(groups_col))
    return long_df, canonical_names, unique_groups


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
    long_df, canonical_names, unique_groups = _load_aggexp_long(
        TRANSCRIPT_TRACE_PSEUDOBULK
    )

    # Sample-key validation.
    samplekey = _load_samplekey(TRANSCRIPT_TRACE_SAMPLEKEY)
    sk_groups = set(samplekey.keys())
    agg_groups = set(unique_groups)
    missing_in_sk = agg_groups - sk_groups
    if missing_in_sk:
        raise ValueError(
            f"transcript_trace: aggexp Group values {sorted(missing_in_sk)[:5]} "
            f"absent from sample-key SCRNA_ID column at "
            f"{TRANSCRIPT_TRACE_SAMPLEKEY}"
        )

    # Pathway-side cluster discovery + coverage check.
    incytr_idx = os.path.join(EDGE_SLICES_INCYTR_PATHWAYS_DIR, "index.json")
    pathway_clusters = _load_pathway_clusters(incytr_idx)
    canonical_set = set(canonical_names)
    missing = pathway_clusters - canonical_set
    if missing:
        raise ValueError(
            f"transcript_trace: pathway output references cluster(s) "
            f"{sorted(missing)} that are absent from the pseudobulk at "
            f"{TRANSCRIPT_TRACE_PSEUDOBULK}. Substrate drift — regenerate "
            f"aggexp.csv with the active cluster vocabulary or restrict the "
            f"pathway run."
        )
    print(f"  transcript_trace: {len(pathway_clusters)} pathway clusters "
          f"(all present in pseudobulk)", flush=True)

    # Per-cluster shard write.
    # Decoder lookup → arrays parallel to the group categorical for fast join.
    group_to_meta = {
        g: samplekey[g] if g in samplekey else _decode_group(g)
        for g in unique_groups
    }

    shards_written: dict[str, str] = {}
    for cluster in sorted(pathway_clusters):
        sub = long_df[long_df["cluster"] == cluster].copy()
        if sub.empty:
            raise ValueError(
                f"transcript_trace: cluster {cluster!r} resolved zero rows "
                f"in pseudobulk; substrate is inconsistent."
            )
        sub = sub.drop(columns=["cluster"])
        sub["group"] = sub["group"].astype(str)
        meta_rows = sub["group"].map(samplekey)
        sub["sex"] = meta_rows.map(lambda m: m["sex"])
        sub["timepoint"] = meta_rows.map(lambda m: m["timepoint"])
        sub["genotype"] = meta_rows.map(lambda m: m["genotype"])
        sub = sub[["gene", "group", "sex", "timepoint", "genotype", "value"]]

        slug = _sanitize_celltype(cluster)
        out_path = os.path.join(TRANSCRIPT_TRACE_DIR, f"{slug}.parquet")
        sub.to_parquet(out_path, index=False, compression="zstd")
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

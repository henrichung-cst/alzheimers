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


def _load_aggexp_wide(pseudobulk_path: str):
    """Return (values, row_clusters, row_groups, gene_cols, canonical_names,
    unique_groups) — the wide float32 matrix plus resolved metadata.

    Kept wide so the per-cluster loop in ``build()`` can slice a small subset
    and explode only that cluster's ~24 rows × n_genes into long form. The
    previous "explode-the-whole-thing" version peaked at multi-GB on a 1078 ×
    25k input.
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

    sorted_canonical = sorted(canonical_set, key=len, reverse=True)

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

    # Per-(cluster, group) uniqueness check.
    df_key = pd.DataFrame({"cluster": resolved, "group": groups_col})
    dup_mask = df_key.duplicated(keep=False)
    if dup_mask.any():
        sample = df_key[dup_mask].head(5).to_dict("records")
        raise ValueError(
            f"transcript_trace: duplicate (cluster, group) keys after "
            f"resolution: e.g. {sample}"
        )

    gene_cols = [c for c in df.columns if c != "Group"]
    values = df[gene_cols].to_numpy(dtype="float32")
    # Release the original wide DataFrame; we only need the float32 matrix +
    # parallel metadata lists from here on.
    del df
    unique_groups = sorted(set(groups_col))
    return values, resolved, groups_col, gene_cols, canonical_names, unique_groups


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
    values, row_clusters, row_groups, gene_cols, canonical_names, unique_groups = (
        _load_aggexp_wide(TRANSCRIPT_TRACE_PSEUDOBULK)
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

    # Per-cluster shard write — stream one cluster at a time. Each cluster has
    # ~24 rows (sex × tp × genotype groups), so the per-iteration long frame is
    # ~24 × n_genes ≈ 600k rows. Holding only one cluster at a time keeps peak
    # RSS ~50 MB instead of multi-GB.
    row_clusters_arr = np.asarray(row_clusters)
    row_groups_arr = np.asarray(row_groups)
    n_genes = len(gene_cols)
    gene_cols_arr = np.asarray(gene_cols)

    shards_written: dict[str, str] = {}
    for cluster in sorted(pathway_clusters):
        idx = np.flatnonzero(row_clusters_arr == cluster)
        if idx.size == 0:
            raise ValueError(
                f"transcript_trace: cluster {cluster!r} resolved zero rows "
                f"in pseudobulk; substrate is inconsistent."
            )
        sub_groups = row_groups_arr[idx]
        sub_values = values[idx]  # (n_rows_in_cluster, n_genes)
        n_rows = sub_values.shape[0]

        long_gene = np.tile(gene_cols_arr, n_rows)
        long_group = np.repeat(sub_groups, n_genes)
        long_value = sub_values.ravel()

        # Vectorize sex/tp/geno via per-group lookup table → cheap repeat.
        sex_per_row = np.array([samplekey[g]["sex"] for g in sub_groups])
        tp_per_row = np.array([samplekey[g]["timepoint"] for g in sub_groups])
        geno_per_row = np.array([samplekey[g]["genotype"] for g in sub_groups])
        long_sex = np.repeat(sex_per_row, n_genes)
        long_tp = np.repeat(tp_per_row, n_genes)
        long_geno = np.repeat(geno_per_row, n_genes)

        sub = pd.DataFrame({
            "gene": long_gene,
            "group": long_group,
            "sex": long_sex,
            "timepoint": long_tp,
            "genotype": long_geno,
            "value": long_value,
        })

        slug = _sanitize_celltype(cluster)
        out_path = os.path.join(TRANSCRIPT_TRACE_DIR, f"{slug}.parquet")
        sub.to_parquet(out_path, index=False, compression="zstd")
        shards_written[cluster] = os.path.relpath(out_path, UNIFIED_VIEWER_DIR)
        del sub, long_gene, long_group, long_value, long_sex, long_tp, long_geno

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

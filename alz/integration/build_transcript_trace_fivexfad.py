#!/usr/bin/env python3
"""Build per-cluster transcript pseudobulk shards for the 5xFAD contexts of the
Incytr Pathways "Evidence" panel (transcript layer).

5xFAD analog of ``build_transcript_trace.py``. Substrate is the per-cell-type
per-sample linear pseudobulk at
``outputs/reports/kinase_attribution_5xfad/celltype_mea/fivexfad_snrna_pseudobulk_linear.csv.gz``
(wide: rows = (tissue, age_months, genotype, sample_id, cell_type, n_cells,
<gene columns…>)).

Mirroring Song's transcript layer — which is pseudobulk n=1 per arm — the 5xFAD
transcript collapses its ≤2 scRNA samples per (cell_type, genotype, age) to a
single condition mean, rendered as one bar per arm. Output schema matches the
Song transcript_trace shard so ``TranscriptTraceStore`` reads it unchanged:

    gene; group (= "<geno>_<age>mo", e.g. "TG_3mo"); genotype ("TG"|"WT");
    timepoint ("3mo".."12mo"); value (linear pseudobulk condition mean)

Routed to the same evidence genes + pathway clusters as the omics-trace 5xFAD
builder (Ligand→sender, Receptor/EM/Target→receiver).

Schema version: 1 — bump TRANSCRIPT_TRACE_FIVEXFAD_SCHEMA_VERSION in
alz/viewer/paths.py on any schema change.
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
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))  # repo root

from alz.shared import config  # noqa: E402
from alz.incytr_pair.pair_to_receiver_cache import _sanitize_celltype  # noqa: E402
from alz.integration.build_omics_trace_fivexfad import (  # noqa: E402
    _load_evidence_genes,
    _load_pathway_clusters,
    _pathway_index_path,
)
from alz.viewer.paths import (  # noqa: E402
    TRANSCRIPT_TRACE_FIVEXFAD_CORTEX_DIR,
    TRANSCRIPT_TRACE_FIVEXFAD_CORTEX_INDEX,
    TRANSCRIPT_TRACE_FIVEXFAD_HIPPO_DIR,
    TRANSCRIPT_TRACE_FIVEXFAD_HIPPO_INDEX,
    TRANSCRIPT_TRACE_FIVEXFAD_SCHEMA_VERSION,
    UNIFIED_VIEWER_DIR,
)

PSEUDOBULK_CSV = os.path.join(
    config.REPO_ROOT, "outputs", "reports", "kinase_attribution_5xfad",
    "celltype_mea", "fivexfad_snrna_pseudobulk_linear.csv.gz",
)
_META_COLS = ["tissue", "age_months", "genotype", "sample_id", "cell_type", "n_cells"]

_TISSUE_OUT = {
    "cortex": (TRANSCRIPT_TRACE_FIVEXFAD_CORTEX_DIR, TRANSCRIPT_TRACE_FIVEXFAD_CORTEX_INDEX),
    "hippocampus": (TRANSCRIPT_TRACE_FIVEXFAD_HIPPO_DIR, TRANSCRIPT_TRACE_FIVEXFAD_HIPPO_INDEX),
}

_SHARD_COLS = ["gene", "group", "genotype", "timepoint", "value"]


def build_tissue(tissue: str, force: bool = False) -> dict:
    out_dir, index_path = _TISSUE_OUT[tissue]

    if not force and os.path.exists(index_path):
        with open(index_path) as f:
            existing = json.load(f)
        if existing.get("trace_schema_version") == TRANSCRIPT_TRACE_FIVEXFAD_SCHEMA_VERSION:
            return existing

    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    t0 = time.time()
    print(f"\n=== transcript_trace_fivexfad {tissue} ===", flush=True)
    if not os.path.exists(PSEUDOBULK_CSV):
        raise FileNotFoundError(
            f"5xFAD scRNA pseudobulk substrate missing: {PSEUDOBULK_CSV}.")

    pathway_clusters = _load_pathway_clusters(_pathway_index_path(tissue))
    evidence_genes = _load_evidence_genes(tissue)
    evid_all = set().union(*evidence_genes.values()) if evidence_genes else set()

    # Read only the metadata columns + present evidence-gene columns (the wide
    # substrate has thousands of gene columns; routing keeps this bounded).
    header = pd.read_csv(PSEUDOBULK_CSV, nrows=0).columns.tolist()
    gene_cols = [c for c in header if c in evid_all]
    df = pd.read_csv(PSEUDOBULK_CSV, usecols=_META_COLS + gene_cols)
    df = df[df["tissue"].astype(str) == tissue]
    df["cell_type"] = df["cell_type"].astype(str)
    df = df[df["cell_type"].isin(pathway_clusters)]
    print(f"  {len(pathway_clusters)} pathway clusters, {len(gene_cols):,} routed "
          f"genes present, {len(df):,} pseudobulk sample rows", flush=True)

    missing_clusters = pathway_clusters - set(df["cell_type"].unique())
    if missing_clusters:
        print(f"  (warn) {len(missing_clusters)} pathway cluster(s) absent from "
              f"pseudobulk: {sorted(missing_clusters)}", flush=True)

    # Collapse ≤2 scRNA samples per (cell_type, genotype, age) to a mean.
    cond = df.groupby(["cell_type", "genotype", "age_months"], sort=True)[gene_cols].mean()

    shards_written: dict[str, str] = {}
    groups_seen: set[str] = set()
    for cl in sorted(pathway_clusters):
        if cl not in cond.index.get_level_values("cell_type"):
            continue
        genes_cl = sorted(evidence_genes.get(cl, set()) & set(gene_cols))
        if not genes_cl:
            continue
        sub = cond.xs(cl, level="cell_type")  # index (genotype, age_months)
        rows: list[dict] = []
        for (geno, age), gene_means in sub[genes_cl].iterrows():
            group = f"{geno}_{int(age)}mo"
            tp = f"{int(age)}mo"
            groups_seen.add(group)
            for gene, val in gene_means.items():
                if pd.isna(val):
                    continue
                rows.append({"gene": gene, "group": group, "genotype": geno,
                             "timepoint": tp, "value": float(val)})
        if not rows:
            continue
        shard = pd.DataFrame(rows, columns=_SHARD_COLS)
        slug = _sanitize_celltype(cl)
        out_path = os.path.join(out_dir, f"{slug}.parquet")
        shard.to_parquet(out_path, index=False, compression="zstd")
        shards_written[cl] = os.path.relpath(out_path, UNIFIED_VIEWER_DIR)

    if not shards_written:
        raise RuntimeError(
            f"transcript_trace_fivexfad {tissue}: no shards written — logic error.")

    index = {
        "trace_schema_version": TRANSCRIPT_TRACE_FIVEXFAD_SCHEMA_VERSION,
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "tissue": tissue,
        "label": f"5xFAD {tissue} transcript pseudobulk trace",
        "source_pseudobulk": os.path.relpath(PSEUDOBULK_CSV, config.REPO_ROOT),
        "substrate_note": (
            "Per-(cell_type, genotype, age) mean of the linear snRNA pseudobulk "
            "(≤2 libraries per arm, collapsed to a single value — mirrors Song's "
            "n=1-per-arm transcript layer)."
        ),
        "sanitize_rule": "replace('/', '-'); replace(' ', '_')",
        "filename_template": "{cluster}.parquet",
        "relative_path": os.path.relpath(out_dir, UNIFIED_VIEWER_DIR),
        "clusters": sorted(shards_written.keys()),
        "shard_files": shards_written,
        "groups": sorted(groups_seen),
        "n_libraries_per_arm": 1,
        "note": "5xFAD transcript pseudobulk · condition mean · 1 bar per arm",
    }
    with open(index_path, "w") as f:
        json.dump(index, f)
    print(f"  wrote {len(shards_written)} cluster shards in "
          f"{time.time() - t0:.1f}s", flush=True)
    return index


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tissue", choices=["cortex", "hippocampus", "all"],
                    default="all")
    ap.add_argument("--force", action="store_true",
                    help="Rebuild even if index is current.")
    args = ap.parse_args()
    tissues = ["cortex", "hippocampus"] if args.tissue == "all" else [args.tissue]
    for t in tissues:
        build_tissue(t, force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

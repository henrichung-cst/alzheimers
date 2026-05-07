"""Factorial all-pairs substrate-based kinase support scoring.

Per pair x contrast: structural substrate evidence only (MEA-significant
kinases whose substrates land on the pathway's EM/Target nodes, weighted by
IDF and absolute MEA NES). No per-pair cell-type-attribution multiplier — the
pair-mode pipeline applies one via ``apply_pair_weights`` from
``unified_attribution.csv``, but that step (a) re-uses the live pipeline's
WMB-class-keyed attribution scores under SEA-AD-subclass cell-type names
(taxonomy mismatch produces silently-zero weights for every pair), and
(b) bakes cell-type attribution into the support score, where it cannot be
audited or re-joined cleanly downstream. Cell-type attribution is the live
pipeline's job; this sidecar produces a per-pathway structural support that
consumers can join to ``unified_attribution.csv`` post-hoc.

Restore the prior pair-weighted behavior via ``git show
legacy-incytr-storage-cutover:code/integration/sidecar/kinase_pack/compute_kinase_support_factorial.py``.

Output per pair: kinase_support_scores.csv with per-contrast columns:
  mea_kinase_support_score_{contrast}, mea_concordance_flag_{contrast}, etc.

Usage:
  python compute_kinase_support_factorial.py
  python compute_kinase_support_factorial.py --pair-filter "Astrocyte__*"
  python compute_kinase_support_factorial.py --force
"""

import argparse
import fnmatch
import json
import os
import sys
import time

# Sidecar lives at code/integration/sidecar/kinase_pack/; add code/integration/
# and code/integration/adapters/ to sys.path so we can import the in-tree
# helpers (config_integration, common, compute_kinase_support).
_HERE = os.path.dirname(os.path.abspath(__file__))
_INT_DIR = os.path.dirname(os.path.dirname(_HERE))
for _p in (_INT_DIR, os.path.join(_INT_DIR, "adapters")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pandas as pd

from common import (load_mouse_gene_to_kinase_mapping,
                    build_substrate_kinase_map, ensure_intermediates_dir,
                    sanitize_celltype_name)
from compute_kinase_support import (
    _load_mea_kinases, _compute_idf_map,
    build_substrate_edge_table, compute_scores_fast, edges_to_list_form,
)
import config_integration as icfg


# ---------------------------------------------------------------------------
# Shared data loading (per-contrast)
# ---------------------------------------------------------------------------

def load_shared_data(contrast_filter=None):
    """Load pair-independent data for contrasts.

    Builds per-contrast edge tables (different MEA-significant kinases).
    IDF is pair-independent and contrast-independent (substrate promiscuity
    is a static property of the kinase-substrate network).

    Parameters
    ----------
    contrast_filter : str or list[str], optional
        If given, only load data for the specified contrast(s). Reduces
        memory from ~9x to 1x for single-contrast runs like permutation
        tests.
    """
    t0 = time.monotonic()

    if contrast_filter is None:
        contrasts_to_load = list(icfg.FACTORIAL_CONTRASTS.keys())
    elif isinstance(contrast_filter, str):
        contrasts_to_load = [contrast_filter]
    else:
        contrasts_to_load = list(contrast_filter)

    kldata = pd.read_csv(os.path.join(icfg.INTERMEDIATES_DIR, "kldata.csv"))
    sub_to_kins = build_substrate_kinase_map(kldata)
    del kldata
    print(f"  kldata: {len(sub_to_kins)} substrates")

    mouse_to_abbrevs = load_mouse_gene_to_kinase_mapping()
    print(f"  naming bridge: {sum(len(v) for v in mouse_to_abbrevs.values())} mappings")

    # Per-contrast: sig_kinases, edge tables, IDF. Each contrast's edge table
    # is converted to list form once here (no per-pair attribution weighting).
    contrast_data = {}
    for contrast in contrasts_to_load:
        sig_kinases, all_mea_nes = _load_mea_kinases(
            contrast, icfg.PHOSPHO_FDR_GATE)

        # Pair-independent IDF (same across pairs, computed per contrast
        # because IDF depends on which kinases are significant)
        idf_map = _compute_idf_map(sub_to_kins, mouse_to_abbrevs, sig_kinases,
                                   pair_independent=True)

        sub_raw_edges, all_kinase_genes = build_substrate_edge_table(
            sub_to_kins, idf_map, sig_kinases, mouse_to_abbrevs)

        contrast_data[contrast] = {
            "sub_pair": edges_to_list_form(sub_raw_edges),
            "all_kinase_genes": all_kinase_genes,
        }
        print(f"  {contrast}: {len(sig_kinases)} sig kinases, "
              f"{len(sub_raw_edges)} substrates in edge table")

    elapsed = time.monotonic() - t0
    print(f"  shared data loaded in {elapsed:.1f}s")

    return {"contrast_data": contrast_data}


# ---------------------------------------------------------------------------
# Pair discovery
# ---------------------------------------------------------------------------

def discover_pairs(pair_filter=None):
    """Discover pairs from factorial receiver-indexed Parquet files."""
    import glob as globmod
    import pyarrow.parquet as pq

    parquet_dir = icfg.FACTORIAL_ALL_PAIRS_DIR
    parquet_files = sorted(globmod.glob(
        os.path.join(parquet_dir, "recv_*.parquet")))

    if not parquet_files:
        raise FileNotFoundError(
            f"No recv_*.parquet files in {parquet_dir}. "
            "Run the factorial R pipeline first.")

    pairs = []
    for pq_path in parquet_files:
        meta = pq.read_metadata(pq_path)
        file_meta = meta.metadata or {}
        receiver = file_meta.get(b"receiver", b"").decode()
        if not receiver:
            base = os.path.basename(pq_path).replace("recv_", "").replace(".parquet", "")
            receiver = base.replace("_", " ")

        table = pq.read_table(pq_path, columns=["sender"])
        senders = sorted(set(table.column("sender").to_pylist()))

        for sender in senders:
            pairs.append((pq_path, sender, receiver))

    print(f"  {len(parquet_files)} Parquet files, {len(pairs)} pairs")

    if pair_filter:
        pairs = [(p, s, r) for p, s, r in pairs
                 if fnmatch.fnmatch(
                     f"{sanitize_celltype_name(s)}__{sanitize_celltype_name(r)}",
                     pair_filter)]

    return pairs


# ---------------------------------------------------------------------------
# Per-pair processing
# ---------------------------------------------------------------------------

PATHWAY_ID_COLS = ["Path", "EM", "Target", "Receptor", "Ligand"]


def process_one_pair(shared, pq_path, sender, receiver, *, emit_routes=False):
    """Score one pair across all 9 contrasts. Returns summary dict."""
    import pyarrow.parquet as pq

    t0 = time.monotonic()

    # Read pathway structure + per-contrast TPDS/PDS when present.
    tpds_cols = [f"TPDS_{c}" for c in icfg.FACTORIAL_CONTRASTS]
    schema_names = set(pq.read_schema(pq_path).names)
    pds_cols = [
        f"PDS_{c}" for c in icfg.FACTORIAL_CONTRASTS
        if f"PDS_{c}" in schema_names
    ]
    usecols = PATHWAY_ID_COLS + ["sender"] + tpds_cols + pds_cols
    filters = [("sender", "=", sender)]
    table = pq.read_table(pq_path, columns=usecols, filters=filters)
    df = table.drop("sender").to_pandas()
    t_read = time.monotonic() - t0

    n_pathways = len(df)

    # Build pathway base DataFrame (Path, EM, Target, Receptor, Ligand)
    pathways_base = df[PATHWAY_ID_COLS].copy()

    # Score each contrast
    all_scores = {}
    routes_per_contrast = {} if emit_routes else None
    for contrast in icfg.FACTORIAL_CONTRASTS:
        cdata = shared["contrast_data"][contrast]
        sub_pair = cdata["sub_pair"]

        # Build pathways DataFrame with TPDS and PDS columns expected by
        # compute_scores_fast. Newer factorial runs emit native-style PDS
        # columns; older checkpoints fall back to TPDS for compatibility.
        tpds_col = f"TPDS_{contrast}"
        pds_col = f"PDS_{contrast}"
        pathways = pathways_base.copy()
        pathways["TPDS"] = df[tpds_col].values
        if pds_col in df.columns:
            pathways["PDS"] = df[pds_col].values
        else:
            pathways["PDS"] = df[tpds_col].values

        routes_sink = [] if emit_routes else None
        scores_c = compute_scores_fast(
            pathways, sub_pair, cdata["all_kinase_genes"],
            routes_sink=routes_sink)
        if emit_routes:
            routes_per_contrast[contrast] = routes_sink

        # Collect per-contrast columns
        all_scores[f"mea_kinase_support_score_{contrast}"] = scores_c["mea_kinase_support_score"].values
        all_scores[f"mea_kinase_support_sum_{contrast}"] = scores_c["mea_kinase_support_sum"].values
        all_scores[f"mea_n_distinct_kinases_{contrast}"] = scores_c["mea_n_distinct_kinases"].values
        all_scores[f"mea_concordance_flag_{contrast}"] = scores_c["mea_concordance_flag"].values

    t_score = time.monotonic() - t0 - t_read

    # Build output DataFrame
    out = pathways_base.copy()
    for col, vals in all_scores.items():
        out[col] = vals

    # Add per-contrast TPDS for reference
    for contrast in icfg.FACTORIAL_CONTRASTS:
        out[f"TPDS_{contrast}"] = df[f"TPDS_{contrast}"].values
        pds_col = f"PDS_{contrast}"
        if pds_col in df.columns:
            out[pds_col] = df[pds_col].values

    # Write outputs
    t_w = time.monotonic()
    dir_path = os.path.join(
        icfg.FACTORIAL_ALL_PAIRS_DIR,
        f"{sanitize_celltype_name(sender)}__{sanitize_celltype_name(receiver)}")
    os.makedirs(dir_path, exist_ok=True)

    out.to_csv(os.path.join(dir_path, "kinase_support_scores.csv"),
               index=False, float_format="%.6g")

    if emit_routes:
        rows = []
        for contrast, sink in routes_per_contrast.items():
            for path, kinase, contribution, nes_sign in sink:
                rows.append((path, contrast, kinase, contribution, nes_sign))
        if rows:
            routes_df = pd.DataFrame(
                rows,
                columns=["Path", "contrast", "kinase",
                         "mea_support_contribution", "nes_sign"],
            )
            routes_df["mea_support_contribution"] = (
                routes_df["mea_support_contribution"].astype("float32")
            )
            routes_df["nes_sign"] = routes_df["nes_sign"].astype("int8")
            routes_df.to_parquet(
                os.path.join(dir_path, "kinase_routes.parquet"),
                index=False,
                compression="zstd",
            )

    # Summary JSON
    summary_data = {"n_pathways": n_pathways, "contrasts": {}}
    for contrast in icfg.FACTORIAL_CONTRASTS:
        ks = out[f"mea_kinase_support_score_{contrast}"]
        n_nz = int((ks > 0).sum())
        summary_data["contrasts"][contrast] = {
            "n_nonzero": n_nz,
            "pct_nonzero": round(100 * n_nz / max(n_pathways, 1), 1),
            "median_score": round(float(ks[ks > 0].median()), 4) if n_nz else 0.0,
        }
    with open(os.path.join(dir_path, "reranking_summary.json"), "w") as f:
        json.dump(summary_data, f, indent=2)

    t_write = time.monotonic() - t_w
    total = time.monotonic() - t0

    return {
        "sender": sender,
        "receiver": receiver,
        "n_pathways": n_pathways,
        "time_sec": round(total, 1),
        "time_read": round(t_read, 2),
        "time_score": round(t_score, 2),
        "time_write": round(t_write, 2),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Factorial all-pairs kinase support scoring")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing outputs")
    parser.add_argument("--pair-filter", metavar="PATTERN",
                        help="Filter pairs by glob (e.g. 'Astrocyte__*')")
    parser.add_argument("--emit-kinase-routes", action="store_true",
                        help="Also emit per-(backbone, kinase, contrast) "
                             "routes to kinase_routes.parquet alongside "
                             "kinase_support_scores.csv")
    args = parser.parse_args()

    ensure_intermediates_dir()

    print("Loading shared data (9 contrasts)...")
    shared = load_shared_data()

    pairs = discover_pairs(args.pair_filter)
    print(f"\nDiscovered {len(pairs)} pairs")

    if not pairs:
        print("No pairs to process.")
        return

    summaries = []
    n_skipped = 0
    n_total = len(pairs)

    for i, (pq_path, sender, receiver) in enumerate(pairs, 1):
        dir_path = os.path.join(
            icfg.FACTORIAL_ALL_PAIRS_DIR,
            f"{sanitize_celltype_name(sender)}__{sanitize_celltype_name(receiver)}")
        scores_path = os.path.join(dir_path, "kinase_support_scores.csv")

        routes_path = os.path.join(dir_path, "kinase_routes.parquet")
        needs_routes = args.emit_kinase_routes and not os.path.exists(routes_path)
        if os.path.exists(scores_path) and not args.force and not needs_routes:
            n_skipped += 1
            continue

        result = process_one_pair(shared, pq_path, sender, receiver,
                                  emit_routes=args.emit_kinase_routes)
        summaries.append(result)

        if i % 50 == 0 or i == n_total:
            print(f"  [{i}/{n_total}] {sender} -> {receiver}: "
                  f"{result['n_pathways']} pathways, {result['time_sec']}s")

    # Write summary
    if summaries:
        summary_df = pd.DataFrame(summaries)
        summary_path = os.path.join(icfg.FACTORIAL_ALL_PAIRS_DIR,
                                    "kinase_support_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        total_time = summary_df["time_sec"].sum()
        print(f"\n=== Summary ===")
        print(f"  Processed: {len(summaries)}/{n_total} pairs")
        print(f"  Skipped:   {n_skipped}")
        print(f"  Total time: {total_time:.0f}s ({total_time/60:.1f} min)")
        print(f"  Wrote {summary_path}")
    else:
        print(f"\nAll {n_skipped} pairs already scored (use --force to rerun).")

    print("\nFactorial kinase support scoring complete.")


if __name__ == "__main__":
    main()

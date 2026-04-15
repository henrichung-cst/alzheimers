"""All-pairs substrate-based kinase support scoring.

Extends compute_kinase_support.py to process all 462 sender-receiver pairs.
Reads pathway data from receiver-indexed Parquet files (recv_*.parquet),
loads shared data (kldata, MEA, attribution, IDF) once, then iterates over
pairs writing per-pair kinase_support_scores.csv and a cross-pair summary.

Uses pair-independent IDF (promiscuity is a substrate property, not a pair
property).  Attribution relevance is captured by the per-pair multiplicative
attribution_weight in the edge weight formula.

Usage:
  python compute_kinase_support_all_pairs.py                          # all pairs
  python compute_kinase_support_all_pairs.py --profile-pair Microglia-PVM__L5_IT
  python compute_kinase_support_all_pairs.py --pair-filter "Astrocyte__*"
  python compute_kinase_support_all_pairs.py --force                  # overwrite
"""

import argparse
import fnmatch
import json
import os
import time

import pandas as pd

from common import (load_mouse_gene_to_kinase_mapping,
                    build_substrate_kinase_map, ensure_intermediates_dir,
                    sanitize_celltype_name)
from compute_kinase_support import (
    _load_mea_kinases, _compute_idf_map,
    build_substrate_edge_table, apply_pair_weights, compute_scores_fast,
    compute_scores, compute_adjusted_rankings, run_sensitivity_analyses,
)
import config_integration as icfg


# ---------------------------------------------------------------------------
# Shared data loading
# ---------------------------------------------------------------------------

def load_shared_data():
    """Load all pair-independent data structures once.

    Builds a precomputed edge table (substrate -> edge arrays with |NES|*IDF)
    that is shared across all 462 pairs.  Per-pair attribution weights are
    applied later via :func:`apply_pair_weights`.
    """
    t0 = time.monotonic()

    kldata = pd.read_csv(os.path.join(icfg.INTERMEDIATES_DIR, "kldata.csv"))
    sub_to_kins = build_substrate_kinase_map(kldata)
    print(f"  kldata: {len(kldata)} edges, {len(sub_to_kins)} substrates")

    mouse_to_abbrevs = load_mouse_gene_to_kinase_mapping()
    print(f"  naming bridge: {sum(len(v) for v in mouse_to_abbrevs.values())} mappings")

    sig_kinases, all_mea_nes = _load_mea_kinases(
        icfg.CONTRAST, icfg.PHOSPHO_FDR_GATE)
    print(f"  MEA: {len(sig_kinases)} significant, {len(all_mea_nes)} total")

    # Pair-independent IDF
    idf_map = _compute_idf_map(sub_to_kins, mouse_to_abbrevs, sig_kinases,
                               pair_independent=True)
    print(f"  IDF: {len(idf_map)} substrates (pair-independent)")

    # Precomputed edge table (pair-independent, ~50ms)
    sub_raw_edges, all_kinase_genes = build_substrate_edge_table(
        sub_to_kins, idf_map, sig_kinases, mouse_to_abbrevs)
    print(f"  edge table: {len(sub_raw_edges)} substrates, "
          f"{len(all_kinase_genes)} kinase genes")

    # Pre-index attribution by cell type
    attr = pd.read_csv(icfg.UNIFIED_ATTRIBUTION_CSV)
    attr_c = attr[attr["contrast"] == icfg.CONTRAST]
    attr_by_celltype = {}
    for _, row in attr_c.iterrows():
        ct = row["cell_type"]
        attr_by_celltype.setdefault(ct, {})[row["kinase"]] = row["combined_score"]
    print(f"  attribution: {len(attr_by_celltype)} cell types, "
          f"{sum(len(v) for v in attr_by_celltype.values())} kinase-celltype entries")

    elapsed = time.monotonic() - t0
    print(f"  shared data loaded in {elapsed:.1f}s")

    return {
        "sub_to_kins": sub_to_kins,
        "mouse_to_abbrevs": mouse_to_abbrevs,
        "sig_kinases": sig_kinases,
        "all_mea_nes": all_mea_nes,
        "idf_map": idf_map,
        "sub_raw_edges": sub_raw_edges,
        "all_kinase_genes": all_kinase_genes,
        "attr_by_celltype": attr_by_celltype,
    }


# ---------------------------------------------------------------------------
# Pair discovery
# ---------------------------------------------------------------------------

def discover_pairs(pair_filter=None, profile_pair=None):
    """Discover pairs from receiver-indexed Parquet files.

    Returns list of (parquet_path, sender, receiver) tuples.
    """
    import glob as globmod
    import pyarrow.parquet as pq

    parquet_files = sorted(globmod.glob(
        os.path.join(icfg.ALL_PAIRS_DIR, "recv_*.parquet")))

    if not parquet_files:
        raise FileNotFoundError(
            f"No recv_*.parquet files in {icfg.ALL_PAIRS_DIR}. "
            "Run the R pipeline first to produce Parquet output.")

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
            dir_name = f"{sanitize_celltype_name(sender)}__{sanitize_celltype_name(receiver)}"
            pairs.append((pq_path, sender, receiver, dir_name))

    print(f"  {len(parquet_files)} Parquet files, {len(pairs)} pairs")

    if profile_pair:
        pairs = [(p, s, r, d) for p, s, r, d in pairs if d == profile_pair]
        if not pairs:
            raise ValueError(f"No pair matching --profile-pair '{profile_pair}'.")
    elif pair_filter:
        pairs = [(p, s, r, d) for p, s, r, d in pairs
                 if fnmatch.fnmatch(d, pair_filter)]

    return [(p, s, r) for p, s, r, _ in pairs]


# ---------------------------------------------------------------------------
# Per-pair attribution weights
# ---------------------------------------------------------------------------

def compute_pair_attr_weights(attr_by_celltype, sender, receiver,
                              sender_discount):
    """Build per-kinase attribution weights for a specific pair.

    Receiver attribution: combined_score x 1.0
    Sender attribution:   combined_score x sender_discount
    Takes the max per kinase across sender/receiver.
    """
    weights = {}
    for kin, score in attr_by_celltype.get(receiver, {}).items():
        weights[kin] = score * 1.0
    for kin, score in attr_by_celltype.get(sender, {}).items():
        w = score * sender_discount
        if kin not in weights or w > weights[kin]:
            weights[kin] = w
    return weights


# ---------------------------------------------------------------------------
# Process one pair
# ---------------------------------------------------------------------------

SCORING_COLS = ["Path", "EM", "Target", "Receptor", "Ligand", "TPDS", "PDS"]
SENSITIVITY_COLS = SCORING_COLS + ["PhPDS_ps"]


def process_one_pair(shared, pq_path, sender, receiver, run_sensitivity=True):
    """Score one pair from a receiver Parquet file.  Returns summary dict."""
    import pyarrow.parquet as pq

    t0 = time.monotonic()
    usecols = SENSITIVITY_COLS if run_sensitivity else SCORING_COLS
    filters = [("sender", "=", sender)]

    try:
        table = pq.read_table(pq_path, columns=usecols + ["sender"],
                              filters=filters)
        results_full = table.drop("sender").to_pandas()
    except Exception:
        table = pq.read_table(pq_path, columns=SCORING_COLS + ["sender"],
                              filters=filters)
        results_full = table.drop("sender").to_pandas()

    # Output goes to per-pair directory
    dir_path = os.path.join(
        os.path.dirname(pq_path),
        f"{sanitize_celltype_name(sender)}__{sanitize_celltype_name(receiver)}")
    os.makedirs(dir_path, exist_ok=True)

    pathways = results_full[SCORING_COLS].copy()
    t_read = time.monotonic() - t0

    # Per-pair attribution weights -> weighted edge table
    t1 = time.monotonic()
    attr_weights = compute_pair_attr_weights(
        shared["attr_by_celltype"], sender, receiver,
        icfg.SENDER_ATTRIBUTION_DISCOUNT)
    sub_pair = apply_pair_weights(shared["sub_raw_edges"], attr_weights)
    t_attr = time.monotonic() - t1

    # Score pathways (vectorized fast path)
    t2 = time.monotonic()
    scores_df = compute_scores_fast(
        pathways, sub_pair, shared["all_kinase_genes"])
    t_score = time.monotonic() - t2

    # Adjusted rankings
    t3 = time.monotonic()
    adj_df = compute_adjusted_rankings(scores_df, icfg.LAMBDA_VALUES)
    t_rank = time.monotonic() - t3

    # Sensitivity analyses
    summary = {}
    t_sens = 0.0
    if run_sensitivity:
        t4 = time.monotonic()
        summary = run_sensitivity_analyses(
            scores_df, results_full, icfg.LAMBDA_VALUES, adj_df=adj_df)
        t_sens = time.monotonic() - t4

    # Write outputs
    t5 = time.monotonic()
    scores_df.to_csv(os.path.join(dir_path, "kinase_support_scores.csv"),
                     index=False, float_format="%.6g")
    adj_df.to_csv(os.path.join(dir_path, "adjusted_rankings.csv"),
                  index=False, float_format="%.6g")
    summary_path = os.path.join(dir_path, "reranking_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    t_write = time.monotonic() - t5

    total_time = time.monotonic() - t0
    n_nonzero = int((scores_df["kinase_support_score"] > 0).sum())
    ks_vals = scores_df.loc[scores_df["kinase_support_score"] > 0,
                            "kinase_support_score"]

    return {
        "sender": sender,
        "receiver": receiver,
        "n_pathways": len(pathways),
        "n_nonzero_score": n_nonzero,
        "pct_nonzero": round(100 * n_nonzero / max(len(pathways), 1), 1),
        "median_score": round(float(ks_vals.median()), 4) if len(ks_vals) else 0.0,
        "mean_score": round(float(ks_vals.mean()), 4) if len(ks_vals) else 0.0,
        "max_score": round(float(ks_vals.max()), 4) if len(ks_vals) else 0.0,
        "time_sec": round(total_time, 1),
        "time_read": round(t_read, 2),
        "time_attr": round(t_attr, 4),
        "time_score": round(t_score, 2),
        "time_rank": round(t_rank, 2),
        "time_sensitivity": round(t_sens, 2),
        "time_write": round(t_write, 2),
    }


# ---------------------------------------------------------------------------
# Profiling mode
# ---------------------------------------------------------------------------

def profile_single_pair(shared, pq_path, sender, receiver):
    """Profile a single pair with detailed timing and memory tracking."""
    pair_dir = os.path.join(
        os.path.dirname(pq_path),
        f"{sanitize_celltype_name(sender)}__{sanitize_celltype_name(receiver)}")

    print(f"\n{'='*60}")
    print(f"PROFILING: {sender} -> {receiver}")
    print(f"Parquet: {pq_path}")
    print(f"Output:  {pair_dir}")
    print(f"{'='*60}\n")

    # Run scoring WITHOUT tracemalloc (it adds 10-30x overhead)
    result = process_one_pair(shared, pq_path, sender, receiver,
                              run_sensitivity=True)

    # Measure memory separately with a lightweight estimate
    import resource
    mem_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

    print(f"\n--- Timing breakdown ---")
    print(f"  Read CSV:        {result['time_read']:.2f}s")
    print(f"  Attr + edges:    {result['time_attr']:.4f}s")
    print(f"  Scoring:         {result['time_score']:.2f}s")
    print(f"  Rankings:        {result['time_rank']:.2f}s")
    print(f"  Sensitivity:     {result['time_sensitivity']:.2f}s")
    print(f"  Write outputs:   {result['time_write']:.2f}s")
    print(f"  TOTAL:           {result['time_sec']:.1f}s")

    print(f"\n--- Memory ---")
    print(f"  Process RSS:  {mem_mb:.0f} MB")

    print(f"\n--- Score distribution ---")
    print(f"  Pathways:  {result['n_pathways']}")
    print(f"  Nonzero:   {result['n_nonzero_score']} ({result['pct_nonzero']}%)")
    print(f"  Median:    {result['median_score']}")
    print(f"  Mean:      {result['mean_score']}")
    print(f"  Max:       {result['max_score']}")

    print(f"\n--- Extrapolation (462 pairs) ---")
    est_total = result['time_sec'] * 462
    print(f"  Estimated total: {est_total:.0f}s ({est_total/60:.0f} min)")

    # Validation: compare vectorized vs original iterrows scoring
    print(f"\n--- Vectorized vs original validation ---")
    import pyarrow.parquet as pq_mod
    _tbl = pq_mod.read_table(pq_path, columns=SCORING_COLS + ["sender"],
                              filters=[("sender", "=", sender)])
    pathways = _tbl.drop("sender").to_pandas()
    attr_weights = compute_pair_attr_weights(
        shared["attr_by_celltype"], sender, receiver,
        icfg.SENDER_ATTRIBUTION_DISCOUNT)

    t0 = time.monotonic()
    orig_scores = compute_scores(
        pathways, shared["sub_to_kins"], shared["idf_map"],
        shared["sig_kinases"], attr_weights, shared["mouse_to_abbrevs"])
    t_orig = time.monotonic() - t0

    new_scores = pd.read_csv(os.path.join(pair_dir, "kinase_support_scores.csv"))

    import numpy as np
    diff = np.abs(
        orig_scores["kinase_support_score"].values
        - new_scores["kinase_support_score"].values)
    print(f"  Original iterrows: {t_orig:.2f}s")
    print(f"  Vectorized:        {result['time_score']:.2f}s")
    print(f"  Speedup:           {t_orig / max(result['time_score'], 0.001):.1f}x")
    print(f"  Max score diff:    {diff.max():.2e}")
    print(f"  Exact match:       {np.allclose(diff, 0, atol=1e-10)}")

    # IDF refactor comparison: pair-independent vs pair-dependent
    print(f"\n--- IDF refactor validation ---")
    old_idf = _compute_idf_map(
        shared["sub_to_kins"], shared["mouse_to_abbrevs"],
        shared["sig_kinases"], attr_weights=attr_weights,
        pair_independent=False)
    old_scores = compute_scores(
        pathways, shared["sub_to_kins"], old_idf,
        shared["sig_kinases"], attr_weights, shared["mouse_to_abbrevs"])

    from scipy import stats
    for n in [20, 50, 100]:
        old_top = set(old_scores.nlargest(n, "kinase_support_score")["Path"])
        new_top = set(new_scores.nlargest(n, "kinase_support_score")["Path"])
        pct = len(old_top & new_top) / n * 100
        print(f"  Top-{n:>3d} overlap (old vs new IDF): {pct:.0f}%")

    merged = old_scores[["Path", "kinase_support_score"]].merge(
        new_scores[["Path", "kinase_support_score"]],
        on="Path", suffixes=("_old", "_new"))
    mask = (merged["kinase_support_score_old"] > 0) | (merged["kinase_support_score_new"] > 0)
    if mask.sum() > 10:
        rho, _ = stats.spearmanr(
            merged.loc[mask, "kinase_support_score_old"],
            merged.loc[mask, "kinase_support_score_new"])
        print(f"  Spearman rho (nonzero scores): {rho:.4f}")

    print(f"\nProfile complete. Outputs written to {pair_dir}/")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="All-pairs substrate-based kinase support scoring")
    parser.add_argument("--profile-pair", metavar="DIRNAME",
                        help="Profile a single pair (e.g. Microglia-PVM__L5_IT)")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing kinase_support_scores.csv")
    parser.add_argument("--no-sensitivity", action="store_true",
                        help="Skip per-pair sensitivity analyses")
    parser.add_argument("--pair-filter", metavar="PATTERN",
                        help="Filter pairs by glob (e.g. 'Astrocyte__*')")
    args = parser.parse_args()

    ensure_intermediates_dir()

    # 1. Load shared data
    print("Loading shared data...")
    shared = load_shared_data()

    # 2. Discover pairs
    pairs = discover_pairs(args.pair_filter, args.profile_pair)
    print(f"\nDiscovered {len(pairs)} pairs")

    if not pairs:
        print("No pairs to process.")
        return

    # 3. Profile mode
    if args.profile_pair:
        pq_path, sender, receiver = pairs[0]
        profile_single_pair(shared, pq_path, sender, receiver)
        return

    # 4. Process all pairs
    run_sensitivity = not args.no_sensitivity
    summaries = []
    n_skipped = 0
    n_total = len(pairs)

    for i, (pq_path, sender, receiver) in enumerate(pairs, 1):
        # Checkpoint: skip if output exists
        pair_dir = os.path.join(
            os.path.dirname(pq_path),
            f"{sanitize_celltype_name(sender)}__{sanitize_celltype_name(receiver)}")
        scores_path = os.path.join(pair_dir, "kinase_support_scores.csv")
        if os.path.exists(scores_path) and not args.force:
            n_skipped += 1
            continue

        result = process_one_pair(shared, pq_path, sender, receiver,
                                  run_sensitivity=run_sensitivity)
        summaries.append(result)

        print(f"  [{i}/{n_total}] {sender} -> {receiver}: "
              f"{result['n_pathways']} pathways, "
              f"{result['n_nonzero_score']} nonzero, "
              f"{result['time_sec']}s")

    if n_skipped:
        print(f"\nSkipped {n_skipped} pairs with existing output "
              f"(use --force to overwrite)")

    # 5. Write cross-pair summary
    if summaries:
        summary_df = pd.DataFrame(summaries)
        summary_path = os.path.join(icfg.ALL_PAIRS_DIR,
                                    "kinase_support_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        print(f"\nWrote {summary_path} ({len(summaries)} pairs)")

        # Quick aggregate stats
        print(f"\n--- Aggregate ---")
        print(f"  Pairs processed:   {len(summaries)}")
        print(f"  Total pathways:    {summary_df['n_pathways'].sum():,}")
        print(f"  Median nonzero %:  {summary_df['pct_nonzero'].median():.1f}%")
        print(f"  Total time:        {summary_df['time_sec'].sum():.0f}s "
              f"({summary_df['time_sec'].sum()/60:.0f} min)")
        print(f"  Median time/pair:  {summary_df['time_sec'].median():.1f}s")

    print("\nAll-pairs kinase support scoring complete.")


if __name__ == "__main__":
    main()

"""All-pairs substrate-based kinase support scoring.

Per pair: structural substrate evidence only (MEA-significant kinases whose
substrates land on the pathway's EM/Target nodes, weighted by IDF and
absolute MEA NES). No per-pair cell-type-attribution multiplier — see the
factorial sidecar docstring for the full rationale (taxonomy mismatch
between WMB-keyed ``unified_attribution.csv`` and SEA-AD-keyed Incytr pair
names, plus a Group-C concern about baking attribution into the support
score where it cannot be audited or re-joined cleanly downstream).

Restore the prior pair-weighted behavior via ``git show
legacy-incytr-storage-cutover:code/integration/adapters/compute_kinase_support_all_pairs.py``.

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
from normalization import resolve_paths, write_pathway_scores_for_pair
from compute_kinase_support import (
    _load_mea_kinases, _compute_idf_map,
    build_substrate_edge_table, edges_to_list_form, compute_scores_fast,
    compute_adjusted_rankings, run_sensitivity_analyses,
)
import config_integration as icfg


# ---------------------------------------------------------------------------
# Shared data loading
# ---------------------------------------------------------------------------

def load_shared_data():
    """Load all pair-independent data structures once. Builds a precomputed
    edge table (substrate -> edge arrays with |NES|*IDF) and converts it to
    the list form that ``compute_scores_fast`` expects, with no per-pair
    attribution weighting."""
    t0 = time.monotonic()

    kldata = pd.read_csv(os.path.join(icfg.INTERMEDIATES_DIR, "kldata.csv"))
    sub_to_kins = build_substrate_kinase_map(kldata)
    print(f"  kldata: {len(kldata)} edges, {len(sub_to_kins)} substrates")

    mouse_to_abbrevs = load_mouse_gene_to_kinase_mapping()
    print(f"  naming bridge: {sum(len(v) for v in mouse_to_abbrevs.values())} mappings")

    sig_kinases, all_mea_nes = _load_mea_kinases(
        icfg.CONTRAST, icfg.PHOSPHO_FDR_GATE)
    print(f"  MEA: {len(sig_kinases)} significant, {len(all_mea_nes)} total")

    idf_map = _compute_idf_map(sub_to_kins, mouse_to_abbrevs, sig_kinases,
                               pair_independent=True)
    print(f"  IDF: {len(idf_map)} substrates (pair-independent)")

    sub_raw_edges, all_kinase_genes = build_substrate_edge_table(
        sub_to_kins, idf_map, sig_kinases, mouse_to_abbrevs)
    print(f"  edge table: {len(sub_raw_edges)} substrates, "
          f"{len(all_kinase_genes)} kinase genes")

    sub_pair = edges_to_list_form(sub_raw_edges)

    elapsed = time.monotonic() - t0
    print(f"  shared data loaded in {elapsed:.1f}s")

    return {
        "sub_pair": sub_pair,
        "all_kinase_genes": all_kinase_genes,
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

    t2 = time.monotonic()
    scores_df = compute_scores_fast(
        pathways, shared["sub_pair"], shared["all_kinase_genes"])
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
    npaths = resolve_paths()
    if os.path.exists(os.path.join(npaths.universe_dir, "pathways.parquet")):
        pair_name = f"{sanitize_celltype_name(sender)}__{sanitize_celltype_name(receiver)}"
        write_pathway_scores_for_pair(
            scores_df,
            universe_dir=npaths.universe_dir,
            scoring_dir=npaths.scoring_dir,
            pair_name=pair_name,
            contrast_name=icfg.CONTRAST,
        )
    summary_path = os.path.join(dir_path, "reranking_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    t_write = time.monotonic() - t5

    total_time = time.monotonic() - t0
    n_nonzero = int((scores_df["mea_kinase_support_score"] > 0).sum())
    ks_vals = scores_df.loc[scores_df["mea_kinase_support_score"] > 0,
                            "mea_kinase_support_score"]

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

"""Cross-pair aggregation of all-pairs Incytr pipeline results.

Phase 3 of the receiver-centric refactor. Reads receiver-indexed Parquet
files (recv_*.parquet) via DuckDB and produces three aggregation views:

  3a. Backbone recurrence: R-EM-T triples shared across senders
  3b. Cell-type hub matrix: 22x22 sender x receiver signaling summary
  3c. Target gene convergence: genes targeted by multiple senders/routes

Optionally (--permutations), runs backbone-level dual null model permutation
tests on the substrate-based kinase support score.  This tests whether kinase
evidence concentrates in specific receiver signaling backbones beyond chance,
using receiver-only attribution weights (since backbone identity is receiver-
determined).

Usage:
  python aggregate_cross_pair.py
  python aggregate_cross_pair.py --permutations
  python aggregate_cross_pair.py --pds-threshold 0.15
  python aggregate_cross_pair.py --receiver Chandelier
  python aggregate_cross_pair.py --force
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone

# Ensure the integration directory is importable (config_integration lives there)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import duckdb
import numpy as np
import pandas as pd

import config_integration as icfg


def _build_filter(receiver_filter, pds_threshold):
    """Build WHERE clause and parameter lists for optional receiver filter.

    Returns (where_clause, pre_params, post_params) where pre_params go
    before threshold params in the SQL parameter binding order, and
    post_params go after.  DuckDB binds ``?`` marks in left-to-right
    textual order, and SELECT-clause ``?`` marks appear before
    FROM/WHERE ``?`` marks in the query text.
    """
    where = ""
    post_params = []
    if receiver_filter:
        where = "WHERE receiver = ?"
        post_params.append(receiver_filter)
    return where, pds_threshold, post_params


def create_connection(data_dir):
    """Create DuckDB connection and register all receiver Parquets as one view.

    The view normalizes column names: Receiver.group -> receiver, and
    projects only the columns needed for aggregation.
    """
    assert "'" not in data_dir, f"data_dir contains single quote: {data_dir}"

    con = duckdb.connect()
    parquet_glob = os.path.join(data_dir, "recv_*.parquet")
    con.execute(f"""
        CREATE VIEW recv_all AS
        SELECT sender,
               "Receiver.group" AS receiver,
               Ligand, Receptor, EM, Target, Path,
               pathway_evidence, imputed_nodes,
               TPDS, PDS, kinase_boost
        FROM read_parquet('{parquet_glob}')
    """)

    return con


def aggregate_backbone_recurrence(con, pds_threshold, receiver_filter=None):
    """3a. Backbone recurrence: R-EM-T triples shared across senders."""
    where, threshold, post_params = _build_filter(receiver_filter, pds_threshold)
    # SQL ? order: WHERE receiver (in CTE), then n_senders_significant threshold
    params = post_params + [threshold]

    sql = f"""
    WITH base AS (
        SELECT sender, receiver, Receptor, EM, Target, PDS, kinase_boost
        FROM recv_all {where}
    ),
    per_sender AS (
        SELECT sender, receiver, Receptor, EM, Target,
               AVG(PDS) AS sender_mean_pds,
               AVG(kinase_boost) AS sender_mean_kb
        FROM base
        GROUP BY sender, receiver, Receptor, EM, Target
    ),
    with_group_mean AS (
        SELECT *,
            AVG(sender_mean_pds) OVER (PARTITION BY receiver, Receptor, EM, Target)
                AS group_mean
        FROM per_sender
    )
    SELECT
        receiver, Receptor, EM, Target,
        COUNT(*) AS n_senders,
        COUNT(CASE WHEN ABS(sender_mean_pds) > ? THEN 1 END)
            AS n_senders_significant,
        ROUND(AVG(sender_mean_pds), 6) AS mean_pds,
        ROUND(MEDIAN(sender_mean_pds), 6) AS median_pds,
        ROUND(STDDEV_SAMP(sender_mean_pds), 6) AS std_pds,
        ROUND(MAX(ABS(sender_mean_pds)), 6) AS max_abs_pds,
        ROUND(AVG(sender_mean_kb), 6) AS mean_kinase_boost,
        ROUND(
            SUM(CASE WHEN SIGN(sender_mean_pds) = SIGN(group_mean) THEN 1 ELSE 0 END)
                ::FLOAT / COUNT(*),
            4) AS pds_direction_consistency,
        STRING_AGG(sender, ',' ORDER BY sender) AS sender_list
    FROM with_group_mean
    GROUP BY receiver, Receptor, EM, Target
    ORDER BY receiver, n_senders_significant DESC, max_abs_pds DESC
    """

    return con.execute(sql, params).fetchdf()


def aggregate_hub_matrix(con, pds_threshold, receiver_filter=None):
    """3b. Cell-type hub analysis: 22x22 sender x receiver matrix."""
    where, threshold, post_params = _build_filter(receiver_filter, pds_threshold)
    # SQL ? order: 4 threshold refs in SELECT, then WHERE receiver
    params = [threshold, threshold, threshold, -threshold] + post_params

    sql = f"""
    SELECT
        sender, receiver,
        COUNT(*) AS n_pathways,
        COUNT(CASE WHEN ABS(PDS) > ? THEN 1 END) AS n_significant,
        ROUND(COUNT(CASE WHEN ABS(PDS) > ? THEN 1 END)::FLOAT
              / COUNT(*), 4) AS pct_significant,
        ROUND(AVG(ABS(PDS)), 6) AS mean_abs_pds,
        ROUND(AVG(PDS), 6) AS mean_pds,
        ROUND(AVG(kinase_boost), 6) AS mean_kinase_boost,
        COUNT(CASE WHEN PDS > ? THEN 1 END) AS n_upregulated,
        COUNT(CASE WHEN PDS < ? THEN 1 END) AS n_downregulated
    FROM recv_all {where}
    GROUP BY sender, receiver
    ORDER BY sender, receiver
    """

    return con.execute(sql, params).fetchdf()


def pivot_hub_matrix(hub_df, metric="mean_abs_pds"):
    """Pivot 462-row hub matrix to 22x22 wide format for a given metric."""
    wide = hub_df.pivot(index="sender", columns="receiver", values=metric)
    wide = wide.fillna(0)
    wide = wide.sort_index(axis=0).sort_index(axis=1)
    return wide


def aggregate_target_convergence(con, pds_threshold, receiver_filter=None):
    """3c. Gene-level convergence per receiver."""
    where, threshold, post_params = _build_filter(receiver_filter, pds_threshold)
    # SQL ? order: WHERE receiver (in CTE), then n_senders_significant threshold
    params = post_params + [threshold]

    sql = f"""
    WITH base AS (
        SELECT sender, receiver, Target, Receptor, EM, PDS, kinase_boost
        FROM recv_all {where}
    ),
    per_sender AS (
        SELECT receiver, Target, sender,
               AVG(PDS) AS sender_mean_pds
        FROM base
        GROUP BY receiver, Target, sender
    ),
    route_counts AS (
        SELECT receiver, Target,
               COUNT(DISTINCT Receptor || '::' || EM) AS n_routes,
               COUNT(*) AS n_pathways,
               ROUND(AVG(kinase_boost), 6) AS mean_kinase_boost
        FROM base
        GROUP BY receiver, Target
    )
    SELECT
        ps.receiver, ps.Target,
        COUNT(DISTINCT ps.sender) AS n_senders,
        COUNT(DISTINCT CASE WHEN ABS(ps.sender_mean_pds) > ?
              THEN ps.sender END) AS n_senders_significant,
        rc.n_routes,
        rc.n_pathways,
        ROUND(AVG(ps.sender_mean_pds), 6) AS mean_pds,
        ROUND(MEDIAN(ps.sender_mean_pds), 6) AS median_pds,
        rc.mean_kinase_boost,
        ARG_MAX(ps.sender, ABS(ps.sender_mean_pds)) AS top_sender
    FROM per_sender ps
    JOIN route_counts rc ON ps.receiver = rc.receiver AND ps.Target = rc.Target
    GROUP BY ps.receiver, ps.Target, rc.n_routes, rc.n_pathways,
             rc.mean_kinase_boost
    ORDER BY ps.receiver, n_senders_significant DESC, rc.n_routes DESC
    """

    return con.execute(sql, params).fetchdf()


# ---------------------------------------------------------------------------
# Backbone-level permutation tests
# ---------------------------------------------------------------------------

def _build_backbone_edges(backbone_df, sub_raw_edges, attr_by_celltype):
    """Build per-backbone edge arrays using receiver-only attribution weights.

    For each unique (receiver, EM, Target), looks up kinase-substrate edges
    in the shared edge table and applies the receiver's attribution weights.
    Backbones sharing the same (receiver, EM, Target) get identical edges,
    so we deduplicate at that level.

    Returns
    -------
    bb_edge_weights : list[np.ndarray]
        Per-backbone weighted edge values (|NES| * IDF * attr_weight).
    bb_idf_coeffs : list[np.ndarray]
        Per-backbone IDF coefficients (for null models that keep IDF fixed).
    kin_list : list[str]
        Ordered list of all kinase abbreviations that appear in any edge.
    """
    receivers = backbone_df["receiver"].values
    em_arr = backbone_df["EM"].values
    tg_arr = backbone_df["Target"].values
    n_bb = len(backbone_df)

    # Build receiver-only attribution weights per receiver
    recv_weights = {}
    for recv in backbone_df["receiver"].unique():
        recv_weights[recv] = attr_by_celltype.get(recv, {})

    # Cache per (receiver, EM, Target) to avoid recomputing
    cache = {}

    bb_edge_weights = []
    bb_idf_coeffs = []

    for i in range(n_bb):
        recv = receivers[i]
        em = em_arr[i]
        tg = tg_arr[i]
        key = (recv, em, tg)

        if key in cache:
            ew, idf_c = cache[key]
            bb_edge_weights.append(ew)
            bb_idf_coeffs.append(idf_c)
            continue

        weights = recv_weights[recv]
        edge_ws = []
        idf_cs = []
        seen = set()

        for sub_gene in (em, tg):
            raw = sub_raw_edges.get(sub_gene)
            if raw is None:
                continue
            for j, abbrev in enumerate(raw["abbrevs"]):
                aw = weights.get(abbrev, 0.0)
                if aw <= 0:
                    continue
                edge_key = (abbrev, sub_gene)
                if edge_key in seen:
                    continue
                seen.add(edge_key)

                edge_ws.append(raw["abs_nes_idf"][j] * aw)
                # IDF = abs_nes_idf / abs_nes (the IDF multiplier)
                idf_cs.append(raw["abs_nes_idf"][j] / raw["abs_nes"][j]
                              if raw["abs_nes"][j] > 0 else 1.0)

        ew = np.array(edge_ws, dtype=np.float64) if edge_ws else np.empty(0)
        idf_c = np.array(idf_cs, dtype=np.float64) if idf_cs else np.empty(0)

        cache[key] = (ew, idf_c)
        bb_edge_weights.append(ew)
        bb_idf_coeffs.append(idf_c)

    print(f"    {len(cache):,} unique (receiver, EM, Target) edge structures "
          f"(of {n_bb:,} backbones)")

    return bb_edge_weights, bb_idf_coeffs


def run_backbone_permutations(backbone_df, shared, n_permutations):
    """Dual permutation null models for backbone-level kinase support.

    Operates on unique (receiver, EM, Target) backbones from the cross-pair
    aggregation.  Uses receiver-only attribution weights (backbone identity
    is receiver-determined; sender variation is not tested here).

    Null 1 (enrichment null): For each backbone with N edges, sample N
      kinases from the full MEA universe. Tests whether the backbone's
      score reflects concentration of disease-significant, receiver-
      attributed kinases.

    Null 2 (wiring null): Reassign each backbone's edges to random kinases
      from the full MEA universe, keeping IDF coefficients fixed. Tests
      whether specific kinase-substrate wiring matters.
    """
    from statsmodels.stats.multitest import multipletests

    print(f"\n=== Backbone-Level Permutation Tests ({n_permutations:,} iterations) ===")

    # Background kinase pool (full MEA universe)
    all_mea_nes = shared["all_mea_nes"]
    bg_kinases = sorted(all_mea_nes.keys())
    n_bg = len(bg_kinases)
    bg_nes_vec = np.array([all_mea_nes[k] for k in bg_kinases])
    print(f"  Background kinase pool: {n_bg} (full MEA universe)")

    # Build per-backbone edge arrays
    print("  Building backbone edge structures...")
    bb_edge_weights, bb_idf_coeffs = _build_backbone_edges(
        backbone_df, shared["sub_raw_edges"], shared["attr_by_celltype"])

    # Compute observed scores (median of edge weights per backbone)
    n_bb = len(backbone_df)
    observed = np.zeros(n_bb)
    for i in range(n_bb):
        ew = bb_edge_weights[i]
        if len(ew) > 0:
            observed[i] = np.median(ew)

    # Identify active backbones (those with at least one edge)
    has_edges = np.array([len(ew) > 0 for ew in bb_edge_weights])
    active_idx = np.where(has_edges)[0]
    n_active = len(active_idx)
    print(f"  {n_active:,} active backbones with edges (of {n_bb:,} total)")

    if n_active == 0:
        print("  No active backbones — skipping permutation tests")
        return _empty_permutation_result(backbone_df)

    # Collect attribution weights across all receivers for Null 2 sampling
    all_attr_weights = []
    for recv_dict in shared["attr_by_celltype"].values():
        all_attr_weights.extend(recv_dict.values())
    attr_weight_arr = np.array(all_attr_weights, dtype=np.float64)
    median_attr_weight = float(np.median(attr_weight_arr))
    print(f"  Uniform attribution weight (Null 1): {median_attr_weight:.4f}")
    print(f"  Attribution weight pool (Null 2): {len(attr_weight_arr)} values")

    # Degree-bucketed vectorization
    pw_degrees = np.array([len(bb_edge_weights[i]) for i in active_idx])
    pw_idf = [bb_idf_coeffs[i] for i in active_idx]
    obs_active = observed[active_idx]

    degree_groups = defaultdict(list)
    for k in range(n_active):
        degree_groups[pw_degrees[k]].append((k, pw_idf[k]))

    degree_group_data = {}
    for d, members in degree_groups.items():
        local_indices = np.array([m[0] for m in members], dtype=np.intp)
        idf_arrays = np.array([m[1] for m in members])  # (n_members, d)
        obs_slice = obs_active[local_indices]
        degree_group_data[d] = (local_indices, idf_arrays, obs_slice)

    n_degree_groups = len(degree_group_data)
    print(f"  {n_degree_groups} degree groups "
          f"(range {min(degree_groups)}-{max(degree_groups)})")

    batch_size = min(500, n_permutations)
    rng = np.random.default_rng(42)

    def _run_null(label, make_aw):
        print(f"  {label}...")
        ge = np.zeros(n_active)
        n_done = 0
        while n_done < n_permutations:
            b = min(batch_size, n_permutations - n_done)
            for d, (idx, idf_arr, obs_sl) in degree_group_data.items():
                n_members = len(idx)
                drawn = rng.integers(0, n_bg, size=(b, n_members, d))
                nes_drawn = bg_nes_vec[drawn]
                aw = make_aw(b, n_members, d)
                ews = idf_arr[np.newaxis, :, :] * nes_drawn * aw
                perm_medians = np.median(ews, axis=2)
                ge[idx] += (perm_medians >= obs_sl[np.newaxis, :]).sum(axis=0)
            n_done += b
            if n_done % 2000 == 0 or n_done == n_permutations:
                print(f"    {n_done}/{n_permutations}")
        return ge

    null1_ge = _run_null(
        "Null 1 (enrichment null)",
        lambda b, nm, d: median_attr_weight,  # scalar broadcasts
    )
    null2_ge = _run_null(
        "Null 2 (wiring null)",
        lambda b, nm, d: attr_weight_arr[
            rng.integers(0, len(attr_weight_arr), size=(b, nm, d))],
    )

    # Assemble p-values
    pval_null1 = np.ones(n_bb)
    pval_null2 = np.ones(n_bb)
    pval_null1[active_idx] = (null1_ge + 1) / (n_permutations + 1)
    pval_null2[active_idx] = (null2_ge + 1) / (n_permutations + 1)

    _, fdr_null1, _, _ = multipletests(pval_null1, method="fdr_bh")
    _, fdr_null2, _, _ = multipletests(pval_null2, method="fdr_bh")

    fdr_gate = icfg.PHOSPHO_FDR_GATE

    results = pd.DataFrame({
        "receiver": backbone_df["receiver"].values,
        "Receptor": backbone_df["Receptor"].values,
        "EM": backbone_df["EM"].values,
        "Target": backbone_df["Target"].values,
        "observed_score": np.round(observed, 6),
        "n_edges": np.array([len(bb_edge_weights[i]) for i in range(n_bb)]),
        "pval_null1": pval_null1,
        "pval_null2": pval_null2,
        "fdr_null1": fdr_null1,
        "fdr_null2": fdr_null2,
        "significant_null1": fdr_null1 < fdr_gate,
        "significant_null2": fdr_null2 < fdr_gate,
        "significant_both": (fdr_null1 < fdr_gate) & (fdr_null2 < fdr_gate),
    })

    n_sig1 = results["significant_null1"].sum()
    n_sig2 = results["significant_null2"].sum()
    n_both = results["significant_both"].sum()
    print(f"\n  Null 1 significant (FDR<{fdr_gate}): {n_sig1:,}")
    print(f"  Null 2 significant (FDR<{fdr_gate}): {n_sig2:,}")
    print(f"  Significant under both: {n_both:,}")

    return results


def _empty_permutation_result(backbone_df):
    """Return an empty-valued permutation result for backbones with no edges."""
    n = len(backbone_df)
    return pd.DataFrame({
        "receiver": backbone_df["receiver"].values,
        "Receptor": backbone_df["Receptor"].values,
        "EM": backbone_df["EM"].values,
        "Target": backbone_df["Target"].values,
        "observed_score": np.zeros(n),
        "n_edges": np.zeros(n, dtype=int),
        "pval_null1": np.ones(n),
        "pval_null2": np.ones(n),
        "fdr_null1": np.ones(n),
        "fdr_null2": np.ones(n),
        "significant_null1": np.zeros(n, dtype=bool),
        "significant_null2": np.zeros(n, dtype=bool),
        "significant_both": np.zeros(n, dtype=bool),
    })


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def write_outputs(backbone_df, hub_df, hub_wide, target_df,
                  output_dir, metadata):
    """Write all aggregation outputs."""
    os.makedirs(output_dir, exist_ok=True)

    backbone_df.to_csv(os.path.join(output_dir, "backbone_recurrence.csv"),
                       index=False)
    hub_df.to_csv(os.path.join(output_dir, "hub_matrix.csv"),
                  index=False)
    hub_wide.to_csv(os.path.join(output_dir, "hub_matrix_wide.csv"))
    target_df.to_csv(os.path.join(output_dir, "target_convergence.csv"),
                     index=False)

    meta_path = os.path.join(output_dir, "aggregation_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Cross-pair aggregation of all-pairs Incytr results")
    parser.add_argument("--pds-threshold", type=float,
                        default=icfg.PDS_SIGNIFICANCE_THRESHOLD,
                        help="PDS threshold for significance (default: %(default)s)")
    parser.add_argument("--receiver", metavar="NAME",
                        help="Aggregate only this receiver")
    parser.add_argument("--permutations", action="store_true",
                        help="Run backbone-level permutation null models")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing aggregation outputs")
    args = parser.parse_args()

    output_dir = icfg.AGGREGATION_DIR
    perm_path = os.path.join(output_dir, "backbone_permutation_pvalues.csv")

    # Check for existing outputs (aggregation and permutations separately)
    agg_exists = os.path.exists(
        os.path.join(output_dir, "aggregation_metadata.json"))
    perm_exists = os.path.exists(perm_path)

    if agg_exists and not args.force and not args.permutations:
        print("Aggregation outputs already exist. Use --force to overwrite.")
        return

    if args.permutations and perm_exists and not args.force:
        print("Permutation outputs already exist. Use --force to overwrite.")
        return

    print("=== Cross-Pair Aggregation ===")
    print(f"  PDS threshold: {args.pds_threshold}")
    if args.receiver:
        print(f"  Receiver filter: {args.receiver}")

    t0 = time.monotonic()

    # Run aggregation (or reload if it exists and we're only doing perms)
    if agg_exists and not args.force and args.permutations:
        # Aggregation already done -- just load backbone_recurrence for perms
        print("\nAggregation outputs exist; loading for permutation tests...")
        backbone_df = pd.read_csv(
            os.path.join(output_dir, "backbone_recurrence.csv"))
        print(f"  {len(backbone_df):,} backbones loaded")
    else:
        print("\nLoading Parquet data...")
        con = create_connection(icfg.ALL_PAIRS_DIR)

        try:
            # Hub matrix first -- derives data_stats without extra scans
            print("\n3b. Cell-type hub matrix...")
            t2 = time.monotonic()
            hub_df = aggregate_hub_matrix(con, args.pds_threshold, args.receiver)
            hub_wide = pivot_hub_matrix(hub_df)
            print(f"  {len(hub_df)} pairs, {hub_wide.shape[0]}x{hub_wide.shape[1]} "
                  f"matrix ({time.monotonic() - t2:.1f}s)")

            total_pathways = int(hub_df["n_pathways"].sum())
            n_pairs = len(hub_df)
            n_receivers = int(hub_df["receiver"].nunique())
            print(f"  {total_pathways:,} pathways across {n_pairs} pairs "
                  f"({n_receivers} receivers)")

            print("\n3a. Backbone recurrence...")
            t1 = time.monotonic()
            backbone_df = aggregate_backbone_recurrence(
                con, args.pds_threshold, args.receiver)
            print(f"  {len(backbone_df):,} backbones "
                  f"({time.monotonic() - t1:.1f}s)")

            print("\n3c. Target gene convergence...")
            t3 = time.monotonic()
            target_df = aggregate_target_convergence(
                con, args.pds_threshold, args.receiver)
            print(f"  {len(target_df):,} target-receiver pairs "
                  f"({time.monotonic() - t3:.1f}s)")
        finally:
            con.close()

        total_time = time.monotonic() - t0
        metadata = {
            "pds_threshold": args.pds_threshold,
            "receiver_filter": args.receiver,
            "n_backbones": len(backbone_df),
            "n_hub_pairs": n_pairs,
            "n_target_genes": len(target_df),
            "total_pathways": total_pathways,
            "n_receivers": n_receivers,
            "n_pairs": n_pairs,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_time_sec": round(total_time, 1),
        }

        print(f"\nWriting outputs to {output_dir}/")
        write_outputs(backbone_df, hub_df, hub_wide, target_df,
                      output_dir, metadata)

        print(f"\nAggregation complete ({total_time:.1f}s)")
        print(f"  Backbones:       {len(backbone_df):,}")
        print(f"  Hub matrix:      {hub_wide.shape[0]}x{hub_wide.shape[1]}")
        print(f"  Target genes:    {len(target_df):,}")

    # Run permutation tests if requested
    if args.permutations:
        from compute_kinase_support_all_pairs import load_shared_data

        print("\nLoading shared kinase data for permutation tests...")
        shared = load_shared_data()

        perm_df = run_backbone_permutations(
            backbone_df, shared, icfg.N_PERMUTATIONS_AGGREGATE)

        perm_df.to_csv(perm_path, index=False)
        print(f"\n  Wrote {perm_path}")
        print(f"  Permutation tests complete "
              f"({time.monotonic() - t0:.1f}s total)")


if __name__ == "__main__":
    main()

"""Cross-pair aggregation of all-pairs Incytr pipeline results.

Phase 3 of the receiver-centric refactor. Reads receiver-indexed Parquet
files (recv_*.parquet) via DuckDB and produces three aggregation views:

  3a. Backbone recurrence: R-EM-T triples shared across senders
  3b. Cell-type hub matrix: 22x22 sender x receiver signaling summary
  3c. Target gene convergence: genes targeted by multiple senders/routes

Usage:
  python aggregate_cross_pair.py
  python aggregate_cross_pair.py --pds-threshold 0.15
  python aggregate_cross_pair.py --receiver Chandelier
  python aggregate_cross_pair.py --force
"""

import argparse
import json
import os
import time
from datetime import datetime, timezone

import duckdb

import config_integration as icfg


def _build_filter(receiver_filter, pds_threshold):
    """Build WHERE clause and parameter list for optional receiver filter."""
    where = ""
    params = []
    if receiver_filter:
        where = "WHERE receiver = ?"
        params.append(receiver_filter)
    params.append(pds_threshold)
    return where, params


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
    where, params = _build_filter(receiver_filter, pds_threshold)

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
    where, params = _build_filter(receiver_filter, pds_threshold)
    # Need threshold 3 more times for n_significant, pct, n_up, n_down
    params.extend([params[-1]] * 2 + [-params[-1]])

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
    where, params = _build_filter(receiver_filter, pds_threshold)

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
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing aggregation outputs")
    args = parser.parse_args()

    output_dir = icfg.AGGREGATION_DIR
    if os.path.exists(os.path.join(output_dir, "aggregation_metadata.json")) \
            and not args.force:
        print("Aggregation outputs already exist. Use --force to overwrite.")
        return

    print("=== Cross-Pair Aggregation ===")
    print(f"  PDS threshold: {args.pds_threshold}")
    if args.receiver:
        print(f"  Receiver filter: {args.receiver}")

    t0 = time.monotonic()

    print("\nLoading Parquet data...")
    con = create_connection(icfg.ALL_PAIRS_DIR)

    try:
        # Hub matrix first — derives data_stats without extra scans
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


if __name__ == "__main__":
    main()

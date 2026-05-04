"""Factorial cross-pair aggregation of Incytr pipeline results.

Reads receiver-indexed Parquet files and per-pair kinase support CSVs via
DuckDB. Produces per-contrast aggregation tables and diagnostic plots.

Processes one contrast at a time (per-contrast iteration) to stay within
memory limits on 30GB systems. Each query scans the wide-format Parquet
directly without unnesting, avoiding the 9x row expansion that OOMs.

Optionally (--permutations), runs backbone-level dual null model permutation
tests on the substrate-based kinase support score, independently per contrast.
Uses the same enrichment/wiring null design as aggregate_cross_pair.py but
with per-contrast MEA background pools and attribution weights.

Usage:
  python aggregate_factorial.py
  python aggregate_factorial.py --pvalue-threshold 0.01
  python aggregate_factorial.py --permutations
  python aggregate_factorial.py --permutations --n-permutations 1000
  python aggregate_factorial.py --force
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from glob import glob

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import duckdb
import numpy as np
import pandas as pd

import config_integration as icfg

CONTRASTS = list(icfg.FACTORIAL_CONTRASTS.keys())
GENOTYPES = ["App", "Tau", "ApTt"]
TIMEPOINTS = ["2mo", "4mo", "6mo"]


# ---------------------------------------------------------------------------
# DuckDB setup
# ---------------------------------------------------------------------------

def _build_receiver_lookup(data_dir):
    """Build filename -> receiver name lookup from Parquet file metadata."""
    import glob as globmod
    import pyarrow.parquet as pq

    lookup = {}
    for f in sorted(globmod.glob(os.path.join(data_dir, "recv_*.parquet"))):
        meta = pq.read_metadata(f).metadata or {}
        recv = meta.get(b"receiver", b"").decode()
        if recv:
            lookup[f] = recv
    return lookup


def _resolve_duckdb_settings(memory_limit_gb=None, duckdb_threads=None):
    """Resolve DuckDB resource settings from CLI/env with safe defaults."""
    if memory_limit_gb is None:
        memory_limit_gb = float(os.getenv("MEMORY_LIMIT_GB", "6"))
    if duckdb_threads is None:
        duckdb_threads = int(os.getenv("DUCKDB_THREADS", "4"))
    return {
        "memory_limit_gb": memory_limit_gb,
        "duckdb_threads": max(1, duckdb_threads),
        "preserve_insertion_order": False,
    }


def create_connection(data_dir, *, memory_limit_gb=None, duckdb_threads=None):
    """Create DuckDB connection and register Parquet + kinase CSV views."""
    assert "'" not in data_dir

    settings = _resolve_duckdb_settings(memory_limit_gb, duckdb_threads)
    con = duckdb.connect()
    con.execute(f"SET memory_limit='{settings['memory_limit_gb']}GB'")
    con.execute(f"SET threads={settings['duckdb_threads']}")
    con.execute("SET preserve_insertion_order=false")

    # Build receiver lookup from Parquet metadata
    recv_lookup = _build_receiver_lookup(data_dir)
    recv_df = pd.DataFrame([
        {"filename": k, "receiver": v} for k, v in recv_lookup.items()
    ])
    con.register("recv_lookup", recv_df)

    # Register Parquet view with receiver name from lookup
    parquet_glob = os.path.join(data_dir, "recv_*.parquet")
    con.execute(f"""
        CREATE VIEW recv_all AS
        SELECT p.*, rl.receiver
        FROM read_parquet('{parquet_glob}', filename=true) p
        JOIN recv_lookup rl ON p.filename = rl.filename
    """)

    # Register kinase support CSV view with sender/receiver from filepath
    csv_glob = os.path.join(data_dir, "*", "kinase_support_scores.csv")
    con.execute(f"""
        CREATE VIEW kinase_all AS
        SELECT *,
            regexp_extract(filename, '([^/]+)__([^/]+)/kinase', 1) AS sender,
            regexp_extract(filename, '([^/]+)__([^/]+)/kinase', 2) AS receiver
        FROM read_csv_auto('{csv_glob}', filename=true)
    """)

    return con, settings


def build_backbone_provenance_table(data_dir, out_dir):
    """Build static backbone provenance once from receiver parquet files.

    Provenance is contrast-invariant in the current receiver parquet schema, so
    this helper computes one compact row per (receiver, Receptor, EM, Target)
    and persists it for Q4 reuse and debugging.
    """
    recv_lookup = _build_receiver_lookup(data_dir)
    parts = []
    columns = ["Receptor", "EM", "Target", "pathway_evidence", "imputed_nodes"]

    for path in sorted(glob(os.path.join(data_dir, "recv_*.parquet"))):
        receiver = recv_lookup.get(path)
        if not receiver:
            continue
        df = pd.read_parquet(path, columns=columns)
        if df.empty:
            continue
        df["receiver"] = receiver
        nodes = df["imputed_nodes"].fillna("").astype(str)
        df["imp_receptor"] = nodes.str.contains("Receptor", regex=False)
        df["imp_em"] = nodes.str.contains("EM", regex=False)
        df["imp_target"] = nodes.str.contains("Target", regex=False)
        # Older factorial receiver parquets wrote every pathway_evidence value
        # as expression-confirmed while still carrying correct imputed_nodes.
        # Treat imputed_nodes as the source of truth for provenance.
        is_kin_imp = nodes.ne("")
        df["is_expr"] = (~is_kin_imp).astype(np.int32)
        df["is_kin_imp"] = is_kin_imp.astype(np.int32)

        grouped = (
            df.groupby(["receiver", "Receptor", "EM", "Target"], sort=False)
              .agg(
                  n_expression_confirmed=("is_expr", "sum"),
                  n_kinase_imputed=("is_kin_imp", "sum"),
                  imp_receptor=("imp_receptor", "max"),
                  imp_em=("imp_em", "max"),
                  imp_target=("imp_target", "max"),
              )
              .reset_index()
        )
        parts.append(grouped)

    if parts:
        provenance = pd.concat(parts, ignore_index=True)
    else:
        provenance = pd.DataFrame(columns=[
            "receiver", "Receptor", "EM", "Target",
            "n_expression_confirmed", "n_kinase_imputed",
            "imp_receptor", "imp_em", "imp_target",
        ])

    provenance["pathway_evidence_backbone"] = np.select(
        [
            (provenance["n_expression_confirmed"] > 0)
            & (provenance["n_kinase_imputed"] > 0),
            provenance["n_kinase_imputed"] > 0,
        ],
        ["mixed", "kinase-imputed"],
        default="expression-confirmed",
    )

    def _join_nodes(row):
        out = []
        if bool(row["imp_receptor"]):
            out.append("Receptor")
        if bool(row["imp_em"]):
            out.append("EM")
        if bool(row["imp_target"]):
            out.append("Target")
        return ";".join(out)

    provenance["imputed_nodes_union"] = provenance.apply(_join_nodes, axis=1)
    provenance = provenance[[
        "receiver", "Receptor", "EM", "Target",
        "pathway_evidence_backbone",
        "n_expression_confirmed", "n_kinase_imputed",
        "imputed_nodes_union",
    ]]

    prov_path = os.path.join(out_dir, "backbone_provenance.csv")
    provenance.to_csv(prov_path, index=False)
    return provenance, prov_path


def _count_csv_rows(path):
    """Return row count excluding header for a CSV file."""
    with open(path, "r", encoding="utf-8") as f:
        return max(sum(1 for _ in f) - 1, 0)


# ---------------------------------------------------------------------------
# Aggregation queries
# ---------------------------------------------------------------------------

def q1_hub_matrix(con, pval_threshold):
    """Q1: 22x22x9 hub matrix — n_pathways, n_significant, mean_abs_tpds.

    Processes one contrast at a time to avoid OOM on the 366M-row unnest.
    """
    parts = []
    for contrast in CONTRASTS:
        sql = f"""
        SELECT '{contrast}' AS contrast, sender, receiver,
               COUNT(*) AS n_pathways,
               COUNT(*) FILTER (pvalue_{contrast} < ?) AS n_significant,
               ROUND(AVG(ABS(TPDS_{contrast})), 6) AS mean_abs_tpds,
               ROUND(AVG(TPDS_{contrast}), 6) AS mean_tpds,
               ROUND(MEDIAN(TPDS_{contrast}), 6) AS median_tpds
        FROM recv_all
        GROUP BY sender, receiver
        ORDER BY sender, receiver
        """
        df = con.execute(sql, [pval_threshold]).fetchdf()
        parts.append(df)
        print(f"    {contrast}: {len(df)} pairs")

    return pd.concat(parts, ignore_index=True)


def q3_temporal_dynamics(con, pval_threshold):
    """Q3: Temporal dynamics — significance by genotype x timepoint per pair.

    Processes one contrast at a time to avoid OOM on the 366M-row unnest.
    """
    parts = []
    for contrast in CONTRASTS:
        geno, tp = contrast.split("_", 1)
        sql = f"""
        SELECT sender, receiver,
               '{geno}' AS genotype,
               '{tp}' AS timepoint,
               '{contrast}' AS contrast,
               COUNT(*) AS n_pathways,
               COUNT(*) FILTER (pvalue_{contrast} < ?) AS n_significant,
               ROUND(100.0 * COUNT(*) FILTER (pvalue_{contrast} < ?) / COUNT(*), 2) AS pct_significant,
               ROUND(AVG(ABS(TPDS_{contrast})), 6) AS mean_abs_tpds
        FROM recv_all
        GROUP BY sender, receiver
        ORDER BY sender, receiver
        """
        df = con.execute(sql, [pval_threshold, pval_threshold]).fetchdf()
        parts.append(df)
        print(f"    {contrast}: {len(df)} pairs")

    return pd.concat(parts, ignore_index=True)


def q4_backbone_recurrence(con, pval_threshold, out_path):
    """Q4: Backbone recurrence — R-EM-T triples significant across senders.

    Processes one contrast at a time to avoid OOM on the 366M-row unnest.
    """
    total_rows = 0
    wrote_header = False
    for contrast in CONTRASTS:
        sql = f"""
        WITH per_sender AS (
            SELECT sender, receiver, Receptor, EM, Target,
                   AVG(TPDS_{contrast}) AS sender_mean_tpds,
                   MIN(pvalue_{contrast}) AS sender_min_pval
            FROM recv_all
            GROUP BY sender, receiver, Receptor, EM, Target
        )
        SELECT '{contrast}' AS contrast,
               ps.receiver, ps.Receptor, ps.EM, ps.Target,
               COUNT(DISTINCT ps.sender) AS n_senders,
               COUNT(DISTINCT ps.sender) FILTER (ps.sender_min_pval < ?) AS n_senders_significant,
               ROUND(AVG(ps.sender_mean_tpds), 6) AS mean_tpds,
               ROUND(MAX(ABS(ps.sender_mean_tpds)), 6) AS max_abs_tpds,
               MIN(ps.sender_min_pval) AS tpds_pvalue,
               STRING_AGG(DISTINCT ps.sender, ',' ORDER BY ps.sender) AS sender_list,
               p.pathway_evidence_backbone,
               p.n_expression_confirmed,
               p.n_kinase_imputed,
               p.imputed_nodes_union
        FROM per_sender ps
        JOIN backbone_provenance p
          ON ps.receiver = p.receiver
         AND ps.Receptor = p.Receptor
         AND ps.EM = p.EM
         AND ps.Target = p.Target
        GROUP BY ps.receiver, ps.Receptor, ps.EM, ps.Target,
                 p.pathway_evidence_backbone, p.n_expression_confirmed,
                 p.n_kinase_imputed, p.imputed_nodes_union
        HAVING COUNT(DISTINCT ps.sender) FILTER (ps.sender_min_pval < ?) >= 2
        ORDER BY n_senders_significant DESC, max_abs_tpds DESC
        """
        df = con.execute(sql, [pval_threshold, pval_threshold]).fetchdf()
        df.to_csv(out_path, mode="a" if wrote_header else "w",
                  header=not wrote_header, index=False)
        wrote_header = True
        total_rows += len(df)
        print(f"    {contrast}: {len(df)} backbones")
    return total_rows


def q5_kinase_integration(con, pval_threshold):
    """Q5: Per-contrast kinase support + TPDS integration summary.

    Processes one contrast at a time to avoid OOM on the full unnest join.
    """
    parts = []
    for contrast in CONTRASTS:
        sql = f"""
        WITH kinase_norm AS (
            SELECT Path,
                   replace(replace(sender, '_', ' '), '-', '/') AS sender,
                   replace(replace(receiver, '_', ' '), '-', '/') AS receiver,
                   kinase_support_score_{contrast} AS kss,
                   n_distinct_kinases_{contrast} AS n_kinases,
                   concordance_flag_{contrast} AS concordance
            FROM kinase_all
        ),
        joined AS (
            SELECT r.sender, r.receiver, r.Path,
                   r.TPDS_{contrast} AS TPDS,
                   r.pvalue_{contrast} AS pvalue,
                   k.kss, k.n_kinases, k.concordance
            FROM recv_all r
            JOIN kinase_norm k
              ON r.sender = k.sender
             AND r.receiver = k.receiver
             AND r.Path = k.Path
        )
        SELECT '{contrast}' AS contrast,
               COUNT(*) AS n_total,
               COUNT(*) FILTER (pvalue < ?) AS n_sig_tpds,
               COUNT(*) FILTER (kss > 0) AS n_has_kinase,
               COUNT(*) FILTER (pvalue < ? AND kss > 0) AS n_sig_with_kinase,
               ROUND(AVG(kss), 6) AS mean_kss_all,
               ROUND(AVG(kss) FILTER (pvalue < ?), 6) AS mean_kss_significant,
               ROUND(AVG(n_kinases), 2) AS mean_n_kinases,
               COUNT(*) FILTER (pvalue < ? AND concordance = 'concordant') AS n_concordant,
               COUNT(*) FILTER (pvalue < ? AND concordance = 'discordant') AS n_discordant,
               COUNT(*) FILTER (pvalue < ? AND concordance = 'mixed') AS n_mixed
        FROM joined
        """
        df = con.execute(sql, [pval_threshold] * 6).fetchdf()
        parts.append(df)
        row = df.iloc[0]
        print(f"    {contrast}: {int(row['n_sig_with_kinase'])} sig+kinase "
              f"/ {int(row['n_sig_tpds'])} sig")

    return pd.concat(parts, ignore_index=True)


def q6_target_convergence(con, pval_threshold):
    """Q6: Target genes hit by multiple senders with significant effects.

    Processes one contrast at a time to avoid OOM.
    """
    parts = []
    for contrast in CONTRASTS:
        sql = f"""
        WITH per_sender_target AS (
            SELECT receiver, Target, sender,
                   AVG(TPDS_{contrast}) AS sender_mean_tpds,
                   MIN(pvalue_{contrast}) AS sender_min_pval
            FROM recv_all
            GROUP BY receiver, Target, sender
        )
        SELECT '{contrast}' AS contrast,
               receiver, Target,
               COUNT(DISTINCT sender) AS n_senders,
               COUNT(DISTINCT sender) FILTER (sender_min_pval < ?) AS n_senders_significant,
               ROUND(AVG(sender_mean_tpds), 6) AS mean_tpds,
               STRING_AGG(DISTINCT sender, ',' ORDER BY sender)
                   FILTER (sender_min_pval < ?) AS significant_senders
        FROM per_sender_target
        GROUP BY receiver, Target
        HAVING COUNT(DISTINCT sender) FILTER (sender_min_pval < ?) >= 3
        ORDER BY n_senders_significant DESC
        """
        df = con.execute(sql, [pval_threshold] * 3).fetchdf()
        parts.append(df)
        print(f"    {contrast}: {len(df)} targets")

    return pd.concat(parts, ignore_index=True)


# ---------------------------------------------------------------------------
# Backbone-level permutation tests (per-contrast)
# ---------------------------------------------------------------------------

def _build_backbone_edges(backbone_df, sub_raw_edges, attr_by_celltype):
    """Build per-backbone edge arrays using receiver-only attribution weights.

    Adapted from aggregate_cross_pair._build_backbone_edges for factorial mode.

    Returns:
        bb_edge_weights: list of np.ndarray, per-backbone composite edge weights
        bb_idf_coeffs: list of np.ndarray, per-backbone IDF coefficients
        bb_recv_attr: list of np.ndarray, per-backbone receiver attr weight pools
            (for within-receiver null model)
    """
    receivers = backbone_df["receiver"].values
    em_arr = backbone_df["EM"].values
    tg_arr = backbone_df["Target"].values
    n_bb = len(backbone_df)

    recv_weights = {}
    for recv in backbone_df["receiver"].unique():
        recv_weights[recv] = attr_by_celltype.get(recv, {})

    # Per-receiver attr weight arrays for within-receiver null model
    recv_attr_pools = {}
    for recv in backbone_df["receiver"].unique():
        w = attr_by_celltype.get(recv, {})
        recv_attr_pools[recv] = (
            np.array(list(w.values()), dtype=np.float64) if w
            else np.array([1.0]))

    cache = {}
    bb_edge_weights = []
    bb_idf_coeffs = []
    bb_recv_attr = []

    for i in range(n_bb):
        recv = receivers[i]
        em = em_arr[i]
        tg = tg_arr[i]
        key = (recv, em, tg)

        bb_recv_attr.append(recv_attr_pools[recv])

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
                idf_cs.append(raw["abs_nes_idf"][j] / raw["abs_nes"][j]
                              if raw["abs_nes"][j] > 0 else 1.0)

        ew = np.array(edge_ws, dtype=np.float64) if edge_ws else np.empty(0)
        idf_c = np.array(idf_cs, dtype=np.float64) if idf_cs else np.empty(0)

        cache[key] = (ew, idf_c)
        bb_edge_weights.append(ew)
        bb_idf_coeffs.append(idf_c)

    return bb_edge_weights, bb_idf_coeffs, bb_recv_attr


def _degree_median(ews, d):
    """Median along axis=2 with short-circuits for small degree."""
    if d == 1:
        return ews[:, :, 0]
    if d == 2:
        return (ews[:, :, 0] + ews[:, :, 1]) * 0.5
    return np.median(ews, axis=2)


def _storey_qvalue(pvals, lambda_val=0.5):
    """Storey's q-value with fixed-lambda pi0 estimation.

    More powerful than BH when a substantial fraction of tests are true
    alternatives (pi0 < 1). Safe for large test counts (>10K) with
    permutation-derived p-values.
    """
    m = len(pvals)
    if m == 0:
        return np.array([]), 1.0

    pi0 = min(1.0, (pvals > lambda_val).sum() / (m * (1 - lambda_val)))
    pi0 = max(pi0, 1 / m)  # floor to avoid pi0=0

    idx = np.argsort(pvals)
    pvals_sorted = pvals[idx]
    ranks = np.arange(1, m + 1, dtype=np.float64)

    qvals_sorted = pi0 * m * pvals_sorted / ranks
    qvals_sorted = np.minimum.accumulate(qvals_sorted[::-1])[::-1]
    qvals_sorted = np.minimum(qvals_sorted, 1.0)

    qvals = np.empty(m)
    qvals[idx] = qvals_sorted
    return qvals, pi0


def _run_permutation_one_contrast(backbone_c, all_mea_nes, sub_raw_edges,
                                  attr_by_celltype, n_permutations, contrast):
    """Run dual null model permutation test for one contrast's backbones.

    Returns a DataFrame with p-values and Storey q-values for both null models.
    """

    bg_kinases = sorted(all_mea_nes.keys())
    n_bg = len(bg_kinases)
    if n_bg == 0:
        print(f"    {contrast}: no background kinases — skipping")
        return _empty_perm_result(backbone_c, contrast)

    bg_nes_vec = np.array([all_mea_nes[k] for k in bg_kinases])

    bb_edge_weights, bb_idf_coeffs, bb_recv_attr = _build_backbone_edges(
        backbone_c, sub_raw_edges, attr_by_celltype)

    n_bb = len(backbone_c)
    n_edges = np.array([len(ew) for ew in bb_edge_weights])
    observed = np.zeros(n_bb)
    for i in range(n_bb):
        if n_edges[i] > 0:
            observed[i] = np.median(bb_edge_weights[i])

    active_idx = np.where(n_edges > 0)[0]
    n_active = len(active_idx)

    if n_active == 0:
        print(f"    {contrast}: no active backbones")
        return _empty_perm_result(backbone_c, contrast)

    # Median attr weight for Null 1 (enrichment null, fixed attr)
    all_attr_weights = []
    for recv_dict in attr_by_celltype.values():
        all_attr_weights.extend(recv_dict.values())
    attr_weight_arr = np.array(all_attr_weights, dtype=np.float64)
    median_attr_weight = float(np.median(attr_weight_arr)) if len(attr_weight_arr) else 1.0

    pw_degrees = n_edges[active_idx]
    pw_idf = [bb_idf_coeffs[i] for i in active_idx]
    pw_recv_attr = [bb_recv_attr[i] for i in active_idx]
    obs_active = observed[active_idx]

    degree_groups = defaultdict(list)
    for k in range(n_active):
        degree_groups[pw_degrees[k]].append((k, pw_idf[k]))

    degree_group_data = {}
    for d, members in degree_groups.items():
        local_indices = np.array([m[0] for m in members], dtype=np.intp)
        idf_arrays = np.array([m[1] for m in members])
        obs_slice = obs_active[local_indices]
        degree_group_data[d] = (local_indices, idf_arrays, obs_slice)

    batch_size = min(100, n_permutations)
    # Peak memory per batch step: batch * members * degree * 8bytes * 3arrays
    # Budget: 400MB max -> members = 4e8 / (batch * degree * 24)
    MEM_BUDGET = 400_000_000  # 400MB
    rng = np.random.default_rng(42)

    def _run_null1():
        """Null 1 (enrichment): shuffle kinase identity, fix attr at median."""
        ge = np.zeros(n_active)
        n_done = 0
        while n_done < n_permutations:
            b = min(batch_size, n_permutations - n_done)
            for d, (idx, idf_arr, obs_sl) in degree_group_data.items():
                n_members = len(idx)
                member_chunk = max(100, int(MEM_BUDGET / (b * max(d, 1) * 24)))
                for m_start in range(0, n_members, member_chunk):
                    m_end = min(m_start + member_chunk, n_members)
                    m_idx = idx[m_start:m_end]
                    m_idf = idf_arr[m_start:m_end]
                    m_obs = obs_sl[m_start:m_end]
                    m_n = m_end - m_start
                    drawn = rng.integers(0, n_bg, size=(b, m_n, d))
                    nes_drawn = bg_nes_vec[drawn]
                    ews = m_idf[np.newaxis, :, :] * nes_drawn * median_attr_weight
                    perm_medians = _degree_median(ews, d)
                    ge[m_idx] += (perm_medians >= m_obs[np.newaxis, :]).sum(axis=0)
            n_done += b
        return ge

    def _run_null2():
        """Null 2 (wiring): shuffle kinase identity + within-receiver attr.

        Each backbone draws attr weights from its own receiver's kinase
        attribution pool, not the global pool. This conditions on receiver
        identity and avoids cross-cell-type attr variance that drowns signal
        for contrasts with few significant kinases.
        """
        ge = np.zeros(n_active)
        n_done = 0
        while n_done < n_permutations:
            b = min(batch_size, n_permutations - n_done)
            for d, (idx, idf_arr, obs_sl) in degree_group_data.items():
                n_members = len(idx)
                member_chunk = max(100, int(MEM_BUDGET / (b * max(d, 1) * 24)))
                for m_start in range(0, n_members, member_chunk):
                    m_end = min(m_start + member_chunk, n_members)
                    m_idx = idx[m_start:m_end]
                    m_idf = idf_arr[m_start:m_end]
                    m_obs = obs_sl[m_start:m_end]
                    m_n = m_end - m_start
                    drawn = rng.integers(0, n_bg, size=(b, m_n, d))
                    nes_drawn = bg_nes_vec[drawn]
                    # Within-receiver attr shuffle
                    aw = np.ones((b, m_n, d))
                    for j in range(m_n):
                        pool = pw_recv_attr[m_start + j]
                        aw[:, j, :] = pool[
                            rng.integers(0, len(pool), size=(b, d))]
                    ews = m_idf[np.newaxis, :, :] * nes_drawn * aw
                    perm_medians = _degree_median(ews, d)
                    ge[m_idx] += (perm_medians >= m_obs[np.newaxis, :]).sum(axis=0)
            n_done += b
        return ge

    null1_ge = _run_null1()
    null2_ge = _run_null2()

    pval_null1 = np.ones(n_bb)
    pval_null2 = np.ones(n_bb)
    pval_null1[active_idx] = (null1_ge + 1) / (n_permutations + 1)
    pval_null2[active_idx] = (null2_ge + 1) / (n_permutations + 1)

    # Compute Storey's q on active backbones only — inactive (p=1) dilute pi0
    qval_null1 = np.ones(n_bb)
    qval_null2 = np.ones(n_bb)
    q1_active, pi0_1 = _storey_qvalue(pval_null1[active_idx])
    q2_active, pi0_2 = _storey_qvalue(pval_null2[active_idx])
    qval_null1[active_idx] = q1_active
    qval_null2[active_idx] = q2_active

    fdr_gate = icfg.PHOSPHO_FDR_GATE

    result = pd.DataFrame({
        "contrast": contrast,
        "receiver": backbone_c["receiver"].values,
        "Receptor": backbone_c["Receptor"].values,
        "EM": backbone_c["EM"].values,
        "Target": backbone_c["Target"].values,
        "observed_score": np.round(observed, 6),
        "n_edges": n_edges,
        "pval_null1": pval_null1,
        "pval_null2": pval_null2,
        "qval_null1": qval_null1,
        "qval_null2": qval_null2,
        "pi0_null1": pi0_1,
        "pi0_null2": pi0_2,
        "significant_null1": qval_null1 < fdr_gate,
        "significant_null2": qval_null2 < fdr_gate,
        "significant_both": (qval_null1 < fdr_gate) & (qval_null2 < fdr_gate),
    })

    n_sig1 = result["significant_null1"].sum()
    n_sig2 = result["significant_null2"].sum()
    print(f"    {contrast}: {n_active:,} active, "
          f"Null1 sig={n_sig1:,}, Null2 sig={n_sig2:,} "
          f"(pi0_1={pi0_1:.3f}, pi0_2={pi0_2:.3f}, bg={n_bg} kinases)")

    return result


def _empty_perm_result(backbone_c, contrast):
    """Return empty permutation result for a contrast with no edges."""
    n = len(backbone_c)
    return pd.DataFrame({
        "contrast": contrast,
        "receiver": backbone_c["receiver"].values,
        "Receptor": backbone_c["Receptor"].values,
        "EM": backbone_c["EM"].values,
        "Target": backbone_c["Target"].values,
        "observed_score": np.zeros(n),
        "n_edges": np.zeros(n, dtype=int),
        "pval_null1": np.ones(n),
        "pval_null2": np.ones(n),
        "qval_null1": np.ones(n),
        "qval_null2": np.ones(n),
        "pi0_null1": 1.0,
        "pi0_null2": 1.0,
        "significant_null1": np.zeros(n, dtype=bool),
        "significant_null2": np.zeros(n, dtype=bool),
        "significant_both": np.zeros(n, dtype=bool),
    })


def run_factorial_permutations(backbone_path, n_permutations, out_path):
    """Run backbone-level permutation tests for all 9 contrasts.

    Loads per-contrast kinase data from compute_kinase_support_factorial,
    then runs dual null model (enrichment + wiring) independently per contrast.
    Writes results incrementally per contrast to avoid OOM.

    Memory strategy: loads backbone CSV once, slices per contrast, frees
    the full DataFrame before running permutations. Each contrast uses
    ~500MB peak (edge structures + permutation arrays).
    """
    import gc
    from compute_kinase_support_factorial import load_shared_data

    print(f"\n=== Backbone-Level Permutation Tests "
          f"({n_permutations:,} iterations x 9 contrasts) ===")

    print("\nLoading shared kinase data (9 contrasts)...")
    shared = load_shared_data()

    # Pre-split backbone CSV by contrast to avoid holding 782MB in memory
    # during permutation runs
    print(f"\nSplitting backbone recurrence by contrast...")
    backbone_df = pd.read_csv(backbone_path,
                              dtype={"contrast": str, "receiver": str,
                                     "Receptor": str, "EM": str,
                                     "Target": str})
    contrast_backbones = {}
    for contrast in CONTRASTS:
        backbone_c = backbone_df[backbone_df["contrast"] == contrast].copy()
        backbone_c = backbone_c.reset_index(drop=True)
        contrast_backbones[contrast] = backbone_c
        print(f"  {contrast}: {len(backbone_c):,} backbones")
    del backbone_df
    gc.collect()

    first = True
    total_rows = 0
    for contrast in CONTRASTS:
        backbone_c = contrast_backbones[contrast]
        cdata = shared["contrast_data"][contrast]
        attr_by_ct = shared["attr_by_contrast_ct"][contrast]

        result = _run_permutation_one_contrast(
            backbone_c,
            all_mea_nes=cdata["all_mea_nes"],
            sub_raw_edges=cdata["sub_raw_edges"],
            attr_by_celltype=attr_by_ct,
            n_permutations=n_permutations,
            contrast=contrast,
        )

        result.to_csv(out_path, mode="w" if first else "a",
                      header=first, index=False)
        total_rows += len(result)
        first = False

        # Free memory between contrasts
        del result
        contrast_backbones[contrast] = None
        gc.collect()

    print(f"\n  Wrote {out_path} ({total_rows:,} rows)")
    return total_rows


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_hub_heatmaps(hub_df, out_dir):
    """3x3 grid of hub heatmaps (genotype x timepoint), colored by mean_abs_tpds."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 3, figsize=(24, 20))

    for i, geno in enumerate(GENOTYPES):
        for j, tp in enumerate(TIMEPOINTS):
            ax = axes[i, j]
            contrast = f"{geno}_{tp}"
            c_data = hub_df[hub_df["contrast"] == contrast]

            if c_data.empty:
                ax.set_title(f"{contrast} (no data)")
                ax.axis("off")
                continue

            pivot = c_data.pivot(index="sender", columns="receiver",
                                values="mean_abs_tpds")
            pivot = pivot.fillna(0)

            im = ax.imshow(pivot.values, cmap="YlOrRd", aspect="auto")
            ax.set_xticks(range(len(pivot.columns)))
            ax.set_xticklabels(pivot.columns, rotation=90, fontsize=6)
            ax.set_yticks(range(len(pivot.index)))
            ax.set_yticklabels(pivot.index, fontsize=6)
            ax.set_title(f"{contrast}\n(n_sig median: "
                         f"{int(c_data['n_significant'].median())})")
            plt.colorbar(im, ax=ax, shrink=0.6, label="mean |TPDS|")

    plt.suptitle("Hub matrices by genotype x timepoint", fontsize=14, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(out_dir, "hub_heatmap_grid.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Wrote {path}")


def plot_temporal_dynamics(temporal_df, out_dir, pval_threshold):
    """Line plot: n_significant vs timepoint, faceted by genotype."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Aggregate across all pairs per genotype x timepoint
    agg = temporal_df.groupby(["genotype", "timepoint"]).agg(
        total_sig=("n_significant", "sum"),
        total_pathways=("n_pathways", "sum"),
        mean_pct_sig=("pct_significant", "mean"),
    ).reset_index()

    # Sort timepoints
    tp_order = {"2mo": 0, "4mo": 1, "6mo": 2}
    agg["tp_idx"] = agg["timepoint"].map(tp_order)
    agg = agg.sort_values(["genotype", "tp_idx"])

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    colors = {"App": "#e74c3c", "Tau": "#3498db", "ApTt": "#2ecc71"}

    for idx, geno in enumerate(GENOTYPES):
        ax = axes[idx]
        g_data = agg[agg["genotype"] == geno]
        ax.plot(g_data["timepoint"], g_data["total_sig"],
                marker="o", color=colors[geno], linewidth=2, markersize=8)
        ax.set_title(f"{geno} vs WT", fontsize=12)
        ax.set_xlabel("Timepoint")
        if idx == 0:
            ax.set_ylabel(f"Total significant pathways (p<{pval_threshold})")

        # Add percentage labels
        for _, row in g_data.iterrows():
            ax.annotate(f"{row['mean_pct_sig']:.1f}%",
                        (row["timepoint"], row["total_sig"]),
                        textcoords="offset points", xytext=(0, 10),
                        ha="center", fontsize=9)

    plt.suptitle("Temporal dynamics of pathway significance", fontsize=14)
    plt.tight_layout()
    path = os.path.join(out_dir, "temporal_dynamics.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Wrote {path}")


def plot_kinase_coverage(kinase_df, out_dir):
    """Bar chart: kinase support coverage across 9 contrasts."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: fraction of significant pathways with kinase support
    kinase_df["kinase_coverage_pct"] = (
        100 * kinase_df["n_sig_with_kinase"] / kinase_df["n_sig_tpds"].clip(lower=1)
    )
    colors = []
    for c in kinase_df["contrast"]:
        if c.startswith("App"): colors.append("#e74c3c")
        elif c.startswith("Tau"): colors.append("#3498db")
        else: colors.append("#2ecc71")

    ax = axes[0]
    ax.bar(kinase_df["contrast"], kinase_df["kinase_coverage_pct"], color=colors)
    ax.set_ylabel("% of significant pathways\nwith kinase support")
    ax.set_title("Kinase coverage of significant pathways")
    ax.set_xticklabels(kinase_df["contrast"], rotation=45, ha="right")

    # Right: concordance breakdown
    ax = axes[1]
    bottom = [0] * len(kinase_df)
    for label, col, color in [
        ("concordant", "n_concordant", "#27ae60"),
        ("discordant", "n_discordant", "#c0392b"),
        ("mixed", "n_mixed", "#95a5a6"),
    ]:
        vals = kinase_df[col].values
        ax.bar(kinase_df["contrast"], vals, bottom=bottom,
               label=label, color=color)
        bottom = [b + v for b, v in zip(bottom, vals)]

    ax.set_ylabel("Count (significant pathways)")
    ax.set_title("Concordance of kinase evidence")
    ax.set_xticklabels(kinase_df["contrast"], rotation=45, ha="right")
    ax.legend()

    plt.tight_layout()
    path = os.path.join(out_dir, "kinase_coverage.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Wrote {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Factorial cross-pair aggregation")
    parser.add_argument("--pvalue-threshold", type=float, default=0.05,
                        help="P-value threshold for significance (default 0.05)")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing outputs")
    parser.add_argument("--skip-kinase-join", action="store_true",
                        help="Skip Q5 kinase integration (heaviest query)")
    parser.add_argument("--skip-plots", action="store_true",
                        help="Skip plot generation")
    parser.add_argument("--permutations", action="store_true",
                        help="Run backbone-level permutation null models (per contrast)")
    parser.add_argument("--n-permutations", type=int,
                        default=icfg.N_PERMUTATIONS_AGGREGATE,
                        help="Number of permutation iterations (default: %(default)s)")
    parser.add_argument("--memory-limit-gb", type=float, default=None,
                        help="DuckDB memory limit in GB (default: MEMORY_LIMIT_GB env or 6)")
    parser.add_argument("--duckdb-threads", type=int, default=None,
                        help="DuckDB thread count (default: DUCKDB_THREADS env or 4)")
    args = parser.parse_args()

    data_dir = icfg.FACTORIAL_ALL_PAIRS_DIR
    out_dir = os.path.join(data_dir, "aggregation")
    os.makedirs(out_dir, exist_ok=True)
    pval = args.pvalue_threshold

    print(f"Factorial aggregation: {data_dir}")
    print(f"  P-value threshold: {pval}")
    print(f"  Output: {out_dir}")
    duckdb_settings = _resolve_duckdb_settings(
        args.memory_limit_gb, args.duckdb_threads
    )
    print("  DuckDB settings: "
          f"memory_limit={duckdb_settings['memory_limit_gb']}GB, "
          f"threads={duckdb_settings['duckdb_threads']}, "
          f"preserve_insertion_order={str(duckdb_settings['preserve_insertion_order']).lower()}")

    t0 = time.monotonic()

    # Check if all aggregation queries are already cached
    def _should_run(path):
        return args.force or not os.path.exists(path)

    all_cached = not any(_should_run(os.path.join(out_dir, f)) for f in [
        "hub_matrix_by_contrast.csv", "contrast_comparison.csv",
        "temporal_dynamics.csv", "backbone_recurrence_by_contrast.csv",
        "backbone_provenance.csv",
        "target_convergence_by_contrast.csv",
    ]) and (args.skip_kinase_join or not _should_run(
        os.path.join(out_dir, "kinase_tpds_integration.csv")))

    # Skip DuckDB connection if all queries are cached (saves ~4GB memory
    # for permutation-only runs)
    if all_cached:
        print("  All aggregation queries cached — skipping DuckDB")
        con = None
        n_rows = 0
        prov_rows = _count_csv_rows(os.path.join(out_dir, "backbone_provenance.csv"))
    else:
        con, _ = create_connection(
            data_dir,
            memory_limit_gb=duckdb_settings["memory_limit_gb"],
            duckdb_threads=duckdb_settings["duckdb_threads"],
        )
        n_rows = con.execute("SELECT COUNT(*) FROM recv_all").fetchone()[0]
        n_pairs = con.execute(
            "SELECT COUNT(DISTINCT sender) FROM recv_all").fetchone()[0]
        print(f"  Parquet: {n_rows:,} total rows, {n_pairs} senders")

        prov_path = os.path.join(out_dir, "backbone_provenance.csv")
        if _should_run(prov_path):
            print("\nQ0: Backbone provenance...")
            t0_prov = time.monotonic()
            provenance_df, prov_path = build_backbone_provenance_table(data_dir, out_dir)
            prov_rows = len(provenance_df)
            print(f"  {prov_rows} rows ({time.monotonic()-t0_prov:.1f}s) -> {prov_path}")
        else:
            print("\nQ0: Backbone provenance (cached)")
            provenance_df = pd.read_csv(prov_path)
            prov_rows = len(provenance_df)
        con.register("backbone_provenance_df", provenance_df)
        con.execute("CREATE OR REPLACE TEMP VIEW backbone_provenance AS SELECT * FROM backbone_provenance_df")

    metadata = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "pvalue_threshold": pval,
        "n_contrasts": len(CONTRASTS),
        "contrasts": CONTRASTS,
        "total_pathways": n_rows,
        "backbone_provenance_rows": prov_rows,
        "duckdb_settings": duckdb_settings,
    }

    # --- Q1: Hub matrix ---
    hub_path = os.path.join(out_dir, "hub_matrix_by_contrast.csv")
    if _should_run(hub_path):
        print("\nQ1: Hub matrix...")
        t1 = time.monotonic()
        hub_df = q1_hub_matrix(con, pval)
        hub_df.to_csv(hub_path, index=False)
        metadata["hub_matrix_rows"] = len(hub_df)
        print(f"  {len(hub_df)} rows ({time.monotonic()-t1:.1f}s) -> {hub_path}")
    else:
        print("\nQ1: Hub matrix (cached)")
        hub_df = pd.read_csv(hub_path)

    # --- Q2: Contrast comparison (derived from Q1 in pandas) ---
    comp_path = os.path.join(out_dir, "contrast_comparison.csv")
    if _should_run(comp_path):
        print("\nQ2: Contrast comparison...")
        comparison_rows = []
        for c in CONTRASTS:
            c_data = hub_df[hub_df["contrast"] == c].copy()
            c_data["pct_significant"] = (
                100 * c_data["n_significant"] / c_data["n_pathways"].clip(lower=1))
            top = c_data.nlargest(10, "mean_abs_tpds")
            top = top[["contrast", "sender", "receiver", "n_pathways",
                        "n_significant", "pct_significant", "mean_abs_tpds"]].copy()
            top["rank"] = range(1, len(top) + 1)
            comparison_rows.append(top)
        comparison_df = pd.concat(comparison_rows, ignore_index=True)
        comparison_df.to_csv(comp_path, index=False)
        metadata["contrast_comparison_rows"] = len(comparison_df)
        print(f"  {len(comparison_df)} rows -> {comp_path}")
    else:
        print("\nQ2: Contrast comparison (cached)")

    # --- Q3: Temporal dynamics ---
    temp_path = os.path.join(out_dir, "temporal_dynamics.csv")
    if _should_run(temp_path):
        print("\nQ3: Temporal dynamics...")
        t3 = time.monotonic()
        temporal_df = q3_temporal_dynamics(con, pval)
        temporal_df.to_csv(temp_path, index=False)
        metadata["temporal_dynamics_rows"] = len(temporal_df)
        print(f"  {len(temporal_df)} rows ({time.monotonic()-t3:.1f}s) -> {temp_path}")
    else:
        print("\nQ3: Temporal dynamics (cached)")
        temporal_df = pd.read_csv(temp_path)

    # --- Q4: Backbone recurrence ---
    bb_path = os.path.join(out_dir, "backbone_recurrence_by_contrast.csv")
    if _should_run(bb_path):
        print("\nQ4: Backbone recurrence...")
        t4 = time.monotonic()
        backbone_rows = q4_backbone_recurrence(con, pval, bb_path)
        metadata["backbone_recurrence_rows"] = backbone_rows
        print(f"  {backbone_rows} rows ({time.monotonic()-t4:.1f}s) -> {bb_path}")
    else:
        print("\nQ4: Backbone recurrence (cached)")
        metadata["backbone_recurrence_rows"] = _count_csv_rows(bb_path)

    # --- Q5: Kinase integration ---
    kin_path = os.path.join(out_dir, "kinase_tpds_integration.csv")
    kinase_df = None
    if args.skip_kinase_join:
        print("\nQ5: Skipped (--skip-kinase-join)")
    elif _should_run(kin_path):
        print("\nQ5: Kinase integration...")
        t5 = time.monotonic()
        kinase_df = q5_kinase_integration(con, pval)
        kinase_df.to_csv(kin_path, index=False)
        metadata["kinase_integration_rows"] = len(kinase_df)
        print(f"  {len(kinase_df)} rows ({time.monotonic()-t5:.1f}s) -> {kin_path}")
    else:
        print("\nQ5: Kinase integration (cached)")
        kinase_df = pd.read_csv(kin_path)

    # --- Q6: Target convergence ---
    tgt_path = os.path.join(out_dir, "target_convergence_by_contrast.csv")
    if _should_run(tgt_path):
        print("\nQ6: Target convergence...")
        t6 = time.monotonic()
        target_df = q6_target_convergence(con, pval)
        target_df.to_csv(tgt_path, index=False)
        metadata["target_convergence_rows"] = len(target_df)
        print(f"  {len(target_df)} rows ({time.monotonic()-t6:.1f}s) -> {tgt_path}")
    else:
        print("\nQ6: Target convergence (cached)")

    # --- Metadata ---
    metadata["total_time_sec"] = round(time.monotonic() - t0, 1)
    meta_path = os.path.join(out_dir, "aggregation_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\nMetadata -> {meta_path}")

    # --- Plots ---
    if not args.skip_plots:
        print("\nGenerating plots...")
        plot_hub_heatmaps(hub_df, out_dir)
        plot_temporal_dynamics(temporal_df, out_dir, pval)
        if kinase_df is not None and not kinase_df.empty:
            plot_kinase_coverage(kinase_df, out_dir)

    if con is not None:
        con.close()
        del con

    # --- Permutation tests ---
    if args.permutations:
        perm_path = os.path.join(out_dir,
                                 "backbone_permutation_pvalues_by_contrast.csv")
        if not args.force and os.path.exists(perm_path):
            print("\nPermutation outputs already exist. Use --force to rerun.")
        else:
            bb_path = os.path.join(out_dir,
                                   "backbone_recurrence_by_contrast.csv")
            if not os.path.exists(bb_path):
                print("\nERROR: backbone recurrence must be computed first "
                      "(run without --permutations)")
            else:
                total_rows = run_factorial_permutations(
                    bb_path, args.n_permutations, perm_path)

                # Summary from the written file
                perm_df = pd.read_csv(perm_path)
                for c in CONTRASTS:
                    c_data = perm_df[perm_df["contrast"] == c]
                    n_active = (c_data["n_edges"] > 0).sum()
                    n1 = c_data["significant_null1"].sum()
                    n2 = c_data["significant_null2"].sum()
                    print(f"  {c}: {n_active:,} active, "
                          f"Null1={n1:,} ({100*n1/max(n_active,1):.1f}%), "
                          f"Null2={n2:,} ({100*n2/max(n_active,1):.1f}%)")
                del perm_df

    total = time.monotonic() - t0
    print(f"\nFactorial aggregation complete ({total:.0f}s).")


if __name__ == "__main__":
    main()

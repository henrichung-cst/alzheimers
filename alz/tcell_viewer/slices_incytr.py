"""Incytr pair-mode shard writers and celltype-QC builders for the T-cell viewer."""

from __future__ import annotations

import glob
import gzip
import json
import os
import re
import shutil
import sys

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from alz.shared import config  # noqa: E402
from alz.viewer.shared.payload_helpers import (  # noqa: E402
    _sanitize,
    _configure_duckdb_tempdir,
    _INCYTR_FC_NODES,
    _build_incytr_gene_node_index,
    _write_incytr_sidechain_slice,
    _write_gene_node_index_shard,
)
from alz.viewer.shared.incytr_index import (  # noqa: E402
    _INCYTR_LABEL_NODES,
    _INCYTR_LABEL_COLS,
    _INCYTR_LABEL_VOCAB,
    _INCYTR_SCORE_COLS_BASE,
    _INCYTR_SCORE_COLS_OPTIONAL,
    _SIGN_VEC_LABELS,
    _active_optional_score_cols,
    _idx_label_bits,
    _idx_traj_bits,
    write_incytr_backbone_grains,
)
from alz.tcell_viewer.paths import (  # noqa: E402
    EDGE_SLICES_INCYTR_BACKBONE_DIR,
    EDGE_SLICES_INCYTR_PATHWAYS_DIR,
    INCYTR_PAIR_MODE_TCELLS_DIR,
    SCHEMA_VERSION,
    TCELLS_INCYTR_INPUTS_DIR,
    UNIFIED_VIEWER_DIR,
)
from alz.tcell_viewer.common import DONORS, _incytr_sanitize  # noqa: E402
from alz.tcell_viewer.state_contract import validate_pathway_states  # noqa: E402

# ---------------------------------------------------------------------------
# Module-local constants
# ---------------------------------------------------------------------------

_INCYTR_FC_METRICS = ("sclog2FC", "pr_log2FC", "ps_log2FC", "py_log2FC")
_INCYTR_FC_COLS = tuple(
    f"{node}_{metric}" for node in _INCYTR_FC_NODES for metric in _INCYTR_FC_METRICS
)
_INCYTR_LABEL_SRC = tuple(f"{n}.label" for n in _INCYTR_LABEL_NODES)

# Pre-aggregated heatmap thresholds (matches mouse-cohort viewer contract).
_INCYTR_PATHWAY_PVALUES = (0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0)
_INCYTR_PATHWAY_ABS_PDS = (0.0, 0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0)
_INCYTR_TOP_INSTANCE_LIMIT = 5000
_INCYTR_INDEX_FILENAME = "incytr_index.bin.gz"

_PAIR_FILE_RE = re.compile(r"(d\d+_d\d+)_incytr_output\.parquet$")
_BACKBONE_FILE_RE = re.compile(r"(d\d+_d\d+)_backbone_output\.parquet$")
_KINASE_SIDECHAIN_EDGE_DIR = os.path.join(
    config.REPO_ROOT, "outputs", "reports", "kinase_kinase_edges"
)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _contrast_days(contrasts: list[str]) -> list[int]:
    """Days participating in T-cell day-vs-baseline Incytr contrasts."""
    days: set[int] = set()
    for contrast in contrasts:
        for part in str(contrast).split("_"):
            if part.startswith("d") and part[1:].isdigit():
                days.add(int(part[1:]))
    return sorted(days)


def _contrast_from_filename(fname: str) -> str | None:
    m = _PAIR_FILE_RE.search(fname)
    return m.group(1) if m else None


def _backbone_contrast_from_filename(fname: str) -> str | None:
    """`d13_d2_backbone_output.parquet` -> `d13_d2`."""
    m = _BACKBONE_FILE_RE.search(fname)
    return m.group(1) if m else None


def _timepoint_label(contrast: str) -> str:
    """`d13_d2` → `d13` (the disease timepoint; d2 is the baseline)."""
    return contrast.split("_", 1)[0] if "_" in contrast else contrast


def _contrast_pvalue_column(column_names: set[str], contrast: str) -> str | None:
    """Return the follow-up-day p-value column for a day-vs-baseline contrast."""
    follow_up = _timepoint_label(contrast)
    column = f"p_value_{follow_up}"
    return column if column in column_names else None


def _pvalue_filter_sql(threshold: float, has_pvalue: bool) -> str:
    """SQL for a precomputed p-value band; the final band is truly ungated."""
    if not has_pvalue or threshold >= 1.0:
        return "TRUE"
    return f"pvalue < {threshold}"


# ---------------------------------------------------------------------------
# Cell-type QC
# ---------------------------------------------------------------------------

def _read_tcell_incytr_celltype_qc(
    donor: str,
    celltypes: list[str],
    contrasts: list[str],
) -> dict:
    """Cell-count QC metadata for the donor-specific T-cell Incytr view."""
    counts_path = os.path.join(
        TCELLS_INCYTR_INPUTS_DIR, donor, "scrna", "cell_counts.csv"
    )
    scope_days = _contrast_days(contrasts)
    out = {
        "source": os.path.relpath(counts_path, config.REPO_ROOT),
        "sample_scope": (
            "donor-specific evidence-backed state counts over Incytr contrast days "
            f"{scope_days}"
        ),
        "units": "cells",
        "days": [f"d{d}" for d in scope_days],
        "by_celltype": {},
    }
    if not os.path.exists(counts_path):
        print(f"  ({donor}) (warn) cell-count QC file not found: {counts_path}",
              flush=True)
        return out

    counts = pd.read_csv(counts_path)
    required = {"state", "day", "n_cells"}
    if not required.issubset(counts.columns):
        print(f"  ({donor}) (warn) cell-count QC missing columns: "
              f"{sorted(required - set(counts.columns))}", flush=True)
        return out

    counts["state"] = counts["state"].astype(str)
    counts["day"] = pd.to_numeric(counts["day"], errors="coerce")
    counts["n_cells"] = pd.to_numeric(counts["n_cells"], errors="coerce")
    counts = counts.dropna(subset=["state", "day", "n_cells"])
    counts["day"] = counts["day"].astype(int)
    if scope_days:
        counts = counts[counts["day"].isin(scope_days)]
    state_day_counts = counts.groupby(["state", "day"])["n_cells"].sum()

    by_celltype: dict[str, dict] = {}
    for ct in sorted(set(celltypes)):
        if scope_days:
            by_day = {
                f"d{day}": int(state_day_counts.get((ct, day), 0))
                for day in scope_days
            }
            values = list(by_day.values())
            rec = {
                "median_n": float(np.median(values)),
                "mean_n": float(np.mean(values)),
                "min_n": int(min(values)),
                "total_n": int(sum(values)),
                "n_timepoints": len(scope_days),
                "by_day": by_day,
            }
        else:
            rec = {
                "median_n": None,
                "mean_n": None,
                "min_n": None,
                "total_n": 0,
                "n_timepoints": 0,
                "by_day": {},
            }
        by_celltype[ct] = rec

    out["by_celltype"] = by_celltype
    print(f"  ({donor}) incytr celltype_qc: raw counts for "
          f"{len(by_celltype)} state(s)", flush=True)
    return out


def _build_tcell_celltype_pathway_qc(con, celltype_qc: dict) -> dict:
    """Scatterplot rows: median cell count vs Incytr pathway burden."""
    receiver = con.execute("""
        SELECT receiver AS cell_type,
               COUNT(*)::INTEGER AS receiver_paths_all,
               SUM(CASE WHEN ABS(PDS) > 1 THEN 1 ELSE 0 END)::INTEGER
                 AS receiver_paths_abs_pds_gt1
        FROM src
        GROUP BY receiver
    """).fetchdf()
    sender = con.execute("""
        SELECT sender AS cell_type,
               COUNT(*)::INTEGER AS sender_paths_all,
               SUM(CASE WHEN ABS(PDS) > 1 THEN 1 ELSE 0 END)::INTEGER
                 AS sender_paths_abs_pds_gt1
        FROM src
        GROUP BY sender
    """).fetchdf()
    df = receiver.merge(sender, on="cell_type", how="outer").fillna(0)
    rows: list[dict] = []
    by_celltype = celltype_qc.get("by_celltype") or {}
    for rec in df.to_dict("records"):
        ct = str(rec["cell_type"])
        qc = by_celltype.get(ct, {})
        receiver_gt1 = int(rec.get("receiver_paths_abs_pds_gt1") or 0)
        sender_gt1 = int(rec.get("sender_paths_abs_pds_gt1") or 0)
        rows.append({
            "cell_type": ct,
            "median_n": qc.get("median_n"),
            "mean_n": qc.get("mean_n"),
            "min_n": qc.get("min_n"),
            "total_n": qc.get("total_n"),
            "n_timepoints": qc.get("n_timepoints"),
            "receiver_paths_all": int(rec.get("receiver_paths_all") or 0),
            "receiver_paths_abs_pds_gt1": receiver_gt1,
            "sender_paths_all": int(rec.get("sender_paths_all") or 0),
            "sender_paths_abs_pds_gt1": sender_gt1,
            "endpoint_paths_abs_pds_gt1": receiver_gt1 + sender_gt1,
        })
    rows.sort(key=lambda r: (
        -(r["receiver_paths_abs_pds_gt1"] or 0),
        r["median_n"] if r["median_n"] is not None else float("inf"),
    ))
    return {
        "schema_version": 1,
        "x_metric": "median_n",
        "y_metric": "receiver_paths_abs_pds_gt1",
        "y_transform": "log10(y + 1)",
        "pds_gate": "abs(PDS) > 1",
        "rows": _sanitize(rows),
    }


# ---------------------------------------------------------------------------
# Incytr pair-mode shards (per donor × contrast × sender × receiver)
# ---------------------------------------------------------------------------

def _write_donor_pair_pathways(donor: str) -> dict | None:
    """Shard donor's pair-mode parquets by (sender, receiver).

    Output filenames: `{donor}__{sender}__{receiver}.parquet`. Returns the
    per-donor `incytr_pathways` payload sub-block, or None if no parquets exist.
    """
    import duckdb

    input_dir = os.path.join(INCYTR_PAIR_MODE_TCELLS_DIR, donor, "wide")
    if not os.path.isdir(input_dir):
        print(f"  ({donor}) no pair-mode dir: {input_dir}", flush=True)
        return None
    parquet_files = sorted(glob.glob(os.path.join(input_dir, "*_incytr_output.parquet")))
    if not parquet_files:
        print(f"  ({donor}) no pair-mode parquets in {input_dir}", flush=True)
        return None

    file_to_contrast: list[tuple[str, str]] = []
    for fpath in parquet_files:
        contrast = _contrast_from_filename(os.path.basename(fpath))
        if contrast is not None:
            file_to_contrast.append((fpath, contrast))
    if not file_to_contrast:
        return None

    present_contrasts = sorted({c for _, c in file_to_contrast},
                               key=lambda x: int(x.split("_")[0][1:]))
    contrast_to_idx = {c: i for i, c in enumerate(present_contrasts)}
    print(f"  ({donor}) contrasts = {present_contrasts}", flush=True)

    con = duckdb.connect()
    con.execute("PRAGMA threads=8; PRAGMA memory_limit='12GB';")
    _configure_duckdb_tempdir(con)

    sample_schema = pq.read_schema(file_to_contrast[0][0])
    src_cols = {f.name for f in sample_schema}
    dir_flag_cols = [c for c in ("pr_up", "pr_down", "ps_up", "ps_down",
                                  "py_up", "py_down") if c in src_cols]
    extra_path_cols = [c for c in ("log2FC",) if c in src_cols]
    # Optional PTM-track score columns (Ack/KGG/Rme1). The t-cell cohort has no
    # acetylation/ubiquitination assay, so this self-gates empty; the wiring is
    # kept symmetric with the 5xFAD/Song builders so it surfaces automatically
    # if a donor ever carries PTM tracks.
    optional_in_schema = [c for c in _INCYTR_SCORE_COLS_OPTIONAL if c in src_cols]

    selects = []
    has_pvalue = False
    for fpath, contrast in file_to_contrast:
        sch = pq.read_schema(fpath)
        names = {f.name for f in sch}
        pcol_disease = _contrast_pvalue_column(names, contrast)
        has_pvalue = has_pvalue or pcol_disease is not None
        if pcol_disease is None:
            print(
                f"    (warn) no follow-up p-value column for {contrast}; using NULL",
                flush=True,
            )
        pcol_clause = f'CAST("{pcol_disease}" AS DOUBLE)' if pcol_disease else "CAST(NULL AS DOUBLE)"

        # SiK_score is emitted per-condition (SiK_score_<cond1> / SiK_score_d2);
        # collapse to the treatment arm (cond1, the later day), baseline is d2.
        cond1 = contrast.split("_")[0]
        sik_disease = f"SiK_score_{cond1}"
        if sik_disease in names:
            sik_clause = f'CAST("{sik_disease}" AS DOUBLE) AS SiK_score'
        else:
            print(f"    (warn) no disease-arm SiK_score col ({sik_disease}) in "
                  f"{os.path.basename(fpath)}; using NULL", flush=True)
            sik_clause = "CAST(NULL AS DOUBLE) AS SiK_score"

        base_nonsik = [c for c in _INCYTR_SCORE_COLS_BASE if c != "SiK_score"]
        generic_scores = base_nonsik + [c for c in optional_in_schema if c in names]
        score_clauses = ",\n          ".join(
            f"CAST({c} AS DOUBLE) AS {c}" for c in generic_scores if c in names
        )
        missing_scores = [c for c in base_nonsik if c not in names]
        missing_score_clauses = ",\n          ".join(
            f"CAST(NULL AS DOUBLE) AS {c}" for c in missing_scores
        )
        dir_clauses = ",\n          ".join(
            f"CAST({c} AS DOUBLE) AS {c}" for c in dir_flag_cols
        )
        path_clauses = ",\n          ".join(
            f"CAST({c} AS DOUBLE) AS {c}" for c in extra_path_cols
        )
        fc_clauses = ",\n          ".join(
            (f'CAST("{c}" AS DOUBLE) AS "{c}"' if c in names
             else f'CAST(NULL AS DOUBLE) AS "{c}"')
            for c in _INCYTR_FC_COLS
        )
        label_clauses = ",\n          ".join(
            (f'CAST("{src}" AS VARCHAR) AS "{dst}"' if src in names
             else f'CAST(NULL AS VARCHAR) AS "{dst}"')
            for src, dst in zip(_INCYTR_LABEL_SRC, _INCYTR_LABEL_COLS)
        )
        clauses = [score_clauses, missing_score_clauses, sik_clause,
                   dir_clauses, path_clauses, fc_clauses, label_clauses]
        extra_select = ",\n          ".join(c for c in clauses if c)

        selects.append(f"""
        SELECT
          "Sender.group"   AS sender,
          "Receiver.group" AS receiver,
          Path, Ligand, Receptor, EM, Target,
          '{contrast}'      AS contrast,
          {pcol_clause}     AS pvalue,
          CAST(PDS AS DOUBLE) AS PDS,
          {extra_select}
        FROM read_parquet('{fpath}')
        """)
    con.execute(f"CREATE VIEW src AS {' UNION ALL '.join(selects)}")
    n_src = con.execute("SELECT COUNT(*) FROM src").fetchone()[0]
    print(f"  ({donor}) loaded {n_src:,} rows across "
          f"{len(file_to_contrast)} contrast(s)", flush=True)

    # Which optional PTM score cols have any non-zero value for this donor?
    # Only these enter the index, shards, and score_columns; all-zero/absent
    # cols are dropped (honesty rule). Empty for every current t-cell donor.
    active_optional = _active_optional_score_cols(set(optional_in_schema), con, "src")
    effective_score_cols: tuple[str, ...] = _INCYTR_SCORE_COLS_BASE + active_optional

    senders_canonical = sorted({r[0] for r in con.execute(
        "SELECT DISTINCT sender FROM src").fetchall()})
    receivers_canonical = sorted({r[0] for r in con.execute(
        "SELECT DISTINCT receiver FROM src").fetchall()})
    canonical_roster = validate_pathway_states(
        donor, senders_canonical, receivers_canonical
    )
    sender_to_idx = {s: i for i, s in enumerate(senders_canonical)}
    receiver_to_idx = {r: i for i, r in enumerate(receivers_canonical)}
    n_s, n_r, n_c = len(senders_canonical), len(receivers_canonical), len(present_contrasts)
    print(f"  ({donor}) senders={n_s}, receivers={n_r}, contrasts={n_c}", flush=True)
    celltype_qc = _read_tcell_incytr_celltype_qc(
        donor,
        canonical_roster,
        list(present_contrasts),
    )
    celltype_pathway_qc = _build_tcell_celltype_pathway_qc(con, celltype_qc)

    # Heatmap counts cube.
    n_thr = len(_INCYTR_PATHWAY_PVALUES)
    n_ap = len(_INCYTR_PATHWAY_ABS_PDS)
    def pval_filter(threshold: float) -> str:
        return _pvalue_filter_sql(threshold, has_pvalue)
    hm_clauses = ", ".join(
        f"COUNT(*) FILTER (WHERE {pval_filter(tp)} AND COALESCE(ABS(PDS), 0) >= {tap}) AS c_{ip}_{iap}"
        for ip, tp in enumerate(_INCYTR_PATHWAY_PVALUES)
        for iap, tap in enumerate(_INCYTR_PATHWAY_ABS_PDS)
    )
    hm_rows = con.execute(f"""
        SELECT sender, receiver, contrast, {hm_clauses}
        FROM src GROUP BY sender, receiver, contrast
    """).fetchall()
    grid = np.zeros((n_s, n_r, n_c, n_thr, n_ap), dtype=np.uint32)
    for row in hm_rows:
        s_raw, r_raw, c = row[0], row[1], row[2]
        if s_raw not in sender_to_idx or r_raw not in receiver_to_idx:
            continue
        if c not in contrast_to_idx:
            continue
        s_i, r_i, c_i = sender_to_idx[s_raw], receiver_to_idx[r_raw], contrast_to_idx[c]
        offset = 3
        for ip in range(n_thr):
            for iap in range(n_ap):
                grid[s_i, r_i, c_i, ip, iap] = int(row[offset])
                offset += 1
    totals = np.zeros((n_thr, n_ap), dtype=np.uint64)
    for ip in range(n_thr):
        for iap in range(n_ap):
            totals[ip, iap] = int(grid[:, :, :, ip, iap].sum())
    heatmap_counts = {
        "thresholds": list(_INCYTR_PATHWAY_PVALUES),
        "abs_pds_thresholds": list(_INCYTR_PATHWAY_ABS_PDS),
        "shape": [n_s, n_r, n_c, n_thr, n_ap],
        "counts": grid.flatten().tolist(),
        "total_by_threshold": totals.tolist(),
    }
    hm_signed_rows = con.execute(f"""
        SELECT sender, receiver, contrast,
               CASE WHEN PDS > 0 THEN 2 WHEN PDS < 0 THEN 0 ELSE 1 END AS s,
               {hm_clauses}
        FROM src GROUP BY sender, receiver, contrast, s
    """).fetchall()
    signed_grid = np.zeros((n_s, n_r, n_c, 3, n_thr, n_ap), dtype=np.uint32)
    for row in hm_signed_rows:
        s_raw, r_raw, c, sign_i = row[0], row[1], row[2], int(row[3])
        if s_raw not in sender_to_idx or r_raw not in receiver_to_idx:
            continue
        if c not in contrast_to_idx:
            continue
        s_i, r_i, c_i = sender_to_idx[s_raw], receiver_to_idx[r_raw], contrast_to_idx[c]
        offset = 4
        for ip in range(n_thr):
            for iap in range(n_ap):
                signed_grid[s_i, r_i, c_i, sign_i, ip, iap] = int(row[offset])
                offset += 1
    signed_totals = np.zeros((3, n_thr, n_ap), dtype=np.uint64)
    for sign_i in range(3):
        for ip in range(n_thr):
            for iap in range(n_ap):
                signed_totals[sign_i, ip, iap] = int(signed_grid[:, :, :, sign_i, ip, iap].sum())
    heatmap_counts_signed = {
        "thresholds": list(_INCYTR_PATHWAY_PVALUES),
        "abs_pds_thresholds": list(_INCYTR_PATHWAY_ABS_PDS),
        "shape": [n_s, n_r, n_c, 3, n_thr, n_ap],
        "counts": signed_grid.flatten().tolist(),
        "total_by_sign_threshold": signed_totals.tolist(),
        "sign_source": "PDS",
    }

    # Pathway counts cube (per contrast x sign x thresholds).
    def _build_pathway_counts(where_extra: str = "") -> dict:
        thr_clauses = ", ".join(
            f"COUNT(*) FILTER (WHERE {pval_filter(tp)} AND COALESCE(ABS(PDS), 0) >= {tap}) AS c_{ip}_{iap}"
            for ip, tp in enumerate(_INCYTR_PATHWAY_PVALUES)
            for iap, tap in enumerate(_INCYTR_PATHWAY_ABS_PDS)
        )
        where_parts = []
        if where_extra:
            where_parts.append(where_extra)
        where_clause = "WHERE " + " AND ".join(where_parts) if where_parts else ""
        pathway_rows = con.execute(f"""
            SELECT contrast,
                   CASE WHEN PDS > 0 THEN 2 WHEN PDS < 0 THEN 0 ELSE 1 END AS s,
                   {thr_clauses}
            FROM src {where_clause} GROUP BY contrast, s
        """).fetchall()
        pathway_arr = np.zeros((n_c, 3, n_thr, n_ap), dtype=np.uint32)
        for row in pathway_rows:
            contrast, s_idx = row[0], int(row[1])
            if contrast not in contrast_to_idx:
                continue
            c_idx = contrast_to_idx[contrast]
            for ip in range(n_thr):
                for iap in range(n_ap):
                    pathway_arr[c_idx, s_idx, ip, iap] = int(row[2 + ip * n_ap + iap])
        return {
            "thresholds": list(_INCYTR_PATHWAY_PVALUES),
            "abs_pds_thresholds": list(_INCYTR_PATHWAY_ABS_PDS),
            "contrasts": list(present_contrasts),
            "counts": pathway_arr.flatten().tolist(),
            "shape": [n_c, 3, n_thr, n_ap],
            "sign_source": "PDS",
        }

    pathway_counts = _build_pathway_counts()

    # Annotate against the complete donor-local long table before selecting
    # top instances or streaming pair shards. The completeness gate must see
    # every follow-up day, not only rows that survive an output cut.
    ordered_days = tuple(sorted(
        {_timepoint_label(c) for c in present_contrasts},
        key=lambda day: int(day[1:]) if day.startswith("d") else day,
    ))
    from alz.viewer.shared.trajectory import annotate_trajectory_columns
    trajectory_source = con.execute("""
        SELECT sender, receiver, Ligand, Receptor, EM, Target, Path, contrast, PDS
        FROM src
        WHERE PDS IS NOT NULL
    """).fetchdf()
    trajectory_source, _, traj_summary = annotate_trajectory_columns(
        trajectory_source,
        series_key=lambda frame: pd.Series(donor, index=frame.index),
        axis_value=lambda frame: frame["contrast"].map(_timepoint_label),
        ordered_axis=ordered_days,
        valid_series={donor},
        source_label=f"tcell/{donor}",
    )
    # Key on the path component columns, not the joined Path string: Incytr's
    # Path uses "*" as its separator, so any client-side re-join would silently
    # miss every row.
    _traj_merge_cols = ["sender", "receiver", "Ligand", "Receptor", "EM", "Target"]
    trajectory_map = trajectory_source[
        [*_traj_merge_cols, "traj_labels", "sign_vec"]
    ].drop_duplicates(_traj_merge_cols)

    # Complete donor-local top-mode index. This mirrors the unified viewer's
    # packed-column contract so the shared Incytr Pathways tab can filter over
    # every row instead of a pre-capped top_instances table.
    incytr_index_columns = (
        [("PDS", "f4"), ("pvalue", "f4")]
        + [(sc, "u2") for sc in effective_score_cols]
        + [("ligandId", "u2"), ("receptorId", "u2"),
           ("emId", "u2"), ("targetId", "u2")]
        + [("senderId", "u1"), ("receiverId", "u1"), ("contrastId", "u1"),
           ("labelBits", "u1"), ("trajBits", "u1")]
        + [("kinaseEdges", "u2")]
    )
    idx_gene_to_id: dict[str, int] = {}
    idx_gene_vocab: list[str] = []
    idx_chunks: list[dict] = []

    def _idx_gene_ids(series) -> np.ndarray:
        cat = series.astype(str)
        for gene in cat.unique():
            if gene not in idx_gene_to_id:
                idx_gene_to_id[gene] = len(idx_gene_vocab)
                idx_gene_vocab.append(gene)
        return cat.map(idx_gene_to_id).to_numpy(dtype="<u2")

    def _accumulate_index(s_name: str, r_name: str, frame: pd.DataFrame) -> None:
        n = len(frame)
        if n == 0:
            return
        chunk = {
            "senderId": np.full(n, sender_to_idx[s_name], dtype="<u1"),
            "receiverId": np.full(n, receiver_to_idx[r_name], dtype="<u1"),
            "contrastId": frame["contrast"].map(contrast_to_idx).to_numpy(dtype="<u1"),
            "ligandId": _idx_gene_ids(frame["Ligand"]),
            "receptorId": _idx_gene_ids(frame["Receptor"]),
            "emId": _idx_gene_ids(frame["EM"]),
            "targetId": _idx_gene_ids(frame["Target"]),
            "labelBits": _idx_label_bits(frame),
            "trajBits": _idx_traj_bits(frame["traj_labels"]),
            "PDS": frame["PDS"].to_numpy(dtype="<f4"),
            "pvalue": frame["pvalue"].to_numpy(dtype="<f4"),
            "kinaseEdges": frame["kinase_edges"].to_numpy(dtype="<u2"),
        }
        for sc in effective_score_cols:
            chunk[sc] = (
                frame[sc].to_numpy(dtype="float16").view("<u2")
                if sc in frame.columns else np.zeros(n, dtype="<u2")
            )
        idx_chunks.append(chunk)

    top = con.execute(f"""
        SELECT
          sender, receiver, Path, Ligand, Receptor, EM, Target,
          contrast, pvalue, PDS, ABS(PDS) AS abs_pds,
          {", ".join(effective_score_cols)},
          {", ".join(f'"{c}"' for c in _INCYTR_LABEL_COLS)}
        FROM src
        WHERE PDS IS NOT NULL
        ORDER BY ABS(PDS) DESC NULLS LAST
        LIMIT {_INCYTR_TOP_INSTANCE_LIMIT}
    """).fetchdf()
    if len(top):
        top["rank"] = np.arange(1, len(top) + 1, dtype=int)
        top = top.merge(
            trajectory_map,
            on=_traj_merge_cols,
            how="left",
        )
        top["traj_labels"] = top["traj_labels"].fillna("")
        top["sign_vec"] = top["sign_vec"].fillna("")
    top_cols = [
        "rank", "sender", "receiver", "Path", "Ligand", "Receptor", "EM",
        "Target", "contrast", "pvalue", "PDS", "abs_pds", "traj_labels",
        "sign_vec", *effective_score_cols,
        *_INCYTR_LABEL_COLS,
    ]
    top_instances = {
        "rank_by": "abs(PDS)",
        "limit": _INCYTR_TOP_INSTANCE_LIMIT,
        "rows": _sanitize(top[top_cols].to_dict("records")) if len(top) else [],
    }
    gene_node_index = _build_incytr_gene_node_index(con)
    print(
        f"  ({donor}) gene_node_index: "
        f"{len(gene_node_index['gene_id']):,} gene-role-pair entries; "
        f"{len(gene_node_index['genes']):,} genes",
        flush=True,
    )

    # Shard the long table per (sender, receiver) — donor-scoped output dir.
    shard_select_cols = (
        ["Ligand", "Receptor", "EM", "Target", "contrast", "pvalue", "PDS"]
        + list(effective_score_cols) + list(dir_flag_cols)
        + list(extra_path_cols) + list(_INCYTR_FC_COLS) + list(_INCYTR_LABEL_COLS)
    )
    float32_cols = ["pvalue"]
    float16_cols = (["PDS"] + list(effective_score_cols) + list(_INCYTR_FC_COLS)
                    + list(dir_flag_cols) + list(extra_path_cols))
    float_cols = float32_cols + float16_cols

    # Per-receiver distinct-kinase counts keyed by (contrast, role, node gene) —
    # the exact edges the sidechain graph draws for a pathway row. Empty for
    # donors without a within-cohort kinase attribution artifact (e.g. donor2).
    kin_lut = _build_terminal_kinase_lookup(donor)

    present_pairs: list[list[str]] = []
    pair_row_counts: dict[str, int] = {}
    total_rows = 0

    def _flush(key: tuple[str, str], frames: list[pd.DataFrame]) -> None:
        nonlocal total_rows
        if not frames:
            return
        sub = pd.concat(frames, ignore_index=True, copy=False)
        for col in _INCYTR_LABEL_COLS:
            if col in sub.columns:
                sub[col] = pd.Categorical(sub[col], categories=_INCYTR_LABEL_VOCAB)
        s, r = key
        sub["sender"] = s
        sub["receiver"] = r
        sub = sub.merge(
            trajectory_map,
            on=_traj_merge_cols,
            how="left",
        )
        sub["traj_labels"] = sub["traj_labels"].fillna("")
        sub["sign_vec"] = sub["sign_vec"].fillna("")
        # Kinase-edge count per pathway row: distinct kinase→node edges the
        # sidechain graph draws (Target/EM/Receptor, contrast-matched,
        # owning_cluster == this receiver). Kinases never attach at Ligand.
        recv_lut = kin_lut.get(r, {})
        n_rows = len(sub)
        kinase_edges = np.zeros(n_rows, dtype=np.int64)
        if recv_lut:
            contrasts = sub["contrast"].astype(str).to_numpy()
            for role in ("Target", "EM", "Receptor"):
                genes = sub[role].astype(str).to_numpy()
                kinase_edges += np.fromiter(
                    (recv_lut.get((contrasts[i], role, genes[i]), 0)
                     for i in range(n_rows)),
                    dtype=np.int64, count=n_rows,
                )
        sub["kinase_edges"] = kinase_edges.astype(np.int32)
        _accumulate_index(s, r, sub)
        sub = sub.drop(columns=["sender", "receiver"])
        for col in float32_cols:
            if col in sub.columns:
                sub[col] = sub[col].astype("float32")
        for col in float16_cols:
            if col in sub.columns:
                sub[col] = sub[col].astype("float16")
        path_sort_cols = [c for c in ("Ligand", "Receptor", "EM", "Target", "contrast")
                          if c in sub.columns]
        if path_sort_cols:
            sub = sub.sort_values(path_sort_cols, kind="stable",
                                  na_position="last").reset_index(drop=True)
        fname = f"{donor}__{_incytr_sanitize(s)}__{_incytr_sanitize(r)}.parquet"
        path = os.path.join(EDGE_SLICES_INCYTR_PATHWAYS_DIR, fname)
        present_floats = [c for c in float_cols if c in sub.columns]
        bss_cols = {c: "BYTE_STREAM_SPLIT" for c in present_floats}
        dict_cols = [c for c in sub.columns if c not in bss_cols]
        pq.write_table(
            pa.Table.from_pandas(sub, preserve_index=False),
            path, compression="zstd",
            column_encoding=bss_cols if bss_cols else None,
            use_dictionary=dict_cols if bss_cols else True,
        )
        present_pairs.append([s, r])
        pair_row_counts[fname] = len(sub)
        total_rows += len(sub)

    stream_cols = ["receiver"] + shard_select_cols
    for s in senders_canonical:
        reader = con.execute(
            f"""SELECT {', '.join(stream_cols)} FROM src
                WHERE sender = ? ORDER BY receiver""", [s],
        ).fetch_record_batch(1_000_000)
        cur_receiver: str | None = None
        buf: list[pd.DataFrame] = []
        for batch in reader:
            bdf = batch.to_pandas()
            receivers = bdf["receiver"].to_numpy()
            starts = [0]
            for i in range(1, len(receivers)):
                if receivers[i] != receivers[i - 1]:
                    starts.append(i)
            starts.append(len(receivers))
            for j in range(len(starts) - 1):
                a, b = starts[j], starts[j + 1]
                r = receivers[a]
                seg = bdf.iloc[a:b].drop(columns=["receiver"])
                if cur_receiver is None:
                    cur_receiver = r
                elif r != cur_receiver:
                    _flush((s, cur_receiver), buf)
                    buf = []
                    cur_receiver = r
                buf.append(seg)
        if buf and cur_receiver is not None:
            _flush((s, cur_receiver), buf)
    con.close()

    print(f"  ({donor}) wrote {len(present_pairs)} shards ({total_rows:,} rows)",
          flush=True)

    global_index = None
    if idx_chunks:
        if sys.byteorder != "little":
            raise RuntimeError("T-cell Incytr global index assumes little-endian")
        cols = {
            name: np.concatenate([chunk[name] for chunk in idx_chunks])
            for name, _dt in incytr_index_columns
        }
        idx_chunks.clear()
        n_idx = int(len(cols["PDS"]))
        perm = np.argsort(-np.abs(cols["PDS"]), kind="stable")
        buf = bytearray()
        columns_manifest = []
        for name, dt in incytr_index_columns:
            arr = np.ascontiguousarray(
                cols[name][perm],
                dtype=np.dtype("<" + dt[0] + dt[1]),
            )
            columns_manifest.append({
                "name": name,
                "type": dt,
                "bytes": int(arr.nbytes),
            })
            buf += arr.tobytes()
        raw_bin = bytes(buf)
        gz_bin = gzip.compress(raw_bin, compresslevel=6)
        index_fname = f"{donor}__{_INCYTR_INDEX_FILENAME}"
        with open(os.path.join(EDGE_SLICES_INCYTR_PATHWAYS_DIR, index_fname), "wb") as f:
            f.write(gz_bin)
        global_index = {
            "url": f"edge_slices/incytr_pathways/{index_fname}",
            "nrows": n_idx,
            "rank_by": "abs(PDS)",
            "byteorder": "little",
            "sender_vocab": senders_canonical,
            "receiver_vocab": receivers_canonical,
            "contrast_vocab": list(present_contrasts),
            "gene_vocab": idx_gene_vocab,
            "traj_label_vocab": list(_SIGN_VEC_LABELS),
            "label_states": ["", *_INCYTR_LABEL_VOCAB],
            "label_nodes": list(_INCYTR_LABEL_NODES),
            "score_columns": list(effective_score_cols),
            "columns": columns_manifest,
            "raw_bytes": len(raw_bin),
            "gzip_bytes": len(gz_bin),
        }
        print(
            f"  ({donor}) incytr global_index: {n_idx:,} rows × "
            f"{len(columns_manifest)} cols, {len(idx_gene_vocab):,} genes; "
            f"{len(raw_bin)/1e6:.1f} MB raw -> {len(gz_bin)/1e6:.1f} MB gz",
            flush=True,
        )

    # B-3 / B-4: per-grain backbone entity tables + heatmap tensors, keyed to
    # this donor's backbone parquets + donor-scoped output subdir (shard
    # filenames are `{sender}__{receiver}.parquet` with no context in the
    # name, so donors must not share a dir). Uses the Full-grain sender/
    # receiver/contrast vocabs so grid indices align across grains.
    backbone_grains = write_incytr_backbone_grains(
        backbone_pair_mode_dir=os.path.join(INCYTR_PAIR_MODE_TCELLS_DIR, donor, "backbone"),
        edge_slices_backbone_dir=os.path.join(EDGE_SLICES_INCYTR_BACKBONE_DIR, donor),
        unified_viewer_dir=UNIFIED_VIEWER_DIR,
        contrast_from_filename=_backbone_contrast_from_filename,
        senders_canonical=senders_canonical,
        receivers_canonical=receivers_canonical,
        present_contrasts=list(present_contrasts),
        contrast_to_idx=contrast_to_idx,
        schema_version=SCHEMA_VERSION,
    )

    # T-cell contrasts are of the form "<day>_d2" — expose the follow-up day
    # axis for the shared Incytr trend chart and controls.
    contrasts_list = list(present_contrasts)
    days = list(ordered_days)
    return {
        "donor": donor,
        "contrasts": contrasts_list,
        "diseases": days,
        "timepoints": days,
        "senders": senders_canonical,
        "receivers": receivers_canonical,
        "celltype_qc": celltype_qc,
        "celltype_pathway_qc": celltype_pathway_qc,
        "heatmap_counts": heatmap_counts,
        "heatmap_counts_signed": heatmap_counts_signed,
        "pathway_counts": pathway_counts,
        "present_pairs": sorted(present_pairs),
        "n_total_rows": total_rows,
        "pair_row_counts": pair_row_counts,
        "score_columns": list(effective_score_cols),
        "label_columns": list(_INCYTR_LABEL_COLS),
        "label_nodes": list(_INCYTR_LABEL_NODES),
        "label_vocab": list(_INCYTR_LABEL_VOCAB),
        "trajectory_summary": traj_summary,
        "direction_flag_columns": list(dir_flag_cols),
        "path_metric_columns": list(extra_path_cols),
        "top_instances": top_instances,
        "global_index": global_index,
        "gene_node_index_shard": _write_gene_node_index_shard(
            gene_node_index, EDGE_SLICES_INCYTR_PATHWAYS_DIR,
            f"{donor}__gene_node_index.json.gz",
        ),
        "slice_index": {
            "schema_version": SCHEMA_VERSION,
            "filename_template": "{context}__{sender}__{receiver}.parquet",
            "sanitize_rule": "replace('/', '-'); replace(' ', '_'); replace('.', '')",
            "present": sorted(present_pairs),
            "n_total_rows": total_rows,
            "pair_row_counts": pair_row_counts,
        },
        # B-3 / B-4: per-grain backbone entity tables + heatmap tensors.
        # Absent when backbone parquets have not been produced for this donor.
        **({"backbone_grains": backbone_grains} if backbone_grains else {}),
    }


_TERMINAL_CONTRAST_RE = re.compile(r"^D\d+_(d\d+)_vs_(d\d+)$")


def _terminal_contrast_to_row(contrast: str) -> str:
    """`D1_d13_vs_d2` -> `d13_d2`, the pathways-row contrast vocabulary.

    The backend terminal edges carry the donor-prefixed MEA contrast; the
    pathways table rows key on the filename-derived `d<x>_d<y>` form
    (`_contrast_from_filename`). The sidechain tab joins on exact equality, so
    the shard must speak the row vocabulary. Unrecognized values pass through.
    """
    m = _TERMINAL_CONTRAST_RE.match(str(contrast))
    return f"{m.group(1)}_{m.group(2)}" if m else str(contrast)


def _build_terminal_kinase_lookup(
    donor: str,
) -> dict[str, dict[tuple[str, str, str], int]]:
    """Distinct-kinase count per (contrast, role, node gene) for each receiver.

    Keyed ``lut[owning_cluster][(contrast_row, role, gene)] -> n_distinct_kinase``,
    counting terminal edges exactly as the sidechain graph draws them
    (`incytr_sidechains.js`): one edge per (kinase, node) when the edge's
    contrast, role-node gene, and owning_cluster all match the pathway row.
    Kinases attach only at Target/EM/Receptor, so owning_cluster is the
    receiver. Empty for donors with no within-cohort kinase attribution.
    """
    import duckdb

    te_path = os.path.join(
        _KINASE_SIDECHAIN_EDGE_DIR, f"tcells_{donor}", "terminal_edges.csv"
    )
    if not os.path.exists(te_path):
        return {}
    rows = duckdb.sql(f"""
        SELECT owning_cluster, contrast, role, target_gene,
               COUNT(DISTINCT kinase) AS n_kin
        FROM read_csv_auto('{te_path}')
        GROUP BY owning_cluster, contrast, role, target_gene
    """).fetchall()
    lut: dict[str, dict[tuple[str, str, str], int]] = {}
    for owning_cluster, contrast, role, gene, n_kin in rows:
        crow = _terminal_contrast_to_row(str(contrast))
        lut.setdefault(str(owning_cluster), {})[
            (crow, str(role), str(gene))
        ] = int(n_kin)
    return lut


def _write_tcell_sidechain_slices() -> None:
    """Write the donor1-only kinase-sidechain shard for the T-cell viewer.

    Donor2 has no within-cohort kinase attribution and therefore no backend
    motif source or sidechain artifact. It is absent rather than represented by
    an empty shard.
    """
    donor = "donor1"
    _write_incytr_sidechain_slice(
        os.path.join(_KINASE_SIDECHAIN_EDGE_DIR, "tcells_donor1"),
        EDGE_SLICES_INCYTR_PATHWAYS_DIR,
        f"{donor}__sidechains.json.gz",
        donor,
        contrast_transform=_terminal_contrast_to_row,
    )


def _write_tcell_pair_pathways() -> dict:
    """Wipe and recreate the shard directory, then process each donor in turn."""
    shutil.rmtree(EDGE_SLICES_INCYTR_PATHWAYS_DIR, ignore_errors=True)
    os.makedirs(EDGE_SLICES_INCYTR_PATHWAYS_DIR, exist_ok=True)
    shutil.rmtree(EDGE_SLICES_INCYTR_BACKBONE_DIR, ignore_errors=True)
    os.makedirs(EDGE_SLICES_INCYTR_BACKBONE_DIR, exist_ok=True)

    by_context: dict[str, dict] = {}
    all_senders: set[str] = set()
    all_receivers: set[str] = set()
    all_contrasts: set[str] = set()
    for donor in DONORS:
        block = _write_donor_pair_pathways(donor)
        if block is None:
            continue
        by_context[donor] = block
        all_senders.update(block["senders"])
        all_receivers.update(block["receivers"])
        all_contrasts.update(block["contrasts"])

    # Single index.json listing every donor-scoped shard.
    index = {
        "schema_version": SCHEMA_VERSION,
        "filename_template": "{context}__{sender}__{receiver}.parquet",
        "sanitize_rule": "replace('/', '-'); replace(' ', '_'); replace('.', '')",
        "contexts": sorted(by_context.keys()),
        "by_context": {
            d: dict(block["slice_index"])
            for d, block in by_context.items()
        },
    }
    with open(os.path.join(EDGE_SLICES_INCYTR_PATHWAYS_DIR, "index.json"), "w") as f:
        json.dump(index, f)

    # Merged score-column contract = base + any optional PTM channel active for
    # at least one donor (self-gates to base-5 for the current PTM-free cohort).
    agg_score_cols = list(_INCYTR_SCORE_COLS_BASE) + [
        c for c in _INCYTR_SCORE_COLS_OPTIONAL
        if any(c in b.get("score_columns", []) for b in by_context.values())
    ]

    return {
        "schema_version": SCHEMA_VERSION,
        "version": 4,   # v4: backbone_grains (B-3/B-4)
        "source": f"pair_mode_tcells ({os.path.relpath(INCYTR_PAIR_MODE_TCELLS_DIR, config.REPO_ROOT)})",
        "source_mode": "pair_mode_tcells",
        "donors": sorted(by_context.keys()),
        "contexts": sorted(by_context.keys()),
        "by_context": by_context,
        "senders": sorted(all_senders),
        "receivers": sorted(all_receivers),
        "contrasts": sorted(all_contrasts, key=lambda x: int(x.split("_")[0][1:])),
        "score_columns": agg_score_cols,
        "label_columns": list(_INCYTR_LABEL_COLS),
        "label_nodes": list(_INCYTR_LABEL_NODES),
        "label_vocab": list(_INCYTR_LABEL_VOCAB),
        "slice_index": index,
    }

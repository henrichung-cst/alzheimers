"""Incytr binary filter-index format: vocab constants + the pure per-pair encoders shared by
both viewer builders' shard writers (phase 5d-2). One source of truth for the `.bin.gz` layout
decoded by incytr_global_index.js."""
import glob
import gzip
import json
import os
import shutil
import sys
from typing import Callable

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from alz.shared import config
from alz.viewer.shared.payload_helpers import (
    _INCYTR_FC_NODES,
    _configure_duckdb_tempdir,
)


def incytr_celltype_groups(names: set[str]) -> dict | None:
    """Cluster → coarse-group map for the heatmap's axis-grouping filter.

    Rolls the native Levy-spine clusters up to their WMB **tissue category** via
    the canonical crosswalk (`cluster_to_wmb_class.csv` → `CLASS_TO_TISSUE_CATEGORY`),
    e.g. the inhibitory-neuron subtypes collapse to "Interneurons". Shared by every
    cohort whose Incytr axes are Levy-spine cluster names (Song + 5xFAD). Heatmap-only:
    the table tab keeps the native spine. Returns None when the crosswalk does not
    cover every axis name (partial grouping is omitted, not faked), so the selector
    self-gates for cohorts with a foreign vocabulary (t-cell)."""
    try:
        cluster_to_class = config.load_cluster_to_wmb_class_map()
    except (FileNotFoundError, OSError):
        return None
    tissue = config.CLASS_TO_TISSUE_CATEGORY
    by_name: dict[str, str] = {}
    for n in names:
        wmb = cluster_to_class.get(n)
        cat = tissue.get(wmb) if wmb else None
        if not cat:
            return None   # incomplete crosswalk → omit the feature entirely
        by_name[n] = cat
    present = set(by_name.values())
    order = [c for c in config.TISSUE_ORDER if c in present]
    order += sorted(present - set(order))   # any category outside TISSUE_ORDER, stable
    return {"tissue": by_name, "tissue_order": order}

_INCYTR_LABEL_NODES = _INCYTR_FC_NODES
_INCYTR_LABEL_COLS = tuple(f"{n}_label" for n in _INCYTR_LABEL_NODES)
_INCYTR_LABEL_VOCAB = ("DEG", "prG", "KsG")

# Base score columns — always emitted by every cohort.
_INCYTR_SCORE_COLS_BASE = ("TPDS", "PPDS", "PhPDS_ps", "PhPDS_py", "SiK_score")
# Optional PTM-track score columns — only emitted by cohorts that ran the
# corresponding assay (5xFAD acetylation / ubiquitination).  Surfaced in
# score_columns and the binary index ONLY when the source parquets have at
# least one non-zero value; all-zero columns are never shipped (honesty rule).
_INCYTR_SCORE_COLS_OPTIONAL = ("Ack_score", "KGG_score", "Rme1_score")

# ---------------------------------------------------------------------------
# Backbone grain definitions (shared by all cohort builders).
# ---------------------------------------------------------------------------
# Surviving node columns per grain.  "Full" (the existing pathway grain)
# uses all four nodes and is handled by the existing pathway builder.
BACKBONE_GRAIN_NODES: dict[str, tuple[str, ...]] = {
    "R-EM":   ("Receptor", "EM"),
    "L-R-EM": ("Ligand", "Receptor", "EM"),
    "R-EM-T": ("Receptor", "EM", "Target"),
}
# inline  = global binary index only (all rows ship as one file; no per-pair shards).
# sharded = global binary index (Top mode) + per-(sender,receiver) parquet shards
#           (Cell Type mode).  R-EM-T is sharded because 2.78M rows at full scale
#           cannot be loaded in a single fetch for a per-pair drill-down.
BACKBONE_GRAIN_MODE: dict[str, str] = {
    "R-EM":   "inline",
    "L-R-EM": "inline",
    "R-EM-T": "sharded",
}
# Backbone binary index sidecar filename (parallel to _INCYTR_INDEX_FILENAME
# for Full pathways; kept distinct to avoid collision in the same output dir).
_BACKBONE_INDEX_FILENAME = "incytr_backbone_index.bin.gz"

# FC metric suffixes emitted by the incytr driver (one per node × assay channel).
_INCYTR_FC_METRICS = ("sclog2FC", "pr_log2FC", "ps_log2FC", "py_log2FC",
                      "Ack_log2FC", "KGG_log2FC")
_INCYTR_FC_COLS = tuple(
    f"{node}_{metric}" for node in _INCYTR_FC_NODES for metric in _INCYTR_FC_METRICS
)
# Label source column → canonical label column rename (raw driver → viewer payload).
_INCYTR_LABEL_SRC = tuple(f"{n}.label" for n in _INCYTR_LABEL_NODES)

# Pre-aggregation threshold grids — user input is snapped to the nearest entry.
_INCYTR_PATHWAY_PVALUES = (0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0)
_INCYTR_PATHWAY_ABS_PDS = (0.0, 0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0)

# On-disk sidecar filenames (same for every cohort).
_INCYTR_INDEX_FILENAME = "incytr_index.bin.gz"
_INCYTR_GENE_NODE_INDEX_FILENAME = "gene_node_index.json.gz"

_SIGN_VEC_LABELS = (
    "always-up",        # uuu — every PDS > 0
    "always-down",      # ddd — every PDS < 0
    "monotonic-up",     # PDS[2] < PDS[4] < PDS[6] (strictly)
    "monotonic-down",   # PDS[2] > PDS[4] > PDS[6] (strictly)
    "mixed",            # sign changes across timepoints
)


def _active_optional_score_cols(
    src_cols: set[str],
    con,
    view_name: str = "src",
) -> tuple[str, ...]:
    """Return optional score columns that are present in src AND have at least
    one non-zero row.  All-zero columns are excluded (honesty rule — never ship
    an empty score channel for a cohort that didn't run the assay)."""
    active: list[str] = []
    for c in _INCYTR_SCORE_COLS_OPTIONAL:
        if c not in src_cols:
            continue
        n = con.execute(
            f'SELECT COUNT(*) FROM {view_name} WHERE "{c}" IS NOT NULL AND "{c}" != 0'
        ).fetchone()[0]
        if n > 0:
            active.append(c)
    return tuple(active)


def _idx_label_bits(frame) -> np.ndarray:
    # 2 bits per node (Ligand/Receptor/EM/Target): 0=none, 1=DEG, 2=prG.
    bits = np.zeros(len(frame), dtype="<u1")
    for shift, col in zip((0, 2, 4, 6), _INCYTR_LABEL_COLS):
        if col not in frame.columns:
            continue
        c = frame[col]
        codes = (c.cat.codes.to_numpy()
                 if isinstance(c.dtype, pd.CategoricalDtype)
                 else pd.Categorical(c, categories=_INCYTR_LABEL_VOCAB).codes)
        # cat.codes: -1 NaN, 0 DEG, 1 prG → +1 → 0/1/2
        bits |= ((codes + 1).astype("<u1") << shift).astype("<u1")
    return bits


def _idx_traj_bits(series) -> np.ndarray:
    s = series.fillna("").astype(str)
    bits = np.zeros(len(s), dtype="<u1")
    for i, label in enumerate(_SIGN_VEC_LABELS):  # exact tokens, no collisions
        bits |= (s.str.contains(label, regex=False).to_numpy().astype("<u1") << i)
    return bits


def _backbone_sanitize(name: str) -> str:
    """Shard-filename sanitize (matches alz/integration/load.R:sanitize_celltype)."""
    return name.replace("/", "-").replace(" ", "_")


def write_incytr_backbone_grains(
    *,
    backbone_pair_mode_dir: str,
    edge_slices_backbone_dir: str,
    unified_viewer_dir: str,
    contrast_from_filename: Callable[[str], "str | None"],
    senders_canonical: list[str],
    receivers_canonical: list[str],
    present_contrasts: list[str],
    contrast_to_idx: dict[str, int],
    schema_version,
) -> dict:
    """Build B-3 heatmap count tensors + B-4 entity payload for all backbone grains.

    Cohort-agnostic: every cohort/tissue supplies its own backbone source dir,
    output dir, contrast-from-filename parser, and canonical sender/receiver/
    contrast vocabs (established from the Full pathway grain so grid indices
    align across grains).  Called from each cohort's pathway builder after the
    Full pathway view is built.

    Parameters:
      backbone_pair_mode_dir  — source dir holding one subdir per grain
                                (R-EM/, L-R-EM/, R-EM-T/) of *_backbone_output.parquet.
      edge_slices_backbone_dir — output root; one subdir per grain is written here.
      unified_viewer_dir      — viewer root; global_index URLs are relative to it.
      contrast_from_filename  — maps a backbone parquet basename → contrast key
                                (cohort-specific; returns None to skip a file).
      schema_version          — stamped into the sharded-grain slice index.

    Returns a dict mapping grain key → payload block for each grain that has data.
    Grains with no backbone parquets are silently omitted.

    Memory safety:
      - R-EM and L-R-EM are "inline" (all rows materialized; ~20k rows at full scale).
      - R-EM-T is "sharded" (streaming pass; ~2.78M rows at full scale).
    All heavy I/O uses DuckDB with a spill directory cap.
    """
    if not os.path.isdir(backbone_pair_mode_dir):
        print(f"  backbone_grains: backbone dir not found "
              f"({backbone_pair_mode_dir}); skipping", flush=True)
        return {}

    import duckdb as _duckdb

    result: dict[str, dict] = {}
    n_s = len(senders_canonical)
    n_r = len(receivers_canonical)
    n_c = len(present_contrasts)
    n_thr = len(_INCYTR_PATHWAY_PVALUES)
    n_ap = len(_INCYTR_PATHWAY_ABS_PDS)
    sender_to_idx_g = {s: i for i, s in enumerate(senders_canonical)}
    receiver_to_idx_g = {r: i for i, r in enumerate(receivers_canonical)}

    for grain, grain_nodes in BACKBONE_GRAIN_NODES.items():
        grain_dir = os.path.join(backbone_pair_mode_dir, grain)
        if not os.path.isdir(grain_dir):
            continue

        parquet_files = sorted(
            glob.glob(os.path.join(grain_dir, "*_backbone_output.parquet"))
        )
        if not parquet_files:
            continue

        # Pair each file with its contrast (skip if contrast not in canonical list).
        file_to_contrast_bb: list[tuple[str, str]] = []
        for fpath in parquet_files:
            c = contrast_from_filename(os.path.basename(fpath))
            if c is not None and c in contrast_to_idx:
                file_to_contrast_bb.append((fpath, c))
        if not file_to_contrast_bb:
            continue

        grain_mode = BACKBONE_GRAIN_MODE[grain]
        print(f"  backbone_grains: grain={grain}  parquets={len(file_to_contrast_bb)}"
              f"  mode={grain_mode}", flush=True)

        grain_con = _duckdb.connect()
        grain_con.execute("PRAGMA threads=4; PRAGMA memory_limit='8GB';")
        _configure_duckdb_tempdir(grain_con)

        # Schema detection from first file.
        first_schema = pq.read_schema(file_to_contrast_bb[0][0])
        bb_src_cols = {f.name for f in first_schema}
        # Optional score cols present in the backbone parquets' schema.
        optional_in_schema_bb = [c for c in _INCYTR_SCORE_COLS_OPTIONAL
                                  if c in bb_src_cols]
        base_nonsik_bb = [c for c in _INCYTR_SCORE_COLS_BASE if c != "SiK_score"]

        # Build per-file SELECT clauses unioning into bb_src VIEW.
        bb_selects: list[str] = []
        for fpath, contrast in file_to_contrast_bb:
            sch = pq.read_schema(fpath)
            names = {f.name for f in sch}

            # SiK_score: collapse to disease arm (consistent with pathway builder).
            sik_col = next(
                (n for n in names
                 if n.startswith("SiK_score_") and not n.endswith("_WTyp")),
                None,
            )
            sik_clause = (
                f'CAST("{sik_col}" AS DOUBLE) AS SiK_score' if sik_col
                else "CAST(NULL AS DOUBLE) AS SiK_score"
            )

            # Base non-SiK scores: present → cast; absent → NULL-fill.
            present_base = [c for c in base_nonsik_bb if c in names]
            missing_base = [c for c in base_nonsik_bb if c not in names]
            parts: list[str] = []
            if present_base:
                parts.append(",\n              ".join(
                    f"CAST({c} AS DOUBLE) AS {c}" for c in present_base
                ))
            if missing_base:
                parts.append(",\n              ".join(
                    f"CAST(NULL AS DOUBLE) AS {c}" for c in missing_base
                ))
            parts.append(sik_clause)
            # Optional cols: present → cast; absent → exclude (no NULL fill).
            opt_present = [c for c in optional_in_schema_bb if c in names]
            if opt_present:
                parts.append(",\n              ".join(
                    f"CAST({c} AS DOUBLE) AS {c}" for c in opt_present
                ))
            extra_select_bb = ",\n              ".join(p for p in parts if p)

            bb_selects.append(f"""
            SELECT
              "Sender.group" AS sender,
              "Receiver.group" AS receiver,
              Ligand, Receptor, EM, Target,
              '{contrast}' AS contrast,
              CAST(PDS AS DOUBLE) AS PDS,
              CAST(n_paths AS INTEGER) AS n_paths,
              {extra_select_bb}
            FROM read_parquet('{fpath}')
            """)

        union_bb = "\nUNION ALL\n".join(bb_selects)
        grain_con.execute(f"CREATE VIEW bb_src AS {union_bb}")

        n_bb_total = grain_con.execute("SELECT COUNT(*) FROM bb_src").fetchone()[0]
        print(f"    {grain}: {n_bb_total:,} rows", flush=True)
        if n_bb_total == 0:
            grain_con.close()
            continue

        # B-5: detect active optional score cols for this grain (non-zero check).
        active_opt_bb = _active_optional_score_cols(
            set(optional_in_schema_bb), grain_con, "bb_src"
        )
        grain_score_cols: tuple[str, ...] = _INCYTR_SCORE_COLS_BASE + active_opt_bb
        if active_opt_bb:
            print(f"    {grain}: optional channels active: {list(active_opt_bb)}",
                  flush=True)

        # B-3: heatmap count tensors.
        # Backbone has no pvalue → all pvalue-threshold bands give the same count;
        # the tensor keeps the same shape as Full so the JS can reuse one decoder.
        hm_thr_clauses_list = []
        for ip, _tp in enumerate(_INCYTR_PATHWAY_PVALUES):
            for iap, tap in enumerate(_INCYTR_PATHWAY_ABS_PDS):
                hm_thr_clauses_list.append(
                    f"COUNT(*) FILTER (WHERE COALESCE(ABS(PDS), 0) >= {tap}) AS c_{ip}_{iap}"
                )
        hm_thr_clauses = ", ".join(hm_thr_clauses_list)

        hm_rows = grain_con.execute(f"""
            SELECT sender, receiver, contrast, {hm_thr_clauses}
            FROM bb_src
            GROUP BY sender, receiver, contrast
        """).fetchall()
        grid = np.zeros((n_s, n_r, n_c, n_thr, n_ap), dtype=np.uint32)
        for row in hm_rows:
            s_raw, r_raw, c_val = row[0], row[1], row[2]
            if s_raw not in sender_to_idx_g or r_raw not in receiver_to_idx_g:
                continue
            if c_val not in contrast_to_idx:
                continue
            s_i = sender_to_idx_g[s_raw]
            r_i = receiver_to_idx_g[r_raw]
            c_i = contrast_to_idx[c_val]
            offset = 3
            for ip in range(n_thr):
                for iap in range(n_ap):
                    grid[s_i, r_i, c_i, ip, iap] = int(row[offset])
                    offset += 1

        totals = np.zeros((n_thr, n_ap), dtype=np.uint64)
        for ip in range(n_thr):
            for iap in range(n_ap):
                totals[ip, iap] = int(grid[:, :, :, ip, iap].sum())

        heatmap_counts_bb = {
            "thresholds": list(_INCYTR_PATHWAY_PVALUES),
            "abs_pds_thresholds": list(_INCYTR_PATHWAY_ABS_PDS),
            "shape": [n_s, n_r, n_c, n_thr, n_ap],
            "counts": grid.flatten().tolist(),
            "total_by_threshold": totals.tolist(),
        }

        hm_signed_rows = grain_con.execute(f"""
            SELECT sender, receiver, contrast,
                   CASE WHEN PDS > 0 THEN 2 WHEN PDS < 0 THEN 0 ELSE 1 END AS s,
                   {hm_thr_clauses}
            FROM bb_src
            GROUP BY sender, receiver, contrast, s
        """).fetchall()
        signed_grid = np.zeros((n_s, n_r, n_c, 3, n_thr, n_ap), dtype=np.uint32)
        for row in hm_signed_rows:
            s_raw, r_raw, c_val, sign_i = row[0], row[1], row[2], int(row[3])
            if s_raw not in sender_to_idx_g or r_raw not in receiver_to_idx_g:
                continue
            if c_val not in contrast_to_idx:
                continue
            s_i = sender_to_idx_g[s_raw]
            r_i = receiver_to_idx_g[r_raw]
            c_i = contrast_to_idx[c_val]
            offset = 4
            for ip in range(n_thr):
                for iap in range(n_ap):
                    signed_grid[s_i, r_i, c_i, sign_i, ip, iap] = int(row[offset])
                    offset += 1

        signed_totals = np.zeros((3, n_thr, n_ap), dtype=np.uint64)
        for sign_i in range(3):
            for ip in range(n_thr):
                for iap in range(n_ap):
                    signed_totals[sign_i, ip, iap] = int(
                        signed_grid[:, :, :, sign_i, ip, iap].sum()
                    )

        heatmap_counts_signed_bb = {
            "thresholds": list(_INCYTR_PATHWAY_PVALUES),
            "abs_pds_thresholds": list(_INCYTR_PATHWAY_ABS_PDS),
            "shape": [n_s, n_r, n_c, 3, n_thr, n_ap],
            "counts": signed_grid.flatten().tolist(),
            "total_by_sign_threshold": signed_totals.tolist(),
            "sign_source": "PDS",
        }
        del grid, signed_grid  # free numpy arrays

        # B-4: entity payload — binary index (all grains) + shard files (R-EM-T only).
        grain_out_dir = os.path.join(edge_slices_backbone_dir, grain)
        shutil.rmtree(grain_out_dir, ignore_errors=True)
        os.makedirs(grain_out_dir, exist_ok=True)

        # Surviving node columns for this grain.
        node_src_cols = [n for n in ("Ligand", "Receptor", "EM", "Target")
                         if n in grain_nodes]
        # ID column names: lowercase node name + "Id" (receptorId, emId, …).
        node_id_cols = [f"{n.lower()}Id" for n in node_src_cols]

        # Binary index column layout (f4 → u2 → u1 for alignment).
        BB_INDEX_COLUMNS: list[tuple[str, str]] = (
            [("PDS", "f4"), ("n_paths", "f4")]
            + [(sc, "u2") for sc in grain_score_cols]
            + [(col, "u2") for col in node_id_cols]
            + [("senderId", "u1"), ("receiverId", "u1"), ("contrastId", "u1")]
        )

        bb_gene_to_id: dict[str, int] = {}
        bb_gene_vocab: list[str] = []
        bb_idx_chunks: list[dict] = []

        def _bb_gene_id(val: object) -> int:
            s = str(val) if val is not None and str(val) != "None" else ""
            if s not in bb_gene_to_id:
                bb_gene_to_id[s] = len(bb_gene_vocab)
                bb_gene_vocab.append(s)
            return bb_gene_to_id[s]

        def _bb_accumulate(frame: "pd.DataFrame",
                           s_idx: int, r_idx: int) -> None:
            n = len(frame)
            if n == 0:
                return
            chunk: dict = {
                "PDS":        frame["PDS"].to_numpy(dtype="<f4"),
                "n_paths":    frame["n_paths"].fillna(0).to_numpy(dtype="<f4"),
                "senderId":   np.full(n, s_idx, dtype="<u1"),
                "receiverId": np.full(n, r_idx, dtype="<u1"),
                "contrastId": frame["contrast"].map(contrast_to_idx).to_numpy(dtype="<u1"),
            }
            for sc in grain_score_cols:
                chunk[sc] = (
                    frame[sc].to_numpy(dtype="float16").view("<u2")
                    if sc in frame.columns
                    else np.zeros(n, dtype="<u2")
                )
            for node_col, id_col in zip(node_src_cols, node_id_cols):
                ids = np.array(
                    [_bb_gene_id(v) for v in frame[node_col]],
                    dtype="<u2",
                )
                chunk[id_col] = ids
            bb_idx_chunks.append(chunk)

        # Shard select cols: surviving nodes + per-entity scalars.
        bb_shard_select = (
            list(grain_nodes)
            + ["contrast", "PDS", "n_paths"]
            + list(grain_score_cols)
        )

        total_bb_rows = 0
        present_pairs_bb: list[list[str]] = []
        pair_row_counts_bb: dict[str, int] = {}

        if grain_mode == "sharded":
            # Streaming pass: ORDER BY sender, receiver to flush at pair boundaries.
            bb_stream_cols = ["sender", "receiver"] + bb_shard_select
            reader = grain_con.execute(
                f"SELECT {', '.join(bb_stream_cols)} FROM bb_src"
                f" ORDER BY sender, receiver"
            ).fetch_record_batch(500_000)

            cur_s_bb: str | None = None
            cur_r_bb: str | None = None
            buf_bb: list["pd.DataFrame"] = []

            def _bb_flush(key: tuple[str, str],
                          frames: list["pd.DataFrame"]) -> None:
                nonlocal total_bb_rows
                if not frames:
                    return
                s_raw, r_raw = key
                sub = pd.concat(frames, ignore_index=True, copy=False)
                s_idx = sender_to_idx_g.get(s_raw, 0)
                r_idx = receiver_to_idx_g.get(r_raw, 0)
                _bb_accumulate(sub, s_idx, r_idx)
                # Float16 compress score cols + PDS for parquet storage.
                for c in (["PDS"] + list(grain_score_cols)):
                    if c in sub.columns:
                        sub[c] = sub[c].astype("float16")
                fname = f"{_backbone_sanitize(s_raw)}__{_backbone_sanitize(r_raw)}.parquet"
                fpath_out = os.path.join(grain_out_dir, fname)
                pq.write_table(
                    pa.Table.from_pandas(sub, preserve_index=False),
                    fpath_out, compression="zstd",
                )
                present_pairs_bb.append([s_raw, r_raw])
                pair_row_counts_bb[fname] = len(sub)
                total_bb_rows += len(sub)

            for batch in reader:
                bdf = batch.to_pandas()
                senders_arr = bdf["sender"].to_numpy()
                receivers_arr = bdf["receiver"].to_numpy()
                starts = [0]
                for i in range(1, len(senders_arr)):
                    if (senders_arr[i] != senders_arr[i - 1]
                            or receivers_arr[i] != receivers_arr[i - 1]):
                        starts.append(i)
                starts.append(len(senders_arr))
                for j in range(len(starts) - 1):
                    a, b = starts[j], starts[j + 1]
                    s_val = senders_arr[a]
                    r_val = receivers_arr[a]
                    seg = bdf.iloc[a:b].drop(columns=["sender", "receiver"])
                    if cur_s_bb is None:
                        cur_s_bb, cur_r_bb = s_val, r_val
                    elif s_val != cur_s_bb or r_val != cur_r_bb:
                        _bb_flush((cur_s_bb, cur_r_bb), buf_bb)
                        buf_bb = []
                        cur_s_bb, cur_r_bb = s_val, r_val
                    buf_bb.append(seg)
            if buf_bb and cur_s_bb is not None and cur_r_bb is not None:
                _bb_flush((cur_s_bb, cur_r_bb), buf_bb)

        else:
            # Inline mode: small grain — single materialise pass.
            bb_all = grain_con.execute(
                f"SELECT sender, receiver, {', '.join(bb_shard_select)} FROM bb_src"
            ).fetchdf()
            total_bb_rows = len(bb_all)
            for (s_raw, r_raw), group in bb_all.groupby(
                    ["sender", "receiver"], sort=False):
                s_idx = sender_to_idx_g.get(s_raw, 0)
                r_idx = receiver_to_idx_g.get(r_raw, 0)
                _bb_accumulate(group.drop(columns=["sender", "receiver"]), s_idx, r_idx)

        grain_con.close()

        # Build and write global binary index for this grain.
        global_index_bb: dict | None = None
        if bb_idx_chunks:
            assert sys.byteorder == "little", "backbone index assumes little-endian"
            bb_cols_map = {
                name: np.concatenate([c[name] for c in bb_idx_chunks])
                for name, _dt in BB_INDEX_COLUMNS
            }
            bb_idx_chunks.clear()
            n_bb_idx = int(len(bb_cols_map["PDS"]))
            perm_bb = np.argsort(-np.abs(bb_cols_map["PDS"]), kind="stable")
            buf_bb_bin = bytearray()
            bb_columns_manifest: list[dict] = []
            for name, dt in BB_INDEX_COLUMNS:
                arr = np.ascontiguousarray(
                    bb_cols_map[name][perm_bb],
                    dtype=np.dtype("<" + dt[0] + dt[1]),
                )
                bb_columns_manifest.append({
                    "name": name, "type": dt, "bytes": int(arr.nbytes)
                })
                buf_bb_bin += arr.tobytes()
            del bb_cols_map
            raw_bb_bin = bytes(buf_bb_bin)
            gz_bb_bin = gzip.compress(raw_bb_bin, compresslevel=6)
            idx_path = os.path.join(grain_out_dir, _BACKBONE_INDEX_FILENAME)
            with open(idx_path, "wb") as f:
                f.write(gz_bb_bin)
            url_prefix = (
                os.path.relpath(grain_out_dir, unified_viewer_dir).replace(os.sep, "/")
                + "/"
            )
            global_index_bb = {
                "url": f"{url_prefix}{_BACKBONE_INDEX_FILENAME}",
                "nrows": n_bb_idx,
                "rank_by": "abs(PDS)",
                "byteorder": "little",
                "sender_vocab": senders_canonical,
                "receiver_vocab": receivers_canonical,
                "contrast_vocab": list(present_contrasts),
                "gene_vocab": bb_gene_vocab,
                "node_id_columns": node_id_cols,
                "score_columns": list(grain_score_cols),
                "columns": bb_columns_manifest,
                "raw_bytes": len(raw_bb_bin),
                "gzip_bytes": len(gz_bb_bin),
            }
            print(
                f"    {grain} index: {n_bb_idx:,} rows × "
                f"{len(bb_columns_manifest)} cols; "
                f"{len(raw_bb_bin) / 1e6:.1f} MB raw → "
                f"{len(gz_bb_bin) / 1e6:.1f} MB gz",
                flush=True,
            )

        grain_payload: dict = {
            "grain": grain,
            "nodes": list(grain_nodes),
            "mode": grain_mode,
            "heatmap_counts": heatmap_counts_bb,
            "heatmap_counts_signed": heatmap_counts_signed_bb,
            "score_columns": list(grain_score_cols),
            "global_index": global_index_bb,
        }

        if grain_mode == "sharded" and present_pairs_bb:
            shard_index = {
                "schema_version": schema_version,
                "filename_template": "{sender}__{receiver}.parquet",
                "sanitize_rule": "replace('/', '-'); replace(' ', '_')",
                "present": sorted(present_pairs_bb),
                "n_total_rows": total_bb_rows,
                "pair_row_counts": pair_row_counts_bb,
            }
            with open(os.path.join(grain_out_dir, "index.json"), "w") as f:
                json.dump(shard_index, f)
            grain_payload["slice_index"] = shard_index

        result[grain] = grain_payload
        print(f"    {grain}: done ({total_bb_rows:,} entity rows total)", flush=True)

    if result:
        print(f"  backbone_grains: built {len(result)} grain(s): "
              f"{sorted(result)}", flush=True)
    else:
        print(f"  backbone_grains: no backbone data found in "
              f"{backbone_pair_mode_dir}", flush=True)

    return result

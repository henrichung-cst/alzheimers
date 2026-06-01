#!/usr/bin/env python3
"""T-cell viewer builder: single-file HTML deliverable for the T-cell cohort.

Reads the T-cell bulk MEA (donor1 only; donor2 has no IMAC) and the per-donor
pair-mode Incytr wide outputs (`outputs/reports/incytr_pair_mode_tcells/`).
Emits `outputs/reports/tcell_viewer/index.html` with the columnar payload
inlined as `<script type="application/json" id="payload-data">` plus per-pair
parquet shards under `edge_slices/incytr_pathways/` fetched on demand.

The mouse-cohort and human-cohort builders live at `alz/build_unified_viewer.py`.
This is a fully independent builder — no shared code paths flag-gated on cohort.

Usage:
    python alz/build_tcell_viewer.py              # payload + html (default)
    python alz/build_tcell_viewer.py --summary    # input row counts
    python alz/build_tcell_viewer.py --payload    # JSON only
    python alz/build_tcell_viewer.py --html       # write HTML (needs payload)
    python alz/build_tcell_viewer.py --validate   # write report md
"""

from __future__ import annotations

import argparse
import glob
import gzip
import json
import os
import re
import resource
import shutil
import sys
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(HERE)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, HERE)

from alz.shared import config  # noqa: E402

from tcell_viewer.paths import (  # noqa: E402
    AUDIT_PREVIEW_ROWS,
    AUDIT_SOURCES_DIR,
    EDGE_SLICES_INCYTR_PATHWAYS_DIR,
    INCYTR_PAIR_MODE_TCELLS_DIR,
    KINASE_ATTRIBUTION_TCELLS_DIR,
    PAYLOAD_JSON,
    PAYLOAD_JSON_GZ,
    REPORT_MD,
    SCHEMA_VERSION,
    TCELLS_INCYTR_INPUTS_DIR,
    UNIFIED_VIEWER_DIR,
    UNIFIED_VIEWER_HTML,
)

# ---------------------------------------------------------------------------
# T-cell cohort constants
# ---------------------------------------------------------------------------
DONORS = ("donor1", "donor2")

# Per-donor MEA presence — donor2 had no stoichiometry matrix (no IMAC) and no
# pY motif → all four MEA variants skipped per the donor2 mea_manifest.json.
DONOR_WITH_MEA = ("donor1",)

# Sequential viridis progression replaces the mouse 3-disease palette.
# Sampled from the matplotlib viridis colormap at evenly spaced points.
TIMEPOINT_COLOR_MAP = {
    "d2":  "#440154",
    "d5":  "#482878",
    "d7":  "#3e4a89",
    "d9":  "#31688e",
    "d11": "#26828e",
    "d13": "#1f9e89",
    "d15": "#35b779",
    "d17": "#6dcd59",
    "d19": "#b4de2c",
    "d20": "#fde725",
}

_INCYTR_SCORE_COLS = ("TPDS", "PPDS", "PhPDS_ps", "PhPDS_py", "SiK_score")
_INCYTR_FC_NODES = ("Ligand", "Receptor", "EM", "Target")
_INCYTR_FC_METRICS = ("sclog2FC", "pr_log2FC", "ps_log2FC", "py_log2FC")
_INCYTR_FC_COLS = tuple(
    f"{node}_{metric}" for node in _INCYTR_FC_NODES for metric in _INCYTR_FC_METRICS
)
_INCYTR_LABEL_NODES = _INCYTR_FC_NODES
_INCYTR_LABEL_VOCAB = ("DEG", "prG")
_INCYTR_LABEL_SRC = tuple(f"{n}.label" for n in _INCYTR_LABEL_NODES)
_INCYTR_LABEL_COLS = tuple(f"{n}_label" for n in _INCYTR_LABEL_NODES)

# Pre-aggregated heatmap thresholds (matches mouse-cohort viewer contract).
_INCYTR_PATHWAY_PVALUES = (0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0)
_INCYTR_PATHWAY_ABS_PDS = (0.0, 0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0)


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _sanitize(obj: Any, decimals: int = 4):
    """JSON-safe: NaN/Inf -> None, numpy -> native, floats rounded."""
    if isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return round(obj, decimals)
    if isinstance(obj, dict):
        return {k: _sanitize(v, decimals) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v, decimals) for v in obj]
    if isinstance(obj, np.ndarray):
        return _sanitize(obj.tolist(), decimals)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        x = float(obj)
        if np.isnan(x) or np.isinf(x):
            return None
        return round(x, decimals)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if obj is pd.NA:
        return None
    return obj


def _peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _incytr_sanitize(name: str) -> str:
    """Match the upstream sanitize in alz/integration/load.R:sanitize_celltype."""
    return name.replace("/", "-").replace(" ", "_").replace(".", "")


def _configure_duckdb_tempdir(con) -> None:
    d = os.environ.get("DUCKDB_TEMP_DIR", os.path.expanduser("~/.cache/duckdb"))
    os.makedirs(d, exist_ok=True)
    con.execute(f"SET temp_directory='{d}';")
    con.execute("SET max_temp_directory_size='40GiB';")


def _count_csv_rows(path: str) -> int:
    with open(path, "rb") as f:
        n = sum(1 for _ in f)
    return max(0, n - 1)


_PAIR_FILE_RE = re.compile(r"(d\d+_d\d+)_incytr_output\.parquet$")


def _contrast_from_filename(fname: str) -> str | None:
    m = _PAIR_FILE_RE.search(fname)
    return m.group(1) if m else None


def _timepoint_label(contrast: str) -> str:
    """`d13_d2` → `d13` (the disease timepoint; d2 is the baseline)."""
    return contrast.split("_", 1)[0] if "_" in contrast else contrast


# ---------------------------------------------------------------------------
# Kinase slice (donor1 MEA only)
# ---------------------------------------------------------------------------

def _load_donor_kinase_attribution(donor: str) -> dict | None:
    """Load NES/FDR tables for one donor across {ST, pY} × {stoich, raw}.

    Returns None if no MEA outputs exist for the donor (donor2's expected case).
    """
    mea_dir = os.path.join(KINASE_ATTRIBUTION_TCELLS_DIR, donor, "mea")
    manifest_path = os.path.join(mea_dir, "mea_manifest.json")
    if not os.path.exists(manifest_path):
        return None
    with open(manifest_path) as f:
        manifest = json.load(f)
    if not manifest.get("mea_ran"):
        return None

    tracks: dict[str, dict] = {}
    for track_suffix, residue_label in (("", "ST"), ("_pY", "Y")):
        for variant in ("", "_raw"):
            nes_path = os.path.join(
                mea_dir, f"kinase_timepoint_nes{variant}{track_suffix}.csv"
            )
            fdr_path = os.path.join(
                mea_dir, f"kinase_timepoint_fdr{variant}{track_suffix}.csv"
            )
            if not (os.path.exists(nes_path) and os.path.exists(fdr_path)):
                continue
            nes_df = pd.read_csv(nes_path)
            fdr_df = pd.read_csv(fdr_path)
            key = f"{residue_label}{variant}"
            tracks[key] = {"nes": nes_df, "fdr": fdr_df}

    if not tracks:
        return None
    return {"manifest": manifest, "tracks": tracks}


def _build_donor_kinases_slice(donor: str) -> dict | None:
    """Columnar kinases table for one donor.

    Schema mirrors the mouse-cohort kinases slice: `id`, `name`, `gene_symbol`,
    `residue_type`, `trajectory`, `peak_contrast`, `peak_NES`, `n_sig_contrasts`,
    `NES_<contrast>`, `FDR_<contrast>` — with cell-type attribution fields
    omitted (no per-cluster MEA on T-cells).
    """
    attribution = _load_donor_kinase_attribution(donor)
    if attribution is None:
        return None

    # Build a unified long-form (kinase × contrast) table across tracks.
    # Default track is ST/stoich (residue=ST, variant="").
    primary = attribution["tracks"].get("ST") or next(iter(attribution["tracks"].values()))
    nes = primary["nes"].copy()
    fdr = primary["fdr"].copy()
    if "kinase" not in nes.columns:
        return None
    contrasts = [c for c in nes.columns if c != "kinase"]
    # MEA columns: `D1_d13` style — strip the donor prefix to align with the
    # pair-mode `d13_d2` vocabulary for the timepoint legend.
    contrast_to_short = {
        c: re.sub(r"^D\d+_", "", c) for c in contrasts
    }
    short_contrasts = [contrast_to_short[c] for c in contrasts]

    kinases = nes["kinase"].astype(str).tolist()
    kid = {k: i for i, k in enumerate(kinases)}

    cols: dict[str, list] = {
        "id": list(range(len(kinases))),
        "name": kinases,
        "gene_symbol": kinases,
        "residue_type": ["ST"] * len(kinases),
        "trajectory": [""] * len(kinases),
        "peak_contrast": [],
        "peak_NES": [],
        "n_sig_contrasts": [],
        "top_celltype_1": [""] * len(kinases),
    }
    for sc in short_contrasts:
        cols[f"NES_{sc}"] = []
        cols[f"FDR_{sc}"] = []

    nes_idx = nes.set_index("kinase")
    fdr_idx = fdr.set_index("kinase")
    fdr_thresh = float(attribution["manifest"].get("mea_fdr_thresh", 0.25))

    for k in kinases:
        nes_row = nes_idx.loc[k] if k in nes_idx.index else None
        fdr_row = fdr_idx.loc[k] if k in fdr_idx.index else None
        nes_vec, fdr_vec = [], []
        for c in contrasts:
            n_val = float(nes_row[c]) if nes_row is not None and pd.notna(nes_row[c]) else float("nan")
            f_val = float(fdr_row[c]) if fdr_row is not None and pd.notna(fdr_row[c]) else float("nan")
            nes_vec.append(n_val)
            fdr_vec.append(f_val)
        sc = short_contrasts
        for i, scn in enumerate(sc):
            cols[f"NES_{scn}"].append(nes_vec[i])
            cols[f"FDR_{scn}"].append(fdr_vec[i])
        # Peak: largest |NES| among contrasts with finite FDR.
        finite = [(i, nes_vec[i]) for i in range(len(contrasts))
                  if not (np.isnan(nes_vec[i]) or np.isnan(fdr_vec[i]))]
        if finite:
            i_peak = max(finite, key=lambda t: abs(t[1]))[0]
            cols["peak_contrast"].append(sc[i_peak])
            cols["peak_NES"].append(nes_vec[i_peak])
            cols["n_sig_contrasts"].append(
                int(sum(1 for j in range(len(contrasts))
                        if not np.isnan(fdr_vec[j]) and fdr_vec[j] < fdr_thresh))
            )
        else:
            cols["peak_contrast"].append("")
            cols["peak_NES"].append(float("nan"))
            cols["n_sig_contrasts"].append(0)

    return {
        "kinases_slice": cols,
        "kinase_names": kinases,
        "contrasts": short_contrasts,
        "kid": kid,
        "fdr_threshold": fdr_thresh,
    }


# ---------------------------------------------------------------------------
# Celltypes slice (ProjecTILs labels per donor)
# ---------------------------------------------------------------------------

def _load_donor_clusters(donor: str) -> list[str]:
    """Sorted ProjecTILs cluster names for one donor (alphanumeric-sanitized)."""
    pred_path = os.path.join(
        TCELLS_INCYTR_INPUTS_DIR, donor, "scrna", "projectils_predictions.csv"
    )
    if not os.path.exists(pred_path):
        return []
    df = pd.read_csv(pred_path, usecols=["functional.cluster"])
    raw = df["functional.cluster"].dropna().astype(str).unique().tolist()
    # The decompose pipeline writes the sanitized form to disk (no dots/dashes);
    # ProjecTILs predictions still carry the dotted source labels. Sanitize here
    # so the celltypes slice matches the pair-mode `Sender.group`/`Receiver.group`
    # values found in the wide parquets.
    return sorted(_incytr_sanitize(c) for c in raw)


def _build_celltypes_slice(donor_clusters: dict[str, list[str]]) -> dict:
    """Union of per-donor ProjecTILs cluster names; one row per cluster.

    The viewer's existing `celltypes` consumer indexes by id; per-donor
    membership is exposed via `available_donors` so the donor selector can
    hide empty clusters.
    """
    seen: dict[str, set[str]] = {}
    for donor, clusters in donor_clusters.items():
        for c in clusters:
            seen.setdefault(c, set()).add(donor)
    ordered = sorted(seen.keys())
    return {
        "id": list(range(len(ordered))),
        "name": ordered,
        "tissue_category": ["T-cell"] * len(ordered),
        "available_donors": [sorted(seen[c]) for c in ordered],
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

    selects = []
    has_pvalue = False
    for fpath, contrast in file_to_contrast:
        sch = pq.read_schema(fpath)
        names = {f.name for f in sch}
        pcol_disease = None
        for n in names:
            if n.startswith("p_value_") and not n.endswith("_WTyp"):
                pcol_disease = n
                has_pvalue = True
                break
        pcol_clause = f'CAST("{pcol_disease}" AS DOUBLE)' if pcol_disease else "CAST(NULL AS DOUBLE)"

        score_clauses = ",\n          ".join(
            f"CAST({c} AS DOUBLE) AS {c}" for c in _INCYTR_SCORE_COLS if c in names
        )
        missing_scores = [c for c in _INCYTR_SCORE_COLS if c not in names]
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
        clauses = [score_clauses, missing_score_clauses,
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

    senders_canonical = sorted({r[0] for r in con.execute(
        "SELECT DISTINCT sender FROM src").fetchall()})
    receivers_canonical = sorted({r[0] for r in con.execute(
        "SELECT DISTINCT receiver FROM src").fetchall()})
    sender_to_idx = {s: i for i, s in enumerate(senders_canonical)}
    receiver_to_idx = {r: i for i, r in enumerate(receivers_canonical)}
    n_s, n_r, n_c = len(senders_canonical), len(receivers_canonical), len(present_contrasts)
    print(f"  ({donor}) senders={n_s}, receivers={n_r}, contrasts={n_c}", flush=True)

    # Heatmap counts cube.
    n_thr = len(_INCYTR_PATHWAY_PVALUES)
    n_ap = len(_INCYTR_PATHWAY_ABS_PDS)
    pval_filter = (lambda tp: f"pvalue < {tp}") if has_pvalue else (lambda tp: "TRUE")
    pval_where = "WHERE pvalue IS NOT NULL" if has_pvalue else ""
    hm_clauses = ", ".join(
        f"COUNT(*) FILTER (WHERE {pval_filter(tp)} AND COALESCE(ABS(PDS), 0) >= {tap}) AS c_{ip}_{iap}"
        for ip, tp in enumerate(_INCYTR_PATHWAY_PVALUES)
        for iap, tap in enumerate(_INCYTR_PATHWAY_ABS_PDS)
    )
    hm_rows = con.execute(f"""
        SELECT sender, receiver, contrast, {hm_clauses}
        FROM src {pval_where} GROUP BY sender, receiver, contrast
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

    # Pathway counts cube (per contrast × sign × thresholds).
    thr_clauses = ", ".join(
        f"COUNT(*) FILTER (WHERE {pval_filter(tp)} AND COALESCE(ABS(PDS), 0) >= {tap}) AS c_{ip}_{iap}"
        for ip, tp in enumerate(_INCYTR_PATHWAY_PVALUES)
        for iap, tap in enumerate(_INCYTR_PATHWAY_ABS_PDS)
    )
    pathway_rows = con.execute(f"""
        SELECT contrast,
               CASE WHEN PDS > 0 THEN 2 WHEN PDS < 0 THEN 0 ELSE 1 END AS s,
               {thr_clauses}
        FROM src {pval_where} GROUP BY contrast, s
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
    pathway_counts = {
        "thresholds": list(_INCYTR_PATHWAY_PVALUES),
        "abs_pds_thresholds": list(_INCYTR_PATHWAY_ABS_PDS),
        "contrasts": list(present_contrasts),
        "counts": pathway_arr.flatten().tolist(),
        "shape": [n_c, 3, n_thr, n_ap],
        "sign_source": "PDS",
    }

    # Shard the long table per (sender, receiver) — donor-scoped output dir.
    shard_select_cols = (
        ["Ligand", "Receptor", "EM", "Target", "contrast", "pvalue", "PDS"]
        + list(_INCYTR_SCORE_COLS) + list(dir_flag_cols)
        + list(extra_path_cols) + list(_INCYTR_FC_COLS) + list(_INCYTR_LABEL_COLS)
    )
    float32_cols = ["pvalue"]
    float16_cols = (["PDS"] + list(_INCYTR_SCORE_COLS) + list(_INCYTR_FC_COLS)
                    + list(dir_flag_cols) + list(extra_path_cols))
    float_cols = float32_cols + float16_cols

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
        s, r = key
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

    # T-cell contrasts are of the form "<day>_d2" — derive the day axis once
    # so the JS doesn't have to parse contrast strings at render time. "Disease"
    # in the unified-viewer JS vocabulary maps to the variable day for T-cell;
    # "timepoint" is the baseline (degenerate single-valued).
    contrasts_list = list(present_contrasts)
    days = []
    baselines = []
    for c in contrasts_list:
        parts = c.split("_", 1)
        head = parts[0]
        tail = parts[1] if len(parts) > 1 else c
        if head not in days:
            days.append(head)
        if tail not in baselines:
            baselines.append(tail)
    return {
        "donor": donor,
        "contrasts": contrasts_list,
        "diseases": days,
        "timepoints": baselines,
        "senders": senders_canonical,
        "receivers": receivers_canonical,
        "heatmap_counts": heatmap_counts,
        "pathway_counts": pathway_counts,
        "present_pairs": sorted(present_pairs),
        "n_total_rows": total_rows,
        "pair_row_counts": pair_row_counts,
        "score_columns": list(_INCYTR_SCORE_COLS),
        "label_columns": list(_INCYTR_LABEL_COLS),
        "label_nodes": list(_INCYTR_LABEL_NODES),
        "label_vocab": list(_INCYTR_LABEL_VOCAB),
        "direction_flag_columns": list(dir_flag_cols),
        "path_metric_columns": list(extra_path_cols),
        "slice_index": {
            "schema_version": SCHEMA_VERSION,
            "filename_template": "{context}__{sender}__{receiver}.parquet",
            "sanitize_rule": "replace('/', '-'); replace(' ', '_'); replace('.', '')",
            "present": sorted(present_pairs),
            "n_total_rows": total_rows,
            "pair_row_counts": pair_row_counts,
        },
    }


def _write_tcell_pair_pathways() -> dict:
    """Wipe and recreate the shard directory, then process each donor in turn."""
    shutil.rmtree(EDGE_SLICES_INCYTR_PATHWAYS_DIR, ignore_errors=True)
    os.makedirs(EDGE_SLICES_INCYTR_PATHWAYS_DIR, exist_ok=True)

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

    return {
        "schema_version": SCHEMA_VERSION,
        "version": 1,
        "source": f"pair_mode_tcells ({os.path.relpath(INCYTR_PAIR_MODE_TCELLS_DIR, config.REPO_ROOT)})",
        "source_mode": "pair_mode_tcells",
        "donors": sorted(by_context.keys()),
        "contexts": sorted(by_context.keys()),
        "by_context": by_context,
        "senders": sorted(all_senders),
        "receivers": sorted(all_receivers),
        "contrasts": sorted(all_contrasts, key=lambda x: int(x.split("_")[0][1:])),
        "score_columns": list(_INCYTR_SCORE_COLS),
        "label_columns": list(_INCYTR_LABEL_COLS),
        "label_nodes": list(_INCYTR_LABEL_NODES),
        "label_vocab": list(_INCYTR_LABEL_VOCAB),
        "slice_index": index,
    }


# ---------------------------------------------------------------------------
# Audit manifest (small T-cell version)
# ---------------------------------------------------------------------------

def _copy_audit_source(src: str, key: str) -> str | None:
    if not os.path.exists(src):
        return None
    os.makedirs(AUDIT_SOURCES_DIR, exist_ok=True)
    dest_name = f"{key}{os.path.splitext(src)[1]}"
    dest = os.path.join(AUDIT_SOURCES_DIR, dest_name)
    shutil.copyfile(src, dest)
    return os.path.relpath(dest, UNIFIED_VIEWER_DIR)


def _json_preview(path: str) -> tuple[list[str], list[dict], int]:
    with open(path) as f:
        obj = json.load(f)
    if isinstance(obj, dict):
        rows = [{"key": k, "value": json.dumps(v, ensure_ascii=False)}
                for k, v in list(obj.items())[:AUDIT_PREVIEW_ROWS]]
        return ["key", "value"], rows, len(obj)
    if isinstance(obj, list):
        rows = obj[:AUDIT_PREVIEW_ROWS]
        cols = sorted({k for r in rows if isinstance(r, dict) for k in r})
        return cols, rows, len(obj)
    return ["value"], [{"value": json.dumps(obj, ensure_ascii=False)}], 1


def _tcell_audit_specs() -> list[tuple[str, str, str]]:
    specs = []
    for donor in DONORS:
        mea_dir = os.path.join(KINASE_ATTRIBUTION_TCELLS_DIR, donor, "mea")
        specs.append((
            f"{donor}_mea_manifest", f"{donor} MEA manifest",
            os.path.join(mea_dir, "mea_manifest.json"),
        ))
        for stem, label in [
            ("kinase_timepoint_nes", "Kinase timepoint NES"),
            ("kinase_timepoint_fdr", "Kinase timepoint FDR"),
            ("recurrence", "MEA recurrence"),
            ("mea_global_shift", "MEA global shift"),
        ]:
            specs.append((
                f"{donor}_{stem}", f"{donor} {label}",
                os.path.join(mea_dir, f"{stem}.csv"),
            ))
        specs.append((
            f"{donor}_extract_manifest", f"{donor} scRNA extract manifest",
            os.path.join(TCELLS_INCYTR_INPUTS_DIR, donor, "scrna",
                         "extract_manifest.json"),
        ))
        specs.append((
            f"{donor}_decompose_manifest", f"{donor} decompose manifest",
            os.path.join(TCELLS_INCYTR_INPUTS_DIR, donor, "scrna",
                         "decompose_manifest.json"),
        ))
    return specs


AUDIT_TABLE_SPECS = _tcell_audit_specs()


# ---------------------------------------------------------------------------
# Kinase audit drawer wiring (donor1 only; donor2 has no IMAC). The audit
# drawer in alz/tcell_viewer/template/js/tabs/kinase_audit.js loads a fixed
# set of unprefixed keys (mea_stoichiometry, mea_substrate_sets, ...) via
# AuditDataStore.load(). We register donor1's per-donor mea/ outputs under
# those keys, with the `contrast` column rewritten from `D1_d13_vs_d2` to
# the viewer's short token (`d13`) so `r.contrast === contrast` matches the
# CONTRASTS vocab. Keys the bulk T-cell pipeline doesn't produce
# (site_level_ols, unified_attribution*, normalization_summary,
# sample_mapping, mea_raw_phospho, atlas crosswalks) get empty shims so
# Promise.all([...]) in the drawer doesn't reject — most panels degrade to
# empty tables; the running-ES walk is fed by a synthesized site_level_ols
# (see _synthesize_site_level_ols).
# ---------------------------------------------------------------------------

_TCELL_CONTRAST_RE = re.compile(r"^D\d+_(d\d+)_vs_d2$")


def _short_contrast(s: object) -> str:
    """Rewrite `D1_d13_vs_d2` -> `d13` (the viewer's CONTRASTS token).

    Pass-through for already-short tokens. Strict regex so a typo in the
    pipeline output surfaces as an unchanged value, not a silent shim.
    """
    if s is None:
        return ""
    m = _TCELL_CONTRAST_RE.match(str(s))
    return m.group(1) if m else str(s)


def _audit_csv_meta(dest_path: str, label: str, key: str,
                    extra_note: str = "") -> dict:
    """Build a manifest entry for an on-disk CSV in AUDIT_SOURCES_DIR."""
    header = pd.read_csv(dest_path, nrows=0)
    columns = list(header.columns)
    preview_df = pd.read_csv(dest_path, nrows=AUDIT_PREVIEW_ROWS)
    preview = preview_df.where(pd.notna(preview_df), None).to_dict("records")
    return {
        "key": key,
        "label": label + (f" — {extra_note}" if extra_note else ""),
        "type": "csv",
        "row_count": int(_count_csv_rows(dest_path)),
        "column_count": int(len(columns)),
        "columns": [{"raw": c, "label": c, "definition": "", "format": "text"}
                    for c in columns],
        "preview": _sanitize(preview),
        "relative_path": os.path.relpath(dest_path, UNIFIED_VIEWER_DIR),
        "source_path": os.path.relpath(dest_path, config.REPO_ROOT),
    }


def _rewrite_contrast_csv(src: str, dest_name: str, label: str,
                          key: str, drop_cols: tuple[str, ...] = ()
                          ) -> dict | None:
    """Copy CSV with contrast column rewritten to short form.

    drop_cols is honored verbatim to slim large tables (e.g. mea_substrate_sets
    is ~99 MB and the drawer reads only kinase/contrast/motif/kl_percentile).
    """
    if not os.path.exists(src):
        return None
    os.makedirs(AUDIT_SOURCES_DIR, exist_ok=True)
    dest = os.path.join(AUDIT_SOURCES_DIR, dest_name)
    # Stream in chunks — mea_substrate_sets is 2.2M rows.
    first = True
    for chunk in pd.read_csv(src, chunksize=200_000):
        if "contrast" in chunk.columns:
            chunk["contrast"] = chunk["contrast"].map(_short_contrast)
        if drop_cols:
            chunk = chunk.drop(columns=[c for c in drop_cols if c in chunk.columns])
        chunk.to_csv(dest, index=False, mode="w" if first else "a",
                     header=first)
        first = False
    return _audit_csv_meta(dest, label, key,
                           extra_note=f"donor1; contrast normalized to short token")


def _synthesize_site_level_ols(stoich_matrix_path: str,
                               key: str = "site_level_ols",
                               label: str = "Site-level LFC (donor1)"
                               ) -> dict | None:
    """Derive per-site centered LFC from stoichiometry_matrix.csv.

    The T-cell MEA pipeline (alz/ingest/tcells_perdonor.py::_run_mea) takes
    raw `D1_dXX − D1_d2` deltas, median-centers, winsorizes, then GSEA-prerank.
    The drawer's _ensurePreRank applies the centering + winsorization itself
    (lines 115-116 in kinase_audit.js), so we only need raw stoich_lfc per
    site here. Columns: site_id, gene_symbol, motif, stoich_lfc_<short>.
    """
    if not os.path.exists(stoich_matrix_path):
        return None
    df = pd.read_csv(stoich_matrix_path)
    baseline_col = next((c for c in df.columns if c.endswith("_d2")), None)
    if baseline_col is None:
        return None
    keep = ["site_id", "gene_symbol", "motif"]
    out = df[[c for c in keep if c in df.columns]].copy()
    out["matched_protein"] = df.get("protein_id", "")
    # n_obs_stoich = count of non-NA across non-baseline timepoint columns.
    tp_cols = [c for c in df.columns
               if c.startswith(baseline_col.split("_")[0] + "_d")
               and c != baseline_col]
    if not tp_cols:
        return None
    out["n_obs_stoich"] = df[tp_cols].notna().sum(axis=1).astype(int)
    base_vals = df[baseline_col]
    for tp in tp_cols:
        short = _short_contrast(f"{tp}_vs_d2")  # `D1_d13` -> `d13`
        if short == f"{tp}_vs_d2":
            # regex miss — try direct split.
            short = tp.split("_", 1)[1] if "_" in tp else tp
        out[f"stoich_lfc_{short}"] = df[tp] - base_vals
    os.makedirs(AUDIT_SOURCES_DIR, exist_ok=True)
    dest = os.path.join(AUDIT_SOURCES_DIR, f"{key}.csv")
    out.to_csv(dest, index=False)
    return _audit_csv_meta(dest, label, key,
                           extra_note="synthesized at build time from "
                           "donor1/stoichiometry_matrix.csv "
                           "(no per-site OLS fit in the T-cell pipeline)")


def _shim_audit_entry(key: str, label: str, reason: str) -> dict:
    """Empty shim. AuditDataStore.load returns meta.preview ([]) when
    relative_path is absent — keeps Promise.all([...]) from rejecting."""
    return {
        "key": key,
        "label": label + f" — n/a ({reason})",
        "type": "csv",
        "row_count": 0,
        "column_count": 0,
        "columns": [],
        "preview": [],
        "missing": True,
    }


# Keys the drawer awaits unconditionally — must all resolve.
# Maps unprefixed-key → (filename under donor1/mea/ or donor1/, drop_cols).
_KINASE_AUDIT_FILES: tuple[tuple[str, str, str, tuple[str, ...]], ...] = (
    ("mea_stoichiometry",     "donor1/mea/mea_timecourse.csv",    "MEA stoichiometry (donor1; mea_timecourse rows)",       ()),
    ("mea_stoichiometry_pY",  "donor1/mea/mea_timecourse_pY.csv", "MEA stoichiometry pY (donor1)",                          ()),
    ("mea_substrate_sets",    "donor1/mea/mea_substrate_sets.csv","MEA substrate sets (donor1; ST)",                        ("residue_type", "track")),
    ("mea_substrate_sets_pY", "donor1/mea/mea_substrate_sets_pY.csv","MEA substrate sets (donor1; pY)",                     ("residue_type", "track")),
    ("winsorized_sites",      "donor1/mea/winsorized_sites.csv",  "Winsorized sites (donor1)",                              ()),
    ("winsorized_sites_pY",   "donor1/mea/winsorized_sites_pY.csv","Winsorized sites pY (donor1)",                          ()),
    ("mea_global_shift",      "donor1/mea/mea_global_shift.csv",  "MEA global shift (donor1)",                              ()),
    ("mea_global_shift_pY",   "donor1/mea/mea_global_shift_pY.csv","MEA global shift pY (donor1)",                          ()),
)

# Per-site matrices live one level up from mea/. No contrast rewrite needed
# (no contrast column) but registered under unprefixed keys.
_KINASE_AUDIT_MATRICES: tuple[tuple[str, str, str], ...] = (
    ("raw_phospho_normalized",    "donor1/raw_phospho_normalized.csv",    "Raw phospho normalized (donor1)"),
    ("raw_phospho_normalized_pY", "donor1/raw_phospho_normalized_pY.csv", "Raw phospho normalized pY (donor1)"),
    ("stoichiometry_matrix",      "donor1/stoichiometry_matrix.csv",      "Stoichiometry matrix (donor1)"),
    ("stoichiometry_matrix_pY",   "donor1/stoichiometry_matrix_pY.csv",   "Stoichiometry matrix pY (donor1)"),
)

# Keys the drawer awaits but the T-cell bulk pipeline doesn't produce.
# Empty shims keep Promise.all from rejecting; affected sub-panels render
# empty tables (degraded but honest).
_KINASE_AUDIT_SHIMS: tuple[tuple[str, str], ...] = (
    ("mea_raw_phospho",         "raw-phospho MEA track not run for T-cell"),
    ("mea_raw_phospho_pY",      "raw-phospho MEA track (pY) not run for T-cell"),
    ("unified_attribution",     "T-cell bulk MEA has no decomposition / attribution step"),
    ("unified_attribution_full","T-cell bulk MEA has no decomposition / attribution step"),
    ("normalization_summary",   "T-cell normalization is per-batch (Forperseus); no summary JSON"),
    ("sample_mapping",          "T-cell sample mapping not surfaced (per-donor TMT plex)"),
    ("wmb_kinase_expression",   "WMB atlas is mouse-only; not applicable to human T-cell"),
    ("sea_ad_supertype_lfc",    "SEA-AD atlas is brain cortex; not applicable to T-cell"),
)


def _register_kinase_audit_tables(tables: dict) -> None:
    donor1_root = os.path.join(KINASE_ATTRIBUTION_TCELLS_DIR, "donor1")
    for key, relpath, label, drop_cols in _KINASE_AUDIT_FILES:
        src = os.path.join(KINASE_ATTRIBUTION_TCELLS_DIR, relpath)
        meta = _rewrite_contrast_csv(src, f"{key}.csv", label, key, drop_cols)
        if meta is None:
            tables[key] = _shim_audit_entry(key, label, "source CSV missing")
        else:
            tables[key] = meta
    for key, relpath, label in _KINASE_AUDIT_MATRICES:
        src = os.path.join(KINASE_ATTRIBUTION_TCELLS_DIR, relpath)
        if not os.path.exists(src):
            tables[key] = _shim_audit_entry(key, label, "source CSV missing")
            continue
        dest = os.path.join(AUDIT_SOURCES_DIR, f"{key}.csv")
        os.makedirs(AUDIT_SOURCES_DIR, exist_ok=True)
        shutil.copyfile(src, dest)
        tables[key] = _audit_csv_meta(dest, label, key)
    # Synthesize per-site LFC for ST + pY from on-disk stoich matrices.
    for key, matrix_rel, label in (
        ("site_level_ols",
         "donor1/stoichiometry_matrix.csv",
         "Site-level LFC (donor1; ST)"),
        ("site_level_ols_pY",
         "donor1/stoichiometry_matrix_pY.csv",
         "Site-level LFC (donor1; pY)"),
    ):
        meta = _synthesize_site_level_ols(
            os.path.join(KINASE_ATTRIBUTION_TCELLS_DIR, matrix_rel),
            key=key, label=label,
        )
        if meta is None:
            tables[key] = _shim_audit_entry(
                key, label, "stoichiometry matrix unavailable")
        else:
            tables[key] = meta
    for key, reason in _KINASE_AUDIT_SHIMS:
        # Use a stable label even for shims so the drawer's "source: ..." text
        # reads coherently when the panel renders empty.
        tables[key] = _shim_audit_entry(key, key, reason)


def build_tcell_audit_manifest() -> dict:
    tables = {}
    for key, label, src in AUDIT_TABLE_SPECS:
        if not os.path.exists(src):
            tables[key] = {
                "key": key, "label": label, "missing": True,
                "source_path": os.path.relpath(src, config.REPO_ROOT),
            }
            continue
        rel = _copy_audit_source(src, key)
        ext = os.path.splitext(src)[1].lower()
        if ext == ".csv":
            header = pd.read_csv(src, nrows=0)
            columns = list(header.columns)
            preview_df = pd.read_csv(src, nrows=AUDIT_PREVIEW_ROWS)
            preview = preview_df.where(pd.notna(preview_df), None).to_dict("records")
            row_count = _count_csv_rows(src)
        elif ext == ".json":
            columns, preview, row_count = _json_preview(src)
        else:
            columns, preview, row_count = [], [], 0
        tables[key] = {
            "key": key,
            "label": label,
            "type": ext.lstrip("."),
            "row_count": int(row_count),
            "column_count": int(len(columns)),
            "columns": [{"raw": c, "label": c, "definition": "", "format": "text"}
                        for c in columns],
            "preview": _sanitize(preview),
            "relative_path": rel,
            "source_path": os.path.relpath(src, config.REPO_ROOT),
        }
    _register_kinase_audit_tables(tables)
    return {"preview_rows": AUDIT_PREVIEW_ROWS, "tables": tables}


# ---------------------------------------------------------------------------
# Per-cluster transcript pseudobulk shards (Incytr Pathways · Evidence tab).
#
# Source: data/derived/tcells_incytr_inputs/<donor>/scrna/aggexp_data.csv
# Header: `gene`, then `<cluster>__<day>` columns (N=1 per (cluster, day)).
#
# Output: outputs/reports/tcell_viewer/audit_sources/transcript_trace/<slug>.parquet
# Schema: {gene: string, group: string, value: float} where group = day token.
#
# Per-cluster shards are written once per donor's aggexp; if both donors share
# a cluster name (no overlap in current cohort), the second write would
# overwrite. The viewer is donor-scoped at render time (it loads shards
# matching whichever donor is active), so cross-donor name collisions would
# be a data-shape problem upstream, not here.
# ---------------------------------------------------------------------------

def _write_tcell_transcript_trace() -> dict:
    """Generate per-donor, per-cluster transcript pseudobulk parquets.

    Returns {by_context: {<donor-context>: {clusters, relative_path}}, ...}.
    Donor scoping disambiguates clusters whose names appear in both donors
    (e.g. CD4Naive) but whose pseudobulk values differ.
    """
    rel_path = "audit_sources/transcript_trace"
    out_dir_base = os.path.join(UNIFIED_VIEWER_DIR, rel_path)

    by_context: dict[str, dict] = {}
    for donor in DONORS:
        agg_path = os.path.join(
            TCELLS_INCYTR_INPUTS_DIR, donor, "scrna", "aggexp_data.csv"
        )
        donor_rel = f"{rel_path}/{donor}"
        donor_out = os.path.join(UNIFIED_VIEWER_DIR, donor_rel)
        os.makedirs(donor_out, exist_ok=True)
        if not os.path.exists(agg_path):
            print(f"  ({donor}) no aggexp_data.csv; skipping transcript_trace",
                  flush=True)
            by_context[donor] = {"clusters": [], "relative_path": donor_rel}
            continue
        df = pd.read_csv(agg_path)
        if "gene" not in df.columns:
            print(f"  ({donor}) aggexp_data.csv missing `gene` column; skip",
                  flush=True)
            by_context[donor] = {"clusters": [], "relative_path": donor_rel}
            continue
        col_split: dict[str, list[tuple[str, str]]] = {}
        for c in df.columns:
            if c == "gene" or "__" not in c:
                continue
            cluster, day = c.rsplit("__", 1)
            col_split.setdefault(cluster, []).append((day, c))
        donor_slugs: list[str] = []
        for cluster, pairs in col_split.items():
            frames = []
            for day, col in pairs:
                sub = df[["gene", col]].rename(columns={col: "value"}).copy()
                sub["group"] = day
                frames.append(sub[["gene", "group", "value"]])
            long_df = pd.concat(frames, ignore_index=True)
            long_df = long_df.dropna(subset=["gene"])
            long_df = long_df[long_df["gene"].astype(str).str.len() > 0]
            slug = cluster  # already alphanumeric from extract pipeline
            out_path = os.path.join(donor_out, f"{slug}.parquet")
            pq.write_table(pa.Table.from_pandas(long_df, preserve_index=False),
                           out_path, compression="zstd")
            donor_slugs.append(slug)
        by_context[donor] = {
            "clusters": sorted(donor_slugs),
            "relative_path": donor_rel,
        }
        print(f"  ({donor}) wrote {len(donor_slugs)} transcript_trace shard(s)",
              flush=True)
    return {"by_context": by_context}


# ---------------------------------------------------------------------------
# Payload assembly
# ---------------------------------------------------------------------------

def build_tcell_payload() -> dict:
    """Assemble the T-cell payload — donor-scoped data nested under by_context."""
    print("[build_tcell_payload] kinase slices per donor:", flush=True)
    kinases_by_context: dict[str, dict] = {}
    contrast_union: list[str] = []
    fdr_thresh = 0.25
    for donor in DONORS:
        block = _build_donor_kinases_slice(donor)
        if block is None:
            print(f"  {donor}: no MEA (expected for donor2)", flush=True)
            continue
        kinases_by_context[donor] = block["kinases_slice"]
        for c in block["contrasts"]:
            if c not in contrast_union:
                contrast_union.append(c)
        fdr_thresh = block["fdr_threshold"]
        print(f"  {donor}: {len(block['kinase_names'])} kinases × "
              f"{len(block['contrasts'])} contrasts", flush=True)

    # Empty donor slice: same column names, zero rows. Keeps the JS contract
    # stable (donor swap toggles between two slice objects, never null).
    if kinases_by_context:
        template_cols = next(iter(kinases_by_context.values()))
        empty_slice = {k: [] for k in template_cols}
        for donor in DONORS:
            kinases_by_context.setdefault(donor, empty_slice)

    kinases_slice = {
        "by_context": kinases_by_context,
    }

    print("[build_tcell_payload] celltypes slice:", flush=True)
    donor_clusters = {d: _load_donor_clusters(d) for d in DONORS}
    celltypes_slice = _build_celltypes_slice(donor_clusters)
    celltype_id_by_name = {
        name: idx for idx, name in zip(celltypes_slice["id"], celltypes_slice["name"])
    }
    celltypes_by_context: dict[str, dict] = {}
    for donor, clusters in donor_clusters.items():
        ordered = [c for c in celltypes_slice["name"] if c in set(clusters)]
        celltypes_by_context[donor] = {
            "id": [celltype_id_by_name[c] for c in ordered],
            "name": ordered,
            "tissue_category": ["T-cell"] * len(ordered),
            "available_donors": [[donor] for _ in ordered],
        }
    celltypes_slice["by_context"] = celltypes_by_context
    print(f"  {len(celltypes_slice['id'])} cluster(s) across {len(DONORS)} donors",
          flush=True)

    print("[build_tcell_payload] pair-mode shards:", flush=True)
    incytr_pathways_block = _write_tcell_pair_pathways()

    print("[build_tcell_payload] transcript_trace shards:", flush=True)
    transcript_trace_meta = _write_tcell_transcript_trace()
    total = sum(len(b["clusters"])
                for b in transcript_trace_meta.get("by_context", {}).values())
    print(f"  {total} cluster shard(s) total across {len(DONORS)} donors",
          flush=True)

    family_map: dict[str, str] = {}
    union_kinases: list[str] = []
    seen: set[str] = set()
    for donor in DONORS:
        for k in (kinases_by_context.get(donor, {}).get("name") or []):
            if k not in seen:
                seen.add(k)
                union_kinases.append(k)
    if union_kinases:
        try:
            from kinase_library.modules import data as kl_data
            family_map = {
                str(k): str(v) for k, v in
                kl_data.get_kinase_family(union_kinases).to_dict().items()
                if v is not None and str(v) != "nan"
            }
        except Exception as e:
            print(f"  (warn) family resolve failed: {e}; using empty map",
                  flush=True)

    # Empty per-tab payloads the existing viewer JS dereferences blindly.
    kinase_motifs = {"name": [], "motifs": []}
    audit_tables = build_tcell_audit_manifest()

    # Timepoints actually seen across both donors → palette subset.
    timepoint_set: set[str] = set()
    for block in incytr_pathways_block.get("by_context", {}).values():
        for c in block.get("contrasts", []):
            timepoint_set.add(_timepoint_label(c))
            timepoint_set.add(c.split("_", 1)[1] if "_" in c else c)
    timepoint_set.update(contrast_union)
    palette = {tp: TIMEPOINT_COLOR_MAP.get(tp, "#808080")
               for tp in sorted(timepoint_set)}

    contexts: list[dict] = []
    for donor in DONORS:
        ip_block = incytr_pathways_block.get("by_context", {}).get(donor, {})
        donor_contrasts = ip_block.get("contrasts") or []
        capabilities = {
            "kinases": donor in DONOR_WITH_MEA and len(
                kinases_by_context.get(donor, {}).get("id", [])
            ) > 0,
            "celltypes": len(celltypes_by_context.get(donor, {}).get("id", [])) > 0,
            "incytr": bool(ip_block.get("slice_index", {}).get("present")),
            "decomp_ols": False,
            "song_concordance": False,
            "human_reference": False,
            "subclass_breakdown": False,
            "audit_tables": True,
            "transcript_trace": donor in transcript_trace_meta.get("by_context", {}),
            "omics_trace": False,
        }
        notes = []
        if not capabilities["kinases"]:
            notes.append("No IMAC kinase MEA is available for this donor.")
        contexts.append({
            "id": donor,
            "label": donor.replace("donor", "Donor "),
            "cohort": "tcell",
            "axis_kind": "donor",
            "contrasts": donor_contrasts,
            "contrast_axis": {
                "primary": "day",
                "baseline": "d2",
                "groups": ip_block.get("diseases", []),
                "timepoints": ip_block.get("timepoints", []),
            },
            "celltypes": celltypes_by_context.get(donor, {}).get("name", []),
            "capabilities": capabilities,
            "notes": notes,
        })

    meta = {
        "schema_version": SCHEMA_VERSION,
        "viewer_payload_schema_version": 2,
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "cohort": "tcell",
        "default_context": "donor1",
        "contexts": contexts,
        "capabilities": {
            "contexts": True,
            "kinases": any(c["capabilities"]["kinases"] for c in contexts),
            "celltypes": any(c["capabilities"]["celltypes"] for c in contexts),
            "incytr": any(c["capabilities"]["incytr"] for c in contexts),
            "decomp_ols": False,
            "song_concordance": False,
            "human_reference": False,
            "subclass_breakdown": False,
            "audit_tables": True,
            "transcript_trace": any(c["capabilities"]["transcript_trace"] for c in contexts),
            "omics_trace": False,
        },
        "donors": list(DONORS),
        "donors_with_mea": list(DONOR_WITH_MEA),
        "contrasts": contrast_union or sorted(timepoint_set),
        "timepoints": sorted(timepoint_set),
        "timepoint_color_map": palette,
        "familyMap": family_map,
        "fdr_threshold": fdr_thresh,
        "mea_kinase_donor": "donor1",
        "transcript_trace": transcript_trace_meta,
        "notes": {
            "donor2_mea": "Donor 2 has no IMAC; kinase MEA unavailable for donor 2.",
        },
    }

    payload = {
        "kinases": kinases_slice,
        "kinase_motifs": kinase_motifs,
        "celltypes": celltypes_slice,
        "audit_tables": audit_tables,
        "edge_slice_ref": {
            "schema_version": SCHEMA_VERSION,
            "incytr_pathways_url": "edge_slices/incytr_pathways/",
            "incytr_pathways_index": "edge_slices/incytr_pathways/index.json",
        },
        "incytr_pathways": incytr_pathways_block,
        "meta": meta,
    }
    return _sanitize(payload)


def write_payload(payload: dict) -> dict:
    os.makedirs(UNIFIED_VIEWER_DIR, exist_ok=True)
    json_str = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    raw = json_str.encode("utf-8")
    with open(PAYLOAD_JSON, "wb") as f:
        f.write(raw)
    gz = gzip.compress(raw, compresslevel=6)
    with open(PAYLOAD_JSON_GZ, "wb") as f:
        f.write(gz)
    return {"raw_bytes": len(raw), "gzip_bytes": len(gz), "json_str": json_str}


# ---------------------------------------------------------------------------
# HTML rendering
# ---------------------------------------------------------------------------

from jinja2 import Environment, FileSystemLoader  # noqa: E402

_TEMPLATE_DIR = os.path.join(HERE, "tcell_viewer", "template")
_SHARED_TEMPLATE_DIR = os.path.join(HERE, "viewer_shared", "template")


def _render_template() -> str:
    def _raw(path: str) -> str:
        local_path = os.path.join(_TEMPLATE_DIR, path)
        shared_path = os.path.join(_SHARED_TEMPLATE_DIR, path)
        source = local_path if os.path.exists(local_path) else shared_path
        with open(source) as f:
            return f.read()

    env = Environment(
        loader=FileSystemLoader(_TEMPLATE_DIR),
        keep_trailing_newline=True,
    )
    env.globals["raw"] = _raw
    return env.get_template("index.html.j2").render()


def write_html(payload: dict, json_str: str | None = None) -> dict:
    os.makedirs(UNIFIED_VIEWER_DIR, exist_ok=True)
    if json_str is None:
        json_str = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    html = _render_template()
    palette = payload.get("meta", {}).get("timepoint_color_map", {})
    d13 = palette.get("d13", TIMEPOINT_COLOR_MAP["d13"])
    d17 = palette.get("d17", TIMEPOINT_COLOR_MAP["d17"])
    d20 = palette.get("d20", TIMEPOINT_COLOR_MAP["d20"])
    # styles.css lifts verbatim from unified_viewer, which carries mouse-
    # genotype color sentinels (__APP_COLOR__ etc). Map them to day colors:
    # mid (App→d17) / early (Tau→d13) / late (ApTt→d20).
    for sentinel, value in (
        ("__D13_COLOR__", d13),
        ("__D17_COLOR__", d17),
        ("__D20_COLOR__", d20),
        ("__APP_COLOR__", d17),
        ("__TAU_COLOR__", d13),
        ("__APTT_COLOR__", d20),
        ("__PAYLOAD_SENTINEL__", json_str),
    ):
        html = html.replace(sentinel, value)
    raw = html.encode("utf-8")
    with open(UNIFIED_VIEWER_HTML, "wb") as f:
        f.write(raw)
    return {"html_bytes": len(raw), "output": UNIFIED_VIEWER_HTML}


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate(payload: dict | None = None) -> str:
    errors: list[str] = []
    warnings: list[str] = []

    if not os.path.exists(PAYLOAD_JSON):
        errors.append(f"payload JSON missing: {PAYLOAD_JSON}")
        raw_bytes = gzip_bytes = 0
    else:
        raw_bytes = os.path.getsize(PAYLOAD_JSON)
        gzip_bytes = os.path.getsize(PAYLOAD_JSON_GZ) if os.path.exists(PAYLOAD_JSON_GZ) else 0
        if payload is None:
            with open(PAYLOAD_JSON) as f:
                payload = json.load(f)

    if raw_bytes >= 100 * 1024 * 1024:
        errors.append(f"payload raw {raw_bytes/1e6:.1f} MB exceeds 100 MB cap")
    if gzip_bytes >= 20 * 1024 * 1024:
        errors.append(f"payload gzip {gzip_bytes/1e6:.1f} MB exceeds 20 MB cap")

    if payload is not None:
        meta = payload.get("meta", {})
        if meta.get("viewer_payload_schema_version") != 2:
            errors.append("meta.viewer_payload_schema_version != 2")
        context_ids = [c.get("id") for c in meta.get("contexts", [])]
        if meta.get("default_context") not in context_ids:
            errors.append("meta.default_context is not present in meta.contexts")
        for key in ("kinases", "celltypes", "incytr_pathways"):
            if "by_context" not in (payload.get(key) or {}):
                errors.append(f"{key}.by_context missing")

        # Donor1 must have MEA.
        kinases_by_context = payload.get("kinases", {}).get("by_context", {})
        d1_rows = len(kinases_by_context.get("donor1", {}).get("id", []))
        if d1_rows == 0:
            errors.append("donor1 kinases slice is empty — expected MEA outputs")

        # Both donors must have Incytr pair-mode pathways.
        ip_donors = set(payload.get("incytr_pathways", {}).get("donors", []))
        for d in DONORS:
            if d not in ip_donors:
                errors.append(f"{d} missing from incytr_pathways block")

        ip_idx_path = os.path.join(EDGE_SLICES_INCYTR_PATHWAYS_DIR, "index.json")
        if not os.path.exists(ip_idx_path):
            errors.append(f"missing edge_slices/incytr_pathways/index.json")

    peak_mb = _peak_rss_mb()
    lines = [
        "# T-cell Viewer Payload Report",
        "",
        f"_Generated {pd.Timestamp.utcnow().isoformat()}_",
        "",
        "## Sizes",
        "",
        f"- Payload JSON (raw): {raw_bytes/1e6:.2f} MB (cap 100)",
        f"- Payload JSON (gzip): {gzip_bytes/1e6:.2f} MB (cap 20)",
        "",
        f"- Peak RSS: {peak_mb:.0f} MB",
        "",
        "## Invariants",
        "",
    ]
    if errors:
        lines.append("### FAIL")
        for e in errors:
            lines.append(f"- {e}")
    else:
        lines.append("All structural invariants pass.")
    if warnings:
        lines.append("")
        lines.append("### Warnings")
        for w in warnings:
            lines.append(f"- {w}")
    report = "\n".join(lines) + "\n"

    os.makedirs(os.path.dirname(REPORT_MD), exist_ok=True)
    with open(REPORT_MD, "w") as f:
        f.write(report)
    print(report)
    if errors:
        raise SystemExit(f"validation failed: {len(errors)} error(s)")
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", action="store_true", help="Print input row counts")
    ap.add_argument("--payload", action="store_true", help="Write JSON payload")
    ap.add_argument("--html", action="store_true", help="Write tcell_viewer HTML (requires payload)")
    ap.add_argument("--validate", action="store_true", help="Write payload validation report")
    args = ap.parse_args(argv)

    if not any([args.summary, args.payload, args.html, args.validate]):
        args.payload = True
        args.html = True

    if args.summary:
        for donor in DONORS:
            mea_dir = os.path.join(KINASE_ATTRIBUTION_TCELLS_DIR, donor, "mea")
            manifest_path = os.path.join(mea_dir, "mea_manifest.json")
            mea_state = "n/a"
            if os.path.exists(manifest_path):
                with open(manifest_path) as f:
                    mea_state = f"{len(json.load(f).get('mea_ran', []))} tracks"
            wide_dir = os.path.join(INCYTR_PAIR_MODE_TCELLS_DIR, donor, "wide")
            wide_files = (sorted(glob.glob(os.path.join(wide_dir, "*_incytr_output.parquet")))
                          if os.path.isdir(wide_dir) else [])
            print(f"  {donor}: MEA={mea_state}, "
                  f"wide_parquets={len(wide_files)} "
                  f"({[os.path.basename(f) for f in wide_files]})")

    payload = None
    json_str = None
    if args.payload:
        payload = build_tcell_payload()
        sizes = write_payload(payload)
        json_str = sizes.pop("json_str")
        print(f"  payload raw={sizes['raw_bytes']/1e6:.2f} MB "
              f"gzip={sizes['gzip_bytes']/1e6:.2f} MB")

    if args.html:
        if payload is None:
            if not os.path.exists(PAYLOAD_JSON):
                raise SystemExit(f"payload missing at {PAYLOAD_JSON}; run --payload first")
            with open(PAYLOAD_JSON) as f:
                json_str = f.read()
            payload = json.loads(json_str)
        info = write_html(payload, json_str=json_str)
        print(f"  html {info['html_bytes']/1e6:.2f} MB -> {info['output']}")

    if args.validate:
        validate(payload)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

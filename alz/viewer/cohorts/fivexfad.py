"""5xFAD supporting-cohort unified-viewer slice adapter."""

from __future__ import annotations

import gzip
import json
import os
import re
import shutil
import uuid
from functools import cmp_to_key
from typing import TYPE_CHECKING, Any

import glob
import gzip
import sys

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from alz.bulk_mea.confidence import DECOMP_FDR_AGREEMENT
from alz.shared import config
from alz.viewer.paths import (
    EDGE_SLICES_INCYTR_PATHWAYS_5XFAD_CORTEX_DIR,
    EDGE_SLICES_INCYTR_PATHWAYS_5XFAD_HIPPO_DIR,
    FIVEXFAD_KINASE_DIR,
    SCHEMA_VERSION,
    UNIFIED_VIEWER_DIR,
)
from alz.viewer.shared.build_cache import _input_signature, _load_build_cache, _write_build_cache
from alz.viewer.shared.cohort_slice import CohortViewerSlice
from alz.viewer.shared.incytr_index import (
    _INCYTR_FC_COLS,
    _INCYTR_GENE_NODE_INDEX_FILENAME,
    _INCYTR_INDEX_FILENAME,
    _INCYTR_LABEL_COLS,
    _INCYTR_LABEL_NODES,
    _INCYTR_LABEL_SRC,
    _INCYTR_LABEL_VOCAB,
    _INCYTR_PATHWAY_ABS_PDS,
    _INCYTR_PATHWAY_PVALUES,
    _INCYTR_SCORE_COLS,
    _SIGN_VEC_LABELS,
    _idx_label_bits,
    _idx_traj_bits,
)
from alz.viewer.shared.payload_helpers import (
    _INCYTR_FC_NODES,
    _build_incytr_gene_node_index,
    _configure_duckdb_tempdir,
    _sanitize,
    _write_gene_node_index_shard,
)

if TYPE_CHECKING:
    from alz.build_unified_viewer import UnifiedData


FIVEXFAD_DETAIL_DIR = os.path.join(
    UNIFIED_VIEWER_DIR, "edge_slices", "fivexfad_detail"
)
FIVEXFAD_CELLTYPE_DIR = os.path.join(FIVEXFAD_KINASE_DIR, "celltype_mea")
FIVEXFAD_CELLTYPE_OLS_DIR = os.path.join(
    UNIFIED_VIEWER_DIR, "edge_slices", "fivexfad_celltype_ols"
)
FIVEXFAD_CELLTYPE_MEA_DIR = os.path.join(
    UNIFIED_VIEWER_DIR, "edge_slices", "fivexfad_celltype_mea"
)
FIVEXFAD_ATTRIBUTION_DIR = os.path.join(
    UNIFIED_VIEWER_DIR, "edge_slices", "fivexfad_attribution"
)
# Whole-list index sidecars (P1/P2) live directly under edge_slices/, not in a per-kinase
# subdir, since they are fetched and iterated in full on first render.
FIVEXFAD_INDEX_DIR = os.path.join(UNIFIED_VIEWER_DIR, "edge_slices")
FIVEXFAD_DETAIL_SITES_PER_CONTRAST = 12
FIVEXFAD_DETAIL_MAX_SITES = 40
FIVEXFAD_RUNNING_DISPLAY_POINTS = 450

_F5_CONF_RANK = {"very_high": 4, "high": 3, "moderate": 2, "low": 1, "none": 0}
_F5_MIN_CELLS_PER_CONTRAST = 3
_F5_MECHANISM_COLUMNS = [
    "cohort",
    "tissue",
    "track",
    "contrast",
    "kinase",
    "stoich_NES",
    "stoich_FDR",
    "raw_NES",
    "raw_FDR",
    "stoich_significant",
    "raw_significant",
    "sign_relation",
    "mechanism_call",
    "skip_reason",
]


def _subs_fraction_counts(value: Any) -> tuple[int | None, int | None]:
    text = str(value or "")
    m = re.match(r"^\s*(\d+)\s*/\s*(\d+)\s*$", text)
    if not m:
        return None, None
    return int(m.group(1)), int(m.group(2))


def _f5_json_value(v):
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except Exception:
        pass
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return float(v)
    if isinstance(v, np.bool_):
        return bool(v)
    return v


def _f5_records(df: pd.DataFrame, cols: list[str] | None = None) -> list[dict]:
    if df is None or df.empty:
        return []
    if cols is not None:
        cols = [c for c in cols if c in df.columns]
        df = df[cols]
    return [
        {str(k): _f5_json_value(v) for k, v in row.items()}
        for row in df.to_dict(orient="records")
    ]


def _load_fivexfad_mechanism_attribution() -> list[dict]:
    rows: list[dict] = []
    for tissue in ("cortex", "hippocampus"):
        for track in ("st", "py"):
            path = os.path.join(FIVEXFAD_KINASE_DIR, f"{tissue}_{track}_mechanism_attribution.csv")
            if not os.path.exists(path):
                continue
            df = pd.read_csv(path)
            if df.empty or "mechanism_call" not in df.columns or "mechanism_score" in df.columns:
                continue
            rows.extend(_f5_records(df, _F5_MECHANISM_COLUMNS))
    return rows


def _f5_shard_name(key: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", key).strip("_") + ".json"


def _f5_group_key(kinase: str, tissue: str, assay: str, analysis_track: str) -> str:
    return "|".join([kinase or "", tissue or "", assay or "", analysis_track or ""])


def _f5_manifest_assay_key(assay: str) -> str:
    v = str(assay or "").strip().lower()
    if v == "py":
        return "py"
    if v == "imac":
        return "imac"
    return v


def _f5_norm_motif(v: Any) -> str:
    return str(v or "").strip().upper()


def _f5_float_or_none(v: Any) -> float | None:
    v = _f5_json_value(v)
    if v is None:
        return None
    try:
        out = float(v)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def _f5_celltype_contrast_cell_counts() -> dict[tuple[str, int, str], int]:
    """Local snRNA cell support keyed by tissue, age, and cell type."""
    candidates = [
        os.path.join(FIVEXFAD_CELLTYPE_DIR, "fivexfad_snrna_pseudobulk_counts.csv"),
        os.path.join(FIVEXFAD_KINASE_DIR, "fivexfad_snrna_cell_counts.csv"),
    ]
    path = next((p for p in candidates if os.path.exists(p)), None)
    if path is None:
        return {}
    try:
        counts = pd.read_csv(path, usecols=["tissue", "age_months", "cell_type", "n_cells"])
    except Exception:
        return {}
    counts["age_months"] = pd.to_numeric(counts["age_months"], errors="coerce")
    counts["n_cells"] = pd.to_numeric(counts["n_cells"], errors="coerce").fillna(0)
    counts = counts.dropna(subset=["tissue", "age_months", "cell_type"])
    grouped = counts.groupby(["tissue", "age_months", "cell_type"], sort=False)["n_cells"].sum()
    out: dict[tuple[str, int, str], int] = {}
    for key, value in grouped.items():
        tissue, age, cell_type = key
        out[(str(tissue), int(age), str(cell_type))] = int(value)
    return out


def _f5_kinase_gene_map() -> dict[str, str]:
    path = config.MAPPING_CACHE_FILE
    if not os.path.exists(path):
        return {}
    try:
        df = pd.read_csv(path)
    except Exception:
        return {}
    if not {"kinase_abbreviation", "gene_symbol"}.issubset(df.columns):
        return {}
    out: dict[str, str] = {}
    for _, row in df.iterrows():
        kinase = str(row.get("kinase_abbreviation", "") or "").strip()
        gene = str(row.get("gene_symbol", "") or "").strip()
        if kinase and gene:
            out[kinase] = gene
    return out


def _f5_site_label_value(row: Any) -> str:
    """Compact viewer label for 5xFAD sites while preserving site_id as join key."""
    get = row.get if hasattr(row, "get") else (lambda _k, _d=None: _d)
    gene = str(get("gene_symbol", "") or "").strip()
    residue = str(get("residue_type", "") or get("PTM.SiteAA", "") or "").strip().upper()
    pos = str(get("site_position", "") or get("PTM.SiteLocation", "") or "").strip()
    if gene and residue and pos:
        return f"{gene}_{residue}{pos}"

    site_id = str(get("site_id", "") or "").strip()
    match = re.search(r"_([STY])(\d+)(?:_[^_\s]+)*$", site_id, flags=re.IGNORECASE)
    if gene and match:
        return f"{gene}_{match.group(1).upper()}{match.group(2)}"
    if site_id:
        symbol_match = re.search(r"gene_symbol:([A-Za-z0-9_.-]+)", site_id)
        if symbol_match and match:
            return f"{symbol_match.group(1)}_{match.group(1).upper()}{match.group(2)}"
    return site_id


def _build_fivexfad_attribution_rows(rows: list[dict], data: UnifiedData | None) -> list[dict]:
    """Native 5xFAD snRNA attribution rows keyed by kinase, tissue, and age."""
    path = os.path.join(FIVEXFAD_KINASE_DIR, "fivexfad_snrna_attribution.csv")
    if not os.path.exists(path):
        return []
    kinases = {str(r.get("kinase", "")) for r in rows if str(r.get("kinase", ""))}
    if not kinases:
        return []
    ev = pd.read_csv(path)
    ev = ev[ev["kinase"].astype(str).isin(kinases)].copy()
    if "cell_type" in ev.columns:
        ev = ev[~ev["cell_type"].astype(str).str.match(r"^cluster-\d+$", na=False)].copy()
    if ev.empty:
        return []
    if data is not None and not data.celltype_evidence.empty:
        ref_cols = [
            "kinase", "cell_type", "wmb_specificity", "wmb_fold_over_uniform",
            "sea_ad_lfc", "seaad_location_score", "hbca_location_score",
            "human_location_score", "wmb_tier",
        ]
        ref = data.celltype_evidence.copy()
        for col in ref_cols:
            if col not in ref.columns:
                ref[col] = float("nan") if col not in {"kinase", "cell_type", "wmb_tier"} else ""
        ref = ref[ref_cols].drop_duplicates(["kinase", "cell_type"], keep="first")
        ev = ev.merge(ref, on=["kinase", "cell_type"], how="left", suffixes=("", "_ref"))
    for col, default in [
        ("gene_symbol", ""),
        ("cell_type", ""),
        ("confidence_tier", "none"),
        ("confidence_basis", ""),
        ("tissue", ""),
        ("age_months", 0),
        ("wmb_specificity", float("nan")),
        ("wmb_fold_over_uniform", float("nan")),
        ("fivexfad_specificity", float("nan")),
        ("fivexfad_fold_over_uniform", float("nan")),
        ("fivexfad_tau", float("nan")),
        ("fivexfad_top_cluster", ""),
        ("fivexfad_lfc", float("nan")),
        ("fivexfad_pval", float("nan")),
        ("fivexfad_fdr", float("nan")),
        ("n_snrna_samples_wt", float("nan")),
        ("n_snrna_samples_tg", float("nan")),
        ("n_cells_wt", float("nan")),
        ("n_cells_tg", float("nan")),
        ("cluster_source", "new_clusters"),
        ("sea_ad_lfc", float("nan")),
        ("seaad_location_score", float("nan")),
        ("hbca_location_score", float("nan")),
        ("human_location_score", float("nan")),
        ("wmb_tier", "none"),
    ]:
        if col not in ev.columns:
            ev[col] = default
    ev["_rank"] = ev["confidence_tier"].map(_F5_CONF_RANK).fillna(0)
    ev["_f5"] = pd.to_numeric(ev["fivexfad_specificity"], errors="coerce").fillna(-1.0)
    ev["_wmb"] = pd.to_numeric(ev["wmb_specificity"], errors="coerce").fillna(-1.0)
    ev = ev.sort_values(["kinase", "tissue", "age_months", "cell_type", "_rank", "_f5", "_wmb"], ascending=[True, True, True, True, False, False, False])
    ev = ev.drop_duplicates(["kinase", "tissue", "age_months", "cell_type"], keep="first")
    ev = ev.sort_values(["kinase", "tissue", "age_months", "_rank", "_f5", "_wmb"], ascending=[True, True, True, False, False, False])
    cols = [
        "kinase", "gene_symbol", "tissue", "age_months", "cell_type",
        "confidence_tier", "confidence_basis", "wmb_specificity",
        "wmb_fold_over_uniform", "fivexfad_specificity",
        "fivexfad_fold_over_uniform", "fivexfad_tau",
        "fivexfad_top_cluster", "fivexfad_lfc", "fivexfad_pval",
        "fivexfad_fdr", "n_snrna_samples_wt", "n_snrna_samples_tg",
        "n_cells_wt", "n_cells_tg", "cluster_source", "sea_ad_lfc",
        "seaad_location_score", "hbca_location_score", "human_location_score",
        "wmb_tier",
    ]
    return _f5_records(ev, cols)


def _build_fivexfad_attribution_summary_index(attribution_rows: list[dict]) -> list[dict]:
    """Compact 5xFAD attribution summaries for first-load table rendering."""
    if not attribution_rows:
        return []
    grouped: dict[tuple[str, str, int], list[dict]] = {}
    for row in attribution_rows:
        kinase = str(row.get("kinase", ""))
        tissue = str(row.get("tissue", ""))
        age = row.get("age_months")
        if not kinase or not tissue or age is None:
            continue
        grouped.setdefault((kinase, tissue, int(age)), []).append(row)

    out: list[dict] = []
    for (kinase, tissue, age), rows_for_group in sorted(grouped.items()):
        rows_for_group = sorted(rows_for_group, key=cmp_to_key(_f5_attr_record_cmp))
        display_rows = [
            r for r in rows_for_group
            if str(r.get("confidence_tier", "")) in {"very_high", "high", "moderate"}
        ]
        best = rows_for_group[0] if rows_for_group else {}
        best_display = display_rows[0] if display_rows else best
        celltypes = []
        for r in rows_for_group:
            celltypes.append({
                "cell_type": r.get("cell_type"),
                "confidence_tier": r.get("confidence_tier"),
                "fivexfad_specificity": r.get("fivexfad_specificity"),
                "fivexfad_fold_over_uniform": r.get("fivexfad_fold_over_uniform"),
                "fivexfad_tau": r.get("fivexfad_tau"),
                "fivexfad_top_cluster": r.get("fivexfad_top_cluster"),
                "fivexfad_lfc": r.get("fivexfad_lfc"),
                "wmb_specificity": r.get("wmb_specificity"),
                "sea_ad_lfc": r.get("sea_ad_lfc"),
            })
        out.append({
            "kinase": kinase,
            "gene_symbol": best.get("gene_symbol") or kinase,
            "tissue": tissue,
            "age_months": age,
            "high_moderate_celltype_count": len({r.get("cell_type") for r in display_rows if r.get("cell_type")}),
            "best_confidence_tier": best.get("confidence_tier", "none"),
            "top_cell_type": best_display.get("cell_type"),
            "top_fivexfad_specificity": best_display.get("fivexfad_specificity"),
            "top_fivexfad_fold_over_uniform": best_display.get("fivexfad_fold_over_uniform"),
            "top_fivexfad_tau": best_display.get("fivexfad_tau"),
            "top_fivexfad_cluster": best_display.get("fivexfad_top_cluster"),
            "top_fivexfad_lfc": best_display.get("fivexfad_lfc"),
            "top_wmb_specificity": best_display.get("wmb_specificity"),
            "top_sea_ad_lfc": best_display.get("sea_ad_lfc"),
            "celltypes": celltypes,
        })
    return _sanitize(out)


def _assign_fivexfad_song_aligned_confidence(
    attribution_rows: list[dict],
    bulk_rows: list[dict],
) -> list[dict]:
    """Apply Song-style confidence semantics to 5xFAD attribution rows.

    Native 5xFAD snRNA location remains raw evidence. The categorical confidence
    tier requires significant bulk MEA plus snRNA disease-direction support,
    matching the convention users see in the Song attribution pane.
    """
    if not attribution_rows:
        return attribution_rows

    bulk_by_key: dict[tuple[str, str, int], list[dict]] = {}
    for row in bulk_rows:
        if row.get("analysis_track") not in ("", None, "stoichiometry"):
            continue
        age = row.get("age_months")
        if age is None:
            continue
        fdr = _f5_float_or_none(row.get("FDR"))
        nes = _f5_float_or_none(row.get("NES"))
        if fdr is None or nes is None or nes == 0 or fdr >= config.MEA_FDR_THRESH:
            continue
        key = (
            str(row.get("kinase", "")),
            str(row.get("tissue", "")),
            int(age),
        )
        bulk_by_key.setdefault(key, []).append(row)

    out: list[dict] = []
    for row in attribution_rows:
        rec = dict(row)
        age = rec.get("age_months")
        key = (
            str(rec.get("kinase", "")),
            str(rec.get("tissue", "")),
            int(age) if age is not None else -1,
        )
        fold = _f5_float_or_none(rec.get("fivexfad_fold_over_uniform"))
        lfc = _f5_float_or_none(rec.get("fivexfad_lfc"))
        bulk_sig_rows = bulk_by_key.get(key, [])
        sparse_basis = "Fewer than " in str(rec.get("confidence_basis", ""))

        rec["decomp_agrees_bulk"] = False
        if sparse_basis:
            rec["confidence_tier"] = "none"
        elif fold is None:
            rec["confidence_tier"] = "none"
            rec["confidence_basis"] = "No usable 5xFAD snRNA location evidence; confidence tier not applied"
        elif not bulk_sig_rows:
            rec["confidence_tier"] = "none"
            rec["confidence_basis"] = "5xFAD bulk MEA is not significant; Song-aligned confidence tier not applied"
        elif lfc is None or abs(lfc) <= config.SONG_LFC_MIN:
            rec["confidence_tier"] = "none"
            rec["confidence_basis"] = "5xFAD snRNA LFC does not pass the Song direction-support gate"
        else:
            direction_support = any(
                (lfc > 0) == (_f5_float_or_none(bulk.get("NES")) > 0)
                for bulk in bulk_sig_rows
                if _f5_float_or_none(bulk.get("NES")) is not None
            )
            if not direction_support:
                rec["confidence_tier"] = "none"
                rec["confidence_basis"] = "5xFAD snRNA LFC direction does not match significant bulk MEA"
            elif fold >= 2.0:
                rec["confidence_tier"] = "high"
                rec["confidence_basis"] = "5xFAD snRNA direction + tissue-specific high location"
            else:
                rec["confidence_tier"] = "moderate"
                rec["confidence_basis"] = "5xFAD snRNA direction with sub-high tissue-specific location"
        out.append(rec)
    return out


def _promote_fivexfad_attribution_confidence(
    attribution_rows: list[dict],
    bulk_rows: list[dict],
    celltype_rows: list[dict],
) -> list[dict]:
    """Mirror Song confidence promotion for 5xFAD native attribution rows.

    A native 5xFAD high-confidence location row becomes very_high when the
    matching per-cell-type MEA row agrees in sign with the bulk kinase MEA under
    the same decomposition FDR agreement gate used by the Song attribution
    model. This is a categorical cross-check; it does not create or expose a
    synthetic score.
    """
    if not attribution_rows or not celltype_rows:
        return attribution_rows

    bulk_by_key: dict[tuple[str, str, str, int], dict] = {}
    for row in bulk_rows:
        if row.get("analysis_track") not in ("", None, "stoichiometry"):
            continue
        age = row.get("age_months")
        if age is None:
            continue
        key = (
            str(row.get("kinase", "")),
            str(row.get("tissue", "")),
            str(row.get("track", "")),
            int(age),
        )
        bulk_by_key[key] = row

    decomp_by_key: dict[tuple[str, str, int, str], list[dict]] = {}
    for row in celltype_rows:
        age = row.get("age_months")
        if age is None:
            continue
        key = (
            str(row.get("kinase", "")),
            str(row.get("tissue", "")),
            int(age),
            str(row.get("cell_type", "")),
        )
        decomp_by_key.setdefault(key, []).append(row)

    promoted = 0
    out: list[dict] = []
    for row in attribution_rows:
        rec = dict(row)
        if str(rec.get("confidence_tier", "")) != "high":
            out.append(rec)
            continue
        age = rec.get("age_months")
        if age is None:
            out.append(rec)
            continue
        key = (
            str(rec.get("kinase", "")),
            str(rec.get("tissue", "")),
            int(age),
            str(rec.get("cell_type", "")),
        )
        agrees = False
        best: dict | None = None
        for drow in decomp_by_key.get(key, []):
            track = str(drow.get("track", ""))
            bulk = bulk_by_key.get((key[0], key[1], track, key[2]), {})
            bulk_nes = _f5_float_or_none(bulk.get("NES"))
            decomp_nes = _f5_float_or_none(drow.get("NES"))
            decomp_fdr = _f5_float_or_none(drow.get("FDR"))
            if (
                bulk_nes is None
                or decomp_nes is None
                or decomp_fdr is None
                or bulk_nes == 0
                or decomp_nes == 0
                or decomp_fdr >= DECOMP_FDR_AGREEMENT
            ):
                continue
            if (bulk_nes > 0) == (decomp_nes > 0):
                agrees = True
                best = drow
                break
        if agrees:
            rec["confidence_tier"] = "very_high"
            rec["confidence_basis"] = "5xFAD snRNA high + decomp agreement"
            rec["decomp_agrees_bulk"] = True
            rec["decomp_nes"] = _f5_json_value((best or {}).get("NES"))
            rec["decomp_fdr"] = _f5_json_value((best or {}).get("FDR"))
            promoted += 1
        out.append(rec)
    if promoted:
        print(f"  supporting_5xfad_attribution: {promoted:,} high rows promoted to very_high", flush=True)
    return out


def _build_fivexfad_celltype_mea_plot_index(rows: list[dict]) -> list[dict]:
    """Compact embedded decomp rows used for bars and no-fetch fallback views."""
    keep = [
        "kinase", "tissue", "track", "cell_type", "age_months", "NES", "FDR",
        "substrate_hits", "substrate_universe",
    ]
    out = []
    for row in rows:
        rec = {k: row.get(k) for k in keep}
        if rec.get("kinase") and rec.get("tissue") and rec.get("track") and rec.get("cell_type"):
            out.append(rec)
    return _sanitize(out)


def _f5_attr_record_cmp(a: dict, b: dict) -> int:
    cr = _F5_CONF_RANK.get(str(b.get("confidence_tier", "none")), 0) - _F5_CONF_RANK.get(str(a.get("confidence_tier", "none")), 0)
    if cr:
        return cr
    af5 = _f5_sort_num(a.get("fivexfad_specificity"))
    bf5 = _f5_sort_num(b.get("fivexfad_specificity"))
    if af5 != bf5:
        return -1 if bf5 < af5 else 1
    awmb = _f5_sort_num(a.get("wmb_specificity"))
    bwmb = _f5_sort_num(b.get("wmb_specificity"))
    if awmb != bwmb:
        return -1 if bwmb < awmb else 1
    acell = str(a.get("cell_type", "")).lower()
    bcell = str(b.get("cell_type", "")).lower()
    if acell < bcell:
        return -1
    if acell > bcell:
        return 1
    return 0


def _f5_sort_num(v: Any) -> float:
    try:
        if v is None or pd.isna(v):
            return -1.0
        return float(v)
    except Exception:
        return -1.0


def _write_fivexfad_attribution_shards(rows: list[dict]) -> dict[str, str]:
    """Write full per-kinase 5xFAD cell-type attribution sidecars."""
    if not rows:
        return {}
    _attr_src = os.path.join(FIVEXFAD_KINASE_DIR, "fivexfad_snrna_attribution.csv")
    _attr_sig = _input_signature(
        "fivexfad_attribution",
        [__file__, _attr_src],
        {"schema_version": 1, "builder_version": 1},
    )
    _attr_cached = _load_build_cache(
        "fivexfad_attribution", _attr_sig, FIVEXFAD_ATTRIBUTION_DIR
    )
    if _attr_cached is not None:
        return _attr_cached

    tmp_dir = f"{FIVEXFAD_ATTRIBUTION_DIR}.tmp.{os.getpid()}.{uuid.uuid4().hex[:8]}"
    shutil.rmtree(tmp_dir, ignore_errors=True)
    os.makedirs(tmp_dir, exist_ok=True)

    shard_index: dict[str, str] = {}
    by_kinase: dict[str, list[dict]] = {}
    for row in rows:
        kinase = str(row.get("kinase", ""))
        if kinase:
            by_kinase.setdefault(kinase, []).append(row)

    for kinase, kinase_rows in sorted(by_kinase.items()):
        kinase_rows = sorted(kinase_rows, key=cmp_to_key(_f5_attr_record_cmp))
        fname = _f5_shard_name(kinase)
        payload = {
            "schema_version": 1,
            "kinase": kinase,
            "rows": kinase_rows,
        }
        with open(os.path.join(tmp_dir, fname), "w") as f:
            json.dump(_sanitize(payload), f, allow_nan=False, separators=(",", ":"))
        shard_index[kinase] = os.path.relpath(
            os.path.join(FIVEXFAD_ATTRIBUTION_DIR, fname),
            UNIFIED_VIEWER_DIR,
        )

    if shard_index:
        with open(os.path.join(tmp_dir, "index.json"), "w") as f:
            json.dump({"schema_version": 1, "shards": shard_index}, f, separators=(",", ":"))
        shutil.rmtree(FIVEXFAD_ATTRIBUTION_DIR, ignore_errors=True)
        shutil.move(tmp_dir, FIVEXFAD_ATTRIBUTION_DIR)
        _attr_dir_rel = [_f5_shard_name(k) for k in shard_index] + ["index.json"]
        _write_build_cache("fivexfad_attribution", _attr_sig, FIVEXFAD_ATTRIBUTION_DIR,
                           _attr_dir_rel, shard_index)
    else:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    print(f"  supporting_5xfad_attribution: {len(shard_index):,} shards", flush=True)
    return shard_index


def _f5_track_assay(track: str) -> str:
    return "pY" if str(track) == "py" else "IMAC"


def _f5_track_residue(track: str) -> str:
    return "Y" if str(track) == "py" else "ST"


def _build_fivexfad_celltype_mea_rows(rows: list[dict]) -> list[dict]:
    path = os.path.join(FIVEXFAD_CELLTYPE_DIR, "fivexfad_celltype_mea.parquet")
    if not os.path.exists(path):
        return []
    kinases = {str(r.get("kinase", "")) for r in rows if str(r.get("kinase", ""))}
    if not kinases:
        return []
    cols = [
        "tissue", "track", "cell_type", "kinase", "contrast", "NES", "FDR",
        "ES", "p-value", "Subs fraction", "residue_type",
    ]
    schema_names = set(pq.read_schema(path).names)
    table = pq.read_table(path, columns=[c for c in cols if c in schema_names])
    df = table.to_pandas()
    if df.empty:
        return []
    df = df[df["kinase"].astype(str).isin(kinases)].copy()
    if "cell_type" in df.columns:
        df = df[~df["cell_type"].astype(str).str.match(r"^cluster-\d+$", na=False)].copy()
    if df.empty:
        return []
    df["age_months"] = df["contrast"].map(_age_from_contrast_label)
    support = _f5_celltype_contrast_cell_counts()
    if support:
        local_cells = [
            support.get((
                str(row.get("tissue", "")),
                int(row.get("age_months")),
                str(row.get("cell_type", "")),
            ), 0) if pd.notna(row.get("age_months")) else 0
            for row in df.to_dict(orient="records")
        ]
        df = df[np.asarray(local_cells) >= _F5_MIN_CELLS_PER_CONTRAST].copy()
        if df.empty:
            return []
    for col in ["ES", "NES", "FDR", "p-value"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "Subs fraction" in df.columns:
        counts = df["Subs fraction"].map(_subs_fraction_counts)
        df["substrate_hits"] = [x[0] for x in counts]
        df["substrate_universe"] = [x[1] for x in counts]
    else:
        df["substrate_hits"] = np.nan
        df["substrate_universe"] = np.nan
    df["assay"] = df["track"].map(_f5_track_assay)
    df["residue_type"] = df["track"].map(_f5_track_residue)
    out_cols = [
        "kinase", "tissue", "track", "assay", "residue_type", "cell_type",
        "contrast", "age_months", "NES", "FDR", "ES", "p-value",
        "substrate_hits", "substrate_universe",
    ]
    df = df.sort_values(["kinase", "tissue", "track", "contrast", "cell_type"])
    return _f5_records(df, out_cols)


def _build_fivexfad_celltype_agreement_index(
    bulk_rows: list[dict],
    celltype_rows: list[dict],
) -> list[dict]:
    """Compact categorical agreement calls for the 5xFAD main table."""
    bulk_by_key: dict[tuple[str, str, str, int], dict] = {}
    for row in bulk_rows:
        if row.get("analysis_track") not in ("", None, "stoichiometry"):
            continue
        age = row.get("age_months")
        if age is None:
            continue
        key = (
            str(row.get("kinase", "")),
            str(row.get("tissue", "")),
            str(row.get("track", "")),
            int(age),
        )
        bulk_by_key[key] = row

    by_key: dict[tuple[str, str, str, int], list[dict]] = {}
    for row in celltype_rows:
        age = row.get("age_months")
        if age is None:
            continue
        key = (
            str(row.get("kinase", "")),
            str(row.get("tissue", "")),
            str(row.get("track", "")),
            int(age),
        )
        by_key.setdefault(key, []).append(row)

    fdr_gate = config.MEA_FDR_THRESH
    out: list[dict] = []
    for key in sorted(set(bulk_by_key) | set(by_key)):
        kinase, tissue, track, age = key
        bulk = bulk_by_key.get(key, {})
        rows_for_key = by_key.get(key, [])
        bulk_nes = _f5_json_value(bulk.get("NES"))
        bulk_fdr = _f5_json_value(bulk.get("FDR"))
        bulk_sig = (
            bulk_nes is not None
            and bulk_fdr is not None
            and float(bulk_fdr) < fdr_gate
        )
        sig_rows = [
            r for r in rows_for_key
            if _f5_json_value(r.get("NES")) not in (None, 0)
            and _f5_json_value(r.get("FDR")) is not None
            and float(r.get("FDR")) < fdr_gate
        ]
        same = [
            r for r in sig_rows
            if bulk_nes is not None and (float(r.get("NES")) > 0) == (float(bulk_nes) > 0)
        ]
        opposite = [
            r for r in sig_rows
            if bulk_nes is not None and (float(r.get("NES")) > 0) != (float(bulk_nes) > 0)
        ]
        if not bulk_sig and not sig_rows:
            state = "none"
        elif not bulk_sig:
            state = "decomp_only"
        elif not sig_rows:
            state = "bulk_only"
        elif same and not opposite:
            state = "agree"
        else:
            state = "mixed" if same else "disagree"

        top = None
        if sig_rows:
            top = sorted(sig_rows, key=lambda r: abs(float(r.get("NES") or 0)), reverse=True)[0]
        hits, universe = _subs_fraction_counts(bulk.get("Subs fraction"))
        out.append({
            "kinase": kinase,
            "tissue": tissue,
            "track": track,
            "assay": _f5_track_assay(track),
            "residue_type": _f5_track_residue(track),
            "contrast": bulk.get("contrast") or (top or {}).get("contrast") or f"TG_vs_WT_{age}mo",
            "age_months": age,
            "agreement_state": state,
            "bulk_NES": bulk_nes,
            "bulk_FDR": bulk_fdr,
            "bulk_substrate_hits": bulk.get("substrate_hits", hits),
            "bulk_substrate_universe": bulk.get("substrate_universe", universe),
            "decomp_celltype_count": len(rows_for_key),
            "decomp_sig_celltype_count": len(sig_rows),
            "decomp_same_direction_count": len(same),
            "decomp_opposite_direction_count": len(opposite),
            "top_cell_type": (top or {}).get("cell_type"),
            "top_celltype_NES": _f5_json_value((top or {}).get("NES")),
            "top_celltype_FDR": _f5_json_value((top or {}).get("FDR")),
            "top_celltype_substrate_hits": (top or {}).get("substrate_hits"),
            "top_celltype_substrate_universe": (top or {}).get("substrate_universe"),
        })
    return _sanitize(out)


def _write_fivexfad_celltype_mea_shards(rows: list[dict]) -> dict[str, str]:
    """Write full per-kinase 5xFAD cell-type MEA sidecars for lazy detail views."""
    if not rows:
        return {}
    _mea_src = os.path.join(FIVEXFAD_CELLTYPE_DIR, "fivexfad_celltype_mea.parquet")
    _mea_sig = _input_signature(
        "fivexfad_celltype_mea",
        [__file__, _mea_src],
        {"schema_version": 1, "builder_version": 1},
    )
    _mea_cached = _load_build_cache(
        "fivexfad_celltype_mea", _mea_sig, FIVEXFAD_CELLTYPE_MEA_DIR
    )
    if _mea_cached is not None:
        return _mea_cached

    tmp_dir = f"{FIVEXFAD_CELLTYPE_MEA_DIR}.tmp.{os.getpid()}.{uuid.uuid4().hex[:8]}"
    shutil.rmtree(tmp_dir, ignore_errors=True)
    os.makedirs(tmp_dir, exist_ok=True)

    shard_index: dict[str, str] = {}
    by_kinase: dict[str, list[dict]] = {}
    for row in rows:
        kinase = str(row.get("kinase", ""))
        if kinase:
            by_kinase.setdefault(kinase, []).append(row)

    for kinase, kinase_rows in sorted(by_kinase.items()):
        fname = _f5_shard_name(kinase)
        payload = {
            "schema_version": 1,
            "kinase": kinase,
            "rows": kinase_rows,
        }
        with open(os.path.join(tmp_dir, fname), "w") as f:
            json.dump(_sanitize(payload), f, allow_nan=False, separators=(",", ":"))
        shard_index[kinase] = os.path.relpath(
            os.path.join(FIVEXFAD_CELLTYPE_MEA_DIR, fname),
            UNIFIED_VIEWER_DIR,
        )

    if shard_index:
        with open(os.path.join(tmp_dir, "index.json"), "w") as f:
            json.dump({"schema_version": 1, "shards": shard_index}, f, separators=(",", ":"))
        shutil.rmtree(FIVEXFAD_CELLTYPE_MEA_DIR, ignore_errors=True)
        shutil.move(tmp_dir, FIVEXFAD_CELLTYPE_MEA_DIR)
        _mea_dir_rel = [_f5_shard_name(k) for k in shard_index] + ["index.json"]
        _write_build_cache("fivexfad_celltype_mea", _mea_sig, FIVEXFAD_CELLTYPE_MEA_DIR,
                           _mea_dir_rel, shard_index)
    else:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    print(f"  supporting_5xfad_celltype_mea: {len(shard_index):,} shards", flush=True)
    return shard_index


def _write_fivexfad_index_shard(rows: list[dict], filename: str, label: str) -> str | None:
    """Write a whole 5xFAD index list to one gzipped sidecar; return its viewer-relative URL.

    These indexes are iterated in full at first 5xFAD/Crosstable render rather than
    accessed per-kinase, so a single fetched-on-demand file keeps them out of the
    upfront payload parse without forcing a per-kinase fan-out. Returns None when there
    are no rows (the JS treats a missing path as an empty index)."""
    if not rows:
        return None
    os.makedirs(FIVEXFAD_INDEX_DIR, exist_ok=True)
    out_path = os.path.join(FIVEXFAD_INDEX_DIR, filename)
    tmp_path = f"{out_path}.tmp.{os.getpid()}.{uuid.uuid4().hex[:8]}"
    raw = json.dumps(
        _sanitize({"schema_version": 1, "rows": rows}),
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")
    with gzip.open(tmp_path, "wb", compresslevel=6) as f:
        f.write(raw)
    os.replace(tmp_path, out_path)
    rel = os.path.relpath(out_path, UNIFIED_VIEWER_DIR)
    print(f"  {label}: {len(rows):,} rows -> {rel}", flush=True)
    return rel


def _write_fivexfad_celltype_ols_shards(rows: list[dict]) -> dict[str, str]:
    """Write per-kinase 5xFAD cell-type substrate-site OLS sidecars."""
    site_path = os.path.join(FIVEXFAD_CELLTYPE_DIR, "fivexfad_celltype_site_level_ols.parquet")
    subs_path = os.path.join(FIVEXFAD_CELLTYPE_DIR, "fivexfad_celltype_substrate_sets.csv")
    if not rows or not os.path.exists(site_path) or not os.path.exists(subs_path):
        return {}

    kinases = sorted({str(r.get("kinase", "")) for r in rows if r.get("kinase")})
    if not kinases:
        return {}

    existing_index = os.path.join(FIVEXFAD_CELLTYPE_OLS_DIR, "index.json")
    if os.path.exists(existing_index):
        try:
            with open(existing_index) as f:
                existing = json.load(f)
            shards = existing.get("shards", {})
            if isinstance(shards, dict) and len(shards) > 1:
                print(f"  supporting_5xfad_celltype_ols: {len(shards):,} shards (reused)", flush=True)
                return shards
            if isinstance(shards, dict) and shards:
                print(
                    "  supporting_5xfad_celltype_ols: existing shard index is incomplete; "
                    "skipping heavy rebuild unless FIVEXFAD_REBUILD_CELLTYPE_OLS=1",
                    flush=True,
                )
                if os.environ.get("FIVEXFAD_REBUILD_CELLTYPE_OLS") != "1":
                    shutil.rmtree(FIVEXFAD_CELLTYPE_OLS_DIR, ignore_errors=True)
                    return {}
        except (OSError, json.JSONDecodeError):
            pass
    elif os.environ.get("FIVEXFAD_REBUILD_CELLTYPE_OLS") != "1":
        print(
            "  supporting_5xfad_celltype_ols: skipped heavy shard build "
            "(set FIVEXFAD_REBUILD_CELLTYPE_OLS=1 to regenerate)",
            flush=True,
        )
        return {}

    subs = pd.read_csv(subs_path)
    subs = subs[subs["kinase"].astype(str).isin(kinases)].copy()
    if "cell_type" in subs.columns:
        subs = subs[~subs["cell_type"].astype(str).str.match(r"^cluster-\d+$", na=False)].copy()
    if subs.empty:
        return {}
    subs["motif_norm"] = subs["motif"].map(_f5_norm_motif)
    key_cols = ["tissue", "track", "cell_type", "contrast", "motif_norm"]
    selector = subs[key_cols + ["kinase"]].dropna(subset=["motif_norm"]).drop_duplicates()

    site_cols = [
        "tissue", "track", "cell_type", "contrast", "site_id", "gene_symbol",
        "motif", "lfc", "se", "t", "pval", "fdr", "n_obs", "n_wt", "n_tg",
    ]
    site = pq.read_table(site_path, columns=site_cols).to_pandas()
    site = site[~site["cell_type"].astype(str).str.match(r"^cluster-\d+$", na=False)].copy()
    site["motif_norm"] = site["motif"].map(_f5_norm_motif)
    joined = site.merge(selector, on=key_cols, how="inner")
    if joined.empty:
        return {}

    tmp_dir = f"{FIVEXFAD_CELLTYPE_OLS_DIR}.tmp.{os.getpid()}.{uuid.uuid4().hex[:8]}"
    shutil.rmtree(tmp_dir, ignore_errors=True)
    os.makedirs(tmp_dir, exist_ok=True)

    shard_index: dict[str, str] = {}
    keep_cols = [
        "tissue", "track", "cell_type", "contrast", "site_id", "gene_symbol",
        "motif", "lfc", "se", "t", "pval", "fdr", "n_obs", "n_wt", "n_tg",
    ]
    for kinase, g in joined.groupby("kinase", sort=True):
        out = g[keep_cols].drop_duplicates().copy()
        out = out.sort_values(["tissue", "track", "contrast", "cell_type", "lfc"], ascending=[True, True, True, True, False])
        fname = _f5_shard_name(str(kinase))
        payload = {
            "schema_version": 1,
            "kinase": str(kinase),
            "rows": _f5_records(out),
        }
        with open(os.path.join(tmp_dir, fname), "w") as f:
            json.dump(payload, f, allow_nan=False, separators=(",", ":"))
        shard_index[str(kinase)] = os.path.relpath(os.path.join(FIVEXFAD_CELLTYPE_OLS_DIR, fname), UNIFIED_VIEWER_DIR)

    if shard_index:
        with open(os.path.join(tmp_dir, "index.json"), "w") as f:
            json.dump({"schema_version": 1, "shards": shard_index}, f, separators=(",", ":"))
        shutil.rmtree(FIVEXFAD_CELLTYPE_OLS_DIR, ignore_errors=True)
        shutil.move(tmp_dir, FIVEXFAD_CELLTYPE_OLS_DIR)
    else:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    print(f"  supporting_5xfad_celltype_ols: {len(shard_index):,} shards", flush=True)
    return shard_index


def _f5_prerank_for_contrast(
    ols: pd.DataFrame,
    track_shift: pd.DataFrame,
    track_winsor: pd.DataFrame,
    effect_prefix: str,
    contrast: str,
) -> pd.DataFrame:
    lfc_col = f"{effect_prefix}_lfc_{contrast}"
    if lfc_col not in ols.columns:
        return pd.DataFrame()
    p_col = f"{effect_prefix}_pval_{contrast}"
    fdr_col = f"{effect_prefix}_fdr_{contrast}"
    wt_col = f"{effect_prefix}_n_wt_{contrast}"
    tg_col = f"{effect_prefix}_n_tg_{contrast}"
    cols = [
        "site_id", "gene_symbol", "motif", "site_position", "residue_type",
        "matched_protein", "n_obs_stoich", "n_obs_raw", lfc_col, p_col,
        fdr_col, wt_col, tg_col,
    ]
    out = ols[[c for c in cols if c in ols.columns]].copy()
    if "site_label" not in out.columns:
        out["site_label"] = out.apply(_f5_site_label_value, axis=1)
    out = out.rename(columns={
        lfc_col: "lfc",
        p_col: "p_value",
        fdr_col: "fdr",
        wt_col: "n_wt",
        tg_col: "n_tg",
    })
    out["lfc"] = pd.to_numeric(out["lfc"], errors="coerce")
    out = out[np.isfinite(out["lfc"])].copy()
    if out.empty:
        return out

    shift_val = 0.0
    if not track_shift.empty and "contrast" in track_shift.columns:
        shift_rows = track_shift[track_shift["contrast"] == contrast]
        if not shift_rows.empty and "median_shift" in shift_rows.columns:
            val = pd.to_numeric(shift_rows["median_shift"].iloc[0], errors="coerce")
            if pd.notna(val):
                shift_val = float(val)
    out["centered_lfc"] = out["lfc"] - shift_val

    lower = upper = None
    clipped_site_ids: set[str] = set()
    if not track_winsor.empty and "contrast" in track_winsor.columns:
        w = track_winsor[track_winsor["contrast"] == contrast]
        if not w.empty:
            if "site_id" in w.columns:
                clipped_site_ids = set(w["site_id"].dropna().astype(str))
            if "lower_bound" in w.columns and "upper_bound" in w.columns:
                lo = pd.to_numeric(w["lower_bound"].dropna(), errors="coerce").dropna()
                hi = pd.to_numeric(w["upper_bound"].dropna(), errors="coerce").dropna()
                if not lo.empty and not hi.empty:
                    lower = float(lo.iloc[0])
                    upper = float(hi.iloc[0])
    if lower is not None and upper is not None:
        out["clipped_lfc"] = out["centered_lfc"].clip(lower, upper)
    else:
        out["clipped_lfc"] = out["centered_lfc"]
    out["was_winsorized"] = np.where(
        out["site_id"].astype(str).isin(clipped_site_ids)
        | ~np.isclose(out["clipped_lfc"], out["centered_lfc"], equal_nan=True),
        "yes",
        "no",
    )
    out = out.sort_values("clipped_lfc", ascending=False, kind="mergesort").reset_index(drop=True)
    out["rank_in_contrast"] = np.arange(1, len(out) + 1)
    out["contrast"] = contrast
    return out


def _f5_running_enrichment(prerank: pd.DataFrame, motif_set: set[str]) -> dict[str, Any] | None:
    if prerank.empty or not motif_set:
        return None
    motifs = {_f5_norm_motif(m) for m in motif_set if _f5_norm_motif(m)}
    if not motifs:
        return None
    ranked_cols = ["rank_in_contrast", "site_id", "site_label", "gene_symbol", "motif", "clipped_lfc"]
    ranked = prerank[[c for c in ranked_cols if c in prerank.columns]].copy()
    if "site_label" not in ranked.columns:
        ranked["site_label"] = ranked.apply(_f5_site_label_value, axis=1)
    ranked["motif_norm"] = ranked["motif"].map(_f5_norm_motif)
    is_hit = ranked["motif_norm"].isin(motifs).to_numpy()
    n = len(ranked)
    nh = int(is_hit.sum())
    if n == 0 or nh == 0:
        return None
    weights = pd.to_numeric(ranked["clipped_lfc"], errors="coerce").abs().fillna(0.0).to_numpy()
    hit_sum = float(weights[is_hit].sum())
    if hit_sum <= 0:
        return None
    miss_rate = 1.0 / max(n - nh, 1)
    running: list[float] = []
    hit_indices: list[int] = []
    cur = 0.0
    peak = 0.0
    peak_idx = 0
    for i in range(n):
        if is_hit[i]:
            cur += float(weights[i]) / hit_sum
            hit_indices.append(i)
        else:
            cur -= miss_rate
        running.append(cur)
        if abs(cur) > abs(peak):
            peak = cur
            peak_idx = i

    if n <= FIVEXFAD_RUNNING_DISPLAY_POINTS:
        line_idx = list(range(n))
    else:
        line_idx = sorted(set(np.linspace(0, n - 1, FIVEXFAD_RUNNING_DISPLAY_POINTS, dtype=int).tolist() + [peak_idx]))
    line = [
        {"rank": int(ranked["rank_in_contrast"].iloc[i]), "running_es": _f5_json_value(running[i])}
        for i in line_idx
    ]
    hits = []
    for i in hit_indices:
        r = ranked.iloc[i]
        hits.append({
            "rank": int(r["rank_in_contrast"]),
            "running_es": _f5_json_value(running[i]),
            "site_id": _f5_json_value(r.get("site_id")),
            "site_label": _f5_json_value(r.get("site_label")),
            "gene_symbol": _f5_json_value(r.get("gene_symbol")),
            "motif": _f5_json_value(r.get("motif")),
            "clipped_lfc": _f5_json_value(r.get("clipped_lfc")),
        })
    leading_edge_count = (
        sum(1 for i in hit_indices if i <= peak_idx)
        if peak >= 0 else sum(1 for i in hit_indices if i >= peak_idx)
    )
    return {
        "n_ranked": n,
        "n_hits": nh,
        "peak_rank": int(ranked["rank_in_contrast"].iloc[peak_idx]),
        "peak_es": _f5_json_value(peak),
        "leading_edge_count": int(leading_edge_count),
        "line": line,
        "hits": hits,
    }


def _write_fivexfad_detail_shards(
    track_specs: list[tuple[str, str, str, str]],
    rows: list[dict],
    manifest_path: str,
) -> dict[str, str]:
    """Write per-kinase 5xFAD audit sidecars for the detail workbench."""
    if not rows:
        return {}
    _detail_source_files = [__file__, manifest_path]
    for tissue, track, _assay, _residue in track_specs:
        prefix = f"{tissue}_{track}"
        for suffix in (
            "_raw_phospho_normalized.csv",
            "_matched_total_protein.csv",
            "_stoichiometry_matrix.csv",
            "_site_level_ols.csv",
            "_mea_substrate_sets.csv",
            "_winsorized_sites.csv",
            "_mea_global_shift.csv",
        ):
            _detail_source_files.append(os.path.join(FIVEXFAD_KINASE_DIR, f"{prefix}{suffix}"))
    _detail_sig = _input_signature(
        "fivexfad_detail",
        _detail_source_files,
        {"schema_version": 2, "builder_version": 1},
    )
    _detail_cached = _load_build_cache("fivexfad_detail", _detail_sig, FIVEXFAD_DETAIL_DIR)
    if _detail_cached is not None:
        return _detail_cached

    tmp_dir = f"{FIVEXFAD_DETAIL_DIR}.tmp.{os.getpid()}.{uuid.uuid4().hex[:8]}"
    shutil.rmtree(tmp_dir, ignore_errors=True)
    os.makedirs(tmp_dir, exist_ok=True)

    manifest = pd.read_csv(manifest_path) if os.path.exists(manifest_path) else pd.DataFrame()
    manifest_by_assay: dict[tuple[str, str], pd.DataFrame] = {}
    if not manifest.empty:
        primary = manifest[manifest["analysis_action"] == "primary"].copy()
        for (tissue, assay), g in primary.groupby(["tissue", "assay"], sort=False):
            manifest_by_assay[
                (str(tissue), _f5_manifest_assay_key(str(assay)))
            ] = g.drop_duplicates("biological_sample_id")

    rows_by_group: dict[tuple[str, str, str, str, str], list[dict]] = {}
    for row in rows:
        key = (
            str(row.get("tissue", "")),
            str(row.get("track", "")),
            str(row.get("assay", "")),
            str(row.get("analysis_track", "")),
            str(row.get("kinase", "")),
        )
        rows_by_group.setdefault(key, []).append(row)

    # In-memory accumulator: kinase → {key: payload_dict}
    detail_parts_by_kinase: dict[str, dict[str, dict]] = {}
    for tissue, track, assay, residue in track_specs:
        prefix = f"{tissue}_{track}"
        paths = {
            "raw": os.path.join(FIVEXFAD_KINASE_DIR, f"{prefix}_raw_phospho_normalized.csv"),
            "protein": os.path.join(FIVEXFAD_KINASE_DIR, f"{prefix}_matched_total_protein.csv"),
            "stoich": os.path.join(FIVEXFAD_KINASE_DIR, f"{prefix}_stoichiometry_matrix.csv"),
            "ols": os.path.join(FIVEXFAD_KINASE_DIR, f"{prefix}_site_level_ols.csv"),
            "subs": os.path.join(FIVEXFAD_KINASE_DIR, f"{prefix}_mea_substrate_sets.csv"),
            "winsor": os.path.join(FIVEXFAD_KINASE_DIR, f"{prefix}_winsorized_sites.csv"),
            "shift": os.path.join(FIVEXFAD_KINASE_DIR, f"{prefix}_mea_global_shift.csv"),
        }
        if not all(os.path.exists(paths[k]) for k in ["raw", "protein", "stoich", "ols", "subs"]):
            continue

        raw = pd.read_csv(paths["raw"])
        protein = pd.read_csv(paths["protein"])
        stoich = pd.read_csv(paths["stoich"])
        ols = pd.read_csv(paths["ols"])
        subs = pd.read_csv(paths["subs"])
        winsor = pd.read_csv(paths["winsor"]) if os.path.exists(paths["winsor"]) else pd.DataFrame()
        shift = pd.read_csv(paths["shift"]) if os.path.exists(paths["shift"]) else pd.DataFrame()

        meta_cols = ["site_id", "gene_symbol", "motif", "site_position", "residue_type", "matched_protein"]
        stoich_meta = stoich[[c for c in meta_cols if c in stoich.columns]].copy()
        ols = ols.merge(
            stoich_meta[[c for c in ["site_id", "motif", "site_position", "residue_type"] if c in stoich_meta.columns]],
            on="site_id",
            how="left",
        )
        sample_cols = [
            c for c in stoich.columns
            if c not in {"site_id", "gene_symbol", "motif", "site_position", "residue_type", "matched_protein"}
        ]
        sample_meta = {}
        sm = manifest_by_assay.get((tissue, _f5_manifest_assay_key(assay)), pd.DataFrame())
        if not sm.empty:
            for _, srow in sm.iterrows():
                sid = str(srow.get("biological_sample_id", ""))
                sample_meta[sid] = {
                    "sample": sid,
                    "age_months": int(srow.get("age_months", 0) or 0),
                    "age": str(srow.get("age", "")),
                    "genotype": str(srow.get("genotype", "")),
                }

        raw_by_site = raw.set_index("site_id", drop=False)
        protein_by_site = protein.set_index("site_id", drop=False)
        stoich_by_site = stoich.set_index("site_id", drop=False)

        for analysis_track in ["stoichiometry", "raw_phospho"]:
            effect_prefix = "stoich" if analysis_track == "stoichiometry" else "raw"
            track_subs = subs[subs["analysis_track"] == analysis_track] if "analysis_track" in subs.columns else subs.iloc[0:0]
            track_winsor = winsor[winsor["analysis_track"] == analysis_track] if "analysis_track" in winsor.columns else pd.DataFrame()
            track_shift = shift[shift["analysis_track"] == analysis_track] if "analysis_track" in shift.columns else pd.DataFrame()
            kinases = sorted({
                key[4] for key in rows_by_group
                if key[0] == tissue and key[1] == track and key[3] == analysis_track
            })
            contrasts = sorted({
                str(r.get("contrast", ""))
                for key, group_rows in rows_by_group.items()
                if key[0] == tissue and key[1] == track and key[3] == analysis_track
                for r in group_rows
            })
            prerank_by_contrast = {
                contrast: _f5_prerank_for_contrast(
                    ols, track_shift, track_winsor, effect_prefix, contrast
                )
                for contrast in contrasts
            }

            for kinase in kinases:
                group_rows = rows_by_group.get((tissue, track, assay, analysis_track, kinase), [])
                site_frames = []
                prepared_frames = []
                running_rows = []
                for grow in group_rows:
                    contrast = str(grow.get("contrast", ""))
                    motif_set = set(
                        track_subs[
                            (track_subs["kinase"] == kinase)
                            & (track_subs["contrast"] == contrast)
                        ]["motif"].dropna().astype(str)
                    )
                    kl_by_motif: dict[str, float] = {}
                    if "kl_percentile" in track_subs.columns:
                        sub_rows = track_subs[
                            (track_subs["kinase"] == kinase)
                            & (track_subs["contrast"] == contrast)
                        ]
                        for _, srow in sub_rows.iterrows():
                            motif_key = _f5_norm_motif(srow.get("motif"))
                            pct = pd.to_numeric(srow.get("kl_percentile"), errors="coerce")
                            if motif_key and pd.notna(pct):
                                kl_by_motif[motif_key] = float(pct)
                    if not motif_set:
                        continue
                    prerank = prerank_by_contrast.get(contrast, pd.DataFrame())
                    if prerank.empty:
                        continue
                    motif_norm = {_f5_norm_motif(m) for m in motif_set}
                    sub = prerank[prerank["motif"].map(_f5_norm_motif).isin(motif_norm)].copy()
                    if sub.empty:
                        continue
                    leading_motifs = {
                        _f5_norm_motif(x) for x in str(grow.get("leading_substrates", "")).split(";")
                        if _f5_norm_motif(x)
                    }
                    sub["in_leading_edge"] = np.where(
                        sub["motif"].map(_f5_norm_motif).isin(leading_motifs),
                        "yes",
                        "no",
                    )
                    sub["kl_percentile"] = sub["motif"].map(
                        lambda m: kl_by_motif.get(_f5_norm_motif(m))
                    )
                    if "site_label" not in sub.columns:
                        sub["site_label"] = sub.apply(_f5_site_label_value, axis=1)
                    sub = sub.sort_values("rank_in_contrast").head(FIVEXFAD_DETAIL_SITES_PER_CONTRAST)
                    site_frames.append(sub)
                    prepared_frames.append(sub)
                    running = _f5_running_enrichment(prerank, motif_set)
                    if running is not None:
                        running_rows.append({"contrast": contrast, **running})

                site_stats = pd.concat(site_frames, ignore_index=True) if site_frames else pd.DataFrame()
                prepared_input = pd.concat(prepared_frames, ignore_index=True) if prepared_frames else pd.DataFrame()
                if site_stats.empty:
                    site_ids = []
                else:
                    site_stats = site_stats.sort_values(["contrast", "rank_in_contrast"])
                    site_ids = list(dict.fromkeys(site_stats["site_id"].astype(str).tolist()))[:FIVEXFAD_DETAIL_MAX_SITES]
                    site_stats = site_stats[site_stats["site_id"].astype(str).isin(site_ids)]
                    if not prepared_input.empty:
                        prepared_input = prepared_input[prepared_input["site_id"].astype(str).isin(site_ids)]
                kl_by_site: dict[tuple[str, str], Any] = {}
                if not site_stats.empty and "kl_percentile" in site_stats.columns:
                    for _, srow in site_stats.iterrows():
                        kl_by_site[(str(srow.get("contrast", "")), str(srow.get("site_id", "")))] = _f5_json_value(srow.get("kl_percentile"))

                measurement_rows = []
                for site_id in site_ids:
                    if site_id not in stoich_by_site.index:
                        continue
                    srow = stoich_by_site.loc[site_id]
                    rrow = raw_by_site.loc[site_id] if site_id in raw_by_site.index else {}
                    prow = protein_by_site.loc[site_id] if site_id in protein_by_site.index else {}
                    for sample in sample_cols:
                        smeta = sample_meta.get(sample, {"sample": sample, "age_months": None, "age": "", "genotype": ""})
                        contrast_label = f"TG_vs_WT_{smeta.get('age_months')}mo" if smeta.get("age_months") else ""
                        measurement_rows.append({
                            "site_id": site_id,
                            "site_label": _f5_site_label_value(srow),
                            "gene_symbol": _f5_json_value(srow.get("gene_symbol")),
                            "motif": _f5_json_value(srow.get("motif")),
                            "kl_percentile": kl_by_site.get((contrast_label, site_id)),
                            "sample": sample,
                            "age_months": smeta.get("age_months"),
                            "age": smeta.get("age"),
                            "genotype": smeta.get("genotype"),
                            "raw_phospho": _f5_json_value(rrow.get(sample) if hasattr(rrow, "get") else None),
                            "matched_total_protein": _f5_json_value(prow.get(sample) if hasattr(prow, "get") else None),
                            "stoichiometry": _f5_json_value(srow.get(sample)),
                        })

                substrate_summary = (
                    track_subs[track_subs["kinase"] == kinase]
                    .groupby("contrast")["motif"]
                    .nunique()
                    .rename("substrate_motifs")
                    .reset_index()
                ) if not track_subs.empty else pd.DataFrame()
                key = _f5_group_key(kinase, tissue, assay, analysis_track)
                payload = {
                    "schema_version": 1,
                    "key": key,
                    "kinase": kinase,
                    "tissue": tissue,
                    "track": track,
                    "assay": assay,
                    "residue_type": residue,
                    "analysis_track": analysis_track,
                    "sample_columns": [sample_meta.get(s, {"sample": s}) for s in sample_cols],
                    "measurement_trace": measurement_rows,
                    "site_stats": _f5_records(site_stats),
                    "prepared_mea_input": _f5_records(prepared_input),
                    "running_enrichment": running_rows,
                    "global_shift": _f5_records(track_shift),
                    "winsorized_sites": _f5_records(
                        track_winsor[track_winsor["site_id"].astype(str).isin(site_ids)].assign(
                            site_label=lambda d: d.apply(_f5_site_label_value, axis=1),
                        ) if not track_winsor.empty and site_ids else pd.DataFrame()
                    ),
                    "substrate_summary": _f5_records(substrate_summary),
                    "source_files": [os.path.basename(paths[k]) for k in paths if os.path.exists(paths[k])],
                }
                detail_parts_by_kinase.setdefault(kinase, {})[key] = _sanitize(payload)

    detail_index: dict[str, str] = {}
    for kinase, parts in sorted(detail_parts_by_kinase.items()):
        details: dict[str, dict] = {key: parts[key] for key in sorted(parts)}
        fname = _f5_shard_name(kinase) + ".gz"
        bundle = {
            "schema_version": 2,
            "layout": "per_kinase_bundle",
            "kinase": kinase,
            "details": details,
        }
        raw = json.dumps(bundle, allow_nan=False, separators=(",", ":")).encode("utf-8")
        with gzip.open(os.path.join(tmp_dir, fname), "wb", compresslevel=6) as f:
            f.write(raw)
        detail_index[kinase] = os.path.relpath(os.path.join(FIVEXFAD_DETAIL_DIR, fname), UNIFIED_VIEWER_DIR)

    if detail_index:
        with open(os.path.join(tmp_dir, "index.json"), "w") as f:
            json.dump(
                {
                    "schema_version": 2,
                    "layout": "per_kinase_bundle_v2",
                    "shards": detail_index,
                },
                f,
                separators=(",", ":"),
            )
        shutil.rmtree(FIVEXFAD_DETAIL_DIR, ignore_errors=True)
        shutil.move(tmp_dir, FIVEXFAD_DETAIL_DIR)
        _detail_dir_rel = [
            os.path.basename(v) for v in detail_index.values()
        ] + ["index.json"]
        _write_build_cache("fivexfad_detail", _detail_sig, FIVEXFAD_DETAIL_DIR,
                           _detail_dir_rel, detail_index)
    else:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    print(f"  supporting_5xfad_detail: {len(detail_index):,} shards", flush=True)
    return detail_index


def build_supporting_5xfad_slice(data: UnifiedData | None = None) -> dict | None:
    """Supporting 5xFAD kinase-enrichment slice.

    5xFAD is intentionally not a schema-v2 context. Tissue is a filter
    attribute inside this supporting block so Song remains the default AD
    context and cortex/hippocampus do not become viewer tabs.
    """
    if not os.path.isdir(FIVEXFAD_KINASE_DIR):
        return None

    track_specs = [
        ("cortex", "st", "IMAC", "ST"),
        ("cortex", "py", "pY", "Y"),
        ("hippocampus", "st", "IMAC", "ST"),
        ("hippocampus", "py", "pY", "Y"),
    ]
    analysis_files = [
        ("stoichiometry", "mea_stoichiometry"),
        ("raw_phospho", "mea_raw_phospho"),
    ]

    qc_lookup: dict[tuple[str, str, str], dict] = {}
    qc_rows: list[dict] = []
    for tissue, track, assay, residue in track_specs:
        qc_path = os.path.join(FIVEXFAD_KINASE_DIR, f"{tissue}_{track}_contrast_qc.csv")
        if not os.path.exists(qc_path):
            continue
        qdf = pd.read_csv(qc_path)
        for _, row in qdf.iterrows():
            rec = {
                "tissue": tissue,
                "track": track,
                "assay": assay,
                "residue_type": residue,
                "contrast": str(row.get("contrast", "")),
                "age_months": int(row.get("age_months", 0) or 0),
                "n_wt": int(row.get("n_wt", 0) or 0),
                "n_tg": int(row.get("n_tg", 0) or 0),
                "contrast_status": str(row.get("contrast_status", "")),
            }
            qc_rows.append(rec)
            qc_lookup[(tissue, track, rec["contrast"])] = rec

    rows: list[dict] = []
    source_files: list[str] = []
    kinase_gene_map = _f5_kinase_gene_map()
    for tissue, track, assay, residue in track_specs:
        for analysis_track, basename in analysis_files:
            path = os.path.join(FIVEXFAD_KINASE_DIR, f"{tissue}_{track}_{basename}.csv")
            if not os.path.exists(path):
                continue
            source_files.append(os.path.relpath(path, UNIFIED_VIEWER_DIR))
            df = pd.read_csv(path)
            for _, row in df.iterrows():
                contrast = str(row.get("contrast", ""))
                qc = qc_lookup.get((tissue, track, contrast), {})
                hits, universe = _subs_fraction_counts(row.get("Subs fraction"))
                leading = str(row.get("Leading substrates", ""))
                leading_count = 0 if not leading else len([x for x in leading.split(";") if x])
                gene_symbol = row.get("gene_symbol", row.get("kinase", ""))
                if pd.isna(gene_symbol):
                    gene_symbol = row.get("kinase", "")
                kinase = str(row.get("kinase", ""))
                if not str(gene_symbol or "").strip() or str(gene_symbol) == kinase:
                    gene_symbol = kinase_gene_map.get(kinase, gene_symbol)
                rows.append({
                    "kinase": kinase,
                    "gene_symbol": str(gene_symbol),
                    "tissue": tissue,
                    "track": track,
                    "analysis_track": analysis_track,
                    "assay": assay,
                    "residue_type": residue,
                    "contrast": contrast,
                    "age_months": int(qc.get("age_months", _age_from_contrast_label(contrast)) or 0),
                    "NES": float(row.get("NES")) if pd.notna(row.get("NES")) else None,
                    "FDR": float(row.get("FDR")) if pd.notna(row.get("FDR")) else None,
                    "ES": float(row.get("ES")) if pd.notna(row.get("ES")) else None,
                    "p_value": float(row.get("p-value")) if pd.notna(row.get("p-value")) else None,
                    "substrate_hits": hits,
                    "substrate_universe": universe,
                    "leading_substrate_count": leading_count,
                    "leading_substrates": leading,
                    "n_wt": qc.get("n_wt"),
                    "n_tg": qc.get("n_tg"),
                    "contrast_status": qc.get("contrast_status", ""),
                })

    if not rows and not qc_rows:
        return None

    sample_counts: list[dict] = []
    manifest_path = os.path.join(FIVEXFAD_KINASE_DIR, "sample_manifest.csv")
    if os.path.exists(manifest_path):
        manifest = pd.read_csv(manifest_path)
        primary = manifest[
            (manifest["analysis_action"] == "primary")
            & (manifest["analysis_scope"] == "kinase_mea_v1")
        ].drop_duplicates(["tissue", "assay", "biological_sample_id"])
        if not primary.empty:
            grouped = (
                primary.groupby(["tissue", "assay", "age", "genotype"])
                .size()
                .rename("n_biological_samples")
                .reset_index()
            )
            sample_counts = grouped.to_dict(orient="records")
        source_files.append(os.path.relpath(manifest_path, UNIFIED_VIEWER_DIR))

    detail_shards = _write_fivexfad_detail_shards(track_specs, rows, manifest_path)
    rows_payload = [
        {
            k: v for k, v in row.items()
            if k not in {"leading_substrates", "leading_substrate_count"}
        }
        for row in rows
    ]
    celltype_mea_rows = _build_fivexfad_celltype_mea_rows(rows)
    attribution_rows = _build_fivexfad_attribution_rows(rows, data)
    attribution_rows = _assign_fivexfad_song_aligned_confidence(attribution_rows, rows)
    attribution_rows = _promote_fivexfad_attribution_confidence(
        attribution_rows,
        rows,
        celltype_mea_rows,
    )
    celltype_attribution_summary_index = _build_fivexfad_attribution_summary_index(attribution_rows)
    celltype_attribution_summary_shard = _write_fivexfad_index_shard(
        celltype_attribution_summary_index,
        "fivexfad_attribution_summary.json.gz",
        "supporting_5xfad_attribution_summary",
    )
    celltype_attribution_shards = _write_fivexfad_attribution_shards(attribution_rows)
    celltype_mea_plot_index = _build_fivexfad_celltype_mea_plot_index(celltype_mea_rows)
    celltype_mea_plot_index_shard = _write_fivexfad_index_shard(
        celltype_mea_plot_index,
        "fivexfad_celltype_mea_index.json.gz",
        "supporting_5xfad_celltype_mea_index",
    )
    celltype_agreement_index = _build_fivexfad_celltype_agreement_index(rows, celltype_mea_rows)
    celltype_mea_shards = _write_fivexfad_celltype_mea_shards(celltype_mea_rows)
    celltype_ols_shards = _write_fivexfad_celltype_ols_shards(celltype_mea_rows)
    mechanism_attribution = _load_fivexfad_mechanism_attribution()
    for extra in ("fivexfad_snrna_attribution.csv", "fivexfad_snrna_cell_counts.csv"):
        extra_path = os.path.join(FIVEXFAD_KINASE_DIR, extra)
        if os.path.exists(extra_path):
            source_files.append(os.path.relpath(extra_path, UNIFIED_VIEWER_DIR))
    for extra in (
        "fivexfad_snrna_pseudobulk_linear.csv.gz",
        "fivexfad_snrna_pseudobulk_counts.csv",
        "fivexfad_snrna_gene_map.csv",
        "fivexfad_celltype_mea.parquet",
        "fivexfad_celltype_site_level_ols.parquet",
        "fivexfad_celltype_mea_global_shift.csv",
        "fivexfad_celltype_winsorized_sites.csv",
        "fivexfad_celltype_substrate_sets.csv",
        "fivexfad_celltype_counts.csv",
        "fivexfad_celltype_mea_audit.json",
    ):
        extra_path = os.path.join(FIVEXFAD_CELLTYPE_DIR, extra)
        if os.path.exists(extra_path):
            source_files.append(os.path.relpath(extra_path, UNIFIED_VIEWER_DIR))

    print(f"  supporting_5xfad: {len(rows):,} MEA rows", flush=True)
    payload = {
        "schema_version": 1,
        "cohort": "5xFAD",
        "role": "supporting_ad_cohort",
        "filters": {
            "tissue": ["cortex", "hippocampus"],
            "age_months": [3, 6, 9, 12],
        },
        "rows": rows_payload,
        "celltype_attribution_summary_shard": celltype_attribution_summary_shard,
        "celltype_attribution_shards": celltype_attribution_shards,
        "celltype_agreement_index": celltype_agreement_index,
        "celltype_mea_plot_index_shard": celltype_mea_plot_index_shard,
        "celltype_mea_shards": celltype_mea_shards,
        "contrast_qc": qc_rows,
        "sample_counts": sample_counts,
        "detail_shards": detail_shards,
        "celltype_ols_shards": celltype_ols_shards,
        "source_files": sorted(set(source_files)),
    }
    if mechanism_attribution:
        payload["mechanism_attribution"] = mechanism_attribution
    return payload


# ---------------------------------------------------------------------------
# 5xFAD incytr pair-mode constants
# ---------------------------------------------------------------------------
_5XFAD_INCYTR_CONTRASTS = ("TG_3mo", "TG_6mo", "TG_9mo", "TG_12mo")
_5XFAD_TRAJ_TIMEPOINTS = ("3mo", "6mo", "9mo", "12mo")

_5XFAD_INCYTR_TISSUE = {
    "cortex": {
        "edge_dir": EDGE_SLICES_INCYTR_PATHWAYS_5XFAD_CORTEX_DIR,
        "context_id": "fivexfad_cortex",
        "url_prefix": "edge_slices/incytr_pathways_fivexfad_cortex/",
    },
    "hippocampus": {
        "edge_dir": EDGE_SLICES_INCYTR_PATHWAYS_5XFAD_HIPPO_DIR,
        "context_id": "fivexfad_hippocampus",
        "url_prefix": "edge_slices/incytr_pathways_fivexfad_hippocampus/",
    },
}


def _5xfad_incytr_sanitize(name: str) -> str:
    return name.replace("/", "-").replace(" ", "_")


def _5xfad_pair_mode_contrast_from_filename(fname: str) -> str | None:
    """`TG_3mo_WT_3mo_incytr_output.parquet` → `TG_3mo`."""
    m = re.match(r"(TG)_(\d+)mo_WT_\d+mo_incytr_output\.parquet$", fname)
    if not m:
        return None
    return f"TG_{m.group(2)}mo"


def _5xfad_annotate_trajectory_columns(
    df: "pd.DataFrame",
    source_label: str = "pair_mode",
) -> "tuple[pd.DataFrame, dict, dict]":
    """Trajectory annotation for 5xFAD: 4 timepoints (3/6/9/12mo), disease=TG."""
    df = df.copy()
    df["_path_str"] = (
        df["sender"].astype(str) + "||"
        + df["receiver"].astype(str) + "||"
        + df["Path"].astype(str)
    )
    split = df["contrast"].str.split("_", n=1, expand=True)
    df["_disease"] = split[0].fillna("")
    df["_timepoint"] = split[1].fillna("")

    if df.empty:
        df["traj_labels"] = ""
        df["sign_vec"] = ""
        return df, {}, {}

    pds_col = df["PDS"].astype(float)
    sign_ser = pd.Series("", index=df.index, dtype="str")
    sign_ser.loc[pds_col > 0] = "u"
    sign_ser.loc[pds_col < 0] = "d"
    df["_sign"] = sign_ser
    df["_pds"] = pds_col

    valid_tp = set(_5XFAD_TRAJ_TIMEPOINTS)
    valid_dis = {"TG"}
    pivot_mask = (
        df["_disease"].isin(valid_dis)
        & df["_timepoint"].isin(valid_tp)
        & df["_sign"].isin(["u", "d"])
    )
    sub = df.loc[pivot_mask, ["_path_str", "_disease", "_timepoint", "_sign", "_pds"]]

    if sub.empty:
        print(f"  trajectory ({source_label}): no canonical contrasts; skipping",
              flush=True)
        df["traj_labels"] = ""
        df["sign_vec"] = ""
        return df, {}, {}

    sign_pivot = sub.pivot_table(
        index=["_path_str", "_disease"],
        columns="_timepoint",
        values="_sign",
        aggfunc="first",
    )
    pds_pivot = sub.pivot_table(
        index=["_path_str", "_disease"],
        columns="_timepoint",
        values="_pds",
        aggfunc="first",
    )
    for tp in _5XFAD_TRAJ_TIMEPOINTS:
        if tp not in sign_pivot.columns: sign_pivot[tp] = pd.NA
        if tp not in pds_pivot.columns:  pds_pivot[tp] = pd.NA
    sign_pivot = sign_pivot[list(_5XFAD_TRAJ_TIMEPOINTS)]
    pds_pivot = pds_pivot[list(_5XFAD_TRAJ_TIMEPOINTS)]
    complete_mask = sign_pivot.notna().all(axis=1) & pds_pivot.notna().all(axis=1)
    sign_pivot = sign_pivot.loc[complete_mask]
    pds_pivot = pds_pivot.loc[complete_mask]

    if sign_pivot.empty:
        df["traj_labels"] = ""
        df["sign_vec"] = ""
        return df, {}, {}

    s3, s6, s9, s12 = (sign_pivot[tp] for tp in _5XFAD_TRAJ_TIMEPOINTS)
    v3, v6, v9, v12 = (pds_pivot[tp] for tp in _5XFAD_TRAJ_TIMEPOINTS)
    out = pd.DataFrame(index=sign_pivot.index)
    out["sign_vec"] = s3 + s6 + s9 + s12
    out["always_up"] = (s3 == "u") & (s6 == "u") & (s9 == "u") & (s12 == "u")
    out["always_down"] = (s3 == "d") & (s6 == "d") & (s9 == "d") & (s12 == "d")
    out["monotonic_up"] = (v3 < v6) & (v6 < v9) & (v9 < v12)
    out["monotonic_down"] = (v3 > v6) & (v6 > v9) & (v9 > v12)
    out["mixed"] = ~(out["always_up"] | out["always_down"])

    def _join_labels(row):
        names = []
        if row["always_up"]:      names.append("always-up")
        if row["always_down"]:    names.append("always-down")
        if row["monotonic_up"]:   names.append("monotonic-up")
        if row["monotonic_down"]: names.append("monotonic-down")
        if row["mixed"]:          names.append("mixed")
        return ";".join(names)

    out["traj_labels"] = out.apply(_join_labels, axis=1)

    traj_map = out[["sign_vec", "traj_labels"]].reset_index()
    df = df.merge(traj_map, on=["_path_str", "_disease"], how="left")
    df["traj_labels"] = df["traj_labels"].fillna("")
    df["sign_vec"] = df["sign_vec"].fillna("")

    sig_pivot = out.reset_index()[["_path_str", "_disease"]]
    recur_index: dict = {}
    if len(sig_pivot):
        recur_series = sig_pivot.groupby("_path_str", sort=False)["_disease"].agg(list)
        recur_index = {str(pid): dis for pid, dis in recur_series.items()}

    traj_summary: dict = {lbl: int(out[lbl.replace("-", "_")].sum())
                          for lbl in _SIGN_VEC_LABELS}

    n_paths = len(out.index.get_level_values("_path_str").unique())
    print(f"  trajectory ({source_label}): {n_paths:,} unique paths annotated; "
          f"{len(recur_index):,} recur in ≥1 disease; "
          f"label dist = {dict(sorted(traj_summary.items()))}", flush=True)

    df.drop(columns=["_path_str", "_disease", "_timepoint", "_sign", "_pds"],
            inplace=True, errors="ignore")
    return df, recur_index, traj_summary


def _write_5xfad_incytr_pair_pathways(tissue: str) -> dict | None:
    """Shard the 5xFAD pair-mode Incytr output for one tissue.

    Reads `outputs/reports/incytr_pair_mode_5xfad/{tissue}/wide/*.parquet`
    and emits one parquet per (sender, receiver) pair under a tissue-specific
    edge_slices subdirectory. Returns the block for `incytr_pathways.by_context`
    or None when the input dir is absent.
    """
    if tissue not in _5XFAD_INCYTR_TISSUE:
        raise ValueError(f"unknown 5xFAD tissue: {tissue!r}")

    out_dir = _5XFAD_INCYTR_TISSUE[tissue]["edge_dir"]
    url_prefix = _5XFAD_INCYTR_TISSUE[tissue]["url_prefix"]

    input_dir = os.path.join(
        config.REPO_ROOT, "outputs", "reports",
        "incytr_pair_mode_5xfad", tissue, "wide",
    )
    if not os.path.isdir(input_dir):
        print(f"  (warn) 5xFAD incytr input dir not found: {input_dir}; "
              f"skipping {tissue}", flush=True)
        return None

    parquet_files = sorted(glob.glob(os.path.join(input_dir, "*_incytr_output.parquet")))
    if not parquet_files:
        print(f"  (warn) no 5xFAD pair-mode parquets in {input_dir}; "
              f"skipping {tissue}", flush=True)
        return None

    file_to_contrast: list[tuple[str, str]] = []
    for fpath in parquet_files:
        contrast = _5xfad_pair_mode_contrast_from_filename(os.path.basename(fpath))
        if contrast is not None:
            file_to_contrast.append((fpath, contrast))
    if not file_to_contrast:
        print(f"  (warn) no parseable 5xFAD parquets in {input_dir}; "
              f"skipping {tissue}", flush=True)
        return None

    _incytr_5xfad_sig = _input_signature(
        f"fivexfad_incytr_{tissue}",
        [__file__] + [fp for fp, _ in file_to_contrast],
        {"tissue": tissue, "builder_version": 1},
    )
    _incytr_5xfad_cached = _load_build_cache(
        f"fivexfad_incytr_{tissue}", _incytr_5xfad_sig, out_dir
    )
    if _incytr_5xfad_cached is not None:
        return _incytr_5xfad_cached

    present_contrasts = [c for c in _5XFAD_INCYTR_CONTRASTS
                         if c in {c2 for _, c2 in file_to_contrast}]
    contrast_to_idx = {c: i for i, c in enumerate(present_contrasts)}
    print(f"  5xfad incytr ({tissue}): {len(file_to_contrast)} parquet(s); "
          f"contrasts = {present_contrasts}", flush=True)

    import duckdb

    con = duckdb.connect()
    con.execute("PRAGMA threads=8; PRAGMA memory_limit='12GB';")
    _configure_duckdb_tempdir(con)

    sample_schema = pq.read_schema(file_to_contrast[0][0])
    src_cols = {f.name for f in sample_schema}
    dir_flag_cols = [c for c in ("pr_up", "pr_down", "ps_up", "ps_down",
                                  "py_up", "py_down")
                     if c in src_cols]
    extra_path_cols = [c for c in ("log2FC",) if c in src_cols]
    if not dir_flag_cols:
        print(f"    (warn) no direction-flag columns in {tissue}; "
              f"downstream UI badges will be empty", flush=True)

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
        if pcol_disease is None:
            print(f"    (warn) no disease-arm p_value col in "
                  f"{os.path.basename(fpath)}; using NULL", flush=True)
            pcol_clause = "CAST(NULL AS DOUBLE)"
        else:
            pcol_clause = f'CAST("{pcol_disease}" AS DOUBLE)'

        sik_disease = next(
            (n for n in names if n.startswith("SiK_score_") and not n.endswith("_WTyp")),
            None,
        )
        if sik_disease is None:
            sik_clause = "CAST(NULL AS DOUBLE) AS SiK_score"
        else:
            sik_clause = f'CAST("{sik_disease}" AS DOUBLE) AS SiK_score'

        dir_clauses = ",\n          ".join(
            f"CAST({c} AS DOUBLE) AS {c}" for c in dir_flag_cols
        )
        path_clauses = ",\n          ".join(
            f"CAST({c} AS DOUBLE) AS {c}" for c in extra_path_cols
        )
        generic_scores = [c for c in _INCYTR_SCORE_COLS if c != "SiK_score"]
        score_clauses = ",\n          ".join(
            f"CAST({c} AS DOUBLE) AS {c}" for c in generic_scores if c in names
        )
        missing_scores = [c for c in generic_scores if c not in names]
        missing_score_clauses = ",\n          ".join(
            f"CAST(NULL AS DOUBLE) AS {c}" for c in missing_scores
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

        where_clause = (
            f"WHERE {pcol_clause} IS NULL OR {pcol_clause} <= 0.75"
            if pcol_disease is not None else ""
        )
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
        {where_clause}
        """)

    union_sql = "\nUNION ALL\n".join(selects)
    con.execute(f"CREATE VIEW src AS {union_sql}")
    n_src = con.execute("SELECT COUNT(*) FROM src").fetchone()[0]
    print(f"  5xfad incytr ({tissue}): loaded {n_src:,} rows across "
          f"{len(file_to_contrast)} contrast(s)", flush=True)

    senders_canonical = sorted({r[0] for r in con.execute(
        "SELECT DISTINCT sender FROM src").fetchall()})
    receivers_canonical = sorted({r[0] for r in con.execute(
        "SELECT DISTINCT receiver FROM src").fetchall()})
    sender_to_idx = {s: i for i, s in enumerate(senders_canonical)}
    receiver_to_idx = {r: i for i, r in enumerate(receivers_canonical)}
    n_s, n_r, n_c = len(senders_canonical), len(receivers_canonical), len(present_contrasts)
    print(f"    senders={n_s}, receivers={n_r}, contrasts={n_c} "
          f"(pair count={n_s * n_r})", flush=True)

    n_thr = len(_INCYTR_PATHWAY_PVALUES)
    n_ap = len(_INCYTR_PATHWAY_ABS_PDS)
    pval_filter = (lambda tp: f"pvalue < {tp}") if has_pvalue else (lambda tp: "TRUE")
    pval_where = "WHERE pvalue IS NOT NULL" if has_pvalue else ""
    hm_thr_clauses_list = []
    for ip, tp in enumerate(_INCYTR_PATHWAY_PVALUES):
        for iap, tap in enumerate(_INCYTR_PATHWAY_ABS_PDS):
            hm_thr_clauses_list.append(
                f"COUNT(*) FILTER (WHERE {pval_filter(tp)} "
                f"AND COALESCE(ABS(PDS), 0) >= {tap}) AS c_{ip}_{iap}"
            )
    hm_thr_clauses = ", ".join(hm_thr_clauses_list)
    hm_rows = con.execute(f"""
        SELECT sender, receiver, contrast, {hm_thr_clauses}
        FROM src {pval_where}
        GROUP BY sender, receiver, contrast
    """).fetchall()
    grid = np.zeros((n_s, n_r, n_c, n_thr, n_ap), dtype=np.uint32)
    for row in hm_rows:
        s_raw, r_raw, c = row[0], row[1], row[2]
        if s_raw not in sender_to_idx or r_raw not in receiver_to_idx: continue
        if c not in contrast_to_idx: continue
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
               CASE WHEN PDS > 0 THEN 2
                    WHEN PDS < 0 THEN 0
                    ELSE 1 END AS s,
               {hm_thr_clauses}
        FROM src {pval_where}
        GROUP BY sender, receiver, contrast, s
    """).fetchall()
    signed_grid = np.zeros((n_s, n_r, n_c, 3, n_thr, n_ap), dtype=np.uint32)
    for row in hm_signed_rows:
        s_raw, r_raw, c, sign_i = row[0], row[1], row[2], int(row[3])
        if s_raw not in sender_to_idx or r_raw not in receiver_to_idx: continue
        if c not in contrast_to_idx: continue
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
                signed_totals[sign_i, ip, iap] = int(
                    signed_grid[:, :, :, sign_i, ip, iap].sum()
                )
    heatmap_counts_signed = {
        "thresholds": list(_INCYTR_PATHWAY_PVALUES),
        "abs_pds_thresholds": list(_INCYTR_PATHWAY_ABS_PDS),
        "shape": [n_s, n_r, n_c, 3, n_thr, n_ap],
        "counts": signed_grid.flatten().tolist(),
        "total_by_sign_threshold": signed_totals.tolist(),
        "sign_source": "PDS",
    }

    def _build_pathway_counts(where_extra: str = "") -> dict:
        thr_clauses = []
        for ip, tp in enumerate(_INCYTR_PATHWAY_PVALUES):
            for iap, tap in enumerate(_INCYTR_PATHWAY_ABS_PDS):
                thr_clauses.append(
                    f"COUNT(*) FILTER (WHERE {pval_filter(tp)} "
                    f"AND COALESCE(ABS(PDS), 0) >= {tap}) AS c_{ip}_{iap}"
                )
        where_parts = []
        if has_pvalue: where_parts.append("pvalue IS NOT NULL")
        if where_extra: where_parts.append(where_extra)
        where_clause = "WHERE " + " AND ".join(where_parts) if where_parts else ""
        pathway_rows = con.execute(f"""
            SELECT contrast,
                   CASE WHEN PDS > 0 THEN 2
                        WHEN PDS < 0 THEN 0
                        ELSE 1 END AS s,
                   {", ".join(thr_clauses)}
            FROM src {where_clause}
            GROUP BY contrast, s
        """).fetchall()
        pathway_arr = np.zeros((n_c, 3, n_thr, n_ap), dtype=np.uint32)
        for row in pathway_rows:
            contrast, s_idx = row[0], int(row[1])
            if contrast not in contrast_to_idx: continue
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

    src_cols_view = set(
        con.execute("DESCRIBE SELECT * FROM src LIMIT 0").fetchdf()["column_name"]
    )

    INCYTR_INDEX_COLUMNS = (
        [("PDS", "f4"), ("pvalue", "f4")]
        + [(sc, "u2") for sc in _INCYTR_SCORE_COLS]
        + [("ligandId", "u2"), ("receptorId", "u2"),
           ("emId", "u2"), ("targetId", "u2")]
        + [("senderId", "u1"), ("receiverId", "u1"), ("contrastId", "u1"),
           ("labelBits", "u1"), ("trajBits", "u1")]
    )
    idx_gene_to_id: dict[str, int] = {}
    idx_gene_vocab: list[str] = []
    idx_chunks: list[dict] = []

    def _idx_gene_ids(series) -> np.ndarray:
        cat = series.astype(str)
        for g in cat.unique():
            if g not in idx_gene_to_id:
                idx_gene_to_id[g] = len(idx_gene_vocab)
                idx_gene_vocab.append(g)
        return cat.map(idx_gene_to_id).to_numpy(dtype="<u2")

    def _accumulate_index(s_name: str, r_name: str, frame) -> None:
        n = len(frame)
        if n == 0: return
        chunk = {
            "senderId":   np.full(n, sender_to_idx[s_name], dtype="<u1"),
            "receiverId": np.full(n, receiver_to_idx[r_name], dtype="<u1"),
            "contrastId": frame["contrast"].map(contrast_to_idx).to_numpy(dtype="<u1"),
            "ligandId":   _idx_gene_ids(frame["Ligand"]),
            "receptorId": _idx_gene_ids(frame["Receptor"]),
            "emId":       _idx_gene_ids(frame["EM"]),
            "targetId":   _idx_gene_ids(frame["Target"]),
            "labelBits":  _idx_label_bits(frame),
            "trajBits":   _idx_traj_bits(frame["traj_labels"]),
            "PDS":        frame["PDS"].to_numpy(dtype="<f4"),
            "pvalue":     frame["pvalue"].to_numpy(dtype="<f4"),
        }
        for sc in _INCYTR_SCORE_COLS:
            chunk[sc] = (frame[sc].to_numpy(dtype="float16").view("<u2")
                         if sc in frame.columns else np.zeros(n, dtype="<u2"))
        idx_chunks.append(chunk)

    gene_node_index = _build_incytr_gene_node_index(con)
    print(
        f"    gene_node_index ({tissue}): {len(gene_node_index['gene_id']):,} "
        f"gene-role-pair entries; {len(gene_node_index['genes']):,} genes",
        flush=True,
    )

    shutil.rmtree(out_dir, ignore_errors=True)
    os.makedirs(out_dir, exist_ok=True)

    fc_select = [
        f'"{c}"' if c in src_cols_view else f'CAST(NULL AS DOUBLE) AS "{c}"'
        for c in _INCYTR_FC_COLS
    ]
    label_select = [
        f'"{dst}"' if dst in src_cols_view else f'CAST(NULL AS VARCHAR) AS "{dst}"'
        for dst in _INCYTR_LABEL_COLS
    ]
    shard_select_cols = (
        ["Ligand", "Receptor", "EM", "Target", "contrast", "pvalue", "PDS"]
        + list(_INCYTR_SCORE_COLS)
        + dir_flag_cols
        + extra_path_cols
        + list(_INCYTR_FC_COLS)
        + list(_INCYTR_LABEL_COLS)
    )
    float_cols = (
        ["pvalue", "PDS"]
        + list(_INCYTR_SCORE_COLS)
        + dir_flag_cols
        + extra_path_cols
        + list(_INCYTR_FC_COLS)
    )
    float32_cols = [c for c in float_cols if c in ("pvalue", "PDS", "log2FC")]
    float16_cols = [c for c in float_cols if c not in float32_cols]

    present_pairs: list[list[str]] = []
    pair_row_counts: dict[str, int] = {}
    total_rows = 0
    max_shard_bytes = 0
    max_shard_name = ""
    recur_index: dict = {}
    traj_summary: dict = {}

    def _flush(key: tuple[str, str], frames: list[pd.DataFrame]) -> None:
        nonlocal total_rows, max_shard_bytes, max_shard_name
        if not frames: return
        sub = pd.concat(frames, ignore_index=True, copy=False)
        for col in _INCYTR_LABEL_COLS:
            if col in sub.columns:
                sub[col] = pd.Categorical(sub[col], categories=_INCYTR_LABEL_VOCAB)
        s_key, r_key = key
        sub["sender"] = s_key
        sub["receiver"] = r_key
        sub["Path"] = (sub["Ligand"].astype(str) + "|"
                       + sub["Receptor"].astype(str) + "|"
                       + sub["EM"].astype(str) + "|"
                       + sub["Target"].astype(str))
        sub, pair_recur, pair_traj = _5xfad_annotate_trajectory_columns(
            sub, source_label=f"pair_mode/{tissue}",
        )
        recur_index.update(pair_recur)
        for label, count in pair_traj.items():
            traj_summary[label] = traj_summary.get(label, 0) + int(count)
        _accumulate_index(s_key, r_key, sub)
        sub = sub.drop(columns=["sender", "receiver", "Path"])
        for col in float32_cols:
            if col in sub.columns: sub[col] = sub[col].astype("float32")
        for col in float16_cols:
            if col in sub.columns: sub[col] = sub[col].astype("float16")
        path_sort_cols = [c for c in ("Ligand", "Receptor", "EM", "Target", "contrast")
                          if c in sub.columns]
        if path_sort_cols:
            sub = sub.sort_values(path_sort_cols, kind="stable",
                                  na_position="last").reset_index(drop=True)
        s, r = key
        fname = f"{_5xfad_incytr_sanitize(s)}__{_5xfad_incytr_sanitize(r)}.parquet"
        path = os.path.join(out_dir, fname)
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
        sz = os.path.getsize(path)
        if sz > max_shard_bytes:
            max_shard_bytes = sz
            max_shard_name = fname

    stream_cols = ["receiver"] + shard_select_cols
    for s in senders_canonical:
        reader = con.execute(
            f"""SELECT {', '.join(stream_cols)}
                FROM src
                WHERE sender = ?
                ORDER BY receiver""",
            [s],
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

    index = {
        "schema_version": SCHEMA_VERSION,
        "filename_template": "{sender}__{receiver}.parquet",
        "sanitize_rule": "replace('/', '-'); replace(' ', '_')",
        "present": sorted(present_pairs),
        "n_total_rows": total_rows,
        "pair_row_counts": pair_row_counts,
        "base_url": url_prefix,
    }
    with open(os.path.join(out_dir, "index.json"), "w") as f:
        json.dump(index, f)

    total_bytes = sum(
        os.path.getsize(os.path.join(out_dir, fn))
        for fn in os.listdir(out_dir) if fn.endswith(".parquet")
    )
    print(f"  5xfad incytr ({tissue}): wrote {len(present_pairs)} shards "
          f"({total_rows:,} rows; {total_bytes/1e6:.1f} MB total; "
          f"max {max_shard_bytes/1e6:.2f} MB → {max_shard_name})", flush=True)

    assert sys.byteorder == "little", "global index assumes little-endian"
    global_index = None
    if idx_chunks:
        cols = {name: np.concatenate([c[name] for c in idx_chunks])
                for name, _dt in INCYTR_INDEX_COLUMNS}
        idx_chunks.clear()
        n_idx = int(len(cols["PDS"]))
        perm = np.argsort(-np.abs(cols["PDS"]), kind="stable")
        buf_bytes = bytearray()
        columns_manifest = []
        for name, dt in INCYTR_INDEX_COLUMNS:
            arr = np.ascontiguousarray(cols[name][perm], dtype=np.dtype("<" + dt[0] + dt[1]))
            columns_manifest.append({"name": name, "type": dt, "bytes": int(arr.nbytes)})
            buf_bytes += arr.tobytes()
        del cols
        raw_bin = bytes(buf_bytes)
        gz_bin = gzip.compress(raw_bin, compresslevel=6)
        with open(os.path.join(out_dir, _INCYTR_INDEX_FILENAME), "wb") as f:
            f.write(gz_bin)
        global_index = {
            "url": f"{url_prefix}{_INCYTR_INDEX_FILENAME}",
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
            "score_columns": list(_INCYTR_SCORE_COLS),
            "columns": columns_manifest,
            "raw_bytes": len(raw_bin),
            "gzip_bytes": len(gz_bin),
        }
        print(f"  5xfad incytr global_index ({tissue}): {n_idx:,} rows × "
              f"{len(columns_manifest)} cols, {len(idx_gene_vocab):,} genes; "
              f"{len(raw_bin)/1e6:.1f} MB raw → {len(gz_bin)/1e6:.1f} MB gz",
              flush=True)

    celltypes = sorted(set(senders_canonical) | set(receivers_canonical))
    block = {
        "schema_version": SCHEMA_VERSION,
        "version": 3,
        "source": f"pair_mode (5xfad/{tissue}/wide)",
        "source_mode": "pair_mode",
        "contrasts": list(present_contrasts),
        "senders": senders_canonical,
        "receivers": receivers_canonical,
        "celltypes": celltypes,
        "empty_deg_celltypes": [],
        "celltype_qc": None,
        "low_signal_celltypes": [],
        "heatmap_counts": heatmap_counts,
        "heatmap_counts_signed": heatmap_counts_signed,
        "pathway_counts": pathway_counts,
        "pathway_counts_low_signal_excluded": None,
        "slice_index": index,
        "score_columns": list(_INCYTR_SCORE_COLS),
        "label_columns": list(_INCYTR_LABEL_COLS),
        "label_nodes": list(_INCYTR_LABEL_NODES),
        "label_vocab": list(_INCYTR_LABEL_VOCAB),
        "direction_flag_columns": list(dir_flag_cols),
        "path_metric_columns": list(extra_path_cols),
        "global_index": global_index,
        "gene_node_index_shard": _write_gene_node_index_shard(
            gene_node_index, out_dir,
            _INCYTR_GENE_NODE_INDEX_FILENAME,
            url_prefix=url_prefix,
        ),
        "trajectory_summary": traj_summary,
    }
    _incytr_5xfad_output_files = (
        [os.path.basename(p) for p in os.listdir(out_dir) if p.endswith(".parquet")]
        + [_INCYTR_INDEX_FILENAME, _INCYTR_GENE_NODE_INDEX_FILENAME, "index.json"]
    )
    _write_build_cache(
        f"fivexfad_incytr_{tissue}", _incytr_5xfad_sig, out_dir,
        _incytr_5xfad_output_files, block,
    )
    return block


def build_5xfad_incytr_blocks() -> dict[str, dict]:
    """Build pair-mode incytr blocks for both 5xFAD tissues.

    Returns a dict mapping context_id → payload block, containing only the
    tissues where data is present.  Callers merge results into
    ``PAYLOAD.incytr_pathways.by_context``.
    """
    blocks = {}
    for tissue in ("cortex", "hippocampus"):
        context_id = _5XFAD_INCYTR_TISSUE[tissue]["context_id"]
        block = _write_5xfad_incytr_pair_pathways(tissue)
        if block is not None:
            blocks[context_id] = block
    return blocks


def _age_from_contrast_label(contrast: str) -> int | None:
    m = re.search(r"_(3|6|9|12)mo$", str(contrast))
    return int(m.group(1)) if m else None


def build_fivexfad_viewer_slice(data: UnifiedData | None = None) -> CohortViewerSlice | None:
    """5xFAD supporting-cohort contribution to the unified viewer payload.
    Returns None when the 5xFAD attribution inputs are absent (mouse-only / non-5xFAD
    build stays byte-equivalent — the caller omits PAYLOAD.supporting_5xfad)."""
    block = build_supporting_5xfad_slice(data)
    if block is None:
        return None
    return CohortViewerSlice(
        cohort_id="fivexfad",
        context_ids=("song_ad",),
        owned_sections={"supporting_5xfad": block},
        capabilities={"supporting_5xfad": True},
        kinase_names=tuple(r.get("kinase", "") for r in block.get("rows", [])),
        provenance={"source_dir": FIVEXFAD_KINASE_DIR},
    )

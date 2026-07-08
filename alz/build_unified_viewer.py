#!/usr/bin/env python3
"""Unified viewer builder for the kinase pipeline.

Reads the unified attribution + recovery tables (mouse), the Levy-t5 decomposition
shards (per-cluster MEA + per-animal site OLS), the pair-mode Incytr output
(`outputs/reports/incytr_pair_mode/`), and the human per-donor MEA + SEA-AD
agreement chain. Emits `outputs/reports/unified_viewer/index.html` plus
`unified_viewer.payload.json(.gz)` sidecars and per-entity shards under
`edge_slices/{human_perdonor,decomp_ols,song_concordance,incytr_pathways}/`
fetched on demand by the JS tabs.

Usage:
    python alz/build_unified_viewer.py              # payload + html (default)
    python alz/build_unified_viewer.py --summary    # input row counts
    python alz/build_unified_viewer.py --payload    # JSON only
    python alz/build_unified_viewer.py --html       # write HTML (needs payload)
    python alz/build_unified_viewer.py --validate   # write report md
"""

from __future__ import annotations

import argparse
import filecmp
import glob
import gzip
import hashlib
import json
import os
import re
import resource
import shutil
import sys
import time
import uuid
import warnings
from dataclasses import dataclass, field
from functools import cmp_to_key
from typing import Any

# Some dependency paths import Matplotlib for motif/logo helpers. In managed
# environments HOME may not be writable, which makes every build pay a temp
# cache setup penalty and emit a warning.
os.environ.setdefault("MPLCONFIGDIR", os.path.join("/tmp", "alz-matplotlib"))

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(HERE)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "integration"))
sys.path.insert(0, os.path.join(HERE, "bulk_mea"))
sys.path.insert(0, os.path.join(HERE, "cross_reference"))

from alz.shared import config  # noqa: E402
from alz.bulk_mea.confidence import DECOMP_FDR_AGREEMENT  # noqa: E402
import config_integration as icfg  # noqa: E402
import normalize as kattr  # noqa: E402  (alz.bulk_mea.normalize — Stage 1 phospho-track + IRS helpers)
from alz.viewer.shared.payload_helpers import (_sanitize, _configure_duckdb_tempdir, _json_clean_value, _INCYTR_FC_NODES, _build_incytr_gene_node_index, _build_kinase_motifs)  # noqa: E402
from alz.viewer.shared.incytr_index import (_INCYTR_LABEL_NODES, _INCYTR_LABEL_COLS, _INCYTR_LABEL_VOCAB, _SIGN_VEC_LABELS, _idx_label_bits, _idx_traj_bits)  # noqa: E402
from alz.viewer.shared.build_cache import _input_signature, _load_build_cache, _write_build_cache  # noqa: E402
try:
    from human_celltype_attribution import build_celltype_specificity_payload  # noqa: E402
    _HAS_HUMAN_CELLTYPE = True
except ImportError:
    _HAS_HUMAN_CELLTYPE = False

# ---------------------------------------------------------------------------
# Paths — re-exported from viewer.paths so existing references in this module
# continue to resolve. The pathway-side payload builders in
# `viewer.pathway_payload` import directly from `viewer.paths`.
# ---------------------------------------------------------------------------

from alz.viewer.cohorts.mukesh import build_mukesh_viewer_slice  # noqa: E402
from alz.viewer.cohorts.fivexfad import (  # noqa: E402
    build_5xfad_incytr_blocks,
    build_fivexfad_viewer_slice,
)
from alz.viewer.cohorts.substrate_compare import (  # noqa: E402
    build_substrate_compare_slice,
)
from alz.viewer.shared.compose import compose_viewer_slices  # noqa: E402
from alz.viewer.cohorts.song import (  # noqa: E402
    SongBuild,
    _as_single_context_block,
    _annotate_trajectory_columns,
    _build_agreement_index,
    _build_celltypes_slice,
    _build_kinases_slice,
    _build_subclass_breakdown,
    _incytr_celltype_qc_counts_path,
    _incytr_sanitize,
    _norm_motif,
    _pair_mode_contrast_from_filename,
    _read_empty_deg_celltypes,
    _read_incytr_celltype_qc,
    _to_float32_estimable,
    _write_decomp_ols_slices,
    _write_incytr_pair_pathways,
    _write_song_concordance_slices,
    build_song_viewer_slice,
)
from viewer.paths import (  # noqa: E402
    AUDIT_PREVIEW_ROWS,
    AUDIT_SOURCES_DIR,
    DECOMP_OLS_PARQUET,
    EDGE_SLICES_DECOMP_OLS_DIR,
    EDGE_SLICES_INCYTR_PATHWAYS_DIR,
    EDGE_SLICES_SONG_CONCORDANCE_DIR,
    FIVEXFAD_KINASE_DIR,
    INCYTR_PAIR_MODE_OUTPUTS_DIR,
    MEASUREMENT_TRACE_DIR,
    MEASUREMENT_TRACE_INDEX,
    MEASUREMENT_TRACE_SCHEMA_VERSION,
    OMICS_TRACE_INDEX,
    OMICS_TRACE_SCHEMA_VERSION,
    OMICS_TRACE_FIVEXFAD_SCHEMA_VERSION,
    OMICS_TRACE_NORMALIZED_INDEX,
    OMICS_TRACE_NORMALIZED_SCHEMA_VERSION,
    TRANSCRIPT_TRACE_INDEX,
    TRANSCRIPT_TRACE_SCHEMA_VERSION,
    TRANSCRIPT_TRACE_FIVEXFAD_SCHEMA_VERSION,
    PAYLOAD_JSON,
    PAYLOAD_JSON_GZ,
    PIPELINE_OVERVIEW_DEST,
    PIPELINE_OVERVIEW_SRC,
    REPORT_MD,
    SCHEMA_VERSION,
    TOP_N_KINASES,
    UNIFIED_VIEWER_DIR,
    UNIFIED_VIEWER_HTML,
    UNIFIED_VIEWER_OUTPUT_DIR,
)

# ---------------------------------------------------------------------------
# Cohort display names — single source of truth for context/audit labels.
# JS counterpart: COHORT_LABELS in 01_state.js.
# ---------------------------------------------------------------------------
COHORT_DISPLAY = {
    "song": "MouseC1",
    "fivexfad": "MouseC2",
    "mukesh": "HumanC1",
}


# Per-track audit tables: identical schema, separate files per analysis track.
# ST (serine/threonine) is the default suffix (none); pY tables carry the
# `_pY` suffix produced by `config.track_output`.
_PER_TRACK_AUDIT = [
    ("raw_phospho_normalized", "Raw phospho normalized"),
    ("stoichiometry_matrix", "Stoichiometry matrix"),
    ("site_level_ols", "Site-level OLS"),
    ("mea_global_shift", "MEA global shift"),
    ("mea_substrate_sets", "MEA substrate sets"),
    ("winsorized_sites", "Winsorized sites"),
    ("mea_stoichiometry", "MEA stoichiometry"),
    ("mea_raw_phospho", "MEA raw phospho"),
]

# (key, label, suffix) — tracks defined in config.PHOSPHO_TRACKS.
# ST = "" suffix, pY = "_pY".
_AUDIT_TRACKS = [("", "ST"), ("_pY", "Y")]


def _audit_specs() -> list[tuple[str, str, str]]:
    specs: list[tuple[str, str, str]] = [
        ("sample_mapping", "Sample mapping",
         os.path.join(config.REPO_ROOT, "outputs", "reports", "data_ingest", "sample_mapping.csv")),
        ("normalization_summary", "Normalization summary",
         os.path.join(config.KINASE_ATTRIBUTION_OUTPUT_DIR, "normalization_summary.json")),
    ]
    for suffix, residue in _AUDIT_TRACKS:
        for base, label in _PER_TRACK_AUDIT:
            key = base + suffix
            full_label = label if suffix == "" else f"{label} ({residue})"
            specs.append((
                key, full_label,
                os.path.join(config.KINASE_ATTRIBUTION_OUTPUT_DIR, f"{base}{suffix}.csv"),
            ))
    specs.extend([
        ("5xfad_sample_manifest", f'{COHORT_DISPLAY["fivexfad"]} sample manifest',
         os.path.join(FIVEXFAD_KINASE_DIR, "sample_manifest.csv")),
        ("5xfad_dataset_index", f'{COHORT_DISPLAY["fivexfad"]} dataset index',
         os.path.join(FIVEXFAD_KINASE_DIR, "dataset_index.csv")),
        ("unified_attribution", "Unified attribution",
         os.path.join(config.KINASE_ATTRIBUTION_OUTPUT_DIR, "unified_attribution.csv")),
        ("unified_attribution_full", "Unified attribution (31 Levy-t5 clusters)",
         os.path.join(config.KINASE_ATTRIBUTION_OUTPUT_DIR, "unified_attribution_full.csv")),
        ("wmb_kinase_expression", "WMB kinase expression",
         config.WMB_EXPRESSION_FILE),
        # song_concordance lives in edge_slices/song_concordance/ (per-gene shards).
        ("sea_ad_supertype_lfc", "SEA-AD supertype LFCs",
         os.path.join(config.KINASE_ATTRIBUTION_OUTPUT_DIR, "sea_ad_supertype_lfc.csv")),
        ("kinase_activity_matrix", "Kinase activity matrix",
         os.path.join(config.ATTRIBUTION_RECOVERY_OUTPUT_DIR, "kinase_activity_matrix.csv")),
        ("celltype_evidence_table", "Cell-type evidence table",
         os.path.join(config.ATTRIBUTION_RECOVERY_OUTPUT_DIR, "celltype_evidence_table.csv")),
        ("kinase_hypothesis_table", "Kinase hypothesis table",
         os.path.join(config.ATTRIBUTION_RECOVERY_OUTPUT_DIR, "kinase_hypothesis_table.csv")),
        ("kinase_decomposition", "Kinase decomposition (Levy-T5)",
         os.path.join(config.REPO_ROOT, "outputs", "reports", "decomposition",
                      "levy_t5", "mea_per_cluster.parquet")),
    ])
    return specs


AUDIT_TABLE_SPECS = _audit_specs()

COLUMN_DEFINITIONS = {
    "kinase": "Kinase identifier used by the enrichment and attribution tables.",
    "gene_symbol": "Gene symbol associated with the site, kinase, or evidence row.",
    "contrast": "Disease genotype and timepoint comparison versus matched WT control.",
    "ES": "Raw enrichment score from motif enrichment analysis.",
    "NES": "Normalized enrichment score from motif enrichment analysis. Sign: + NES = kinase substrates concentrated at the high-stoichiometry end of the ranked β list = kinase more active in disease vs WT. Matches the Incytr convention (+ sclog2FC / + PDS = up in disease).",
    "p-value": "Nominal enrichment p-value.",
    "FDR": "False discovery rate for the enrichment result.",
    "Subs fraction": "Number of kinase substrates contributing to the enrichment divided by the substrate universe.",
    "Leading substrates": "Substrate motifs forming the leading-edge enrichment set.",
    "site_id": "Stable phosphosite identifier used to join site matrices and model outputs.",
    "motif": "Peptide motif centered on the phosphorylated residue.",
    "matched_protein": "Whether the phosphosite had a matched parent-protein measurement for stoichiometry normalization.",
    "n_obs_stoich": "Total number of biological sample columns with a usable stoichiometry value for this phosphosite; this is a site-level availability count, not a contrast-specific statistic.",
    "residue_type": "Phosphorylated residue class included in the analysis track.",
    "track": "Analysis track label, usually ST for serine/threonine.",
    "cell_type": "Cell type assigned to a kinase attribution evidence row.",
    "confidence_tier": "Canonical song-first confidence tier assigned in Python.",
    "confidence_basis": "Short summary of the evidence route supporting the confidence tier.",
    "sample_column": "Normalized sample column name used in phospho and stoichiometry matrices.",
    "raw_phospho": "Raw phosphosite intensity from the phospho source workbook before IRS normalization.",
    "raw_protein": "Raw parent-protein intensity from the total proteome source workbook before IRS normalization.",
    "irs_phospho": "IRS-normalized phosphosite intensity for the selected sample.",
    "irs_protein": "IRS-normalized parent-protein intensity for the selected sample.",
    "log2_irs_phospho": "Log2 of the IRS-normalized phosphosite intensity.",
    "log2_irs_protein": "Log2 of the IRS-normalized parent-protein intensity.",
    "stoichiometry": "Protein-corrected site value: log2(IRS phosphosite) minus log2(IRS parent protein).",
    "protein_gene": "Parent-protein gene symbol matched to the phosphosite for stoichiometry correction.",
    "normalization_method": "Normalization method recorded by the phosphoproteomics pipeline.",
    "raw_source": "Original source file or workbook reference for provenance.",
    "raw_lfc": "Site-level stoichiometry log fold change for the selected contrast, taken directly from site_level_ols.csv.",
    "median_shift": "Median stoichiometry LFC across all ranked sites for this contrast; subtracted before GSEA so the prerank is centered at zero.",
    "centered_lfc": "raw_lfc minus median_shift. The value handed to winsorization. Derived at view time.",
    "winsor_lower": "Lower clipping bound for this contrast (1st percentile of centered LFCs across all ranked sites).",
    "winsor_upper": "Upper clipping bound for this contrast (99th percentile of centered LFCs across all ranked sites).",
    "clipped_lfc": "centered_lfc clipped to [winsor_lower, winsor_upper]; this is the value passed to GSEA prerank. Derived at view time.",
    "was_winsorized": "True when the centered LFC fell outside the winsor bounds and was clipped.",
    "rank_in_contrast": "1-based descending rank of clipped_lfc within all ranked sites for this contrast (recomputed at view time; the prerank list itself is not persisted).",
    "in_leading_edge": "True when this site's motif is reported in the kinase's Leading substrates field for this contrast.",
    "original_lfc": "Pre-clip centered LFC value (equals centered_lfc; logged in winsorized_sites.csv only for sites that were clipped).",
    "lower_bound": "Lower clipping bound applied to centered LFCs in this contrast.",
    "upper_bound": "Upper clipping bound applied to centered LFCs in this contrast.",
    "mean_before": "Mean stoichiometry LFC across ranked sites before median-centering.",
    "pct_pos_before": "Percent of ranked sites with positive LFC before centering.",
    "pct_pos_after": "Percent of ranked sites with positive LFC after centering.",
}


def _clean_label(col: str) -> str:
    tokens = col.replace("_", " ").replace("-", " ").split()
    out = []
    specials = {
        "lfc": "LFC", "fdr": "FDR", "pval": "p-value",
        "pvalue": "p-value", "nes": "NES", "es": "ES",
        "ols": "OLS", "irs": "IRS", "id": "ID",
    }
    for tok in tokens:
        out.append(specials.get(tok.lower(), tok if tok.isupper() else tok.capitalize()))
    return " ".join(out)


def _column_definition(col: str) -> str:
    if col in COLUMN_DEFINITIONS:
        return COLUMN_DEFINITIONS[col]
    if col.startswith("stoich_lfc_"):
        return "Stoichiometry-normalized log fold change for this contrast."
    if col.startswith("stoich_fdr_"):
        return "FDR for the stoichiometry-normalized site-level contrast."
    if col.startswith("stoich_pval_"):
        return "Nominal p-value for the stoichiometry-normalized site-level contrast."
    if col.startswith("raw_lfc_"):
        return "Raw phosphosite log fold change for this contrast."
    if col.startswith("raw_fdr_"):
        return "FDR for the raw phosphosite site-level contrast."
    if col.endswith("_sn_mean") or col.startswith("plex"):
        return "Sample-level normalized intensity column used by the phosphoproteomics matrices."
    return f"Source column `{col}` from the audit table."


def _numeric_hint(col: str) -> str:
    cl = col.lower()
    if cl in {"site_id", "n_obs_stoich"} or cl.startswith("n_"):
        return "integer"
    if any(tok in cl for tok in ["nes", "es", "fdr", "p-value", "pval", "lfc", "score", "value", "fold"]):
        return "float"
    if cl.endswith("_sn_mean") or cl.startswith("plex"):
        return "float"
    return "text"


def _audit_columns(columns: list[str]) -> list[dict]:
    return [
        {
            "raw": c,
            "label": _clean_label(c),
            "definition": _column_definition(c),
            "format": _numeric_hint(c),
        }
        for c in columns
    ]


def _copy_audit_source(src: str, key: str) -> str | None:
    if not os.path.exists(src):
        return None
    os.makedirs(AUDIT_SOURCES_DIR, exist_ok=True)
    ext = os.path.splitext(src)[1]
    dest_name = f"{key}{ext}"
    dest = os.path.join(AUDIT_SOURCES_DIR, dest_name)
    _copy_if_different(src, dest)
    return os.path.relpath(dest, UNIFIED_VIEWER_DIR)


def _copy_if_different(src: str, dest: str) -> bool:
    """Copy src to dest only when dest is missing or content differs.

    Returns True when file bytes were copied. Identical files still get source
    metadata via copystat so future checks can use the cheap shallow path.
    """
    if os.path.exists(dest):
        src_stat = os.stat(src)
        dest_stat = os.stat(dest)
        if (
            src_stat.st_size == dest_stat.st_size
            and src_stat.st_mtime_ns == dest_stat.st_mtime_ns
        ):
            return False
        if src_stat.st_size == dest_stat.st_size and filecmp.cmp(
            src, dest, shallow=False,
        ):
            shutil.copystat(src, dest)
            return False
    shutil.copy2(src, dest)
    return True
















def _count_csv_rows(path: str) -> int:
    with open(path, "rb") as f:
        n = sum(1 for _ in f)
    return max(0, n - 1)


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


def _log2_series(values: pd.Series) -> pd.Series:
    vals = pd.to_numeric(values, errors="coerce")
    out = pd.Series(np.nan, index=values.index, dtype="float64")
    positive = vals > 0
    out.loc[positive] = np.log2(vals.loc[positive])
    return out


def _measurement_trace_columns() -> list[str]:
    return [
        "site_id", "gene_symbol", "motif", "protein_gene", "matched_protein",
        "raw_phospho", "raw_protein", "irs_phospho", "irs_protein",
        "log2_irs_phospho", "log2_irs_protein", "stoichiometry",
    ]


def ensure_measurement_trace_sources() -> dict:
    """Create per-sample sidecar receipts for raw→IRS→stoichiometry auditing."""
    _trace_source_files = [kattr.TOTAL_PROTEOME_FILE] + [
        config.resolve_track(tk)["input_file"]
        for tk in config.PHOSPHO_TRACKS
    ]
    _trace_sig = _input_signature(
        "measurement_trace",
        _trace_source_files,
        {"builder_version": 1},
    )
    _trace_cached = _load_build_cache("measurement_trace", _trace_sig, MEASUREMENT_TRACE_DIR)
    if _trace_cached is not None:
        return _trace_cached

    if os.path.exists(MEASUREMENT_TRACE_DIR):
        shutil.rmtree(MEASUREMENT_TRACE_DIR)
    os.makedirs(MEASUREMENT_TRACE_DIR, exist_ok=True)
    mapping = config.load_sample_mapping()
    sample_to_plex = dict(zip(mapping["column_name"], mapping["plex"]))
    bio_cols = mapping["column_name"].tolist()

    # Total proteome raw + IRS-normalized matrices.
    tp = pd.read_excel(kattr.TOTAL_PROTEOME_FILE, header=1)
    tp_gene = tp["Gene Symbol"].fillna("").astype(str).str.upper()
    ref_cols_tp = {
        plex: kattr._proteome_ref_col(plex)
        for plex in sorted(mapping["plex"].unique())
        if kattr._proteome_ref_col(plex) in tp.columns
    }
    tp_cols = [c for c in bio_cols + list(ref_cols_tp.values()) if c in tp.columns]
    tp_quant_raw = tp[tp_cols].apply(pd.to_numeric, errors="coerce")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="divide by zero encountered in log2", category=RuntimeWarning)
        if len(ref_cols_tp) >= 4:
            tp_quant_norm = kattr._irs_normalize(tp_quant_raw, ref_cols_tp, sample_to_plex)
            norm_method = "IRS"
        else:
            tp_quant_norm = kattr._median_center_normalize(tp_quant_raw, sample_to_plex)
            norm_method = "median_centering"
    tp_norm = tp_quant_norm[[c for c in bio_cols if c in tp_quant_norm.columns]]

    # Shared phospho→protein matching infrastructure.
    gene_to_tp_idx: dict[str, int] = {}
    for idx, gene in enumerate(tp_gene):
        if gene and gene != "0" and gene not in gene_to_tp_idx:
            gene_to_tp_idx[gene] = idx

    columns = _measurement_trace_columns()
    tracks_index: dict[str, dict] = {}
    first_preview: list[dict] = []
    _trace_written_files: list[str] = []  # paths relative to MEASUREMENT_TRACE_DIR

    # Per-track: load that track's phospho workbook, IRS-normalize, write per-sample
    # measurement-trace CSVs into a track subdir. ST keeps the legacy unsuffixed
    # filenames for backward compatibility; pY lands under measurement_trace/py/.
    for track_key, residue_label in (("st", "ST"), ("py", "Y")):
        track_cfg = config.resolve_track(track_key)
        try:
            sq = kattr._load_phospho_track(track_cfg)
        except FileNotFoundError:
            continue
        phospho_bio_cols = [kattr._proteome_to_phospho_col(c) for c in bio_cols]
        phospho_bio_cols = [c for c in phospho_bio_cols if c in sq.columns]
        ref_cols_ph = {
            plex: kattr._phospho_ref_col(plex)
            for plex in sorted(mapping["plex"].unique())
            if kattr._phospho_ref_col(plex) in sq.columns
        }
        phospho_s2p = {}
        for tp_col, plex in sample_to_plex.items():
            ph_col = kattr._proteome_to_phospho_col(tp_col)
            if ph_col in sq.columns:
                phospho_s2p[ph_col] = plex
        ph_cols = [c for c in phospho_bio_cols + list(ref_cols_ph.values()) if c in sq.columns]
        sq_quant_raw = sq[ph_cols].apply(pd.to_numeric, errors="coerce")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="divide by zero encountered in log2", category=RuntimeWarning)
            if len(ref_cols_ph) >= 4:
                sq_quant_norm = kattr._irs_normalize(sq_quant_raw, ref_cols_ph, phospho_s2p)
            else:
                sq_quant_norm = kattr._median_center_normalize(sq_quant_raw, phospho_s2p)

        sq_genes = sq["gene_symbol"].fillna("").astype(str).str.upper()
        tp_row_for_site = sq_genes.map(gene_to_tp_idx).fillna(-1).astype(int).to_numpy()
        matched = tp_row_for_site >= 0
        protein_gene = np.where(matched, sq_genes.to_numpy(), "")

        base = pd.DataFrame({
            "site_id": sq["site_id"].values,
            "gene_symbol": sq["gene_symbol"].values,
            "motif": sq["motif"].values,
            "protein_gene": protein_gene,
            "matched_protein": matched,
        })

        track_subdir = MEASUREMENT_TRACE_DIR if track_key == "st" else os.path.join(MEASUREMENT_TRACE_DIR, "py")
        os.makedirs(track_subdir, exist_ok=True)
        sample_files: dict[str, str] = {}
        preview: list[dict] = []
        for sample in bio_cols:
            ph_col = kattr._proteome_to_phospho_col(sample)
            raw_ph = sq_quant_raw[ph_col] if ph_col in sq_quant_raw.columns else pd.Series(np.nan, index=sq.index)
            irs_ph = sq_quant_norm[ph_col] if ph_col in sq_quant_norm.columns else pd.Series(np.nan, index=sq.index)

            raw_prot = pd.Series(np.nan, index=sq.index, dtype="float64")
            irs_prot = pd.Series(np.nan, index=sq.index, dtype="float64")
            if sample in tp_quant_raw.columns and sample in tp_norm.columns:
                valid = np.where(matched)[0]
                raw_vals = tp_quant_raw[sample].to_numpy()
                norm_vals = tp_norm[sample].to_numpy()
                raw_prot.iloc[valid] = raw_vals[tp_row_for_site[valid]]
                irs_prot.iloc[valid] = norm_vals[tp_row_for_site[valid]]

            log_ph = _log2_series(irs_ph)
            log_prot = _log2_series(irs_prot)
            trace = base.copy()
            trace["raw_phospho"] = raw_ph
            trace["raw_protein"] = raw_prot
            trace["irs_phospho"] = irs_ph
            trace["irs_protein"] = irs_prot
            trace["log2_irs_phospho"] = log_ph
            trace["log2_irs_protein"] = log_prot
            trace["stoichiometry"] = log_ph - log_prot
            trace = trace[columns]

            dest = os.path.join(track_subdir, f"{sample}.csv")
            trace.to_csv(dest, index=False)
            sample_files[sample] = os.path.relpath(dest, UNIFIED_VIEWER_DIR)
            _trace_written_files.append(os.path.relpath(dest, MEASUREMENT_TRACE_DIR))
            if not preview:
                preview = trace.head(AUDIT_PREVIEW_ROWS).where(
                    pd.notna(trace.head(AUDIT_PREVIEW_ROWS)), None
                ).to_dict("records")

        tracks_index[residue_label] = {
            "residue": residue_label,
            "track": track_cfg.get("name", track_key),
            "normalization_method": norm_method,
            "row_count_per_sample": int(len(base)),
            "matched_site_count": int(matched.sum()),
            "unmatched_site_count": int((~matched).sum()),
            "sample_count": int(len(sample_files)),
            "sample_files": sample_files,
            "preview": _sanitize(preview),
        }
        if not first_preview:
            first_preview = _sanitize(preview)

    # Top-level fields keep the v2 shape (sample_files, preview) defaulting to ST
    # so any legacy reader still works; new readers should consult `tracks`.
    st_block = tracks_index.get("ST", {})
    index = {
        "label": "Measurement trace",
        "normalization_method": norm_method,
        "trace_schema_version": MEASUREMENT_TRACE_SCHEMA_VERSION,
        "default_track": "ST",
        "tracks": tracks_index,
        "row_count_per_sample": st_block.get("row_count_per_sample", 0),
        "matched_site_count": st_block.get("matched_site_count", 0),
        "unmatched_site_count": st_block.get("unmatched_site_count", 0),
        "sample_count": st_block.get("sample_count", 0),
        "columns": _audit_columns(columns),
        "preview": st_block.get("preview", first_preview),
        "sample_files": st_block.get("sample_files", {}),
        "relative_path": os.path.relpath(MEASUREMENT_TRACE_DIR, UNIFIED_VIEWER_DIR),
        "source_path": "derived from phospho and total-proteome workbooks",
    }
    with open(MEASUREMENT_TRACE_INDEX, "w") as f:
        json.dump(index, f)
    _trace_index_rel = os.path.relpath(MEASUREMENT_TRACE_INDEX, MEASUREMENT_TRACE_DIR)
    _write_build_cache(
        "measurement_trace", _trace_sig, MEASUREMENT_TRACE_DIR,
        [_trace_index_rel] + _trace_written_files, index,
    )
    return index


def ensure_transcript_trace_sources() -> dict:
    """Build (or reuse) per-cluster transcript pseudobulk shards backing the
    Incytr Pathways "Measurement Trace" panel. Hard-fails on missing substrate
    — no silent skip, no empty panel."""
    # Local import to keep the heavy aggexp loader out of import-time.
    from integration import build_transcript_trace as btt  # noqa: E402

    rebuild = True
    if os.path.exists(TRANSCRIPT_TRACE_INDEX):
        with open(TRANSCRIPT_TRACE_INDEX) as f:
            existing = json.load(f)
        if existing.get("trace_schema_version") == TRANSCRIPT_TRACE_SCHEMA_VERSION:
            rebuild = False
            index = existing
    if rebuild:
        index = btt.build(force=True)
    return {
        "schema_version": TRANSCRIPT_TRACE_SCHEMA_VERSION,
        "relative_path": index.get("relative_path"),
        "clusters": index.get("clusters", []),
        "sample_groups": index.get("groups", []),
        "n_libraries_per_arm": index.get("n_libraries_per_arm", 1),
        "note": index.get("note", ""),
        "filename_template": index.get("filename_template",
                                       "{cluster}.parquet"),
        "sanitize_rule": index.get("sanitize_rule",
                                   "replace('/', '-'); replace(' ', '_')"),
    }


def ensure_omics_trace_sources() -> dict:
    """Build (or reuse) per-cluster protein + phospho raw-value shards for the
    Incytr Pathways Evidence tab. Hard-fails on missing substrate — no silent
    skip, no empty panel."""
    from integration import build_omics_trace as bot  # noqa: E402

    rebuild = True
    if os.path.exists(OMICS_TRACE_INDEX):
        with open(OMICS_TRACE_INDEX) as f:
            existing = json.load(f)
        if existing.get("omics_schema_version") == OMICS_TRACE_SCHEMA_VERSION:
            rebuild = False
            index = existing
    if rebuild:
        index = bot.build(force=True)
    return {
        "schema_version": OMICS_TRACE_SCHEMA_VERSION,
        "relative_path": index.get("relative_path"),
        "clusters": index.get("clusters", []),
        "layers": index.get("layers", ["protein", "phospho_ps", "phospho_py"]),
        "filename_template": index.get("filename_template", "{cluster}.parquet"),
        "sanitize_rule": index.get("sanitize_rule",
                                   "replace('/', '-'); replace(' ', '_')"),
        "gene_scope": index.get("gene_scope", "all_measured_genes"),
        "n_routed_cluster_gene_pairs": index.get(
            "n_routed_cluster_gene_pairs", None
        ),
        "log2_value_note": index.get("log2_value_note", ""),
    }


def ensure_omics_trace_normalized_sources() -> dict:
    """Build (or reuse) per-cluster limma-normalized condition means backing
    the Evidence tab's right-edge LFC recomputation. Hard-fails if the
    build-time round-trip vs Incytr's stored ``*_log2FC`` exceeds 1e-4."""
    from integration import build_normalized_substrate as bns  # noqa: E402

    rebuild = True
    if os.path.exists(OMICS_TRACE_NORMALIZED_INDEX):
        with open(OMICS_TRACE_NORMALIZED_INDEX) as f:
            existing = json.load(f)
        if existing.get("schema_version") == OMICS_TRACE_NORMALIZED_SCHEMA_VERSION:
            rebuild = False
            index = existing
    if rebuild:
        index = bns.build(force=True)
    return {
        "schema_version": OMICS_TRACE_NORMALIZED_SCHEMA_VERSION,
        "relative_path": index.get("relative_path"),
        "clusters": index.get("clusters", []),
        "contrasts": index.get("contrasts", []),
        "layers": index.get("layers", ["protein", "phospho_ps", "phospho_py"]),
        "epsilon": index.get("epsilon"),
        "epsilon_note": index.get("epsilon_note", ""),
        "filename_template": index.get("filename_template", "{cluster}.parquet"),
        "sanitize_rule": index.get("sanitize_rule",
                                   "replace('/', '-'); replace(' ', '_')"),
    }


def ensure_5xfad_omics_trace_sources(tissue: str) -> dict:
    """Build (or reuse) the per-sample 5xFAD protein + phospho deconvolution
    shards for one tissue's Evidence panel. Asserts the per-sample mean
    reconciles to the condition-level deconvoluted value (rel 1e-6) at build
    time. Returns the per-context block for ``meta.omics_trace.by_context``."""
    from integration import build_omics_trace_fivexfad as botf  # noqa: E402

    index = botf.build_tissue(tissue)
    return {
        "schema_version": OMICS_TRACE_FIVEXFAD_SCHEMA_VERSION,
        "relative_path": index.get("relative_path"),
        "clusters": index.get("clusters", []),
        "layers": index.get("layers", ["protein", "phospho_ps", "phospho_py"]),
        "filename_template": index.get("filename_template", "{cluster}.parquet"),
        "sanitize_rule": index.get("sanitize_rule",
                                   "replace('/', '-'); replace(' ', '_')"),
        "gene_scope": index.get("gene_scope",
                                "routed_incytr_pathway_evidence_genes"),
        "reconciliation_max_rel_err": index.get("reconciliation_max_rel_err"),
    }


def ensure_5xfad_transcript_trace_sources(tissue: str) -> dict:
    """Build (or reuse) the per-cluster 5xFAD transcript pseudobulk shards for
    one tissue's Evidence panel. Returns the per-context block for
    ``meta.transcript_trace.by_context``."""
    from integration import build_transcript_trace_fivexfad as bttf  # noqa: E402

    index = bttf.build_tissue(tissue)
    return {
        "schema_version": TRANSCRIPT_TRACE_FIVEXFAD_SCHEMA_VERSION,
        "relative_path": index.get("relative_path"),
        "clusters": index.get("clusters", []),
        "sample_groups": index.get("groups", []),
        "n_libraries_per_arm": index.get("n_libraries_per_arm", 1),
        "note": index.get("note", ""),
        "filename_template": index.get("filename_template", "{cluster}.parquet"),
        "sanitize_rule": index.get("sanitize_rule",
                                   "replace('/', '-'); replace(' ', '_')"),
    }


def assert_pathway_fc_round_trips() -> None:
    """Default-mode round-trip assertion: spot-check ~100 rows per contrast.

    Recomputes every node's ``*_log2FC`` from the canonical per-cluster
    substrates and asserts agreement with stored values in the
    ``edge_slices/incytr_pathways/`` shards to within 1e-4. Catches:
      - Cluster-routing bugs (the original EM→sender bug).
      - Substrate drift (Incytr ran against an older decomposition).
      - Sign-flip drift (pair_to_receiver_cache.py sign re-introduced).
      - Aggregation mismatch (phospho per-gene rule divergence).

    Hard-fails with a named (contrast, sender, receiver, node, layer, gene,
    stored, recomputed, delta) message on any drift. See
    ``alz/integration/verify_pathway_round_trip.py`` for the full verifier
    including ``--strict`` mode.

    Skipped gracefully when the edge_slices or substrate dirs do not yet exist
    (first build or partial build — the build itself will create them).
    """
    from integration import verify_pathway_round_trip as vpr  # noqa: E402

    from viewer.paths import (  # noqa: E402
        EDGE_SLICES_INCYTR_PATHWAYS_DIR,
        OMICS_TRACE_NORMALIZED_DIR,
    )
    transcript_dir = os.path.join(UNIFIED_VIEWER_DIR, "audit_sources", "transcript_trace")

    if (
        not os.path.exists(EDGE_SLICES_INCYTR_PATHWAYS_DIR)
        or not glob.glob(os.path.join(EDGE_SLICES_INCYTR_PATHWAYS_DIR, "*.parquet"))
    ):
        print(
            "  assert_pathway_fc_round_trips: SKIP — edge_slices/incytr_pathways/ "
            "not yet built (will be created this run)",
            flush=True,
        )
        return
    if not os.path.exists(OMICS_TRACE_NORMALIZED_DIR):
        print(
            "  assert_pathway_fc_round_trips: SKIP — omics_trace_normalized/ "
            "not yet built",
            flush=True,
        )
        return
    if not os.path.exists(transcript_dir):
        print(
            "  assert_pathway_fc_round_trips: SKIP — transcript_trace/ "
            "not yet built",
            flush=True,
        )
        return

    print("  assert_pathway_fc_round_trips: running default-mode spot-check …",
          flush=True)
    result = vpr.verify(strict=False)
    print(
        f"  assert_pathway_fc_round_trips: "
        f"slices={result['slices_checked']} "
        f"rows={result['rows_checked']:,} "
        f"failures={result['failures']} "
        f"runtime={result['runtime_s']:.1f}s",
        flush=True,
    )
    if result["failures"]:
        msgs = "\n  ".join(result["failure_msgs"][:20])
        raise RuntimeError(
            f"assert_pathway_fc_round_trips: {result['failures']} "
            f"round-trip failure(s) exceed tol=1e-4. "
            f"First {min(20, result['failures'])} failures:\n  {msgs}"
        )


def build_audit_manifest() -> dict:
    """Metadata + small previews for full audit tables copied beside HTML."""
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
            "columns": _audit_columns(columns),
            "preview": _sanitize(preview),
            "relative_path": rel,
            "source_path": os.path.relpath(src, config.REPO_ROOT),
        }
    return {
        "preview_rows": AUDIT_PREVIEW_ROWS,
        "tables": tables,
        "measurement_trace": ensure_measurement_trace_sources(),
    }


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class UnifiedData:
    """All non-edge inputs + a lazy handle to the full edge parquet."""

    # Kinase-side
    kinase_activity: pd.DataFrame
    celltype_evidence: pd.DataFrame
    kinase_hypothesis: pd.DataFrame
    mea_stoichiometry: pd.DataFrame

    unified_attribution: pd.DataFrame
    unified_attribution_full: pd.DataFrame
    decomposition: pd.DataFrame

    # Vocabulary spine for kinase_id / contrast_id / cell_type ordering;
    # derived in load_all_data from the live-pipeline outputs.
    edge_metadata: dict = field(default_factory=dict)

    def summary(self) -> dict:
        md = self.edge_metadata
        return {
            "kinases": len(md.get("kinases", [])),
            "celltypes": len(md.get("celltypes", [])),
            "contrasts": len(md.get("contrasts", [])),
            "edges": md.get("n_edges", 0),
            "kinase_activity_rows": len(self.kinase_activity),
            "celltype_evidence_rows": len(self.celltype_evidence),
            "kinase_hypothesis_rows": len(self.kinase_hypothesis),
            "mea_rows": len(self.mea_stoichiometry),
            "unified_attribution_rows": len(self.unified_attribution),
            "decomposition_rows": len(self.decomposition),
        }


def load_all_data() -> UnifiedData:
    """Load every non-edge input; open the edge parquet as a lazy handle."""
    ar_dir = config.ATTRIBUTION_RECOVERY_OUTPUT_DIR
    ka_dir = config.KINASE_ATTRIBUTION_OUTPUT_DIR

    kinase_activity = pd.read_csv(os.path.join(ar_dir, "kinase_activity_matrix.csv"))
    celltype_evidence = pd.read_csv(os.path.join(ar_dir, "celltype_evidence_table.csv"))
    kinase_hypothesis = pd.read_csv(os.path.join(ar_dir, "kinase_hypothesis_table.csv"))

    # Concatenate ST + pY MEA outputs so the kinase browser, NES trajectory,
    # and audit context all see a unified set of kinases. Each row carries
    # `residue_type` (ST/Y) so the viewer can route kinase-specific lookups
    # to the matching track-suffixed audit files.
    mea_frames = []
    for suffix, residue in _AUDIT_TRACKS:
        path = os.path.join(ka_dir, f"mea_stoichiometry{suffix}.csv")
        if not os.path.exists(path):
            continue
        cols = ["kinase", "NES", "FDR", "contrast"]
        head = pd.read_csv(path, nrows=0).columns
        if "residue_type" in head:
            cols.append("residue_type")
        df = pd.read_csv(path, usecols=cols)
        if "residue_type" not in df.columns:
            df["residue_type"] = residue
        mea_frames.append(df)
    mea = pd.concat(mea_frames, ignore_index=True) if mea_frames else pd.DataFrame(
        columns=["kinase", "NES", "FDR", "contrast", "residue_type"])

    unified_attribution = pd.read_csv(
        os.path.join(config.KINASE_ATTRIBUTION_OUTPUT_DIR, "unified_attribution.csv"),
        usecols=[
            "kinase", "gene_symbol", "contrast", "cell_type",
            "NES", "FDR", "confidence_tier",
        ],
    )
    # Load full attribution table (all tiers incl. low/none) as the single
    # source of truth for the attribution_index payload.
    _ua_full_path = os.path.join(config.KINASE_ATTRIBUTION_OUTPUT_DIR,
                                 "unified_attribution_full.csv")
    _ua_full_cols = [
        "kinase", "contrast", "cell_type",
        "confidence_tier", "confidence_basis",
        # Recalculated confidence = within-cohort cell-type exclusivity over the
        # curated specificity units (config.load_specificity_unit_map: collapse
        # over-split WMB classes, keep distinct cell types split). The dominant
        # unit label names the pill, the home cluster is its top child, and
        # specificity_collapsed flags an expandable parent. The prior
        # disease-direction tier is kept as a secondary signal for the tooltip.
        "specificity_unit", "specificity_unit_label",
        "specificity_celltype", "specificity_collapsed",
        "direction_tier", "direction_basis",
        "song_direction_support", "human_location_tier", "decomp_agrees_bulk",
        # WMB expression evidence: per-class detection plus all-class
        # concentration / tier.
        "wmb_detected", "wmb_concentration", "wmb_concentration_tier",
        "wmb_mean_log2_expression", "wmb_fraction_cells_expressing",
        "wmb_binary_expressed",
        "sea_ad_lfc", "song_lfc", "concordance_source",
        "seaad_location_score", "hbca_location_score", "human_location_score",
        "decomp_nes", "decomp_fdr",
        # Song expression evidence (favored mouse signal): per-cluster detection
        # plus all-cluster concentration / tier + per-gene effective number of
        # cell types.
        "song_detected", "song_concentration", "song_concentration_of_total",
        "song_concentration_tier",
        "song_fraction_cells_expressing", "song_effective_n",
        "song_unit_effective_n",
        "song_top_celltype", "song_top_concentration",
        "song_pval", "song_fdr",
        "NES", "FDR",
    ]
    unified_attribution_full = pd.read_csv(
        _ua_full_path, usecols=_ua_full_cols,
    ) if os.path.exists(_ua_full_path) else pd.DataFrame(columns=_ua_full_cols)

    # `cluster` is renamed to `wmb_class` so downstream agreement_index /
    # decomposition_index code paths don't need updating; values are Levy-T5.
    _decomp_path = os.path.join(
        config.REPO_ROOT, "outputs", "reports", "decomposition",
        "levy_t5", "mea_per_cluster.parquet",
    )
    _decomp_out_cols = ["kinase", "wmb_class", "contrast", "NES", "FDR"]
    if os.path.exists(_decomp_path):
        _decomp_read_cols = ["kinase", "cluster", "contrast", "NES", "FDR", "track"]
        decomposition = pq.read_table(
            _decomp_path, columns=_decomp_read_cols,
        ).to_pandas()
        decomposition = decomposition[decomposition["track"] == "st"].drop(columns=["track"])
        decomposition = decomposition.rename(columns={"cluster": "wmb_class"})
    else:
        decomposition = pd.DataFrame(columns=_decomp_out_cols)

    # Vocabulary spine: derived from the live pipeline. kinase IDs follow
    # activity-matrix order; contrasts follow the MEA output; cell-type IDs
    # follow config.WMB_CLASSES (the canonical 34-class taxonomy).
    edge_metadata = {
        "kinases": list(dict.fromkeys(kinase_activity["kinase"].astype(str).tolist())),
        "celltypes": list(config.WMB_CLASSES),
        "contrasts": sorted(mea["contrast"].dropna().astype(str).unique().tolist()),
    }

    return UnifiedData(
        kinase_activity=kinase_activity,
        celltype_evidence=celltype_evidence,
        kinase_hypothesis=kinase_hypothesis,
        mea_stoichiometry=mea,
        unified_attribution=unified_attribution,
        unified_attribution_full=unified_attribution_full,
        decomposition=decomposition,
        edge_metadata=edge_metadata,
    )


















































def build_payload(data: UnifiedData) -> dict:
    """Assemble the full JSON payload (no edges — that's the sidecar)."""
    from kinase_library.modules import data as kl_data

    sb = build_song_viewer_slice(data)

    # Kinase family map
    try:
        fam = kl_data.get_kinase_family(data.edge_metadata["kinases"]).to_dict()
    except Exception as e:
        print(f"  (warn) family resolve failed: {e}; using empty map", flush=True)
        fam = {}

    contrasts = data.edge_metadata["contrasts"]
    context_id = "song_ad"

    # Attribution is detection-gated across cohorts (standard metric); the Song
    # and WMB share fold-pill baselines (song_uniform) are gone. wmb_uniform is
    # kept as the even-split baseline the viewer badge tooltip uses to translate
    # the detection-based wmb_concentration_tier into its concentration threshold
    # (tier × uniform). See docs/foundation/standard_attribution_metric.md.
    meta = {
        "schema_version": SCHEMA_VERSION,
        "viewer_payload_schema_version": 2,
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "wmb_uniform": config.wmb_specificity_uniform(),
        "cohort": "song_ad",
        "default_context": context_id,
        "contexts": [
            {
                "id": context_id,
                "label": COHORT_DISPLAY["song"],
                "cohort": "song_ad",
                "axis_kind": "cohort",
                "contrasts": list(contrasts),
                "contrast_axis": {
                    "primary": "disease_timepoint",
                    "groups": list(config.DISEASE_GROUPS),
                    "timepoints": list(config.TIMEPOINTS),
                },
                "celltypes": list(data.edge_metadata["celltypes"]),
                "capabilities": {
                    "kinases": True,
                    "celltypes": True,
                    "incytr": sb.incytr_present,
                    "decomp_ols": sb.decomp_ols_slice_count > 0,
                    "song_concordance": bool(sb.song_concordance_present_genes),
                    "human_reference": False,
                    "supporting_5xfad": False,
                    "subclass_breakdown": True,
                    "audit_tables": True,
                    "transcript_trace": True,
                    "omics_trace": True,
                },
                "notes": [],
            }
        ],
        "capabilities": {
            "contexts": True,
            "kinases": True,
            "celltypes": True,
            "incytr": sb.incytr_present,
            "decomp_ols": sb.decomp_ols_slice_count > 0,
            "song_concordance": bool(sb.song_concordance_present_genes),
            "human_reference": False,
            "supporting_5xfad": False,
            "subclass_breakdown": True,
            "audit_tables": True,
            "transcript_trace": True,
            "omics_trace": True,
        },
        "contrasts": contrasts,
        "diseaseGroups": list(config.DISEASE_GROUPS),
        "timepoints": list(config.TIMEPOINTS),
        "diseaseColors": dict(config.DISEASE_COLORS),
        "familyMap": fam,
        # by_context so a 5xFAD context resolves its own shards — flat would
        # return Song's index under a 5xFAD context and the overlapping cluster
        # names would load the wrong shards. 5xFAD tissues are merged in below,
        # after their pathway indices exist. omics_trace_normalized stays flat
        # (build-time gate only; never read by the viewer).
        "transcript_trace": {"by_context": {context_id: ensure_transcript_trace_sources()}},
        "omics_trace": {"by_context": {context_id: ensure_omics_trace_sources()}},
        "omics_trace_normalized": ensure_omics_trace_normalized_sources(),
    }

    mukesh_slice = build_mukesh_viewer_slice()
    if mukesh_slice is not None:
        meta["contexts"][0]["capabilities"]["human_reference"] = True
        _human_block = mukesh_slice.owned_sections["human"]
        print(f"  human slice: {len(_human_block['kinases']['id']):,} kinase rows "
              f"× {len(_human_block['donors'])} donors", flush=True)
    fivexfad_slice = build_fivexfad_viewer_slice(data)
    if fivexfad_slice is not None:
        meta["contexts"][0]["capabilities"]["supporting_5xfad"] = True

    # Substrate Conservation (D1) — cross-cohort human↔5xFAD substrate comparison.
    # Only when both cohorts are present, since the tab's direction glyphs join
    # against the human + supporting_5xfad sections.
    substrate_slice = None
    if mukesh_slice is not None and fivexfad_slice is not None:
        substrate_slice = build_substrate_compare_slice()
        if substrate_slice is not None:
            meta["contexts"][0]["capabilities"]["substrate_compare"] = True

    # 5xFAD incytr: build per-tissue blocks; merge into incytr_pathways.by_context
    # AFTER composition, since Song already owns the incytr_pathways owned section.
    fivexfad_incytr_blocks = build_5xfad_incytr_blocks()
    _5xfad_incytr_ctx_labels = {
        "fivexfad_cortex": f'{COHORT_DISPLAY["fivexfad"]} Cortex',
        "fivexfad_hippocampus": f'{COHORT_DISPLAY["fivexfad"]} Hippocampus',
    }
    _5xfad_incytr_capabilities = {
        "kinases": False, "celltypes": False, "incytr": True,
        "decomp_ols": False, "song_concordance": False,
        "human_reference": False, "supporting_5xfad": False,
        "subclass_breakdown": False, "audit_tables": False,
        "transcript_trace": False, "omics_trace": False,
    }
    for ctx_id, block in fivexfad_incytr_blocks.items():
        contrasts = block.get("contrasts", [])
        groups = sorted({c.split("_")[0] for c in contrasts if "_" in c})
        timepoints = [c.split("_", 1)[1] for c in contrasts if "_" in c]
        tissue = ctx_id.replace("fivexfad_", "")

        # Per-sample evidence shards for this tissue (built now that the 5xFAD
        # pathway index exists); merge into the by_context meta + flip caps.
        ot_block = ensure_5xfad_omics_trace_sources(tissue)
        tt_block = ensure_5xfad_transcript_trace_sources(tissue)
        meta["omics_trace"]["by_context"][ctx_id] = ot_block
        meta["transcript_trace"]["by_context"][ctx_id] = tt_block
        caps = {**_5xfad_incytr_capabilities}
        caps["omics_trace"] = bool(ot_block.get("clusters"))
        caps["transcript_trace"] = bool(tt_block.get("clusters"))

        meta["contexts"].append({
            "id": ctx_id,
            "label": _5xfad_incytr_ctx_labels[ctx_id],
            "cohort": "fivexfad",
            "axis_kind": "timepoint",
            "contrasts": contrasts,
            "contrast_axis": {
                "primary": "timepoint",
                "groups": groups,
                "timepoints": timepoints,
                "baseline": "WT",
            },
            "capabilities": caps,
            "celltypes": block.get("celltypes", []),
            "notes": [],
        })

    slices = [sb.slice]
    if mukesh_slice is not None:
        slices.append(mukesh_slice)
    if fivexfad_slice is not None:
        slices.append(fivexfad_slice)
    if substrate_slice is not None:
        slices.append(substrate_slice)

    edge_slice_ref_base = {
        "schema_version": SCHEMA_VERSION,
        "human_perdonor_url": "edge_slices/human_perdonor/",
        "human_perdonor_index": "edge_slices/human_perdonor/index.json",
        "present_human_perdonor_kinase_ids": [],
    }

    payload = compose_viewer_slices(
        slices,
        meta=meta,
        audit_manifest_base=build_audit_manifest(),
        edge_slice_ref_base=edge_slice_ref_base,
        kinase_motifs_builder=_build_kinase_motifs,
    )

    # Merge 5xFAD incytr blocks into incytr_pathways.by_context (Song already
    # owns this section; post-composition merge avoids the owned-section collision).
    if fivexfad_incytr_blocks:
        ip = payload.setdefault("incytr_pathways", {})
        by_ctx = ip.setdefault("by_context", {})
        by_ctx.update(fivexfad_incytr_blocks)
        if payload.get("meta", {}).get("capabilities", {}).get("incytr") is False:
            payload["meta"]["capabilities"]["incytr"] = True

    return _sanitize(payload)


def write_payload(payload: dict) -> dict:
    os.makedirs(UNIFIED_VIEWER_DIR, exist_ok=True)
    json_str = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    raw = json_str.encode("utf-8")
    gz = gzip.compress(raw, compresslevel=6)
    with open(PAYLOAD_JSON_GZ, "wb") as f:
        f.write(gz)
    return {"raw_bytes": len(raw), "gzip_bytes": len(gz), "json_str": json_str}


def refresh_supporting_5xfad_payload() -> dict:
    """Refresh only the 5xFAD supporting block in an existing viewer payload."""
    if not os.path.exists(PAYLOAD_JSON_GZ):
        raise SystemExit(
            f"payload missing at {PAYLOAD_JSON_GZ}; run the full viewer build once before "
            "using --supporting-5xfad-only"
        )
    with gzip.open(PAYLOAD_JSON_GZ, "rt", encoding="utf-8") as f:
        payload = json.load(f)

    _fivexfad_slice = build_fivexfad_viewer_slice(data=None)
    supporting_5xfad = (
        _fivexfad_slice.owned_sections["supporting_5xfad"]
        if _fivexfad_slice is not None else None
    )
    meta = payload.setdefault("meta", {})
    capabilities = meta.setdefault("capabilities", {})
    contexts = meta.setdefault("contexts", [])
    if supporting_5xfad is None:
        payload.pop("supporting_5xfad", None)
        capabilities["supporting_5xfad"] = False
        for ctx in contexts:
            ctx.setdefault("capabilities", {})["supporting_5xfad"] = False
    else:
        payload["supporting_5xfad"] = supporting_5xfad
        capabilities["supporting_5xfad"] = True
        for ctx in contexts:
            ctx.setdefault("capabilities", {})["supporting_5xfad"] = True

    sizes = write_payload(_sanitize(payload))
    html = write_html(inline_payload=False)
    return {
        "raw_bytes": sizes["raw_bytes"],
        "gzip_bytes": sizes["gzip_bytes"],
        "html_bytes": html["html_bytes"],
        "output": html["output"],
    }


# ---------------------------------------------------------------------------
# HTML shell — rendered from alz/viewer/template/index.html.j2 via Jinja.
# Sentinel substitution (__APP_COLOR__, __PAYLOAD_SENTINEL__, etc.) happens
# in write_html() AFTER Jinja render. Jinja's role here is purely chunk
# inclusion; CSS/JS braces are never reparsed because raw() returns opaque
# strings that aren't passed back through the Jinja parser.
# ---------------------------------------------------------------------------

from jinja2 import Environment, FileSystemLoader  # noqa: E402

_TEMPLATE_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "viewer", "template"
)
_SHARED_TEMPLATE_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "viewer_shared", "template"
)
_VIEWER_SPECIFIC_TAB_INCLUDES = [
    "js/tabs/attribution_manifest_song.js",
    "js/tabs/attribution_manifest_fivexfad.js",
    "js/tabs/kinase_human.js",
    "js/tabs/kinase_fivexfad.js",
    "js/tabs/kinase_crosstable.js",
    "js/tabs/substrate_compare.js",
]


def _render_template() -> str:
    def _raw(path: str) -> str:
        local_path = os.path.join(_TEMPLATE_DIR, path)
        shared_path = os.path.join(_SHARED_TEMPLATE_DIR, path)
        source = local_path if os.path.exists(local_path) else shared_path
        with open(source) as f:
            return f.read()

    env = Environment(
        loader=FileSystemLoader([_TEMPLATE_DIR, _SHARED_TEMPLATE_DIR]),
        keep_trailing_newline=True,
    )
    env.globals["raw"] = _raw
    return env.get_template("index.html.j2").render(
        viewer_specific_tab_includes=_VIEWER_SPECIFIC_TAB_INCLUDES
    )




def write_html(
    payload: dict | None = None,
    json_str: str | None = None,
    *,
    inline_payload: bool = False,
) -> dict:
    """Emit the unified viewer HTML at UNIFIED_VIEWER_DIR/index.html.

    Sibling dirs (edge_slices/, edge_summaries/) are written by
    build_edge_shards.py; this function only writes the HTML.
    """
    os.makedirs(UNIFIED_VIEWER_DIR, exist_ok=True)
    if inline_payload and json_str is None:
        if payload is None:
            raise ValueError("payload or json_str is required for inline HTML")
        json_str = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    payload_text = json_str if inline_payload else "null"
    # Default hosted/S3 mode keeps index.html small and loads
    # unified_viewer.payload.json.gz via client-side DecompressionStream.
    # --inline-payload preserves the archival/offline single-file mode.
    html = _render_template()
    for sentinel, value in (
        ("__APP_COLOR__", config.DISEASE_COLORS["App"]),
        ("__TAU_COLOR__", config.DISEASE_COLORS["Tau"]),
        ("__APTT_COLOR__", config.DISEASE_COLORS["ApTt"]),
        ("__PAYLOAD_SENTINEL__", payload_text),
    ):
        html = html.replace(sentinel, value)
    raw = html.encode("utf-8")
    with open(UNIFIED_VIEWER_HTML, "wb") as f:
        f.write(raw)
    methods_bytes = 0
    if os.path.exists(PIPELINE_OVERVIEW_SRC):
        _copy_if_different(PIPELINE_OVERVIEW_SRC, PIPELINE_OVERVIEW_DEST)
        methods_bytes = os.path.getsize(PIPELINE_OVERVIEW_DEST)
    else:
        print(f"WARNING: {PIPELINE_OVERVIEW_SRC} not found; "
              "Methods tab will 404. Render docs/methods/pipeline_overview.qmd first "
              "(quarto render docs/methods/pipeline_overview.qmd --to html).",
              file=sys.stderr)
    return {"html_bytes": len(raw), "output": UNIFIED_VIEWER_HTML,
            "methods_bytes": methods_bytes}


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def validate(data: UnifiedData) -> str:
    """Write pipeline_notes/phase2_payload_report.md. Returns the md string."""
    errors: list[str] = []
    warnings: list[str] = []

    md = data.edge_metadata
    n_kinases = len(md["kinases"])
    n_celltypes = len(md["celltypes"])
    n_contrasts = len(md["contrasts"])

    # Payload size
    if not os.path.exists(PAYLOAD_JSON_GZ):
        errors.append(f"payload JSON missing: {PAYLOAD_JSON_GZ}")
        raw_bytes = gzip_bytes = 0
        payload = None
    else:
        gzip_bytes = os.path.getsize(PAYLOAD_JSON_GZ)
        with gzip.open(PAYLOAD_JSON_GZ, "rt", encoding="utf-8") as f:
            payload_str = f.read()
        raw_bytes = len(payload_str.encode("utf-8"))
        payload = json.loads(payload_str)

    if raw_bytes >= 100 * 1024 * 1024:
        errors.append(f"payload raw {raw_bytes/1e6:.1f} MB exceeds 100 MB cap")
    if gzip_bytes >= 20 * 1024 * 1024:
        errors.append(f"payload gzip {gzip_bytes/1e6:.1f} MB exceeds 20 MB cap")

    # Structural
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

        default_context = meta.get("default_context")
        pk = payload["kinases"]["by_context"].get(default_context, {})
        pc_ = payload["celltypes"]["by_context"].get(default_context, {})

        if len(pk["id"]) != n_kinases:
            errors.append(f"kinases rows {len(pk['id'])} != vocab {n_kinases}")
        if len(pc_["id"]) != n_celltypes:
            errors.append(f"celltypes rows {len(pc_['id'])} != vocab {n_celltypes}")

        audit_tables = payload.get("audit_tables", {}).get("tables", {})
        expected_audit = {k for k, _, _ in AUDIT_TABLE_SPECS}
        missing_audit = expected_audit - set(audit_tables)
        if missing_audit:
            errors.append(f"missing audit table manifest entries: {sorted(missing_audit)}")
        for key, _, src in AUDIT_TABLE_SPECS:
            meta = audit_tables.get(key)
            if not meta or meta.get("missing"):
                warnings.append(f"audit source missing: {key} ({src})")
                continue
            rel = meta.get("relative_path")
            if rel and not os.path.exists(os.path.join(UNIFIED_VIEWER_DIR, rel)):
                errors.append(f"audit source copy missing for {key}: {rel}")
            if src.endswith(".csv") and os.path.exists(src):
                actual_rows = _count_csv_rows(src)
                if int(meta.get("row_count", -1)) != actual_rows:
                    errors.append(
                        f"audit table {key} rows {meta.get('row_count')} != source {actual_rows}"
                    )
                actual_cols = len(pd.read_csv(src, nrows=0).columns)
                if int(meta.get("column_count", -1)) != actual_cols:
                    errors.append(
                        f"audit table {key} columns {meta.get('column_count')} != source {actual_cols}"
                    )
            for col in meta.get("columns", []):
                if not col.get("label") or not col.get("raw") or not col.get("definition"):
                    errors.append(f"audit table {key} has incomplete column tooltip metadata")

    peak_mb = _peak_rss_mb()

    lines = [
        "# Phase 2 Payload Report",
        "",
        f"_Generated {pd.Timestamp.utcnow().isoformat()}_",
        "",
        "## Sizes",
        "",
        f"- Payload JSON (raw): {raw_bytes/1e6:.2f} MB (cap 100)",
        f"- Payload JSON (gzip): {gzip_bytes/1e6:.2f} MB (cap 20)",
        "",
        "## Counts",
        "",
        f"- kinases: {n_kinases}",
        f"- celltypes: {n_celltypes}",
        f"- contrasts: {n_contrasts}",
        "",
        "## Memory",
        "",
        f"- Peak RSS (this process): {peak_mb:.0f} MB",
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

def _assert_input_provenance(skip_verify: bool = False) -> None:
    """Hardfail if upstream artifacts disagree on scope/spine/cohort.

    Reads sidecar JSONs from the WMB expression, decomposition, and enrichment
    stages and aborts the viewer build if any of them disagree with the
    expected scope (`config.WMB_REGION_SCOPE`), spine (`levy_t5`), or analysis
    mode (`config.ANALYSIS_MODE`). Aborts cleanly before any HTML is written.
    """
    wmb_scope_path = os.path.join(
        config.REPO_ROOT, "outputs", "reports", "wmb_expression",
        "wmb_kinase_expression.scope.json",
    )
    if os.path.exists(wmb_scope_path):
        with open(wmb_scope_path) as f:
            wmb_meta = json.load(f)
        wmb_scope = wmb_meta.get("scope")
        if wmb_scope and wmb_scope != config.WMB_REGION_SCOPE:
            raise SystemExit(
                f"WMB expression scope mismatch: file says {wmb_scope!r}, "
                f"config.WMB_REGION_SCOPE is {config.WMB_REGION_SCOPE!r}. "
                f"Re-run alz/reference/wmb_expression.py with the expected scope."
            )

    decomp_dir = os.path.join(
        config.REPO_ROOT, "outputs", "reports", "decomposition", config.CLUSTER_SPINE_NAME,
    )
    enrich_audit_path = os.path.join(decomp_dir, "enrich_audit.json")

    verify_path = os.path.join(decomp_dir, "verification.json")
    if skip_verify:
        print(f"[provenance] --skip-verify: bypassing {verify_path} gate",
              flush=True)
    elif os.path.exists(verify_path):
        with open(verify_path) as f:
            verif = json.load(f)
        hard_pass = verif.get("hard_pass", verif.get("all_pass"))
        if hard_pass is False:
            failed = [
                c.get("check") for c in (verif.get("checks") or [])
                if (
                    isinstance(c, dict)
                    and c.get("severity", "hard") == "hard"
                    and not c.get("pass", False)
                )
            ]
            raise SystemExit(
                f"Decomposition hard verification did not pass: failed checks = "
                f"{failed}. See {verify_path}."
            )

    # Cross-summary provenance: every summary that stamps `analysis_mode` /
    # `spine` must agree with config. Missing fields are allowed (older
    # artifacts predate the stamp); contradictions abort.
    summaries = [
        os.path.join(config.KINASE_ATTRIBUTION_OUTPUT_DIR, "normalization_summary.json"),
        os.path.join(config.KINASE_ATTRIBUTION_OUTPUT_DIR, "normalization_summary_pY.json"),
        os.path.join(config.KINASE_ATTRIBUTION_OUTPUT_DIR, "attribution_summary.json"),
        os.path.join(config.REPO_ROOT, "outputs", "reports", "attribution_recovery",
                     "recovery_summary.json"),
        os.path.join(decomp_dir, "per_animal", "site_level_ols.audit.json"),
        enrich_audit_path,
    ]
    for sp in summaries:
        if not os.path.exists(sp):
            continue
        with open(sp) as f:
            meta = json.load(f)
        amode = meta.get("analysis_mode")
        if amode and amode != config.ANALYSIS_MODE:
            raise SystemExit(
                f"analysis_mode mismatch in {sp}: {amode!r} != "
                f"{config.ANALYSIS_MODE!r} — chains disagree on cohort."
            )
        spine_field = meta.get("spine")
        if spine_field and spine_field != config.CLUSTER_SPINE_NAME:
            raise SystemExit(
                f"spine mismatch in {sp}: {spine_field!r} != "
                f"{config.CLUSTER_SPINE_NAME!r}."
            )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", action="store_true", help="Print input counts (Unit 2.1 smoke test)")
    ap.add_argument("--payload", action="store_true", help="Write JSON payload")
    ap.add_argument("--html", action="store_true", help="Write unified_viewer.html (requires payload)")
    ap.add_argument("--validate", action="store_true", help="Write Phase 2 validation report")
    ap.add_argument("--supporting-5xfad-only", action="store_true",
                    help="Refresh only payload.supporting_5xfad and the static HTML shell "
                         "from existing viewer outputs. Does not rebuild Song/Mukesh/Incytr "
                         "payload sections or sidecar shards.")
    ap.add_argument("--inline-payload", action="store_true",
                    help="Embed the full JSON payload in index.html for a "
                         "single-file archival/offline artifact. Default "
                         "hosted mode writes a small HTML shell that fetches "
                         "unified_viewer.payload.json.gz beside it.")
    ap.add_argument("--skip-roundtrip", action="store_true",
                    help="Skip the Item 3.5 build-time LFC round-trip assertion "
                         "(default mode, ~60s). Use only for fast iteration.")
    ap.add_argument("--skip-verify", action="store_true",
                    help="Bypass the decomposition verification.json gate. Use "
                         "when a failing check is known-orthogonal to viewer "
                         "outputs (e.g. per_cluster_vs_bulk_mea).")
    ap.add_argument("--strict-roundtrip", action="store_true",
                    help="Run the Item 3.5 round-trip in full-grid strict mode "
                         "(checks every (contrast, pair, node, layer) cell). "
                         "Intended for pre-publish / CI builds.")
    args = ap.parse_args(argv)

    if args.supporting_5xfad_only:
        info = refresh_supporting_5xfad_payload()
        print(f"  payload raw={info['raw_bytes']/1e6:.2f} MB "
              f"gzip={info['gzip_bytes']/1e6:.2f} MB")
        print(f"  html {info['html_bytes']/1e6:.2f} MB -> {info['output']}")
        return 0

    if not any([args.summary, args.payload, args.html, args.validate]):
        args.payload = True
        args.html = True

    needs_data = args.summary or args.payload or args.validate
    data = None
    if needs_data:
        _assert_input_provenance(skip_verify=args.skip_verify)
        data = load_all_data()

    if args.summary:
        assert data is not None
        print(json.dumps(data.summary(), indent=2))

    payload = None
    json_str = None
    if args.payload:
        assert data is not None
        payload = build_payload(data)
        sizes = write_payload(payload)
        json_str = sizes.pop("json_str")
        print(f"  payload raw={sizes['raw_bytes']/1e6:.2f} MB "
              f"gzip={sizes['gzip_bytes']/1e6:.2f} MB")

    if args.html:
        if args.inline_payload and payload is None:
            if not os.path.exists(PAYLOAD_JSON_GZ):
                raise SystemExit(
                    f"payload missing at {PAYLOAD_JSON_GZ}; run --payload first"
                )
            with gzip.open(PAYLOAD_JSON_GZ, "rt", encoding="utf-8") as f:
                json_str = f.read()
            payload = json.loads(json_str)
        elif not args.inline_payload:
            if not os.path.exists(PAYLOAD_JSON_GZ):
                raise SystemExit(
                    f"payload sidecar missing; run --payload first: {PAYLOAD_JSON_GZ}"
                )
            json_str = None
        info = write_html(
            payload,
            json_str=json_str,
            inline_payload=args.inline_payload,
        )
        print(f"  html {info['html_bytes']/1e6:.2f} MB -> {info['output']}")

    if args.validate:
        assert data is not None
        validate(data)

    if args.payload and not args.skip_roundtrip:
        from integration import verify_pathway_round_trip as vprt  # noqa: E402
        vprt.verify(strict=args.strict_roundtrip)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

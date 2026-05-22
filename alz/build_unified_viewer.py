#!/usr/bin/env python3
"""Unified viewer builder: single-file HTML deliverable for the kinase pipeline.

Reads the unified attribution + recovery tables (mouse), the Levy-t5 decomposition
shards (per-cluster MEA + per-animal site OLS), the pair-mode Incytr output
(`outputs/reports/incytr_pair_mode/`), and the human per-donor MEA + SEA-AD
agreement chain. Emits `outputs/reports/unified_viewer/index.html` with the
columnar payload inlined as `<script type="application/json" id="payload-data">`
plus per-entity shards under `edge_slices/{human_perdonor,decomp_ols,
song_concordance,incytr_pathways}/` fetched on demand by the JS tabs.

Usage:
    python alz/build_unified_viewer.py              # payload + html (default)
    python alz/build_unified_viewer.py --summary    # input row counts
    python alz/build_unified_viewer.py --payload    # JSON only
    python alz/build_unified_viewer.py --html       # write HTML (needs payload)
    python alz/build_unified_viewer.py --validate   # write report md
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
import time
import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "integration"))
sys.path.insert(0, os.path.join(HERE, "bulk_mea"))
sys.path.insert(0, os.path.join(HERE, "cross_reference"))

import config  # noqa: E402
import config_integration as icfg  # noqa: E402
import normalize as kattr  # noqa: E402  (alz.bulk_mea.normalize — Stage 1 helpers + load_sample_mapping)
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

from viewer.paths import (  # noqa: E402
    AUDIT_PREVIEW_ROWS,
    AUDIT_SOURCES_DIR,
    DECOMP_OLS_PARQUET,
    EDGE_SLICES_DECOMP_OLS_DIR,
    EDGE_SLICES_HUMAN_PERDONOR_DIR,
    EDGE_SLICES_INCYTR_PATHWAYS_DIR,
    EDGE_SLICES_SONG_CONCORDANCE_DIR,
    INCYTR_PAIR_MODE_OUTPUTS_DIR,
    MEASUREMENT_TRACE_DIR,
    MEASUREMENT_TRACE_INDEX,
    MEASUREMENT_TRACE_SCHEMA_VERSION,
    OMICS_TRACE_INDEX,
    OMICS_TRACE_SCHEMA_VERSION,
    OMICS_TRACE_NORMALIZED_INDEX,
    OMICS_TRACE_NORMALIZED_SCHEMA_VERSION,
    TRANSCRIPT_TRACE_INDEX,
    TRANSCRIPT_TRACE_SCHEMA_VERSION,
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

TISSUE_CATEGORIES = {
    "Excitatory": ["01 IT-ET Glut", "02 NP-CT-L6b Glut", "03 OB-CR Glut",
                   "04 DG-IMN Glut"],
    "Inhibitory": ["05 OB-IMN GABA", "06 CTX-CGE GABA", "07 CTX-MGE GABA",
                   "08 CNU-MGE GABA", "09 CNU-LGE GABA"],
    "Subcortical": ["10 LSX GABA", "11 CNU-HYa GABA", "12 HY GABA",
                    "13 CNU-HYa Glut", "14 HY Glut", "15 HY Gnrh1 Glut",
                    "16 HY MM Glut", "17 MH-LH Glut", "18 TH Glut"],
    "Brainstem": ["19 MB Glut", "20 MB GABA", "21 MB Dopa", "22 MB-HB Sero",
                  "23 P Glut", "24 MY Glut", "25 Pineal Glut",
                  "26 P GABA", "27 MY GABA"],
    "Cerebellum": ["28 CB GABA", "29 CB Glut"],
    "Non-neuronal": ["30 Astro-Epen", "31 OPC-Oligo", "32 OEC",
                     "33 Vascular", "34 Immune"],
}
RECEIVER_TO_TISSUE = {r: t for t, rs in TISSUE_CATEGORIES.items() for r in rs}

# Per-track audit tables: identical schema, separate files per analysis track.
# ST (serine/threonine) is the default suffix (none); pY tables carry the
# `_pY` suffix produced by `kinase_attribution._track_output`.
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
    "combined_score": "Combined cell-type attribution score across transcriptomic evidence sources.",
    "combined_confidence": "Confidence tier assigned to the combined attribution row.",
    "evidence_basis": "Evidence sources supporting the attribution row.",
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
    shutil.copyfile(src, dest)
    return os.path.relpath(dest, UNIFIED_VIEWER_DIR)


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


def _configure_duckdb_tempdir(con) -> None:
    d = os.environ.get("DUCKDB_TEMP_DIR", os.path.expanduser("~/.cache/duckdb"))
    os.makedirs(d, exist_ok=True)
    con.execute(f"SET temp_directory='{d}';")


def _measurement_trace_columns() -> list[str]:
    return [
        "site_id", "gene_symbol", "motif", "protein_gene", "matched_protein",
        "raw_phospho", "raw_protein", "irs_phospho", "irs_protein",
        "log2_irs_phospho", "log2_irs_protein", "stoichiometry",
    ]


def ensure_measurement_trace_sources() -> dict:
    """Create per-sample sidecar receipts for raw→IRS→stoichiometry auditing."""
    if os.path.exists(MEASUREMENT_TRACE_INDEX):
        with open(MEASUREMENT_TRACE_INDEX) as f:
            existing = json.load(f)
        if existing.get("trace_schema_version") == MEASUREMENT_TRACE_SCHEMA_VERSION:
            return existing
        shutil.rmtree(MEASUREMENT_TRACE_DIR)

    os.makedirs(MEASUREMENT_TRACE_DIR, exist_ok=True)
    mapping = kattr.load_sample_mapping()
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

    # Per-track: load that track's phospho workbook, IRS-normalize, write per-sample
    # measurement-trace CSVs into a track subdir. ST keeps the legacy unsuffixed
    # filenames for backward compatibility; pY lands under measurement_trace/py/.
    for track_key, residue_label in (("st", "ST"), ("py", "Y")):
        track_cfg = kattr._resolve_track(track_key)
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
            "NES", "FDR", "combined_score", "combined_confidence",
        ],
    )
    # Load full attribution table (all tiers incl. low/none) as the single
    # source of truth for the attribution_index payload.
    _ua_full_path = os.path.join(config.KINASE_ATTRIBUTION_OUTPUT_DIR,
                                 "unified_attribution_full.csv")
    _ua_full_cols = [
        "kinase", "contrast", "cell_type",
        "combined_confidence", "combined_score",
        "wmb_specificity", "wmb_mean_log2_expression",
        "wmb_fraction_cells_expressing", "wmb_binary_expressed",
        "sea_ad_lfc", "song_lfc", "concordance_source",
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


# ---------------------------------------------------------------------------
# Step C — JSON payload
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
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        x = float(obj)
        if np.isnan(x) or np.isinf(x):
            return None
        return round(x, decimals)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if obj is pd.NA:
        return None
    return obj


def _build_kinases_slice(data: UnifiedData) -> dict:
    """Columnar kinases table. IDs follow edge_metadata['kinases'] ordering."""
    kinases = data.edge_metadata["kinases"]
    kid = {k: i for i, k in enumerate(kinases)}

    ka = data.kinase_activity.set_index("kinase")
    hyp = data.kinase_hypothesis.set_index("kinase")
    contrasts = data.edge_metadata["contrasts"]

    cols: dict[str, list] = {
        "id": [], "name": [], "gene_symbol": [],
        "residue_type": [],
        "trajectory": [], "peak_contrast": [], "peak_NES": [],
        "n_sig_contrasts": [],
        "top_celltype_1": [], "top_celltype_2": [], "top_celltype_3": [],
        "top_celltype_1_wmb_fold": [],
        "top_celltype_1_sea_ad_lfc": [],
        "top_celltype_1_song_lfc": [],
        "n_celltype_candidates": [],
    }
    for c in contrasts:
        cols[f"NES_{c}"] = []
        cols[f"FDR_{c}"] = []

    for k in kinases:
        cols["id"].append(kid[k])
        cols["name"].append(k)
        ka_row = ka.loc[k] if k in ka.index else None
        hyp_row = hyp.loc[k] if k in hyp.index else None

        def _get(r, col, default=None):
            if r is None or col not in r.index:
                return default
            v = r[col]
            return default if pd.isna(v) else v

        cols["gene_symbol"].append(_get(ka_row, "gene_symbol", ""))
        cols["residue_type"].append(_get(ka_row, "residue_type", "ST"))
        cols["trajectory"].append(_get(ka_row, "trajectory_label", ""))
        cols["peak_contrast"].append(_get(ka_row, "peak_contrast", ""))
        cols["peak_NES"].append(_get(ka_row, "peak_NES"))
        cols["n_sig_contrasts"].append(_get(ka_row, "n_sig_contrasts", 0))
        cols["top_celltype_1"].append(_get(hyp_row, "top_celltype_1", ""))
        cols["top_celltype_2"].append(_get(hyp_row, "top_celltype_2", ""))
        cols["top_celltype_3"].append(_get(hyp_row, "top_celltype_3", ""))
        cols["top_celltype_1_wmb_fold"].append(_get(hyp_row, "top_celltype_1_wmb_fold"))
        cols["top_celltype_1_sea_ad_lfc"].append(_get(hyp_row, "top_celltype_1_sea_ad_lfc"))
        cols["top_celltype_1_song_lfc"].append(_get(hyp_row, "top_celltype_1_song_lfc"))
        cols["n_celltype_candidates"].append(_get(hyp_row, "n_celltype_candidates", 0))
        for c in contrasts:
            cols[f"NES_{c}"].append(_get(ka_row, f"{c}_NES"))
            cols[f"FDR_{c}"].append(_get(ka_row, f"{c}_FDR"))
    return cols


HUMAN_PERDONOR_DIR = os.path.join(
    config.REPO_ROOT, "outputs", "reports",
    "kinase_attribution_human", "perdonor",
)
HUMAN_TRACK_SUFFIXES = [("", "ST"), ("_pY", "Y")]


def _write_human_perdonor_substrate_slices(
    perdonor_rows: list[tuple],
) -> dict:
    """Per-kinase shards for `leading_substrates` + `substrate_motifs`.

    Together these two columns added ~50 MB to the inlined PAYLOAD; they are
    only consumed by the human Audit drawer (Trace + Running Enrichment sub-
    tabs) when a specific kinase is selected. Shard format: one parquet per
    kinase id with rows (donor, leading_substrates, substrate_motifs). Rows
    with both fields empty are skipped — the JS treats a missing row as
    "no leading edge" the same way it used to treat an empty string.
    """
    shutil.rmtree(EDGE_SLICES_HUMAN_PERDONOR_DIR, ignore_errors=True)
    os.makedirs(EDGE_SLICES_HUMAN_PERDONOR_DIR, exist_ok=True)

    by_kid: dict[int, list[tuple[str, str, str, str]]] = {}
    for r in perdonor_rows:
        kid = int(r[0])
        donor = str(r[1])
        leading = str(r[4]) if r[4] is not None else ""
        motifs = str(r[11]) if r[11] is not None else ""
        kl_pcts = str(r[12]) if len(r) > 12 and r[12] is not None else ""
        if not leading and not motifs:
            continue
        by_kid.setdefault(kid, []).append((donor, leading, motifs, kl_pcts))

    template = "{kinase_id:03d}.parquet"
    present: list[int] = []
    total_rows = 0
    for kid in sorted(by_kid):
        df = pd.DataFrame(
            by_kid[kid],
            columns=["donor", "leading_substrates", "substrate_motifs", "substrate_kl_percentiles"],
        )
        path = os.path.join(
            EDGE_SLICES_HUMAN_PERDONOR_DIR,
            template.format(kinase_id=kid),
        )
        pq.write_table(
            pa.Table.from_pandas(df, preserve_index=False),
            path, compression="zstd",
        )
        present.append(kid)
        total_rows += len(df)

    index = {
        "schema_version": SCHEMA_VERSION,
        "slice_count": len(present),
        "present_kinase_ids": present,
        "filename_template": template,
        "n_total_rows": total_rows,
    }
    with open(os.path.join(EDGE_SLICES_HUMAN_PERDONOR_DIR, "index.json"), "w") as f:
        json.dump(index, f)
    print(f"  human_perdonor_substrate: wrote {len(present)} shards "
          f"({total_rows:,} total rows)", flush=True)
    return index


def _human_track_load(suffix: str, residue: str) -> dict | None:
    """Load all per-donor CSVs for one track. Returns None if missing.

    Both the stoichiometry and raw-phospho MEA outputs are loaded when
    present. Raw-phospho is the sensitivity check (analogous to mouse
    ``alz/bulk_mea/mechanism.py``); when only stoichiometry is available the
    raw entries are empty DataFrames and the viewer hides the comparison.
    """
    rec_path = os.path.join(HUMAN_PERDONOR_DIR, f"recurrence{suffix}.csv")
    rec_ctrl_path = os.path.join(HUMAN_PERDONOR_DIR, f"recurrence_ctrl{suffix}.csv")
    nes_path = os.path.join(HUMAN_PERDONOR_DIR, f"kinase_donor_nes{suffix}.csv")
    fdr_path = os.path.join(HUMAN_PERDONOR_DIR, f"kinase_donor_fdr{suffix}.csv")
    mea_path = os.path.join(HUMAN_PERDONOR_DIR, f"mea_perdonor{suffix}.csv")
    shift_path = os.path.join(HUMAN_PERDONOR_DIR, f"mea_global_shift{suffix}.csv")
    winsor_path = os.path.join(HUMAN_PERDONOR_DIR, f"winsorized_sites{suffix}.csv")
    # Raw-phospho counterparts (optional; populated when ingest_mukesh_perdonor.py
    # has been re-run with the raw-track export enabled).
    raw_mea_path = os.path.join(HUMAN_PERDONOR_DIR, f"mea_perdonor_raw{suffix}.csv")
    raw_nes_path = os.path.join(HUMAN_PERDONOR_DIR, f"kinase_donor_nes_raw{suffix}.csv")
    raw_fdr_path = os.path.join(HUMAN_PERDONOR_DIR, f"kinase_donor_fdr_raw{suffix}.csv")
    subs_path = os.path.join(HUMAN_PERDONOR_DIR, f"mea_substrate_sets{suffix}.csv")
    # Site-level matrices live one directory up from perdonor/
    parent = os.path.dirname(HUMAN_PERDONOR_DIR)
    stoich_path = os.path.join(parent, f"stoichiometry_matrix{suffix}.csv")
    raw_path = os.path.join(parent, f"raw_phospho_normalized{suffix}.csv")
    for p in (rec_path, nes_path, fdr_path):
        if not os.path.exists(p):
            return None
    rec = pd.read_csv(rec_path)
    rec_ctrl = pd.read_csv(rec_ctrl_path) if os.path.exists(rec_ctrl_path) else pd.DataFrame()
    nes = pd.read_csv(nes_path).set_index("kinase")
    fdr = pd.read_csv(fdr_path).set_index("kinase")
    mea = pd.read_csv(mea_path) if os.path.exists(mea_path) else pd.DataFrame()
    shift = pd.read_csv(shift_path) if os.path.exists(shift_path) else pd.DataFrame()
    winsor = pd.read_csv(winsor_path) if os.path.exists(winsor_path) else pd.DataFrame()
    stoich = pd.read_csv(stoich_path) if os.path.exists(stoich_path) else pd.DataFrame()
    raw = pd.read_csv(raw_path) if os.path.exists(raw_path) else pd.DataFrame()
    raw_mea = pd.read_csv(raw_mea_path) if os.path.exists(raw_mea_path) else pd.DataFrame()
    raw_nes = pd.read_csv(raw_nes_path).set_index("kinase") if os.path.exists(raw_nes_path) else pd.DataFrame()
    raw_fdr = pd.read_csv(raw_fdr_path).set_index("kinase") if os.path.exists(raw_fdr_path) else pd.DataFrame()
    subs = pd.read_csv(subs_path) if os.path.exists(subs_path) else pd.DataFrame()
    return {
        "residue": residue, "rec": rec, "rec_ctrl": rec_ctrl,
        "nes": nes, "fdr": fdr,
        "mea": mea, "shift": shift, "winsor": winsor,
        "stoich": stoich, "raw": raw,
        "raw_mea": raw_mea, "raw_nes": raw_nes, "raw_fdr": raw_fdr,
        "subs": subs,
    }


def build_human_slice() -> tuple[dict, dict] | tuple[None, None]:
    """Per-donor human (NBB / Mukesh) kinase slice.

    Returns (human_slice, substrate_slice_index). Both are None when the
    perdonor outputs are absent — caller omits the PAYLOAD.human block so
    the mouse-only artifact stays byte-equivalent.
    """
    if not os.path.isdir(HUMAN_PERDONOR_DIR):
        return None, None
    tracks: list[dict] = []
    for suffix, residue in HUMAN_TRACK_SUFFIXES:
        t = _human_track_load(suffix, residue)
        if t is not None:
            tracks.append(t)
    if not tracks:
        return None, None

    # Donor membership — prefer the explicit donor_groups.json sidecar
    # (written by ingest_mukesh_perdonor.py); fall back to name-prefix
    # inference for older perdonor outputs that predate the sidecar.
    donor_groups_path = os.path.join(HUMAN_PERDONOR_DIR, "donor_groups.json")
    if os.path.exists(donor_groups_path):
        with open(donor_groups_path) as fh:
            _dg = json.load(fh)
        ad_set = {str(x) for x in _dg.get("ad", [])}
        ctrl_set = {str(x) for x in _dg.get("ctrl", [])}
    else:
        ad_set = set()
        ctrl_set = set()

    # AD donors drive the rendering axis ("donors"). CTRL donors are
    # rendered side-by-side as a separate column group.
    donors: list[str] = []          # AD only
    ctrl_donors: list[str] = []     # CTRL only
    seen: set[str] = set()
    for t in tracks:
        for d in list(t["nes"].columns):
            ds = str(d)
            if ds in seen:
                continue
            seen.add(ds)
            if ctrl_set:
                if ds in ad_set:
                    donors.append(ds)
                elif ds in ctrl_set:
                    ctrl_donors.append(ds)
                else:
                    # Unknown — treat as AD so we don't silently drop it.
                    donors.append(ds)
            else:
                # No sidecar: legacy prefix inference.
                (ctrl_donors if ds.upper().startswith("CTRL") else donors).append(ds)
    all_donors_axis = donors + ctrl_donors
    contrasts = [f"{d}_vs_CTRLmean" for d in all_donors_axis]

    # Gene-symbol map from the cached mapping CSV; fall back to identity.
    gene_map: dict[str, str] = {}
    gene_csv = os.path.join(
        config.REPO_ROOT, "data", "datasets", "song", "analysis_cache",
        "kinase_to_gene_mapping.csv",
    )
    if os.path.exists(gene_csv):
        gm = pd.read_csv(gene_csv)
        for k, g in zip(gm["kinase_abbreviation"], gm["gene_symbol"]):
            if pd.notna(g):
                gene_map[str(k)] = str(g)

    # Optional: SEA-AD per-kinase agreement (one cohort-level LFC per kinase,
    # collapsed across SEA-AD supertypes; no cell-type bridge because the
    # human AD samples have no per-cell-type resolution).
    seaad_csv = os.path.join(
        config.REPO_ROOT, "outputs", "reports", "kinase_attribution",
        "human_seaad_agreement.csv",
    )
    seaad_lookup: dict[tuple[str, str], dict] = {}
    if os.path.exists(seaad_csv):
        sdf = pd.read_csv(seaad_csv)
        for _, srow in sdf.iterrows():
            key = (str(srow["kinase"]), str(srow["residue_type"]))
            seaad_lookup[key] = {
                "lfc": (float(srow["sea_ad_lfc_median"])
                        if pd.notna(srow["sea_ad_lfc_median"]) else None),
                "n": int(srow["n_supertypes_finite"]) if pd.notna(srow["n_supertypes_finite"]) else 0,
                "agreement": (float(srow["sea_ad_direction_agreement"])
                              if pd.notna(srow["sea_ad_direction_agreement"]) else None),
            }

    # Build (kinase, residue) rows. Same kinase can appear once per track.
    cols: dict[str, list] = {
        "id": [], "name": [], "gene_symbol": [], "residue_type": [],
        "n_donors_sig": [], "n_donors_up": [], "n_donors_down": [],
        "n_donors_tested": [],
        "median_nes": [], "median_nes_sig_only": [],
        "sea_ad_lfc": [], "sea_ad_n_supertypes": [], "sea_ad_direction_agreement": [],
    }
    for d in all_donors_axis:
        cols[f"NES_{d}_vs_CTRLmean"] = []
        cols[f"FDR_{d}_vs_CTRLmean"] = []

    perdonor_rows: list[tuple] = []  # (kid, donor, NES, FDR, lead, ES, subs_fraction, p_value, raw_NES, raw_FDR, raw_p, substrate_motifs, substrate_kl_percentiles)
    next_id = 0
    for t in tracks:
        residue = t["residue"]
        rec = t["rec"].set_index("kinase")
        nes = t["nes"]
        fdr = t["fdr"]
        mea = t["mea"]
        raw_mea = t.get("raw_mea", pd.DataFrame())
        raw_lookup: dict[tuple[str, str], tuple[float, float, float]] = {}
        if not raw_mea.empty:
            for _, rrow in raw_mea.iterrows():
                k = str(rrow.get("kinase", ""))
                contrast = str(rrow.get("contrast", ""))
                donor = contrast.replace("_vs_CTRLmean", "")
                raw_lookup[(k, donor)] = (
                    float(rrow.get("NES")) if pd.notna(rrow.get("NES")) else float("nan"),
                    float(rrow.get("FDR")) if pd.notna(rrow.get("FDR")) else float("nan"),
                    float(rrow.get("p-value")) if pd.notna(rrow.get("p-value")) else float("nan"),
                )
        subs = t.get("subs", pd.DataFrame())
        # Per-(kinase, donor) substrate motif set used by GSEA as the hit set.
        # Mirrors the mouse `mea_substrate_sets.csv` contract; persisted by
        # `ingest_mukesh_perdonor.py`. Used at view time to replay the GSEA walk
        # against the full substrate set (not the leading-edge subset).
        subs_lookup: dict[tuple[str, str], list[str]] = {}
        subs_pct_lookup: dict[tuple[str, str], list[float]] = {}
        if not subs.empty:
            for _, srow in subs.iterrows():
                k = str(srow.get("kinase", ""))
                contrast = str(srow.get("contrast", ""))
                donor = contrast.replace("_vs_CTRLmean", "")
                motif = str(srow.get("motif", ""))
                if motif:
                    subs_lookup.setdefault((k, donor), []).append(motif)
                    pct_val = srow.get("kl_percentile")
                    subs_pct_lookup.setdefault((k, donor), []).append(
                        float(pct_val) if pd.notna(pct_val) else float("nan")
                    )
        mea_lookup: dict[tuple[str, str], tuple[float, float, str, float, str, float]] = {}
        if not mea.empty:
            for _, row in mea.iterrows():
                k = str(row.get("kinase", ""))
                contrast = str(row.get("contrast", ""))
                donor = contrast.replace("_vs_CTRLmean", "")
                lead = str(row.get("Leading substrates", ""))
                mea_lookup[(k, donor)] = (
                    float(row.get("NES")) if pd.notna(row.get("NES")) else float("nan"),
                    float(row.get("FDR")) if pd.notna(row.get("FDR")) else float("nan"),
                    lead,
                    float(row.get("ES")) if pd.notna(row.get("ES")) else float("nan"),
                    str(row.get("Subs fraction", "")),
                    float(row.get("p-value")) if pd.notna(row.get("p-value")) else float("nan"),
                )

        for k in nes.index.astype(str):
            kid = next_id
            next_id += 1
            cols["id"].append(kid)
            cols["name"].append(k)
            cols["gene_symbol"].append(gene_map.get(k, k))
            cols["residue_type"].append(residue)
            rrow = rec.loc[k] if k in rec.index else None

            def _gr(c, default=0):
                if rrow is None or c not in rrow.index:
                    return default
                v = rrow[c]
                return default if pd.isna(v) else v

            cols["n_donors_sig"].append(int(_gr("n_donors_sig", 0)))
            cols["n_donors_up"].append(int(_gr("n_donors_up", 0)))
            cols["n_donors_down"].append(int(_gr("n_donors_down", 0)))
            cols["n_donors_tested"].append(int(_gr("n_donors_tested", 0)))
            cols["median_nes"].append(_gr("median_nes", float("nan")))
            cols["median_nes_sig_only"].append(_gr("median_nes_sig_only", float("nan")))
            sea = seaad_lookup.get((k, residue))
            cols["sea_ad_lfc"].append(round(sea["lfc"], 4) if (sea and sea["lfc"] is not None) else None)
            cols["sea_ad_n_supertypes"].append(sea["n"] if sea else 0)
            cols["sea_ad_direction_agreement"].append(
                round(sea["agreement"], 3) if (sea and sea["agreement"] is not None) else None
            )

            for d in all_donors_axis:
                if d in nes.columns:
                    v_nes = nes.loc[k, d] if k in nes.index else float("nan")
                    v_fdr = fdr.loc[k, d] if k in fdr.index else float("nan")
                else:
                    v_nes = float("nan")
                    v_fdr = float("nan")
                cols[f"NES_{d}_vs_CTRLmean"].append(float(v_nes) if pd.notna(v_nes) else float("nan"))
                cols[f"FDR_{d}_vs_CTRLmean"].append(float(v_fdr) if pd.notna(v_fdr) else float("nan"))
                if (k, d) in mea_lookup:
                    n_, f_, lead, es_, subs_, p_ = mea_lookup[(k, d)]
                    rn_, rf_, rp_ = raw_lookup.get((k, d), (float("nan"), float("nan"), float("nan")))
                    motifs = ";".join(subs_lookup.get((k, d), []))
                    kl_pcts = ";".join(
                        ("" if not (v == v) else f"{v:.2f}")  # NaN check via self-cmp
                        for v in subs_pct_lookup.get((k, d), [])
                    )
                    perdonor_rows.append((kid, d, n_, f_, lead, es_, subs_, p_, rn_, rf_, rp_, motifs, kl_pcts))

    # Per-donor scalar index (one row per (kinase, donor) with an MEA record).
    # `leading_substrates` and `substrate_motifs` are sharded out to
    # edge_slices/human_perdonor/ and loaded on demand via
    # SliceCache.loadHumanPerdonorSubstrate.
    perdonor_index = {
        "kinase_id":   [r[0] for r in perdonor_rows],
        "donor":       [r[1] for r in perdonor_rows],
        "NES":         [round(r[2], 4) if pd.notna(r[2]) else None for r in perdonor_rows],
        "FDR":         [round(r[3], 4) if pd.notna(r[3]) else None for r in perdonor_rows],
        "ES":          [round(r[5], 4) if pd.notna(r[5]) else None for r in perdonor_rows],
        "subs_fraction": [r[6] for r in perdonor_rows],
        "p_value":     [round(r[7], 6) if pd.notna(r[7]) else None for r in perdonor_rows],
        "raw_NES":     [round(r[8], 4) if pd.notna(r[8]) else None for r in perdonor_rows],
        "raw_FDR":     [round(r[9], 4) if pd.notna(r[9]) else None for r in perdonor_rows],
        "raw_p_value": [round(r[10], 6) if pd.notna(r[10]) else None for r in perdonor_rows],
    }
    human_perdonor_substrate_slice_index = (
        _write_human_perdonor_substrate_slices(perdonor_rows)
    )

    # Global-shift diagnostics (track-tagged).
    shift_rows: list[dict] = []
    for t in tracks:
        if t["shift"].empty:
            continue
        for _, row in t["shift"].iterrows():
            contrast = str(row.get("contrast", ""))
            donor = contrast.replace("_vs_CTRLmean", "")
            shift_rows.append({
                "contrast": contrast,
                "donor": donor,
                "residue_type": t["residue"],
                "median_shift": float(row.get("median_shift")) if pd.notna(row.get("median_shift")) else None,
                "mean_before": float(row.get("mean_before")) if pd.notna(row.get("mean_before")) else None,
                "pct_pos_before": float(row.get("pct_pos_before")) if pd.notna(row.get("pct_pos_before")) else None,
                "pct_pos_after": float(row.get("pct_pos_after")) if pd.notna(row.get("pct_pos_after")) else None,
            })

    # Winsorization receipts: per-(donor, track) clipped sites.
    winsor_cols: dict[str, list] = {
        "donor": [], "residue_type": [], "site_id": [], "gene_symbol": [],
        "original_lfc": [], "clipped_lfc": [], "lower_bound": [], "upper_bound": [],
    }
    for t in tracks:
        if t["winsor"].empty:
            continue
        for _, row in t["winsor"].iterrows():
            contrast = str(row.get("contrast", ""))
            donor = contrast.replace("_vs_CTRLmean", "")
            winsor_cols["donor"].append(donor)
            winsor_cols["residue_type"].append(t["residue"])
            winsor_cols["site_id"].append(str(row.get("site_id", "")))
            winsor_cols["gene_symbol"].append(str(row.get("gene_symbol", "")))
            for k in ("original_lfc", "clipped_lfc", "lower_bound", "upper_bound"):
                v = row.get(k)
                winsor_cols[k].append(round(float(v), 4) if pd.notna(v) else None)

    # Site-level measurement-trace matrices (per-site stoichiometry + raw phospho).
    # Donor axis = union of donor + CTRL columns across tracks.
    sites_cols: dict[str, list] = {
        "site_id": [], "motif": [], "gene_symbol": [], "site_position": [],
        "residue_type": [],
    }
    donors_all: list[str] = []
    seen_all: set[str] = set()
    META_COLS = {"site_id", "protein_id", "gene_symbol", "site_position", "motif"}
    for t in tracks:
        for df in (t["stoich"], t["raw"]):
            if df is None or df.empty:
                continue
            for c in df.columns:
                if c in META_COLS:
                    continue
                if c not in seen_all:
                    seen_all.add(c)
                    donors_all.append(str(c))
    # Reorder donors_all so case (AD) donors land before CTRL — matches the
    # rendering axis order and keeps wide loaders that key off column
    # position stable.
    if ctrl_donors:
        ctrl_set_axis = set(ctrl_donors)
        donors_all = (
            [d for d in donors_all if d not in ctrl_set_axis]
            + [d for d in donors_all if d in ctrl_set_axis]
        )
    case_donors = [d for d in donors_all if d not in set(ctrl_donors)]

    stoich_by_site: list[list[float | None]] = []
    raw_by_site: list[list[float | None]] = []
    for t in tracks:
        residue = t["residue"]
        sdf = t["stoich"]
        rdf = t["raw"]
        if sdf is None or sdf.empty:
            continue
        # Index raw by site_id for quick join (raw may differ in coverage).
        raw_idx = rdf.set_index("site_id") if (rdf is not None and not rdf.empty) else None
        for _, srow in sdf.iterrows():
            sid = str(srow.get("site_id", ""))
            sites_cols["site_id"].append(sid)
            sites_cols["motif"].append(str(srow.get("motif", "")))
            sites_cols["gene_symbol"].append(str(srow.get("gene_symbol", "")))
            sites_cols["site_position"].append(str(srow.get("site_position", "")))
            sites_cols["residue_type"].append(residue)
            srow_vals: list[float | None] = []
            rrow_vals: list[float | None] = []
            for d in donors_all:
                v = srow.get(d) if d in srow.index else None
                srow_vals.append(round(float(v), 4) if (v is not None and pd.notna(v)) else None)
                if raw_idx is not None and sid in raw_idx.index and d in raw_idx.columns:
                    rv = raw_idx.loc[sid, d]
                    rrow_vals.append(round(float(rv), 4) if pd.notna(rv) else None)
                else:
                    rrow_vals.append(None)
            stoich_by_site.append(srow_vals)
            raw_by_site.append(rrow_vals)

    # CTRL recurrence sidecar (kinase × CTRL-donor sig counts). Always
    # AD-only on the kinase columns — the CTRL block here is a separate
    # diagnostic surface. Track-tagged.
    ctrl_rec_rows: list[dict] = []
    for t in tracks:
        residue = t["residue"]
        rec_ctrl = t.get("rec_ctrl")
        if rec_ctrl is None or rec_ctrl.empty:
            continue
        for _, row in rec_ctrl.iterrows():
            ctrl_rec_rows.append({
                "kinase": str(row.get("kinase", "")),
                "residue_type": residue,
                "n_donors_sig": int(row.get("n_donors_sig", 0) or 0),
                "n_donors_up": int(row.get("n_donors_up", 0) or 0),
                "n_donors_down": int(row.get("n_donors_down", 0) or 0),
                "n_donors_tested": int(row.get("n_donors_tested", 0) or 0),
                "median_nes": (round(float(row["median_nes"]), 4)
                               if pd.notna(row.get("median_nes")) else None),
            })

    # Cell-type specificity block (CR03). Populated when human_reference_expression.py
    # has been run and the specificity CSVs exist. Gracefully absent in phase-1 runs.
    celltype_specificity: dict | None = None
    if _HAS_HUMAN_CELLTYPE:
        try:
            celltype_specificity = build_celltype_specificity_payload()
        except Exception as exc:
            print(f"  human celltype_specificity: skipped ({exc})", flush=True)

    human_slice: dict = {
        "schema_version": 2,  # bumped: adds ctrl_donors, NES_<ctrl>_vs_CTRLmean cols
        "kinases": cols,
        "donors": donors,
        "ctrl_donors": ctrl_donors,
        "contrasts": contrasts,
        "perdonor_index": perdonor_index,
        "global_shift": shift_rows,
        "winsor": winsor_cols,
        "sites": sites_cols,
        "donors_all": donors_all,
        "case_donors": case_donors,
        "recurrence_ctrl": ctrl_rec_rows,
        "stoich_by_site": stoich_by_site,
        "raw_phospho_by_site": raw_by_site,
    }
    if celltype_specificity is not None:
        human_slice["celltype_specificity"] = celltype_specificity
    return human_slice, human_perdonor_substrate_slice_index


def _build_celltypes_slice(data: UnifiedData) -> dict:
    celltypes = data.edge_metadata["celltypes"]
    return {
        "id": list(range(len(celltypes))),
        "name": list(celltypes),
        "tissue_category": [RECEIVER_TO_TISSUE.get(c, "Other") for c in celltypes],
    }



def _build_subclass_breakdown(kid: dict[str, int]) -> dict:
    """Per-kinase subclass composition tooltips for verdict-table rows.

    For each (kinase, WMB class) where the class spans ≥2 WMB subclasses with
    detectable expression, returns the top-3 contributing subclasses ranked by
    mean log2 expression. Lets a viewer user see what subclass-level structure
    is collapsed behind a class-level call (e.g., "07 CTX-MGE GABA" → Pvalb +
    Sst + Chandelier).
    """
    sub_path = config.WMB_EXPRESSION_SUBCLASS_FILE
    map_path = config.WMB_SUBCLASS_TO_CLASS_FILE
    if not (os.path.exists(sub_path) and os.path.exists(map_path)):
        print(f"  subclass_breakdown: skipped (missing {sub_path} or {map_path})",
              flush=True)
        return {}
    sub = pd.read_csv(sub_path)
    sc2cls = pd.read_csv(map_path)
    sub = sub.merge(sc2cls, left_on="wmb_subclass", right_on="subclass", how="left")
    sub = sub[sub["class"].notna() & sub["kinase_id"].isin(kid)]
    # Keep only subclasses with detectable expression
    sub = sub[sub["mean_log2_expression"] > 0.5]
    if len(sub) == 0:
        return {}
    sub = sub.sort_values(["kinase_id", "class", "mean_log2_expression"],
                           ascending=[True, True, False])
    out: dict[str, dict[str, str]] = {}
    for (kin, cls), g in sub.groupby(["kinase_id", "class"], sort=False):
        if len(g) < 2:
            continue
        top = g.head(3)
        parts = [
            f"{r['wmb_subclass']} (mean={r['mean_log2_expression']:.2f}, "
            f"frac={r['fraction_cells_expressing']:.2f})"
            for _, r in top.iterrows()
        ]
        n_more = len(g) - len(top)
        text = "; ".join(parts)
        if n_more > 0:
            text += f"; +{n_more} more"
        out.setdefault(str(kid[kin]), {})[str(cls)] = text
    print(f"  subclass_breakdown: {len(out)} kinases × "
          f"{sum(len(v) for v in out.values())} (kinase,class) tooltips",
          flush=True)
    return out


_AGREEMENT_STATE_CODES = {
    "neither_sig": 0,
    "agree":       1,
    "mixed":       2,
    "disagree":    3,
    "bulk_only":   4,
    "decomp_only": 5,
}


def _build_agreement_index(
    mea: pd.DataFrame,
    decomp: pd.DataFrame,
    kid: dict,
    contrast_to_id: dict,
    fdr_thresh: float,
) -> dict:
    """Per-(kinase, contrast) agreement state between bulk MEA and per-cell decomp MEA.

    For each (kinase, contrast) where bulk and/or decomp data exist, classify:
      - agree:       bulk sig, ≥1 cell sig, all sig cells match bulk sign
      - mixed:       bulk sig, sig cells split (some match, some oppose)
      - disagree:    bulk sig, sig cells all oppose bulk sign
      - bulk_only:   bulk sig, no cell sig
      - decomp_only: bulk insig, ≥1 cell sig
      - neither_sig: bulk insig, no cell sig (NOT emitted; absence == this state)

    Also reports the top decomp cell (largest |NES| among that kinase×contrast's
    decomp rows) for the scatter plot.
    """
    if mea.empty or decomp.empty:
        return {"kinase_id": [], "contrast_id": [], "state": [],
                "bulk_nes": [], "bulk_fdr": [],
                "top_cell": [], "top_cell_nes": [], "top_cell_fdr": [],
                "n_cells_match": [], "n_cells_oppose": []}

    b = mea[mea["kinase"].isin(kid) & mea["contrast"].isin(contrast_to_id)].copy()
    b = b.rename(columns={"NES": "bulk_NES", "FDR": "bulk_FDR"})
    b["bulk_sig"] = b["bulk_FDR"] < fdr_thresh
    b["bulk_dir"] = np.sign(b["bulk_NES"])

    d = decomp[decomp["kinase"].isin(kid) & decomp["contrast"].isin(contrast_to_id)].copy()
    d["dec_sig"] = d["FDR"] < fdr_thresh
    d["dec_dir"] = np.sign(d["NES"])
    d["abs_nes"] = d["NES"].abs()

    # Outer join so bulk-only and decomp-only rows are kept.
    m = d.merge(b[["kinase", "contrast", "bulk_NES", "bulk_FDR", "bulk_sig", "bulk_dir"]],
                on=["kinase", "contrast"], how="outer")
    # Decomp side may be NaN for bulk-only (kinase, contrast); fill safe defaults.
    m["dec_sig"] = m["dec_sig"].fillna(False)
    m["dec_dir"] = m["dec_dir"].fillna(0)
    m["bulk_sig"] = m["bulk_sig"].fillna(False)
    m["bulk_dir"] = m["bulk_dir"].fillna(0)

    # For sign comparisons, only count cells where bulk_dir != 0.
    m["match"] = m["dec_sig"] & (m["bulk_dir"] != 0) & (m["dec_dir"] == m["bulk_dir"])
    m["oppose"] = m["dec_sig"] & (m["bulk_dir"] != 0) & (m["dec_dir"] == -m["bulk_dir"])

    # Top decomp cell by |NES| per (kinase, contrast).
    has_dec = m["wmb_class"].notna()
    top_idx = m[has_dec].groupby(["kinase", "contrast"])["abs_nes"].idxmax()
    top = m.loc[top_idx, ["kinase", "contrast", "wmb_class", "NES", "FDR"]].rename(
        columns={"wmb_class": "top_cell", "NES": "top_cell_nes", "FDR": "top_cell_fdr"}
    )

    agg = m.groupby(["kinase", "contrast"]).agg(
        bulk_NES=("bulk_NES", "first"),
        bulk_FDR=("bulk_FDR", "first"),
        bulk_sig=("bulk_sig", "first"),
        n_match=("match", "sum"),
        n_oppose=("oppose", "sum"),
        n_dec_sig=("dec_sig", "sum"),
    ).reset_index()
    agg = agg.merge(top, on=["kinase", "contrast"], how="left")

    def _state(r):
        if r.bulk_sig:
            if r.n_dec_sig == 0:
                return "bulk_only"
            if r.n_match > 0 and r.n_oppose == 0:
                return "agree"
            if r.n_match == 0 and r.n_oppose > 0:
                return "disagree"
            return "mixed"
        if r.n_dec_sig > 0:
            return "decomp_only"
        return "neither_sig"

    agg["state"] = agg.apply(_state, axis=1)
    # Drop neither_sig — absence in lookup table == that state.
    agg = agg[agg["state"] != "neither_sig"].reset_index(drop=True)

    print(f"  agreement_index: {len(agg):,} (kinase, contrast) cells "
          f"(states: {agg['state'].value_counts().to_dict()})", flush=True)

    state_codes = agg["state"].map(_AGREEMENT_STATE_CODES).astype("uint8").tolist()
    return {
        "kinase_id":   agg["kinase"].map(kid).astype("uint16").tolist(),
        "contrast_id": agg["contrast"].map(contrast_to_id).astype("uint8").tolist(),
        "state":       state_codes,
        "bulk_nes":    agg["bulk_NES"].astype(float).round(4).tolist(),
        "bulk_fdr":    agg["bulk_FDR"].astype(float).round(4).tolist(),
        "top_cell":    agg["top_cell"].fillna("").astype(str).tolist(),
        "top_cell_nes": agg["top_cell_nes"].astype(float).round(4).tolist(),
        "top_cell_fdr": agg["top_cell_fdr"].astype(float).round(4).tolist(),
        "n_cells_match":  agg["n_match"].astype(int).tolist(),
        "n_cells_oppose": agg["n_oppose"].astype(int).tolist(),
        "_state_codes": _AGREEMENT_STATE_CODES,
    }


def _norm_motif(s: str) -> str:
    return str(s or "").strip("_").upper()


def _write_decomp_ols_slices(kid: dict, contrast_to_id: dict) -> dict:
    """Per-kinase shard of per-cell-type OLS at substrate sites.

    Reads `outputs/reports/decomposition/{spine}/per_animal/site_level_ols.parquet`
    (per-cell_type × per-contrast × per-track site rows in Levy-t5
    vocabulary), filters each kinase to its substrate-set motifs (across
    all contrasts/tracks), and writes one parquet per kinase to
    `edge_slices/decomp_ols/{kid:03d}.parquet`.

    The drawer in the Attribution tab fetches one shard on demand and
    filters client-side by current contrast + cell_type to populate the
    substrate-level evidence table for the per-cell pseudo-deconv NES.
    """
    if not os.path.exists(DECOMP_OLS_PARQUET):
        print(f"  (warn) decomp OLS parquet missing: {DECOMP_OLS_PARQUET}; "
              f"skipping decomp_ols slice generation", flush=True)
        return {"slice_count": 0, "present_kinase_ids": [], "filename_template":
                "{kinase_id:03d}.parquet"}

    # Wipe-and-recreate so an aborted previous run can't leave partial
    # shards alongside the new ones (mismatched slice_count vs.
    # present_kinase_ids in index.json).
    shutil.rmtree(EDGE_SLICES_DECOMP_OLS_DIR, ignore_errors=True)
    os.makedirs(EDGE_SLICES_DECOMP_OLS_DIR, exist_ok=True)

    # Substrate sets — st + py tracks. Both files share schema.
    ss_paths = [
        os.path.join(AUDIT_SOURCES_DIR, "mea_substrate_sets.csv"),
        os.path.join(AUDIT_SOURCES_DIR, "mea_substrate_sets_pY.csv"),
    ]
    ss_frames = []
    for p in ss_paths:
        if os.path.exists(p):
            ss_frames.append(pd.read_csv(p, usecols=["kinase", "motif", "track"]))
    if not ss_frames:
        print(f"  (warn) substrate-set tables not found under {AUDIT_SOURCES_DIR}; "
              f"skipping decomp_ols slice generation", flush=True)
        return {"slice_count": 0, "present_kinase_ids": [], "filename_template":
                "{kinase_id:03d}.parquet"}
    ss = pd.concat(ss_frames, ignore_index=True)
    ss["motif_norm"] = ss["motif"].map(_norm_motif)
    ss = ss[ss["kinase"].isin(kid)]
    # kinase -> set of (motif_norm, track) substrate keys
    kinase_subs: dict[str, set] = {}
    for k, g in ss.groupby("kinase"):
        kinase_subs[k] = set(zip(g["motif_norm"], g["track"]))
    print(f"  decomp_ols: {len(kinase_subs)} kinases with substrate sets", flush=True)

    print(f"  decomp_ols: loading {DECOMP_OLS_PARQUET} "
          f"({os.path.getsize(DECOMP_OLS_PARQUET) / 1e6:.1f} MB)", flush=True)
    cols = ["site_id", "gene_symbol", "motif", "cell_type",
            "contrast", "lfc", "se", "pval", "track"]
    pcdf = pq.read_table(DECOMP_OLS_PARQUET, columns=cols).to_pandas()
    pcdf = pcdf[pcdf["contrast"].isin(contrast_to_id)].copy()
    pcdf["motif_norm"] = pcdf["motif"].astype(str).map(_norm_motif)
    pcdf["contrast_id"] = pcdf["contrast"].map(contrast_to_id).astype("uint8")
    pcdf = pcdf.drop(columns=["contrast"])
    print(f"  decomp_ols: {len(pcdf):,} per-cell rows after contrast filter", flush=True)

    # Index by (motif_norm, track) for fast per-kinase slicing.
    pc_index = pcdf.set_index(["motif_norm", "track"], drop=False).sort_index()

    template = "{kinase_id:03d}.parquet"
    present = []
    total_rows = 0
    for k, kid_int in kid.items():
        keys = kinase_subs.get(k)
        if not keys:
            continue
        # Build a small DataFrame of (motif_norm, track) selectors and join.
        sel_keys = list(keys)
        try:
            sub = pc_index.loc[sel_keys]
        except KeyError:
            sub = pc_index.loc[pc_index.index.intersection(sel_keys)]
        if isinstance(sub, pd.Series):
            continue
        if sub.empty:
            continue
        out = sub.reset_index(drop=True)[
            ["contrast_id", "cell_type", "site_id", "gene_symbol",
             "motif", "lfc", "se", "pval", "track"]
        ].copy()
        out["lfc"] = out["lfc"].astype("float32")
        out["se"] = out["se"].astype("float32")
        out["pval"] = out["pval"].astype("float32")
        path = os.path.join(EDGE_SLICES_DECOMP_OLS_DIR,
                            template.format(kinase_id=int(kid_int)))
        pq.write_table(pa.Table.from_pandas(out, preserve_index=False), path,
                       compression="zstd")
        present.append(int(kid_int))
        total_rows += len(out)

    present.sort()
    index = {
        "schema_version": SCHEMA_VERSION,
        "slice_count": len(present),
        "present_kinase_ids": present,
        "filename_template": template,
        "n_total_rows": total_rows,
    }
    with open(os.path.join(EDGE_SLICES_DECOMP_OLS_DIR, "index.json"), "w") as f:
        json.dump(index, f)
    print(f"  decomp_ols: wrote {len(present)} shards "
          f"({total_rows:,} total rows)", flush=True)
    return index


def _write_song_concordance_slices(genes_of_interest: set[str]) -> dict:
    """Per-gene shards of `song_concordance.csv`.

    The full file is ~210 MB; the Attribution drawer only ever filters to a
    single gene, so the JS fetches one shard on demand instead.
    """
    src = config.SONG_CONCORDANCE_FILE
    if not os.path.exists(src):
        print(f"  (warn) song_concordance source missing: {src}; skipping",
              flush=True)
        return {"slice_count": 0, "present_genes": [],
                "filename_template": "{gene}.parquet"}

    shutil.rmtree(EDGE_SLICES_SONG_CONCORDANCE_DIR, ignore_errors=True)
    os.makedirs(EDGE_SLICES_SONG_CONCORDANCE_DIR, exist_ok=True)

    print(f"  song_concordance: loading {src} "
          f"({os.path.getsize(src) / 1e6:.1f} MB)", flush=True)
    cols = ["gene_symbol", "cell_type", "contrast",
            "song_lfc", "song_se", "song_pval", "song_fdr", "n_animals"]
    df = pd.read_csv(src, usecols=lambda c: c in cols)
    df["gene_upper"] = df["gene_symbol"].astype(str).str.upper()
    gset = {str(g).upper() for g in genes_of_interest if g}
    if gset:
        df = df[df["gene_upper"].isin(gset)]
    print(f"  song_concordance: {len(df):,} rows after gene filter "
          f"({df['gene_upper'].nunique()} genes)", flush=True)

    float_cols = [c for c in ("song_lfc", "song_se", "song_pval", "song_fdr")
                  if c in df.columns]
    has_n_animals = "n_animals" in df.columns
    # Skip names that would need URL escaping on the client.
    def _safe(g: str) -> bool:
        return bool(g) and all(c.isalnum() or c in ("-", "_") for c in g)

    present = []
    total_rows = 0
    template = "{gene}.parquet"
    for gene_upper, g in df.groupby("gene_upper", sort=False):
        if not _safe(gene_upper):
            continue
        out = g.drop(columns=["gene_upper"])
        for c in float_cols:
            out[c] = out[c].astype("float32")
        if has_n_animals:
            out["n_animals"] = out["n_animals"].fillna(0).astype("int16")
        path = os.path.join(EDGE_SLICES_SONG_CONCORDANCE_DIR,
                            template.format(gene=gene_upper))
        pq.write_table(pa.Table.from_pandas(out, preserve_index=False), path,
                       compression="zstd")
        present.append(gene_upper)
        total_rows += len(out)

    present.sort()
    index = {
        "schema_version": SCHEMA_VERSION,
        "slice_count": len(present),
        "present_genes": present,
        "filename_template": template,
        "n_total_rows": total_rows,
    }
    with open(os.path.join(EDGE_SLICES_SONG_CONCORDANCE_DIR, "index.json"), "w") as f:
        json.dump(index, f)
    print(f"  song_concordance: wrote {len(present)} shards "
          f"({total_rows:,} total rows)", flush=True)
    return index


_INCYTR_CONTRASTS = tuple(config.CONTRAST_COEFS.keys())

# ---------------------------------------------------------------------------
# CR-04: trajectory_index + recur_index computation
# ---------------------------------------------------------------------------
# Trajectory is a property of the path's raw PDS sign vector — NOT of any
# significance gate. A (path, disease) earns a label only when the Incytr
# pipeline produced a row at all three timepoints (2/4/6 mo); incomplete
# paths get no label (rendered as "—" in the viewer). Sign chars are 'u'
# (PDS > 0) or 'd' (PDS < 0) — no 'f', no hardcoded flat threshold. The
# viewer's pvalue/|PDS| sliders filter visible rows but do not redefine
# the trajectory.
_TRAJ_TIMEPOINTS = ("2mo", "4mo", "6mo")


# Trajectory label vocabulary — non-exclusive: a (path, disease) tuple can
# carry multiple labels (e.g. "always-up" + "monotonic-up" if PDS is
# positive and strictly increasing). Stored as a semicolon-joined string in
# the shard so a single column carries the set.
_SIGN_VEC_LABELS = (
    "always-up",        # uuu — every PDS > 0
    "always-down",      # ddd — every PDS < 0
    "monotonic-up",     # PDS[2] < PDS[4] < PDS[6] (strictly)
    "monotonic-down",   # PDS[2] > PDS[4] > PDS[6] (strictly)
    "mixed",            # sign changes across timepoints
)


def _annotate_trajectory_columns(
    df: "pd.DataFrame",
    source_label: str = "factorial",
) -> "tuple[pd.DataFrame, dict, dict]":
    """Add traj_labels and sign_vec columns to the long-form shard DataFrame.

    Fully vectorised (no Python-level row loops) — handles 10M+ rows in a
    few seconds via pandas pivot + string ops.

    Returns the annotated DataFrame plus two small payload-level summaries:
      - recur_index: { path_string → [disease, …] } — diseases with ≥1 non-flat
        timepoint. path_string = sender||receiver||Path. Small enough to inline.
      - traj_summary: { label → count } aggregate across all paths.

    traj_labels and sign_vec are added as columns to every shard row.
    The JS reads them directly from loaded shards (no separate index needed).
    The full per-path index is NOT inlined in the main payload to keep it small.

    Phase-1 / mock note
    -------------------
    When ``source_label == "factorial"``, the contrasts come from the
    factorial OLS cache (App_2mo … ApTt_6mo). The classification logic is
    identical to pair-mode data. Swap point for phase 2: just change which
    ``df`` is passed to this function. No other code changes needed.
    """
    # Build path string and parse disease / timepoint from contrast.
    df = df.copy()
    df["_path_str"] = (
        df["sender"].astype(str) + "||"
        + df["receiver"].astype(str) + "||"
        + df["Path"].astype(str)
    )
    split = df["contrast"].str.split("_", n=1, expand=True)
    df["_disease"] = split[0].fillna("")
    df["_timepoint"] = split[1].fillna("")

    if df is None or df.empty:
        df["traj_labels"] = ""
        df["sign_vec"] = ""
        return df, {}, {}

    # ---- 1. Per-row sign char + raw PDS (no flat threshold) --------------
    pds_col = df["PDS"].astype(float)
    sign_ser = pd.Series("", index=df.index, dtype="str")
    sign_ser.loc[pds_col > 0] = "u"
    sign_ser.loc[pds_col < 0] = "d"
    df["_sign"] = sign_ser
    df["_pds"]  = pds_col

    # ---- 2. Pivot: (path, disease) × timepoint → sign & PDS --------------
    valid_tp = set(_TRAJ_TIMEPOINTS)
    valid_dis = {"App", "Tau", "ApTt"}
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
    for tp in _TRAJ_TIMEPOINTS:
        if tp not in sign_pivot.columns: sign_pivot[tp] = pd.NA
        if tp not in pds_pivot.columns:  pds_pivot[tp]  = pd.NA
    sign_pivot = sign_pivot[list(_TRAJ_TIMEPOINTS)]
    pds_pivot  = pds_pivot[list(_TRAJ_TIMEPOINTS)]
    complete_mask = sign_pivot.notna().all(axis=1) & pds_pivot.notna().all(axis=1)
    sign_pivot = sign_pivot.loc[complete_mask]
    pds_pivot  = pds_pivot.loc[complete_mask]

    if sign_pivot.empty:
        df["traj_labels"] = ""
        df["sign_vec"] = ""
        return df, {}, {}

    # ---- 3. Vectorised label derivation (non-exclusive) ------------------
    s2, s4, s6 = sign_pivot["2mo"], sign_pivot["4mo"], sign_pivot["6mo"]
    v2, v4, v6 = pds_pivot["2mo"], pds_pivot["4mo"], pds_pivot["6mo"]
    out = pd.DataFrame(index=sign_pivot.index)
    out["sign_vec"]    = s2 + s4 + s6
    out["always_up"]   = (s2 == "u") & (s4 == "u") & (s6 == "u")
    out["always_down"] = (s2 == "d") & (s4 == "d") & (s6 == "d")
    out["monotonic_up"]   = (v2 < v4) & (v4 < v6)
    out["monotonic_down"] = (v2 > v4) & (v4 > v6)
    # "mixed" = sign changes (sign_vec uses both 'u' and 'd').
    out["mixed"] = ~(out["always_up"] | out["always_down"])

    def _join_labels(row):
        names = []
        if row["always_up"]:       names.append("always-up")
        if row["always_down"]:     names.append("always-down")
        if row["monotonic_up"]:    names.append("monotonic-up")
        if row["monotonic_down"]:  names.append("monotonic-down")
        if row["mixed"]:           names.append("mixed")
        return ";".join(names)

    out["traj_labels"] = out.apply(_join_labels, axis=1)

    # ---- 4. Back-join onto every shard row -------------------------------
    traj_map = out[["sign_vec", "traj_labels"]].reset_index()
    df = df.merge(traj_map, on=["_path_str", "_disease"], how="left")
    df["traj_labels"] = df["traj_labels"].fillna("")
    df["sign_vec"]    = df["sign_vec"].fillna("")

    # ---- 5. recur_index — path → list of diseases with complete trajectory
    sig_pivot = out.reset_index()[["_path_str", "_disease"]]
    recur_index: dict = {}
    if len(sig_pivot):
        recur_series = sig_pivot.groupby("_path_str", sort=False)["_disease"].agg(list)
        recur_index = {str(pid): dis for pid, dis in recur_series.items()}

    # ---- 6. Trajectory summary (per-label counts across (path, disease)) -
    traj_summary: dict = {lbl: int(out[lbl.replace("-", "_")].sum())
                          for lbl in _SIGN_VEC_LABELS}

    n_paths = len(out.index.get_level_values("_path_str").unique())
    print(f"  trajectory ({source_label}): {n_paths:,} unique paths annotated; "
          f"{len(recur_index):,} recur in ≥1 disease; "
          f"label dist = {dict(sorted(traj_summary.items()))}", flush=True)

    # Drop temp columns.
    df.drop(columns=["_path_str", "_disease", "_timepoint", "_sign", "_pds"],
            inplace=True, errors="ignore")
    return df, recur_index, traj_summary


# Fixed pvalue grid for the Temporal v2 pathway layer. User-entered pvalue is
# snapped down to the nearest threshold; counts are pre-aggregated per
# (contrast, sign(PDS), pvalue_threshold, abs_pds_threshold) to keep the
# payload tiny (9 × 3 × 8 × 8 × 4B ≈ 7 KB).
_INCYTR_PATHWAY_PVALUES = (0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0)
# |PDS| ≥ threshold (so 0 = no effect-size gate). PDS is the composite
# multimodel Pathway Disturbance Score — OLS-derived, consistent with the
# rest of the viewer. 0.01 is the "real signal" floor; 0.5 is "very strong".
# Replaced the legacy sigprob_max filter (outlier-driven mean ratio) on
# 2026-05-12.
_INCYTR_PATHWAY_ABS_PDS = (0.0, 0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0)

# Pathway scoring stack — surfaced as table columns in the pathway tab.
# PDS is already in the shard; these are net-new. log2FC and sigprob_max
# were retired 2026-05-12 (mean-driven, inconsistent with the OLS pipeline).
_INCYTR_SCORE_COLS = ("TPDS", "PPDS", "PhPDS_ps", "PhPDS_py", "SiK_score")

# Per-node log2 fold-change columns. Only the four genuine log2FC metrics are
# shipped — the Hill-shrunk aFC variants emitted by Incytr are NOT log2 fold-
# changes (β × adjustment in (0,1]) and were dropped from the viewer payload
# to avoid mixed-units confusion. 4 nodes × 4 metrics = 16 columns.
_INCYTR_FC_NODES = ("Ligand", "Receptor", "EM", "Target")
_INCYTR_FC_METRICS = (
    "sclog2FC",
    "pr_log2FC",
    "ps_log2FC",
    "py_log2FC",
)
_INCYTR_FC_COLS = tuple(
    f"{node}_{metric}" for node in _INCYTR_FC_NODES for metric in _INCYTR_FC_METRICS
)

# Per-node evidence-source labels — each of Ligand / Receptor / EM / Target is
# tagged with the seed list that admitted it: "DEG" (single-cell DEG) or "prG"
# (proteomics-significant gene). Source columns are dotted in the upstream
# parquet (`<Node>.label`); we alias them to `<Node>_label` on the way out so
# downstream JS can use plain dot-access.
_INCYTR_LABEL_NODES = _INCYTR_FC_NODES
_INCYTR_LABEL_VOCAB = ("DEG", "prG")
_INCYTR_LABEL_SRC = tuple(f"{n}.label" for n in _INCYTR_LABEL_NODES)
_INCYTR_LABEL_COLS = tuple(f"{n}_label" for n in _INCYTR_LABEL_NODES)

def _incytr_sanitize(name: str) -> str:
    """Match the upstream sanitize in alz/integration/load.R:sanitize_celltype."""
    return name.replace("/", "-").replace(" ", "_")


_INCYTR_PAIR_GENO_NORMALIZE = {
    "AppP": "App", "App": "App",
    "Ttau": "Tau", "Tau": "Tau", "TtauP": "Tau",
    "ApTt": "ApTt", "AppPTtau": "ApTt", "AppTtau": "ApTt",
}


def _pair_mode_contrast_from_filename(fname: str) -> str | None:
    """`ma_2mo_AppP_ma_2mo_WTyp_incytr_output.parquet` → 'App_2mo'."""
    m = re.match(
        r"ma_(\d+)mo_([A-Za-z]+)_ma_\1mo_WTyp_incytr_output\.parquet$", fname
    )
    if not m:
        return None
    age, geno_token = m.group(1), m.group(2)
    geno = _INCYTR_PAIR_GENO_NORMALIZE.get(geno_token)
    if geno is None:
        print(f"    (warn) unknown geno token '{geno_token}' in {fname}; skipping",
              flush=True)
        return None
    return f"{geno}_{age}mo"


def _write_incytr_pair_pathways() -> dict | None:
    """Shard the pair-mode Incytr output by (sender, receiver) — unfiltered.

    Reads `outputs/reports/incytr_pair_mode/wide/*.parquet` (one file per
    contrast, Levy-t5 spine, 31² = 961 sender×receiver pairs) and emits one
    parquet per pair under `edge_slices/incytr_pathways/` plus the
    `incytr_pathways` payload block. No build-time significance gate; the
    UI thresholds live.
    """
    import duckdb

    input_dir = os.environ.get(
        "INCYTR_PAIR_MODE_INPUT_DIR",
        os.path.join(config.REPO_ROOT, "outputs", "reports", "incytr_pair_mode", "wide"),
    )
    if not os.path.isdir(input_dir):
        print(f"  (warn) pair-mode input dir not found: {input_dir}; "
              f"skipping incytr_pathways", flush=True)
        return None

    parquet_files = sorted(glob.glob(os.path.join(input_dir, "*_incytr_output.parquet")))
    if not parquet_files:
        print(f"  (warn) no pair-mode parquets in {input_dir}; "
              f"skipping incytr_pathways", flush=True)
        return None

    # Pair each file with its contrast label. Drop files we can't parse.
    file_to_contrast: list[tuple[str, str]] = []
    for fpath in parquet_files:
        contrast = _pair_mode_contrast_from_filename(os.path.basename(fpath))
        if contrast is not None:
            file_to_contrast.append((fpath, contrast))
    if not file_to_contrast:
        print(f"  (warn) no parseable pair-mode parquets in {input_dir}; "
              f"skipping incytr_pathways", flush=True)
        return None

    # Contrast list: subset of the canonical 9, in canonical order.
    present_contrasts = [c for c in _INCYTR_CONTRASTS
                         if c in {c2 for _, c2 in file_to_contrast}]
    contrast_to_idx = {c: i for i, c in enumerate(present_contrasts)}
    print(f"  incytr_pair_pathways: {len(file_to_contrast)} parquet(s); "
          f"contrasts = {present_contrasts}", flush=True)

    con = duckdb.connect()
    con.execute("PRAGMA threads=8; PRAGMA memory_limit='12GB';")
    _configure_duckdb_tempdir(con)

    # Detect optional path-level direction-flag columns once.
    sample_schema = pq.read_schema(file_to_contrast[0][0])
    src_cols = {f.name for f in sample_schema}
    dir_flag_cols = [c for c in ("pr_up", "pr_down", "ps_up", "ps_down",
                                  "py_up", "py_down")
                     if c in src_cols]
    # aFC is byte-identical to log2FC in every row (verified across all 9
    # contrasts) — drop the duplicate to save ~17% of shard storage. log2FC
    # is the canonical path-level fold-change column.
    extra_path_cols = [c for c in ("log2FC",) if c in src_cols]
    if not dir_flag_cols:
        print(f"    (warn) no direction-flag columns; downstream UI badges "
              f"will be empty", flush=True)

    # Build per-file SELECTs unioning into a single src table. Use disease-arm
    # p-value (`p_value_ma_<age>mo_<geno>`) as the `pvalue` proxy — pair-mode
    # has no factorial-OLS pooled p-value.
    selects = []
    for fpath, contrast in file_to_contrast:
        sch = pq.read_schema(fpath)
        names = {f.name for f in sch}
        pcol_disease = None
        for n in names:
            if n.startswith("p_value_") and not n.endswith("_WTyp"):
                pcol_disease = n
                break
        if pcol_disease is None:
            print(f"    (warn) no disease-arm p_value col in "
                  f"{os.path.basename(fpath)}; using NULL", flush=True)
            pcol_clause = "CAST(NULL AS DOUBLE)"
        else:
            pcol_clause = f'CAST("{pcol_disease}" AS DOUBLE)'

        dir_clauses = ",\n          ".join(
            f"CAST({c} AS DOUBLE) AS {c}" for c in dir_flag_cols
        )
        path_clauses = ",\n          ".join(
            f"CAST({c} AS DOUBLE) AS {c}" for c in extra_path_cols
        )
        score_clauses = ",\n          ".join(
            f"CAST({c} AS DOUBLE) AS {c}" for c in _INCYTR_SCORE_COLS
            if c in names
        )
        # Stable column order; missing scores are NULL.
        missing_scores = [c for c in _INCYTR_SCORE_COLS if c not in names]
        missing_score_clauses = ",\n          ".join(
            f"CAST(NULL AS DOUBLE) AS {c}" for c in missing_scores
        )
        # Per-node FC columns (Ligand_sclog2FC, …): NULL-fill if absent in
        # this driver's output. Names match _INCYTR_FC_COLS exactly.
        fc_clauses = ",\n          ".join(
            (f'CAST("{c}" AS DOUBLE) AS "{c}"' if c in names
             else f'CAST(NULL AS DOUBLE) AS "{c}"')
            for c in _INCYTR_FC_COLS
        )
        # Per-node evidence labels: source columns use dot notation
        # ("Ligand.label"), aliased to underscore ("Ligand_label") in the
        # output shard. NULL-fill if upstream driver omitted them.
        label_clauses = ",\n          ".join(
            (f'CAST("{src}" AS VARCHAR) AS "{dst}"' if src in names
             else f'CAST(NULL AS VARCHAR) AS "{dst}"')
            for src, dst in zip(_INCYTR_LABEL_SRC, _INCYTR_LABEL_COLS)
        )
        clauses = [score_clauses, missing_score_clauses,
                   dir_clauses, path_clauses, fc_clauses, label_clauses]
        extra_select = ",\n          ".join(c for c in clauses if c)

        # Pre-filter: keep rows with |PDS| >= 0.01 (gating signal — pvalue is
        # untrustworthy in pair-mode) AND drop rows with clearly-no-signal
        # disease-arm pvalue > 0.75 (coarse no-signal cut, not a ranking gate).
        pval_drop_clause = (
            f"AND ({pcol_clause} IS NULL OR {pcol_clause} <= 0.75)"
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
        WHERE COALESCE(ABS(CAST(PDS AS DOUBLE)), 0) >= 0.30
          {pval_drop_clause}
        """)
    union_sql = "\nUNION ALL\n".join(selects)
    # VIEW (not TEMP TABLE) — DuckDB re-reads parquet on demand for each query,
    # avoiding a 57M-row in-memory materialization that OOMs the box. The
    # per-file WHERE pre-filter (|PDS| >= 0.30 AND disease-arm pvalue <= 0.75)
    # cuts ~22% of rows that carry no actionable signal. Pair-mode pvalue is
    # untrustworthy for ranking, but pvalue > 0.75 is a coarse no-signal cut.
    con.execute(f"CREATE VIEW src AS {union_sql}")
    n_src = con.execute("SELECT COUNT(*) FROM src").fetchone()[0]
    print(f"  incytr_pair_pathways: loaded {n_src:,} rows across "
          f"{len(file_to_contrast)} contrast(s) (pre-filtered |PDS|>=0.30 AND p<=0.75)",
          flush=True)

    # Sender/receiver canonical lists from the data itself (Levy-t5, already
    # sanitized — no display↔sanitized indirection).
    senders_canonical = sorted({r[0] for r in con.execute(
        "SELECT DISTINCT sender FROM src").fetchall()})
    receivers_canonical = sorted({r[0] for r in con.execute(
        "SELECT DISTINCT receiver FROM src").fetchall()})
    sender_to_idx = {s: i for i, s in enumerate(senders_canonical)}
    receiver_to_idx = {r: i for i, r in enumerate(receivers_canonical)}
    n_s, n_r, n_c = len(senders_canonical), len(receivers_canonical), len(present_contrasts)
    print(f"    senders={n_s}, receivers={n_r}, contrasts={n_c} "
          f"(pair count={n_s * n_r})", flush=True)

    # --- heatmap_counts cube (same shape contract as factorial) ----------
    n_thr = len(_INCYTR_PATHWAY_PVALUES)
    n_ap = len(_INCYTR_PATHWAY_ABS_PDS)
    hm_thr_clauses_list = []
    for ip, tp in enumerate(_INCYTR_PATHWAY_PVALUES):
        for iap, tap in enumerate(_INCYTR_PATHWAY_ABS_PDS):
            hm_thr_clauses_list.append(
                f"COUNT(*) FILTER (WHERE pvalue < {tp} "
                f"AND COALESCE(ABS(PDS), 0) >= {tap}) AS c_{ip}_{iap}"
            )
    hm_thr_clauses = ", ".join(hm_thr_clauses_list)
    hm_rows = con.execute(f"""
        SELECT sender, receiver, contrast, {hm_thr_clauses}
        FROM src
        WHERE pvalue IS NOT NULL
        GROUP BY sender, receiver, contrast
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
    p_005_idx = _INCYTR_PATHWAY_PVALUES.index(0.05)
    ap_zero_idx = _INCYTR_PATHWAY_ABS_PDS.index(0.0)
    ap_001_idx = _INCYTR_PATHWAY_ABS_PDS.index(0.01)
    print(f"    heatmap_counts: total at pvalue<0.05 & |PDS|>=0    = "
          f"{int(totals[p_005_idx, ap_zero_idx]):>9,}; "
          f"at pvalue<0.05 & |PDS|>=0.01 = {int(totals[p_005_idx, ap_001_idx]):>9,}",
          flush=True)

    # --- pathway_counts cube ---------------------------------------------
    thr_clauses = []
    for ip, tp in enumerate(_INCYTR_PATHWAY_PVALUES):
        for iap, tap in enumerate(_INCYTR_PATHWAY_ABS_PDS):
            thr_clauses.append(
                f"COUNT(*) FILTER (WHERE pvalue < {tp} "
                f"AND COALESCE(ABS(PDS), 0) >= {tap}) AS c_{ip}_{iap}"
            )
    pathway_rows = con.execute(f"""
        SELECT contrast,
               CASE WHEN PDS > 0 THEN 2
                    WHEN PDS < 0 THEN 0
                    ELSE 1 END AS s,
               {", ".join(thr_clauses)}
        FROM src
        WHERE pvalue IS NOT NULL
        GROUP BY contrast, s
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

    # --- shard the long table per (sender, receiver) ---------------------
    # Wipe-and-recreate to avoid mixing old/new shards on an aborted run.
    shutil.rmtree(EDGE_SLICES_INCYTR_PATHWAYS_DIR, ignore_errors=True)
    os.makedirs(EDGE_SLICES_INCYTR_PATHWAYS_DIR, exist_ok=True)

    # Materialize once; group + write. Pair-mode now supplies per-node FC
    # (driver Cal_scFC + post-hoc reconstruct_node_fc.R) and labels (driver
    # DEG/prG assignment); fall back to NULL if the source parquet was
    # produced by an older driver.
    src_cols_pair = set(
        con.execute("DESCRIBE SELECT * FROM src LIMIT 0").fetchdf()["column_name"]
    )
    fc_select = [
        f'"{c}"' if c in src_cols_pair
        else f'CAST(NULL AS DOUBLE) AS "{c}"'
        for c in _INCYTR_FC_COLS
    ]
    # The union SELECT above already aliased `<Node>.label` → `<Node>_label`,
    # so the view exposes the underscore form. Reference the dst name.
    label_select = [
        f'"{dst}"' if dst in src_cols_pair
        else f'CAST(NULL AS VARCHAR) AS "{dst}"'
        for dst in _INCYTR_LABEL_COLS
    ]
    # Per-pair shard columns: sender/receiver are constant within a shard so
    # they're omitted (matches the factorial path's contract). `Path` is also
    # omitted — it's just `Ligand|Receptor|EM|Target` concatenated and is
    # reconstructed client-side in incytr_pathways.js (~10% disk savings).
    shard_select_cols = (
        ["Ligand", "Receptor", "EM", "Target",
         "contrast", "pvalue", "PDS"]
        + list(_INCYTR_SCORE_COLS)
        + dir_flag_cols
        + extra_path_cols
        + fc_select
        + label_select
    )
    # pvalue stays float32 to preserve dynamic range for small values
    # (float16's smallest normal is ~6e-8). All other floats compress to
    # float16 — ~3 decimal digits of precision, matching the UI display.
    float32_cols = ["pvalue"]
    float16_cols = (["PDS"]
                    + list(_INCYTR_SCORE_COLS)
                    + list(_INCYTR_FC_COLS)
                    + dir_flag_cols + extra_path_cols)
    float_cols = float32_cols + float16_cols  # for BSS encoding selection

    present_pairs: list[list[str]] = []
    pair_row_counts: dict[str, int] = {}
    total_rows = 0
    max_shard_bytes = 0
    max_shard_name = ""
    # CR-04: trajectory annotation runs per-pair inside _flush. Per-pair
    # path_strings (sender||receiver||Path) are globally unique, so per-pair
    # recur dicts and traj label counts compose by simple union / sum.
    recur_index: dict = {}
    traj_summary: dict = {}

    # Single streaming pass over the union view, ordered by (sender, receiver)
    # so each pair's rows arrive contiguously. Previous implementation issued
    # one DuckDB query per (sender, receiver) — 961 queries × 9 parquet rescans
    # each = ~8.6k parquet reads. This pass reads each parquet once.
    def _flush(key: tuple[str, str], frames: list[pd.DataFrame]) -> None:
        nonlocal total_rows, max_shard_bytes, max_shard_name
        if not frames:
            return
        sub = pd.concat(frames, ignore_index=True, copy=False)
        for col in _INCYTR_LABEL_COLS:
            if col in sub.columns:
                sub[col] = pd.Categorical(sub[col], categories=_INCYTR_LABEL_VOCAB)
        # CR-04: annotate traj_labels + sign_vec. Add sender/receiver/Path as
        # temp columns (constant within a shard) so the annotation function
        # can key on (sender, receiver, Path); drop sender/receiver after
        # annotation (Path is rebuilt client-side, so also dropped).
        s_key, r_key = key
        sub["sender"] = s_key
        sub["receiver"] = r_key
        sub["Path"] = (sub["Ligand"].astype(str) + "|"
                       + sub["Receptor"].astype(str) + "|"
                       + sub["EM"].astype(str) + "|"
                       + sub["Target"].astype(str))
        sub, pair_recur, pair_traj = _annotate_trajectory_columns(
            sub, source_label="pair_mode",
        )
        recur_index.update(pair_recur)
        for label, count in pair_traj.items():
            traj_summary[label] = traj_summary.get(label, 0) + int(count)
        sub = sub.drop(columns=["sender", "receiver", "Path"])
        # Float dtype compression runs after annotation so traj_labels
        # (string) and sign_vec (str) are unaffected.
        for col in float32_cols:
            if col in sub.columns:
                sub[col] = sub[col].astype("float32")
        for col in float16_cols:
            if col in sub.columns:
                sub[col] = sub[col].astype("float16")
        # Sort by (Ligand, Receptor, EM, Target, contrast) so the path-string
        # columns form long RLE runs in the parquet dictionary encoding.
        # 4,130 distinct paths × 9 contrasts repeated → 5-10× compression on
        # the string columns vs the previous (contrast, pvalue) sort, which
        # scattered identical paths across the file.
        path_sort_cols = [c for c in ("Ligand", "Receptor", "EM", "Target", "contrast")
                          if c in sub.columns]
        if path_sort_cols:
            sub = sub.sort_values(path_sort_cols, kind="stable",
                                  na_position="last").reset_index(drop=True)
        s, r = key
        fname = f"{_incytr_sanitize(s)}__{_incytr_sanitize(r)}.parquet"
        path = os.path.join(EDGE_SLICES_INCYTR_PATHWAYS_DIR, fname)
        # byte_stream_split on float columns improves zstd compression of
        # float bit-patterns by ~15-25% (parquet docs §encoding). pyarrow
        # refuses column_encoding when use_dictionary=True, so pass
        # use_dictionary as an explicit list of non-float columns (which keeps
        # RLE_DICTIONARY on string/categorical columns) and BYTE_STREAM_SPLIT
        # on the float columns via column_encoding.
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

    # Per-sender streaming: 31 queries (one per sender) instead of one global
    # sort over 42M rows (OOMs DuckDB) or one query per (sender, receiver)
    # pair (961 queries × 9 parquet rescans). Each sender query is ~1.4M rows,
    # well within memory; sort is local to one sender.
    stream_cols = ["receiver"] + shard_select_cols
    for s in senders_canonical:
        df = con.execute(
            f"""SELECT {', '.join(stream_cols)}
                FROM src
                WHERE sender = ?
                ORDER BY receiver, contrast, pvalue NULLS LAST""",
            [s],
        ).fetchdf()
        if df.empty:
            continue
        receivers = df["receiver"].tolist()
        # Run-length partition by receiver.
        starts = [0]
        for i in range(1, len(receivers)):
            if receivers[i] != receivers[i - 1]:
                starts.append(i)
        starts.append(len(receivers))
        for j in range(len(starts) - 1):
            a, b = starts[j], starts[j + 1]
            r = receivers[a]
            run = df.iloc[a:b].drop(columns=["receiver"])
            _flush((s, r), [run])
        del df
    con.close()

    index = {
        "schema_version": SCHEMA_VERSION,
        "filename_template": "{sender}__{receiver}.parquet",
        "sanitize_rule": "replace('/', '-'); replace(' ', '_')",
        "present": sorted(present_pairs),
        "n_total_rows": total_rows,
        "pair_row_counts": pair_row_counts,
    }
    with open(os.path.join(EDGE_SLICES_INCYTR_PATHWAYS_DIR, "index.json"), "w") as f:
        json.dump(index, f)

    total_bytes = sum(
        os.path.getsize(os.path.join(EDGE_SLICES_INCYTR_PATHWAYS_DIR, fn))
        for fn in os.listdir(EDGE_SLICES_INCYTR_PATHWAYS_DIR)
        if fn.endswith(".parquet")
    )
    print(f"  incytr_pair_pathways: wrote {len(present_pairs)} shards "
          f"({total_rows:,} rows; {total_bytes/1e6:.1f} MB total; "
          f"max {max_shard_bytes/1e6:.2f} MB → {max_shard_name})", flush=True)

    return {
        "schema_version": SCHEMA_VERSION,
        "version": 3,           # CR-04: v3 = multi-label traj_labels (semicolon-joined)
        "source": f"pair_mode ({os.path.relpath(input_dir, config.REPO_ROOT)})",
        "source_mode": "pair_mode",
        "contrasts": list(present_contrasts),
        "senders": senders_canonical,
        "receivers": receivers_canonical,
        "empty_deg_celltypes": [],
        "heatmap_counts": heatmap_counts,
        "pathway_counts": pathway_counts,
        "slice_index": index,
        "score_columns": list(_INCYTR_SCORE_COLS),
        "label_columns": list(_INCYTR_LABEL_COLS),
        "label_nodes": list(_INCYTR_LABEL_NODES),
        "label_vocab": list(_INCYTR_LABEL_VOCAB),
        "direction_flag_columns": list(dir_flag_cols),
        "path_metric_columns": list(extra_path_cols),
        # CR-04: traj_labels/sign_vec live in shard rows; summary inline.
        "trajectory_summary": traj_summary,
    }


def _read_empty_deg_celltypes() -> list[str]:
    """Read the list of cell types with no DEGs from the upstream MANIFEST.

    `compute_seed_lists.R` writes `deg_cell_type_status` into
    `<INCYTR_PAIR_MODE_OUTPUTS_DIR>/MANIFEST.json`. Cell types whose status
    indicates an empty DEG set are surfaced in the heatmap as hatched cells
    (visually distinct from "0 candidates pass the gate"). Returns `[]` if
    the manifest is absent or doesn't carry the field — the heatmap will
    just not render the hatched overlay in that case.
    """
    manifest_path = os.path.join(INCYTR_PAIR_MODE_OUTPUTS_DIR, "MANIFEST.json")
    if not os.path.exists(manifest_path):
        return []
    try:
        with open(manifest_path) as f:
            manifest = json.load(f)
    except (OSError, json.JSONDecodeError):
        return []
    status = manifest.get("deg_cell_type_status") or {}
    empty: list[str] = []
    for ct, info in status.items():
        if isinstance(info, dict):
            n = info.get("n_degs") or info.get("n_DEGs") or 0
            state = (info.get("status") or "").lower()
            if (isinstance(n, (int, float)) and n == 0) or state in {"empty", "no_degs"}:
                empty.append(ct)
        elif isinstance(info, str) and info.lower() in {"empty", "no_degs"}:
            empty.append(ct)
    return sorted(empty)


def _build_kinase_motifs(kinase_names: list) -> dict:
    """Per-kinase library PSSM (normalized probabilities) + ST favorability.

    Output is keyed by kinase name. Each entry:
        {
          "kin_type": "ser_thr" | "tyrosine",
          "positions": [-5, -4, ..., 4]  (or +5 for tyrosine),
          "amino_acids": [..., 23 entries ...],
          "matrix":  [[...], ...]  (n_aa x n_positions, normalized probs),
          "st_fav":  {"S": float, "T": float} | null,
        }
    Sequence-logo widget on the viewer side scales letter heights by
    information content (log2(20) - entropy) at each position.
    """
    import kinase_library as kl
    out: dict[str, dict] = {}
    skipped: list[str] = []
    for name in kinase_names:
        try:
            mat = kl.get_matrix(name, mat_type="norm")
            kin_type = kl.get_kinase_type(name)
        except Exception as e:
            skipped.append(f"{name} ({e})")
            continue
        st_fav: dict | None = None
        if kin_type == "ser_thr":
            try:
                sf = kl.get_st_fav(name)
                st_fav = {"S": float(sf.loc[name, "S"]),
                          "T": float(sf.loc[name, "T"])}
            except Exception:
                st_fav = None
        out[name] = {
            "kin_type": kin_type,
            "positions": [int(c) for c in mat.columns],
            "amino_acids": [str(a) for a in mat.index],
            "matrix": [[round(float(v), 4) for v in row] for row in mat.values],
            "st_fav": st_fav,
        }
    if skipped:
        print(f"  kinase_motifs: skipped {len(skipped)} kinases "
              f"(first 3: {skipped[:3]})", flush=True)
    print(f"  kinase_motifs: emitted PSSM for {len(out):,} kinases", flush=True)
    return out


def build_payload(data: UnifiedData) -> dict:
    """Assemble the full JSON payload (no edges — that's the sidecar)."""
    from kinase_library.modules import data as kl_data

    kinases_slice = _build_kinases_slice(data)
    celltypes_slice = _build_celltypes_slice(data)

    # Kinase family map
    try:
        fam = kl_data.get_kinase_family(data.edge_metadata["kinases"]).to_dict()
    except Exception as e:
        print(f"  (warn) family resolve failed: {e}; using empty map", flush=True)
        fam = {}

    contrasts = data.edge_metadata["contrasts"]

    # Decomp-OLS slices: per-kinase per-cell-type substrate-site OLS, fetched on
    # demand by the Attribution drawer to back the per-cell pseudo-deconv NES.
    _kid_for_slices = {k: i for i, k in enumerate(data.edge_metadata["kinases"])}
    _contrast_to_id_for_slices = {c: i for i, c in enumerate(contrasts)}
    decomp_ols_slice_index = _write_decomp_ols_slices(
        _kid_for_slices, _contrast_to_id_for_slices,
    )

    _kinase_genes = set(data.kinase_activity["gene_symbol"].dropna().astype(str))
    _kinase_genes |= set(data.edge_metadata["kinases"])
    song_concordance_slice_index = _write_song_concordance_slices(_kinase_genes)

    # Incytr pathway shards: one parquet per (sender, receiver), backing the
    # significant-pathway heatmap + table tabs. Pair-mode (Levy-t5) is the
    # only active source — factorial Incytr was archived 2026-05-18.
    incytr_pathways_block = _write_incytr_pair_pathways()

    meta = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "contrasts": contrasts,
        "diseaseGroups": list(config.DISEASE_GROUPS),
        "timepoints": list(config.TIMEPOINTS),
        "diseaseColors": dict(config.DISEASE_COLORS),
        "familyMap": fam,
        "transcript_trace": ensure_transcript_trace_sources(),
        "omics_trace": ensure_omics_trace_sources(),
        "omics_trace_normalized": ensure_omics_trace_normalized_sources(),
    }
    # Build-time round-trip assertion: recompute stored *_log2FC from substrate
    # and verify agreement to <=1e-4. Catches routing bugs, substrate drift,
    # sign flips, and aggregation mismatches. Skipped gracefully on first build
    # (before edge_slices/incytr_pathways/ exists). See Item 3.5.
    assert_pathway_fc_round_trips()

    kid = {k: i for i, k in enumerate(data.edge_metadata["kinases"])}
    ev = data.celltype_evidence[
        data.celltype_evidence["kinase"].isin(kid)
    ].copy()
    ev["kinase_id"] = ev["kinase"].map(kid).astype("uint16")
    kinase_celltype_evidence = {
        "kinase_id":  ev["kinase_id"].tolist(),
        "cell_type":  ev["cell_type"].tolist(),
        "wmb_fold":   ev["wmb_fold_over_uniform"].astype(float).round(3).tolist(),
        "sea_ad_lfc": ev["sea_ad_lfc"].astype(float).round(3).tolist(),
        "song_lfc":   ev["song_lfc"].astype(float).round(3).tolist(),
        "wmb_tier":   ev["wmb_tier"].astype(str).tolist(),
        "evidence_basis": ev["evidence_basis"].fillna("").astype(str).tolist(),
        "concordance_direction": ev["concordance_direction"].fillna("").astype(str).tolist(),
    }

    # Attribution index — single source of truth, built from unified_attribution_full.csv.
    # Contains ALL confidence tiers (high/moderate/low/none) so JS can derive
    # pills, cell-type counts, and verdict rows from one consistent source.
    # Extra columns (wmb_specificity, sea_ad_lfc, song_lfc, concordance_source)
    # are denormalized here so the verdict table no longer needs a separate CSV fetch.
    contrast_to_id = {c: i for i, c in enumerate(contrasts)}
    # Use full table if available, fall back to high+moderate subset.
    ua_src = data.unified_attribution_full if len(data.unified_attribution_full) > 0 \
        else data.unified_attribution
    ua = ua_src[ua_src["kinase"].isin(kid)
                & ua_src["contrast"].isin(contrast_to_id)].copy()
    # Ensure expected columns exist (fall-back for legacy unified_attribution.csv).
    for _col, _default in [
        ("wmb_specificity", float("nan")),
        ("wmb_mean_log2_expression", float("nan")),
        ("wmb_fraction_cells_expressing", float("nan")),
        ("wmb_binary_expressed", False),
        ("sea_ad_lfc", float("nan")),
        ("song_lfc", float("nan")),
        ("song_pval", float("nan")),
        ("song_fdr", float("nan")),
        ("concordance_source", ""),
        ("NES", float("nan")),
        ("FDR", float("nan")),
    ]:
        if _col not in ua.columns:
            ua[_col] = _default
    attribution_index = {
        "kinase_id":   ua["kinase"].map(kid).astype("uint16").tolist(),
        "contrast_id": ua["contrast"].map(contrast_to_id).astype("uint8").tolist(),
        "cell_type":   ua["cell_type"].astype(str).tolist(),
        "combined_confidence": ua["combined_confidence"].astype(str).tolist(),
        "combined_score": ua["combined_score"].astype(float).round(3).tolist(),
        "wmb_specificity": ua["wmb_specificity"].astype(float).round(4).tolist(),
        "wmb_mean_log2_expression": ua["wmb_mean_log2_expression"].astype(float).round(3).tolist(),
        "wmb_fraction_cells_expressing": ua["wmb_fraction_cells_expressing"].astype(float).round(3).tolist(),
        "wmb_binary_expressed": [bool(v) for v in ua["wmb_binary_expressed"].fillna(False)],
        "sea_ad_lfc": ua["sea_ad_lfc"].astype(float).round(4).tolist(),
        "song_lfc": ua["song_lfc"].astype(float).round(4).tolist(),
        "song_pval": ua["song_pval"].astype(float).round(4).tolist(),
        "song_fdr": ua["song_fdr"].astype(float).round(4).tolist(),
        "concordance_source": ua["concordance_source"].fillna("").astype(str).tolist(),
        "nes": ua["NES"].astype(float).round(4).tolist(),
        "fdr": ua["FDR"].astype(float).round(4).tolist(),
    }
    print(f"  attribution_index: {len(ua):,} rows "
          f"({ua['combined_confidence'].value_counts().to_dict()})",
          flush=True)

    decomp = data.decomposition
    decomp = decomp[decomp["kinase"].isin(kid)
                    & decomp["contrast"].isin(contrast_to_id)].copy()
    decomposition_index = {
        "kinase_id":   decomp["kinase"].map(kid).astype("uint16").tolist(),
        "contrast_id": decomp["contrast"].map(contrast_to_id).astype("uint8").tolist(),
        "cell_type":   decomp["wmb_class"].astype(str).tolist(),
        "decomp_nes":  decomp["NES"].astype(float).round(4).tolist(),
        "decomp_fdr":  decomp["FDR"].astype(float).round(4).tolist(),
    }
    print(f"  decomposition_index: {len(decomp):,} rows", flush=True)

    agreement_index = _build_agreement_index(
        data.mea_stoichiometry, data.decomposition,
        kid, contrast_to_id, config.MEA_FDR_THRESH,
    )

    human_slice, human_perdonor_substrate_index = build_human_slice()
    if human_slice is not None:
        print(f"  human slice: {len(human_slice['kinases']['id']):,} kinase rows "
              f"× {len(human_slice['donors'])} donors", flush=True)

    motif_names = set(kinases_slice.get("name", []))
    if human_slice is not None:
        motif_names.update(human_slice["kinases"].get("name", []))
    kinase_motifs = _build_kinase_motifs(sorted(motif_names))

    payload = {
        "kinases": kinases_slice,
        "kinase_motifs": kinase_motifs,
        "celltypes": celltypes_slice,
        "kinase_celltype_evidence": kinase_celltype_evidence,
        "attribution_index": attribution_index,
        "decomposition_index": decomposition_index,
        "agreement_index": agreement_index,
        "subclass_breakdown": _build_subclass_breakdown(kid),
        "audit_tables": build_audit_manifest(),
        "edge_slice_ref": {
            "schema_version": SCHEMA_VERSION,
            "decomp_ols_url": "edge_slices/decomp_ols/",
            "decomp_ols_index": "edge_slices/decomp_ols/index.json",
            "n_decomp_ols_slices": decomp_ols_slice_index.get("slice_count", 0),
            "present_decomp_ols_kinase_ids": decomp_ols_slice_index.get(
                "present_kinase_ids", []
            ),
            "incytr_pathways_url": "edge_slices/incytr_pathways/",
            "incytr_pathways_index": "edge_slices/incytr_pathways/index.json",
            "song_concordance_url": "edge_slices/song_concordance/",
            "song_concordance_index": "edge_slices/song_concordance/index.json",
            "present_song_concordance_genes": song_concordance_slice_index.get(
                "present_genes", []
            ),
            "human_perdonor_url": "edge_slices/human_perdonor/",
            "human_perdonor_index": "edge_slices/human_perdonor/index.json",
            "present_human_perdonor_kinase_ids": [],
        },
        "incytr_pathways": incytr_pathways_block,
        "meta": meta,
    }
    if human_slice is not None:
        if human_perdonor_substrate_index:
            payload["edge_slice_ref"]["present_human_perdonor_kinase_ids"] = (
                human_perdonor_substrate_index.get("present_kinase_ids", [])
            )
        payload["human"] = human_slice
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


def _render_template() -> str:
    def _raw(path: str) -> str:
        with open(os.path.join(_TEMPLATE_DIR, path)) as f:
            return f.read()

    env = Environment(
        loader=FileSystemLoader(_TEMPLATE_DIR),
        keep_trailing_newline=True,
    )
    env.globals["raw"] = _raw
    return env.get_template("index.html.j2").render()




def write_html(payload: dict, json_str: str | None = None) -> dict:
    """Emit the unified viewer HTML at UNIFIED_VIEWER_DIR/index.html.

    Sibling dirs (edge_slices/, edge_summaries/) are written by
    build_edge_shards.py; this function only writes the HTML.
    """
    os.makedirs(UNIFIED_VIEWER_DIR, exist_ok=True)
    if json_str is None:
        json_str = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    # PAYLOAD inlined inside <script type="application/json" id="payload-data">.
    # Trade-off: index.html grows to ~85 MB (vs ~1 MB with async fetch), but
    # the viewer is a single self-contained file with no second-fetch failure
    # mode (S3 ACL / file:// / CORS / etc). JS still supports the async-fetch
    # path as a fallback when the inline payload is empty/null.
    html = _render_template()
    for sentinel, value in (
        ("__APP_COLOR__", config.DISEASE_COLORS["App"]),
        ("__TAU_COLOR__", config.DISEASE_COLORS["Tau"]),
        ("__APTT_COLOR__", config.DISEASE_COLORS["ApTt"]),
        ("__PAYLOAD_SENTINEL__", json_str),
    ):
        html = html.replace(sentinel, value)
    raw = html.encode("utf-8")
    with open(UNIFIED_VIEWER_HTML, "wb") as f:
        f.write(raw)
    methods_bytes = 0
    if os.path.exists(PIPELINE_OVERVIEW_SRC):
        shutil.copyfile(PIPELINE_OVERVIEW_SRC, PIPELINE_OVERVIEW_DEST)
        methods_bytes = os.path.getsize(PIPELINE_OVERVIEW_DEST)
    else:
        print(f"WARNING: {PIPELINE_OVERVIEW_SRC} not found; "
              "Methods tab will 404. Render docs/pipeline_overview.qmd first "
              "(quarto render docs/pipeline_overview.qmd --to html).",
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
    if not os.path.exists(PAYLOAD_JSON):
        errors.append(f"payload JSON missing: {PAYLOAD_JSON}")
        raw_bytes = gzip_bytes = 0
        payload = None
    else:
        raw_bytes = os.path.getsize(PAYLOAD_JSON)
        gzip_bytes = os.path.getsize(PAYLOAD_JSON_GZ) if os.path.exists(PAYLOAD_JSON_GZ) else 0
        with open(PAYLOAD_JSON) as f:
            payload = json.load(f)

    if raw_bytes >= 100 * 1024 * 1024:
        errors.append(f"payload raw {raw_bytes/1e6:.1f} MB exceeds 100 MB cap")
    if gzip_bytes >= 20 * 1024 * 1024:
        errors.append(f"payload gzip {gzip_bytes/1e6:.1f} MB exceeds 20 MB cap")

    # Structural
    if payload is not None:
        pk = payload["kinases"]
        pc_ = payload["celltypes"]

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
                f"Re-run alz/wmb_expression.py with the expected scope."
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
        if verif.get("all_pass") is False:
            failed = [c.get("check") for c in (verif.get("checks") or [])
                      if isinstance(c, dict) and not c.get("pass", False)]
            raise SystemExit(
                f"Decomposition verification did not pass: failed checks = "
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

    if not any([args.summary, args.payload, args.html, args.validate]):
        args.payload = True
        args.html = True

    _assert_input_provenance(skip_verify=args.skip_verify)

    data = load_all_data()

    if args.summary:
        print(json.dumps(data.summary(), indent=2))

    payload = None
    json_str = None
    if args.payload:
        payload = build_payload(data)
        sizes = write_payload(payload)
        json_str = sizes.pop("json_str")
        print(f"  payload raw={sizes['raw_bytes']/1e6:.2f} MB "
              f"gzip={sizes['gzip_bytes']/1e6:.2f} MB")

    if args.html:
        if payload is None:
            if not os.path.exists(PAYLOAD_JSON):
                raise SystemExit(
                    f"payload missing at {PAYLOAD_JSON}; run --payload first"
                )
            with open(PAYLOAD_JSON) as f:
                json_str = f.read()
            payload = json.loads(json_str)
        info = write_html(payload, json_str=json_str)
        print(f"  html {info['html_bytes']/1e6:.2f} MB -> {info['output']}")

    if args.validate:
        validate(data)

    if args.payload and not args.skip_roundtrip:
        from integration import verify_pathway_round_trip as vprt  # noqa: E402
        vprt.verify(strict=args.strict_roundtrip)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Unified viewer builder: single entry point for kinase + pathway views.

Phase 3 adds the HTML shell (template + CSS + JS Store + Overview tab).
Phase 2 artifacts produced:

  - kinase_backbone_edges_sig.parquet  — edges filtered to backbones that
    pass both null permutation tests (significant_both). Sidecar artifact
    (not embedded in HTML; fetched by future tabs only if needed).
  - unified_viewer.payload.json (+ .gz) — columnar JSON payload with
    stable integer IDs for kinases, celltypes, and backbones. Embedded
    inline in the HTML via a <script type="application/json"> tag so the
    viewer is a single-file deliverable usable over file://.

The full 7.14 GB / 2.23B-row edge parquet is streamed via
ParquetFile.iter_batches — it is never materialized in memory.

Usage:
    python code/build_unified_viewer.py              # build + html (default)
    python code/build_unified_viewer.py --summary    # input row counts
    python code/build_unified_viewer.py --sidecar    # sig parquet only
    python code/build_unified_viewer.py --payload    # JSON only (needs sidecar)
    python code/build_unified_viewer.py --build      # sidecar + payload
    python code/build_unified_viewer.py --html       # write HTML (needs payload)
    python code/build_unified_viewer.py --validate   # write report md
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
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

import config  # noqa: E402
import config_integration as icfg  # noqa: E402
import kinase_attribution as kattr  # noqa: E402

# ---------------------------------------------------------------------------
# Paths — re-exported from viewer.paths so existing references in this module
# continue to resolve. The pathway-side payload builders in
# `viewer.pathway_payload` import directly from `viewer.paths`.
# ---------------------------------------------------------------------------

from viewer.paths import (  # noqa: E402
    AGGREGATION_DIR,
    AUDIT_PREVIEW_ROWS,
    AUDIT_SOURCES_DIR,
    BACKBONE_PERM_CSV,
    BACKBONE_REC_CSV,
    DECOMP_OLS_PARQUET,
    EDGE_META_JSON,
    EDGE_SLICES_BACKBONE_DIR,
    EDGE_SLICES_DECOMP_OLS_DIR,
    EDGE_SLICES_KINASE_DIR,
    EDGE_STREAM_BATCH,
    EDGES_PARQUET,
    MEASUREMENT_TRACE_DIR,
    MEASUREMENT_TRACE_INDEX,
    MEASUREMENT_TRACE_SCHEMA_VERSION,
    PAYLOAD_JSON,
    PAYLOAD_JSON_GZ,
    PER_BACKBONE_SUMMARY,
    PER_KINASE_SUMMARY,
    PIPELINE_OVERVIEW_DEST,
    PIPELINE_OVERVIEW_SRC,
    REPORT_MD,
    SCHEMA_VERSION,
    SIDECAR_PARQUET,
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
        ("unified_attribution_full", "Unified attribution (all 34 WMB classes)",
         os.path.join(config.KINASE_ATTRIBUTION_OUTPUT_DIR, "unified_attribution_full.csv")),
        ("wmb_kinase_expression", "WMB kinase expression",
         config.WMB_EXPRESSION_FILE),
        ("song_concordance", "Song within-cohort concordance",
         config.SONG_CONCORDANCE_FILE),
        ("sea_ad_supertype_lfc", "SEA-AD supertype LFCs",
         os.path.join(config.KINASE_ATTRIBUTION_OUTPUT_DIR, "sea_ad_supertype_lfc.csv")),
        ("kinase_activity_matrix", "Kinase activity matrix",
         os.path.join(config.ATTRIBUTION_RECOVERY_OUTPUT_DIR, "kinase_activity_matrix.csv")),
        ("celltype_evidence_table", "Cell-type evidence table",
         os.path.join(config.ATTRIBUTION_RECOVERY_OUTPUT_DIR, "celltype_evidence_table.csv")),
        ("kinase_hypothesis_table", "Kinase hypothesis table",
         os.path.join(config.ATTRIBUTION_RECOVERY_OUTPUT_DIR, "kinase_hypothesis_table.csv")),
        ("kinase_decomposition", "Kinase decomposition (WMB)",
         os.path.join(config.REPO_ROOT, "outputs", "reports", "deconvolution",
                      "per_animal", "kinase_enrichment_wmb.csv")),
    ])
    return specs


AUDIT_TABLE_SPECS = _audit_specs()

COLUMN_DEFINITIONS = {
    "kinase": "Kinase identifier used by the enrichment and attribution tables.",
    "gene_symbol": "Gene symbol associated with the site, kinase, or evidence row.",
    "contrast": "Disease genotype and timepoint comparison versus matched WT control.",
    "ES": "Raw enrichment score from motif enrichment analysis.",
    "NES": "Normalized enrichment score from motif enrichment analysis.",
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

    # Pathway-side
    backbone_sig: pd.DataFrame
    backbone_recurrence: pd.DataFrame
    unified_attribution: pd.DataFrame
    unified_attribution_full: pd.DataFrame
    decomposition: pd.DataFrame

    # Edge parquet — metadata only, streamed at use time
    edges_pf: pq.ParquetFile
    edge_metadata: dict = field(default_factory=dict)

    def summary(self) -> dict:
        md = self.edge_metadata
        return {
            "kinases": len(md.get("kinases", [])),
            "celltypes": len(md.get("celltypes", [])),
            "contrasts": len(md.get("contrasts", [])),
            "backbones": md.get("backbones_n", 0),
            "edges": self.edges_pf.metadata.num_rows,
            "kinase_activity_rows": len(self.kinase_activity),
            "celltype_evidence_rows": len(self.celltype_evidence),
            "kinase_hypothesis_rows": len(self.kinase_hypothesis),
            "mea_rows": len(self.mea_stoichiometry),
            "backbone_sig_rows": len(self.backbone_sig),
            "backbone_recurrence_rows": len(self.backbone_recurrence),
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

    backbone_sig = pd.read_csv(
        BACKBONE_PERM_CSV,
        usecols=[
            "contrast", "receiver", "Receptor", "EM", "Target",
            "observed_score", "pi0_null1", "pi0_null2", "significant_both",
        ],
        engine="pyarrow",
    )
    backbone_sig = backbone_sig[backbone_sig["significant_both"]].drop(
        columns=["significant_both"]
    ).reset_index(drop=True)
    backbone_recurrence = pd.read_csv(
        BACKBONE_REC_CSV,
        usecols=[
            "contrast", "receiver", "Receptor", "EM", "Target",
            "n_senders", "n_senders_significant",
            "mean_tpds", "max_abs_tpds", "tpds_pvalue", "sender_list",
            "pathway_evidence_backbone",
            "n_expression_confirmed", "n_kinase_imputed",
            "imputed_nodes_union",
        ],
        dtype={"sender_list": "string", "imputed_nodes_union": "string",
               "pathway_evidence_backbone": "string"},
    )
    # Hard significance gate: only keep recurrence rows whose (contrast,
    # receiver, Receptor, EM, Target) tuple passed both permutation nulls.
    # This drops ~90% of rows and removes spurious backbones from every tab.
    _before = len(backbone_recurrence)
    backbone_recurrence = backbone_recurrence.merge(
        backbone_sig[["contrast", "receiver", "Receptor", "EM", "Target"]],
        on=["contrast", "receiver", "Receptor", "EM", "Target"],
        how="inner",
    ).reset_index(drop=True)
    print(f"  recurrence sig-gate: kept {len(backbone_recurrence):,}/{_before:,} "
          f"rows ({100*len(backbone_recurrence)/_before:.1f}% pass both nulls)",
          flush=True)
    unified_attribution = pd.read_csv(
        icfg.UNIFIED_ATTRIBUTION_CSV,
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

    _decomp_path = os.path.join(
        config.REPO_ROOT, "outputs", "reports", "deconvolution",
        "per_animal", "kinase_enrichment_wmb.csv",
    )
    _decomp_cols = ["kinase", "wmb_class", "contrast", "NES", "FDR"]
    decomposition = pd.read_csv(
        _decomp_path, usecols=_decomp_cols,
    ) if os.path.exists(_decomp_path) else pd.DataFrame(columns=_decomp_cols)

    edges_pf = pq.ParquetFile(EDGES_PARQUET)
    with open(EDGE_META_JSON) as f:
        edge_metadata = json.load(f)

    # Extend the kinase universe to the union of edge_metadata kinases (which
    # anchor edge slice IDs 0..N-1) and every kinase carried in the activity
    # matrix — including pY kinases which post-date the Incytr edge build and
    # therefore have no edges yet. New kinases append after the edge-anchored
    # block, so existing kinase_id values in the edges parquet stay valid.
    edge_kinases = list(edge_metadata.get("kinases", []))
    edge_kinase_set = set(edge_kinases)
    activity_kinases = list(dict.fromkeys(kinase_activity["kinase"].astype(str).tolist()))
    extra_kinases = [k for k in activity_kinases if k not in edge_kinase_set]
    if extra_kinases:
        edge_metadata["kinases"] = edge_kinases + extra_kinases
        edge_metadata["edge_kinase_count"] = len(edge_kinases)
    else:
        edge_metadata.setdefault("edge_kinase_count", len(edge_kinases))

    return UnifiedData(
        kinase_activity=kinase_activity,
        celltype_evidence=celltype_evidence,
        kinase_hypothesis=kinase_hypothesis,
        mea_stoichiometry=mea,
        backbone_sig=backbone_sig,
        backbone_recurrence=backbone_recurrence,
        unified_attribution=unified_attribution,
        unified_attribution_full=unified_attribution_full,
        decomposition=decomposition,
        edges_pf=edges_pf,
        edge_metadata=edge_metadata,
    )


# ---------------------------------------------------------------------------
# Pathway-side builders are imported from viewer.pathway_payload.
# Anything below the kinase-side `_build_kinases_slice` / `_build_celltypes_slice`
# pair was extracted from this file as part of the pathway-redesign code split.
# ---------------------------------------------------------------------------

from viewer.paths import BACKBONE_VOCAB_CACHE  # noqa: E402, F401
from viewer.pathway_payload import (  # noqa: E402
    _build_backbones_slice,
    _build_overview_slice,
    _build_sender_matrix_slice,
    _build_tpds_distribution_slice,
    _encode_sender_mask,
    _extract_pi0,
    build_backbone_index,
    compute_sig_sets,
    write_sig_sidecar,
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

    edge_kinase_count = int(data.edge_metadata.get("edge_kinase_count", len(kinases)))
    cols: dict[str, list] = {
        "id": [], "name": [], "gene_symbol": [],
        "residue_type": [], "has_edges": [],
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
        cols["has_edges"].append(kid[k] < edge_kinase_count)
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

    Reads `outputs/reports/deconvolution/per_animal/site_level_ols.parquet`
    (3.77M rows: site × wmb_class × contrast × track), filters each kinase
    to its substrate-set motifs (across all contrasts/tracks), and writes
    one parquet per kinase to `edge_slices/decomp_ols/{kid:03d}.parquet`.

    The drawer in the Attribution tab fetches one shard on demand and
    filters client-side by current contrast + cell_type to populate the
    substrate-level evidence table for the per-cell pseudo-deconv NES.
    """
    if not os.path.exists(DECOMP_OLS_PARQUET):
        print(f"  (warn) decomp OLS parquet missing: {DECOMP_OLS_PARQUET}; "
              f"skipping decomp_ols slice generation", flush=True)
        return {"slice_count": 0, "present_kinase_ids": [], "filename_template":
                "{kinase_id:03d}.parquet"}

    os.makedirs(EDGE_SLICES_DECOMP_OLS_DIR, exist_ok=True)
    # Clear stale shards so an aborted previous run doesn't leave a mismatched
    # slice_count vs. present_kinase_ids.
    for f in os.listdir(EDGE_SLICES_DECOMP_OLS_DIR):
        if f.endswith(".parquet"):
            os.remove(os.path.join(EDGE_SLICES_DECOMP_OLS_DIR, f))

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
    cols = ["site_id", "gene_symbol", "motif", "wmb_class",
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
            ["contrast_id", "wmb_class", "site_id", "gene_symbol",
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


def build_payload(data: UnifiedData) -> dict:
    """Assemble the full JSON payload (no edges — that's the sidecar)."""
    from kinase_library.utils._global_vars import family_colors as KL_FAMILY_COLORS
    from kinase_library.modules import data as kl_data

    bb_index = build_backbone_index(data.backbone_recurrence)

    kinases_slice = _build_kinases_slice(data)
    celltypes_slice = _build_celltypes_slice(data)
    backbones_slice, sender_order = _build_backbones_slice(data, bb_index)

    # Kinase family map
    try:
        fam = kl_data.get_kinase_family(data.edge_metadata["kinases"]).to_dict()
    except Exception as e:
        print(f"  (warn) family resolve failed: {e}; using empty map", flush=True)
        fam = {}

    contrasts = data.edge_metadata["contrasts"]

    # Tier-1 edge summary (embedded). Tier-2 slices are loaded lazily by the
    # browser from edge_slices/ — not embedded.
    if not os.path.exists(PER_KINASE_SUMMARY):
        raise SystemExit(
            f"per_kinase_summary missing: {PER_KINASE_SUMMARY}. "
            f"Run: pixi run python code/integration/adapters/build_edge_shards.py"
        )
    pk_summary_tbl = pq.read_table(PER_KINASE_SUMMARY)
    per_kinase_summary = {name: pk_summary_tbl[name].to_pylist()
                          for name in pk_summary_tbl.column_names}

    # Distinct-backbone count per kinase (across contrasts). The per_kinase_summary
    # table is keyed by (kinase, contrast) so summing n_backbones over-counts when a
    # backbone supports the kinase in multiple contrasts. Compute distinct counts
    # from the source edge parquet so the kinase table's #Backbones column matches
    # the row count in the "Backbones supported" detail panel.
    src_kb = pq.read_table(SIDECAR_PARQUET,
                           columns=["kinase_id", "backbone_id"]).to_pandas()
    _distinct = src_kb.groupby("kinase_id")["backbone_id"].nunique()
    kinase_distinct_backbones = {
        "kinase_id": _distinct.index.astype(int).tolist(),
        "n_distinct_backbones": _distinct.values.astype(int).tolist(),
    }
    del src_kb, _distinct

    kinase_index_path = os.path.join(EDGE_SLICES_KINASE_DIR, "index.json")
    backbone_index_path = os.path.join(EDGE_SLICES_BACKBONE_DIR, "index.json")
    with open(kinase_index_path) as f:
        kinase_slice_index = json.load(f)
    with open(backbone_index_path) as f:
        backbone_slice_index = json.load(f)

    # Decomp-OLS slices: per-kinase per-cell-type substrate-site OLS, fetched on
    # demand by the Attribution drawer to back the per-cell pseudo-deconv NES.
    _kid_for_slices = {k: i for i, k in enumerate(data.edge_metadata["kinases"])}
    _contrast_to_id_for_slices = {c: i for i, c in enumerate(contrasts)}
    decomp_ols_slice_index = _write_decomp_ols_slices(
        _kid_for_slices, _contrast_to_id_for_slices,
    )

    meta = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "contrasts": contrasts,
        "diseaseGroups": list(config.DISEASE_GROUPS),
        "timepoints": list(config.TIMEPOINTS),
        "diseaseColors": dict(config.DISEASE_COLORS),
        "tissueOrder": list(config.TISSUE_ORDER),
        "tissueCategories": TISSUE_CATEGORIES,
        "receiverToTissue": RECEIVER_TO_TISSUE,
        "senderOrder": sender_order,
        "fdrThreshDefault": config.MEA_FDR_THRESH,
        "specificityHigh": round(config.SPECIFICITY_HIGH, 4),
        "specificityLow": round(config.SPECIFICITY_LOW, 4),
        "seaAdLfcMin": config.SEA_AD_LFC_MIN,
        "familyMap": fam,
        "familyColors": dict(KL_FAMILY_COLORS),
        "pi0": _extract_pi0(data.backbone_sig, contrasts),
        "contrast_sig_backbone_counts": {
            c: int(data.backbone_recurrence.loc[
                data.backbone_recurrence["contrast"] == c,
                ["receiver", "Receptor", "EM", "Target"],
            ].drop_duplicates().shape[0])
            for c in contrasts
        },
    }

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

    payload = {
        "kinases": kinases_slice,
        "celltypes": celltypes_slice,
        "backbones": backbones_slice,
        "overview": _build_overview_slice(data),
        "tpdsDistribution": _build_tpds_distribution_slice(),
        "senderMatrix": _build_sender_matrix_slice(data, sender_order),
        "per_kinase_summary": per_kinase_summary,
        "kinase_distinct_backbones": kinase_distinct_backbones,
        "kinase_celltype_evidence": kinase_celltype_evidence,
        "attribution_index": attribution_index,
        "decomposition_index": decomposition_index,
        "agreement_index": agreement_index,
        "subclass_breakdown": _build_subclass_breakdown(kid),
        "audit_tables": build_audit_manifest(),
        "edge_slice_ref": {
            "kinase_url": "edge_slices/kinase/",
            "backbone_url": "edge_slices/backbone/",
            "kinase_index": "edge_slices/kinase/index.json",
            "backbone_index": "edge_slices/backbone/index.json",
            "backbone_summary_url": "edge_summaries/per_backbone_summary.parquet",
            "bucket_size": backbone_slice_index["bucket_size"],
            "schema_version": SCHEMA_VERSION,
            "n_kinase_slices": kinase_slice_index["slice_count"],
            "n_backbone_buckets": backbone_slice_index["bucket_count"],
            "present_kinase_ids": kinase_slice_index["present_kinase_ids"],
            "source_sha256": kinase_slice_index.get("source_sha256"),
            "decomp_ols_url": "edge_slices/decomp_ols/",
            "decomp_ols_index": "edge_slices/decomp_ols/index.json",
            "n_decomp_ols_slices": decomp_ols_slice_index.get("slice_count", 0),
            "present_decomp_ols_kinase_ids": decomp_ols_slice_index.get(
                "present_kinase_ids", []
            ),
        },
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
# HTML shell (Phase 3)
# ---------------------------------------------------------------------------

# Raw string + sentinel replacement avoids f-string collisions with CSS/JS braces.
HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Kinase + Pathway Viewer</title>
<script src="https://cdn.plot.ly/plotly-2.35.0.min.js"></script>
<script src="https://unpkg.com/cytoscape@3.30.4/dist/cytoscape.min.js"></script>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet">
<script type="module">
  // hyparquet: ESM-only parquet reader. Attach to window for non-module code.
  // hyparquet-compressors supplies the ZSTD/SNAPPY decoders not bundled in
  // hyparquet itself — required because edge slices are written with zstd.
  import { parquetReadObjects } from "https://cdn.jsdelivr.net/npm/hyparquet@1.8.0/+esm";
  import { compressors } from "https://cdn.jsdelivr.net/npm/hyparquet-compressors@1.0.0/+esm";
  window.hyparquet = { parquetReadObjects, compressors };
</script>
<style>
:root {
  --app-red:__APP_COLOR__; --tau-blue:__TAU_COLOR__; --aptt-purple:__APTT_COLOR__;
  --up-red:__APP_COLOR__; --down-blue:__TAU_COLOR__;
  --receptor-color:#1b5e20; --em-color:#e65100; --target-color:#4a148c;
  --bg:#fafafa; --card-bg:#ffffff; --border:#e0e0e0;
  --text:#212121; --text-muted:#757575;
  --near-miss-bg:#fff8e1; --sub-thresh-bg:#f5f5f5;
  --selected-border:#1976d2;
  --font-body: "IBM Plex Sans", system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  --font-mono: "IBM Plex Mono", ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  --space-1:4px; --space-2:8px; --space-3:12px; --space-4:16px;
  --space-5:24px; --space-6:32px;
}
* { box-sizing:border-box; }
html, body { margin:0; padding:0; background:var(--bg); color:var(--text);
  font:13px/1.45 var(--font-body); font-feature-settings: "ss01", "cv11";
  -webkit-font-smoothing: antialiased; -moz-osx-font-smoothing: grayscale; }
table.data-table { font-variant-numeric: tabular-nums; }
table.data-table td:not(:first-child):not(:nth-child(2)) { font-feature-settings: "tnum"; }
header#app-header { background:#455a64; color:#fff; padding:8px 14px;
  display:flex; gap:10px; align-items:center; flex-wrap:wrap;
  border-bottom:1px solid #37474f; }
header#app-header h1 { margin:0 16px 0 0; font-size:15px; font-weight:600;
  letter-spacing:0.2px; }
header#app-header label { display:flex; gap:4px; align-items:center;
  font-size:12px; color:#cfd8dc; }
header#app-header select, header#app-header input[type=number] {
  background:#fff; color:var(--text); border:1px solid #37474f;
  border-radius:3px; padding:2px 5px; font-size:12px; }
header#app-header button#glossary-toggle {
  margin-left:auto; background:#263238; color:#fff; border:1px solid #37474f;
  border-radius:3px; padding:3px 10px; cursor:pointer; font-size:12px; }
header#app-header button#glossary-toggle:hover { background:#37474f; }
button:focus-visible, select:focus-visible, input:focus-visible,
tr[tabindex]:focus-visible {
  outline:2px solid #90caf9; outline-offset:2px;
}
nav#tab-bar { background:#37474f; display:flex; gap:0; padding:0 10px;
  border-bottom:1px solid #263238; overflow-x:auto; align-items:stretch; }
nav#tab-bar button { background:transparent; color:#cfd8dc; border:none;
  border-bottom:3px solid transparent; padding:9px 14px; cursor:pointer;
  font-size:13px; font-weight:500; letter-spacing:0.2px; }
nav#tab-bar button:hover { color:#fff; background:#455a64; }
nav#tab-bar button.active, nav#tab-bar button[aria-selected="true"] {
  color:#fff; border-bottom-color:#42a5f5;
}
nav#tab-bar .tab-group-label {
  display:inline-flex; align-items:center; padding:0 8px 0 4px;
  color:#90a4ae; font-size:10px; font-weight:600; letter-spacing:1px;
  text-transform:uppercase; pointer-events:none;
}
nav#tab-bar .tab-group-divider {
  display:inline-block; width:1px; margin:6px 6px;
  background:#546e7a; align-self:stretch;
}
.filter-label.dimmed { opacity:0.35; }
.filter-label.dimmed select, .filter-label.dimmed input { cursor:not-allowed; }
.prereq-card { display:flex; align-items:center; gap:14px;
  background:#fffde7; border:1px solid #ffe082; border-radius:6px;
  padding:14px 18px; margin:14px 0; color:#5d4037; font-size:13px; }
.prereq-card .prereq-icon { font-size:22px; flex:0 0 auto; }
.prereq-card .prereq-msg { flex:1; line-height:1.5; }
.prereq-card .prereq-msg strong { color:#3e2723; }
.prereq-card .prereq-action { background:#1976d2; color:#fff; border:none;
  border-radius:4px; padding:7px 14px; font-size:12px; font-weight:600;
  cursor:pointer; letter-spacing:0.3px; white-space:nowrap; }
.prereq-card .prereq-action:hover { background:#1565c0; }
#content-shell { display:flex; align-items:stretch;
  height:calc(100vh - var(--shell-top, 120px)); overflow:hidden; }
#content-shell > main#app-main { flex:1 1 auto; min-width:0;
  overflow-y:auto; }
#drawer-resizer { flex:0 0 5px; cursor:col-resize; background:transparent;
  border-left:1px solid var(--border); transition:background 0.15s;
  z-index:10; }
#drawer-resizer:hover, #drawer-resizer.dragging { background:#cfd8dc; }
#content-shell > aside#howto-drawer { flex:0 0 auto; width:340px;
  min-width:180px; max-width:600px;
  background:#fafafa; padding:16px 18px 24px; overflow-y:auto;
  font-size:12.5px; line-height:1.55; color:#37474f; }
#content-shell.drawer-collapsed > aside#howto-drawer { display:none; }
#content-shell.drawer-collapsed > #drawer-resizer { display:none; }
#howto-drawer-toggle { position:fixed; right:8px; top:calc(var(--shell-top, 120px) + 8px);
  width:28px; height:28px; border-radius:14px; border:1px solid var(--border);
  background:#fff; color:#455a64; font-size:14px; font-weight:700; cursor:pointer;
  box-shadow:0 1px 3px rgba(0,0,0,0.08); z-index:20; display:flex;
  align-items:center; justify-content:center; padding:0; }
#howto-drawer-toggle:hover { background:#eceff1; }
#howto-drawer-toggle.expanded { background:#1976d2; color:#fff; border-color:#1976d2; }
#howto-drawer h3 { margin:0 0 10px; font-size:14px; color:#263238;
  border-bottom:1px solid var(--border); padding-bottom:8px; font-weight:600; }
#howto-drawer h4 { margin:14px 0 4px; font-size:10px;
  text-transform:uppercase; letter-spacing:0.7px; color:#78909c; font-weight:700; }
#howto-drawer p { margin:0 0 8px; }
#howto-drawer ul { margin:0; padding-left:18px; }
#howto-drawer li { margin-bottom:5px; }
#howto-drawer .ht-cue { margin-bottom:6px; }
#howto-drawer .ht-cue .ht-metric { font-weight:600; color:#263238; }
#howto-drawer .ht-cue .ht-when { color:#546e7a; }
#howto-drawer .ht-pitfall { background:#fff8e1; border-left:3px solid #ffb300;
  padding:7px 9px; border-radius:0 3px 3px 0; margin-bottom:6px; }
#howto-drawer .ht-preamble { margin:0 0 10px; color:#546e7a; }
#howto-drawer .ht-conclusion { margin-bottom:8px; }
@media (max-width:900px) {
  #content-shell { flex-direction:column; height:auto; overflow:visible; }
  #content-shell > main#app-main { overflow-y:visible; }
  #drawer-resizer { display:none; }
  #content-shell > aside#howto-drawer { flex:1 1 auto; width:100% !important;
    max-width:100%; border-top:1px solid var(--border); }
}
main#app-main { padding:14px; }
.tab-panel { display:none; }
.tab-panel.active { display:block; }
.card { background:var(--card-bg); border:1px solid var(--border);
  border-radius:4px; padding:12px 14px; margin-bottom:12px; }
.card h2 { margin:0 0 6px; font-size:14px; font-weight:600; }
.muted { color:var(--text-muted); font-size:12px; }
.notice {
  display:none; background:#fff3cd; color:#5f4300; border:1px solid #f0c36d;
  border-radius:4px; padding:8px 12px; margin-bottom:12px; font-size:12px;
}
.notice.show { display:block; }
.panel-note { color:#455a64; font-size:12px; max-width:980px; }
.panel-note strong { color:#263238; font-weight:600; }
.guide-grid { display:grid; grid-template-columns:repeat(auto-fit, minmax(240px, 1fr));
  gap:12px; margin-top:12px; }
.guide-card { border:1px solid var(--border); border-radius:4px; padding:12px;
  background:#fff; }
.guide-card h3 { margin:0 0 6px; font-size:13px; color:#263238; }
.guide-card p { margin:0 0 8px; color:#455a64; font-size:12px; line-height:1.45; }
.guide-card ul { margin:6px 0 0 18px; padding:0; color:#455a64;
  font-size:12px; line-height:1.45; }
.guide-card li { margin:3px 0; }
.callout { border-left:3px solid #78909c; padding:8px 10px;
  background:#f7fafb; color:#455a64; font-size:12px; line-height:1.45;
  margin-top:10px; }
.callout strong { color:#263238; }
.metric-help {
  display:inline-flex; align-items:center; justify-content:center;
  width:16px; height:16px; margin-left:4px; border-radius:50%;
  border:1px solid #b0bec5; color:#455a64; background:#fff; font-size:11px;
  font-weight:700; cursor:help; vertical-align:middle;
}
.sr-only {
  position:absolute; width:1px; height:1px; padding:0; margin:-1px;
  overflow:hidden; clip:rect(0,0,0,0); white-space:nowrap; border:0;
}
#overview-plot { width:100%; height:560px; }
#glossary-panel { position:fixed; top:0; right:0; width:340px; height:100%;
  background:#fff; border-left:1px solid var(--border); padding:16px 18px;
  box-shadow:-4px 0 12px rgba(0,0,0,0.08); transform:translateX(100%);
  transition:transform 0.15s ease-out; z-index:50; overflow-y:auto; }
#glossary-panel.open { transform:translateX(0); }
#glossary-panel h3 { margin-top:0; font-size:14px; }
#glossary-panel dl { margin:0; font-size:12px; }
#glossary-panel dt { font-weight:600; margin-top:8px; color:#37474f; }
#glossary-panel dd { margin:2px 0 0; color:var(--text-muted); }
.tab-stub { color:var(--text-muted); font-style:italic; padding:40px;
  text-align:center; border:1px dashed var(--border); border-radius:4px; }
.explorer-layout { display:grid; grid-template-columns:minmax(0,1.6fr) minmax(320px,1fr);
  gap:12px; align-items:start; }
.kinase-audit-layout { display:flex; gap:0; align-items:stretch; grid-template-columns:none; }
.kinase-audit-layout > .card { flex:0 0 auto; width:420px; min-width:280px; max-width:60%;
  overflow:auto; }
.kinase-audit-layout > .ka-splitter { flex:0 0 6px; cursor:col-resize; background:transparent;
  border-left:1px solid var(--border); border-right:1px solid var(--border);
  margin:0 4px; transition:background 0.15s; }
.kinase-audit-layout > .ka-splitter:hover, .kinase-audit-layout > .ka-splitter.dragging { background:#cfd8dc; }
.kinase-audit-layout > .detail-card { flex:1 1 auto; min-width:0;
  position:sticky; top:8px; align-self:flex-start;
  max-height:calc(100vh - 24px); overflow-y:auto; overflow-x:hidden; }
.kinase-workbench-header { display:flex; align-items:flex-start; justify-content:space-between;
  gap:12px; padding-bottom:10px; border-bottom:1px solid var(--border); margin-bottom:10px; }
.kinase-workbench-title h3 { margin:0 0 3px; }
.kinase-workbench-controls { display:flex; gap:8px; align-items:center; flex-wrap:wrap;
  justify-content:flex-end; font-size:12px; }
.kinase-workbench-controls select { max-width:220px; padding:3px 6px;
  border:1px solid var(--border); border-radius:3px; font-size:12px; }
.kinase-audit-tabs { display:flex; gap:4px; overflow-x:auto; padding-bottom:8px;
  margin-bottom:10px; border-bottom:1px solid var(--border); }
.kinase-audit-tabs button { flex:0 0 auto; border:1px solid var(--border);
  background:#fff; border-radius:3px; padding:5px 8px; cursor:pointer;
  font-size:11px; color:#37474f; }
.kinase-audit-tabs button.active { background:#e3f2fd; border-color:var(--selected-border);
  color:#0d47a1; font-weight:600; }
.kinase-audit-tab-body { min-height:360px; }
.kinase-stage-note { color:#455a64; font-size:12px; line-height:1.45; max-width:980px;
  margin:0 0 10px; }
.audit-grid { display:grid; grid-template-columns:repeat(2, minmax(0, 1fr)); gap:10px; }
.audit-grid .audit-wide { grid-column:1 / -1; }
.audit-panel { border:1px solid var(--border); border-radius:4px; padding:10px;
  background:#fff; min-width:0; }
.audit-panel h4 { margin-top:0; }
.audit-controls { display:flex; gap:8px; align-items:center; flex-wrap:wrap;
  margin:0 0 8px; font-size:12px; }
.audit-controls input, .audit-controls select { padding:3px 6px;
  border:1px solid var(--border); border-radius:3px; font-size:12px; }
.audit-table-wrap { max-height:260px; overflow:auto; border:1px solid var(--border);
  border-radius:3px; }
.audit-table-wrap .data-table thead th { top:0; }
.audit-pager { display:flex; gap:8px; align-items:center; justify-content:flex-end;
  margin-top:6px; color:var(--text-muted); font-size:11px; }
.audit-pager button, .audit-controls button { border:1px solid #b0bec5; background:#fff;
  border-radius:3px; padding:2px 8px; cursor:pointer; font-size:11px; }
.audit-pager button:disabled { opacity:0.4; cursor:not-allowed; }
.audit-source-catalog { display:grid; grid-template-columns:280px minmax(0,1fr);
  gap:10px; }
.audit-source-list { max-height:330px; overflow:auto; border:1px solid var(--border); }
.audit-source-list button { display:block; width:100%; text-align:left; background:#fff;
  border:0; border-bottom:1px solid var(--border); padding:7px 8px; cursor:pointer; }
.audit-source-list button.active { background:#e3f2fd; box-shadow:inset 3px 0 0 var(--selected-border); }
.audit-kv { display:grid; grid-template-columns:150px minmax(0,1fr); gap:4px 8px;
  font-size:12px; margin-bottom:8px; }
.audit-kv dt { color:#546e7a; font-weight:600; }
.audit-kv dd { margin:0; overflow-wrap:anywhere; }
.numeric-cell { text-align:right !important; font-family:var(--font-mono); }

/* Attribution drawer */
.attr-verdict-table { width:100%; border-collapse:collapse; font-size:12px; }
.attr-verdict-table th { background:#f8fafc; font-weight:600; text-align:left;
  padding:6px 8px; border-bottom:1px solid var(--border); cursor:help; user-select:none; }
.attr-verdict-table th.attr-verdict-th { cursor:pointer; }
.attr-verdict-table th.attr-verdict-th:hover { background:#eef2f7; }
.attr-verdict-table td { padding:5px 8px; border-bottom:1px solid #f1f5f9; }
.attr-verdict-table tr.attr-verdict-row { cursor:pointer; }
.attr-verdict-table tr.attr-verdict-row:hover { background:#f8fafc; }
.attr-verdict-table tr.attr-verdict-selected { background:#e3f2fd; }
.attr-verdict-table tr.attr-verdict-selected:hover { background:#dbeafe; }
.attr-verdict-table thead tr.attr-verdict-supergroup th {
  background:#eef2f7; border-bottom:1px solid #cbd5e1; font-weight:600;
  text-align:center; font-size:11px; color:#334155; padding:4px 6px;
  cursor:default;
}
.attr-verdict-table thead tr.attr-verdict-supergroup th.attr-supergroup-attr { background:#fef3c7; }
.attr-verdict-table thead tr.attr-verdict-supergroup th.attr-supergroup-decomp { background:#dbeafe; }
.attr-verdict-table thead tr.attr-verdict-supergroup th.attr-supergroup-spacer { background:transparent; border-bottom:none; }
.attr-bulk-anchor { padding:4px 0 8px; font-size:12px; color:#334155; }
.attr-bulk-anchor .attr-bulk-pill { display:inline-block; padding:2px 8px; border-radius:10px;
  background:#f1f5f9; border:1px solid #cbd5e1; font-weight:600; }
.attr-bulk-anchor .attr-bulk-up { color:#b91c1c; }
.attr-bulk-anchor .attr-bulk-down { color:#1d4ed8; }
.attr-bulk-anchor .attr-bulk-ns { color:#64748b; font-weight:500; }
.attr-verdict-toggle { padding:6px 0 0; font-size:11px; }
.attr-verdict-toggle label { cursor:pointer; user-select:none; }
.attr-explainer { margin-top:10px; padding:8px 12px; background:#f8fafc; border:1px solid var(--border);
  border-radius:6px; font-size:12px; line-height:1.5; }
.attr-explainer summary { cursor:pointer; font-weight:600; color:#334155; padding:2px 0;
  user-select:none; list-style:revert; }
.attr-explainer summary:hover { color:#0f172a; }
.attr-explainer-body { padding-top:8px; }
.attr-explainer-body p { margin:6px 0; }
.attr-explainer-body ul { margin:6px 0 6px 18px; padding:0; }
.attr-explainer-body li { margin:3px 0; }
.attr-explainer-table { border-collapse:collapse; margin:8px 0; font-size:11px; }
.attr-explainer-table th { background:#f1f5f9; text-align:left; padding:4px 8px;
  border-bottom:1px solid var(--border); font-weight:600; }
.attr-explainer-table td { padding:3px 8px; border-bottom:1px solid #f1f5f9; }
.attr-explainer-table td.num { text-align:right; font-family:var(--font-mono); }
.attr-allen-link-secondary { color:#64748b !important; font-size:11px; }
.attr-num { text-align:right; font-family:var(--font-mono); color:#0f172a; }
.attr-num-lfc { font-weight:600; color:#0f172a; }
.attr-celltype { font-weight:500; }
.attr-subclass-marker { color:#94a3b8; font-size:10px; cursor:help; margin-left:2px; }
.attr-empty { color:#94a3b8; }
.attr-conf { display:inline-block; padding:1px 7px; border-radius:9px; font-size:11px;
  font-weight:600; text-transform:lowercase; }
.attr-conf-very-high { background:#14532d; color:#f0fdf4; }
.attr-conf-high { background:#dcfce7; color:#166534; }
.attr-conf-moderate { background:#fef3c7; color:#92400e; }
.attr-conf-low { background:#e2e8f0; color:#475569; }
.attr-conf-none { background:#f1f5f9; color:#94a3b8; }
.attr-badge { display:inline-block; padding:0 6px; border-radius:8px; font-size:10px;
  margin-left:4px; vertical-align:middle; cursor:help; }
.attr-badge-warn { background:#fee2e2; color:#991b1b; }
.attr-badge-info { background:#dbeafe; color:#1e40af; }
.attr-verdict-legend { padding:8px 0 2px; font-size:11px; line-height:1.45; }
.attr-drawer-header { padding:6px 0 8px; border-bottom:1px solid var(--border);
  margin-bottom:8px; }
.attr-allen-link { color:#2b6cb0; text-decoration:none; font-size:12px; }
.attr-allen-link:hover { text-decoration:underline; }
.attr-drawer-grid { display:grid; grid-template-columns:repeat(auto-fit, minmax(360px, 1fr));
  gap:12px; }
.attr-section { background:#fafafa; border:1px solid var(--border); border-radius:4px;
  padding:8px 10px; }
.attr-section h5 { margin:0 0 4px; font-size:13px; }
.attr-caption { font-size:11px; margin:0 0 6px; line-height:1.4; }
.attr-svg { display:block; max-width:100%; height:auto; }
.attr-dot-label { fill:#374151; font-family:var(--font-sans); }
.attr-dot-label-target { fill:#111827; font-weight:700; }
.attr-hm-label-target { font-weight:700; fill:#111827; }
.attr-song-table { width:100%; border-collapse:collapse; font-size:12px; }
.attr-song-table th { background:#f8fafc; font-weight:600; padding:5px 8px;
  text-align:left; border-bottom:1px solid var(--border); cursor:help; }
.attr-song-table td { padding:4px 8px; border-bottom:1px solid #f1f5f9; }
.attr-song-table tr.attr-song-selected { background:#e3f2fd; }
.attr-section-wide { margin-top:12px; }
.attr-section-wide .audit-scroll { max-height:340px; overflow:auto; }
.attr-decomp-ols-table { width:100%; border-collapse:collapse; font-size:12px; }
.attr-decomp-ols-table th { background:#f8fafc; font-weight:600; padding:5px 8px;
  text-align:left; border-bottom:1px solid var(--border); cursor:help;
  position:sticky; top:0; z-index:1; }
.attr-decomp-ols-table td { padding:4px 8px; border-bottom:1px solid #f1f5f9; }
.attr-decomp-ols-table td.attr-num { text-align:right; font-variant-numeric:tabular-nums; }
.attr-decomp-ols-table td.motif-mono { font-family:var(--font-mono, monospace); font-size:11px; }
@media (max-width:1100px) {
  .kinase-audit-layout, .audit-source-catalog { grid-template-columns:1fr; }
  .audit-grid { grid-template-columns:1fr; }
}
.detail-card { background:var(--card-bg); border:1px solid var(--border);
  border-radius:4px; padding:12px 14px; position:sticky; top:10px;
  max-height:calc(100vh - 80px); overflow-y:auto; }
.detail-card h3 { margin:0 0 4px; font-size:14px; font-weight:600; }
.detail-card .meta { color:var(--text-muted); font-size:11px; margin-bottom:8px; }
.detail-card h4 { margin:12px 0 4px; font-size:12px; font-weight:600;
  color:#37474f; text-transform:uppercase; letter-spacing:0.3px; }
.ke-toolbar { display:flex; gap:10px; align-items:center; margin-bottom:8px; }
.ke-toolbar.ke-toolbar-multi { flex-wrap:wrap; row-gap:6px; }
.ke-toolbar input { padding:3px 6px; border:1px solid var(--border);
  border-radius:3px; font-size:12px; width:220px; }
.ke-filter-label { display:inline-flex; align-items:center; gap:4px;
  font-size:11px; color:#546e7a; }
.ke-filter-label select { padding:2px 4px; border:1px solid var(--border);
  border-radius:3px; font-size:11px; max-width:160px; }
.ke-filter-reset { padding:2px 8px; font-size:11px; border:1px solid #b0bec5;
  border-radius:3px; background:#fff; color:#546e7a; cursor:pointer; }
.ke-filter-reset:hover { background:#eceff1; }
.ms-wrap { position:relative; display:inline-block; }
.ms-button { padding:2px 18px 2px 6px; border:1px solid var(--border);
  border-radius:3px; background:#fff; font-size:11px; cursor:pointer;
  color:#37474f; min-width:90px; max-width:180px; text-align:left;
  white-space:nowrap; overflow:hidden; text-overflow:ellipsis;
  background-image:linear-gradient(45deg, transparent 50%, #78909c 50%),
                    linear-gradient(135deg, #78909c 50%, transparent 50%);
  background-position:calc(100% - 9px) calc(50% - 1px), calc(100% - 5px) calc(50% - 1px);
  background-size:4px 4px, 4px 4px; background-repeat:no-repeat; }
.ms-button:hover { background-color:#f5f7f8; }
.ms-button[data-active="1"] { border-color:#1976d2; background-color:#e3f2fd; }
.ms-panel { position:absolute; top:calc(100% + 2px); left:0; z-index:10;
  background:#fff; border:1px solid #b0bec5; border-radius:3px;
  box-shadow:0 4px 12px rgba(0,0,0,0.12); padding:4px 0; min-width:140px;
  max-height:240px; overflow-y:auto; display:none; }
.ms-panel.open { display:block; }
.ms-opt { display:flex; align-items:center; gap:6px; padding:3px 10px;
  font-size:11px; color:#37474f; cursor:pointer; user-select:none;
  white-space:nowrap; }
.ms-opt:hover { background:#f0f4f7; }
.ms-opt input { margin:0; }
.ms-divider { border-top:1px solid #eceff1; margin:3px 0; }
.ms-action { padding:3px 10px; font-size:10px; color:#1976d2; cursor:pointer;
  user-select:none; }
.ms-action:hover { background:#f0f4f7; }
.ctx-chip { display:inline-block; padding:0 5px; margin-left:3px; border-radius:8px;
  font-size:9px; font-weight:600; letter-spacing:0.2px; background:#eceff1;
  color:#37474f; border:1px solid #cfd8dc; vertical-align:middle; }
.ctx-chip.vhi { background:#14532d; color:#f0fdf4; border-color:#14532d; }
.ctx-chip.hi { background:#c8e6c9; color:#1b5e20; border-color:#a5d6a7; }
.ctx-chip.mid { background:#fff3cd; color:#7a5a00; border-color:#ffe082; }
.ctx-overflow { display:inline-block; margin-left:3px; font-size:9px; color:#78909c;
  font-style:italic; vertical-align:middle; }
.scope-label { font-size:10px; color:#78909c; font-style:italic;
  margin-bottom:2px; display:block; }
.nes-profile-cell { display:inline-grid; grid-template-columns:repeat(3, 12px);
  grid-auto-rows:12px; gap:1px; padding:1px; background:#eceff1;
  border:1px solid #cfd8dc; border-radius:2px; vertical-align:middle; }
.nes-profile-cell .npc { width:12px; height:12px; box-sizing:border-box; }
.nes-profile-cell .npc.sig { outline:1.5px solid #000; outline-offset:-1px; z-index:1; }
.nes-profile-row-labels { display:inline-flex; flex-direction:column; gap:1px;
  margin-right:3px; vertical-align:middle; padding:1px 0; font-size:9px;
  color:#78909c; line-height:13px; }
.nes-profile-col-labels { display:flex; gap:1px; padding:0 1px; font-size:9px;
  color:#78909c; line-height:1; margin-bottom:1px; justify-content:space-around; }
.nes-profile-wrap { display:inline-flex; align-items:center; gap:2px; }
.agreement-profile-cell { display:inline-grid; grid-template-columns:repeat(3, 12px);
  grid-auto-rows:12px; gap:1px; padding:1px; background:#eceff1;
  border:1px solid #cfd8dc; border-radius:2px; vertical-align:middle; }
.agreement-profile-cell .apc { width:12px; height:12px; box-sizing:border-box; background:#fff; }
.agreement-profile-cell .apc.agree { background:#43a047; }
.agreement-profile-cell .apc.disagree { background:#ef6c00; }
.agreement-profile-wrap { display:inline-flex; align-items:center; gap:2px; }
.ke-table-wrap { max-height:70vh; overflow-y:auto; }
.data-table { border-collapse:collapse; font-size:12px; width:100%; }
.data-table th, .data-table td { padding:4px 8px; border-bottom:1px solid var(--border);
  text-align:left; vertical-align:top; white-space:nowrap; }
.data-table thead th { position:sticky; top:0; background:#eceff1; cursor:pointer;
  user-select:none; z-index:1; font-weight:600; }
.data-table thead th:hover { background:#cfd8dc; }
.data-table tbody tr { cursor:pointer; }
.data-table tbody tr:hover { background:#f5f5f5; }
.data-table tbody tr.selected { background:#e3f2fd; box-shadow:inset 3px 0 0 var(--selected-border); }
.data-table tbody tr.sub-thresh { color:var(--text-muted); }
.data-table tbody tr.driver { box-shadow:inset 3px 0 0 #e65100; }
.badge { display:inline-block; padding:1px 6px; border-radius:9px; font-size:10px;
  font-weight:600; letter-spacing:0.2px; }
.badge.vhi { background:#14532d; color:#f0fdf4; }
.badge.hi { background:#c8e6c9; color:#1b5e20; }
.badge.lo { background:#eceff1; color:#546e7a; }
.badge.mid { background:#fff3cd; color:#7a5a00; }
.badge.expr { background:#dcedc8; color:#33691e; }
.badge.imp { background:#ffe0b2; color:#bf360c; }
.badge.mix { background:#d1c4e9; color:#4527a0; }
.track-badge { display:inline-block; padding:0 5px; border-radius:6px; font-size:9px;
  font-weight:700; letter-spacing:0.3px; vertical-align:1px; }
.track-badge.track-y { background:#ede7f6; color:#4527a0; border:1px solid #d1c4e9; }
#ke-detail-nes { height:160px; }
.mea-scorecard { display:flex; gap:24px; align-items:flex-start; flex-wrap:wrap; padding:6px 0; }
.mea-score-nes { display:flex; flex-direction:column; align-items:flex-start; min-width:140px; }
.mea-score-label { font-size:11px; color:#78909c; letter-spacing:0.06em; text-transform:uppercase; }
.mea-score-value { font-size:38px; font-weight:600; line-height:1; margin:2px 0 6px; font-variant-numeric:tabular-nums; }
.mea-score-chip { font-size:11px; font-weight:600; padding:2px 8px; border-radius:10px; }
.mea-score-chip.chip-pass { background:#c8e6c9; color:#1b5e20; }
.mea-score-chip.chip-borderline { background:#fff3e0; color:#bf360c; }
.mea-score-chip.chip-fail { background:#eceff1; color:#546e7a; }
.mea-score-chip.muted { background:#eceff1; color:#90a4ae; }
.mea-score-stats { display:grid; grid-template-columns:auto 1fr; column-gap:14px; row-gap:4px; margin:0; font-size:13px; }
.mea-score-stats dt { color:#607d8b; font-weight:500; }
.mea-score-stats dd { margin:0; font-variant-numeric:tabular-nums; }
#pe-detail-cross { height:180px; }
.pe-chip { display:inline-block; padding:1px 5px; margin:0 2px 2px 0;
  border-radius:3px; font-size:10px; font-weight:600; background:#eceff1; color:#546e7a; }
.pe-chip.on { background:#c8e6c9; color:#1b5e20; }
.pe-chip.expr { background:#dcedc8; color:#33691e; }
.pe-chip.imp { background:#ffe0b2; color:#bf360c; }
.pe-chip.mix { background:#d1c4e9; color:#4527a0; }
.detail-chips { display:flex; gap:12px; align-items:center; margin-bottom:8px;
  flex-wrap:wrap; font-size:12px; }
.detail-chips label { display:flex; gap:4px; align-items:center; }
.chip { background:#fff3cd; color:#8a6d3b; border:1px solid #f0ad4e;
  border-radius:3px; padding:2px 8px; font-size:11px; cursor:pointer; }
.tv2-row { display:flex; gap:8px; align-items:center; padding:6px 8px;
  background:#f5f5f5; border:1px solid var(--border); border-radius:4px;
  font-size:11px; flex-wrap:wrap; }
.tv2-row label { display:flex; gap:4px; align-items:center; }
.tv2-row select, .tv2-row input { font-size:11px; padding:1px 4px; }
.tv2-row .tv2-rm { background:#fff; color:#b71c1c; border:1px solid #b71c1c;
  border-radius:3px; padding:0 6px; font-size:11px; cursor:pointer;
  margin-left:auto; }
.tv2-row .tv2-rm:hover { background:#b71c1c; color:#fff; }
.tv2-row .tv2-label { font-weight:600; min-width:54px; color:#37474f; }
.tv2-row[data-disabled~="cells"] .tv2-cells { opacity:0.4; pointer-events:none; }
.tv2-row[data-disabled~="agree"] .tv2-agree { opacity:0.4; pointer-events:none; }
#graph-container { display:grid; grid-template-columns:1fr 320px; gap:12px;
  height:calc(100vh - 180px); }
#cy { background:#fafafa; border:1px solid var(--border); border-radius:4px;
  min-height:400px; }
#graph-detail { position:relative; top:0; }
.graph-legend { display:inline-flex; gap:10px; align-items:center; margin-left:auto;
  font-size:11px; color:var(--text-muted); }
.graph-legend-item { display:inline-flex; gap:4px; align-items:center; }
.graph-legend-swatch { display:inline-block; width:10px; height:10px; border-radius:50%;
  border:1px solid rgba(0,0,0,0.15); }
.graph-placeholder { display:flex; align-items:center; justify-content:center;
  height:100%; color:var(--text-muted); font-style:italic; text-align:center;
  padding:20px; }
.pe-cset { display:flex; flex-wrap:wrap; gap:6px 12px; align-items:center;
  margin:6px 0 8px; padding:6px 8px; background:#f7f8fa; border:1px solid var(--border);
  border-radius:4px; font-size:11px; }
.pe-cset-label { font-weight:600; color:#37474f; }
.pe-cset-chips { display:flex; flex-wrap:wrap; gap:4px; }
.pe-cset-chip { background:#eceff1; color:#37474f; border:1px solid #cfd8dc;
  border-radius:12px; padding:1px 8px; font-size:10px; font-weight:600; cursor:pointer; }
.pe-cset-chip.on { background:#1976d2; color:#fff; border-color:#1976d2; }
.pe-cset-mode { display:flex; gap:8px; }
.pe-cset-mode label { display:flex; gap:3px; align-items:center; cursor:pointer; }
#pe-cset-clear { padding:1px 8px; }
.pe-cchip-cell { white-space:nowrap; }
.pe-cchip { display:inline-block; padding:1px 5px; margin:0 2px 2px 0;
  border-radius:3px; font-size:10px; font-weight:600; background:#e3f2fd; color:#0d47a1; }
.pe-cchip-more { background:#cfd8dc; color:#37474f; cursor:help; }
</style>
</head>
<body>
<header id="app-header">
  <h1>Kinase + Pathway Viewer</h1>
  <label class="filter-label" data-filter="receiver" data-metric="receiver">Receiver <select id="f-receiver" aria-label="Receiver filter"></select></label>
  <label class="filter-label" data-filter="pathwayEvidence" data-metric="pathwayEvidence">Support <select id="f-pathway-evidence" aria-label="Pathway support filter">
    <option value="ALL">All</option>
    <option value="expression-confirmed">Expression-confirmed</option>
    <option value="kinase-imputed">Kinase-imputed</option>
    <option value="mixed">Mixed</option>
  </select></label>
  <label class="filter-label" data-filter="fdr" data-metric="fdr">FDR &lt; <input id="f-fdr" aria-label="FDR threshold" type="number" step="0.05" min="0" max="1"
    value="0.25" style="width:60px;"></label>
  <label class="filter-label" data-filter="tpdsSig" data-metric="tpdsSig"
    title="Threshold on chain-level TPDS p-value. Off = no gate (every enumerated chain). 0.10/0.05/0.01 = require the chain's TPDS magnitude to be significantly non-zero at that level (min over senders). Distinct from the chain over-representation test, which gates kinase specificity rather than TPDS magnitude.">TPDS p &lt; <select id="f-tpds-sig" aria-label="TPDS p-value threshold">
    <option value="OFF">off</option>
    <option value="0.10">0.10</option>
    <option value="0.05">0.05</option>
    <option value="0.01">0.01</option>
  </select></label>
  <button id="glossary-toggle" aria-controls="glossary-panel" aria-expanded="false" title="Open metric definitions and abbreviations">Glossary</button>
  <button id="f-graph-nodes-clear" class="chip" hidden>Clear graph-node filter</button>
  <button id="f-sender-clear" class="chip" hidden>Clear sender filter</button>
  <button id="f-selection-kinase-clear" class="chip" hidden>Clear kinase selection</button>
  <button id="f-selection-celltype-clear" class="chip" hidden>Clear cell-type selection</button>
</header>
<nav id="tab-bar" role="tablist" aria-label="Viewer panels">
  <span class="tab-group-label">Landscape</span>
  <button id="tabbtn-signal" role="tab" aria-selected="true" aria-controls="tab-signal" data-tab="signal" data-tab-group="landscape" class="active">Signal Map</button>
  <button id="tabbtn-senders" role="tab" aria-selected="false" aria-controls="tab-senders" data-tab="senders" data-tab-group="landscape">Sender&times;Receiver</button>
  <button id="tabbtn-temporal" role="tab" aria-selected="false" aria-controls="tab-temporal" data-tab="temporal" data-tab-group="landscape">Temporal</button>
  <button id="tabbtn-additivity" role="tab" aria-selected="false" aria-controls="tab-additivity" data-tab="additivity" data-tab-group="landscape">Additivity</button>
  <button id="tabbtn-temporalv2" role="tab" aria-selected="false" aria-controls="tab-temporalv2" data-tab="temporalv2" data-tab-group="landscape">Temporal v2</button>
  <span class="tab-group-divider" aria-hidden="true"></span>
  <span class="tab-group-label">Drill-down</span>
  <button id="tabbtn-kinase" role="tab" aria-selected="false" aria-controls="tab-kinase" data-tab="kinase" data-tab-group="drilldown">Kinase</button>
  <button id="tabbtn-pathway" role="tab" aria-selected="false" aria-controls="tab-pathway" data-tab="pathway" data-tab-group="drilldown">Pathway</button>
  <button id="tabbtn-graph" role="tab" aria-selected="false" aria-controls="tab-graph" data-tab="graph" data-tab-group="drilldown">Graph</button>
  <span class="tab-group-divider" aria-hidden="true"></span>
  <span class="tab-group-label">Reference</span>
  <button id="tabbtn-methods" role="tab" aria-selected="false" aria-controls="tab-methods" data-tab="methods" data-tab-group="reference">Methods</button>
</nav>
<div id="content-shell">
<main id="app-main">
  <div id="file-mode-notice" class="notice">
    This viewer is open as a local file. Some kinase and pathway drill-down
    panels fetch parquet sidecar files, which browsers block under file://.
    Serve this directory over HTTP if those drill-downs are needed.
  </div>
  <div id="tab-signal" class="tab-panel active" role="tabpanel" aria-labelledby="tabbtn-signal">
    <div class="card">
      <h2>Receiver &times; Contrast</h2>
      </div>
    <div class="card">
      <div id="overview-plot" role="img" aria-label="Heatmap of significant pathway backbones by receiver and contrast"></div>
    </div>
  </div>
  <div id="tab-kinase" class="tab-panel" role="tabpanel" aria-labelledby="tabbtn-kinase" hidden>
    <div class="explorer-layout kinase-audit-layout">
      <div class="card">
        <div class="ke-toolbar ke-toolbar-multi" id="ke-unified-toolbar">
          <label class="sr-only" for="ke-search">Search kinase or gene</label>
          <input id="ke-search" aria-label="Search kinase or gene" placeholder="Search kinase or gene…"/>
          <span class="ke-filter-label" id="ke-ms-disease"
                data-key="disease" data-label="Disease"
                title="Filter by disease. Multi-select: kinase must have ≥1 row in EVERY selected disease (cross-grid AND with timepoint and cell type)."></span>
          <span class="ke-filter-label" id="ke-ms-timepoint"
                data-key="timepoint" data-label="Timepoint"
                title="Filter by timepoint. Multi-select: kinase must have ≥1 row in EVERY selected timepoint."></span>
          <span class="ke-filter-label" id="ke-ms-celltype"
                data-key="celltype" data-label="Cell type"
                title="Filter by attributed cell type. Multi-select: kinase must have ≥1 row in EVERY selected cell type."></span>
          <span class="ke-filter-label" id="ke-ms-confidence-wrap">
            <label class="ke-filter-label">Confidence
              <select id="ke-filter-confidence" aria-label="Minimum confidence tier"
                      title="Minimum combined tier. Threshold (≥): very high = upgraded by decomp agreement only; high = high+very high; moderate = +moderate; low = +low.">
                <option value="">Any</option>
                <option value="very_high">very high (only)</option>
                <option value="high">high+</option>
                <option value="moderate">moderate+</option>
                <option value="low">low+</option>
              </select>
            </label>
          </span>
          <span class="ke-filter-label" id="ke-ms-wmb-wrap">
            <label class="ke-filter-label">WMB
              <select id="ke-filter-wmb" aria-label="Minimum WMB specificity tier"
                      title="Minimum WMB specificity tier (× uniform 1/34 ≈ 0.029). Kinase passes if any attribution row in scope has wmb_specificity ≥ threshold.">
                <option value="0">Any</option>
                <option value="1">≥1× (0.029)</option>
                <option value="2">≥2× (0.059)</option>
                <option value="5">≥5× (0.147)</option>
                <option value="10">≥10× (0.294)</option>
              </select>
            </label>
          </span>
          <label class="ke-filter-label" title="Minimum number of significant contrasts (FDR < threshold) the kinase must reach within the active disease/timepoint scope. 0 = no constraint.">n_sig &geq;
            <input id="ke-filter-nsig-min" type="number" min="0" max="9" step="1" value="0" style="width:48px;"
                   aria-label="Minimum number of significant contrasts"/>
          </label>
          <button id="ke-filter-reset" class="ke-filter-reset" title="Reset all kinase filters">Reset</button>
          <span class="muted" id="ke-count"></span>
        </div>
        <div class="ke-table-wrap">
          <table class="data-table" id="ke-table">
            <thead><tr>
              <th data-col="name" data-metric="kinaseName" title="Display label: Kinase&#10;Raw column: name&#10;Definition: Kinase identifier from the MEA / integration tables.">Kinase</th>
              <th data-col="gene_symbol" data-metric="kinaseGene" title="Display label: Gene&#10;Raw column: gene_symbol&#10;Definition: Gene symbol associated with the kinase.">Gene</th>
              <th data-col="nes_profile" data-metric="nesProfile" title="NES profile across 9 contrasts (3 diseases × 3 timepoints). Color = direction (red up, blue down), saturation = |NES|, black outline = FDR < threshold. Click to sort by max |NES| in scope.">NES Profile</th>
              <th data-col="agreement_profile" data-metric="agreementProfile" title="Bulk-vs-decomp agreement across 9 contrasts. Green = both pipelines significant, same direction; orange = disagreement (decomp_only / bulk_only / mixed / pure disagreement); empty = neither significant. Click to sort by count of disagreement contrasts (in scope).">Agreement</th>
              <th data-col="peak_NES" data-metric="peakNes" title="Peak |NES| — max absolute NES across the active disease/timepoint scope (or all 9 contrasts if no scope). Click to sort.">|NES|</th>
              <th data-col="n_sig" data-metric="nSig" title="n_sig — count of contrasts where FDR < threshold, within the active disease/timepoint scope (or out of 9 if no scope). Click to sort.">n_sig</th>
              <th data-col="n_attributed_celltypes" data-metric="nAttributed" title="Cell types attributed at high or moderate confidence in the active scope. Click to sort by count (matches the pill column).">Cell types</th>
              <th data-col="wmb_max_tier" data-metric="wmbMaxTier" title="Maximum WMB specificity tier reached by any attribution row in scope (×uniform 1/34). 10× = ≥0.294, 5× = ≥0.147, 2× = ≥0.059, 1× = ≥0.029. Click to sort by max tier.">WMB</th>
              <th data-col="conf" data-metric="highConfAttr" title="Highest confidence tier reached in the active scope (high &gt; moderate &gt; low &gt; none). Click to sort by max tier — kinases with at least one HIGH-confidence row come first, ties broken by table order.">Conf</th>
            </tr></thead>
            <tbody></tbody>
          </table>
        </div>
      </div>
      <div class="ka-splitter" id="ka-splitter" title="Drag to resize"></div>
      <section class="detail-card" id="ke-detail">
        <div class="muted">Select a kinase to see details.</div>
      </section>
    </div>
  </div>
  <div id="tab-pathway" class="tab-panel" role="tabpanel" aria-labelledby="tabbtn-pathway" hidden>
    <div class="explorer-layout">
      <div class="card">
        <div class="ke-toolbar">
          <label class="sr-only" for="pe-search">Search Receptor, EM, or Target</label>
          <input id="pe-search" aria-label="Search Receptor, EM, or Target" placeholder="Search Receptor / EM / Target…"/>
          <label>|TPDS| &geq;
            <input id="pe-tpds-min" type="number" step="0.1" min="0" value="0"
                   aria-label="Minimum absolute TPDS for pathway rows"
                   title="Hide backbones whose maximum |TPDS| across passing contrasts falls below this value."
                   style="width:64px;">
          </label>
          <span class="muted" id="pe-count"></span>
        </div>
        <div class="pe-cset" role="group" aria-label="Filter by trajectory">
          <span class="pe-cset-label">Show backbones passing in:</span>
          <span id="pe-traj-buttons" class="pe-cset-chips"></span>
        </div>
        <div class="ke-table-wrap">
          <table class="data-table" id="pe-table">
            <thead><tr>
              <th data-col="receiver" data-metric="receiverCol" title="Display label: Receiver&#10;Raw column: receiver&#10;Definition: Receiver cell type for the backbone.">Receiver</th>
              <th data-col="Receptor" data-metric="receptorCol" title="Display label: Receptor&#10;Raw column: Receptor&#10;Definition: Receptor gene in the backbone.">Receptor</th>
              <th data-col="EM" data-metric="emCol" title="Display label: EM&#10;Raw column: EM&#10;Definition: Extracellular-matrix or intermediate molecule in the backbone.">EM</th>
              <th data-col="Target" data-metric="targetCol" title="Display label: Target&#10;Raw column: Target&#10;Definition: Downstream target gene in the backbone.">Target</th>
              <th data-col="pathway_evidence" data-metric="pathwayEvidence" title="Display label: Support&#10;Raw column: pathway_evidence_backbone&#10;Definition: How a backbone's chain was assembled.">Support</th>
              <th data-col="tpds" data-metric="tpds" title="Display label: TPDS&#10;Raw column: mean_tpds&#10;Definition: Transcript-level pathway differential score for the selected contrast.">TPDS</th>
              <th data-col="passing_contrasts" data-metric="passingContrasts" title="Display label: Passing contrasts&#10;Raw column: significant_both_mask&#10;Definition: Genotype-by-timepoint contrasts where this backbone passed both permutation nulls.">Passing contrasts</th>
              <th data-col="n_senders" data-metric="nSenders" title="Display label: Senders&#10;Raw column: n_senders_significant&#10;Definition: Number of significant sender cell types detected for this backbone.">Senders</th>
              <th data-col="max_abs_tpds" data-metric="maxAbsTpds" title="Display label: Max |TPDS|&#10;Raw column: max_abs_tpds&#10;Definition: Largest absolute TPDS observed across contrasts.">Max|TPDS|</th>
            </tr></thead>
            <tbody></tbody>
          </table>
        </div>
      </div>
      <aside class="detail-card" id="pe-detail">
        <div class="muted">Select a backbone to see details.</div>
      </aside>
    </div>
  </div>
  <div id="tab-graph" class="tab-panel" role="tabpanel" aria-labelledby="tabbtn-graph" hidden>
    <div id="graph-controls" class="detail-chips">
      <label>Genotype:
        <select id="graph-genotype" aria-label="Graph genotype" title="Pick which disease genotype's network is shown.">
          <option value="App">App</option>
          <option value="Tau">Tau</option>
          <option value="ApTt">ApTt</option>
        </select>
      </label>
      <label>Timepoint:
        <select id="graph-timepoint" aria-label="Graph timepoint" title="Pick which timepoint's snapshot to render. Step with the slider or arrow keys.">
          <option value="2mo">2mo</option>
          <option value="4mo">4mo</option>
          <option value="6mo">6mo</option>
        </select>
      </label>
      <label>Layout:
        <select id="graph-layout" aria-label="Graph layout" title="Choose how receptor, EM, and target nodes are arranged.">
          <option value="concentric">Concentric (R → EM → T)</option>
          <option value="flow">Flow (column-snapped)</option>
          <option value="force">Force-directed</option>
        </select>
      </label>
      <label>Min degree:
        <select id="graph-min-degree" aria-label="Minimum graph node degree" title="Hide nodes connected to fewer than this many backbones.">
          <option value="1">1</option>
          <option value="2">2</option>
          <option value="5">5</option>
          <option value="10">10</option>
          <option value="20">20</option>
          <option value="50">50</option>
        </select>
      </label>
      <label>|TPDS| &geq;
        <input id="graph-tpds-min" type="number" step="0.1" min="0" value="0"
               aria-label="Minimum absolute TPDS for graph edges"
               title="Hide edges whose mean |TPDS| is below this value. Drag down to reveal weaker edges."
               style="width:64px;">
      </label>
      <label>Max edges:
        <input id="graph-top-n" type="number" step="50" min="0" placeholder="off"
               aria-label="Maximum number of edges to render"
               title="Optional rendering cap: keep only the top N edges by |TPDS|. Leave blank for no cap."
               style="width:72px;">
      </label>
      <span id="graph-stats" class="muted"></span>
      <button id="graph-focus-clear" class="chip" hidden>Clear focus</button>
      <span class="graph-legend" aria-label="Node role legend">
        <span class="graph-legend-item"><span class="graph-legend-swatch" style="background:var(--receptor-color);"></span>Receptor</span>
        <span class="graph-legend-item"><span class="graph-legend-swatch" style="background:var(--em-color);"></span>EM (effector)</span>
        <span class="graph-legend-item"><span class="graph-legend-swatch" style="background:var(--target-color);"></span>Target</span>
      </span>
    </div>
    <div id="graph-container">
      <div id="cy" role="img" aria-label="Interactive receptor, EM, and target pathway graph"></div>
      <aside class="detail-card" id="graph-detail">
        <div class="muted">Click a node for details.</div>
      </aside>
    </div>
  </div>
  <div id="tab-senders" class="tab-panel" role="tabpanel" aria-labelledby="tabbtn-senders" hidden>
    <div class="card">
      <div class="detail-chips">
        <label>Compare:
          <select id="sm-axis" aria-label="Sender receiver compare axis" title="Compare timepoints holds genotype fixed and shows three timepoints side-by-side. Compare genotypes holds timepoint fixed and shows the three genotypes.">
            <option value="timepoint">Timepoints (fixed genotype)</option>
            <option value="genotype">Genotypes (fixed timepoint)</option>
          </select>
        </label>
        <label id="sm-anchor-label">Anchor:
          <select id="sm-anchor" aria-label="Sender receiver anchor"></select>
        </label>
        <label>Mode:
          <select id="sm-mode" aria-label="Sender receiver matrix mode" title="Count mode shows log-scaled backbone counts; direction mode shows up minus down backbones.">
            <option value="count">Count (log10 1+n)</option>
            <option value="direction">Direction (n_up − n_down)</option>
          </select>
        </label>
        <span class="muted" id="sm-subtitle"></span>
        <span class="muted" id="sm-keyhint" title="Step the anchor with left and right; flip the compare axis with up and down.">
          <kbd>←</kbd> <kbd>→</kbd> step anchor · <kbd>↑</kbd> <kbd>↓</kbd> flip axis
        </span>
      </div>
      <div id="sender-matrix-grid" style="display:grid; grid-template-columns:repeat(3, 1fr); gap:12px;">
        <div id="sender-matrix-plot-0" role="img" aria-label="Sender by receiver heatmap, panel 1" style="width:100%; height:560px;"></div>
        <div id="sender-matrix-plot-1" role="img" aria-label="Sender by receiver heatmap, panel 2" style="width:100%; height:560px;"></div>
        <div id="sender-matrix-plot-2" role="img" aria-label="Sender by receiver heatmap, panel 3" style="width:100%; height:560px;"></div>
      </div>
    </div>
  </div>
  <div id="tab-temporal" class="tab-panel" role="tabpanel" aria-labelledby="tabbtn-temporal" hidden>
    <div class="card">
      <div class="detail-chips">
        <label>Level:
          <select id="tm-level" aria-label="Temporal aggregation level" title="Switch between kinase-level and pathway-backbone-level temporal summaries.">
            <option value="kinase">Kinase</option>
            <option value="backbone">Backbone</option>
          </select>
        </label>
        <label id="tm-metric-label">Metric:
          <select id="tm-metric" aria-label="Temporal backbone metric" title="Choose the backbone summary metric plotted across disease timepoints.">
            <option value="count">Count</option>
            <option value="mean_score">Mean observed score</option>
            <option value="mean_tpds">Mean |TPDS|</option>
            <option value="pct_up">% upregulated</option>
          </select>
        </label>
        <label id="tm-tissue-label">Tissue:
          <select id="tm-tissue" aria-label="Temporal kinase cell type scope" title="Restrict kinase temporal counts by top attributed cell type or tissue class."></select>
        </label>
        <label id="tm-score-label">|TPDS| &geq;
          <input id="tm-score-min" type="number" step="0.1" min="0" value="0"
                 aria-label="Minimum absolute TPDS for backbone-mode rows"
                 title="In backbone mode, drop rows whose contrast |TPDS| falls below this value before counting or averaging."
                 style="width:64px;">
        </label>
        <span class="muted" id="tm-subtitle"></span>
      </div>
      <div id="temporal-plot" role="img" aria-label="Temporal summary plot" style="width:100%; height:480px;"></div>
    </div>
  </div>
  <div id="tab-additivity" class="tab-panel" role="tabpanel" aria-labelledby="tabbtn-additivity" hidden>
    <div class="card">
      <div class="detail-chips">
        <label>Level:
          <select id="add-level" aria-label="Additivity aggregation level" title="Switch between kinase NES additivity and backbone observed-score additivity.">
            <option value="kinase">Kinase</option>
            <option value="backbone">Backbone</option>
          </select>
        </label>
        <label>Timepoint:
          <select id="add-tp" aria-label="Additivity timepoint" title="Show one timepoint or all three timepoints side by side.">
            <option value="ALL">All (2/4/6mo)</option>
            <option value="2mo">2mo</option>
            <option value="4mo">4mo</option>
            <option value="6mo">6mo</option>
          </select>
        </label>
        <label>Score &geq;
          <input id="add-score-min" type="number" step="0.1" min="0" value="0"
                 aria-label="Minimum observed score for additivity rows"
                 title="Drop backbones whose observed score in either component contrast falls below this value."
                 style="width:64px;">
        </label>
        <span class="muted" id="add-subtitle"></span>
      </div>
      <div id="add-plot" role="img" aria-label="Additivity scatter plot" style="width:100%; height:520px;"></div>
      <div class="muted" id="add-stats"></div>
    </div>
  </div>
  <div id="tab-temporalv2" class="tab-panel" role="tabpanel" aria-labelledby="tabbtn-temporalv2" hidden>
    <div class="card">
      <div class="detail-chips" id="tv2-presets" style="flex-wrap:wrap;">
        <span class="muted" style="margin-right:6px;">Presets:</span>
        <button class="chip" data-tv2-preset="bulk_only">Bulk only</button>
        <button class="chip" data-tv2-preset="bulk_corrob_contest">Bulk · Corroborated · Contested</button>
        <button class="chip" data-tv2-preset="bulk_vs_decomp">Bulk vs Decomp-summed</button>
        <button class="chip" data-tv2-preset="bulk_attr_vs_decomp">Bulk · Bulk+attr≥high · Decomp</button>
        <button class="chip" data-tv2-preset="celltype_sweep">Per-cell-type sweep</button>
        <span style="flex:1;"></span>
        <button class="chip" id="tv2-add-series" title="Add a new series row">+ Add series</button>
        <button class="chip" id="tv2-clear" title="Remove all series">Clear</button>
        <label class="muted" style="margin-left:8px;" title="When on, every subplot uses the same y-axis range so heights are visually comparable across series.">
          <input type="checkbox" id="tv2-share-y"> shared y-axis
        </label>
      </div>
      <div id="tv2-series-list" style="display:flex; flex-direction:column; gap:4px; margin:8px 0;"></div>
      <div class="muted" id="tv2-subtitle" style="margin:4px 0 8px 0;"></div>
      <div id="tv2-plot" role="img" aria-label="Temporal series builder plot" style="width:100%;"></div>
      <details style="margin-top:10px;">
        <summary class="muted">How this view counts kinases</summary>
        <div class="muted" style="margin-top:6px; line-height:1.5;">
          <p>Each <b>series</b> defines a predicate over (kinase, contrast) pairs. Bar height = <b>count of unique kinases</b> passing the predicate at that (genotype, timepoint). Series stack as small multiples (one row per series).</p>
          <ul style="margin:4px 0 0 16px; padding:0;">
            <li><b>Layer = bulk</b>: kinase passes if its bulk-MEA FDR at this contrast &lt; gate.</li>
            <li><b>Layer = decomp</b>: kinase passes if at least one cell type in scope has decomp FDR &lt; gate (sign-agree with bulk optional).</li>
            <li><b>Layer = bulk ∩ decomp</b>: both must pass; "Agree" requires NES sign match.</li>
            <li><b>Layer = bulk \\ decomp</b>: bulk passes AND no cell type in scope passes decomp at the gate (bulk-only signal).</li>
            <li><b>Sign = signed</b>: split bars (up at +y, down at −y). <b>up / down / either</b>: only one direction or unsigned total.</li>
          </ul>
          <p style="margin-top:6px;">Cell-type scope is one of: <b>any (OR)</b> across the 24 decomp-present WMB classes, or a single class. The decomp pipeline does not cover the remaining 10 WMB classes (sampling gaps).</p>
        </div>
      </details>
    </div>
  </div>
  <div id="tab-methods" class="tab-panel" role="tabpanel" aria-labelledby="tabbtn-methods" hidden>
    <iframe id="methods-frame" src="pipeline_overview.html"
      style="width:100%;height:calc(100vh - 120px);border:1px solid var(--border);
             border-radius:4px;background:#fff;"
      title="Pipeline overview and methods"></iframe>
  </div>
</main>
<div id="drawer-resizer" title="Drag to resize"></div>
<button id="howto-drawer-toggle" type="button" aria-label="Toggle how-to drawer" aria-expanded="false" title="Show/hide the 'How to read' drawer">?</button>
<aside id="howto-drawer"></aside>
</div>
<aside id="glossary-panel" aria-hidden="true">
  <h3>Glossary</h3>
  <dl>
    <dt>Backbone</dt><dd>A receiver-specific Receptor &rarr; EM &rarr; Target route. It summarizes the receiver-side signaling route shared across one or more sender ligands.</dd>
    <dt>Receiver</dt><dd>The cell type whose receptor, EM, and target genes define the intracellular side of the modeled pathway.</dd>
    <dt>Sender</dt><dd>The cell type providing the ligand side of a sender-to-receiver pathway.</dd>
    <dt>TPDS</dt><dd>Transcript-level pathway differential score: modeled disease-associated change in log signaling probability. Positive and negative signs indicate opposite directions of modeled signaling change.</dd>
    <dt>Passed both nulls</dt><dd>For a specific backbone and contrast, both permutation checks passed: kinase enrichment and receiver-specific wiring.</dd>
    <dt>Passing contrasts</dt><dd>Count of genotype-by-timepoint contrasts, out of 9, where the backbone passed both nulls. A value of 3 means three contrasts passed, not three null models.</dd>
    <dt>NES</dt><dd>Normalized Enrichment Score (MEA on stoichiometry).</dd>
    <dt>FDR</dt><dd>False discovery rate used to threshold kinase MEA significance.</dd>
    <dt>Pathway support</dt><dd>Backbone-level provenance derived from full pathways: expression-confirmed, kinase-imputed, or mixed when both appear across contributing paths or contrasts.</dd>
    <dt>Support</dt><dd>Kinase contribution to a pathway backbone edge; larger absolute values indicate stronger phosphoproteomic support for that modeled route.</dd>
    <dt>Observed score</dt><dd>Permutation-tested pathway score used for backbone-level additivity and graph edge summaries.</dd>
    <dt>Additivity</dt><dd>Comparison of the observed ApTt signal with the App plus Tau expectation. Above the diagonal is stronger than additive; below is weaker than additive.</dd>
    <dt>Contrast</dt><dd>disease &times; timepoint: App/Tau/ApTt at 2/4/6mo.</dd>
  </dl>
</aside>

<script type="application/json" id="payload-data">__PAYLOAD_SENTINEL__</script>
<script>
"use strict";

// ---------------------------------------------------------------------------
// Payload
// ---------------------------------------------------------------------------
const PAYLOAD = JSON.parse(document.getElementById("payload-data").textContent);
const META = PAYLOAD.meta;
const CONTRASTS = META.contrasts;
const RECEIVERS = PAYLOAD.celltypes.name;
const TISSUE_CAT = PAYLOAD.celltypes.tissue_category;
const DISEASE_COLORS = META.diseaseColors;
const PATHWAY_EVIDENCE_LABELS = {
  "expression-confirmed": "Expression-confirmed",
  "kinase-imputed": "Kinase-imputed",
  "mixed": "Mixed",
};

// ---------------------------------------------------------------------------
// Store — reducer-style with {selection, filters, view} slices
// ---------------------------------------------------------------------------
const INITIAL_STATE = {
  selection: { kinase:null, backbone:null, celltype:null },
  filters:   { contrast:"ALL", direction:"ALL", receiver:"ALL", sender:null,
               pathwayEvidence:"ALL", fdr:0.25, score:0.0, graphNodeIds:null,
               tpdsSig:"OFF" },
  view:      { activeTab:"signal", overviewMode:"count",
               overviewSort:"tissue", glossaryOpen:false,
               graphLayout:"concentric", graphMinDegree:1,
               graphGenotype:"App", graphTimepoint:"2mo",
               graphTpdsMin:0, graphTopN:null,
               senderMatrixMode:"count",
               senderMatrixAxis:"timepoint", senderMatrixAnchor:"ApTt",
               senderMatrixLastAnchorByAxis:{ genotype:"2mo", timepoint:"ApTt" },
               kinaseAuditTab:"measurement-trace",
               temporalLevel:"kinase", temporalMetric:"count",
               temporalTissue:"ALL",
               additivityLevel:"kinase", additivityTimepoint:"ALL",
               temporalScoreMin:0, additivityScoreMin:0, pathwayScoreMin:0 },
};

const _clone = (typeof structuredClone === "function")
  ? structuredClone
  : (v) => JSON.parse(JSON.stringify(v));

function reducer(state, action) {
  const s = _clone(state);
  if (action.type === "SET_FILTER") s.filters[action.key] = action.value;
  else if (action.type === "SET_SELECTION") s.selection[action.key] = action.value;
  else if (action.type === "SET_VIEW") s.view[action.key] = action.value;
  else return state;
  return s;
}

const Store = (function(){
  let state = _clone(INITIAL_STATE);
  const subs = [];
  return {
    get state() { return state; },
    subscribe(fn) { subs.push(fn); return () => {
      const i = subs.indexOf(fn); if (i >= 0) subs.splice(i, 1);
    }; },
    dispatch(action) {
      const prev = state;
      const next = reducer(state, action);
      if (next === prev) return;
      state = next;
      for (const fn of subs) fn(next, prev);
    },
  };
})();
window.Store = Store;  // expose for console smoke test

// ---------------------------------------------------------------------------
// Canonical metric glossary — single source of truth for tooltips, column
// header labels, and the per-tab "How to read" drawer. Static HTML uses
// `data-metric="<key>"` to reference an entry; applyMetricTooltips() stamps
// the .short text into `title=` at boot. Dynamic render functions read
// METRIC_DEFS[key].short directly.
// ---------------------------------------------------------------------------
const METRIC_DEFS = {
  // Global filters
  contrast: {
    label: "Contrast",
    short: "Disease-by-timepoint comparison (e.g. App_4mo). Pick one to scope panels that need a single contrast; All shows pooled views where supported.",
    howToRead: "Pick a contrast first; the rest of the bar narrows from there." },
  direction: {
    label: "Direction",
    short: "Up- vs down-regulated in disease. Filters by signed TPDS for pathways and signed NES for kinases.",
    howToRead: "Use to isolate gain-of-activity vs loss-of-activity drivers." },
  receiver: {
    label: "Receiver",
    short: "Downstream cell type that hosts the pathway. Restricts backbones to one receiver.",
    howToRead: "Useful when investigating a single cell type's signaling." },
  pathwayEvidence: {
    label: "Support",
    short: "How a backbone's chain was assembled: every protein detected, kinase-imputed, or mixed.",
    howToRead: "Expression-confirmed across multiple contrasts is the strongest evidence; imputed is exploratory." },
  fdr: {
    label: "FDR",
    short: "False-discovery-rate threshold for significant kinase activity (NES vs WT).",
    howToRead: "Lower = stricter. Default 0.25 follows GSEA convention." },
  score: {
    label: "|Score|",
    short: "Minimum absolute pathway score (TPDS or observed) to keep a backbone.",
    howToRead: "Raise to focus on high-magnitude pathways." },

  // Kinase explorer columns
  kinaseName:    { label: "Kinase",        short: "Kinase identifier from the MEA / integration tables." },
  kinaseFamily:  { label: "Family",        short: "Kinase family annotation." },
  kinaseGene:    { label: "Gene",          short: "Gene symbol associated with the kinase." },
  nSig:          { label: "Sig vs WT",     short: "Number of contrasts where this kinase's MEA FDR is below the header threshold." },
  peakNES:       { label: "Peak NES",      short: "Largest |NES| across contrasts. Sign indicates direction." },
  topCelltype:   { label: "Top cell type", short: "Top attributed receiver cell type from the attribution evidence table." },
  highConfAttr:  { label: "Conf",          short: "Whether the kinase has high-confidence cell-type attribution." },
  nBackbones:    { label: "#Backbones",    short: "Number of distinct pathway backbones with significant support from this kinase, across all contrasts." },

  // Pathway browser columns
  receiverCol:     { label: "Receiver",         short: "Receiver cell type for the backbone." },
  receptorCol:     { label: "Receptor",         short: "Receptor gene in the backbone." },
  emCol:           { label: "EM",               short: "Extracellular-matrix or intermediate molecule in the backbone." },
  targetCol:       { label: "Target",           short: "Downstream target gene in the backbone." },
  tpds:            { label: "TPDS",
                     short: "Transcript-level pathway differential score for the selected contrast (max |TPDS| when All is selected).",
                     howToRead: "Magnitude tells you how strongly the chain shifts; sign tells you which way." },
  passingContrasts:{ label: "Passing contrasts",
                     short: "Genotype-by-timepoint contrasts where this backbone passed both permutation nulls.",
                     howToRead: "More contrasts = more reproducible. Use the contrast-set chips above to combine exact sets." },
  nSenders:        { label: "Senders",          short: "Number of significant sender cell types detected for this backbone." },
  maxAbsTpds:      { label: "Max |TPDS|",       short: "Largest absolute TPDS observed across contrasts." },

  // Pathway-detail h4 sections
  passedNulls:    { label: "Passed both nulls by contrast",
                    short: "Conditions where this pathway passed both significance tests (kinase-enrichment null and receiver-specific wiring null).",
                    howToRead: "More chips = more reproducible. Only pathways passing in ≥1 contrast appear in the viewer." },
  pathwaySupportH:{ label: "Pathway support by contrast",
                    short: "Whether each chain step was directly measured or imputed, per contrast.",
                    howToRead: "Expression-confirmed across multiple contrasts is the strongest evidence." },
  tpdsCross:      { label: "TPDS across contrasts",
                    short: "Signed pathway score per contrast.",
                    howToRead: "Red = up in disease, blue = down. Black outline marks contrasts that passed both nulls — those bars are the trustworthy ones." },
  drivingKinasesH:{ label: "Driving kinases",
                    short: "Kinases ranked by how much signal they push into this pathway.",
                    howToRead: "Top rows are the strongest driver candidates. Direction tells you whether the drive is up or down in disease." },

  // Driving-kinase columns
  support:         { label: "Support",
                     short: "Total signal a kinase pushes into this pathway. Bigger = stronger driver.",
                     howToRead: "Use this to rank top driver candidates." },
  drivingDirection:{ label: "Direction",
                     short: "Signed Support: + = more active in disease, − = less, ~0 = mixed evidence.",
                     howToRead: "High Support + strong sign = clean driver. Near-zero relative to Support = weaker candidate." },
  trend:           { label: "Trend",
                     short: "Quick-read direction: ↑ mostly up, ↓ mostly down, — balanced. Counts in parens are (up-evidence / down-evidence).",
                     howToRead: "Counts evidence, not magnitude — use Direction for magnitude." },
};

function _metricShort(key) {
  const m = METRIC_DEFS[key];
  return m ? m.short : "";
}

// Stamp data-metric -> title on every element with a known key. Idempotent;
// safe to call after dynamic re-renders.
function applyMetricTooltips(root) {
  const scope = root || document;
  scope.querySelectorAll("[data-metric]").forEach(el => {
    const key = el.dataset.metric;
    const m = METRIC_DEFS[key];
    if (m && m.short) {
      const raw = el.dataset.col || key;
      el.title = `Display label: ${m.label || el.textContent.trim()}\nRaw column: ${raw}\nDefinition: ${m.short}`;
      el.setAttribute("aria-label", el.title);
    }
  });
}
window.applyMetricTooltips = applyMetricTooltips;

// ---------------------------------------------------------------------------
// Per-tab "How to read" drawer content. Each entry distills purpose,
// primary-view orientation, metric cues (joined with METRIC_DEFS), and
// conclusions. Keep copy declarative — don't repeat tab labels.
// ---------------------------------------------------------------------------

const TAB_GUIDE = {
  signal: {
    preamble: "Rows are receiver cell types in the cortex. Columns are nine disease contexts: three genotypes (App, Tau, and the App-Tau double knock-in) measured at three timepoints (2, 4, and 6 months). Each cell's color encodes the number of receptor → effector → target gene chains, inside that receiver cell type, that the analysis flagged as disease-linked under that genotype and timepoint. Brighter cells mean more flagged chains; blank cells mean none cleared the test.",
    method: [
      "Kinases are enzymes that phosphorylate proteins, so changes in kinase activity show up as changes in how much of a particular site on a particular protein is phosphorylated. The phosphoproteomics in this study measured those phosphorylation levels across thousands of sites in App, Tau, and ApTt (App-Tau double knock-in) mice and compared them with controls. Combining the sites that moved with already-published kinase–substrate relationships, the analysis inferred which kinases must be unusually active or inactive to explain the observed pattern in each genotype-by-timepoint combination.",
      "Separately, single-nucleus RNA-seq identified which receptors, internal effector molecules, and target genes are expressed by each receiver cell type in the cortex. For every cell type, the analysis listed every receptor → effector → target chain in which the cell expresses all three proteins — chains the cell type could plausibly run as a signaling route.",
      "For each chain, the analysis asked whether the kinases flagged as disease-active in that genotype-by-timepoint context happen to be the same kinases that phosphorylate the receptor, effector, or target proteins in the chain — more often than they would if we drew kinases at random. A second check repeated the question with cell-type labels shuffled, asking whether the chain's kinase support is specific to this receiver or could come from any cell type. A chain is counted on the heatmap only if both checks gave positive results at FDR < 0.25 (false-discovery rate; fewer than one in four flagged chains is expected to be a chance result).",
    ],
    shows: {
      lead: "A bright row means one cell type carries flagged chains across many disease contexts — broadly affected. A bright column means one disease context produces flagged chains in many cell types — a widespread effect. An isolated bright cell points to a context-specific lead.",
      bullets: [
        "The double genotype (App-Tau, abbreviated ApTt) drives 32,356 chain-by-context entries — more than App and Tau combined. 25,839 of those chains never appear in App or Tau alone, so the combined pathology produces signaling-chain disturbance that neither single pathology generates by itself.",
        "App ramps up then resolves: 6,854 chains at 2 months, peaking at 9,726 at 4 months, declining to 5,644 at 6 months. The amyloid response concentrates at mid-disease.",
        "Tau is front-loaded: 10,027 chains at 2 months, then zero at 4 months and 6 months. The blank Tau columns are not biological silence. At those later timepoints, 152 then 180 different kinases were flagged as disease-active — nearly half of all kinases in the reference library. When that many kinases look active everywhere, a randomly-drawn kinase set overlaps a chain's substrates almost as well as the truly disease-active set, and no individual chain can pass the over-representation test. Tau_2mo (the Tau genotype at 2 months) passes because only 74 kinases are flagged as active, leaving room for specific chains to score above random. Tau biology starts focused and broadens until specific chains can no longer be distinguished.",
        "L5 IT is the sharpest Tau-early receiver: 2,987 Tau_2mo (Tau genotype, 2 months) chains and 2,763 ApTt_2mo (App-Tau double knock-in, 2 months) chains converge there, against 1,410 App chains summed across all three timepoints.",
        "Lamp5 Lhx6 is the most broadly affected receiver — 13,827 chains across all genotypes — making it a convergence point regardless of which pathology drives the disease.",
      ],
    },
    howTo: "Click any cell to pin that receiver as a filter; the selection carries through to every other tab, so the same cell type becomes the subject of the Pathway, Kinase, and Graph drill-downs. The color scale is log-compressed, so a block twice as bright does not represent twice as many chains. Treat chain count as a prioritization signal rather than a measure of biological importance — a single chain with a strong disease-vs-control phosphorylation shift is often more informative than many weakly-supported ones.",
    conclusions: [
      "Disease-linked signaling-chain disturbance in this cohort concentrates in the double genotype and arrives early. The combined App-Tau pathology produces the broadest disturbance and the most genotype-unique chains; Tau drives an early, focused signal that broadens until individual chains can no longer be resolved by 4 months; App rises and falls across the time course. Combined with the receiver concentration in Lamp5 Lhx6 and L5 IT, the map points the first round of follow-up toward early-disease, multi-genotype effects on specific neuronal subclasses rather than a uniform pan-cell-type response.",
    ],
    toggles: [
      { name: "Contrast filter", desc: "pick one disease context (one genotype at one timepoint) to focus the map on a single column." },
      { name: "Direction mode", desc: "switch from \"any flagged chain\" to a directional score that separates chains where phosphorylation went up in disease from chains where it went down, useful for asking whether a bright cell represents activated or suppressed signaling." },
      { name: "FDR threshold (false-discovery rate, the proportion of flagged chains expected to be chance results)", desc: "raise to require stronger statistical support, lower to capture weaker signal. The default of 0.25 is hypothesis-generation territory; 0.10 is closer to confirmatory." },
    ],
  },
  senders: {
    preamble: "Three 22×22 sender-by-receiver grids are shown side-by-side, each a real disease contrast — no averaging across panels. Rows are sender cell types; columns are receiver cell types. Both axes list the same 22 cortical cell types, since any cell type can play either role. Each cell's color encodes the number of receptor → effector → target gene chains, in that panel's disease context, in which the ligand at the start of the chain came from the sender cell type and the receptor that catches it sits on the receiver cell type. The Compare control above the grid sets which dimension is varied across the three panels: hold the genotype fixed and vary the timepoint, or hold the timepoint fixed and vary the genotype.",
    method: [
      "A signaling chain in this analysis runs from one cell type to another: the sender cell type expresses and releases a ligand, the receiver cell type expresses a receptor that binds it, an internal effector molecule in the receiver passes the signal along, and a target gene at the end of the chain is switched on or off. Single-nucleus RNA-seq identified which cell types express which ligands, receptors, effectors, and target genes; for each sender-receiver pair, the analysis listed every chain in which the sender expresses the ligand and the receiver expresses the receptor, effector, and target.",
      "Each chain was then filtered the same way as in the Signal Map. Phosphoproteomics measured which protein sites moved up or down in the chosen disease genotype-and-timepoint compared with controls. Combining the moved sites with already-published kinase–substrate relationships, the analysis inferred which kinases must be unusually active or inactive to explain the pattern. For each chain it asked whether those disease-active kinases happen to be the same kinases that phosphorylate the receptor, effector, or target proteins in the chain — more often than they would if we drew kinases at random — and a second check repeated the question with cell-type labels shuffled. Chains that passed both checks at FDR < 0.25 (false-discovery rate; fewer than one in four flagged chains is expected to be a chance result) are counted in the cell for that sender-receiver pair.",
    ],
    shows: {
      lead: [
        "Within any one panel, a dense row means one sender cell type is the source of many flagged chains landing across many receiver cell types — a broadly broadcasting cell in that disease context. A dense column means one receiver cell type is on the receiving end of flagged chains from many senders — a cell with broad incoming remodeling. Bright cells on the diagonal point to within-cell-type signaling, where the same cell type plays both roles. The point of showing three panels at once is comparison: read across the row of panels and ask which sender or receiver patterns persist, change, or appear only in one slice.",
        "The default view (ApTt at 2mo, 4mo, 6mo) was chosen because the trajectory of the double genotype across time is the most biologically loaded read in this dataset. Three other reading paths follow from flipping the axis or stepping the anchor:",
      ],
      bullets: [
        "App across time (App at 2mo, 4mo, 6mo). Step the anchor to App. Watch which sender rows thicken into mid-disease at 4 months and collapse back at 6 months — those are the cell types whose ligand output rises and resolves with the amyloid response. Receiver columns that stay dense across all three panels mark cell types whose incoming remodeling is sustained.",
        "Tau across time (Tau at 2mo, 4mo, 6mo). Step the anchor to Tau. Tau_4mo and Tau_6mo render as blank panels for the same reason as in the Signal Map: at those timepoints nearly half the kinase library is flagged as disease-active, so no individual sender-receiver pair stands above the random baseline. The interpretable Tau structure is at 2 months alone.",
        "Cross-section at one timepoint (App, Tau, ApTt at 2mo). Flip the axis to Compare genotypes and set the anchor to 2mo. The early-disease snapshot lets you ask which sender or receiver patterns are shared across all three genotypes versus which are genotype-specific.",
        "Persistent dense rows or columns across all three panels — whatever the axis — identify sender or receiver cell types whose role is structural rather than stage- or genotype-specific.",
      ],
    },
    howTo: "The Compare control sets the comparison axis: Compare timepoints holds the genotype fixed (the anchor) and shows that genotype at 2mo, 4mo, 6mo; Compare genotypes holds the timepoint fixed and shows App, Tau, ApTt at that timepoint. Step the anchor with ← and →; flip the comparison axis with ↑ or ↓. Each axis remembers the last anchor you used on it, so flipping back returns to where you were. The color scale is pinned across all nine contrasts, so brightness is directly comparable across panels and across anchor steps — a cell that looks faint in one panel really is fainter than the same cell in another, not just rescaled. Click any cell to pin its receiver as a global filter; the choice carries through to the Pathway, Kinase, and Graph drill-downs. The color scale in count mode is log-compressed, so a cell twice as bright does not represent twice as many chains. A dense cell can come from many distinct chains or from repeated use of a few highly-shared ligand–receptor combinations — to distinguish, drill into that cell's chains in the Pathway tab.",
    conclusions: [
      "Showing three real disease contrasts side-by-side, never an average, is the design choice that lets sender-receiver structure be read as a trajectory rather than a single snapshot. Senders with dense rows in all three panels are broadcasting cell types whose role does not depend on the dimension being varied; receivers with dense columns across all three are absorption hubs in the same sense. Stage-specific or genotype-specific patterns — a sender row dense only at 4 months, a receiver column that thickens only when amyloid pathology is added — are the most informative for asking what each cell type does at a particular disease moment.",
    ],
    toggles: [
      { name: "Compare", desc: "selects the axis varied across the three panels. Compare timepoints holds the genotype fixed (the anchor) and shows that genotype at 2mo, 4mo, 6mo; Compare genotypes holds the timepoint fixed and shows App, Tau, ApTt at that timepoint." },
      { name: "Anchor", desc: "the dimension held fixed across the three panels. Its options switch with the axis: when comparing timepoints the anchor is one of App, Tau, ApTt; when comparing genotypes it is one of 2mo, 4mo, 6mo. Each axis remembers the last anchor you used on it." },
      { name: "Mode", desc: "switch between count (number of flagged chains per sender-receiver pair, log-scaled so the brightest cells do not crowd out moderate ones) and direction (chains where disease-vs-control phosphorylation went up minus chains where it went down, useful for asking whether a pair is dominated by activation or suppression)." },
      { name: "Arrow keys", desc: "← and → step the anchor within the current axis; ↑ and ↓ flip the axis. The color scale is pinned across all nine contrasts so brightness is comparable as you step." },
    ],
  },
  temporal: {
    preamble: "For each genotype, three points trace how the kinase or pathway signal evolves across 2, 4, and 6 months of age. In kinase mode, the y-axis is the count of kinases whose substrate phosphosites shift coherently in disease versus control at that timepoint. In backbone mode, the y-axis is the count of receptor → effector → target chains that cleared both permutation null tests at that timepoint. Three colored lines, one per genotype (App, Tau, ApTt), so flat versus rising versus peaked trajectories can be compared in one read.",
    method: [
      "Phosphoproteomics measured how much of each protein site is phosphorylated in each disease mouse line at each timepoint, normalized to the parent protein's abundance so changes in total protein do not show up as apparent kinase activity changes. Combining the sites that moved with already-published kinase–substrate relationships, the analysis inferred which kinases must be unusually active or inactive to explain the pattern in each genotype-by-timepoint context — those are the kinases counted in kinase mode at that timepoint.",
      "For backbone mode, single-nucleus RNA-seq listed every receptor → effector → target chain in which the receiver cell type expresses all three proteins. For each chain the analysis asked whether the disease-active kinases happen to be the same kinases that phosphorylate the receptor, effector, or target proteins in the chain — more often than they would if we drew kinases at random — and a second check repeated the question with cell-type labels shuffled. Chains that passed both checks at FDR < 0.25 (false-discovery rate; fewer than one in four flagged chains is expected to be a chance result) are counted in backbone mode at that timepoint.",
    ],
    shows: {
      lead: "Backbone mode exposes two readouts that are designed to disagree in the diffuse phase. The count line — passing chains per timepoint — drops to zero in late Tau because the chain test loses its statistical handle when the disease-active kinase set grows toward half the library. The mean |TPDS| line — pathway dysregulation magnitude across every enumerated chain in the receiver slice, with no significance gate — keeps climbing across the same timepoints, because TPDS measures how much pathway flux is shifted regardless of whether any individual route can be picked out from random. The two readouts read from different data: count and mean observed score iterate the chain-test-passing payload (the same chains the rest of the viewer shows), while mean |TPDS| and percent upregulated read a build-time summary computed across every chain the cell type expresses, including the diffuse-phase chains the viewer otherwise filters out. Read count for 'how many specific routes resolve' and mean |TPDS| for 'how much pathway burden is present'; their divergence is the focal-to-diffuse signature, not a contradiction. Pinning a specific sender or selecting a kinase falls back to the chain-test-passing payload for both metrics — magnitude in those scoped views is necessarily restricted to passing chains.",
      bullets: [
        "Kinase counts and chain counts often move in opposite directions across the time course — most visibly in Tau, where 74 disease-active kinases at 2mo grow to 152 at 4mo and 180 at 6mo while passing chains collapse from 10,027 to zero. This is not a contradiction. The two modes ask different questions: kinase mode counts enzymes whose substrate sites cluster coherently in disease (an enrichment test that gets stronger as more enzymes are weakly perturbed), while backbone mode counts chains where the disease-active kinases are specifically the ones that phosphorylate this chain's proteins more often than a random kinase draw would (an over-representation test that loses power as the active set approaches half the kinase library). Climbing kinases plus collapsing chains describes a shift from focal early signaling — a few specific axes resolvable by the chain test — to diffuse late remodeling that the chain test cannot pin to particular routes. Treat the divergence as the trajectory's headline read, not an artifact.",
        "App rises and falls in backbone mode: 6,854 chains at 2mo, peaking at 9,726 at 4mo, declining to 5,644 at 6mo. The amyloid response concentrates at mid-disease.",
        "Tau diverges between count and magnitude. The count metric is front-loaded: 10,027 passing chains at 2mo, then zero at 4mo and 6mo — the chain test cannot resolve specific routes once randomly drawn kinase sets overlap any chain almost as well as the truly disease-active set. The mean |TPDS| metric, applied across all enumerated chains rather than only passing ones, climbs across the same trajectory: median |TPDS| 0.014 → 0.017 → 0.018 and p95 0.097 → 0.106 → 0.120 from 2mo to 6mo. The blank count line is the diffuse-phase symptom; the climbing TPDS line is what's actually happening underneath.",
        "ApTt (App-Tau double knock-in) front-loads: 15,610 chains at 2mo, declining through 4mo (10,048) and 6mo (6,698). The trajectory resembles Tau's early-loading more than App's gradual build, suggesting the double genotype's early dynamic is largely Tau-shaped.",
        "Restricting to one receiver tests whether the genotype-specific timing is a property of one cell type or a cohort-wide pattern.",
      ],
    },
    howTo: "Switch between kinase mode and backbone mode with the toggle above the plot. In backbone mode the metric selector chooses between four readouts with deliberately different gating: 'count' is significance-gated (passing chains only); 'mean observed score' is significance-gated by definition (observed score is undefined for non-passing chains); 'mean |TPDS|' and '% upregulated' are not significance-gated and operate on every enumerated chain in the receiver/sender slice. The hover on any line shows both the passing-chain count and the chain-with-TPDS count so you can see how the two are weighted. The local |TPDS| ≥ control thins all metrics to chains whose mean total pathway dysregulation score (TPDS — the integrated shift in modeled signaling probability for the chain in disease versus control) clears the chosen value; default zero keeps every chain. The global FDR slider (false-discovery rate; fewer than one in four flagged chains or kinases is expected to be a chance result) tightens upstream kinase selection. Restrict by receiver in the global filter bar to ask whether timing is cell-type-specific. A flat count line can mean stable remodeling or absence of signal — read mean |TPDS| at the same timepoints to distinguish the two.",
    conclusions: [
      "The trajectory's two modes stratify the disease into a focal phase and a diffuse phase. In the focal phase — App across its full course, Tau at 2mo, ApTt at 2mo — the kinase landscape is specific enough that the chain test resolves particular receptor → effector → target routes, and kinase counts and chain counts move together. In the diffuse phase — Tau at 4mo and 6mo, where kinase counts climb past half the library — the chain test loses its statistical handle and chain counts collapse despite continued (broader, weaker) phosphoproteomic remodeling. Pathway remodeling is also not synchronous across genotypes: App builds to a mid-disease peak and partially resolves; ApTt and Tau front-load at 2 months, with ApTt resembling Tau's early dynamic more than App's gradual build. The first follow-up is to read late-phase signal through TPDS magnitude on the Pathway tab — a magnitude-based score that does not depend on active-set specificity — rather than asking the chain count to describe a regime where it is by design uninformative.",
    ],
    toggles: [
      { name: "Mode", desc: "switch between kinase mode (count of kinases passing at each timepoint) and backbone mode (count of receptor → effector → target chains clearing both permutation tests)." },
      { name: "|TPDS| ≥", desc: "local cut on chains in backbone mode. Default zero counts every passing chain; raise to keep only chains whose mean total pathway dysregulation score clears the chosen value." },
      { name: "FDR threshold (false-discovery rate)", desc: "applies to upstream kinase or chain selection. Tighten to focus on robust signals; loosen to capture weaker early signals." },
      { name: "Receiver", desc: "restricts both modes to one receiver cell type. Use to ask whether the trajectory is shared across cells or specific to one." },
    ],
  },
  additivity: {
    preamble: "A scatter that asks whether the App-Tau double genotype (ApTt) behaves like the sum of App and Tau or whether the two pathologies interact. The y-axis is the signal in ApTt; the x-axis is the predicted signal if App and Tau add linearly. Each point is one kinase (in kinase mode) or one receptor → effector → target chain (in backbone mode), shown separately at 2, 4, and 6 months. The diagonal is pure additivity — points above mean the double genotype exceeds the prediction, points below mean it falls short.",
    method: [
      "For each kinase, the analysis took the kinase's enrichment score (NES — normalized enrichment score; positive means the kinase's substrates are more phosphorylated in disease, negative means less) in the App, Tau, and ApTt contrasts at one timepoint. The x-axis plots App's NES + Tau's NES; the y-axis plots ApTt's NES. A point on the diagonal would mean the double genotype's enrichment is what you'd get from stacking App and Tau side-by-side.",
      "In backbone mode, each point is one receptor → effector → target chain in the current filter, plotted with x = the chain's observed score in App + its observed score in Tau, and y = its observed score in ApTt, at one timepoint. The observed score is the chain's permutation-tested pathway score (mean kinase support across the chain's nodes). No FDR filter is applied to the scatter itself — every chain admitted by the active support and receiver filters contributes a point — so the cloud is dense and a sub-additive bias is read from the bulk distribution, not from individual labeled points. Distance above or below the diagonal is the deviation from the additive prediction.",
    ],
    shows: {
      lead: "Points above the diagonal are supra-additive — the double genotype produces more signal than App and Tau together would predict. Points below are sub-additive — the two pathologies partly cancel or share a saturating mechanism. Spread along the diagonal at any one timepoint reveals magnitude; consistency across all three timepoints reveals whether the interaction is a stable feature or a stage-specific one.",
      bullets: [
        "Kinase mode shows only kinases that pass FDR (false-discovery rate; fewer than one in four flagged kinases is expected to be a chance result) in at least one of App, Tau, or ApTt at the chosen timepoint, color-coded by which subset of contrasts they clear: App only, Tau only, ApTt only, or Multi (≥2). Kinases that fail FDR everywhere are not plotted, matching the filtering convention used by the rest of the viewer.",
        "At 2 months in backbone mode, the bulk of the chain cloud sits below the diagonal — the double genotype's per-chain pathway score is mildly sub-additive relative to App + Tau. A separate count-based view of the same effect: 15,610 chains pass FDR in ApTt against 16,881 expected from App's 6,854 + Tau's 10,027, a ratio of 0.92×.",
        "At 4 and 6 months, Tau contributes essentially zero passing chains and a near-zero observed-score distribution, so the predicted (App + Tau) score collapses to App's score and the scatter cannot distinguish additivity from independence. Read these timepoints as confirming the double genotype is not silenced by Tau pathology, not as evidence for or against synergy.",
        "The kinase-level and backbone-level scatters can disagree on the same entity: a kinase can be supra-additive in NES while its supported chains are sub-additive in observed score, or vice versa. Both readings are legitimate, because they measure different layers — kinase-level enrichment versus per-chain pathway score.",
      ],
    },
    howTo: "Switch between kinase mode and backbone mode with the toggle. The local score-min control thins to points where either App, Tau, or ApTt's signal magnitude clears the chosen value, removing low-signal noise from the diagonal cloud. The global FDR slider tightens upstream kinase or chain selection. Step through the three timepoints — 2mo, 4mo, 6mo — before treating any single point as a stable interaction; an interaction that holds at one timepoint and inverts at another is a stage-specific phenomenon, not an additivity failure of the double genotype.",
    conclusions: [
      "The most interpretable additivity reading is at 2 months, when both single genotypes generate their own passing chains and their sum is a meaningful prediction. The 0.92× backbone-level ratio there is a mild sub-additive signal, suggesting partial mechanistic overlap or competition for shared signaling machinery rather than true independence. At later timepoints, Tau's collapse to zero passing chains makes the prediction degenerate; ApTt tracking App is consistent with either additivity or the double genotype defaulting to the App-driven mid-disease arc. The kinase-level scatter at each timepoint nominates individual enzymes whose interaction signature should be cross-checked against their cell-type attribution and supported chains in the Kinase and Pathway tabs.",
    ],
    toggles: [
      { name: "Mode", desc: "kinase mode plots NES (normalized enrichment score) per kinase; backbone mode plots mean kinase support score across passing chains." },
      { name: "Score min", desc: "drops points whose App, Tau, or ApTt signal magnitude falls below the chosen value. Use to thin the cloud near the origin and emphasize interactions among strongly-active entities." },
      { name: "FDR threshold (false-discovery rate)", desc: "tightens upstream kinase or chain selection before the scatter is computed." },
    ],
  },
  kinase: {
    preamble: "A ranked table of the 240 kinases whose substrate phosphosites shift coherently in at least one disease contrast. Each row is one kinase. NES (normalized enrichment score) columns capture the direction and magnitude of that shift in each genotype-by-timepoint context. Cell-type columns place that activity onto cortical subclasses using independent transcriptomic evidence. The backbone count is how many passing receptor → effector → target chains the kinase appears among the inferred drivers of.",
    method: [
      "Phosphoproteomics measured how much of each protein site is phosphorylated in App, Tau, and ApTt (App-Tau double knock-in) mice at each timepoint, normalized to the parent protein's abundance so changes in total protein do not show up as apparent kinase activity changes. For each disease contrast, the analysis ranked every measured site by its disease-versus-control change and asked, for each kinase in the reference library, whether that kinase's known substrate sites cluster toward the top or bottom of the ranking more strongly than they would if we drew sites at random — a positive NES means the substrates concentrate among the upregulated sites, a negative NES means they concentrate among the downregulated sites.",
      "Independently, single-nucleus RNA-seq from a separate human Alzheimer's cohort and a mouse brain reference atlas provided per-cell-type expression and disease-direction concordance for each kinase; those become the cell-type columns. The backbone count comes from the same chain analysis used elsewhere in the viewer: a chain passing at FDR < 0.25 (false-discovery rate; fewer than one in four flagged chains is expected to be a chance result) and naming this kinase among the over-represented substrate-phosphorylators contributes one to its count.",
    ],
    shows: {
      lead: "NES columns answer where in the disease landscape the kinase's substrates are most coherently shifted. Cell-type columns answer where in the cortex the kinase's transcript is concordantly differentially expressed in disease. Backbone count answers how broadly the kinase appears among inferred drivers of passing chains — a structural prevalence signal, not a per-chain magnitude.",
      bullets: [
        "240 kinases pass FDR < 0.25 in at least one of nine disease contrasts. Of those, 124 (52%) follow a peaked trajectory — enrichment rises and falls across the time course — while only three remain sustained across all three timepoints of one genotype.",
        "Peak enrichment concentrates in the double genotype: 125 kinases peak in an ApTt (App-Tau double knock-in) contrast versus 69 in App and four in Tau alone. The strongest individual signals by NES magnitude — AKT1, AKT2, AKT3 — are negative and peak at App_4mo (App genotype, 4 months), meaning their substrate phosphorylation is reduced relative to protein abundance specifically in amyloid disease at mid-disease. This AKT hypoactivity signature is absent from the Tau genotype.",
        "The broadest backbone supporters — CAMK2D (15,028 chains), CDK1 (14,776), CHK1 (13,098) — have moderate NES across many contrasts. They are structural participants in many chains rather than strong disease-specific signals.",
        "High NES with weak cell-type attribution is not evidence against the kinase; it is evidence that the transcriptomic side has less to say about where it acts. The reverse is also true.",
      ],
    },
    howTo: "Sort by any column to surface kinases by enrichment magnitude, cell-type concordance, or backbone breadth. Click a row to pin that kinase across the viewer — its trajectory across timepoints opens in the side panel, the Pathway tab restricts to chains it drives, and any cell-type filter on Signal Map or Sender × Receiver applies the same constraint. The global FDR slider (false-discovery rate) tightens upstream kinase selection: at 0.25, roughly one in four flagged kinases is a false positive, which is hypothesis-generation territory; at 0.10, the count falls but each remaining kinase is closer to a confirmatory call.",
    conclusions: [
      "The peaked-trajectory majority is the headline structural feature of the kinase landscape — most disease-active kinases turn on and off with stage rather than accumulating. Concentration of peaks in the double genotype, combined with the App-specific AKT hypoactivity signature and the broad-but-moderate enrichment of CAMK2 / CDK1 / CHK1, points the first round of follow-up to two questions: what is the cell-type origin of the AKT suppression in App_4mo, and do the structural backbone supporters carry chain-level direction information that the per-kinase NES summary obscures. Both questions chain directly into the Pathway and Sender × Receiver tabs.",
    ],
    toggles: [
      { name: "FDR threshold (false-discovery rate)", desc: "sets the cutoff for which kinases enter the table. 0.25 is hypothesis generation; 0.10 is closer to confirmatory." },
      { name: "Receiver, Support", desc: "when set in the global filter bar, restrict the backbone count column to chains landing on that receiver or carrying that support type, so the rank reflects the kinase's role in the chosen subset rather than its total prevalence." },
    ],
  },
  pathway: {
    preamble: "A scrollable list of the receptor → effector → target chains that passed both permutation null tests in at least one disease contrast. Each row is one chain. TPDS (total pathway dysregulation score) columns measure how strongly the chain's modeled signaling probability shifts in each disease context — positive means more activity in disease, negative less. The Passing contrasts column lists the disease contexts where this chain cleared both null tests. Click a chain to expand its driving kinases — the kinases whose disease-active substrate signature put the chain over the threshold in each contrast.",
    method: [
      "Single-nucleus RNA-seq identified which receptors, intracellular effector molecules (EM — signaling components linking receptor binding to gene expression), and target genes are expressed by each receiver cell type. For every receiver, the analysis listed every receptor → EM → target chain in which the cell expresses all three components.",
      "Phosphoproteomics measured which protein sites moved up or down in each genotype-by-timepoint context, normalized to the parent protein's abundance so changes in total protein do not show up as apparent kinase activity changes. For each chain, the analysis asked whether the kinases inferred as unusually active in disease happen to be the same kinases that phosphorylate the receptor, effector, or target proteins in the chain — more often than they would if we drew kinases at random. A second test repeated the question with cell-type labels shuffled, asking whether the chain's kinase support is specific to its receiver or could come from any cell type. Chains that cleared both tests at FDR < 0.25 (false-discovery rate; fewer than one in four flagged chains is expected to be a chance result) appear here.",
      "TPDS is computed independently of the over-representation test: it integrates the modeled signaling-probability shift along the entire chain — receptor, effector, and target components together — into one signed magnitude per chain per contrast.",
    ],
    shows: {
      lead: "Each row is one chain that passed both null tests somewhere. The Passing contrasts column shows where; the TPDS columns show how strongly the chain's modeled signaling probability shifted there. The trajectory buttons let you ask which chains pass under specific contrast patterns — one genotype's three timepoints (App, Tau, or ApTt trajectory), one timepoint across all three genotypes (2mo, 4mo, or 6mo cross-section), or every contrast.",
      bullets: [
        "Of 55,859 unique chains in the dataset, 230 pass in all three genotypes — the most reproducible leads across disease contexts.",
        "Most chains are genotype-specific: 25,839 pass only in ApTt (App-Tau double knock-in), 16,232 only in App, 8,061 only in Tau. Genotype-specific chains are the hypothesis-rich zone — each clears stringent permutation thresholds in one context but not others.",
        "Recurrence stratifies quality: a chain passing in six of nine contrasts is more likely to reflect stable structural remodeling than one passing in one. Use the trajectory buttons to require multi-contrast passage before drilling in.",
        "TPDS magnitude and recurrence answer different questions. A chain can pass many contrasts at modest TPDS (broad reproducible structure) or pass one contrast at large TPDS (a single-context shift large enough that the test catches it on its own). Both are worth examining for different reasons.",
      ],
    },
    howTo: "Filter contrast patterns with the seven trajectory buttons above the table. The three trajectory buttons (App / Tau / ApTt) keep chains that pass any of that genotype's three timepoints. The three cross-section buttons (2mo / 4mo / 6mo) keep chains that pass any of that timepoint's three genotypes. The All button removes the contrast-pattern restriction. Tighten the local |TPDS| ≥ control to keep only chains whose maximum |TPDS| across passing contrasts clears the chosen value — a magnitude floor on top of the permutation-pass requirement. The global Receiver and Support filters restrict to one receiver cell type or to one chain-support category (expression-confirmed routes have direct transcriptomic evidence for receptor, effector, and target; kinase-imputed routes are inferred from substrate evidence). Click a row to open its driving kinases panel.",
    conclusions: [
      "The chain catalog is dominated by genotype-specific routes — the double genotype alone contributes nearly half the unique chains — with a small core of 230 chains reproducible across all three genotypes. Recurrence is the primary quality stratifier here; the seven trajectory buttons let you go after specific patterns (one genotype's full trajectory, one timepoint's cross-section, all contrasts) without juggling contrast lists by hand. From any row, the driving kinases panel names the enzymes responsible for the chain clearing the over-representation test in each contrast — that is the link back to the Kinase tab, and the path from a chain hypothesis to a testable molecular driver.",
    ],
    toggles: [
      { name: "Trajectory buttons", desc: "App / Tau / ApTt keep chains that pass any of that genotype's three timepoints. 2mo / 4mo / 6mo keep chains that pass any of that timepoint's three genotypes. All removes the restriction." },
      { name: "|TPDS| ≥", desc: "local magnitude floor. Keeps chains whose maximum |TPDS| across passing contrasts reaches the chosen value. Use to add a magnitude requirement on top of the permutation-pass requirement." },
      { name: "Receiver", desc: "global filter; restricts to chains whose receiver cell type matches." },
      { name: "Support", desc: "global filter; switches between expression-confirmed (direct transcriptomic evidence for receptor, effector, and target) and kinase-imputed (inferred from substrate evidence)." },
      { name: "FDR threshold (false-discovery rate)", desc: "applies to the upstream chain selection. Tighten to require stronger statistical support." },
    ],
  },
  graph: {
    preamble: "One genotype-by-timepoint snapshot at a time, built from the routes that passed both permutation null tests in that contrast. Nodes are the receptor, EM, and target genes of those routes; edges connect genes that co-appear in the same passing route. Pick a genotype with the dropdown above the graph and step through the three timepoints with the Timepoint control or arrow keys; each step is an independent snapshot, not a fade between time-collapsed views. Tau_4mo and Tau_6mo render empty because at those timepoints nearly half the kinase library is flagged as disease-active and no individual route stands above the random baseline.",
    method: [
      "Two upstream filters define the universe of routes shown. First, single-nucleus RNA-seq identified which cell types in the cortex express each receptor, intracellular effector, and target protein, and Incytr enumerated every chain in which the receiver cell type expresses all three components. Second, phosphoproteomics measured which protein sites moved up or down in the chosen disease genotype-and-timepoint compared with controls, and the analysis asked, for each chain, whether the kinases flagged as unusually active in disease happen to be the same kinases that phosphorylate the receptor, effector, or target proteins in the chain — more often than they would if we drew kinases at random. A second test repeated the question with cell-type labels shuffled. Only chains that passed both tests at FDR < 0.25 (false-discovery rate; fewer than one in four flagged chains is expected to be a chance result) flow into this view.",
      "The remaining filters are local rendering choices. Min degree drops nodes connected to fewer than that many surviving routes. The |TPDS| ≥ X cut hides edges whose mean total pathway dysregulation score (TPDS — the integrated shift in modeled signaling probability for the chain in disease versus control) falls below the chosen value; default zero shows everything in the passing-both-nulls universe, drag up to thin out weak-signal edges. The optional Max edges cap is a separate safety net: when on, only the top N edges by |TPDS| are drawn.",
    ],
    shows: {
      lead: [
        "Each edge in the graph encodes shared route membership: two genes connect when they appear together in a chain that passed both permutation tests at this snapshot. Edge color comes from the mean TPDS across those shared chains — red for routes pointing up in disease (more modeled signaling activity than control), blue for routes pointing down, grey near zero. Read the network for two things at once: structure (which genes converge into hubs, which sit at the periphery) and direction (whether those hubs sit on red, blue, or mixed-color edges).",
      ],
      bullets: [
        "A receptor (R-prefix) gene with high degree means many distinct routes start at the same incoming signal — a converging-input hub. A target (T-prefix) gene with high degree means many routes converge on the same downstream effector — a converging-output hub. Either pattern points to a small number of molecular focal points carrying the disease signal in this snapshot.",
        "EM (E-prefix) genes with very high degree should be interpreted with one caveat: the Incytr effector database is densely curated for some EM genes, so their connectivity partly reflects how many curated substrate links exist rather than how biologically central they are at this disease moment. Cross-check EM hubs against the Pathway tab.",
        "Stepping through the three timepoints of one genotype shows whether the network is structurally stable (the same hubs appear at each step) or stage-shifting (hubs appear, change, or disappear across 2mo → 4mo → 6mo). Switching genotypes asks whether App, Tau, and ApTt converge on the same molecular focal points or address disease through different routes.",
      ],
    },
    howTo: "Pick a genotype, then step the timepoint with the dropdown or with ← / → on the keyboard. Each step is a clean rebuild of the network for that single contrast. Click any node to focus its closed neighborhood — its direct neighbors stay coloured, everything else fades. Click empty space to clear the focus. The detail panel on the right shows how many passing chains the selected node sits in and the up-versus-down breakdown of those chains. Adjust min-degree to thin sparsely-connected genes, raise |TPDS| to drop weak-signal edges, or set Max edges as a hard rendering cap when a snapshot stalls. Layout choice is a presentation control, not an analytic one: concentric forces R → EM → T into rings, flow snaps the same three roles into columns, force-directed lets the network find its own layout based on edge counts.",
    conclusions: [
      "Each snapshot answers one question — which genes are tied together by passing-both-nulls routes in this genotype at this timepoint, and in what direction. Stepping the timepoint shows trajectory; switching the genotype tests whether the disease arc has a shared molecular substrate. Convergent routes sharing a receptor suggest a common incoming signal; convergent routes sharing a target suggest a common downstream effector; scattered routes with no shared hubs suggest broad remodeling without a single focal point. Empty graphs (notably Tau_4mo and Tau_6mo) are not a failure of the test — they are the predicted consequence of broad kinase activation overwhelming the per-chain over-representation signal, the same dynamic visible in the Signal Map and Sender × Receiver tabs.",
    ],
    toggles: [
      { name: "Genotype", desc: "selects which disease model's network is rendered: App, Tau, or ApTt (App-Tau double knock-in)." },
      { name: "Timepoint", desc: "selects which timepoint of the chosen genotype is rendered: 2mo, 4mo, or 6mo. Stepping forward or backward triggers a clean rebuild for the new snapshot." },
      { name: "Layout", desc: "presentation only; concentric arranges nodes in R → EM → T rings, flow snaps the three roles into columns, force-directed runs an unconstrained spring layout." },
      { name: "Min degree", desc: "drops nodes connected to fewer than this many passing routes. Raise to thin out genes that participate in only a few chains and emphasize convergence hubs." },
      { name: "|TPDS| ≥", desc: "hides edges whose mean |TPDS| falls below this value. Default zero shows everything in the passing-both-nulls universe; drag up to reveal only the strongest-signal edges." },
      { name: "Max edges", desc: "optional rendering cap. When set to a number, keeps only the top N edges by |TPDS|. Leave blank for no cap; flip on if a snapshot stalls or exceeds what the layout can handle clearly." },
      { name: "Arrow keys", desc: "← and → step the timepoint within the current genotype. Switch genotype with the dropdown." },
    ],
  },
  methods: {
    preamble: "This panel contains the long-form methods documentation: pipeline stages, statistical model specifications, metric definitions, and integration design decisions. It is a reference companion to the analytical tabs, not an analytical view itself.",
    purpose: "Long-form methods reference: pipeline stages, statistical models, and metric definitions in full detail.",
    primary: "Start with the Key viewer concepts and Stage 6 Incytr integration sections when a term in another tab needs more context. Stage 7 covers cross-pair aggregation and the backbone permutation tests.",
  },
};

function _escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
}

function _auditManifest() {
  return (PAYLOAD.audit_tables && PAYLOAD.audit_tables.tables) || {};
}

function _measurementTraceManifest() {
  return (PAYLOAD.audit_tables && PAYLOAD.audit_tables.measurement_trace) || {};
}

function _isLikelyNumericColumn(col) {
  const c = String(col).toLowerCase();
  return /(^n_|_n$|nes|^es$|fdr|p-value|pval|lfc|score|fold|value|_sn_mean$|site_id)/.test(c);
}

const MEA_PREP_COL_DEFS = {
  site_id: {label:"Site ID", definition:"Stable phosphosite identifier used to join site matrices and model outputs.", format:"integer"},
  gene_symbol: {label:"Gene", definition:"Gene symbol associated with the phosphosite.", format:"text"},
  motif: {label:"Motif", definition:"Peptide motif centered on the phosphorylated residue.", format:"text"},
  n_obs_stoich: {label:"N obs", definition:"Number of biological samples with usable stoichiometry for this site (site-level availability count).", format:"integer"},
  raw_lfc: {label:"Raw LFC", definition:"Site-level stoichiometry log fold change for the selected contrast (site_level_ols.stoich_lfc_<contrast>).", format:"float"},
  centered_lfc: {label:"Centered LFC", definition:"raw_lfc minus the contrast's median shift. Derived at view time.", format:"float"},
  clipped_lfc: {label:"Clipped LFC", definition:"centered_lfc clipped to the contrast's winsor bounds; the value passed to GSEA prerank. Derived at view time.", format:"float"},
  was_winsorized: {label:"Winsorized?", definition:"True when the centered LFC was clipped to the bounds.", format:"text"},
  rank_in_contrast: {label:"Rank", definition:"Descending rank of clipped_lfc across all ranked sites for the contrast (1 = most up-shifted; recomputed at view time).", format:"integer"},
  in_leading_edge: {label:"Leading edge?", definition:"Annotation from MEA output: true when the site's motif appears in this kinase's Leading substrates for the contrast.", format:"text"},
};

const MEA_CMP_COL_DEFS = {
  metric: {label:"Metric", definition:"MEA output metric being compared between tracks.", format:"text"},
  stoich: {label:"Stoichiometry (primary)", definition:"Value from mea_stoichiometry.csv for the selected kinase × contrast.", format:"text"},
  raw: {label:"Raw phospho (sensitivity)", definition:"Value from mea_raw_phospho.csv for the selected kinase × contrast. Empty rows mean the kinase has no row in the raw track for this contrast.", format:"text"},
  delta: {label:"Δ (stoich − raw)", definition:"Signed difference, stoichiometry minus raw. — for non-numeric metrics.", format:"text"},
};

function _auditColMeta(tableKey, raw) {
  if (tableKey === "mea_input_derived" && MEA_PREP_COL_DEFS[raw]) {
    return {raw, ...MEA_PREP_COL_DEFS[raw]};
  }
  if (tableKey === "mea_track_comparison" && MEA_CMP_COL_DEFS[raw]) {
    return {raw, ...MEA_CMP_COL_DEFS[raw]};
  }
  const t = tableKey === "measurement_trace" ? _measurementTraceManifest() : (_auditManifest()[tableKey] || {});
  const cols = t.columns || [];
  return cols.find(c => c.raw === raw) || {
    raw, label: raw, definition: "Source column " + raw + ".",
    format: _isLikelyNumericColumn(raw) ? "float" : "text",
  };
}

function _auditHeaderHtml(tableKey, raw) {
  const m = _auditColMeta(tableKey, raw);
  const tip = `Display label: ${m.label}\nRaw column: ${m.raw}\nDefinition: ${m.definition}`;
  return `<th title="${_escapeHtml(tip)}" aria-label="${_escapeHtml(tip)}" data-raw="${_escapeHtml(raw)}">${_escapeHtml(m.label)}</th>`;
}

function _formatAuditValue(v, col) {
  if (v == null || v === "") return "";
  if (_isLikelyNumericColumn(col)) {
    const n = Number(v);
    if (Number.isFinite(n)) {
      if (Number.isInteger(n) && Math.abs(n) < 100000) return String(n);
      return Math.abs(n) >= 1000 ? n.toFixed(1) : n.toPrecision(4);
    }
  }
  const s = String(v);
  return s.length > 90 ? s.slice(0, 87) + "..." : s;
}

function _parseCsv(text) {
  const rows = [];
  let row = [], cur = "", inQ = false;
  for (let i = 0; i < text.length; i++) {
    const ch = text[i], nx = text[i + 1];
    if (inQ) {
      if (ch === '"' && nx === '"') { cur += '"'; i++; }
      else if (ch === '"') inQ = false;
      else cur += ch;
    } else {
      if (ch === '"') inQ = true;
      else if (ch === ",") { row.push(cur); cur = ""; }
      else if (ch === "\n") { row.push(cur); rows.push(row); row = []; cur = ""; }
      else if (ch !== "\r") cur += ch;
    }
  }
  if (cur.length || row.length) { row.push(cur); rows.push(row); }
  if (!rows.length) return [];
  const header = rows.shift();
  return rows.filter(r => r.some(v => v !== "")).map(r => {
    const obj = {};
    header.forEach((h, i) => { obj[h] = r[i] == null ? "" : r[i]; });
    return obj;
  });
}

const AuditDataStore = (() => {
  const cache = new Map();
  const fileMode = location.protocol === "file:";
  async function load(tableKey) {
    if (cache.has(tableKey)) return cache.get(tableKey);
    const meta = _auditManifest()[tableKey];
    if (!meta) throw new Error("Unknown audit table: " + tableKey);
    if (fileMode || !meta.relative_path) {
      const preview = meta.preview || [];
      cache.set(tableKey, preview);
      return preview;
    }
    const resp = await fetch(meta.relative_path);
    if (!resp.ok) throw new Error(`HTTP ${resp.status} loading ${meta.relative_path}`);
    const text = await resp.text();
    let rows;
    if (meta.type === "json") {
      const obj = JSON.parse(text);
      rows = Array.isArray(obj) ? obj : Object.entries(obj).map(([key, value]) => ({key, value: JSON.stringify(value)}));
    } else {
      rows = _parseCsv(text);
    }
    cache.set(tableKey, rows);
    return rows;
  }
  return { load, fileMode };
})();

const MeasurementTraceStore = (() => {
  const cache = new Map();
  // Track-aware lookup: ST kinases pull from manifest.sample_files (legacy);
  // pY kinases pull from manifest.tracks.Y.sample_files (per-track sidecars).
  async function load(sample, residueType) {
    const manifest = _measurementTraceManifest();
    const tracks = manifest.tracks || {};
    const block = (residueType && tracks[residueType]) || tracks.ST || manifest;
    const files = block.sample_files || {};
    const key = (residueType || "ST") + "|" + sample;
    if (!files[sample]) {
      if (AuditDataStore.fileMode) return block.preview || manifest.preview || [];
      throw new Error("No measurement trace source for sample: " + sample);
    }
    if (cache.has(key)) return cache.get(key);
    if (AuditDataStore.fileMode) {
      const preview = block.preview || manifest.preview || [];
      cache.set(key, preview);
      return preview;
    }
    const resp = await fetch(files[sample]);
    if (!resp.ok) throw new Error(`HTTP ${resp.status} loading ${files[sample]}`);
    const rows = _parseCsv(await resp.text());
    cache.set(key, rows);
    return rows;
  }
  return { load };
})();

class AuditTable {
  constructor(hostId, opts) {
    this.host = document.getElementById(hostId);
    this.tableKey = opts.tableKey || "adhoc";
    this.columns = opts.columns || null;
    this.rows = opts.rows || [];
    this.pageSize = opts.pageSize || 20;
    this.page = 0;
    this.query = "";
    this.sortCol = null;
    this.sortAsc = true;
    this.title = opts.title || "";
    this.fullSourceKey = opts.fullSourceKey === false ? null : (opts.fullSourceKey || this.tableKey);
  }
  setRows(rows, columns) {
    this.rows = rows || [];
    if (columns) this.columns = columns;
    this.page = 0;
    this.render();
  }
  visibleColumns() {
    if (this.columns && this.columns.length) return this.columns;
    return Object.keys(this.rows[0] || {});
  }
  filteredRows() {
    const q = this.query.trim().toLowerCase();
    let rows = this.rows;
    if (q) rows = rows.filter(r => Object.values(r).some(v => String(v ?? "").toLowerCase().includes(q)));
    if (this.sortCol) {
      const c = this.sortCol, asc = this.sortAsc;
      rows = rows.slice().sort((a, b) => {
        const an = Number(a[c]), bn = Number(b[c]);
        let cmp = Number.isFinite(an) && Number.isFinite(bn)
          ? an - bn : String(a[c] ?? "").localeCompare(String(b[c] ?? ""));
        return asc ? cmp : -cmp;
      });
    }
    return rows;
  }
  exportRows(rows, cleanHeaders) {
    const cols = this.visibleColumns();
    const headers = cleanHeaders ? cols.map(c => _auditColMeta(this.tableKey, c).label) : cols;
    const esc = v => {
      const s = String(v == null ? "" : v);
      return /[",\n]/.test(s) ? '"' + s.replace(/"/g, '""') + '"' : s;
    };
    return [headers.map(esc).join(",")].concat(rows.map(r => cols.map(c => esc(r[c])).join(","))).join("\n");
  }
  downloadCsv(rows, label, cleanHeaders) {
    const blob = new Blob([this.exportRows(rows, cleanHeaders)], {type:"text/csv"});
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url; a.download = label + ".csv";
    document.body.appendChild(a); a.click(); document.body.removeChild(a);
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
  render() {
    if (!this.host) return;
    const cols = this.visibleColumns();
    const rows = this.filteredRows();
    const pages = Math.max(1, Math.ceil(rows.length / this.pageSize));
    if (this.page >= pages) this.page = pages - 1;
    const start = this.page * this.pageSize;
    const pageRows = rows.slice(start, start + this.pageSize);
    const cleanId = `${this.host.id}-clean`;
    const fullButton = this.fullSourceKey ? `<button data-action="export-full">Export full source</button>` : "";
    const body = pageRows.map(r => `<tr>${cols.map(c => {
      const cls = _isLikelyNumericColumn(c) ? ' class="numeric-cell"' : "";
      const raw = r[c] == null ? "" : String(r[c]);
      return `<td${cls} title="${_escapeHtml(raw)}">${_escapeHtml(_formatAuditValue(raw, c))}</td>`;
    }).join("")}</tr>`).join("");
    const fileNotice = AuditDataStore.fileMode
      ? '<div class="notice show">Full audit table loading requires serving outputs/reports/unified_viewer/ over HTTP. Showing embedded previews and selected in-payload data.</div>'
      : "";
    this.host.innerHTML =
      `${fileNotice}<div class="audit-controls">` +
      `<input type="search" placeholder="Search rows" aria-label="Search ${_escapeHtml(this.title || this.tableKey)}">` +
      `<button data-action="export-filtered">Export filtered</button>` +
      fullButton +
      `<label><input type="checkbox" id="${cleanId}"> Clean headers</label>` +
      `<span class="muted">${rows.length.toLocaleString()} rows</span></div>` +
      `<div class="audit-table-wrap"><table class="data-table"><thead><tr>${cols.map(c => _auditHeaderHtml(this.tableKey, c)).join("")}</tr></thead><tbody>${body}</tbody></table></div>` +
      `<div class="audit-pager"><button data-action="prev"${this.page === 0 ? " disabled" : ""}>Prev</button>` +
      `<span>${rows.length ? start + 1 : 0}-${Math.min(start + this.pageSize, rows.length)} of ${rows.length}</span>` +
      `<button data-action="next"${this.page >= pages - 1 ? " disabled" : ""}>Next</button></div>`;
    const search = this.host.querySelector('input[type="search"]');
    search.value = this.query;
    search.addEventListener("input", ev => { this.query = ev.target.value; this.page = 0; this.render(); });
    this.host.querySelectorAll("th").forEach(th => th.addEventListener("click", () => {
      const c = th.dataset.raw;
      if (this.sortCol === c) this.sortAsc = !this.sortAsc;
      else { this.sortCol = c; this.sortAsc = true; }
      this.render();
    }));
    this.host.querySelector('[data-action="prev"]').addEventListener("click", () => { this.page--; this.render(); });
    this.host.querySelector('[data-action="next"]').addEventListener("click", () => { this.page++; this.render(); });
    this.host.querySelector('[data-action="export-filtered"]').addEventListener("click", () => {
      this.downloadCsv(rows, `${this.tableKey}_filtered`, document.getElementById(cleanId).checked);
    });
    const fullBtn = this.host.querySelector('[data-action="export-full"]');
    if (fullBtn) fullBtn.addEventListener("click", async () => {
      const full = await AuditDataStore.load(this.fullSourceKey);
      this.downloadCsv(full, `${this.fullSourceKey}_full`, document.getElementById(cleanId).checked);
    });
  }
}

function renderHowToDrawer(tab) {
  const drawer = document.getElementById("howto-drawer");
  if (!drawer) return;
  const guide = TAB_GUIDE[tab];
  const manifest = TAB_MANIFEST[tab];
  const label = (manifest && manifest.label) || tab;
  if (!guide) {
    drawer.innerHTML = `<h3>${_escapeHtml(label)}</h3><p class="muted">No guide for this tab.</p>`;
    return;
  }
  const parts = [`<h3>${_escapeHtml(label)}</h3>`];
  if (guide.preamble) {
    parts.push(`<p class="ht-preamble">${_escapeHtml(guide.preamble)}</p>`);
  }
  const isNewSchema = guide.method || guide.shows || guide.howTo || guide.toggles;
  if (isNewSchema) {
    if (guide.method) {
      const paras = Array.isArray(guide.method) ? guide.method : [guide.method];
      parts.push(`<h4>How it was generated</h4>`);
      paras.forEach(p => parts.push(`<p>${_escapeHtml(p)}</p>`));
    }
    if (guide.shows) {
      parts.push(`<h4>What it shows</h4>`);
      if (typeof guide.shows === "string") {
        parts.push(`<p>${_escapeHtml(guide.shows)}</p>`);
      } else {
        if (guide.shows.lead) {
          const leads = Array.isArray(guide.shows.lead) ? guide.shows.lead : [guide.shows.lead];
          leads.forEach(l => parts.push(`<p>${_escapeHtml(l)}</p>`));
        }
        if (guide.shows.bullets && guide.shows.bullets.length) {
          const lis = guide.shows.bullets.map(b => `<li>${_escapeHtml(b)}</li>`).join("");
          parts.push(`<ul>${lis}</ul>`);
        }
      }
    }
    if (guide.howTo) {
      parts.push(`<h4>How to read it</h4><p>${_escapeHtml(guide.howTo)}</p>`);
    }
    if (guide.conclusions && guide.conclusions.length) {
      const cl = guide.conclusions.map(c => `<div class="ht-conclusion">${_escapeHtml(c)}</div>`).join("");
      parts.push(`<h4>Conclusions</h4>${cl}`);
    }
    if (guide.toggles && guide.toggles.length) {
      const lis = guide.toggles.map(t => {
        const name = t.name ? `<strong>${_escapeHtml(t.name)}</strong>` : "";
        const desc = t.desc ? ` — ${_escapeHtml(t.desc)}` : "";
        return `<li>${name}${desc}</li>`;
      }).join("");
      parts.push(`<h4>Adjustable toggles</h4><ul class="ht-toggles">${lis}</ul>`);
    }
  } else {
    if (guide.purpose) {
      parts.push(`<h4>What this tab answers</h4><p>${_escapeHtml(guide.purpose)}</p>`);
    }
    if (guide.primary) {
      parts.push(`<h4>How to read it</h4><p>${_escapeHtml(guide.primary)}</p>`);
    }
    if (guide.cues && guide.cues.length) {
      const cueRows = guide.cues.map(cue => {
        const m = METRIC_DEFS[cue.metric];
        const name = m ? m.label : cue.metric;
        const hr = m && m.howToRead ? m.howToRead : (m ? m.short : "");
        const when = cue.when ? ` <span class="ht-when">— ${_escapeHtml(cue.when)}</span>` : "";
        return `<div class="ht-cue"><span class="ht-metric">${_escapeHtml(name)}</span>${when}<br><span class="ht-when">${_escapeHtml(hr)}</span></div>`;
      }).join("");
      parts.push(`<h4>Metrics to watch</h4>${cueRows}`);
    }
    if (guide.conclusions && guide.conclusions.length) {
      const cl = guide.conclusions.map(c => `<div class="ht-conclusion">${_escapeHtml(c)}</div>`).join("");
      parts.push(`<h4>Conclusions</h4>${cl}`);
    }
  }
  drawer.innerHTML = parts.join("");
}

// ---------------------------------------------------------------------------
// View export — copy the current on-screen view (filters, methods preamble,
// visible rows) as Markdown for pasting into an AI chatbot. Scope: kinase,
// pathway, temporal, additivity, senders, graph. Reads DOM and Store state
// directly so the export tracks exactly what is rendered.
// ---------------------------------------------------------------------------
const EXPORT_TABS = ["kinase","pathway","temporal","additivity","senders","graph"];

function _exportFilterMap(tab) {
  // Returns alphabetized {Label: "value"} for the filters this tab consumes,
  // plus any tab-local thresholds.
  const m = TAB_MANIFEST[tab] || { filters: [] };
  const f = Store.state.filters;
  const out = {};
  if (m.filters.includes("receiver")) out["Receiver"] = f.receiver || "ALL";
  if (m.filters.includes("pathwayEvidence")) out["Support"] = f.pathwayEvidence || "any";
  if (m.filters.includes("fdr")) out["FDR"] = "< " + f.fdr;
  if (f.tpdsSig && f.tpdsSig !== "OFF") out["TPDS p"] = "< " + f.tpdsSig;
  // Tab-local thresholds (read from DOM so they reflect what the user set).
  const grab = (id) => { const e = document.getElementById(id); return e ? e.value : null; };
  if (tab === "temporal") {
    out["Mode"] = Store.state.view.temporalLevel;
    const v = grab("tm-score-min"); if (v != null && Number(v) > 0) out["|TPDS| min"] = v;
    const tiss = grab("tm-tissue"); if (tiss && tiss !== "ALL") out["Tissue"] = tiss;
    const met = grab("tm-metric"); if (met && Store.state.view.temporalLevel === "backbone") out["Metric"] = met;
  }
  if (tab === "additivity") {
    out["Mode"] = grab("add-level") || "kinase";
    out["Timepoint"] = grab("add-tp") || "ALL";
    const v = grab("add-score-min"); if (v != null && Number(v) > 0) out["Score min"] = v;
  }
  if (tab === "pathway") {
    const v = grab("pe-tpds-min"); if (v != null && Number(v) > 0) out["|TPDS| min"] = v;
    out["Trajectory"] = (typeof peTrajectory !== "undefined" && peTrajectory) ? peTrajectory : "all";
  }
  if (tab === "senders") {
    out["Compare"] = grab("sm-axis") || "timepoint";
    out["Anchor"] = grab("sm-anchor") || "";
    out["Mode"] = grab("sm-mode") || "count";
  }
  if (tab === "graph") {
    out["Genotype"] = grab("graph-genotype") || "";
    out["Timepoint"] = grab("graph-timepoint") || "";
    out["Layout"] = grab("graph-layout") || "";
    out["Min degree"] = grab("graph-min-degree") || "1";
    const v = grab("graph-tpds-min"); if (v != null && Number(v) > 0) out["|TPDS| min"] = v;
    const top = grab("graph-top-n"); if (top) out["Max edges"] = top;
  }
  // Sort alphabetically by key.
  const sorted = {};
  Object.keys(out).sort().forEach(k => { sorted[k] = out[k]; });
  return sorted;
}

function _exportSelectionChips() {
  const sel = Store.state.selection;
  const chips = [];
  if (sel.kinase != null) {
    _ensureKinaseIdx();
    const ki = _kinaseIdxById.get(sel.kinase);
    chips.push("kinase=" + (ki != null ? PAYLOAD.kinases.name[ki] : ("kid:" + sel.kinase)));
  }
  if (sel.backbone != null) chips.push("backbone=BB_" + sel.backbone);
  if (sel.celltype != null) chips.push("celltype=" + (RECEIVERS[sel.celltype] || ("cid:" + sel.celltype)));
  return chips;
}

function _exportDenominator(tab) {
  // Pull whichever subtitle/count element the tab already maintains.
  const ids = {
    kinase: "ke-count",
    pathway: "pe-count",
    temporal: "tm-subtitle",
    additivity: "add-subtitle",
    senders: "sm-subtitle",
    graph: "graph-stats",
  };
  const el = document.getElementById(ids[tab]);
  return el ? el.textContent.trim() : "";
}

function _exportMethods(tab) {
  const g = TAB_GUIDE[tab];
  if (!g) return "";
  const lines = [];
  if (Array.isArray(g.method)) g.method.forEach(p => lines.push(p));
  if (g.shows && g.shows.lead) {
    if (Array.isArray(g.shows.lead)) g.shows.lead.forEach(p => lines.push(p));
    else lines.push(g.shows.lead);
  }
  return lines.join("\n\n");
}

function _exportTableFromDom(tableId) {
  const tbl = document.getElementById(tableId);
  if (!tbl) return null;
  const headers = Array.from(tbl.querySelectorAll("thead th"))
    .map(th => th.textContent.replace(/[ ▲▼]+$/, "").trim());
  const rows = Array.from(tbl.querySelectorAll("tbody tr"))
    .map(tr => Array.from(tr.children).map(td => td.textContent.trim()));
  return { headers, rows };
}

function _exportTableFromPlotly(elId) {
  const el = document.getElementById(elId);
  if (!el || !el.data || !el.data.length) return null;
  // Generic flattening: for each trace, emit (trace_name, x, y).
  const headers = ["Series", "X", "Y"];
  const rows = [];
  for (const tr of el.data) {
    const name = tr.name || "";
    const xs = tr.x || [];
    const ys = tr.y || [];
    const n = Math.max(xs.length, ys.length);
    for (let i = 0; i < n; i++) {
      const x = xs[i] != null ? String(xs[i]) : "";
      const y = ys[i] != null ? (typeof ys[i] === "number" ? ys[i].toFixed(3) : String(ys[i])) : "";
      rows.push([name, x, y]);
    }
  }
  return { headers, rows };
}

function _exportTableFromHeatmaps(elIds) {
  // Three Plotly heatmaps for senders. Each carries z (matrix), x (receivers),
  // y (senders), name (panel label). Flatten cells with any non-null value.
  const headers = ["Panel", "Sender", "Receiver", "Value"];
  const rows = [];
  for (const elId of elIds) {
    const el = document.getElementById(elId);
    if (!el || !el.data || !el.data.length) continue;
    const tr = el.data[0];
    const z = tr.z || [];
    const xs = tr.x || [];
    const ys = tr.y || [];
    const panel = (el.layout && el.layout.title && el.layout.title.text) || tr.name || elId;
    for (let i = 0; i < z.length; i++) {
      for (let j = 0; j < (z[i] || []).length; j++) {
        const v = z[i][j];
        if (v == null) continue;
        rows.push([String(panel), String(ys[i] || i), String(xs[j] || j), typeof v === "number" ? v.toFixed(3) : String(v)]);
      }
    }
  }
  return { headers, rows };
}

function _exportTableFromGraph() {
  if (!_cyInstance) return null;
  const headers = ["Type", "Id", "Label", "Degree/Weight", "Extra"];
  const rows = [];
  _cyInstance.nodes(":visible").forEach(n => {
    rows.push(["node", n.id(), n.data("label") || "", String(n.degree(false)), n.data("kind") || ""]);
  });
  _cyInstance.edges(":visible").forEach(e => {
    const w = e.data("weight");
    rows.push(["edge", e.id(),
      (e.source().data("label") || e.source().id()) + " → " + (e.target().data("label") || e.target().id()),
      w == null ? "" : (typeof w === "number" ? w.toFixed(3) : String(w)),
      e.data("genotype") || ""]);
  });
  return { headers, rows };
}

function _exportTable(tab) {
  if (tab === "kinase")     return _exportTableFromDom("ke-table");
  if (tab === "pathway")    return _exportTableFromDom("pe-table");
  if (tab === "temporal")   return _exportTableFromPlotly("temporal-plot");
  if (tab === "additivity") return _exportTableFromPlotly("add-plot");
  if (tab === "senders")    return _exportTableFromHeatmaps(["sender-matrix-plot-0","sender-matrix-plot-1","sender-matrix-plot-2"]);
  if (tab === "graph")      return _exportTableFromGraph();
  return null;
}

function _exportEscapeMd(s) {
  return String(s).replace(/\|/g, "\\|").replace(/\n/g, " ");
}

function _exportRenderTable(table) {
  if (!table || !table.headers || !table.headers.length) return "_(no table data captured for this view)_";
  const head = "| " + table.headers.map(_exportEscapeMd).join(" | ") + " |";
  const sep  = "| " + table.headers.map(() => "---").join(" | ") + " |";
  const body = table.rows.map(r => "| " + r.map(_exportEscapeMd).join(" | ") + " |").join("\n");
  return [head, sep, body].join("\n");
}

function _exportAssemble(tab) {
  const label = (TAB_MANIFEST[tab] && TAB_MANIFEST[tab].label) || tab;
  const filters = _exportFilterMap(tab);
  const sels = _exportSelectionChips();
  const denom = _exportDenominator(tab);
  const methods = _exportMethods(tab);
  const table = _exportTable(tab);

  const lines = [];
  lines.push("# Unified Viewer export — " + label + " tab");
  lines.push("");
  lines.push("## Active view");
  Object.entries(filters).forEach(([k, v]) => lines.push("- **" + k + ":** " + v));
  if (sels.length) lines.push("- **Selection:** " + sels.join(", "));
  if (denom) {
    lines.push("");
    lines.push("**Denominator:** " + denom);
  }
  lines.push("");
  if (methods) {
    lines.push("## How this view was generated");
    lines.push("");
    lines.push(methods);
    lines.push("");
  }
  lines.push("## Visible rows");
  lines.push("");
  lines.push(_exportRenderTable(table));
  lines.push("");
  lines.push("_Generated by build_unified_viewer.py · view-scoped export. For full underlying data see outputs/reports/._");
  return lines.join("\n");
}

function _exportFlash(btn, msg) {
  const orig = btn.textContent;
  btn.textContent = msg;
  btn.disabled = true;
  setTimeout(() => { btn.textContent = orig; btn.disabled = false; }, 1400);
}

function _exportDownload(md, tab) {
  const blob = new Blob([md], { type: "text/markdown" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url; a.download = "viewer_" + tab + "_view.md";
  document.body.appendChild(a); a.click(); document.body.removeChild(a);
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

async function exportTab(tab, mode, btn) {
  try {
    const md = _exportAssemble(tab);
    if (mode === "download") {
      _exportDownload(md, tab);
      if (btn) _exportFlash(btn, "Downloaded ✓");
      return;
    }
    if (navigator.clipboard && navigator.clipboard.writeText) {
      await navigator.clipboard.writeText(md);
      if (btn) _exportFlash(btn, "Copied ✓");
    } else {
      _exportDownload(md, tab);
      if (btn) _exportFlash(btn, "Downloaded ✓");
    }
  } catch (e) {
    console.error("export failed", e);
    if (btn) _exportFlash(btn, "Failed — see console");
  }
}

function _exportInjectButton(tab, hostSelector) {
  const host = document.querySelector(hostSelector);
  if (!host || host.querySelector(".export-view-btn")) return;
  const wrap = document.createElement("span");
  wrap.className = "export-view-wrap";
  wrap.style.marginLeft = "auto";
  wrap.innerHTML = `<button type="button" class="chip export-view-btn" title="Copy this view as Markdown for an AI chatbot">⧉ Export view</button>` +
                   `<button type="button" class="chip export-view-dl" title="Download this view as a .md file" style="margin-left:4px;">⤓</button>`;
  host.appendChild(wrap);
  wrap.querySelector(".export-view-btn").addEventListener("click", e => exportTab(tab, "clipboard", e.currentTarget));
  wrap.querySelector(".export-view-dl").addEventListener("click", e => exportTab(tab, "download", e.currentTarget));
}

function wireExportButtons() {
  _exportInjectButton("kinase",     "#tab-kinase .ke-toolbar");
  _exportInjectButton("pathway",    "#tab-pathway .ke-toolbar");
  _exportInjectButton("temporal",   "#tab-temporal .detail-chips");
  _exportInjectButton("additivity", "#tab-additivity .detail-chips");
  _exportInjectButton("senders",    "#tab-senders .detail-chips");
  _exportInjectButton("graph",      "#graph-controls");
}

function wireDrawerResizer() {
  const resizer = document.getElementById("drawer-resizer");
  const drawer  = document.getElementById("howto-drawer");
  const shell   = document.getElementById("content-shell");
  if (!resizer || !drawer || !shell) return;

  // Restore saved width.
  try {
    const saved = localStorage.getItem("howtoDrawer.width");
    if (saved) { const w = parseInt(saved, 10); if (w >= 180 && w <= 800) drawer.style.width = w + "px"; }
  } catch (_) {}

  let startX = 0, startW = 0;
  resizer.addEventListener("mousedown", e => {
    startX = e.clientX;
    startW = drawer.getBoundingClientRect().width;
    resizer.classList.add("dragging");
    document.body.style.cursor = "col-resize";
    document.body.style.userSelect = "none";

    function onMove(ev) {
      const delta = startX - ev.clientX;   // dragging left = narrower main, wider drawer
      const newW = Math.min(600, Math.max(180, startW + delta));
      drawer.style.width = newW + "px";
    }
    function onUp() {
      resizer.classList.remove("dragging");
      document.body.style.cursor = "";
      document.body.style.userSelect = "";
      try { localStorage.setItem("howtoDrawer.width", parseInt(drawer.style.width, 10)); } catch (_) {}
      document.removeEventListener("mousemove", onMove);
      document.removeEventListener("mouseup", onUp);
    }
    document.addEventListener("mousemove", onMove);
    document.addEventListener("mouseup", onUp);
    e.preventDefault();
  });

  // Calibrate --shell-top so content-shell height fills the viewport below the header.
  function calibrateShellHeight() {
    const header = document.querySelector("header#app-header");
    const nav    = document.querySelector("nav#tab-bar");
    let top = 0;
    if (header) top += header.getBoundingClientRect().height;
    if (nav)    top += nav.getBoundingClientRect().height;
    shell.style.setProperty("--shell-top", top + "px");
    shell.style.height = "calc(100vh - " + top + "px)";
  }
  calibrateShellHeight();
  window.addEventListener("resize", calibrateShellHeight);

  // Drawer collapse toggle. Default: collapsed. Persisted in localStorage.
  const toggleBtn = document.getElementById("howto-drawer-toggle");
  if (toggleBtn) {
    let expanded = false;
    try { expanded = localStorage.getItem("howtoDrawer.expanded") === "1"; } catch (_) {}
    function applyExpanded(e) {
      shell.classList.toggle("drawer-collapsed", !e);
      toggleBtn.classList.toggle("expanded", e);
      toggleBtn.setAttribute("aria-expanded", e ? "true" : "false");
    }
    applyExpanded(expanded);
    toggleBtn.addEventListener("click", () => {
      expanded = !expanded;
      applyExpanded(expanded);
      try { localStorage.setItem("howtoDrawer.expanded", expanded ? "1" : "0"); } catch (_) {}
    });
  }

  // Kinase-tab splitter between ranked-kinase table and audit detail.
  const kaSplitter = document.getElementById("ka-splitter");
  if (kaSplitter) {
    const leftPanel = kaSplitter.previousElementSibling;
    if (leftPanel) {
      try {
        const savedW = localStorage.getItem("kinaseTab.leftWidth");
        if (savedW) {
          const w = parseInt(savedW, 10);
          if (w >= 280 && w <= 1200) leftPanel.style.width = w + "px";
        }
      } catch (_) {}
      let kStartX = 0, kStartW = 0;
      kaSplitter.addEventListener("mousedown", e => {
        kStartX = e.clientX;
        kStartW = leftPanel.getBoundingClientRect().width;
        kaSplitter.classList.add("dragging");
        document.body.style.cursor = "col-resize";
        document.body.style.userSelect = "none";
        function onMove(ev) {
          const newW = Math.min(1200, Math.max(280, kStartW + (ev.clientX - kStartX)));
          leftPanel.style.width = newW + "px";
        }
        function onUp() {
          kaSplitter.classList.remove("dragging");
          document.body.style.cursor = "";
          document.body.style.userSelect = "";
          try { localStorage.setItem("kinaseTab.leftWidth", parseInt(leftPanel.style.width, 10)); } catch (_) {}
          document.removeEventListener("mousemove", onMove);
          document.removeEventListener("mouseup", onUp);
        }
        document.addEventListener("mousemove", onMove);
        document.addEventListener("mouseup", onUp);
        e.preventDefault();
      });
    }
  }
}

// ---------------------------------------------------------------------------
// Per-tab manifest — declares which filters each tab consumes and what
// prerequisites must be met before content can render. Single source of
// truth for the filter-bar dim/hide logic and prerequisite empty states.
// ---------------------------------------------------------------------------
const TAB_MANIFEST = {
  signal:     { group:"landscape", label:"Signal Map",
                filters:[], requires:[] },
  senders:    { group:"landscape", label:"Sender×Receiver",
                filters:[], requires:[] },
  temporal:   { group:"landscape", label:"Temporal",
                filters:["fdr","pathwayEvidence","receiver"],
                requires:[] },
  additivity: { group:"landscape", label:"Additivity",
                filters:["fdr","receiver","pathwayEvidence"],
                requires:[] },
  kinase:     { group:"drilldown", label:"Kinase",
                filters:["fdr"], requires:[] },
  pathway:    { group:"drilldown", label:"Pathway",
                filters:["receiver","pathwayEvidence","fdr"],
                requires:[] },
  graph:      { group:"drilldown", label:"Graph",
                filters:["receiver","pathwayEvidence"],
                requires:[] },
  methods:    { group:"reference", label:"Methods",
                filters:[], requires:[] },
};

function syncFilterBarToTab(tab) {
  const manifest = TAB_MANIFEST[tab];
  const consumed = new Set(manifest ? manifest.filters : []);
  document.querySelectorAll(".filter-label").forEach(lab => {
    const key = lab.dataset.filter;
    lab.hidden = !consumed.has(key);
  });
}

// ---------------------------------------------------------------------------
// Unified prerequisite empty state — replaces bespoke placeholders for
// tabs that need a prior selection or filter to be set. Reads the active
// tab's manifest.requires[] and renders an actionable card if any
// requirement is unmet. Returns true if a prerequisite was rendered (caller
// should bail).
// ---------------------------------------------------------------------------
function _checkRequirement(req) {
  const f = Store.state.filters;
  const sel = Store.state.selection;
  if (req.type === "filter") {
    if (req.notEqual !== undefined) return f[req.key] !== req.notEqual;
    if (req.equal !== undefined) return f[req.key] === req.equal;
    return f[req.key] != null;
  }
  if (req.type === "selection") return sel[req.key] != null;
  return true;
}

function renderUnmetPrerequisite(panelEl, tab) {
  const manifest = TAB_MANIFEST[tab];
  if (!manifest || !manifest.requires || manifest.requires.length === 0)
    return false;
  const unmet = manifest.requires.find(r => !_checkRequirement(r));
  if (!unmet) return false;
  const card = document.createElement("div");
  card.className = "prereq-card";
  card.innerHTML =
    '<span class="prereq-icon" aria-hidden="true">&#9888;</span>' +
    '<div class="prereq-msg"></div>' +
    '<button type="button" class="prereq-action"></button>';
  card.querySelector(".prereq-msg").textContent = unmet.message;
  const btn = card.querySelector(".prereq-action");
  btn.textContent = unmet.cta;
  btn.addEventListener("click", () => {
    if (unmet.goTo) {
      Store.dispatch({type:"SET_VIEW", key:"activeTab", value:unmet.goTo});
    } else if (unmet.focus) {
      const el = document.getElementById(unmet.focus);
      if (el) { el.focus(); if (el.click) try { el.click(); } catch(_){} }
    }
  });
  panelEl.innerHTML = "";
  panelEl.appendChild(card);
  return true;
}

// ---------------------------------------------------------------------------
// URL-hash sync — serialize tab/filters/selection into location.hash so
// reload and back/forward restore state. Only non-default keys are emitted
// to keep the URL short. Suppresses re-broadcast while applying inbound.
// ---------------------------------------------------------------------------
const _HASH_DEFAULTS = {
  t: "signal", r: "ALL", s: "ALL",
  fdr: 0.25,
  k: null, b: null, ct: null,
};
let _hashApplying = false;

function _serializeHash() {
  const v = Store.state.view, f = Store.state.filters, s = Store.state.selection;
  const cur = {
    t: v.activeTab, r: f.receiver,
    s: f.pathwayEvidence, fdr: f.fdr,
    k: s.kinase, b: s.backbone, ct: s.celltype,
  };
  const parts = [];
  for (const k in cur) {
    const val = cur[k];
    if (val == null) continue;
    if (val === _HASH_DEFAULTS[k]) continue;
    parts.push(encodeURIComponent(k) + "=" + encodeURIComponent(String(val)));
  }
  return parts.length ? "#" + parts.join("&") : "";
}

function pushHash() {
  if (_hashApplying) return;
  const h = _serializeHash();
  if (h === window.location.hash) return;
  // Use replaceState so each filter twiddle doesn't pollute history; only
  // tab changes create a new history entry.
  history.replaceState(null, "", h || window.location.pathname + window.location.search);
}

function applyHash() {
  const raw = (window.location.hash || "").replace(/^#/, "");
  if (!raw) return;
  const parts = raw.split("&");
  const map = {};
  parts.forEach(p => {
    const [k, v] = p.split("=").map(decodeURIComponent);
    if (k) map[k] = v;
  });
  _hashApplying = true;
  try {
    if (map.r != null) Store.dispatch({type:"SET_FILTER", key:"receiver", value:map.r});
    if (map.s != null) Store.dispatch({type:"SET_FILTER", key:"pathwayEvidence", value:map.s});
    if (map.fdr != null) Store.dispatch({type:"SET_FILTER", key:"fdr", value:parseFloat(map.fdr)});
    if (map.k != null) Store.dispatch({type:"SET_SELECTION", key:"kinase", value:parseInt(map.k,10)});
    if (map.b != null) Store.dispatch({type:"SET_SELECTION", key:"backbone", value:parseInt(map.b,10)});
    if (map.ct != null) Store.dispatch({type:"SET_SELECTION", key:"celltype", value:parseInt(map.ct,10)});
    if (map.t != null) Store.dispatch({type:"SET_VIEW", key:"activeTab", value:map.t});
  } finally {
    _hashApplying = false;
  }
}

// ---------------------------------------------------------------------------
// Derived-array memoization — keyed on JSON signature of filters slice
// ---------------------------------------------------------------------------
let _filteredCache = { key:null, gnRef:null, indices:null };

function _computeFilteredIndices() {
  const f = Store.state.filters;
  const sel = Store.state.selection;
  const BB = PAYLOAD.backbones;
  const n = BB.id.length;
  const cIdx = CONTRASTS.indexOf(f.contrast);
  const tpdsCol = cIdx >= 0 ? BB["mean_tpds_" + f.contrast] : null;
  const sigCol = BB.significant_both_mask;
  const evidenceCol = cIdx >= 0
    ? BB["pathway_evidence_backbone_" + f.contrast]
    : BB.all_contrasts_pathway_evidence;
  const rIdx = (f.receiver === "ALL") ? -1 : RECEIVERS.indexOf(f.receiver);
  // graphNodeIds is a transient filter applied after a Pathway Graph node
  // click. Stored as a Set of backbone_id for O(1) membership.
  const gnSet = (f.graphNodeIds && f.graphNodeIds.length)
    ? new Set(f.graphNodeIds) : null;
  const senderBit = (f.sender == null) ? 0 : (1 << f.sender);
  const senderMaskCol = BB.sender_mask;
  // selection.kinase mask is null until the slice loads — the SET_SELECTION
  // subscriber re-renders on resolve so the unconstrained pass is transient.
  const kSet = (sel.kinase != null)
    ? SliceCache.kinaseBackboneSetSync(sel.kinase) : null;
  const ctIdx = (sel.celltype != null) ? sel.celltype : -1;
  // TPDS-magnitude significance: gates on whether the chain's TPDS is
  // distinguishable from zero, distinct from the kinase chain test gated
  // via significant_both_mask. Threshold via UI dropdown (off / 0.10 /
  // 0.05 / 0.01).
  const tpdsSigCol = (f.tpdsSig === "0.01") ? BB.tpds_sig_001_mask
                   : (f.tpdsSig === "0.05") ? BB.tpds_sig_005_mask
                   : (f.tpdsSig === "0.10") ? BB.tpds_sig_010_mask
                   : null;
  const out = [];
  for (let i = 0; i < n; i++) {
    if (rIdx >= 0 && BB.receiver_id[i] !== rIdx) continue;
    if (ctIdx >= 0 && BB.receiver_id[i] !== ctIdx) continue;
    if (senderBit && !(senderMaskCol[i] & senderBit)) continue;
    if (f.pathwayEvidence !== "ALL") {
      const ev = evidenceCol ? evidenceCol[i] : null;
      if (ev !== f.pathwayEvidence) continue;
    }
    if (cIdx >= 0) {
      if (!((sigCol[i] >> cIdx) & 1)) continue;
      const t = tpdsCol[i];
      if (t == null) continue;
    }
    if (tpdsSigCol !== null) {
      // contrast=ALL ⇒ require TPDS significance in any contrast.
      // contrast=specific ⇒ require it in that contrast.
      if (cIdx >= 0) {
        if (!((tpdsSigCol[i] >> cIdx) & 1)) continue;
      } else {
        if (tpdsSigCol[i] === 0) continue;
      }
    }
    if (gnSet !== null && !gnSet.has(BB.id[i])) continue;
    if (kSet !== null && !kSet.has(BB.id[i])) continue;
    out.push(i);
  }
  return out;
}

function getFilteredIndices() {
  const f = Store.state.filters;
  const sel = Store.state.selection;
  // graphNodeIds array identity changes on each SET_FILTER dispatch (reducer
  // deep-clones state) — use identity, not stringify, to avoid scanning the
  // full array on every read.
  const gnKey = f.graphNodeIds ? ("gn:" + f.graphNodeIds.length) : "gn:null";
  const gnRef = f.graphNodeIds;  // also compare by identity
  // kLoaded distinguishes pre-load (no mask) from post-load (mask applied).
  const kLoaded = (sel.kinase != null
    && SliceCache.kinaseBackboneSetSync(sel.kinase) !== null) ? "1" : "0";
  const key = f.contrast + "|" + f.receiver + "|"
            + f.pathwayEvidence + "|" + f.fdr + "|" + gnKey + "|s:" + (f.sender ?? "")
            + "|k:" + (sel.kinase ?? "") + "/" + kLoaded
            + "|c:" + (sel.celltype ?? "")
            + "|t:" + (f.tpdsSig ?? "OFF");
  if (key !== _filteredCache.key || gnRef !== _filteredCache.gnRef) {
    _filteredCache = {
      key, gnRef, indices: _computeFilteredIndices(),
    };
  }
  return _filteredCache.indices;
}
function invalidateFilterCache(){ _filteredCache.key = null; }
window.getFilteredIndices = getFilteredIndices;

// ---------------------------------------------------------------------------
// SliceCache — lazy loader for per-entity edge parquets (Unit E).
// Kinase slices and backbone-bucket slices are fetched on demand via the
// URLs in PAYLOAD.edge_slice_ref. LRU-capped to avoid unbounded memory.
// Parquet decoding uses hyparquet (CDN-loaded) when available; falls back
// to reporting an error message on the selected entity's side panel.
// ---------------------------------------------------------------------------
const SliceCache = (function(){
  const ESR = PAYLOAD.edge_slice_ref || {};
  const BUCKET_SIZE = ESR.bucket_size || 256;
  const MAX = 16;                          // LRU cap (per side)
  const kCache = new Map();                // kinase_id -> {backbone_id, contrast_id, support_contribution, concordance}
  const bCache = new Map();                // bucket_id -> same shape + kinase_id

  function _lruTouch(cache, key, value){
    if (cache.has(key)) cache.delete(key);
    cache.set(key, value);
    while (cache.size > MAX) cache.delete(cache.keys().next().value);
  }

  async function _fetchParquet(url){
    let resp;
    try {
      resp = await fetch(url);
    } catch (e) {
      if (window.location.protocol === "file:") {
        throw new Error(
          "Browser blocked local sidecar fetches under file://. " +
          "Serve outputs/reports/unified_viewer over HTTP and open that URL."
        );
      }
      throw e;
    }
    if (!resp.ok) throw new Error(`fetch ${url} → ${resp.status}`);
    const buf = await resp.arrayBuffer();
    if (typeof hyparquet === "undefined") {
      throw new Error("parquet reader not loaded (hyparquet missing)");
    }
    return await hyparquet.parquetReadObjects({
      file: buf, compressors: hyparquet.compressors,
    });
  }

  // Persistent (non-LRU) sets of backbone_ids per kinase, for sync filter use.
  const kBbSets = new Map();
  function _populateBbSet(kinase_id, rows){
    const s = new Set();
    for (const r of rows) s.add(r.backbone_id);
    kBbSets.set(kinase_id, s);
  }
  function kinaseBackboneSetSync(kinase_id){
    return kBbSets.has(kinase_id) ? kBbSets.get(kinase_id) : null;
  }

  async function loadKinase(kinase_id){
    if (kCache.has(kinase_id)) {
      const v = kCache.get(kinase_id); _lruTouch(kCache, kinase_id, v); return v;
    }
    const pad = String(kinase_id).padStart(3, "0");
    const url = `${ESR.kinase_url}${pad}.parquet`;
    const rows = await _fetchParquet(url);
    _lruTouch(kCache, kinase_id, rows);
    if (!kBbSets.has(kinase_id)) _populateBbSet(kinase_id, rows);
    return rows;
  }

  async function loadBackboneBucket(backbone_id){
    const bkt = Math.floor(backbone_id / BUCKET_SIZE);
    if (bCache.has(bkt)) {
      const v = bCache.get(bkt); _lruTouch(bCache, bkt, v); return v;
    }
    const pad = String(bkt).padStart(3, "0");
    const url = `${ESR.backbone_url}${pad}.parquet`;
    const rows = await _fetchParquet(url);
    _lruTouch(bCache, bkt, rows);
    return rows;
  }

  async function backboneEdges(backbone_id){
    const rows = await loadBackboneBucket(backbone_id);
    return rows.filter(r => r.backbone_id === backbone_id);
  }

  // per_backbone_summary.parquet — fetched once, indexed by backbone_id.
  // ~64,592 rows × 7 cols ≈ ~3 MB; small enough to keep wholesale in memory.
  let _bbSummaryAll = null;
  let _bbSummaryIdx = null;
  let _bbSummaryPromise = null;
  async function _loadBackboneSummary(){
    if (_bbSummaryAll) return _bbSummaryAll;
    if (_bbSummaryPromise) return _bbSummaryPromise;
    const url = ESR.backbone_summary_url;
    if (!url) throw new Error("backbone_summary_url missing in edge_slice_ref");
    _bbSummaryPromise = (async () => {
      const rows = await _fetchParquet(url);
      _bbSummaryAll = rows;
      const idx = new Map();
      for (let i = 0; i < rows.length; i++) {
        const bid = rows[i].backbone_id;
        let arr = idx.get(bid);
        if (!arr) { arr = []; idx.set(bid, arr); }
        arr.push(i);
      }
      _bbSummaryIdx = idx;
      _bbSummaryPromise = null;
      return rows;
    })();
    return _bbSummaryPromise;
  }
  async function backboneSummary(backbone_id){
    const rows = await _loadBackboneSummary();
    const arr = _bbSummaryIdx.get(backbone_id) || [];
    return arr.map(i => rows[i]);
  }

  // Decomp-OLS shards: per-kinase substrate-site OLS for every (contrast, wmb_class).
  // Backs the Attribution drawer's per-cell evidence section.
  const dCache = new Map();              // kinase_id -> rows[]
  const dPresent = new Set((ESR.present_decomp_ols_kinase_ids || []).map(Number));
  async function loadDecompOls(kinase_id){
    if (!dPresent.has(Number(kinase_id))) return [];
    if (dCache.has(kinase_id)) {
      const v = dCache.get(kinase_id); _lruTouch(dCache, kinase_id, v); return v;
    }
    if (!ESR.decomp_ols_url) return [];
    const pad = String(kinase_id).padStart(3, "0");
    const url = `${ESR.decomp_ols_url}${pad}.parquet`;
    const rows = await _fetchParquet(url);
    _lruTouch(dCache, kinase_id, rows);
    return rows;
  }

  return { loadKinase, loadBackboneBucket, backboneEdges, backboneSummary,
           kinaseBackboneSetSync, loadDecompOls,
           get kinaseCacheSize(){ return kCache.size; },
           get backboneCacheSize(){ return bCache.size; },
           get decompOlsCacheSize(){ return dCache.size; } };
})();
window.SliceCache = SliceCache;

// ---------------------------------------------------------------------------
// Header wiring
// ---------------------------------------------------------------------------
function populateHeader() {
  const fileNotice = document.getElementById("file-mode-notice");
  if (fileNotice && window.location.protocol === "file:")
    fileNotice.classList.add("show");
  const fr = document.getElementById("f-receiver");
  fr.innerHTML = ['<option value="ALL">All</option>']
    .concat(RECEIVERS.map(r => `<option value="${r}">${r}</option>`)).join("");
  fr.addEventListener("change", e => Store.dispatch({
    type:"SET_FILTER", key:"receiver", value:e.target.value}));
  document.getElementById("f-pathway-evidence").addEventListener("change", e =>
    Store.dispatch({type:"SET_FILTER", key:"pathwayEvidence", value:e.target.value}));
  document.getElementById("f-fdr").addEventListener("change", e =>
    Store.dispatch({type:"SET_FILTER", key:"fdr", value:parseFloat(e.target.value)}));
  const tps = document.getElementById("f-tpds-sig");
  if (tps) tps.addEventListener("change", e =>
    Store.dispatch({type:"SET_FILTER", key:"tpdsSig", value:e.target.value}));
  document.getElementById("glossary-toggle").addEventListener("click", () =>
    Store.dispatch({type:"SET_VIEW", key:"glossaryOpen",
      value:!Store.state.view.glossaryOpen}));
  const gnClear = document.getElementById("f-graph-nodes-clear");
  if (gnClear) gnClear.addEventListener("click", () =>
    Store.dispatch({type:"SET_FILTER", key:"graphNodeIds", value:null}));
  const sClear = document.getElementById("f-sender-clear");
  if (sClear) sClear.addEventListener("click", () =>
    Store.dispatch({type:"SET_FILTER", key:"sender", value:null}));
  const skClear = document.getElementById("f-selection-kinase-clear");
  if (skClear) skClear.addEventListener("click", () =>
    Store.dispatch({type:"SET_SELECTION", key:"kinase", value:null}));
  const scClear = document.getElementById("f-selection-celltype-clear");
  if (scClear) scClear.addEventListener("click", () =>
    Store.dispatch({type:"SET_SELECTION", key:"celltype", value:null}));
}

function _activateRowOnKey(ev, selector, handler) {
  if (ev.key !== "Enter" && ev.key !== " ") return;
  const tr = ev.target.closest(selector);
  if (!tr) return;
  ev.preventDefault();
  handler(tr);
}

function syncHeaderFromStore() {
  const f = Store.state.filters;
  const ids = ["f-receiver","f-pathway-evidence"];
  const vals = [f.receiver, f.pathwayEvidence];
  for (let i = 0; i < ids.length; i++) {
    const el = document.getElementById(ids[i]);
    if (el && el.value !== String(vals[i])) el.value = vals[i];
  }
  document.getElementById("f-fdr").value = f.fdr;
  const tps = document.getElementById("f-tpds-sig");
  if (tps && tps.value !== String(f.tpdsSig || "OFF")) tps.value = f.tpdsSig || "OFF";
  const gnClear = document.getElementById("f-graph-nodes-clear");
  if (gnClear) {
    const on = !!(f.graphNodeIds && f.graphNodeIds.length);
    gnClear.hidden = !on;
    if (on) gnClear.textContent = "Clear graph-node filter ("
      + f.graphNodeIds.length + " backbones)";
  }
  const sClear = document.getElementById("f-sender-clear");
  if (sClear) {
    const on = f.sender != null;
    sClear.hidden = !on;
    if (on) {
      const SENDERS = META.senderOrder || [];
      sClear.textContent = "Clear sender filter (" +
        (SENDERS[f.sender] || ("sid:" + f.sender)) + ")";
    }
  }
  const sel = Store.state.selection;
  const skClear = document.getElementById("f-selection-kinase-clear");
  if (skClear) {
    const on = sel.kinase != null;
    skClear.hidden = !on;
    if (on) {
      _ensureKinaseIdx();
      const K = PAYLOAD.kinases;
      const ki = _kinaseIdxById.get(sel.kinase);
      const name = ki != null ? K.name[ki] : ("kid:" + sel.kinase);
      skClear.textContent = "Clear kinase selection (" + name + ")";
    }
  }
  const scClear = document.getElementById("f-selection-celltype-clear");
  if (scClear) {
    const on = sel.celltype != null;
    scClear.hidden = !on;
    if (on) {
      const name = RECEIVERS[sel.celltype] || ("cid:" + sel.celltype);
      scClear.textContent = "Clear cell-type selection (" + name + ")";
    }
  }
}

// ---------------------------------------------------------------------------
// Tabs
// ---------------------------------------------------------------------------
function wireTabs() {
  const tabs = Array.from(document.querySelectorAll("nav#tab-bar button"));
  tabs.forEach((btn, idx) => {
    btn.addEventListener("click", () => {
      Store.dispatch({type:"SET_VIEW", key:"activeTab", value:btn.dataset.tab});
    });
    btn.addEventListener("keydown", ev => {
      const key = ev.key;
      if (!["ArrowRight", "ArrowLeft", "Home", "End"].includes(key)) return;
      ev.preventDefault();
      let nextIdx = idx;
      if (key === "ArrowRight") nextIdx = (idx + 1) % tabs.length;
      else if (key === "ArrowLeft") nextIdx = (idx - 1 + tabs.length) % tabs.length;
      else if (key === "Home") nextIdx = 0;
      else if (key === "End") nextIdx = tabs.length - 1;
      tabs[nextIdx].focus();
      Store.dispatch({type:"SET_VIEW", key:"activeTab", value:tabs[nextIdx].dataset.tab});
    });
  });
}

function syncTabsFromStore() {
  const active = Store.state.view.activeTab;
  document.querySelectorAll("nav#tab-bar button").forEach(btn => {
    const on = btn.dataset.tab === active;
    btn.classList.toggle("active", on);
    btn.setAttribute("aria-selected", on ? "true" : "false");
    btn.tabIndex = on ? 0 : -1;
  });
  document.querySelectorAll(".tab-panel").forEach(p => {
    const on = p.id === "tab-" + active;
    p.classList.toggle("active", on);
    p.hidden = !on;
  });
}

// ---------------------------------------------------------------------------
// Signal Map tab — receiver × contrast heatmap
// ---------------------------------------------------------------------------
function _tissueSortedPairs(names, tissueOf) {
  const pairs = names.map((n, i) => [n, tissueOf(n, i) || "zzz", i]);
  pairs.sort((a, b) => {
    if (a[1] !== b[1]) return a[1] < b[1] ? -1 : 1;
    return a[0] < b[0] ? -1 : (a[0] > b[0] ? 1 : 0);
  });
  return pairs;
}

function receiverOrder() {
  return _tissueSortedPairs(RECEIVERS, (_, i) => TISSUE_CAT[i]).map(p => p[0]);
}

function renderOverview() {
  const el = document.getElementById("overview-plot");
  if (!el) return;
  const f = Store.state.filters;
  const mode = Store.state.view.overviewMode;  // 'count' | 'direction'
  const rows = receiverOrder();
  const cols = CONTRASTS;

  // Build z matrix + hover + customdata.
  const z = [], hover = [], cd = [];
  for (const r of rows) {
    const zrow = [], hrow = [], crow = [];
    for (const c of cols) {
      const cell = PAYLOAD.overview[c + "|" + r];
      if (!cell || cell.n === 0) {
        zrow.push(null); hrow.push(`${r} | ${c}<br>(no sig backbones)`);
        crow.push({receiver:r, contrast:c, n:0});
      } else {
        let v;
        if (mode === "direction") v = cell.n_up - cell.n_down;
        else v = Math.log10(1 + cell.n);
        zrow.push(v);
        hrow.push(
          `${r} | ${c}<br>n=${cell.n} (up=${cell.n_up}, down=${cell.n_down})` +
          `<br>mean TPDS=${cell.mean_tpds}`);
        crow.push({receiver:r, contrast:c, n:cell.n});
      }
    }
    z.push(zrow); hover.push(hrow); cd.push(crow);
  }

  // Contrast filter: dim non-selected columns by blanking cells.
  if (f.contrast !== "ALL") {
    const keep = cols.indexOf(f.contrast);
    for (let i = 0; i < z.length; i++)
      for (let j = 0; j < z[i].length; j++)
        if (j !== keep) z[i][j] = null;
  }

  const colorscale = (mode === "direction")
    ? [[0, DISEASE_COLORS.Tau], [0.5, "#ffffff"], [1, DISEASE_COLORS.App]]
    : "YlOrRd";
  const trace = {
    type:"heatmap", x:cols, y:rows, z, text:hover,
    hovertemplate:"%{text}<extra></extra>", customdata:cd,
    colorscale, showscale:true,
    zmid: (mode === "direction") ? 0 : undefined,
  };
  const layout = {
    margin:{l:130, r:20, t:10, b:90},
    xaxis:{tickangle:-30, automargin:true},
    yaxis:{automargin:true, autorange:"reversed"},
    height:560,
  };
  Plotly.react(el, [trace], layout, {displaylogo:false, responsive:true});

  // Plotly.react preserves the DOM node, so detach prior listeners first.
  el.removeAllListeners && el.removeAllListeners("plotly_click");
  el.on && el.on("plotly_click", ev => {
    if (!ev.points || !ev.points.length) return;
    const d = ev.points[0].customdata;
    if (!d || d.n === 0) return;
    Store.dispatch({type:"SET_SELECTION", key:"backbone", value:null});
    Store.dispatch({type:"SET_FILTER", key:"receiver", value:d.receiver});
  });
}

// ---------------------------------------------------------------------------
// Sender × Receiver tab
// ---------------------------------------------------------------------------
let _senderOrderCache = null;
function _senderOrder() {
  if (_senderOrderCache) return _senderOrderCache;
  const SENDERS = META.senderOrder || [];
  const toTissue = META.receiverToTissue || {};
  _senderOrderCache = _tissueSortedPairs(SENDERS, (s) => toTissue[s]);
  return _senderOrderCache;
}

const SENDER_GENOTYPES = ["App", "Tau", "ApTt"];
const SENDER_TIMEPOINTS = ["2mo", "4mo", "6mo"];
let _senderMatrixScaleCache = null;

function _senderMatrixGlobalScale() {
  if (_senderMatrixScaleCache) return _senderMatrixScaleCache;
  const SM = PAYLOAD.senderMatrix || {};
  let maxCount = 0;
  let maxAbsDir = 0;
  for (const key of Object.keys(SM)) {
    const cell = SM[key];
    if (!cell || cell.n === 0) continue;
    const cv = Math.log10(1 + cell.n);
    if (cv > maxCount) maxCount = cv;
    const dv = Math.abs(cell.n_up - cell.n_down);
    if (dv > maxAbsDir) maxAbsDir = dv;
  }
  if (maxCount === 0) maxCount = 1;
  if (maxAbsDir === 0) maxAbsDir = 1;
  _senderMatrixScaleCache = { maxCount, maxAbsDir };
  return _senderMatrixScaleCache;
}

// Compare-axis design: render three 22×22 matrices side-by-side. The active
// axis determines what's varied across panels (the three timepoints, or the
// three genotypes). The anchor is the dimension held fixed.
const SENDER_AXIS_PANELS = {
  timepoint: ["2mo", "4mo", "6mo"],   // fixed genotype, vary timepoint
  genotype:  ["App", "Tau", "ApTt"],  // fixed timepoint, vary genotype
};
const SENDER_ANCHOR_OPTIONS = {
  timepoint: ["App", "Tau", "ApTt"],
  genotype:  ["2mo", "4mo", "6mo"],
};

function _senderPanelContrast(axis, anchor, panelValue) {
  // Reconstruct the contrast key {genotype}_{timepoint} given which axis we
  // are varying across panels. When axis="timepoint", anchor is the genotype
  // and panelValue is the timepoint; when axis="genotype" it's reversed.
  if (axis === "timepoint") return `${anchor}_${panelValue}`;
  return `${panelValue}_${anchor}`;
}

function _setSenderAxis(nextAxis) {
  const view = Store.state.view;
  if (view.senderMatrixAxis === nextAxis) return;
  const map = view.senderMatrixLastAnchorByAxis || {};
  const restored = map[nextAxis] ||
    (nextAxis === "timepoint" ? "ApTt" : "2mo");
  Store.dispatch({type:"SET_VIEW", key:"senderMatrixAxis", value: nextAxis});
  Store.dispatch({type:"SET_VIEW", key:"senderMatrixAnchor", value: restored});
}

function _setSenderAnchor(nextAnchor) {
  const view = Store.state.view;
  if (view.senderMatrixAnchor === nextAnchor) return;
  const map = Object.assign({}, view.senderMatrixLastAnchorByAxis || {});
  map[view.senderMatrixAxis] = nextAnchor;
  Store.dispatch({type:"SET_VIEW", key:"senderMatrixAnchor", value: nextAnchor});
  Store.dispatch({type:"SET_VIEW", key:"senderMatrixLastAnchorByAxis", value: map});
}

function _stepSenderAnchor(delta) {
  const view = Store.state.view;
  const opts = SENDER_ANCHOR_OPTIONS[view.senderMatrixAxis];
  const i = opts.indexOf(view.senderMatrixAnchor);
  const ni = ((i + delta) % opts.length + opts.length) % opts.length;
  _setSenderAnchor(opts[ni]);
}

function _flipSenderAxis() {
  const cur = Store.state.view.senderMatrixAxis;
  _setSenderAxis(cur === "timepoint" ? "genotype" : "timepoint");
}

function _renderSenderPanel(slotIdx, contrast, panelLabel) {
  const el = document.getElementById("sender-matrix-plot-" + slotIdx);
  if (!el) return;
  const mode = Store.state.view.senderMatrixMode;
  const SM = PAYLOAD.senderMatrix || {};
  const sRows = _senderOrder();
  const rCols = receiverOrder();
  const ctid = {};
  for (let i = 0; i < RECEIVERS.length; i++) ctid[RECEIVERS[i]] = i;

  const z = [], hover = [], cd = [];
  for (const [sname, , sid] of sRows) {
    const zrow = [], hrow = [], crow = [];
    for (const rname of rCols) {
      const rid = ctid[rname];
      const cell = SM[contrast + "|" + sid + "|" + rid];
      if (!cell || cell.n === 0) {
        zrow.push(null);
        hrow.push(`${sname} → ${rname}<br>(no backbones)`);
        crow.push({sender_id: sid, receiver: rname, n: 0});
      } else {
        let v;
        if (mode === "direction") v = cell.n_up - cell.n_down;
        else v = Math.log10(1 + cell.n);
        zrow.push(v);
        hrow.push(
          `${sname} → ${rname}<br>${contrast}<br>n=${cell.n} ` +
          `(up=${cell.n_up}, down=${cell.n_down})` +
          `<br>mean TPDS=${cell.mean_tpds}`);
        crow.push({sender_id: sid, receiver: rname, n: cell.n});
      }
    }
    z.push(zrow); hover.push(hrow); cd.push(crow);
  }

  const scale = _senderMatrixGlobalScale();
  const colorscale = (mode === "direction")
    ? [[0, DISEASE_COLORS.Tau], [0.5, "#ffffff"], [1, DISEASE_COLORS.App]]
    : "YlOrRd";
  // Show the colorbar only on the rightmost panel to save space.
  const showscale = (slotIdx === 2);
  const trace = {
    type: "heatmap",
    x: rCols, y: sRows.map(p => p[0]), z,
    text: hover, hovertemplate: "%{text}<extra></extra>",
    customdata: cd, colorscale, showscale,
    zmin: (mode === "direction") ? -scale.maxAbsDir : 0,
    zmax: (mode === "direction") ?  scale.maxAbsDir : scale.maxCount,
    zmid: (mode === "direction") ? 0 : undefined,
  };
  // Only the leftmost panel shows the y-axis sender labels to save space.
  const showY = (slotIdx === 0);
  const layout = {
    title: { text: panelLabel, font: { size: 13 } },
    margin: { l: showY ? 130 : 30, r: showscale ? 60 : 8, t: 30, b: 110 },
    xaxis: { tickangle:-45, automargin:true, tickfont:{size:9} },
    yaxis: { automargin:true, autorange:"reversed",
             dtick:1, tickfont:{size:9}, showticklabels: showY },
    height: 560,
  };
  Plotly.react(el, [trace], layout, {displaylogo:false, responsive:true});

  el.removeAllListeners && el.removeAllListeners("plotly_click");
  el.on && el.on("plotly_click", ev => {
    if (!ev.points || !ev.points.length) return;
    const d = ev.points[0].customdata;
    if (!d || d.n === 0) return;
    Store.dispatch({type:"SET_FILTER", key:"sender", value: d.sender_id});
    Store.dispatch({type:"SET_FILTER", key:"receiver", value: d.receiver});
    Store.dispatch({type:"SET_VIEW", key:"activeTab", value:"pathway"});
  });
}

function renderSenderMatrix() {
  const view = Store.state.view;
  const axis = view.senderMatrixAxis;
  const anchor = view.senderMatrixAnchor;
  const panels = SENDER_AXIS_PANELS[axis];
  for (let i = 0; i < 3; i++) {
    const c = _senderPanelContrast(axis, anchor, panels[i]);
    _renderSenderPanel(i, c, c);
  }
  const sub = document.getElementById("sm-subtitle");
  if (sub) {
    sub.textContent = (axis === "timepoint")
      ? `${anchor} at ${panels.join(", ")} — color scale pinned across all nine contrasts.`
      : `${panels.join(", ")} at ${anchor} — color scale pinned across all nine contrasts.`;
  }
}

function _populateSenderAnchorSelect() {
  const sel = document.getElementById("sm-anchor");
  if (!sel) return;
  const view = Store.state.view;
  const opts = SENDER_ANCHOR_OPTIONS[view.senderMatrixAxis];
  sel.innerHTML = opts.map(o => `<option value="${o}">${o}</option>`).join("");
  sel.value = view.senderMatrixAnchor;
  const lab = document.getElementById("sm-anchor-label");
  if (lab) {
    lab.firstChild.nodeValue =
      (view.senderMatrixAxis === "timepoint") ? "Genotype: " : "Timepoint: ";
  }
}

function wireSenderMatrix() {
  const modeSel = document.getElementById("sm-mode");
  if (modeSel) {
    modeSel.value = Store.state.view.senderMatrixMode;
    modeSel.addEventListener("change", ev => {
      Store.dispatch({type:"SET_VIEW", key:"senderMatrixMode",
                      value: ev.target.value});
    });
  }
  const axisSel = document.getElementById("sm-axis");
  if (axisSel) {
    axisSel.value = Store.state.view.senderMatrixAxis;
    axisSel.addEventListener("change", ev => _setSenderAxis(ev.target.value));
  }
  const anchorSel = document.getElementById("sm-anchor");
  _populateSenderAnchorSelect();
  if (anchorSel) {
    anchorSel.addEventListener("change", ev => _setSenderAnchor(ev.target.value));
  }
}

function wireSenderMatrixKeyboard() {
  document.addEventListener("keydown", ev => {
    if (Store.state.view.activeTab !== "senders") return;
    const tag = (ev.target && ev.target.tagName) || "";
    if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;
    if (ev.metaKey || ev.ctrlKey || ev.altKey) return;
    let handled = false;
    if (ev.key === "ArrowLeft")       { _stepSenderAnchor(-1); handled = true; }
    else if (ev.key === "ArrowRight") { _stepSenderAnchor(+1); handled = true; }
    else if (ev.key === "ArrowUp" ||
             ev.key === "ArrowDown")  { _flipSenderAxis(); handled = true; }
    if (handled) ev.preventDefault();
  });
}

// ---------------------------------------------------------------------------
// Temporal Dynamics tab — merged kinase Direction-over-Time + pathway Temporal.
// Level toggle picks which entity is aggregated; both render to #temporal-plot.
// ---------------------------------------------------------------------------
function _parseContrast(c) {
  const ix = c.lastIndexOf("_");
  return { geno: c.slice(0, ix), tp: c.slice(ix + 1) };
}

function _temporalKinaseMemberMask() {
  // Returns a Uint8Array over kinase ids indicating whether each kinase belongs
  // to the selected tissue scope (top_celltype_1 membership).
  const K = PAYLOAD.kinases;
  const scope = Store.state.view.temporalTissue;
  const n = K.id.length;
  const mask = new Uint8Array(n);
  if (scope === "ALL") { mask.fill(1); return mask; }
  const tissues = META.tissueCategories || {};
  if (scope.startsWith("t:")) {
    const name = scope.slice(2);
    const rs = new Set(tissues[name] || []);
    for (let i = 0; i < n; i++) mask[i] = rs.has(K.top_celltype_1[i]) ? 1 : 0;
  } else if (scope.startsWith("r:")) {
    const r = scope.slice(2);
    for (let i = 0; i < n; i++) mask[i] = (K.top_celltype_1[i] === r) ? 1 : 0;
  } else {
    mask.fill(1);
  }
  return mask;
}

function renderTemporalKinase() {
  const el = document.getElementById("temporal-plot");
  const sub = document.getElementById("tm-subtitle");
  const K = PAYLOAD.kinases;
  const fdr = Store.state.filters.fdr;
  const mask = _temporalKinaseMemberMask();
  const DG = META.diseaseGroups;
  const TPS = META.timepoints;
  const counts = {};
  for (const g of DG) counts[g] = {};
  for (const g of DG) for (const t of TPS) counts[g][t] = { up: 0, down: 0 };

  const n = K.id.length;
  let nScope = 0;
  for (let i = 0; i < n; i++) if (mask[i]) nScope++;

  for (const g of DG) {
    for (const t of TPS) {
      const c = g + "_" + t;
      const nesCol = K["NES_" + c];
      const fdrCol = K["FDR_" + c];
      if (!nesCol || !fdrCol) continue;
      let up = 0, down = 0;
      for (let i = 0; i < n; i++) {
        if (!mask[i]) continue;
        const q = fdrCol[i], nes = nesCol[i];
        if (q == null || nes == null) continue;
        if (q >= fdr) continue;
        if (nes > 0) up++;
        else if (nes < 0) down++;
      }
      counts[g][t] = { up, down };
    }
  }

  const traces = [];
  for (const g of DG) {
    const color = (META.diseaseColors || {})[g] || "#555";
    traces.push({
      type: "bar", name: g + " up",
      x: TPS, y: TPS.map(t => counts[g][t].up),
      marker: { color }, legendgroup: g,
      hovertemplate: `${g} up @ %{x}: %{y}<extra></extra>`,
    });
    traces.push({
      type: "bar", name: g + " down",
      x: TPS, y: TPS.map(t => -counts[g][t].down),
      marker: { color, opacity: 0.55 }, legendgroup: g, showlegend: true,
      hovertemplate: `${g} down @ %{x}: %{customdata}<extra></extra>`,
      customdata: TPS.map(t => counts[g][t].down),
    });
  }
  const layout = {
    barmode: "group", bargap: 0.25,
    margin: { l: 60, r: 20, t: 10, b: 40 },
    xaxis: { title: "Timepoint" },
    yaxis: { title: "Sig kinases (up − down)", zeroline: true },
    legend: { orientation: "h", y: -0.15 },
    height: 480,
    shapes: [{ type: "line", x0: -0.5, x1: TPS.length - 0.5, y0: 0, y1: 0,
               xref: "x", yref: "y", line: { color: "#000", width: 1 } }],
  };
  Plotly.react(el, traces, layout, { displaylogo: false, responsive: true });
  if (sub) sub.textContent =
    `${nScope} kinases in scope · FDR < ${fdr} · diverging bars show up (+) vs down (−) counts.`;
}

// Receiver/sender/kinase-selection filter without chain-significance gating.
// Used by magnitude-based temporal metrics (mean_tpds, pct_up) so the diffuse
// phase — where the chain test reports zero passing chains but pathway burden
// is still elevated — does not get suppressed alongside the count.
function _temporalUngatedIndices() {
  const f = Store.state.filters;
  const sel = Store.state.selection;
  const BB = PAYLOAD.backbones;
  const n = BB.id.length;
  const rIdx = (f.receiver === "ALL") ? -1 : RECEIVERS.indexOf(f.receiver);
  const gnSet = (f.graphNodeIds && f.graphNodeIds.length)
    ? new Set(f.graphNodeIds) : null;
  const senderBit = (f.sender == null) ? 0 : (1 << f.sender);
  const senderMaskCol = BB.sender_mask;
  const kSet = (sel.kinase != null)
    ? SliceCache.kinaseBackboneSetSync(sel.kinase) : null;
  const ctIdx = (sel.celltype != null) ? sel.celltype : -1;
  const tpdsSigCol = (f.tpdsSig === "0.01") ? BB.tpds_sig_001_mask
                   : (f.tpdsSig === "0.05") ? BB.tpds_sig_005_mask
                   : (f.tpdsSig === "0.10") ? BB.tpds_sig_010_mask
                   : null;
  const out = [];
  for (let i = 0; i < n; i++) {
    if (rIdx >= 0 && BB.receiver_id[i] !== rIdx) continue;
    if (ctIdx >= 0 && BB.receiver_id[i] !== ctIdx) continue;
    if (senderBit && !(senderMaskCol[i] & senderBit)) continue;
    if (tpdsSigCol !== null && tpdsSigCol[i] === 0) continue;
    if (gnSet !== null && !gnSet.has(BB.id[i])) continue;
    if (kSet !== null && !kSet.has(BB.id[i])) continue;
    out.push(i);
  }
  return out;
}

function renderTemporalBackbone() {
  const el = document.getElementById("temporal-plot");
  const sub = document.getElementById("tm-subtitle");
  const BB = PAYLOAD.backbones;
  const metric = Store.state.view.temporalMetric;
  const DG = META.diseaseGroups;
  const TPS = META.timepoints;
  const f = Store.state.filters;
  const sel = Store.state.selection;
  const TD = PAYLOAD.tpdsDistribution || {};
  // The backbone payload is hard-gated to chains passing the chain test.
  // For count and mean_score that's appropriate (both are defined on
  // passing chains). For mean_tpds and pct_up — magnitude readouts that
  // should reflect every enumerated chain, including diffuse-phase late-
  // Tau where nothing passes — read from the build-time tpdsDistribution
  // summary. The summary aggregates per (receiver, contrast) across all
  // senders, so it answers the broad-scope question; if the user pins a
  // sender or a kinase selection, we fall back to the BB iteration with a
  // subtitle note that magnitude is now restricted to passing chains.
  const useSummary = (metric === "mean_tpds" || metric === "pct_up")
                     && f.sender == null && sel.kinase == null
                     && (!f.graphNodeIds || !f.graphNodeIds.length);

  if (useSummary) {
    const traces = [];
    const recvKeys = (f.receiver === "ALL")
      ? RECEIVERS.slice() : [f.receiver];
    for (const g of DG) {
      const color = (META.diseaseColors || {})[g] || "#555";
      const y = [];
      const cust = [];
      const totalN = [];
      for (const t of TPS) {
        const c = g + "_" + t;
        let nSum = 0, sumAbs = 0, nUp = 0, nDown = 0;
        for (const r of recvKeys) {
          const cell = TD[c + "|" + r];
          if (!cell) continue;
          nSum  += cell.n;
          sumAbs += cell.mean_abs * cell.n;
          nUp   += cell.n_up;
          nDown += cell.n_down;
        }
        if (nSum === 0) { y.push(null); cust.push([0, 0]); totalN.push(0); continue; }
        if (metric === "mean_tpds") y.push(sumAbs / nSum);
        else if (metric === "pct_up") y.push(100 * nUp / nSum);
        else y.push(null);
        cust.push([nUp, nDown]);
        totalN.push(nSum);
      }
      traces.push({
        type: "scatter", mode: "lines+markers", name: g,
        x: TPS, y, customdata: TPS.map((_, i) => [totalN[i], cust[i][0], cust[i][1]]),
        line: { color, width: 2 }, marker: { color, size: 8 },
        hovertemplate:
          "<b>" + g + "</b> %{x}<br>" +
          "value: %{y:.4f}<br>" +
          "n chains: %{customdata[0]}<br>" +
          "up / down: %{customdata[1]} / %{customdata[2]}<extra></extra>",
      });
    }
    const yTitle = (metric === "mean_tpds")
      ? "Mean |TPDS| (all enumerated chains)"
      : "% upregulated (TPDS > 0, all enumerated chains)";
    Plotly.react(el, traces, {
      margin: { l: 70, r: 20, t: 10, b: 40 },
      xaxis: { title: "Timepoint" },
      yaxis: { title: yTitle, zeroline: true },
      legend: { orientation: "h", y: -0.15 },
      height: 480,
    }, { displaylogo: false, responsive: true });
    if (sub) sub.textContent =
      `metric = ${metric} · reading per-(receiver, contrast) summary built` +
      ` from every enumerated chain · receiver=${f.receiver}.`;
    return;
  }

  // Count or mean_score, OR a sender/kinase selection is active (which the
  // summary cannot answer). Iterate the gated BB payload.
  const idx = _temporalUngatedIndices();
  const sigMaskCol = BB.significant_both_mask;
  const tpdsMin = Math.max(0, Number(Store.state.view.temporalScoreMin) || 0);

  const agg = {};
  for (const g of DG) { agg[g] = {}; for (const t of TPS)
    agg[g][t] = { countSig: 0, sumScore: 0, nFinite: 0,
                   nMagnitude: 0, sumAbsTpds: 0, nUp: 0 }; }
  for (const g of DG) {
    for (const t of TPS) {
      const c = g + "_" + t;
      const cIdx = CONTRASTS.indexOf(c);
      const tpdsCol = BB["mean_tpds_" + c];
      const obsCol = BB["observed_score_" + c];
      if (!tpdsCol || cIdx < 0) continue;
      const a = agg[g][t];
      for (let j = 0; j < idx.length; j++) {
        const i = idx[j];
        const tp = tpdsCol[i];
        if (tp == null) continue;
        if (tpdsMin > 0 && Math.abs(tp) < tpdsMin) continue;
        a.nMagnitude++;
        a.sumAbsTpds += Math.abs(tp);
        if (tp > 0) a.nUp++;
        const isSig = ((sigMaskCol[i] >> cIdx) & 1) === 1;
        if (isSig) {
          a.countSig++;
          const os = obsCol ? obsCol[i] : null;
          if (os != null) { a.sumScore += os; a.nFinite++; }
        }
      }
    }
  }

  const traces = [];
  for (const g of DG) {
    const color = (META.diseaseColors || {})[g] || "#555";
    const y = TPS.map(t => {
      const a = agg[g][t];
      if (metric === "count") return a.countSig;
      if (metric === "mean_score") return a.nFinite ? a.sumScore / a.nFinite : null;
      if (metric === "mean_tpds") return a.nMagnitude ? a.sumAbsTpds / a.nMagnitude : null;
      if (metric === "pct_up") return a.nMagnitude ? (100 * a.nUp / a.nMagnitude) : null;
      return null;
    });
    const customdata = TPS.map(t => {
      const a = agg[g][t];
      return [a.countSig, a.nMagnitude];
    });
    traces.push({
      type: "scatter", mode: "lines+markers", name: g,
      x: TPS, y, customdata,
      line: { color, width: 2 }, marker: { color, size: 8 },
      hovertemplate:
        "<b>" + g + "</b> %{x}<br>" +
        "value: %{y}<br>" +
        "passing chains: %{customdata[0]}<br>" +
        "chains with TPDS: %{customdata[1]}<extra></extra>",
    });
  }
  const yTitle = ({
    count: "Passing-chain count",
    mean_score: "Mean observed score (over passing chains)",
    mean_tpds: "Mean |TPDS| (passing chains only — selection active)",
    pct_up: "% upregulated (passing chains only — selection active)",
  })[metric] || "";
  Plotly.react(el, traces, {
    margin: { l: 70, r: 20, t: 10, b: 40 },
    xaxis: { title: "Timepoint" },
    yaxis: { title: yTitle, zeroline: true },
    legend: { orientation: "h", y: -0.15 },
    height: 480,
  }, { displaylogo: false, responsive: true });
  const restrict = (metric === "mean_tpds" || metric === "pct_up")
    ? " · magnitude restricted to passing chains because a sender/kinase selection is active"
    : "";
  if (sub) sub.textContent =
    `${idx.length.toLocaleString()} chains in current filter · metric = ${metric}` + restrict + ".";
}

function renderTemporal() {
  const el = document.getElementById("temporal-plot");
  if (!el) return;
  const level = Store.state.view.temporalLevel;
  const metricLabel = document.getElementById("tm-metric-label");
  const tissueLabel = document.getElementById("tm-tissue-label");
  if (metricLabel) metricLabel.style.display = (level === "backbone") ? "" : "none";
  if (tissueLabel) tissueLabel.style.display = (level === "kinase") ? "" : "none";
  if (level === "kinase") renderTemporalKinase();
  else renderTemporalBackbone();
}

function wireTemporalControls() {
  const levelSel = document.getElementById("tm-level");
  const metricSel = document.getElementById("tm-metric");
  const tissueSel = document.getElementById("tm-tissue");
  if (!levelSel || !metricSel || !tissueSel) return;

  // Populate tissue dropdown: All + tissue groups + per-receiver leaves.
  const opts = ['<option value="ALL">All cell types</option>'];
  const tissues = META.tissueCategories || {};
  for (const tname of Object.keys(tissues)) {
    opts.push(`<option value="t:${tname}">${tname} (tissue)</option>`);
    for (const r of tissues[tname])
      opts.push(`<option value="r:${r}">&nbsp;&nbsp;${r}</option>`);
  }
  tissueSel.innerHTML = opts.join("");

  levelSel.value = Store.state.view.temporalLevel;
  metricSel.value = Store.state.view.temporalMetric;
  tissueSel.value = Store.state.view.temporalTissue;

  levelSel.addEventListener("change", ev =>
    Store.dispatch({type:"SET_VIEW", key:"temporalLevel", value: ev.target.value}));
  metricSel.addEventListener("change", ev =>
    Store.dispatch({type:"SET_VIEW", key:"temporalMetric", value: ev.target.value}));
  tissueSel.addEventListener("change", ev =>
    Store.dispatch({type:"SET_VIEW", key:"temporalTissue", value: ev.target.value}));
  const scoreInp = document.getElementById("tm-score-min");
  if (scoreInp) {
    scoreInp.value = Store.state.view.temporalScoreMin || 0;
    scoreInp.addEventListener("change", ev =>
      Store.dispatch({type:"SET_VIEW", key:"temporalScoreMin",
                      value: Math.max(0, parseFloat(ev.target.value) || 0)}));
  }
}

// ---------------------------------------------------------------------------
// Additivity tab — merged kinase NES + backbone TPDS ApTt-additivity scatter.
// Predicted = App + Tau; observed = ApTt. y=x means perfectly additive; points
// below the diagonal = sub-additive (standing sanity check).
// ---------------------------------------------------------------------------
const _ADD_COLORS = {
  "App only":   "#d1495b",
  "Tau only":   "#2e86ab",
  "ApTt only":  "#8338ec",
  "Multi":      "#444",
};
const _ADD_CATEGORIES = ["App only", "Tau only", "ApTt only", "Multi"];
const _ADD_BACKBONE_MAX_POINTS = 20000;

const _axSuf = (k) => (k === 0) ? "" : String(k + 1);
const _newAcc = () => ({ n:0, sx:0, sy:0, sxx:0, syy:0, sxy:0 });
function _accAdd(a, x, y) {
  a.n++; a.sx += x; a.sy += y; a.sxx += x*x; a.syy += y*y; a.sxy += x*y;
}
function _pearson(a) {
  if (a.n < 3) return { r: null, n: a.n };
  const num = a.n*a.sxy - a.sx*a.sy;
  const den = Math.sqrt((a.n*a.sxx - a.sx*a.sx) * (a.n*a.syy - a.sy*a.sy));
  return { r: den > 0 ? num/den : null, n: a.n };
}

function _addTimepointsInScope() {
  const tp = Store.state.view.additivityTimepoint;
  const TPS = META.timepoints;
  return (tp === "ALL") ? TPS.slice() : [tp];
}

function _addDiagonalShapes(tps, xRange) {
  const shapes = [];
  const annotations = [];
  for (let k = 0; k < tps.length; k++) {
    const s = _axSuf(k);
    shapes.push({
      type: "line", xref: "x" + s, yref: "y" + s,
      x0: xRange[0], x1: xRange[1], y0: xRange[0], y1: xRange[1],
      line: { color: "#888", width: 1, dash: "dash" },
    });
    annotations.push({
      xref: "x" + s + " domain", yref: "y" + s + " domain",
      x: 0.03, y: 0.97, xanchor: "left", yanchor: "top", showarrow: false,
      text: "Synergistic", font: { size: 10, color: "#888" },
    });
    annotations.push({
      xref: "x" + s + " domain", yref: "y" + s + " domain",
      x: 0.97, y: 0.03, xanchor: "right", yanchor: "bottom", showarrow: false,
      text: "Sub-additive", font: { size: 10, color: "#888" },
    });
    annotations.push({
      xref: "x" + s + " domain", yref: "y" + s + " domain",
      x: 0.5, y: 1.08, xanchor: "center", yanchor: "bottom", showarrow: false,
      text: "<b>" + tps[k] + "</b>", font: { size: 13 },
    });
  }
  return { shapes, annotations };
}

function _addAxesLayout(tps, axRange, xTitle, yTitle) {
  const layout = {
    margin: { l: 60, r: 20, t: 40, b: 50 },
    grid: { rows: 1, columns: tps.length, pattern: "independent" },
    height: 520, hovermode: "closest",
  };
  for (let k = 0; k < tps.length; k++) {
    const s = _axSuf(k);
    layout["xaxis" + s] = { title: xTitle, range: axRange, zeroline: true };
    layout["yaxis" + s] = { title: (k === 0) ? yTitle : "",
                             range: axRange, zeroline: true,
                             scaleanchor: "x" + s, scaleratio: 1 };
  }
  return layout;
}

function _addCategory(fApp, fTau, fApTt, thresh) {
  const sApp = (fApp != null && fApp < thresh);
  const sTau = (fTau != null && fTau < thresh);
  const sAp  = (fApTt != null && fApTt < thresh);
  const n = (sApp ? 1 : 0) + (sTau ? 1 : 0) + (sAp ? 1 : 0);
  if (n === 0) return null;
  if (n >= 2) return "Multi";
  if (sApp) return "App only";
  if (sTau) return "Tau only";
  return "ApTt only";
}

function _writeStats(stats, tps, accs) {
  if (!stats) return;
  stats.textContent = tps.map((t, k) => {
    const r = _pearson(accs[k]);
    return `${t}: n=${r.n}, Pearson r=${r.r == null ? "–" : r.r.toFixed(3)}`;
  }).join("  ·  ");
}

function renderAdditivityKinase() {
  const el = document.getElementById("add-plot");
  const sub = document.getElementById("add-subtitle");
  const stats = document.getElementById("add-stats");
  const K = PAYLOAD.kinases;
  const fdr = Store.state.filters.fdr;
  const recv = Store.state.filters.receiver;
  const n = K.id.length;
  const tps = _addTimepointsInScope();

  const buckets = tps.map(() => {
    const b = {};
    for (const c of _ADD_CATEGORIES) b[c] = { x: [], y: [], text: [], customdata: [] };
    return b;
  });
  const accs = tps.map(_newAcc);
  let xMin = -0.1, xMax = 0.1, yMin = -0.1, yMax = 0.1;

  for (let k = 0; k < tps.length; k++) {
    const t = tps[k];
    const nAppCol = K["NES_App_" + t],  nTauCol = K["NES_Tau_" + t],  nApCol = K["NES_ApTt_" + t];
    const fAppCol = K["FDR_App_" + t],  fTauCol = K["FDR_Tau_" + t],  fApCol = K["FDR_ApTt_" + t];
    if (!nAppCol || !nTauCol || !nApCol) continue;
    const bucket = buckets[k];
    const acc = accs[k];
    for (let i = 0; i < n; i++) {
      if (recv !== "ALL" && K.top_celltype_1 && K.top_celltype_1[i] !== recv) continue;
      const nApp = nAppCol[i], nTau = nTauCol[i], nAp = nApCol[i];
      if (nApp == null || nTau == null || nAp == null) continue;
      const fApp = fAppCol[i], fTau = fTauCol[i], fAp = fApCol[i];
      const x = nApp + nTau, y = nAp;
      const cat = _addCategory(fApp, fTau, fAp, fdr);
      if (cat == null) continue;
      const b = bucket[cat];
      b.x.push(x); b.y.push(y);
      b.text.push(K.name[i]);
      b.customdata.push([nApp, nTau, nAp, fApp, fTau, fAp]);
      _accAdd(acc, x, y);
      if (x < xMin) xMin = x; if (x > xMax) xMax = x;
      if (y < yMin) yMin = y; if (y > yMax) yMax = y;
    }
  }

  const axRange = [Math.min(xMin, yMin) - 0.2, Math.max(xMax, yMax) + 0.2];
  const traces = [];
  for (let k = 0; k < tps.length; k++) {
    const s = _axSuf(k);
    for (const cat of _ADD_CATEGORIES) {
      const b = buckets[k][cat];
      if (!b.x.length) continue;
      traces.push({
        type: "scattergl", mode: "markers", name: cat,
        legendgroup: cat, showlegend: (k === 0),
        x: b.x, y: b.y, text: b.text, customdata: b.customdata,
        xaxis: "x" + s, yaxis: "y" + s,
        marker: { color: _ADD_COLORS[cat], size: 7, opacity: 0.75,
                  line: { width: 0.5, color: "#fff" } },
        hovertemplate:
          "<b>%{text}</b><br>App NES: %{customdata[0]:.2f} (q=%{customdata[3]:.2g})" +
          "<br>Tau NES: %{customdata[1]:.2f} (q=%{customdata[4]:.2g})" +
          "<br>ApTt NES: %{customdata[2]:.2f} (q=%{customdata[5]:.2g})" +
          "<br>Pred (App+Tau): %{x:.2f}<br>Obs (ApTt): %{y:.2f}<extra></extra>",
      });
    }
  }
  const { shapes, annotations } = _addDiagonalShapes(tps, axRange);
  const layout = _addAxesLayout(tps, axRange, "App + Tau NES", "ApTt NES (observed)");
  layout.showlegend = true;
  layout.legend = { orientation: "h", y: -0.18 };
  layout.shapes = shapes;
  layout.annotations = annotations;
  Plotly.react(el, traces, layout, { displaylogo: false, responsive: true });

  _writeStats(stats, tps, accs);
  if (sub) sub.textContent =
    `Kinase level · predicted = App NES + Tau NES · observed = ApTt NES · FDR < ${fdr}` +
    (recv !== "ALL" ? ` · receiver=${recv}` : "");
}

function renderAdditivityBackbone() {
  const el = document.getElementById("add-plot");
  const sub = document.getElementById("add-subtitle");
  const stats = document.getElementById("add-stats");
  const BB = PAYLOAD.backbones;
  const tps = _addTimepointsInScope();
  const idx = getFilteredIndices();

  let sampleIdx = idx;
  let thinned = false;
  if (idx.length > _ADD_BACKBONE_MAX_POINTS) {
    const stride = idx.length / _ADD_BACKBONE_MAX_POINTS;
    sampleIdx = new Int32Array(_ADD_BACKBONE_MAX_POINTS);
    for (let j = 0; j < _ADD_BACKBONE_MAX_POINTS; j++)
      sampleIdx[j] = idx[Math.floor(j * stride)];
    thinned = true;
  }

  const perTp = tps.map(() => ({ x: [], y: [] }));
  const accs = tps.map(_newAcc);
  let xMin = 0, xMax = 0, yMin = 0, yMax = 0;
  for (let k = 0; k < tps.length; k++) {
    const t = tps[k];
    const oApp = BB["observed_score_App_" + t];
    const oTau = BB["observed_score_Tau_" + t];
    const oAp  = BB["observed_score_ApTt_" + t];
    if (!oApp || !oTau || !oAp) continue;
    const dst = perTp[k];
    const acc = accs[k];
    const sMin = Math.max(0, Number(Store.state.view.additivityScoreMin) || 0);
    for (let j = 0; j < sampleIdx.length; j++) {
      const i = sampleIdx[j];
      const a = oApp[i], tv = oTau[i], av = oAp[i];
      if (a == null || tv == null || av == null) continue;
      if (sMin > 0 && (a < sMin && tv < sMin)) continue;
      const x = a + tv;
      dst.x.push(x); dst.y.push(av);
      _accAdd(acc, x, av);
      if (x < xMin) xMin = x; if (x > xMax) xMax = x;
      if (av < yMin) yMin = av; if (av > yMax) yMax = av;
    }
  }
  const axRange = [Math.min(xMin, yMin) * 1.05 - 0.1,
                   Math.max(xMax, yMax) * 1.05 + 0.1];

  const traces = [];
  for (let k = 0; k < tps.length; k++) {
    const s = _axSuf(k);
    const p = perTp[k];
    const npts = p.x.length;
    const mSize = npts > 10000 ? 3 : npts > 2000 ? 5 : 8;
    const mOpacity = npts > 10000 ? 0.35 : npts > 2000 ? 0.5 : 0.75;
    traces.push({
      type: "scattergl", mode: "markers", name: tps[k], showlegend: false,
      x: p.x, y: p.y, xaxis: "x" + s, yaxis: "y" + s,
      marker: { color: "#2e86ab", size: mSize, opacity: mOpacity },
      hovertemplate: "Pred: %{x:.3f}<br>Obs: %{y:.3f}<extra></extra>",
    });
  }
  const { shapes, annotations } = _addDiagonalShapes(tps, axRange);
  const layout = _addAxesLayout(tps, axRange, "App + Tau score", "ApTt score (observed)");
  layout.showlegend = false;
  layout.shapes = shapes;
  layout.annotations = annotations;
  Plotly.react(el, traces, layout, { displaylogo: false, responsive: true });

  _writeStats(stats, tps, accs);
  if (sub) sub.textContent =
    `Backbone level · ${idx.length.toLocaleString()} in current filter` +
    (thinned ? ` (showing ${_ADD_BACKBONE_MAX_POINTS.toLocaleString()} sampled)` : "") +
    ` · predicted = App + Tau observed_score · observed = ApTt observed_score.`;
}

function renderAdditivity() {
  const el = document.getElementById("add-plot");
  if (!el) return;
  const level = Store.state.view.additivityLevel;
  if (level === "kinase") renderAdditivityKinase();
  else renderAdditivityBackbone();
}

function wireAdditivityControls() {
  const levelSel = document.getElementById("add-level");
  const tpSel = document.getElementById("add-tp");
  if (!levelSel || !tpSel) return;
  levelSel.value = Store.state.view.additivityLevel;
  tpSel.value = Store.state.view.additivityTimepoint;
  levelSel.addEventListener("change", ev =>
    Store.dispatch({type:"SET_VIEW", key:"additivityLevel", value: ev.target.value}));
  tpSel.addEventListener("change", ev =>
    Store.dispatch({type:"SET_VIEW", key:"additivityTimepoint", value: ev.target.value}));
  const scoreInp = document.getElementById("add-score-min");
  if (scoreInp) {
    scoreInp.value = Store.state.view.additivityScoreMin || 0;
    scoreInp.addEventListener("change", ev =>
      Store.dispatch({type:"SET_VIEW", key:"additivityScoreMin",
                      value: Math.max(0, parseFloat(ev.target.value) || 0)}));
  }
}

// ---------------------------------------------------------------------------
// Temporal v2 — series builder (draft)
// ---------------------------------------------------------------------------
// Each series is a predicate over (kinase, contrast). Bar height = unique
// kinases passing per (genotype, timepoint), split by NES sign when
// requested. Series stack as small multiples (one row per series) so y-scales
// stay independent.
let _tv2State = null;
let _tv2DecompCellsCache = null;
let _tv2AttrTierByKinCtx = null;  // Map<`${kid}|${cidx}`, Array<{cell, rank}>>

function _tv2EnsureAttrIndex() {
  if (_tv2AttrTierByKinCtx) return;
  _ensureKinaseIndexes();
  const AI = PAYLOAD.attribution_index || {kinase_id:[]};
  const m = new Map();
  for (let j = 0; j < AI.kinase_id.length; j++) {
    const kid = AI.kinase_id[j];
    const cidx = AI.contrast_id[j];
    const cell = AI.cell_type[j];
    const tier = _combinedTierFor(kid, cidx, cell, AI.combined_confidence[j]);
    const rank = _CONF_RANK[tier] || 0;
    const key = kid + "|" + cidx;
    let arr = m.get(key);
    if (!arr) { arr = []; m.set(key, arr); }
    arr.push({ cell, rank });
  }
  _tv2AttrTierByKinCtx = m;
}

function _tv2AttrPasses(ctxKey, cellsScope, threshold) {
  // Returns true if at least one attribution row at (kid, contrastIdx) within
  // the cell-type scope reaches the requested tier rank. threshold "" → pass.
  if (!threshold) return true;
  const wantRank = _CONF_RANK[threshold] || 0;
  if (wantRank <= 0) return true;
  const arr = _tv2AttrTierByKinCtx.get(ctxKey);
  if (!arr) return false;
  for (const r of arr) {
    if (cellsScope !== "ALL" && r.cell !== cellsScope) continue;
    if (r.rank >= wantRank) return true;
  }
  return false;
}

function _tv2DecompCellTypes() {
  if (_tv2DecompCellsCache) return _tv2DecompCellsCache;
  const D = PAYLOAD.decomposition_index || {cell_type:[]};
  const s = new Set();
  for (const c of D.cell_type) s.add(c);
  _tv2DecompCellsCache = Array.from(s).sort();
  return _tv2DecompCellsCache;
}

function _tv2DefaultSeries(layer) {
  return {
    layer: layer || "bulk",
    cells: "ALL",
    sign: "signed",
    fdrBulk: 0.25,
    fdrDecomp: 0.25,
    agree: true,
    attrTier: "",   // "" any | "low" | "moderate" | "high" | "very_high"
  };
}

function _tv2InitState() {
  if (_tv2State) return;
  _tv2State = { series: [_tv2DefaultSeries("bulk")], shareY: false };
}

function _tv2Eval(series, kid, contrastIdx) {
  // Returns null if the kinase fails the predicate at this contrast,
  // else { sign: -1|0|+1 } based on bulk NES (or decomp NES when bulk absent).
  const K = PAYLOAD.kinases;
  const cName = CONTRASTS[contrastIdx];
  const bulkNesCol = K["NES_" + cName];
  const bulkFdrCol = K["FDR_" + cName];
  const bulkNes = bulkNesCol ? bulkNesCol[kid] : null;
  const bulkFdr = bulkFdrCol ? bulkFdrCol[kid] : null;
  const bulkSig = bulkFdr != null && isFinite(bulkFdr) && bulkFdr < series.fdrBulk;

  const ctxKey = kid + "|" + contrastIdx;
  let dRows = (_decompByKinCtx && _decompByKinCtx.get(ctxKey)) || [];
  if (series.cells !== "ALL") dRows = dRows.filter(r => r.cell_type === series.cells);
  // For each decomp row at this kinase × contrast: sig + sign-vs-bulk.
  let decompAnyPass = false;        // any decomp row sig at fdrDecomp (sign-agnostic)
  let decompAgreePass = false;      // ≥1 decomp row sig AND sign-matches bulk
  let decompDisagreePass = false;   // ≥1 decomp row sig AND sign-disagrees with bulk
  let decompSignNes = null;
  for (const r of dRows) {
    if (r.fdr == null || !isFinite(r.fdr) || r.fdr >= series.fdrDecomp) continue;
    if (r.nes == null || !isFinite(r.nes) || r.nes === 0) continue;
    decompAnyPass = true;
    if (decompSignNes == null || Math.abs(r.nes) > Math.abs(decompSignNes)) {
      decompSignNes = r.nes;
    }
    if (bulkNes != null && bulkNes !== 0) {
      if ((r.nes > 0) === (bulkNes > 0)) decompAgreePass = true;
      else decompDisagreePass = true;
    }
  }

  let pass, refNes;
  if (series.layer === "bulk") { pass = bulkSig; refNes = bulkNes; }
  else if (series.layer === "decomp") { pass = decompAnyPass; refNes = decompSignNes; }
  else if (series.layer === "intersect") {
    pass = bulkSig && (series.agree ? decompAgreePass : decompAnyPass);
    refNes = bulkNes;
  }
  else if (series.layer === "contested") { pass = bulkSig && decompDisagreePass; refNes = bulkNes; }
  else if (series.layer === "diff") { pass = bulkSig && !decompAnyPass; refNes = bulkNes; }
  else { pass = false; refNes = null; }
  if (!pass) return null;
  // Attribution-tier gate: applies to any series, scoped to the same cells set.
  if (series.attrTier && !_tv2AttrPasses(ctxKey, series.cells, series.attrTier)) {
    return null;
  }

  const sign = (refNes == null || refNes === 0) ? 0 : (refNes > 0 ? 1 : -1);
  if (series.sign === "up" && sign < 0) return null;
  if (series.sign === "down" && sign > 0) return null;
  return { sign };
}

function _tv2Counts(series) {
  // Returns counts[g][t] = { up, down, total, upIds, downIds, totalIds } of unique kinases.
  _tv2EnsureAttrIndex();
  const K = PAYLOAD.kinases;
  const DG = META.diseaseGroups;
  const TPS = META.timepoints;
  const counts = {};
  // Hoist (g, t) → contrast-index lookup out of the per-kinase loop.
  const gtPairs = [];
  for (const g of DG) {
    counts[g] = {};
    for (const t of TPS) {
      counts[g][t] = {
        up: 0, down: 0, total: 0,
        upIds: [], downIds: [], totalIds: [],
      };
      const cIdx = CONTRASTS.indexOf(g + "_" + t);
      if (cIdx >= 0) gtPairs.push({ g, t, cIdx });
    }
  }
  const n = K.id.length;
  for (let i = 0; i < n; i++) {
    const kid = K.id[i];
    for (const p of gtPairs) {
      const r = _tv2Eval(series, kid, p.cIdx);
      if (!r) continue;
      const cell = counts[p.g][p.t];
      cell.total++;
      cell.totalIds.push(kid);
      if (r.sign > 0) { cell.up++; cell.upIds.push(kid); }
      else if (r.sign < 0) { cell.down++; cell.downIds.push(kid); }
    }
  }
  return counts;
}

function _tv2SeriesLabel(series) {
  const layerLabels = { bulk: "Bulk", decomp: "Decomp",
                         intersect: "Bulk ∩ Decomp", contested: "Bulk vs Decomp (contested)",
                         diff: "Bulk \\ Decomp" };
  const parts = [layerLabels[series.layer] || series.layer];
  if (series.layer !== "bulk") {
    parts.push(series.cells === "ALL" ? "any cell type" : series.cells);
  }
  if (series.layer !== "decomp") parts.push(`bulk FDR<${series.fdrBulk}`);
  if (series.layer !== "bulk") parts.push(`decomp FDR<${series.fdrDecomp}`);
  if (series.layer === "intersect") parts.push(series.agree ? "sign agree" : "any sign");
  if (series.attrTier) {
    const lbl = { very_high: "attr=very_high", high: "attr≥high",
                  moderate: "attr≥moderate", low: "attr≥low" };
    parts.push(lbl[series.attrTier] || ("attr≥" + series.attrTier));
  }
  if (series.sign !== "signed") parts.push(series.sign);
  return parts.join(" · ");
}

function _tv2RenderSeriesRow(series, idx) {
  const cells = _tv2DecompCellTypes();
  const cellOpts = ['<option value="ALL">any (OR)</option>']
    .concat(cells.map(c => `<option value="${c}">${c}</option>`)).join("");
  const layerOpts = [
    ['bulk', 'bulk'], ['decomp', 'decomp'],
    ['intersect', 'bulk ∩ decomp (corroborated)'],
    ['contested', 'bulk ∩ decomp (contested)'],
    ['diff', 'bulk \\ decomp (bulk-only)'],
  ].map(([v, l]) => `<option value="${v}">${l}</option>`).join("");
  const signOpts = [
    ['signed', 'signed (up/down)'], ['up', 'up only'],
    ['down', 'down only'], ['either', 'either (total)'],
  ].map(([v, l]) => `<option value="${v}">${l}</option>`).join("");
  const attrOpts = [
    ['', 'Any'], ['very_high', 'very high (only)'], ['high', 'high+'],
    ['moderate', 'moderate+'], ['low', 'low+'],
  ].map(([v, l]) => `<option value="${v}">${l}</option>`).join("");
  const cellsDisabled = (series.layer === "bulk");
  const agreeDisabled = (series.layer !== "intersect");
  const showBulkFdr = (series.layer !== "decomp");
  const showDecompFdr = (series.layer !== "bulk");
  const disParts = [];
  if (cellsDisabled) disParts.push("cells");
  if (agreeDisabled) disParts.push("agree");
  const disAttr = disParts.length ? ` data-disabled="${disParts.join(' ')}"` : '';
  return `<div class="tv2-row" data-idx="${idx}"${disAttr}>
    <span class="tv2-label">Series ${idx + 1}</span>
    <label>Layer <select class="tv2-layer">${layerOpts}</select></label>
    <label class="tv2-cells">Cells <select class="tv2-cells-sel">${cellOpts}</select></label>
    <label>Sign <select class="tv2-sign">${signOpts}</select></label>
    ${showBulkFdr ? `<label>Bulk FDR<input class="tv2-fdr-bulk" type="number" min="0" max="1" step="0.01" style="width:54px;"></label>` : ''}
    ${showDecompFdr ? `<label>Decomp FDR<input class="tv2-fdr-decomp" type="number" min="0" max="1" step="0.01" style="width:54px;"></label>` : ''}
    <label title="Require ≥1 attribution row in scope reaching this confidence tier (very_high = high+decomp agree, high = WMB+concordance, etc.).">Attr <select class="tv2-attr">${attrOpts}</select></label>
    <label class="tv2-agree" title="When on, decomp row must match bulk NES sign to count as corroboration."><input class="tv2-agree-cb" type="checkbox"> sign agree</label>
    <button class="tv2-rm" title="Remove this series">×</button>
  </div>`;
}

function _tv2WireSeriesRow(rowEl, idx) {
  const s = _tv2State.series[idx];
  const layerSel = rowEl.querySelector(".tv2-layer");
  const cellsSel = rowEl.querySelector(".tv2-cells-sel");
  const signSel = rowEl.querySelector(".tv2-sign");
  const fdrB = rowEl.querySelector(".tv2-fdr-bulk");
  const fdrD = rowEl.querySelector(".tv2-fdr-decomp");
  const agreeCb = rowEl.querySelector(".tv2-agree-cb");
  const attrSel = rowEl.querySelector(".tv2-attr");
  const rmBtn = rowEl.querySelector(".tv2-rm");
  layerSel.value = s.layer;
  cellsSel.value = s.cells;
  signSel.value = s.sign;
  if (fdrB) fdrB.value = s.fdrBulk;
  if (fdrD) fdrD.value = s.fdrDecomp;
  agreeCb.checked = !!s.agree;
  if (attrSel) attrSel.value = s.attrTier || "";
  layerSel.addEventListener("change", () => {
    s.layer = layerSel.value;
    if (s.layer === "bulk") s.cells = "ALL";
    _tv2RenderUI(); renderTemporalV2();
  });
  cellsSel.addEventListener("change", () => { s.cells = cellsSel.value; renderTemporalV2(); });
  signSel.addEventListener("change", () => { s.sign = signSel.value; renderTemporalV2(); });
  if (fdrB) fdrB.addEventListener("change", () => {
    const v = parseFloat(fdrB.value); if (isFinite(v) && v > 0 && v <= 1) {
      s.fdrBulk = v; renderTemporalV2();
    } else { fdrB.value = s.fdrBulk; }
  });
  if (fdrD) fdrD.addEventListener("change", () => {
    const v = parseFloat(fdrD.value); if (isFinite(v) && v > 0 && v <= 1) {
      s.fdrDecomp = v; renderTemporalV2();
    } else { fdrD.value = s.fdrDecomp; }
  });
  agreeCb.addEventListener("change", () => { s.agree = agreeCb.checked; renderTemporalV2(); });
  if (attrSel) attrSel.addEventListener("change", () => {
    s.attrTier = attrSel.value; renderTemporalV2();
  });
  rmBtn.addEventListener("click", () => {
    _tv2State.series.splice(idx, 1);
    if (_tv2State.series.length === 0) _tv2State.series.push(_tv2DefaultSeries("bulk"));
    _tv2RenderUI(); renderTemporalV2();
  });
}

function _tv2RenderUI() {
  const list = document.getElementById("tv2-series-list");
  if (!list) return;
  list.innerHTML = _tv2State.series.map((s, i) => _tv2RenderSeriesRow(s, i)).join("");
  list.querySelectorAll(".tv2-row").forEach((row, i) => _tv2WireSeriesRow(row, i));
}

function _tv2ApplyPreset(name) {
  const cells = _tv2DecompCellTypes();
  if (name === "bulk_only") {
    _tv2State.series = [_tv2DefaultSeries("bulk")];
  } else if (name === "bulk_corrob_contest") {
    const corrob = _tv2DefaultSeries("intersect"); corrob.agree = true;
    const contest = _tv2DefaultSeries("contested");
    _tv2State.series = [_tv2DefaultSeries("bulk"), corrob, contest];
  } else if (name === "bulk_vs_decomp") {
    _tv2State.series = [_tv2DefaultSeries("bulk"), _tv2DefaultSeries("decomp")];
  } else if (name === "bulk_attr_vs_decomp") {
    const bulkAttr = _tv2DefaultSeries("bulk"); bulkAttr.attrTier = "high";
    _tv2State.series = [
      _tv2DefaultSeries("bulk"),
      bulkAttr,
      _tv2DefaultSeries("decomp"),
    ];
  } else if (name === "celltype_sweep") {
    _tv2State.series = cells.slice(0, Math.min(4, cells.length)).map(c => {
      const s = _tv2DefaultSeries("decomp"); s.cells = c; return s;
    });
    if (_tv2State.series.length === 0) _tv2State.series = [_tv2DefaultSeries("decomp")];
  }
  _tv2RenderUI();
  renderTemporalV2();
}

function renderTemporalV2() {
  const el = document.getElementById("tv2-plot");
  const sub = document.getElementById("tv2-subtitle");
  if (!el) return;
  _ensureKinaseIndexes();
  const series = _tv2State ? _tv2State.series : [];
  if (!series.length) {
    Plotly.purge(el);
    if (sub) sub.textContent = "No series defined. Click + Add series or pick a preset.";
    return;
  }
  const DG = META.diseaseGroups;
  const TPS = META.timepoints;
  const traces = [];
  const layout = {
    grid: { rows: series.length, columns: 1, pattern: "independent" },
    margin: { l: 70, r: 20, t: 20, b: 50 },
    height: Math.max(220, 200 * series.length + 40),
    barmode: "group", bargap: 0.25,
    legend: { orientation: "h", y: -0.1 / series.length },
    annotations: [],
  };
  // First pass: compute counts per series and (if shared y) the global range.
  const allCounts = series.map(ser => _tv2Counts(ser));
  let sharedRange = null;
  if (_tv2State.shareY) {
    let lo = 0, hi = 0;
    for (let s = 0; s < series.length; s++) {
      const ser = series[s];
      const counts = allCounts[s];
      for (const g of DG) for (const t of TPS) {
        const cell = counts[g][t];
        if (ser.sign === "signed") {
          if (cell.up > hi) hi = cell.up;
          if (-cell.down < lo) lo = -cell.down;
        } else {
          if (cell.total > hi) hi = cell.total;
        }
      }
    }
    const pad = Math.max(1, Math.ceil(Math.max(hi, -lo) * 0.05));
    sharedRange = [lo - (lo < 0 ? pad : 0), hi + pad];
  }
  for (let s = 0; s < series.length; s++) {
    const ser = series[s];
    const counts = allCounts[s];
    const sfx = (s === 0) ? "" : String(s + 1);
    const xAxis = "x" + sfx, yAxis = "y" + sfx;
    const showLegend = (s === 0);
    for (const g of DG) {
      const color = (META.diseaseColors || {})[g] || "#555";
      if (ser.sign === "signed") {
        traces.push({
          type: "bar", name: g + " up",
          x: TPS, y: TPS.map(t => counts[g][t].up),
          marker: { color }, legendgroup: g + "-up",
          offsetgroup: g, alignmentgroup: "v" + s,
          xaxis: xAxis, yaxis: yAxis, showlegend: showLegend,
          customdata: TPS.map(t => [counts[g][t].up, counts[g][t].upIds]),
          meta: { s, g, sign: "up" },
          hovertemplate: `[S${s+1}] ${g} up @ %{x}: %{customdata[0]} · click to open in Kinase tab<extra></extra>`,
        });
        traces.push({
          type: "bar", name: g + " down",
          x: TPS, y: TPS.map(t => -counts[g][t].down),
          marker: { color, opacity: 0.55 }, legendgroup: g + "-down",
          offsetgroup: g, alignmentgroup: "v" + s,
          xaxis: xAxis, yaxis: yAxis, showlegend: showLegend,
          customdata: TPS.map(t => [counts[g][t].down, counts[g][t].downIds]),
          meta: { s, g, sign: "down" },
          hovertemplate: `[S${s+1}] ${g} down @ %{x}: %{customdata[0]} · click to open in Kinase tab<extra></extra>`,
        });
      } else {
        traces.push({
          type: "bar", name: g,
          x: TPS, y: TPS.map(t => counts[g][t].total),
          marker: { color }, legendgroup: g,
          offsetgroup: g, alignmentgroup: "v" + s,
          xaxis: xAxis, yaxis: yAxis, showlegend: showLegend,
          customdata: TPS.map(t => [counts[g][t].total, counts[g][t].totalIds]),
          meta: { s, g, sign: "total" },
          hovertemplate: `[S${s+1}] ${g} @ %{x}: %{customdata[0]} · click to open in Kinase tab<extra></extra>`,
        });
      }
    }
    layout["xaxis" + sfx] = {
      title: (s === series.length - 1) ? "Timepoint" : "",
      anchor: "y" + sfx,
    };
    layout["yaxis" + sfx] = {
      title: "n kinases",
      zeroline: true,
      anchor: "x" + sfx,
    };
    if (ser.sign === "signed") {
      layout["yaxis" + sfx].zerolinecolor = "#000";
      layout["yaxis" + sfx].zerolinewidth = 1;
    }
    if (sharedRange) layout["yaxis" + sfx].range = sharedRange;
    layout.annotations.push({
      xref: "paper", yref: "paper",
      x: 0, xanchor: "left",
      y: 1 - (s / series.length) - 0.02 / series.length,
      yanchor: "top",
      text: `<b>S${s + 1}</b> · ${_tv2SeriesLabel(ser)}`,
      showarrow: false, font: { size: 11, color: "#37474f" },
    });
  }
  Plotly.react(el, traces, layout, { displaylogo: false, responsive: true });
  el.removeAllListeners && el.removeAllListeners("plotly_click");
  el.on && el.on("plotly_click", ev => {
    if (!ev.points || !ev.points.length) return;
    const p = ev.points[0];
    const cd = p.customdata;
    const meta = (p.data && p.data.meta) || {};
    if (!cd || !Array.isArray(cd) || !cd[1] || !cd[1].length) return;
    const ids = cd[1];
    const ser = (_tv2State.series || [])[meta.s] || {};
    const label = `Temporal v2 · ${_tv2SeriesLabel(ser)} · ${meta.g}_${p.x}` +
                  (meta.sign === "total" ? "" : ` · ${meta.sign}`);
    _openKinaseDeepDiveWithWhitelist(ids, label, {
      genotype: meta.g, timepoint: p.x,
      cells: ser.cells, attrTier: ser.attrTier,
    });
  });
  if (sub) {
    sub.textContent = `${series.length} series · y = unique kinases per (genotype, timepoint) · `
      + `signed series split up at +y, down at −y · scales independent across rows · `
      + `click any bar to open the Kinase deep dive filtered to that kinase set.`;
  }
}

function _openKinaseDeepDiveWithWhitelist(kinaseIds, sourceLabel, ctx) {
  if (typeof KinaseFilter === "undefined" || !KinaseFilter.setWhitelist) {
    console.warn("KinaseFilter whitelist not available");
    return;
  }
  // Prefill the filter dropdowns from the bar's context so the user can see
  // and edit the implied scope. The whitelist is stored separately and ANDs
  // with these filters when "Stack with filters" is toggled on; otherwise
  // the dropdowns are visible-but-inactive (the whitelist takes precedence).
  if (ctx) {
    const patch = {
      disease:    ctx.genotype  ? [ctx.genotype]  : [],
      timepoint:  ctx.timepoint ? [ctx.timepoint] : [],
      celltype:   (ctx.cells && ctx.cells !== "ALL") ? [ctx.cells] : [],
      confidence: ctx.attrTier || "",
      // n_sig isn't part of the bar context; leave whatever the user had.
    };
    KinaseFilter.set(patch);
  }
  // New whitelists default to bypass mode (stack=false) so the user sees the
  // full clicked set first, then opts into stacking with the toggle.
  KinaseFilter.setWhitelist(kinaseIds.slice(), sourceLabel);
  KinaseFilter.setWhitelistStack(false);
  Store.dispatch({type:"SET_VIEW", key:"activeTab", value:"kinase"});
  // Push the prefilled state into the visible toolbar inputs after the tab
  // has been swapped in. Defer to next frame because syncTabsFromStore runs
  // synchronously inside the dispatch handler and may unhide the panel.
  if (typeof _syncKinaseFilterUI === "function") {
    setTimeout(_syncKinaseFilterUI, 0);
  }
}

function wireTemporalV2() {
  _tv2InitState();
  _tv2RenderUI();
  document.querySelectorAll("#tv2-presets [data-tv2-preset]").forEach(btn => {
    btn.addEventListener("click", () => _tv2ApplyPreset(btn.dataset.tv2Preset));
  });
  const addBtn = document.getElementById("tv2-add-series");
  if (addBtn) addBtn.addEventListener("click", () => {
    _tv2State.series.push(_tv2DefaultSeries("bulk"));
    _tv2RenderUI(); renderTemporalV2();
  });
  const clrBtn = document.getElementById("tv2-clear");
  if (clrBtn) clrBtn.addEventListener("click", () => {
    _tv2State.series = [_tv2DefaultSeries("bulk")];
    _tv2RenderUI(); renderTemporalV2();
  });
  const shareCb = document.getElementById("tv2-share-y");
  if (shareCb) {
    shareCb.checked = !!_tv2State.shareY;
    shareCb.addEventListener("change", () => {
      _tv2State.shareY = shareCb.checked;
      renderTemporalV2();
    });
  }
}

// ---------------------------------------------------------------------------
// Kinase Explorer tab
// ---------------------------------------------------------------------------

// Single filter-state object replacing scattered module-level vars and
// window._keFilters. Backed by localStorage key kinaseFilter.v4.
// (v1/v2/v3 keys are intentionally ignored — schema changed.)
window.KinaseFilter = (function() {
  const _KEY = "kinaseFilter.v4";
  const _defaults = {
    search: "",
    disease: [],      // [] = any; otherwise array of "App"|"Tau"|"ApTt"
    timepoint: [],    // [] = any; otherwise array of "2mo"|"4mo"|"6mo"
    celltype: [],     // [] = any; otherwise array of subclass strings
    confidence: "",   // "" | "high" | "moderate" | "low" — ordinal threshold (≥)
    nSigMin: 0,       // minimum n_sig (count of significant contrasts in scope)
    wmbMin: 0,        // 0 = any; 1/2/5/10 = minimum WMB specificity tier (× uniform)
    fdr: 0.25, sortCol: "nes_profile", sortAsc: false,
  };
  const _arrKeys = new Set(["disease","timepoint","celltype"]);
  let _state = Object.assign({}, _defaults);
  try {
    const saved = JSON.parse(localStorage.getItem(_KEY) || "null");
    if (saved && typeof saved === "object") {
      for (const k of Object.keys(_defaults)) {
        if (k in saved) {
          if (_arrKeys.has(k)) _state[k] = Array.isArray(saved[k]) ? saved[k].slice() : [];
          else _state[k] = saved[k];
        }
      }
    }
  } catch(e) {}
  const _subs = [];
  // Whitelist is in-memory only (NOT persisted) — survives tab switches but not
  // page reloads. Set by cross-tab handoffs (e.g. Temporal v2 bar click); when
  // active, the kinase explorer bypasses attribution / n_sig / confidence gates
  // and shows exactly the listed kinase IDs.
  let _whitelist = null;       // null | Set<number>
  let _whitelistLabel = "";    // human-readable source description
  let _whitelistStack = false; // false = whitelist bypass; true = AND with dropdowns
  function _save() {
    try { localStorage.setItem(_KEY, JSON.stringify(_state)); } catch(e) {}
  }
  return {
    get: function(k) { return k ? _state[k] : Object.assign({}, _state); },
    getWhitelist: function() {
      return _whitelist ? { ids: _whitelist, label: _whitelistLabel,
                            stack: _whitelistStack } : null;
    },
    setWhitelist: function(ids, label) {
      _whitelist = new Set(ids);
      _whitelistLabel = label || "";
      for (const fn of _subs) fn();
    },
    clearWhitelist: function() {
      if (_whitelist === null) return;
      _whitelist = null; _whitelistLabel = ""; _whitelistStack = false;
      for (const fn of _subs) fn();
    },
    setWhitelistStack: function(on) {
      const v = !!on;
      if (_whitelistStack === v) return;
      _whitelistStack = v;
      for (const fn of _subs) fn();
    },
    set: function(patch) {
      let changed = false;
      for (const k of Object.keys(patch)) {
        const nv = patch[k];
        if (_arrKeys.has(k)) {
          const cur = _state[k] || [];
          const a = Array.isArray(nv) ? nv.slice() : [];
          if (cur.length !== a.length || cur.some((v,i) => v !== a[i])) {
            _state[k] = a; changed = true;
          }
        } else if (_state[k] !== nv) {
          _state[k] = nv; changed = true;
        }
      }
      if (changed) { _save(); for (const fn of _subs) fn(); }
    },
    reset: function() {
      _state = JSON.parse(JSON.stringify(_defaults));
      _state.fdr = Store.state.filters.fdr || 0.25;
      _save();
      for (const fn of _subs) fn();
    },
    subscribe: function(fn) { _subs.push(fn); },
  };
})();

// Back-compat shim so any code reading window._keFilters still works.
// Multiselect: collapse to single selection if exactly one chosen, else "".
Object.defineProperty(window, "_keFilters", {
  get: function() {
    const f = KinaseFilter.get();
    const one = a => (Array.isArray(a) && a.length === 1) ? a[0] : "";
    return { disease: one(f.disease), tp: one(f.timepoint),
             celltype: one(f.celltype), trajectory: "" };
  },
  configurable: true,
});

let _keRows = null;
let _keSigFdr = null;
let _kinaseIdxById = null;
let _backboneIdxById = null;
let _evidenceByKinase = null;
let _presentKinaseSet = null;
let _decompByKey = null;
let _decompByKinCtx = null;
let _agreementByKey = null;
const _AGREEMENT_STATE_NAMES = ["neither_sig","agree","mixed","disagree","bulk_only","decomp_only"];

// ---------------------------------------------------------------------------
// Scoped attribution helpers (single source of truth: PAYLOAD.attribution_index)
// ---------------------------------------------------------------------------

// Coerce a filter dimension value to a Set of selected values.
// Accepts: undefined/null/"" → empty (any), string (single) → {string},
// array → set of array entries.
function _filterSet(v) {
  const s = new Set();
  if (v == null || v === "") return s;
  if (Array.isArray(v)) { for (const x of v) if (x !== "" && x != null) s.add(x); return s; }
  s.add(v);
  return s;
}

function getScopedContrastIds(filter) {
  // Returns Set of contrast indices matching the filter's disease × timepoint
  // selection sets. Empty set on a dimension = any.
  const ds = _filterSet(filter.disease);
  const ts = _filterSet(filter.timepoint);
  const ids = new Set();
  for (let ci = 0; ci < CONTRASTS.length; ci++) {
    const c = CONTRASTS[ci];
    const d = c.split("_")[0];
    const m = c.match(/_(\d+mo)$/);
    const t = m ? m[1] : "";
    if (ds.size && !ds.has(d)) continue;
    if (ts.size && !ts.has(t)) continue;
    ids.add(ci);
  }
  return ids;
}

// Confidence threshold: a row passes if its tier rank ≥ requested rank.
// "" / undefined → no constraint.
const _CONF_RANK = {very_high: 4, high: 3, moderate: 2, low: 1, none: 0};
function _confPass(rowConf, threshold) {
  if (!threshold) return true;
  return (_CONF_RANK[rowConf] || 0) >= (_CONF_RANK[threshold] || 0);
}

// WMB specificity tiers expressed as multiples of uniform (1/34 ≈ 0.0294 across
// 34 WMB classes). 10× / 5× / 2× / 1× → 0.294 / 0.147 / 0.059 / 0.029.
const _WMB_UNIFORM = 1 / 34;
const _WMB_TIER_VALUES = [10, 5, 2, 1];
function _wmbTier(s) {
  if (s == null || !isFinite(s)) return 0;
  for (const t of _WMB_TIER_VALUES) {
    if (s >= t * _WMB_UNIFORM) return t;
  }
  return 0;
}
function _wmbTierLabel(t) { return t > 0 ? "≥" + t + "×" : ""; }
function _wmbTierBadge(t) {
  if (!t) return '<span class="muted">—</span>';
  const cls = t >= 10 ? "vhi" : (t >= 5 ? "hi" : (t >= 2 ? "mid" : "lo"));
  return '<span class="badge ' + cls + '" title="WMB specificity ≥ ' + t +
         '× uniform (' + (t * _WMB_UNIFORM).toFixed(3) + ')">' + _wmbTierLabel(t) + '</span>';
}

// Decomp step ordinal vs bulk MEA direction:
//   3 strong-agree (FDR<0.10), 2 sig-agree (FDR<0.25), 1 nominal,
//   0 absent, -2 sig-disagree (FDR<0.25, sign opposes bulk).
// bulkNes may be passed in pre-fetched; otherwise read from PAYLOAD.kinases.
function _decompStep(decompNes, decompFdr, bulkNes) {
  if (decompNes == null || !isFinite(decompNes) || decompNes === 0) return 0;
  if (bulkNes == null || !isFinite(bulkNes) || bulkNes === 0) return 1;
  const agree = (decompNes > 0) === (bulkNes > 0);
  const sig = decompFdr != null && isFinite(decompFdr) && decompFdr < 0.25;
  const strong = decompFdr != null && isFinite(decompFdr) && decompFdr < 0.10;
  if (agree) return strong ? 3 : (sig ? 2 : 1);
  return sig ? -2 : 1;
}

function _decompStepFor(kid, contrastId, cellType) {
  if (!_decompByKey) return 0;
  const d = _decompByKey.get(`${kid}|${contrastId}|${cellType}`);
  if (!d) return 0;
  const cName = CONTRASTS[contrastId];
  const _K = PAYLOAD.kinases;
  const bulkNes = (_K && cName && _K["NES_" + cName]) ? _K["NES_" + cName][kid] : null;
  return _decompStep(d.nes, d.fdr, bulkNes);
}

// Apply the very_high upgrade rule: a "high" attribution row whose decomp
// significantly agrees with the bulk direction is promoted.
function _upgradeTier(attrConf, decompStep) {
  if (attrConf === "high" && decompStep >= 2) return "very_high";
  return attrConf || "none";
}

function _combinedTierFor(kid, contrastId, cellType, attrConf) {
  if (attrConf !== "high") return attrConf || "none";
  return _upgradeTier(attrConf, _decompStepFor(kid, contrastId, cellType));
}

function getScopedAttribution(kinaseId, filter) {
  // Returns filtered rows from PAYLOAD.attribution_index for one kinase.
  // filter: { disease, timepoint, celltype, confidence } where dimension values
  // may be string ("" = any) or array ([] = any).
  const AI = PAYLOAD.attribution_index || {};
  if (!AI.kinase_id) return [];
  const scopedCtx = getScopedContrastIds(filter);
  const ctSet = _filterSet(filter.celltype);
  const confidence = filter.confidence || "";
  const out = [];
  for (let j = 0; j < AI.kinase_id.length; j++) {
    if (AI.kinase_id[j] !== kinaseId) continue;
    if (scopedCtx.size > 0 && !scopedCtx.has(AI.contrast_id[j])) continue;
    if (ctSet.size && !ctSet.has(AI.cell_type[j]))                continue;
    const _attrConf = AI.combined_confidence[j];
    const _tier = _combinedTierFor(kinaseId, AI.contrast_id[j], AI.cell_type[j], _attrConf);
    // Confidence threshold tests the upgraded tier so "very_high" filters work.
    if (!_confPass(_tier, confidence))                            continue;
    out.push({
      contrast_id:               AI.contrast_id[j],
      cell_type:                 AI.cell_type[j],
      combined_confidence:       _attrConf,
      combined_tier:             _tier,
      combined_score:            AI.combined_score[j],
      wmb_specificity:           AI.wmb_specificity            ? AI.wmb_specificity[j]            : null,
      wmb_mean_log2_expression:  AI.wmb_mean_log2_expression   ? AI.wmb_mean_log2_expression[j]   : null,
      wmb_fraction_cells_expressing: AI.wmb_fraction_cells_expressing ? AI.wmb_fraction_cells_expressing[j] : null,
      wmb_binary_expressed:      AI.wmb_binary_expressed       ? AI.wmb_binary_expressed[j]       : false,
      sea_ad_lfc:                AI.sea_ad_lfc                 ? AI.sea_ad_lfc[j]                 : null,
      song_lfc:                  AI.song_lfc                   ? AI.song_lfc[j]                   : null,
      song_pval:                 AI.song_pval                  ? AI.song_pval[j]                  : null,
      song_fdr:                  AI.song_fdr                   ? AI.song_fdr[j]                   : null,
      concordance_source:        AI.concordance_source         ? AI.concordance_source[j]         : "",
      nes:                       AI.nes                        ? AI.nes[j]                        : null,
      fdr:                       AI.fdr                        ? AI.fdr[j]                        : null,
    });
  }
  return out;
}

// Cross-grid AND coverage: kinase passes iff for every cell of the selected
// sub-grid (selected diseases × timepoints × cell types) ≥1 attribution row
// exists at the requested confidence threshold. Empty selection on a dimension
// = wildcard for that axis. Trajectory is per-kinase scalar (OR), checked
// upstream. Search is also handled upstream.
function kinaseQualifies(kinaseId, filter) {
  const rows = getScopedAttribution(kinaseId, filter);
  if (rows.length === 0) return false;
  const dSet = _filterSet(filter.disease);
  const tSet = _filterSet(filter.timepoint);
  const cSet = _filterSet(filter.celltype);
  if (!dSet.size && !tSet.size && !cSet.size) return true;
  const Ds = dSet.size ? Array.from(dSet) : [null];
  const Ts = tSet.size ? Array.from(tSet) : [null];
  const Cs = cSet.size ? Array.from(cSet) : [null];
  // Pre-decode contrast → (disease, timepoint) once.
  const decoded = new Array(rows.length);
  for (let i = 0; i < rows.length; i++) {
    const ctx = CONTRASTS[rows[i].contrast_id] || "";
    const d = ctx.split("_")[0];
    const m = ctx.match(/_(\d+mo)$/);
    decoded[i] = { d, t: m ? m[1] : "", c: rows[i].cell_type };
  }
  for (const d of Ds) {
    for (const t of Ts) {
      for (const c of Cs) {
        let ok = false;
        for (let i = 0; i < decoded.length; i++) {
          const e = decoded[i];
          if (d != null && e.d !== d) continue;
          if (t != null && e.t !== t) continue;
          if (c != null && e.c !== c) continue;
          ok = true; break;
        }
        if (!ok) return false;
      }
    }
  }
  return true;
}

function _buildKinaseRowModel() {
  const K = PAYLOAD.kinases;
  const KDB = PAYLOAD.kinase_distinct_backbones || {kinase_id:[], n_distinct_backbones:[]};
  const famMap = META.familyMap || {};
  const bbByK = new Array(K.id.length).fill(0);
  for (let i = 0; i < KDB.kinase_id.length; i++) {
    bbByK[KDB.kinase_id[i]] = KDB.n_distinct_backbones[i];
  }
  const idxById = new Map();
  const out = [];
  for (let i = 0; i < K.id.length; i++) {
    idxById.set(K.id[i], i);
    out.push({
      id: K.id[i],
      name: K.name[i],
      gene_symbol: K.gene_symbol[i] || "",
      family: famMap[K.name[i]] || "",
      residue_type: (K.residue_type && K.residue_type[i]) || "ST",
      has_edges: K.has_edges ? !!K.has_edges[i] : true,
      trajectory: K.trajectory[i] || "",
      peak_contrast: K.peak_contrast[i] || "",
      peak_NES: K.peak_NES[i],
      top_celltype_1: K.top_celltype_1[i] || "",
      n_backbones: bbByK[i],
      _fdr: CONTRASTS.map(c => K["FDR_" + c][i]),
      _nes: CONTRASTS.map(c => K["NES_" + c][i]),
    });
  }
  _kinaseIdxById = idxById;
  return out;
}

function _ensureBackboneIdx() {
  if (_backboneIdxById !== null) return;
  const BB = PAYLOAD.backbones;
  const m = new Map();
  for (let i = 0; i < BB.id.length; i++) m.set(BB.id[i], i);
  _backboneIdxById = m;
}

function _ensureKinaseIdx() {
  if (_kinaseIdxById !== null) return;
  const K = PAYLOAD.kinases;
  const m = new Map();
  for (let i = 0; i < K.id.length; i++) m.set(K.id[i], i);
  _kinaseIdxById = m;
}

function _ensureKinaseIndexes() {
  if (_keRows === null) _keRows = _buildKinaseRowModel();
  _ensureBackboneIdx();
  if (_evidenceByKinase === null) {
    const EV = PAYLOAD.kinase_celltype_evidence || {kinase_id:[]};
    const m = new Map();
    for (let k = 0; k < EV.kinase_id.length; k++) {
      const kid = EV.kinase_id[k];
      let arr = m.get(kid);
      if (!arr) { arr = []; m.set(kid, arr); }
      arr.push(k);
    }
    _evidenceByKinase = m;
  }
  if (_presentKinaseSet === null) {
    const esr = PAYLOAD.edge_slice_ref || {};
    _presentKinaseSet = new Set(esr.present_kinase_ids || []);
  }
  if (_decompByKey === null) {
    const D = PAYLOAD.decomposition_index || {kinase_id:[]};
    const m = new Map();
    const m2 = new Map();
    for (let k = 0; k < D.kinase_id.length; k++) {
      const key = `${D.kinase_id[k]}|${D.contrast_id[k]}|${D.cell_type[k]}`;
      m.set(key, {nes: D.decomp_nes[k], fdr: D.decomp_fdr[k]});
      const k2 = `${D.kinase_id[k]}|${D.contrast_id[k]}`;
      let arr = m2.get(k2);
      if (!arr) { arr = []; m2.set(k2, arr); }
      arr.push({cell_type: D.cell_type[k], nes: D.decomp_nes[k], fdr: D.decomp_fdr[k]});
    }
    _decompByKey = m;
    _decompByKinCtx = m2;
  }
  if (_agreementByKey === null) {
    const A = PAYLOAD.agreement_index || {kinase_id:[]};
    const m = new Map();
    for (let k = 0; k < A.kinase_id.length; k++) {
      const key = `${A.kinase_id[k]}|${A.contrast_id[k]}`;
      m.set(key, {
        state: A.state[k],
        bulk_nes: A.bulk_nes[k],
        bulk_fdr: A.bulk_fdr[k],
        top_cell: A.top_cell[k],
        top_cell_nes: A.top_cell_nes[k],
        top_cell_fdr: A.top_cell_fdr[k],
        n_match: A.n_cells_match[k],
        n_oppose: A.n_cells_oppose[k],
      });
    }
    _agreementByKey = m;
  }
}

function _refreshSigCounts(fdr) {
  if (_keSigFdr === fdr) return;
  for (const r of _keRows) {
    let n = 0;
    for (const v of r._fdr) if (v != null && v < fdr) n++;
    r._sigCount = n;
  }
  _keSigFdr = fdr;
}

function _kineMaxAbsNesScoped(r, scopedCtxIds) {
  // Returns max |NES| among contrast indices in scopedCtxIds (all if empty Set).
  let best = null;
  for (let ci = 0; ci < CONTRASTS.length; ci++) {
    if (scopedCtxIds.size > 0 && !scopedCtxIds.has(ci)) continue;
    const v = r._nes[ci];
    if (v == null) continue;
    const a = Math.abs(v);
    if (best == null || a > best) best = a;
  }
  return best;
}

// Max WMB specificity tier across attribution rows for this kinase under the
// active filter scope. Returns 0 when no qualifying rows have wmb_specificity.
function _kineMaxWmbTierScoped(kinaseId, filter) {
  let best = 0;
  for (const e of getScopedAttribution(kinaseId, filter)) {
    const t = _wmbTier(Number(e.wmb_specificity));
    if (t > best) best = t;
  }
  return best;
}

function _kineSigCountScoped(r, fdr, scopedCtxIds) {
  let n = 0;
  for (let ci = 0; ci < CONTRASTS.length; ci++) {
    if (scopedCtxIds.size > 0 && !scopedCtxIds.has(ci)) continue;
    const q = r._fdr[ci];
    if (q != null && q < fdr) n++;
  }
  return n;
}

// Legacy lens array kept for any remaining references; new code uses
// getScopedContrastIds via KinaseFilter.
const NES_PROFILE_LENSES = [
  {key:"any",  label:"any disease"},
  {key:"App",  label:"App"},
  {key:"Tau",  label:"Tau"},
  {key:"ApTt", label:"ApTt"},
  {key:"nsig", label:"# sig contrasts"},
];

// Legacy helpers — kept for any surviving call sites (e.g. _renderMeaTrajectory
// still uses the ctx.contrast which uses _selectedAuditContrast).
function _kineMaxAbsNesIn(r, diseasePrefix, tpFilter) {
  let best = null;
  for (let ci = 0; ci < CONTRASTS.length; ci++) {
    const c = CONTRASTS[ci];
    if (diseasePrefix && c.indexOf(diseasePrefix) !== 0) continue;
    if (tpFilter && c.indexOf(tpFilter) < 0) continue;
    const v = r._nes[ci];
    if (v == null) continue;
    const a = Math.abs(v);
    if (best == null || a > best) best = a;
  }
  return best;
}

function _kineSigCountIn(r, fdr, diseasePrefix, tpFilter) {
  let n = 0;
  for (let ci = 0; ci < CONTRASTS.length; ci++) {
    const c = CONTRASTS[ci];
    if (diseasePrefix && c.indexOf(diseasePrefix) !== 0) continue;
    if (tpFilter && c.indexOf(tpFilter) < 0) continue;
    const q = r._fdr[ci];
    if (q != null && q < fdr) n++;
  }
  return n;
}

// _keCompare is called with a pre-computed scopedCtxIds set injected via closure.
// We wrap it in a factory called from renderKinaseExplorer.
function _makeKeCompare(scopedCtxIds) {
  const kf = KinaseFilter.get();
  const col = kf.sortCol || "nes_profile";
  const asc = !!kf.sortAsc;
  const fdr = kf.fdr || Store.state.filters.fdr || 0.25;
  return function(a, b) {
    let va, vb;
    if (col === "nes_profile") {
      va = _kineMaxAbsNesScoped(a, scopedCtxIds);
      vb = _kineMaxAbsNesScoped(b, scopedCtxIds);
      if (va == null) va = -Infinity;
      if (vb == null) vb = -Infinity;
    }
    else if (col === "n_attributed_celltypes") {
      // Match what the Cell types pill column displays: dedup by cell_type
      // keeping best tier, then count rows at moderate-or-better.
      const _bestTierByCT = (kid) => {
        const m = new Map();
        for (const e of getScopedAttribution(kid, kf)) {
          const r = _CONF_RANK[e.combined_tier] || 0;
          if (r > (m.get(e.cell_type) || 0)) m.set(e.cell_type, r);
        }
        let n = 0;
        for (const r of m.values()) if (r >= 2) n++;
        return n;
      };
      va = _bestTierByCT(a.id);
      vb = _bestTierByCT(b.id);
    }
    else if (col === "conf") {
      // Sort by max tier reached in scope: very_high(4) > high(3) > moderate(2) > low(1) > none(0).
      const _maxTier = (kid) => {
        let m = 0;
        for (const e of getScopedAttribution(kid, kf)) {
          const r = _CONF_RANK[e.combined_tier] || 0;
          if (r > m) m = r;
        }
        return m;
      };
      va = _maxTier(a.id);
      vb = _maxTier(b.id);
    }
    else if (col === "n_sig") {
      va = _kineSigCountScoped(a, fdr, scopedCtxIds);
      vb = _kineSigCountScoped(b, fdr, scopedCtxIds);
    }
    else if (col === "wmb_max_tier") {
      va = _kineMaxWmbTierScoped(a.id, kf);
      vb = _kineMaxWmbTierScoped(b.id, kf);
    }
    else if (col === "agreement_profile") {
      va = _kineDisagreeCountScoped(a, scopedCtxIds);
      vb = _kineDisagreeCountScoped(b, scopedCtxIds);
    }
    else if (col === "peak_NES") {
      // Scope-aware to match the column's displayed value.
      va = _kineMaxAbsNesScoped(a, scopedCtxIds);
      vb = _kineMaxAbsNesScoped(b, scopedCtxIds);
      if (va == null) va = -Infinity;
      if (vb == null) vb = -Infinity;
    }
    else { va = a[col]; vb = b[col]; }
    if (va == null && vb == null) return 0;
    if (va == null) return 1;
    if (vb == null) return -1;
    if (typeof va === "string") return asc
      ? va.localeCompare(vb) : vb.localeCompare(va);
    return asc ? (va - vb) : (vb - va);
  };
}

// Render the NES profile mini-heatmap (3 diseases × 3 timepoints) for one row.
// Always shows all 9 cells — this glyph IS the cross-contrast comparison.
function _renderNesProfile(r, fdrThresh, maxAbs) {
  const DG = META.diseaseGroups || ["App","Tau","ApTt"];
  const TPS = META.timepoints || ["2mo","4mo","6mo"];
  const cells = [];
  for (const d of DG) {
    for (const t of TPS) {
      const c = `${d}_${t}`;
      const ci = CONTRASTS.indexOf(c);
      const nes = ci >= 0 ? r._nes[ci] : null;
      const fdrV = ci >= 0 ? r._fdr[ci] : null;
      const sig = fdrV != null && fdrV < fdrThresh;
      let bg = "#fff";
      if (nes != null && isFinite(nes) && maxAbs > 0) {
        const a = Math.min(1, Math.abs(nes) / maxAbs);
        const rgb = nes >= 0 ? [197,48,48] : [43,108,176];
        bg = `rgba(${rgb[0]},${rgb[1]},${rgb[2]},${(0.15 + 0.85 * a).toFixed(3)})`;
      }
      const tip = nes == null ? `${c}: n/a`
        : `${c}: NES ${nes.toFixed(2)}${fdrV != null ? `, FDR ${fdrV.toExponential(1)}` : ""}${sig ? " (sig)" : ""}`;
      cells.push(`<div class="npc${sig ? " sig" : ""}" style="background:${bg};" title="${_escapeHtml(tip)}"></div>`);
    }
  }
  // Layout: rows = diseases (App/Tau/ApTt), cols = timepoints.
  const rowLabels = DG.map(d => `<span>${_escapeHtml(d)}</span>`).join("");
  return `<div class="nes-profile-wrap">` +
    `<div class="nes-profile-row-labels">${rowLabels}</div>` +
    `<div class="nes-profile-cell">${cells.join("")}</div>` +
    `</div>`;
}

function _agreementStateFor(kid, ci) {
  if (!_agreementByKey) return null;
  return _agreementByKey.get(`${kid}|${ci}`) || null;
}

function _kineDisagreeCountScoped(r, scopedCtxIds) {
  let n = 0;
  for (let ci = 0; ci < CONTRASTS.length; ci++) {
    if (scopedCtxIds && scopedCtxIds.size > 0 && !scopedCtxIds.has(ci)) continue;
    const a = _agreementStateFor(r.id, ci);
    if (a && a.state >= 2) n++;
  }
  return n;
}

function _renderAgreementProfile(r) {
  const DG = META.diseaseGroups || ["App","Tau","ApTt"];
  const TPS = META.timepoints || ["2mo","4mo","6mo"];
  const cells = [];
  for (const d of DG) {
    for (const t of TPS) {
      const c = `${d}_${t}`;
      const ci = CONTRASTS.indexOf(c);
      const a = ci >= 0 ? _agreementStateFor(r.id, ci) : null;
      let cls = "";
      let tip;
      if (!a) {
        tip = `${c}: neither pipeline significant`;
      } else {
        const stateName = _AGREEMENT_STATE_NAMES[a.state] || "?";
        if (a.state === 1) {
          cls = " agree";
          tip = `${c}: agree — bulk and decomp both significant, same direction`;
        } else {
          cls = " disagree";
          let detail;
          if (stateName === "decomp_only") detail = "bulk null, ≥1 decomp class significant";
          else if (stateName === "bulk_only") detail = "bulk significant, no decomp class significant";
          else if (stateName === "mixed") detail = "bulk significant, decomp classes split (some match, some oppose)";
          else if (stateName === "disagree") detail = "bulk significant, all sig decomp classes oppose bulk sign";
          else detail = stateName;
          tip = `${c}: ${stateName} — ${detail}`;
          if (a.top_cell) tip += ` · top decomp ${a.top_cell} NES ${Number(a.top_cell_nes).toFixed(2)}`;
          if (a.bulk_nes != null && isFinite(a.bulk_nes)) tip += ` · bulk NES ${Number(a.bulk_nes).toFixed(2)}`;
        }
      }
      cells.push(`<div class="apc${cls}" title="${_escapeHtml(tip)}"></div>`);
    }
  }
  const rowLabels = DG.map(d => `<span>${_escapeHtml(d)}</span>`).join("");
  return `<div class="agreement-profile-wrap">` +
    `<div class="nes-profile-row-labels">${rowLabels}</div>` +
    `<div class="agreement-profile-cell">${cells.join("")}</div>` +
    `</div>`;
}

function _renderCellTypesCell(r, filter) {
  // Reflect rows in the active filter scope; if filter is empty, scope = all 9 contrasts.
  const rows = getScopedAttribution(r.id, filter || {});
  const byCell = new Map();
  for (const e of rows) {
    const prev = byCell.get(e.cell_type);
    if (!prev || e.combined_score > prev.combined_score) byCell.set(e.cell_type, e);
  }
  const displayRows = Array.from(byCell.values()).filter(e =>
    e.combined_tier === "very_high" || e.combined_tier === "high" || e.combined_tier === "moderate");
  // Sort: tier first (very_high → high → moderate), then score desc within tier.
  displayRows.sort((a, b) => {
    const dt = (_CONF_RANK[b.combined_tier] || 0) - (_CONF_RANK[a.combined_tier] || 0);
    if (dt !== 0) return dt;
    return b.combined_score - a.combined_score;
  });
  const n = displayRows.length;
  if (n === 0) return `<span class="muted">—</span>`;
  const top = displayRows.slice(0, 3);
  const tip = displayRows.map(e => `${e.cell_type} (${(e.combined_tier || '').replace('_', ' ')}, ${e.combined_score.toFixed(2)})`).join("\n");
  const topNames = top.map(e => {
    const cls = e.combined_tier === "very_high" ? "vhi"
              : e.combined_tier === "high"      ? "hi"
              : "mid";
    return `<span class="badge ${cls}">${_escapeHtml(e.cell_type)}</span>`;
  }).join(" ");
  return `<span title="${_escapeHtml(tip)}"><strong>${n}</strong> ${topNames}${displayRows.length > 3 ? ` <span class="muted">+${displayRows.length - 3}</span>` : ""}</span>`;
}

function _renderKinaseWhitelistBanner(wl) {
  const wrap = document.querySelector(".ke-table-wrap");
  if (!wrap) return;
  let banner = document.getElementById("ke-whitelist-banner");
  if (!wl) {
    if (banner) banner.remove();
    return;
  }
  if (!banner) {
    banner = document.createElement("div");
    banner.id = "ke-whitelist-banner";
    banner.style.cssText = "background:#fff3cd; border:1px solid #f0ad4e; "
      + "color:#8a6d3b; padding:6px 10px; font-size:11px; border-radius:3px; "
      + "margin-bottom:6px; display:flex; align-items:center; gap:10px; "
      + "flex-wrap:wrap;";
    wrap.parentNode.insertBefore(banner, wrap);
  }
  const n = wl.ids.size;
  const lbl = wl.label || "external whitelist";
  const stackHint = wl.stack
    ? "Dropdowns AND with this set — turning them off broadens the result."
    : "Dropdowns are pre-filled with the click context but inactive. Toggle stack to apply them.";
  banner.innerHTML =
    `<span><b>Filtered to ${n} kinases</b> from ${_escapeHtml(lbl)}.</span>`
    + `<label style="display:flex; gap:4px; align-items:center;">`
    +   `<input type="checkbox" id="ke-whitelist-stack"${wl.stack ? " checked" : ""}> stack with filters`
    + `</label>`
    + `<span class="muted" style="flex:1; min-width:240px;">${stackHint}</span>`
    + `<button id="ke-whitelist-clear" class="chip">Clear filter</button>`;
  const stackCb = document.getElementById("ke-whitelist-stack");
  if (stackCb) stackCb.onchange = () => {
    KinaseFilter.setWhitelistStack(stackCb.checked);
    renderKinaseExplorer();
  };
  const btn = document.getElementById("ke-whitelist-clear");
  if (btn) btn.onclick = () => {
    KinaseFilter.clearWhitelist();
    renderKinaseExplorer();
  };
}

function renderKinaseExplorer() {
  const tbody = document.querySelector("#ke-table tbody");
  if (!tbody) return;
  _ensureKinaseIndexes();
  const kf = KinaseFilter.get();
  const fdr = kf.fdr || Store.state.filters.fdr || 0.25;
  const selKid = Store.state.selection.kinase;
  const q = (kf.search || "").trim().toLowerCase();
  const wl = KinaseFilter.getWhitelist();
  _renderKinaseWhitelistBanner(wl);

  _refreshSigCounts(fdr);

  // Scoped contrast IDs from the list filter (disease + timepoint) — used for
  // row inclusion (require ≥1 sig contrast in scope) and sort keys, NOT for
  // visualization scoping inside a row.
  const scopedCtxIds = getScopedContrastIds(kf);

  // Whether any attribution-grid filter is active (drives full qualification).
  const dSet = _filterSet(kf.disease);
  const tSet = _filterSet(kf.timepoint);
  const cSet = _filterSet(kf.celltype);
  const gridActive = dSet.size > 0 || tSet.size > 0 || cSet.size > 0 || !!kf.confidence;
  const nSigMin = Math.max(0, parseInt(kf.nSigMin, 10) || 0);
  const wmbMin = Math.max(0, parseInt(kf.wmbMin, 10) || 0);
  const wmbMinScore = wmbMin > 0 ? wmbMin * _WMB_UNIFORM : 0;

  // Whitelist mode (cross-tab handoff) has two sub-modes:
  //   stack=false (default): whitelist bypasses every other gate. Decomp-only
  //     kinases that would normally fail the attribution grid still appear.
  //   stack=true: whitelist ANDs with the normal filter chain. Useful for
  //     narrowing within a click-through set, but the attribution grid will
  //     drop kinases that lack attribution rows (interpretable empties).
  const visible = [];
  for (const r of _keRows) {
    if (wl) {
      if (!wl.ids.has(r.id)) continue;
      if (!wl.stack) { visible.push(r); continue; }
      // Stack mode: fall through to the normal predicate chain below.
    }
    // Text search
    if (q && !(r.name.toLowerCase().includes(q) ||
               r.gene_symbol.toLowerCase().includes(q))) continue;
    const scopedSig = _kineSigCountScoped(r, fdr, scopedCtxIds);
    if (!q) {
      // n_sig minimum (numeric filter).
      if (scopedSig < nSigMin) continue;
      // Disease/timepoint scope: require ≥1 sig contrast in scope.
      if (scopedCtxIds.size > 0 && scopedSig === 0) continue;
    }
    // Attribution grid: cross-product AND coverage on disease × timepoint × celltype,
    // with confidence as ordinal threshold (≥). Skipped when text search is
    // active so a targeted lookup (e.g. "EGFR") still surfaces the kinase even
    // if persisted localStorage filters would otherwise disqualify it.
    if (!q && gridActive && !kinaseQualifies(r.id, kf)) continue;
    // WMB tier minimum: kinase passes if any attribution row in scope has
    // wmb_specificity ≥ threshold. Independent of grid filters — uses the same
    // disease/timepoint/celltype scope getScopedAttribution honors.
    if (!q && wmbMin > 0) {
      const _rows = getScopedAttribution(r.id, kf);
      let _ok = false;
      for (const e of _rows) {
        const s = Number(e.wmb_specificity);
        if (isFinite(s) && s >= wmbMinScore) { _ok = true; break; }
      }
      if (!_ok) continue;
    }
    visible.push(r);
  }

  // maxAbsNes computed across all 9 contrasts on visible kinases — color is
  // a global comparison, not scope-restricted.
  let maxAbsNes = 0;
  for (const r of visible) {
    for (let ci = 0; ci < CONTRASTS.length; ci++) {
      const v = r._nes[ci];
      if (v != null && isFinite(v)) {
        const a = Math.abs(v);
        if (a > maxAbsNes) maxAbsNes = a;
      }
    }
  }
  if (maxAbsNes <= 0) maxAbsNes = 1;

  visible.sort(_makeKeCompare(scopedCtxIds));

  // Header arrows: show sort col + direction.
  document.querySelectorAll("#ke-table thead th").forEach(th => {
    const c = th.dataset.col;
    const sortCol = kf.sortCol || "nes_profile";
    const sortAsc = !!kf.sortAsc;
    th.textContent = th.textContent.replace(/[ ▲▼]+$/, "");
    th.textContent = th.textContent.replace(/\s*\(.*\)\s*$/, "");
    if (c === sortCol) th.textContent += sortAsc ? " ▲" : " ▼";
  });

  // Filter scope passed to per-row column renderers. If no grid filter is
  // active, fall back to all-9-contrasts scope for those columns.
  const colFilter = gridActive ? kf
    : {disease:[], timepoint:[], celltype:[], confidence:""};
  const shortContrast = c => c.replace(/_(\d+)mo$/, "·$1").replace(/^ApTt/, "AT");

  const parts = [];
  const drvSet = _highlightKinaseIds;
  const sigDenom = scopedCtxIds.size > 0 ? scopedCtxIds.size : CONTRASTS.length;
  for (const r of visible) {
    const selCls = r.id === selKid ? " selected" : "";
    // sub-thresh: 0 sig contrasts in the scoped set.
    const scopedSig = _kineSigCountScoped(r, fdr, scopedCtxIds);
    const peakAbsNes = _kineMaxAbsNesScoped(r, scopedCtxIds);
    const subCls = scopedSig === 0 ? " sub-thresh" : "";
    const drvCls = (drvSet && drvSet.has(r.id)) ? " driver" : "";

    // Conf pill: highest tier present in scope, with contributing contrasts as chips.
    const scopedRows = getScopedAttribution(r.id, colFilter);
    const ctxByTier = {very_high: new Set(), high: new Set(), moderate: new Set()};
    for (const e of scopedRows) {
      if (ctxByTier[e.combined_tier]) ctxByTier[e.combined_tier].add(e.contrast_id);
    }
    const tierSpec = [
      {tier:"very_high", cls:"vhi", label:"VERY HIGH", suffix:" (attribution + decomp agreement)"},
      {tier:"high",      cls:"hi",  label:"HIGH",      suffix:""},
      {tier:"moderate",  cls:"mid", label:"MOD",       suffix:""},
    ];
    let confBadge;
    const hit = tierSpec.find(s => ctxByTier[s.tier].size > 0);
    if (hit) {
      const ctxs = Array.from(ctxByTier[hit.tier]).map(ci => CONTRASTS[ci]);
      const shown = ctxs.slice(0, 3).map(c => `<span class="ctx-chip ${hit.cls}">${shortContrast(c)}</span>`).join("");
      const overflow = ctxs.length > 3 ? `<span class="ctx-overflow">+${ctxs.length - 3}</span>` : "";
      const tip = `${hit.label} in ${ctxs.length} contrast${ctxs.length===1?"":"s"}${hit.suffix}: ${ctxs.join(", ")}`;
      confBadge = `<span class="badge ${hit.cls}" title="${_escapeHtml(tip)}">${hit.label}</span>${shown}${overflow}`;
    } else {
      const tipScope = gridActive ? "in active filter scope" : "across all 9 contrasts";
      confBadge = `<span class="badge lo" title="No HIGH or MODERATE attribution ${tipScope}.">low</span>`;
    }

    const residueBadge = r.residue_type === "Y"
      ? ' <span class="track-badge track-y" title="Tyrosine kinase (pY track)">pY</span>'
      : "";
    const profile = _renderNesProfile(r, fdr, maxAbsNes);
    const agreementProfile = _renderAgreementProfile(r);
    parts.push(
      `<tr class="ke-row${selCls}${subCls}${drvCls}" data-kid="${r.id}" ` +
      `tabindex="0" aria-label="Kinase ${r.name}; ${scopedSig} sig contrasts in scope">` +
      `<td>${r.name}${residueBadge}</td>` +
      `<td>${r.gene_symbol}</td>` +
      `<td>${profile}</td>` +
      `<td>${agreementProfile}</td>` +
      `<td class="attr-num">${peakAbsNes != null ? peakAbsNes.toFixed(2) : '<span class="muted">—</span>'}</td>` +
      `<td class="attr-num">${scopedSig}<span class="muted" style="font-size:10px;"> / ${sigDenom}</span></td>` +
      `<td>${_renderCellTypesCell(r, colFilter)}</td>` +
      `<td>${_wmbTierBadge(_kineMaxWmbTierScoped(r.id, colFilter))}</td>` +
      `<td>${confBadge}</td>` +
      `</tr>`
    );
  }
  tbody.innerHTML = parts.join("");
  const countEl = document.getElementById("ke-count");
  if (countEl) countEl.textContent = `${visible.length} / ${_keRows.length} kinases`;
}

function _updateRowSelection(tableSel, rowCls, dataAttr, value) {
  const tbody = document.querySelector(`${tableSel} tbody`);
  if (!tbody) return;
  const prev = tbody.querySelector(`tr.${rowCls}.selected`);
  if (prev) prev.classList.remove("selected");
  if (value == null) return;
  const row = tbody.querySelector(`tr.${rowCls}[${dataAttr}="${value}"]`);
  if (row) row.classList.add("selected");
}

function _updateKinaseRowSelection(kid) {
  _updateRowSelection("#ke-table", "ke-row", "data-kid", kid);
}

function _diseaseColorFor(contrast) {
  for (const d of ["App","Tau","ApTt"])
    if (contrast.indexOf(d) === 0) return DISEASE_COLORS[d];
  return "#90a4ae";
}

let _kinaseAuditSeq = 0;
let _sourceCatalogKey = "mea_stoichiometry";

function _normMotif(s) {
  return String(s || "").replace(/^_+|_+$/g, "").toUpperCase();
}

function _selectedAuditContrast(K, ki) {
  // Audit panel's Contrast picker drives this. Falls back to peak_NES when
  // picker = "ALL". Independent of the left-list KinaseFilter.
  const f = Store.state && Store.state.filters && Store.state.filters.contrast;
  if (f && f !== "ALL" && CONTRASTS.indexOf(f) >= 0) return f;
  return K.peak_contrast[ki] || CONTRASTS[0];
}

function _siteOlsColumns(contrast) {
  return ["site_id", "gene_symbol", "motif", "stoich_lfc_" + contrast,
          "stoich_fdr_" + contrast, "raw_lfc_" + contrast, "raw_fdr_" + contrast,
          "n_obs_stoich", "matched_protein"];
}

async function _renderAuditTable(hostId, tableKey, rows, columns, sourceKey) {
  const t = new AuditTable(hostId, {tableKey, rows, columns, fullSourceKey: sourceKey === false ? false : (sourceKey || tableKey)});
  t.render();
  return t;
}

const KINASE_AUDIT_TABS = [
  {id:"measurement-trace", label:"Measurement Trace"},
  {id:"site-stats", label:"OLS Details"},
  {id:"mea-input", label:"MEA Preparation"},
  {id:"mea-score", label:"MEA Score"},
  {id:"attribution", label:"Attribution"},
];

function _activeKinaseAuditTab() {
  const id = Store.state.view.kinaseAuditTab || KINASE_AUDIT_TABS[0].id;
  return KINASE_AUDIT_TABS.some(t => t.id === id) ? id : KINASE_AUDIT_TABS[0].id;
}

function _selectedAuditSample() {
  return document.getElementById("audit-sample-select")?.value || "plex2_130c_sn_mean";
}

function _selectedAuditSite() {
  return document.getElementById("audit-site-select")?.value || "2488";
}

function _existingCols(rows, cols) {
  return cols.filter(c => rows.some(r => Object.prototype.hasOwnProperty.call(r, c)));
}

function _selectedSiteRows(rows, siteIds, limit) {
  const sid = _selectedAuditSite();
  let out = rows.filter(r => String(r.site_id) === String(sid));
  if (!out.length && siteIds && siteIds.size) out = rows.filter(r => siteIds.has(String(r.site_id)));
  return out.slice(0, limit || 200);
}

function _substrateSiteRows(rows, siteIds, limit) {
  const sid = _selectedAuditSite();
  let out = (siteIds && siteIds.size) ? rows.filter(r => siteIds.has(String(r.site_id))) : [];
  if (!out.length) out = rows.filter(r => String(r.site_id) === String(sid));
  out.sort((a, b) => {
    const as = String(a.site_id) === String(sid) ? 1 : 0;
    const bs = String(b.site_id) === String(sid) ? 1 : 0;
    return bs - as;
  });
  return out.slice(0, limit || 500);
}

function _leadingSubstrateRows(leadRow) {
  return String(leadRow["Leading substrates"] || "")
    .split(";").map(_normMotif).filter(Boolean)
    .map((motif, i) => ({rank:i + 1, substrate_motif:motif}));
}

function _shiftFor(globalRows, contrast) {
  const r = (globalRows || []).find(x => x.contrast === contrast);
  if (!r) return null;
  const v = Number(r.median_shift);
  return Number.isFinite(v) ? v : null;
}

function _winsorBoundsFor(winsorRows, contrast) {
  const r = (winsorRows || []).find(x => x.contrast === contrast);
  if (!r) return null;
  const lo = Number(r.lower_bound), hi = Number(r.upper_bound);
  if (!Number.isFinite(lo) || !Number.isFinite(hi)) return null;
  return [lo, hi];
}

function _winsorContrastSummary(winsorRows, contrast) {
  const filt = (winsorRows || []).filter(x => x.contrast === contrast);
  return {n: filt.length, rows: filt};
}

const _preRankCache = new Map();
function _ensurePreRank(contrast, olsRows, stoichMatrix, shift, bounds) {
  const key = contrast;
  if (_preRankCache.has(key)) return _preRankCache.get(key);
  if (shift == null || !bounds || !Array.isArray(olsRows) || !Array.isArray(stoichMatrix)) {
    return null;
  }
  const motifBySite = new Map();
  for (const r of stoichMatrix) {
    const m = _normMotif(r.motif);
    if (m) motifBySite.set(String(r.site_id), m);
  }
  const lfcCol = "stoich_lfc_" + contrast;
  const [lo, hi] = bounds;
  const ranked = [];
  for (const r of olsRows) {
    const sid = String(r.site_id);
    const motif = motifBySite.get(sid);
    if (!motif) continue;
    const v = r[lfcCol];
    if (v == null || v === "") continue;
    const raw = Number(v);
    if (!Number.isFinite(raw)) continue;
    const centered = raw - shift;
    const clipped = Math.min(Math.max(centered, lo), hi);
    ranked.push({sid, motif, clipped, gene_symbol: r.gene_symbol || ""});
  }
  ranked.sort((a, b) => b.clipped - a.clipped);
  const rankMap = new Map();
  for (let i = 0; i < ranked.length; i++) rankMap.set(ranked[i].sid, i + 1);
  const out = {rankMap, ranked, total: ranked.length};
  _preRankCache.set(key, out);
  return out;
}

function _computeRunningES(ranked, substrateMotifs) {
  const N = ranked.length;
  if (!N || !substrateMotifs || !substrateMotifs.size) return null;
  const hits = new Array(N);
  let Nh = 0, sumHitWeights = 0;
  for (let i = 0; i < N; i++) {
    const isHit = substrateMotifs.has(_normMotif(ranked[i].motif));
    hits[i] = isHit;
    if (isHit) { Nh += 1; sumHitWeights += Math.abs(ranked[i].clipped); }
  }
  if (Nh === 0 || Nh === N) return null;
  const missStep = 1 / (N - Nh);
  const running = new Array(N);
  const hitIndices = [];
  let es = 0, peakES = 0, peakIdx = 0;
  for (let i = 0; i < N; i++) {
    if (hits[i]) {
      es += sumHitWeights > 0 ? Math.abs(ranked[i].clipped) / sumHitWeights : 0;
      hitIndices.push(i);
    } else {
      es -= missStep;
    }
    running[i] = es;
    if (Math.abs(es) > Math.abs(peakES)) { peakES = es; peakIdx = i; }
  }
  const leadingEdge = peakES >= 0
    ? hitIndices.filter(i => i <= peakIdx)
    : hitIndices.filter(i => i >= peakIdx);
  return {running, hitIndices, peakES, peakIdx, leadingEdge, N, Nh};
}

function _buildMeaComparisonRows(leadRow, rawRow) {
  const num = (v) => {
    if (v == null || v === "") return null;
    const n = Number(v);
    return Number.isFinite(n) ? n : null;
  };
  const fmt = (v, d=3) => v == null ? "—" : v.toFixed(d);
  const fmtSigned = (v, d=3) => v == null ? "—" : (v > 0 ? "+" : "") + v.toFixed(d);
  const stoichVals = {
    ES: num(leadRow && leadRow.ES),
    NES: num(leadRow && leadRow.NES),
    p: num(leadRow && leadRow["p-value"]),
    FDR: num(leadRow && leadRow.FDR),
    subs: leadRow && leadRow["Subs fraction"] || "",
  };
  const rawVals = {
    ES: num(rawRow && rawRow.ES),
    NES: num(rawRow && rawRow.NES),
    p: num(rawRow && rawRow["p-value"]),
    FDR: num(rawRow && rawRow.FDR),
    subs: rawRow && rawRow["Subs fraction"] || "",
  };
  const delta = (a, b) => (a == null || b == null) ? null : a - b;
  return [
    {metric:"ES",                  stoich: fmt(stoichVals.ES),  raw: fmt(rawVals.ES),  delta: fmtSigned(delta(stoichVals.ES,  rawVals.ES))},
    {metric:"NES",                 stoich: fmt(stoichVals.NES, 2), raw: fmt(rawVals.NES, 2), delta: fmtSigned(delta(stoichVals.NES, rawVals.NES), 2)},
    {metric:"p-value",             stoich: fmt(stoichVals.p, 4), raw: fmt(rawVals.p, 4), delta: fmtSigned(delta(stoichVals.p,   rawVals.p), 4)},
    {metric:"FDR",                 stoich: fmt(stoichVals.FDR, 3), raw: fmt(rawVals.FDR, 3), delta: fmtSigned(delta(stoichVals.FDR, rawVals.FDR), 3)},
    {metric:"Substrates tested",   stoich: stoichVals.subs || "—", raw: rawVals.subs || "—", delta: "—"},
  ];
}

function _diagnoseRawAbsence(ctx, rawRow) {
  if (rawRow && rawRow.contrast) return null;
  const meaRaw = ctx.meaRaw || [];
  if (!meaRaw.length) {
    return {kind:"file_missing", note:"No raw-phospho MEA loaded for this kinase. Run <code>pixi run python code/kinase_attribution.py --mechanism-annotation</code> to generate <code>mea_raw_phospho.csv</code> (and <code>mea_raw_phospho_pY.csv</code> for tyrosine kinases)."};
  }
  const contrasts = new Set(meaRaw.map(r => r.contrast));
  if (!contrasts.has(ctx.contrast)) {
    return {kind:"contrast_missing", note:`Raw-phospho MEA exists for ${meaRaw.length} other contrast(s) of this kinase but not <strong>${_escapeHtml(ctx.contrast)}</strong>. The raw-phospho file may have been generated under an older contrast set; rerun <code>--mechanism-annotation</code> to refresh.`};
  }
  return {kind:"unknown", note:"Raw-phospho row not found for the selected kinase × contrast."};
}

function _renderMeaScorecard(hostId, leadRow, rawRow, ctx) {
  const host = document.getElementById(hostId);
  if (!host) return;
  const fdrThresh = (Store.state.filters && Store.state.filters.fdr) || 0.25;
  const fmt = (v, d=3) => {
    if (v == null || v === "") return "—";
    const n = Number(v);
    return Number.isFinite(n) ? n.toFixed(d) : String(v);
  };
  const tier = (() => {
    const f = Number(leadRow && leadRow.FDR);
    if (!Number.isFinite(f)) return {label:"no FDR", cls:"muted"};
    if (f < fdrThresh) return {label:`FDR ${f.toFixed(3)} · passes ${fdrThresh}`, cls:"chip-pass"};
    if (f < fdrThresh * 2) return {label:`FDR ${f.toFixed(3)} · borderline`, cls:"chip-borderline"};
    return {label:`FDR ${f.toFixed(3)} · fails ${fdrThresh}`, cls:"chip-fail"};
  })();
  const nesVal = leadRow ? Number(leadRow.NES) : null;
  const nesColor = (nesVal == null || !Number.isFinite(nesVal)) ? "#666"
    : (nesVal > 0 ? "#1f77b4" : "#d62728");
  const subsFrac = leadRow ? leadRow["Subs fraction"] : "";
  const rawNes = rawRow ? rawRow.NES : null;
  const rawFdr = rawRow ? rawRow.FDR : null;
  host.innerHTML = `
    <div class="mea-scorecard">
      <div class="mea-score-nes" style="color:${nesColor}">
        <div class="mea-score-label">NES</div>
        <div class="mea-score-value">${nesVal == null || !Number.isFinite(nesVal) ? "—" : nesVal.toFixed(2)}</div>
        <div class="mea-score-chip ${tier.cls}">${_escapeHtml(tier.label)}</div>
      </div>
      <dl class="mea-score-stats">
        <dt>ES</dt><dd>${fmt(leadRow && leadRow.ES)}</dd>
        <dt>p-value</dt><dd>${fmt(leadRow && leadRow["p-value"], 4)}</dd>
        <dt>Substrates tested</dt><dd>${_escapeHtml(subsFrac || "—")}<span class="muted"> (kinase substrates &cap; contrast prerank)</span></dd>
        <dt>Raw phospho NES</dt><dd>${fmt(rawNes)}<span class="muted"> · FDR ${fmt(rawFdr, 3)}</span></dd>
      </dl>
    </div>`;
}

function _renderRunningEnrichmentPlot(hostId, ctx) {
  const host = document.getElementById(hostId);
  if (!host) return;
  const shift = _shiftFor(ctx.globalRows, ctx.contrast);
  const bounds = _winsorBoundsFor(ctx.winsorRows, ctx.contrast);
  const prerank = _ensurePreRank(ctx.contrast, ctx.olsRows, ctx.stoichMatrix, shift, bounds);
  if (!prerank || !prerank.ranked || !prerank.ranked.length) {
    host.innerHTML = `<div class="muted" style="padding:1em">Running enrichment requires the full prerank list (site_level_ols + mea_global_shift + winsorized_sites). Under file:// the audit tables are preview-only — serve the viewer directory over HTTP to view this plot.</div>`;
    return;
  }
  if (!ctx.substrateMotifs || !ctx.substrateMotifs.size) {
    host.innerHTML = `<div class="muted" style="padding:1em">No substrate-set motifs found for ${_escapeHtml(ctx.name)} on ${_escapeHtml(ctx.contrast)} (mea_substrate_sets.csv).</div>`;
    return;
  }
  const r = _computeRunningES(prerank.ranked, ctx.substrateMotifs);
  if (!r) {
    host.innerHTML = `<div class="muted" style="padding:1em">Running enrichment unavailable: kinase has no substrate hits in the contrast prerank.</div>`;
    return;
  }
  const ranks = new Array(r.N);
  for (let i = 0; i < r.N; i++) ranks[i] = i + 1;
  const hitX = r.hitIndices.map(i => i + 1);
  const hitY = r.hitIndices.map(i => r.running[i]);
  const hitText = r.hitIndices.map(i => {
    const e = prerank.ranked[i];
    return `rank ${i + 1}<br>${_escapeHtml(e.gene_symbol || "")} · ${_escapeHtml(e.motif)}<br>clipped LFC ${e.clipped.toFixed(3)}<br>running ES ${r.running[i].toFixed(3)}`;
  });
  const peakX = r.peakIdx + 1;
  const peakY = r.peakES;
  const leShape = r.peakES >= 0
    ? {x0: 1, x1: peakX, y0: 0, y1: 1}
    : {x0: peakX, x1: r.N, y0: 0, y1: 1};
  Plotly.react(hostId, [
    {type:"scatter", mode:"lines", x: ranks, y: r.running,
     line:{color:"#1f77b4", width:1.5}, name:"running ES", hoverinfo:"skip"},
    {type:"scatter", mode:"markers", x: hitX, y: hitY,
     marker:{color:"#1f77b4", size:5, opacity:0.9}, name:"substrate hit",
     text: hitText, hovertemplate:"%{text}<extra></extra>"},
    {type:"scatter", mode:"markers", x:[peakX], y:[peakY],
     marker:{color:"#000", size:9, symbol:"diamond"}, name:"peak ES",
     hovertemplate:`peak ES ${peakES_safe(peakY)} at rank ${peakX}<extra></extra>`},
  ], {
    margin:{l:50, r:10, t:30, b:40}, height:300,
    showlegend:false,
    annotations:[{
      x: peakX, y: peakY, xref:"x", yref:"y",
      text: `peak ES ${peakY.toFixed(3)} at rank ${peakX}<br>leading edge: ${r.leadingEdge.length} of ${r.Nh} hits`,
      showarrow:true, arrowhead:2, ax: 30, ay: peakY >= 0 ? -40 : 40,
      font:{size:11},
    }],
    shapes:[{
      type:"rect", xref:"x", yref:"paper",
      x0: leShape.x0, x1: leShape.x1, y0: 0, y1: 1,
      fillcolor:"#1f77b4", opacity:0.08, line:{width:0},
    }, {
      type:"line", xref:"x", yref:"y",
      x0: 1, x1: r.N, y0: 0, y1: 0,
      line:{color:"#999", width:1, dash:"dot"},
    }],
    xaxis:{title:"prerank rank (1 = most up-shifted)", range:[1, r.N]},
    yaxis:{title:"running ES", zeroline:false},
  }, {displaylogo:false, responsive:true});
}
function peakES_safe(v) { return Number.isFinite(v) ? v.toFixed(3) : "—"; }

function _renderMeaTrajectory(hostId, kinase_id, ctx) {
  const host = document.getElementById(hostId);
  if (!host) return;
  _ensureKinaseIndexes();
  const K = PAYLOAD.kinases;
  const i = _kinaseIdxById.get(kinase_id);
  if (i == null) return;
  const fdrThresh = (Store.state.filters && Store.state.filters.fdr) || 0.25;
  const stoichNes = CONTRASTS.map(c => K["NES_" + c][i]);
  const stoichFdr = CONTRASTS.map(c => K["FDR_" + c][i]);
  const rawByContrast = new Map((ctx.meaRaw || []).map(r => [r.contrast, r]));
  const rawNes = CONTRASTS.map(c => {
    const r = rawByContrast.get(c);
    if (!r) return null;
    const v = Number(r.NES);
    return Number.isFinite(v) ? v : null;
  });
  const _hexToRgba = (hex, alpha) => {
    const m = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex || "");
    if (!m) return hex;
    return `rgba(${parseInt(m[1],16)},${parseInt(m[2],16)},${parseInt(m[3],16)},${alpha})`;
  };
  const colors = CONTRASTS.map((c, ci) => {
    const base = _diseaseColorFor(c);
    const sig = stoichFdr[ci] != null && stoichFdr[ci] < fdrThresh;
    return sig ? base : _hexToRgba(base, 0.28);
  });
  const selectedIdx = CONTRASTS.indexOf(ctx.contrast);
  const outlines = CONTRASTS.map((_, i) => i === selectedIdx ? "#000" : "rgba(0,0,0,0)");
  const barLineWidth = CONTRASTS.map((_, i) => i === selectedIdx ? 2.5 : 0);
  Plotly.react(hostId, [
    {type:"bar", x: CONTRASTS, y: stoichNes,
     marker:{color: colors, line:{color: outlines, width: barLineWidth}},
     name:"stoichiometry NES",
     hovertemplate:"%{x}<br>stoich NES %{y:.2f}<extra></extra>"},
    {type:"scatter", mode:"markers", x: CONTRASTS, y: rawNes,
     marker:{color:"#000", size:9, symbol:"diamond-open", line:{width:1.5, color:"#000"}},
     name:"raw phospho NES",
     hovertemplate:"%{x}<br>raw NES %{y:.2f}<extra></extra>"},
  ], {
    margin:{l:40, r:10, t:10, b:60}, height:220,
    yaxis:{zeroline:true, zerolinecolor:"#bbb", title:"NES"},
    xaxis:{tickangle:-35},
    showlegend:false,
  }, {displaylogo:false, responsive:true}).then(() => {
    if (host.on && !host.__meaTrajWired) {
      host.__meaTrajWired = true;
      host.on("plotly_click", (ev) => {
        const pts = ev && ev.points ? ev.points : null;
        if (!pts || !pts[0]) return;
        const target = pts[0].x;
        const select = document.getElementById("audit-contrast-select");
        if (select && Array.from(select.options).some(o => o.value === target)) {
          select.value = target;
          select.dispatchEvent(new Event("change"));
        }
      });
    }
  });
}

function _renderDecompPanel(hostId, kinase_id, ctx, leadRow) {
  const host = document.getElementById(hostId);
  if (!host) return;
  _ensureKinaseIndexes();
  const cid = CONTRASTS.indexOf(ctx.contrast);
  if (cid < 0) {
    host.innerHTML = `<div class="muted">No decomposition data for this contrast.</div>`;
    return;
  }
  const rows = (_decompByKinCtx && _decompByKinCtx.get(`${kinase_id}|${cid}`)) || [];
  if (!rows.length) {
    host.innerHTML = `<div class="muted">No decomposition rows for this kinase &times; contrast.</div>`;
    return;
  }
  const sorted = rows.slice().sort((a, b) => (a.nes ?? 0) - (b.nes ?? 0));
  const fdrThresh = (Store.state.filters && Store.state.filters.fdr) || 0.25;
  const bulkNes = leadRow && Number.isFinite(Number(leadRow.NES)) ? Number(leadRow.NES) : null;
  const bulkFdr = leadRow && Number.isFinite(Number(leadRow.FDR)) ? Number(leadRow.FDR) : null;
  const bulkSig = bulkFdr != null && bulkFdr < fdrThresh;
  const cellTypes = sorted.map(r => r.cell_type);
  const nes = sorted.map(r => (r.nes == null || !isFinite(r.nes)) ? 0 : r.nes);
  const fdrs = sorted.map(r => r.fdr);
  const sigMask = fdrs.map(v => v != null && isFinite(v) && v < fdrThresh);
  const colors = nes.map((v, i) => {
    const base = v >= 0 ? "#c8261c" : "#1f5fa6";
    if (sigMask[i]) return base;
    return v >= 0 ? "rgba(200,38,28,0.22)" : "rgba(31,95,166,0.22)";
  });
  const outlines = sigMask.map(s => s ? "#000" : "rgba(0,0,0,0)");
  const lineWidths = sigMask.map(s => s ? 1.2 : 0);
  const hovers = sorted.map((r, i) =>
    `${r.cell_type}<br>decomp NES ${nes[i].toFixed(2)}` +
    (r.fdr != null && isFinite(r.fdr) ? `<br>FDR ${Number(r.fdr).toExponential(2)}${sigMask[i] ? " (sig)" : ""}` : "")
  );
  const traces = [{
    type: "bar", orientation: "h",
    x: nes, y: cellTypes,
    marker: {color: colors, line: {color: outlines, width: lineWidths}},
    hovertemplate: "%{customdata}<extra></extra>",
    customdata: hovers,
    name: "decomp NES",
  }];
  const shapes = [];
  const annotations = [];
  if (bulkNes != null) {
    shapes.push({
      type: "line", xref: "x", yref: "paper",
      x0: bulkNes, x1: bulkNes, y0: 0, y1: 1,
      line: {color: bulkSig ? "#000" : "#888", width: 2, dash: bulkSig ? "solid" : "dash"},
    });
    annotations.push({
      xref: "x", yref: "paper", x: bulkNes, y: 1.04,
      text: `bulk NES ${bulkNes.toFixed(2)}${bulkSig ? "" : " (ns)"}`,
      showarrow: false, font: {size: 11, color: "#000"},
      xanchor: bulkNes >= 0 ? "left" : "right",
    });
  }
  const height = Math.max(220, 22 * cellTypes.length + 60);
  Plotly.react(hostId, traces, {
    margin: {l: 180, r: 30, t: 30, b: 40},
    height,
    xaxis: {title: "NES", zeroline: true, zerolinecolor: "#bbb"},
    yaxis: {automargin: true, tickfont: {size: 11}},
    shapes, annotations,
    showlegend: false,
  }, {displaylogo: false, responsive: true});
}

function _buildPreparedMeaInput(ctx) {
  const shift = _shiftFor(ctx.globalRows, ctx.contrast);
  const bounds = _winsorBoundsFor(ctx.winsorRows, ctx.contrast);
  const winsorSitesForContrast = new Set(
    (ctx.winsorRows || []).filter(r => r.contrast === ctx.contrast)
      .map(r => String(r.site_id)));
  const leadingEdgeSiteIds = ctx.siteIds || new Set();
  const lfcCol = "stoich_lfc_" + ctx.contrast;
  const prerank = _ensurePreRank(ctx.contrast, ctx.olsRows, ctx.stoichMatrix, shift, bounds);
  const sourceRows = (ctx.substrateSiteRows && ctx.substrateSiteRows.length)
    ? ctx.substrateSiteRows : (ctx.siteRows || []);
  const sourceMode = (ctx.substrateSiteRows && ctx.substrateSiteRows.length) ? "substrate" : "leading_edge_fallback";
  const rows = [];
  for (const sr of sourceRows) {
    const sid = String(sr.site_id);
    const v = sr[lfcCol];
    const raw = (v == null || v === "") ? null : Number(v);
    const rawNum = Number.isFinite(raw) ? raw : null;
    const centered = (rawNum == null || shift == null) ? null : rawNum - shift;
    let clipped = centered;
    if (centered != null && bounds) clipped = Math.min(Math.max(centered, bounds[0]), bounds[1]);
    const wasWin = winsorSitesForContrast.has(sid) ||
      (centered != null && clipped != null && Math.abs(centered - clipped) > 1e-12);
    rows.push({
      site_id: sr.site_id,
      gene_symbol: sr.gene_symbol || "",
      motif: sr.motif || "",
      n_obs_stoich: sr.n_obs_stoich,
      raw_lfc: rawNum,
      centered_lfc: centered,
      clipped_lfc: clipped,
      was_winsorized: wasWin ? "yes" : "no",
      rank_in_contrast: prerank ? (prerank.rankMap.get(sid) ?? null) : null,
      in_leading_edge: leadingEdgeSiteIds.has(sid) ? "yes" : "no",
    });
  }
  rows.sort((a, b) => {
    const ar = a.rank_in_contrast == null ? Infinity : a.rank_in_contrast;
    const br = b.rank_in_contrast == null ? Infinity : b.rank_in_contrast;
    return ar - br;
  });
  return {rows, shift, bounds, prerank, sourceMode};
}

function _renderKinaseNesPlot(hostId, kinase_id) {
  const host = document.getElementById(hostId);
  if (!host) return;
  _ensureKinaseIndexes();
  const K = PAYLOAD.kinases;
  const i = _kinaseIdxById.get(kinase_id);
  if (i == null) return;
  const fdr = Store.state.filters.fdr;
  const nes = CONTRASTS.map(c => K["NES_" + c][i]);
  const fdrs = CONTRASTS.map(c => K["FDR_" + c][i]);
  const colors = CONTRASTS.map(_diseaseColorFor);
  const outlines = fdrs.map(v => (v != null && v < fdr) ? "#000" : "rgba(0,0,0,0)");
  Plotly.react(hostId, [{
    type: "bar", x: CONTRASTS, y: nes,
    marker: { color: colors, line: { color: outlines, width: 1.5 } },
    hovertemplate: "%{x}<br>NES %{y:.2f}<extra></extra>",
  }], {
    margin:{l:40,r:10,t:6,b:60}, height:180,
    yaxis:{zeroline:true, zerolinecolor:"#bbb"},
    xaxis:{tickangle:-35},
  }, {displaylogo:false, responsive:true});
}

function _renderKinaseCelltypeEvidence(hostId, kinase_id) {
  const host = document.getElementById(hostId);
  if (!host) return;
  const EV = PAYLOAD.kinase_celltype_evidence || {kinase_id:[]};
  const evIdx = _evidenceByKinase.get(kinase_id) || [];
  const rows = evIdx.map(k => ({
    cell_type: EV.cell_type[k],
    wmb_fold: EV.wmb_fold[k],
    sea_ad_lfc: EV.sea_ad_lfc[k],
    song_lfc: EV.song_lfc[k],
    wmb_tier: EV.wmb_tier[k],
    evidence_basis: EV.evidence_basis ? EV.evidence_basis[k] : "",
    concordance_direction: EV.concordance_direction ? EV.concordance_direction[k] : "",
  }));
  rows.sort((a, b) => {
    const av = a.wmb_fold == null ? -Infinity : a.wmb_fold;
    const bv = b.wmb_fold == null ? -Infinity : b.wmb_fold;
    return bv - av;
  });
  _renderAuditTable(hostId, "celltype_evidence_table", rows,
    ["cell_type","wmb_fold","sea_ad_lfc","song_lfc","wmb_tier","evidence_basis","concordance_direction"],
    "celltype_evidence_table");
}

// ---- Attribution drawer helpers ----------------------------------------
// The Attribution subtab uses three audit sources read directly so the
// reviewer sees the underlying evidence in idiomatic single-cell-biology
// shapes (Seurat dot plot for WMB; supertype LFC heatmap for SEA-AD; OLS
// coefficient table for Song) rather than a synthesized score.

function _attrPathwayFromContrast(contrast) {
  return String(contrast || "").split("_")[0] || "";
}

function _attrLfcColor(lfc) {
  if (lfc == null || !isFinite(lfc) || lfc === 0) return "#f3f4f6";
  const m = Math.min(Math.abs(lfc), 1.0);
  const alpha = 0.08 + 0.32 * m;
  return lfc > 0
    ? `rgba(197, 48, 48, ${alpha.toFixed(3)})`
    : `rgba(43, 108, 176, ${alpha.toFixed(3)})`;
}

function _attrConfidenceClass(conf) {
  if (conf === "very_high") return "attr-conf attr-conf-very-high";
  if (conf === "high") return "attr-conf attr-conf-high";
  if (conf === "moderate") return "attr-conf attr-conf-moderate";
  if (conf === "low") return "attr-conf attr-conf-low";
  return "attr-conf attr-conf-none";
}

function _allenABALink(gene) {
  if (!gene) return "";
  const abc = "https://knowledge.brain-map.org/abcatlas";
  const ctxHpf = `https://celltypes.brain-map.org/rnaseq/mouse_ctx-hpf_10x?selectedVisualization=Scatter+Plot&colorByFeature=Gene+Expression&colorByFeatureValue=${encodeURIComponent(gene)}`;
  return (
    `<a href="${abc}" target="_blank" rel="noopener" class="attr-allen-link" ` +
    `title="ABC Atlas (whole brain) — same Allen WMB 10Xv3 dataset our specificity score is computed on. Search '${_escapeHtml(gene)}' to verify against the same data we used.">` +
    `Verify in ABC Atlas (whole brain) →</a>` +
    ` <a href="${ctxHpf}" target="_blank" rel="noopener" class="attr-allen-link attr-allen-link-secondary" ` +
    `title="Allen Cortex+HPF Transcriptomics Explorer — different dataset (cortex + hippocampal formation only, ~1.1M cells). Useful for high-resolution per-cell intensity in cortical/HPF cell types, but does not contain striatum, olfactory bulb, thalamus, or cerebellum.">` +
    `ctx+HPF (partial tissue)</a>`
  );
}

const ATTR_VERDICT_COLS = [
  {key:"cell_type",                    label:"Cell type",   type:"str", group:"id",
   title:""},
  {key:"cross_rank",                   label:"Conf",        type:"num", group:"attr",
   title:"Combined confidence tier. Starts from the attribution-only tier (high / moderate / low / none). Upgraded to 'very high' when the decomposition layer significantly agrees (Decomp FDR < 0.25 with sign matching bulk MEA). Sort uses cross_rank: tier first, decomposition step as tie-breaker."},
  {key:"wmb_specificity",              label:"WMB enrich",  type:"num", group:"attr",
   title:"WMB enrichment: cell type's share of total log2 expression across 34 WMB classes (uniform = 1/34 ≈ 0.029). Higher = more concentrated in this cell type."},
  {key:"wmb_tier",                     label:"WMB tier",    type:"num", group:"attr",
   title:"WMB specificity expressed as a multiple of uniform (1/34 ≈ 0.029): ≥10× / ≥5× / ≥2× / ≥1×. Empty = below 1× uniform."},
  {key:"wmb_mean_log2_expression",     label:"log2 expr",   type:"num", group:"attr",
   title:"WMB mean log2 expression in this cell type (Allen Whole Mouse Brain 10Xv3, pooled across 13 regions). Absolute level — low values flag the score as potentially noise-driven."},
  {key:"wmb_fraction_cells_expressing",label:"% cells",     type:"num", group:"attr",
   title:"WMB fraction of cells of this cell type with non-zero counts for this gene."},
  {key:"sea_ad_lfc",                   label:"SEA-AD LFC",  type:"num", group:"attr",
   title:"SEA-AD log2 fold change in human AD vs control, median across SEA-AD supertypes mapped to this subclass. Stratum (early / late / full CPS) is selected from the contrast pathway. Color: red = up in AD, blue = down."},
  {key:"song_lfc",                     label:"Song LFC",    type:"num", group:"attr",
   title:"Song log2 fold change from within-cohort snRNA-seq factorial OLS (β at this contrast — 10-param design, time-resolved). Color: red = up in disease genotype, blue = down."},
  {key:"combined_score",               label:"Score",       type:"num", group:"attr",
   title:"Combined attribution score: effective concordance × (0.5 + WMB specificity). The unified attribution uses this for confidence tiers."},
  {key:"decomp_nes",                   label:"Decomp NES",  type:"num", group:"decomp",
   title:"Decomposition NES from the CTM-native proportional decomposition (per-cell-type kinase MEA on bulk phospho ranking weighted by snRNA share for the kinase's substrate set). Same join key as Song LFC. Hypothesis-strength signal — see Methods."},
  {key:"decomp_fdr",                   label:"Decomp FDR",  type:"num", group:"decomp",
   title:"Decomposition MEA FDR for this (kinase, contrast, cell type) row. < 0.25 is the standard MEA gate."},
  {key:"bulk_match",                   label:"vs Bulk",     type:"num", group:"decomp",
   title:"Sign agreement between Decomp NES and the bulk MEA NES for this kinase × contrast. Bold ✓/✗ when Decomp FDR < 0.25; muted when not. Hover any cell for the underlying values."},
];
function _attrVerdictCmp(a, b, key, type, asc) {
  let va, vb;
  if (type === "num") {
    va = a[key]; vb = b[key];
    va = (va == null || !isFinite(va)) ? null : Number(va);
    vb = (vb == null || !isFinite(vb)) ? null : Number(vb);
  } else if (type === "conf") {
    va = _CONF_RANK[a[key]] ?? -1;
    vb = _CONF_RANK[b[key]] ?? -1;
  } else {
    va = (a[key] || "").toString();
    vb = (b[key] || "").toString();
  }
  if (va == null && vb == null) return 0;
  if (va == null) return 1;
  if (vb == null) return -1;
  if (typeof va === "string") return asc ? va.localeCompare(vb) : vb.localeCompare(va);
  return asc ? (va - vb) : (vb - va);
}

function _renderAttributionVerdict(hostId, ctx) {
  const host = document.getElementById(hostId);
  if (!host) return;

  // Verdict reads attribution_index for the audit picker's contrast. Independent
  // of the left-list KinaseFilter — the detail panel's job is to inspect this kinase
  // at the contrast the user picked.
  const verdictFilter = {
    disease:   (ctx.contrast || "").split("_")[0] || "",
    timepoint: ((ctx.contrast || "").match(/_(\d+mo)$/) || ["",""])[1] || "",
    celltype: "", confidence: "",
  };
  const allRows = ctx.kinase_id != null
    ? getScopedAttribution(ctx.kinase_id, verdictFilter)
    : [];

  if (allRows.length === 0) {
    host.innerHTML = `<div class="muted">No attribution rows in ${_escapeHtml(ctx.contrast || "")}.</div>`;
    return;
  }

  // Deduplicate by (contrast_id, cell_type), keeping best-score row.
  const rowKey = r => `${r.contrast_id}|${r.cell_type}`;
  const deduped = new Map();
  for (const r of allRows) {
    const k = rowKey(r);
    const prev = deduped.get(k);
    if (!prev || r.combined_score > prev.combined_score) deduped.set(k, r);
  }
  const rows = Array.from(deduped.values());

  // Attach decomposition NES/FDR for sorting + render. Also compute bulk-NES
  // sign agreement once (same bulk NES/FDR for every row in this kinase × contrast).
  const _K = PAYLOAD.kinases;
  const _bulkNes = (_K && _K["NES_" + ctx.contrast]) ? _K["NES_" + ctx.contrast][ctx.kinase_id] : null;
  const _bulkFdr = (_K && _K["FDR_" + ctx.contrast]) ? _K["FDR_" + ctx.contrast][ctx.kinase_id] : null;
  for (const r of rows) {
    const dk = `${ctx.kinase_id}|${r.contrast_id}|${r.cell_type}`;
    const d = _decompByKey ? _decompByKey.get(dk) : null;
    r.decomp_nes = d ? d.nes : null;
    r.decomp_fdr = d ? d.fdr : null;
    // bulk_match: +2 sig-agree, +1 nonsig-agree, -1 nonsig-disagree, -2 sig-disagree,
    // null when either side is missing. "Sig" here = Decomp FDR < 0.25.
    if (r.decomp_nes == null || !isFinite(r.decomp_nes) || r.decomp_nes === 0
        || _bulkNes == null || !isFinite(_bulkNes) || _bulkNes === 0) {
      r.bulk_match = null;
    } else {
      const agree = (r.decomp_nes > 0) === (_bulkNes > 0);
      const sig = r.decomp_fdr != null && isFinite(r.decomp_fdr) && r.decomp_fdr < 0.25;
      r.bulk_match = agree ? (sig ? 2 : 1) : (sig ? -2 : -1);
    }
    const decompStep = _decompStep(r.decomp_nes, r.decomp_fdr, _bulkNes);
    r.decomp_step = decompStep;
    r.combined_tier = _upgradeTier(r.combined_confidence, decompStep);
    r.wmb_tier = _wmbTier(Number(r.wmb_specificity));
    // cross_rank: combine combined_tier (0..4) and decomp step (-2..3) so
    // reinforcing rows sort first, conflicts demoted, single-layer in between.
    r.cross_rank = (_CONF_RANK[r.combined_tier] || 0) * 6 + decompStep;
  }

  const sortKey = host.dataset.sortKey || "combined_score";
  const sortAsc = host.dataset.sortAsc === "1";
  const sortCol = ATTR_VERDICT_COLS.find(c => c.key === sortKey)
    || ATTR_VERDICT_COLS.find(c => c.key === "combined_score")
    || ATTR_VERDICT_COLS[ATTR_VERDICT_COLS.length - 1];
  rows.sort((a, b) => _attrVerdictCmp(a, b, sortCol.key, sortCol.type, sortAsc));
  const showAllId = `${hostId}-show-all`;
  const showAll = !!(host.dataset.showAll === "1");
  const visibleRows = showAll
    ? rows
    : rows.filter(r => r.combined_tier === "very_high"
                    || r.combined_tier === "high"
                    || r.combined_tier === "moderate");
  const hiddenCount = rows.length - visibleRows.length;
  const num = (v, d=3) => (v == null || !isFinite(v)) ? "" : Number(v).toFixed(d);
  const tbody = visibleRows.map((r, i) => {
    const seaCell = r.sea_ad_lfc == null || !isFinite(r.sea_ad_lfc)
      ? `<td class="attr-num attr-empty">—</td>`
      : `<td class="attr-num attr-num-lfc" style="background:${_attrLfcColor(r.sea_ad_lfc)}">${num(r.sea_ad_lfc, 3)}</td>`;
    const songCell = r.song_lfc == null || !isFinite(r.song_lfc)
      ? `<td class="attr-num attr-empty">—</td>`
      : `<td class="attr-num attr-num-lfc" style="background:${_attrLfcColor(r.song_lfc)}">${num(r.song_lfc, 3)}</td>`;
    const decompNesCell = r.decomp_nes == null || !isFinite(r.decomp_nes)
      ? `<td class="attr-num attr-empty">—</td>`
      : `<td class="attr-num attr-num-lfc" style="background:${_attrLfcColor(r.decomp_nes)}">${num(r.decomp_nes, 2)}</td>`;
    const decompFdrSig = r.decomp_fdr != null && isFinite(r.decomp_fdr) && r.decomp_fdr < 0.25;
    const decompFdrCell = r.decomp_fdr == null || !isFinite(r.decomp_fdr)
      ? `<td class="attr-num attr-empty">—</td>`
      : `<td class="attr-num"${decompFdrSig ? ' style="font-weight:600"' : ''}>${num(r.decomp_fdr, 3)}</td>`;
    let bulkMatchCell;
    if (r.bulk_match == null) {
      bulkMatchCell = `<td class="attr-num attr-empty">—</td>`;
    } else {
      const agree = r.bulk_match > 0;
      const sig = Math.abs(r.bulk_match) === 2;
      const glyph = agree ? "✓" : "✗";
      const color = agree ? "#15803d" : "#b91c1c";
      const style = sig
        ? `color:${color};font-weight:700`
        : `color:#94a3b8;font-weight:500`;
      const tip = `Bulk NES = ${num(_bulkNes, 2)}` +
        (_bulkFdr != null && isFinite(_bulkFdr) ? ` (FDR ${num(_bulkFdr, 3)})` : "") +
        ` · Decomp NES = ${num(r.decomp_nes, 2)}` +
        (r.decomp_fdr != null && isFinite(r.decomp_fdr) ? ` (FDR ${num(r.decomp_fdr, 3)})` : "") +
        (sig ? "" : " · Decomp not significant (FDR ≥ 0.25)");
      bulkMatchCell = `<td class="attr-num" style="${style};text-align:center" title="${_escapeHtml(tip)}">${glyph}</td>`;
    }
    const binFlag = r.wmb_binary_expressed === true || String(r.wmb_binary_expressed).toLowerCase() === "true";
    const expBadge = binFlag
      ? ""
      : `<span class="attr-badge attr-badge-warn" title="Mean log2 expression < 1 OR fewer than 10% of cells detect the gene in this cell type. The enrichment score may be elevated because the gene is barely expressed anywhere.">low expr</span>`;
    const _sbk = (PAYLOAD.subclass_breakdown || {})[String(ctx.kinase_id)] || {};
    const _sbTip = _sbk[r.cell_type] || "";
    const _sbAttr = _sbTip ? ` title="WMB subclass breakdown: ${_escapeHtml(_sbTip)}"` : "";
    const scoreCell = `<td class="attr-num">${num(r.combined_score, 3)}</td>`;
    return `<tr data-cell-type="${_escapeHtml(r.cell_type)}" class="attr-verdict-row${i === 0 ? ' attr-verdict-selected' : ''}">` +
      `<td class="attr-celltype"${_sbAttr}>${_escapeHtml(r.cell_type)}${_sbTip ? ' <span class="attr-subclass-marker" aria-hidden="true">ⓘ</span>' : ''} ${expBadge}</td>` +
      `<td><span class="${_attrConfidenceClass(r.combined_tier)}" title="${_escapeHtml('Attribution: ' + (r.combined_confidence || 'none') + (r.combined_tier === 'very_high' ? ' · upgraded to very_high by significant decomp agreement' : ''))}">${_escapeHtml((r.combined_tier || '').replace('_', ' '))}</span></td>` +
      `<td class="attr-num">${num(r.wmb_specificity, 3)}</td>` +
      `<td class="attr-num">${_wmbTierBadge(_wmbTier(Number(r.wmb_specificity)))}</td>` +
      `<td class="attr-num">${num(r.wmb_mean_log2_expression, 2)}</td>` +
      `<td class="attr-num">${num(r.wmb_fraction_cells_expressing, 2)}</td>` +
      seaCell +
      songCell +
      scoreCell +
      decompNesCell +
      decompFdrCell +
      bulkMatchCell +
      `</tr>`;
  }).join("");
  const headCells = ATTR_VERDICT_COLS.map(c => {
    const arrow = (c.key === sortCol.key) ? (sortAsc ? " ▲" : " ▼") : "";
    const title = c.title ? ` title="${_escapeHtml(c.title)}"` : "";
    return `<th class="attr-verdict-th" data-sort-key="${c.key}"${title}>${c.label}${arrow}</th>`;
  }).join("");
  // Super-header groups the columns into Layer-1 (attribution) and Layer-2 (decomp).
  const _grpCounts = ATTR_VERDICT_COLS.reduce((acc, c) => { acc[c.group] = (acc[c.group]||0)+1; return acc; }, {});
  const superHead =
    `<tr class="attr-verdict-supergroup">` +
      `<th class="attr-supergroup-spacer" colspan="${_grpCounts.id || 0}"></th>` +
      `<th class="attr-supergroup-attr" colspan="${_grpCounts.attr || 0}" title="Cell-type attribution evidence. Each component is compared against the bulk MEA direction at this contrast.">Attribution (vs bulk direction)</th>` +
      `<th class="attr-supergroup-decomp" colspan="${_grpCounts.decomp || 0}" title="Per-cell-type pseudo-deconvolution MEA. A second look at the bulk phospho ranking re-projected by snRNA share.">Decomposition cross-check</th>` +
    `</tr>`;
  // Bulk anchor — both layers compare against this kinase's bulk MEA at this contrast.
  const _bulkSig = _bulkFdr != null && isFinite(_bulkFdr) && _bulkFdr < 0.25;
  const _bulkDir = (_bulkNes != null && isFinite(_bulkNes))
    ? (_bulkNes > 0 ? `<span class="attr-bulk-up">↑ NES = +${num(_bulkNes, 2)}</span>`
                    : `<span class="attr-bulk-down">↓ NES = ${num(_bulkNes, 2)}</span>`)
    : `<span class="attr-bulk-ns">NES n/a</span>`;
  const _bulkFdrTxt = (_bulkFdr != null && isFinite(_bulkFdr))
    ? `FDR = ${num(_bulkFdr, 3)}${_bulkSig ? "" : " (n.s.)"}` : "FDR n/a";
  const bulkAnchor =
    `<div class="attr-bulk-anchor">Bulk MEA anchor for ${_escapeHtml(ctx.contrast || "")}: ` +
    `<span class="attr-bulk-pill">${_bulkDir} · ${_bulkFdrTxt}</span> ` +
    `<span class="muted">— sign of the bulk NES is the reference direction every column below is checked against.</span></div>`;
  host.innerHTML =
    bulkAnchor +
    `<table class="attr-verdict-table">` +
      `<thead>${superHead}<tr>${headCells}</tr></thead><tbody>${tbody}</tbody>` +
    `</table>` +
    (hiddenCount > 0
      ? `<div class="attr-verdict-toggle"><label><input type="checkbox" id="${showAllId}"${showAll ? " checked" : ""}> Show all 34 WMB classes <span class="muted">(${hiddenCount} hidden — low/none confidence)</span></label></div>`
      : (showAll && rows.length > 0
        ? `<div class="attr-verdict-toggle"><label><input type="checkbox" id="${showAllId}" checked> Showing all cell types</label></div>`
        : "")) +
    `<details class="attr-explainer"><summary>How to read <em>Score</em> vs. <em>Confidence</em> in this table</summary>` +
      `<div class="attr-explainer-body">` +
      `<p>Score and tier come from the same three evidence sources but answer different questions:</p>` +
      `<table class="attr-explainer-table" style="margin-bottom:8px;">` +
        `<thead><tr><th>Source</th><th>What it tells you</th></tr></thead><tbody>` +
        `<tr><td><strong>Song</strong></td><td>Does this gene go up or down in our own mice? (within-cohort snRNA-seq from the same animals)</td></tr>` +
        `<tr><td><strong>SEA-AD</strong></td><td>Does this gene go up or down in human Alzheimer's brains? (human postmortem reference)</td></tr>` +
        `<tr><td><strong>WMB</strong></td><td>Is this gene normally on in this cell type, in a healthy mouse? (used as a sanity check, not a direction)</td></tr>` +
        `<tr><td><strong>Decomp NES / FDR</strong></td><td>A per-cell-type version of the bulk phospho signal, reweighted toward each cell type using the snRNA data. Uses the same snRNA data as Song, so treat it as a second look, not independent evidence.</td></tr>` +
        `</tbody></table>` +
      `<p><strong>Confidence tier</strong> grades <em>which sources agree</em>, not how strong any one signal is:</p>` +
      `<ul>` +
        `<li><strong><span class="attr-conf attr-conf-very-high">very high</span></strong> — a <em>high</em> attribution row that is also corroborated by the decomposition layer: Decomp FDR < 0.25 with the same sign as the bulk MEA NES. Both evidence streams reinforce one another.</li>` +
        `<li><strong><span class="badge hi">high</span></strong> — all three of these hold: <em>(a)</em> within-cohort Song supports the direction, <em>(b)</em> the gene is clearly cell-type-specific in WMB (specificity ≥ 2× uniform, i.e. ≈ 0.059 for 34 WMB classes), and <em>(c)</em> at least one reference shows real movement (|Song LFC| or |SEA-AD LFC| > 0.1).</li>` +
        `<li><strong><span class="badge mid">moderate</span></strong> — meaningful evidence but missing one strict gate. Two ways to land here: Song-supported but WMB specificity falls below the high threshold, <em>or</em> only SEA-AD reached concordance (no Song). SEA-AD-only is <strong>always</strong> capped at moderate — we won't promote a cross-species call to high.</li>` +
        `<li><strong><span class="badge lo">low</span></strong> — concordance is positive but the gene isn't expression-specific in WMB and no reference LFC clears the magnitude bar.</li>` +
        `<li><strong>none</strong> — concordance ≤ 0 (signs disagree). Row is excluded from <code>unified_attribution.csv</code> entirely.</li>` +
      `</ul>` +
      `<p><strong>Higher score does not imply higher tier.</strong> A row with strong magnitudes but no within-cohort Song evidence stays at moderate regardless of score; a row with a modest score but Song support + WMB specificity ≥ 0.059 reaches high. Read tier as evidence <em>type</em>, score as evidence <em>weight</em>.</p>` +
      `<p><strong>Combined score</strong> = <code>effective_concordance × (0.5 + wmb_specificity)</code> where <code>effective_concordance = sign(NES) × (3·song_lfc + 1·sea_ad_lfc) / 4</code>. Continuous; used to rank cell types within a kinase (tie-break within tier) and to weight kinase support in the Incytr cell–cell integration.</p>` +
      `</div></details>`;
  host.querySelectorAll("tr.attr-verdict-row").forEach(tr => tr.addEventListener("click", () => {
    host.querySelectorAll("tr.attr-verdict-row").forEach(r => r.classList.remove("attr-verdict-selected"));
    tr.classList.add("attr-verdict-selected");
    _renderAttributionDrawer("attr-drawer", ctx, tr.dataset.cellType);
  }));
  host.querySelectorAll("th.attr-verdict-th").forEach(th => th.addEventListener("click", () => {
    const k = th.dataset.sortKey;
    if (host.dataset.sortKey === k) {
      host.dataset.sortAsc = host.dataset.sortAsc === "1" ? "0" : "1";
    } else {
      host.dataset.sortKey = k;
      // Numeric/conf cols default to descending (largest first); strings ascending.
      const col = ATTR_VERDICT_COLS.find(c => c.key === k);
      host.dataset.sortAsc = (col && col.type === "str") ? "1" : "0";
    }
    _renderAttributionVerdict(hostId, ctx);
  }));
  const toggleEl = document.getElementById(showAllId);
  if (toggleEl) {
    toggleEl.addEventListener("change", () => {
      host.dataset.showAll = toggleEl.checked ? "1" : "0";
      _renderAttributionVerdict(hostId, ctx);
    });
  }
  // Open drawer on the top row by default
  if (rows[0]) _renderAttributionDrawer("attr-drawer", ctx, rows[0].cell_type);
}

function _renderAttributionDrawer(hostId, ctx, cellType) {
  const host = document.getElementById(hostId);
  if (!host) return;
  const gene = ctx.gene || "";
  host.innerHTML =
    `<div class="attr-drawer-header"><strong>${_escapeHtml(cellType)}</strong>` +
    ` &middot; <span class="muted">${_escapeHtml(gene)} / ${_escapeHtml(ctx.contrast)}</span>` +
    ` &middot; ${_allenABALink(gene)}</div>` +
    `<div class="attr-drawer-grid">` +
      `<section class="attr-section"><h5>WMB expression across 34 WMB classes <span class="muted">(wmb_kinase_expression.csv)</span></h5>` +
        `<p class="muted attr-caption">Seurat-style dot plot for ${_escapeHtml(gene)} in the Allen Whole Mouse Brain reference. Color = mean log2 expression, dot size = fraction of cells expressing. Target cell type is outlined.</p>` +
        `<div id="attr-wmb-dotplot"></div></section>` +
      `<section class="attr-section"><h5>SEA-AD supertype log2 fold change <span class="muted">(sea_ad_supertype_lfc.csv)</span></h5>` +
        `<p class="muted attr-caption">Per-supertype LFC for ${_escapeHtml(gene)} in human AD donors, grouped by subclass. Stratum (early / late / full CPS) follows the contrast pathway. Subclass median is used in the verdict table.</p>` +
        `<div id="attr-seaad-heatmap"></div></section>` +
      `<section class="attr-section"><h5>Song within-cohort OLS <span class="muted">(song_concordance.csv)</span></h5>` +
        `<p class="muted attr-caption">Factorial OLS coefficient on the per-animal pseudobulk for this cell type and pathway. Pathway is derived from the contrast prefix (App / Tau / ApTt).</p>` +
        `<div id="attr-song-table"></div></section>` +
    `</div>` +
    `<section class="attr-section attr-section-wide"><h5>Per-cell substrate-site OLS <span class="muted">(deconvolution/per_animal/site_level_ols.parquet)</span></h5>` +
      `<p class="muted attr-caption">Per-(site, contrast, cell type) β / SE / p from the CTM-native pseudo-deconvolution OLS, restricted to ${_escapeHtml(ctx.name || "")}'s substrate set in ${_escapeHtml(cellType)}. Shows what is driving the Decomp NES in the row above. Bulk β is the same site's stoichiometry β before share-reweighting; |Δβ| measures how much the per-cell estimate diverges from bulk.</p>` +
      `<div id="attr-decomp-ols-table" class="audit-scroll"></div></section>`;
  _renderWMBDotPlot("attr-wmb-dotplot", ctx, cellType);
  _renderSEAADHeatmap("attr-seaad-heatmap", ctx, cellType);
  _renderSongOLSPanel("attr-song-table", ctx, cellType);
  _renderDecompOlsTable("attr-decomp-ols-table", ctx, cellType);
}

function _renderDecompOlsTable(hostId, ctx, cellType) {
  const host = document.getElementById(hostId);
  if (!host) return;
  const cId = CONTRASTS.indexOf(ctx.contrast);
  if (ctx.kinase_id == null || cId < 0) {
    host.innerHTML = `<div class="muted">No contrast resolved.</div>`;
    return;
  }
  if (!SliceCache || typeof SliceCache.loadDecompOls !== "function") {
    host.innerHTML = `<div class="muted">Decomp OLS shards unavailable in this build.</div>`;
    return;
  }
  host.innerHTML = `<div class="muted">Loading per-cell OLS shard…</div>`;
  const reqGene = ctx.gene;
  const reqContrast = ctx.contrast;
  const reqCell = cellType;
  SliceCache.loadDecompOls(ctx.kinase_id).then(rows => {
    // Bail if the user moved on while we were fetching.
    if (ctx.gene !== reqGene || ctx.contrast !== reqContrast) return;
    const stillThis = document.getElementById(hostId);
    if (!stillThis || stillThis !== host) return;
    if (!Array.isArray(rows) || rows.length === 0) {
      host.innerHTML = `<div class="muted">No per-cell OLS shard for this kinase.</div>`;
      return;
    }
    const sub = rows.filter(r => Number(r.contrast_id) === cId
                              && String(r.wmb_class) === String(reqCell));
    if (!sub.length) {
      host.innerHTML = `<div class="muted">No substrate sites for ${_escapeHtml(reqCell)} in ${_escapeHtml(reqContrast)}.</div>`;
      return;
    }
    const lfcCol = "stoich_lfc_" + reqContrast;
    const pCol = "stoich_pval_" + reqContrast;
    const bulkBySite = new Map();
    for (const r of (ctx.olsRows || [])) {
      bulkBySite.set(String(r.site_id), {bulk_lfc: r[lfcCol], bulk_pval: r[pCol]});
    }
    sub.sort((a, b) => (Number(b.lfc) || 0) - (Number(a.lfc) || 0));
    const num = (v, d=3) => (v == null || !isFinite(v)) ? "—" : Number(v).toFixed(d);
    const rowsHtml = sub.map(r => {
      const sid = String(r.site_id);
      const bulk = bulkBySite.get(sid) || {};
      const blfc = bulk.bulk_lfc != null && isFinite(bulk.bulk_lfc) ? Number(bulk.bulk_lfc) : null;
      const dlfc = (blfc != null && isFinite(r.lfc)) ? Math.abs(Number(r.lfc) - blfc) : null;
      const pcSig = isFinite(r.pval) && Number(r.pval) < 0.05;
      const bulkSig = bulk.bulk_pval != null && isFinite(bulk.bulk_pval) && Number(bulk.bulk_pval) < 0.05;
      return `<tr>` +
        `<td>${_escapeHtml(r.gene_symbol || "")}</td>` +
        `<td class="attr-num">${_escapeHtml(sid)}</td>` +
        `<td class="motif-mono">${_escapeHtml(r.motif || "")}</td>` +
        `<td>${_escapeHtml(r.track || "")}</td>` +
        `<td class="attr-num"${pcSig ? ' style="font-weight:600"' : ''}>${num(r.lfc, 3)}</td>` +
        `<td class="attr-num">${num(r.se, 3)}</td>` +
        `<td class="attr-num"${pcSig ? ' style="font-weight:600"' : ''}>${num(r.pval, 3)}</td>` +
        `<td class="attr-num"${bulkSig ? ' style="font-weight:600"' : ''}>${num(blfc, 3)}</td>` +
        `<td class="attr-num">${num(dlfc, 3)}</td>` +
      `</tr>`;
    }).join("");
    host.innerHTML =
      `<div class="muted" style="font-size:11px;margin-bottom:4px;">${sub.length} substrate sites · sorted by per-cell β (largest first)</div>` +
      `<table class="attr-decomp-ols-table"><thead><tr>` +
        `<th>Gene</th><th>Site</th><th>Motif</th><th>Track</th>` +
        `<th title="Per-cell β: substrate-site stoichiometry coefficient from the per-(group, wmb_class) OLS, on the deconvoluted phospho signal. Bold when per-cell p < 0.05.">Per-cell β</th>` +
        `<th>SE</th>` +
        `<th title="Per-cell p-value (uncorrected). Bold at p < 0.05.">p</th>` +
        `<th title="Bulk β: same site's stoichiometry β from the bulk MEA pipeline before share-reweighting. Bold when bulk p < 0.05.">Bulk β</th>` +
        `<th title="|per-cell β − bulk β|. Large values mean the cell-type estimate diverges materially from the bulk estimate at this site.">|Δβ|</th>` +
      `</tr></thead><tbody>${rowsHtml}</tbody></table>`;
  }).catch(err => {
    console.error("decomp OLS shard fetch failed", err);
    host.innerHTML = `<div class="muted">Failed to load per-cell OLS shard: ${_escapeHtml(String(err && err.message || err))}</div>`;
  });
}

function _renderWMBDotPlot(hostId, ctx, targetCellType) {
  const host = document.getElementById(hostId);
  if (!host) return;
  const rows = (ctx.wmbRows || []).slice();
  if (rows.length === 0) {
    host.innerHTML = `<div class="muted">No WMB rows for ${_escapeHtml(ctx.gene || '')}.</div>`;
    return;
  }
  rows.sort((a, b) => (Number(b.mean_log2_expression) || 0) - (Number(a.mean_log2_expression) || 0));
  const maxExpr = Math.max(...rows.map(r => Number(r.mean_log2_expression) || 0), 1);
  const W = 720, H = 18 * rows.length + 60, padL = 160, padT = 30, padR = 40;
  const innerW = W - padL - padR;
  const x0 = padL, x1 = padL + innerW;
  const colorAt = (v) => {
    const t = Math.max(0, Math.min(1, v / maxExpr));
    // grey → deep blue ramp
    const r = Math.round(240 - 180 * t), g = Math.round(240 - 130 * t), b = Math.round(240 - 50 * t);
    return `rgb(${r},${g},${b})`;
  };
  const sizeAt = (frac) => {
    const f = Math.max(0, Math.min(1, Number(frac) || 0));
    return 2 + 9 * Math.sqrt(f);
  };
  const tickValues = [0, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0].filter(v => v <= maxExpr * 1.05);
  const xScale = (v) => x0 + (Math.max(0, Math.min(maxExpr, v)) / maxExpr) * innerW;
  const ticks = tickValues.map(v => `<line x1="${xScale(v)}" x2="${xScale(v)}" y1="${padT - 4}" y2="${padT}" stroke="#9ca3af" stroke-width="1"/>` +
    `<text x="${xScale(v)}" y="${padT - 8}" font-size="10" text-anchor="middle" fill="#6b7280">${v}</text>`).join("");
  const dots = rows.map((r, i) => {
    const expr = Number(r.mean_log2_expression) || 0;
    const frac = Number(r.fraction_cells_expressing) || 0;
    const cx = xScale(expr);
    const cy = padT + 18 * i + 9;
    const isTarget = r.cell_type === targetCellType;
    const stroke = isTarget ? "#111827" : "#cbd5e0";
    const strokeW = isTarget ? 2 : 0.8;
    const labelClass = isTarget ? "attr-dot-label attr-dot-label-target" : "attr-dot-label";
    const title = `${r.cell_type}: log2 expr = ${expr.toFixed(2)}, fraction = ${frac.toFixed(2)}, specificity = ${(Number(r.specificity_score) || 0).toFixed(3)}`;
    return `<g><title>${_escapeHtml(title)}</title>` +
      `<text x="${x0 - 8}" y="${cy + 3.5}" text-anchor="end" font-size="11" class="${labelClass}">${_escapeHtml(r.cell_type)}</text>` +
      `<line x1="${x0}" x2="${x1}" y1="${cy}" y2="${cy}" stroke="#e5e7eb" stroke-dasharray="2,2"/>` +
      `<circle cx="${cx}" cy="${cy}" r="${sizeAt(frac).toFixed(1)}" fill="${colorAt(expr)}" stroke="${stroke}" stroke-width="${strokeW}"/>` +
      `</g>`;
  }).join("");
  const legend = `<g transform="translate(${padL}, ${H - 22})">` +
    `<text x="0" y="0" font-size="10" fill="#6b7280">Color: log2 expression (0 → ${maxExpr.toFixed(1)})  ·  Size: fraction of cells expressing (0 → 1)</text>` +
    `</g>`;
  host.innerHTML = `<svg viewBox="0 0 ${W} ${H}" width="100%" preserveAspectRatio="xMidYMid meet" class="attr-svg">` +
    `<line x1="${x0}" x2="${x1}" y1="${padT}" y2="${padT}" stroke="#9ca3af" stroke-width="1"/>` +
    ticks + dots + legend +
    `</svg>`;
}

function _renderSEAADHeatmap(hostId, ctx, targetCellType) {
  const host = document.getElementById(hostId);
  if (!host) return;
  const stratumByPathway = {App: "early", Tau: "late", ApTt: "full"};
  const pathway = _attrPathwayFromContrast(ctx.contrast);
  const stratum = stratumByPathway[pathway] || "full";
  const rows = (ctx.seaSuperRows || []).filter(r => r.stratum === stratum);
  if (rows.length === 0) {
    host.innerHTML = `<div class="muted">No SEA-AD supertype rows for ${_escapeHtml(ctx.gene || '')} (stratum: ${_escapeHtml(stratum)}).</div>`;
    return;
  }
  // Group by subclass
  const bySubclass = new Map();
  for (const r of rows) {
    const sc = r.subclass || "(unknown)";
    if (!bySubclass.has(sc)) bySubclass.set(sc, []);
    bySubclass.get(sc).push(r);
  }
  const subclasses = Array.from(bySubclass.keys()).sort((a, b) => {
    if (a === targetCellType) return -1;
    if (b === targetCellType) return 1;
    return a.localeCompare(b);
  });
  const allLfcs = rows.map(r => Number(r.supertype_lfc) || 0);
  const maxAbs = Math.max(...allLfcs.map(Math.abs), 0.5);
  const cellW = 22, cellH = 16, padL = 170;
  let maxCols = 0;
  for (const arr of bySubclass.values()) maxCols = Math.max(maxCols, arr.length);
  const W = padL + cellW * maxCols + 30;
  const H = subclasses.length * cellH + 50;
  const lfcColor = (v) => {
    const m = Math.min(Math.abs(v) / maxAbs, 1);
    const alpha = 0.15 + 0.75 * m;
    if (v > 0) return `rgba(197, 48, 48, ${alpha.toFixed(3)})`;
    if (v < 0) return `rgba(43, 108, 176, ${alpha.toFixed(3)})`;
    return "#f3f4f6";
  };
  const cells = subclasses.map((sc, i) => {
    const arr = bySubclass.get(sc).slice().sort((a, b) => (Number(b.supertype_lfc) || 0) - (Number(a.supertype_lfc) || 0));
    const isTarget = sc === targetCellType;
    const labelClass = isTarget ? "attr-hm-label-target" : "";
    const median = arr.map(r => Number(r.supertype_lfc) || 0).sort((a, b) => a - b)[Math.floor(arr.length / 2)] || 0;
    const cellsRow = arr.map((r, j) => {
      const v = Number(r.supertype_lfc) || 0;
      const x = padL + j * cellW;
      const y = i * cellH + 30;
      return `<g><title>${_escapeHtml(r.supertype)}: LFC = ${v.toFixed(3)}</title>` +
        `<rect x="${x}" y="${y}" width="${cellW - 1}" height="${cellH - 1}" fill="${lfcColor(v)}" stroke="#fff"/>` +
        `</g>`;
    }).join("");
    const median_str = `median ${median.toFixed(2)} (n=${arr.length})`;
    return `<g><text x="${padL - 8}" y="${i * cellH + 30 + 11}" text-anchor="end" font-size="11" class="${labelClass}">${_escapeHtml(sc)}</text>` +
      cellsRow +
      `<text x="${padL + maxCols * cellW + 6}" y="${i * cellH + 30 + 11}" font-size="10" fill="#6b7280">${median_str}</text></g>`;
  }).join("");
  const legend = `<g transform="translate(${padL}, ${H - 14})"><text x="0" y="0" font-size="10" fill="#6b7280">stratum: ${_escapeHtml(stratum)} CPS · color: red = up in AD, blue = down · one square per supertype, grouped by subclass</text></g>`;
  host.innerHTML = `<svg viewBox="0 0 ${W} ${H}" width="100%" preserveAspectRatio="xMidYMid meet" class="attr-svg">` +
    cells + legend + `</svg>`;
}

function _renderSongOLSPanel(hostId, ctx, targetCellType) {
  const host = document.getElementById(hostId);
  if (!host) return;
  // Schema migrated from 3-pathway to 9-contrast. Fall back to legacy pathway
  // key if the contrast column isn't present on the loaded rows.
  const _useContrast = (ctx.songCdRows || []).some(r => r.contrast != null);
  const targetKey = _useContrast ? ctx.contrast : _attrPathwayFromContrast(ctx.contrast);
  const keyCol = _useContrast ? "contrast" : "pathway";
  const rows = (ctx.songCdRows || []).filter(r => r.cell_type === targetCellType);
  if (rows.length === 0) {
    host.innerHTML = `<div class="muted">No Song concordance rows for ${_escapeHtml(ctx.gene || '')} × ${_escapeHtml(targetCellType)}.</div>`;
    return;
  }
  const num = (v, d=3) => (v == null || !isFinite(Number(v))) ? "—" : Number(v).toFixed(d);
  const sciNum = (v) => (v == null || !isFinite(Number(v))) ? "—" : Number(v).toExponential(2);
  const tbody = rows.map(r => {
    const isTarget = r[keyCol] === targetKey;
    return `<tr${isTarget ? ' class="attr-song-selected"' : ''}>` +
      `<td>${_escapeHtml(r[keyCol])}${isTarget ? ' <span class="attr-badge attr-badge-info">selected</span>' : ''}</td>` +
      `<td class="attr-num" style="background:${_attrLfcColor(Number(r.song_lfc))}">${num(r.song_lfc, 3)}</td>` +
      `<td class="attr-num">${num(r.song_se, 3)}</td>` +
      `<td class="attr-num">${sciNum(r.song_pval)}</td>` +
      `<td class="attr-num">${num(r.song_fdr, 3)}</td>` +
      `<td class="attr-num">${num(r.n_animals, 0)}</td>` +
      `</tr>`;
  }).join("");
  const headerLabel = _useContrast ? "Contrast" : "Pathway";
  const lfcTitle = _useContrast
    ? "Factorial OLS coefficient at this contrast (10-param design with timepoint interactions). Pseudobulk log2(CPM+1), males only."
    : "Factorial OLS coefficient: App = β_App; Tau = β_Tau; ApTt = β_App + β_Tau + β_Int. Pseudobulk log2(CPM+1), males only, pooled across timepoints.";
  const pvalTitle = _useContrast
    ? "Two-sided p-value for the OLS contrast t-statistic with df_resid = n_animals − 10."
    : "Two-sided p-value for the OLS coefficient with df_resid = n_animals − 4.";
  const fdrTitle = `Benjamini–Hochberg FDR computed within (cell type, ${_useContrast ? "contrast" : "pathway"}).`;
  host.innerHTML =
    `<table class="attr-song-table">` +
      `<thead><tr>` +
        `<th>${headerLabel}</th>` +
        `<th title="${lfcTitle}">β (log2 LFC)</th>` +
        `<th title="Standard error of β.">SE</th>` +
        `<th title="${pvalTitle}">p-value</th>` +
        `<th title="${fdrTitle}">FDR</th>` +
        `<th title="Animals contributing to the OLS fit for this cell type.">n animals</th>` +
      `</tr></thead><tbody>${tbody}</tbody>` +
    `</table>`;
}

function _setAuditSelectors(ctx) {
  const siteSelect = document.getElementById("audit-site-select");
  if (siteSelect) {
    const current = siteSelect.value;
    const siteRows = ctx.siteRows || [];
    siteSelect.innerHTML = siteRows.slice(0, 300).map(r =>
      `<option value="${_escapeHtml(r.site_id)}">${_escapeHtml(r.site_id)} · ${_escapeHtml(r.gene_symbol || "")}</option>`
    ).join("");
    if (current && siteRows.some(r => String(r.site_id) === current)) siteSelect.value = current;
    else if (siteRows.some(r => String(r.site_id) === "2488")) siteSelect.value = "2488";
    else if (siteRows[0]) siteSelect.value = siteRows[0].site_id;
    siteSelect.onchange = () => renderActiveKinaseAuditTab(Store.state.selection.kinase);
  }
  const sampleSelect = document.getElementById("audit-sample-select");
  if (sampleSelect) {
    const current = sampleSelect.value || "plex2_130c_sn_mean";
    const cols = Object.keys((ctx.rawMatrix || [])[0] || {}).filter(c => c.endsWith("_sn_mean"));
    sampleSelect.innerHTML = cols.map(c => `<option value="${_escapeHtml(c)}">${_escapeHtml(c)}</option>`).join("");
    sampleSelect.value = cols.includes(current) ? current : (cols.includes("plex2_130c_sn_mean") ? "plex2_130c_sn_mean" : cols[0] || "");
    sampleSelect.onchange = () => renderActiveKinaseAuditTab(Store.state.selection.kinase);
  }
}

async function _loadKinaseAuditContext(kinase_id, seq) {
  _ensureKinaseIndexes();
  const K = PAYLOAD.kinases;
  const ki = _kinaseIdxById.get(kinase_id);
  if (ki == null) return null;
  const name = K.name[ki];
  const contrast = _selectedAuditContrast(K, ki);
  // Resolve track-suffixed audit keys: ST kinases load the unsuffixed files
  // (raw_phospho_normalized, mea_stoichiometry, ...), pY kinases load the
  // _pY siblings produced by kinase_attribution._track_output.
  const residueType = (K.residue_type && K.residue_type[ki]) || "ST";
  const tk = (base) => residueType === "Y" ? base + "_pY" : base;

  const [stoichRows, rawRows, olsRows, rawMatrix, stoichMatrix, uaRows,
         normRows, sampleRows, winsorRows, globalRows, subsRows,
         wmbAllRows, songCdAllRows, seaSuperAllRows] = await Promise.all([
    AuditDataStore.load(tk("mea_stoichiometry")),
    AuditDataStore.load(tk("mea_raw_phospho")).catch(() => []),
    AuditDataStore.load(tk("site_level_ols")),
    AuditDataStore.load(tk("raw_phospho_normalized")),
    AuditDataStore.load(tk("stoichiometry_matrix")),
    AuditDataStore.load("unified_attribution_full").catch(() => AuditDataStore.load("unified_attribution")),
    AuditDataStore.load("normalization_summary"),
    AuditDataStore.load("sample_mapping"),
    AuditDataStore.load(tk("winsorized_sites")),
    AuditDataStore.load(tk("mea_global_shift")),
    AuditDataStore.load(tk("mea_substrate_sets")).catch(() => []),
    AuditDataStore.load("wmb_kinase_expression").catch(() => []),
    AuditDataStore.load("song_concordance").catch(() => []),
    AuditDataStore.load("sea_ad_supertype_lfc").catch(() => []),
  ]);
  if (seq !== _kinaseAuditSeq || Store.state.selection.kinase !== kinase_id) return null;

  const meaStoich = stoichRows.filter(r => r.kinase === name);
  const meaRaw = rawRows.filter(r => r.kinase === name || r.kinase === K.gene_symbol[ki]);
  const leadRow = meaStoich.find(r => r.contrast === contrast) || meaStoich[0] || {};
  const motifs = new Set(String(leadRow["Leading substrates"] || "")
    .split(";").map(_normMotif).filter(Boolean));
  const motifBySite = new Map();
  for (const r of rawMatrix) if (motifs.has(_normMotif(r.motif))) motifBySite.set(String(r.site_id), r.motif);
  for (const r of stoichMatrix) if (motifs.has(_normMotif(r.motif))) motifBySite.set(String(r.site_id), r.motif);
  const siteIds = new Set(motifBySite.keys());
  const siteRows = olsRows.filter(r => siteIds.has(String(r.site_id))).map(r => ({
    ...r,
    motif: motifBySite.get(String(r.site_id)) || r.motif || "",
  }));
  const attrRows = uaRows.filter(r => r.kinase === name || r.gene_symbol === K.gene_symbol[ki]);
  const geneUpper = String(K.gene_symbol[ki] || "").toUpperCase();
  const wmbRows = (wmbAllRows || []).filter(r =>
    String(r.gene_symbol || "").toUpperCase() === geneUpper);
  const songCdRows = (songCdAllRows || []).filter(r =>
    String(r.gene_symbol || "").toUpperCase() === geneUpper);
  const seaSuperRows = (seaSuperAllRows || []).filter(r =>
    String(r.gene_symbol || "").toUpperCase() === geneUpper);

  // Substrate-set sites for this kinase + contrast (kinase library's substrate
  // gene set restricted to this contrast's prerank universe). This is what GSEA
  // walks for this kinase, and is upstream of the MEA leading-edge result.
  const substrateMotifs = new Set();
  for (const r of (subsRows || [])) {
    if (r.kinase === name && r.contrast === contrast) substrateMotifs.add(_normMotif(r.motif));
  }
  const substrateMotifBySite = new Map();
  for (const r of stoichMatrix) {
    const m = _normMotif(r.motif);
    if (m && substrateMotifs.has(m)) substrateMotifBySite.set(String(r.site_id), r.motif);
  }
  const substrateSiteIds = new Set(substrateMotifBySite.keys());
  const substrateSiteRows = olsRows.filter(r => substrateSiteIds.has(String(r.site_id))).map(r => ({
    ...r,
    motif: substrateMotifBySite.get(String(r.site_id)) || r.motif || "",
  }));

  return {
    kinase_id, ki, name, gene:K.gene_symbol[ki], contrast,
    residueType,
    meaStoich, meaRaw, leadRow, siteIds, siteRows, attrRows, olsRows,
    wmbRows, songCdRows, seaSuperRows,
    substrateMotifs, substrateSiteIds, substrateSiteRows, subsRows,
    rawMatrix, stoichMatrix, normRows, sampleRows, winsorRows, globalRows,
  };
}

function renderNumberTrace(rawMatrix, stoichMatrix, normRows, sampleRows, hostId) {
  const host = document.getElementById(hostId || "audit-number-trace");
  if (!host) return;
  const sid = _selectedAuditSite();
  const sample = _selectedAuditSample();
  const raw = rawMatrix.find(r => String(r.site_id) === String(sid)) || {};
  const sto = stoichMatrix.find(r => String(r.site_id) === String(sid)) || {};
  const sm = sampleRows.find(r => r.column_name === sample) || {};
  const norm = Object.fromEntries((normRows || []).map(r => [r.key, r.value]));
  const rows = [
    {step:"Selected site", source:"site_id", value:sid},
    {step:"Sample column", source:"sample_mapping.csv", value:sample},
    {step:"Animal / condition", source:"sample_mapping.csv", value:[sm.animal_id, sm.genotype, sm.timepoint].filter(Boolean).join(" / ")},
    {step:"Normalized phospho", source:"raw_phospho_normalized.csv", value:raw[sample]},
    {step:"Stoichiometry", source:"stoichiometry_matrix.csv", value:sto[sample]},
    {step:"Matched protein", source:"stoichiometry_matrix.csv", value:sto.matched_protein},
    {step:"IRS method", source:"normalization_summary.json", value:norm.normalization_method || ""},
    {step:"Raw workbook reference", source:"pipeline provenance", value:"Referenced by generated CSV lineage; raw workbooks are not embedded in v1."},
  ];
  _renderAuditTable(hostId || "audit-number-trace", "number_trace", rows, ["step","source","value"], "stoichiometry_matrix");
}

function renderSourceCatalog(listId, detailId) {
  const list = document.getElementById(listId || "audit-source-list");
  const detail = document.getElementById(detailId || "audit-source-detail");
  if (!list || !detail) return;
  const tables = _auditManifest();
  const keys = Object.keys(tables);
  if (!keys.includes(_sourceCatalogKey)) _sourceCatalogKey = keys[0];
  list.innerHTML = keys.map(k => {
    const t = tables[k];
    return `<button class="${k === _sourceCatalogKey ? "active" : ""}" data-key="${_escapeHtml(k)}">` +
      `<strong>${_escapeHtml(t.label || k)}</strong><br><span class="muted">${(t.row_count || 0).toLocaleString()} rows · ${(t.column_count || 0).toLocaleString()} cols</span></button>`;
  }).join("");
  list.querySelectorAll("button").forEach(btn => btn.addEventListener("click", () => {
    _sourceCatalogKey = btn.dataset.key;
    renderSourceCatalog(listId, detailId);
  }));
  const t = tables[_sourceCatalogKey] || {};
  detail.innerHTML = `<dl class="audit-kv">` +
    `<dt>Raw path</dt><dd>${_escapeHtml(t.source_path || "")}</dd>` +
    `<dt>Viewer path</dt><dd>${_escapeHtml(t.relative_path || "")}</dd>` +
    `<dt>Rows / columns</dt><dd>${(t.row_count || 0).toLocaleString()} / ${(t.column_count || 0).toLocaleString()}</dd>` +
    `<dt>Searchable columns</dt><dd>${_escapeHtml((t.columns || []).map(c => c.raw).join(", "))}</dd>` +
    `</dl><div id="audit-source-preview"></div>`;
  _renderAuditTable("audit-source-preview", _sourceCatalogKey, t.preview || [],
    (t.columns || []).slice(0, 12).map(c => c.raw), _sourceCatalogKey);
}

async function renderActiveKinaseAuditTab(kinase_id) {
  const body = document.getElementById("kinase-audit-body");
  if (!body || kinase_id == null) return;
  const tab = _activeKinaseAuditTab();
  document.querySelectorAll(".kinase-audit-tabs button").forEach(btn =>
    btn.classList.toggle("active", btn.dataset.auditTab === tab));
  const seq = _kinaseAuditSeq;
  body.innerHTML = '<div class="muted">Loading audit data...</div>';
  try {
    const ctx = await _loadKinaseAuditContext(kinase_id, seq);
    if (!ctx || seq !== _kinaseAuditSeq) return;
    _setAuditSelectors(ctx);
    const sample = _selectedAuditSample();
    const siteCols = _siteOlsColumns(ctx.contrast);
    const existingSiteCols = _existingCols(ctx.siteRows, siteCols);
    const wantedMea = ["kinase", "contrast", "ES", "NES", "p-value", "FDR", "Subs fraction", "Leading substrates"];

    if (tab === "measurement-trace") {
      body.innerHTML = `<p class="kinase-stage-note">Raw-to-stoichiometry receipt for the selected kinase and contrast's leading-substrate sites. The Sample control selects one animal/channel column; each row shows raw PTM, raw parent protein, IRS-normalized values, log2 transforms, and the stoichiometry subtraction used downstream.</p><div id="audit-measurement-trace"></div>`;
      const traceRows = await MeasurementTraceStore.load(sample, ctx.residueType);
      if (seq !== _kinaseAuditSeq) return;
      const rows = _substrateSiteRows(traceRows, ctx.siteIds, 500);
      _renderAuditTable("audit-measurement-trace", "measurement_trace", rows,
        ["site_id","gene_symbol","motif","protein_gene","matched_protein","raw_phospho","raw_protein","irs_phospho","irs_protein","log2_irs_phospho","log2_irs_protein","stoichiometry"],
        false);
    } else if (tab === "site-stats") {
      body.innerHTML = `<p class="kinase-stage-note">OLS contrast details for the selected kinase's leading-substrate phosphosites. Each row is one phosphosite, not one sample. The selected contrast controls which stoichiometry and raw-phospho effect columns are shown; n_obs_stoich is the total count of usable stoichiometry sample columns available for that site.</p><div id="audit-site-stats"></div>`;
      _renderAuditTable("audit-site-stats", "site_level_ols", ctx.siteRows, existingSiteCols, "site_level_ols");
    } else if (tab === "mea-input") {
      const prep = _buildPreparedMeaInput(ctx);
      const shiftRow = (ctx.globalRows || []).find(r => r.contrast === ctx.contrast) || null;
      const shiftVal = shiftRow ? Number(shiftRow.median_shift) : null;
      const winsorAll = _winsorContrastSummary(ctx.winsorRows, ctx.contrast);
      const totalSites = prep.prerank ? prep.prerank.total : null;
      const rankNote = AuditDataStore.fileMode
        ? `<span class="muted"> &middot; rank_in_contrast unavailable under file:// (serve over HTTP for the full prerank).</span>`
        : (prep.prerank ? ` &middot; ranked across ${totalSites.toLocaleString()} sites` : `<span class="muted"> &middot; rank could not be computed (missing shift or winsor bounds)</span>`);
      const formula = `<code>OLS &beta;<sub>stoich</sub> &minus; median shift &rarr; centered &rarr; winsorize [lo, hi] &rarr; clipped &rarr; GSEA prerank</code>`;
      const winsorClippedInSet = prep.rows.filter(r => r.was_winsorized === "yes").length;
      const winsorHeadline = (prep.bounds
          ? `bounds [${prep.bounds[0].toFixed(3)}, ${prep.bounds[1].toFixed(3)}] &middot; ${winsorAll.n.toLocaleString()} sites clipped across the contrast &middot; ${winsorClippedInSet} of this kinase's substrate sites clipped`
          : `<span class="muted">No winsorization receipts for this contrast.</span>`);
      const subsCount = (ctx.substrateMotifs && ctx.substrateMotifs.size) || 0;
      const leSubset = prep.rows.filter(r => r.in_leading_edge === "yes").length;
      const fallbackNote = (prep.sourceMode === "leading_edge_fallback")
        ? `<div class="muted" style="margin-top:.4em">mea_substrate_sets.csv unavailable; falling back to the leading-edge subset. Run <code>pixi run enrich</code> to materialize substrate-set receipts.</div>`
        : "";
      body.innerHTML =
        `<section class="audit-panel"><h4>Step 1 &middot; Global shift <span class="muted">(mea_global_shift.csv)</span></h4>` +
        `<p class="kinase-stage-note">Median stoichiometry LFC across the contrast's ranked sites${shiftVal != null ? `: <strong>${shiftVal.toFixed(4)}</strong>` : ""}. Subtracted from every ranked site before GSEA so the prerank is centered at zero. Contrast-level, not kinase-specific.</p>` +
        `<p class="kinase-stage-note muted"><strong>Why center?</strong> GSEA scores how a kinase's substrate set is concentrated at the top vs. bottom of the ranked list. If the entire contrast has a global up- or down-shift (e.g. a small bulk-level imbalance in normalization), that shift moves <em>every</em> kinase's ranks in one direction and inflates one tail's NES regardless of biology. Subtracting the contrast-level median forces equal numbers of positive and negative ranks so a significant NES reflects substrate-set concentration relative to background, not the global drift.</p>` +
        `<div id="audit-mea-shift"></div></section>` +
        `<section class="audit-panel"><h4>Step 2 &middot; Winsorization <span class="muted">(winsorized_sites.csv)</span></h4>` +
        `<p class="kinase-stage-note">Centered LFCs clipped to the 1st/99th percentile so individual sites cannot dominate the prerank. ${winsorHeadline}</p></section>` +
        `<section class="audit-panel audit-wide"><h4>Step 3 &middot; Prepared MEA input for this kinase <span class="muted">(mea_substrate_sets.csv &times; site_level_ols)</span></h4>` +
        `<p class="kinase-stage-note">One row per site whose motif is in this kinase's substrate set per the kinase library at threshold KL_THRESH. <strong>${subsCount.toLocaleString()}</strong> motifs &rarr; <strong>${prep.rows.length.toLocaleString()}</strong> sites in this contrast's prerank universe; ${leSubset} flagged as leading edge in the MEA result. Each row walks from the OLS &beta; through the median-shift correction and the winsor clip into the prerank position GSEA used to score this kinase. Sort by Rank to walk the ranked list as GSEA did.${fallbackNote}</p>` +
        `<div id="audit-mea-prepared"></div></section>`;
      _renderAuditTable("audit-mea-shift", "mea_global_shift", shiftRow ? [shiftRow] : [],
        ["contrast","median_shift","mean_before","pct_pos_before","pct_pos_after"], "mea_global_shift");
      _renderAuditTable("audit-mea-prepared", "mea_input_derived", prep.rows,
        ["rank_in_contrast","site_id","gene_symbol","motif","n_obs_stoich","raw_lfc","centered_lfc","clipped_lfc","was_winsorized","in_leading_edge"],
        false);
    } else if (tab === "mea-score") {
      const leadRow = ctx.leadRow || {};
      const rawRow = (ctx.meaRaw || []).find(r => r.contrast === ctx.contrast) || {};
      const compactMea = ["kinase","contrast","ES","NES","p-value","FDR","Subs fraction"];
      const fileNote = AuditDataStore.fileMode
        ? `<div class="muted" style="margin-top:.4em">Running enrichment requires the full prerank — serve over HTTP to render it.</div>`
        : "";
      body.innerHTML = `<p class="kinase-stage-note">The score for ${_escapeHtml(ctx.name)} on ${_escapeHtml(ctx.contrast)}: how the kinase's substrate set (Step 3) concentrates in the contrast prerank. Stoichiometry track is the primary signal; raw phospho is shown alongside for cross-track sanity.${fileNote}</p>` +
        `<section class="audit-panel"><h4>Score for ${_escapeHtml(ctx.contrast)}</h4>` +
        `<div id="audit-mea-scorecard"></div></section>` +
        `<section class="audit-panel"><h4>Running enrichment for ${_escapeHtml(ctx.contrast)}</h4>` +
        `<p class="kinase-stage-note">GSEA walk recomputed at view time from the cached prerank. The curve steps up at every substrate hit (weighted by |clipped LFC|) and down at every miss. Peak ES and the leading-edge prefix are marked. Tie-breaking among ~2% of sites with duplicated clipped values may differ from gseapy's internal order.</p>` +
        `<div id="audit-mea-running" style="height:300px"></div></section>` +
        `<section class="audit-panel"><h4>NES across all contrasts</h4>` +
        `<p class="kinase-stage-note">Stoichiometry NES bars: full-saturation when FDR &lt; threshold, faded when not significant. The selected contrast is outlined in black. Raw phospho NES shown as paired open diamonds. Click a bar to switch the selected contrast.</p>` +
        `<div id="audit-mea-trajectory" style="height:220px"></div></section>` +
        `<section class="audit-panel"><h4>Stoichiometry vs raw phospho for ${_escapeHtml(ctx.contrast)} <span class="muted">(mea_stoichiometry.csv vs mea_raw_phospho.csv)</span></h4>` +
        `<p class="kinase-stage-note">Per-metric comparison of the same kinase &times; contrast scored against two preprocessing tracks. Stoichiometry is primary; raw phospho is the sensitivity check. Δ = stoichiometry − raw. Sign-flipping or significance divergence flags abundance-driven vs activity-driven signals.</p>` +
        `<div id="audit-mea-comparison"></div></section>` +
        `<section class="audit-panel audit-wide"><h4>Per-cell-type decomposition for ${_escapeHtml(ctx.contrast)} <span class="muted">(kinase_enrichment_wmb.csv)</span></h4>` +
        `<p class="kinase-stage-note">Pseudo-deconvoluted MEA NES per WMB class for this kinase &times; contrast, sorted by NES. Bars are filled when FDR &lt; threshold, faded otherwise. The vertical line marks the bulk NES from the live pipeline (solid black = bulk significant, dashed gray = ns). Comparing the spread of class bars to the bulk line shows whether the bulk signal localizes to a class, is averaged across many, or is masked by canceling classes.</p>` +
        `<div id="audit-mea-decomp"></div></section>`;
      _renderMeaScorecard("audit-mea-scorecard", leadRow, rawRow, ctx);
      _renderRunningEnrichmentPlot("audit-mea-running", ctx);
      _renderMeaTrajectory("audit-mea-trajectory", kinase_id, ctx);
      try { _renderDecompPanel("audit-mea-decomp", kinase_id, ctx, leadRow); }
      catch (decompErr) {
        console.error("decomp panel failed", decompErr);
        const dh = document.getElementById("audit-mea-decomp");
        if (dh) dh.innerHTML = `<div class="muted">Decomposition panel failed: ${_escapeHtml(String(decompErr && decompErr.message || decompErr))}</div>`;
      }
      const cmpRows = _buildMeaComparisonRows(leadRow, rawRow);
      const diag = _diagnoseRawAbsence(ctx, rawRow);
      const diagBanner = diag
        ? `<div class="kinase-stage-note muted" style="margin-bottom:.6em">⚠ ${diag.note}</div>`
        : "";
      const cmpHost = document.getElementById("audit-mea-comparison");
      if (cmpHost) {
        cmpHost.innerHTML = diagBanner + `<div id="audit-mea-cmp-table"></div>`;
        _renderAuditTable("audit-mea-cmp-table", "mea_track_comparison", cmpRows,
          ["metric","stoich","raw","delta"], false);
      }
    } else if (tab === "attribution") {
      body.innerHTML =
        `<section class="audit-panel audit-wide"><h4>Verdict across cell types <span class="muted">for ${_escapeHtml(ctx.name)} / ${_escapeHtml(ctx.contrast)}</span></h4>` +
        `<div id="attr-verdict"></div></section>` +
        `<section class="audit-panel audit-wide"><h4>Evidence drawer</h4>` +
        `<div id="attr-drawer"></div></section>` +
        `<section class="audit-panel"><h4>Raw attribution rows <span class="muted">(unified_attribution.csv)</span></h4>` +
        `<div id="audit-attribution"></div></section>`;
      _renderAttributionVerdict("attr-verdict", ctx);
      _renderAuditTable("audit-attribution", "unified_attribution", ctx.attrRows,
        ["kinase","gene_symbol","contrast","cell_type","combined_confidence","wmb_specificity","wmb_mean_log2_expression","wmb_fraction_cells_expressing","sea_ad_lfc","song_lfc","combined_score","evidence_basis"],
        "unified_attribution");
    }
  } catch (e) {
    if (seq !== _kinaseAuditSeq) return;
    console.error("audit tab failed", e);
    const msg = e && (e.message || e.toString && e.toString()) || String(e);
    body.innerHTML = `<div class="muted">Audit table load failed: ${_escapeHtml(msg)}</div>`;
  }
}

function renderKinaseDetail(kinase_id) {
  const el = document.getElementById("ke-detail");
  if (!el) return;
  if (kinase_id == null) {
    el.innerHTML = '<div class="muted">Select a kinase to open the audit workbench.</div>';
    return;
  }
  _ensureKinaseIndexes();
  const K = PAYLOAD.kinases;
  const i = _kinaseIdxById.get(kinase_id);
  if (i == null) {
    el.innerHTML = '<div class="muted">Kinase not found.</div>';
    return;
  }
  const name = K.name[i];
  ++_kinaseAuditSeq;
  const tabButtons = KINASE_AUDIT_TABS.map(t =>
    `<button type="button" data-audit-tab="${t.id}" class="${t.id === _activeKinaseAuditTab() ? "active" : ""}">${t.label}</button>`
  ).join("");

  el.innerHTML =
    `<div class="kinase-workbench-header">` +
    `<div class="kinase-workbench-title"><h3>${name}</h3></div>` +
    `<div class="kinase-workbench-controls">` +
    `<label>Contrast <select id="audit-contrast-select"><option value="ALL">Auto peak/global</option>${CONTRASTS.map(c => `<option value="${c}">${c}</option>`).join("")}</select></label>` +
    `<label>Sample <select id="audit-sample-select"></select></label>` +
    `</div></div>` +
    `<div class="kinase-audit-tabs" role="tablist" aria-label="Kinase audit walkthrough">${tabButtons}</div>` +
    `<div class="kinase-audit-tab-body" id="kinase-audit-body"></div>`;

  const contrastSelect = document.getElementById("audit-contrast-select");
  if (contrastSelect) {
    contrastSelect.value = Store.state.filters.contrast || "ALL";
    contrastSelect.onchange = ev => Store.dispatch({type:"SET_FILTER", key:"contrast", value:ev.target.value});
  }
  document.querySelectorAll(".kinase-audit-tabs button").forEach(btn => {
    btn.addEventListener("click", () => {
      Store.dispatch({type:"SET_VIEW", key:"kinaseAuditTab", value:btn.dataset.auditTab});
    });
  });
  renderActiveKinaseAuditTab(kinase_id);
}

async function renderKinaseBackbones(kinase_id) {
  const container = document.getElementById("ke-detail-backbones");
  if (!container) return;
  _ensureKinaseIndexes();
  if (!_presentKinaseSet.has(kinase_id)) {
    container.innerHTML = '<div class="muted">No significant edges for this kinase.</div>';
    container.classList.remove("muted");
    return;
  }
  let rows;
  try {
    rows = await SliceCache.loadKinase(kinase_id);
  } catch (e) {
    if (Store.state.selection.kinase !== kinase_id) return;
    container.innerHTML = `<div class="muted">Failed to load: ${e.message}</div>`;
    return;
  }
  if (Store.state.selection.kinase !== kinase_id) return;

  const f = Store.state.filters;
  const cIdx = CONTRASTS.indexOf(f.contrast);
  const filtered = (cIdx >= 0)
    ? rows.filter(r => r.contrast_id === cIdx)
    : rows;

  const BB = PAYLOAD.backbones;
  const bbIdxById = _backboneIdxById;
  const K = PAYLOAD.kinases;
  const ki = _kinaseIdxById.get(kinase_id);
  const kinaseName = ki != null ? K.name[ki] : "";

  // Group edge rows by backbone, keeping per-contrast support+concordance.
  const groups = new Map();
  for (const r of filtered) {
    let g = groups.get(r.backbone_id);
    if (!g) {
      g = { backbone_id: r.backbone_id, byContrast: new Array(CONTRASTS.length).fill(null), maxAbs: 0 };
      groups.set(r.backbone_id, g);
    }
    g.byContrast[r.contrast_id] = {
      support: r.support_contribution,
      concordance: r.concordance,
    };
    const m = Math.abs(r.support_contribution);
    if (m > g.maxAbs) g.maxAbs = m;
  }

  // For each group, determine per-contrast role (imputed step vs enrichment-only)
  // by checking whether this kinase appears in BB.imputed_nodes_union_<contrast>.
  const imputedColsCache = CONTRASTS.map(c => BB["imputed_nodes_union_" + c]);
  const grouped = Array.from(groups.values());
  for (const g of grouped) {
    const bi = bbIdxById.get(g.backbone_id);
    let nImp = 0, nEnr = 0;
    for (let ci = 0; ci < CONTRASTS.length; ci++) {
      const cell = g.byContrast[ci];
      if (!cell) continue;
      const raw = (bi != null && imputedColsCache[ci]) ? (imputedColsCache[ci][bi] || "") : "";
      const isImputed = raw && kinaseName &&
        raw.split(";").some(s => s.trim() === kinaseName);
      cell.role = isImputed ? "imp" : "enr";
      if (isImputed) nImp++; else nEnr++;
    }
    g.role = (nImp > 0 && nEnr > 0) ? "mixed" : (nImp > 0 ? "imp" : "enr");
    g.nContrasts = nImp + nEnr;
  }

  grouped.sort((a, b) => b.maxAbs - a.maxAbs);
  const TOP = 100;
  const shown = grouped.slice(0, TOP);

  const roleLabel = { imp: "imputed-step", enr: "enrichment", mixed: "mixed" };
  const roleClass = { imp: "imp", enr: "expr", mixed: "mix" };

  const supCell = (cell) => {
    if (!cell) return '<td class="bb-sup"></td>';
    const v = cell.support;
    const m = Math.abs(v);
    const dir = cell.concordance > 0 ? "↑" : (cell.concordance < 0 ? "↓" : "·");
    const color = v > 0 ? "var(--up-red, #c53030)" : (v < 0 ? "var(--down-blue, #2b6cb0)" : "#999");
    const mark = cell.role === "imp" ? "★" : "";
    const title = `${v.toFixed(3)} (${cell.role === "imp" ? "imputed step" : "enrichment"})`;
    return `<td class="bb-sup" title="${title}" style="color:${color};white-space:nowrap;padding:2px 4px;font-size:11px;text-align:center;">` +
      `<span style="font-weight:600;">${dir}${m.toFixed(2)}</span>` +
      (mark ? `<span style="margin-left:2px;">${mark}</span>` : "") +
      `</td>`;
  };

  const shortContrast = (c) => c.replace(/_(\d+)mo$/, "·$1").replace(/^ApTt/, "AT");
  const headContrasts = CONTRASTS.map(c =>
    `<th title="Display label: ${shortContrast(c)}\nRaw column: support_contribution_${c}\nDefinition: Kinase support contribution for ${c}." style="padding:2px 4px;font-size:11px;text-align:center;white-space:nowrap;">${shortContrast(c)}</th>`
  ).join("");
  const parts = [
    `<div class="muted" style="margin-bottom:4px;">Showing top ${shown.length} of ${grouped.length} backbones` +
    (cIdx >= 0 ? ` (contrast ${f.contrast})` : "") +
    ` · ★ = kinase imputed as a pathway step; otherwise support is from enrichment of substrates.</div>`,
    '<div style="overflow-x:auto;max-width:100%;">',
    '<table class="data-table" style="font-size:11px;"><thead><tr>',
    '<th title="Display label: Receiver\nRaw column: receiver\nDefinition: Receiver cell type for the pathway backbone.">Receiver</th>',
    '<th title="Display label: Receptor\nRaw column: Receptor\nDefinition: Receptor gene in the pathway backbone.">Receptor</th>',
    '<th title="Display label: EM\nRaw column: EM\nDefinition: Extracellular-matrix or intermediate effector molecule in the pathway backbone.">EM</th>',
    '<th title="Display label: Target\nRaw column: Target\nDefinition: Target gene in the pathway backbone.">Target</th>',
    headContrasts,
    '<th title="Display label: Role\nRaw column: pathway_evidence_backbone\nDefinition: Whether this kinase is an imputed pathway step or substrate-enrichment support.">Role</th>',
    '</tr></thead><tbody>',
  ];
  for (const g of shown) {
    const bi = bbIdxById.get(g.backbone_id);
    const rcv = bi != null ? RECEIVERS[BB.receiver_id[bi]] : "?";
    const rcp = bi != null ? BB.Receptor[bi] : "?";
    const em  = bi != null ? BB.EM[bi] : "?";
    const tgt = bi != null ? BB.Target[bi] : "?";
    const supCells = g.byContrast.map(supCell).join("");
    const cls = roleClass[g.role] || "lo";
    parts.push(
      `<tr><td>${rcv}</td><td>${rcp}</td><td>${em}</td><td>${tgt}</td>` +
      supCells +
      `<td><span class="badge ${cls}">${roleLabel[g.role]}</span></td></tr>`
    );
  }
  parts.push("</tbody></table></div>");
  container.innerHTML = parts.join("");
}

// Module-level handle so _syncKinaseFilterUI can re-render multi-selects from
// outside wireKinaseTable's closure (e.g. after a cross-tab handoff).
let _kineRenderMultiselect = null;

function wireKinaseTable() {
  const tbl = document.getElementById("ke-table");
  if (!tbl) return;

  tbl.querySelectorAll("thead th").forEach(th => {
    th.addEventListener("click", () => {
      const col = th.dataset.col;
      const cur = KinaseFilter.get();
      if (col === "nes_profile") {
        // Sort by NES profile: first click selects it descending.
        if (cur.sortCol !== col) KinaseFilter.set({sortCol: col, sortAsc: false});
        else KinaseFilter.set({sortAsc: !cur.sortAsc});
        renderKinaseExplorer();
        return;
      }
      if (cur.sortCol === col) KinaseFilter.set({sortAsc: !cur.sortAsc});
      else KinaseFilter.set({sortCol: col, sortAsc: false});
      renderKinaseExplorer();
    });
  });
  tbl.querySelector("tbody").addEventListener("click", ev => {
    const tr = ev.target.closest("tr.ke-row");
    if (!tr) return;
    const kid = parseInt(tr.dataset.kid, 10);
    Store.dispatch({type:"SET_SELECTION", key:"kinase", value: kid});
  });
  tbl.querySelector("tbody").addEventListener("keydown", ev =>
    _activateRowOnKey(ev, "tr.ke-row", tr => {
      const kid = parseInt(tr.dataset.kid, 10);
      Store.dispatch({type:"SET_SELECTION", key:"kinase", value: kid});
    }));
  const search = document.getElementById("ke-search");
  if (search) {
    search.value = KinaseFilter.get("search");
    search.addEventListener("input", ev => {
      KinaseFilter.set({search: ev.target.value});
      renderKinaseExplorer();
    });
  }

  // Multiselect option sources.
  const AI = PAYLOAD.attribution_index || {};
  const allCells = Array.from(new Set(AI.cell_type || [])).sort();
  const MS_OPTS = {
    disease:    ["App","Tau","ApTt"],
    timepoint:  ["2mo","4mo","6mo"],
    celltype:   allCells,
  };

  // Render a multiselect into its placeholder span. Idempotent.
  function _renderMultiselect(key) {
    const host = document.getElementById("ke-ms-" + key);
    if (!host) return;
    const label = host.dataset.label || key;
    const opts = MS_OPTS[key] || [];
    const cur = (KinaseFilter.get(key) || []).slice();
    const curSet = new Set(cur);
    const summary = cur.length === 0 ? "Any"
      : cur.length <= 2 ? cur.join(", ")
      : `${cur.length} selected`;
    const optsHtml = opts.map(v => {
      const checked = curSet.has(v) ? " checked" : "";
      return `<label class="ms-opt"><input type="checkbox" data-val="${_escapeHtml(v)}"${checked}/>${_escapeHtml(v)}</label>`;
    }).join("");
    host.innerHTML =
      `<span style="margin-right:4px;">${_escapeHtml(label)}</span>` +
      `<span class="ms-wrap">` +
        `<button type="button" class="ms-button" data-active="${cur.length ? 1 : 0}" ` +
          `aria-haspopup="true" aria-expanded="false">${_escapeHtml(summary)}</button>` +
        `<div class="ms-panel" role="listbox" aria-multiselectable="true">` +
          `<div class="ms-action" data-action="clear">Clear</div>` +
          `<div class="ms-divider"></div>` +
          optsHtml +
        `</div>` +
      `</span>`;
    const wrap = host.querySelector(".ms-wrap");
    const btn  = wrap.querySelector(".ms-button");
    const panel = wrap.querySelector(".ms-panel");
    btn.addEventListener("click", ev => {
      ev.stopPropagation();
      const open = panel.classList.toggle("open");
      btn.setAttribute("aria-expanded", open ? "true" : "false");
      // Close other open panels.
      document.querySelectorAll(".ms-panel.open").forEach(p => {
        if (p !== panel) {
          p.classList.remove("open");
          const b = p.parentElement && p.parentElement.querySelector(".ms-button");
          if (b) b.setAttribute("aria-expanded", "false");
        }
      });
    });
    panel.addEventListener("click", ev => ev.stopPropagation());
    panel.querySelectorAll('input[type="checkbox"]').forEach(cb => {
      cb.addEventListener("change", () => {
        const next = (KinaseFilter.get(key) || []).slice();
        const v = cb.dataset.val;
        const i = next.indexOf(v);
        if (cb.checked && i < 0) next.push(v);
        else if (!cb.checked && i >= 0) next.splice(i, 1);
        KinaseFilter.set({[key]: next});
        _renderMultiselect(key);
        renderKinaseExplorer();
      });
    });
    const clearBtn = panel.querySelector('[data-action="clear"]');
    if (clearBtn) clearBtn.addEventListener("click", () => {
      KinaseFilter.set({[key]: []});
      _renderMultiselect(key);
      renderKinaseExplorer();
    });
  }
  // Close panels on outside click (one-time wiring).
  if (!window._msOutsideWired) {
    document.addEventListener("click", () => {
      document.querySelectorAll(".ms-panel.open").forEach(p => {
        p.classList.remove("open");
        const b = p.parentElement && p.parentElement.querySelector(".ms-button");
        if (b) b.setAttribute("aria-expanded", "false");
      });
    });
    window._msOutsideWired = true;
  }

  ["disease","timepoint","celltype"].forEach(_renderMultiselect);
  _kineRenderMultiselect = _renderMultiselect;

  // Confidence (single, ordinal threshold).
  const confSel = document.getElementById("ke-filter-confidence");
  if (confSel) {
    confSel.value = KinaseFilter.get("confidence") || "";
    confSel.addEventListener("change", () => {
      KinaseFilter.set({confidence: confSel.value});
      renderKinaseExplorer();
    });
  }

  // WMB specificity tier minimum (single, ordinal threshold).
  const wmbSel = document.getElementById("ke-filter-wmb");
  if (wmbSel) {
    wmbSel.value = String(KinaseFilter.get("wmbMin") || 0);
    wmbSel.addEventListener("change", () => {
      KinaseFilter.set({wmbMin: parseInt(wmbSel.value, 10) || 0});
      renderKinaseExplorer();
    });
  }

  // n_sig minimum (numeric input).
  const nsigInp = document.getElementById("ke-filter-nsig-min");
  if (nsigInp) {
    nsigInp.value = String(KinaseFilter.get("nSigMin") || 0);
    nsigInp.addEventListener("change", () => {
      const v = Math.max(0, Math.min(9, parseInt(nsigInp.value, 10) || 0));
      nsigInp.value = String(v);
      KinaseFilter.set({nSigMin: v});
      renderKinaseExplorer();
    });
  }

  const resetBtn = document.getElementById("ke-filter-reset");
  if (resetBtn) {
    resetBtn.addEventListener("click", () => {
      KinaseFilter.reset();
      _syncKinaseFilterUI();
      renderKinaseExplorer();
    });
  }
}

// Re-pushes the persisted KinaseFilter state into all the toolbar inputs.
// Used after programmatic mutations (e.g. cross-tab handoff prefilling
// disease/timepoint from a Temporal v2 click) so the dropdowns reflect the
// new state without a full page rebuild.
function _syncKinaseFilterUI() {
  const inp = document.getElementById("ke-search");
  if (inp) inp.value = KinaseFilter.get("search") || "";
  if (_kineRenderMultiselect) {
    ["disease","timepoint","celltype"].forEach(k => _kineRenderMultiselect(k));
  }
  const confSel = document.getElementById("ke-filter-confidence");
  if (confSel) confSel.value = KinaseFilter.get("confidence") || "";
  const wmbSel = document.getElementById("ke-filter-wmb");
  if (wmbSel) wmbSel.value = String(KinaseFilter.get("wmbMin") || 0);
  const nsigInp = document.getElementById("ke-filter-nsig-min");
  if (nsigInp) nsigInp.value = String(KinaseFilter.get("nSigMin") || 0);
}

// ---------------------------------------------------------------------------
// Pathway Explorer tab
// ---------------------------------------------------------------------------
let peSortCol = "tpds";
let peSortAsc = false;
let peSearch = "";
let peTrajectory = "all";          // "all" | "App" | "Tau" | "ApTt" | "2mo" | "4mo" | "6mo"
let _peRows = null;
let _peSearchTimer = null;
let _peTrajMaskCache = null;       // bitmask cached for the active trajectory

const PE_TRAJECTORIES = {
  all:  { label: "All contrasts", contrasts: [] },
  App:  { label: "App trajectory",   contrasts: ["App_2mo","App_4mo","App_6mo"] },
  Tau:  { label: "Tau trajectory",   contrasts: ["Tau_2mo","Tau_4mo","Tau_6mo"] },
  ApTt: { label: "ApTt trajectory",  contrasts: ["ApTt_2mo","ApTt_4mo","ApTt_6mo"] },
  "2mo": { label: "2mo cross-section", contrasts: ["App_2mo","Tau_2mo","ApTt_2mo"] },
  "4mo": { label: "4mo cross-section", contrasts: ["App_4mo","Tau_4mo","ApTt_4mo"] },
  "6mo": { label: "6mo cross-section", contrasts: ["App_6mo","Tau_6mo","ApTt_6mo"] },
};

function _peTrajMask() {
  if (_peTrajMaskCache != null) return _peTrajMaskCache;
  let m = 0;
  for (const c of (PE_TRAJECTORIES[peTrajectory] || PE_TRAJECTORIES.all).contrasts) {
    const idx = CONTRASTS.indexOf(c);
    if (idx >= 0) m |= (1 << idx);
  }
  _peTrajMaskCache = m;
  return m;
}

function _peCsetMatch(rowMask) {
  if (peTrajectory === "all") return true;
  const sel = _peTrajMask();
  // Implicit "any": the backbone passes both nulls in at least one of the
  // contrasts named by the active trajectory.
  return (rowMask & sel) !== 0;
}

function _peContrastChips(mask) {
  // Render passing contrasts as small inline chips. Up to 3 visible, +N overflow tooltip.
  const passing = [];
  for (let i = 0; i < CONTRASTS.length; i++) {
    if (mask & (1 << i)) passing.push(CONTRASTS[i]);
  }
  if (passing.length === 0) return '<span class="muted">—</span>';
  const SHOW = 3;
  const head = passing.slice(0, SHOW)
    .map(c => `<span class="pe-cchip">${c}</span>`).join("");
  const tail = passing.length > SHOW
    ? `<span class="pe-cchip pe-cchip-more" title="${passing.join(", ")}">+${passing.length - SHOW}</span>`
    : "";
  return head + tail;
}

function _popcount(m) {
  m = m - ((m >> 1) & 0x55555555);
  m = (m & 0x33333333) + ((m >> 2) & 0x33333333);
  return (((m + (m >> 4)) & 0x0f0f0f0f) * 0x01010101) >> 24;
}

function _pathwayEvidenceLabel(v) {
  return PATHWAY_EVIDENCE_LABELS[v] || "Unknown";
}

function _pathwayEvidenceClass(v) {
  if (v === "expression-confirmed") return "expr";
  if (v === "kinase-imputed") return "imp";
  if (v === "mixed") return "mix";
  return "lo";
}

function _pathwayEvidenceRank(v) {
  if (v === "expression-confirmed") return 0;
  if (v === "kinase-imputed") return 1;
  if (v === "mixed") return 2;
  return 3;
}

function _pathwayEvidenceBadge(v) {
  return `<span class="badge ${_pathwayEvidenceClass(v)}">${_pathwayEvidenceLabel(v)}</span>`;
}

function _pathwayEvidenceChip(v, label) {
  return `<span class="pe-chip ${_pathwayEvidenceClass(v)}">${label}</span>`;
}

function _buildPathwayRowModel() {
  const BB = PAYLOAD.backbones;
  const n = BB.id.length;
  const tpdsCols = CONTRASTS.map(c => BB["mean_tpds_" + c]);
  const out = new Array(n);
  for (let i = 0; i < n; i++) {
    const tpds = new Array(CONTRASTS.length);
    for (let c = 0; c < CONTRASTS.length; c++) tpds[c] = tpdsCols[c][i];
    out[i] = {
      idx: i,
      id: BB.id[i],
      receiver_id: BB.receiver_id[i],
      receiver: RECEIVERS[BB.receiver_id[i]],
      Receptor: BB.Receptor[i] || "",
      EM: BB.EM[i] || "",
      Target: BB.Target[i] || "",
      sender_mask: BB.sender_mask[i],
      n_senders: BB.n_senders[i],
      n_senders_sig: BB.n_senders_significant[i],
      max_abs_tpds: BB.max_abs_tpds[i],
      sig_mask: BB.significant_both_mask[i],
      sig_count: _popcount(BB.significant_both_mask[i]),
      pathway_evidence_all: BB.all_contrasts_pathway_evidence[i] || "expression-confirmed",
      all_imputed_nodes_union: BB.all_imputed_nodes_union[i] || "",
      all_n_expression_confirmed: BB.all_n_expression_confirmed[i] || 0,
      all_n_kinase_imputed: BB.all_n_kinase_imputed[i] || 0,
      _tpds: tpds,
    };
  }
  return out;
}

function _ensurePathwayIndexes() {
  if (_peRows === null) _peRows = _buildPathwayRowModel();
  _ensureBackboneIdx();
}

function _peCompare(a, b, cIdx) {
  const col = peSortCol;
  let va, vb;
  if (col === "tpds") {
    va = cIdx >= 0 ? a._tpds[cIdx] : a.max_abs_tpds;
    vb = cIdx >= 0 ? b._tpds[cIdx] : b.max_abs_tpds;
    if (va == null) va = -Infinity;
    if (vb == null) vb = -Infinity;
  }
  else if (col === "passing_contrasts") { va = a.sig_count; vb = b.sig_count; }
  else if (col === "receiver") { va = a.receiver; vb = b.receiver; }
  else if (col === "pathway_evidence") {
    va = _pathwayEvidenceRank(cIdx >= 0
      ? (PAYLOAD.backbones["pathway_evidence_backbone_" + CONTRASTS[cIdx]][a.idx] || "expression-confirmed")
      : a.pathway_evidence_all);
    vb = _pathwayEvidenceRank(cIdx >= 0
      ? (PAYLOAD.backbones["pathway_evidence_backbone_" + CONTRASTS[cIdx]][b.idx] || "expression-confirmed")
      : b.pathway_evidence_all);
  }
  else { va = a[col]; vb = b[col]; }
  if (va == null && vb == null) return 0;
  if (va == null) return 1;
  if (vb == null) return -1;
  if (typeof va === "string") return peSortAsc
    ? va.localeCompare(vb) : vb.localeCompare(va);
  return peSortAsc ? (va - vb) : (vb - va);
}

function renderPathwayExplorer() {
  const tbody = document.querySelector("#pe-table tbody");
  if (!tbody) return;
  _ensurePathwayIndexes();
  const f = Store.state.filters;
  const cIdx = CONTRASTS.indexOf(f.contrast);
  const selBid = Store.state.selection.backbone;
  const q = peSearch.trim().toLowerCase();
  const baseIdx = getFilteredIndices();

  const tpdsMin = Math.max(0, Number(Store.state.view.pathwayScoreMin) || 0);
  const visible = [];
  for (const i of baseIdx) {
    const r = _peRows[i];
    if (r.sig_count === 0) continue;  // hard sig-both gate (formerly checkbox)
    if (!_peCsetMatch(r.sig_mask)) continue;
    if (tpdsMin > 0) {
      const mt = r.max_abs_tpds;
      if (mt == null || mt < tpdsMin) continue;
    }
    if (q && !(r.Receptor.toLowerCase().includes(q) ||
               r.EM.toLowerCase().includes(q) ||
               r.Target.toLowerCase().includes(q))) continue;
    visible.push(r);
  }
  visible.sort((a, b) => _peCompare(a, b, cIdx));

  document.querySelectorAll("#pe-table thead th").forEach(th => {
    const c = th.dataset.col;
    th.textContent = th.textContent.replace(/[ ▲▼]+$/, "");
    if (c === peSortCol) th.textContent += peSortAsc ? " ▲" : " ▼";
  });

  const CAP = 2000;
  const shown = visible.slice(0, CAP);
  const parts = [];
  for (const r of shown) {
    const selCls = r.id === selBid ? " selected" : "";
    const t = cIdx >= 0 ? r._tpds[cIdx] : r.max_abs_tpds;
    const tStr = (t == null) ? "—" : t.toFixed(3);
    const evidence = cIdx >= 0
      ? (PAYLOAD.backbones["pathway_evidence_backbone_" + f.contrast][r.idx] || "expression-confirmed")
      : r.pathway_evidence_all;
    const evidenceLabel = _pathwayEvidenceLabel(evidence);
    r.pathway_evidence_label = evidenceLabel;
    parts.push(
      `<tr class="pe-row${selCls}" data-bid="${r.id}" tabindex="0" ` +
      `aria-label="Backbone ${r.Receptor} to ${r.EM} to ${r.Target}; receiver ${r.receiver}; support ${evidenceLabel}; TPDS ${tStr}; ${r.sig_count} passing-null contrasts">` +
      `<td>${r.receiver}</td>` +
      `<td>${r.Receptor}</td>` +
      `<td>${r.EM}</td>` +
      `<td>${r.Target}</td>` +
      `<td>${_pathwayEvidenceBadge(evidence)}</td>` +
      `<td>${tStr}</td>` +
      `<td class="pe-cchip-cell">${_peContrastChips(r.sig_mask)}</td>` +
      `<td>${r.n_senders_sig}/${r.n_senders}</td>` +
      `<td>${r.max_abs_tpds == null ? "—" : r.max_abs_tpds.toFixed(3)}</td>` +
      `</tr>`
    );
  }
  tbody.innerHTML = parts.join("");
  const countEl = document.getElementById("pe-count");
  if (countEl) {
    const cap = visible.length > CAP ? ` (first ${CAP} shown)` : "";
    countEl.textContent = `${visible.length.toLocaleString()} / ${_peRows.length.toLocaleString()} backbones${cap}`;
  }
}

function _updatePathwayRowSelection(bid) {
  _updateRowSelection("#pe-table", "pe-row", "data-bid", bid);
}

function renderPathwayDetail(backbone_id) {
  const el = document.getElementById("pe-detail");
  if (!el) return;
  if (backbone_id == null) {
    el.innerHTML = '<div class="muted">Select a backbone to see details.</div>';
    return;
  }
  _ensurePathwayIndexes();
  const BB = PAYLOAD.backbones;
  const i = _backboneIdxById.get(backbone_id);
  if (i == null) {
    el.innerHTML = '<div class="muted">Backbone not found.</div>';
    return;
  }
  const receiver = RECEIVERS[BB.receiver_id[i]];
  const rcp = BB.Receptor[i] || "—";
  const em = BB.EM[i] || "—";
  const tgt = BB.Target[i] || "—";
  const sigMask = BB.significant_both_mask[i];
  const evidenceSummary = BB.all_contrasts_pathway_evidence[i] || "expression-confirmed";
  const allImputedNodes = BB.all_imputed_nodes_union[i] || "";
  const nExpr = BB.all_n_expression_confirmed[i] || 0;
  const nImp = BB.all_n_kinase_imputed[i] || 0;
  const chips = CONTRASTS.map((c, ci) => {
    const on = ((sigMask >> ci) & 1) ? " on" : "";
    return `<span class="pe-chip${on}">${c}</span>`;
  }).join("");
  const evidenceChips = CONTRASTS.map((c) => {
    const ev = BB["pathway_evidence_backbone_" + c][i];
    if (!ev) return "";
    return _pathwayEvidenceChip(ev, `${c}: ${_pathwayEvidenceLabel(ev)}`);
  }).join("");
  const nSendSig = BB.n_senders_significant[i];
  const nSend = BB.n_senders[i];

  el.innerHTML =
    `<h3>${rcp} → ${em} → ${tgt}</h3>` +
    `<div class="meta">Receiver: ${receiver} · Senders: ${nSendSig}/${nSend} sig · Support: ${_pathwayEvidenceLabel(evidenceSummary)} · Expression-confirmed paths: ${nExpr} · Kinase-imputed paths: ${nImp} · Imputed positions observed: ${allImputedNodes || "none"}</div>` +
    `<div class="detail-chips">${_pathwayEvidenceBadge(evidenceSummary)}</div>` +
    `<h4>Passed both nulls by contrast <span class="metric-help" tabindex="0" data-metric="passedNulls" title="${_metricShort('passedNulls')}">i</span></h4><div>${chips}</div>` +
    `<h4>Pathway support by contrast <span class="metric-help" tabindex="0" data-metric="pathwaySupportH" title="${_metricShort('pathwaySupportH')}">i</span></h4><div>${evidenceChips || '<span class="muted">No support provenance available.</span>'}</div>` +
    `<h4>TPDS across contrasts <span class="metric-help" tabindex="0" data-metric="tpdsCross" title="${_metricShort('tpdsCross')}">i</span></h4><div id="pe-detail-cross"></div>` +
    `<h4>Driving kinases <span class="metric-help" tabindex="0" data-metric="drivingKinasesH" title="${_metricShort('drivingKinasesH')}">i</span></h4><div id="pe-detail-kinases" class="muted">loading…</div>`;

  const tpds = CONTRASTS.map(c => BB["mean_tpds_" + c][i]);
  const barColors = tpds.map(v => {
    if (v == null || v === 0) return "#cfd8dc";
    return v > 0 ? "var(--up-red)" : "var(--down-blue)";
  });
  const outlines = CONTRASTS.map((_, ci) =>
    ((sigMask >> ci) & 1) ? "#000" : "rgba(0,0,0,0)");
  Plotly.react("pe-detail-cross", [{
    type: "bar", x: CONTRASTS, y: tpds.map(v => v == null ? 0 : v),
    marker: { color: barColors, line: { color: outlines, width: 1.5 } },
    hovertemplate: "%{x}<br>TPDS %{y:.3f}<extra></extra>",
  }], {
    margin:{l:40,r:10,t:6,b:60}, height:180,
    yaxis:{zeroline:true, zerolinecolor:"#bbb"},
    xaxis:{tickangle:-35},
  }, {displaylogo:false, responsive:true});

  renderPathwayKinases(backbone_id);
}

async function renderPathwayKinases(backbone_id) {
  const container = document.getElementById("pe-detail-kinases");
  if (!container) return;
  _ensurePathwayIndexes();
  const bi = _backboneIdxById.get(backbone_id);
  if (bi == null) {
    container.innerHTML = '<div class="muted">Backbone not found.</div>';
    return;
  }
  if (PAYLOAD.backbones.significant_both_mask[bi] === 0) {
    container.innerHTML = '<div class="muted">No significant kinase edges.</div>';
    return;
  }
  let rows;
  try {
    rows = await SliceCache.backboneEdges(backbone_id);
  } catch (e) {
    if (Store.state.selection.backbone !== backbone_id) return;
    container.innerHTML = `<div class="muted">Failed to load: ${e.message}</div>`;
    return;
  }
  if (Store.state.selection.backbone !== backbone_id) return;

  const f = Store.state.filters;
  const cIdx = CONTRASTS.indexOf(f.contrast);
  const filtered = (cIdx >= 0)
    ? rows.filter(r => r.contrast_id === cIdx)
    : rows;

  const byK = new Map();
  for (const r of filtered) {
    let g = byK.get(r.kinase_id);
    if (!g) { g = { sum_abs:0, net:0, up:0, down:0, n:0 }; byK.set(r.kinase_id, g); }
    const s = Math.abs(r.support_contribution);
    const sign = (r.concordance > 0) ? 1 : (r.concordance < 0 ? -1 : 0);
    g.sum_abs += s;
    g.net += sign * s;
    if (sign > 0) g.up++;
    else if (sign < 0) g.down++;
    g.n++;
  }
  const groups = Array.from(byK.entries()).map(([kid, g]) => ({ kid, ...g }));
  groups.sort((a, b) => b.sum_abs - a.sum_abs);

  _ensureKinaseIdx();
  const K = PAYLOAD.kinases;
  const famMap = META.familyMap || {};

  const TOP = 200;
  const shown = groups.slice(0, TOP);
  const header = cIdx >= 0
    ? `Showing ${shown.length} of ${groups.length} kinases (contrast ${f.contrast}).`
    : `Showing ${shown.length} of ${groups.length} kinases (all contrasts).`;
  const parts = [
    `<div class="muted">${header}</div>`,
    '<table class="data-table"><thead><tr>',
    `<th data-metric="kinaseName" title="${_metricShort('kinaseName')}">Kinase</th>`,
    `<th data-metric="kinaseFamily" title="${_metricShort('kinaseFamily')}">Family</th>`,
    `<th data-metric="support" title="${_metricShort('support')}">Support</th>`,
    `<th data-metric="drivingDirection" title="${_metricShort('drivingDirection')}">Direction</th>`,
    `<th data-metric="trend" title="${_metricShort('trend')}">Trend</th>`,
    '</tr></thead><tbody>',
  ];
  for (const g of shown) {
    const kIdx = _kinaseIdxById.get(g.kid);
    const name = kIdx != null ? K.name[kIdx] : `kid:${g.kid}`;
    const fam = famMap[name] || "";
    const conc = (g.up > g.down) ? "↑" : (g.down > g.up ? "↓" : "—");
    parts.push(
      `<tr><td>${name}</td><td>${fam}</td>` +
      `<td>${g.sum_abs.toFixed(3)}</td>` +
      `<td>${g.net.toFixed(3)}</td>` +
      `<td>${conc} (${g.up}/${g.down})</td></tr>`
    );
  }
  parts.push("</tbody></table>");
  parts.push('<div class="muted" style="margin-top:6px;">Top 200 kinases shown. Open the How-to-read drawer for column meanings.</div>');
  container.innerHTML = parts.join("");
}

function wirePathwayTable() {
  const tbl = document.getElementById("pe-table");
  if (!tbl) return;
  tbl.querySelectorAll("thead th").forEach(th => {
    th.addEventListener("click", () => {
      const col = th.dataset.col;
      if (peSortCol === col) peSortAsc = !peSortAsc;
      else { peSortCol = col; peSortAsc = false; }
      renderPathwayExplorer();
    });
  });
  tbl.querySelector("tbody").addEventListener("click", ev => {
    const tr = ev.target.closest("tr.pe-row");
    if (!tr) return;
    const bid = parseInt(tr.dataset.bid, 10);
    Store.dispatch({type:"SET_SELECTION", key:"backbone", value: bid});
  });
  tbl.querySelector("tbody").addEventListener("keydown", ev =>
    _activateRowOnKey(ev, "tr.pe-row", tr => {
      const bid = parseInt(tr.dataset.bid, 10);
      Store.dispatch({type:"SET_SELECTION", key:"backbone", value: bid});
    }));
  const search = document.getElementById("pe-search");
  if (search) search.addEventListener("input", ev => {
    const val = ev.target.value;
    if (_peSearchTimer) clearTimeout(_peSearchTimer);
    _peSearchTimer = setTimeout(() => {
      peSearch = val;
      renderPathwayExplorer();
    }, 250);
  });
  _renderPeTrajectoryButtons();
  const tpdsInp = document.getElementById("pe-tpds-min");
  if (tpdsInp) {
    tpdsInp.value = Store.state.view.pathwayScoreMin || 0;
    tpdsInp.addEventListener("change", ev => {
      const v = Math.max(0, parseFloat(ev.target.value) || 0);
      Store.dispatch({type:"SET_VIEW", key:"pathwayScoreMin", value: v});
    });
  }
}

function _renderPeTrajectoryButtons() {
  const host = document.getElementById("pe-traj-buttons");
  if (!host) return;
  const order = ["all", "App", "Tau", "ApTt", "2mo", "4mo", "6mo"];
  host.innerHTML = order.map(k => {
    const t = PE_TRAJECTORIES[k];
    const on = peTrajectory === k;
    const tip = t.contrasts.length
      ? `Show backbones passing in any of ${t.contrasts.join(", ")}.`
      : "Show all passing backbones (no trajectory filter).";
    return `<button type="button" class="pe-cset-chip${on ? " on" : ""}" data-k="${k}" aria-pressed="${on}" title="${tip}">${t.label}</button>`;
  }).join("");
  host.querySelectorAll(".pe-cset-chip").forEach(btn => {
    btn.addEventListener("click", () => {
      peTrajectory = btn.dataset.k;
      _peTrajMaskCache = null;
      _renderPeTrajectoryButtons();
      renderPathwayExplorer();
    });
  });
}

// ---------------------------------------------------------------------------
// Pathway Graph (Cytoscape) — aggregates filtered backbones into an
// R → EM → T node DAG where each node is a unique gene across many backbones.
// ---------------------------------------------------------------------------
const GRAPH_MAX_NODES = 600;
const GRAPH_COLORS = { "Receptor":"#43a047", "EM":"#fb8c00", "Target":"#5c6bc0" };

let _cyInstance = null;
let _nodeInfo = null;  // Map<nodeId, {bbs:number[], scoreSum, scoreN, nUp, nDown}>

function _destroyCy() {
  if (_cyInstance) { try { _cyInstance.destroy(); } catch(e) {} _cyInstance = null; }
  _nodeInfo = null;
}

function _graphPlaceholder(msg) {
  const el = document.getElementById("cy");
  if (!el) return;
  el.innerHTML = '<div class="graph-placeholder">' + msg + "</div>";
}

function _buildGraphData(indices, contrast) {
  const BB = PAYLOAD.backbones;
  const scoreCol = BB["observed_score_" + contrast];
  const tpdsCol = BB["mean_tpds_" + contrast];
  const nodeDeg = new Map();
  const nodeType = new Map();
  const nodeInfo = new Map();
  const edgeScores = new Map();
  const edgeTpds = new Map();
  const edgeCounts = new Map();

  for (const i of indices) {
    const bid = BB.id[i];
    const rGene = BB.Receptor[i];
    const emGene = BB.EM[i];
    const tGene = BB.Target[i];
    const rId = "R:" + rGene;
    const eId = "E:" + emGene;
    const tId = "T:" + tGene;
    const score = scoreCol ? scoreCol[i] : null;
    const tpds = tpdsCol ? tpdsCol[i] : null;

    for (const [nid, type] of [[rId, "Receptor"], [eId, "EM"], [tId, "Target"]]) {
      nodeDeg.set(nid, (nodeDeg.get(nid) || 0) + 1);
      if (!nodeType.has(nid)) nodeType.set(nid, type);
      let info = nodeInfo.get(nid);
      if (!info) {
        info = {bbs:[], scoreSum:0, scoreN:0, nUp:0, nDown:0};
        nodeInfo.set(nid, info);
      }
      info.bbs.push(bid);
      if (score != null) { info.scoreSum += score; info.scoreN++; }
      if (tpds != null) { if (tpds > 0) info.nUp++; else if (tpds < 0) info.nDown++; }
    }

    const rek = rId + ">" + eId;
    const etk = eId + ">" + tId;
    const s = (score == null) ? 0 : score;
    const t = (tpds == null) ? 0 : tpds;
    edgeScores.set(rek, Math.max(edgeScores.get(rek) || 0, s));
    edgeScores.set(etk, Math.max(edgeScores.get(etk) || 0, s));
    edgeTpds.set(rek, (edgeTpds.get(rek) || 0) + t);
    edgeTpds.set(etk, (edgeTpds.get(etk) || 0) + t);
    edgeCounts.set(rek, (edgeCounts.get(rek) || 0) + 1);
    edgeCounts.set(etk, (edgeCounts.get(etk) || 0) + 1);
  }

  // Min-degree filter
  const minDeg = Store.state.view.graphMinDegree | 0;
  let keepIds = [...nodeDeg.keys()].filter(id => nodeDeg.get(id) >= minDeg);
  // Node cap (degree-sorted)
  if (keepIds.length > GRAPH_MAX_NODES) {
    keepIds.sort((a,b) => nodeDeg.get(b) - nodeDeg.get(a));
    keepIds = keepIds.slice(0, GRAPH_MAX_NODES);
  }
  const keep = new Set(keepIds);

  // Local |TPDS| threshold and optional top-N edge cap. Threshold drops edges
  // whose mean |TPDS| falls below the user's value; top-N keeps only the
  // strongest |TPDS| edges as a separate rendering safety net.
  const tpdsMin = Math.max(0, Number(Store.state.view.graphTpdsMin) || 0);
  const topN = Number(Store.state.view.graphTopN) || 0;

  const maxDeg = keepIds.reduce((m, id) => Math.max(m, nodeDeg.get(id)), 1);
  const maxScore = [...edgeScores.values()].reduce((m,v) => Math.max(m,v), 0) || 1;

  // Build candidate edges (after min-degree node filter and |TPDS| threshold)
  let candidates = [];
  let tpdsBelowCount = 0;
  for (const [key, score] of edgeScores.entries()) {
    const [src, tgt] = key.split(">");
    if (!keep.has(src) || !keep.has(tgt)) continue;
    const count = edgeCounts.get(key) || 1;
    const avgTpds = (edgeTpds.get(key) || 0) / count;
    if (Math.abs(avgTpds) < tpdsMin) { tpdsBelowCount++; continue; }
    candidates.push({ key, src, tgt, score, avgTpds });
  }
  // Top-N cap by |TPDS| (descending)
  let topNApplied = false;
  if (topN > 0 && candidates.length > topN) {
    candidates.sort((a,b) => Math.abs(b.avgTpds) - Math.abs(a.avgTpds));
    candidates = candidates.slice(0, topN);
    topNApplied = true;
  }

  // Restrict surviving nodes to those touched by surviving edges.
  const nodesUsed = new Set();
  for (const c of candidates) { nodesUsed.add(c.src); nodesUsed.add(c.tgt); }
  const finalIds = keepIds.filter(id => nodesUsed.has(id));

  const nodes = finalIds.map(id => {
    const type = nodeType.get(id);
    const deg = nodeDeg.get(id);
    const sz = 10 + 30 * Math.sqrt(deg / maxDeg);
    const rank = type === "Receptor" ? 0 : type === "EM" ? 1 : 2;
    return { data: {
      id, label: id.slice(2), type, deg, size: sz,
      color: GRAPH_COLORS[type], rank,
    }};
  });

  const edges = candidates.map(c => {
    const w = 0.5 + 3 * (c.score / maxScore);
    const op = 0.2 + 0.6 * (c.score / maxScore);
    const col = c.avgTpds > 0 ? "#c62828"
              : c.avgTpds < 0 ? "#1565c0" : "#999";
    return { data: {
      id: c.key, source: c.src, target: c.tgt,
      score: c.score, width: w, opacity: op, edgeColor: col,
    }};
  });

  const finalInfo = new Map();
  for (const id of finalIds) finalInfo.set(id, nodeInfo.get(id));

  return { nodes, edges, nodeInfo: finalInfo,
           totalNodes: nodeDeg.size, keptNodes: finalIds.length,
           tpdsBelowCount, topNApplied };
}

function _applyFlowSnap(cy) {
  const w = cy.width() || 800;
  const cols = { "Receptor": w * 0.15, "EM": w * 0.50, "Target": w * 0.85 };
  cy.nodes().forEach(n => {
    const xTarget = cols[n.data("type")];
    const xCur = n.position("x");
    n.position("x", xCur * 0.15 + xTarget * 0.85);
  });
}

function _layoutConfig(layoutName, nNodes) {
  if (layoutName === "concentric") {
    return { name:"concentric",
             concentric: node => 3 - (node.data("rank") || 0),
             levelWidth: () => 1,
             minNodeSpacing: 8, animate:false };
  }
  const cose = { name:"cose", animate:false, randomize:true,
                 nodeRepulsion: () => nNodes > 200 ? 80000 : 40000,
                 idealEdgeLength: () => nNodes > 200
                   ? (layoutName === "flow" ? 60 : 50)
                   : (layoutName === "flow" ? 80 : 70),
                 gravity: layoutName === "flow" ? 0.3 : 0.25,
                 nodeOverlap:20 };
  return cose;
}

function _renderNodeDetail(nodeData) {
  const det = document.getElementById("graph-detail");
  if (!det) return;
  const nodeId = nodeData.id;
  const info = (_nodeInfo && _nodeInfo.get(nodeId))
    || {bbs:[], scoreSum:0, scoreN:0, nUp:0, nDown:0};
  const avgScore = info.scoreN ? (info.scoreSum / info.scoreN) : 0;
  det.innerHTML = "<h3>" + nodeData.label
    + ' <span class="meta">(' + nodeData.type + ")</span></h3>"
    + '<div class="meta">Backbones: ' + nodeData.deg
    + " &middot; avg score: " + avgScore.toFixed(3)
    + " &middot; ↑" + info.nUp + " / ↓" + info.nDown + "</div>"
    + '<button id="graph-filter-btn" class="chip" style="margin-top:8px;">'
    + "Filter Pathway Explorer to this gene</button>";
  const btn = document.getElementById("graph-filter-btn");
  if (btn) btn.addEventListener("click", () => {
    const gene = nodeData.label;
    peSearch = gene;
    const search = document.getElementById("pe-search");
    if (search) search.value = gene;
    Store.dispatch({type:"SET_VIEW", key:"activeTab", value:"pathway"});
  });
}

function _graphActiveContrast() {
  const v = Store.state.view;
  return `${v.graphGenotype}_${v.graphTimepoint}`;
}

function renderGraph() {
  const el = document.getElementById("cy");
  if (!el) return;
  const contrast = _graphActiveContrast();
  // Graph is contrast-driven via its own genotype + timepoint controls; sync
  // the legacy filters.contrast slice so getFilteredIndices uses this snapshot.
  if (Store.state.filters.contrast !== contrast) {
    Store.dispatch({type:"SET_FILTER", key:"contrast", value: contrast});
  }
  el.innerHTML = "";
  const indices = getFilteredIndices();
  const built = _buildGraphData(indices, contrast);
  _nodeInfo = built.nodeInfo;
  const stats = document.getElementById("graph-stats");
  if (stats) {
    let s = `${contrast} · ${built.keptNodes} / ${built.totalNodes} nodes`
      + ` (min-deg ${Store.state.view.graphMinDegree}`;
    if (built.totalNodes > built.keptNodes) s += `, degree-capped at ${GRAPH_MAX_NODES}`;
    s += `), ${built.edges.length} edges`;
    if (built.tpdsBelowCount > 0) s += ` · ${built.tpdsBelowCount} hidden by |TPDS| ≥ ${Store.state.view.graphTpdsMin}`;
    if (built.topNApplied) s += ` · capped at top ${Store.state.view.graphTopN} by |TPDS|`;
    stats.textContent = s;
  }
  if (!built.nodes.length) {
    _destroyCy();
    _graphPlaceholder("No backbones for the current filters.");
    return;
  }

  _destroyCy();
  const layoutName = Store.state.view.graphLayout || "concentric";
  const nNodes = built.nodes.length;
  const layoutCfg = _layoutConfig(layoutName, nNodes);
  _cyInstance = cytoscape({
    container: el,
    elements: { nodes: built.nodes, edges: built.edges },
    style: [
      { selector:"node", style: {
        label:"data(label)", width:"data(size)", height:"data(size)",
        "background-color":"data(color)", "font-size":8,
        "text-valign":"bottom", "text-margin-y":4,
        "text-outline-color":"#fff", "text-outline-width":1,
        "min-zoomed-font-size":6,
      }},
      { selector:"edge", style: {
        width:"data(width)", "line-color":"data(edgeColor)",
        "target-arrow-color":"data(edgeColor)",
        "target-arrow-shape":"triangle", "curve-style":"bezier",
        opacity:"data(opacity)", "arrow-scale":0.6,
      }},
      { selector:"node.highlighted", style: {
        "border-width":3, "border-color":"#e53935",
        "font-weight":"bold", "font-size":10, "z-index":999,
      }},
      { selector:"node.faded", style: { opacity:0.15 } },
      { selector:"edge.faded", style: { opacity:0.05 } },
      { selector:"node.focus-center", style: {
        "border-width":4, "border-color":"#ff6f00", "border-style":"double",
      }},
    ],
    layout: layoutCfg,
    wheelSensitivity: 0.3,
  });
  if (layoutName === "flow") {
    _cyInstance.one("layoutstop", () => _applyFlowSnap(_cyInstance));
  }

  _cyInstance.on("tap", "node", evt => {
    const n = evt.target;
    _cyInstance.elements().removeClass("highlighted faded focus-center");
    const nbh = n.closedNeighborhood();
    _cyInstance.elements().not(nbh).addClass("faded");
    nbh.nodes().addClass("highlighted");
    n.addClass("focus-center");
    _renderNodeDetail(n.data());
  });
  _cyInstance.on("tap", evt => {
    if (evt.target === _cyInstance) {
      _cyInstance.elements().removeClass("highlighted faded focus-center");
      const det = document.getElementById("graph-detail");
      if (det) det.innerHTML = '<div class="muted">Click a node for details.</div>';
    }
  });
}

function wireGraphControls() {
  const v = Store.state.view;
  const genoSel = document.getElementById("graph-genotype");
  if (genoSel) {
    genoSel.value = v.graphGenotype;
    genoSel.addEventListener("change", ev => {
      Store.dispatch({type:"SET_VIEW", key:"graphGenotype", value: ev.target.value});
    });
  }
  const tpSel = document.getElementById("graph-timepoint");
  if (tpSel) {
    tpSel.value = v.graphTimepoint;
    tpSel.addEventListener("change", ev => {
      Store.dispatch({type:"SET_VIEW", key:"graphTimepoint", value: ev.target.value});
    });
  }
  const layoutSel = document.getElementById("graph-layout");
  if (layoutSel) {
    layoutSel.value = v.graphLayout;
    layoutSel.addEventListener("change", ev => {
      Store.dispatch({type:"SET_VIEW", key:"graphLayout", value: ev.target.value});
    });
  }
  const degSel = document.getElementById("graph-min-degree");
  if (degSel) {
    degSel.value = String(v.graphMinDegree);
    degSel.addEventListener("change", ev => {
      Store.dispatch({type:"SET_VIEW", key:"graphMinDegree",
                      value: parseInt(ev.target.value, 10)});
    });
  }
  const tpdsInp = document.getElementById("graph-tpds-min");
  if (tpdsInp) {
    tpdsInp.value = v.graphTpdsMin;
    tpdsInp.addEventListener("change", ev => {
      Store.dispatch({type:"SET_VIEW", key:"graphTpdsMin",
                      value: Math.max(0, parseFloat(ev.target.value) || 0)});
    });
  }
  const topNInp = document.getElementById("graph-top-n");
  if (topNInp) {
    topNInp.value = v.graphTopN == null ? "" : v.graphTopN;
    topNInp.addEventListener("change", ev => {
      const raw = ev.target.value.trim();
      const val = raw === "" ? null : Math.max(0, parseInt(raw, 10) || 0);
      Store.dispatch({type:"SET_VIEW", key:"graphTopN", value: val});
    });
  }
}

function wireGraphKeyboard() {
  const TPS = ["2mo", "4mo", "6mo"];
  document.addEventListener("keydown", ev => {
    if (Store.state.view.activeTab !== "graph") return;
    const tag = (ev.target && ev.target.tagName) || "";
    if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;
    if (ev.metaKey || ev.ctrlKey || ev.altKey) return;
    let handled = false;
    if (ev.key === "ArrowLeft" || ev.key === "ArrowRight") {
      const cur = TPS.indexOf(Store.state.view.graphTimepoint);
      const ni = ((cur + (ev.key === "ArrowLeft" ? -1 : 1)) % TPS.length + TPS.length) % TPS.length;
      Store.dispatch({type:"SET_VIEW", key:"graphTimepoint", value: TPS[ni]});
      handled = true;
    }
    if (handled) ev.preventDefault();
  });
}

// ---------------------------------------------------------------------------
// Glossary
// ---------------------------------------------------------------------------
function syncGlossary() {
  const open = Store.state.view.glossaryOpen;
  const panel = document.getElementById("glossary-panel");
  const toggle = document.getElementById("glossary-toggle");
  if (panel) {
    panel.classList.toggle("open", open);
    panel.setAttribute("aria-hidden", open ? "false" : "true");
  }
  if (toggle) toggle.setAttribute("aria-expanded", open ? "true" : "false");
}

function _kinaseRerenderForFilter(activeTab){
  if (activeTab === "pathway") renderPathwayExplorer();
  if (activeTab === "graph") renderGraph();
  if (activeTab === "temporal" && Store.state.view.temporalLevel === "backbone")
    renderTemporal();
  if (activeTab === "additivity" && Store.state.view.additivityLevel === "backbone")
    renderAdditivity();
  if (activeTab === "kinase") renderKinaseExplorer();
}

// Backbone selection → highlight kinases that drive it. Loaded async from the
// per-backbone edge slice; updates a Set used by the table renderer.
let _highlightKinaseIds = null;
let _highlightForBid = null;
async function _refreshHighlightForBackbone(bid){
  if (bid == null) {
    _highlightKinaseIds = null; _highlightForBid = null;
    return;
  }
  if (bid === _highlightForBid) return;
  _highlightForBid = bid;
  try {
    const rows = await SliceCache.backboneEdges(bid);
    if (Store.state.selection.backbone !== bid) return;
    const s = new Set();
    for (const r of rows) s.add(r.kinase_id);
    _highlightKinaseIds = s;
    if (Store.state.view.activeTab === "kinase") renderKinaseExplorer();
  } catch (e) { console.warn("highlight fetch failed", e); }
}

// ---------------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------------
function boot() {
  populateHeader();
  wireTabs();
  wireKinaseTable();
  wirePathwayTable();
  wireGraphControls();
  wireGraphKeyboard();
  wireSenderMatrix();
  wireSenderMatrixKeyboard();
  wireTemporalControls();
  wireAdditivityControls();
  wireTemporalV2();
  syncHeaderFromStore();
  syncTabsFromStore();
  applyMetricTooltips();
  syncFilterBarToTab(Store.state.view.activeTab);
  syncGlossary();
  wireDrawerResizer();
  wireExportButtons();
  renderHowToDrawer(Store.state.view.activeTab);
  applyHash();
  window.addEventListener("popstate", applyHash);
  window.addEventListener("hashchange", () => {
    if (_serializeHash() !== window.location.hash) applyHash();
  });

  Store.subscribe((next, prev) => {
    const activeTab = next.view.activeTab;
    pushHash();
    if (next.filters !== prev.filters) {
      syncHeaderFromStore();
      if (activeTab === "signal") renderOverview();
      if (activeTab === "kinase") {
        renderKinaseExplorer();
        if (next.selection.kinase != null)
          renderKinaseDetail(next.selection.kinase);
      }
      if (activeTab === "pathway") {
        renderPathwayExplorer();
        if (next.selection.backbone != null)
          renderPathwayDetail(next.selection.backbone);
      }
      if (activeTab === "graph") renderGraph();
      if (activeTab === "senders") renderSenderMatrix();
      if (activeTab === "temporal") renderTemporal();
      if (activeTab === "additivity") renderAdditivity();
      if (activeTab === "temporalv2") renderTemporalV2();
    }
    if (next.selection.kinase !== prev.selection.kinase) {
      syncHeaderFromStore();
      const kid = next.selection.kinase;
      if (kid != null && SliceCache.kinaseBackboneSetSync(kid) === null) {
        SliceCache.loadKinase(kid).then(() => {
          if (Store.state.selection.kinase !== kid) return;
          invalidateFilterCache();
          _kinaseRerenderForFilter(Store.state.view.activeTab);
        }).catch(e => console.warn("kinase slice load failed", e));
      } else {
        invalidateFilterCache();
      }
      if (activeTab === "kinase") {
        _updateKinaseRowSelection(next.selection.kinase);
        renderKinaseDetail(next.selection.kinase);
      }
      if (activeTab !== "kinase") _kinaseRerenderForFilter(activeTab);
    }
    if (next.selection.celltype !== prev.selection.celltype) {
      syncHeaderFromStore();
      invalidateFilterCache();
      _kinaseRerenderForFilter(activeTab);
    }
    if (next.selection.backbone !== prev.selection.backbone) {
      if (activeTab === "pathway") {
        _updatePathwayRowSelection(next.selection.backbone);
        renderPathwayDetail(next.selection.backbone);
      }
      _refreshHighlightForBackbone(next.selection.backbone);
    }
    if (next.view !== prev.view) {
      if (next.view.activeTab !== prev.view.activeTab) {
        syncTabsFromStore();
        syncFilterBarToTab(activeTab);
        renderHowToDrawer(activeTab);
        if (activeTab === "kinase") {
          renderKinaseExplorer();
          if (next.selection.kinase != null)
            renderKinaseDetail(next.selection.kinase);
        }
        if (activeTab === "pathway") {
          renderPathwayExplorer();
          if (next.selection.backbone != null)
            renderPathwayDetail(next.selection.backbone);
        }
        if (activeTab === "graph") renderGraph();
        if (activeTab === "signal") renderOverview();
        if (activeTab === "senders") renderSenderMatrix();
        if (activeTab === "temporal") renderTemporal();
        if (activeTab === "additivity") renderAdditivity();
        if (activeTab === "temporalv2") renderTemporalV2();
        if (prev.view.activeTab === "graph" && activeTab !== "graph")
          _destroyCy();
      }
      if (next.view.glossaryOpen !== prev.view.glossaryOpen) syncGlossary();
      if (next.view.overviewMode !== prev.view.overviewMode &&
          activeTab === "signal") renderOverview();
      if ((next.view.graphLayout !== prev.view.graphLayout ||
           next.view.graphMinDegree !== prev.view.graphMinDegree ||
           next.view.graphGenotype !== prev.view.graphGenotype ||
           next.view.graphTimepoint !== prev.view.graphTimepoint ||
           next.view.graphTpdsMin !== prev.view.graphTpdsMin ||
           next.view.graphTopN !== prev.view.graphTopN) &&
          activeTab === "graph") {
        const genoSel = document.getElementById("graph-genotype");
        if (genoSel && genoSel.value !== next.view.graphGenotype)
          genoSel.value = next.view.graphGenotype;
        const tpSel = document.getElementById("graph-timepoint");
        if (tpSel && tpSel.value !== next.view.graphTimepoint)
          tpSel.value = next.view.graphTimepoint;
        renderGraph();
      }
      if (activeTab === "senders" &&
          (next.view.senderMatrixMode !== prev.view.senderMatrixMode ||
           next.view.senderMatrixAxis !== prev.view.senderMatrixAxis ||
           next.view.senderMatrixAnchor !== prev.view.senderMatrixAnchor)) {
        if (next.view.senderMatrixAxis !== prev.view.senderMatrixAxis) {
          _populateSenderAnchorSelect();
        } else {
          const anchorSel = document.getElementById("sm-anchor");
          if (anchorSel && anchorSel.value !== next.view.senderMatrixAnchor) {
            anchorSel.value = next.view.senderMatrixAnchor;
          }
        }
        renderSenderMatrix();
      }
      if ((next.view.temporalLevel !== prev.view.temporalLevel ||
           next.view.temporalMetric !== prev.view.temporalMetric ||
           next.view.temporalTissue !== prev.view.temporalTissue ||
           next.view.temporalScoreMin !== prev.view.temporalScoreMin) &&
          activeTab === "temporal") renderTemporal();
      if ((next.view.additivityLevel !== prev.view.additivityLevel ||
           next.view.additivityTimepoint !== prev.view.additivityTimepoint ||
           next.view.additivityScoreMin !== prev.view.additivityScoreMin) &&
          activeTab === "additivity") renderAdditivity();
      if (next.view.kinaseAuditTab !== prev.view.kinaseAuditTab &&
          activeTab === "kinase" && next.selection.kinase != null)
        renderActiveKinaseAuditTab(next.selection.kinase);
      if (next.view.pathwayScoreMin !== prev.view.pathwayScoreMin &&
          activeTab === "pathway") renderPathwayExplorer();
    }
  });
}

if (document.readyState === "loading")
  document.addEventListener("DOMContentLoaded", boot);
else boot();
</script>
</body>
</html>
"""


def write_html(payload: dict, json_str: str | None = None) -> dict:
    """Emit the unified viewer HTML at UNIFIED_VIEWER_DIR/index.html.

    Sibling dirs (edge_slices/, edge_summaries/) are written by
    build_edge_shards.py; this function only writes the HTML.
    """
    os.makedirs(UNIFIED_VIEWER_DIR, exist_ok=True)
    if json_str is None:
        json_str = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    # Escape </ so an embedded "</script>" in the JSON can't terminate the tag.
    safe = json_str.replace("</", "<\\/")
    html = HTML_TEMPLATE
    for sentinel, value in (
        ("__APP_COLOR__", config.DISEASE_COLORS["App"]),
        ("__TAU_COLOR__", config.DISEASE_COLORS["Tau"]),
        ("__APTT_COLOR__", config.DISEASE_COLORS["ApTt"]),
        ("__PAYLOAD_SENTINEL__", safe),
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

    # Edge-summary artifacts (Tier 1, embedded in payload)
    pk_summary_rows = pk_summary_bytes = 0
    pb_summary_rows = pb_summary_bytes = 0
    if not os.path.exists(PER_KINASE_SUMMARY):
        errors.append(f"per_kinase_summary missing: {PER_KINASE_SUMMARY}")
    else:
        pk_summary_rows = pq.ParquetFile(PER_KINASE_SUMMARY).metadata.num_rows
        pk_summary_bytes = os.path.getsize(PER_KINASE_SUMMARY)
    if not os.path.exists(PER_BACKBONE_SUMMARY):
        errors.append(f"per_backbone_summary missing: {PER_BACKBONE_SUMMARY}")
    else:
        pb_summary_rows = pq.ParquetFile(PER_BACKBONE_SUMMARY).metadata.num_rows
        pb_summary_bytes = os.path.getsize(PER_BACKBONE_SUMMARY)

    # Tier-2 slice directories (lazy-loaded; not embedded)
    n_kinase_slices = n_backbone_buckets = 0
    slices_bytes = 0
    if not os.path.isdir(EDGE_SLICES_KINASE_DIR):
        errors.append(f"kinase slice dir missing: {EDGE_SLICES_KINASE_DIR}")
    else:
        n_kinase_slices = sum(1 for f in os.listdir(EDGE_SLICES_KINASE_DIR)
                              if f.endswith(".parquet"))
        slices_bytes += sum(os.path.getsize(os.path.join(EDGE_SLICES_KINASE_DIR, f))
                            for f in os.listdir(EDGE_SLICES_KINASE_DIR)
                            if f.endswith(".parquet"))
    if not os.path.isdir(EDGE_SLICES_BACKBONE_DIR):
        errors.append(f"backbone slice dir missing: {EDGE_SLICES_BACKBONE_DIR}")
    else:
        n_backbone_buckets = sum(1 for f in os.listdir(EDGE_SLICES_BACKBONE_DIR)
                                 if f.endswith(".parquet"))
        slices_bytes += sum(os.path.getsize(os.path.join(EDGE_SLICES_BACKBONE_DIR, f))
                            for f in os.listdir(EDGE_SLICES_BACKBONE_DIR)
                            if f.endswith(".parquet"))

    # Structural
    if payload is not None:
        pk = payload["kinases"]
        pc_ = payload["celltypes"]
        pb = payload["backbones"]

        if len(pk["id"]) != n_kinases:
            errors.append(f"kinases rows {len(pk['id'])} != vocab {n_kinases}")
        if len(pc_["id"]) != n_celltypes:
            errors.append(f"celltypes rows {len(pc_['id'])} != vocab {n_celltypes}")
        # backbones[] only covers recurrence-aware backbones (superset of sig),
        # a strict subset of the edge parquet's 832K vocab. See docstring of
        # build_backbone_index() for why the two sets differ.
        if len(pb["id"]) > md["backbones_n"]:
            errors.append(
                f"backbones rows {len(pb['id'])} > edge vocab {md['backbones_n']}"
            )
        if len(set(pb["id"])) != len(pb["id"]):
            errors.append("duplicate backbone ids in payload")
        bb_index = build_backbone_index(data.backbone_recurrence)
        sig_bb_ids, _ = compute_sig_sets(data, bb_index)
        payload_bb_ids = set(pb["id"])
        missing_sig = [int(b) for b in sig_bb_ids if int(b) not in payload_bb_ids]
        if missing_sig:
            errors.append(
                f"{len(missing_sig)} sig backbone_id(s) absent from payload "
                f"backbones[] (first 3: {missing_sig[:3]})"
            )

        bad = [rid for rid in pb["receiver_id"]
               if rid < 0 or rid >= n_celltypes]
        if bad:
            errors.append(f"{len(bad)} orphan receiver_id(s) in backbones")

        n_bb = len(pb["id"])
        sig_mask_arr = np.asarray(pb["significant_both_mask"], dtype=np.int64)
        for ci, c in enumerate(md["contrasts"]):
            obs_key = f"observed_score_{c}"
            tpds_key = f"mean_tpds_{c}"
            if obs_key not in pb:
                errors.append(f"missing backbones[{obs_key}]")
                continue
            if len(pb[obs_key]) != n_bb:
                errors.append(
                    f"{obs_key} length {len(pb[obs_key])} != id length {n_bb}"
                )
                continue
            obs_notnull = np.array([v is not None for v in pb[obs_key]])
            tpds_notnull = np.array([v is not None for v in pb[tpds_key]])
            bad_rows = int(np.sum(obs_notnull & ~tpds_notnull))
            if bad_rows:
                errors.append(
                    f"{obs_key}: {bad_rows} rows have sig observed_score but "
                    f"no recurrence mean_tpds (sig should imply recurrence)"
                )
            sig_bit = ((sig_mask_arr >> ci) & 1).astype(bool)
            missing_obs = int(np.sum(sig_bit & ~obs_notnull))
            if missing_obs:
                errors.append(
                    f"{obs_key}: {missing_obs} sig-both rows missing "
                    f"observed_score"
                )

        # Tier-1 summary embedded in payload
        pks = payload.get("per_kinase_summary", {})
        if len(pks.get("kinase_id", [])) != pk_summary_rows:
            errors.append(
                f"per_kinase_summary rows in payload "
                f"{len(pks.get('kinase_id', []))} != parquet {pk_summary_rows}"
            )

        # Tier-2 slice reference
        esr = payload.get("edge_slice_ref", {})
        if esr.get("n_kinase_slices") != n_kinase_slices:
            errors.append(
                f"edge_slice_ref.n_kinase_slices={esr.get('n_kinase_slices')} "
                f"but {n_kinase_slices} parquet files on disk"
            )
        if esr.get("n_backbone_buckets") != n_backbone_buckets:
            errors.append(
                f"edge_slice_ref.n_backbone_buckets={esr.get('n_backbone_buckets')} "
                f"but {n_backbone_buckets} parquet files on disk"
            )

        # Every kinase_id referenced in per_kinase_summary must be in kinases[]
        summary_kids = set(pks.get("kinase_id", []))
        payload_kids = set(payload["kinases"]["id"])
        missing = summary_kids - payload_kids
        if missing:
            errors.append(
                f"{len(missing)} per_kinase_summary kinase_id(s) absent from "
                f"kinases[] (first 3: {sorted(missing)[:3]})"
            )

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
        f"- per_kinase_summary: {pk_summary_bytes/1e3:.1f} KB, {pk_summary_rows:,} rows",
        f"- per_backbone_summary: {pb_summary_bytes/1e3:.1f} KB, {pb_summary_rows:,} rows",
        f"- Edge slice shards total: {slices_bytes/1e6:.1f} MB "
        f"({n_kinase_slices} kinase + {n_backbone_buckets} backbone buckets)",
        "",
        "## Counts",
        "",
        f"- kinases: {n_kinases}",
        f"- celltypes: {n_celltypes}",
        f"- contrasts: {n_contrasts}",
        f"- backbones: {md['backbones_n']:,}",
        f"- full edges (Phase 1): {md['n_edges']:,}",
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

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", action="store_true", help="Print input counts (Unit 2.1 smoke test)")
    ap.add_argument("--sidecar", action="store_true", help="Stream-filter full edges to _sig sidecar parquet")
    ap.add_argument("--payload", action="store_true", help="Write JSON payload (requires sidecar)")
    ap.add_argument("--build", action="store_true", help="Sidecar then payload")
    ap.add_argument("--html", action="store_true", help="Write unified_viewer.html (requires payload)")
    ap.add_argument("--validate", action="store_true", help="Write Phase 2 validation report")
    args = ap.parse_args(argv)

    if not any([args.summary, args.sidecar, args.payload, args.build,
                args.html, args.validate]):
        args.build = True
        args.html = True

    data = load_all_data()

    if args.summary:
        print(json.dumps(data.summary(), indent=2))

    if args.sidecar:
        bb_index = build_backbone_index(data.backbone_recurrence)
        write_sig_sidecar(data, bb_index)

    payload = None
    json_str = None
    if args.payload or args.build:
        if not os.path.exists(PER_KINASE_SUMMARY):
            raise SystemExit(
                f"edge shards missing; run: "
                f"pixi run python code/integration/adapters/build_edge_shards.py"
            )
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

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

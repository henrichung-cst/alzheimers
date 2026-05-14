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
import kinase_normalize as kattr  # noqa: E402  (Stage 1 helpers + load_sample_mapping)

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
    EDGE_SLICES_INCYTR_PATHWAYS_DIR,
    INCYTR_FACTORIAL_INPUTS_DIR,
    INCYTR_FACTORIAL_OUTPUTS_DIR,
    MEASUREMENT_TRACE_DIR,
    MEASUREMENT_TRACE_INDEX,
    MEASUREMENT_TRACE_SCHEMA_VERSION,
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

    _decomp_path = os.path.join(
        config.REPO_ROOT, "outputs", "reports", "deconvolution",
        "per_animal", "kinase_enrichment_wmb.csv",
    )
    _decomp_cols = ["kinase", "wmb_class", "contrast", "NES", "FDR"]
    decomposition = pd.read_csv(
        _decomp_path, usecols=_decomp_cols,
    ) if os.path.exists(_decomp_path) else pd.DataFrame(columns=_decomp_cols)

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


_INCYTR_CONTRASTS = (
    "App_2mo", "App_4mo", "App_6mo",
    "Tau_2mo", "Tau_4mo", "Tau_6mo",
    "ApTt_2mo", "ApTt_4mo", "ApTt_6mo",
)

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
_INCYTR_SCORE_COLS = ("TPDS", "PPDS", "PhPDS_ps", "PhPDS_py", "multimodel_score")

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


def _write_incytr_pathways() -> dict | None:
    """Shard the long-form factorial output by (sender, receiver) — unfiltered.

    Reads `outputs/reports/incytr_factorial/receiver_cache/receiver=*/*.parquet`
    (WMB-labelled run, 4.76M rows, 164 cols, 9 contrasts × 528,790 paths) and
    emits one parquet per (sender, receiver) pair under
    `edge_slices/incytr_pathways/` plus the `incytr_pathways` payload block.
    No build-time significance gate is applied — the shards carry raw metrics
    so the UI can threshold live.

    `heatmap_counts` is a sender × receiver × contrast × pvalue × |PDS| grid
    (no significance gate at build time — client picks the threshold pair).
    Filter axes are pvalue (Wald t on the factorial OLS β for SigProb) and
    |PDS| (composite multimodel effect-size magnitude). The legacy
    `sigprob_max` mean-ratio filter was retired 2026-05-12.
    """
    import duckdb  # local to keep top-of-file imports lean

    cache_glob = os.path.join(
        INCYTR_FACTORIAL_OUTPUTS_DIR,
        "receiver_cache", "receiver=*", "*.parquet",
    )
    if not glob.glob(cache_glob):
        print(f"  (warn) receiver_cache empty under "
              f"{INCYTR_FACTORIAL_OUTPUTS_DIR}; skipping incytr_pathways",
              flush=True)
        return None

    pm_path = os.path.join(INCYTR_FACTORIAL_OUTPUTS_DIR, "pair_metadata.parquet")
    if not os.path.exists(pm_path):
        print(f"  (warn) {pm_path} missing; cannot resolve cell-type names",
              flush=True)
        return None
    pm = pd.read_parquet(pm_path, columns=["sender", "receiver"])
    senders_canonical = sorted(set(pm["sender"].tolist()))
    receivers_canonical = sorted(set(pm["receiver"].tolist()))
    sanitized_to_display = {_incytr_sanitize(n): n for n in receivers_canonical}

    con = duckdb.connect()
    con.execute("PRAGMA threads=8; PRAGMA memory_limit='12GB';")
    con.execute("SET temp_directory='/home/hchung/.cache/duckdb';")
    extra_score_select = ",\n          ".join(
        f"CAST({c} AS DOUBLE) AS {c}" for c in _INCYTR_SCORE_COLS
    )
    extra_fc_select = ",\n          ".join(
        f"CAST({c} AS DOUBLE) AS {c}" for c in _INCYTR_FC_COLS
    )
    label_select = ",\n          ".join(
        f'CAST("{src}" AS VARCHAR) AS {dst}'
        for src, dst in zip(_INCYTR_LABEL_SRC, _INCYTR_LABEL_COLS)
    )
    con.execute(f"""
        CREATE TEMP TABLE src AS
        SELECT
          sender, receiver, Path, Ligand, Receptor, EM, Target,
          contrast,
          CAST(pvalue   AS DOUBLE) AS pvalue,
          CAST(PDS      AS DOUBLE) AS PDS,
          {extra_score_select},
          {extra_fc_select},
          {label_select}
        FROM read_parquet('{cache_glob}', hive_partitioning = true)
    """)
    n_src = con.execute("SELECT COUNT(*) FROM src").fetchone()[0]
    print(f"  incytr_pathways: loaded receiver_cache ({n_src:,} rows)", flush=True)
    sender_to_idx = {s: i for i, s in enumerate(senders_canonical)}
    receiver_to_idx = {r: i for i, r in enumerate(receivers_canonical)}
    contrast_to_idx = {c: i for i, c in enumerate(_INCYTR_CONTRASTS)}
    n_s, n_r, n_c = len(senders_canonical), len(receivers_canonical), len(_INCYTR_CONTRASTS)

    # Heatmap counts: (sender × receiver × contrast × pvalue × |PDS|). Every
    # path is replicated across all 9 contrasts in the long-form schema, so
    # *unfiltered* counts are identical across contrasts — only pvalue / |PDS|
    # gating makes the per-contrast view differ. We pre-compute the same
    # 8 pvalue × 8 |PDS| cutoff grid used by the pathway-count cube so the
    # heatmap can re-threshold client-side without re-fetching shards.
    # NULL PDS rows count as |PDS|=0 (kept by the 0 threshold, dropped by all
    # others) — same convention as the sign bucket fall-through.
    n_thr_hm = len(_INCYTR_PATHWAY_PVALUES)
    n_ap_hm = len(_INCYTR_PATHWAY_ABS_PDS)
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
    grid = np.zeros((n_s, n_r, n_c, n_thr_hm, n_ap_hm), dtype=np.uint32)
    skipped_pairs: set[tuple[str, str]] = set()
    for row in hm_rows:
        s_raw, r_raw, c = row[0], row[1], row[2]
        r_disp = sanitized_to_display.get(r_raw, r_raw)
        if s_raw not in sender_to_idx or r_disp not in receiver_to_idx:
            skipped_pairs.add((s_raw, r_raw))
            continue
        if c not in contrast_to_idx:
            continue
        s_i = sender_to_idx[s_raw]
        r_i = receiver_to_idx[r_disp]
        c_i = contrast_to_idx[c]
        offset = 3
        for ip in range(n_thr_hm):
            for iap in range(n_ap_hm):
                grid[s_i, r_i, c_i, ip, iap] = int(row[offset])
                offset += 1
    # total_by_threshold becomes 2D [n_pvalues × n_abs_pds] so the client can
    # show "X paths across all contrasts at this (pvalue, |PDS|) pair".
    totals = np.zeros((n_thr_hm, n_ap_hm), dtype=np.uint64)
    for ip in range(n_thr_hm):
        for iap in range(n_ap_hm):
            totals[ip, iap] = int(grid[:, :, :, ip, iap].sum())
    heatmap_counts = {
        "thresholds": list(_INCYTR_PATHWAY_PVALUES),
        "abs_pds_thresholds": list(_INCYTR_PATHWAY_ABS_PDS),
        "shape": [n_s, n_r, n_c, n_thr_hm, n_ap_hm],
        "counts": grid.flatten().tolist(),
        "total_by_threshold": totals.tolist(),
    }
    if skipped_pairs:
        print(f"    (warn) heatmap_counts: skipped unknown pairs "
              f"{sorted(skipped_pairs)[:5]}{'...' if len(skipped_pairs) > 5 else ''}",
              flush=True)
    # Diagnostic: report total at the default (pvalue<0.05, |PDS|>=0) and at
    # the typical filter (pvalue<0.05, |PDS|>=0.01).
    ap_zero_idx = _INCYTR_PATHWAY_ABS_PDS.index(0.0)
    ap_001_idx = _INCYTR_PATHWAY_ABS_PDS.index(0.01)
    p_005_idx = _INCYTR_PATHWAY_PVALUES.index(0.05)
    print(f"    heatmap_counts: total at pvalue<0.05 & |PDS|>=0  = "
          f"{int(totals[p_005_idx, ap_zero_idx]):>9,}; "
          f"at pvalue<0.05 & |PDS|>=0.01 = {int(totals[p_005_idx, ap_001_idx]):>9,}", flush=True)

    # Pathway counts indexed by (contrast, sign(PDS), pvalue_threshold,
    # abs_pds_threshold) for the Temporal v2 pathway layer. Sign source is
    # the composite Pathway Disturbance Score (multimodel β across all omics
    # layers). Sign bucket: 0=down, 1=zero/NA, 2=up. NULL PDS falls through
    # to bucket 1 ("zero/NA"). |PDS| bucket: counts include rows with
    # |PDS| >= threshold (so threshold=0 keeps all rows; NULL PDS treated
    # as |PDS|=0 via COALESCE).
    n_thr = len(_INCYTR_PATHWAY_PVALUES)
    n_ap = len(_INCYTR_PATHWAY_ABS_PDS)
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
        # Cells laid out as (p_threshold × abs_pds_threshold) in row-major order.
        for ip in range(n_thr):
            for iap in range(n_ap):
                pathway_arr[c_idx, s_idx, ip, iap] = int(row[2 + ip * n_ap + iap])
    pathway_counts = {
        "thresholds": list(_INCYTR_PATHWAY_PVALUES),
        "abs_pds_thresholds": list(_INCYTR_PATHWAY_ABS_PDS),
        "contrasts": list(_INCYTR_CONTRASTS),
        "counts": pathway_arr.flatten().tolist(),
        "shape": [n_c, 3, n_thr, n_ap],
        "sign_source": "PDS",
    }
    # Diagnostic prints: total at pvalue<1.0, |PDS|>=0 (all rows) and a
    # meaningful gate (pvalue<0.05, |PDS|>=0.01). Also report the share of
    # paths landing in the "zero/NA" sign bucket (PDS = 0 or PDS IS NULL).
    p05_idx = _INCYTR_PATHWAY_PVALUES.index(0.05)
    ap01_idx = _INCYTR_PATHWAY_ABS_PDS.index(0.01)
    pds_na_pct = con.execute(
        "SELECT 100.0 * COUNT(*) FILTER (WHERE PDS IS NULL) / NULLIF(COUNT(*), 0) FROM src"
    ).fetchone()[0] or 0.0
    pds_zero_pct = con.execute(
        "SELECT 100.0 * COUNT(*) FILTER (WHERE PDS = 0) / NULLIF(COUNT(*), 0) FROM src"
    ).fetchone()[0] or 0.0
    print(f"    pathway_counts: {int(pathway_arr[:, :, -1, 0].sum()):>9,} rows "
          f"at pvalue<1.0 (all signs, no |PDS| gate); "
          f"{int(pathway_arr[:, :, p05_idx, ap01_idx].sum()):>6,} "
          f"at pvalue<0.05 & |PDS|>=0.01 · sign source = PDS "
          f"(NULL: {pds_na_pct:.1f}%, =0: {pds_zero_pct:.1f}% — these land in "
          f"the 'neither' bucket)", flush=True)

    # Reset / re-create the shard directory.
    os.makedirs(EDGE_SLICES_INCYTR_PATHWAYS_DIR, exist_ok=True)
    for f in os.listdir(EDGE_SLICES_INCYTR_PATHWAYS_DIR):
        if f.endswith(".parquet") or f == "index.json":
            os.remove(os.path.join(EDGE_SLICES_INCYTR_PATHWAYS_DIR, f))

    # Materialize all rows once, then groupby+write per pair. The per-node
    # seed-list labels (`<Node>.label` upstream → `<Node>_label` in the shard,
    # vocab = {"DEG","prG"}) returned with the seed-list rerun and surface in
    # the table as evidence-source badges next to each gene name.
    extra_cols = list(_INCYTR_SCORE_COLS) + list(_INCYTR_FC_COLS)
    extra_select = ", ".join(extra_cols)
    label_cols = list(_INCYTR_LABEL_COLS)
    label_select = ", ".join(label_cols)
    df = con.execute(f"""
        SELECT sender, receiver, Path, Ligand, Receptor, EM, Target,
               contrast, pvalue, PDS,
               {extra_select},
               {label_select}
        FROM src
    """).fetchdf()
    con.close()

    # Receivers arrive sanitized (hive partition); senders raw.
    df["receiver"] = df["receiver"].map(lambda r: sanitized_to_display.get(r, r))
    float_cols = ["pvalue", "PDS"] + extra_cols
    for col in float_cols:
        df[col] = df[col].astype("float32")
    # Labels are categoricals with a tiny fixed vocab — shrink to category for
    # ~1/4 the parquet bytes vs raw strings.
    for col in label_cols:
        df[col] = df[col].astype("category")

    shard_cols = [
        "Path", "Ligand", "Receptor", "EM", "Target",
        "contrast", "pvalue", "PDS",
        *extra_cols,
        *label_cols,
    ]
    present_pairs: list[list[str]] = []
    pair_row_counts: dict[str, int] = {}
    total_rows = 0
    max_shard_bytes = 0
    max_shard_name = ""
    for (s, r), g in df.groupby(["sender", "receiver"], sort=True):
        sub = g[shard_cols].sort_values(
            ["contrast", "pvalue"], kind="mergesort", na_position="last",
        )
        fname = f"{_incytr_sanitize(s)}__{_incytr_sanitize(r)}.parquet"
        path = os.path.join(EDGE_SLICES_INCYTR_PATHWAYS_DIR, fname)
        pq.write_table(
            pa.Table.from_pandas(sub, preserve_index=False),
            path, compression="zstd",
        )
        present_pairs.append([s, r])
        pair_row_counts[fname] = len(sub)
        total_rows += len(sub)
        sz = os.path.getsize(path)
        if sz > max_shard_bytes:
            max_shard_bytes = sz
            max_shard_name = fname

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
    print(f"  incytr_pathways: wrote {len(present_pairs)} shards "
          f"({total_rows:,} rows; {total_bytes/1e6:.1f} MB total; "
          f"max {max_shard_bytes/1e6:.2f} MB → {max_shard_name})",
          flush=True)

    return {
        "schema_version": SCHEMA_VERSION,
        "source": "receiver_cache/ (unfiltered)",
        "contrasts": list(_INCYTR_CONTRASTS),
        "senders": senders_canonical,
        "receivers": receivers_canonical,
        "empty_deg_celltypes": _read_empty_deg_celltypes(),
        "heatmap_counts": heatmap_counts,
        "pathway_counts": pathway_counts,
        "slice_index": index,
        "score_columns": list(_INCYTR_SCORE_COLS),
        "fc_nodes": list(_INCYTR_FC_NODES),
        "fc_metrics": list(_INCYTR_FC_METRICS),
        "label_columns": list(_INCYTR_LABEL_COLS),
        "label_nodes": list(_INCYTR_LABEL_NODES),
        "label_vocab": list(_INCYTR_LABEL_VOCAB),
    }


def _read_empty_deg_celltypes() -> list[str]:
    """Read the list of WMB classes with no DEGs from the upstream MANIFEST.

    `compute_seed_lists.R` writes `deg_cell_type_status` into
    `<INCYTR_FACTORIAL_OUTPUTS_DIR>/MANIFEST.json`. Cell types whose status
    indicates an empty DEG set are surfaced in the heatmap as hatched cells
    (visually distinct from "0 candidates pass the gate"). Returns `[]` if
    the manifest is absent or doesn't carry the field — the heatmap will
    just not render the hatched overlay in that case.
    """
    manifest_path = os.path.join(INCYTR_FACTORIAL_OUTPUTS_DIR, "MANIFEST.json")
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

    # Incytr pathway shards: one parquet per (sender, receiver), backing the
    # significant-pathway heatmap + table tabs.
    incytr_pathways_block = _write_incytr_pathways()

    meta = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "contrasts": contrasts,
        "diseaseGroups": list(config.DISEASE_GROUPS),
        "timepoints": list(config.TIMEPOINTS),
        "diseaseColors": dict(config.DISEASE_COLORS),
        "familyMap": fam,
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
    # Escape </ so an embedded "</script>" in the JSON can't terminate the tag.
    safe = json_str.replace("</", "<\\/")
    html = _render_template()
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

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", action="store_true", help="Print input counts (Unit 2.1 smoke test)")
    ap.add_argument("--payload", action="store_true", help="Write JSON payload")
    ap.add_argument("--html", action="store_true", help="Write unified_viewer.html (requires payload)")
    ap.add_argument("--validate", action="store_true", help="Write Phase 2 validation report")
    args = ap.parse_args(argv)

    if not any([args.summary, args.payload, args.html, args.validate]):
        args.payload = True
        args.html = True

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

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

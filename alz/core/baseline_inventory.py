"""Phase-0 baseline inventory generator.

Read-only. Never creates, modifies, or overwrites any canonical output.

CLI:
    python -m alz.core.baseline_inventory --cohort song --output path/to/out.json
    python -m alz.core.baseline_inventory --all --output-dir outputs/reports/refactor_audit/phase_0/

Memory-safety rules (shared box):
- Never json.load large files. Large JSON/payload: sha256 + size + mtime ONLY.
- Parquet: use pyarrow.parquet.ParquetFile(path).metadata for row count,
  .schema_arrow for column names. Never read the table.
- CSV: read header only for column names; count rows via streamed line count.
  If file > SIZE_LIMIT_CSV_PARSE, only stream-count lines, do not read into pandas.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

try:
    import pyarrow.parquet as pq  # type: ignore
except ImportError:
    pq = None  # type: ignore

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Files larger than this (bytes) will not be column-parsed — only streamed line count.
SIZE_LIMIT_CSV_PARSE = 50 * 1024 * 1024  # 50 MB
# Files larger than this (bytes) — hash + size + mtime ONLY, no row/col parse.
SIZE_LIMIT_HASH_ONLY = 50 * 1024 * 1024  # same threshold; JSON payloads are 100+ MB

# ---------------------------------------------------------------------------
# Parity tolerance constants (Phase-0 policy; ratified 2026-06-17)
# ---------------------------------------------------------------------------
PARITY_RTOL: float = 1e-6
PARITY_ATOL: float = 1e-9


# ---------------------------------------------------------------------------
# Canonical output roots
# ---------------------------------------------------------------------------
OUTPUT_ROOTS: dict[str, list[str]] = {
    "song": [
        "outputs/reports/kinase_attribution",
        "outputs/reports/decomposition/levy_t5",
        "outputs/reports/attribution_recovery",
    ],
    "mukesh": [
        "outputs/reports/kinase_attribution_human",
    ],
    "tcells": [
        "outputs/reports/kinase_attribution_tcells",
    ],
    "fivexfad": [
        "outputs/reports/kinase_attribution_5xfad",
    ],
    "incytr": [
        "outputs/reports/incytr_pair_mode",
        "outputs/reports/incytr_pair_mode_tcells",
    ],
    "viewer": [
        "outputs/reports/unified_viewer",
        "outputs/reports/tcell_viewer",
    ],
}


# ---------------------------------------------------------------------------
# Protected file specs: path (relative to PROJECT_ROOT), notes, key columns
# ---------------------------------------------------------------------------

def _p(rel: str, notes: str = "", key_columns: list[str] | None = None) -> dict:
    return {"rel_path": rel, "notes": notes, "key_columns": key_columns or []}


def _build_protected_files() -> dict[str, list[dict]]:
    """Return protected file specs grouped by cohort.

    Surface tightened 2026-06-17 per the protected-surface audit
    (docs/audits/cohort_abstraction_refactor/phase_0_protected_surface_audit.md):
    68 derived/transform/orphan entries de-protected (kept on disk, no parity
    contract), 18 actively-read 5xFAD files added. Net 215 -> 165. De-protected
    files carry an inline "# DE-PROTECTED:" note at their former site. Ceasing to
    *produce* the never-read sidecars is deferred to the Phase 2/3 producer
    refactor (5xFAD exempt — unrecoverable behind the .sne hold).
    """
    root = PROJECT_ROOT

    # ---- helpers to enumerate files on disk ----

    def _glob_exists(pattern: str) -> list[str]:
        """Return sorted list of rel paths matching glob."""
        return sorted(
            str(p.relative_to(root))
            for p in root.glob(pattern)
            if p.is_file()
        )

    # ---- SONG ----
    song_mea_root = "outputs/reports/kinase_attribution"
    song_decomp_root = "outputs/reports/decomposition/levy_t5"
    song_recovery_root = "outputs/reports/attribution_recovery"

    song_files: list[dict] = []
    # MEA long
    for name, notes in [
        ("mea_raw_phospho.csv", "MEA long, stoichiometry track"),
        ("mea_raw_phospho_pY.csv", "MEA long, pY track"),
        ("mea_stoichiometry.csv", "MEA long, stoichiometry track (stoich)"),
        ("mea_stoichiometry_pY.csv", "MEA long, pY track (stoich)"),
        ("mea_substrate_sets.csv", "substrate sets"),
        ("mea_substrate_sets_pY.csv", "substrate sets pY"),
    ]:
        song_files.append(_p(
            f"{song_mea_root}/{name}", notes,
            key_columns=["kinase", "contrast", "residue_type", "track"],
        ))
    # audit
    for name, notes in [
        ("mea_global_shift.csv", "audit: global shift"),
        ("mea_global_shift_pY.csv", "audit: global shift pY"),
        ("winsorized_sites.csv", "audit: winsorized sites"),
        ("winsorized_sites_pY.csv", "audit: winsorized sites pY"),
    ]:
        song_files.append(_p(f"{song_mea_root}/{name}", notes, key_columns=["kinase"]))
    # OLS/effect
    for name, notes in [
        ("site_level_ols.csv", "OLS/effect table (no NES/FDR for Song)"),
        ("site_level_ols_pY.csv", "OLS/effect table pY"),
    ]:
        song_files.append(_p(
            f"{song_mea_root}/{name}", notes,
            key_columns=["site_id", "gene_symbol"],
        ))
    # attribution
    for name, notes in [
        ("unified_attribution.csv", "attribution table"),
        ("unified_attribution_full.csv", "attribution table full"),
    ]:
        song_files.append(_p(
            f"{song_mea_root}/{name}", notes,
            key_columns=["kinase", "contrast", "cell_type"],
        ))
    # normalized matrices
    for name in [
        "stoichiometry_matrix.csv", "stoichiometry_matrix_pY.csv",
        "raw_phospho_normalized.csv", "raw_phospho_normalized_pY.csv",
        "total_proteome_normalized.csv",
        # DE-PROTECTED: total_proteome_normalized_pY.csv — duplicate of the
        # track-invariant ST total proteome; no code consumer.
    ]:
        song_files.append(_p(f"{song_mea_root}/{name}", "normalized matrix", key_columns=["kinase"]))

    # Song decomposition subtree
    # NOTE: the per-cluster audit sidecars (mea_substrate_sets/mea_global_shift/
    # winsorized_sites _per_cluster, ST+pY = 6 files), coverage_report.csv, and
    # proportions_provenance.csv are DE-PROTECTED — written by enrich_celltype.py /
    # snrna_proportions.py with no downstream reader (audit per 2026-06-17).
    for name, notes, keys in [
        ("mea_per_cluster.parquet", "decomp MEA per cluster", ["kinase", "cluster", "contrast"]),
        ("mea_per_cluster_pY.parquet", "decomp MEA per cluster pY", ["kinase", "cluster", "contrast"]),
        ("site_level_ols_per_cluster.parquet", "decomp OLS per cluster", ["site_id", "cluster"]),
        ("site_level_ols_per_cluster_pY.parquet", "decomp OLS per cluster pY", ["site_id", "cluster"]),
        ("proportions.parquet", "cell-type proportions", []),
        ("verification.json", "decomp verification (hard checks)", []),
        ("phospho_per_cluster.parquet", "phospho per cluster (provenance)", []),
        ("phospho_per_cluster_pY.parquet", "phospho per cluster pY", []),
        ("protein_per_cluster.parquet", "protein per cluster", []),
        ("transcript_per_cluster.parquet", "transcript per cluster", []),
    ]:
        song_files.append(_p(f"{song_decomp_root}/{name}", notes, key_columns=keys))

    # Song recovery subtree
    for name, notes, keys in [
        ("celltype_evidence_table.csv", "recovery: celltype evidence", ["kinase", "cell_type", "contrast"]),
        ("kinase_activity_matrix.csv", "recovery: kinase activity matrix", ["kinase", "residue_type"]),
        ("kinase_hypothesis_table.csv", "recovery: kinase hypothesis table", ["kinase", "residue_type"]),
    ]:
        song_files.append(_p(f"{song_recovery_root}/{name}", notes, key_columns=keys))

    # ---- MUKESH ----
    mukesh_root = "outputs/reports/kinase_attribution_human"
    mukesh_perdonor = f"{mukesh_root}/perdonor"

    mukesh_files: list[dict] = []
    # root normalized matrices
    for name in [
        "raw_phospho_normalized.csv", "raw_phospho_normalized_pY.csv",
        "stoichiometry_matrix.csv", "stoichiometry_matrix_pY.csv",
        "celltype_specificity.csv",
        # DE-PROTECTED: raw_phospho_normalized_all.csv, stoichiometry_matrix_all.csv
        # — write-only `_concat` outputs of mukesh.py with no downstream reader.
    ]:
        mukesh_files.append(_p(f"{mukesh_root}/{name}", "normalized matrix / specificity", key_columns=["kinase"]))
    # perdonor NES/FDR matrices
    for name, notes in [
        ("kinase_donor_nes.csv", "NES matrix per donor (stoich)"),
        ("kinase_donor_nes_pY.csv", "NES matrix per donor (pY)"),
        ("kinase_donor_nes_raw.csv", "NES matrix per donor (raw)"),
        ("kinase_donor_nes_raw_pY.csv", "NES matrix per donor (raw pY)"),
        ("kinase_donor_fdr.csv", "FDR matrix per donor (stoich)"),
        ("kinase_donor_fdr_pY.csv", "FDR matrix per donor (pY)"),
        ("kinase_donor_fdr_raw.csv", "FDR matrix per donor (raw)"),
        ("kinase_donor_fdr_raw_pY.csv", "FDR matrix per donor (raw pY)"),
    ]:
        mukesh_files.append(_p(f"{mukesh_perdonor}/{name}", notes, key_columns=["kinase"]))
    # perdonor MEA long
    for name, notes in [
        ("mea_perdonor.csv", "MEA long per donor (stoich)"),
        ("mea_perdonor_pY.csv", "MEA long per donor (pY)"),
        ("mea_perdonor_raw.csv", "MEA long per donor (raw)"),
        ("mea_perdonor_raw_pY.csv", "MEA long per donor (raw pY)"),
    ]:
        mukesh_files.append(_p(
            f"{mukesh_perdonor}/{name}", notes,
            key_columns=["kinase", "contrast", "residue_type", "track"],
        ))
    # perdonor audit — only the ""/_pY track variants are read by the viewer's
    # _human_track_load; the _raw-infix sidecars are DE-PROTECTED (written by
    # mukesh_perdonor.py, never loaded).
    for name in [
        "mea_global_shift.csv", "mea_global_shift_pY.csv",
        "winsorized_sites.csv", "winsorized_sites_pY.csv",
    ]:
        mukesh_files.append(_p(f"{mukesh_perdonor}/{name}", "audit", key_columns=["kinase"]))
    # substrate sets (DE-PROTECTED: _raw, _raw_pY)
    for name in [
        "mea_substrate_sets.csv", "mea_substrate_sets_pY.csv",
    ]:
        mukesh_files.append(_p(f"{mukesh_perdonor}/{name}", "substrate sets"))
    # recurrence (DE-PROTECTED: recurrence_raw, _raw_pY, recurrence_ctrl_raw, _raw_pY)
    for name, notes in [
        ("recurrence.csv", "recurrence (stoich)"),
        ("recurrence_pY.csv", "recurrence (pY)"),
        ("recurrence_ctrl.csv", "recurrence ctrl (stoich)"),
        ("recurrence_ctrl_pY.csv", "recurrence ctrl (pY)"),
    ]:
        mukesh_files.append(_p(f"{mukesh_perdonor}/{name}", notes, key_columns=["kinase"]))

    # ---- T-CELLS ----
    tcell_root = "outputs/reports/kinase_attribution_tcells"
    d1 = f"{tcell_root}/donor1"
    d1_mea = f"{d1}/mea"
    d2 = f"{tcell_root}/donor2"

    tcell_files: list[dict] = []
    # donor1 normalized
    for name in [
        "raw_phospho_normalized.csv", "raw_phospho_normalized_pY.csv",
        "stoichiometry_matrix.csv", "stoichiometry_matrix_pY.csv",
        "total_proteome_normalized.csv",
        "tcell_concordance.csv", "tcell_enrichment.csv",
        "unified_attribution_tcells.csv",
    ]:
        tcell_files.append(_p(f"{d1}/{name}", "donor1 normalized/attribution"))
    # donor1/mea MEA long (DE-PROTECTED: _raw, _raw_pY — the tcell viewer shims
    # out the raw-phospho track; _KINASE_AUDIT_FILES wires only ""/_pY)
    for name, notes in [
        ("mea_timecourse.csv", "MEA long per timepoint (stoich)"),
        ("mea_timecourse_pY.csv", "MEA long per timepoint (pY)"),
    ]:
        tcell_files.append(_p(
            f"{d1_mea}/{name}", notes,
            key_columns=["kinase", "contrast", "residue_type", "track"],
        ))
    # donor1/mea NES/FDR
    for name, notes in [
        ("kinase_timepoint_nes.csv", "NES per timepoint (stoich)"),
        ("kinase_timepoint_nes_pY.csv", "NES per timepoint (pY)"),
        ("kinase_timepoint_nes_raw.csv", "NES per timepoint (raw)"),
        ("kinase_timepoint_nes_raw_pY.csv", "NES per timepoint (raw pY)"),
        ("kinase_timepoint_fdr.csv", "FDR per timepoint (stoich)"),
        ("kinase_timepoint_fdr_pY.csv", "FDR per timepoint (pY)"),
        ("kinase_timepoint_fdr_raw.csv", "FDR per timepoint (raw)"),
        ("kinase_timepoint_fdr_raw_pY.csv", "FDR per timepoint (raw pY)"),
    ]:
        tcell_files.append(_p(f"{d1_mea}/{name}", notes, key_columns=["kinase"]))
    # donor1/mea audit + substrate sets + recurrence (DE-PROTECTED: all _raw
    # variants — raw-phospho track shimmed out; and recurrence_pY — the audit_specs
    # loop emits only base recurrence.csv)
    for name in [
        "mea_global_shift.csv", "mea_global_shift_pY.csv",
        "winsorized_sites.csv", "winsorized_sites_pY.csv",
    ]:
        tcell_files.append(_p(f"{d1_mea}/{name}", "audit"))
    for name in [
        "mea_substrate_sets.csv", "mea_substrate_sets_pY.csv",
    ]:
        tcell_files.append(_p(f"{d1_mea}/{name}", "substrate sets"))
    for name in [
        "recurrence.csv",
    ]:
        tcell_files.append(_p(f"{d1_mea}/{name}", "recurrence", key_columns=["kinase"]))
    # donor1/mea manifest
    tcell_files.append(_p(f"{d1_mea}/mea_manifest.json", "donor1 MEA manifest"))

    # donor2 — PARTIAL BY DESIGN (pY normalized only; no full long table).
    # The pY matrices stay protected: their presence is what makes the MEA skip
    # reason "no_motif" rather than "matrix_absent" in the manifest.
    # DE-PROTECTED: donor2/total_proteome_normalized.csv — no reader
    # (_build_tcell_measurement_trace hardcodes donor1).
    for name in [
        "raw_phospho_normalized_pY.csv",
        "stoichiometry_matrix_pY.csv",
    ]:
        tcell_files.append(_p(
            f"{d2}/{name}",
            "donor2 pY-only normalized (partial by design)",
        ))
    tcell_files.append(_p(
        f"{d2}/mea/mea_manifest.json",
        "donor2 MEA manifest (pY only)",
    ))
    # donor2 expected-but-absent full long table — recorded as expected missing
    tcell_files.append(_p(
        f"{d2}/mea/mea_timecourse.csv",
        "absent_by_design: donor2 pY-only, no IMAC",
    ))

    # ---- 5xFAD ----
    fx_root = "outputs/reports/kinase_attribution_5xfad"
    fx_ct = f"{fx_root}/celltype_mea"

    fivexfad_files: list[dict] = []
    for region in ["cortex", "hippocampus"]:
        for mod in ["st", "py"]:
            prefix = f"{fx_root}/{region}_{mod}"
            for suffix, notes, keys in [
                ("_mea_raw_phospho.csv", f"MEA long raw phospho {region}/{mod}", ["kinase", "contrast"]),
                ("_mea_stoichiometry.csv", f"MEA long stoichiometry {region}/{mod}", ["kinase", "contrast"]),
                ("_mea_substrate_sets.csv", f"substrate sets {region}/{mod}", []),
                ("_mea_global_shift.csv", f"audit global shift {region}/{mod}", []),
                ("_winsorized_sites.csv", f"audit winsorized sites {region}/{mod}", []),
                ("_site_level_ols.csv", f"OLS/effect table (no NES/FDR) {region}/{mod}", ["site_id"]),
                # ADDED 2026-06-17: actively read by the viewer's supporting_5xfad
                # slice + detail-shard build, previously unprotected (audit gap).
                ("_contrast_qc.csv", f"contrast QC {region}/{mod}", ["contrast"]),
                ("_raw_phospho_normalized.csv", f"normalized raw phospho {region}/{mod}", ["site_id"]),
                ("_stoichiometry_matrix.csv", f"stoichiometry matrix {region}/{mod}", ["site_id"]),
                ("_matched_total_protein.csv", f"matched total protein {region}/{mod}", ["site_id"]),
            ]:
                fivexfad_files.append(_p(f"{prefix}{suffix}", notes, key_columns=keys))
        # region-level proteome
        fivexfad_files.append(_p(
            f"{fx_root}/{region}_total_proteome_normalized.csv",
            f"normalized total proteome {region}",
        ))
    # ADDED 2026-06-17: sample manifest — read for the sample-counts table.
    fivexfad_files.append(_p(
        f"{fx_root}/sample_manifest.csv",
        "5xFAD sample manifest (viewer sample counts)",
    ))

    fivexfad_files.append(_p(
        f"{fx_root}/fivexfad_snrna_attribution.csv",
        "snRNA attribution table",
        key_columns=["kinase", "cell_type", "tissue", "age_months"],
    ))
    fivexfad_files.append(_p(
        f"{fx_root}/fivexfad_snrna_cell_counts.csv",
        "snRNA cell counts",
    ))

    # celltype_mea (DE-PROTECTED: fivexfad_celltype_mea_global_shift.csv,
    # fivexfad_celltype_winsorized_sites.csv — source_files list only, never
    # parsed by the viewer. ADDED: fivexfad_snrna_pseudobulk_counts.csv — the
    # preferred cell-count source, read before the fallback.)
    for name, notes, keys in [
        # key_columns corrected per Phase-1 DEV-03: these parquets key on
        # cell_type, not cluster (verified against schema 2026-06-17).
        ("fivexfad_celltype_mea.parquet", "celltype MEA parquet (long)", ["kinase", "contrast", "residue_type", "track", "cell_type"]),
        ("fivexfad_celltype_site_level_ols.parquet", "celltype site-level OLS parquet", ["site_id", "cell_type", "contrast", "tissue", "track"]),
        ("fivexfad_celltype_substrate_sets.csv", "celltype substrate sets", []),
        ("fivexfad_snrna_pseudobulk_counts.csv", "snRNA pseudobulk cell counts (preferred source)", []),
    ]:
        fivexfad_files.append(_p(f"{fx_ct}/{name}", notes, key_columns=keys))

    # ---- INCYTR ----
    incytr_files: list[dict] = []
    # AD wide parquets
    for p in (root / "outputs/reports/incytr_pair_mode/wide").glob("*.parquet"):
        incytr_files.append(_p(
            str(p.relative_to(root)),
            "incytr wide AD contrast parquet",
            key_columns=["Sender", "Receiver", "Pathway"],
        ))
    # DE-PROTECTED: receiver_cache/receiver=*/data_0.parquet — a lossless
    # repartition-by-receiver of wide/, regenerable via pair_to_receiver_cache.py.
    # Protect the source (wide/), not the reshape.
    # tcell incytr wide
    for donor in ["donor1", "donor2"]:
        wide_dir = root / f"outputs/reports/incytr_pair_mode_tcells/{donor}/wide"
        if wide_dir.exists():
            for p in sorted(wide_dir.glob("*.parquet")):
                incytr_files.append(_p(
                    str(p.relative_to(root)),
                    f"incytr tcell wide {donor}",
                    key_columns=["Sender", "Receiver", "Pathway"],
                ))

    # ---- VIEWER ----
    viewer_files: list[dict] = []
    for viewer_dir, label in [
        ("outputs/reports/unified_viewer", "unified"),
        ("outputs/reports/tcell_viewer", "tcell"),
    ]:
        vp = root / viewer_dir
        for fname in ["index.html"]:
            viewer_files.append(_p(f"{viewer_dir}/{fname}", f"{label} viewer HTML"))
        # Only the unified uncompressed payload carries an independent parity
        # signal. DE-PROTECTED: unified .payload.json.gz (deterministic gzip of
        # the .json), and both tcell payload files (.json + .gz) — the tcell
        # payload is inlined in tcell_viewer/index.html, which is hashed above.
        if label == "unified":
            viewer_files.append(_p(
                f"{viewer_dir}/{label}_viewer.payload.json",
                f"{label} viewer payload (hash+size only — >50 MB)",
            ))
        # edge slice index.json files (one per family)
        edge_dir = vp / "edge_slices"
        if edge_dir.exists():
            for idx in sorted(edge_dir.rglob("index.json")):
                viewer_files.append(_p(
                    str(idx.relative_to(root)),
                    f"{label} edge slice index",
                ))

    return {
        "song": song_files,
        "mukesh": mukesh_files,
        "tcells": tcell_files,
        "fivexfad": fivexfad_files,
        "incytr": incytr_files,
        "viewer": viewer_files,
    }


# ---------------------------------------------------------------------------
# File inventory helpers
# ---------------------------------------------------------------------------

def _sha256_stream(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _csv_header(path: Path) -> list[str]:
    """Read only the header row of a CSV."""
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        try:
            return next(reader)
        except StopIteration:
            return []


def _csv_row_count(path: Path) -> int:
    """Count data rows (excluding header) via streamed line count."""
    count = 0
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            count += chunk.count(b"\n")
    # subtract 1 for header; handle trailing newline
    return max(0, count - 1)


def _classify_columns(
    col_names: list[str],
) -> tuple[list[str], list[str]]:
    """Heuristic split into numeric vs categorical column name lists.

    We classify by naming convention only — no data reads.
    """
    numeric_suffixes = (
        "_nes", "_fdr", "_pval", "_lfc", "_es", "p-value", "_bytes",
        "n_obs", "n_donors", "n_timepoints", "n_sig", "median_",
        "_fold", "_tau", "_score", "_size", "_count", "_fraction",
        "_log2", "_mean", "_std", "_pct",
    )
    numeric_names = {
        "ES", "NES", "p-value", "FDR", "file_size_bytes", "row_count",
    }
    numeric_cols = []
    categorical_cols = []
    for col in col_names:
        col_lower = col.lower()
        if col in numeric_names or any(col_lower.endswith(s) or col_lower.startswith(s)
                                       for s in numeric_suffixes):
            numeric_cols.append(col)
        else:
            categorical_cols.append(col)
    return numeric_cols, categorical_cols


def _parquet_info(path: Path) -> tuple[int, list[str]]:
    """Return (row_count, column_names) from parquet metadata without loading data."""
    if pq is None:
        return -1, []
    pf = pq.ParquetFile(str(path))
    meta = pf.metadata
    row_count = meta.num_rows
    schema = pf.schema_arrow
    col_names = [schema.field(i).name for i in range(len(schema))]
    return row_count, col_names


# Producer commands: known mappings from output path fragment to command.
_PRODUCER_MAP: list[tuple[str, str]] = [
    ("kinase_attribution/", "pixi run song (or python alz/bulk_mea/song.py)"),
    ("kinase_attribution_human/", "pixi run mukesh (or python alz/bulk_mea/mukesh_perdonor.py)"),
    ("kinase_attribution_tcells/", "pixi run ingest-tcells / pixi run tcells"),
    ("kinase_attribution_5xfad/", "pixi run 5xfad"),
    ("decomposition/levy_t5/", "pixi run decompose (or python alz/decomposition_mea/...)"),
    ("attribution_recovery/", "pixi run recover (or python alz/integration/recover.py)"),
    ("incytr_pair_mode/", "pixi run incytr-pair (or Rscript alz/incytr_pair/incytr_commandline.R)"),
    ("incytr_pair_mode_tcells/", "pixi run tcells-incytr"),
    ("unified_viewer/", "pixi run viewer (or python alz/build_unified_viewer.py)"),
    ("tcell_viewer/", "pixi run tcell-viewer (or python alz/tcell_viewer/build_tcell_viewer.py)"),
]


def _producer_command(rel_path: str) -> str:
    for fragment, cmd in _PRODUCER_MAP:
        if fragment in rel_path:
            return cmd
    return ""


# ---------------------------------------------------------------------------
# Core inventory function for a single file
# ---------------------------------------------------------------------------

def inventory_file(spec: dict) -> dict[str, Any]:
    rel = spec["rel_path"]
    path = PROJECT_ROOT / rel
    notes = spec.get("notes", "")
    key_columns = spec.get("key_columns", [])

    record: dict[str, Any] = {
        "path": rel,
        "exists": path.exists(),
        "file_size_bytes": None,
        "sha256": None,
        "mtime": None,
        "row_count": None,
        "column_names": None,
        "key_columns": key_columns,
        "key_unique": None,
        "numeric_columns": None,
        "categorical_columns": None,
        "producer_command": _producer_command(rel),
        "notes": notes,
    }

    if not path.exists():
        return record

    stat = path.stat()
    size = stat.st_size
    mtime = stat.st_mtime

    record["file_size_bytes"] = size
    record["mtime"] = mtime
    record["sha256"] = _sha256_stream(path)

    suffix = path.suffix.lower()

    if suffix == ".json":
        # JSON: hash + size + mtime only (may be large payload; never parse)
        return record

    if suffix == ".html":
        # HTML viewer: hash + size only
        return record

    if suffix == ".gz":
        # Compressed payload: hash + size only
        return record

    if suffix == ".parquet":
        if pq is not None:
            try:
                row_count, col_names = _parquet_info(path)
                record["row_count"] = row_count
                record["column_names"] = col_names
                numeric_cols, categorical_cols = _classify_columns(col_names)
                record["numeric_columns"] = numeric_cols
                record["categorical_columns"] = categorical_cols
            except Exception as exc:
                record["notes"] = (notes + f" | parquet read error: {exc}").strip(" |")
        else:
            record["notes"] = (notes + " | pyarrow not available").strip(" |")
        return record

    if suffix == ".csv":
        # Always do a streamed row count
        try:
            row_count = _csv_row_count(path)
            record["row_count"] = row_count
        except Exception as exc:
            record["notes"] = (notes + f" | row count error: {exc}").strip(" |")

        # Column parsing: only if under size limit
        if size <= SIZE_LIMIT_CSV_PARSE:
            try:
                col_names = _csv_header(path)
                record["column_names"] = col_names
                numeric_cols, categorical_cols = _classify_columns(col_names)
                record["numeric_columns"] = numeric_cols
                record["categorical_columns"] = categorical_cols
            except Exception as exc:
                record["notes"] = (notes + f" | header read error: {exc}").strip(" |")
        else:
            record["notes"] = (notes + " | >50 MB: column names skipped").strip(" |")

        return record

    # Unknown extension: hash + size only
    return record


# ---------------------------------------------------------------------------
# Cohort inventory
# ---------------------------------------------------------------------------

def run_inventory(cohort: str) -> list[dict[str, Any]]:
    protected = _build_protected_files()
    specs = protected.get(cohort)
    if specs is None:
        raise ValueError(f"Unknown cohort: {cohort!r}. Valid: {list(protected)}")
    return [inventory_file(spec) for spec in specs]


def write_json(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, sort_keys=False)


def write_csv(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        path.write_text("")
        return
    # flatten: column_names, key_columns, numeric_columns, categorical_columns as |-joined strings
    flat_records = []
    for r in records:
        row = dict(r)
        for key in ("column_names", "key_columns", "numeric_columns", "categorical_columns"):
            val = row.get(key)
            if isinstance(val, list):
                row[key] = "|".join(val)
        flat_records.append(row)
    fieldnames = list(flat_records[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flat_records)


# ---------------------------------------------------------------------------
# JSON-serializable output roots
# ---------------------------------------------------------------------------

def build_output_roots() -> dict:
    return {
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "project_root": str(PROJECT_ROOT),
        "cohorts": {
            cohort: [str(PROJECT_ROOT / r) for r in roots]
            for cohort, roots in OUTPUT_ROOTS.items()
        },
        "roots_relative": OUTPUT_ROOTS,
    }


def build_protected_files_manifest() -> dict:
    protected = _build_protected_files()
    return {
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "parity_policy": {
            "structural_fields": "exact",
            "numeric_fields": f"numpy.isclose(rtol={PARITY_RTOL}, atol={PARITY_ATOL})",
            "nan_positions": "exact",
            "ratified": "2026-06-17",
        },
        "cohorts": {
            cohort: [
                {"path": s["rel_path"], "notes": s["notes"], "key_columns": s["key_columns"]}
                for s in specs
            ]
            for cohort, specs in protected.items()
        },
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

COHORTS = ["song", "mukesh", "tcells", "fivexfad", "incytr", "viewer"]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Phase-0 baseline inventory generator (read-only)."
    )
    parser.add_argument("--cohort", choices=COHORTS, help="Inventory a single cohort.")
    parser.add_argument("--all", action="store_true", help="Inventory all cohorts.")
    parser.add_argument("--output", help="Output JSON path (single cohort mode).")
    parser.add_argument(
        "--output-dir",
        default="outputs/reports/refactor_audit/phase_0",
        help="Output directory (--all mode). Default: outputs/reports/refactor_audit/phase_0",
    )
    parser.add_argument(
        "--roots", action="store_true",
        help="Write output_roots.json and protected_files.json, then exit.",
    )
    args = parser.parse_args(argv)

    out_dir = PROJECT_ROOT / args.output_dir

    if args.roots:
        roots_path = out_dir / "output_roots.json"
        protected_path = out_dir / "protected_files.json"
        write_json([build_output_roots()], roots_path)  # wrap in list for consistency
        # write as plain dict (not list)
        roots_path.parent.mkdir(parents=True, exist_ok=True)
        with open(roots_path, "w", encoding="utf-8") as f:
            json.dump(build_output_roots(), f, indent=2)
        with open(protected_path, "w", encoding="utf-8") as f:
            json.dump(build_protected_files_manifest(), f, indent=2)
        print(f"Wrote {roots_path}")
        print(f"Wrote {protected_path}")
        return

    cohorts_to_run: list[str] = []
    if args.all:
        cohorts_to_run = COHORTS
    elif args.cohort:
        cohorts_to_run = [args.cohort]
    else:
        parser.error("Specify --cohort <name> or --all (or --roots).")

    for cohort in cohorts_to_run:
        records = run_inventory(cohort)
        present = sum(1 for r in records if r["exists"])
        absent = len(records) - present
        absent_by_design = sum(
            1 for r in records
            if not r["exists"] and "absent_by_design" in r.get("notes", "")
        )
        print(
            f"{cohort}: {len(records)} protected files | "
            f"{present} present | {absent} absent "
            f"({absent_by_design} absent_by_design)"
        )

        if args.cohort and args.output:
            json_path = Path(args.output)
        else:
            json_path = out_dir / f"{cohort}_inventory.json"
        csv_path = json_path.with_suffix(".csv")

        write_json(records, json_path)
        write_csv(records, csv_path)
        print(f"  -> {json_path}")
        print(f"  -> {csv_path}")


if __name__ == "__main__":
    main()

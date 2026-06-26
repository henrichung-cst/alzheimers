"""Audit manifest builders for the T-cell viewer."""

from __future__ import annotations

import json
import os
import shutil

import pandas as pd

from alz.shared import config  # noqa: E402
from alz.viewer.shared.payload_helpers import _sanitize  # noqa: E402
from alz.tcell_viewer.paths import (  # noqa: E402
    AUDIT_PREVIEW_ROWS,
    AUDIT_SOURCES_DIR,
    KINASE_ATTRIBUTION_TCELLS_DIR,
    TCELLS_INCYTR_INPUTS_DIR,
    UNIFIED_VIEWER_DIR,
)
from alz.tcell_viewer.common import DONORS, _short_contrast  # noqa: E402
from alz.tcell_viewer.slices_traces import _build_tcell_measurement_trace  # noqa: E402

# ---------------------------------------------------------------------------
# Audit helpers (only used here)
# ---------------------------------------------------------------------------

def _count_csv_rows(path: str) -> int:
    with open(path, "rb") as f:
        n = sum(1 for _ in f)
    return max(0, n - 1)


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


# ---------------------------------------------------------------------------
# Audit spec table
# ---------------------------------------------------------------------------

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
# Kinase audit drawer
# ---------------------------------------------------------------------------

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
    ("normalization_summary",   "T-cell normalization is per-batch (Forperseus); no summary JSON"),
    ("sample_mapping",          "T-cell sample mapping not surfaced (per-donor TMT plex)"),
)


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

    The T-cell MEA pipeline (alz/cohorts/tcells/mea.py::_run_mea) takes
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
    # Within-cohort attribution rows (donor1) — the Attribution subtab's raw
    # table. ONE file: the full kinase × state × day grid (every row shipped, no
    # gate); `tcell_concordant` is a shown label, not a filter.
    for key, fname, label in (
        ("unified_attribution",      "unified_attribution_tcells.csv",
         "Within-cohort attribution (donor1; full grid, all rows)"),
    ):
        src = os.path.join(donor1_root, fname)
        if not os.path.exists(src):
            tables[key] = _shim_audit_entry(
                key, label, "run alz/cross_reference/tcell_within_cohort.py first")
            continue
        dest = os.path.join(AUDIT_SOURCES_DIR, f"{key}.csv")
        os.makedirs(AUDIT_SOURCES_DIR, exist_ok=True)
        shutil.copyfile(src, dest)
        tables[key] = _audit_csv_meta(dest, label, key)
    # NSCLC 10x cell-type reference — the human cohort's independent detector.
    # Per-(kinase, cell_type) mean_log2(CPM+1) + fraction_cells_expressing +
    # detection + specificity_score (concentration of expression in type) +
    # specificity_count (N-of-7 coarse groups at ≥10% prevalence floor) +
    # nsclc_enrichment_<state> (ProjecTILs 14-state share, post Stage-1 regen).
    # The NSCLC lineage strip and verdict table read from this single canonical CSV.
    nsclc_src = config.NSCLC_KINASE_EXPRESSION_FILE
    if os.path.exists(nsclc_src):
        os.makedirs(AUDIT_SOURCES_DIR, exist_ok=True)
        dest = os.path.join(AUDIT_SOURCES_DIR, "nsclc_kinase_expression.csv")
        shutil.copyfile(nsclc_src, dest)
        tables["nsclc_kinase_expression"] = _audit_csv_meta(
            dest, "NSCLC 10x cell-type expression reference", "nsclc_kinase_expression")
    else:
        tables["nsclc_kinase_expression"] = _shim_audit_entry(
            "nsclc_kinase_expression", "NSCLC 10x cell-type expression reference",
            "run pixi run nsclc-expression first")
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
    manifest = {"preview_rows": AUDIT_PREVIEW_ROWS, "tables": tables}
    trace = _build_tcell_measurement_trace()
    if trace is not None:
        manifest["measurement_trace"] = trace
    return manifest

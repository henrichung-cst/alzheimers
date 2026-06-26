"""Packet 1B — Phase-1 cohort validators.

CLI:
    python -m alz.core.validate_cohort --cohort {song,mukesh,tcells,fivexfad}
    python -m alz.core.validate_cohort --all

Validates:
- File existence (present / absent_by_design)
- Required columns per artifact kind
- Sample-column presence in normalized matrices (cross-ref to discovered sample IDs)
- Track-suffix conventions (output_suffix = "" for ST, "_pY" for pY)
- MEA manifest: skip reason correctness (no_motif vs matrix_absent)
- Duplicate key behavior (using baseline_inventory key_columns)
- Numeric coercibility of MEA numeric columns (ES, NES, FDR, p-value)
- Sign-convention metadata presence (mea_fdr_thresh in manifests matches config)
- Absent-by-design entries are reported as SKIP, not FAIL

Memory-safety: no large file loads. See validation.py for thresholds.

IMPORTANT: this module is read-only — it does NOT modify any canonical output.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]

from alz.core.validation import (
    ValidationReport,
    check_file_exists,
    check_no_duplicate_keys,
    check_numeric_columns,
    check_required_columns,
    csv_header,
    file_exists,
    file_size,
    json_load_safe,
    parquet_columns,
    write_reports,
    SIZE_LIMIT_NUMERIC_CHECK,
)
from alz.core.phospho_schema import (
    MEA_FDR_THRESH,
    MEA_LONG_TABLE,
    FIVEXFAD_MEA_LONG_TABLE,
    FIVEXFAD_CELLTYPE_MEA,
    SONG_DECOMP_MEA_PARQUET,
    NES_FDR_MATRIX,
    SONG_OLS_TABLE,
    FIVEXFAD_OLS_TABLE,
    SONG_DECOMP_OLS_PARQUET,
    FIVEXFAD_CELLTYPE_OLS_PARQUET,
    MUKESH_RECURRENCE,
    TCELL_RECURRENCE,
    FIVEXFAD_SAMPLE_MANIFEST,
    FIVEXFAD_CONTRAST_QC,
    TCELL_MEA_MANIFEST,
    SONG_STOICH_MATRIX,
    SONG_RAW_NORM_MATRIX,
    HUMAN_NORM_MATRIX,
    FIVEXFAD_STOICH_MATRIX,
    FIVEXFAD_RAW_NORM_MATRIX,
    FIVEXFAD_MATCHED_PROTEIN_MATRIX,
)
from alz.core.baseline_inventory import (
    _build_protected_files,
    inventory_file,
    PROJECT_ROOT as _BASELINE_ROOT,
)
from alz.core.cohort_manifest import ALL_COHORTS, get_cohort

REPORT_DIR = PROJECT_ROOT / "outputs/reports/refactor_audit/phase_1"

# MEA long table numeric columns to check.
# NOTE: "Subs fraction" is GSEApy's "n_leading/n_total" fractional string (e.g. "136/715")
# — it is NOT a float-coercible column. ES/NES/p-value/FDR are the numeric columns.
MEA_NUMERIC_COLS = ["ES", "NES", "p-value", "FDR"]
MECHANISM_REQUIRED_COLS = [
    "cohort",
    "track",
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
MECHANISM_FORBIDDEN_COLS = ["mechanism_score"]
MECHANISM_ALLOWED_CALLS = {
    "both",
    "activity_driven",
    "abundance_driven",
    "discordant",
    "not_significant",
    "not_evaluable",
}
MECHANISM_ALLOWED_SIGN_RELATIONS = {
    "same",
    "opposite",
    "stoich_only",
    "raw_only",
    "none",
    "not_evaluable",
}
MECHANISM_NUMERIC_COLS = ["stoich_NES", "stoich_FDR", "raw_NES", "raw_FDR"]
PROJECTED_STATE_MEA_REQUIRED_MANIFEST_COLUMNS = [
    "donor",
    "state",
    "track",
    "kind",
    "baseline_day",
    "days_available",
    "days_run",
    "n_cells_by_day",
    "n_sites",
    "n_motif_sites",
    "input_files",
    "skip_reason",
]
PROJECTED_STATE_MEA_LONG_REQUIRED_COLUMNS = [
    "state",
    "timepoint",
    "kinase",
    "NES",
    "FDR",
    "donor",
    "track",
    "kind",
]
PROJECTED_STATE_MEA_AGGREGATE_FILES = [
    "kinase_state_timepoint_nes.csv",
    "kinase_state_timepoint_fdr.csv",
    "kinase_state_timepoint_nes_raw.csv",
    "kinase_state_timepoint_fdr_raw.csv",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _protected_specs(cohort: str) -> list[dict]:
    """Return the list of protected file specs for a cohort from baseline_inventory."""
    return _build_protected_files().get(cohort, [])


def _spec_map(cohort: str) -> dict[str, dict]:
    """Return {rel_path: spec} dict for the cohort."""
    return {s["rel_path"]: s for s in _protected_specs(cohort)}


def _validate_csv_schema(
    report: ValidationReport,
    rel_path: str,
    required_cols: list[str],
    numeric_cols: list[str],
    key_cols: list[str],
    absent_by_design: bool = False,
    skip_dup_check_reason: str | None = None,
) -> bool:
    """Standard pipeline: existence → columns → numeric → dup-key.

    Returns True if file is present (regardless of column failures).
    """
    present = check_file_exists(
        report, rel_path, check_name="file_exists",
        absent_by_design=absent_by_design,
    )
    if not present:
        return False

    header = csv_header(rel_path)
    check_required_columns(
        report, rel_path, required_cols, header,
        check_name="required_columns",
    )

    if numeric_cols:
        check_numeric_columns(
            report, rel_path, numeric_cols,
            check_name="numeric_coercibility",
        )

    if key_cols:
        if skip_dup_check_reason:
            report.add(rel_path, "no_duplicate_keys", "SKIP", skip_dup_check_reason)
        else:
            check_no_duplicate_keys(report, rel_path, key_cols)

    return True


def _validate_parquet_schema(
    report: ValidationReport,
    rel_path: str,
    required_cols: list[str],
    key_cols: list[str],
) -> bool:
    """Parquet: existence → columns (via pyarrow metadata).  No numeric coercibility check."""
    present = check_file_exists(report, rel_path, check_name="file_exists")
    if not present:
        return False

    cols = parquet_columns(rel_path)
    if cols is None:
        report.add(rel_path, "required_columns", "SKIP",
                   "pyarrow not available or file unreadable; column check skipped.")
    else:
        check_required_columns(
            report, rel_path, required_cols, cols,
            check_name="required_columns",
        )
        if key_cols:
            missing_keys = [k for k in key_cols if k not in cols]
            if missing_keys:
                report.add(rel_path, "no_duplicate_keys", "SKIP",
                           f"Key columns missing from parquet: {missing_keys}")
            else:
                report.add(rel_path, "no_duplicate_keys", "SKIP",
                           "Parquet dup-key check deferred (would need full load).")

    return True


def _check_track_suffix_convention(
    report: ValidationReport,
    rel_path: str,
    expected_suffix: str,
) -> None:
    """Check that a file path ends with the expected track suffix before the extension."""
    stem = Path(rel_path).stem  # filename without extension
    if expected_suffix == "":
        # ST track: must NOT end with _pY, _raw_pY
        if stem.endswith("_pY"):
            report.add(rel_path, "track_suffix_convention", "FAIL",
                       f"Expected ST suffix ('') but filename stem ends with '_pY': {stem}")
        else:
            report.add(rel_path, "track_suffix_convention", "PASS",
                       "ST track: no _pY suffix (correct).")
    else:
        if stem.endswith(expected_suffix.lstrip("_")) or (
            expected_suffix in stem
        ):
            report.add(rel_path, "track_suffix_convention", "PASS",
                       f"pY track: suffix '{expected_suffix}' present in filename.")
        else:
            report.add(rel_path, "track_suffix_convention", "FAIL",
                       f"Expected pY suffix '{expected_suffix}' but not found in: {stem}")


def _check_mea_fdr_thresh_in_manifest(
    report: ValidationReport,
    rel_path: str,
    manifest_data: Any,
) -> None:
    """Verify mea_fdr_thresh in manifest matches config value."""
    if not isinstance(manifest_data, dict):
        report.add(rel_path, "mea_fdr_thresh", "FAIL",
                   "Manifest not a JSON object.")
        return
    thresh = manifest_data.get("mea_fdr_thresh")
    if thresh is None:
        report.add(rel_path, "mea_fdr_thresh", "SKIP",
                   "mea_fdr_thresh key absent from manifest.")
    elif abs(float(thresh) - MEA_FDR_THRESH) < 1e-9:
        report.add(rel_path, "mea_fdr_thresh", "PASS",
                   f"mea_fdr_thresh = {thresh} matches config ({MEA_FDR_THRESH}).")
    else:
        report.add(rel_path, "mea_fdr_thresh", "FAIL",
                   f"mea_fdr_thresh = {thresh} != config {MEA_FDR_THRESH}.")


def _check_sample_columns_non_empty(
    report: ValidationReport,
    rel_path: str,
    header: list[str] | None,
    prefix_cols: list[str],
) -> None:
    """Check that there is at least one sample column beyond the prefix."""
    if header is None:
        return
    prefix_set = set(prefix_cols)
    sample_cols = [c for c in header if c not in prefix_set]
    if sample_cols:
        report.add(rel_path, "sample_columns_present", "PASS",
                   f"{len(sample_cols)} sample columns found.")
    else:
        report.add(rel_path, "sample_columns_present", "FAIL",
                   "No sample columns found beyond prefix columns.")


def _csv_has_data_rows(rel_path: str) -> bool | None:
    """Return whether a CSV has at least one post-header data row."""
    p = PROJECT_ROOT / rel_path
    if not p.exists():
        return None
    try:
        with open(p, newline="", encoding="utf-8", errors="replace") as f:
            reader = csv.reader(f)
            if next(reader, None) is None:
                return False
            for row in reader:
                if any(cell.strip() for cell in row):
                    return True
            return False
    except Exception:
        return None


def _check_optional_file_exists(
    report: ValidationReport,
    rel_path: str,
    check_name: str = "file_exists",
    *,
    present_message: str = "Optional file present.",
    missing_message: str = "Optional mechanism output file absent.",
) -> bool:
    """Emit PASS/ SKIP for optional files."""
    if file_exists(rel_path):
        report.add(rel_path, check_name, "PASS", present_message)
        return True
    report.add(rel_path, check_name, "SKIP", missing_message)
    return False


def _check_allowed_values_in_column(
    report: ValidationReport,
    rel_path: str,
    column: str,
    allowed_values: set[str],
    *,
    check_name: str,
) -> None:
    """Validate that each non-empty value in a column is in an allowed set."""
    try:
        p = PROJECT_ROOT / rel_path
        with open(p, newline="", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames or column not in reader.fieldnames:
                report.add(
                    rel_path, check_name, "FAIL",
                    f"{column} column missing; cannot validate values.",
                )
                return

            bad: set[str] = set()
            for row in reader:
                value = row.get(column)
                if value is None:
                    bad.add("<missing>")
                    continue
                normalized = str(value).strip()
                if normalized == "":
                    bad.add("<empty>")
                elif normalized not in allowed_values:
                    bad.add(normalized)
            if bad:
                report.add(
                    rel_path, check_name, "FAIL",
                    f"{column} has invalid values: {sorted(bad)}",
                )
            else:
                report.add(
                    rel_path, check_name, "PASS",
                    f"All {column} values in allowed vocabulary.",
                )
    except Exception:
        report.add(
            rel_path, check_name, "FAIL",
            f"Could not read {column} for vocabulary validation.",
        )


def _validate_mechanism_attribution_file(
    report: ValidationReport,
    rel_path: str,
    required_context_columns: list[str],
    *,
    check_prefix: str = "mechanism_attribution",
) -> bool:
    """Validate optional mechanism attribution outputs (additive, never required)."""
    if not _check_optional_file_exists(report, rel_path, check_name=f"{check_prefix}_file_exists"):
        return False

    required_cols = list(dict.fromkeys(required_context_columns + MECHANISM_REQUIRED_COLS))
    header = csv_header(rel_path)
    required_ok = check_required_columns(
        report, rel_path, required=required_cols, actual=header,
        check_name="required_columns",
    )
    if not required_ok or header is None:
        return False

    check_numeric_columns(
        report, rel_path, numeric_columns=MECHANISM_NUMERIC_COLS,
        check_name="numeric_coercibility",
    )

    forbidden = [col for col in MECHANISM_FORBIDDEN_COLS if col in header]
    if forbidden:
        report.add(
            rel_path,
            "forbidden_columns",
            "FAIL",
            f"Forbidden mechanism attribution columns present: {forbidden}",
        )
    else:
        report.add(
            rel_path,
            "forbidden_columns",
            "PASS",
            "No forbidden mechanism score columns present.",
        )

    _check_allowed_values_in_column(
        report,
        rel_path,
        "mechanism_call",
        MECHANISM_ALLOWED_CALLS,
        check_name="allowed_mechanism_call",
    )
    _check_allowed_values_in_column(
        report,
        rel_path,
        "sign_relation",
        MECHANISM_ALLOWED_SIGN_RELATIONS,
        check_name="allowed_sign_relation",
    )
    return True


def _validate_projected_state_mea_directory(
    report: ValidationReport,
    rel_dir: str,
) -> None:
    """Validate optional T-cell projected-state MEA outputs under a donor directory."""
    if not file_exists(rel_dir):
        report.add(
            rel_dir,
            "projected_state_mea_directory",
            "SKIP",
            "Optional projected-state MEA directory not found.",
        )
        return

    report.add(
        rel_dir,
        "projected_state_mea_directory",
        "PASS",
        "Projected-state MEA directory present.",
    )

    # Manifest (optional)
    rel_manifest = f"{rel_dir}/projected_state_mea_manifest.json"
    manifest_present = _check_optional_file_exists(
        report,
        rel_manifest,
        check_name="projected_state_mea_manifest_exists",
        present_message="Optional projected-state MEA manifest present.",
        missing_message="Optional projected-state MEA manifest absent.",
    )
    if manifest_present:
        data = json_load_safe(rel_manifest)
        if data is None:
            report.add(
                rel_manifest,
                "projected_state_mea_manifest_valid_json",
                "FAIL",
                "Could not parse projected_state_mea_manifest.json.",
            )
        elif not isinstance(data, dict):
            report.add(
                rel_manifest,
                "projected_state_mea_manifest_valid_json",
                "FAIL",
                "projected_state_mea_manifest.json is not a JSON object.",
            )
        else:
            check_required_columns(
                report,
                rel_manifest,
                required=PROJECTED_STATE_MEA_REQUIRED_MANIFEST_COLUMNS,
                actual=list(data.keys()),
                check_name="projected_state_mea_manifest_required_columns",
            )

    # Long tables
    for fname in ("mea_projected_state.csv", "mea_projected_state_raw.csv"):
        rel = f"{rel_dir}/{fname}"
        long_present = _check_optional_file_exists(
            report,
            rel,
            check_name="projected_state_mea_long_file_exists",
            present_message="Optional projected-state MEA long file present.",
            missing_message="Optional projected-state MEA long file absent.",
        )
        if not long_present:
            continue
        header = csv_header(rel)
        check_required_columns(
            report, rel,
            required=PROJECTED_STATE_MEA_LONG_REQUIRED_COLUMNS,
            actual=header,
            check_name="projected_state_mea_long_required_columns",
        )

    # Optional aggregate tables
    aggregate_present = False
    for fname in PROJECTED_STATE_MEA_AGGREGATE_FILES:
        rel = f"{rel_dir}/{fname}"
        if _check_optional_file_exists(
            report,
            rel,
            check_name="projected_state_mea_aggregate_file_exists",
            present_message="Optional projected-state aggregate file present.",
            missing_message="Optional projected-state aggregate file absent.",
        ):
            aggregate_present = True
            header = csv_header(rel)
            required_ok = check_required_columns(
                report,
                rel,
                required=["kinase"],
                actual=header,
                check_name="projected_state_mea_aggregate_required_columns",
            )
            if not required_ok or header is None:
                continue
            has_rows = _csv_has_data_rows(rel)
            if has_rows is None:
                report.add(
                    rel,
                    "projected_state_mea_aggregate_state_timepoint_columns",
                    "FAIL",
                    "Could not determine aggregate table data rows.",
                )
            elif has_rows:
                extra_columns = [col for col in header if col != "kinase"]
                if extra_columns:
                    report.add(
                        rel,
                        "projected_state_mea_aggregate_state_timepoint_columns",
                        "PASS",
                        "Aggregate table has kinase plus state/timepoint columns.",
                    )
                else:
                    report.add(
                        rel,
                        "projected_state_mea_aggregate_state_timepoint_columns",
                        "FAIL",
                        "Aggregate table has no state/timepoint columns beyond kinase.",
                    )
            else:
                report.add(
                    rel,
                    "projected_state_mea_aggregate_state_timepoint_columns",
                    "PASS",
                    "Aggregate table present but contains no data rows.",
                )
    if aggregate_present:
        rel_recurrence = f"{rel_dir}/recurrence_projected_state_deferred.txt"
        check_file_exists(
            report,
            rel_recurrence,
            check_name="projected_state_mea_recurrence_deferred_note_exists",
        )

    # Mechanism attribution (optional)
    rel_mech = f"{rel_dir}/mechanism_attribution_projected_state.csv"
    _validate_mechanism_attribution_file(
        report,
        rel_mech,
        required_context_columns=[
            "cohort",
            "donor",
            "track",
            "state",
            "timepoint",
            "kinase",
        ],
        check_prefix="tcells_projected_state_mechanism_attribution",
    )
# ---------------------------------------------------------------------------
# Song validator
# ---------------------------------------------------------------------------

def validate_song(report: ValidationReport) -> None:
    root = "outputs/reports/kinase_attribution"
    decomp = "outputs/reports/decomposition/levy_t5"
    recovery = "outputs/reports/attribution_recovery"

    # ---- MEA long tables ----
    for suffix, track_suffix, track_label in [
        ("", "", "ST"),
        ("_pY", "_pY", "pY"),
    ]:
        for kind, kind_suffix in [("raw_phospho", ""), ("stoichiometry", "")]:
            if kind == "raw_phospho":
                fname = f"mea_raw_phospho{suffix}.csv"
            else:
                fname = f"mea_stoichiometry{suffix}.csv"
            rel = f"{root}/{fname}"
            _validate_csv_schema(
                report, rel,
                required_cols=MEA_LONG_TABLE.required_columns,
                numeric_cols=MEA_NUMERIC_COLS,
                key_cols=[],  # MEA long is multi-row per kinase by design (contrast × track)
                skip_dup_check_reason=(
                    "MEA long table: multiple rows per kinase (one per contrast × track) — "
                    "kinase alone is not unique. Accepted by design."
                ),
            )
            _check_track_suffix_convention(report, rel, track_suffix)

    # ---- audit files ----
    for fname, suffix in [
        ("mea_global_shift.csv", ""),
        ("mea_global_shift_pY.csv", "_pY"),
        ("winsorized_sites.csv", ""),
        ("winsorized_sites_pY.csv", "_pY"),
    ]:
        rel = f"{root}/{fname}"
        header = None
        present = check_file_exists(report, rel, check_name="file_exists")
        if present:
            header = csv_header(rel)
            if fname.startswith("mea_global_shift"):
                check_required_columns(
                    report, rel,
                    required=["median_shift"],
                    actual=header,
                    check_name="required_columns",
                )
            elif fname.startswith("winsorized"):
                check_required_columns(
                    report, rel,
                    required=["site_id", "gene_symbol", "original_lfc", "clipped_lfc"],
                    actual=header,
                    check_name="required_columns",
                )
        _check_track_suffix_convention(report, rel, suffix)

    # ---- OLS tables ----
    for fname, suffix in [
        ("site_level_ols.csv", ""),
        ("site_level_ols_pY.csv", "_pY"),
    ]:
        rel = f"{root}/{fname}"
        _validate_csv_schema(
            report, rel,
            required_cols=SONG_OLS_TABLE.required_columns,
            numeric_cols=[],
            key_cols=["site_id"],
            skip_dup_check_reason=None,
        )
        _check_track_suffix_convention(report, rel, suffix)

    # ---- substrate sets ----
    for fname, suffix in [
        ("mea_substrate_sets.csv", ""),
        ("mea_substrate_sets_pY.csv", "_pY"),
    ]:
        rel = f"{root}/{fname}"
        _validate_csv_schema(
            report, rel,
            required_cols=["kinase", "contrast", "motif", "residue_type", "track", "kl_percentile"],
            numeric_cols=["kl_percentile"],
            key_cols=[],
            skip_dup_check_reason=(
                "Substrate sets: multiple rows per kinase (one per motif) — "
                "not expected to be unique on kinase alone."
            ),
        )
        _check_track_suffix_convention(report, rel, suffix)

    # ---- attribution ----
    for fname in ["unified_attribution.csv", "unified_attribution_full.csv"]:
        rel = f"{root}/{fname}"
        _validate_csv_schema(
            report, rel,
            required_cols=["kinase", "contrast", "NES", "FDR"],
            numeric_cols=["NES", "FDR"],
            key_cols=["kinase", "contrast", "cell_type"],
            skip_dup_check_reason=None,
        )

    # ---- optional mechanism attribution ----
    for fname, suffix in [
        ("mechanism_attribution.csv", ""),
        ("mechanism_attribution_pY.csv", "_pY"),
    ]:
        rel = f"{root}/{fname}"
        if _validate_mechanism_attribution_file(
            report, rel,
            required_context_columns=["cohort", "track", "contrast", "kinase"],
            check_prefix="song_mechanism_attribution",
        ):
            _check_track_suffix_convention(report, rel, suffix)

    # ---- normalized matrices ----
    for fname, schema, suffix in [
        ("stoichiometry_matrix.csv", SONG_STOICH_MATRIX, ""),
        ("stoichiometry_matrix_pY.csv", SONG_STOICH_MATRIX, "_pY"),
        ("raw_phospho_normalized.csv", SONG_RAW_NORM_MATRIX, ""),
        ("raw_phospho_normalized_pY.csv", SONG_RAW_NORM_MATRIX, "_pY"),
        ("total_proteome_normalized.csv", None, ""),
    ]:
        rel = f"{root}/{fname}"
        present = check_file_exists(report, rel, check_name="file_exists")
        if present and schema is not None:
            header = csv_header(rel)
            check_required_columns(
                report, rel, schema.required_prefix_columns, header,
                check_name="required_columns",
            )
            _check_sample_columns_non_empty(
                report, rel, header, schema.required_prefix_columns
            )
        elif present:
            # total_proteome_normalized: just existence + non-empty columns
            header = csv_header(rel)
            if header:
                report.add(rel, "required_columns", "PASS",
                           f"Total proteome normalized: columns present ({len(header)}).")
            else:
                report.add(rel, "required_columns", "FAIL",
                           "Could not read total_proteome_normalized header.")
        _check_track_suffix_convention(report, rel, suffix)

    # ---- decomposition parquets ----
    for fname, required, keys in [
        ("mea_per_cluster.parquet", SONG_DECOMP_MEA_PARQUET.required_columns,
         ["kinase", "cluster", "contrast"]),
        ("mea_per_cluster_pY.parquet", SONG_DECOMP_MEA_PARQUET.required_columns,
         ["kinase", "cluster", "contrast"]),
        ("site_level_ols_per_cluster.parquet", SONG_DECOMP_OLS_PARQUET.required_columns,
         ["site_id", "cluster"]),
        ("site_level_ols_per_cluster_pY.parquet", SONG_DECOMP_OLS_PARQUET.required_columns,
         ["site_id", "cluster"]),
    ]:
        rel = f"{decomp}/{fname}"
        _validate_parquet_schema(report, rel, required_cols=required, key_cols=keys)

    # ---- other decomposition files ----
    for fname in [
        "proportions.parquet", "phospho_per_cluster.parquet",
        "phospho_per_cluster_pY.parquet", "protein_per_cluster.parquet",
        "transcript_per_cluster.parquet",
    ]:
        check_file_exists(report, f"{decomp}/{fname}", check_name="file_exists")

    # verification.json: just existence
    check_file_exists(report, f"{decomp}/verification.json", check_name="file_exists")

    # ---- recovery ----
    for fname, required, keys in [
        ("celltype_evidence_table.csv",
         ["kinase", "cell_type", "contrast", "confidence_tier"],
         ["kinase", "cell_type", "contrast"]),
        ("kinase_activity_matrix.csv",
         ["kinase", "residue_type"],
         ["kinase", "residue_type"]),
        ("kinase_hypothesis_table.csv",
         ["kinase", "residue_type", "n_sig_contrasts"],
         ["kinase", "residue_type"]),
    ]:
        rel = f"{recovery}/{fname}"
        _validate_csv_schema(
            report, rel,
            required_cols=required,
            numeric_cols=[],
            key_cols=keys,
        )


# ---------------------------------------------------------------------------
# Mukesh validator
# ---------------------------------------------------------------------------

def validate_mukesh(report: ValidationReport) -> None:
    root = "outputs/reports/kinase_attribution_human"
    perdonor = f"{root}/perdonor"

    # ---- normalized matrices (root) ----
    for fname, schema, suffix in [
        ("raw_phospho_normalized.csv", HUMAN_NORM_MATRIX, ""),
        ("raw_phospho_normalized_pY.csv", HUMAN_NORM_MATRIX, "_pY"),
        ("stoichiometry_matrix.csv", HUMAN_NORM_MATRIX, ""),
        ("stoichiometry_matrix_pY.csv", HUMAN_NORM_MATRIX, "_pY"),
        ("celltype_specificity.csv", None, ""),
    ]:
        rel = f"{root}/{fname}"
        present = check_file_exists(report, rel, check_name="file_exists")
        if present:
            header = csv_header(rel)
            if schema is not None:
                check_required_columns(
                    report, rel, schema.required_prefix_columns, header,
                    check_name="required_columns",
                )
                _check_sample_columns_non_empty(
                    report, rel, header, schema.required_prefix_columns,
                )
            else:
                # celltype_specificity
                check_required_columns(
                    report, rel,
                    required=["kinase", "reference", "celltype", "specificity_score"],
                    actual=header,
                    check_name="required_columns",
                )
        _check_track_suffix_convention(report, rel, suffix)

    # ---- NES/FDR matrices (perdonor) ----
    for fname, suffix in [
        ("kinase_donor_nes.csv", ""),
        ("kinase_donor_nes_pY.csv", "_pY"),
        ("kinase_donor_nes_raw.csv", ""),
        ("kinase_donor_nes_raw_pY.csv", "_pY"),
        ("kinase_donor_fdr.csv", ""),
        ("kinase_donor_fdr_pY.csv", "_pY"),
        ("kinase_donor_fdr_raw.csv", ""),
        ("kinase_donor_fdr_raw_pY.csv", "_pY"),
    ]:
        rel = f"{perdonor}/{fname}"
        present = check_file_exists(report, rel, check_name="file_exists")
        if present:
            header = csv_header(rel)
            check_required_columns(
                report, rel, NES_FDR_MATRIX.required_prefix_columns, header,
                check_name="required_columns",
            )
            _check_sample_columns_non_empty(
                report, rel, header, NES_FDR_MATRIX.required_prefix_columns,
            )
        _check_track_suffix_convention(report, rel, suffix)

    # ---- MEA long tables (perdonor) ----
    for fname, suffix in [
        ("mea_perdonor.csv", ""),
        ("mea_perdonor_pY.csv", "_pY"),
        ("mea_perdonor_raw.csv", ""),
        ("mea_perdonor_raw_pY.csv", "_pY"),
    ]:
        rel = f"{perdonor}/{fname}"
        _validate_csv_schema(
            report, rel,
            required_cols=MEA_LONG_TABLE.required_columns,
            numeric_cols=MEA_NUMERIC_COLS,
            key_cols=[],
            skip_dup_check_reason=(
                "MEA long: multiple rows per kinase (one per contrast × donor) — "
                "not unique on kinase alone."
            ),
        )
        _check_track_suffix_convention(report, rel, suffix)

    # ---- audit (perdonor) ----
    for fname, suffix in [
        ("mea_global_shift.csv", ""),
        ("mea_global_shift_pY.csv", "_pY"),
        ("winsorized_sites.csv", ""),
        ("winsorized_sites_pY.csv", "_pY"),
    ]:
        rel = f"{perdonor}/{fname}"
        present = check_file_exists(report, rel, check_name="file_exists")
        if present:
            header = csv_header(rel)
            if fname.startswith("mea_global_shift"):
                check_required_columns(
                    report, rel, ["median_shift"], header,
                    check_name="required_columns",
                )
            else:
                check_required_columns(
                    report, rel,
                    ["site_id", "gene_symbol", "original_lfc", "clipped_lfc"],
                    header, check_name="required_columns",
                )
        _check_track_suffix_convention(report, rel, suffix)

    # ---- substrate sets (perdonor) ----
    for fname, suffix in [
        ("mea_substrate_sets.csv", ""),
        ("mea_substrate_sets_pY.csv", "_pY"),
    ]:
        rel = f"{perdonor}/{fname}"
        _validate_csv_schema(
            report, rel,
            required_cols=["kinase", "contrast", "motif", "residue_type", "track", "kl_percentile"],
            numeric_cols=["kl_percentile"],
            key_cols=[],
            skip_dup_check_reason="Substrate sets: multiple rows per kinase by design.",
        )
        _check_track_suffix_convention(report, rel, suffix)

    # ---- recurrence (perdonor) ----
    for fname, suffix in [
        ("recurrence.csv", ""),
        ("recurrence_pY.csv", "_pY"),
        ("recurrence_ctrl.csv", ""),
        ("recurrence_ctrl_pY.csv", "_pY"),
    ]:
        rel = f"{perdonor}/{fname}"
        _validate_csv_schema(
            report, rel,
            required_cols=MUKESH_RECURRENCE.required_columns,
            numeric_cols=["n_donors_sig", "n_donors_up", "n_donors_down",
                          "n_donors_tested", "median_nes"],
            key_cols=["kinase"],
        )
        _check_track_suffix_convention(report, rel, suffix)

    # ---- optional mechanism attribution (per-donor) ----
    for fname, suffix in [
        ("mechanism_attribution.csv", ""),
        ("mechanism_attribution_pY.csv", "_pY"),
    ]:
        rel = f"{perdonor}/{fname}"
        if _validate_mechanism_attribution_file(
            report, rel,
            required_context_columns=["cohort", "track", "donor", "kinase"],
            check_prefix="mukesh_mechanism_attribution",
        ):
            _check_track_suffix_convention(report, rel, suffix)


# ---------------------------------------------------------------------------
# T-cells validator
# ---------------------------------------------------------------------------

def validate_tcells(report: ValidationReport) -> None:
    root = "outputs/reports/kinase_attribution_tcells"
    d1 = f"{root}/donor1"
    d1_mea = f"{d1}/mea"
    d1_state_mea = f"{d1}/state_mea"
    d2 = f"{root}/donor2"
    d2_mea = f"{d2}/mea"
    d2_state_mea = f"{d2}/state_mea"

    # ---- donor1 normalized ----
    for fname, schema, suffix in [
        ("raw_phospho_normalized.csv", HUMAN_NORM_MATRIX, ""),
        ("raw_phospho_normalized_pY.csv", HUMAN_NORM_MATRIX, "_pY"),
        ("stoichiometry_matrix.csv", HUMAN_NORM_MATRIX, ""),
        ("stoichiometry_matrix_pY.csv", HUMAN_NORM_MATRIX, "_pY"),
        ("total_proteome_normalized.csv", None, ""),
    ]:
        rel = f"{d1}/{fname}"
        present = check_file_exists(report, rel, check_name="file_exists")
        if present:
            header = csv_header(rel)
            if schema is not None:
                check_required_columns(
                    report, rel, schema.required_prefix_columns, header,
                    check_name="required_columns",
                )
                _check_sample_columns_non_empty(
                    report, rel, header, schema.required_prefix_columns,
                )
            else:
                if header:
                    report.add(rel, "required_columns", "PASS",
                               f"total_proteome_normalized columns present ({len(header)}).")
        _check_track_suffix_convention(report, rel, suffix)

    # ---- donor1 concordance / specificity / attribution ----
    for fname, required in [
        # tcell_concordance: gene-level DEG concordance table (not site-level)
        ("tcell_concordance.csv", ["gene", "state", "tcell_lfc"]),
        # tcell_enrichment: gene-level within-cohort enrichment metric (not kinase-level)
        ("tcell_enrichment.csv",
         ["gene", "state", "tcell_detected", "tcell_state_enrichment"]),
        ("unified_attribution_tcells.csv",
         ["kinase", "residue_type", "contrast", "NES", "FDR"]),
    ]:
        rel = f"{d1}/{fname}"
        _validate_csv_schema(
            report, rel,
            required_cols=required,
            numeric_cols=[],
            key_cols=[],
            skip_dup_check_reason="No key columns defined for this file.",
        )

    # ---- donor1 MEA manifest ----
    rel_d1_manifest = f"{d1_mea}/mea_manifest.json"
    present = check_file_exists(report, rel_d1_manifest, check_name="file_exists")
    if present:
        data = json_load_safe(rel_d1_manifest)
        if data is None:
            report.add(rel_d1_manifest, "manifest_required_keys", "FAIL",
                       "Could not parse donor1 mea_manifest.json.")
        else:
            check_required_columns(
                report, rel_d1_manifest,
                required=TCELL_MEA_MANIFEST.required_columns,
                actual=list(data.keys()),
                check_name="manifest_required_keys",
            )
            _check_mea_fdr_thresh_in_manifest(report, rel_d1_manifest, data)
            # donor1 should have run all tracks
            mea_ran = data.get("mea_ran", [])
            mea_skipped = data.get("mea_skipped", [])
            if not isinstance(mea_ran, list) or len(mea_ran) == 0:
                report.add(rel_d1_manifest, "donor1_mea_ran_non_empty", "FAIL",
                           f"donor1 mea_ran is empty: {mea_ran}")
            else:
                report.add(rel_d1_manifest, "donor1_mea_ran_non_empty", "PASS",
                           f"donor1 mea_ran: {mea_ran}")
            if mea_skipped:
                report.add(rel_d1_manifest, "donor1_mea_skipped_empty", "DEVIATION",
                           f"donor1 has skipped tracks: {mea_skipped}. "
                           "Expected all tracks ran for donor1.")
            else:
                report.add(rel_d1_manifest, "donor1_mea_skipped_empty", "PASS",
                           "donor1 mea_skipped is empty (correct).")

    # ---- donor1 MEA long tables ----
    for fname, suffix in [
        ("mea_timecourse.csv", ""),
        ("mea_timecourse_pY.csv", "_pY"),
    ]:
        rel = f"{d1_mea}/{fname}"
        _validate_csv_schema(
            report, rel,
            required_cols=MEA_LONG_TABLE.required_columns,
            numeric_cols=MEA_NUMERIC_COLS,
            key_cols=[],
            skip_dup_check_reason=(
                "MEA long: multiple rows per kinase by design (one per contrast × timepoint)."
            ),
        )
        _check_track_suffix_convention(report, rel, suffix)

    # ---- donor1 optional mechanism attribution ----
    for fname, suffix in [
        ("mechanism_attribution.csv", ""),
        ("mechanism_attribution_pY.csv", "_pY"),
    ]:
        rel = f"{d1_mea}/{fname}"
        if _validate_mechanism_attribution_file(
            report, rel,
            required_context_columns=["cohort", "donor", "track", "timepoint", "kinase"],
            check_prefix="tcells_donor1_mechanism_attribution",
        ):
            _check_track_suffix_convention(report, rel, suffix)

    # ---- donor1 NES/FDR matrices ----
    for fname, suffix in [
        ("kinase_timepoint_nes.csv", ""),
        ("kinase_timepoint_nes_pY.csv", "_pY"),
        ("kinase_timepoint_nes_raw.csv", ""),
        ("kinase_timepoint_nes_raw_pY.csv", "_pY"),
        ("kinase_timepoint_fdr.csv", ""),
        ("kinase_timepoint_fdr_pY.csv", "_pY"),
        ("kinase_timepoint_fdr_raw.csv", ""),
        ("kinase_timepoint_fdr_raw_pY.csv", "_pY"),
    ]:
        rel = f"{d1_mea}/{fname}"
        present = check_file_exists(report, rel, check_name="file_exists")
        if present:
            header = csv_header(rel)
            check_required_columns(
                report, rel, NES_FDR_MATRIX.required_prefix_columns, header,
                check_name="required_columns",
            )
            _check_sample_columns_non_empty(
                report, rel, header, NES_FDR_MATRIX.required_prefix_columns,
            )
        _check_track_suffix_convention(report, rel, suffix)

    # ---- donor1 audit + substrate sets + recurrence ----
    for fname, suffix in [
        ("mea_global_shift.csv", ""),
        ("mea_global_shift_pY.csv", "_pY"),
        ("winsorized_sites.csv", ""),
        ("winsorized_sites_pY.csv", "_pY"),
    ]:
        rel = f"{d1_mea}/{fname}"
        present = check_file_exists(report, rel, check_name="file_exists")
        if present:
            header = csv_header(rel)
            if fname.startswith("mea_global"):
                check_required_columns(report, rel, ["median_shift"], header,
                                       check_name="required_columns")
            else:
                check_required_columns(
                    report, rel,
                    ["site_id", "gene_symbol", "original_lfc", "clipped_lfc"],
                    header, check_name="required_columns",
                )
        _check_track_suffix_convention(report, rel, suffix)

    for fname, suffix in [
        ("mea_substrate_sets.csv", ""),
        ("mea_substrate_sets_pY.csv", "_pY"),
    ]:
        rel = f"{d1_mea}/{fname}"
        _validate_csv_schema(
            report, rel,
            required_cols=["kinase", "contrast", "motif", "residue_type", "track"],
            numeric_cols=[],
            key_cols=[],
            skip_dup_check_reason="Substrate sets: multiple rows per kinase by design.",
        )
        _check_track_suffix_convention(report, rel, suffix)

    rel_rec = f"{d1_mea}/recurrence.csv"
    _validate_csv_schema(
        report, rel_rec,
        required_cols=TCELL_RECURRENCE.required_columns,
        numeric_cols=["n_timepoints_sig", "n_timepoints_up", "n_timepoints_down",
                      "n_timepoints_tested", "median_nes"],
        key_cols=["kinase"],
    )

    # ---- donor2 — partial by design ----
    # pY matrices: present and valid
    for fname, suffix in [
        ("raw_phospho_normalized_pY.csv", "_pY"),
        ("stoichiometry_matrix_pY.csv", "_pY"),
    ]:
        rel = f"{d2}/{fname}"
        present = check_file_exists(report, rel, check_name="file_exists")
        if present:
            header = csv_header(rel)
            check_required_columns(
                report, rel, HUMAN_NORM_MATRIX.required_prefix_columns, header,
                check_name="required_columns",
            )
            _check_sample_columns_non_empty(
                report, rel, header, HUMAN_NORM_MATRIX.required_prefix_columns,
            )
        _check_track_suffix_convention(report, rel, suffix)

    # donor2 mea_timecourse.csv: absent_by_design
    rel_d2_long = f"{d2_mea}/mea_timecourse.csv"
    check_file_exists(
        report, rel_d2_long,
        check_name="file_exists",
        absent_by_design=True,
    )

    # ---- donor2 MEA manifest ----
    rel_d2_manifest = f"{d2_mea}/mea_manifest.json"
    present = check_file_exists(report, rel_d2_manifest, check_name="file_exists")
    if present:
        data = json_load_safe(rel_d2_manifest)
        if data is None:
            report.add(rel_d2_manifest, "manifest_required_keys", "FAIL",
                       "Could not parse donor2 mea_manifest.json.")
        else:
            check_required_columns(
                report, rel_d2_manifest,
                required=TCELL_MEA_MANIFEST.required_columns,
                actual=list(data.keys()),
                check_name="manifest_required_keys",
            )
            _check_mea_fdr_thresh_in_manifest(report, rel_d2_manifest, data)

            # Verify donor2 pY skip reason is "no_motif" (not "matrix_absent")
            skipped = data.get("mea_skipped", [])
            py_skips = [
                s for s in skipped
                if isinstance(s, dict) and s.get("track") == "py"
            ]
            if not py_skips:
                report.add(rel_d2_manifest, "donor2_py_skip_reason", "FAIL",
                           "donor2 has no pY skip entries in mea_skipped.")
            else:
                all_no_motif = all(
                    s.get("reason") == "no_motif" for s in py_skips
                )
                if all_no_motif:
                    report.add(
                        rel_d2_manifest, "donor2_py_skip_reason", "PASS",
                        "donor2 pY MEA skip reason = 'no_motif' (matrix present, "
                        "motif absent) — correct."
                    )
                else:
                    reasons = [s.get("reason") for s in py_skips]
                    report.add(
                        rel_d2_manifest, "donor2_py_skip_reason", "FAIL",
                        f"donor2 pY skip reason(s) = {reasons}; expected 'no_motif'."
                    )

            # Verify donor2 ST skip reason is "matrix_absent"
            st_skips = [
                s for s in skipped
                if isinstance(s, dict) and s.get("track") == "st"
            ]
            if not st_skips:
                report.add(rel_d2_manifest, "donor2_st_skip_reason", "FAIL",
                           "donor2 has no ST skip entries in mea_skipped.")
            else:
                all_matrix_absent = all(
                    s.get("reason") == "matrix_absent" for s in st_skips
                )
                if all_matrix_absent:
                    report.add(
                        rel_d2_manifest, "donor2_st_skip_reason", "PASS",
                        "donor2 ST MEA skip reason = 'matrix_absent' — correct."
                    )
                else:
                    reasons = [s.get("reason") for s in st_skips]
                    report.add(
                        rel_d2_manifest, "donor2_st_skip_reason", "FAIL",
                        f"donor2 ST skip reason(s) = {reasons}; expected 'matrix_absent'."
                    )

            # donor2 mea_ran should be empty
            mea_ran = data.get("mea_ran", [])
            if mea_ran:
                report.add(rel_d2_manifest, "donor2_mea_ran_empty", "FAIL",
                           f"donor2 mea_ran is non-empty: {mea_ran}")
            else:
                report.add(rel_d2_manifest, "donor2_mea_ran_empty", "PASS",
                           "donor2 mea_ran is empty (correct — no MEA ran).")

    # ---- donor2 optional mechanism attribution ----
    for fname, suffix in [
        ("mechanism_attribution.csv", ""),
        ("mechanism_attribution_pY.csv", "_pY"),
    ]:
        rel = f"{d2_mea}/{fname}"
        if _validate_mechanism_attribution_file(
            report, rel,
            required_context_columns=["cohort", "donor", "track", "timepoint", "kinase"],
            check_prefix="tcells_donor2_mechanism_attribution",
        ):
            _check_track_suffix_convention(report, rel, suffix)

    _validate_projected_state_mea_directory(report, d1_state_mea)
    _validate_projected_state_mea_directory(report, d2_state_mea)


# ---------------------------------------------------------------------------
# 5xFAD validator
# ---------------------------------------------------------------------------

def validate_fivexfad(report: ValidationReport) -> None:
    root = "outputs/reports/kinase_attribution_5xfad"
    ct = f"{root}/celltype_mea"

    # ---- sample manifest ----
    rel_sm = f"{root}/sample_manifest.csv"
    _validate_csv_schema(
        report, rel_sm,
        required_cols=FIVEXFAD_SAMPLE_MANIFEST.required_columns,
        numeric_cols=[],
        key_cols=[],
        skip_dup_check_reason="Sample manifest: rows are samples, no single-column key.",
    )

    # ---- per-region × per-track files ----
    for region in ["cortex", "hippocampus"]:
        for track_short, track_suffix, track_fname in [
            ("st", "", "st"),
            ("py", "_pY", "py"),
        ]:
            prefix = f"{root}/{region}_{track_fname}"

            # MEA long
            for kind, kind_fname in [
                ("raw_phospho", "mea_raw_phospho"),
                ("stoichiometry", "mea_stoichiometry"),
            ]:
                rel = f"{prefix}_{kind_fname}.csv"
                _validate_csv_schema(
                    report, rel,
                    required_cols=FIVEXFAD_MEA_LONG_TABLE.required_columns,
                    numeric_cols=MEA_NUMERIC_COLS,
                    key_cols=[],
                    skip_dup_check_reason=(
                        "5xFAD MEA long: multiple rows per kinase by design "
                        "(one per contrast)."
                    ),
                )
                # Track suffix: 5xFAD uses _st_ / _py_ in filename, not _pY suffix
                # Validate that the 'track' column value is sensible after loading header
                # (the suffix check uses file path convention, not column values)
                _check_fivexfad_track_fname_convention(
                    report, rel, region, track_fname
                )

            # substrate sets
            rel_ss = f"{prefix}_mea_substrate_sets.csv"
            _validate_csv_schema(
                report, rel_ss,
                required_cols=["kinase", "contrast", "motif", "residue_type",
                               "track", "kl_percentile"],
                numeric_cols=["kl_percentile"],
                key_cols=[],
                skip_dup_check_reason="Substrate sets: multiple rows per kinase.",
            )
            _check_fivexfad_track_fname_convention(
                report, rel_ss, region, track_fname
            )

            # audit: global shift
            rel_gs = f"{prefix}_mea_global_shift.csv"
            present = check_file_exists(report, rel_gs, check_name="file_exists")
            if present:
                header = csv_header(rel_gs)
                check_required_columns(
                    report, rel_gs,
                    required=["tissue", "analysis_track", "contrast", "median_shift"],
                    actual=header,
                    check_name="required_columns",
                )
            _check_fivexfad_track_fname_convention(
                report, rel_gs, region, track_fname
            )

            # audit: winsorized sites
            rel_ws = f"{prefix}_winsorized_sites.csv"
            present = check_file_exists(report, rel_ws, check_name="file_exists")
            if present:
                header = csv_header(rel_ws)
                check_required_columns(
                    report, rel_ws,
                    required=["tissue", "analysis_track", "site_id", "gene_symbol",
                               "original_lfc", "clipped_lfc"],
                    actual=header,
                    check_name="required_columns",
                )
            _check_fivexfad_track_fname_convention(
                report, rel_ws, region, track_fname
            )

            # OLS table
            rel_ols = f"{prefix}_site_level_ols.csv"
            _validate_csv_schema(
                report, rel_ols,
                required_cols=FIVEXFAD_OLS_TABLE.required_columns,
                numeric_cols=[],
                key_cols=["site_id"],
            )
            _check_fivexfad_track_fname_convention(
                report, rel_ols, region, track_fname
            )

            # optional mechanism attribution
            rel_mech = f"{prefix}_mechanism_attribution.csv"
            if _validate_mechanism_attribution_file(
                report, rel_mech,
                required_context_columns=["cohort", "tissue", "track", "contrast", "kinase"],
                check_prefix="fivexfad_mechanism_attribution",
            ):
                _check_fivexfad_track_fname_convention(
                    report, rel_mech, region, track_fname
                )

            # contrast QC
            rel_qc = f"{prefix}_contrast_qc.csv"
            _validate_csv_schema(
                report, rel_qc,
                required_cols=FIVEXFAD_CONTRAST_QC.required_columns,
                numeric_cols=[],
                key_cols=["contrast"],
            )
            _check_fivexfad_track_fname_convention(
                report, rel_qc, region, track_fname
            )

            # normalized raw phospho
            rel_rn = f"{prefix}_raw_phospho_normalized.csv"
            present = check_file_exists(report, rel_rn, check_name="file_exists")
            if present:
                header = csv_header(rel_rn)
                check_required_columns(
                    report, rel_rn,
                    FIVEXFAD_RAW_NORM_MATRIX.required_prefix_columns,
                    header, check_name="required_columns",
                )
                _check_sample_columns_non_empty(
                    report, rel_rn, header,
                    FIVEXFAD_RAW_NORM_MATRIX.required_prefix_columns,
                )
            _check_fivexfad_track_fname_convention(
                report, rel_rn, region, track_fname
            )

            # stoichiometry matrix
            rel_sm2 = f"{prefix}_stoichiometry_matrix.csv"
            present = check_file_exists(report, rel_sm2, check_name="file_exists")
            if present:
                header = csv_header(rel_sm2)
                check_required_columns(
                    report, rel_sm2,
                    FIVEXFAD_STOICH_MATRIX.required_prefix_columns,
                    header, check_name="required_columns",
                )
                _check_sample_columns_non_empty(
                    report, rel_sm2, header,
                    FIVEXFAD_STOICH_MATRIX.required_prefix_columns,
                )
            _check_fivexfad_track_fname_convention(
                report, rel_sm2, region, track_fname
            )

            # matched total protein
            rel_mp = f"{prefix}_matched_total_protein.csv"
            present = check_file_exists(report, rel_mp, check_name="file_exists")
            if present:
                header = csv_header(rel_mp)
                check_required_columns(
                    report, rel_mp,
                    FIVEXFAD_MATCHED_PROTEIN_MATRIX.required_prefix_columns,
                    header, check_name="required_columns",
                )
            _check_fivexfad_track_fname_convention(
                report, rel_mp, region, track_fname
            )

        # region-level total proteome
        rel_tp = f"{root}/{region}_total_proteome_normalized.csv"
        present = check_file_exists(report, rel_tp, check_name="file_exists")
        if present:
            header = csv_header(rel_tp)
            if header:
                report.add(rel_tp, "required_columns", "PASS",
                           f"total_proteome_normalized columns present ({len(header)}).")

    # ---- snRNA attribution + cell counts ----
    for fname, required, keys in [
        ("fivexfad_snrna_attribution.csv",
         ["kinase", "gene_symbol", "tissue", "age_months", "cell_type",
          "confidence_tier"],
         ["kinase", "cell_type", "tissue", "age_months"]),
        ("fivexfad_snrna_cell_counts.csv",
         ["tissue", "age_months", "genotype", "cell_type", "n_cells"],
         []),
    ]:
        rel = f"{root}/{fname}"
        _validate_csv_schema(
            report, rel, required_cols=required,
            numeric_cols=[], key_cols=keys,
        )

    # ---- celltype MEA parquets ----
    # Note: baseline_inventory key_columns uses "cluster" for these files, but the
    # actual parquets use "cell_type". The key_columns annotation in baseline_inventory
    # is conceptually correct (cluster = cell_type here); we validate with the real
    # column name. See phase_1_legacy_deviations.md: cluster_vs_cell_type_key_annotation.
    _validate_parquet_schema(
        report,
        f"{ct}/fivexfad_celltype_mea.parquet",
        required_cols=FIVEXFAD_CELLTYPE_MEA.required_columns,
        key_cols=["kinase", "cell_type", "contrast"],
    )
    _validate_parquet_schema(
        report,
        f"{ct}/fivexfad_celltype_site_level_ols.parquet",
        required_cols=FIVEXFAD_CELLTYPE_OLS_PARQUET.required_columns,
        key_cols=["site_id", "cell_type"],
    )

    # celltype substrate sets
    rel_ct_ss = f"{ct}/fivexfad_celltype_substrate_sets.csv"
    _validate_csv_schema(
        report, rel_ct_ss,
        required_cols=["kinase", "contrast", "motif", "residue_type",
                       "track", "kl_percentile", "cell_type", "tissue"],
        numeric_cols=["kl_percentile"],
        key_cols=[],
        skip_dup_check_reason="Substrate sets: multiple rows per kinase.",
    )

    # pseudobulk cell counts
    rel_pb = f"{ct}/fivexfad_snrna_pseudobulk_counts.csv"
    _validate_csv_schema(
        report, rel_pb,
        required_cols=["tissue", "age_months", "genotype", "sample_id",
                       "cell_type", "n_cells"],
        numeric_cols=["n_cells"],
        key_cols=[],
        skip_dup_check_reason="Cell counts: no single key column.",
    )


def _check_fivexfad_track_fname_convention(
    report: ValidationReport,
    rel_path: str,
    region: str,
    track_fname: str,  # "st" or "py"
) -> None:
    """Check that 5xFAD file path contains '<region>_<track>' prefix."""
    fname = Path(rel_path).name
    expected_prefix = f"{region}_{track_fname}_"
    if fname.startswith(expected_prefix):
        report.add(rel_path, "track_filename_convention", "PASS",
                   f"Filename starts with expected prefix '{expected_prefix}'.")
    else:
        report.add(rel_path, "track_filename_convention", "FAIL",
                   f"Expected filename prefix '{expected_prefix}', got: {fname}")


# ---------------------------------------------------------------------------
# Dispatch + CLI
# ---------------------------------------------------------------------------

COHORT_VALIDATORS = {
    "song": validate_song,
    "mukesh": validate_mukesh,
    "tcells": validate_tcells,
    "fivexfad": validate_fivexfad,
}

ALL_COHORTS_ORDERED = ["song", "mukesh", "tcells", "fivexfad"]


def run_cohort(cohort: str, output_dir: Path) -> ValidationReport:
    if cohort not in COHORT_VALIDATORS:
        raise ValueError(f"Unknown cohort {cohort!r}. Valid: {sorted(COHORT_VALIDATORS)}")

    report = ValidationReport(cohort=cohort)
    COHORT_VALIDATORS[cohort](report)
    write_reports(report, output_dir)
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Phase-1 cohort validator (read-only)."
    )
    parser.add_argument(
        "--cohort", choices=ALL_COHORTS_ORDERED,
        help="Validate a single cohort.",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Validate all cohorts.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(REPORT_DIR),
        help="Output directory for reports.",
    )
    args = parser.parse_args(argv)

    cohorts_to_run: list[str] = []
    if args.all:
        cohorts_to_run = ALL_COHORTS_ORDERED
    elif args.cohort:
        cohorts_to_run = [args.cohort]
    else:
        parser.error("Specify --cohort <name> or --all.")

    output_dir = Path(args.output_dir)
    all_ok = True

    for cohort in cohorts_to_run:
        report = run_cohort(cohort, output_dir)
        c = report.counts
        status_line = (
            f"{cohort}: PASS={c['PASS']} FAIL={c['FAIL']} "
            f"SKIP={c['SKIP']} DEVIATION={c['DEVIATION']}"
        )
        print(status_line)
        if c["FAIL"] > 0:
            all_ok = False
            for f in report.findings:
                if f.status == "FAIL":
                    print(f"  FAIL  {f.artifact_path} | {f.check_name} | {f.detail}")

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()

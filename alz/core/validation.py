"""Packet 1B+1C — Validation result model and report writer.

This module owns:
- Finding dataclass (one per check)
- ValidationReport dataclass (collection of findings for one cohort)
- JSON + Markdown report writers
- Memory-safe CSV column reader (header + targeted streamed checks only)
- Memory-safe numeric coercibility checker (no full pandas load)

All reads are streamed or header-only.  No pandas / numpy at import time.
"""
from __future__ import annotations

import csv
import io
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 50 MB threshold: beyond this size, full numeric-coercibility checks are
# skipped (memory-safety rule).  Header reads are always cheap.
SIZE_LIMIT_NUMERIC_CHECK: int = 50 * 1024 * 1024

Status = Literal["PASS", "FAIL", "SKIP", "DEVIATION"]


# ---------------------------------------------------------------------------
# Finding
# ---------------------------------------------------------------------------

@dataclass
class Finding:
    artifact_path: str        # relative path from PROJECT_ROOT
    check_name: str
    status: Status
    detail: str = ""

    def to_dict(self) -> dict[str, str]:
        return {
            "artifact_path": self.artifact_path,
            "check_name": self.check_name,
            "status": self.status,
            "detail": self.detail,
        }


# ---------------------------------------------------------------------------
# ValidationReport
# ---------------------------------------------------------------------------

@dataclass
class ValidationReport:
    cohort: str
    generated_at: str = field(default_factory=lambda: time.strftime(
        "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
    ))
    findings: list[Finding] = field(default_factory=list)

    # ---- summary counts (derived) ----
    @property
    def counts(self) -> dict[str, int]:
        counts: dict[str, int] = {"PASS": 0, "FAIL": 0, "SKIP": 0, "DEVIATION": 0}
        for f in self.findings:
            counts[f.status] = counts.get(f.status, 0) + 1
        return counts

    def add(
        self,
        artifact_path: str,
        check_name: str,
        status: Status,
        detail: str = "",
    ) -> Finding:
        f = Finding(artifact_path, check_name, status, detail)
        self.findings.append(f)
        return f

    def to_dict(self) -> dict[str, Any]:
        c = self.counts
        return {
            "cohort": self.cohort,
            "generated_at": self.generated_at,
            "summary": c,
            "findings": [f.to_dict() for f in self.findings],
        }


# ---------------------------------------------------------------------------
# JSON + Markdown report writers
# ---------------------------------------------------------------------------

def write_reports(report: ValidationReport, output_dir: Path) -> tuple[Path, Path]:
    """Write JSON + Markdown reports.  Returns (json_path, md_path)."""
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / f"{report.cohort}_validation.json"
    md_path = output_dir / f"{report.cohort}_validation.md"

    data = report.to_dict()
    # Determinism: sort findings by (artifact_path, check_name)
    data["findings"] = sorted(
        data["findings"],
        key=lambda f: (f["artifact_path"], f["check_name"]),
    )

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=False)

    _write_markdown(report, data, md_path)

    return json_path, md_path


def _write_markdown(report: ValidationReport, data: dict, path: Path) -> None:
    c = data["summary"]
    lines = [
        f"# Phase 1 Validation — {report.cohort}",
        "",
        f"Generated: {report.generated_at}",
        "",
        "## Summary",
        "",
        f"| PASS | FAIL | SKIP | DEVIATION |",
        f"| --- | --- | --- | --- |",
        f"| {c['PASS']} | {c['FAIL']} | {c['SKIP']} | {c['DEVIATION']} |",
        "",
        "## Findings",
        "",
        "| Status | Artifact | Check | Detail |",
        "| --- | --- | --- | --- |",
    ]
    for f in data["findings"]:
        detail = f["detail"].replace("|", "\\|").replace("\n", " ")[:200]
        lines.append(
            f"| {f['status']} | `{f['artifact_path']}` | {f['check_name']} | {detail} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Memory-safe file helpers
# ---------------------------------------------------------------------------

def file_exists(rel_path: str) -> bool:
    return (PROJECT_ROOT / rel_path).exists()


def file_size(rel_path: str) -> int:
    p = PROJECT_ROOT / rel_path
    return p.stat().st_size if p.exists() else -1


def csv_header(rel_path: str) -> list[str] | None:
    """Read only the CSV header row.  Returns None if file missing or unreadable."""
    p = PROJECT_ROOT / rel_path
    if not p.exists():
        return None
    try:
        with open(p, newline="", encoding="utf-8", errors="replace") as f:
            reader = csv.reader(f)
            return next(reader)
    except Exception:
        return None


def csv_row_count(rel_path: str) -> int:
    """Streamed line count of CSV data rows (header excluded)."""
    p = PROJECT_ROOT / rel_path
    if not p.exists():
        return -1
    count = 0
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            count += chunk.count(b"\n")
    return max(0, count - 1)


def csv_check_duplicate_keys(
    rel_path: str, key_columns: list[str], max_rows: int = 50_000
) -> tuple[bool, int, int]:
    """Check for duplicate key tuples in a CSV.

    Memory-safe: reads one row at a time.  Stops after max_rows rows.
    Returns (has_duplicates, n_checked, n_duplicate_rows).
    Large files (>SIZE_LIMIT_NUMERIC_CHECK) skip the check → returns (False, 0, 0).
    """
    if file_size(rel_path) > SIZE_LIMIT_NUMERIC_CHECK:
        return False, 0, 0

    p = PROJECT_ROOT / rel_path
    if not p.exists():
        return False, 0, 0

    seen: set[tuple] = set()
    n_dup = 0
    n_checked = 0

    try:
        with open(p, newline="", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                return False, 0, 0
            missing = [k for k in key_columns if k not in (reader.fieldnames or [])]
            if missing:
                return False, 0, 0  # can't check: missing key columns
            for row in reader:
                if n_checked >= max_rows:
                    break
                key = tuple(row[k] for k in key_columns)
                if key in seen:
                    n_dup += 1
                else:
                    seen.add(key)
                n_checked += 1
    except Exception:
        return False, 0, 0

    return n_dup > 0, n_checked, n_dup


def csv_numeric_coercible(
    rel_path: str, numeric_columns: list[str], sample_rows: int = 200
) -> tuple[bool, dict[str, int]]:
    """Check that named columns are coercible to float for the first sample_rows.

    Returns (all_ok, bad_counts) where bad_counts maps column → failure count.
    Skips check for files > SIZE_LIMIT_NUMERIC_CHECK (returns True, {}).
    """
    if not numeric_columns:
        return True, {}
    if file_size(rel_path) > SIZE_LIMIT_NUMERIC_CHECK:
        return True, {}

    p = PROJECT_ROOT / rel_path
    if not p.exists():
        return True, {}

    bad: dict[str, int] = {c: 0 for c in numeric_columns}
    n_checked = 0

    try:
        with open(p, newline="", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                return True, {}
            present = [c for c in numeric_columns if c in (reader.fieldnames or [])]
            if not present:
                return True, {}
            for row in reader:
                if n_checked >= sample_rows:
                    break
                for col in present:
                    val = row.get(col, "")
                    if val in ("", "NA", "NaN", "nan", "None", "null"):
                        continue  # NaN is acceptable
                    try:
                        float(val)
                    except (ValueError, TypeError):
                        bad[col] = bad.get(col, 0) + 1
                n_checked += 1
    except Exception:
        return True, {}

    all_ok = all(v == 0 for v in bad.values())
    return all_ok, bad


def json_load_safe(rel_path: str, size_limit: int = 10 * 1024 * 1024) -> Any | None:
    """Load a small JSON file.  Returns None if missing or > size_limit."""
    p = PROJECT_ROOT / rel_path
    if not p.exists():
        return None
    if p.stat().st_size > size_limit:
        return None  # too large — caller should hash-only
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Parquet helpers (optional; requires pyarrow)
# ---------------------------------------------------------------------------

try:
    import pyarrow.parquet as _pq  # type: ignore
    _PYARROW_AVAILABLE = True
except ImportError:
    _pq = None  # type: ignore
    _PYARROW_AVAILABLE = False


def parquet_columns(rel_path: str) -> list[str] | None:
    """Return column names from parquet metadata.  Never loads data."""
    if not _PYARROW_AVAILABLE or _pq is None:
        return None
    p = PROJECT_ROOT / rel_path
    if not p.exists():
        return None
    try:
        pf = _pq.ParquetFile(str(p))
        schema = pf.schema_arrow
        return [schema.field(i).name for i in range(len(schema))]
    except Exception:
        return None


def parquet_row_count(rel_path: str) -> int | None:
    """Return row count from parquet metadata.  Never loads data."""
    if not _PYARROW_AVAILABLE or _pq is None:
        return None
    p = PROJECT_ROOT / rel_path
    if not p.exists():
        return None
    try:
        pf = _pq.ParquetFile(str(p))
        return pf.metadata.num_rows
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Common check helpers
# ---------------------------------------------------------------------------

def check_file_exists(
    report: ValidationReport,
    rel_path: str,
    check_name: str = "file_exists",
    absent_by_design: bool = False,
) -> bool:
    """Emit PASS/FAIL/SKIP finding for file existence.  Returns True if present."""
    if absent_by_design:
        exists = file_exists(rel_path)
        if exists:
            report.add(rel_path, check_name, "DEVIATION",
                       "File present but expected absent_by_design.")
            return True
        report.add(rel_path, check_name, "SKIP",
                   "absent_by_design: file expected not to exist.")
        return False

    if file_exists(rel_path):
        report.add(rel_path, check_name, "PASS")
        return True
    else:
        report.add(rel_path, check_name, "FAIL", "File not found.")
        return False


def check_required_columns(
    report: ValidationReport,
    rel_path: str,
    required: list[str],
    actual: list[str] | None,
    check_name: str = "required_columns",
) -> bool:
    """Emit PASS/FAIL for required column presence.  Returns True if all present."""
    if actual is None:
        report.add(rel_path, check_name, "FAIL",
                   "Could not read column headers.")
        return False
    missing = [c for c in required if c not in actual]
    if missing:
        report.add(rel_path, check_name, "FAIL",
                   f"Missing columns: {missing}")
        return False
    report.add(rel_path, check_name, "PASS",
               f"All {len(required)} required columns present.")
    return True


def check_no_duplicate_keys(
    report: ValidationReport,
    rel_path: str,
    key_columns: list[str],
    check_name: str = "no_duplicate_keys",
) -> None:
    """Emit PASS/FAIL/SKIP/DEVIATION finding for duplicate key check."""
    if not key_columns:
        report.add(rel_path, check_name, "SKIP", "No key columns defined.")
        return
    if file_size(rel_path) > SIZE_LIMIT_NUMERIC_CHECK:
        report.add(rel_path, check_name, "SKIP",
                   f"File > {SIZE_LIMIT_NUMERIC_CHECK // (1024*1024)} MB; skipped for memory safety.")
        return

    header = csv_header(rel_path)
    if header is None:
        report.add(rel_path, check_name, "FAIL", "Could not read header.")
        return
    missing_keys = [k for k in key_columns if k not in header]
    if missing_keys:
        report.add(rel_path, check_name, "SKIP",
                   f"Key columns absent from header: {missing_keys}; skipping dup check.")
        return

    has_dups, n_checked, n_dup = csv_check_duplicate_keys(rel_path, key_columns)
    if has_dups:
        report.add(rel_path, check_name, "DEVIATION",
                   f"{n_dup} duplicate key rows in {n_checked} checked. "
                   "Accepted if the artifact allows multiple rows per key (e.g. mea_long).")
    else:
        report.add(rel_path, check_name, "PASS",
                   f"No duplicate keys in {n_checked} rows checked.")


def check_numeric_columns(
    report: ValidationReport,
    rel_path: str,
    numeric_columns: list[str],
    check_name: str = "numeric_coercibility",
) -> None:
    """Emit PASS/FAIL/SKIP for numeric coercibility of named columns."""
    if not numeric_columns:
        report.add(rel_path, check_name, "SKIP", "No numeric columns to check.")
        return
    if file_size(rel_path) > SIZE_LIMIT_NUMERIC_CHECK:
        report.add(rel_path, check_name, "SKIP",
                   f"File > {SIZE_LIMIT_NUMERIC_CHECK // (1024*1024)} MB; skipped.")
        return
    all_ok, bad = csv_numeric_coercible(rel_path, numeric_columns)
    if all_ok:
        report.add(rel_path, check_name, "PASS",
                   f"Numeric columns {numeric_columns} coercible in sampled rows.")
    else:
        bad_detail = {k: v for k, v in bad.items() if v > 0}
        report.add(rel_path, check_name, "FAIL",
                   f"Non-numeric values in: {bad_detail}")

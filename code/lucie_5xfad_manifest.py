#!/usr/bin/env python3

"""Build a reproducible manifest for local Lucie 5xFAD `.sne` files."""

from __future__ import annotations

import json
import re
import tarfile
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "lucie_proteomics"
OUT_PATH = ROOT / "docs" / "integrations" / "5xfad-lucie-manifest.json"
SCAN_BYTES = 65536


@dataclass
class ManifestEntry:
    path: str
    size_bytes: int
    modality_guess: str
    tissue_guess: str
    xic_variant: bool
    pooled_runs_present: bool
    age_labels: list[str]
    embedded_raw_run_count: int
    embedded_raw_runs: list[str]
    genotype_labels_present_in_runs: bool
    sex_labels_present_in_runs: bool
    zip_container: bool
    tar_container: bool
    spectronaut_field_terms_found: list[str]
    first_utf16_header_excerpt: str
    extraction_readiness: str
    recommended_extraction_path: str
    notes: list[str]


def modality_guess(path: Path) -> str:
    name = path.name.lower()
    folder = path.parent.name.lower()
    if "total" in name or "total" in folder:
        return "pr"
    if "py" in name or "py" in folder:
        return "py"
    if "ack" in name or "ack" in folder:
        return "AcK"
    return "unknown"


def tissue_guess(path: Path) -> str:
    full = str(path).lower()
    if "cortex" in full or "cort_" in full or "ctx_" in full:
        return "cortex"
    if "hippocampus" in full or "hip_" in full:
        return "hippocampus"
    return "unknown"


def decode_header(path: Path) -> str:
    with path.open("rb") as handle:
        buf = handle.read(SCAN_BYTES)
    return buf.decode("utf-16le", errors="ignore").replace("\x00", "")


def extract_runs(text: str) -> list[str]:
    raw_runs = re.findall(r"([0-9A-Za-z]{6,}_LD_[0-9A-Za-z_\-]+\.raw)", text)
    normalized = []
    for run in raw_runs:
        prefix, rest = run.split("_LD_", 1)
        digits = "".join(ch for ch in prefix if ch.isdigit())
        if len(digits) >= 6:
            prefix = digits[-6:]
        normalized.append(f"{prefix}_LD_{rest}")
    return list(dict.fromkeys(normalized))


def extract_ages(runs: list[str]) -> list[str]:
    ages = sorted({match.group(1) for run in runs for match in re.finditer(r"_M(\d+)_", run)})
    return [f"M{age}" for age in ages]


def build_entry(path: Path) -> ManifestEntry:
    header = decode_header(path)
    runs = extract_runs(header)
    field_terms = [
        term
        for term in [
            "Quantity",
            "Protein",
            "Gene",
            "Spectronaut",
            "Report",
            "PG.",
            "FG.",
            "XIC",
            "PTM",
            "Intensity",
            "LFQ",
        ]
        if term in header
    ]
    notes = [
        "Local file is not plain-text tabular input for the current R parser.",
        "Manifest reflects embedded UTF-16LE-readable header content only.",
    ]
    if not zipfile.is_zipfile(path):
        notes.append("Archive probe: not a ZIP container.")
    if not tarfile.is_tarfile(path):
        notes.append("Archive probe: not a TAR container.")
    if not field_terms:
        notes.append("No obvious Spectronaut report field names recovered from the scanned header window.")
    if not runs:
        notes.append("No embedded `.raw` run names recovered from the scanned header window.")
    if not re.search(r"\b(WT|WildT|5XFAD|5X)\b", " ".join(runs), flags=re.IGNORECASE):
        notes.append("Genotype labels are not explicit in recovered run names.")
    return ManifestEntry(
        path=str(path.relative_to(ROOT)),
        size_bytes=path.stat().st_size,
        modality_guess=modality_guess(path),
        tissue_guess=tissue_guess(path),
        xic_variant="xic" in path.name.lower(),
        pooled_runs_present=any("pool" in run.lower() for run in runs),
        age_labels=extract_ages(runs),
        embedded_raw_run_count=len(runs),
        embedded_raw_runs=runs,
        genotype_labels_present_in_runs=bool(
            re.search(r"\b(WT|WildT|5XFAD|5X)\b", " ".join(runs), flags=re.IGNORECASE)
        ),
        sex_labels_present_in_runs=bool(re.search(r"\b(M|male|F|female)\b", " ".join(runs), flags=re.IGNORECASE)),
        zip_container=zipfile.is_zipfile(path),
        tar_container=tarfile.is_tarfile(path),
        spectronaut_field_terms_found=field_terms,
        first_utf16_header_excerpt=header[:500],
        extraction_readiness=(
            "blocked_on_export_or_binary_extraction"
            if runs
            else "blocked_on_format_identification"
        ),
        recommended_extraction_path=(
            "export_readable_report_from_spectronaut_or_use_vendor_supported_export_step"
        ),
        notes=notes,
    )


def main() -> None:
    sne_files = sorted(DATA_DIR.glob("5xFAD*/*.sne"))
    entries = [build_entry(path) for path in sne_files]
    payload = {
        "generated_from": str(Path(__file__).resolve().relative_to(ROOT)),
        "data_root": str(DATA_DIR.relative_to(ROOT)),
        "scan_bytes": SCAN_BYTES,
        "file_count": len(entries),
        "entries": [asdict(entry) for entry in entries],
    }
    OUT_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

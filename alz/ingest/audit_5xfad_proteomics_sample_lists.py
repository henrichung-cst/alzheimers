#!/usr/bin/env python3
"""Audit delivered 5xFAD proteomics DOCX sample lists against the genotype map."""

from __future__ import annotations

import argparse
import csv
import re
import sys
import zipfile
from pathlib import Path
from xml.etree import ElementTree

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from alz.cohorts.fivexfad.ingest import GENOTYPE_BY_AGE_SAMPLE  # noqa: E402


REPO = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO / "outputs/reports/5xfad_proteomics_sample_list_audit.csv"
DOCX_SOURCES = {
    "cortex": REPO / "data/raw/external/lucie_proteomics/reports/Male Cortex/5xFAD Male Cortex Sample List.docx",
    "hippocampus": REPO / "data/raw/external/lucie_proteomics/reports/Male Hippocampus/Sample IDs 5xFAD Male Hippocampus.docx",
}


FIELDS = [
    "tissue",
    "source_docx",
    "record_type",
    "age_months",
    "sample",
    "delivered_genotype",
    "code_genotype",
    "code_matches_delivered",
    "pool_members",
    "provenance_note",
]


def docx_text(path: Path) -> str:
    with zipfile.ZipFile(path) as archive:
        xml = archive.read("word/document.xml")
    root = ElementTree.fromstring(xml)
    words = [
        node.text
        for node in root.iter()
        if node.tag.endswith("}t") and node.text is not None
    ]
    return " ".join(words)


def parse_individuals(text: str) -> list[tuple[int, str, str]]:
    compact = re.sub(r"\s+", " ", text)
    rows: list[tuple[int, str, str]] = []
    patterns = [
        r"M(?P<age>3|6|9|12)\s*#\s*(?P<sample>\d+)\s*(?P<genotype>WT|TG)",
        r"M(?P<age>3|6|9|12)\s*_\s*(?P<sample>\d+)\s*_\s*(?P<genotype>WT|TG)",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, compact, flags=re.IGNORECASE):
            rows.append((
                int(match.group("age")),
                str(int(match.group("sample"))),
                match.group("genotype").upper(),
            ))
    return sorted(set(rows))


def parse_pools(text: str) -> list[tuple[int, str, str]]:
    compact = re.sub(r"\s+", " ", text)
    pools: list[tuple[int, str, str]] = []
    pattern = r"M(?P<age>3|6|9|12)\s+Pool\s*\((?P<members>[^)]+)\)\s*(?P<genotype>WT|TG)"
    for match in re.finditer(pattern, compact, flags=re.IGNORECASE):
        members = ",".join(re.findall(r"\d+", match.group("members")))
        pools.append((int(match.group("age")), members, match.group("genotype").upper()))
    return pools


def build_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for tissue, path in DOCX_SOURCES.items():
        text = docx_text(path)
        source = str(path.relative_to(REPO))
        for age, sample, delivered in parse_individuals(text):
            code = GENOTYPE_BY_AGE_SAMPLE.get(age, {}).get(sample, "")
            rows.append({
                "tissue": tissue,
                "source_docx": source,
                "record_type": "individual",
                "age_months": str(age),
                "sample": sample,
                "delivered_genotype": delivered,
                "code_genotype": code,
                "code_matches_delivered": str(code == delivered),
                "pool_members": "",
                "provenance_note": "Delivered DOCX individual sample label.",
            })
        for age, members, delivered in parse_pools(text):
            rows.append({
                "tissue": tissue,
                "source_docx": source,
                "record_type": "pool",
                "age_months": str(age),
                "sample": "",
                "delivered_genotype": delivered,
                "code_genotype": "",
                "code_matches_delivered": "",
                "pool_members": members,
                "provenance_note": "Delivered DOCX pooled sample label; retained as provenance, not an independent mouse replicate.",
            })
    return sorted(rows, key=lambda r: (r["tissue"], r["record_type"], int(r["age_months"]), r["sample"], r["pool_members"]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    rows = build_rows()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    mismatches = [
        row for row in rows
        if row["record_type"] == "individual" and row["code_matches_delivered"] != "True"
    ]
    print(f"[audit-5xfad-proteomics-sample-lists] wrote {args.out}")
    print(f"[audit-5xfad-proteomics-sample-lists] rows={len(rows)} individual_mismatches={len(mismatches)}")
    if mismatches:
        for row in mismatches:
            print(
                "[audit-5xfad-proteomics-sample-lists] mismatch "
                f"{row['tissue']} M{row['age_months']} sample {row['sample']}: "
                f"delivered={row['delivered_genotype']} code={row['code_genotype']}"
            )
        raise SystemExit(1)


if __name__ == "__main__":
    main()

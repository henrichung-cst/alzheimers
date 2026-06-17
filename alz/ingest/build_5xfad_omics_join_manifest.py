#!/usr/bin/env python3
"""Build the 5xFAD transcriptomics/proteomics sample join manifest."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
DEFAULT_OBS = REPO / "data/datasets/5xFAD/primary/scrna/obs_df.csv"
DEFAULT_PROTEOMICS = REPO / "outputs/reports/kinase_attribution_5xfad/sample_manifest.csv"
DEFAULT_OUT = REPO / "data/datasets/5xFAD/metadata/omics_join_manifest.csv"


AUDIT_HOLD_CANDIDATES = {
    "WildT_06mo_C_11": (
        "cortex_6mo_WT_11",
        "audit_hold_pooled_only",
        "delivered_cortex_sample_list_pool_only",
        "exclude_until_resolved",
        "The delivered cortex proteomics sample list identifies M6 sample 11 as WT, but only inside the M6 Pool (11 + 14) WT entry. No individual cortex proteomics raw run for sample 11 is present in the local reports, so this row is excluded from per-animal integration.",
    ),
}


FIELDS = [
    "transcriptomics_sample_id",
    "proposed_proteomics_biological_sample_id",
    "tissue",
    "age",
    "transcriptomics_genotype",
    "local_proteomics_genotype",
    "join_status",
    "evidence_class",
    "per_animal_integration_action",
    "provenance_note",
]


def parse_tx_sample(sample: str) -> dict[str, str]:
    match = re.fullmatch(r"(5XFAD|WildT)_(\d{2}mo)_([CH])_(\d+)", sample)
    if not match:
        raise ValueError(f"unexpected transcriptomics sample ID: {sample!r}")
    genotype_raw, age_raw, tissue_raw, sample_no_raw = match.groups()
    genotype = "TG" if genotype_raw == "5XFAD" else "WT"
    tissue = "cortex" if tissue_raw == "C" else "hippocampus"
    age = f"{int(age_raw[:2])}mo"
    sample_no = str(int(sample_no_raw))
    normalized = f"{tissue}_{age}_{genotype}_{sample_no}"
    return {
        "tissue": tissue,
        "age": age,
        "transcriptomics_genotype": genotype,
        "normalized_id": normalized,
    }


def load_tx_samples(path: Path) -> list[str]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if "sample" not in (reader.fieldnames or []):
            raise ValueError(f"{path} has no 'sample' column")
        return sorted({row["sample"] for row in reader if row.get("sample")})


def load_proteomics_by_id(path: Path) -> dict[str, str]:
    by_id: dict[str, str] = {}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"biological_sample_id", "genotype"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} missing columns: {sorted(missing)}")
        for row in reader:
            bio_id = row.get("biological_sample_id", "")
            genotype = row.get("genotype", "")
            if bio_id and genotype:
                by_id.setdefault(bio_id, genotype)
    return by_id


def direct_note(sample: str, parsed: dict[str, str]) -> str:
    tissue_code = "C" if parsed["tissue"] == "cortex" else "H"
    return (
        "Direct normalized sample-label match: "
        f"{sample.split('_', 1)[0]}->{parsed['transcriptomics_genotype']}, "
        f"{tissue_code}->{parsed['tissue']}, zero-padded sample number collapsed where present."
    )


def build_rows(obs: Path, proteomics: Path) -> list[dict[str, str]]:
    samples = load_tx_samples(obs)
    proteomics_by_id = load_proteomics_by_id(proteomics)
    rows: list[dict[str, str]] = []

    for sample in samples:
        parsed = parse_tx_sample(sample)
        target = parsed["normalized_id"]
        if target in proteomics_by_id:
            status = "direct"
            evidence_class = "direct_normalized_name_match"
            action = "use"
            note = direct_note(sample, parsed)
        elif sample in AUDIT_HOLD_CANDIDATES:
            target, status, evidence_class, action, note = AUDIT_HOLD_CANDIDATES[sample]
        else:
            status = "unresolved"
            evidence_class = "unresolved"
            action = "exclude_until_resolved"
            note = "No direct or audited local proteomics mapping is currently defined."

        rows.append(
            {
                "transcriptomics_sample_id": sample,
                "proposed_proteomics_biological_sample_id": target,
                "tissue": parsed["tissue"],
                "age": parsed["age"],
                "transcriptomics_genotype": parsed["transcriptomics_genotype"],
                "local_proteomics_genotype": proteomics_by_id.get(target, ""),
                "join_status": status,
                "evidence_class": evidence_class,
                "per_animal_integration_action": action,
                "provenance_note": note,
            }
        )

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--obs", type=Path, default=DEFAULT_OBS)
    parser.add_argument("--proteomics-manifest", type=Path, default=DEFAULT_PROTEOMICS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    rows = build_rows(args.obs, args.proteomics_manifest)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    counts: dict[str, int] = {}
    for row in rows:
        counts[row["join_status"]] = counts.get(row["join_status"], 0) + 1
    print(f"[5xfad-omics-join] wrote {args.out}")
    print(f"[5xfad-omics-join] rows={len(rows)} status_counts={counts}")


if __name__ == "__main__":
    main()

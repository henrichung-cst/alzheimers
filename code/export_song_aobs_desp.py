#!/usr/bin/env python3
"""Export Song-format deconvoluted omics tables using A_obs + DESP."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, Iterable, List, Optional, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]

DESP_RUNNER = REPO_ROOT / "deconv" / "code" / "run_desp_baseline.R"
RSCRIPT_WRAPPER = REPO_ROOT / "deconv" / "code" / "rscript_mamba.sh"
SONG_PROTEOMICS_DIR = REPO_ROOT / "data" / "incytr_collections" / "song" / "proteomics"
SONG_SOURCE_DIR = SONG_PROTEOMICS_DIR / "source"
SONG_LEGACY_DIR = SONG_PROTEOMICS_DIR / "legacy"
SONG_ROOT_DIR = SONG_PROTEOMICS_DIR.parent
AOBS_SOURCE = (
    REPO_ROOT
    / "data"
    / "incytr_collections"
    / "song"
    / "method_records"
    / "aobs_desp_standardized"
    / "inputs"
    / "A_obs_fractions.tsv"
)
SAMPLE_KEY = SONG_ROOT_DIR / "source" / "metadata" / "yuyu_samplekey.csv"

AOBS_LABEL_MAP = {
    "Astrocytes": "Astrocytes",
    "Endothelial cells": "Endothelial cells",
    "Excitatory neurons": "Glut",
    "Interneurons": "Gaba",
    "Medium spiny neurons": "Medium spiny neurons",
    "Microglia": "Microglia",
    "Oligodendrocytes": "Oligodendrocytes",
    "OPCs": "OPCs",
}

MODALITY_SPECS = {
    "pr": {
        "bulk_path": SONG_SOURCE_DIR / "pr_median.csv",
        "template_path": SONG_LEGACY_DIR / "pr_yuyu_deconvoluted.csv",
        "metadata_fields": ["protein_id", "Gene Symbol", "geneID"],
        "sample_start": 5,
        "default_output_path": SONG_PROTEOMICS_DIR / "pr_yuyu_deconvoluted.csv",
    },
    "ps": {
        "bulk_path": SONG_SOURCE_DIR / "imac_median.csv",
        "template_path": SONG_LEGACY_DIR / "ps_yuyu_deconvoluted.csv",
        "metadata_fields": ["site_id", "protein_id", "gene_symbol", "prot_description", "site_position", "motif"],
        "sample_start": 8,
        "default_output_path": SONG_PROTEOMICS_DIR / "ps_yuyu_deconvoluted.csv",
    },
    "py": {
        "bulk_path": SONG_SOURCE_DIR / "py_median.csv",
        "template_path": SONG_LEGACY_DIR / "py_yuyu_deconvoluted.csv",
        "metadata_fields": ["protein_id", "gene_symbol", "prot_description", "site_position", "motif", "gene_id"],
        "sample_start": 8,
        "default_output_path": SONG_PROTEOMICS_DIR / "py_yuyu_deconvoluted.csv",
    },
}


def load_csv_rows(path: Path) -> Tuple[List[str], List[List[str]]]:
    with path.open(newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = [row for row in reader]
    return header, rows


def load_sample_map(path: Path) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            mapping[row["MS_ID"]] = row["Group"]
    return mapping


def parse_optional_float(raw: str) -> Optional[float]:
    if raw in ("", "NA", "NaN", "nan"):
        return None
    return float(raw)


def load_aobs_song(path: Path) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    with path.open(newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            sample_id = row["sample_id"]
            vals: Dict[str, float] = defaultdict(float)
            for src_label, dst_label in AOBS_LABEL_MAP.items():
                vals[dst_label] += float(row[src_label])
            row_sum = sum(vals.values())
            if row_sum <= 0:
                raise SystemExit(f"A_obs row has non-positive retained mass for sample {sample_id}")
            out[sample_id] = {label: vals[label] / row_sum for label in vals}
    return out


def ordered_unique(items: Iterable[str]) -> List[str]:
    seen = set()
    out = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def parse_template_columns(header: Sequence[str], sample_start: int) -> Tuple[List[str], List[str]]:
    data_cols = list(header[sample_start:])
    sample_ids = ordered_unique(col.rsplit("_", 1)[0] for col in data_cols)
    celltypes = ordered_unique(col.rsplit("_", 1)[1] for col in data_cols)
    return sample_ids, celltypes


def build_template_key(row: Sequence[str], metadata_fields: Sequence[str]) -> Tuple[str, ...]:
    offset = 2
    return tuple(row[offset + i] for i in range(len(metadata_fields)))


def build_raw_key(row: Dict[str, str], metadata_fields: Sequence[str]) -> Tuple[str, ...]:
    return tuple(row[field] for field in metadata_fields)


def write_tsv_matrix(path: Path, header_name: str, col_ids: Sequence[str], rows: Sequence[Tuple[str, Sequence[float]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow([header_name, *col_ids])
        for row_id, values in rows:
            writer.writerow([row_id, *values])


def run_desp(bulk_tsv: Path, proportions_tsv: Path, out_dir: Path, desp_lambda: float, desp_beta: float) -> None:
    cmd = [
        "bash",
        str(RSCRIPT_WRAPPER),
        str(DESP_RUNNER),
        "--lambda",
        str(desp_lambda),
        "--beta",
        str(desp_beta),
        "--bulk",
        str(bulk_tsv),
        "--proportions",
        str(proportions_tsv),
        "--out-dir",
        str(out_dir),
    ]
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), check=False)
    if proc.returncode != 0:
        raise SystemExit(f"DESP command failed ({proc.returncode}): {' '.join(cmd)}")


def read_prediction_matrix(path: Path) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    with path.open(newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)
        celltypes = header[1:]
        for row in reader:
            feature_id = row[0]
            out[feature_id] = {celltype: float(value) for celltype, value in zip(celltypes, row[1:])}
    return out


def materialize_modality(
    modality: str,
    aobs_song: Dict[str, Dict[str, float]],
    sample_map: Dict[str, str],
    desp_lambda: float,
    desp_beta: float,
    keep_intermediates: bool,
) -> Dict[str, object]:
    spec = MODALITY_SPECS[modality]
    template_header, template_rows = load_csv_rows(spec["template_path"])
    template_samples, template_celltypes = parse_template_columns(template_header, spec["sample_start"])

    raw_queue: DefaultDict[Tuple[str, ...], List[Dict[str, str]]] = defaultdict(list)
    with spec["bulk_path"].open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw_queue[build_raw_key(row, spec["metadata_fields"])].append(row)
    raw_row_count = sum(len(queue) for queue in raw_queue.values())

    paired_rows: List[Tuple[List[str], Dict[str, str]]] = []
    missing_keys: List[Tuple[str, ...]] = []
    for template_row in template_rows:
        key = build_template_key(template_row, spec["metadata_fields"])
        if raw_queue[key]:
            paired_rows.append((template_row, raw_queue[key].pop(0)))
        else:
            missing_keys.append(key)

    if missing_keys:
        raise SystemExit(f"{modality}: missing {len(missing_keys)} template rows in raw bulk input")
    unused_raw_rows = sum(len(queue) for queue in raw_queue.values())

    work_root = SONG_PROTEOMICS_DIR / "aobs_desp_work" / modality
    if work_root.exists():
        shutil.rmtree(work_root)
    work_root.mkdir(parents=True, exist_ok=True)

    feature_ids = [f"feature_{idx:06d}" for idx in range(1, len(paired_rows) + 1)]
    sample_items = list(sample_map.items())
    bulk_rows: List[Tuple[str, Sequence[float]]] = []
    missing_mask: Dict[str, Dict[str, bool]] = {}
    n_missing_cells = 0
    for feature_id, (_, raw_row) in zip(feature_ids, paired_rows):
        observed_by_group: Dict[str, Optional[float]] = {}
        observed_values: List[float] = []
        for ms_id, group_id in sample_items:
            value = parse_optional_float(raw_row[ms_id])
            observed_by_group[group_id] = value
            if value is not None:
                observed_values.append(value)
            else:
                n_missing_cells += 1
        fill_value = sum(observed_values) / len(observed_values) if observed_values else 0.0
        bulk_rows.append(
            (
                feature_id,
                [
                    observed_by_group[group_id] if observed_by_group[group_id] is not None else fill_value
                    for _, group_id in sample_items
                ],
            )
        )
        missing_mask[feature_id] = {
            group_id: observed_by_group[group_id] is None for _, group_id in sample_items
        }

    proportions_rows = []
    for _, sample_id in sample_items:
        if sample_id not in aobs_song:
            raise SystemExit(f"{modality}: sample {sample_id} missing from A_obs source")
        proportions_rows.append((sample_id, [aobs_song[sample_id][celltype] for celltype in template_celltypes]))

    bulk_tsv = work_root / "bulk_features_by_samples.tsv"
    proportions_tsv = work_root / "A_obs_song_samples_by_celltypes.tsv"
    desp_out = work_root / "desp_results"
    write_tsv_matrix(bulk_tsv, "feature_id", [group_id for _, group_id in sample_items], bulk_rows)
    write_tsv_matrix(proportions_tsv, "sample_id", template_celltypes, proportions_rows)

    run_desp(bulk_tsv, proportions_tsv, desp_out, desp_lambda, desp_beta)
    prediction = read_prediction_matrix(desp_out / "desp_prediction_features_by_celltypes.tsv")

    output_path = spec["default_output_path"]
    min_value = math.inf
    max_value = -math.inf
    with output_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(template_header)
        for feature_id, (template_row, _) in zip(feature_ids, paired_rows):
            contribs = []
            pred_row = prediction[feature_id]
            for data_col in template_header[spec["sample_start"]:]:
                sample_id, celltype = data_col.rsplit("_", 1)
                if missing_mask[feature_id][sample_id]:
                    contribs.append("NA")
                    continue
                value = aobs_song[sample_id][celltype] * pred_row[celltype]
                if abs(value) < 1e-12:
                    value = 0.0
                min_value = min(min_value, value)
                max_value = max(max_value, value)
                contribs.append(value)
            writer.writerow(template_row[:spec["sample_start"]] + contribs)

    if not keep_intermediates:
        shutil.rmtree(work_root)

    return {
        "modality": modality,
        "bulk_path": str(spec["bulk_path"]),
        "template_path": str(spec["template_path"]),
        "output_path": str(output_path),
        "n_raw_rows": raw_row_count,
        "n_template_rows": len(template_rows),
        "n_unused_raw_rows": unused_raw_rows,
        "n_rows": len(paired_rows),
        "n_samples": len(template_samples),
        "n_celltypes": len(template_celltypes),
        "n_missing_bulk_cells_restored_as_na": n_missing_cells,
        "min_output_value": min_value,
        "max_output_value": max_value,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate Song-format deconvoluted omics with A_obs + DESP")
    parser.add_argument("--modalities", default="pr,ps,py")
    parser.add_argument("--lambda", dest="desp_lambda", type=float, default=1e-7)
    parser.add_argument("--beta", dest="desp_beta", type=float, default=1e-4)
    parser.add_argument("--keep-intermediates", action="store_true")
    args = parser.parse_args()

    requested = [token.strip() for token in args.modalities.split(",") if token.strip()]
    unsupported = [token for token in requested if token not in MODALITY_SPECS]
    if unsupported:
        raise SystemExit(
            "Unsupported or unavailable modalities requested: "
            + ", ".join(unsupported)
            + ". Supported modalities are: "
            + ", ".join(sorted(MODALITY_SPECS))
        )

    aobs_song = load_aobs_song(AOBS_SOURCE)
    sample_map = load_sample_map(SAMPLE_KEY)

    results = [
        materialize_modality(
            modality=modality,
            aobs_song=aobs_song,
            sample_map=sample_map,
            desp_lambda=args.desp_lambda,
            desp_beta=args.desp_beta,
            keep_intermediates=args.keep_intermediates,
        )
        for modality in requested
    ]

    status = {
        "workflow": "A_obs + DESP",
        "source_dir": str(SONG_SOURCE_DIR),
        "legacy_dir": str(SONG_LEGACY_DIR),
        "aobs_source": str(AOBS_SOURCE),
        "sample_key": str(SAMPLE_KEY),
        "modalities_regenerated": results,
        "hyperparameters": {
            "desp_lambda": args.desp_lambda,
            "desp_beta": args.desp_beta,
        },
    }
    status_path = SONG_PROTEOMICS_DIR / "aobs_desp_status.json"
    status_path.write_text(json.dumps(status, indent=2) + "\n")
    print(f"Wrote: {status_path}")
    for item in results:
        print(f"Wrote: {item['output_path']}")


if __name__ == "__main__":
    main()

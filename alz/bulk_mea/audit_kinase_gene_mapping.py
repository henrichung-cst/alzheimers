#!/usr/bin/env python3
"""Audit generated AD/Song kinase tables against the canonical gene map."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from alz.shared import config


DEFAULT_CSV_PATHS = (
    os.path.join(config.KINASE_ATTRIBUTION_OUTPUT_DIR, "unified_attribution_full.csv"),
    os.path.join(config.KINASE_ATTRIBUTION_OUTPUT_DIR, "unified_attribution.csv"),
    os.path.join(config.ATTRIBUTION_RECOVERY_OUTPUT_DIR, "kinase_activity_matrix.csv"),
    os.path.join(config.ATTRIBUTION_RECOVERY_OUTPUT_DIR, "kinase_hypothesis_table.csv"),
    os.path.join(config.ATTRIBUTION_RECOVERY_OUTPUT_DIR, "celltype_evidence_table.csv"),
)
DEFAULT_PAYLOAD_PATH = os.path.join(
    config.REPO_ROOT, "outputs", "reports", "unified_viewer",
    "unified_viewer.payload.json",
)


def _load_gene_map() -> dict[str, str]:
    if not os.path.exists(config.MAPPING_CACHE_FILE):
        raise FileNotFoundError(
            f"canonical kinase mapping cache not found: {config.MAPPING_CACHE_FILE}"
        )
    df = pd.read_csv(config.MAPPING_CACHE_FILE)
    required = {"kinase_abbreviation", "gene_symbol"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{config.MAPPING_CACHE_FILE} missing required columns: {sorted(missing)}"
        )
    return dict(zip(df["kinase_abbreviation"].astype(str),
                    df["gene_symbol"].astype(str)))


def _audit_pairs(label: str, pairs: pd.DataFrame,
                 gene_map: dict[str, str]) -> pd.DataFrame:
    if pairs.empty:
        return pd.DataFrame(columns=["source", "kinase", "gene_symbol", "expected"])
    pairs = pairs[["kinase", "gene_symbol"]].drop_duplicates().copy()
    pairs["kinase"] = pairs["kinase"].astype(str)
    pairs["gene_symbol"] = pairs["gene_symbol"].astype(str)
    pairs["expected"] = pairs["kinase"].map(lambda k: gene_map.get(k, k))
    mismatches = pairs[pairs["gene_symbol"] != pairs["expected"]].copy()
    mismatches.insert(0, "source", label)
    return mismatches.sort_values(["source", "kinase"])


def _read_csv_pairs(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        print(f"SKIP missing {path}")
        return pd.DataFrame(columns=["kinase", "gene_symbol"])
    header = pd.read_csv(path, nrows=0)
    if not {"kinase", "gene_symbol"}.issubset(header.columns):
        print(f"SKIP no kinase/gene_symbol columns {path}")
        return pd.DataFrame(columns=["kinase", "gene_symbol"])
    return pd.read_csv(path, usecols=["kinase", "gene_symbol"])


def _read_payload_pairs(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        print(f"SKIP missing {path}")
        return pd.DataFrame(columns=["kinase", "gene_symbol"])
    with open(path) as f:
        payload = json.load(f)
    kinases = payload.get("kinases", {})
    blocks = kinases.get("by_context") if isinstance(kinases, dict) else None
    rows = []
    if blocks:
        for ctx_id, block in blocks.items():
            names = block.get("name", [])
            genes = block.get("gene_symbol", [])
            for kinase, gene in zip(names, genes):
                rows.append({
                    "context": ctx_id,
                    "kinase": kinase,
                    "gene_symbol": gene,
                })
    else:
        names = kinases.get("name", []) if isinstance(kinases, dict) else []
        genes = kinases.get("gene_symbol", []) if isinstance(kinases, dict) else []
        for kinase, gene in zip(names, genes):
            rows.append({"kinase": kinase, "gene_symbol": gene})
    return pd.DataFrame(rows, columns=["kinase", "gene_symbol"])


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check generated AD/Song kinase outputs against the canonical "
                    "kinase-to-gene mapping cache.",
    )
    parser.add_argument("--csv", action="append", default=[],
                        help="Additional CSV file with kinase and gene_symbol columns.")
    parser.add_argument("--payload", default=DEFAULT_PAYLOAD_PATH,
                        help="Viewer payload JSON to audit.")
    parser.add_argument("--no-defaults", action="store_true",
                        help="Only audit explicitly provided --csv/--payload paths.")
    parser.add_argument("--warn-only", action="store_true",
                        help="Report mismatches but exit 0.")
    args = parser.parse_args()

    gene_map = _load_gene_map()
    reports = []
    csv_paths = list(args.csv) if args.no_defaults else list(DEFAULT_CSV_PATHS) + list(args.csv)
    for path in csv_paths:
        pairs = _read_csv_pairs(path)
        reports.append(_audit_pairs(path, pairs, gene_map))
    if args.payload:
        pairs = _read_payload_pairs(args.payload)
        reports.append(_audit_pairs(args.payload, pairs, gene_map))

    mismatches = pd.concat(reports, ignore_index=True) if reports else pd.DataFrame()
    if mismatches.empty:
        print("PASS kinase-gene mappings match canonical cache")
        return 0

    print(f"FAIL {len(mismatches)} kinase-gene mapping mismatch row(s)")
    print(mismatches.to_string(index=False))
    return 0 if args.warn_only else 1


if __name__ == "__main__":
    raise SystemExit(main())

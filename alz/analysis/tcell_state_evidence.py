#!/usr/bin/env python3
"""Summarize auditable evidence for cycle-independent per-cell T-cell labels."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from alz.analysis.tcell_marker_sets import (  # noqa: E402
    COLLAPSED_STATE_LABELS,
    PER_CELL_STATE_DEFINITIONS,
)
from alz.analysis.tcell_percell_auroc import mann_whitney_auroc  # noqa: E402


ROOT = Path("outputs/reports/tcell_labeling")
OUTPUT = ROOT / "percell_evidence"
DONORS = ("donor1", "donor2")


def _expression_column(gene: str) -> str:
    return f"{gene}_log_normalized_expression"


def summarize_donor(donor: str, cells: pd.DataFrame) -> dict[str, pd.DataFrame]:
    state_by_day = (
        cells.groupby(["donor", "lineage", "day", "label", "type"], dropna=False)
        .size()
        .rename("n_cells")
        .reset_index()
    )
    panel_rows = []
    gene_rows = []
    for state, definition in PER_CELL_STATE_DEFINITIONS.items():
        if state in COLLAPSED_STATE_LABELS:
            continue
        lineage_cells = cells[cells["lineage"].eq(definition.lineage)]
        target = lineage_cells["label"].eq(state)
        comparison = ~target
        if not target.any() or not comparison.any():
            continue
        modules = [
            (module, "higher") for module in definition.positive_modules
        ] + [
            (module, "lower") for module in definition.negative_modules
        ]
        for module, direction in modules:
            columns = [_expression_column(gene) for gene in module.genes]
            module_value = lineage_cells[columns].mean(axis=1)
            oriented = module_value if direction == "higher" else -module_value
            panel_rows.append(
                {
                    "donor": donor,
                    "lineage": definition.lineage,
                    "state": state,
                    "module": module.name,
                    "expected_direction": direction,
                    "markers": ";".join(module.genes),
                    "n_markers": len(module.genes),
                    "comparison": f"other marker-derived {definition.lineage} states",
                    "n_cells_target": int(target.sum()),
                    "n_cells_comparison": int(comparison.sum()),
                    "oriented_panel_auroc": mann_whitney_auroc(
                        oriented.to_numpy(), target.to_numpy()
                    ),
                }
            )
            for gene, column in zip(module.genes, columns):
                value = lineage_cells[column]
                oriented_gene = value if direction == "higher" else -value
                gene_rows.append(
                    {
                        "donor": donor,
                        "lineage": definition.lineage,
                        "state": state,
                        "module": module.name,
                        "gene": gene,
                        "expected_direction": direction,
                        "marker_value_unit": "log-normalized RNA expression",
                        "comparison": f"other marker-derived {definition.lineage} states",
                        "n_cells_target": int(target.sum()),
                        "n_cells_comparison": int(comparison.sum()),
                        "target_detection_fraction": float(value[target].gt(0).mean()),
                        "comparison_detection_fraction": float(value[comparison].gt(0).mean()),
                        "target_mean_marker_value": float(value[target].mean()),
                        "comparison_mean_marker_value": float(value[comparison].mean()),
                        "oriented_gene_auroc": mann_whitney_auroc(
                            oriented_gene.to_numpy(), target.to_numpy()
                        ),
                    }
                )
    projected = cells[cells["projectils_state"].notna()]
    corroboration = (
        projected.groupby(
            ["donor", "lineage", "label", "projectils_state", "projectils_quality"],
            dropna=False,
        )
        .agg(
            n_cells=("barcode", "size"),
            median_projectils_confidence=("projectils_confidence", "median"),
        )
        .reset_index()
    )
    return {
        "state_by_day": state_by_day,
        "state_panel_evidence": pd.DataFrame(panel_rows),
        "state_gene_evidence": pd.DataFrame(gene_rows),
        "projectils_corroboration": corroboration,
    }


def marker_definitions() -> pd.DataFrame:
    rows = []
    describe = lambda modules: " | ".join(
        f"{module.name}: {','.join(module.genes)}" for module in modules
    )
    for state, definition in PER_CELL_STATE_DEFINITIONS.items():
        if state in COLLAPSED_STATE_LABELS:
            continue
        rows.append(
            {
                "lineage": definition.lineage,
                "state": state,
                "type": definition.type_name,
                "positive_modules_expected_higher": describe(definition.positive_modules),
                "negative_modules_expected_lower": describe(definition.negative_modules),
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    combined: dict[str, list[pd.DataFrame]] = {}
    for donor in DONORS:
        cells = pd.read_csv(ROOT / "cells" / f"{donor}_state_labels.csv")
        for name, table in summarize_donor(donor, cells).items():
            combined.setdefault(name, []).append(table)
    for name, tables in combined.items():
        pd.concat(tables, ignore_index=True).to_csv(args.output / f"{name}.csv", index=False)
    marker_definitions().to_csv(args.output / "state_marker_definitions.csv", index=False)
    print(f"wrote per-cell evidence tables to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

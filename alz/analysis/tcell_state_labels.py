#!/usr/bin/env python3
"""Assign cycle-independent T-cell states to individual cells.

Raw CITE-seq CD4/CD8 counts establish lineage, with the donor's native-cluster
lineage used only when antibody counts are inconclusive. Within each donor and
lineage, non-cycle RNA markers are standardized across cells. A named state is
eligible only when every expected-high and expected-low module points in its
declared direction relative to the donor-lineage mean. The eligible state whose
weakest required module has the strongest support is assigned. If none is
eligible, or the best support is tied, the cell remains simply ``CD4`` or ``CD8``.

ProjecTILs calls are retained as reference evidence and never determine the label.
Internal signed module values are classification mechanics and are not exported
as biological scores.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from alz.analysis.tcell_marker_sets import (  # noqa: E402
    COLLAPSED_STATE_LABELS,
    PER_CELL_STATE_DEFINITIONS,
    SIGNATURES,
    PerCellStateDefinition,
    per_cell_marker_genes,
)


DONORS = ("donor1", "donor2")
REPORT_ROOT = Path("outputs/reports/tcell_labeling")
EXPRESSION_ROOT = REPORT_ROOT / "auroc"
ADT_ROOT = REPORT_ROOT / "adt"
UMAP_ROOT = REPORT_ROOT / "umap"
OUTPUT_ROOT = REPORT_ROOT / "cells"
PROJECTILS_ROOT = Path("data/derived/tcells_incytr_inputs")

REQUIRED_ADT_COLUMNS = (
    "CD4_protein_umi",
    "CD8_protein_umi",
    "mouse_isotype_umi",
)

CLUSTER_LINEAGE = {
    "donor1": {
        0: "CD4", 1: "CD4", 2: "CD4", 3: "CD4", 4: "CD8", 5: "CD4",
        6: "CD8", 7: "CD4", 8: "CD8", 9: "CD8", 10: "CD4", 11: "CD8",
        12: "contaminant", 13: "contaminant",
    },
    "donor2": {
        0: "CD8", 1: "CD4", 2: "CD4", 3: "CD4", 4: "CD8", 5: "CD8",
        6: "CD4", 7: "CD8", 8: "CD8", 9: "CD8", 10: "CD4",
        11: "contaminant", 12: "contaminant",
    },
}

ANALYSIS_TYPES = {
    **{
        label: definition.type_name
        for label, definition in PER_CELL_STATE_DEFINITIONS.items()
        if label not in COLLAPSED_STATE_LABELS
    },
    "CD4": "CD4",
    "CD8": "CD8",
}


def _assert_unique(frame: pd.DataFrame, name: str) -> None:
    if "barcode" not in frame:
        raise ValueError(f"{name} lacks a barcode column")
    if frame["barcode"].isna().any() or frame["barcode"].duplicated().any():
        raise ValueError(f"{name} barcodes must be non-null and unique")


def _standardize(expression: pd.DataFrame) -> pd.DataFrame:
    standard_deviation = expression.std(axis=0, ddof=0)
    variable = standard_deviation.gt(0)
    standardized = pd.DataFrame(0.0, index=expression.index, columns=expression.columns)
    standardized.loc[:, variable] = expression.loc[:, variable].subtract(
        expression.loc[:, variable].mean(axis=0), axis=1
    ).divide(standard_deviation.loc[variable], axis=1)
    return standardized


def classify_lineage(row: pd.Series, cluster_lineage: str) -> tuple[str, str]:
    """Resolve CD4/CD8 identity from raw ADT counts with cluster fallback."""
    if cluster_lineage == "contaminant":
        return "contaminant", "cluster contaminant"
    cd4 = row["CD4_protein_umi"]
    cd8 = row["CD8_protein_umi"]
    background = row["mouse_isotype_umi"]
    if cd4 > cd8 and cd4 > background:
        return "CD4", "ADT CD4-dominant"
    if cd8 > cd4 and cd8 > background:
        return "CD8", "ADT CD8-dominant"
    return cluster_lineage, "cluster fallback"


def classify_projectils_quality(state: object, confidence: object) -> str:
    """Retain exact reference-neighborhood unanimity without a tuned cutoff."""
    if pd.isna(state) or pd.isna(confidence):
        return "not projected"
    return "unanimous" if float(confidence) == 1.0 else "projected"


def collapse_state_label(label: str) -> str:
    """Apply conservative biological collapses before labels are exported."""
    return COLLAPSED_STATE_LABELS.get(label, label)


def assign_marker_states(
    signed_module_support: pd.DataFrame,
    *,
    fallback: str = "lineage",
) -> str:
    """Choose one state from a single cell's signed module-support table.

    Rows are required modules and columns are candidate states. A state is
    eligible only when every finite entry in its column is strictly positive.
    Its support is the weakest required module. Ties retain the lineage fallback.
    """
    support = {}
    for state in signed_module_support:
        values = signed_module_support[state].dropna()
        if len(values) and values.gt(0).all():
            support[state] = float(values.min())
    if not support:
        return fallback
    best = max(support.values())
    winners = [state for state, value in support.items() if value == best]
    return winners[0] if len(winners) == 1 else fallback


def _state_module_support(
    standardized: pd.DataFrame,
    definition: PerCellStateDefinition,
) -> pd.DataFrame:
    columns = {}
    for module in definition.positive_modules:
        columns[f"higher:{module.name}"] = standardized[list(module.genes)].mean(axis=1)
    for module in definition.negative_modules:
        columns[f"lower:{module.name}"] = -standardized[list(module.genes)].mean(axis=1)
    return pd.DataFrame(columns, index=standardized.index)


def _assign_lineage_states(cells: pd.DataFrame, lineage: str) -> pd.Series:
    definitions = {
        label: definition
        for label, definition in PER_CELL_STATE_DEFINITIONS.items()
        if definition.lineage == lineage
    }
    genes = list(per_cell_marker_genes())
    standardized = _standardize(cells[genes].apply(pd.to_numeric, errors="raise"))
    state_support = {
        label: _state_module_support(standardized, definition)
        for label, definition in definitions.items()
    }
    labels = []
    for cell_index in standardized.index:
        signed = pd.DataFrame(
            {
                label: support.loc[cell_index]
                for label, support in state_support.items()
            }
        )
        labels.append(assign_marker_states(signed, fallback=lineage))
    return pd.Series(labels, index=standardized.index, dtype="object")


def load_donor_cells(
    donor: str,
    *,
    expression_root: Path = EXPRESSION_ROOT,
    adt_root: Path = ADT_ROOT,
    umap_root: Path = UMAP_ROOT,
    projectils_root: Path = PROJECTILS_ROOT,
) -> pd.DataFrame:
    genes = list(per_cell_marker_genes())
    expression = pd.read_csv(expression_root / f"{donor}_marker_cell_expr.csv")
    adt = pd.read_csv(adt_root / f"{donor}_adt_evidence.csv")
    native = pd.read_csv(
        umap_root / f"{donor}_native_umap_coords.csv",
        usecols=["barcode", "seurat_clusters", "day_label"],
    )
    missing_expression = sorted(set(genes) - set(expression))
    if missing_expression:
        raise ValueError(f"{donor}: marker expression needs re-extraction: {missing_expression}")
    missing_adt = sorted(set(REQUIRED_ADT_COLUMNS) - set(adt))
    if missing_adt:
        raise ValueError(f"{donor}: ADT evidence lacks columns {missing_adt}")
    expression = expression[["barcode", *genes]]
    adt_columns = [column for column in adt if column == "barcode" or column.endswith("_umi")]
    adt = adt[adt_columns]
    for name, frame in (("RNA", expression), ("ADT", adt), ("native", native)):
        _assert_unique(frame, f"{donor} {name}")
    cells = native.merge(expression, on="barcode", validate="one_to_one")
    cells = cells.merge(adt, on="barcode", validate="one_to_one")
    if len(cells) != len(native) or len(expression) != len(native) or len(adt) != len(native):
        raise ValueError(
            f"{donor}: cell accounting mismatch native={len(native)} "
            f"RNA={len(expression)} ADT={len(adt)} joined={len(cells)}"
        )

    projection_path = projectils_root / donor / "scrna" / "projectils_predictions.csv"
    if projection_path.exists():
        projection = pd.read_csv(
            projection_path,
            usecols=["barcode", "functional.cluster", "functional.cluster.conf"],
        ).rename(
            columns={
                "functional.cluster": "projectils_state",
                "functional.cluster.conf": "projectils_confidence",
            }
        )
        _assert_unique(projection, f"{donor} ProjecTILs")
        cells = cells.merge(projection, on="barcode", how="left", validate="one_to_one")
    else:
        cells["projectils_state"] = pd.NA
        cells["projectils_confidence"] = pd.NA
    return cells


def assign_per_cell_labels(donor: str, cells: pd.DataFrame) -> pd.DataFrame:
    """Assign one cycle-independent marker state independently to every cell."""
    cluster_map = CLUSTER_LINEAGE[donor]
    assigned = cells.copy()
    assigned["seurat_cluster"] = pd.to_numeric(
        assigned["seurat_clusters"], errors="raise"
    ).astype(int)
    observed = set(assigned["seurat_cluster"])
    if observed != set(cluster_map):
        raise ValueError(
            f"{donor}: native cluster roster mismatch observed={sorted(observed)} "
            f"expected={sorted(cluster_map)}"
        )
    assigned["cluster_lineage"] = assigned["seurat_cluster"].map(cluster_map)
    lineage_calls = assigned.apply(
        lambda row: classify_lineage(row, row["cluster_lineage"]), axis=1
    )
    assigned["lineage"] = lineage_calls.map(lambda call: call[0])
    assigned["lineage_source"] = lineage_calls.map(lambda call: call[1])
    assigned["label"] = "contaminant"
    for lineage in ("CD4", "CD8"):
        selected = assigned["lineage"].eq(lineage)
        assigned.loc[selected, "label"] = _assign_lineage_states(
            assigned.loc[selected], lineage
        )
    assigned["label"] = assigned["label"].map(collapse_state_label)
    assigned["type"] = assigned["label"].map(ANALYSIS_TYPES)
    assigned["day"] = (
        assigned["day_label"].astype(str).str.extract(r"Day_(\d+)", expand=False).astype(int)
    )
    assigned["donor"] = donor
    assigned["projectils_quality"] = assigned.apply(
        lambda row: classify_projectils_quality(
            row["projectils_state"], row["projectils_confidence"]
        ),
        axis=1,
    )
    if assigned.loc[assigned["lineage"].isin(["CD4", "CD8"]), "type"].isna().any():
        raise ValueError(f"{donor}: non-contaminant cells lack an Incytr type")
    if assigned.loc[assigned["lineage"].eq("contaminant"), "type"].notna().any():
        raise ValueError(f"{donor}: contaminants unexpectedly have an Incytr type")

    rename = {gene: f"{gene}_log_normalized_expression" for gene in per_cell_marker_genes()}
    assigned = assigned.rename(columns=rename)
    output_columns = [
        "barcode", "donor", "seurat_cluster", "day", "lineage", "lineage_source",
        "label", "type", "projectils_state", "projectils_confidence",
        "projectils_quality", *rename.values(),
        *[column for column in assigned if column.endswith("_umi")],
    ]
    return assigned[output_columns].sort_values("barcode").reset_index(drop=True)


def marker_genes_for_export() -> list[str]:
    historical_context = {
        gene
        for panel in ("tcell_core", "cd4_lineage", "cd8_lineage")
        for gene in SIGNATURES[panel]
    }
    return sorted(historical_context | set(per_cell_marker_genes()))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--donor", choices=(*DONORS, "all"), default="all")
    parser.add_argument("--write-markers", type=Path)
    parser.add_argument("--expression-root", type=Path, default=EXPRESSION_ROOT)
    parser.add_argument("--adt-root", type=Path, default=ADT_ROOT)
    parser.add_argument("--umap-root", type=Path, default=UMAP_ROOT)
    parser.add_argument("--projectils-root", type=Path, default=PROJECTILS_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    if args.write_markers:
        genes = marker_genes_for_export()
        args.write_markers.parent.mkdir(parents=True, exist_ok=True)
        args.write_markers.write_text("\n".join(genes) + "\n")
        print(f"wrote {len(genes)} non-cycle per-cell markers to {args.write_markers}")
        return 0

    args.output_root.mkdir(parents=True, exist_ok=True)
    donors = DONORS if args.donor == "all" else (args.donor,)
    for donor in donors:
        labels = assign_per_cell_labels(
            donor,
            load_donor_cells(
                donor,
                expression_root=args.expression_root,
                adt_root=args.adt_root,
                umap_root=args.umap_root,
                projectils_root=args.projectils_root,
            ),
        )
        output = args.output_root / f"{donor}_state_labels.csv"
        labels.to_csv(output, index=False)
        print(f"[{donor}] wrote {len(labels)} per-cell labels to {output}")
        print(labels.groupby(["label", "type"], dropna=False).size().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

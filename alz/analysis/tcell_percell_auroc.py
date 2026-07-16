#!/usr/bin/env python3
"""Reproduce the original per-cell T-cell marker AUROC analysis.

The historical ProjecTILs ``functional.cluster`` call is the label being
evaluated. Marker expression is log-normalized RNA expression, and the
original all-other-cells background is retained so the AUROCs displayed in
the original report can be reproduced exactly.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tcell_marker_sets import (  # noqa: E402
    CORE_PANELS,
    SIGNATURES,
    _marker_class,
    per_cell_marker_genes,
)


DONORS = ("donor1", "donor2")
REPORT_ROOT = Path("outputs/reports/tcell_labeling")
EXPRESSION_ROOT = REPORT_ROOT / "auroc"
EMBEDDING_ROOT = Path("data/derived/tcells_incytr_inputs")
REPRODUCTION_FIXTURE = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "tcell_original_report_expected.json"
)

PANEL_LABELS: dict[str, set[str]] = {
    "exhaustion": {"CD8.TEX", "CD4.CTL_Exh"},
    "progenitor_exhaustion": {"CD8.TPEX"},
    "cytotoxic": {
        "CD8.EM",
        "CD8.TEMRA",
        "CD8.TEX",
        "CD8.MAIT",
        "CD4.CTL_GNLY",
        "CD4.CTL_EOMES",
    },
    "th17": {"CD4.Th17"},
    "tfh": {"CD4.Tfh"},
    "treg": {"CD4.Treg"},
    "naive_memory": {"CD8.NaiveLike", "CD8.CM", "CD4.NaiveLike"},
}

TYPE_PANELS = {"cd4_lineage": "CD4", "cd8_lineage": "CD8"}
PROJECTION_COLUMNS = {
    "projection_reference",
    "reduction",
    "functional.cluster",
    "functional.cluster.conf",
}


def lineage_for_state(state: str) -> str:
    """Return the CD4/CD8 lineage encoded by a ProjecTILs state."""
    if state.startswith("CD4."):
        return "CD4"
    if state.startswith("CD8."):
        return "CD8"
    raise ValueError(f"unsupported ProjecTILs functional.cluster label: {state!r}")


def mann_whitney_auroc(values: np.ndarray, positive: np.ndarray) -> float:
    """Return P(a random positive ranks above a random negative), ties = 0.5."""
    values = np.asarray(values, dtype=float)
    positive = np.asarray(positive, dtype=bool)
    if values.shape != positive.shape:
        raise ValueError("values and positive must have identical shapes")
    valid = np.isfinite(values)
    values = values[valid]
    positive = positive[valid]
    n_positive = int(positive.sum())
    n_negative = int(positive.size - n_positive)
    if not n_positive or not n_negative:
        return np.nan
    ranks = rankdata(values)
    u_statistic = ranks[positive].sum() - n_positive * (n_positive + 1) / 2
    return float(u_statistic / (n_positive * n_negative))


def _read_unique_csv(path: Path, *, name: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "barcode" not in frame:
        raise ValueError(f"{name} lacks required barcode column: {path}")
    if frame["barcode"].isna().any() or frame["barcode"].duplicated().any():
        raise ValueError(f"{name} barcodes must be non-null and unique: {path}")
    return frame.set_index("barcode")


def _collapse_projection_rows(path: Path) -> pd.DataFrame:
    """Collapse PCA/UMAP duplicates while preserving the original projection choice.

    Within a lineage-specific projection the PCA and UMAP rows must agree on the
    state and confidence.  A small number of cells passed both CD4 and CD8 gates
    historically; the original analysis kept the first PCA projection in file order.
    We retain that behavior only to make the original report reproducible.
    """
    projections = pd.read_csv(path)
    required = {"barcode", *PROJECTION_COLUMNS}
    missing = required - set(projections.columns)
    if missing:
        raise ValueError(f"projection file lacks columns {sorted(missing)}: {path}")
    projections = projections[projections["functional.cluster"].notna()].copy()
    if projections.empty:
        raise ValueError(f"projection file contains no functional.cluster labels: {path}")

    agreement = projections.groupby(
        ["barcode", "projection_reference"], sort=False, dropna=False
    ).agg(
        n_states=("functional.cluster", "nunique"),
        n_confidences=("functional.cluster.conf", "nunique"),
    )
    if (agreement["n_states"] != 1).any() or (agreement["n_confidences"] != 1).any():
        raise ValueError(f"PCA/UMAP projection rows disagree within a reference: {path}")

    pca = projections[projections["reduction"].eq("pca")].copy()
    if pca.empty:
        pca = projections.drop_duplicates(["barcode", "projection_reference"], keep="first")
    collapsed = pca.drop_duplicates("barcode", keep="first").set_index("barcode")
    if not collapsed.index.is_unique:
        raise AssertionError("collapsed projection barcodes are not unique")
    return collapsed


def load_donor_data(
    donor: str,
    *,
    expression_root: Path = EXPRESSION_ROOT,
    embedding_root: Path = EMBEDDING_ROOT,
) -> pd.DataFrame:
    """Load and strictly join projected labels to marker RNA by barcode."""
    expression_path = expression_root / f"{donor}_marker_cell_expr.csv"
    projection_path = (
        embedding_root / donor / "scrna" / "projectils_embeddings.csv"
    )
    expression = _read_unique_csv(expression_path, name="marker expression")
    projection = _collapse_projection_rows(projection_path)

    projected_barcodes = set(projection.index)
    missing_expression = projected_barcodes - set(expression.index)
    if missing_expression:
        raise ValueError(
            f"{donor}: projected barcodes missing from marker expression="
            f"{len(missing_expression)}"
        )

    joined = projection.join(expression, how="left", validate="one_to_one")
    if len(joined) != len(projection) or not joined.index.is_unique:
        raise AssertionError(f"{donor}: barcode join did not preserve projected cells one-to-one")
    if joined["functional.cluster"].isna().any():
        raise ValueError(f"{donor}: incomplete projected-state join")
    joined["lineage"] = joined["functional.cluster"].map(lineage_for_state)
    joined.attrs["expression_input"] = str(expression_path)
    joined.attrs["projection_input"] = str(projection_path)
    return joined


def _panel_class(panel: str) -> str:
    if panel in TYPE_PANELS:
        return "type"
    if panel in CORE_PANELS:
        return "core"
    return "state"


def _is_matching_panel(panel: str, state: str) -> bool:
    if panel in TYPE_PANELS:
        return TYPE_PANELS[panel] == lineage_for_state(state)
    if panel in CORE_PANELS:
        return True
    return state in PANEL_LABELS.get(panel, set())


def _comparison_mask(
    state_values: pd.Series,
    state: str,
    marker_class: str,
    type_background: str,
) -> tuple[np.ndarray, np.ndarray, str]:
    positive = state_values.eq(state).to_numpy()
    lineage = lineage_for_state(state)
    lineages = state_values.map(lineage_for_state)
    if marker_class == "type" and type_background == "opposite_lineage":
        other = "CD4" if lineage == "CD8" else "CD8"
        comparison = lineages.eq(other).to_numpy()
        label = f"opposite-lineage {other} cells"
    elif marker_class == "type" and type_background == "all_other_cells":
        comparison = ~positive
        label = "all other projected T cells"
    else:
        comparison = lineages.eq(lineage).to_numpy() & ~positive
        label = f"same-lineage sibling {lineage} states"
    eligible = positive | comparison
    return positive, eligible, label


def _zscore_columns(expression: pd.DataFrame) -> pd.DataFrame:
    standard_deviation = expression.std(axis=0, ddof=0).replace(0, np.nan)
    return expression.subtract(expression.mean(axis=0), axis=1).divide(
        standard_deviation, axis=1
    )


def calculate_evidence_tables(
    donor: str,
    cells: pd.DataFrame,
    *,
    signatures: Mapping[str, Sequence[str]] = SIGNATURES,
    type_background: str = "all_other_cells",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate per-gene and z-scored panel AUROC for every projected state."""
    if type_background not in {"opposite_lineage", "all_other_cells"}:
        raise ValueError(f"unsupported type background: {type_background}")
    if "functional.cluster" not in cells:
        raise ValueError("cells must contain functional.cluster")
    states = cells["functional.cluster"].astype(str)
    present_states = sorted(states.unique())
    marker_columns = [
        gene
        for panel_genes in signatures.values()
        for gene in panel_genes
        if gene in cells.columns
    ]
    marker_columns = list(dict.fromkeys(marker_columns))
    expression = cells[marker_columns].apply(pd.to_numeric, errors="raise")
    standardized = _zscore_columns(expression)

    marker_rows: list[dict[str, object]] = []
    panel_rows: list[dict[str, object]] = []
    for panel, declared_genes in signatures.items():
        marker_class = _panel_class(panel)
        present_genes = [gene for gene in declared_genes if gene in expression]
        for state in present_states:
            positive, eligible, comparison_label = _comparison_mask(
                states, state, marker_class, type_background
            )
            comparison = eligible & ~positive
            matching = _is_matching_panel(panel, state)
            for gene in declared_genes:
                present = gene in expression
                values = (
                    expression[gene].to_numpy(dtype=float)
                    if present
                    else np.full(len(cells), np.nan)
                )
                marker_rows.append(
                    {
                        "donor": donor,
                        "state": state,
                        "lineage": lineage_for_state(state),
                        "panel": panel,
                        "gene": gene,
                        "marker_class": marker_class,
                        "is_matching_panel": matching,
                        "present": present,
                        "marker_value_unit": "log-normalized RNA expression",
                        "comparison": comparison_label,
                        "n_cells_target": int(positive.sum()),
                        "n_cells_comparison": int(comparison.sum()),
                        "target_detection_fraction": (
                            float(np.mean(values[positive] > 0))
                            if present
                            else np.nan
                        ),
                        "comparison_detection_fraction": (
                            float(np.mean(values[comparison] > 0))
                            if present
                            else np.nan
                        ),
                        "target_mean_marker_value": (
                            float(np.mean(values[positive])) if present else np.nan
                        ),
                        "comparison_mean_marker_value": (
                            float(np.mean(values[comparison])) if present else np.nan
                        ),
                        "gene_auroc": (
                            mann_whitney_auroc(values[eligible], positive[eligible])
                            if present
                            else np.nan
                        ),
                    }
                )

            if present_genes:
                score = standardized[present_genes].mean(axis=1).to_numpy(dtype=float)
                signature_auroc = mann_whitney_auroc(
                    score[eligible], positive[eligible]
                )
            else:
                signature_auroc = np.nan
            panel_rows.append(
                {
                    "donor": donor,
                    "state": state,
                    "lineage": lineage_for_state(state),
                    "panel": panel,
                    "marker_class": marker_class,
                    "is_matching_panel": matching,
                    "markers_present": ";".join(present_genes),
                    "n_markers": len(present_genes),
                    "comparison": comparison_label,
                    "n_cells_target": int(positive.sum()),
                    "n_cells_comparison": int(comparison.sum()),
                    "signature_auroc": signature_auroc,
                }
            )
    return pd.DataFrame(marker_rows), pd.DataFrame(panel_rows)


def _write_donor_results(donor: str, cells: pd.DataFrame, output_root: Path) -> None:
    declared_markers = list(
        dict.fromkeys(gene for genes in SIGNATURES.values() for gene in genes)
    )
    present_markers = [gene for gene in declared_markers if gene in cells]
    unadjusted_matt_marker, unadjusted_matt_panel = calculate_evidence_tables(
        donor, cells, type_background="all_other_cells"
    )

    unadjusted_dir = output_root / "reproduced_unadjusted"
    unadjusted_dir.mkdir(parents=True, exist_ok=True)

    unadjusted_matt_marker.to_csv(
        unadjusted_dir / f"{donor}_historical_percell_marker_auroc.csv", index=False
    )
    unadjusted_matt_panel.to_csv(
        unadjusted_dir / f"{donor}_historical_percell_panel_auroc.csv", index=False
    )
    inventory = pd.DataFrame(
        [
            {
                "donor": donor,
                "expression_input": cells.attrs.get("expression_input", "unknown"),
                "projection_input": cells.attrs.get("projection_input", "unknown"),
                "n_projected_cells": len(cells),
                "n_marker_genes": len(present_markers),
            }
        ]
    )
    inventory.to_csv(unadjusted_dir / f"{donor}_input_inventory.csv", index=False)
    print(
        f"[{donor}] {len(cells)} projected cells, {len(present_markers)} markers"
    )


def verify_historical_reproduction(
    output_root: Path,
    *,
    fixture_path: Path = REPRODUCTION_FIXTURE,
    donors: Sequence[str] = DONORS,
) -> pd.DataFrame:
    """Verify the original recorded panels and displayed AUROCs from versioned inputs."""
    fixture = json.loads(fixture_path.read_text())
    displayed_tables = fixture.get("displayed_tables", [])
    expected_table_count = fixture.get("displayed_table_count")
    if (
        not isinstance(expected_table_count, int)
        or len(displayed_tables) != expected_table_count
        or any("rows" not in table for table in displayed_tables)
    ):
        raise ValueError("original report fixture does not inventory every displayed table")
    fixture_panels = fixture["marker_panels"]
    implemented_panels = {
        panel: list(genes)
        for panel, genes in SIGNATURES.items()
        if panel in fixture_panels
    }
    if implemented_panels != fixture_panels:
        raise ValueError("versioned marker panels do not match the original report fixture")

    rows: list[dict[str, object]] = []
    for donor in donors:
        results_path = (
            output_root
            / "reproduced_unadjusted"
            / f"{donor}_historical_percell_panel_auroc.csv"
        )
        results = pd.read_csv(results_path).set_index(["state", "panel"])
        inventory_path = (
            output_root / "reproduced_unadjusted" / f"{donor}_input_inventory.csv"
        )
        inventory = pd.read_csv(inventory_path).iloc[0]
        expected_inputs = fixture["inputs"]
        expected_expression = expected_inputs["expression_pattern"].format(donor=donor)
        expected_projection = expected_inputs["projection_pattern"].format(donor=donor)
        if (
            inventory["expression_input"] != expected_expression
            or inventory["projection_input"] != expected_projection
            or int(inventory["n_projected_cells"])
            != int(expected_inputs["projected_cells"][donor])
            or int(inventory["n_marker_genes"])
            != int(expected_inputs["marker_genes_present"][donor])
        ):
            raise ValueError(f"{donor}: declared inputs or inventory do not match fixture")
        for key, expected in fixture["displayed_panel_auroc_rounded_3dp"][donor].items():
            state, panel = key.split("|", maxsplit=1)
            result = results.loc[(state, panel)]
            observed = round(float(result["signature_auroc"]), 3)
            observed_count = int(result["n_cells_target"])
            expected_count = int(fixture["displayed_state_target_counts"][donor][state])
            rows.append(
                {
                    "donor": donor,
                    "state": state,
                    "panel": panel,
                    "expected_displayed_auroc": expected,
                    "observed_reproduced_auroc": observed,
                    "expected_target_cells": expected_count,
                    "observed_target_cells": observed_count,
                    "matches": observed == expected and observed_count == expected_count,
                }
            )
    check = pd.DataFrame(rows)
    check_path = output_root / "reproduced_unadjusted" / "reproduction_check.csv"
    check.to_csv(check_path, index=False)
    if not check["matches"].all():
        mismatches = check.loc[~check["matches"]].to_dict("records")
        raise ValueError(f"historical AUROC reproduction mismatches: {mismatches}")
    return check


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-markers", type=Path)
    parser.add_argument("--donor", choices=(*DONORS, "all"), default="all")
    parser.add_argument("--expression-root", type=Path, default=EXPRESSION_ROOT)
    parser.add_argument("--embedding-root", type=Path, default=EMBEDDING_ROOT)
    parser.add_argument("--output-root", type=Path, default=REPORT_ROOT)
    parser.add_argument("--reproduction-fixture", type=Path, default=REPRODUCTION_FIXTURE)
    args = parser.parse_args()
    if args.write_markers:
        genes = sorted(
            {
                gene
                for genes in SIGNATURES.values()
                for gene in genes
            }
            | set(per_cell_marker_genes())
        )
        args.write_markers.parent.mkdir(parents=True, exist_ok=True)
        args.write_markers.write_text("\n".join(genes) + "\n")
        print(f"wrote {len(genes)} marker genes to {args.write_markers}")
        return 0

    donors = DONORS if args.donor == "all" else (args.donor,)
    for donor in donors:
        cells = load_donor_data(
            donor,
            expression_root=args.expression_root,
            embedding_root=args.embedding_root,
        )
        _write_donor_results(donor, cells, args.output_root)
    check = verify_historical_reproduction(
        args.output_root,
        fixture_path=args.reproduction_fixture,
        donors=donors,
    )
    print(f"historical reproduction: {int(check['matches'].sum())}/{len(check)} AUROCs match")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Reproduce Matt's per-cell T-cell AUROC analysis with cycle regression.

The historical ProjecTILs ``functional.cluster`` call remains the label being
evaluated.  Marker expression is evaluated twice: once as log-normalized RNA
expression and once after donor-wide ordinary least-squares regression on the
continuous Seurat ``S.Score`` and ``G2M.Score`` covariates.

The primary type-panel comparison is a state against cells of the opposite
lineage.  A separate historical reproduction uses Matt's implemented
all-other-cells background so the values displayed in the historical HTML can
be reproduced exactly despite the HTML describing an opposite-lineage test.
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
from tcell_marker_sets import CORE_PANELS, SIGNATURES, _marker_class  # noqa: E402


DONORS = ("donor1", "donor2")
REPORT_ROOT = Path("outputs/reports/tcell_labeling")
EXPRESSION_ROOT = REPORT_ROOT / "auroc"
EMBEDDING_ROOT = Path("data/derived/tcells_incytr_inputs")
CYCLE_ROOT = REPORT_ROOT / "clusters"
REPRODUCTION_FIXTURE = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "tcell_matt_report_expected.json"
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

MEMORY_GENES = ("TCF7", "LEF1", "SELL", "CCR7", "IL7R")
TYPE_PANELS = {"cd4_lineage": "CD4", "cd8_lineage": "CD8"}
PROJECTION_COLUMNS = {
    "projection_reference",
    "reduction",
    "functional.cluster",
    "functional.cluster.conf",
}
CYCLE_COLUMNS = ("Phase", "S.Score", "G2M.Score")


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
    """Collapse PCA/UMAP duplicates while preserving Matt's projection choice.

    Within a lineage-specific projection the PCA and UMAP rows must agree on the
    state and confidence.  A small number of cells passed both CD4 and CD8 gates
    historically; Matt's analysis kept the first PCA projection in file order.
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
    cycle_root: Path = CYCLE_ROOT,
) -> pd.DataFrame:
    """Load and strictly join projected labels, marker RNA, and cycle scores."""
    expression_path = expression_root / f"{donor}_marker_cell_expr.csv"
    projection_path = (
        embedding_root / donor / "scrna" / "projectils_embeddings.csv"
    )
    cycle_path = cycle_root / f"{donor}_cc_recluster_cells.csv"

    expression = _read_unique_csv(expression_path, name="marker expression")
    projection = _collapse_projection_rows(projection_path)
    if set(CYCLE_COLUMNS).issubset(expression.columns):
        cycle = expression[list(CYCLE_COLUMNS)].copy()
        expression = expression.drop(columns=list(CYCLE_COLUMNS))
        cycle_score_source = expression_path
    else:
        cycle = _read_unique_csv(cycle_path, name="cell-cycle scores")
        cycle_score_source = cycle_path
        missing_cycle = set(CYCLE_COLUMNS) - set(cycle.columns)
        if missing_cycle:
            raise ValueError(
                f"cell-cycle file lacks columns {sorted(missing_cycle)}: {cycle_path}"
            )

    projected_barcodes = set(projection.index)
    missing_expression = projected_barcodes - set(expression.index)
    missing_scores = projected_barcodes - set(cycle.index)
    if missing_expression or missing_scores:
        raise ValueError(
            f"{donor}: projected barcodes missing from expression={len(missing_expression)} "
            f"or cell-cycle scores={len(missing_scores)}"
        )

    joined = projection.join(expression, how="left", validate="one_to_one")
    joined = joined.join(cycle[list(CYCLE_COLUMNS)], how="left", validate="one_to_one")
    if len(joined) != len(projection) or not joined.index.is_unique:
        raise AssertionError(f"{donor}: barcode join did not preserve projected cells one-to-one")
    if joined[["functional.cluster", "S.Score", "G2M.Score"]].isna().any().any():
        raise ValueError(f"{donor}: incomplete label or cycle-score join")
    joined["lineage"] = joined["functional.cluster"].map(lineage_for_state)
    joined.attrs["expression_input"] = str(expression_path)
    joined.attrs["projection_input"] = str(projection_path)
    joined.attrs["cycle_score_input"] = str(cycle_score_source)
    return joined


def residualize_marker_expression(
    expression: pd.DataFrame,
    cycle_scores: pd.DataFrame,
) -> pd.DataFrame:
    """Remove donor-wide linear S.Score and G2M.Score effects from each gene."""
    if not expression.index.equals(cycle_scores.index):
        raise ValueError("expression and cycle-score rows must have identical indexes")
    required = ["S.Score", "G2M.Score"]
    missing = set(required) - set(cycle_scores.columns)
    if missing:
        raise ValueError(f"cycle scores lack columns: {sorted(missing)}")

    response = expression.apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)
    covariates = cycle_scores[required].apply(pd.to_numeric, errors="raise")
    design = np.column_stack([np.ones(len(covariates)), covariates.to_numpy(dtype=float)])
    if not np.isfinite(response).all() or not np.isfinite(design).all():
        raise ValueError("cycle regression requires finite expression and covariates")
    if np.linalg.matrix_rank(design) != design.shape[1]:
        raise ValueError("cycle-regression design matrix is not full rank")

    coefficients, *_ = np.linalg.lstsq(design, response, rcond=None)
    residuals = response - design @ coefficients
    return pd.DataFrame(residuals, index=expression.index, columns=expression.columns)


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
        label = "all other projected T cells (historical implementation)"
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
    type_background: str = "opposite_lineage",
    value_kind: str = "log_normalized_expression",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate per-gene and z-scored panel AUROC for every projected state."""
    if type_background not in {"opposite_lineage", "all_other_cells"}:
        raise ValueError(f"unsupported type background: {type_background}")
    value_units = {
        "log_normalized_expression": "log-normalized RNA expression",
        "cycle_regressed_residual": "cycle-regressed log-normalized-expression residual",
    }
    if value_kind not in value_units:
        raise ValueError(f"unsupported marker value kind: {value_kind}")
    is_raw_expression = value_kind == "log_normalized_expression"
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
                        "marker_value_unit": value_units[value_kind],
                        "comparison": comparison_label,
                        "n_cells_target": int(positive.sum()),
                        "n_cells_comparison": int(comparison.sum()),
                        "target_detection_fraction": (
                            float(np.mean(values[positive] > 0))
                            if present and is_raw_expression
                            else np.nan
                        ),
                        "comparison_detection_fraction": (
                            float(np.mean(values[comparison] > 0))
                            if present and is_raw_expression
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


def _comparison_table(
    unadjusted: pd.DataFrame,
    adjusted: pd.DataFrame,
    *,
    value_column: str,
) -> pd.DataFrame:
    count_columns = ["n_cells_target", "n_cells_comparison"]
    identity = [
        column
        for column in unadjusted.columns
        if column not in {*count_columns, value_column}
        and column != "marker_value_unit"
        and not column.startswith("target_")
        and not column.startswith("comparison_")
    ]
    left = unadjusted[identity + count_columns + [value_column]].rename(
        columns={value_column: "unadjusted_auroc"}
    )
    right = adjusted[identity + count_columns + [value_column]].rename(
        columns={value_column: "cycle_regressed_auroc"}
    )
    merged = left.merge(
        right,
        on=identity + count_columns,
        how="outer",
        validate="one_to_one",
    )
    merged["auroc_difference"] = (
        merged["cycle_regressed_auroc"] - merged["unadjusted_auroc"]
    )
    return merged


def _tpex_tex_evidence(
    donor: str,
    raw: pd.DataFrame,
    residuals: pd.DataFrame,
    states: pd.Series,
) -> pd.DataFrame:
    eligible = states.isin(["CD8.TPEX", "CD8.TEX"]).to_numpy()
    positive = states.eq("CD8.TPEX").to_numpy()
    rows: list[dict[str, object]] = []
    for gene in MEMORY_GENES:
        if gene not in raw:
            continue
        raw_values = raw[gene].to_numpy(dtype=float)
        adjusted_values = residuals[gene].to_numpy(dtype=float)
        rows.append(
            {
                "donor": donor,
                "gene": gene,
                "target": "CD8.TPEX",
                "comparison": "CD8.TEX",
                "n_cells_target": int((positive & eligible).sum()),
                "n_cells_comparison": int((~positive & eligible).sum()),
                "tpex_raw_detection_fraction": float(
                    np.mean(raw_values[positive & eligible] > 0)
                ),
                "tex_raw_detection_fraction": float(
                    np.mean(raw_values[~positive & eligible] > 0)
                ),
                "unadjusted_auroc": mann_whitney_auroc(
                    raw_values[eligible], positive[eligible]
                ),
                "cycle_regressed_auroc": mann_whitney_auroc(
                    adjusted_values[eligible], positive[eligible]
                ),
            }
        )
    result = pd.DataFrame(rows)
    result["auroc_difference"] = (
        result["cycle_regressed_auroc"] - result["unadjusted_auroc"]
    )
    return result


def _loss_of_memory_sensitivity(
    donor: str,
    raw: pd.DataFrame,
    residuals: pd.DataFrame,
    states: pd.Series,
) -> pd.DataFrame:
    target = states.eq("CD8.TEX").to_numpy()
    eligible = states.str.startswith("CD8.").to_numpy()
    rows: list[dict[str, object]] = []
    for gene in MEMORY_GENES:
        if gene not in raw:
            continue
        row: dict[str, object] = {
            "donor": donor,
            "target": "CD8.TEX",
            "comparison": "same-lineage sibling CD8 states",
            "gene": gene,
            "n_cells_target": int(target.sum()),
            "n_cells_comparison": int((eligible & ~target).sum()),
        }
        for layer, values in (
            ("unadjusted", raw[gene].to_numpy(dtype=float)),
            ("cycle_regressed", residuals[gene].to_numpy(dtype=float)),
        ):
            higher_in_tex = mann_whitney_auroc(values[eligible], target[eligible])
            row[f"{layer}_loss_of_memory_auroc"] = 1.0 - higher_in_tex
        rows.append(row)
    result = pd.DataFrame(rows)
    result["auroc_difference"] = (
        result["cycle_regressed_loss_of_memory_auroc"]
        - result["unadjusted_loss_of_memory_auroc"]
    )
    return result


def _write_donor_results(donor: str, cells: pd.DataFrame, output_root: Path) -> None:
    declared_markers = list(
        dict.fromkeys(gene for genes in SIGNATURES.values() for gene in genes)
    )
    present_markers = [gene for gene in declared_markers if gene in cells]
    raw = cells[present_markers].apply(pd.to_numeric, errors="raise")
    residuals = residualize_marker_expression(raw, cells[list(CYCLE_COLUMNS)])

    unadjusted_marker, unadjusted_panel = calculate_evidence_tables(donor, cells)
    adjusted_cells = cells.drop(columns=present_markers).join(residuals)
    adjusted_marker, adjusted_panel = calculate_evidence_tables(
        donor, adjusted_cells, value_kind="cycle_regressed_residual"
    )
    historical_marker, historical_panel = calculate_evidence_tables(
        donor, cells, type_background="all_other_cells"
    )

    unadjusted_dir = output_root / "reproduced_unadjusted"
    adjusted_dir = output_root / "cycle_regressed"
    unadjusted_dir.mkdir(parents=True, exist_ok=True)
    adjusted_dir.mkdir(parents=True, exist_ok=True)

    unadjusted_marker.to_csv(
        unadjusted_dir / f"{donor}_percell_marker_auroc.csv", index=False
    )
    unadjusted_panel.to_csv(
        unadjusted_dir / f"{donor}_percell_panel_auroc.csv", index=False
    )
    historical_marker.to_csv(
        unadjusted_dir / f"{donor}_historical_percell_marker_auroc.csv", index=False
    )
    historical_panel.to_csv(
        unadjusted_dir / f"{donor}_historical_percell_panel_auroc.csv", index=False
    )
    adjusted_marker.to_csv(
        adjusted_dir / f"{donor}_percell_marker_auroc.csv", index=False
    )
    adjusted_panel.to_csv(
        adjusted_dir / f"{donor}_percell_panel_auroc.csv", index=False
    )
    _comparison_table(
        unadjusted_marker, adjusted_marker, value_column="gene_auroc"
    ).to_csv(adjusted_dir / f"{donor}_percell_marker_auroc_comparison.csv", index=False)
    _comparison_table(
        unadjusted_panel, adjusted_panel, value_column="signature_auroc"
    ).to_csv(adjusted_dir / f"{donor}_percell_panel_auroc_comparison.csv", index=False)

    values = pd.DataFrame(index=cells.index)
    values.index.name = "barcode"
    values["functional_cluster"] = cells["functional.cluster"].astype(str)
    values["lineage"] = cells["lineage"].astype(str)
    for column in CYCLE_COLUMNS:
        values[column] = cells[column]
    for gene in present_markers:
        values[f"{gene}_log_normalized_expression"] = raw[gene]
        values[f"{gene}_cycle_regressed_residual"] = residuals[gene]
    values.to_csv(adjusted_dir / f"{donor}_marker_cell_values.csv")
    _tpex_tex_evidence(
        donor, raw, residuals, cells["functional.cluster"].astype(str)
    ).to_csv(adjusted_dir / f"{donor}_tpex_tex_gene_evidence.csv", index=False)
    _loss_of_memory_sensitivity(
        donor, raw, residuals, cells["functional.cluster"].astype(str)
    ).to_csv(adjusted_dir / f"{donor}_loss_of_memory_sensitivity.csv", index=False)

    design = np.column_stack(
        [np.ones(len(cells)), cells[["S.Score", "G2M.Score"]].to_numpy(dtype=float)]
    )
    max_orthogonality_error = float(np.abs(design.T @ residuals.to_numpy()).max())
    residual_cycle_correlations = residuals.apply(
        lambda gene: max(
            abs(gene.corr(cells["S.Score"])),
            abs(gene.corr(cells["G2M.Score"])),
        )
    )
    inventory = pd.DataFrame(
        [
            {
                "donor": donor,
                "expression_input": cells.attrs.get("expression_input", "unknown"),
                "projection_input": cells.attrs.get("projection_input", "unknown"),
                "cycle_score_input": cells.attrs.get("cycle_score_input", "unknown"),
                "n_projected_cells": len(cells),
                "n_marker_genes": len(present_markers),
                "max_abs_design_transpose_residual": max_orthogonality_error,
                "max_abs_residual_cycle_correlation": float(
                    residual_cycle_correlations.max()
                ),
            }
        ]
    )
    inventory.to_csv(unadjusted_dir / f"{donor}_input_inventory.csv", index=False)
    print(
        f"[{donor}] {len(cells)} projected cells, {len(present_markers)} markers; "
        f"max |X' residual|={max_orthogonality_error:.3g}"
    )


def verify_historical_reproduction(
    output_root: Path,
    *,
    fixture_path: Path = REPRODUCTION_FIXTURE,
    donors: Sequence[str] = DONORS,
) -> pd.DataFrame:
    """Verify Matt's recorded panels and displayed AUROCs from versioned inputs."""
    fixture = json.loads(fixture_path.read_text())
    displayed_tables = fixture.get("displayed_tables", [])
    expected_table_count = fixture.get("displayed_table_count")
    if (
        not isinstance(expected_table_count, int)
        or len(displayed_tables) != expected_table_count
        or any("rows" not in table for table in displayed_tables)
    ):
        raise ValueError("Matt report fixture does not inventory every displayed table")
    fixture_panels = fixture["marker_panels"]
    implemented_panels = {
        panel: list(genes)
        for panel, genes in SIGNATURES.items()
        if panel in fixture_panels
    }
    if implemented_panels != fixture_panels:
        raise ValueError("versioned marker panels do not match the Matt report fixture")

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
    parser.add_argument("--cycle-root", type=Path, default=CYCLE_ROOT)
    parser.add_argument("--output-root", type=Path, default=REPORT_ROOT)
    parser.add_argument("--reproduction-fixture", type=Path, default=REPRODUCTION_FIXTURE)
    args = parser.parse_args()
    if args.write_markers:
        genes = sorted({gene for genes in SIGNATURES.values() for gene in genes})
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
            cycle_root=args.cycle_root,
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

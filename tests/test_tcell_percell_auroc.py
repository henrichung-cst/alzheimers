from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import json

import numpy as np
import pandas as pd

from alz.analysis.tcell_percell_auroc import (
    calculate_evidence_tables,
    load_donor_data,
    mann_whitney_auroc,
    residualize_marker_expression,
    verify_historical_reproduction,
)


class MannWhitneyAurocTests(unittest.TestCase):
    def test_assigns_half_credit_to_ties(self) -> None:
        self.assertEqual(
            mann_whitney_auroc(
                np.array([3.0, 2.0, 2.0, 1.0]),
                np.array([True, True, False, False]),
            ),
            0.875,
        )

    def test_ignores_nonfinite_values_and_their_labels(self) -> None:
        actual = mann_whitney_auroc(
            np.array([3.0, np.nan, 1.0, np.inf]),
            np.array([True, True, False, False]),
        )
        self.assertEqual(actual, 1.0)


class CycleRegressionTests(unittest.TestCase):
    def test_returns_the_known_residual_after_removing_both_cycle_axes(self) -> None:
        cycle = pd.DataFrame(
            {
                "S.Score": [-1.0, -1.0, 1.0, 1.0],
                "G2M.Score": [-1.0, 1.0, -1.0, 1.0],
            },
            index=["a", "b", "c", "d"],
        )
        expression = pd.DataFrame(
            {"GENE1": [7.0, -1.0, 9.0, 5.0]},
            index=cycle.index,
        )

        residuals = residualize_marker_expression(expression, cycle)

        np.testing.assert_allclose(
            residuals["GENE1"].to_numpy(),
            np.array([1.0, -1.0, -1.0, 1.0]),
            atol=1e-12,
        )
        design = np.column_stack(
            [np.ones(len(cycle)), cycle["S.Score"], cycle["G2M.Score"]]
        )
        np.testing.assert_allclose(design.T @ residuals.to_numpy(), 0.0, atol=1e-12)

    def test_binary_phase_cannot_change_the_adjusted_values(self) -> None:
        expression = pd.DataFrame({"GENE1": [0.0, 1.0, 4.0, 9.0]})
        first = pd.DataFrame(
            {
                "S.Score": [-0.2, 0.1, 0.4, 0.7],
                "G2M.Score": [0.6, 0.2, -0.1, -0.5],
                "Phase": ["G1", "S", "S", "G2M"],
            }
        )
        second = first.copy()
        second["Phase"] = list(reversed(second["Phase"]))

        pd.testing.assert_frame_equal(
            residualize_marker_expression(expression, first),
            residualize_marker_expression(expression, second),
        )


class EvidenceTableTests(unittest.TestCase):
    def test_type_question_uses_only_opposite_lineage_as_comparison(self) -> None:
        cells = pd.DataFrame(
            {
                "functional.cluster": [
                    "CD8.TEX",
                    "CD8.EM",
                    "CD4.Th17",
                    "CD4.Tfh",
                ],
                "CD8A": [3.0, 2.0, 0.0, 0.0],
                "CD4": [0.0, 0.0, 2.0, 3.0],
            },
            index=["a", "b", "c", "d"],
        )
        signatures = {"cd8_lineage": ["CD8A"], "cd4_lineage": ["CD4"]}

        _, panels = calculate_evidence_tables("test", cells, signatures=signatures)
        row = panels[
            (panels["state"] == "CD8.TEX")
            & (panels["panel"] == "cd8_lineage")
        ].iloc[0]

        self.assertEqual(row["comparison"], "opposite-lineage CD4 cells")
        self.assertEqual(row["n_cells_target"], 1)
        self.assertEqual(row["n_cells_comparison"], 2)

    def test_state_question_uses_only_same_lineage_siblings(self) -> None:
        cells = pd.DataFrame(
            {
                "functional.cluster": [
                    "CD8.TEX",
                    "CD8.EM",
                    "CD4.Th17",
                    "CD4.Tfh",
                ],
                "HAVCR2": [3.0, 0.0, 1.0, 2.0],
            },
            index=["a", "b", "c", "d"],
        )

        markers, panels = calculate_evidence_tables(
            "test", cells, signatures={"exhaustion": ["HAVCR2"]}
        )
        row = panels[panels.state.eq("CD8.TEX")].iloc[0]

        self.assertEqual(row["comparison"], "same-lineage sibling CD8 states")
        self.assertEqual(row["n_cells_target"], 1)
        self.assertEqual(row["n_cells_comparison"], 1)
        self.assertTrue(markers[["n_cells_target", "n_cells_comparison"]].notna().all().all())
        self.assertTrue(panels[["n_cells_target", "n_cells_comparison"]].notna().all().all())

    def test_residual_tables_do_not_call_above_zero_values_detected_rna(self) -> None:
        cells = pd.DataFrame(
            {
                "functional.cluster": ["CD8.TEX", "CD8.EM"],
                "HAVCR2": [0.25, -0.25],
            }
        )

        markers, _ = calculate_evidence_tables(
            "test",
            cells,
            signatures={"exhaustion": ["HAVCR2"]},
            value_kind="cycle_regressed_residual",
        )

        self.assertTrue(markers["target_detection_fraction"].isna().all())
        self.assertTrue(markers["comparison_detection_fraction"].isna().all())
        self.assertEqual(
            markers.loc[0, "marker_value_unit"],
            "cycle-regressed log-normalized-expression residual",
        )


class DonorInputTests(unittest.TestCase):
    def test_barcode_join_preserves_each_projected_cell_once(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            expression_root = root / "expression"
            embedding_root = root / "embedding"
            cycle_root = root / "cycle"
            expression_root.mkdir()
            (embedding_root / "donor" / "scrna").mkdir(parents=True)
            cycle_root.mkdir()

            pd.DataFrame(
                {"barcode": ["a", "b", "c"], "GENE1": [1.0, 2.0, 3.0]}
            ).to_csv(expression_root / "donor_marker_cell_expr.csv", index=False)
            pd.DataFrame(
                {
                    "barcode": ["a", "a", "b", "b"],
                    "projection_reference": ["CD8", "CD8", "CD4", "CD4"],
                    "reduction": ["pca", "umap", "pca", "umap"],
                    "functional.cluster": ["CD8.TEX", "CD8.TEX", "CD4.Th17", "CD4.Th17"],
                    "functional.cluster.conf": [0.9, 0.9, 0.8, 0.8],
                }
            ).to_csv(
                embedding_root / "donor" / "scrna" / "projectils_embeddings.csv",
                index=False,
            )
            pd.DataFrame(
                {
                    "barcode": ["a", "b", "c"],
                    "Phase": ["S", "G1", "G2M"],
                    "S.Score": [0.5, -0.2, 0.1],
                    "G2M.Score": [0.0, -0.1, 0.7],
                }
            ).to_csv(cycle_root / "donor_cc_recluster_cells.csv", index=False)

            joined = load_donor_data(
                "donor",
                expression_root=expression_root,
                embedding_root=embedding_root,
                cycle_root=cycle_root,
            )

            self.assertEqual(joined.index.tolist(), ["a", "b"])
            self.assertTrue(joined.index.is_unique)
            self.assertEqual(joined.loc["a", "functional.cluster"], "CD8.TEX")
            self.assertEqual(joined.loc["b", "S.Score"], -0.2)

    def test_cycle_scores_embedded_by_the_extractor_need_no_cluster_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            expression_root = root / "expression"
            embedding_root = root / "embedding"
            cycle_root = root / "missing-cycle-directory"
            expression_root.mkdir()
            (embedding_root / "donor" / "scrna").mkdir(parents=True)
            pd.DataFrame(
                {
                    "barcode": ["a", "b"],
                    "GENE1": [1.0, 2.0],
                    "Phase": ["S", "G1"],
                    "S.Score": [0.12345, -0.23456],
                    "G2M.Score": [-0.34567, 0.45678],
                }
            ).to_csv(expression_root / "donor_marker_cell_expr.csv", index=False)
            pd.DataFrame(
                {
                    "barcode": ["a", "b"],
                    "projection_reference": ["CD8", "CD4"],
                    "reduction": ["pca", "pca"],
                    "functional.cluster": ["CD8.TEX", "CD4.Th17"],
                    "functional.cluster.conf": [0.9, 0.8],
                }
            ).to_csv(
                embedding_root / "donor" / "scrna" / "projectils_embeddings.csv",
                index=False,
            )

            joined = load_donor_data(
                "donor",
                expression_root=expression_root,
                embedding_root=embedding_root,
                cycle_root=cycle_root,
            )

            self.assertEqual(joined.loc["a", "S.Score"], 0.12345)
            self.assertEqual(joined.loc["b", "G2M.Score"], 0.45678)


class HistoricalReproductionTests(unittest.TestCase):
    def test_verifies_one_donor_from_the_method_fixture(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = root / "fixture.json"
            fixture.write_text(
                json.dumps(
                    {
                        "marker_panels": {},
                        "displayed_table_count": 1,
                        "displayed_tables": [
                            {"table_index": 0, "rows": [["header"]]}
                        ],
                        "inputs": {
                            "expression_pattern": "expression/{donor}.csv",
                            "projection_pattern": "projection/{donor}.csv",
                            "projected_cells": {"donor1": 4},
                            "marker_genes_present": {"donor1": 1},
                        },
                        "displayed_state_target_counts": {
                            "donor1": {"CD8.TEX": 3}
                        },
                        "displayed_panel_auroc_rounded_3dp": {
                            "donor1": {"CD8.TEX|exhaustion": 0.672}
                        },
                    }
                )
            )
            results_dir = root / "reproduced_unadjusted"
            results_dir.mkdir()
            pd.DataFrame(
                {
                    "state": ["CD8.TEX"],
                    "panel": ["exhaustion"],
                    "signature_auroc": [0.671976],
                    "n_cells_target": [3],
                }
            ).to_csv(
                results_dir / "donor1_historical_percell_panel_auroc.csv",
                index=False,
            )
            pd.DataFrame(
                {
                    "expression_input": ["expression/donor1.csv"],
                    "projection_input": ["projection/donor1.csv"],
                    "cycle_score_input": ["cycle/donor1.csv"],
                    "n_projected_cells": [4],
                    "n_marker_genes": [1],
                }
            ).to_csv(results_dir / "donor1_input_inventory.csv", index=False)

            check = verify_historical_reproduction(
                root, fixture_path=fixture, donors=("donor1",)
            )
            self.assertEqual(len(check), 1)
            self.assertTrue(check.loc[0, "matches"])


if __name__ == "__main__":
    unittest.main()

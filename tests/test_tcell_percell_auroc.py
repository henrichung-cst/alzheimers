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


class EvidenceTableTests(unittest.TestCase):
    def test_type_question_reproduces_matts_all_other_cells_comparison(self) -> None:
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

        self.assertEqual(row["comparison"], "all other projected T cells")
        self.assertEqual(row["n_cells_target"], 1)
        self.assertEqual(row["n_cells_comparison"], 3)

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

class DonorInputTests(unittest.TestCase):
    def test_barcode_join_preserves_each_projected_cell_once(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            expression_root = root / "expression"
            embedding_root = root / "embedding"
            expression_root.mkdir()
            (embedding_root / "donor" / "scrna").mkdir(parents=True)

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
            joined = load_donor_data(
                "donor",
                expression_root=expression_root,
                embedding_root=embedding_root,
            )

            self.assertEqual(joined.index.tolist(), ["a", "b"])
            self.assertTrue(joined.index.is_unique)
            self.assertEqual(joined.loc["a", "functional.cluster"], "CD8.TEX")
            self.assertEqual(joined.loc["b", "GENE1"], 2.0)


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

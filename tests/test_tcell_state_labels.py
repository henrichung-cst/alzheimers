from __future__ import annotations

import unittest

import pandas as pd

from alz.analysis.tcell_state_labels import (
    assign_marker_states,
    classify_lineage,
    classify_projectils_quality,
    collapse_state_label,
)


class PerCellMarkerStateTests(unittest.TestCase):
    def test_best_state_requires_every_signed_module_to_support_it(self) -> None:
        signed = pd.DataFrame(
            {
                "state_a": [0.5, 0.5],
                "state_b": [0.8, -0.1],
            },
            index=["positive", "negative"],
        )

        self.assertEqual(assign_marker_states(signed), "state_a")

    def test_no_eligible_state_retains_lineage_name(self) -> None:
        signed = pd.DataFrame(
            {"state_a": [-0.1, 0.4], "state_b": [0.0, 0.8]},
            index=["positive", "negative"],
        )

        self.assertEqual(assign_marker_states(signed, fallback="CD8"), "CD8")

    def test_exact_tie_retains_lineage_name(self) -> None:
        signed = pd.DataFrame(
            {"state_a": [0.5, 0.4], "state_b": [0.4, 0.4]},
            index=["positive", "negative"],
        )

        self.assertEqual(assign_marker_states(signed, fallback="CD4"), "CD4")

    def test_lineage_uses_raw_adt_before_cluster_fallback(self) -> None:
        cell = pd.Series(
            {"CD4_protein_umi": 4, "CD8_protein_umi": 80, "mouse_isotype_umi": 5}
        )

        self.assertEqual(classify_lineage(cell, "CD4"), ("CD8", "ADT CD8-dominant"))

    def test_projection_confidence_one_is_categorical_corroboration_only(self) -> None:
        self.assertEqual(classify_projectils_quality("CD8.TEX", 1.0), "unanimous")
        self.assertEqual(classify_projectils_quality("CD8.TEX", 0.99), "projected")

    def test_provisional_tpex_is_collapsed_to_lineage_only_cd8(self) -> None:
        self.assertEqual(collapse_state_label("CD8 precursor exhausted (TPEX)"), "CD8")


if __name__ == "__main__":
    unittest.main()

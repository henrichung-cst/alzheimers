from __future__ import annotations

import unittest

import pandas as pd

from alz.analysis.tcell_marker_sets import PER_CELL_STATE_DEFINITIONS
from alz.analysis.tcell_state_labels import (
    assign_marker_states,
    classify_lineage,
    classify_projectils_quality,
    collapse_state_label,
    hierarchical_exhaustion_call,
)


class PerCellMarkerStateTests(unittest.TestCase):
    def test_naive_like_requires_positive_stemness_and_homing_modules(self) -> None:
        definition = PER_CELL_STATE_DEFINITIONS["CD8 naive-like"]
        self.assertEqual(
            [module.name for module in definition.positive_modules],
            ["naive stemness", "naive homing"],
        )

    def test_resting_memory_has_its_own_positive_identity_module(self) -> None:
        definition = PER_CELL_STATE_DEFINITIONS["CD8 resting/memory"]
        self.assertEqual(
            [module.name for module in definition.positive_modules],
            ["resting/memory identity"],
        )
        self.assertEqual(definition.positive_modules[0].genes, ("IL7R", "CD27"))

    def test_exhaustion_uses_one_late_receptor_and_tf_signature(self) -> None:
        definition = PER_CELL_STATE_DEFINITIONS["CD8 exhausted (TEX)"]
        self.assertEqual(
            [module.name for module in definition.positive_modules],
            ["late exhaustion signature"],
        )
        self.assertEqual(
            definition.positive_modules[0].genes,
            ("HAVCR2", "LAG3", "ENTPD1", "TOX", "NR4A1"),
        )

    def test_activation_requires_acute_and_effector_programs(self) -> None:
        definition = PER_CELL_STATE_DEFINITIONS["CD8 activated/effector"]
        self.assertEqual(
            [module.name for module in definition.positive_modules],
            ["acute activation", "effector function"],
        )

    def test_hierarchical_exhaustion_must_exceed_both_counterweights(self) -> None:
        self.assertTrue(hierarchical_exhaustion_call(0.5, 0.2, 0.4))
        self.assertFalse(hierarchical_exhaustion_call(0.5, 0.6, 0.1))
        self.assertFalse(hierarchical_exhaustion_call(0.5, 0.1, 0.6))
        self.assertFalse(hierarchical_exhaustion_call(-0.1, -0.3, -0.2))

    def test_best_state_requires_every_signed_module_to_support_it(self) -> None:
        signed = pd.DataFrame(
            {
                "state_a": [0.5, 0.5],
                "state_b": [0.8, -0.1],
            },
            index=["higher:identity", "higher:support"],
        )

        self.assertEqual(
            assign_marker_states(
                signed,
                eligible_states={"state_a", "state_b"},
            ),
            "state_a",
        )

    def test_no_eligible_state_retains_lineage_name(self) -> None:
        signed = pd.DataFrame(
            {"state_a": [-0.1, 0.4], "state_b": [0.0, 0.8]},
            index=["positive", "negative"],
        )

        self.assertEqual(
            assign_marker_states(signed, fallback="CD8", eligible_states=set()),
            "CD8",
        )

    def test_positive_eligibility_can_override_imperfect_negative_evidence(self) -> None:
        signed = pd.DataFrame(
            {"resting/memory": [0.6, -0.2], "activated": [0.8, 0.4]},
            index=["higher:identity", "lower:exhaustion"],
        )
        self.assertEqual(
            assign_marker_states(
                signed,
                fallback="CD8",
                eligible_states={"resting/memory"},
            ),
            "resting/memory",
        )

    def test_exact_tie_retains_lineage_name(self) -> None:
        signed = pd.DataFrame(
            {"state_a": [0.5, 0.4], "state_b": [0.5, 0.4]},
            index=["higher:identity", "lower:counter-program"],
        )

        self.assertEqual(
            assign_marker_states(
                signed,
                fallback="CD4",
                eligible_states={"state_a", "state_b"},
            ),
            "CD4",
        )

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

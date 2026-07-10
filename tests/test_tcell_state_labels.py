from __future__ import annotations

import unittest

import pandas as pd

from alz.analysis.tcell_state_labels import (
    classify_cd4_state,
    classify_cd8_state,
    classify_lineage,
    classify_projectils_quality,
    exhaustion_corroboration,
)


def _cd8_cell(**overrides: object) -> pd.Series:
    values: dict[str, object] = {
        "Phase": "G1",
        "HAVCR2": 0.0,
        "LAG3": 0.0,
        "ENTPD1": 0.0,
        "PDCD1": 0.0,
        "TCF7": 0.0,
        "LEF1": 0.0,
        "SELL": 0.0,
        "CCR7": 0.0,
        "IL7R": 0.0,
        "GZMB": 0.0,
        "GZMH": 0.0,
        "GNLY": 0.0,
        "PRF1": 0.0,
    }
    values.update(overrides)
    return pd.Series(values)


class Cd8StateClassificationTests(unittest.TestCase):
    def test_checkpoint_memory_label_requires_coherent_memory_rna(self) -> None:
        cell = _cd8_cell(
            TCF7=1.0,
            LEF1=1.0,
            HAVCR2=1.0,
            LAG3=1.0,
        )

        self.assertEqual(
            classify_cd8_state(cell),
            "CD8 precursor exhausted",
        )

    def test_checkpoint_coexpression_is_called_exhausted(self) -> None:
        cell = _cd8_cell(HAVCR2=1.0, ENTPD1=1.0)

        self.assertEqual(classify_cd8_state(cell), "CD8 exhausted")

    def test_memory_label_does_not_require_checkpoint_expression(self) -> None:
        cell = _cd8_cell(
            TCF7=1.0,
            CCR7=1.0,
        )

        self.assertEqual(classify_cd8_state(cell), "CD8 memory")

    def test_cytotoxic_label_requires_perforin_and_granzyme_evidence(self) -> None:
        cell = _cd8_cell(PRF1=1.0, GZMB=1.0)

        self.assertEqual(classify_cd8_state(cell), "CD8 cytotoxic")

    def test_cell_without_specific_state_evidence_is_called_effector(self) -> None:
        self.assertEqual(classify_cd8_state(_cd8_cell()), "CD8 effector")


class Cd4StateClassificationTests(unittest.TestCase):
    def test_cell_cycle_is_assigned_per_cell(self) -> None:
        cell = _cd8_cell(Phase="S")

        self.assertEqual(classify_cd4_state(cell), "CD4 proliferating")

    def test_noncycling_memory_program_is_described_without_helper_subtype(self) -> None:
        cell = _cd8_cell(TCF7=1.0, IL7R=1.0)

        self.assertEqual(classify_cd4_state(cell), "CD4 memory")

    def test_noncycling_cytotoxic_program_is_described_directly(self) -> None:
        cell = _cd8_cell(PRF1=1.0, GZMH=1.0)

        self.assertEqual(classify_cd4_state(cell), "CD4 cytotoxic")

    def test_noncycling_cell_without_program_is_reported_as_g1_scored(self) -> None:
        self.assertEqual(classify_cd4_state(_cd8_cell()), "CD4 resting")


class LineageClassificationTests(unittest.TestCase):
    def test_cd4_dominant_adt_assigns_cd4(self) -> None:
        cell = pd.Series(
            {"CD4_protein_umi": 120, "CD8_protein_umi": 8, "mouse_isotype_umi": 5}
        )

        self.assertEqual(classify_lineage(cell, "CD8"), ("CD4", "ADT CD4-dominant"))

    def test_cd8_dominant_adt_assigns_cd8(self) -> None:
        cell = pd.Series(
            {"CD4_protein_umi": 7, "CD8_protein_umi": 95, "mouse_isotype_umi": 5}
        )

        self.assertEqual(classify_lineage(cell, "CD4"), ("CD8", "ADT CD8-dominant"))

    def test_cluster_lineage_is_fallback_when_adt_is_not_above_background(self) -> None:
        cell = pd.Series(
            {"CD4_protein_umi": 4, "CD8_protein_umi": 3, "mouse_isotype_umi": 5}
        )

        self.assertEqual(classify_lineage(cell, "CD4"), ("CD4", "cluster fallback"))


class ProjectilsEvidenceTests(unittest.TestCase):
    def test_exact_unanimity_has_a_distinct_categorical_call(self) -> None:
        self.assertEqual(classify_projectils_quality("CD8.TEX", 1.0), "unanimous")

    def test_weighted_majority_is_supported_without_an_arbitrary_high_cutoff(self) -> None:
        self.assertEqual(
            classify_projectils_quality("CD8.TEX", 0.51),
            "majority-supported",
        )

    def test_half_or_less_is_ambiguous(self) -> None:
        self.assertEqual(classify_projectils_quality("CD8.TEX", 0.5), "ambiguous")

    def test_missing_projection_is_reported_categorically(self) -> None:
        self.assertEqual(classify_projectils_quality(pd.NA, pd.NA), "not projected")

    def test_unanimous_tex_corroborates_direct_exhaustion_call(self) -> None:
        self.assertEqual(
            exhaustion_corroboration(
                label="CD8 exhausted",
                lineage="CD8",
                projectils_state="CD8.TEX",
                projectils_quality="unanimous",
            ),
            "unanimous CD8.TEX",
        )

    def test_tpex_reference_is_retained_but_does_not_overwrite_state(self) -> None:
        self.assertEqual(
            exhaustion_corroboration(
                label="CD8 precursor exhausted",
                lineage="CD8",
                projectils_state="CD8.TPEX",
                projectils_quality="majority-supported",
            ),
            "CD8.TPEX reference support",
        )


if __name__ == "__main__":
    unittest.main()

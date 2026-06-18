from __future__ import annotations

from pathlib import Path
import tempfile
import sys
import unittest

import pandas as pd

_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from alz.core.mechanism_attribution import (
    classify_mechanism_files,
    classify_mechanisms,
    main,
)
from alz.shared import config


class MechanismAttributionTests(unittest.TestCase):
    def _run(
        self,
        stoich: pd.DataFrame,
        raw: pd.DataFrame,
        context_cols: list[str] | None = None,
    ) -> pd.DataFrame:
        return classify_mechanisms(
            stoich,
            raw,
            context_cols=context_cols or ["contrast"],
            fdr_thresh=config.MEA_FDR_THRESH,
        )

    def test_both_significant_same_sign(self) -> None:
        stoich = pd.DataFrame(
            {
                "kinase": ["K1"],
                "NES": [2.0],
                "FDR": [0.01],
                "contrast": ["c1"],
            }
        )
        raw = pd.DataFrame(
            {
                "kinase": ["K1"],
                "NES": [1.0],
                "FDR": [0.02],
                "contrast": ["c1"],
            }
        )

        result = self._run(stoich, raw)
        row = result.iloc[0]
        self.assertEqual(row["mechanism_call"], "both")
        self.assertEqual(row["sign_relation"], "same")
        self.assertTrue(row["stoich_significant"])
        self.assertTrue(row["raw_significant"])

    def test_stoich_only(self) -> None:
        stoich = pd.DataFrame(
            {
                "kinase": ["K1"],
                "NES": [-2.0],
                "FDR": [0.01],
                "contrast": ["c1"],
            }
        )
        raw = pd.DataFrame(
            {
                "kinase": ["K1"],
                "NES": [-1.0],
                "FDR": [0.50],
                "contrast": ["c1"],
            }
        )

        result = self._run(stoich, raw)
        row = result.iloc[0]
        self.assertEqual(row["mechanism_call"], "activity_driven")
        self.assertEqual(row["sign_relation"], "stoich_only")
        self.assertTrue(row["stoich_significant"])
        self.assertFalse(row["raw_significant"])

    def test_raw_only(self) -> None:
        stoich = pd.DataFrame(
            {
                "kinase": ["K1"],
                "NES": [2.0],
                "FDR": [0.80],
                "contrast": ["c1"],
            }
        )
        raw = pd.DataFrame(
            {
                "kinase": ["K1"],
                "NES": [1.0],
                "FDR": [0.02],
                "contrast": ["c1"],
            }
        )

        result = self._run(stoich, raw)
        row = result.iloc[0]
        self.assertEqual(row["mechanism_call"], "abundance_driven")
        self.assertEqual(row["sign_relation"], "raw_only")
        self.assertFalse(row["stoich_significant"])
        self.assertTrue(row["raw_significant"])

    def test_both_significant_opposite_sign(self) -> None:
        stoich = pd.DataFrame(
            {
                "kinase": ["K1"],
                "NES": [2.0],
                "FDR": [0.01],
                "contrast": ["c1"],
            }
        )
        raw = pd.DataFrame(
            {
                "kinase": ["K1"],
                "NES": [-1.0],
                "FDR": [0.02],
                "contrast": ["c1"],
            }
        )

        result = self._run(stoich, raw)
        row = result.iloc[0]
        self.assertEqual(row["mechanism_call"], "discordant")
        self.assertEqual(row["sign_relation"], "opposite")
        self.assertTrue(row["stoich_significant"])
        self.assertTrue(row["raw_significant"])

    def test_not_significant_neither_side(self) -> None:
        stoich = pd.DataFrame(
            {
                "kinase": ["K1"],
                "NES": [2.0],
                "FDR": [0.30],
                "contrast": ["c1"],
            }
        )
        raw = pd.DataFrame(
            {
                "kinase": ["K1"],
                "NES": [1.0],
                "FDR": [0.40],
                "contrast": ["c1"],
            }
        )

        result = self._run(stoich, raw)
        row = result.iloc[0]
        self.assertEqual(row["mechanism_call"], "not_significant")
        self.assertEqual(row["sign_relation"], "none")
        self.assertFalse(row["stoich_significant"])
        self.assertFalse(row["raw_significant"])

    def test_missing_raw_row(self) -> None:
        stoich = pd.DataFrame(
            {
                "kinase": ["K1"],
                "NES": [2.0],
                "FDR": [0.01],
                "contrast": ["c1"],
            }
        )
        raw = pd.DataFrame(columns=["kinase", "NES", "FDR", "contrast"])

        result = self._run(stoich, raw)
        row = result.iloc[0]
        self.assertEqual(row["mechanism_call"], "not_evaluable")
        self.assertEqual(row["sign_relation"], "not_evaluable")
        self.assertEqual(row["skip_reason"], "missing_raw_row")

    def test_missing_stoich_row(self) -> None:
        stoich = pd.DataFrame(columns=["kinase", "NES", "FDR", "contrast"])
        raw = pd.DataFrame(
            {
                "kinase": ["K1"],
                "NES": [1.0],
                "FDR": [0.01],
                "contrast": ["c1"],
            }
        )

        result = self._run(stoich, raw)
        row = result.iloc[0]
        self.assertEqual(row["mechanism_call"], "not_evaluable")
        self.assertEqual(row["sign_relation"], "not_evaluable")
        self.assertEqual(row["skip_reason"], "missing_stoich_row")

    def test_duplicate_rows_are_not_evaluable(self) -> None:
        stoich = pd.DataFrame(
            {
                "kinase": ["K1", "K1"],
                "NES": [2.0, 1.5],
                "FDR": [0.01, 0.02],
                "contrast": ["c1", "c1"],
            }
        )
        raw = pd.DataFrame(
            {
                "kinase": ["K1"],
                "NES": [1.0],
                "FDR": [0.01],
                "contrast": ["c1"],
            }
        )

        result = self._run(stoich, raw)
        self.assertEqual(len(result), 1)
        row = result.iloc[0]
        self.assertEqual(row["mechanism_call"], "not_evaluable")
        self.assertEqual(row["skip_reason"], "duplicate_pair_rows")

    def test_custom_context_columns_are_preserved(self) -> None:
        stoich = pd.DataFrame(
            {
                "kinase": ["K1", "K2"],
                "NES": [2.0, -1.5],
                "FDR": [0.01, 0.10],
                "contrast": ["c1", "c2"],
                "donor": ["d1", "d2"],
                "tissue": ["tA", "tB"],
                "track": ["st", "st"],
            }
        )
        raw = pd.DataFrame(
            {
                "kinase": ["K1", "K2"],
                "NES": [2.0, 1.0],
                "FDR": [0.02, 0.20],
                "contrast": ["c1", "c2"],
                "donor": ["d1", "d2"],
                "tissue": ["tA", "tB"],
                "track": ["st", "st"],
            }
        )

        result = self._run(stoich, raw, context_cols=["contrast", "donor", "tissue", "track"])
        self.assertEqual(set(result.columns), set(
            [
                "contrast",
                "donor",
                "tissue",
                "track",
                "kinase",
                "stoich_NES",
                "stoich_FDR",
                "raw_NES",
                "raw_FDR",
                "stoich_significant",
                "raw_significant",
                "sign_relation",
                "mechanism_call",
                "skip_reason",
            ]
        ))
        self.assertEqual(result.loc[result["kinase"] == "K1", "mechanism_call"].iloc[0], "both")
        self.assertEqual(result.loc[result["kinase"] == "K2", "mechanism_call"].iloc[0], "discordant")

    def test_classify_mechanism_files_writes_output_with_constants(self) -> None:
        stoich = pd.DataFrame(
            {
                "kinase": ["K1", "K2"],
                "NES": [2.0, -2.0],
                "FDR": [0.01, 0.02],
                "contrast": ["c1", "c1"],
                "donor": ["d1", "d1"],
            }
        )
        raw = pd.DataFrame(
            {
                "kinase": ["K1", "K2"],
                "NES": [1.0, 2.5],
                "FDR": [0.02, 0.50],
                "contrast": ["c1", "c1"],
                "donor": ["d1", "d1"],
            }
        )

        with tempfile.TemporaryDirectory() as td:
            stoich_path = Path(td) / "stoich.csv"
            raw_path = Path(td) / "raw.csv"
            out_path = Path(td) / "out.csv"
            stoich.to_csv(stoich_path, index=False)
            raw.to_csv(raw_path, index=False)

            result = classify_mechanism_files(
                stoich_path,
                raw_path,
                out_path,
                context_cols=["contrast", "donor"],
                cohort="song",
                extra_constant_cols={"track": "st"},
            )

            self.assertEqual(result["cohort"].iloc[0], "song")
            self.assertEqual(result["track"].iloc[0], "st")
            self.assertTrue(out_path.exists())
            from_disk = pd.read_csv(out_path)
            pd.testing.assert_frame_equal(result.fillna(""), from_disk.fillna(""))

    def test_main_help_smoke(self) -> None:
        with self.assertRaises(SystemExit):
            main(["--help"])


if __name__ == "__main__":
    unittest.main()

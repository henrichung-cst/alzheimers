from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import sys
import unittest
from unittest.mock import patch

import pandas as pd

_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import alz.tcell_viewer.slices_kinase as slices_kinase


class ProjectedStateViewerPayloadTests(unittest.TestCase):
    def test_projected_state_payload_absent_when_files_absent(self) -> None:
        with TemporaryDirectory() as td:
            with patch.object(slices_kinase, "KINASE_ATTRIBUTION_TCELLS_DIR", td):
                self.assertIsNone(slices_kinase._load_projected_state_mea_payload())

    def test_projected_state_payload_reads_existing_optional_files(self) -> None:
        with TemporaryDirectory() as td:
            root = Path(td)
            out_dir = root / "donor1" / "state_mea"
            out_dir.mkdir(parents=True)
            pd.DataFrame(
                {
                    "kinase": ["AKT1"],
                    "track": ["st"],
                    "state": ["CD8Tex"],
                    "timepoint": ["d13"],
                    "contrast": ["d13_vs_d2"],
                    "NES": [1.2],
                    "FDR": [0.05],
                }
            ).to_csv(out_dir / "mea_projected_state.csv", index=False)
            pd.DataFrame(
                {
                    "cohort": ["tcells"],
                    "donor": ["donor1"],
                    "track": ["st"],
                    "state": ["CD8Tex"],
                    "timepoint": ["d13"],
                    "contrast": ["d13_vs_d2"],
                    "kinase": ["AKT1"],
                    "stoich_NES": [1.2],
                    "stoich_FDR": [0.05],
                    "raw_NES": [0.1],
                    "raw_FDR": [0.8],
                    "stoich_significant": [True],
                    "raw_significant": [False],
                    "sign_relation": ["stoich_only"],
                    "mechanism_call": ["activity_driven"],
                    "skip_reason": [""],
                }
            ).to_csv(out_dir / "mechanism_attribution_projected_state.csv", index=False)

            with patch.object(slices_kinase, "KINASE_ATTRIBUTION_TCELLS_DIR", str(root)):
                payload = slices_kinase._load_projected_state_mea_payload()

        self.assertIsNotNone(payload)
        donor = payload["by_context"]["donor1"]
        self.assertEqual(donor["tracks"], ["st"])
        self.assertEqual(donor["states"], ["CD8Tex"])
        self.assertEqual(donor["timepoints"], ["d13"])
        self.assertEqual(donor["rows"][0]["kind"], "stoich")
        self.assertEqual(
            donor["mechanism_attribution"][0]["mechanism_call"],
            "activity_driven",
        )
        self.assertNotIn("mechanism_score", donor["mechanism_attribution"][0])


if __name__ == "__main__":
    unittest.main()

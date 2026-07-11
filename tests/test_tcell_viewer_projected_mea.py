from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import pandas as pd

from alz.tcell_viewer.slices_kinase import (
    _load_projected_state_mea_payload,
    _load_tcell_attribution,
)
from alz.tcell_viewer.paths import resolve_incytr_pair_mode_tcells_dir


class ProjectedStateMeaViewerTests(unittest.TestCase):
    def test_viewer_defaults_to_completed_positive_negative_marker_run(self) -> None:
        selected = Path(resolve_incytr_pair_mode_tcells_dir({}))

        self.assertEqual(
            selected.name,
            "incytr_pair_mode_tcells_percell_posneg",
        )

    def test_current_attribution_subset_is_retained(self) -> None:
        current_subset = pd.DataFrame([{
            "kinase": "JAK1",
            "cell_type": "CD8Exhausted",
        }])
        with (
            patch("alz.tcell_viewer.slices_kinase.os.path.exists", return_value=True),
            patch(
                "alz.tcell_viewer.slices_kinase.pd.read_csv",
                return_value=current_subset,
            ),
            patch(
                "alz.tcell_viewer.slices_kinase.load_donor_states",
                return_value=["CD8Exhausted", "CD8RestingMemory"],
            ),
        ):
            observed = _load_tcell_attribution("donor1")

        self.assertIs(observed, current_subset)

    def test_stale_attribution_roster_is_not_mixed_with_current_labels(self) -> None:
        stale = pd.DataFrame([{
            "kinase": "JAK1",
            "cell_type": "CD8PrecursorExhausted",
        }])
        with (
            patch("alz.tcell_viewer.slices_kinase.os.path.exists", return_value=True),
            patch("alz.tcell_viewer.slices_kinase.pd.read_csv", return_value=stale),
            patch(
                "alz.tcell_viewer.slices_kinase.load_donor_states",
                return_value=["CD8Exhausted"],
            ),
        ):
            self.assertIsNone(_load_tcell_attribution("donor1"))

    def test_payload_exports_raw_projected_mea_only(self) -> None:
        with TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            row = {
                "kinase": "JAK1",
                "track": "py",
                "state": "CD8Exhausted",
                "timepoint": "d20",
                "contrast": "d20_vs_d2",
                "NES": 1.8,
                "FDR": 0.01,
            }
            pd.DataFrame([row]).to_csv(
                out_dir / "mea_projected_state_raw.csv", index=False
            )
            pd.DataFrame([{**row, "NES": 9.9}]).to_csv(
                out_dir / "mea_projected_state.csv", index=False
            )
            pd.DataFrame([{
                "kinase": "JAK1",
                "state": "CD8Exhausted",
                "stoich_NES": 9.9,
                "raw_NES": 1.8,
                "mechanism_call": "abundance-driven",
            }]).to_csv(
                out_dir / "mechanism_attribution_projected_state.csv", index=False
            )
            (out_dir / "projected_state_mea_manifest.json").write_text(
                json.dumps([{
                    "donor": "donor1",
                    "state": "CD8Exhausted",
                    "track": "py",
                    "kind": "projected_state",
                    "baseline_day": 2,
                    "days_available": [2, 20],
                    "days_run": [20],
                }])
            )

            with (
                patch("alz.tcell_viewer.slices_kinase.DONORS", ("donor1",)),
                patch(
                    "alz.tcell_viewer.slices_kinase._projected_state_candidate_dirs",
                    return_value=[str(out_dir)],
                ),
                patch(
                    "alz.tcell_viewer.slices_kinase.load_donor_states",
                    return_value=["CD8Exhausted"],
                ),
            ):
                payload = _load_projected_state_mea_payload()

        self.assertIsNotNone(payload)
        block = payload["by_context"]["donor1"]
        self.assertEqual(block["rows"], [{**row, "kind": "raw"}])
        self.assertEqual(block["interpretation"], "raw_projected_state_mea")
        self.assertNotIn("mechanism_attribution", block)
        self.assertFalse(any("mechanism_attribution" in p for p in block["source_files"]))
        self.assertFalse(any("mea_projected_state.csv" in p for p in block["source_files"]))


if __name__ == "__main__":
    unittest.main()

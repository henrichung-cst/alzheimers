from __future__ import annotations

from pathlib import Path
import unittest

from alz.tcell_viewer.paths import resolve_incytr_pair_mode_tcells_dir


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "alz" / "analysis" / "run_tcell_original_report.sh"
README = ROOT / "outputs" / "reports" / "tcell_labeling" / "README.md"


class OriginalReportRestorationTests(unittest.TestCase):
    def test_primary_runner_preserves_original_report_and_native_umap(self) -> None:
        self.assertTrue(RUNNER.is_file())
        runner = RUNNER.read_text()

        self.assertIn("tcell_state_labeling_evidence_original.html", runner)
        self.assertIn("umap/umap_label_comparison.png", runner)

    def test_primary_runner_does_not_apply_cycle_adjustment(self) -> None:
        runner = RUNNER.read_text().lower()

        for excluded in (
            "cellcycle",
            "cycle_regressed",
            "s.score",
            "g2m.score",
            "recluster",
        ):
            self.assertNotIn(excluded, runner)

    def test_workspace_declares_original_report_as_preserved_historical_baseline(self) -> None:
        readme = README.read_text().lower()

        self.assertIn("tcell_state_labeling_evidence_original.html", readme)
        self.assertIn("preserved historical", readme)
        self.assertIn("tcell_state_labeling_evidence_percell.html", readme)
        self.assertNotIn("current deliverable is\n[`tcell_state_labeling_evidence_cycle_regressed.html`", readme)

    def test_viewer_defaults_to_completed_current_state_incytr_run(self) -> None:
        expected = (
            ROOT / "outputs" / "reports" /
            "incytr_pair_mode_tcells"
        )

        self.assertEqual(Path(resolve_incytr_pair_mode_tcells_dir({})), expected)


if __name__ == "__main__":
    unittest.main()

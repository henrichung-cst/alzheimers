from __future__ import annotations

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "alz" / "analysis" / "run_tcell_labeling.sh"
REPORT = ROOT / "outputs" / "reports" / "tcell_labeling" / "tcell_state_labeling_evidence_percell.qmd"
STATE_SCRIPT = ROOT / "alz" / "analysis" / "tcell_state_labels.py"


class PerCellReportTests(unittest.TestCase):
    def test_runner_builds_per_cell_labels_evidence_umaps_and_report(self) -> None:
        runner = RUNNER.read_text()
        self.assertIn("tcell_state_labels.py", runner)
        self.assertIn("tcell_state_evidence.py", runner)
        self.assertIn("tcell_native_umap_plots.py", runner)
        self.assertIn("tcell_state_labeling_evidence_percell.qmd", runner)

    def test_report_documents_signed_markers_lineage_fallback_and_cycle_exclusion(self) -> None:
        report = REPORT.read_text().lower()
        self.assertIn("assigned independently to each cell", report)
        self.assertIn("expected to be lower", report)
        self.assertIn("remains simply `cd4` or `cd8`", report)
        self.assertIn("% dividing", report)
        self.assertIn("are not loaded or used", report)

    def test_classifier_does_not_load_cycle_fields_or_use_projection_for_labels(self) -> None:
        source = STATE_SCRIPT.read_text()
        self.assertNotIn('"Phase"', source)
        self.assertNotIn('"S.Score"', source)
        self.assertNotIn('"G2M.Score"', source)
        self.assertIn("never determine the label", source)


if __name__ == "__main__":
    unittest.main()

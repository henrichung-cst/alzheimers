from __future__ import annotations

from pathlib import Path
import re
import unittest

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "alz" / "analysis" / "run_tcell_labeling.sh"
REPORT = ROOT / "outputs" / "reports" / "tcell_labeling" / "tcell_state_labeling_evidence_percell.qmd"
STATE_SCRIPT = ROOT / "alz" / "analysis" / "tcell_state_labels.py"
PANEL_EVIDENCE = (
    ROOT
    / "outputs"
    / "reports"
    / "tcell_labeling"
    / "percell_evidence"
    / "state_panel_evidence.csv"
)


class PerCellReportTests(unittest.TestCase):
    def test_runner_builds_per_cell_labels_evidence_umaps_and_report(self) -> None:
        runner = RUNNER.read_text()
        self.assertIn("tcell_state_labels.py", runner)
        self.assertIn("tcell_state_evidence.py", runner)
        self.assertIn("tcell_native_umap_plots.py", runner)
        self.assertIn("tcell_state_labeling_evidence_percell.qmd", runner)

    def test_report_explains_per_cell_markers_fallback_and_cycle_exclusion(self) -> None:
        report = REPORT.read_text().lower()
        self.assertIn("each cell receives its own state label", report)
        self.assertIn("markers expected to be low", report)
        self.assertIn("direct marker requirements", report)
        self.assertIn("naive-like", report)
        self.assertIn("resting/memory", report)
        self.assertIn("the label remains", report)
        self.assertIn("simply `cd4` or `cd8`", report)
        self.assertIn("% dividing", report)
        self.assertIn("are excluded", report)
        self.assertNotIn("claim boundary", report)

    def test_report_limits_state_separation_auroc_to_defining_markers(self) -> None:
        report = REPORT.read_text()
        prose = " ".join(report.split())
        self.assertIn("Only marker sets that define a state are shown", prose)
        self.assertIn("comparisons made within each cell", prose)
        self.assertNotIn('"Markers expected to be lower"', report)

        chunks = re.findall(r"```\{python\}\n(.*?)```", report, flags=re.DOTALL)
        table_code = next(
            chunk for chunk in chunks if "positive_marker_separation =" in chunk
        )
        namespace = {
            "PANELS": pd.read_csv(PANEL_EVIDENCE),
            "simplify_marker_text": lambda value: value,
        }
        exec(table_code, namespace)
        displayed = namespace["positive_marker_separation"]
        exhausted = displayed[
            displayed["Assigned state"].eq("CD8 exhausted (TEX)")
        ]
        self.assertEqual(
            set(exhausted["Marker set"]),
            {"late exhaustion signature"},
        )

    def test_classifier_does_not_load_cycle_fields_or_use_projection_for_labels(self) -> None:
        source = STATE_SCRIPT.read_text()
        self.assertNotIn('"Phase"', source)
        self.assertNotIn('"S.Score"', source)
        self.assertNotIn('"G2M.Score"', source)
        self.assertIn("never determine the label", source)


if __name__ == "__main__":
    unittest.main()

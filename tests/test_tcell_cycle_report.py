from __future__ import annotations

import json
from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "alz" / "analysis" / "fixtures" / "tcell_matt_report_expected.json"
REPORT = (
    ROOT
    / "outputs"
    / "reports"
    / "tcell_labeling"
    / "tcell_state_labeling_evidence_cycle_regressed.qmd"
)


class MattReportStructureTests(unittest.TestCase):
    def test_cycle_regressed_report_preserves_matts_top_level_sections(self) -> None:
        expected = json.loads(FIXTURE.read_text())["historical_sections"]
        observed = re.findall(r"^## (?!#)(.+)$", REPORT.read_text(), flags=re.MULTILINE)

        self.assertEqual(observed, expected)

    def test_report_is_a_rerun_not_an_audit_of_matt(self) -> None:
        report = REPORT.read_text().lower()

        self.assertNotIn("method-corrected", report)
        self.assertNotIn("historical reproduction", report)
        self.assertNotIn("sensitivity analysis", report)
        self.assertIn("matt's original analysis rerun with one change", report)


if __name__ == "__main__":
    unittest.main()

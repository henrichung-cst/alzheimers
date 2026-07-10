from __future__ import annotations

import os
import subprocess
import tempfile
import time
import unittest
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


REPO_ROOT = Path(__file__).resolve().parents[1]
STATUS_SCRIPT = REPO_ROOT / "alz/incytr_pair/tcells_status.sh"


class TCellStatusTest(unittest.TestCase):
    def run_status(self, root: Path) -> str:
        result = subprocess.run(
            ["bash", str(STATUS_SCRIPT), "--root", str(root)],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout

    @contextmanager
    def fake_incytr_process(self, output_root: Path) -> Iterator[None]:
        environment = os.environ.copy()
        environment["OUTPUT_DIR_OVERRIDE"] = str(output_root)
        process = subprocess.Popen(
            ["bash", "-c", "exec -a incytr_commandline.R sleep 30"],
            cwd=REPO_ROOT,
            env=environment,
        )
        try:
            time.sleep(0.05)
            yield
        finally:
            process.terminate()
            process.wait(timeout=5)

    def test_reports_completed_mea_and_incytr_stages(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "overnight_20260710_220000.log").write_text(
                "\n".join(
                    [
                        "=== 2026-07-10T22:00:00-04:00 overnight t-cell run start ===",
                        "=== 2026-07-10T22:00:01-04:00 [1/2] donor1 projected-state kinase MEA (st + py) ===",
                        "=== 2026-07-10T22:00:02-04:00 [1/2] kinase preflight start ===",
                        "=== 2026-07-10T22:01:01-04:00 [1/2] kinase preflight done ===",
                        "=== 2026-07-10T22:01:02-04:00 [1/2] kinase MEA start ===",
                        "=== 2026-07-10T22:20:01-04:00 [1/2] done ===",
                        "=== 2026-07-10T22:20:01-04:00 [1/2] kinase MEA done ===",
                        "=== 2026-07-10T22:20:02-04:00 [2/2] pair-mode (donor2: 4 contrasts; donor1: 3 contrasts; nboot=100) ===",
                        "=== 2026-07-11T03:00:00-04:00 [2/2] done ===",
                        "=== 2026-07-11T03:00:00-04:00 T-CELL OVERNIGHT RUN COMPLETE ===",
                    ]
                )
                + "\n"
            )
            (root / "pair_run.log").write_text(
                "=== 2026-07-11T02:00:00-04:00 [donor1] d20 vs d2 (nboot=100) ===\n"
            )
            for donor, days in (("donor2", (5, 7, 9, 11)), ("donor1", (13, 17, 20))):
                wide = root / donor / "wide"
                wide.mkdir(parents=True)
                for day in days:
                    (wide / f"d{day}_d2_incytr_output.parquet").write_text("complete\n")

            output = self.run_status(root)

            self.assertIn("T-cell rerun · kinase MEA → Incytr", output)
            self.assertIn("Preflight", output)
            self.assertIn("✓ 0m", output)
            self.assertIn("Kinase MEA", output)
            self.assertIn("✓ 18m", output)
            self.assertIn("Incytr", output)
            self.assertIn("7/7 contrasts complete", output)
            self.assertIn("DONE", output)

    def test_reports_interruption_during_mea(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "overnight_20260710_220000.log").write_text(
                "\n".join(
                    [
                        "=== 2026-07-10T22:00:00-04:00 overnight t-cell run start ===",
                        "=== 2026-07-10T22:00:01-04:00 [1/2] donor1 projected-state kinase MEA (st + py) ===",
                        "=== 2026-07-10T22:00:02-04:00 [1/2] kinase preflight start ===",
                        "=== 2026-07-10T22:01:01-04:00 [1/2] kinase preflight done ===",
                        "=== 2026-07-10T22:01:02-04:00 [1/2] kinase MEA start ===",
                    ]
                )
                + "\n"
            )

            output = self.run_status(root)

            self.assertIn("STOPPED", output)
            self.assertIn("Preflight", output)
            self.assertIn("Kinase MEA", output)
            self.assertIn("⊘ interrupted", output)
            self.assertIn("Incytr", output)
            self.assertIn("· pending", output)

    def test_reports_direct_incytr_completion_without_overnight_log(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for donor, days in (("donor2", (5, 7, 9, 11)), ("donor1", (13, 17, 20))):
                wide = root / donor / "wide"
                wide.mkdir(parents=True)
                for day in days:
                    (wide / f"d{day}_d2_incytr_output.parquet").write_text("complete\n")

            output = self.run_status(root)

            self.assertIn("DONE", output)
            self.assertIn("Incytr        ✓ complete", output)
            self.assertIn("7/7 contrasts complete", output)

    def test_live_direct_incytr_ignores_stale_overnight_log(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "overnight_20260709_220000.log").write_text(
                "\n".join(
                    [
                        "=== 2026-07-09T22:00:00-04:00 overnight t-cell run start ===",
                        "=== 2026-07-09T22:00:01-04:00 [1/2] donor1 projected-state kinase MEA (st + py) ===",
                        "=== 2026-07-09T22:20:01-04:00 [1/2] done ===",
                        "=== 2026-07-10T03:00:00-04:00 T-CELL OVERNIGHT RUN COMPLETE ===",
                    ]
                )
                + "\n"
            )
            (root / "pair_run.log").write_text("direct Incytr activity\n")

            with self.fake_incytr_process(root):
                output = self.run_status(root)

            self.assertIn("RUNNING", output)
            self.assertIn("Preflight     · not scheduled", output)
            self.assertIn("Kinase MEA    · not scheduled", output)
            self.assertIn("Incytr        ▶ running", output)
            self.assertIn("Last: direct Incytr activity", output)
            self.assertNotIn("Run log:", output)

    def test_process_for_another_root_does_not_change_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as other:
            root = Path(tmp)
            (root / "overnight_20260710_220000.log").write_text(
                "\n".join(
                    [
                        "=== 2026-07-10T22:00:00-04:00 overnight t-cell run start ===",
                        "=== 2026-07-10T22:00:01-04:00 [1/2] donor1 projected-state kinase MEA (st + py) ===",
                    ]
                )
                + "\n"
            )

            with self.fake_incytr_process(Path(other)):
                output = self.run_status(root)

            self.assertIn("STOPPED", output)
            self.assertNotIn("RUNNING", output)

    def test_process_for_descendant_root_does_not_match_parent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "overnight_20260710_220000.log").write_text(
                "\n".join(
                    [
                        "=== 2026-07-10T22:00:00-04:00 overnight t-cell run start ===",
                        "=== 2026-07-10T22:00:01-04:00 [1/2] donor1 projected-state kinase MEA (st + py) ===",
                    ]
                )
                + "\n"
            )

            with self.fake_incytr_process(root / "another-run"):
                output = self.run_status(root)

            self.assertIn("STOPPED", output)
            self.assertNotIn("RUNNING", output)

    def test_stopped_direct_run_supersedes_stale_completed_overnight_log(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            overnight_log = root / "overnight_20260709_220000.log"
            overnight_log.write_text(
                "\n".join(
                    [
                        "=== 2026-07-09T22:00:00-04:00 overnight t-cell run start ===",
                        "=== 2026-07-09T22:00:01-04:00 [1/2] donor1 projected-state kinase MEA (st + py) ===",
                        "=== 2026-07-09T22:20:01-04:00 [1/2] done ===",
                        "=== 2026-07-10T03:00:00-04:00 T-CELL OVERNIGHT RUN COMPLETE ===",
                    ]
                )
                + "\n"
            )
            pair_log = root / "pair_run.log"
            pair_log.write_text(
                "=== 2026-07-10T12:00:00-04:00 [donor2] d5 vs d2 (nboot=100) ===\n"
            )
            status_dir = root / "donor2" / "wide"
            status_dir.mkdir(parents=True)
            (status_dir / ".status_d5_d2.txt").write_text(
                "started 2026-07-10T12:00:00-04:00\n"
            )
            os.utime(overnight_log, (100, 100))
            os.utime(pair_log, (200, 200))

            output = self.run_status(root)

            self.assertIn("STOPPED", output)
            self.assertIn("Preflight     · not scheduled", output)
            self.assertIn("Incytr        ⊘ interrupted", output)
            self.assertIn("1 interrupted", output)
            self.assertNotIn("Run log:", output)


if __name__ == "__main__":
    unittest.main()

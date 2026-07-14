from __future__ import annotations

from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCAN = ROOT / "alz" / "analysis" / "tcell_incytr_cellcell_scan.py"

CONTRASTS = {
    "donor1": (13, 17, 20),
    "donor2": (5, 7, 9, 11),
}


def _write_surface(root: Path) -> None:
    for donor, days in CONTRASTS.items():
        wide = root / donor / "wide"
        wide.mkdir(parents=True)
        for index, day in enumerate(days):
            rising_center = 0.4 + (0.2 if donor == "donor2" else 0.3) * index
            falling = -0.4 - 0.1 * index
            erratic = 0.5 if index % 2 == 0 else -0.5
            rows = [
                {
                    "Ligand": "L1",
                    "Receptor": "R1",
                    "EM": "E1",
                    "Target": "T1",
                    "Sender.group": "CD8Exhausted",
                    "Receiver.group": "CD4",
                    "PDS": rising_center - 0.1,
                },
                {
                    "Ligand": "L1",
                    "Receptor": "R1",
                    "EM": "E1",
                    "Target": "T2",
                    "Sender.group": "CD8Exhausted",
                    "Receiver.group": "CD4",
                    "PDS": rising_center + 0.1,
                },
                {
                    "Ligand": "L2",
                    "Receptor": "R2",
                    "EM": "E2",
                    "Target": "T3",
                    "Sender.group": "CD4ActivatedEffector",
                    "Receiver.group": "CD8RestingMemory",
                    "PDS": falling,
                },
                {
                    "Ligand": "L3",
                    "Receptor": "R3",
                    "EM": "E3",
                    "Target": "T4",
                    "Sender.group": "CD4",
                    "Receiver.group": "CD8Exhausted",
                    "PDS": erratic,
                },
            ]
            if index < 2:
                rows.append(
                    {
                        "Ligand": "L4",
                        "Receptor": "R4",
                        "EM": "E4",
                        "Target": "T5",
                        "Sender.group": "CD8RestingMemory",
                        "Receiver.group": "CD4",
                        "PDS": 0.2 + 0.1 * index,
                    }
                )
            pd.DataFrame(rows).to_parquet(
                wide / f"d{day}_d2_incytr_output.parquet", index=False
            )


def _write_labels(root: Path) -> None:
    root.mkdir(parents=True)
    for donor, days in CONTRASTS.items():
        rows = []
        for day in (2, *days):
            counts = {
                "CD8Exhausted": 1 if day == 2 else 2,
                "CD4ActivatedEffector": 4 if day == 2 else 3,
                "CD4": 1,
                "CD8RestingMemory": 2,
            }
            for state, count in counts.items():
                for cell in range(count):
                    rows.append(
                        {
                            "barcode": f"{donor}_{day}_{state}_{cell}",
                            "donor": donor,
                            "day": day,
                            "type": state,
                        }
                    )
            rows.append(
                {
                    "barcode": f"{donor}_{day}_excluded",
                    "donor": donor,
                    "day": day,
                    "type": None,
                }
            )
        pd.DataFrame(rows).to_csv(root / f"{donor}_state_labels.csv", index=False)


class TCellIncytrCellCellScanTests(unittest.TestCase):
    def test_cli_writes_bounded_cell_channel_evidence(self) -> None:
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source = tmp_path / "surface"
            labels = tmp_path / "labels"
            output = tmp_path / "derived"
            _write_surface(source)
            _write_labels(labels)

            subprocess.run(
                [
                    sys.executable,
                    str(SCAN),
                    "--input-dir",
                    str(source),
                    "--labels-dir",
                    str(labels),
                    "--output-dir",
                    str(output),
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )

            macro = pd.read_csv(output / "macro_summary.csv")
            states = pd.read_csv(output / "state_role_summary.csv")
            channels = pd.read_csv(output / "channel_summary.csv")
            trends = pd.read_csv(output / "channel_trends.csv")
            backbones = pd.read_csv(output / "channel_top_backbones.csv")

        d13 = macro[macro.donor.eq("donor1") & macro.day.eq(13)].iloc[0]
        self.assertEqual(d13.path_count, 5)
        self.assertAlmostEqual(d13.up_pds_mass, 1.5)
        self.assertAlmostEqual(d13.down_pds_mass, -0.4)

        baseline = states[
            states.donor.eq("donor1")
            & states.day.eq(2)
            & states.state.eq("CD8Exhausted")
            & states.role.eq("sender")
        ].iloc[0]
        self.assertTrue(baseline.is_baseline)
        self.assertTrue(pd.isna(baseline.path_count))
        self.assertAlmostEqual(baseline.cell_fraction, 1 / 8)

        exhausted_sender = states[
            states.donor.eq("donor1")
            & states.day.eq(13)
            & states.state.eq("CD8Exhausted")
            & states.role.eq("sender")
        ].iloc[0]
        self.assertEqual(exhausted_sender.path_count, 2)
        self.assertEqual(exhausted_sender.up_path_count, 2)
        self.assertEqual(exhausted_sender.down_path_count, 0)
        self.assertAlmostEqual(exhausted_sender.up_pds_mass, 0.8)
        self.assertAlmostEqual(exhausted_sender.cell_fraction, 2 / 8)

        stateless = states[
            states.donor.eq("donor1")
            & states.day.eq(13)
            & states.state.eq("CD4")
            & states.role.eq("sender")
        ].iloc[0]
        self.assertEqual(stateless.state_class, "stateless")
        self.assertEqual(stateless.lineage, "CD4")

        channel = channels[
            channels.donor.eq("donor1")
            & channels.day.eq(13)
            & channels.sender_state.eq("CD8Exhausted")
            & channels.receiver_state.eq("CD4")
        ].iloc[0]
        self.assertEqual(channel.path_count, 2)
        self.assertAlmostEqual(channel.mean_pds, 0.4)
        self.assertEqual(channel.target_count, 2)

        trend = trends[
            trends.donor.eq("donor1")
            & trends.sender_state.eq("CD8Exhausted")
            & trends.receiver_state.eq("CD4")
        ].iloc[0]
        self.assertEqual(trend.direction, "rising")
        self.assertEqual(trend.trend_consistency, "consistent")
        self.assertEqual(trend.driver_rank, 1)
        self.assertEqual(trend.cross_donor_agreement, "same_direction")

        erratic = trends[
            trends.donor.eq("donor1")
            & trends.sender_state.eq("CD4")
            & trends.receiver_state.eq("CD8Exhausted")
        ].iloc[0]
        self.assertEqual(erratic.trend_consistency, "mixed")
        self.assertTrue(pd.isna(erratic.driver_rank))

        partial = trends[
            trends.donor.eq("donor1")
            & trends.sender_state.eq("CD8RestingMemory")
            & trends.receiver_state.eq("CD4")
        ].iloc[0]
        self.assertEqual(partial.trend_consistency, "partial_coverage")
        self.assertTrue(pd.isna(partial.driver_rank))

        backbone = backbones[
            backbones.donor.eq("donor1")
            & backbones.sender_state.eq("CD8Exhausted")
            & backbones.receiver_state.eq("CD4")
            & backbones.ligand.eq("L1")
        ].iloc[0]
        self.assertEqual(backbone.target_fan_count, 2)
        self.assertEqual(backbone.backbone_rank, 1)

    def test_scan_never_materializes_wide_parquet_with_pandas(self) -> None:
        source = SCAN.read_text()

        self.assertIn("read_parquet", source)
        self.assertNotIn("pd.read_parquet", source)
        self.assertNotIn("pandas.read_parquet", source)


if __name__ == "__main__":
    unittest.main()

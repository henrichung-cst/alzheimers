from __future__ import annotations

from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCAN = ROOT / "alz" / "analysis" / "tcell_incytr_scan.py"


def _write_surface(root: Path) -> None:
    contrasts = {
        "donor1": {"d13_d2": -0.2, "d17_d2": -0.4, "d20_d2": -0.6},
        "donor2": {
            "d5_d2": -0.1,
            "d7_d2": -0.2,
            "d9_d2": -0.3,
            "d11_d2": -0.4,
        },
    }
    for donor, donor_contrasts in contrasts.items():
        wide = root / donor / "wide"
        wide.mkdir(parents=True)
        for contrast, il2_pds in donor_contrasts.items():
            rows = [
                {
                    "Ligand": "IL2",
                    "Receptor": "IL2RA",
                    "EM": "JAK1",
                    "Target": "STAT5A",
                    "Sender.group": "CD8Activated",
                    "Receiver.group": "CD8Exhausted",
                    "PDS": il2_pds,
                },
                {
                    "Ligand": "IL2",
                    "Receptor": "IL2RA",
                    "EM": "JAK3",
                    "Target": "STAT5B",
                    "Sender.group": "CD4Activated",
                    "Receiver.group": "CD8Exhausted",
                    "PDS": -0.8 if donor == "donor1" else -0.5,
                },
                {
                    "Ligand": "HMGB1",
                    "Receptor": "HAVCR2",
                    "EM": "SRC",
                    "Target": "NFATC1",
                    "Sender.group": "CD8Exhausted",
                    "Receiver.group": "CD8Exhausted",
                    "PDS": 0.9,
                },
                {
                    "Ligand": "VIM",
                    "Receptor": "CD44",
                    "EM": "SRC",
                    "Target": "PDCD1",
                    "Sender.group": "CD8Exhausted",
                    "Receiver.group": "CD8Exhausted",
                    "PDS": 0.1,
                },
                {
                    "Ligand": "GAPDH",
                    "Receptor": "LAMP1",
                    "EM": "SRC",
                    "Target": "CTLA4",
                    "Sender.group": "CD8Exhausted",
                    "Receiver.group": "CD8Exhausted",
                    "PDS": 0.2,
                },
                {
                    "Ligand": "HLA-DRA",
                    "Receptor": "LAG3",
                    "EM": "SRC",
                    "Target": "TOX",
                    "Sender.group": "CD8Exhausted",
                    "Receiver.group": "CD8Exhausted",
                    "PDS": 1.0,
                },
                {
                    "Ligand": "WBP1",
                    "Receptor": "CTLA4",
                    "EM": "SRC",
                    "Target": "TOX",
                    "Sender.group": "CD8Exhausted",
                    "Receiver.group": "CD8Exhausted",
                    "PDS": 0.3,
                },
            ]
            pd.DataFrame(rows).to_parquet(
                wide / f"{contrast}_incytr_output.parquet", index=False
            )


class TCellIncytrScanTests(unittest.TestCase):
    def test_cli_writes_edge_anchor_novelty_and_concordance_evidence(self) -> None:
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source = tmp_path / "surface"
            output = tmp_path / "derived"
            _write_surface(source)

            result = subprocess.run(
                [
                    sys.executable,
                    str(SCAN),
                    "--input-dir",
                    str(source),
                    "--output-dir",
                    str(output),
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )

            surface = pd.read_csv(output / "surface_summary.csv")
            edge = pd.read_csv(output / "edge_evidence.csv")
            status = pd.read_csv(output / "anchor_status.csv")
            anchor = pd.read_csv(output / "anchor_evidence.csv")
            trends = pd.read_csv(output / "edge_trends.csv")
            concordance = pd.read_csv(output / "cross_donor_concordance.csv")
            novel = pd.read_csv(output / "novel_candidates.csv")

        d13 = surface[
            surface.donor.eq("donor1") & surface.contrast.eq("d13_d2")
        ].iloc[0]
        self.assertEqual(d13.path_count, 7)
        self.assertEqual(d13.edge_count, 6)
        self.assertEqual(d13.sender_count, 3)
        self.assertEqual(d13.receiver_count, 1)

        il2 = edge[
            edge.donor.eq("donor1")
            & edge.contrast.eq("d13_d2")
            & edge.ligand.eq("IL2")
            & edge.receptor.eq("IL2RA")
        ].iloc[0]
        self.assertEqual(il2.path_count, 2)
        self.assertAlmostEqual(il2.best_abs_pds, 0.8)
        self.assertAlmostEqual(il2.mean_abs_pds, 0.5)
        self.assertAlmostEqual(il2.mean_pds, -0.5)
        self.assertEqual(il2.rank_band, "below_top_quartile")

        il2_status = status[
            status.ligand.eq("IL2") & status.receptor.eq("IL2RA")
        ].iloc[0]
        self.assertEqual(il2_status.tier, "tier1_reconstructable")
        self.assertTrue(il2_status.scored)

        pd1_status = status[
            status.ligand.eq("CD274") & status.receptor.eq("PDCD1")
        ].iloc[0]
        self.assertEqual(pd1_status.tier, "tier2_target_observed")
        self.assertFalse(pd1_status.scored)
        self.assertEqual(pd1_status.receptor_as_target_paths, 7)
        self.assertEqual(pd1_status.receptor_role_paths, 0)
        self.assertEqual(pd1_status.receptor_role_status, "target_only")

        ctla4_status = status[
            status.ligand.eq("CD86") & status.receptor.eq("CTLA4")
        ].iloc[0]
        self.assertEqual(ctla4_status.tier, "tier2_target_observed")
        self.assertEqual(ctla4_status.receptor_as_target_paths, 7)
        self.assertEqual(ctla4_status.receptor_role_paths, 7)
        self.assertEqual(ctla4_status.receptor_role_status, "target_and_receptor")

        il15_status = status[
            status.ligand.eq("IL15") & status.receptor.eq("IL15RA")
        ].iloc[0]
        self.assertEqual(il15_status.tier, "absent_no_target_evidence")

        il2_anchor = anchor[
            anchor.donor.eq("donor1")
            & anchor.contrast.eq("d13_d2")
            & anchor.ligand.eq("IL2")
            & anchor.receptor.eq("IL2RA")
        ].iloc[0]
        self.assertTrue(il2_anchor.formed)
        self.assertAlmostEqual(il2_anchor.best_abs_pds, 0.8)

        il2_trend = trends[
            trends.donor.eq("donor1")
            & trends.ligand.eq("IL2")
            & trends.receptor.eq("IL2RA")
        ].iloc[0]
        self.assertEqual(il2_trend.direction, "falling")
        self.assertLess(il2_trend.pds_per_day, 0)
        self.assertFalse(
            trends.ligand.eq("HLA-DRA").any(),
            "MHC-derived edges must not receive temporal scores",
        )

        il2_concordance = concordance[
            concordance.ligand.eq("IL2")
            & concordance.receptor.eq("IL2RA")
        ].iloc[0]
        self.assertEqual(il2_concordance.concordance, "same_direction")
        self.assertFalse(
            concordance.ligand.eq("HLA-DRA").any(),
            "MHC-derived edges must not receive concordance calls",
        )

        hmgb1 = novel[
            novel.ligand.eq("HMGB1") & novel.receptor.eq("HAVCR2")
        ]
        self.assertEqual(set(hmgb1.rank_band), {"top_quartile"})
        self.assertFalse(
            novel.ligand.eq("HLA-DRA").any(),
            "MHC-derived edges are an a-priori family, not novel candidates",
        )
        self.assertIn(
            "donor1 d13_d2 IL2 → IL2RA: best |PDS|=0.800",
            result.stdout,
        )
        self.assertIn("HMGB1 → HAVCR2", result.stdout)

    def test_scan_source_never_materializes_a_wide_parquet_with_pandas(self) -> None:
        source = SCAN.read_text()

        self.assertIn("read_parquet", source)
        self.assertNotIn("pd.read_parquet", source)
        self.assertNotIn("pandas.read_parquet", source)


if __name__ == "__main__":
    unittest.main()

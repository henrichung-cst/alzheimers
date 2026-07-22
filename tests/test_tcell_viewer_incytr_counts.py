from __future__ import annotations

import gzip
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import duckdb
import numpy as np
import pandas as pd

from alz.tcell_viewer import slices_kinase
from alz.tcell_viewer.slices_incytr import (
    _contrast_pvalue_column,
    _pvalue_filter_sql,
)


class TcellViewerIncytrCountTests(unittest.TestCase):
    def test_no_pvalue_gate_counts_every_canonical_row(self) -> None:
        con = duckdb.connect()
        try:
            con.execute("CREATE TABLE paths(pvalue DOUBLE, pds DOUBLE)")
            con.execute(
                "INSERT INTO paths VALUES (0.01, 0.4), (1.0, -0.3), (NULL, 0.2)"
            )

            open_count = con.execute(
                f"SELECT COUNT(*) FILTER (WHERE {_pvalue_filter_sql(1.0, True)} "
                "AND ABS(pds) >= 0.01) FROM paths"
            ).fetchone()[0]
            gated_count = con.execute(
                f"SELECT COUNT(*) FILTER (WHERE {_pvalue_filter_sql(0.05, True)} "
                "AND ABS(pds) >= 0.01) FROM paths"
            ).fetchone()[0]
        finally:
            con.close()

        self.assertEqual(open_count, 3)
        self.assertEqual(gated_count, 1)

    def test_pvalue_column_tracks_the_follow_up_day(self) -> None:
        columns = {"p_value_d2", "p_value_d13", "p_value_d13_WTyp"}

        self.assertEqual(
            _contrast_pvalue_column(columns, "d13_d2"),
            "p_value_d13",
        )
        self.assertIsNone(_contrast_pvalue_column(columns, "d17_d2"))

    def test_terminal_edge_participation_matches_role_contrast_and_receiver(self) -> None:
        arrays = {
            "ligandId": [0, 0, 0, 0, 0],
            "receptorId": [0, 3, 0, 0, 1],
            "emId": [1, 3, 2, 1, 1],
            "targetId": [2, 0, 3, 2, 3],
            "receiverId": [0, 0, 1, 0, 0],
            "contrastId": [0, 0, 0, 1, 0],
        }
        columns = [
            {"name": name, "type": "u2" if name.endswith("Id") and name not in {"receiverId", "contrastId"} else "u1", "bytes": 0}
            for name in arrays
        ]
        for column in columns:
            column["bytes"] = np.asarray(
                arrays[column["name"]], dtype=np.dtype("<" + column["type"])
            ).nbytes
        raw = b"".join(
            np.asarray(arrays[column["name"]], dtype=np.dtype("<" + column["type"])).tobytes()
            for column in columns
        )
        terminal_edges = pd.DataFrame(
            [
                ("K1", "A", "Receptor", "D1_d13_vs_d2", "R1"),
                ("K1", "A", "Receptor", "D1_d13_vs_d2", "R1"),
                ("K1", "D", "Target", "D1_d13_vs_d2", "R2"),
                ("K1", "B", "EM", "D1_d17_vs_d2", "R1"),
                ("K2", "A", "Ligand", "D1_d13_vs_d2", "R1"),
                ("K2", "C", "Target", "D1_d13_vs_d2", "R1"),
                ("K2", "C", "EM", "D1_d13_vs_d2", "R2"),
            ],
            columns=["kinase", "target_gene", "role", "contrast", "owning_cluster"],
        )

        with TemporaryDirectory() as tmp:
            index_path = Path(tmp) / "index.bin.gz"
            with gzip.open(index_path, "wb") as fh:
                fh.write(raw)
            global_index = {
                "url": index_path.name,
                "nrows": len(next(iter(arrays.values()))),
                "receiver_vocab": ["R1", "R2"],
                "contrast_vocab": ["d13_d2", "d17_d2"],
                "gene_vocab": ["A", "B", "C", "D"],
                "columns": columns,
            }
            with patch.object(slices_kinase, "UNIFIED_VIEWER_DIR", tmp), patch.object(
                slices_kinase, "_read_terminal_edges", return_value=terminal_edges
            ):
                pathway, backbone, total = slices_kinase._incytr_pathway_participation(
                    "donor1",
                    {"global_index": global_index},
                    [("K1", "ST"), ("K1", "Y"), ("K2", "ST"), ("K3", "ST")],
                )

        self.assertEqual(total, 5)
        self.assertEqual(pathway, [3, 3, 2, 0])
        self.assertEqual(backbone, [2, 2, 1, 0])
        self.assertTrue(all(b <= p for b, p in zip(backbone, pathway)))


if __name__ == "__main__":
    unittest.main()

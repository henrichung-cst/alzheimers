from __future__ import annotations

import gzip
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import duckdb
import numpy as np

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

    def test_substrates_use_leading_edge_on_index_contrasts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "donor1"
            mea = base / "mea"
            mea.mkdir(parents=True)
            (mea / "mea_timecourse.csv").write_text(
                "kinase,contrast,Leading substrates\n"
                "K1,D1_d13_vs_d2,MOTIF_A;MOTIF_B\n"
                "K1,D1_d15_vs_d2,MOTIF_C\n"
                "K2,D1_d13_vs_d2,MOTIF_D\n",
                encoding="utf-8",
            )
            (base / "raw_phospho_normalized.csv").write_text(
                "motif,gene_symbol\n"
                "motif_a,GeneA\n"
                "motif_b,GeneB\n"
                "motif_c,GeneC\n"
                "motif_d,GeneD\n",
                encoding="utf-8",
            )

            with patch.object(slices_kinase, "KINASE_ATTRIBUTION_TCELLS_DIR", tmp):
                result = slices_kinase._substrate_genes_by_kinase(
                    "donor1",
                    "mea_timecourse.csv",
                    "raw_phospho_normalized.csv",
                    ["d13_d2"],
                )

        self.assertEqual(result, {"K1": {"GENEA", "GENEB"}, "K2": {"GENED"}})

    def test_participation_counts_distinct_pathways_and_backbones(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            index_path = Path(tmp) / "incytr_index.bin.gz"
            columns = [
                ("ligandId", "u2", np.array([0, 0, 0, 1, 1], dtype="<u2")),
                ("receptorId", "u2", np.array([1, 1, 1, 1, 1], dtype="<u2")),
                ("emId", "u2", np.array([2, 2, 2, 2, 2], dtype="<u2")),
                ("targetId", "u2", np.array([3, 4, 3, 3, 5], dtype="<u2")),
            ]
            raw = b"".join(values.tobytes() for _, _, values in columns)
            with gzip.open(index_path, "wb") as handle:
                handle.write(raw)

            manifest = {
                "url": index_path.name,
                "nrows": 5,
                "contrast_vocab": ["d13_d2"],
                "gene_vocab": ["LIG", "REC", "EM", "TGT1", "TGT2", "TGT3"],
                "columns": [
                    {"name": name, "type": dtype, "bytes": int(values.nbytes)}
                    for name, dtype, values in columns
                ],
            }
            mea = Path(tmp) / "donor1" / "mea"
            mea.mkdir(parents=True)
            (mea / "mea_timecourse.csv").write_text(
                "kinase,contrast,Leading substrates\n"
                "K1,D1_d13_vs_d2,MOTIF_LIG;MOTIF_TGT2\n",
                encoding="utf-8",
            )
            (Path(tmp) / "donor1" / "raw_phospho_normalized.csv").write_text(
                "motif,gene_symbol\n"
                "MOTIF_LIG,LIG\n"
                "MOTIF_TGT2,TGT2\n",
                encoding="utf-8",
            )

            with patch.object(slices_kinase, "UNIFIED_VIEWER_DIR", tmp), patch.object(
                slices_kinase, "KINASE_ATTRIBUTION_TCELLS_DIR", tmp
            ):
                result = slices_kinase._incytr_pathway_participation(
                    "donor1",
                    {"global_index": manifest},
                    [("K1", "ST")],
                )

        self.assertEqual(result, ([2], [1], 4, 2))


if __name__ == "__main__":
    unittest.main()

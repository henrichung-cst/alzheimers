from __future__ import annotations

import unittest

import duckdb

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


if __name__ == "__main__":
    unittest.main()

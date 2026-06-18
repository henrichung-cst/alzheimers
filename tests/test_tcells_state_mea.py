from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import math
import json
import sys
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from alz.cohorts.tcells.state_mea import (
    StateDayColumn,
    PROJECTED_ROOT,
    PROTEIN_FILENAME,
    TRACK_FILES,
    build_state_timepoint_matrices,
    write_state_timepoint_aggregates,
    build_missing_input_manifest_records,
    run_projected_state_mea,
    build_manifest_records,
    build_state_matrices,
    should_run_state,
    summarize_state_qc,
    load_projected_inputs,
    write_manifest,
    parse_state_day_columns,
    write_projected_state_mechanism_attribution,
)
import alz.cohorts.tcells.state_mea as state_mea


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


class ParseStateDayColumnsTests(unittest.TestCase):
    def test_parse_state_day_columns_extracts_day_and_state_and_ignores_metadata(self) -> None:
        columns = [
            "site_id",
            "gene_symbol",
            "motif",
            "d2_effector",
            "d13_effector",
            "junk",
            "d20_other_state",
            "dX_bad",
        ]
        parsed = parse_state_day_columns(columns)
        self.assertEqual(
            parsed,
            [
                StateDayColumn(column="d2_effector", day=2, state="effector"),
                StateDayColumn(column="d13_effector", day=13, state="effector"),
                StateDayColumn(column="d20_other_state", day=20, state="other_state"),
            ],
        )


class LoadProjectedInputsTests(unittest.TestCase):
    def _write_inputs(self, root: Path) -> None:
        donor_dir = root / PROJECTED_ROOT / "donor1"
        _write_csv(
            donor_dir / PROTEIN_FILENAME,
            pd.DataFrame(
                {
                    "gene_symbol": ["GENE_A", "GENE_B"],
                    "d2_effector": [10, 20],
                    "d13_effector": [20, 30],
                }
            ),
        )
        _write_csv(
            donor_dir / TRACK_FILES["st"],
            pd.DataFrame(
                {
                    "site_id": ["S1", "S2"],
                    "gene_symbol": ["GENE_A", "GENE_B"],
                    "motif": ["SEQ1", "SEQ2"],
                    "d2_effector": [2, 4],
                    "d13_effector": [4, 8],
                }
            ),
        )
        _write_csv(
            donor_dir / TRACK_FILES["py"],
            pd.DataFrame(
                {
                    "site_id": ["S1", "S2"],
                    "gene_symbol": ["GENE_A", "GENE_B"],
                    "motif": ["SEQ3", "SEQ4"],
                    "d2_effector": [5, 6],
                    "d13_effector": [10, 12],
                }
            ),
        )
        _write_csv(
            donor_dir / "scrna" / "cell_counts.csv",
            pd.DataFrame({"state": ["effector", "effector"], "day": [2, 13], "n_cells": [2, 1]}),
        )

    def test_load_projected_inputs_resolves_track_paths_and_reads_cell_counts(self) -> None:
        with TemporaryDirectory() as td:
            root = Path(td)
            self._write_inputs(root)

            st_inputs = load_projected_inputs("donor1", "st", root=root)
            py_inputs = load_projected_inputs("donor1", "py", root=root)

            self.assertEqual(st_inputs.projected_path.name, TRACK_FILES["st"])
            self.assertEqual(py_inputs.projected_path.name, TRACK_FILES["py"])
            self.assertEqual(st_inputs.protein_path.name, PROTEIN_FILENAME)
            self.assertEqual(py_inputs.protein_path.name, PROTEIN_FILENAME)
            self.assertEqual(st_inputs.track, "st")
            self.assertEqual(py_inputs.track, "py")
            self.assertIsNotNone(st_inputs.cell_counts)
            pd.testing.assert_frame_equal(
                st_inputs.cell_counts,
                py_inputs.cell_counts,
            )


class BuildStateMatricesTests(unittest.TestCase):
    def _write_inputs(self, root: Path, *, include_unmatched: bool = False) -> None:
        donor_dir = root / PROJECTED_ROOT / "donor1"
        protein_rows = [
            {"gene_symbol": "GENE_A", "d2_effector": 2, "d13_effector": 8, "d2_other": 10},
            {"gene_symbol": "GENE_B", "d2_effector": 4, "d13_effector": 16, "d2_other": 12},
        ]
        if include_unmatched:
            protein_rows.append({"gene_symbol": "GENE_C", "d2_effector": 9, "d13_effector": 9, "d2_other": 9})
        _write_csv(
            donor_dir / PROTEIN_FILENAME,
            pd.DataFrame(protein_rows),
        )
        _write_csv(
            donor_dir / TRACK_FILES["st"],
            pd.DataFrame(
                {
                    "site_id": ["S1", "S2", "S3"],
                    "gene_symbol": ["GENE_A", "GENE_B", "GENE_Z"],
                    "motif": ["M1", "M2", "M3"],
                    "d2_effector": [4, 6, 3],
                    "d13_effector": [8, 16, 1],
                    "d2_other": [7, 8, 9],
                }
            ),
        )
        _write_csv(
            donor_dir / "scrna" / "cell_counts.csv",
            pd.DataFrame(
                {
                    "state": ["effector", "effector", "other"],
                    "day": [2, 13, 2],
                    "n_cells": [12, 10, 3],
                }
            ),
        )

    def test_build_state_matrices_raw_and_stoich_shapes_and_values(self) -> None:
        with TemporaryDirectory() as td:
            root = Path(td)
            self._write_inputs(root)
            inputs = load_projected_inputs("donor1", "st", root=root)
            matrices = build_state_matrices(inputs, state="effector", track="st")

            raw = matrices["raw"]
            stoich = matrices["stoich"]
            self.assertEqual(raw.shape, (3, 5))
            self.assertEqual(stoich.shape, (3, 5))
            self.assertListEqual(list(raw.columns), ["site_id", "gene_symbol", "motif", "d2", "d13"])

            expected_raw = pd.DataFrame(
                {
                    "site_id": ["S1", "S2", "S3"],
                    "gene_symbol": ["GENE_A", "GENE_B", "GENE_Z"],
                    "motif": ["M1", "M2", "M3"],
                    "d2": [4, 6, 3],
                    "d13": [8, 16, 1],
                }
            )
            pd.testing.assert_frame_equal(raw, expected_raw)

            expected_stoich = pd.DataFrame(
                {
                    "site_id": ["S1", "S2", "S3"],
                    "gene_symbol": ["GENE_A", "GENE_B", "GENE_Z"],
                    "motif": ["M1", "M2", "M3"],
                    "d2": [1.0, math.log2(6 / 4), np.nan],
                    "d13": [0.0, math.log2(16 / 16), np.nan],
                }
            )
            pd.testing.assert_frame_equal(
                stoich,
                expected_stoich,
            )

    def test_non_positive_values_become_nan_before_log2(self) -> None:
        with TemporaryDirectory() as td:
            root = Path(td)
            donor_dir = root / PROJECTED_ROOT / "donor1"
            _write_csv(
                donor_dir / PROTEIN_FILENAME,
                pd.DataFrame(
                    {
                        "gene_symbol": ["GENE_A", "GENE_B", "GENE_C"],
                        "d2_effector": [2.0, 0.0, -1.0],
                        "d13_effector": [8.0, 4.0, 1.0],
                    }
                ),
            )
            _write_csv(
                donor_dir / TRACK_FILES["st"],
                pd.DataFrame(
                    {
                        "site_id": ["S1", "S2", "S3"],
                        "gene_symbol": ["GENE_A", "GENE_B", "GENE_C"],
                        "motif": ["M1", "M2", "M3"],
                        "d2_effector": [4.0, 2.0, -1.0],
                        "d13_effector": [8.0, 2.0, 8.0],
                    }
                ),
            )
            _write_csv(donor_dir / "scrna" / "cell_counts.csv", pd.DataFrame({"state": ["x"], "day": [1], "n_cells": [3]}))

            inputs = load_projected_inputs("donor1", "st", root=root)
            stoich = build_state_matrices(inputs, state="effector", track="st")["stoich"]

            self.assertTrue(np.isnan(stoich.loc[1, "d2"]))
            self.assertTrue(np.isnan(stoich.loc[2, "d2"]))
            self.assertEqual(stoich.loc[0, "d13"], 0.0)
            self.assertEqual(stoich.loc[1, "d13"],  -1.0)
            self.assertEqual(stoich.loc[2, "d13"], 3.0)

    def test_unmatched_protein_gene_results_in_nan_stoich(self) -> None:
        with TemporaryDirectory() as td:
            root = Path(td)
            self._write_inputs(root, include_unmatched=False)
            inputs = load_projected_inputs("donor1", "st", root=root)
            stoich = build_state_matrices(inputs, state="effector", track="st")["stoich"]

            unmatched_row = stoich.loc[stoich["gene_symbol"] == "GENE_Z"].iloc[0]
            self.assertTrue(math.isnan(unmatched_row["d2"]))
            self.assertTrue(math.isnan(unmatched_row["d13"]))


class StateQcTests(unittest.TestCase):
    def _write_inputs(self, root: Path, *, with_zero_post: bool = False) -> None:
        donor_dir = root / PROJECTED_ROOT / "donor1"
        _write_csv(
            donor_dir / PROTEIN_FILENAME,
            pd.DataFrame(
                {
                    "gene_symbol": ["GENE_A", "GENE_B"],
                    "d2_effector": [2, 2],
                    "d13_effector": [4, 4],
                    "d17_effector": [1, 1],
                    "d2_other": [1, 1],
                }
            ),
        )
        projected = {
            "site_id": ["S1", "S2", "S3"],
            "gene_symbol": ["GENE_A", "GENE_B", "GENE_C"],
            "motif": ["M1", "M2", "M3"],
            "d2_effector": [4, 6, 8],
            "d13_effector": [4, 8, 12],
            "d2_other": [2, 4, 6],
        }
        if with_zero_post:
            projected["d13_effector"] = [1, 1, 1]
            projected["d13_other"] = [1, 1, 1]
        _write_csv(
            donor_dir / TRACK_FILES["st"],
            pd.DataFrame(projected),
        )
        cell_counts = {
            "state": ["effector", "effector", "other", "other"],
            "day": [2, 13, 2, 13],
            "n_cells": [6, 4, 8, 0],
        }
        if with_zero_post:
            cell_counts["n_cells"][1] = 0
            cell_counts["state"].append("other")
            cell_counts["day"].append(13)
            cell_counts["n_cells"].append(0)
        _write_csv(donor_dir / "scrna" / "cell_counts.csv", pd.DataFrame(cell_counts))

    def test_should_run_state_gates(self) -> None:
        self.assertEqual(
            should_run_state(
                n_cells_by_day={2: 6, 13: 4, 17: 0},
                n_motif_sites=3,
                baseline_day=2,
                days_available=[2, 13, 17],
            ),
            (True, None),
        )
        self.assertEqual(
            should_run_state(
                n_cells_by_day={2: 6, 13: 1},
                n_motif_sites=3,
                baseline_day=2,
                days_available=[3, 13],
            ),
            (False, "missing_baseline_day"),
        )
        self.assertEqual(
            should_run_state(
                n_cells_by_day={2: 6, 13: 0},
                n_motif_sites=3,
                baseline_day=2,
                days_available=[2],
            ),
            (False, "no_post_baseline_days"),
        )
        self.assertEqual(
            should_run_state(
                n_cells_by_day={2: 6, 13: 2},
                n_motif_sites=0,
                baseline_day=2,
                days_available=[2, 13],
            ),
            (False, "no_motif_sites"),
        )
        self.assertEqual(
            should_run_state(
                n_cells_by_day={2: 0, 13: 4},
                n_motif_sites=4,
                baseline_day=2,
                days_available=[2, 13],
            ),
            (False, "state_has_no_cells"),
        )
        self.assertEqual(
            should_run_state(
                n_cells_by_day={2: 6},
                n_motif_sites=4,
                baseline_day=2,
                days_available=[2, 13],
                count_days={2},
            ),
            (False, "no_cell_count_for_state_day"),
        )

    def test_summarize_state_qc_includes_skip_reasons_and_counts(self) -> None:
        with TemporaryDirectory() as td:
            root = Path(td)
            self._write_inputs(root)
            inputs = load_projected_inputs("donor1", "st", root=root)
            qc = summarize_state_qc(inputs)

            self.assertEqual(set(qc["state"]), {"effector", "other"})
            effector = qc.loc[qc["state"] == "effector"].iloc[0]
            self.assertEqual(int(effector["baseline_day"]), 2)
            self.assertEqual(effector["days_available"], [2, 13])
            self.assertEqual(effector["n_motif_sites"], 3)
            self.assertEqual(effector["n_sites"], 3)
            self.assertEqual(effector["n_cells_by_day"], {2: 6, 13: 4})
            self.assertIsNone(effector["skip_reason"])

            other = qc.loc[qc["state"] == "other"].iloc[0]
            self.assertEqual(other["skip_reason"], "no_post_baseline_days")

    def test_build_manifest_records_and_write_manifest(self) -> None:
        with TemporaryDirectory() as td:
            root = Path(td)
            self._write_inputs(root, with_zero_post=True)
            inputs = load_projected_inputs("donor1", "st", root=root)
            records = build_manifest_records(inputs, baseline_day=2)

            self.assertEqual(
                {(row["state"], row["skip_reason"]) for row in records},
                {("effector", "state_has_no_cells"), ("other", "state_has_no_cells")},
            )

            effector = next(row for row in records if row["state"] == "effector")
            self.assertEqual(effector["kind"], "projected_state")
            self.assertEqual(effector["days_available"], [2, 13])
            self.assertEqual(effector["days_run"], [])
            self.assertEqual(
                set(effector["input_files"]),
                {
                    str(inputs.projected_path),
                    str(inputs.protein_path),
                    str(inputs.cell_counts_path),
                },
            )
            self.assertEqual(effector["baseline_day"], 2)

            manifest_path = Path(root) / "out" / "manifest.json"
            write_manifest(records, manifest_path)
            self.assertTrue(manifest_path.exists())
            loaded = json.loads(manifest_path.read_text())
            loaded_by_state = {row["state"]: row for row in loaded}
            input_by_state = {row["state"]: row for row in records}
            self.assertEqual(set(loaded_by_state), set(input_by_state))
            for state in loaded_by_state:
                self.assertEqual(loaded_by_state[state]["state"], state)
                self.assertEqual(
                    {int(k): v for k, v in loaded_by_state[state]["n_cells_by_day"].items()},
                    input_by_state[state]["n_cells_by_day"],
                )


class RunProjectedStateMEATests(unittest.TestCase):
    def _write_inputs(self, root: Path) -> None:
        donor_dir = root / PROJECTED_ROOT / "donor1"
        _write_csv(
            donor_dir / PROTEIN_FILENAME,
            pd.DataFrame(
                {
                    "gene_symbol": ["GENE_A", "GENE_B"],
                    "d2_effector": [4, 2],
                    "d13_effector": [8, 8],
                }
            ),
        )
        _write_csv(
            donor_dir / TRACK_FILES["st"],
            pd.DataFrame(
                {
                    "site_id": ["S1", "S2"],
                    "gene_symbol": ["GENE_A", "GENE_B"],
                    "motif": ["M1", "M2"],
                    "d2_effector": [4, 6],
                    "d13_effector": [8, 4],
                }
            ),
        )
        _write_csv(
            donor_dir / "scrna" / "cell_counts.csv",
            pd.DataFrame(
                {
                    "state": ["effector", "effector"],
                    "day": [2, 13],
                    "n_cells": [10, 8],
                }
            ),
        )

    def test_run_projected_state_mea_writes_outputs_and_records_skips(self) -> None:
        with TemporaryDirectory() as td:
            root = Path(td)
            self._write_inputs(root)
            out_dir = root / "out"

            calls: list[dict[str, object]] = []

            def fake_mea_caller(
                *,
                motif_series,
                results_by_contrast,
                lfc_key,
                site_ids,
                gene_symbols,
                track,
            ):
                calls.append({
                    "lfc_key": lfc_key,
                    "results_by_contrast": results_by_contrast,
                })

                contrast = next(iter(results_by_contrast))

                mea_df = pd.DataFrame()
                if lfc_key == "stoich_lfc":
                    mea_df = pd.DataFrame(
                        {
                            "kinase": ["K1"],
                            "ES": [1.0],
                            "NES": [0.5],
                            "p-value": [0.02],
                            "FDR": [0.05],
                            "Subs fraction": ["1/1"],
                            "Leading substrates": ["M1"],
                            "residue_type": ["ST"],
                            "track": [track],
                            "contrast": [contrast],
                        }
                    )

                return (
                    mea_df,
                    pd.DataFrame({"contrast": [contrast], "median_shift": [0.1]}),
                    pd.DataFrame({"contrast": [contrast], "motif": ["M1"], "original_lfc": [0.1], "clipped_lfc": [0.05]}),
                    pd.DataFrame({"kinase": ["K1"], "contrast": [contrast], "motif": ["M1"], "residue_type": ["ST"], "track": [track], "kl_percentile": [1.0]}),
                )

            with patch(
                "alz.cohorts.tcells.state_mea.load_projected_inputs",
                return_value=load_projected_inputs("donor1", "st", root=root),
            ):
                result = run_projected_state_mea(
                    "donor1",
                    "st",
                    out_dir,
                    baseline_day=2,
                    mea_caller=fake_mea_caller,
                )

            records = result["manifest_records"]
            self.assertEqual(len(records), 1)
            self.assertIsNone(records[0]["skip_reason"])
            self.assertIn("skip_reasons", records[0])
            self.assertEqual(records[0]["skip_reasons"].get("raw"), ["empty_mea_result"])

            expected_files = [
                "mea_projected_state.csv",
                "mea_projected_state_raw.csv",
                "mea_global_shift_projected_state.csv",
                "mea_global_shift_projected_state_raw.csv",
                "winsorized_sites_projected_state.csv",
                "winsorized_sites_projected_state_raw.csv",
                "mea_substrate_sets_projected_state.csv",
                "mea_substrate_sets_projected_state_raw.csv",
                "kinase_state_timepoint_nes.csv",
                "kinase_state_timepoint_fdr.csv",
                "kinase_state_timepoint_nes_raw.csv",
                "kinase_state_timepoint_fdr_raw.csv",
                "recurrence_projected_state_deferred.txt",
                "projected_state_mea_manifest.json",
            ]
            for name in expected_files:
                self.assertTrue((out_dir / name).exists(), msg=name)

            self.assertFalse((out_dir / "mechanism_attribution_projected_state.csv").exists())

            mea = pd.read_csv(out_dir / "mea_projected_state.csv")
            self.assertTrue({"donor", "state", "track", "kind", "timepoint"}.issubset(set(mea.columns)))
            self.assertEqual(int(mea.loc[0, "timepoint"] == "d13"), 1)

            raw_mea = pd.read_csv(out_dir / "mea_projected_state_raw.csv")
            self.assertTrue({"donor", "state", "track", "kind", "timepoint"}.issubset(set(raw_mea.columns)))
            self.assertEqual(len(calls), 2)

            nes = pd.read_csv(out_dir / "kinase_state_timepoint_nes.csv")
            self.assertEqual(list(nes.columns), ["kinase", "effector|d13"])
            self.assertEqual(float(nes.loc[0, "effector|d13"]), 0.5)

            fdr = pd.read_csv(out_dir / "kinase_state_timepoint_fdr.csv")
            self.assertEqual(list(fdr.columns), ["kinase", "effector|d13"])
            self.assertEqual(float(fdr.loc[0, "effector|d13"]), 0.05)

            note = (out_dir / "recurrence_projected_state_deferred.txt").read_text()
            self.assertIn("recurrence over (state, timepoint) is deferred", note)

    def test_write_projected_state_mechanism_attribution_success(self) -> None:
        with TemporaryDirectory() as td:
            out_dir = Path(td) / "out"
            out_dir.mkdir(parents=True)
            _write_csv(
                out_dir / "mea_projected_state.csv",
                pd.DataFrame(
                    {
                        "cohort": ["tcells", "tcells"],
                        "donor": ["donor1", "donor1"],
                        "track": ["st", "st"],
                        "state": ["effector", "effector"],
                        "timepoint": ["d13", "d20"],
                        "kinase": ["K1", "K1"],
                        "NES": [1.0, 0.4],
                        "FDR": [0.01, 0.02],
                    }
                ),
            )
            _write_csv(
                out_dir / "mea_projected_state_raw.csv",
                pd.DataFrame(
                    {
                        "cohort": ["tcells", "tcells"],
                        "donor": ["donor1", "donor1"],
                        "track": ["st", "st"],
                        "state": ["effector", "effector"],
                        "timepoint": ["d13", "d20"],
                        "kinase": ["K1", "K1"],
                        "NES": [0.8, 0.7],
                        "FDR": [0.02, 0.40],
                    }
                ),
            )

            attributed = write_projected_state_mechanism_attribution(out_dir)
            self.assertIsNotNone(attributed)
            self.assertTrue((out_dir / "mechanism_attribution_projected_state.csv").exists())

            self.assertEqual(len(attributed), 2)
            one_way = attributed.set_index("timepoint")
            row_d13 = one_way.loc["d13"]
            self.assertEqual(row_d13["mechanism_call"], "both")
            self.assertEqual(row_d13["sign_relation"], "same")
            row_d20 = one_way.loc["d20"]
            self.assertEqual(row_d20["mechanism_call"], "activity_driven")
            self.assertEqual(row_d20["sign_relation"], "stoich_only")

    def test_write_projected_state_mechanism_attribution_skips_without_paired_files(self) -> None:
        with TemporaryDirectory() as td:
            root = Path(td)
            stoich = pd.DataFrame(
                {
                    "cohort": ["tcells"],
                    "donor": ["donor1"],
                    "track": ["st"],
                    "state": ["effector"],
                    "timepoint": ["d13"],
                    "kinase": ["K1"],
                    "NES": [1.0],
                    "FDR": [0.01],
                }
            )
            raw_columns = [
                "cohort",
                "donor",
                "track",
                "state",
                "timepoint",
                "kinase",
                "NES",
                "FDR",
            ]
            raw_empty = pd.DataFrame(columns=raw_columns)

            missing_raw_dir = root / "missing_raw"
            missing_raw_dir.mkdir(parents=True)
            _write_csv(missing_raw_dir / "mea_projected_state.csv", stoich)
            self.assertIsNone(
                write_projected_state_mechanism_attribution(missing_raw_dir),
            )
            self.assertFalse(
                (missing_raw_dir / "mechanism_attribution_projected_state.csv").exists()
            )

            missing_stoich_dir = root / "missing_stoich"
            missing_stoich_dir.mkdir(parents=True)
            _write_csv(missing_stoich_dir / "mea_projected_state_raw.csv", stoich)
            self.assertIsNone(
                write_projected_state_mechanism_attribution(missing_stoich_dir),
            )
            self.assertFalse(
                (missing_stoich_dir / "mechanism_attribution_projected_state.csv").exists()
            )

            empty_raw_dir = root / "empty_raw"
            empty_raw_dir.mkdir(parents=True)
            _write_csv(empty_raw_dir / "mea_projected_state.csv", stoich)
            _write_csv(empty_raw_dir / "mea_projected_state_raw.csv", raw_empty)
            self.assertIsNone(
                write_projected_state_mechanism_attribution(empty_raw_dir),
            )
            self.assertFalse(
                (empty_raw_dir / "mechanism_attribution_projected_state.csv").exists()
            )

    def test_run_projected_state_mea_contrast_math(self) -> None:
        with TemporaryDirectory() as td:
            root = Path(td)
            self._write_inputs(root)
            out_dir = root / "out"

            recorded: dict[str, dict[str, dict[str, np.ndarray]]] = {}

            def fake_mea_caller(
                *,
                motif_series,
                results_by_contrast,
                lfc_key,
                site_ids,
                gene_symbols,
                track,
            ):
                recorded[lfc_key] = results_by_contrast
                contrast = next(iter(results_by_contrast))
                return (
                    pd.DataFrame(
                        {
                            "kinase": ["K1"],
                            "ES": [1.0],
                            "NES": [0.5],
                            "p-value": [0.02],
                            "FDR": [0.05],
                            "Subs fraction": ["1/1"],
                            "Leading substrates": ["M1"],
                            "residue_type": ["ST"],
                            "track": [track],
                            "contrast": [contrast],
                        }
                    ),
                    pd.DataFrame({"contrast": [contrast]}),
                    pd.DataFrame({"contrast": [contrast]}),
                    pd.DataFrame({"kinase": ["K1"], "contrast": [contrast], "motif": ["M1"], "residue_type": ["ST"], "track": [track], "kl_percentile": [1.0]}),
                )

            with patch(
                "alz.cohorts.tcells.state_mea.load_projected_inputs",
                return_value=load_projected_inputs("donor1", "st", root=root),
            ):
                run_projected_state_mea(
                    "donor1",
                    "st",
                    out_dir,
                    baseline_day=2,
                    mea_caller=fake_mea_caller,
                )

            stoich = recorded["stoich_lfc"]["d13_vs_d2"]["stoich_lfc"]
            raw = recorded["raw_lfc"]["d13_vs_d2"]["raw_lfc"]

            np.testing.assert_allclose(stoich, [0.0, -2.584962500721156], rtol=1e-9)
            np.testing.assert_allclose(raw, [1.0, -0.584962500721156], rtol=1e-9)


class AggregateHelpersTests(unittest.TestCase):
    def test_build_state_timepoint_matrices_and_writer(self) -> None:
        with TemporaryDirectory() as td:
            out_dir = Path(td) / "agg"
            long = pd.DataFrame(
                {
                    "kinase": ["K1", "K2", "K1"],
                    "NES": [1.0, 0.2, 1.1],
                    "FDR": [0.05, 0.1, 0.2],
                    "state": ["effector", "effector", "memory"],
                    "timepoint": ["d20", "d13", "d13"],
                    "donor": ["donor1", "donor1", "donor1"],
                }
            )

            nes, fdr = build_state_timepoint_matrices(long)
            self.assertEqual(
                list(nes.columns),
                ["effector|d13", "effector|d20", "memory|d13"],
            )
            self.assertEqual(
                list(fdr.columns),
                ["effector|d13", "effector|d20", "memory|d13"],
            )
            self.assertAlmostEqual(float(nes.loc["K1", "effector|d20"]), 1.0)
            self.assertAlmostEqual(float(fdr.loc["K1", "memory|d13"]), 0.2)

            write_state_timepoint_aggregates(long, out_dir, infix="")
            write_state_timepoint_aggregates(long, out_dir, infix="_raw")

            written_nes = pd.read_csv(out_dir / "kinase_state_timepoint_nes.csv", index_col=0)
            written_fdr = pd.read_csv(out_dir / "kinase_state_timepoint_fdr_raw.csv", index_col=0)
            self.assertEqual(list(written_nes.columns), ["effector|d13", "effector|d20", "memory|d13"])
            self.assertEqual(list(written_fdr.columns), ["effector|d13", "effector|d20", "memory|d13"])

class RunProjectedStateMEACliTests(unittest.TestCase):
    def _write_inputs_for_track(
        self,
        root: Path,
        *,
        donor: str,
        track: str,
        include_motifs: bool = True,
        baseline_day: int = 2,
        post_day: int = 13,
    ) -> None:
        donor_dir = root / PROJECTED_ROOT / donor
        if include_motifs:
            motifs = ["M1", "M2"]
        else:
            motifs = [float("nan"), float("nan")]

        _write_csv(
            donor_dir / PROTEIN_FILENAME,
            pd.DataFrame(
                {
                    "gene_symbol": ["GENE_A", "GENE_B"],
                    f"d{baseline_day}_effector": [2, 2],
                    f"d{post_day}_effector": [4, 4],
                }
            ),
        )
        _write_csv(
            donor_dir / TRACK_FILES[track],
            pd.DataFrame(
                {
                    "site_id": ["S1", "S2"],
                    "gene_symbol": ["GENE_A", "GENE_B"],
                    "motif": motifs,
                    f"d{baseline_day}_effector": [4, 6],
                    f"d{post_day}_effector": [8, 4],
                }
            ),
        )
        _write_csv(
            donor_dir / "scrna" / "cell_counts.csv",
            pd.DataFrame(
                {
                    "state": ["effector", "effector"],
                    "day": [baseline_day, post_day],
                    "n_cells": [10, 8],
                }
            ),
        )

    def test_run_dry_run_both_mode_records_missing_input_and_no_motif_sites(self) -> None:
        with TemporaryDirectory() as td:
            root = Path(td)
            self._write_inputs_for_track(root, donor="donor1", track="st")
            self._write_inputs_for_track(root, donor="donor1", track="py")
            self._write_inputs_for_track(root, donor="donor2", track="py", include_motifs=False)

            manifest_out = root / "out" / "manifest.json"
            with patch.object(state_mea, "_PROJECT_ROOT", root):
                state_mea.main(
                    [
                        "--donor",
                        "both",
                        "--track",
                        "both",
                        "--dry-run",
                        "--manifest-out",
                        str(manifest_out),
                    ]
                )

            records = json.loads(manifest_out.read_text())
            self.assertEqual(
                {(r["donor"], r["track"], r["skip_reason"]) for r in records},
                {
                    ("donor1", "st", None),
                    ("donor1", "py", None),
                    ("donor2", "st", "missing_projected_phospho_file"),
                    ("donor2", "py", "no_motif_sites"),
                },
            )

    def test_build_missing_input_manifest_records_defaults(self) -> None:
        records = build_missing_input_manifest_records(
            "donor2",
            "st",
            reason="missing_projected_phospho_file",
            input_files=["a", "b"],
        )
        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual(record["donor"], "donor2")
        self.assertEqual(record["track"], "st")
        self.assertEqual(record["skip_reason"], "missing_projected_phospho_file")
        self.assertTrue(record["kind"] == "projected_state")
        self.assertIn("input_files", record)

    def test_runner_scratch_dir_both_mode_writes_combo_manifests(self) -> None:
        with TemporaryDirectory() as td:
            root = Path(td)
            self._write_inputs_for_track(root, donor="donor1", track="st")
            self._write_inputs_for_track(root, donor="donor1", track="py")
            self._write_inputs_for_track(root, donor="donor2", track="py")

            out_dir = root / "scratch"

            def fake_run_projected_state_mea(donor, track, out_dir, **kwargs):
                baseline_day = kwargs.get("baseline_day", 2)
                inputs = load_projected_inputs(donor, track, root=root)
                manifest_records = build_manifest_records(inputs, baseline_day=baseline_day)
                state_mea.write_manifest(manifest_records, out_dir / "projected_state_mea_manifest.json")
                return {
                    "manifest_records": manifest_records,
                    "manifest_path": out_dir / "projected_state_mea_manifest.json",
                    "out_dir": out_dir,
                }

            with patch.object(state_mea, "_PROJECT_ROOT", root), patch(
                "alz.cohorts.tcells.state_mea.run_projected_state_mea",
                side_effect=fake_run_projected_state_mea,
            ):
                state_mea.main(
                    [
                        "--donor",
                        "both",
                        "--track",
                        "both",
                        "--runner-scratch-dir",
                        str(out_dir),
                    ]
                )

            expected_dirs = [
                out_dir / "donor1" / "st",
                out_dir / "donor1" / "py",
                out_dir / "donor2" / "st",
                out_dir / "donor2" / "py",
            ]
            for path in expected_dirs:
                self.assertTrue(path.exists())
                manifest = path / "projected_state_mea_manifest.json"
                self.assertTrue(manifest.exists(), msg=str(manifest))

            donor2_st = out_dir / "donor2" / "st" / "projected_state_mea_manifest.json"
            loaded_st = json.loads(donor2_st.read_text())
            self.assertEqual(loaded_st[0]["skip_reason"], "missing_projected_phospho_file")


if __name__ == "__main__":
    unittest.main()

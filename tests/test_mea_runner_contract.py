from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import sys
import unittest

import numpy as np
import pandas as pd

_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from alz.core.mea_runner import MeaRunner, MeaUnit, RunResult, SkipRecord


class _Enrich:
    def __init__(self) -> None:
        self.calls = 0

    def _run_mea(self, **kwargs):
        self.calls += 1
        return (
            pd.DataFrame({"kinase": ["K1"], "NES": [1.0]}),
            pd.DataFrame({"contrast": ["c1"]}),
            pd.DataFrame({"site_id": ["S1"]}),
            pd.DataFrame({"kinase": ["K1"], "motif": ["AAAApSAAAA"]}),
        )


def _unit(tmp_path: Path) -> MeaUnit:
    return MeaUnit(
        track="st",
        kind="stoich",
        lfc_key="lfc",
        out_dir=tmp_path,
        infix="",
        suffix="",
        motif_series=pd.Series(["AAAApSAAAA"]),
        site_ids=np.array(["S1"]),
        gene_symbols=np.array(["GENE1"]),
        results_by_contrast={"c1": {"lfc": np.array([1.0])}},
    )


class _Adapter:
    def __init__(
        self,
        unit: MeaUnit,
        *,
        load_returns_none: bool = False,
        adapter_skip: bool = False,
    ) -> None:
        self.unit = unit
        self.load_returns_none = load_returns_none
        self.adapter_skip = adapter_skip
        self.build_calls = 0
        self.skip_calls = 0
        self.provenance: list[SkipRecord] | None = None

    def iter_units(self):
        yield self.unit

    def load_inputs(self, unit: MeaUnit) -> MeaUnit | None:
        if self.load_returns_none:
            return None
        return unit

    def build_contrasts(self, unit: MeaUnit) -> MeaUnit:
        self.build_calls += 1
        return unit

    def skip_check(self, unit: MeaUnit) -> tuple[bool, str | None]:
        self.skip_calls += 1
        if self.adapter_skip:
            return True, "adapter requested skip"
        return False, None

    def write_aggregates(self, result: RunResult) -> None:
        pass

    def write_provenance(self, skips: list[SkipRecord]) -> None:
        self.provenance = skips


class MeaRunnerContractTests(unittest.TestCase):
    def test_run_all_load_inputs_none_skips_without_building_contrasts(self) -> None:
        with TemporaryDirectory() as tmp:
            enrich = _Enrich()
            runner = MeaRunner(enrich)
            adapter = _Adapter(_unit(Path(tmp)), load_returns_none=True)

            results = runner.run_all(adapter)

        self.assertEqual(results, [])
        self.assertEqual(adapter.build_calls, 0)
        self.assertEqual(adapter.skip_calls, 0)
        self.assertEqual(enrich.calls, 0)
        self.assertEqual(
            [(s.unit_label, s.reason) for s in runner.skips],
            [("st/stoich", "load_inputs returned None")],
        )
        self.assertEqual(adapter.provenance, runner.skips)

    def test_run_unit_skip_check_fn_precedes_adapter_and_mea(self) -> None:
        with TemporaryDirectory() as tmp:
            enrich = _Enrich()
            runner = MeaRunner(enrich)
            adapter = _Adapter(_unit(Path(tmp)))

            result = runner.run_unit(
                adapter.unit,
                adapter,
                skip_check_fn=lambda unit: (True, "caller requested skip"),
            )

        self.assertIsNone(result)
        self.assertEqual(adapter.skip_calls, 0)
        self.assertEqual(enrich.calls, 0)
        self.assertEqual(
            [(s.unit_label, s.reason) for s in runner.skips],
            [("st/stoich", "caller requested skip")],
        )

    def test_run_unit_adapter_skip_still_skips_without_mea(self) -> None:
        with TemporaryDirectory() as tmp:
            enrich = _Enrich()
            runner = MeaRunner(enrich)
            adapter = _Adapter(_unit(Path(tmp)), adapter_skip=True)

            result = runner.run_unit(adapter.unit, adapter)

        self.assertIsNone(result)
        self.assertEqual(adapter.skip_calls, 1)
        self.assertEqual(enrich.calls, 0)
        self.assertEqual(
            [(s.unit_label, s.reason) for s in runner.skips],
            [("st/stoich", "adapter requested skip")],
        )


if __name__ == "__main__":
    unittest.main()

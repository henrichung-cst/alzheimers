"""Mukesh (human NBB) MEA adapter for the Phase-3 shared runner.

Wraps the existing pure functions in `alz/cohorts/mukesh/mea.py` and the
Phase-2 aggregate writers in `alz/core/mea_outputs.py`.  No new logic is
introduced here; the adapter is a thin connector.

Yields 4 MeaUnit objects:
  (st, stoich), (st, raw), (py, stoich), (py, raw)

For each unit:
- load_inputs   → calls _load_track_matrix (raises on missing stoich; returns
                  unit with motif_series=None on missing raw, triggering skip).
- build_contrasts → calls _split_samples + _build_donor_deltas.
- skip_check    → inline matrix-absent guard only (raw track may be absent).
- write_aggregates → calls _write_donor_aggregates (shared Phase-2 helper).
- write_provenance → writes a structured JSON skip log to scratch_dir.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from alz.shared import config
from alz.core.mea_runner import MeaAdapter, MeaUnit, RunResult, SkipRecord
from alz.core.mea_outputs import KIND_SPEC as _SHARED_KIND_SPEC
from alz.cohorts.mukesh.mea import (
    _load_track_matrix,
    _split_samples,
    _build_donor_deltas,
    _write_donor_aggregates,
    _KIND_SPEC as _LOCAL_KIND_SPEC,
    PERDONOR_DIR,
)
from alz.cohorts.mukesh.ingest import SAMPLE_MAPPING_CSV


class MukeshMeaAdapter:
    """Adapter wiring the Mukesh cohort to MeaRunner.

    Parameters
    ----------
    scratch_dir : str | Path
        All runner-driven outputs land here (never in PERDONOR_DIR).
    mapping : pd.DataFrame
        Loaded sample_mapping.csv (caller responsible).
    tracks : list[str]
        Subset of ``["st", "py"]`` to run.
    """

    def __init__(
        self,
        scratch_dir: str | Path,
        mapping: pd.DataFrame,
        tracks: list[str] | None = None,
    ) -> None:
        self.scratch_dir = Path(scratch_dir)
        self.mapping = mapping
        self.tracks = tracks or ["st", "py"]
        # Cached AD/CTRL split (shared across all units; mapping is invariant).
        _ad, _ctrl = _split_samples(mapping)
        self._ad_ids_full = _ad
        self._ctrl_ids_full = _ctrl

    # ------------------------------------------------------------------
    # MeaAdapter protocol
    # ------------------------------------------------------------------

    def iter_units(self) -> Iterable[MeaUnit]:
        """Yield one MeaUnit per (track, kind) combination."""
        for track in self.tracks:
            suffix = config.PHOSPHO_TRACKS[track]["output_suffix"]
            for kind in ("stoich", "raw"):
                spec = _LOCAL_KIND_SPEC[kind]
                out_dir = self.scratch_dir
                yield MeaUnit(
                    track=track,
                    kind=kind,
                    lfc_key=spec["lfc_key"],
                    out_dir=out_dir,
                    infix=spec["infix"],
                    suffix=suffix,
                    # motif_series / site_ids / gene_symbols / results_by_contrast
                    # populated by load_inputs / build_contrasts below.
                )

    def load_inputs(self, unit: MeaUnit) -> MeaUnit | None:
        """Populate motif_series, site_ids, gene_symbols from the track matrix.

        Returns unit with motif_series=None if the input file is absent (raw
        track only; stoich raises FileNotFoundError on missing).
        """
        spec = _LOCAL_KIND_SPEC[unit.kind]
        try:
            matrix = _load_track_matrix(unit.track, spec["matrix_kind"])
        except FileNotFoundError:
            # stoich track missing — propagate; this is a configuration error.
            raise
        if matrix is None:
            # raw track absent — signal skip via None motif_series.
            unit.motif_series = None
            unit.meta["matrix"] = None
            return unit

        # Filter to samples present in the matrix (mirroring canonical path).
        ad_ids = [s for s in self._ad_ids_full if s in matrix.columns]
        ctrl_ids = [s for s in self._ctrl_ids_full if s in matrix.columns]

        unit.motif_series = matrix["motif"].reset_index(drop=True)
        unit.site_ids = matrix["site_id"].values
        unit.gene_symbols = matrix["gene_symbol"].values
        unit.meta["matrix"] = matrix
        unit.meta["ad_ids"] = ad_ids
        unit.meta["ctrl_ids"] = ctrl_ids
        print(
            f"  [{unit.label}] AD donors: {len(ad_ids)}  "
            f"CTRL: {len(ctrl_ids)}  sites: {len(matrix)}"
        )
        return unit

    def build_contrasts(self, unit: MeaUnit) -> MeaUnit:
        """Populate results_by_contrast using _build_donor_deltas."""
        if unit.motif_series is None:
            # Already marked for skip; no-op.
            unit.results_by_contrast = {}
            return unit
        matrix = unit.meta["matrix"]
        ad_ids = unit.meta["ad_ids"]
        ctrl_ids = unit.meta["ctrl_ids"]
        unit.results_by_contrast = _build_donor_deltas(
            matrix, ad_ids, ctrl_ids, unit.lfc_key
        )
        return unit

    def skip_check(self, unit: MeaUnit) -> tuple[bool, str | None]:
        """Skip if raw track input was absent."""
        if unit.motif_series is None:
            return True, f"[{unit.label}] input matrix missing (raw track absent)"
        return False, None

    def write_aggregates(self, result: RunResult) -> None:
        """Write wide NES/FDR + AD/CTRL recurrence tables to scratch."""
        unit = result.unit
        ad_ids = unit.meta["ad_ids"]
        ctrl_ids = unit.meta["ctrl_ids"]
        out_dir = str(unit.out_dir)
        _write_donor_aggregates(
            result.mea_df,
            ad_ids,
            ctrl_ids,
            unit.infix,
            unit.suffix,
            out_dir,
        )

    def write_provenance(self, skips: list[SkipRecord]) -> None:
        """Write a JSON skip log to scratch_dir."""
        if not skips:
            return
        prov_path = self.scratch_dir / "skip_log.json"
        prov_path.parent.mkdir(parents=True, exist_ok=True)
        records = [
            {"unit": s.unit_label, "reason": s.reason, "detail": s.detail}
            for s in skips
        ]
        with open(prov_path, "w") as fh:
            json.dump(records, fh, indent=2)
        print(f"  skip log: {prov_path}  ({len(records)} skip(s))")

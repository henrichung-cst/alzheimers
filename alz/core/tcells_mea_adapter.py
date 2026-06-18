"""T-cell exhaustion MEA adapter for the Phase-3 shared runner.

Wraps the existing pure functions in `alz/ingest/tcells_perdonor.py` and the
Phase-2 aggregate writers in `alz/core/mea_outputs.py`.  No new logic is
introduced here; the adapter is a thin connector.

Yields up to 8 MeaUnit objects across two donors:
  donor1: (st, stoich), (st, raw), (py, stoich), (py, raw)  — all expected to run
  donor2: (st, stoich), (st, raw), (py, stoich), (py, raw)  — all skip by design:
            st/{stoich,raw} -> matrix_absent (no IMAC)
            py/{stoich,raw} -> no_motif (ForPerseus export has no flanking region)

For each unit:
- load_inputs    → reads the track CSV (records matrix_absent skip via motif_series=None).
- build_contrasts → calls _baseline_and_days + _build_timepoint_deltas.
- skip_check     → matrix_absent guard (matrix is None) + n_motif == 0 guard.
- write_aggregates → calls _write_timepoint_aggregates (Phase-2 helper in tcells_perdonor).
- write_provenance → writes mea_manifest.json in T-cell's exact schema, per donor.

Skip records reproduce the EXACT field set that `_run_track_kind` appends:
  matrix_absent: {donor, track, kind, reason, path}
  no_motif:      {donor, track, kind, reason, n_sites}
  mea_empty:     {donor, track, kind, reason, n_sites}
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
from alz.ingest.tcells_perdonor import (
    _KIND_SPEC,
    _baseline_and_days,
    _build_timepoint_deltas,
    _n_motif,
    _matrix_path,
    _write_timepoint_aggregates,
    TRACKS,
)


class TcellsMeaAdapter:
    """Adapter wiring the T-cell exhaustion cohort to MeaRunner.

    Parameters
    ----------
    scratch_dir : str | Path
        Root for all runner-driven outputs.  Per-donor MEA output lands under
        ``<scratch_dir>/<donor>/mea/``.  NEVER writes to the canonical
        ``KINASE_DIR`` tree.
    donors : list[str]
        Donors to iterate over.  Default: ``["donor1", "donor2"]``.
    tracks : list[str]
        Subset of ``["st", "py"]`` to run.  Default: both.
    """

    def __init__(
        self,
        scratch_dir: str | Path,
        donors: list[str] | None = None,
        tracks: list[str] | None = None,
    ) -> None:
        self.scratch_dir = Path(scratch_dir)
        self.donors = donors or ["donor1", "donor2"]
        self.tracks = tracks or list(TRACKS)
        # Accumulate skip details (rich dicts, not bare SkipRecord) per donor so
        # write_provenance can emit the exact canonical manifest schema.
        # Keyed by donor name.
        self._rich_skips: dict[str, list[dict]] = {d: [] for d in self.donors}
        self._ran: dict[str, list[str]] = {d: [] for d in self.donors}

    # ------------------------------------------------------------------
    # MeaAdapter protocol
    # ------------------------------------------------------------------

    def iter_units(self) -> Iterable[MeaUnit]:
        """Yield one MeaUnit per (donor, track, kind) combination."""
        for donor in self.donors:
            for track in self.tracks:
                suffix = config.PHOSPHO_TRACKS[track]["output_suffix"]
                for kind in ("stoich", "raw"):
                    spec = _KIND_SPEC[kind]
                    out_dir = self.scratch_dir / donor / "mea"
                    yield MeaUnit(
                        track=track,
                        kind=kind,
                        lfc_key=spec["lfc_key"],
                        out_dir=out_dir,
                        infix=spec["infix"],
                        suffix=suffix,
                        long_table_stem="mea_timecourse",
                        meta={"donor": donor},
                    )

    def load_inputs(self, unit: MeaUnit) -> MeaUnit:
        """Read the per-donor track matrix and populate motif/site/gene arrays.

        Returns unit with motif_series=None if the CSV is absent (matrix_absent
        skip).  Records the rich skip dict so write_provenance can emit the
        exact canonical manifest schema.
        """
        donor = unit.meta["donor"]
        track = unit.track
        kind = unit.kind
        path = _matrix_path(donor, kind, track)

        if not os.path.exists(path):
            # Record in canonical schema (mirrors _run_track_kind's append).
            self._rich_skips[donor].append({
                "donor": donor,
                "track": track,
                "kind": kind,
                "reason": "matrix_absent",
                "path": os.path.relpath(path, config.REPO_ROOT),
            })
            unit.motif_series = None
            unit.meta["matrix"] = None
            return unit

        matrix = pd.read_csv(path)
        unit.meta["matrix"] = matrix
        unit.meta["n_sites"] = int(len(matrix))
        unit.motif_series = matrix["motif"].reset_index(drop=True)
        unit.site_ids = matrix["site_id"].values
        unit.gene_symbols = matrix["gene_symbol"].values

        n_motif = _n_motif(matrix)
        unit.meta["n_motif"] = n_motif
        tag = f"{donor}/{track}/{kind}"
        print(
            f"  [{tag}] loaded {len(matrix)} sites  motifs={n_motif}  "
            f"path={os.path.relpath(path, config.REPO_ROOT)}"
        )
        return unit

    def build_contrasts(self, unit: MeaUnit) -> MeaUnit:
        """Populate results_by_contrast using _baseline_and_days + _build_timepoint_deltas."""
        donor = unit.meta["donor"]
        matrix = unit.meta.get("matrix")

        if matrix is None:
            # matrix_absent — already recorded; nothing to do.
            unit.results_by_contrast = {}
            return unit

        n_motif = unit.meta.get("n_motif", 0)
        if n_motif == 0:
            # no_motif — skip_check will catch it; still record rich skip here
            # so write_provenance has it in time.
            self._rich_skips[donor].append({
                "donor": donor,
                "track": unit.track,
                "kind": unit.kind,
                "reason": "no_motif",
                "n_sites": unit.meta["n_sites"],
            })
            unit.results_by_contrast = {}
            return unit

        baseline, days = _baseline_and_days(donor, matrix)
        tag = f"{donor}/{unit.track}/{unit.kind}"
        print(f"\n=== Runner time-course MEA: {tag} ===")
        print(
            f"  baseline: {baseline}  timepoints: {len(days)} ({', '.join(days)})"
        )
        unit.results_by_contrast = _build_timepoint_deltas(
            matrix, baseline, days, unit.lfc_key
        )
        return unit

    def skip_check(self, unit: MeaUnit) -> tuple[bool, str | None]:
        """Skip if matrix absent or zero motifs."""
        donor = unit.meta["donor"]
        matrix = unit.meta.get("matrix")

        if matrix is None:
            return True, f"[{donor}/{unit.track}/{unit.kind}] matrix_absent"

        n_motif = unit.meta.get("n_motif", 0)
        if n_motif == 0:
            return True, f"[{donor}/{unit.track}/{unit.kind}] no_motif"

        return False, None

    def write_aggregates(self, result: RunResult) -> None:
        """Write wide NES/FDR + recurrence tables, then record ran/skipped state."""
        unit = result.unit
        donor = unit.meta["donor"]
        mea_df = result.mea_df

        if mea_df.empty:
            # Post-call mea_empty: record rich skip (mirrors _run_track_kind).
            self._rich_skips[donor].append({
                "donor": donor,
                "track": unit.track,
                "kind": unit.kind,
                "reason": "mea_empty",
                "n_sites": unit.meta.get("n_sites", 0),
            })
            return

        out_dir = str(unit.out_dir)
        _write_timepoint_aggregates(mea_df, unit.infix, unit.suffix, out_dir)
        self._ran[donor].append(f"{unit.track}/{unit.kind}")

    def write_provenance(self, skips: list[SkipRecord]) -> None:
        """Write per-donor mea_manifest.json in T-cell's exact canonical schema.

        Schema: {donor, mea_ran, mea_skipped, mea_fdr_thresh, mea_min_sites}
        where mea_skipped entries are the rich dicts recorded during load_inputs /
        build_contrasts / write_aggregates (matching _run_track_kind's exact fields).
        """
        for donor in self.donors:
            out_dir = self.scratch_dir / donor / "mea"
            out_dir.mkdir(parents=True, exist_ok=True)
            manifest = {
                "donor": donor,
                "mea_ran": self._ran[donor],
                "mea_skipped": self._rich_skips[donor],
                "mea_fdr_thresh": config.MEA_FDR_THRESH,
                "mea_min_sites": config.MEA_MIN_SITES,
            }
            manifest_path = out_dir / "mea_manifest.json"
            with open(manifest_path, "w") as fh:
                json.dump(manifest, fh, indent=2)
            print(
                f"  [{donor}] manifest: {manifest_path}  "
                f"ran={self._ran[donor]}  skipped={len(self._rich_skips[donor])}"
            )

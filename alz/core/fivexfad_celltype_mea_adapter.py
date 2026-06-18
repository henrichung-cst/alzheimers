"""5xFAD per-cell-type MEA adapter for the Phase-3 shared runner (Wave 3D).

Routes each per-(tissue, track, cell_type) MEA call through
``fivexfad_celltype_mea.run`` with an injectable MEA caller that wraps
``MeaRunner._call_mea_unit``.  ``run()`` owns all OLS, weighting, contrast
construction, accumulation, and output writes — this module only:

  1. Builds the runner instance and a ``_call_mea_unit`` wrapper that satisfies
     ``_fit_one_celltype``'s ``mea_caller`` signature.
  2. Calls ``run(out_dir=scratch_dir, mea_caller=...)`` with optional filters.

The canonical ``kinase_attribution_5xfad/celltype_mea/`` directory is NEVER
touched.

Outputs are written flat to ``scratch_dir/`` by ``run()``, using the same
file names it would write to the canonical OUT_DIR.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from alz.bulk_mea import enrich as kinase_enrich
from alz.core.mea_runner import MeaRunner, MeaUnit, RunResult, SkipRecord


def run_via_runner(
    scratch_dir: str | Path,
    tissue_filter: set[str] | None = None,
    track_filter: set[str] | None = None,
    celltype_filter: set[str] | None = None,
    max_celltypes: int | None = None,
) -> None:
    """Run 5xFAD per-cell-type MEA through the shared runner, writing all output to scratch_dir.

    Calls ``fivexfad_celltype_mea.run`` with a ``mea_caller`` that wraps
    ``MeaRunner._call_mea_unit`` so the runner records SkipRecords without
    duplicating OLS, weighting, accumulation, or audit logic.

    Parameters
    ----------
    scratch_dir : str | Path
        Destination for all outputs.  Created if absent.
        The canonical ``kinase_attribution_5xfad/celltype_mea/`` directory is
        never touched.
    tissue_filter : set[str] | None
        Subset of TISSUES to run.  ``None`` runs all.
    track_filter : set[str] | None
        Subset of TRACKS to run.  ``None`` runs all.
    celltype_filter : set[str] | None
        Subset of cell-type names to run.  ``None`` runs all.
    max_celltypes : int | None
        Cap on the number of cell types processed per (tissue, track).
    """
    from alz.cohorts.fivexfad.celltype_mea import run

    scratch_dir = Path(scratch_dir)
    scratch_dir.mkdir(parents=True, exist_ok=True)
    runner = MeaRunner(kinase_enrich)

    class _PassAdapter:
        """Adapter stub whose skip_check always passes.

        Input existence is guarded upstream in ``_fit_one_celltype`` (skips
        units with < 8 matched samples or no estimable contrasts); by the time
        ``mea_caller`` is invoked the inputs are confirmed valid.
        """

        def skip_check(self, unit: MeaUnit) -> tuple[bool, str | None]:
            return False, None

        def write_aggregates(self, result: RunResult) -> None:
            pass

        def write_provenance(self, skips: list[SkipRecord]) -> None:
            pass

    _adapter = _PassAdapter()

    def _make_mea_caller():
        """Return a mea_caller closure routing one _run_mea call through the runner.

        The returned function has the same keyword signature as
        ``kinase_enrich._run_mea`` (motif_series, results_by_contrast, lfc_key;
        keyword site_ids, gene_symbols, track) and returns the same 4-tuple
        ``(mea_df, shift_df, wins_df, substrate_df)``.

        ``lfc_key`` is always ``"lfc"`` for the celltype MEA, and ``track``
        is the kl_track (``"st"`` or ``"py"``) passed through from
        ``_fit_one_celltype`` — both are forwarded as-is from the call site,
        not hardcoded here.
        """
        def _caller(
            motif_series,
            results_by_contrast,
            lfc_key,
            *,
            site_ids=None,
            gene_symbols=None,
            track=None,
        ):
            unit = MeaUnit(
                track=track,
                kind="celltype",
                lfc_key=lfc_key,
                out_dir=scratch_dir,
                infix="",
                suffix="",
                motif_series=motif_series,
                site_ids=site_ids,
                gene_symbols=gene_symbols,
                results_by_contrast=results_by_contrast,
            )
            result = runner._call_mea_unit(unit, _adapter)
            if result is None:
                empty = pd.DataFrame()
                return empty, empty, empty, empty
            return (
                result.mea_df,
                result.shift_df,
                result.wins_df,
                result.substrate_df,
            )

        return _caller

    run(
        out_dir=scratch_dir,
        mea_caller=_make_mea_caller(),
        tissue_filter=tissue_filter,
        track_filter=track_filter,
        celltype_filter=celltype_filter,
        max_celltypes=max_celltypes,
    )

    if runner._skips:
        print(f"\n5xFAD celltype runner: {len(runner._skips)} skip(s):")
        for rec in runner._skips:
            print(f"  {rec.unit_label}: {rec.reason}")

"""5xFAD bulk MEA adapter for the Phase-3 shared runner (Wave 3C).

Routes each (tissue, track) through ``fivexfad.fit_track`` with an injectable
MEA caller that wraps ``MeaRunner._call_mea_unit``.  ``fit_track`` owns all
OLS contrast construction, the double _run_mea invocation, and the 7-DataFrame
assembly — this module only:

  1. Builds the runner instance and a ``_call_mea_unit`` wrapper that satisfies
     ``fit_track``'s ``mea_caller`` signature.
  2. Calls ``fit_track(..., mea_caller=...) `` for each (tissue, track).
  3. Writes the returned 7-DataFrame dict to ``scratch_dir``.

The canonical ``kinase_attribution_5xfad/`` directory is NEVER touched.

Outputs per ``{tissue}_{track}`` prefix, flat in ``scratch_dir/``:
  {prefix}_site_level_ols.csv
  {prefix}_mea_stoichiometry.csv
  {prefix}_mea_raw_phospho.csv
  {prefix}_mea_global_shift.csv
  {prefix}_winsorized_sites.csv
  {prefix}_mea_substrate_sets.csv
  {prefix}_contrast_qc.csv
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from alz.bulk_mea import enrich as kinase_enrich
from alz.core.mea_runner import MeaRunner, MeaUnit, RunResult, SkipRecord
from alz.ingest.fivexfad import (
    KINASE_TRACKS,
    TISSUES,
    OUTPUT_DIR,
    _track_prefix,
    fit_track,
)


def run_via_runner(
    scratch_dir: str | Path,
    manifest: pd.DataFrame,
    tissues: list[str] | None = None,
    tracks: list[str] | None = None,
) -> None:
    """Run 5xFAD bulk MEA through the shared runner, writing all output to scratch_dir.

    For each (tissue, track), calls ``fit_track`` with a ``mea_caller`` that wraps
    ``MeaRunner._call_mea_unit`` so the runner records SkipRecords without
    duplicating OLS or concat logic.  The 7-DataFrame dict returned by
    ``fit_track`` is written flat to ``scratch_dir``.

    Parameters
    ----------
    scratch_dir : str | Path
        Destination for all outputs.  Created if absent.
        The canonical ``kinase_attribution_5xfad/`` directory is never touched.
    manifest : pd.DataFrame
        Loaded sample_manifest.csv (caller responsible).
    tissues : list[str] | None
        Subset of TISSUES to run.  ``None`` runs both.
    tracks : list[str] | None
        Subset of KINASE_TRACKS to run.  ``None`` runs both.
    """
    scratch_dir = Path(scratch_dir)
    scratch_dir.mkdir(parents=True, exist_ok=True)
    runner = MeaRunner(kinase_enrich)

    class _PassAdapter:
        """Adapter stub whose skip_check always passes.

        The stoich-matrix existence guard is handled outside via the
        OUTPUT_DIR path check below; once we reach fit_track the inputs
        are confirmed present.
        """

        def skip_check(self, unit: MeaUnit) -> tuple[bool, str | None]:
            return False, None

        def write_aggregates(self, result: RunResult) -> None:
            pass

        def write_provenance(self, skips: list[SkipRecord]) -> None:
            pass

    _adapter = _PassAdapter()

    for tissue in (tissues or list(TISSUES)):
        for track in (tracks or list(KINASE_TRACKS)):
            prefix = _track_prefix(tissue, track)
            kl_track = KINASE_TRACKS[track]["kl_track"]
            print(f"\n=== 5xFAD bulk MEA (runner): {tissue}/{track} ===")

            stoich_path = OUTPUT_DIR / f"{prefix}_stoichiometry_matrix.csv"
            if not stoich_path.exists():
                rec = SkipRecord(
                    unit_label=f"{tissue}/{track}",
                    reason="stoichiometry matrix absent",
                    detail=str(stoich_path),
                )
                runner._skips.append(rec)
                print(f"  SKIP: {rec.reason}: {stoich_path}")
                continue

            def _make_mea_caller(
                tissue_: str,
                kl_track_: str,
            ):
                """Return a mea_caller that routes one _run_mea call through the runner.

                The returned function has the same signature as
                ``kinase_enrich._run_mea`` (positional motif_series,
                results_by_contrast, lfc_key; keyword site_ids, gene_symbols,
                track) and returns the same 4-tuple
                ``(mea_df, shift_df, wins_df, substrate_df)``.

                The ``lfc_key`` and ``track`` arguments are forwarded from
                ``fit_track``'s call site, so the same caller is used for both
                stoich_lfc and raw_lfc invocations.
                """
                def _caller(
                    motif_series,
                    results_by_contrast,
                    lfc_key,
                    *,
                    site_ids=None,
                    gene_symbols=None,
                    track=kl_track_,
                ):
                    infix = "" if lfc_key == "stoich_lfc" else "_raw"
                    unit = MeaUnit(
                        track=kl_track_,
                        kind="stoich" if lfc_key == "stoich_lfc" else "raw",
                        lfc_key=lfc_key,
                        out_dir=scratch_dir,
                        infix=infix,
                        suffix="",
                        motif_series=motif_series,
                        site_ids=site_ids,
                        gene_symbols=gene_symbols,
                        results_by_contrast=results_by_contrast,
                    )
                    result = runner._call_mea_unit(unit, _adapter)
                    if result is None:
                        # Skip recorded by runner; return empty frames so
                        # fit_track's concat/annotate logic stays intact.
                        empty = pd.DataFrame()
                        return empty, empty, empty, empty
                    return (
                        result.mea_df,
                        result.shift_df,
                        result.wins_df,
                        result.substrate_df,
                    )
                return _caller

            mea_caller = _make_mea_caller(tissue, kl_track)
            results = fit_track(tissue, track, manifest, mea_caller=mea_caller)

            for name, df in results.items():
                out_path = scratch_dir / f"{prefix}_{name}.csv"
                df.to_csv(out_path, index=False)
                print(f"  wrote {prefix}_{name}.csv  rows={len(df)}")

    if runner._skips:
        print(f"\n5xFAD bulk runner: {len(runner._skips)} skip(s):")
        for rec in runner._skips:
            print(f"  {rec.unit_label}: {rec.reason}")

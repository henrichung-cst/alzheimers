"""Per-donor time-course MEA kinase enrichment for the T-cell exhaustion cohort.

For each donor independently, builds a per-site delta vector for every later
timepoint against the **Day 2 baseline** and runs MEA against each, on two
preprocessing tracks:

  * stoichiometry (`log2(phospho) − log2(protein)`) — primary signal
  * raw phospho (uncorrected, self-normalized intensity) — sensitivity check

Contrasts are within-donor (timepoints are not comparable across donors; see
``docs/tcell_exhaustion_analysis_summary.md``). Because the D2 matrices are
already log2 + per-run median-centered, the contrast for a later day is a plain
subtraction `value(day) − value(d2)` = log2FC, sign convention `+ = up at the
later timepoint`.

Scope (hard constraints, do not re-litigate — see the cohort memory):
  * Kinase MEA runs on the **bulk** self-normalized substrate ONLY, never on a
    deconvoluted one (same architecture as mouse/Mukesh).
  * Donor 2 has **no IMAC** → no Ser/Thr (`st`) track at all, and its pY export
    carries no flanking motif → no usable MEA. Donor 2 therefore produces no
    kinase enrichment; the skip is recorded numerically in the manifest, not
    hidden.

Inputs (under outputs/reports/kinase_attribution_tcells/donor{1,2}/, written by
``alz/ingest/tcells.py --reshape``):
  stoichiometry_matrix{,_pY}.csv     — stoichiometry track
  raw_phospho_normalized{,_pY}.csv   — raw phospho track
  ../data_ingest_tcells/donor{n}/sample_mapping.csv

Outputs (under outputs/reports/kinase_attribution_tcells/donor{n}/mea/):
  mea_timecourse{,_raw}{suffix}.csv      — long form: timepoint, kinase, NES, FDR, ...
  kinase_timepoint_nes{,_raw}{suffix}.csv — kinase x timepoint wide NES matrix
  kinase_timepoint_fdr{,_raw}{suffix}.csv — kinase x timepoint wide FDR matrix
  recurrence{,_raw}{suffix}.csv          — kinases significant in >= k timepoints
  mea_global_shift{,_raw}{suffix}.csv    — per-timepoint median-centering log
  winsorized_sites{,_raw}{suffix}.csv    — per-timepoint winsorized site log
  mea_substrate_sets{,_raw}{suffix}.csv  — per (timepoint, kinase) substrate motif set
  mea_manifest.json                      — what ran vs was skipped and why
"""

import argparse
import json
import os
import sys
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import pandas as pd

from alz.shared import config
from alz.bulk_mea import enrich as kinase_enrich
from alz.ingest.tcells import KINASE_DIR, INGEST_DIR, META_COLS
from alz.core.mea_outputs import (
    KIND_SPEC as _SHARED_KIND_SPEC,
    build_nes_fdr_matrices,
    build_recurrence_summary,
    mea_output_path,
)

# Tracks to attempt per donor. "st" = Ser/Thr (IMAC), "py" = Tyr (pY).
TRACKS = ["st", "py"]

# stoichiometry (primary) + raw phospho (sensitivity), mirroring mukesh_perdonor.
_KIND_SPEC = {
    "stoich": {"base": "stoichiometry_matrix",   "lfc_key": "stoich_lfc", "infix": ""},
    "raw":    {"base": "raw_phospho_normalized", "lfc_key": "raw_lfc",    "infix": "_raw"},
}


def _mea_dir(donor: str) -> str:
    return os.path.join(KINASE_DIR, donor, "mea")


def _matrix_path(donor: str, kind: str, track: str) -> str:
    suffix = config.PHOSPHO_TRACKS[track]["output_suffix"]
    return os.path.join(KINASE_DIR, donor, f"{_KIND_SPEC[kind]['base']}{suffix}.csv")


def _baseline_and_days(donor: str, matrix: pd.DataFrame) -> tuple[str, list[str]]:
    """Return (baseline_col, [later_day_cols]) restricted to columns present.

    Baseline is the Day-2 sample flagged in this donor's sample_mapping; later
    days are every other sample column the track actually carries (tracks differ
    in day coverage — e.g. donor1 Total has d11 but IMAC/pY do not).
    """
    mapping = pd.read_csv(
        os.path.join(INGEST_DIR, donor, "sample_mapping.csv")
    )
    baseline_ids = mapping.loc[mapping["baseline"], "sample_id"].tolist()
    sample_cols = [c for c in matrix.columns if c not in META_COLS]
    baseline = next((b for b in baseline_ids if b in sample_cols), None)
    if baseline is None:
        raise RuntimeError(
            f"{donor}: no baseline (Day-2) column among track samples {sample_cols}"
        )
    days = [c for c in sample_cols if c != baseline]
    return baseline, days


def _build_timepoint_deltas(
    matrix: pd.DataFrame, baseline: str, days: list[str], lfc_key: str,
) -> dict[str, dict[str, np.ndarray]]:
    """``{f'{day}_vs_d2': {lfc_key: value(day) − value(baseline)}}`` per site."""
    base_vec = matrix[baseline].to_numpy(dtype=float)
    out: dict[str, dict[str, np.ndarray]] = {}
    for day in days:
        out[f"{day}_vs_d2"] = {lfc_key: matrix[day].to_numpy(dtype=float) - base_vec}
    return out


def _n_motif(matrix: pd.DataFrame) -> int:
    m = matrix["motif"]
    return int((m.notna() & (m.astype(str) != "")).sum())


def _write_timepoint_aggregates(
    mea_df: pd.DataFrame,
    infix: str,
    suffix: str,
    out_dir: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Write wide NES/FDR matrices and recurrence table for one (donor, track, kind).

    Derives ``tp_order`` (numeric-day sort) internally from the contrast labels
    present in ``mea_df``.  Single copy of the aggregate-writing logic; called
    from both the canonical ``_run_track_kind`` path and the scratch
    ``regenerate_aggregates_to_scratch`` path so that production and harness
    share identical code.

    Returns ``(nes_wide, fdr_wide)`` so callers can report shapes without
    re-pivoting.
    """
    tp_order = sorted(
        [c.replace("_vs_d2", "") for c in mea_df["contrast"].unique()],
        key=lambda s: int(s.rsplit("_d", 1)[-1]),
    )
    nes_wide, fdr_wide = build_nes_fdr_matrices(
        mea_df,
        entity_col_name="timepoint",
        contrast_suffix="_vs_d2",
        entity_order=tp_order,
    )
    nes_wide.to_csv(mea_output_path(out_dir, "kinase_timepoint_nes", infix, suffix))
    fdr_wide.to_csv(mea_output_path(out_dir, "kinase_timepoint_fdr", infix, suffix))

    rec = build_recurrence_summary(
        nes_wide, fdr_wide,
        subset_ids=tp_order,
        axis_noun="timepoints",
        fdr_thresh=config.MEA_FDR_THRESH,
    )
    rec_path = mea_output_path(out_dir, "recurrence", infix, suffix)
    rec.to_csv(rec_path, index=False)
    print(f"  recurrence: {os.path.relpath(rec_path, config.REPO_ROOT)}  "
          f"(>=1 timepoint sig: {(rec['n_timepoints_sig'] >= 1).sum()}; "
          f">=ceil(N/2): {(rec['n_timepoints_sig'] >= np.ceil(len(tp_order) / 2)).sum()})")

    return nes_wide, fdr_wide


def _run_track_kind(donor: str, track: str, kind: str, skips: list[dict]) -> bool:
    """Run one (donor, track, kind) MEA. Returns True if MEA actually ran."""
    spec = _KIND_SPEC[kind]
    suffix = config.PHOSPHO_TRACKS[track]["output_suffix"]
    tag = f"{donor}/{track}/{kind}"
    path = _matrix_path(donor, kind, track)

    if not os.path.exists(path):
        # Expected for donor2 st (no IMAC). Record, don't warn loudly.
        skips.append({"donor": donor, "track": track, "kind": kind,
                      "reason": "matrix_absent", "path": os.path.relpath(path, config.REPO_ROOT)})
        print(f"  [{tag}] matrix absent -> skip ({os.path.basename(path)})")
        return False

    matrix = pd.read_csv(path)
    n_motif = _n_motif(matrix)
    if n_motif == 0:
        # Expected for donor2 pY (ForPerseus export has no flanking region).
        skips.append({"donor": donor, "track": track, "kind": kind,
                      "reason": "no_motif", "n_sites": int(len(matrix))})
        print(f"  [{tag}] {len(matrix)} sites but 0 motifs -> skip (MEA needs motifs)")
        return False

    baseline, days = _baseline_and_days(donor, matrix)
    print(f"\n=== Per-donor time-course MEA: {tag} ===")
    print(f"  sites: {len(matrix)}  motifs: {n_motif}  "
          f"baseline: {baseline}  timepoints: {len(days)} ({', '.join(days)})")

    results = _build_timepoint_deltas(matrix, baseline, days, spec["lfc_key"])

    mea_df, shift_df, wins_df, substrate_df = kinase_enrich._run_mea(
        motif_series=matrix["motif"],
        results_by_contrast=results,
        lfc_key=spec["lfc_key"],
        site_ids=matrix["site_id"].values,
        gene_symbols=matrix["gene_symbol"].values,
        track=track,
    )

    out_dir = _mea_dir(donor)
    os.makedirs(out_dir, exist_ok=True)
    infix = spec["infix"]

    def _w(df, stem):
        p = os.path.join(out_dir, f"{stem}{infix}{suffix}.csv")
        df.to_csv(p, index=False)
        return p

    mea_path = _w(mea_df, "mea_timecourse")
    _w(shift_df, "mea_global_shift")
    _w(wins_df, "winsorized_sites")
    _w(substrate_df, "mea_substrate_sets")
    print(f"  wrote {os.path.relpath(mea_path, config.REPO_ROOT)}  rows={len(mea_df)}")

    if mea_df.empty:
        skips.append({"donor": donor, "track": track, "kind": kind,
                      "reason": "mea_empty", "n_sites": int(len(matrix))})
        print(f"  [{tag}] WARNING: empty MEA result (no contrast cleared "
              f"MEA_MIN_SITES={config.MEA_MIN_SITES}); no aggregation")
        return True

    _write_timepoint_aggregates(mea_df, infix, out_dir=out_dir, suffix=suffix)
    return True


def _run_donor(donor: str) -> dict:
    print(f"\n########## {donor} ##########")
    skips: list[dict] = []
    ran: list[str] = []
    for track in TRACKS:
        for kind in _KIND_SPEC:
            if _run_track_kind(donor, track, kind, skips):
                ran.append(f"{track}/{kind}")
    manifest = {"donor": donor, "mea_ran": ran, "mea_skipped": skips,
                "mea_fdr_thresh": config.MEA_FDR_THRESH,
                "mea_min_sites": config.MEA_MIN_SITES}
    out_dir = _mea_dir(donor)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "mea_manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"  {donor}: MEA ran {ran or '(none)'}; skipped {len(skips)} combos")
    return manifest


# ---------------------------------------------------------------------------
# Opt-in scratch regeneration (Phase 2 refactor proof harness)
# ---------------------------------------------------------------------------

def regenerate_aggregates_to_scratch(
    donor: str, scratch_dir: str, track: str, kind: str
) -> None:
    """Recompute wide + recurrence tables from the on-disk canonical long table.

    Does NOT run MEA. Reads ``mea_timecourse{infix}{suffix}.csv`` from the
    donor's canonical mea dir and calls ``_write_timepoint_aggregates`` (which
    internally uses ``alz.core.mea_outputs``) to regenerate the
    kinase×timepoint NES/FDR matrices and recurrence table, writing them into
    ``scratch_dir``.

    Audit passthrough tables (shift/wins/substrate) are NOT regenerated here.
    """
    suffix = config.PHOSPHO_TRACKS[track]["output_suffix"]
    infix = _SHARED_KIND_SPEC[kind]["infix"]

    long_path = os.path.join(_mea_dir(donor), f"mea_timecourse{infix}{suffix}.csv")
    if not os.path.exists(long_path):
        print(f"  [tcells scratch/{donor}/{track}/{kind}] long table absent: {long_path}; skip")
        return

    print(f"\n=== Scratch regen: tcells {donor} track={track} kind={kind} ===")
    mea_df = pd.read_csv(long_path)
    print(f"  read {long_path}  rows={len(mea_df)}")

    if mea_df.empty:
        print("  WARNING: empty long table; nothing to regenerate")
        return

    os.makedirs(scratch_dir, exist_ok=True)
    nes_wide, _ = _write_timepoint_aggregates(mea_df, infix, suffix=suffix, out_dir=scratch_dir)
    print(f"  scratch wide: {nes_wide.shape}  wrote to {scratch_dir}/")


# ---------------------------------------------------------------------------
# Opt-in runner entry (Phase 3 refactor scratch path)
# ---------------------------------------------------------------------------

def _run_via_runner(scratch_dir: str, donors: list[str], tracks: list[str]) -> None:
    """Run T-cell MEA through the shared Phase-3 runner to a scratch directory.

    NEVER writes to the canonical KINASE_DIR tree.  Coexists with the inline
    `_run_donor` canonical path during the Phase-3 migration window.
    """
    from alz.bulk_mea import enrich as kinase_enrich
    from alz.core.mea_runner import MeaRunner
    from alz.core.tcells_mea_adapter import TcellsMeaAdapter

    adapter = TcellsMeaAdapter(
        scratch_dir=scratch_dir,
        donors=donors,
        tracks=tracks,
    )
    runner = MeaRunner(kinase_enrich)
    results = runner.run_all(adapter)
    print(f"\n[runner] done: {len(results)} unit(s) ran; "
          f"{len(runner.skips)} skip(s)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--donor", choices=["donor1", "donor2", "both"], default="both",
        help="donor to run (default: both).",
    )
    parser.add_argument(
        "--scratch-dir",
        default=None,
        metavar="DIR",
        help=(
            "If set, regenerate wide + recurrence tables from canonical long "
            "tables into DIR (proof harness — does NOT re-run MEA). "
            "Combine with --donor to restrict to one donor; both tracks and "
            "both kinds are always attempted."
        ),
    )
    parser.add_argument(
        "--runner-scratch-dir",
        default=None,
        metavar="DIR",
        help=(
            "If set, run T-cell MEA through the Phase-3 shared runner and write "
            "outputs to DIR (scratch only — never overwrites canonical outputs). "
            "Combine with --donor and --track to restrict scope."
        ),
    )
    parser.add_argument(
        "--track", choices=["st", "py", "both"], default="both",
        help="track to run in runner mode (default: both). Ignored outside runner mode.",
    )
    args = parser.parse_args(argv)

    if args.runner_scratch_dir is not None:
        # Runner-scratch mode: MEA through the Phase-3 shared runner.
        donors = ["donor1", "donor2"] if args.donor == "both" else [args.donor]
        tracks = ["st", "py"] if args.track == "both" else [args.track]
        _run_via_runner(args.runner_scratch_dir, donors, tracks)
        return 0

    if args.scratch_dir is not None:
        # Scratch-regen mode: no MEA, just recompute aggregates from disk.
        donors = ["donor1", "donor2"] if args.donor == "both" else [args.donor]
        for d in donors:
            for t in TRACKS:
                for k in _KIND_SPEC:
                    regenerate_aggregates_to_scratch(d, args.scratch_dir, t, k)
        return 0

    donors = ["donor1", "donor2"] if args.donor == "both" else [args.donor]
    for d in donors:
        if not os.path.isdir(os.path.join(KINASE_DIR, d)):
            print(f"ERROR: missing {os.path.join(KINASE_DIR, d)}; "
                  f"run `pixi run tcells-reshape` first")
            return 2
        _run_donor(d)
    return 0


if __name__ == "__main__":
    sys.exit(main())

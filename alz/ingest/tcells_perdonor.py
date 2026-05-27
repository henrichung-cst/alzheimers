"""Per-donor time-course MEA kinase enrichment for the T-cell exhaustion cohort.

For each donor independently, builds a per-site delta vector for every later
timepoint against the **Day 2 baseline** and runs MEA against each, on two
preprocessing tracks:

  * stoichiometry (`log2(phospho) − log2(protein)`) — primary signal
  * raw phospho (uncorrected, self-normalized intensity) — sensitivity check

Contrasts are within-donor (timepoints are not comparable across donors; see
``docs/plans/meeting_notes_triage_2026-05-27.md``). Because the D2 matrices are
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

    # Order timepoints by day number, not lexically (d2 < d11, not d11 < d2).
    mea_df = mea_df.copy()
    mea_df["timepoint"] = mea_df["contrast"].str.replace("_vs_d2", "", regex=False)
    tp_order = sorted(
        mea_df["timepoint"].unique(),
        key=lambda s: int(s.rsplit("_d", 1)[-1]),
    )

    nes_wide = mea_df.pivot_table(index="kinase", columns="timepoint",
                                  values="NES", aggfunc="first").reindex(columns=tp_order)
    fdr_wide = mea_df.pivot_table(index="kinase", columns="timepoint",
                                  values="FDR", aggfunc="first").reindex(columns=tp_order)
    nes_wide.to_csv(os.path.join(out_dir, f"kinase_timepoint_nes{infix}{suffix}.csv"))
    fdr_wide.to_csv(os.path.join(out_dir, f"kinase_timepoint_fdr{infix}{suffix}.csv"))

    sig = fdr_wide < config.MEA_FDR_THRESH
    up = sig & (nes_wide > 0)
    dn = sig & (nes_wide < 0)
    rec = pd.DataFrame({
        "kinase": fdr_wide.index,
        "n_timepoints_sig": sig.sum(axis=1).values,
        "n_timepoints_up": up.sum(axis=1).values,
        "n_timepoints_down": dn.sum(axis=1).values,
        "n_timepoints_tested": fdr_wide.notna().sum(axis=1).values,
        "median_nes": nes_wide.median(axis=1, skipna=True).values,
        "median_nes_sig_only": nes_wide.where(sig).median(axis=1, skipna=True).values,
    }).sort_values(["n_timepoints_sig", "n_timepoints_tested"], ascending=[False, False])
    rec_path = os.path.join(out_dir, f"recurrence{infix}{suffix}.csv")
    rec.to_csv(rec_path, index=False)
    print(f"  recurrence: {os.path.relpath(rec_path, config.REPO_ROOT)}  "
          f"(>=1 timepoint sig: {(rec['n_timepoints_sig'] >= 1).sum()}; "
          f">=ceil(N/2): {(rec['n_timepoints_sig'] >= np.ceil(len(tp_order) / 2)).sum()})")
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--donor", choices=["donor1", "donor2", "both"], default="both",
        help="donor to run (default: both).",
    )
    args = parser.parse_args(argv)
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

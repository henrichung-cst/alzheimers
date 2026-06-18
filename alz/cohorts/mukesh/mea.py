"""Per-donor MEA kinase enrichment for the human NBB (Mukesh) cohort.

For each AD donor, builds a per-site delta vector against the CTRL mean
on two preprocessing tracks and runs MEA against each:

  * stoichiometry (`log2(phospho) − log2(protein)`) — primary signal
  * raw phospho (uncorrected, normalized intensity) — sensitivity check

Aggregates each into a kinase x donor NES matrix and a recurrence
summary (kinases significant at FDR < MEA_FDR_THRESH in >= k donors).
The raw-phospho variant mirrors `alz/bulk_mea/mechanism.py` for mouse: it lets
the viewer cross-check whether a per-donor stoichiometry signal is
abundance-driven vs activity-driven.

Inputs (under outputs/reports/kinase_attribution_human/, written by
``python -m alz.cohorts.mukesh.ingest --reshape``):
  stoichiometry_matrix{,_pY}.csv     — stoichiometry track per residue class
  raw_phospho_normalized{,_pY}.csv   — raw phospho track per residue class
  ../data_ingest_human/sample_mapping.csv

Outputs (under outputs/reports/kinase_attribution_human/perdonor/):
  mea_perdonor{,_raw}{suffix}.csv         — long form: donor, kinase, NES, FDR, ...
  kinase_donor_nes{,_raw}{suffix}.csv     — kinase x donor wide NES matrix
  kinase_donor_fdr{,_raw}{suffix}.csv     — kinase x donor wide FDR matrix
  recurrence{,_raw}{suffix}.csv           — kinases significant in >= k donors
  mea_global_shift{,_raw}{suffix}.csv     — per-donor median-centering log
  winsorized_sites{,_raw}{suffix}.csv     — per-donor winsorized site log
  mea_substrate_sets{,_raw}{suffix}.csv   — per (donor, kinase) substrate motif set
                                            used as the GSEA hit set (mirrors the
                                            mouse pipeline; consumed by the viewer
                                            running-enrichment panel)
"""

import argparse
import json
import os
import sys
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parents[3])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import pandas as pd

from alz.shared import config
from alz.bulk_mea import enrich as kinase_enrich
from alz.core.mechanism_attribution import classify_mechanisms
from alz.cohorts.mukesh.ingest import (
    HUMAN_DATA_INGEST_DIR,
    HUMAN_KINASE_DIR,
    SAMPLE_MAPPING_CSV,
)
from alz.core.mea_outputs import (
    KIND_SPEC as _SHARED_KIND_SPEC,
    build_nes_fdr_matrices,
    build_recurrence_summary,
    mea_output_path,
)

PERDONOR_DIR = os.path.join(HUMAN_KINASE_DIR, "perdonor")


def _write_mechanism_attribution(track: str, out_dir: str = PERDONOR_DIR) -> None:
    """Write paired stoich/raw mechanism attribution for one track."""
    suffix = config.PHOSPHO_TRACKS[track]["output_suffix"]
    stoich_path = os.path.join(out_dir, f"mea_perdonor{suffix}.csv")
    raw_path = os.path.join(out_dir, f"mea_perdonor_raw{suffix}.csv")
    if not os.path.exists(stoich_path):
        print(f"  mechanism attribution for track={track}: missing {stoich_path}; skip")
        return
    if not os.path.exists(raw_path):
        print(f"  mechanism attribution for track={track}: missing {raw_path}; skip")
        return

    stoich_df = pd.read_csv(stoich_path)
    raw_df = pd.read_csv(raw_path)

    constants = {"cohort": "mukesh", "track": track}
    for frame in (stoich_df, raw_df):
        for name, value in constants.items():
            frame[name] = value
        if "donor" not in frame.columns:
            if "contrast" not in frame.columns:
                print(
                    "  mechanism attribution requires donor context; "
                    f"{Path(stoich_path).name if frame is stoich_df else Path(raw_path).name} missing both donor and contrast; skip"
                )
                return
            frame["donor"] = frame["contrast"].str.replace(
                r"_vs_CTRLmean$", "", regex=True
            )

    attributed = classify_mechanisms(
        stoich_df,
        raw_df,
        context_cols=["cohort", "track", "donor"],
    )
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"mechanism_attribution{suffix}.csv")
    attributed.to_csv(out_path, index=False)
    print(f"  mechanism attribution written: {out_path}  rows={len(attributed)}")


def _load_track_matrix(track: str, kind: str = "stoich") -> pd.DataFrame | None:
    """Load the per-track matrix; ``kind`` selects stoichiometry vs raw phospho."""
    suffix = config.PHOSPHO_TRACKS[track]["output_suffix"]
    base = "stoichiometry_matrix" if kind == "stoich" else "raw_phospho_normalized"
    path = os.path.join(HUMAN_KINASE_DIR, f"{base}{suffix}.csv")
    if not os.path.exists(path):
        if kind == "raw":
            return None
        raise FileNotFoundError(
            f"missing {path}; run python -m alz.cohorts.mukesh.ingest --reshape"
        )
    return pd.read_csv(path)


def _split_samples(mapping: pd.DataFrame) -> tuple[list[str], list[str]]:
    ad = mapping.loc[mapping["group"] == "AD", "sample_id"].tolist()
    ctrl = mapping.loc[mapping["group"] == "CTRL", "sample_id"].tolist()
    if not ad or not ctrl:
        raise RuntimeError(
            f"sample_mapping missing AD or CTRL: AD={len(ad)} CTRL={len(ctrl)}"
        )
    return sorted(ad), sorted(ctrl)


def _build_donor_deltas(
    matrix: pd.DataFrame, ad_ids: list[str], ctrl_ids: list[str], lfc_key: str,
) -> dict[str, dict[str, np.ndarray]]:
    """Return ``{donor: {lfc_key: np.array per site}}`` for AD + CTRL donors.

    Delta is ``value_donor_i − nanmean(value_CTRL)`` per site, using the
    *full* CTRL block as the reference for both groups (per supervisor
    direction). CTRL donors are scored against the same mean they help
    define; this biases CTRL deltas toward zero by ~1/N but keeps both
    groups on a single, symmetric reference scale.

    Sites with no CTRL coverage are NaN (downstream ``_run_mea`` drops
    them before ranking).
    """
    ctrl_block = matrix[ctrl_ids].to_numpy(dtype=float)
    with np.errstate(all="ignore"):
        ctrl_mean = np.nanmean(ctrl_block, axis=1)
    out: dict[str, dict[str, np.ndarray]] = {}
    for donor in (*ad_ids, *ctrl_ids):
        donor_vec = matrix[donor].to_numpy(dtype=float)
        out[f"{donor}_vs_CTRLmean"] = {lfc_key: donor_vec - ctrl_mean}
    return out


_KIND_SPEC = {
    "stoich": {"matrix_kind": "stoich", "lfc_key": "stoich_lfc", "infix": ""},
    "raw":    {"matrix_kind": "raw",    "lfc_key": "raw_lfc",    "infix": "_raw"},
}


def _write_donor_aggregates(
    mea_df: pd.DataFrame,
    ad_ids: list[str],
    ctrl_ids: list[str],
    infix: str,
    suffix: str,
    out_dir: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Write wide NES/FDR matrices and AD + CTRL recurrence tables.

    Single copy of the aggregate-writing logic; called from both the canonical
    ``_run_track_kind`` path and the scratch ``regenerate_aggregates_to_scratch``
    path so that production and harness share identical code.

    Returns ``(nes_wide, fdr_wide)`` so callers can report shapes without
    re-pivoting.
    """
    donor_order = ad_ids + ctrl_ids
    nes_wide, fdr_wide = build_nes_fdr_matrices(
        mea_df,
        entity_col_name="donor",
        contrast_suffix="_vs_CTRLmean",
        entity_order=donor_order,
    )
    nes_wide.to_csv(mea_output_path(out_dir, "kinase_donor_nes", infix, suffix))
    fdr_wide.to_csv(mea_output_path(out_dir, "kinase_donor_fdr", infix, suffix))

    # AD recurrence — primary, feeds SEA-AD agreement.
    rec = build_recurrence_summary(
        nes_wide, fdr_wide,
        subset_ids=ad_ids,
        axis_noun="donors",
        fdr_thresh=config.MEA_FDR_THRESH,
    )
    rec_path = mea_output_path(out_dir, "recurrence", infix, suffix)
    rec.to_csv(rec_path, index=False)
    print(
        f"  recurrence written: {rec_path}  "
        f"(>=1 donor sig: {(rec['n_donors_sig'] >= 1).sum()}; "
        f">=ceil(N/2) donors sig: "
        f"{(rec['n_donors_sig'] >= np.ceil(len(ad_ids) / 2)).sum()})"
    )

    # CTRL recurrence — sibling table; never feeds SEA-AD agreement. Used by
    # the viewer to surface controls that look case-like after LOO.
    rec_ctrl = build_recurrence_summary(
        nes_wide, fdr_wide,
        subset_ids=ctrl_ids,
        axis_noun="donors",
        fdr_thresh=config.MEA_FDR_THRESH,
    )
    rec_ctrl_path = mea_output_path(out_dir, "recurrence_ctrl", infix, suffix)
    rec_ctrl.to_csv(rec_ctrl_path, index=False)
    print(
        f"  ctrl recurrence written: {rec_ctrl_path}  "
        f"(>=1 ctrl sig: {(rec_ctrl['n_donors_sig'] >= 1).sum() if len(rec_ctrl) else 0})"
    )

    return nes_wide, fdr_wide


def _run_track_kind(track: str, kind: str, mapping: pd.DataFrame) -> bool:
    spec = _KIND_SPEC[kind]
    print(f"\n=== Per-donor MEA: track={track} kind={kind} ===")
    matrix = _load_track_matrix(track, spec["matrix_kind"])
    if matrix is None:
        print(f"  [{track}/{kind}] input matrix missing; skip.")
        return False
    ad_ids, ctrl_ids = _split_samples(mapping)
    # Keep only samples present in the matrix.
    ad_ids = [s for s in ad_ids if s in matrix.columns]
    ctrl_ids = [s for s in ctrl_ids if s in matrix.columns]
    print(f"  AD donors: {len(ad_ids)}  CTRL: {len(ctrl_ids)}  sites: {len(matrix)}")

    results = _build_donor_deltas(matrix, ad_ids, ctrl_ids, spec["lfc_key"])

    motif_series = matrix["motif"]
    site_ids = matrix["site_id"].values
    gene_symbols = matrix["gene_symbol"].values

    mea_df, shift_df, wins_df, substrate_df = kinase_enrich._run_mea(
        motif_series=motif_series,
        results_by_contrast=results,
        lfc_key=spec["lfc_key"],
        site_ids=site_ids,
        gene_symbols=gene_symbols,
        track=track,
    )

    suffix = config.PHOSPHO_TRACKS[track]["output_suffix"]
    infix = spec["infix"]
    os.makedirs(PERDONOR_DIR, exist_ok=True)

    mea_path = os.path.join(PERDONOR_DIR, f"mea_perdonor{infix}{suffix}.csv")
    mea_df.to_csv(mea_path, index=False)
    print(f"  wrote {mea_path}  rows={len(mea_df)}")

    shift_df.to_csv(
        os.path.join(PERDONOR_DIR, f"mea_global_shift{infix}{suffix}.csv"), index=False
    )
    wins_df.to_csv(
        os.path.join(PERDONOR_DIR, f"winsorized_sites{infix}{suffix}.csv"), index=False
    )
    substrate_df.to_csv(
        os.path.join(PERDONOR_DIR, f"mea_substrate_sets{infix}{suffix}.csv"),
        index=False,
    )

    if mea_df.empty:
        print("  WARNING: empty MEA result; skipping aggregation")
        return True

    _write_donor_aggregates(mea_df, ad_ids, ctrl_ids, infix, suffix, PERDONOR_DIR)
    return True


def _run_track(track: str, mapping: pd.DataFrame) -> None:
    stoich_ran = _run_track_kind(track, "stoich", mapping)
    raw_ran = _run_track_kind(track, "raw", mapping)
    if stoich_ran and raw_ran:
        _write_mechanism_attribution(track, PERDONOR_DIR)
    else:
        print(
            f"  mechanism attribution for track={track}: "
            "current stoich/raw run incomplete; skip"
        )


# ---------------------------------------------------------------------------
# Opt-in scratch regeneration (Phase 2 refactor proof harness)
# ---------------------------------------------------------------------------

def regenerate_aggregates_to_scratch(
    scratch_dir: str, track: str, kind: str
) -> None:
    """Recompute wide + recurrence tables from the on-disk canonical long table.

    Does NOT run MEA. Reads ``mea_perdonor{infix}{suffix}.csv`` from
    PERDONOR_DIR and calls ``_write_donor_aggregates`` (which internally uses
    ``alz.core.mea_outputs``) to regenerate the kinase×donor NES/FDR matrices
    and both recurrence tables, writing them into ``scratch_dir``.

    Audit passthrough tables (shift/wins/substrate) are NOT regenerated here;
    they are trivial ``to_csv`` calls and the substrate file is ~100 MB.
    """
    suffix = config.PHOSPHO_TRACKS[track]["output_suffix"]
    infix = _SHARED_KIND_SPEC[kind]["infix"]

    long_path = os.path.join(PERDONOR_DIR, f"mea_perdonor{infix}{suffix}.csv")
    if not os.path.exists(long_path):
        print(f"  [mukesh scratch/{track}/{kind}] long table absent: {long_path}; skip")
        return

    print(f"\n=== Scratch regen: mukesh track={track} kind={kind} ===")
    mea_df = pd.read_csv(long_path)
    print(f"  read {long_path}  rows={len(mea_df)}")

    if mea_df.empty:
        print("  WARNING: empty long table; nothing to regenerate")
        return

    # Recover AD/CTRL donor order from sample_mapping (same small file as
    # the canonical run; restricts to donors actually present in the long
    # table so the scratch reindexing mirrors canonical behaviour).
    mapping = pd.read_csv(SAMPLE_MAPPING_CSV)
    ad_ids, ctrl_ids = _split_samples(mapping)
    contrasts_present = set(mea_df["contrast"].unique())
    ad_ids   = [s for s in ad_ids   if f"{s}_vs_CTRLmean" in contrasts_present]
    ctrl_ids = [s for s in ctrl_ids if f"{s}_vs_CTRLmean" in contrasts_present]

    os.makedirs(scratch_dir, exist_ok=True)
    nes_wide, _ = _write_donor_aggregates(mea_df, ad_ids, ctrl_ids, infix, suffix, scratch_dir)
    print(f"  scratch wide: {nes_wide.shape}  wrote to {scratch_dir}/")


def _run_via_runner(scratch_dir: str, tracks: list[str]) -> int:
    """Opt-in Phase-3 runner path.  Drives all (track, kind) units through
    MeaRunner + MukeshMeaAdapter into ``scratch_dir``.

    The inline ``_run_track_kind`` canonical block is NOT called here; both
    orchestration paths coexist during the Phase-3 migration window.
    """
    from alz.core.mea_runner import MeaRunner
    from alz.core.mukesh_mea_adapter import MukeshMeaAdapter
    from alz.bulk_mea import enrich as kinase_enrich

    if not os.path.exists(SAMPLE_MAPPING_CSV):
        print(
            f"ERROR: missing {SAMPLE_MAPPING_CSV}; "
            "run python -m alz.cohorts.mukesh.ingest --reshape first"
        )
        return 2
    mapping = pd.read_csv(SAMPLE_MAPPING_CSV)

    adapter = MukeshMeaAdapter(
        scratch_dir=scratch_dir,
        mapping=mapping,
        tracks=tracks,
    )
    runner = MeaRunner(kinase_enrich)
    results = runner.run_all(adapter)
    print(f"\nRunner finished: {len(results)} unit(s) completed, "
          f"{len(runner.skips)} skip(s).")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--track",
        choices=["st", "py", "both"],
        default="both",
        help="phospho track to run (default: both).",
    )
    parser.add_argument(
        "--scratch-dir",
        default=None,
        metavar="DIR",
        help=(
            "If set, regenerate wide + recurrence tables from canonical long "
            "tables into DIR (proof harness — does NOT re-run MEA). "
            "Combine with --track to restrict to one track; both kinds "
            "(stoich + raw) are always attempted."
        ),
    )
    parser.add_argument(
        "--runner-scratch-dir",
        default=None,
        metavar="DIR",
        help=(
            "Phase-3 opt-in: drive all (track, kind) units through the shared "
            "MeaRunner + MukeshMeaAdapter into DIR.  Does NOT touch the "
            "canonical PERDONOR_DIR.  Use --track to restrict to one track."
        ),
    )
    args = parser.parse_args(argv)

    if args.runner_scratch_dir is not None:
        # Phase-3 runner-driven path: scratch only, canonical block untouched.
        tracks = ["st", "py"] if args.track == "both" else [args.track]
        return _run_via_runner(args.runner_scratch_dir, tracks)

    if args.scratch_dir is not None:
        # Scratch-regen mode: no MEA, just recompute aggregates from disk.
        if not os.path.exists(SAMPLE_MAPPING_CSV):
            print(
                f"ERROR: missing {SAMPLE_MAPPING_CSV}; "
                "run python -m alz.cohorts.mukesh.ingest --reshape first"
            )
            return 2
        tracks = ["st", "py"] if args.track == "both" else [args.track]
        for t in tracks:
            for k in ("stoich", "raw"):
                regenerate_aggregates_to_scratch(args.scratch_dir, t, k)
        return 0

    if not os.path.exists(SAMPLE_MAPPING_CSV):
        print(
            f"ERROR: missing {SAMPLE_MAPPING_CSV}; "
            "run python -m alz.cohorts.mukesh.ingest --reshape first"
        )
        return 2
    mapping = pd.read_csv(SAMPLE_MAPPING_CSV)

    tracks = ["st", "py"] if args.track == "both" else [args.track]
    for t in tracks:
        _run_track(t, mapping)

    # Donor-group sidecar — saves the viewer build from re-parsing
    # sample_mapping.csv to recover AD/CTRL membership.
    ad_ids, ctrl_ids = _split_samples(mapping)
    os.makedirs(PERDONOR_DIR, exist_ok=True)
    groups_path = os.path.join(PERDONOR_DIR, "donor_groups.json")
    with open(groups_path, "w") as fh:
        json.dump({"ad": ad_ids, "ctrl": ctrl_ids}, fh, indent=2)
    print(f"wrote {groups_path}  (AD={len(ad_ids)} CTRL={len(ctrl_ids)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

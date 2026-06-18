"""Phase-3 parity harness for the Mukesh MEA adapter.

Two checks:
  (A) Input-equivalence (exhaustive): for each of the 4 units, compare the
      _run_mea-input fingerprint built by the runner path vs the canonical
      inline path.  Identical fingerprints + frozen deterministic _run_mea
      => identical outputs.

  (B) End-to-end spot-check (py/stoich only): actually run _run_mea through
      the runner for the smallest unit, write to scratch, and diff the 3 key
      output files against canonical.

Usage:
  pixi run python alz/core/phase3_parity_harness.py --scratch-dir <DIR> [--run-e2e]
"""

from __future__ import annotations

import argparse
import hashlib
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
from alz.core.mea_runner import MeaRunner, _series_fingerprint, _array_fingerprint, _contrast_fingerprint
from alz.core.mukesh_mea_adapter import MukeshMeaAdapter
from alz.ingest.mukesh_perdonor import (
    _load_track_matrix,
    _split_samples,
    _build_donor_deltas,
    _KIND_SPEC as _LOCAL_KIND_SPEC,
    PERDONOR_DIR,
)
from alz.ingest.mukesh import SAMPLE_MAPPING_CSV


# ---------------------------------------------------------------------------
# (A) Input-equivalence
# ---------------------------------------------------------------------------

def build_canonical_fingerprint(track: str, kind: str, mapping: pd.DataFrame) -> dict:
    """Reconstruct the _run_mea kwargs exactly as _run_track_kind does."""
    spec = _LOCAL_KIND_SPEC[kind]
    matrix = _load_track_matrix(track, spec["matrix_kind"])
    if matrix is None:
        return {"unit_label": f"{track}/{kind}", "skip": True, "reason": "matrix absent"}

    ad_ids, ctrl_ids = _split_samples(mapping)
    ad_ids = [s for s in ad_ids if s in matrix.columns]
    ctrl_ids = [s for s in ctrl_ids if s in matrix.columns]

    lfc_key = spec["lfc_key"]
    results = _build_donor_deltas(matrix, ad_ids, ctrl_ids, lfc_key)
    motif_series = matrix["motif"].reset_index(drop=True)
    site_ids = matrix["site_id"].values
    gene_symbols = matrix["gene_symbol"].values
    suffix = config.PHOSPHO_TRACKS[track]["output_suffix"]

    return {
        "unit_label": f"{track}/{kind}",
        "track": track,
        "lfc_key": lfc_key,
        "infix": spec["infix"],
        "suffix": suffix,
        "n_contrasts": len(results),
        "contrast_keys": sorted(results.keys()),
        "motif_series": _series_fingerprint(motif_series, "motif_series"),
        "site_ids": _array_fingerprint(site_ids, "site_ids"),
        "gene_symbols": _array_fingerprint(gene_symbols, "gene_symbols"),
        "contrasts": _contrast_fingerprint(results, lfc_key),
    }


def build_runner_fingerprint(track: str, kind: str, mapping: pd.DataFrame) -> dict:
    """Build the runner's fingerprint for the same unit."""
    adapter = MukeshMeaAdapter(
        scratch_dir="/tmp/phase3_parity_noop",
        mapping=mapping,
        tracks=[track],
    )
    runner = MeaRunner(kinase_enrich)

    # Find the specific unit.
    target_unit = None
    for unit in adapter.iter_units():
        if unit.kind == kind:
            target_unit = unit
            break
    if target_unit is None:
        return {"unit_label": f"{track}/{kind}", "skip": True, "reason": "unit not found"}

    target_unit = adapter.load_inputs(target_unit)
    if target_unit.motif_series is None:
        return {"unit_label": f"{track}/{kind}", "skip": True, "reason": "matrix absent"}
    target_unit = adapter.build_contrasts(target_unit)
    return runner.capture_fingerprint(target_unit)


def compare_fingerprints(canon: dict, runner_fp: dict) -> tuple[bool, list[str]]:
    """Return (all_match, list_of_failures)."""
    failures = []

    if canon.get("skip") or runner_fp.get("skip"):
        # Both should agree on skip.
        if canon.get("skip") == runner_fp.get("skip"):
            return True, []
        return False, [f"skip disagreement: canon={canon.get('skip')} runner={runner_fp.get('skip')}"]

    # Scalar fields.
    for key in ("track", "lfc_key", "infix", "suffix", "n_contrasts"):
        if canon.get(key) != runner_fp.get(key):
            failures.append(f"{key}: canon={canon.get(key)!r} runner={runner_fp.get(key)!r}")

    # Contrast keys.
    if canon["contrast_keys"] != runner_fp["contrast_keys"]:
        failures.append(
            f"contrast_keys differ: "
            f"canon={len(canon['contrast_keys'])} runner={len(runner_fp['contrast_keys'])}"
        )

    # Series fingerprint (motif_series).
    for fname in ("motif_series",):
        cf = canon[fname]
        rf = runner_fp[fname]
        for k in ("length", "null_count", "sha256_16"):
            if cf.get(k) != rf.get(k):
                failures.append(f"{fname}.{k}: canon={cf.get(k)!r} runner={rf.get(k)!r}")

    # Array fingerprints.
    for fname in ("site_ids", "gene_symbols"):
        cf = canon[fname]
        rf = runner_fp[fname]
        for k in ("shape", "dtype", "nan_count", "sha256_16"):
            if cf.get(k) != rf.get(k):
                failures.append(f"{fname}.{k}: canon={cf.get(k)!r} runner={rf.get(k)!r}")

    # Contrast arrays.
    cf_contrasts = canon.get("contrasts", {})
    rf_contrasts = runner_fp.get("contrasts", {})
    for cname in sorted(cf_contrasts.keys()):
        cf_a = cf_contrasts.get(cname, {})
        rf_a = rf_contrasts.get(cname, {})
        for k in ("shape", "dtype", "nan_count", "sha256_16"):
            if cf_a.get(k) != rf_a.get(k):
                failures.append(
                    f"contrasts[{cname!r}].{k}: "
                    f"canon={cf_a.get(k)!r} runner={rf_a.get(k)!r}"
                )

    return len(failures) == 0, failures


def run_input_equivalence(mapping: pd.DataFrame) -> bool:
    """Run input-equivalence for all 4 units. Returns True if all pass."""
    units = [("st", "stoich"), ("st", "raw"), ("py", "stoich"), ("py", "raw")]
    all_pass = True

    print("\n=== (A) Input-equivalence ===")
    print(f"{'Unit':<14} {'Status':<8}  Details")
    print("-" * 70)

    for track, kind in units:
        canon = build_canonical_fingerprint(track, kind, mapping)
        runner_fp = build_runner_fingerprint(track, kind, mapping)

        ok, failures = compare_fingerprints(canon, runner_fp)
        status = "PASS" if ok else "FAIL"
        if not ok:
            all_pass = False
        print(f"  {track}/{kind:<10} {status:<8}  {'; '.join(failures) if failures else 'all fields match'}")

    return all_pass


# ---------------------------------------------------------------------------
# (B) End-to-end spot-check (py/stoich)
# ---------------------------------------------------------------------------

def run_e2e_spotcheck(mapping: pd.DataFrame, scratch_dir: str) -> None:
    """Run the py/stoich unit through the runner and diff against canonical."""
    print("\n=== (B) End-to-end spot-check: py/stoich ===")

    adapter = MukeshMeaAdapter(
        scratch_dir=scratch_dir,
        mapping=mapping,
        tracks=["py"],
    )
    runner = MeaRunner(kinase_enrich)

    results = runner.run_all(adapter)
    if not results:
        print("  ERROR: runner produced no results for py/stoich — check skips above.")
        return

    # Find the stoich unit result.
    stoich_result = next((r for r in results if r.unit.kind == "stoich"), None)
    if stoich_result is None:
        print("  ERROR: py/stoich unit did not complete.")
        return

    unit = stoich_result.unit
    out_dir = str(unit.out_dir)

    # Files to diff.
    checks = [
        (
            "mea_perdonor_pY.csv",
            os.path.join(out_dir, "mea_perdonor_pY.csv"),
            os.path.join(PERDONOR_DIR, "mea_perdonor_pY.csv"),
        ),
        (
            "kinase_donor_nes_pY.csv",
            os.path.join(out_dir, "kinase_donor_nes_pY.csv"),
            os.path.join(PERDONOR_DIR, "kinase_donor_nes_pY.csv"),
        ),
        (
            "recurrence_pY.csv",
            os.path.join(out_dir, "recurrence_pY.csv"),
            os.path.join(PERDONOR_DIR, "recurrence_pY.csv"),
        ),
    ]

    print(f"\n  {'File':<35} {'Rows(new)':<12} {'Rows(canon)':<12} {'Cols match':<12} {'Worst |Δ|'}")
    print("  " + "-" * 90)
    for fname, new_path, canon_path in checks:
        if not os.path.exists(canon_path):
            print(f"  {fname:<35} CANONICAL MISSING")
            continue
        if not os.path.exists(new_path):
            print(f"  {fname:<35} RUNNER OUTPUT MISSING")
            continue

        new_df = pd.read_csv(new_path)
        canon_df = pd.read_csv(canon_path)

        rows_new = len(new_df)
        rows_canon = len(canon_df)
        cols_match = sorted(new_df.columns.tolist()) == sorted(canon_df.columns.tolist())

        # Worst |Δ| on numeric columns (after aligning by same sort key).
        worst_delta = float("nan")
        try:
            # Sort both by all columns to align rows.
            sort_cols = [c for c in new_df.columns if new_df[c].dtype == object][:2]
            if sort_cols:
                new_sorted = new_df.sort_values(sort_cols).reset_index(drop=True)
                canon_sorted = canon_df.sort_values(sort_cols).reset_index(drop=True)
            else:
                new_sorted = new_df.reset_index(drop=True)
                canon_sorted = canon_df.reset_index(drop=True)

            num_cols = new_sorted.select_dtypes(include=[np.number]).columns
            if len(num_cols) and len(new_sorted) == len(canon_sorted):
                deltas = (
                    new_sorted[num_cols].to_numpy(dtype=float)
                    - canon_sorted[num_cols].to_numpy(dtype=float)
                )
                worst_delta = float(np.nanmax(np.abs(deltas)))
        except Exception as exc:
            worst_delta = float("nan")

        print(
            f"  {fname:<35} {rows_new:<12} {rows_canon:<12} "
            f"{'YES' if cols_match else 'NO':<12} {worst_delta:.2e}"
        )

    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scratch-dir",
        required=True,
        metavar="DIR",
        help="Directory for runner-driven scratch outputs.",
    )
    parser.add_argument(
        "--run-e2e",
        action="store_true",
        default=False,
        help="Also run the end-to-end spot-check (py/stoich; takes ~2 min).",
    )
    args = parser.parse_args(argv)

    if not os.path.exists(SAMPLE_MAPPING_CSV):
        print(f"ERROR: {SAMPLE_MAPPING_CSV} not found.")
        return 2

    mapping = pd.read_csv(SAMPLE_MAPPING_CSV)

    all_pass = run_input_equivalence(mapping)

    if not all_pass:
        print("\nINPUT-EQUIVALENCE FAILED — stopping before end-to-end check.")
        return 1

    print("\nInput-equivalence: ALL PASS")

    if args.run_e2e:
        run_e2e_spotcheck(mapping, args.scratch_dir)
    else:
        print("(End-to-end spot-check skipped — pass --run-e2e to enable.)")

    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Export T-cell kinases and substrates that are monotonically changing across timepoints.

Kinase side (NES): donor1 only (donor2 has no IMAC and no pY MEA). Implicit Day-2
baseline NES = 0 is prepended before testing.

Substrate side (raw_phospho_normalized or stoichiometry_matrix): all available
(donor1 ST+pY, donor2 pY). Day-2 baseline column included in the file.

Usage:
    pixi run python -m alz.cohorts.tcells.monotonic_export
    pixi run python -m alz.cohorts.tcells.monotonic_export --track-kind stoich --strict
    pixi run python -m alz.cohorts.tcells.monotonic_export --out-dir /tmp/mono_test/
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

_PROJECT_ROOT = str(Path(__file__).resolve().parents[3])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from alz.shared import config
from alz.cohorts.tcells.ingest import KINASE_DIR


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _day_order(cols: list[str]) -> list[tuple[int, str]]:
    """Return [(day_int, col_name), ...] sorted ascending by day number.

    Columns are expected to match D{n}_d{day} where {day} is the day number.
    """
    parsed = []
    for col in cols:
        m = re.search(r"_d(\d+)$", col)
        if m:
            parsed.append((int(m.group(1)), col))
    return sorted(parsed, key=lambda x: x[0])


def _is_monotone(v: np.ndarray, strict: bool, include_constant: bool) -> str | None:
    """Return 'increasing', 'decreasing', or None.

    v should already be NaN-filtered and in chronological order.
    Excludes constants unless include_constant is True.
    """
    if len(v) < 2:
        return None

    if not include_constant and (np.max(v) - np.min(v)) == 0:
        return None

    diffs = np.diff(v)
    if strict:
        if np.all(diffs > 0):
            return "increasing"
        if np.all(diffs < 0):
            return "decreasing"
    else:
        if np.all(diffs >= 0):
            return "increasing"
        if np.all(diffs <= 0):
            return "decreasing"
    return None


def _ols_slope(days: np.ndarray, values: np.ndarray) -> float:
    """Return OLS slope of values vs day numbers."""
    if len(days) < 2:
        return float("nan")
    slope, _, _, _, _ = stats.linregress(days, values)
    return float(slope)


# ---------------------------------------------------------------------------
# Kinase exporter
# ---------------------------------------------------------------------------

def export_kinases(
    out_dir: Path,
    strict: bool,
    include_constant: bool,
) -> pd.DataFrame:
    nes_files = {
        ("donor1", "st"): os.path.join(KINASE_DIR, "donor1", "mea", "kinase_timepoint_nes.csv"),
        ("donor1", "py"): os.path.join(KINASE_DIR, "donor1", "mea", "kinase_timepoint_nes_pY.csv"),
    }

    rows = []
    summaries = []

    for (donor, track), path in nes_files.items():
        if not os.path.exists(path):
            print(f"[kinases] {donor}/{track}: file not found, skipping — {path}")
            continue

        df = pd.read_csv(path)
        day_cols = _day_order([c for c in df.columns if c != "kinase"])
        days_post = [d for d, _ in day_cols]
        cols_post = [c for _, c in day_cols]

        # Full day sequence including implicit baseline
        all_days = np.array([2] + days_post, dtype=float)

        n_inc = 0
        n_dec = 0

        for _, row in df.iterrows():
            v = np.array([0.0] + [float(row[c]) for c in cols_post])
            # No NaNs expected in NES; still handle gracefully
            mask = ~np.isnan(v)
            v_clean = v[mask]
            d_clean = all_days[mask]

            direction = _is_monotone(v_clean, strict, include_constant)
            if direction is None:
                continue

            if direction == "increasing":
                n_inc += 1
            else:
                n_dec += 1

            out_row: dict = {
                "donor": donor,
                "track": track,
                "kinase": row["kinase"],
                "direction": direction,
                "n_timepoints": len(v_clean),
                "value_d2": 0.0,
            }
            for day, col in zip(days_post, cols_post):
                out_row[f"value_d{day}"] = float(row[col])
            out_row["range"] = float(np.max(v_clean) - np.min(v_clean))
            out_row["slope_ols"] = _ols_slope(d_clean, v_clean)
            rows.append(out_row)

        total = len(df)
        summaries.append((donor, track, total, n_inc, n_dec))
        print(f"[kinases] {donor}/{track}: {total} kinases → {n_inc} increasing, {n_dec} decreasing")

    result = pd.DataFrame(rows)
    if not result.empty:
        result.to_csv(out_dir / "monotonic_kinases.csv", index=False)
    else:
        # Write empty file with header
        pd.DataFrame(columns=[
            "donor", "track", "kinase", "direction", "n_timepoints",
            "value_d2", "range", "slope_ols"
        ]).to_csv(out_dir / "monotonic_kinases.csv", index=False)

    return result


# ---------------------------------------------------------------------------
# Substrate exporter
# ---------------------------------------------------------------------------

def export_substrates(
    out_dir: Path,
    track_kind: str,
    strict: bool,
    min_obs: int,
    include_constant: bool,
) -> pd.DataFrame:
    prefix = "raw_phospho_normalized" if track_kind == "raw" else "stoichiometry_matrix"

    substrate_files: dict[tuple[str, str], str] = {
        ("donor1", "st"): os.path.join(KINASE_DIR, "donor1", f"{prefix}.csv"),
        ("donor1", "py"): os.path.join(KINASE_DIR, "donor1", f"{prefix}_pY.csv"),
        ("donor2", "py"): os.path.join(KINASE_DIR, "donor2", f"{prefix}_pY.csv"),
    }

    meta_cols = ["site_id", "protein_id", "gene_symbol", "site_position", "motif"]
    rows = []

    for (donor, track), path in substrate_files.items():
        if not os.path.exists(path):
            print(f"[substrates] {donor}/{track} {track_kind}: file not found, skipping — {path}")
            continue

        df = pd.read_csv(path)

        # Identify data columns (exclude meta cols that are present)
        present_meta = [c for c in meta_cols if c in df.columns]
        data_cols_raw = [c for c in df.columns if c not in present_meta]
        day_cols = _day_order(data_cols_raw)
        days = np.array([d for d, _ in day_cols], dtype=float)
        cols = [c for _, c in day_cols]

        n_inc = 0
        n_dec = 0
        n_excluded_nan = 0

        for _, row in df.iterrows():
            v = np.array([row[c] for c in cols], dtype=float)
            mask = ~np.isnan(v)
            if mask.sum() < min_obs:
                n_excluded_nan += 1
                continue

            v_clean = v[mask]
            d_clean = days[mask]

            direction = _is_monotone(v_clean, strict, include_constant)
            if direction is None:
                continue

            if direction == "increasing":
                n_inc += 1
            else:
                n_dec += 1

            out_row: dict = {
                "donor": donor,
                "track": track,
                "kind": track_kind,
                "site_id": row.get("site_id", ""),
                "gene_symbol": row.get("gene_symbol", ""),
                "site_position": row.get("site_position", ""),
                "motif": row.get("motif", ""),
                "direction": direction,
                "n_obs": int(mask.sum()),
            }
            for day, col in zip([d for d, _ in day_cols], cols):
                out_row[f"value_d{int(day)}"] = float(row[col]) if not np.isnan(row[col]) else float("nan")
            out_row["range"] = float(np.max(v_clean) - np.min(v_clean))
            out_row["slope_ols"] = _ols_slope(d_clean, v_clean)
            rows.append(out_row)

        total = len(df)
        print(
            f"[substrates] {donor}/{track} {track_kind}: {total} sites → "
            f"{n_inc} increasing, {n_dec} decreasing "
            f"(NaN-excluded sites: {n_excluded_nan})"
        )

    result = pd.DataFrame(rows)
    if not result.empty:
        result.to_csv(out_dir / "monotonic_substrates.csv", index=False)
    else:
        pd.DataFrame(columns=[
            "donor", "track", "kind", "site_id", "gene_symbol", "site_position",
            "motif", "direction", "n_obs", "range", "slope_ols"
        ]).to_csv(out_dir / "monotonic_substrates.csv", index=False)

    return result


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify_outputs(kinases: pd.DataFrame, substrates: pd.DataFrame, strict: bool, min_obs: int) -> None:
    """Run sanity assertions on output dataframes."""
    valid_directions = {"increasing", "decreasing"}

    # Kinase assertions
    if not kinases.empty:
        assert set(kinases["direction"]).issubset(valid_directions), \
            "kinase direction column has invalid values"

        # Verify monotonicity for each row
        val_cols_k = [c for c in kinases.columns if c.startswith("value_d")]
        for _, row in kinases.iterrows():
            v = np.array([float(row[c]) for c in val_cols_k], dtype=float)
            v_clean = v[~np.isnan(v)]
            diffs = np.diff(v_clean)
            if row["direction"] == "increasing":
                if strict:
                    assert np.all(diffs > 0), f"kinase {row['kinase']} not strictly increasing"
                else:
                    assert np.all(diffs >= 0), f"kinase {row['kinase']} not weakly increasing"
            else:
                if strict:
                    assert np.all(diffs < 0), f"kinase {row['kinase']} not strictly decreasing"
                else:
                    assert np.all(diffs <= 0), f"kinase {row['kinase']} not weakly decreasing"
            assert row["range"] > 0, f"kinase {row['kinase']} has zero range"

    # Substrate assertions
    if not substrates.empty:
        assert set(substrates["direction"]).issubset(valid_directions), \
            "substrate direction column has invalid values"
        assert (substrates["n_obs"] >= min_obs).all(), \
            f"substrate rows with n_obs < {min_obs} found"
        assert (substrates["range"] > 0).all(), \
            "substrate rows with zero range found"

    print("[verify] all assertions passed")


# ---------------------------------------------------------------------------
# Spot-check printer
# ---------------------------------------------------------------------------

def _print_spot_check(kinases: pd.DataFrame) -> None:
    if kinases.empty:
        print("[spot-check] no kinase rows to display")
        return

    val_cols = sorted(
        [c for c in kinases.columns if c.startswith("value_d")],
        key=lambda c: int(re.search(r"d(\d+)$", c).group(1))
    )

    inc = kinases[kinases["direction"] == "increasing"].nlargest(3, "slope_ols")
    dec = kinases[kinases["direction"] == "decreasing"].nsmallest(3, "slope_ols")

    print("\n[spot-check] top-3 increasing kinases (by slope_ols):")
    for _, row in inc.iterrows():
        vals = ", ".join(f"{c}={row[c]:.3f}" for c in val_cols if c in row)
        print(f"  {row['track']:2s} | {row['kinase']:12s} | slope={row['slope_ols']:.4f} | {vals}")

    print("[spot-check] top-3 decreasing kinases (by slope_ols):")
    for _, row in dec.iterrows():
        vals = ", ".join(f"{c}={row[c]:.3f}" for c in val_cols if c in row)
        print(f"  {row['track']:2s} | {row['kinase']:12s} | slope={row['slope_ols']:.4f} | {vals}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export monotonically changing T-cell kinases and substrates across timepoints."
    )
    parser.add_argument(
        "--track-kind",
        choices=["raw", "stoich"],
        default="raw",
        help="Substrate preprocessing to use (default: raw = raw_phospho_normalized).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        default=False,
        help="Use strict inequalities (default: weak ≥/≤).",
    )
    parser.add_argument(
        "--min-obs",
        type=int,
        default=3,
        metavar="INT",
        help="Minimum non-NaN timepoints per substrate site (default: 3).",
    )
    parser.add_argument(
        "--include-constant",
        action="store_true",
        default=False,
        help="Include flat (zero-range) series that trivially pass (default: excluded).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(KINASE_DIR) / "monotonic",
        help="Output directory (default: outputs/reports/kinase_attribution_tcells/monotonic/).",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        default=False,
        help="Run sanity assertions on the output after writing.",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    kinases = export_kinases(args.out_dir, args.strict, args.include_constant)
    substrates = export_substrates(
        args.out_dir, args.track_kind, args.strict, args.min_obs, args.include_constant
    )

    print(f"\nwrote monotonic_kinases.csv ({len(kinases)} rows) → {args.out_dir}/")
    print(f"wrote monotonic_substrates.csv ({len(substrates)} rows)")

    _print_spot_check(kinases)

    if args.verify:
        verify_outputs(kinases, substrates, args.strict, args.min_obs)


if __name__ == "__main__":
    main()

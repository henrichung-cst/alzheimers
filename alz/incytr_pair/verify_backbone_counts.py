"""Verify backbone output distinct-spine counts and floor invariants.

Checks:
1. Each grain's output parquets exist and are non-empty.
2. Per-grain distinct (Sender.group, Receiver.group, spine_key) counts match
   the plan's expected values, within a ±20% tolerance:
     R-EM   : ~19,680  distinct (sender, receiver, Receptor, EM) after floor
     L-R-EM : ~27,927  distinct (sender, receiver, Ligand, Receptor, EM) after floor
   These are cross-contrast UNION counts (each unique backbone counted once
   regardless of how many contrasts it appears in).  Derived from the gated
   wide/ population (SigProb>0.1 OR, |PDS|>=0.2).  Re-scoring with backbone
   SigProb may shift these slightly, so ±20% tolerance is applied.
3. All rows satisfy the canonical floor: SigProb > 0.1 (either) AND |PDS| >= 0.2.
4. No row has |PDS| = 0.
5. Schema completeness: TPDS, PPDS, PhPDS_ps, PhPDS_py, PDS columns present.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import duckdb

SIGPROB_CUTOFF = 0.1
ABS_PDS_CUTOFF = 0.2

# Expected CROSS-CONTRAST distinct (sender, receiver, spine) counts from the
# plan (union over all 9 contrasts, after the floor).  Values from the gated
# wide/ population; backbone re-scoring may shift by ±20%.
EXPECTED_COUNTS = {
    "R-EM":   (15_700, 23_600),   # plan ~19,680 distinct (s, r, R, EM) cross-contrast
    "L-R-EM": (22_300, 33_500),   # plan ~27,927 distinct (s, r, L, R, EM) cross-contrast
}

GRAIN_KEYS = {
    "R-EM":   ("Sender.group", "Receiver.group", "Receptor", "EM"),
    "L-R-EM": ("Sender.group", "Receiver.group", "Ligand", "Receptor", "EM"),
    "R-EM-T": ("Sender.group", "Receiver.group", "Receptor", "EM", "Target"),
}

REQUIRED_SCORE_COLS = {"TPDS", "PPDS", "PhPDS_ps", "PhPDS_py", "PDS"}


def verify(backbone_dir: str, grains: list[str]) -> bool:
    spill = os.environ.get(
        "DUCKDB_TEMP_DIR", os.path.join(os.path.expanduser("~"), ".cache", "duckdb")
    )
    os.makedirs(spill, exist_ok=True)
    con = duckdb.connect()
    con.execute("PRAGMA memory_limit='12GB'")
    con.execute(f"SET temp_directory='{spill}'")

    all_ok = True

    def fail(msg: str) -> None:
        nonlocal all_ok
        all_ok = False
        print(f"  FAIL: {msg}", flush=True)

    def ok(msg: str) -> None:
        print(f"  OK:   {msg}", flush=True)

    for grain in grains:
        grain_dir = os.path.join(backbone_dir, grain)
        parquets = sorted(Path(grain_dir).glob("*_backbone_output.parquet")) if os.path.isdir(grain_dir) else []
        print(f"\n[verify] grain={grain}  parquets={len(parquets)}", flush=True)

        if len(parquets) == 0:
            fail(f"no backbone parquets under {grain_dir}/")
            continue

        # Schema check (first file)
        first = str(parquets[0])
        cols_in_file = {
            r[0] for r in con.execute(
                f"DESCRIBE SELECT * FROM read_parquet('{first}') LIMIT 0"
            ).fetchall()
        }
        missing_score = REQUIRED_SCORE_COLS - cols_in_file
        if missing_score:
            fail(f"missing score columns: {missing_score}")
        else:
            ok(f"score columns present: {REQUIRED_SCORE_COLS}")

        # Detect SigProb columns
        sp_cols = sorted(c for c in cols_in_file if c.startswith("SigProb_")
                         and not c.endswith("log2FC") and not c.endswith("aFC"))
        if len(sp_cols) < 2:
            fail(f"expected >=2 SigProb_<cond> columns, found: {sp_cols}")
            continue
        sp1, sp2 = f'"{sp_cols[0]}"', f'"{sp_cols[1]}"'

        # Union all contrast parquets for this grain
        glob_pat = str(Path(grain_dir) / "*_backbone_output.parquet")

        # 1. Floor invariant: every row passes the canonical gate
        n_violate = con.execute(f"""
            SELECT COUNT(*) FROM read_parquet('{glob_pat}', union_by_name=true)
            WHERE NOT (
              ({sp1} > {SIGPROB_CUTOFF} OR {sp2} > {SIGPROB_CUTOFF})
              AND ABS(PDS) >= {ABS_PDS_CUTOFF}
            )
        """).fetchone()[0]
        if n_violate > 0:
            fail(f"floor violation: {n_violate:,} rows below canonical gate")
        else:
            ok(f"floor invariant: 0 violations")

        # 2. No PDS = 0
        n_zero = con.execute(f"""
            SELECT COUNT(*) FROM read_parquet('{glob_pat}', union_by_name=true)
            WHERE PDS = 0
        """).fetchone()[0]
        if n_zero > 0:
            fail(f"{n_zero:,} rows with PDS=0")
        else:
            ok("no PDS=0 rows")

        # 3. Total rows across all contrasts
        n_total = con.execute(f"""
            SELECT COUNT(*) FROM read_parquet('{glob_pat}', union_by_name=true)
        """).fetchone()[0]
        ok(f"total rows (all contrasts): {n_total:,}")

        # 4. Distinct (sender, receiver, spine) cross-contrast
        key_cols = GRAIN_KEYS[grain]
        key_expr = ", ".join(f'"{c}"' for c in key_cols)
        n_distinct = con.execute(f"""
            SELECT COUNT(*) FROM (
              SELECT DISTINCT {key_expr}
              FROM read_parquet('{glob_pat}', union_by_name=true)
            )
        """).fetchone()[0]
        msg = f"distinct ({', '.join(key_cols)}) across all contrasts: {n_distinct:,}"
        if grain in EXPECTED_COUNTS:
            lo, hi = EXPECTED_COUNTS[grain]
            if lo <= n_distinct <= hi:
                ok(f"{msg}  [expected {lo:,}–{hi:,}]")
            else:
                fail(f"{msg}  [expected {lo:,}–{hi:,}]")
        else:
            ok(msg)

        # 5. Per-contrast row counts
        for pq in parquets:
            n_pq = con.execute(
                f"SELECT COUNT(*) FROM read_parquet('{pq}')"
            ).fetchone()[0]
            ok(f"  {pq.name}: {n_pq:,} rows")

    con.close()
    return all_ok


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backbone-dir",
                    default="outputs/reports/incytr_pair_mode/backbone",
                    help="Root directory containing grain subdirs")
    ap.add_argument("--grains", default="R-EM,L-R-EM",
                    help="Comma-separated grains to verify (default: R-EM,L-R-EM)")
    args = ap.parse_args()

    grains = [g.strip() for g in args.grains.split(",") if g.strip()]
    invalid = set(grains) - set(GRAIN_KEYS)
    if invalid:
        sys.exit(f"Unknown grains: {invalid}")

    print(f"Verifying backbone outputs: {grains}", flush=True)
    ok = verify(args.backbone_dir, grains)
    if ok:
        print("\nAll checks passed.", flush=True)
    else:
        print("\nSome checks FAILED.", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

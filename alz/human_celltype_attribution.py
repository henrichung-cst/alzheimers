#!/usr/bin/env python3
"""Human Cell-Type Attribution: top-N specific cell types per kinase.

Derives a ranked "top-N specific cell types" list for each kinase from each
human reference (SEA-AD MTG and Allen HBCA).  Reads the specificity matrices
produced by ``human_reference_expression.py`` and emits a long-form CSV
suitable for the viewer payload.

Output:
  outputs/reports/kinase_attribution_human/celltype_specificity.csv
  Columns: kinase, reference, celltype, specificity_score, rank

Usage:
    python alz/human_celltype_attribution.py          # compute and save
    python alz/human_celltype_attribution.py --summary # print cached results
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

import config

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OUTPUT_DIR = config.HUMAN_CELLTYPE_ATTRIBUTION_OUTPUT_DIR
OUT_FILE = config.CELLTYPE_SPECIFICITY_FILE
TOP_N = config.HUMAN_CELLTYPE_TOP_N

SEAAD_SPEC_FILE = config.SEAAD_KINASE_SPECIFICITY_FILE
HBCA_SPEC_FILE = config.HBCA_KINASE_SPECIFICITY_FILE


# ---------------------------------------------------------------------------
# Core ranking
# ---------------------------------------------------------------------------


def _top_n_for_reference(
    spec_df: pd.DataFrame,
    reference: str,
    top_n: int = TOP_N,
) -> pd.DataFrame:
    """Produce long-form top-N rows for a single reference.

    Parameters
    ----------
    spec_df : pd.DataFrame
        Shape kinase_id × celltype, values = log2(ct_mean / brain_mean).
    reference : str
        Reference label (e.g. "seaad_mtg" or "allen_hbca").
    top_n : int
        Number of top cell types per kinase to retain.

    Returns
    -------
    pd.DataFrame
        Long-form with columns: kinase, reference, celltype, specificity_score, rank.
    """
    rows = []
    for kinase in spec_df.index:
        scores = spec_df.loc[kinase]
        # Sort descending by score; rank is 1-based.
        ranked = scores.sort_values(ascending=False)
        for rank, (celltype, score) in enumerate(ranked.iloc[:top_n].items(), start=1):
            rows.append({
                "kinase": kinase,
                "reference": reference,
                "celltype": celltype,
                "specificity_score": round(float(score), 6) if np.isfinite(score) else 0.0,
                "rank": rank,
            })
    return pd.DataFrame(rows, columns=["kinase", "reference", "celltype",
                                       "specificity_score", "rank"])


# ---------------------------------------------------------------------------
# Main computation
# ---------------------------------------------------------------------------


def compute_human_celltype_attribution(force: bool = False) -> pd.DataFrame:
    """Compute top-N specific cell types per kinase from human references.

    Reads seaad_kinase_specificity.csv and hbca_kinase_specificity.csv,
    ranks cell types within each kinase for each reference, and concatenates
    into a single long-form output at CELLTYPE_SPECIFICITY_FILE.

    Missing reference files are skipped with a warning (allows partial runs
    when only one reference is available).

    Returns the combined DataFrame.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("Human Cell-Type Attribution: top-N specific cell types")
    print("=" * 60)

    if not force and os.path.exists(OUT_FILE):
        print(f"  Cached: {OUT_FILE} (use --force to recompute)")
        return pd.read_csv(OUT_FILE)

    all_parts: list[pd.DataFrame] = []

    for ref_label, spec_path in [
        ("seaad_mtg", SEAAD_SPEC_FILE),
        ("allen_hbca", HBCA_SPEC_FILE),
    ]:
        if not os.path.exists(spec_path):
            print(f"  WARNING: {ref_label} specificity file not found at {spec_path} — skipping")
            print(f"  Run: python alz/human_reference_expression.py --ref "
                  f"{'seaad' if ref_label == 'seaad_mtg' else 'hbca'}")
            continue

        print(f"\n  Processing {ref_label} ...")
        spec_df = pd.read_csv(spec_path, index_col=0)
        print(f"    Shape: {spec_df.shape} (kinases × celltypes)")

        part = _top_n_for_reference(spec_df, ref_label, TOP_N)
        all_parts.append(part)
        print(f"    Top-{TOP_N} rows: {len(part)}")

    if not all_parts:
        raise RuntimeError(
            "No specificity files found for either reference. "
            "Run atlas_reference.py and human_reference_expression.py first."
        )

    df = pd.concat(all_parts, ignore_index=True)
    df.to_csv(OUT_FILE, index=False)
    print(f"\n  Saved {len(df)} rows to {OUT_FILE}")

    # Summary: top cell type per kinase per reference.
    for ref in df["reference"].unique():
        sub = df[df["reference"] == ref]
        top1 = sub[sub["rank"] == 1].head(5)
        print(f"\n  [{ref}] Top-1 cell type for first 5 kinases:")
        for _, row in top1.iterrows():
            print(f"    {row['kinase']}: {row['celltype']} (score={row['specificity_score']:.3f})")

    return df


# ---------------------------------------------------------------------------
# Payload builder helper
# ---------------------------------------------------------------------------


def build_celltype_specificity_payload(
    top_n: int = TOP_N,
) -> dict | None:
    """Build the PAYLOAD.human.celltype_specificity block for the viewer.

    Returns None if neither specificity file exists (phase-2 data absent).

    Schema:
      {
        "references": ["seaad_mtg", "allen_hbca"],
        "seaad_mtg": {
          "celltypes": [...],
          "by_kinase": { kinase_id → [score per celltype] }
        },
        "allen_hbca": { ... }
      }
    """
    ref_map = {
        "seaad_mtg": SEAAD_SPEC_FILE,
        "allen_hbca": HBCA_SPEC_FILE,
    }

    available = {ref: path for ref, path in ref_map.items() if os.path.exists(path)}
    if not available:
        return None

    payload: dict = {"references": list(available.keys())}

    for ref, path in available.items():
        spec_df = pd.read_csv(path, index_col=0)
        celltypes = list(spec_df.columns)
        by_kinase: dict[str, list] = {}
        for kinase in spec_df.index:
            scores = spec_df.loc[kinase].tolist()
            by_kinase[kinase] = [
                round(float(v), 4) if np.isfinite(v) else 0.0
                for v in scores
            ]

        # Precompute top-N convenience list per kinase.
        top_n_by_kinase: dict[str, list[dict]] = {}
        for kinase in spec_df.index:
            scores = spec_df.loc[kinase]
            ranked = scores.sort_values(ascending=False)
            top_n_by_kinase[kinase] = [
                {
                    "celltype": ct,
                    "score": round(float(sc), 4) if np.isfinite(sc) else 0.0,
                    "rank": i + 1,
                }
                for i, (ct, sc) in enumerate(ranked.iloc[:top_n].items())
            ]

        payload[ref] = {
            "celltypes": celltypes,
            "by_kinase": by_kinase,
            "top_n_by_kinase": top_n_by_kinase,
        }

    return payload


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def print_summary() -> None:
    """Print cached attribution results."""
    if not os.path.exists(OUT_FILE):
        print(f"  No cached results at {OUT_FILE}")
        return
    df = pd.read_csv(OUT_FILE)
    print(f"  Celltype specificity: {len(df)} rows")
    for ref in df["reference"].unique():
        sub = df[df["reference"] == ref]
        n_kinases = sub["kinase"].nunique()
        n_ct = sub["celltype"].nunique()
        print(f"    [{ref}] {n_kinases} kinases, {n_ct} cell types")
        # Show top 3 rows for the first kinase.
        first_kinase = sub["kinase"].iloc[0]
        top3 = sub[sub["kinase"] == first_kinase].head(3)
        for _, row in top3.iterrows():
            print(f"      rank={row['rank']} {row['celltype']} "
                  f"(score={row['specificity_score']:.3f})")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Human Cell-Type Attribution: top-N specific cell types per kinase",
    )
    parser.add_argument("--summary", action="store_true",
                        help="Print cached results")
    parser.add_argument("--force", action="store_true",
                        help="Force recomputation even if cached results exist")
    args = parser.parse_args()

    if args.summary:
        print_summary()
    else:
        compute_human_celltype_attribution(force=args.force)


if __name__ == "__main__":
    main()

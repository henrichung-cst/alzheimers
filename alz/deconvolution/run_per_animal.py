#!/usr/bin/env python3
"""Per-animal extension orchestrator.

Multiplies the per-(group, WMB-class, site) decomposition produced by
``build_wmb_decomposition.py`` by per-animal bulk phospho intensity, runs
factorial OLS + MEA + snRNA cross-check + per-row evidence join on the
WMB-class spine.

Outputs land in `outputs/reports/deconvolution/per_animal/`.

Usage:
    python alz/deconvolution/run_per_animal.py --run \
        --cell-types "30 Astro-Epen" "31 OPC-Oligo" \
        --tracks st py --permutations 200
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.dirname(HERE)
REPO_ROOT = os.path.dirname(CODE_DIR)
sys.path.insert(0, CODE_DIR)
sys.path.insert(0, REPO_ROOT)

import pandas as pd

import config
from deconvolution import paths
from deconvolution.load_deconvoluted import (
    load_track, load_wmb_class_sizes,
)
from deconvolution.per_animal_extension import (
    compute_site_fractions, run_per_animal_track, select_male_animals,
)
from deconvolution.mea_per_celltype import run_mea
from deconvolution.snrna_concordance import annotate
from deconvolution.cohort_concordance import (
    compute_cohort_concordance, expression_presence,
)
from deconvolution.confidence import attach_evidence


PA_OUTPUT_DIR = os.path.join(paths.OUTPUT_DIR, "per_animal")
PA_SITE_OLS_FILE = os.path.join(PA_OUTPUT_DIR, "site_level_ols.parquet")
PA_MEA_FILE = os.path.join(PA_OUTPUT_DIR, "kinase_enrichment_raw.csv")
PA_PRIMARY_TABLE = os.path.join(PA_OUTPUT_DIR, "kinase_enrichment_wmb.csv")
PA_COHORT_FILE = os.path.join(PA_OUTPUT_DIR, "cohort_concordance.csv")
PA_GROUP_CLASS_COUNTS_FILE = os.path.join(PA_OUTPUT_DIR, "group_class_counts.csv")
PA_SUMMARY_JSON = os.path.join(PA_OUTPUT_DIR, "summary.json")

RAW_PHOSPHO_BY_TRACK = {
    "st": os.path.join(REPO_ROOT, "outputs", "reports", "kinase_attribution",
                       "raw_phospho_normalized.csv"),
    "py": os.path.join(REPO_ROOT, "outputs", "reports", "kinase_attribution",
                       "raw_phospho_normalized_pY.csv"),
}
SAMPLE_MAPPING_FILE = os.path.join(
    REPO_ROOT, "outputs", "reports", "data_ingest", "sample_mapping.csv",
)
SAMPLE_EXCLUSIONS_FILE = os.path.join(
    REPO_ROOT, "outputs", "reports", "data_ingest", "sample_exclusions.csv",
)


def _print_header(msg: str) -> None:
    print("=" * 72)
    print(msg)
    print("=" * 72)


def main():
    parser = argparse.ArgumentParser(
        description="Per-animal extension of Yuyu's deconvolution → MEA",
    )
    parser.add_argument("--run", action="store_true",
                        help="Run all stages (1-5) end-to-end")
    parser.add_argument("--cell-types", nargs="*", default=None,
                        help="Subset of WMB-class names (default: all 27)")
    parser.add_argument("--tracks", nargs="*", default=["st", "py"],
                        choices=["st", "py"])
    parser.add_argument("--permutations", type=int, default=None,
                        help="Override MEA permutations")
    parser.add_argument("--ols-only", action="store_true",
                        help="Stop after Stage 2; useful for variance audits")
    parser.add_argument("--relabel-only", action="store_true",
                        help="Re-run Stages 4–5 only using existing "
                             "kinase_enrichment_raw.csv (cohort concordance + "
                             "presence + relabel)")
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()

    if args.summary:
        for name, path in [
            ("OLS site-level", PA_SITE_OLS_FILE),
            ("MEA raw", PA_MEA_FILE),
            ("Primary WMB-class table", PA_PRIMARY_TABLE),
        ]:
            if os.path.exists(path):
                df = pd.read_parquet(path) if path.endswith(".parquet") \
                    else pd.read_csv(path)
                print(f"  {name}: {path}  ({len(df):,} rows, {len(df.columns)} cols)")
            else:
                print(f"  {name}: NOT FOUND ({path})")
        return

    if not args.run and not args.ols_only and not args.relabel_only:
        parser.print_help()
        sys.exit(1)

    os.makedirs(PA_OUTPUT_DIR, exist_ok=True)
    t0 = time.time()

    if args.relabel_only:
        _print_header("Relabel-only  Stages 4–5 from existing MEA raw")
        group_class_counts = load_wmb_class_sizes()
        if not os.path.exists(PA_MEA_FILE):
            print(f"  ERROR: {PA_MEA_FILE} missing; run --run first.")
            sys.exit(1)
        mea = pd.read_csv(PA_MEA_FILE)
        print(f"  Loaded MEA raw: {len(mea):,} rows")
        _run_stages_4_to_5(mea, group_class_counts, t0, n_animals=None)
        return

    # Stage 1 — load WMB-class decomposition + per-animal bulk + sample mapping
    _print_header("Stage 1  Load WMB-class decomposition + per-animal bulk")
    tracks = {}
    for t in args.tracks:
        print(f"  Loading {t} track …")
        tracks[t] = load_track(t)
        print(f"    {len(tracks[t].meta):,} sites × "
              f"{len(tracks[t].samples)} groups × "
              f"{len(tracks[t].clusters)} WMB classes")

    group_class_counts = load_wmb_class_sizes()
    group_class_counts.to_csv(PA_GROUP_CLASS_COUNTS_FILE)
    print(f"  Group×WMB-class nucleus counts: {group_class_counts.shape} → "
          f"{PA_GROUP_CLASS_COUNTS_FILE}")

    sample_mapping = pd.read_csv(SAMPLE_MAPPING_FILE)
    print(f"  Per-animal sample mapping: {len(sample_mapping)} animals")

    male_mapping = select_male_animals(sample_mapping, SAMPLE_EXCLUSIONS_FILE)
    print(f"  Males after outlier exclusion: {len(male_mapping)}")
    print(f"  Expected OLS dof: {len(male_mapping) - 10}")

    raw_phospho_by_track = {}
    for t in args.tracks:
        rp_path = RAW_PHOSPHO_BY_TRACK[t]
        raw_phospho_by_track[t] = pd.read_csv(rp_path, low_memory=False)
        print(f"  Raw phospho [{t}]: {raw_phospho_by_track[t].shape}  ({rp_path})")

    # Stage 1b — compute site fractions per track on the WMB axis
    _print_header("Stage 1b Compute site-specific WMB-class fractions")
    fracs = {}
    for t, track in tracks.items():
        fracs[t] = compute_site_fractions(track, group_class_counts)
        print(f"    [{t}] frac: {fracs[t].shape}")

    cell_type_subset = args.cell_types

    # Stage 2 — per-animal factorial OLS per WMB-class per track
    _print_header("Stage 2  Per-animal factorial OLS per WMB-class per track")
    ols_frames = []
    for t, track in tracks.items():
        print(f"  [{t}] running per-animal OLS …")
        df = run_per_animal_track(
            track, fracs[t], raw_phospho_by_track[t], male_mapping,
            cell_types=cell_type_subset,
        )
        if not df.empty:
            ols_frames.append(df)
    if not ols_frames:
        print("  No OLS results produced; exiting.")
        sys.exit(1)
    site_ols = pd.concat(ols_frames, ignore_index=True)
    site_ols.to_parquet(PA_SITE_OLS_FILE, index=False)
    print(f"  Saved {PA_SITE_OLS_FILE} ({len(site_ols):,} rows)")

    if args.ols_only:
        print(f"\n--ols-only set; stopping. Elapsed {time.time()-t0:.1f}s")
        return

    # Stage 3 — two-track MEA per cluster per contrast
    _print_header("Stage 3  Two-track kinase MEA per cluster per contrast")
    mea_frames = []
    for t in args.tracks:
        print(f"  [{t}] running MEA …")
        m = run_mea(site_ols, t, permutation_num=args.permutations)
        if not m.empty:
            mea_frames.append(m)
    if not mea_frames:
        print("  No MEA results produced; exiting.")
        sys.exit(1)
    mea = pd.concat(mea_frames, ignore_index=True)
    mea.to_csv(PA_MEA_FILE, index=False)
    print(f"  Saved {PA_MEA_FILE} ({len(mea):,} rows)")

    _run_stages_4_to_5(mea, group_class_counts, t0,
                       n_animals=len(male_mapping))


def _run_stages_4_to_5(mea: pd.DataFrame, group_class_counts: pd.DataFrame,
                       t0: float, n_animals: int | None) -> None:
    # Stage 4 — snRNA cross-check + cohort concordance
    _print_header("Stage 4  snRNA kinase-gene LFC concordance + cohort test")
    annotated = annotate(mea)
    print(f"  Annotated {len(annotated):,} MEA rows; "
          f"{annotated['kinase_gene_LFC_snRNA'].notna().sum():,} have snRNA match")

    cohort_df = compute_cohort_concordance(annotated)
    cohort_df.to_csv(PA_COHORT_FILE, index=False)
    n_strata = len(cohort_df)
    n_concordant = int(cohort_df["cohort_concordant"].sum()) if n_strata else 0
    median_frac = float(cohort_df["frac_match"].median()) if n_strata else float("nan")
    print(f"  Cohort concordance: {n_concordant}/{n_strata} strata pass "
          f"(cohort_fdr<{paths.COHORT_FDR_THRESH}, frac_match>0.5); "
          f"median frac_match={median_frac:.3f}")
    print(f"  Saved {PA_COHORT_FILE}")

    spec = pd.read_csv(config.SONG_EXPRESSION_FILE)
    expr_mask = expression_presence(annotated, spec)
    print(f"  Expression presence: {int(expr_mask.sum()):,}/{len(expr_mask):,} "
          f"rows above floor {paths.EXPR_PRESENCE_FLOOR}")

    # Stage 5 — attach per-row evidence (numeric/boolean only; no label)
    _print_header("Stage 5  Attach per-row evidence columns")
    primary = attach_evidence(annotated, group_class_counts, cohort_df, expr_mask)
    primary_cols = [
        "kinase", "wmb_class", "contrast", "track",
        "NES", "FDR", "kinase_gene_LFC_snRNA", "kinase_gene_FDR_snRNA",
        "direction_match", "cohort_concordant", "expressed",
        "frac_match", "cohort_fdr", "n_cells_min",
    ]
    keep = [c for c in primary_cols if c in primary.columns]
    drop_cols = {"Leading substrates"}
    extras = [c for c in primary.columns if c not in keep and c not in drop_cols]
    primary = primary[keep + extras]

    primary.to_csv(PA_PRIMARY_TABLE, index=False)
    print(f"  Saved {PA_PRIMARY_TABLE} ({len(primary):,} rows)")

    summary = {
        "n_animals_male_after_exclusion": int(n_animals) if n_animals is not None else None,
        "expected_dof": int(n_animals - 10) if n_animals is not None else None,
        "n_wmb_classes_processed": int(primary["wmb_class"].nunique()),
        "n_kinases": int(primary["kinase"].nunique()),
        "n_contrasts": int(primary["contrast"].nunique()),
        "n_tracks": int(primary["track"].nunique()),
        "n_rows_primary": int(len(primary)),
        "deconv_sig_rows": int((primary["FDR"] < paths.DECON_FDR_THRESH).sum()),
        "n_cohort_strata": int(n_strata),
        "n_cohort_concordant": int(n_concordant),
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    with open(PA_SUMMARY_JSON, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Saved {PA_SUMMARY_JSON}")
    print(f"  Total elapsed: {summary['elapsed_seconds']}s")


if __name__ == "__main__":
    main()

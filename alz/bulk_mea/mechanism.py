"""Optional supplementary stage: mechanism annotation.

Re-runs MEA on raw (uncorrected) phospho LFCs and compares the hit-set against
the stoichiometry MEA from Stage 2. Each (kinase, contrast) is classified as
activity_driven / abundance_driven / both / non_significant. Diagnostic for
reviewers — confirms that stoichiometry correction is doing real work.

Reuses Stage 2 helpers via import from alz.bulk_mea.enrich.

Inputs:
  outputs/reports/kinase_attribution/raw_phospho_normalized{,_pY}.csv
  outputs/reports/kinase_attribution/mea_stoichiometry{,_pY}.csv
  outputs/reports/data_ingest/sample_mapping.csv
  outputs/reports/data_ingest/sample_exclusions.csv (optional)

Outputs (under outputs/reports/kinase_attribution/):
  mea_raw_phospho{,_pY}.csv      — per-track raw-phospho MEA results
  mechanism_annotation.csv       — kinase × contrast classification
  mechanism_attribution.csv      — standardized mechanism attribution (cohort/track-aware)
  unified_attribution.csv        — gets a 'mechanism_annotation' column merged in
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from alz.shared import config
from alz.core.mechanism_attribution import classify_mechanisms
from alz.bulk_mea.enrich import (
    CONTRAST_COEFS,
    _filter_samples,
    _prepare_raw_ols,
    _run_mea,
)

OUTPUT_DIR = config.KINASE_ATTRIBUTION_OUTPUT_DIR


_MECHANISM_COHORT = "song"
_MECHANISM_LEGACY_CALL_MAP = {
    "both": "both",
    "activity_driven": "activity_driven",
    "abundance_driven": "abundance_driven",
    "discordant": "both",
}


def _ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def _run_track_raw_mea(track_cfg, raw_df, mapping):
    """Per-track raw-phospho MEA.

    Returns the per-track ``mea_raw`` DataFrame, or ``None`` when the input
    sites table is empty (caller decides whether to skip the track).
    """
    if raw_df is None or raw_df.empty:
        return None
    bio_cols = mapping["column_name"].tolist()
    print(f"  [{track_cfg['name']}] OLS on raw phospho ({len(raw_df)} sites)...")
    X, X_np, param_names, Y_raw, betas_r, pvals_r, nobs_r, _ = _prepare_raw_ols(
        mapping, bio_cols, raw_df)
    results_by_contrast = {}
    for contrast_name, coefs in CONTRAST_COEFS.items():
        c_vec = np.zeros(len(param_names))
        for param, weight in coefs.items():
            c_vec[param_names.index(param)] = weight
        results_by_contrast[contrast_name] = {"raw_lfc": betas_r @ c_vec}
    print(f"  [{track_cfg['name']}] running MEA on raw phospho...")
    mea_raw, _, _, _ = _run_mea(
        raw_df["motif"], results_by_contrast, "raw_lfc",
        site_ids=raw_df["site_id"].values,
        gene_symbols=raw_df["gene_symbol"].values,
        track=track_cfg["name"])
    return mea_raw


def _ensure_track_column(df, track_name):
    """Return a copy with a populated track column."""
    out = df.copy()
    if "track" not in out.columns:
        out["track"] = track_name
    else:
        out["track"] = out["track"].fillna(track_name)
    return out


def _legacy_mechanism_annotation(classification_df):
    """Convert standardized classification output to legacy Song annotation rows."""
    if classification_df.empty:
        return pd.DataFrame(columns=["kinase", "contrast", "stoich_FDR", "raw_FDR", "mechanism"])

    mech = classification_df.copy()
    mech["legacy_mechanism"] = mech["mechanism_call"].map(_MECHANISM_LEGACY_CALL_MAP)
    mech = mech[mech["legacy_mechanism"].notna()].copy()
    if mech.empty:
        return pd.DataFrame(columns=["kinase", "contrast", "stoich_FDR", "raw_FDR", "mechanism"])

    mech = mech.sort_values(by=["cohort", "kinase", "contrast", "track"])
    rows = []
    for (kinase, contrast), grp in mech.groupby(["kinase", "contrast"], dropna=False):
        has_both = (grp["legacy_mechanism"] == "both").any()
        has_activity = (grp["legacy_mechanism"] == "activity_driven").any()
        has_abundance = (grp["legacy_mechanism"] == "abundance_driven").any()

        if has_both or (has_activity and has_abundance):
            mechanism = "both"
        elif has_activity:
            mechanism = "activity_driven"
        elif has_abundance:
            mechanism = "abundance_driven"
        else:
            continue

        rep = grp.iloc[0]
        rows.append({
            "kinase": kinase,
            "contrast": contrast,
            "stoich_FDR": pd.to_numeric(rep["stoich_FDR"], errors="coerce"),
            "raw_FDR": pd.to_numeric(rep["raw_FDR"], errors="coerce"),
            "mechanism": mechanism,
        })

    return pd.DataFrame(rows)


def _merge_mechanism_into_unified(unified, annotation_df):
    """Return a copy of ``unified`` with a `mechanism_annotation` column added."""
    out = unified.copy()
    mech_map = {(row["kinase"], row["contrast"]): row["mechanism"]
                for _, row in annotation_df.iterrows()}
    out["mechanism_annotation"] = out.apply(
        lambda r: mech_map.get((r["kinase"], r["contrast"]), ""), axis=1)
    return out


def step_mechanism_annotation():
    """Optional: Run raw phospho MEA and classify abundance/activity/both."""
    _ensure_output_dir()
    print(f"\n=== Mechanism Annotation ({config.ANALYSIS_MODE}) ===\n")

    mapping_full = config.load_sample_mapping()
    mapping = _filter_samples(mapping_full)

    tracks = [config.resolve_track(t) for t in ("st", "py")]
    raw_mea_by_track = {}
    for track_cfg in tracks:
        raw_path = config.track_output("raw_phospho_normalized.csv", track_cfg)
        if not os.path.exists(raw_path):
            print(f"  [{track_cfg['name']}] {raw_path} missing; skip raw MEA "
                  "(run --normalize for this track first).")
            continue
        raw_df = pd.read_csv(raw_path)
        mea_raw = _run_track_raw_mea(track_cfg, raw_df, mapping)
        if mea_raw is None:
            print(f"  [{track_cfg['name']}] raw normalized file empty; skip.")
            continue

        mea_raw_path = config.track_output("mea_raw_phospho.csv", track_cfg)
        mea_raw.to_csv(mea_raw_path, index=False)
        print(f"  Saved {mea_raw_path} ({len(mea_raw)} rows)")
        raw_for_attribution = _ensure_track_column(mea_raw, track_cfg["name"])
        raw_for_attribution["cohort"] = _MECHANISM_COHORT
        raw_mea_by_track[track_cfg["name"]] = raw_for_attribution

    if not raw_mea_by_track:
        print("\n  No raw-phospho tracks were processed; skipping mechanism table.")
        return

    mea_raw = pd.concat(list(raw_mea_by_track.values()), ignore_index=True)
    stoich_dfs = []
    for track_cfg in tracks:
        stoich_path = config.track_output("mea_stoichiometry.csv", track_cfg)
        if not os.path.exists(stoich_path):
            continue
        stoich_df = pd.read_csv(stoich_path)
        stoich_df = _ensure_track_column(stoich_df, track_cfg["name"])
        stoich_df["cohort"] = _MECHANISM_COHORT
        stoich_dfs.append(stoich_df)

    if not stoich_dfs:
        raise FileNotFoundError("No mea_stoichiometry*.csv found. Run --enrich first.")

    mea_stoich = pd.concat(stoich_dfs, ignore_index=True)

    annotation_standard = classify_mechanisms(
        mea_stoich,
        mea_raw,
        context_cols=["cohort", "track", "contrast"],
        fdr_thresh=config.MEA_FDR_THRESH,
    )
    std_path = os.path.join(OUTPUT_DIR, "mechanism_attribution.csv")
    annotation_standard.to_csv(std_path, index=False)
    print(f"  Saved {std_path} ({len(annotation_standard)} rows)")

    annotation_df = _legacy_mechanism_annotation(annotation_standard)
    ann_path = os.path.join(OUTPUT_DIR, "mechanism_annotation.csv")
    annotation_df.to_csv(ann_path, index=False)
    print(f"\n  Saved {ann_path} ({len(annotation_df)} rows)")

    if len(annotation_df) > 0:
        print("\n  Mechanism counts:")
        for mech, cnt in annotation_df["mechanism"].value_counts().items():
            print(f"    {mech}: {cnt}")

    unified_path = os.path.join(OUTPUT_DIR, "unified_attribution.csv")
    if os.path.exists(unified_path):
        unified = pd.read_csv(unified_path)
        unified_out = _merge_mechanism_into_unified(unified, annotation_df)
        unified_out.to_csv(unified_path, index=False)
        print(f"  Merged mechanism annotations into {unified_path}")

    print("\n  Mechanism annotation complete.")


def main(argv: list[str] | None = None):
    """Run mechanism annotation directly (no Kedro)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    step_mechanism_annotation()


if __name__ == "__main__":
    main()

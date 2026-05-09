"""Optional supplementary stage: mechanism annotation.

Re-runs MEA on raw (uncorrected) phospho LFCs and compares the hit-set against
the stoichiometry MEA from Stage 2. Each (kinase, contrast) is classified as
activity_driven / abundance_driven / both / non_significant. Diagnostic for
reviewers — confirms that stoichiometry correction is doing real work.

Reuses Stage 2 helpers via import from kinase_enrich.

Inputs:
  outputs/reports/kinase_attribution/raw_phospho_normalized{,_pY}.csv
  outputs/reports/kinase_attribution/mea_stoichiometry{,_pY}.csv
  outputs/reports/data_ingest/sample_mapping.csv
  outputs/reports/data_ingest/sample_exclusions.csv (optional)

Outputs (under outputs/reports/kinase_attribution/):
  mea_raw_phospho{,_pY}.csv      — per-track raw-phospho MEA results
  mechanism_annotation.csv       — kinase × contrast classification
  unified_attribution.csv        — gets a 'mechanism_annotation' column merged in
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from alz import config
from alz.kinase_enrich import (
    CONTRAST_COEFS,
    _filter_samples,
    _prepare_raw_ols,
    _resolve_track,
    _run_mea,
    _track_output,
    load_sample_mapping,
)

OUTPUT_DIR = config.KINASE_ATTRIBUTION_OUTPUT_DIR


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


def _classify_mechanisms(mea_raw, mea_stoich):
    """Per-(kinase, contrast) classification: activity / abundance / both / ns."""
    rows = []
    for contrast_name in CONTRAST_COEFS:
        stoich_c = mea_stoich[mea_stoich["contrast"] == contrast_name]
        raw_c = mea_raw[mea_raw["contrast"] == contrast_name]
        stoich_sig = set(stoich_c[stoich_c["FDR"] < config.MEA_FDR_THRESH]["kinase"])
        raw_sig = set(raw_c[raw_c["FDR"] < config.MEA_FDR_THRESH]["kinase"])

        for kinase in sorted(stoich_sig | raw_sig):
            in_stoich = kinase in stoich_sig
            in_raw = kinase in raw_sig
            if in_stoich and in_raw:
                mechanism = "both"
            elif in_stoich:
                mechanism = "activity_driven"
            else:
                mechanism = "abundance_driven"

            s_fdr = stoich_c[stoich_c["kinase"] == kinase]["FDR"].values
            r_fdr = raw_c[raw_c["kinase"] == kinase]["FDR"].values
            rows.append({
                "kinase": kinase,
                "contrast": contrast_name,
                "stoich_FDR": float(s_fdr[0]) if len(s_fdr) > 0 else np.nan,
                "raw_FDR": float(r_fdr[0]) if len(r_fdr) > 0 else np.nan,
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

    mapping_full = load_sample_mapping()
    mapping = _filter_samples(mapping_full)

    tracks = [_resolve_track(t) for t in ("st", "py")]
    raw_mea_by_track = {}
    for track_cfg in tracks:
        raw_path = _track_output("raw_phospho_normalized.csv", track_cfg)
        if not os.path.exists(raw_path):
            print(f"  [{track_cfg['name']}] {raw_path} missing; skip raw MEA "
                  "(run --normalize for this track first).")
            continue
        raw_df = pd.read_csv(raw_path)
        mea_raw = _run_track_raw_mea(track_cfg, raw_df, mapping)
        if mea_raw is None:
            print(f"  [{track_cfg['name']}] raw normalized file empty; skip.")
            continue
        mea_raw_path = _track_output("mea_raw_phospho.csv", track_cfg)
        mea_raw.to_csv(mea_raw_path, index=False)
        print(f"  Saved {mea_raw_path} ({len(mea_raw)} rows)")
        raw_mea_by_track[track_cfg["name"]] = mea_raw

    if not raw_mea_by_track:
        print("\n  No raw-phospho tracks were processed; skipping mechanism table.")
        return

    mea_raw = pd.concat(list(raw_mea_by_track.values()), ignore_index=True)
    stoich_paths = [_track_output("mea_stoichiometry.csv", t) for t in tracks]
    stoich_paths = [p for p in stoich_paths if os.path.exists(p)]
    if not stoich_paths:
        raise FileNotFoundError("No mea_stoichiometry*.csv found. Run --enrich first.")
    mea_stoich = pd.concat([pd.read_csv(p) for p in stoich_paths], ignore_index=True)

    annotation_df = _classify_mechanisms(mea_raw, mea_stoich)
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


def main():
    """CLI shim: delegates to `kedro run --pipeline=mechanism`."""
    from kedro.framework.session import KedroSession
    from kedro.framework.startup import bootstrap_project

    bootstrap_project(Path(__file__).resolve().parent.parent)
    with KedroSession.create() as session:
        session.run(pipeline_name="mechanism")


if __name__ == "__main__":
    main()

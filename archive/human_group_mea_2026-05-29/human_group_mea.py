"""Single human AD-vs-control kinase activity metric (group-level MEA).

Concern 1: one fold-change-like NES per kinase saying whether activity is higher/lower in AD
humans vs controls -- the human analog of the mouse `kinase_activity_matrix`. Built as a single
group contrast mean(AD) - mean(clean control) per site, run through the same MEA helper the
per-donor pipeline uses (`alz/bulk_mea/enrich._run_mea`: median-center -> winsorize -> GSEA).

Control set EXCLUDES CTRL-07/08/10: the 2026-05-25 audit
(docs/plans/human_ctrl_outlier_audit_findings_2026-05-25.md) showed they carry a genuine AD-like
signature and contaminate the all-control reference mean. Clean controls = CTRL-01/02/03/04.

This is ADDED ALONGSIDE the per-donor outputs (kinase_donor_nes.csv etc.), which answer the
distinct question of donor-level recurrence. It does not replace them.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from alz.shared import config
from alz.bulk_mea import enrich as kinase_enrich
from alz.ingest.mukesh import HUMAN_KINASE_DIR, SAMPLE_MAPPING_CSV
from alz.ingest.mukesh_perdonor import _load_track_matrix

# Audit verdict: these three controls are AD-like; drop from the control baseline.
AD_LIKE_CTRL = {"CTRL-07", "CTRL-08", "CTRL-10"}
OUT_DIR = HUMAN_KINASE_DIR
LFC_KEY = "stoich_lfc"


def _groups(matrix_cols) -> tuple[list[str], list[str]]:
    m = pd.read_csv(SAMPLE_MAPPING_CSV)
    ad = sorted(m.loc[m.group == "AD", "sample_id"])
    ctrl_clean = sorted(s for s in m.loc[m.group == "CTRL", "sample_id"]
                        if s not in AD_LIKE_CTRL)
    ad = [s for s in ad if s in matrix_cols]
    ctrl_clean = [s for s in ctrl_clean if s in matrix_cols]
    return ad, ctrl_clean


def run_track(track: str) -> pd.DataFrame | None:
    matrix = _load_track_matrix(track, "stoich")
    if matrix is None:
        return None
    ad, clean = _groups(matrix.columns)
    print(f"[{track}] AD={len(ad)}  clean CTRL={len(clean)} (excluded {sorted(AD_LIKE_CTRL)})  "
          f"sites={len(matrix)}")

    X = matrix.set_index("site_id")
    lfc = X[ad].astype(float).mean(axis=1) - X[clean].astype(float).mean(axis=1)
    results = {"AD_vs_cleanCTRL": {LFC_KEY: lfc.values}}

    mea_df, _, _, _ = kinase_enrich._run_mea(
        motif_series=matrix["motif"],
        results_by_contrast=results,
        lfc_key=LFC_KEY,
        site_ids=matrix["site_id"].values,
        gene_symbols=matrix["gene_symbol"].values,
        track=track,
    )
    if mea_df.empty:
        print(f"  [{track}] empty MEA")
        return None
    mea_df = mea_df.sort_values("NES", ascending=False)
    mea_df["track"] = track
    return mea_df


def main():
    frames = []
    for track in ("st", "py"):
        df = run_track(track)
        if df is not None:
            frames.append(df)
    if not frames:
        raise RuntimeError("no MEA output produced")
    out = pd.concat(frames, ignore_index=True)
    cols = [c for c in ["kinase", "NES", "ES", "p-value", "FDR", "Subs fraction",
                        "contrast", "track", "residue_type"] if c in out.columns]
    out = out[cols + [c for c in out.columns if c not in cols]]
    path = os.path.join(OUT_DIR, "human_group_mea_clean_ctrl.csv")
    out.to_csv(path, index=False)

    print(f"\nwrote {path}  rows={len(out)}")
    sig = config.MEA_FDR_THRESH
    for track in ("st", "py"):
        t = out[out.track == track]
        up = t[(t.NES > 0) & (t.FDR < sig)]
        dn = t[(t.NES < 0) & (t.FDR < sig)]
        print(f"\n[{track}] FDR<{sig}: {len(up)} up, {len(dn)} down")
        print("  top up:  ", ", ".join(f"{r.kinase}({r.NES:+.2f})"
              for r in up.head(8).itertuples()))
        print("  top down:", ", ".join(f"{r.kinase}({r.NES:+.2f})"
              for r in dn.sort_values('NES').head(8).itertuples()))


if __name__ == "__main__":
    main()

"""Per-donor MEA kinase enrichment for the human NBB (Mukesh) cohort.

For each AD donor, builds a per-site stoichiometry delta vector
(``stoich_AD_i - mean(stoich_CTRL)``) and runs MEA against it, then
aggregates results into a kinase x donor NES matrix and a recurrence
summary (kinases significant at FDR < MEA_FDR_THRESH in >= k donors).

Inputs (under outputs/reports/kinase_attribution_human/, written by
``ingest_mukesh.py --reshape``):
  stoichiometry_matrix.csv       — IMAC (S/T) track
  stoichiometry_matrix_pY.csv    — pY track
  ../data_ingest_human/sample_mapping.csv

Outputs (under outputs/reports/kinase_attribution_human/perdonor/):
  mea_perdonor{suffix}.csv       — long form: donor, kinase, NES, FDR, ...
  kinase_donor_nes{suffix}.csv   — kinase x donor wide NES matrix
  kinase_donor_fdr{suffix}.csv   — kinase x donor wide FDR matrix
  recurrence{suffix}.csv         — kinases significant in >= k donors
  mea_global_shift{suffix}.csv   — per-donor median-centering log
  winsorized_sites{suffix}.csv   — per-donor winsorized site log
"""

import argparse
import os
import sys
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import pandas as pd

from alz import config
from alz import kinase_enrich
from alz.ingest_mukesh import (
    HUMAN_DATA_INGEST_DIR,
    HUMAN_KINASE_DIR,
    SAMPLE_MAPPING_CSV,
)

PERDONOR_DIR = os.path.join(HUMAN_KINASE_DIR, "perdonor")


def _load_track_matrix(track: str) -> pd.DataFrame:
    suffix = config.PHOSPHO_TRACKS[track]["output_suffix"]
    path = os.path.join(HUMAN_KINASE_DIR, f"stoichiometry_matrix{suffix}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"missing {path}; run ingest_mukesh.py --reshape")
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
    matrix: pd.DataFrame, ad_ids: list[str], ctrl_ids: list[str]
) -> dict[str, dict[str, np.ndarray]]:
    """Return ``{donor: {"stoich_lfc": np.array per site}}``.

    Delta is computed as ``stoich_AD_i - nanmean(stoich_CTRL)`` per site;
    sites with no CTRL coverage are NaN (downstream `_run_mea` drops them
    before ranking).
    """
    ctrl_block = matrix[ctrl_ids].to_numpy(dtype=float)
    with np.errstate(all="ignore"):
        ctrl_mean = np.nanmean(ctrl_block, axis=1)
    out: dict[str, dict[str, np.ndarray]] = {}
    for donor in ad_ids:
        donor_vec = matrix[donor].to_numpy(dtype=float)
        delta = donor_vec - ctrl_mean
        out[f"{donor}_vs_CTRLmean"] = {"stoich_lfc": delta}
    return out


def _run_track(track: str, mapping: pd.DataFrame) -> None:
    print(f"\n=== Per-donor MEA: track={track} ===")
    matrix = _load_track_matrix(track)
    ad_ids, ctrl_ids = _split_samples(mapping)
    # Keep only samples present in the matrix.
    ad_ids = [s for s in ad_ids if s in matrix.columns]
    ctrl_ids = [s for s in ctrl_ids if s in matrix.columns]
    print(f"  AD donors: {len(ad_ids)}  CTRL: {len(ctrl_ids)}  sites: {len(matrix)}")

    results = _build_donor_deltas(matrix, ad_ids, ctrl_ids)

    motif_series = matrix["motif"]
    site_ids = matrix["site_id"].values
    gene_symbols = matrix["gene_symbol"].values

    mea_df, shift_df, wins_df, _substrate_df = kinase_enrich._run_mea(
        motif_series=motif_series,
        results_by_contrast=results,
        lfc_key="stoich_lfc",
        site_ids=site_ids,
        gene_symbols=gene_symbols,
        track=track,
    )

    suffix = config.PHOSPHO_TRACKS[track]["output_suffix"]
    os.makedirs(PERDONOR_DIR, exist_ok=True)

    mea_path = os.path.join(PERDONOR_DIR, f"mea_perdonor{suffix}.csv")
    mea_df.to_csv(mea_path, index=False)
    print(f"  wrote {mea_path}  rows={len(mea_df)}")

    shift_df.to_csv(
        os.path.join(PERDONOR_DIR, f"mea_global_shift{suffix}.csv"), index=False
    )
    wins_df.to_csv(
        os.path.join(PERDONOR_DIR, f"winsorized_sites{suffix}.csv"), index=False
    )

    if mea_df.empty:
        print("  WARNING: empty MEA result; skipping aggregation")
        return

    # Strip the "_vs_CTRLmean" suffix for cleaner column labels.
    mea_df = mea_df.copy()
    mea_df["donor"] = mea_df["contrast"].str.replace("_vs_CTRLmean", "", regex=False)

    nes_wide = mea_df.pivot_table(
        index="kinase", columns="donor", values="NES", aggfunc="first"
    )
    fdr_wide = mea_df.pivot_table(
        index="kinase", columns="donor", values="FDR", aggfunc="first"
    )
    nes_wide = nes_wide.reindex(columns=ad_ids)
    fdr_wide = fdr_wide.reindex(columns=ad_ids)
    nes_wide.to_csv(os.path.join(PERDONOR_DIR, f"kinase_donor_nes{suffix}.csv"))
    fdr_wide.to_csv(os.path.join(PERDONOR_DIR, f"kinase_donor_fdr{suffix}.csv"))

    sig_mask = fdr_wide < config.MEA_FDR_THRESH
    up_mask = sig_mask & (nes_wide > 0)
    dn_mask = sig_mask & (nes_wide < 0)
    rec = pd.DataFrame({
        "kinase": fdr_wide.index,
        "n_donors_sig": sig_mask.sum(axis=1).values,
        "n_donors_up": up_mask.sum(axis=1).values,
        "n_donors_down": dn_mask.sum(axis=1).values,
        "n_donors_tested": fdr_wide.notna().sum(axis=1).values,
        "median_nes": nes_wide.median(axis=1, skipna=True).values,
        "median_nes_sig_only": nes_wide.where(sig_mask).median(axis=1, skipna=True).values,
    })
    rec = rec.sort_values(
        ["n_donors_sig", "n_donors_tested"], ascending=[False, False]
    )
    rec_path = os.path.join(PERDONOR_DIR, f"recurrence{suffix}.csv")
    rec.to_csv(rec_path, index=False)
    print(
        f"  recurrence written: {rec_path}  "
        f"(>=1 donor sig: {(rec['n_donors_sig'] >= 1).sum()}; "
        f">=ceil(N/2) donors sig: "
        f"{(rec['n_donors_sig'] >= np.ceil(len(ad_ids) / 2)).sum()})"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--track",
        choices=["st", "py", "both"],
        default="both",
        help="phospho track to run (default: both).",
    )
    args = parser.parse_args(argv)

    if not os.path.exists(SAMPLE_MAPPING_CSV):
        print(f"ERROR: missing {SAMPLE_MAPPING_CSV}; run ingest_mukesh.py --reshape first")
        return 2
    mapping = pd.read_csv(SAMPLE_MAPPING_CSV)

    tracks = ["st", "py"] if args.track == "both" else [args.track]
    for t in tracks:
        _run_track(t, mapping)
    return 0


if __name__ == "__main__":
    sys.exit(main())

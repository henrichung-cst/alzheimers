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
``alz/ingest/mukesh.py --reshape``):
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

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import pandas as pd

from alz.shared import config
from alz.bulk_mea import enrich as kinase_enrich
from alz.ingest.mukesh import (
    HUMAN_DATA_INGEST_DIR,
    HUMAN_KINASE_DIR,
    SAMPLE_MAPPING_CSV,
)

PERDONOR_DIR = os.path.join(HUMAN_KINASE_DIR, "perdonor")


def _load_track_matrix(track: str, kind: str = "stoich") -> pd.DataFrame | None:
    """Load the per-track matrix; ``kind`` selects stoichiometry vs raw phospho."""
    suffix = config.PHOSPHO_TRACKS[track]["output_suffix"]
    base = "stoichiometry_matrix" if kind == "stoich" else "raw_phospho_normalized"
    path = os.path.join(HUMAN_KINASE_DIR, f"{base}{suffix}.csv")
    if not os.path.exists(path):
        if kind == "raw":
            return None
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


def _run_track_kind(track: str, kind: str, mapping: pd.DataFrame) -> None:
    spec = _KIND_SPEC[kind]
    print(f"\n=== Per-donor MEA: track={track} kind={kind} ===")
    matrix = _load_track_matrix(track, spec["matrix_kind"])
    if matrix is None:
        print(f"  [{track}/{kind}] input matrix missing; skip.")
        return
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
    # Keep AD first, then CTRL — column order matters for downstream wide
    # loaders that key off the first N columns being case donors.
    donor_order = ad_ids + ctrl_ids
    nes_wide = nes_wide.reindex(columns=donor_order)
    fdr_wide = fdr_wide.reindex(columns=donor_order)
    nes_wide.to_csv(os.path.join(PERDONOR_DIR, f"kinase_donor_nes{infix}{suffix}.csv"))
    fdr_wide.to_csv(os.path.join(PERDONOR_DIR, f"kinase_donor_fdr{infix}{suffix}.csv"))

    def _recurrence(donor_ids: list[str]) -> pd.DataFrame:
        if not donor_ids:
            return pd.DataFrame(columns=[
                "kinase", "n_donors_sig", "n_donors_up", "n_donors_down",
                "n_donors_tested", "median_nes", "median_nes_sig_only",
            ])
        nes_sub = nes_wide.reindex(columns=donor_ids)
        fdr_sub = fdr_wide.reindex(columns=donor_ids)
        sig = fdr_sub < config.MEA_FDR_THRESH
        up = sig & (nes_sub > 0)
        dn = sig & (nes_sub < 0)
        return pd.DataFrame({
            "kinase": fdr_sub.index,
            "n_donors_sig": sig.sum(axis=1).values,
            "n_donors_up": up.sum(axis=1).values,
            "n_donors_down": dn.sum(axis=1).values,
            "n_donors_tested": fdr_sub.notna().sum(axis=1).values,
            "median_nes": nes_sub.median(axis=1, skipna=True).values,
            "median_nes_sig_only": nes_sub.where(sig).median(axis=1, skipna=True).values,
        }).sort_values(["n_donors_sig", "n_donors_tested"], ascending=[False, False])

    # AD recurrence — primary, feeds SEA-AD agreement.
    rec = _recurrence(ad_ids)
    rec_path = os.path.join(PERDONOR_DIR, f"recurrence{infix}{suffix}.csv")
    rec.to_csv(rec_path, index=False)
    print(
        f"  recurrence written: {rec_path}  "
        f"(>=1 donor sig: {(rec['n_donors_sig'] >= 1).sum()}; "
        f">=ceil(N/2) donors sig: "
        f"{(rec['n_donors_sig'] >= np.ceil(len(ad_ids) / 2)).sum()})"
    )
    # CTRL recurrence — sibling table; never feeds SEA-AD agreement. Used by
    # the viewer to surface controls that look case-like after LOO.
    rec_ctrl = _recurrence(ctrl_ids)
    rec_ctrl_path = os.path.join(PERDONOR_DIR, f"recurrence_ctrl{infix}{suffix}.csv")
    rec_ctrl.to_csv(rec_ctrl_path, index=False)
    print(
        f"  ctrl recurrence written: {rec_ctrl_path}  "
        f"(>=1 ctrl sig: {(rec_ctrl['n_donors_sig'] >= 1).sum() if len(rec_ctrl) else 0})"
    )


def _run_track(track: str, mapping: pd.DataFrame) -> None:
    _run_track_kind(track, "stoich", mapping)
    _run_track_kind(track, "raw", mapping)


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

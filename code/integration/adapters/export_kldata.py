"""Adapter 5.3: Export kinase-substrate reference data in Incytr kldata format.

The kinase-library encodes motif-based kinase-substrate predictions. The Incytr
5xFAD example includes a pre-built kldata file (kldata_pspy.csv) with 101K rows
of mouse gene symbols in the correct format. Since the kinase-library predictions
are a static reference (not disease-specific), we reuse this file and filter to
kinases present in our MEA results.
"""

import os
import shutil

import pandas as pd

from common import (
    load_kinase_to_mouse_gene_mapping,
    ensure_intermediates_dir,
)
import config_integration as icfg


# The 5xFAD kldata is a pre-built export of kinase-library predictions
# in Incytr format (gene, site_pos, motif.geneName) with mouse gene symbols.
FIVEXFAD_KLDATA = os.path.join(
    os.path.dirname(icfg.REPO_ROOT),  # parent of alzheimers repo
    "incytr", "examples", "5xad_data", "kldata_pspy.csv",
)


def main():
    ensure_intermediates_dir()
    out_path = os.path.join(icfg.INTERMEDIATES_DIR, "kldata.csv")

    # ------------------------------------------------------------------
    # 1. Load pre-built kldata from Incytr 5xFAD example
    # ------------------------------------------------------------------
    if not os.path.exists(FIVEXFAD_KLDATA):
        raise FileNotFoundError(
            f"5xFAD kldata not found at {FIVEXFAD_KLDATA}. "
            "This file contains kinase-library substrate predictions in "
            "Incytr format. It should be at incytr/examples/5xad_data/kldata_pspy.csv"
        )

    kldata = pd.read_csv(FIVEXFAD_KLDATA)
    # Drop the unnamed index column if present
    kldata = kldata.drop(columns=[c for c in kldata.columns if "Unnamed" in c],
                         errors="ignore")
    print(f"Loaded kldata: {len(kldata)} rows, "
          f"{kldata['motif.geneName'].nunique()} kinases, "
          f"{kldata['gene'].nunique()} substrates")

    # ------------------------------------------------------------------
    # 2. Verify overlap with our MEA kinases
    # ------------------------------------------------------------------
    kin_to_gene = load_kinase_to_mouse_gene_mapping()
    mea = pd.read_csv(icfg.MEA_STOICHIOMETRY_CSV)
    mea_kinases = set(mea["kinase"].unique())
    mea_gene_symbols = {kin_to_gene.get(k, k) for k in mea_kinases}

    kldata_kinases = set(kldata["motif.geneName"].unique())
    overlap = mea_gene_symbols & kldata_kinases
    print(f"MEA kinases (as mouse gene symbols): {len(mea_gene_symbols)}")
    print(f"kldata kinases: {len(kldata_kinases)}")
    print(f"Overlap: {len(overlap)} ({100*len(overlap)/len(mea_gene_symbols):.0f}%)")

    missing = mea_gene_symbols - kldata_kinases
    if missing and len(missing) <= 20:
        print(f"Missing from kldata: {sorted(missing)}")
    elif missing:
        print(f"Missing from kldata: {len(missing)} kinases")

    # ------------------------------------------------------------------
    # 3. Keep only the columns Incytr expects
    # ------------------------------------------------------------------
    keep_cols = ["gene", "site_pos", "motif.geneName"]
    if "Type" in kldata.columns:
        keep_cols.append("Type")
    kldata = kldata[keep_cols]

    # ------------------------------------------------------------------
    # 4. Write
    # ------------------------------------------------------------------
    kldata.to_csv(out_path, index=False)
    print(f"\nWrote {out_path} ({len(kldata)} rows)")
    print("Adapter 5.3 complete.")


if __name__ == "__main__":
    main()

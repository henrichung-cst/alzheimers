"""Adapter 5.2: Export MEA results as Incytr kl_output format.

Reads MEA enrichment results and cell-type attribution, filters to the
Phase 1 contrast and receiver cell type, and outputs kinase activity
evidence in Incytr's kl_output format.

The kl_output tells Incytr which kinases are *active* in disease (from our
MEA analysis). The kldata (adapter 5.3) provides the static kinase-substrate
reference. Together they drive Incytr's Cal_activity_score().
"""

import os

import pandas as pd

from common import load_kinase_to_mouse_gene_mapping, ensure_intermediates_dir
import config_integration as icfg


def main():
    ensure_intermediates_dir()

    # ------------------------------------------------------------------
    # 1. Load MEA results for the target contrast
    # ------------------------------------------------------------------
    mea = pd.read_csv(icfg.MEA_STOICHIOMETRY_CSV)
    mea_contrast = mea[mea["contrast"] == icfg.CONTRAST].copy()
    print(f"MEA results for {icfg.CONTRAST}: {len(mea_contrast)} kinases")

    # ------------------------------------------------------------------
    # 2. Load unified attribution for tiered filtering
    # ------------------------------------------------------------------
    attr = pd.read_csv(icfg.UNIFIED_ATTRIBUTION_CSV)
    attr_contrast = attr[attr["contrast"] == icfg.CONTRAST].copy()

    # For kl_output, keep kinases attributed at moderate+ confidence to the
    # receiver cell type. Incytr's Cal_activity_score gates by the receiver's
    # EI (Exclusiveness Index), so attributions to the receiver are most relevant.
    receiver_attr = attr_contrast[
        (attr_contrast["cell_type"] == icfg.RECEIVER)
        & (attr_contrast["combined_confidence"].isin(["high", "moderate"]))
    ]
    # Also include sender attributions (kinases in sender can appear in pathways)
    sender_attr = attr_contrast[
        (attr_contrast["cell_type"] == icfg.SENDER)
        & (attr_contrast["combined_confidence"].isin(["high", "moderate"]))
    ]
    attributed_kinases = set(receiver_attr["kinase"]) | set(sender_attr["kinase"])
    print(f"Kinases attributed at moderate+ to {icfg.RECEIVER}: "
          f"{len(receiver_attr)}")
    print(f"Kinases attributed at moderate+ to {icfg.SENDER}: "
          f"{len(sender_attr)}")
    print(f"Total unique attributed kinases: {len(attributed_kinases)}")

    # ------------------------------------------------------------------
    # 3. Filter MEA to attributed kinases
    # ------------------------------------------------------------------
    mea_filtered = mea_contrast[mea_contrast["kinase"].isin(attributed_kinases)]
    print(f"MEA results after attribution filter: {len(mea_filtered)}")

    # ------------------------------------------------------------------
    # 4. Map kinase abbreviations to gene symbols
    # ------------------------------------------------------------------
    kin_to_gene = load_kinase_to_mouse_gene_mapping()

    # ------------------------------------------------------------------
    # 5. Build kl_output: one row per kinase-substrate pair
    #
    # Incytr's kl_output format expects kinase-substrate pairs with scores.
    # The MEA "Leading substrates" are 15-mer motifs, not gene symbols.
    # Instead, we use the kldata (kinase-library reference) to get the
    # substrate gene symbols for each kinase, and attach the MEA score/FDR.
    # ------------------------------------------------------------------
    kldata_path = os.path.join(icfg.INTERMEDIATES_DIR, "kldata.csv")
    if not os.path.exists(kldata_path):
        raise FileNotFoundError(
            f"kldata.csv not found at {kldata_path}. "
            "Run export_kldata.py first."
        )
    kldata = pd.read_csv(kldata_path)

    rows = []
    for _, mea_row in mea_filtered.iterrows():
        kin_abbrev = mea_row["kinase"]
        kin_gene = kin_to_gene.get(kin_abbrev, kin_abbrev)
        nes = mea_row["NES"]
        fdr = mea_row["FDR"]

        # Find substrates for this kinase in kldata
        kin_subs = kldata[kldata["motif.geneName"] == kin_gene]
        if kin_subs.empty:
            # Try case-insensitive match
            kin_subs = kldata[
                kldata["motif.geneName"].str.lower() == kin_gene.lower()
            ]

        for _, sub_row in kin_subs.iterrows():
            rows.append({
                "kinase": kin_gene,
                "substrate": sub_row["gene"],
                "site_pos": sub_row["site_pos"],
                "score": nes,
                "padj": fdr,
            })

    kl_output = pd.DataFrame(rows)
    if kl_output.empty:
        print("WARNING: kl_output is empty — no kinase-substrate pairs found")
    else:
        kl_output = kl_output.drop_duplicates()
        print(f"\nkl_output: {len(kl_output)} rows, "
              f"{kl_output['kinase'].nunique()} kinases, "
              f"{kl_output['substrate'].nunique()} substrates")
        sig = kl_output[kl_output["padj"] < icfg.PHOSPHO_FDR_GATE]
        print(f"  FDR < {icfg.PHOSPHO_FDR_GATE}: {sig['kinase'].nunique()} kinases")

    # ------------------------------------------------------------------
    # 6. Write
    # ------------------------------------------------------------------
    out_path = os.path.join(icfg.INTERMEDIATES_DIR, "kl_output.csv")
    kl_output.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}")
    print("Adapter 5.2 complete.")


if __name__ == "__main__":
    main()

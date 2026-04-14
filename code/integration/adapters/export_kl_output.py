"""Adapter 5.2: Export MEA results as Incytr kl_output format.

Reads MEA enrichment results and cell-type attribution, filters to the
Phase 1 contrast and receiver cell type, and outputs kinase activity
evidence in Incytr's kl_output format.

The kl_output tells Incytr which kinases are *active* in disease (from our
MEA analysis). The kldata (adapter 5.3) provides the static kinase-substrate
reference. Together they drive Incytr's Cal_activity_score().

Flags:
  --all-pairs   Include kinases attributed at moderate+ confidence to ANY
                cell type (for all-pairs pipeline). Default: sender/receiver only.
"""

import argparse
import os

import pandas as pd

from common import load_kinase_to_mouse_gene_mapping, ensure_intermediates_dir
import config_integration as icfg


def main(all_pairs: bool = False):
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

    if all_pairs:
        # All-pairs mode: include kinases attributed at moderate+ to ANY
        # cell type, so every sender-receiver pair has full kinase coverage.
        all_attr = attr_contrast[
            attr_contrast["combined_confidence"].isin(["high", "moderate"])
        ]
        attributed_kinases = set(all_attr["kinase"])
        print(f"All-pairs mode: {len(attributed_kinases)} kinases attributed "
              f"at moderate+ to any cell type")
    else:
        # Single-pair mode: only sender/receiver attributions.
        receiver_attr = attr_contrast[
            (attr_contrast["cell_type"] == icfg.RECEIVER)
            & (attr_contrast["combined_confidence"].isin(["high", "moderate"]))
        ]
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

    # Pre-group kldata by kinase gene name for O(1) lookup
    kl_grouped = dict(list(kldata.groupby("motif.geneName")))
    kl_lower_map = {}
    for name in kldata["motif.geneName"].unique():
        kl_lower_map.setdefault(name.lower(), name)

    rows = []
    for _, mea_row in mea_filtered.iterrows():
        kin_abbrev = mea_row["kinase"]
        kin_gene = kin_to_gene.get(kin_abbrev, kin_abbrev)
        nes = mea_row["NES"]
        fdr = mea_row["FDR"]

        kin_subs = kl_grouped.get(kin_gene)
        if kin_subs is None:
            canonical = kl_lower_map.get(kin_gene.lower())
            if canonical is not None:
                kin_subs = kl_grouped.get(canonical)

        if kin_subs is None:
            continue

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
    parser = argparse.ArgumentParser(description="Export kl_output for Incytr")
    parser.add_argument("--all-pairs", action="store_true",
                        help="Include kinases attributed to any cell type")
    args = parser.parse_args()
    main(all_pairs=args.all_pairs)

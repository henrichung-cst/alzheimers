"""Export MEA results for all 9 contrasts in Incytr kl_output format.

Like export_kl_output.py but produces one kl_output covering all contrasts
in the factorial design. Each row has a ``contrast`` column identifying which
of the 9 genotype x timepoint contrasts the kinase evidence comes from.

Always uses all-pairs mode (kinases attributed at moderate+ to any cell type).
"""

import os

import pandas as pd

from common import load_kinase_to_mouse_gene_mapping, ensure_intermediates_dir
import config_integration as icfg


def main():
    ensure_intermediates_dir()
    out = icfg.FACTORIAL_DIR
    os.makedirs(out, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load MEA results (all contrasts)
    # ------------------------------------------------------------------
    mea = pd.read_csv(icfg.MEA_STOICHIOMETRY_CSV)
    contrasts = list(icfg.FACTORIAL_CONTRASTS.keys())
    mea_fac = mea[mea["contrast"].isin(contrasts)].copy()
    print(f"MEA results for {len(contrasts)} contrasts: "
          f"{len(mea_fac)} kinase x contrast rows")

    for c in contrasts:
        n = (mea_fac["contrast"] == c).sum()
        print(f"  {c}: {n} kinases")

    # ------------------------------------------------------------------
    # 2. Load unified attribution (all contrasts, moderate+ confidence)
    # ------------------------------------------------------------------
    attr = pd.read_csv(icfg.UNIFIED_ATTRIBUTION_CSV)
    attr_fac = attr[
        attr["contrast"].isin(contrasts)
        & attr["combined_confidence"].isin(["high", "moderate"])
    ]

    # Per-contrast attributed kinase sets
    attr_by_contrast = {}
    for c in contrasts:
        attr_c = attr_fac[attr_fac["contrast"] == c]
        attr_by_contrast[c] = set(attr_c["kinase"])
        print(f"  {c}: {len(attr_by_contrast[c])} attributed kinases")

    # ------------------------------------------------------------------
    # 3. Load kldata for substrate lookups
    # ------------------------------------------------------------------
    kldata_path = os.path.join(icfg.INTERMEDIATES_DIR, "kldata.csv")
    if not os.path.exists(kldata_path):
        raise FileNotFoundError(
            f"kldata.csv not found at {kldata_path}. "
            "Run export_kldata.py first."
        )
    kldata = pd.read_csv(kldata_path)
    kl_grouped = dict(list(kldata.groupby("motif.geneName")))
    kl_lower_map = {}
    for name in kldata["motif.geneName"].unique():
        kl_lower_map.setdefault(name.lower(), name)

    # ------------------------------------------------------------------
    # 4. Map kinase abbreviations to mouse gene symbols
    # ------------------------------------------------------------------
    kin_to_gene = load_kinase_to_mouse_gene_mapping()

    # ------------------------------------------------------------------
    # 5. Build kl_output: one row per kinase-substrate pair per contrast
    # ------------------------------------------------------------------
    rows = []
    for c in contrasts:
        mea_c = mea_fac[
            (mea_fac["contrast"] == c)
            & (mea_fac["kinase"].isin(attr_by_contrast[c]))
        ]
        for _, mea_row in mea_c.iterrows():
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
                    "contrast": c,
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
        print(f"\nkl_output: {len(kl_output)} rows across {len(contrasts)} contrasts")
        print(f"  {kl_output['kinase'].nunique()} unique kinases, "
              f"{kl_output['substrate'].nunique()} unique substrates")
        for c in contrasts:
            n = (kl_output["contrast"] == c).sum()
            n_kin = kl_output.loc[kl_output["contrast"] == c, "kinase"].nunique()
            print(f"  {c}: {n} rows, {n_kin} kinases")

    # ------------------------------------------------------------------
    # 6. Write
    # ------------------------------------------------------------------
    out_path = os.path.join(out, "kl_output_all_contrasts.csv")
    kl_output.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}")
    print("export_kl_output_factorial complete.")


if __name__ == "__main__":
    main()

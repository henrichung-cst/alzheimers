"""Adapter 5.5: Export kinase-imputed receiver gene list.

Identifies genes that fail the snRNA-seq expression detection threshold
but have protein-level evidence of kinase activity: the gene is a known
substrate (in kldata) of at least one kinase with significant MEA
enrichment (FDR < threshold) in the current contrast.

These genes are added to the receiver gene list for Incytr pathway
inference, expanding the set of discoverable pathways. Pathways
containing kinase-imputed nodes are labeled separately from
expression-confirmed pathways.

Output:
  intermediates/kinase_imputed_genes.csv
    gene            — mouse gene symbol (substrate)
    n_sig_kinases   — number of significant kinases targeting this gene
    source_kinases  — semicolon-delimited kinase abbreviations
    best_fdr        — lowest FDR among source kinases
"""

import os

import pandas as pd

from common import (ensure_intermediates_dir,
                    load_kinase_to_mouse_gene_mapping,
                    load_mouse_gene_to_kinase_mapping,
                    build_substrate_kinase_map)
import config_integration as icfg


def main():
    ensure_intermediates_dir()

    # ------------------------------------------------------------------
    # 1. Load significant kinases for this contrast
    # ------------------------------------------------------------------
    print("Loading MEA results...")
    mea = pd.read_csv(icfg.MEA_STOICHIOMETRY_CSV)
    mea_contrast = mea[mea["contrast"] == icfg.CONTRAST].copy()
    sig = mea_contrast[mea_contrast["FDR"] < icfg.KINASE_IMPUTATION_FDR]
    print(f"  {len(sig)} significant kinases (FDR < {icfg.KINASE_IMPUTATION_FDR}) "
          f"for {icfg.CONTRAST}")

    sig_kin_info = {}
    for _, r in sig.iterrows():
        sig_kin_info[r["kinase"]] = (r["NES"], r["FDR"])

    # ------------------------------------------------------------------
    # 2. Map kinase abbreviations to mouse gene symbols
    # ------------------------------------------------------------------
    kin_to_gene = load_kinase_to_mouse_gene_mapping()
    gene_to_kins = load_mouse_gene_to_kinase_mapping()

    sig_mouse_genes = {kin_to_gene[k] for k in sig_kin_info if k in kin_to_gene}
    print(f"  {len(sig_mouse_genes)} significant kinases mapped to mouse genes")

    # ------------------------------------------------------------------
    # 3. Find substrates of significant kinases via kldata
    # ------------------------------------------------------------------
    kldata_path = os.path.join(icfg.INTERMEDIATES_DIR, "kldata.csv")
    if not os.path.exists(kldata_path):
        print("  WARNING: kldata.csv not found. Run export_kldata.py first.")
        return

    kldata = pd.read_csv(kldata_path)
    sub_to_kins = build_substrate_kinase_map(kldata)

    # For each substrate gene, find which significant kinases target it
    records = []
    for sub_gene, kin_mouse_genes in sub_to_kins.items():
        sig_abbrevs = set()
        best_fdr = 1.0
        for kg in kin_mouse_genes:
            for abbrev in gene_to_kins.get(kg, set()):
                if abbrev in sig_kin_info:
                    sig_abbrevs.add(abbrev)
                    best_fdr = min(best_fdr, sig_kin_info[abbrev][1])

        if sig_abbrevs:
            records.append({
                "gene": sub_gene,
                "n_sig_kinases": len(sig_abbrevs),
                "source_kinases": ";".join(sorted(sig_abbrevs)),
                "best_fdr": best_fdr,
            })

    imputed = pd.DataFrame(records)
    print(f"  {len(imputed)} substrate genes with significant kinase evidence")

    # ------------------------------------------------------------------
    # 4. Filter to genes in the expression matrix
    # ------------------------------------------------------------------
    expr_genes_path = os.path.join(icfg.INTERMEDIATES_DIR, "expression_genes.csv")
    if os.path.exists(expr_genes_path):
        expr_genes = set(pd.read_csv(expr_genes_path)["gene"])
        before = len(imputed)
        imputed = imputed[imputed["gene"].isin(expr_genes)]
        print(f"  {len(imputed)} in expression matrix "
              f"({before - len(imputed)} excluded)")
    else:
        print("  WARNING: expression_genes.csv not found, keeping all.")

    # ------------------------------------------------------------------
    # 5. Write
    # ------------------------------------------------------------------
    out_path = os.path.join(icfg.INTERMEDIATES_DIR, "kinase_imputed_genes.csv")
    imputed.to_csv(out_path, index=False)
    print(f"\n  Wrote {out_path} ({len(imputed)} genes)")

    # Summary
    print(f"  Median kinases per gene: {imputed['n_sig_kinases'].median():.0f}")
    print(f"  Median best FDR: {imputed['best_fdr'].median():.3f}")


if __name__ == "__main__":
    main()

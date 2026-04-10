"""Adapter 5.4: Export tiered per-cell-type phospho data for Incytr.

Implements the tiered phospho integration scheme:
  - High confidence (Project tier): Winner-take-all, full stoichiometry
    assigned to top-ranked cell type
  - Moderate confidence (Filter tier): No phospho magnitudes passed
    (enters only through kl_output)
  - Low confidence (Exclude): Nothing

Incytr's Integr_multiomics() expects per-condition data frames with
gene_symbol + {celltype}_ps columns containing RAW ABUNDANCE values
(not fold changes). It computes FC internally via Cal_foldchange().

Since our stoichiometry is log2-scale, we convert to linear scale
(2^stoichiometry) so Incytr's internal log2(cond1/cond2) produces
the correct stoichiometry difference.
"""

import os

import numpy as np
import pandas as pd

from common import load_sample_mapping, ensure_intermediates_dir
from config import SEA_AD_SUBCLASSES
import config_integration as icfg


def _get_animal_columns(genotype_code, timepoint, sex="M"):
    """Get stoichiometry matrix column names for matching animals."""
    sm = load_sample_mapping()
    mask = (
        (sm["genotype"] == genotype_code)
        & (sm["timepoint"] == timepoint)
        & (sm["sex"] == sex)
    )
    return list(sm[mask]["column_name"])


def main():
    ensure_intermediates_dir()

    # ------------------------------------------------------------------
    # 1. Load stoichiometry matrix (log2 scale)
    # ------------------------------------------------------------------
    print("Loading stoichiometry matrix...")
    stoich = pd.read_csv(icfg.STOICHIOMETRY_MATRIX_CSV)
    print(f"  {len(stoich)} sites x {stoich.shape[1]} columns")

    # Identify per-animal columns for each condition
    wt_cols = _get_animal_columns("WT", icfg.TIMEPOINT)
    app_cols = _get_animal_columns("APP", icfg.TIMEPOINT)
    print(f"  WT columns ({len(wt_cols)}): {wt_cols}")
    print(f"  App columns ({len(app_cols)}): {app_cols}")

    # Compute per-condition mean stoichiometry (log2 scale)
    stoich["stoich_wt"] = stoich[wt_cols].mean(axis=1)
    stoich["stoich_app"] = stoich[app_cols].mean(axis=1)

    # ------------------------------------------------------------------
    # 2. Aggregate to gene level (max-abs stoichiometry per gene)
    # ------------------------------------------------------------------
    # Multiple phosphosites map to the same gene. For Incytr's per-gene
    # phospho input, keep the site with the largest absolute stoichiometry
    # difference (this is the site that drives MEA ranking).
    stoich["abs_diff"] = (stoich["stoich_app"] - stoich["stoich_wt"]).abs()
    stoich_valid = stoich.dropna(subset=["gene_symbol", "stoich_wt", "stoich_app"])
    idx_max = stoich_valid.groupby("gene_symbol")["abs_diff"].idxmax()
    gene_stoich = stoich_valid.loc[idx_max, ["gene_symbol", "stoich_wt", "stoich_app"]].copy()
    gene_stoich = gene_stoich.set_index("gene_symbol")
    print(f"\n  Gene-level stoichiometry: {len(gene_stoich)} genes")

    # ------------------------------------------------------------------
    # 3. Convert to linear scale for Incytr
    #
    # Incytr's Cal_foldchange() computes log2(cond1/cond2).
    # If we pass 2^stoich, Incytr computes:
    #   log2(2^stoich_app / 2^stoich_wt) = stoich_app - stoich_wt
    # which is exactly the stoichiometry LFC we want.
    # ------------------------------------------------------------------
    gene_stoich["linear_wt"] = np.power(2.0, gene_stoich["stoich_wt"])
    gene_stoich["linear_app"] = np.power(2.0, gene_stoich["stoich_app"])

    # ------------------------------------------------------------------
    # 4. Load unified attribution and kldata for tiered assignment
    #
    # The stoichiometry matrix contains SUBSTRATE genes (phosphosites).
    # Attribution is per KINASE. To assign a substrate's phospho to a
    # cell type, we need: substrate -> kinase (from kldata) -> cell type
    # (from attribution). Winner-take-all: assign to the top cell type
    # of the kinase that phosphorylates this substrate.
    # ------------------------------------------------------------------
    attr = pd.read_csv(icfg.UNIFIED_ATTRIBUTION_CSV)
    attr_contrast = attr[attr["contrast"] == icfg.CONTRAST].copy()

    # Get high-confidence attributions: top cell type per kinase
    high_conf = attr_contrast[attr_contrast["combined_confidence"] == "high"]
    kinase_to_celltype = {}
    if not high_conf.empty:
        top_ct = (
            high_conf.sort_values("combined_score", ascending=False)
            .groupby("kinase")
            .first()
            .reset_index()[["kinase", "gene_symbol", "cell_type"]]
        )
        # Map kinase abbreviation -> top cell type
        kinase_to_celltype = dict(zip(top_ct["kinase"], top_ct["cell_type"]))
        print(f"\n  High-confidence attributions: {len(top_ct)} kinases")
        print(f"  Unique top cell types: {top_ct['cell_type'].nunique()}")
    else:
        print("\n  WARNING: No high-confidence attributions found")

    # Load kldata to get substrate -> kinase mapping
    kldata_path = os.path.join(icfg.INTERMEDIATES_DIR, "kldata.csv")
    if os.path.exists(kldata_path):
        kldata = pd.read_csv(kldata_path)
        # Build substrate gene -> set of kinase gene symbols
        sub_to_kinases = kldata.groupby("gene")["motif.geneName"].apply(set).to_dict()
        print(f"  kldata: {len(sub_to_kinases)} substrate genes")
    else:
        sub_to_kinases = {}
        print("  WARNING: kldata.csv not found")

    # Load kinase-to-gene mapping for abbreviation -> mouse gene symbol
    from common import load_kinase_to_mouse_gene_mapping
    kin_to_gene = load_kinase_to_mouse_gene_mapping()
    # Invert: mouse gene symbol -> set of kinase abbreviations
    gene_to_kin_abbrevs = {}
    for abbrev, gene in kin_to_gene.items():
        gene_to_kin_abbrevs.setdefault(gene, set()).add(abbrev)

    # Build substrate gene -> top cell type mapping (via kinase attribution)
    substrate_to_celltype = {}
    for sub_gene, kinase_genes in sub_to_kinases.items():
        for kin_gene in kinase_genes:
            # Find kinase abbreviations for this gene symbol
            abbrevs = gene_to_kin_abbrevs.get(kin_gene, set())
            for abbrev in abbrevs:
                ct = kinase_to_celltype.get(abbrev)
                if ct is not None:
                    substrate_to_celltype[sub_gene] = ct
                    break  # winner-take-all: first high-conf kinase wins
            if sub_gene in substrate_to_celltype:
                break

    print(f"  Substrates with cell-type assignment: {len(substrate_to_celltype)}")

    # ------------------------------------------------------------------
    # 5. Build per-condition phospho DataFrames
    #
    # Columns: gene_symbol, {subclass}_ps for each of 22 subclasses
    # Values: linear-scale stoichiometry for winner cell type, NA elsewhere
    # ------------------------------------------------------------------
    subclasses = [sc for sc in SEA_AD_SUBCLASSES
                  if sc not in ("Pax6", "L6 IT Car3")]
    ps_cols = [f"{sc}_ps" for sc in subclasses]

    def build_ps_df(value_col):
        """Build a per-condition phospho DataFrame."""
        rows = []
        assigned = 0
        for gene, row in gene_stoich.iterrows():
            rec = {"gene_symbol": gene}
            # Look up cell type via substrate -> kinase -> attribution
            # Use case-insensitive match (stoich has mixed case gene symbols)
            top_ct = substrate_to_celltype.get(gene)
            if top_ct is None:
                # Try mouse-style capitalization
                mouse_gene = gene[0].upper() + gene[1:].lower() if gene else gene
                top_ct = substrate_to_celltype.get(mouse_gene)
            if top_ct is not None and top_ct in subclasses:
                rec[f"{top_ct}_ps"] = row[value_col]
                assigned += 1
            rows.append(rec)

        df = pd.DataFrame(rows)
        for col in ps_cols:
            if col not in df.columns:
                df[col] = np.nan
        df = df[["gene_symbol"] + ps_cols]
        print(f"  {value_col}: {assigned} genes assigned to cell types "
              f"({len(df)} total genes)")
        return df

    ps_wt = build_ps_df("linear_wt")
    ps_app = build_ps_df("linear_app")

    # ------------------------------------------------------------------
    # 6. Write
    # ------------------------------------------------------------------
    wt_path = os.path.join(icfg.INTERMEDIATES_DIR, "ps_condition1.csv")
    app_path = os.path.join(icfg.INTERMEDIATES_DIR, "ps_condition2.csv")
    ps_wt.to_csv(wt_path, index=False)
    ps_app.to_csv(app_path, index=False)
    print(f"\n  Wrote {wt_path}")
    print(f"  Wrote {app_path}")

    # Summary: non-NA entries
    n_nonna = ps_wt[ps_cols].notna().sum().sum()
    print(f"  Non-NA phospho entries (WT): {n_nonna} "
          f"(of {len(ps_wt) * len(ps_cols)} possible)")

    print("\nAdapter 5.4 complete.")


if __name__ == "__main__":
    main()

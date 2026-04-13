"""Adapter 5.4: Export per-cell-type phospho data for Incytr.

Distributes bulk tissue stoichiometry across cell types using
attribution-proportional weighting:

  For each substrate gene:
    1. Find all kinases that phosphorylate it (via kldata)
    2. Sum their attribution scores per cell type
    3. Normalize to proportions → cell-type weights
    4. phospho_celltype = bulk_phospho × proportion

  Genes with no kinase in the attribution pipeline get uniform
  assignment (equal weight to all cell types).

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

from common import (load_sample_mapping, ensure_intermediates_dir,
                    load_mouse_gene_to_kinase_mapping, build_substrate_kinase_map)
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


def _build_attribution_weights(ac, kldata, subclasses, gene_to_kins):
    """Build per-gene per-cell-type attribution weights.

    For each substrate gene, find its kinases via kldata, look up their
    attribution scores per cell type, and normalize to proportions.

    Returns dict: gene -> {cell_type: proportion, ...}
    Genes with no attributed kinases are absent from the dict.
    """
    sub_to_kinases = build_substrate_kinase_map(kldata)

    # Precompute kinase -> [(cell_type, combined_score)] for O(1) lookup
    kinase_ct_scores = {}
    for _, r in ac.iterrows():
        kin = r["kinase"]
        ct = r["cell_type"]
        if ct in subclasses:
            kinase_ct_scores.setdefault(kin, []).append((ct, r["combined_score"]))

    gene_weights = {}
    for sub_gene, kinase_genes in sub_to_kinases.items():
        ct_scores = {}
        for kin_gene in kinase_genes:
            for abbrev in gene_to_kins.get(kin_gene, set()):
                for ct, score in kinase_ct_scores.get(abbrev, []):
                    ct_scores[ct] = ct_scores.get(ct, 0) + score

        if ct_scores:
            total = sum(ct_scores.values())
            if total > 0:
                gene_weights[sub_gene] = {
                    ct: score / total for ct, score in ct_scores.items()
                }

    return gene_weights


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
    stoich["abs_diff"] = (stoich["stoich_app"] - stoich["stoich_wt"]).abs()
    stoich_valid = stoich.dropna(subset=["gene_symbol", "stoich_wt", "stoich_app"])
    idx_max = stoich_valid.groupby("gene_symbol")["abs_diff"].idxmax()
    gene_stoich = stoich_valid.loc[idx_max, ["gene_symbol", "stoich_wt", "stoich_app"]].copy()
    gene_stoich = gene_stoich.set_index("gene_symbol")
    print(f"\n  Gene-level stoichiometry: {len(gene_stoich)} genes")

    # ------------------------------------------------------------------
    # 3. Convert to linear scale for Incytr
    # ------------------------------------------------------------------
    gene_stoich["linear_wt"] = np.power(2.0, gene_stoich["stoich_wt"])
    gene_stoich["linear_app"] = np.power(2.0, gene_stoich["stoich_app"])

    # ------------------------------------------------------------------
    # 4. Build attribution-proportional cell-type weights
    # ------------------------------------------------------------------
    print("\nBuilding attribution-proportional weights...")
    attr = pd.read_csv(icfg.UNIFIED_ATTRIBUTION_CSV)
    attr_contrast = attr[attr["contrast"] == icfg.CONTRAST].copy()

    kldata_path = os.path.join(icfg.INTERMEDIATES_DIR, "kldata.csv")
    if os.path.exists(kldata_path):
        kldata = pd.read_csv(kldata_path)
    else:
        print("  WARNING: kldata.csv not found, using uniform assignment for all genes")
        kldata = pd.DataFrame(columns=["gene", "site_pos", "motif.geneName"])

    gene_to_kins = load_mouse_gene_to_kinase_mapping()

    subclasses = [sc for sc in SEA_AD_SUBCLASSES
                  if sc not in ("Pax6", "L6 IT Car3")]

    gene_weights = _build_attribution_weights(
        attr_contrast, kldata, set(subclasses), gene_to_kins
    )

    n_attributed = sum(1 for g in gene_stoich.index if g in gene_weights)
    n_uniform = len(gene_stoich) - n_attributed
    print(f"  Attributed genes: {n_attributed} (proportional weights)")
    print(f"  Unattributed genes: {n_uniform} (uniform fallback)")

    # ------------------------------------------------------------------
    # 5. Build per-condition phospho DataFrames
    # ------------------------------------------------------------------
    ps_cols = [f"{sc}_ps" for sc in subclasses]
    uniform_prop = 1.0 / len(subclasses)

    def build_ps_df(value_col):
        """Build a per-condition phospho DataFrame."""
        rows = []
        n_attributed_assigned = 0
        n_uniform_assigned = 0
        for gene, row in gene_stoich.iterrows():
            rec = {"gene_symbol": gene}
            val = row[value_col]

            weights = gene_weights.get(gene)
            if weights is not None:
                # Attribution-proportional: distribute by attribution scores.
                # Cell types with attribution get proportional values.
                # Cell types without attribution for this gene get the
                # uniform fallback (tissue average), not NA — Incytr's
                # Cal_foldchange cannot handle NA values.
                for sc in subclasses:
                    prop = weights.get(sc, uniform_prop)
                    rec[f"{sc}_ps"] = val * prop
                n_attributed_assigned += 1
            else:
                # Uniform fallback: equal weight to all cell types
                for sc in subclasses:
                    rec[f"{sc}_ps"] = val * uniform_prop
                n_uniform_assigned += 1

            rows.append(rec)

        df = pd.DataFrame(rows)
        for col in ps_cols:
            if col not in df.columns:
                df[col] = np.nan
        df = df[["gene_symbol"] + ps_cols]

        print(f"  {value_col}: {n_attributed_assigned} proportional, "
              f"{n_uniform_assigned} uniform")
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

    # Summary statistics
    for label, df in [("WT", ps_wt), ("App", ps_app)]:
        n_total = df[ps_cols].size
        n_nonna = df[ps_cols].notna().sum().sum()
        n_zero = (df[ps_cols] == 0).sum().sum()
        print(f"  {label}: {n_nonna}/{n_total} non-NA "
              f"({100*n_nonna/n_total:.1f}%), {n_zero} zeros")

    # Per-cell-type coverage
    print("\n  Per-cell-type non-NA counts:")
    for col in ps_cols:
        n = ps_wt[col].notna().sum()
        if n > 0:
            ct = col.replace("_ps", "")
            print(f"    {ct:20s}: {n:5d}")

    print("\nAdapter 5.4 complete.")


if __name__ == "__main__":
    main()

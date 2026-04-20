"""Adapter 5.5: Export kinase-imputed receiver gene list (refined).

Identifies genes that fail the snRNA-seq expression detection threshold
but have protein-level evidence of kinase activity: the gene is a known
substrate (in kldata) of at least one kinase with significant MEA
enrichment (FDR < KINASE_IMPUTATION_FDR) in the current contrast, AND
(refined behavior) that kinase is sufficiently attributed to the target
receiver cell type per unified_attribution.csv.

Gating rules (see docs/integrations/kinase_incytr_integration.md §5
"Kinase-imputed pathway expansion"):
  - Per-receiver cell-type gating via unified_attribution.combined_score
    > KINASE_IMPUTATION_ATTRIBUTION_TAU. Produces per-receiver files.
  - FDR gate at KINASE_IMPUTATION_FDR (default 0.10) — tighter than
    PHOSPHO_FDR_GATE because imputation multiplies search scope.

Soft rescue using best_fdr, and the expression floor, are applied in
the R wrapper at pathway scoring time.

Output (refined mode):
  intermediates/kinase_imputed_genes__{receiver}.csv     (one per receiver)
    gene             - mouse gene symbol (substrate)
    n_sig_kinases    - number of gated sig kinases targeting this gene
    source_kinases   - semicolon-delimited kinase abbreviations
    best_fdr         - lowest FDR among source kinases
    imputed_weight   - 1 - best_fdr (used by R-side soft rescue)
    receiver         - receiver cell-type label (constant per file)
  intermediates/kinase_imputation_summary.csv
    per-receiver counts of imputed genes, gated kinases, and tau used.

Legacy mode (icfg.KINASE_IMPUTATION_LEGACY=True):
  Emits the pre-refactor flat intermediates/kinase_imputed_genes.csv.
"""

import os

import pandas as pd

from common import (ensure_intermediates_dir,
                    load_kinase_to_mouse_gene_mapping,
                    load_mouse_gene_to_kinase_mapping,
                    build_substrate_kinase_map,
                    sanitize_celltype_name)
import config_integration as icfg


def _load_sig_kinases(mea_df, contrast, fdr_thresh):
    mea_c = mea_df[mea_df["contrast"] == contrast]
    sig = mea_c[mea_c["FDR"] < fdr_thresh]
    return {r["kinase"]: (r["NES"], r["FDR"]) for _, r in sig.iterrows()}


def _compute_flat_records(sig_kin_info, sub_to_kins, gene_to_kins):
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
    return pd.DataFrame(records)


def _compute_gated_records_for_receiver(
    receiver, sig_kin_info, sub_to_kins, gene_to_kins, attr_lookup, tau
):
    records = []
    for sub_gene, kin_mouse_genes in sub_to_kins.items():
        gated_abbrevs = set()
        best_fdr = 1.0
        for kg in kin_mouse_genes:
            for abbrev in gene_to_kins.get(kg, set()):
                if abbrev not in sig_kin_info:
                    continue
                score = attr_lookup.get((receiver, abbrev))
                if score is None or score <= tau:
                    continue
                gated_abbrevs.add(abbrev)
                best_fdr = min(best_fdr, sig_kin_info[abbrev][1])
        if gated_abbrevs:
            records.append({
                "gene": sub_gene,
                "n_sig_kinases": len(gated_abbrevs),
                "source_kinases": ";".join(sorted(gated_abbrevs)),
                "best_fdr": best_fdr,
                "imputed_weight": max(0.0, 1.0 - best_fdr),
                "receiver": receiver,
            })
    return pd.DataFrame(records)


def _filter_by_expression_genes(df, expr_genes):
    if expr_genes is None or df.empty:
        return df
    return df[df["gene"].isin(expr_genes)].copy()


def run_for_contrast(contrast, mea_df, kldata, *, out_dir=None,
                     tau_override=None, fdr_override=None,
                     legacy=False, expr_genes=None,
                     filename_suffix=""):
    if out_dir is None:
        out_dir = icfg.INTERMEDIATES_DIR
    os.makedirs(out_dir, exist_ok=True)

    fdr_thresh = (fdr_override if fdr_override is not None
                  else icfg.KINASE_IMPUTATION_FDR)
    sig_kin_info = _load_sig_kinases(mea_df, contrast, fdr_thresh)
    print(f"  [{contrast}] {len(sig_kin_info)} sig kinases (FDR < {fdr_thresh})")

    gene_to_kins = load_mouse_gene_to_kinase_mapping()
    _ = load_kinase_to_mouse_gene_mapping()  # validate mapping loads
    sub_to_kins = build_substrate_kinase_map(kldata)

    wrote = []

    if legacy:
        df = _compute_flat_records(sig_kin_info, sub_to_kins, gene_to_kins)
        df = _filter_by_expression_genes(df, expr_genes)
        out_path = os.path.join(out_dir, "kinase_imputed_genes.csv")
        df.to_csv(out_path, index=False)
        wrote.append(out_path)
        print(f"  [{contrast}] legacy flat: {len(df)} genes -> {out_path}")
        return ([{"contrast": contrast, "mode": "legacy",
                  "n_genes": int(len(df)), "tau": None}], wrote)

    attr = pd.read_csv(icfg.UNIFIED_ATTRIBUTION_CSV)
    attr_c = attr[attr["contrast"] == contrast]
    attr_c = attr_c[attr_c["combined_score"].notna()]

    tau = tau_override if tau_override is not None else icfg.KINASE_IMPUTATION_ATTRIBUTION_TAU
    if tau is None:
        tau = float(attr_c["combined_score"].median()) if not attr_c.empty else 0.0
    print(f"  [{contrast}] attribution gate tau = {tau:.4f}")

    attr_lookup = {}
    for _, r in attr_c.iterrows():
        key = (str(r["cell_type"]), str(r["kinase"]))
        score = float(r["combined_score"])
        prev = attr_lookup.get(key)
        if prev is None or score > prev:
            attr_lookup[key] = score

    receivers = sorted({ct for (ct, _) in attr_lookup.keys()})
    summary = []
    for recv in receivers:
        df = _compute_gated_records_for_receiver(
            recv, sig_kin_info, sub_to_kins, gene_to_kins, attr_lookup, tau)
        df = _filter_by_expression_genes(df, expr_genes)
        fname = (f"kinase_imputed_genes__{sanitize_celltype_name(recv)}"
                 f"{filename_suffix}.csv")
        out_path = os.path.join(out_dir, fname)
        df.to_csv(out_path, index=False)
        wrote.append(out_path)
        summary.append({
            "contrast": contrast,
            "receiver": recv,
            "n_genes": int(len(df)),
            "n_gated_kinases": int(sum(
                1 for (r, k), s in attr_lookup.items()
                if r == recv and s > tau and k in sig_kin_info)),
            "tau": tau,
            "fdr": fdr_thresh,
        })

    total = sum(row["n_genes"] for row in summary)
    print(f"  [{contrast}] wrote {len(wrote)} per-receiver files, "
          f"{total} gene-receiver pairs total")
    return (summary, wrote)


def main():
    ensure_intermediates_dir()

    mea = pd.read_csv(icfg.MEA_STOICHIOMETRY_CSV)

    kldata_path = os.path.join(icfg.INTERMEDIATES_DIR, "kldata.csv")
    if not os.path.exists(kldata_path):
        print("  WARNING: kldata.csv not found. Run export_kldata.py first.")
        return
    kldata = pd.read_csv(kldata_path)

    expr_genes = None
    expr_genes_path = os.path.join(icfg.INTERMEDIATES_DIR,
                                   "expression_genes.csv")
    if os.path.exists(expr_genes_path):
        expr_genes = set(pd.read_csv(expr_genes_path)["gene"])

    legacy = bool(getattr(icfg, "KINASE_IMPUTATION_LEGACY", False))

    tau_env = os.environ.get("KINASE_IMPUTATION_TAU_OVERRIDE")
    fdr_env = os.environ.get("KINASE_IMPUTATION_FDR_OVERRIDE")
    tau_override = float(tau_env) if tau_env else None
    fdr_override = float(fdr_env) if fdr_env else None

    summary_rows, _ = run_for_contrast(
        icfg.CONTRAST, mea, kldata,
        out_dir=icfg.INTERMEDIATES_DIR,
        legacy=legacy,
        expr_genes=expr_genes,
        tau_override=tau_override,
        fdr_override=fdr_override,
    )
    if not legacy and summary_rows:
        summary_path = os.path.join(icfg.INTERMEDIATES_DIR,
                                    "kinase_imputation_summary.csv")
        pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
        print(f"  Wrote {summary_path}")


if __name__ == "__main__":
    main()

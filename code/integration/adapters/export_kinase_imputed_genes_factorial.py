"""Adapter 5.5-factorial: per-contrast kinase-imputed receiver gene lists.

Loops the refined single-contrast adapter over FACTORIAL_CONTRASTS (9), writing
per-receiver, per-contrast files under intermediates/factorial/:

  kinase_imputed_genes__{receiver}__{contrast}.csv
  kinase_imputation_summary_factorial.csv

Each per-receiver file has the same schema as the single-contrast refined
output: gene, n_sig_kinases, source_kinases, best_fdr, imputed_weight,
receiver. Filtering semantics are per-contrast (each contrast uses its own
imputed substrate set); a union variant would require a second code path.

Run from code/integration/:
  micromamba run -n alzheimers python3 adapters/export_kinase_imputed_genes_factorial.py
"""

import os

import pandas as pd

from common import ensure_intermediates_dir
import config_integration as icfg
from export_kinase_imputed_genes import run_for_contrast


def main():
    ensure_intermediates_dir()
    os.makedirs(icfg.FACTORIAL_DIR, exist_ok=True)

    mea = pd.read_csv(icfg.MEA_STOICHIOMETRY_CSV)

    kldata_path = os.path.join(icfg.INTERMEDIATES_DIR, "kldata.csv")
    if not os.path.exists(kldata_path):
        print("  WARNING: kldata.csv not found. Run export_kldata.py first.")
        return
    kldata = pd.read_csv(kldata_path)

    expr_genes = None
    expr_genes_path = os.path.join(icfg.FACTORIAL_DIR, "expression_genes.csv")
    if os.path.exists(expr_genes_path):
        expr_genes = set(pd.read_csv(expr_genes_path)["gene"])
    else:
        # Fall back to the single-contrast expression gene list (same snRNA-seq
        # base matrix, pre-factorial build).
        alt = os.path.join(icfg.INTERMEDIATES_DIR, "expression_genes.csv")
        if os.path.exists(alt):
            expr_genes = set(pd.read_csv(alt)["gene"])

    tau_env = os.environ.get("KINASE_IMPUTATION_TAU_OVERRIDE")
    fdr_env = os.environ.get("KINASE_IMPUTATION_FDR_OVERRIDE")
    tau_override = float(tau_env) if tau_env else None
    fdr_override = float(fdr_env) if fdr_env else None

    all_summary = []
    for contrast in icfg.FACTORIAL_CONTRASTS.keys():
        rows, _ = run_for_contrast(
            contrast, mea, kldata,
            out_dir=icfg.FACTORIAL_DIR,
            legacy=False,
            expr_genes=expr_genes,
            filename_suffix=f"__{contrast}",
            tau_override=tau_override,
            fdr_override=fdr_override,
        )
        all_summary.extend(rows)

    if all_summary:
        summary_path = os.path.join(
            icfg.FACTORIAL_DIR, "kinase_imputation_summary_factorial.csv")
        pd.DataFrame(all_summary).to_csv(summary_path, index=False)
        print(f"\n  Wrote summary: {summary_path}")
        print(f"  Total (contrast, receiver) rows: {len(all_summary)}")


if __name__ == "__main__":
    main()

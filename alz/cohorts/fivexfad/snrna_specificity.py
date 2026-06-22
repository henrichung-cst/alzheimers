#!/usr/bin/env python3
"""5xFAD within-cohort expression specificity via the standard attribution metric.

Reads the per-(gene, tissue, cell_type) detection + mean log2(x+1) export written by
``alz/ingest/build_5xfad_snrna_attribution.R`` and runs the repo-wide standard metric
(``alz/cross_reference/specificity.py``) once per tissue — mirroring the Song
within-cohort path (``alz/reference/snrna_integration.py:step_specificity``).

This replaces 5xFAD's legacy share/τ localizer (``fivexfad_specificity`` /
``fivexfad_fold_over_uniform`` / ``fivexfad_tau``), which suffered the
share-is-not-presence failure — no detection gate, so a near-zero kinase scored a
high share wherever cross-cluster competition was lowest. One metric, every cohort.

Resolution is the 46-cluster ``new_clusters`` spine, computed per tissue (cortex /
hippocampus) — 5xFAD's existing split, not a coarse rollup. Specificity is
contrast-invariant; the consumer broadcasts it across the age × genotype rows.
"""

from __future__ import annotations

import os
import sys

import pandas as pd

from alz.cross_reference import specificity
from alz.viewer.paths import FIVEXFAD_KINASE_DIR

EXPRESSION_FILE = os.path.join(FIVEXFAD_KINASE_DIR, "fivexfad_snrna_expression.csv")
SPECIFICITY_FILE = os.path.join(
    FIVEXFAD_KINASE_DIR, "fivexfad_expression_specificity.csv"
)


def run() -> None:
    if not os.path.exists(EXPRESSION_FILE):
        print(f"  ERROR: required input not found: {EXPRESSION_FILE}")
        print("  Run `pixi run 5xfad-snrna-attribution` first.")
        sys.exit(1)

    expr = pd.read_csv(EXPRESSION_FILE)
    # The metric is a per-gene property; the export is denormalized per kinase
    # (a gene can back several kinases), so dedup before computing.
    gene_expr = expr.drop_duplicates(["matched_gene", "tissue", "cell_type"])[
        [
            "matched_gene",
            "tissue",
            "cell_type",
            "mean_log2_expression",
            "fraction_cells_expressing",
            "n_cells",
        ]
    ]

    per_label_parts: list[pd.DataFrame] = []
    per_gene_parts: list[pd.DataFrame] = []
    for tissue, sub in gene_expr.groupby("tissue", sort=False):
        per_label, _, per_gene = specificity.compute(
            sub,
            gene_col="matched_gene",
            label_col="cell_type",
            mean_log2_col="mean_log2_expression",
            frac_col="fraction_cells_expressing",
            ncells_col="n_cells",
        )
        per_label["tissue"] = tissue
        per_gene["tissue"] = tissue
        per_label_parts.append(per_label)
        per_gene_parts.append(per_gene)

    per_label = pd.concat(per_label_parts, ignore_index=True)
    per_gene = pd.concat(per_gene_parts, ignore_index=True)

    per_label = per_label.rename(
        columns={
            "detected": "fivexfad_detected",
            "concentration": "fivexfad_concentration",
            "concentration_of_total": "fivexfad_concentration_of_total",
            "concentration_tier": "fivexfad_concentration_tier",
            "fraction_cells_expressing": "fivexfad_fraction_cells_expressing",
        }
    )
    per_gene = per_gene.rename(
        columns={
            "n_detected_native": "fivexfad_n_celltypes_detected",
            "effective_n_native": "fivexfad_effective_n",
            "top_celltype_native": "fivexfad_top_celltype",
            "top_concentration_native": "fivexfad_top_concentration",
        }
    )

    out = per_label.merge(per_gene, on=["matched_gene", "tissue"], how="left")
    # Re-expand to (kinase, gene_symbol): a gene can back several kinases.
    kin_map = expr[["kinase", "gene_symbol", "matched_gene"]].drop_duplicates()
    out = kin_map.merge(out, on="matched_gene", how="inner")

    cols = [
        "kinase",
        "gene_symbol",
        "matched_gene",
        "tissue",
        "cell_type",
        "fivexfad_detected",
        "fivexfad_fraction_cells_expressing",
        "fivexfad_concentration",
        "fivexfad_concentration_of_total",
        "fivexfad_concentration_tier",
        "fivexfad_effective_n",
        "fivexfad_top_celltype",
        "fivexfad_top_concentration",
        "fivexfad_n_celltypes_detected",
    ]
    out = out[[c for c in cols if c in out.columns]]
    os.makedirs(FIVEXFAD_KINASE_DIR, exist_ok=True)
    out.to_csv(SPECIFICITY_FILE, index=False)
    n_det = int(out["fivexfad_detected"].fillna(False).astype(bool).sum())
    print(
        f"[5xfad-snrna-specificity] wrote {SPECIFICITY_FILE} "
        f"rows={len(out):,} detected_rows={n_det:,}"
    )


if __name__ == "__main__":
    run()

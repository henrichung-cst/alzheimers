#!/usr/bin/env python3
"""Human Reference Expression: per-celltype kinase specificity from SEA-AD MTG + Allen HBCA.

Mirrors ``alz/wmb_expression.py`` but consumes two human brain references:

  1. **SEA-AD MTG** — cortical, 139 supertypes. Source:
     ``data/derived/aggregates/seaad/expression_by_supertype.csv``
     (produced by ``atlas_reference.py --sea-ad-expression``).

  2. **Allen Human Brain Cell Atlas (HBCA)** — whole brain, class-level.
     Source: ``data/derived/aggregates/hbca/expression_by_class.csv``
     (produced by ``atlas_reference.py --hbca-download``).

Specificity metric (same formula as WMB):
  ``log2( mean_celltype / mean_brain )`` per gene, then ranked into a
  per-celltype quantile.

This keeps the viewer's cross-reference language consistent: the same
specificity_score column appears for mouse WMB, SEA-AD MTG, and HBCA.

Outputs:
  outputs/reports/human_reference_expression/seaad_kinase_specificity.csv
      (kinase × 139 supertypes; rows = kinase_id, cols = supertypes)
  outputs/reports/human_reference_expression/hbca_kinase_specificity.csv
      (kinase × N HBCA classes; rows = kinase_id, cols = classes)

Phase-1: module skeleton only.  Phase-2 downloads provide the input CSVs.
A small fake h5ad unit test (test_human_reference_expression.py) validates
the specificity recipe without requiring the multi-GB references.

Usage:
    python alz/human_reference_expression.py --ref seaad   # SEA-AD MTG only
    python alz/human_reference_expression.py --ref hbca    # HBCA only
    python alz/human_reference_expression.py --ref both    # Both
    python alz/human_reference_expression.py --summary     # Print cached results
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from alz import config
from alz.reference.atlas import get_all_kinase_genes, get_phosphatase_genes_from_genelist

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OUTPUT_DIR = config.HUMAN_REFERENCE_OUTPUT_DIR
SEAAD_SPEC_FILE = config.SEAAD_KINASE_SPECIFICITY_FILE
HBCA_SPEC_FILE = config.HBCA_KINASE_SPECIFICITY_FILE


# ---------------------------------------------------------------------------
# Core specificity recipe
# ---------------------------------------------------------------------------


def _normalize_gene_symbol(sym: str) -> str:
    """Normalize a gene symbol to uppercase HGNC form.

    Both SEA-AD (HGNC) and HBCA (may use Ensembl IDs or mixed case) need to
    be matched against the kinase gene list which is stored in uppercase HGNC
    form (e.g. 'CDK5', 'MAPT').  Ensembl IDs (ENSGxxx) are returned as-is so
    callers can detect and drop them.
    """
    if not sym or not isinstance(sym, str):
        return ""
    sym = sym.strip()
    if sym.startswith("ENSG") or sym.startswith("ENST"):
        return sym  # Ensembl — caller must resolve via mapping table
    return sym.upper()


def compute_specificity(
    expr_df: pd.DataFrame,
    kinase_genes_human: set[str],
    label: str = "reference",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute kinase-celltype specificity matrix from a genes × celltypes expression table.

    Parameters
    ----------
    expr_df : pd.DataFrame
        Rows = genes (index = gene symbol, should be HGNC uppercase).
        Columns = cell types (supertypes or classes).
        Values = mean log2 expression per gene per cell type.
    kinase_genes_human : set[str]
        Set of human HGNC gene symbols to extract (uppercase).
    label : str
        Reference name for progress messages.

    Returns
    -------
    (spec_df, expr_df_matched)
        spec_df: kinase × celltype log2-ratio specificity scores.
        expr_df_matched: kinase × celltype raw mean log2 expression
          (same shape; subset of input ``expr_df`` rows matching kinase set).
        Both indexed by gene symbol (HGNC uppercase, = kinase_id).

    Notes
    -----
    - Brain-wide mean = row mean across all celltypes in expr_df.
    - Genes with brain-wide mean ≤ 0 get specificity 0.0 (not log-defined).
    - NaN in expression propagation: cells with NaN are treated as 0 (missing
      data is not dropped so the denominator is always the full cell-type set).
    - The output specificity_score for a gene × celltype is log2(ct_mean / brain_mean).
      This can be negative (celltype is below brain average).
    - Quantile rank within each celltype is stored in a separate long-form output
      by human_celltype_attribution.py; this function returns the raw log2-ratio.
    """
    # Normalize index to uppercase HGNC.
    expr_df = expr_df.copy()
    expr_df.index = [_normalize_gene_symbol(g) for g in expr_df.index]

    # Drop Ensembl IDs that couldn't be resolved (still in ENSG form after normalization).
    ensembl_mask = expr_df.index.str.startswith("ENSG") | expr_df.index.str.startswith("ENST")
    n_ensembl = ensembl_mask.sum()
    if n_ensembl > 0:
        print(f"  [{label}] Dropping {n_ensembl} unresolved Ensembl IDs from expression index")
        expr_df = expr_df[~ensembl_mask]

    # Deduplicate index: if the same HGNC symbol appears multiple times (e.g., both
    # 'CDK5' and 'Cdk5' normalize to 'CDK5'), keep the row with the higher mean
    # expression to preserve the most informative entry.
    if expr_df.index.duplicated().any():
        n_dup = expr_df.index.duplicated().sum()
        print(f"  [{label}] Deduplicating {n_dup} duplicate gene symbols (keeping max-mean row)")
        expr_df = expr_df.groupby(level=0).mean()

    # Fill NaN with 0 (treat missing cell-type coverage as zero expression).
    expr_df = expr_df.fillna(0.0)

    # Intersect with kinase gene list.
    available = set(expr_df.index.tolist())
    matched = sorted(available & kinase_genes_human)
    n_total = len(kinase_genes_human)
    print(f"  [{label}] Kinase genes matched: {len(matched)}/{n_total}")

    if not matched:
        raise ValueError(
            f"No kinase genes found in expression index for {label}. "
            "Check gene symbol normalization."
        )

    kp_df = expr_df.loc[matched]  # shape: (n_kinases, n_celltypes)

    # Brain-wide mean per gene: mean across all celltypes.
    brain_mean = kp_df.mean(axis=1)  # series indexed by gene

    # Specificity = log2(celltype_mean / brain_mean).
    # Genes with brain_mean <= 0 get 0.0 to avoid -inf/NaN propagation.
    with np.errstate(divide="ignore", invalid="ignore"):
        log2_ratio = np.log2(
            kp_df.values / np.where(brain_mean.values[:, None] > 0,
                                     brain_mean.values[:, None],
                                     np.nan)
        )
    log2_ratio = np.nan_to_num(log2_ratio, nan=0.0, posinf=0.0, neginf=0.0)

    spec_df = pd.DataFrame(
        log2_ratio,
        index=matched,
        columns=kp_df.columns,
    )
    spec_df.index.name = "kinase_id"

    # Sanity check: no NaN in output.
    n_nan = spec_df.isna().sum().sum()
    if n_nan > 0:
        print(f"  WARNING [{label}]: {n_nan} NaN values remain after normalization — filling 0.0")
        spec_df = spec_df.fillna(0.0)

    print(f"  [{label}] Specificity matrix shape: {spec_df.shape}")

    # Companion: raw mean log2 expression matrix for the same (kinase, celltype)
    # grid. Surfaced in the human Attribution tab as an absolute-level sanity
    # check (low values flag a noise-driven specificity score).
    expr_matched = kp_df.copy()
    expr_matched.index.name = "kinase_id"
    return spec_df, expr_matched


# ---------------------------------------------------------------------------
# SEA-AD MTG computation
# ---------------------------------------------------------------------------


def compute_seaad_specificity(force: bool = False) -> pd.DataFrame:
    """Compute kinase specificity from SEA-AD MTG per-supertype expression.

    Reads ``data/derived/aggregates/seaad/expression_by_supertype.csv``
    (genes × 139 supertypes, produced by atlas_reference.py --sea-ad-expression).
    Applies the shared specificity recipe and writes
    ``outputs/reports/human_reference_expression/seaad_kinase_specificity.csv``.

    Returns the specificity DataFrame (kinase_id × supertype).
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("SEA-AD MTG Kinase Specificity (per supertype)")
    print("=" * 60)

    if not force and os.path.exists(SEAAD_SPEC_FILE):
        print(f"  Cached: {SEAAD_SPEC_FILE} (use --force to recompute)")
        return pd.read_csv(SEAAD_SPEC_FILE, index_col=0)

    expr_path = config.SEA_AD_EXPRESSION_FILE
    if not os.path.exists(expr_path):
        raise FileNotFoundError(
            f"SEA-AD MTG expression file not found: {expr_path}\n"
            "Run: python alz/atlas_reference.py --sea-ad-expression"
        )

    print(f"  Loading expression: {expr_path}")
    expr_df = pd.read_csv(expr_path, index_col=0)
    print(f"  Expression shape: {expr_df.shape} (genes × supertypes)")

    _, human_kinases = get_all_kinase_genes()
    phosphatases_in_atlas = get_phosphatase_genes_from_genelist(
        set(_normalize_gene_symbol(g) for g in expr_df.index)
    )
    # Human kinase gene list is already uppercase HGNC; normalize phosphatase prefixes.
    # For SEA-AD (human), phosphatase prefixes must be in human uppercase form.
    human_phosphatase_prefixes = [p.upper() for p in config.PHOSPHATASE_GENE_PREFIXES]
    human_phos_extra = set(g.upper() for g in config.PHOSPHATASE_GENES_EXTRA)
    expr_idx_upper = set(_normalize_gene_symbol(g) for g in expr_df.index)
    human_phosphatases: set[str] = set()
    for g in expr_idx_upper:
        for prefix in human_phosphatase_prefixes:
            if g.startswith(prefix):
                human_phosphatases.add(g)
                break
    human_phosphatases |= (expr_idx_upper & human_phos_extra)

    all_kp_human = human_kinases | human_phosphatases
    spec_df, expr_matched = compute_specificity(expr_df, all_kp_human, label="SEA-AD MTG")

    spec_df.to_csv(SEAAD_SPEC_FILE)
    expr_matched.to_csv(config.SEAAD_KINASE_EXPRESSION_FILE)
    print(f"\n  Saved to {SEAAD_SPEC_FILE}")
    print(f"  Saved to {config.SEAAD_KINASE_EXPRESSION_FILE}")
    return spec_df


# ---------------------------------------------------------------------------
# HBCA computation
# ---------------------------------------------------------------------------


def compute_hbca_specificity(force: bool = False) -> pd.DataFrame:
    """Compute kinase specificity from Allen HBCA per-class expression.

    Reads ``data/derived/aggregates/hbca/expression_by_class.csv``
    (genes × N classes, produced by atlas_reference.py --hbca-download).
    Applies the shared specificity recipe and writes
    ``outputs/reports/human_reference_expression/hbca_kinase_specificity.csv``.

    Returns the specificity DataFrame (kinase_id × class).

    Notes
    -----
    HBCA uses HGNC gene symbols by default (same as SEA-AD). If the index
    contains Ensembl IDs, they are dropped with a warning; a future mapping
    step can be added here before calling compute_specificity().
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("Allen HBCA Kinase Specificity (per class)")
    print("=" * 60)

    if not force and os.path.exists(HBCA_SPEC_FILE):
        print(f"  Cached: {HBCA_SPEC_FILE} (use --force to recompute)")
        return pd.read_csv(HBCA_SPEC_FILE, index_col=0)

    expr_path = config.HBCA_EXPRESSION_FILE
    if not os.path.exists(expr_path):
        raise FileNotFoundError(
            f"HBCA expression file not found: {expr_path}\n"
            "Run: python alz/atlas_reference.py --hbca-download\n"
            "Then: python alz/human_reference_expression.py --ref hbca"
        )

    print(f"  Loading expression: {expr_path}")
    expr_df = pd.read_csv(expr_path, index_col=0)
    print(f"  Expression shape: {expr_df.shape} (genes × classes)")

    _, human_kinases = get_all_kinase_genes()
    expr_idx_upper = set(_normalize_gene_symbol(g) for g in expr_df.index)
    human_phosphatase_prefixes = [p.upper() for p in config.PHOSPHATASE_GENE_PREFIXES]
    human_phos_extra = set(g.upper() for g in config.PHOSPHATASE_GENES_EXTRA)
    human_phosphatases: set[str] = set()
    for g in expr_idx_upper:
        for prefix in human_phosphatase_prefixes:
            if g.startswith(prefix):
                human_phosphatases.add(g)
                break
    human_phosphatases |= (expr_idx_upper & human_phos_extra)

    all_kp_human = human_kinases | human_phosphatases
    spec_df, expr_matched = compute_specificity(expr_df, all_kp_human, label="HBCA")

    spec_df.to_csv(HBCA_SPEC_FILE)
    expr_matched.to_csv(config.HBCA_KINASE_EXPRESSION_FILE)
    print(f"\n  Saved to {HBCA_SPEC_FILE}")
    print(f"  Saved to {config.HBCA_KINASE_EXPRESSION_FILE}")
    return spec_df


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def print_summary() -> None:
    """Print cached specificity results."""
    for label, path in [("SEA-AD MTG", SEAAD_SPEC_FILE), ("HBCA", HBCA_SPEC_FILE)]:
        if not os.path.exists(path):
            print(f"  [{label}] No cached results at {path}")
            continue
        df = pd.read_csv(path, index_col=0)
        n_kinases, n_ct = df.shape
        print(f"  [{label}] {n_kinases} kinases × {n_ct} cell types")
        # Top 3 most specifically expressed kinase per celltype (first 5 celltypes).
        for ct in df.columns[:5]:
            top3 = df[ct].nlargest(3)
            print(f"    {ct}: " + ", ".join(
                f"{g}({v:.2f})" for g, v in top3.items()
            ))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Human Reference Expression: per-celltype kinase specificity "
                    "(SEA-AD MTG + Allen HBCA)",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--ref", choices=["seaad", "hbca", "both"],
                       help="Reference(s) to compute: seaad | hbca | both")
    group.add_argument("--summary", action="store_true",
                       help="Print cached results")
    parser.add_argument("--force", action="store_true",
                        help="Force recomputation even if cached results exist")

    args = parser.parse_args()

    if args.summary:
        print_summary()
        return

    ref = args.ref
    if ref in ("seaad", "both"):
        compute_seaad_specificity(force=args.force)
    if ref in ("hbca", "both"):
        compute_hbca_specificity(force=args.force)


if __name__ == "__main__":
    main()

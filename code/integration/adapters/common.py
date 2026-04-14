"""Shared utilities for integration adapters."""

import os
import sys

import pandas as pd

# Ensure the integration package and main code are importable
_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
sys.path.insert(0, os.path.join(_REPO_ROOT, "code"))
sys.path.insert(0, os.path.join(_REPO_ROOT, "code", "integration"))

import config_integration as icfg  # noqa: E402


def load_kinase_to_gene_mapping() -> dict:
    """Load kinase abbreviation -> gene symbol mapping as a dict.

    Gene symbols are returned in the original case from the mapping file
    (typically human-style uppercase like GSK3A).
    """
    df = pd.read_csv(icfg.KINASE_TO_GENE_CSV)
    col_kin = [c for c in df.columns if "kinase" in c.lower()][0]
    col_gene = [c for c in df.columns if "gene" in c.lower()][0]
    return dict(zip(df[col_kin], df[col_gene]))


def to_mouse_gene_symbol(human_symbol: str) -> str:
    """Convert human-style gene symbol (GSK3A) to mouse-style (Gsk3a).

    Mouse gene nomenclature: first letter uppercase, rest lowercase.
    Exceptions: genes starting with digits are left as-is.
    """
    if not human_symbol or human_symbol[0].isdigit():
        return human_symbol
    return human_symbol[0].upper() + human_symbol[1:].lower()


def load_kinase_to_mouse_gene_mapping() -> dict:
    """Load kinase abbreviation -> mouse gene symbol mapping.

    Returns mouse-style gene symbols (Gsk3a, Map4k4) for compatibility
    with Incytr's kldata and IncytrDB which use mouse nomenclature.
    """
    human_map = load_kinase_to_gene_mapping()
    return {k: to_mouse_gene_symbol(v) for k, v in human_map.items()}


def load_mouse_gene_to_kinase_mapping() -> dict:
    """Load mouse gene symbol -> set of kinase abbreviations.

    Reverse of load_kinase_to_mouse_gene_mapping(). Multiple abbreviations
    can map to the same gene, so values are sets.
    """
    abbrev_to_mouse = load_kinase_to_mouse_gene_mapping()
    gene_to_kins = {}
    for abbrev, gene in abbrev_to_mouse.items():
        gene_to_kins.setdefault(gene, set()).add(abbrev)
    return gene_to_kins


def build_substrate_kinase_map(kldata) -> dict:
    """Build substrate_gene -> set(kinase_mouse_genes) from kldata DataFrame.

    kldata must have 'gene' (substrate) and 'motif.geneName' (kinase) columns.
    """
    return kldata.groupby("gene")["motif.geneName"].apply(set).to_dict()


def sanitize_celltype_name(name: str) -> str:
    """Convert canonical cell-type name to directory-safe form.

    Matches the R pipeline convention: spaces to underscores, slashes to
    hyphens.  E.g. ``"L2/3 IT"`` -> ``"L2-3_IT"``.
    """
    return name.replace(" ", "_").replace("/", "-")


def ensure_intermediates_dir():
    """Create intermediates directory if it doesn't exist."""
    os.makedirs(icfg.INTERMEDIATES_DIR, exist_ok=True)

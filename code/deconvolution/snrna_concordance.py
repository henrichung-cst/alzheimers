"""Stage 4: snRNA cross-check via kinase gene LFC concordance.

For each row of the Stage 3 (cluster, contrast, kinase) MEA output,
look up the kinase's own gene LFC in the existing
``outputs/reports/snrna_integration/song_concordance.csv`` (males-only
factorial OLS at WMB-class level) by mapping the 46-cluster cell type
through ``yuyu_46_to_wmb_class.csv``.

The WMB-class join is intentional: the live snRNA concordance is at
34 WMB classes, and the snRNA was reclustered at 46 Yuyu clusters
without per-cell labels exported here. Joining at the WMB-class level
gives faithful transcript-level corroboration at the matching
biological resolution; "Unclassified" 46-clusters get NaN snRNA values.

Annotates three columns:
  - kinase_gene_LFC_snRNA   (signed magnitude)
  - kinase_gene_FDR_snRNA   (significance)
  - direction_match         ("match"/"opposite"/"flat"/"n/a")
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

import config
from deconvolution import paths


def _to_mouse_symbol(human_sym: str) -> str:
    """Convert AKT3-style human symbols to Akt3-style mouse symbols.

    Most mouse orthologs follow the convention "first letter uppercase,
    rest lowercase". This is a heuristic; rare exceptions exist but
    are not load-bearing for the cross-check (an unmatched gene becomes
    n/a, which the confidence model handles).
    """
    if not isinstance(human_sym, str) or not human_sym:
        return human_sym
    return human_sym[0].upper() + human_sym[1:].lower()


def _load_kinase_to_gene() -> dict:
    if not os.path.exists(config.MAPPING_CACHE_FILE):
        print(f"  WARN: kinase->gene cache not found at {config.MAPPING_CACHE_FILE}; "
              "treating kinase abbreviation as gene symbol.")
        return {}
    k2g = pd.read_csv(config.MAPPING_CACHE_FILE)
    # snRNA stores mouse symbols (e.g. "Gsk3b"); the live cache stores
    # human-style uppercase ("GSK3B"). Convert here so the join works.
    return {
        k: _to_mouse_symbol(g)
        for k, g in zip(k2g["kinase_abbreviation"], k2g["gene_symbol"])
    }


def _direction_match_vec(nes: np.ndarray, lfc: np.ndarray, fdr: np.ndarray,
                         lfc_flat: float, fdr_high: float) -> np.ndarray:
    out = np.full(len(nes), "n/a", dtype=object)
    snrna_finite = np.isfinite(lfc) & np.isfinite(fdr)
    flat = snrna_finite & ((np.abs(lfc) < lfc_flat) | (fdr >= fdr_high))
    out[flat] = "flat"
    decided = snrna_finite & ~flat & np.isfinite(nes)
    same_sign = np.sign(nes) == np.sign(lfc)
    out[decided & same_sign] = "match"
    out[decided & ~same_sign] = "opposite"
    return out


def annotate(mea_df: pd.DataFrame, mapping_df: pd.DataFrame) -> pd.DataFrame:
    """Join MEA table with snRNA gene LFC; return augmented DataFrame."""
    if not os.path.exists(config.SONG_CONCORDANCE_FILE):
        raise FileNotFoundError(
            f"snRNA concordance file not found: {config.SONG_CONCORDANCE_FILE}. "
            "Run `python code/snrna_integration.py --concordance` first."
        )
    snrna = pd.read_csv(config.SONG_CONCORDANCE_FILE)
    # Keep only the columns we need
    snrna = snrna[["gene_symbol", "cell_type", "pathway", "song_lfc",
                   "song_pval", "song_fdr"]].copy()
    snrna = snrna.rename(columns={"cell_type": "wmb_class"})

    k2g = _load_kinase_to_gene()
    cluster_to_wmb = dict(zip(mapping_df["cluster_name"], mapping_df["wmb_class"]))

    df = mea_df.copy()
    df["gene_symbol"] = df["kinase"].map(
        lambda k: k2g.get(k, _to_mouse_symbol(k))
    )
    df["wmb_class"] = df["cluster"].map(cluster_to_wmb)
    df["pathway"] = df["contrast"].map(paths.CONTRAST_TO_PATHWAY)

    merged = df.merge(
        snrna,
        how="left",
        on=["gene_symbol", "wmb_class", "pathway"],
        suffixes=("", "_snrna"),
    )
    merged = merged.rename(columns={
        "song_lfc": "kinase_gene_LFC_snRNA",
        "song_fdr": "kinase_gene_FDR_snRNA",
        "song_pval": "kinase_gene_pval_snRNA",
    })

    nes = merged["NES"].to_numpy(dtype=float) if "NES" in merged.columns \
        else np.full(len(merged), np.nan)
    merged["direction_match"] = _direction_match_vec(
        nes,
        merged["kinase_gene_LFC_snRNA"].to_numpy(dtype=float),
        merged["kinase_gene_FDR_snRNA"].to_numpy(dtype=float),
        paths.SNRNA_LFC_FLAT,
        paths.SNRNA_FDR_HIGH,
    )
    return merged

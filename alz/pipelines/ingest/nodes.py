"""Kedro node adapters for the ingest (P1) pipeline.

Thin wrappers over the pure ``build_*`` cores in ``alz.ingest.song``: the
catalog supplies pre-loaded inputs and persists the returns, while those same
cores stay callable from ``song.py``'s CLI (``--mapping`` / ``--phospho-match``
/ ``--run``). Single implementation, two callers. See
docs/foundation/live_pipeline_contract.md.
"""
from __future__ import annotations

import pandas as pd

from alz.ingest import song


def sample_mapping(song_sample_list: pd.DataFrame,
                   song_total_proteome: pd.DataFrame) -> pd.DataFrame:
    """§1: TMT channel -> animal mapping, cross-referenced to snRNA-seq.

    The snRNA manifest is optional (returns {} when absent), so it is read
    inside the node rather than cataloged.
    """
    snrna_samples = song._discover_snrna_samples()
    return song.build_sample_mapping(
        song_sample_list, song_total_proteome.columns, snrna_samples)


def phospho_match(song_total_proteome: pd.DataFrame,
                  song_imac_sitequant: pd.DataFrame,
                  song_imac_composite: pd.DataFrame):
    """§2: match phosphosite parent proteins to the total proteome.

    Returns (matching_df, gene_list_text, summary) mapping to the three
    catalog outputs. The proteome gene list is emitted as newline-joined text
    (TextDataset) to match the legacy ``total_proteome_genes.txt`` contract
    consumed by the WMB reference build.
    """
    matching_df, gene_list, summary = song.build_phospho_matching(
        song_total_proteome, song_imac_sitequant, song_imac_composite)
    gene_list_text = "".join(f"{g}\n" for g in gene_list)
    return matching_df, gene_list_text, summary

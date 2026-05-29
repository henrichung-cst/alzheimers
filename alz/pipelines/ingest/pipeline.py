"""P1 — ingest (song / mouse) pipeline definition.

Two pure nodes (Phase 2 scope): sample_mapping and phospho_match. Outlier
exclusions move to bulk_mea (P2, after normalize); the §3 quality diagnostic
and the human (mukesh) ingest stay out (decided 2026-05-27).
"""
from __future__ import annotations

from kedro.pipeline import Pipeline, node, pipeline

from . import nodes


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=nodes.sample_mapping,
                inputs=["song_sample_list", "song_total_proteome"],
                outputs="song_sample_mapping",
                name="sample_mapping",
            ),
            node(
                func=nodes.phospho_match,
                inputs=[
                    "song_total_proteome",
                    "song_imac_sitequant",
                    "song_imac_composite",
                ],
                outputs=[
                    "song_phospho_protein_matching",
                    "song_proteome_gene_list",
                    "song_matching_summary",
                ],
                name="phospho_match",
            ),
        ]
    )

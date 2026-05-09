from kedro.pipeline import Pipeline, node

from .nodes import build_tmt_channel_mapping


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        node(
            func=build_tmt_channel_mapping,
            inputs=[
                "song_tmt_layout",
                "song_total_proteome_columns",
                "params:proof_marker",
            ],
            outputs="sample_mapping",
            name="build_tmt_channel_mapping",
        ),
    ])

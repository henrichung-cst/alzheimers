from kedro.pipeline import Pipeline, node, pipeline as modular_pipeline

from .nodes import classify_mechanisms, mea_raw_phospho_track, merge_into_unified


def _track_template() -> Pipeline:
    return Pipeline([
        node(
            func=mea_raw_phospho_track,
            inputs=["raw_phospho_normalized", "sample_mapping",
                    "params:analysis_mode", "params:track"],
            outputs="mea_raw_phospho",
            name="mea_raw_phospho_track",
        ),
    ])


def _track(namespace: str, track_param: str) -> Pipeline:
    return modular_pipeline(
        _track_template(),
        namespace=namespace,
        inputs={"sample_mapping"},
        parameters={
            "params:analysis_mode": "params:analysis_mode",
            "params:track": f"params:{track_param}",
        },
    )


def create_pipeline(**kwargs) -> Pipeline:
    track_pipelines = _track("st", "track_st") + _track("py", "track_py")
    combine = Pipeline([
        node(
            func=classify_mechanisms,
            inputs=["st.mea_raw_phospho", "py.mea_raw_phospho",
                    "st.mea_stoichiometry", "py.mea_stoichiometry"],
            outputs="mechanism_annotation",
            name="classify_mechanisms",
        ),
        node(
            func=merge_into_unified,
            inputs=["unified_attribution", "mechanism_annotation"],
            outputs="unified_attribution_with_mechanism",
            name="merge_into_unified",
        ),
    ])
    return track_pipelines + combine

"""Track-namespaced normalize pipeline (st + py).

Single template instantiated twice via Kedro's modular `pipeline()` factory.
Shared inputs (`sample_mapping`, `total_proteome_xlsx`) flow through
unprefixed; the per-track phospho sitequant Excel and the four outputs
resolve via namespace prefixing to catalog keys `st.<name>` / `py.<name>`.
"""

from kedro.pipeline import Pipeline, node, pipeline as modular_pipeline

from .nodes import normalize_track


def _normalize_template() -> Pipeline:
    return Pipeline([
        node(
            func=normalize_track,
            inputs=["sample_mapping", "total_proteome_xlsx",
                    "phospho_sitequant_xlsx", "params:track"],
            outputs=["stoichiometry_matrix", "raw_phospho_normalized",
                     "stoichiometry_qc", "normalization_summary"],
            name="normalize_track",
        ),
    ])


def _track(namespace: str, track_param: str) -> Pipeline:
    return modular_pipeline(
        _normalize_template(),
        namespace=namespace,
        inputs={"sample_mapping", "total_proteome_xlsx"},
        parameters={"params:track": f"params:{track_param}"},
    )


def create_pipeline(**kwargs) -> Pipeline:
    return _track("st", "track_st") + _track("py", "track_py")

"""Track-namespaced enrich pipeline (st + py).

Single template instantiated twice via Kedro's modular `pipeline()` factory.
Shared inputs (`sample_mapping`, params) are mapped through; per-track inputs
(`stoichiometry_matrix`, `raw_phospho_normalized`) and outputs resolve via
namespace prefixing to catalog keys `st.<name>` / `py.<name>`.
"""

from kedro.pipeline import Pipeline, node, pipeline as modular_pipeline

from .nodes import filter_samples, fit_and_contrast, run_mea


def _enrich_template() -> Pipeline:
    return Pipeline([
        node(
            func=filter_samples,
            inputs=["sample_mapping", "params:analysis_mode",
                    "params:sample_exclusions_path"],
            outputs="filtered_mapping",
            name="filter_samples",
        ),
        node(
            func=fit_and_contrast,
            inputs=["stoichiometry_matrix", "raw_phospho_normalized",
                    "filtered_mapping", "params:analysis_mode"],
            outputs=["site_level_ols", "results_by_contrast"],
            name="fit_and_contrast",
        ),
        node(
            func=run_mea,
            inputs=["stoichiometry_matrix", "results_by_contrast",
                    "params:track"],
            outputs=["mea_stoichiometry", "mea_global_shift",
                     "winsorized_sites", "mea_substrate_sets"],
            name="run_mea",
        ),
    ])


def _track(namespace: str, track_param: str) -> Pipeline:
    return modular_pipeline(
        _enrich_template(),
        namespace=namespace,
        inputs={"sample_mapping"},
        parameters={
            "params:analysis_mode": "params:analysis_mode",
            "params:sample_exclusions_path": "params:sample_exclusions_path",
            "params:track": f"params:{track_param}",
        },
    )


def create_pipeline(**kwargs) -> Pipeline:
    return _track("st", "track_st") + _track("py", "track_py")

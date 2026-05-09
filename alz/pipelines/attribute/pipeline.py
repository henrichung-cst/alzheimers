"""Single-namespace attribute pipeline.

Consumes both phospho tracks via `st.mea_stoichiometry` + `py.mea_stoichiometry`
catalog keys produced by the enrich pipeline. SEA-AD h5ads and the optional
Song / WMB inputs flow through parameter-injected paths because Kedro's
built-in datasets don't cover anndata or "may-not-exist" files.
"""

from kedro.pipeline import Pipeline, node

from .nodes import assemble_unified, combine_mea_tracks, sea_ad_concordance


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        node(
            func=combine_mea_tracks,
            inputs=["st.mea_stoichiometry", "py.mea_stoichiometry",
                    "kinase_to_gene_mapping"],
            outputs="mea_combined",
            name="combine_mea_tracks",
        ),
        node(
            func=sea_ad_concordance,
            inputs=["mea_combined", "seaad_to_wmb_class",
                    "params:sea_ad_paths"],
            outputs=["sea_ad_concordance_df", "sea_ad_supertype_lfc"],
            name="sea_ad_concordance",
        ),
        node(
            func=assemble_unified,
            inputs=["mea_combined", "sea_ad_concordance_df",
                    "params:wmb_expression_path",
                    "params:song_specificity_path",
                    "params:song_concordance_path"],
            outputs=["unified_attribution", "unified_attribution_full",
                     "attribution_summary"],
            name="assemble_unified",
        ),
    ])

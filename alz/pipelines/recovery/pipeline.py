"""Single-namespace attribution-recovery pipeline.

Combines per-track MEA stoichiometry and assembles the three hypothesis
tables (kinase activity matrix, cell-type evidence, kinase hypothesis
table). All compute lives in pure helpers in `alz.attribution_recovery`.
"""

from kedro.pipeline import Pipeline, node

from .nodes import combine_mea_stoichiometry, compute_recovery_tables


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        node(
            func=combine_mea_stoichiometry,
            inputs=["st.mea_stoichiometry", "py.mea_stoichiometry"],
            outputs="mea_stoichiometry_combined",
            name="combine_mea_stoichiometry",
        ),
        node(
            func=compute_recovery_tables,
            inputs=["mea_stoichiometry_combined",
                    "unified_attribution_full"],
            outputs=["kinase_activity_matrix",
                     "celltype_evidence_table",
                     "kinase_hypothesis_table"],
            name="compute_recovery_tables",
        ),
    ])

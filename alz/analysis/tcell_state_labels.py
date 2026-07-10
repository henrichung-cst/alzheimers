#!/usr/bin/env python3
"""Assign definitive, evidence-backed T-cell state labels per cell.

Lineage is anchored by the CITE-seq CD4/CD8 antibody counts and falls back to
the native-cluster consensus only when neither antibody exceeds the observed
mouse-isotype background. State labels use conventional biological names:

* ``CD8 exhausted`` requires co-detection of checkpoint-associated transcripts;
* ``CD8 precursor exhausted`` additionally requires a coherent TCF7-positive
  memory program;
* memory and cytotoxic states require their corresponding RNA programs;
* proliferation is assigned from the per-cell S/G2M call.

The names are operational state calls for this chronic-stimulation exhaustion
experiment, not proof of functional dysfunction. ``Terminal exhaustion`` is
deliberately not used because many checkpoint-positive cells are proliferating.
Independent ProjecTILs calls are retained as reference evidence and summarized
with categorical confidence semantics. The output retains raw RNA, antibody,
and projection evidence with units; no internal marker count is exported as an
analysis score.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


DONORS = ("donor1", "donor2")
CHECKPOINT_MARKERS = ("HAVCR2", "LAG3", "ENTPD1", "PDCD1")
SUPPORTING_CHECKPOINT_MARKERS = ("TOX", "TIGIT", "CTLA4")
PRECURSOR_RESTING_MARKERS = ("TCF7", "LEF1", "SELL", "CCR7", "IL7R")
MEMORY_COMPANION_MARKERS = ("LEF1", "SELL", "CCR7", "IL7R")
CYTOTOXIC_MARKERS = ("GZMB", "GZMH", "GNLY", "PRF1", "EOMES", "NKG7")
GRANZYME_MARKERS = ("GZMB", "GZMH", "GNLY")
LINEAGE_RNA_MARKERS = ("CD3D", "CD3E", "TRAC", "CD4", "CD8A", "CD8B")
EVIDENCE_MARKERS = tuple(
    dict.fromkeys(
        (
            *CHECKPOINT_MARKERS,
            *SUPPORTING_CHECKPOINT_MARKERS,
            *PRECURSOR_RESTING_MARKERS,
            *CYTOTOXIC_MARKERS,
            *LINEAGE_RNA_MARKERS,
        )
    )
)
DIVIDING_PHASES = {"S", "G2M"}

ADT_EVIDENCE_COLUMNS = (
    "CD3_protein_umi",
    "CD4_protein_umi",
    "CD8_protein_umi",
    "TCF1_protein_umi",
    "Ki67_protein_umi",
    "NCAM1_protein_umi",
    "mouse_isotype_umi",
    "rabbit_isotype_umi",
)
OPTIONAL_ADT_EVIDENCE_COLUMNS = (
    "TOX_protein_umi",
    "BATF_protein_umi",
    "PRDM1_protein_umi",
    "GZMB_protein_umi",
)

REPORT_ROOT = Path("outputs/reports/tcell_labeling")
EXPRESSION_ROOT = REPORT_ROOT / "auroc"
ADT_ROOT = REPORT_ROOT / "adt"
CLUSTER_ROOT = REPORT_ROOT / "clusters"
UMAP_ROOT = REPORT_ROOT / "umap"
OUTPUT_ROOT = REPORT_ROOT / "cells"
PROJECTILS_ROOT = Path("data/derived/tcells_incytr_inputs")


@dataclass(frozen=True)
class ClusterContext:
    lineage: str
    fixed_label: str | None
    description: str


# Cluster-level context remains appropriate for lineage, CD4 state, and QC.
# CD8 clusters intentionally have no fixed state label: their state is resolved
# from per-cell marker detection and cell-cycle phase below.
CLUSTER_CONTEXT = {
    "donor1": {
        0: ClusterContext("CD4", None, "resting-like CD4 cluster"),
        1: ClusterContext("CD4", None, "cycle-dominated T-cell cluster"),
        2: ClusterContext("CD4", None, "cycle-dominated T-cell cluster"),
        3: ClusterContext("CD4", None, "naive-like CD4 cluster"),
        4: ClusterContext("CD8", None, "late CD8; state resolved per cell"),
        5: ClusterContext("CD4", None, "activation-associated T-cell cluster"),
        6: ClusterContext("CD8", None, "late CD8; state resolved per cell"),
        7: ClusterContext("CD4", None, "cycle-dominated T-cell cluster"),
        8: ClusterContext("CD8", None, "day0 baseline cytotoxic CD8"),
        9: ClusterContext("CD8", None, "late CD8; state resolved per cell"),
        10: ClusterContext("CD4", None, "cycle-dominated T-cell cluster"),
        11: ClusterContext("CD8", None, "day2 activated CD8"),
        12: ClusterContext("contaminant", "contaminant", "myeloid / antigen-presenting (not a T cell)"),
        13: ClusterContext("contaminant", "contaminant", "NK / gamma-delta (not a T cell)"),
    },
    "donor2": {
        0: ClusterContext("CD8", None, "late CD8; state resolved per cell"),
        1: ClusterContext("CD4", None, "cycle-dominated T-cell cluster"),
        2: ClusterContext("CD4", None, "stress-associated T-cell cluster"),
        3: ClusterContext("CD4", None, "activation-associated T-cell cluster"),
        4: ClusterContext("CD8", None, "late CD8; state resolved per cell"),
        5: ClusterContext("CD8", None, "late CD8; state resolved per cell"),
        6: ClusterContext("CD4", None, "resting-like CD4 cluster"),
        7: ClusterContext("CD8", None, "day2 activated CD8"),
        8: ClusterContext("CD8", None, "late CD8; state resolved per cell"),
        9: ClusterContext("CD8", None, "dividing CD8"),
        10: ClusterContext("CD4", None, "activation-associated T-cell cluster"),
        11: ClusterContext("contaminant", "contaminant", "myeloid (not a T cell)"),
        12: ClusterContext("contaminant", "contaminant", "mast (not a T cell)"),
    },
}

ANALYSIS_TYPES = {
    "CD8 precursor exhausted": "CD8PrecursorExhausted",
    "CD8 exhausted": "CD8Exhausted",
    "CD8 memory": "CD8Memory",
    "CD8 cytotoxic": "CD8Cytotoxic",
    "CD8 effector": "CD8Effector",
    "CD4 proliferating": "CD4Proliferating",
    "CD4 memory": "CD4Memory",
    "CD4 cytotoxic": "CD4Cytotoxic",
    "CD4 resting": "CD4Resting",
}


def _assert_unique(df: pd.DataFrame, name: str) -> None:
    if df["barcode"].duplicated().any():
        examples = df.loc[df["barcode"].duplicated(), "barcode"].head().tolist()
        raise ValueError(f"{name} has duplicate barcodes: {examples}")


def classify_lineage(row: pd.Series, cluster_lineage: str) -> tuple[str, str]:
    """Resolve CD4/CD8 identity from raw ADT counts with cluster fallback."""
    if cluster_lineage == "contaminant":
        return "contaminant", "cluster contaminant"

    cd4 = row["CD4_protein_umi"]
    cd8 = row["CD8_protein_umi"]
    background = row["mouse_isotype_umi"]
    if cd4 > cd8 and cd4 > background:
        return "CD4", "ADT CD4-dominant"
    if cd8 > cd4 and cd8 > background:
        return "CD8", "ADT CD8-dominant"
    return cluster_lineage, "cluster fallback"


def _state_evidence_calls(row: pd.Series) -> tuple[bool, bool, bool]:
    checkpoint_coexpressed = sum(row[gene] > 0 for gene in CHECKPOINT_MARKERS) >= 2
    memory_like = row["TCF7"] > 0 and any(
        row[gene] > 0 for gene in MEMORY_COMPANION_MARKERS
    )
    cytotoxic_machinery = row["PRF1"] > 0 and any(
        row[gene] > 0 for gene in GRANZYME_MARKERS
    )
    return checkpoint_coexpressed, memory_like, cytotoxic_machinery


def classify_cd8_state(row: pd.Series) -> str:
    """Return a donor-agnostic, operational CD8 biological state."""
    checkpoint_coexpressed, memory_like, cytotoxic_machinery = _state_evidence_calls(row)
    if checkpoint_coexpressed and memory_like:
        return "CD8 precursor exhausted"
    if checkpoint_coexpressed:
        return "CD8 exhausted"
    if memory_like:
        return "CD8 memory"
    if cytotoxic_machinery:
        return "CD8 cytotoxic"
    return "CD8 effector"


def classify_cd4_state(row: pd.Series) -> str:
    """Return a donor-agnostic, operational CD4 biological state."""
    if row["Phase"] in DIVIDING_PHASES:
        return "CD4 proliferating"
    _, memory_like, cytotoxic_machinery = _state_evidence_calls(row)
    if memory_like:
        return "CD4 memory"
    if cytotoxic_machinery:
        return "CD4 cytotoxic"
    return "CD4 resting"


def classify_projectils_quality(state: object, confidence: object) -> str:
    """Translate ProjecTILs neighborhood confidence into semantic categories.

    ProjecTILs confidence is the winning class's weight among five reference
    neighbors. Exactly one therefore means unanimous weighted-neighborhood
    support, while greater than one half has the defensible meaning of majority
    support. No arbitrary high-confidence threshold is introduced.
    """
    if pd.isna(state) or pd.isna(confidence):
        return "not projected"
    confidence = float(confidence)
    if confidence == 1.0:
        return "unanimous"
    if confidence > 0.5:
        return "majority-supported"
    return "ambiguous"


def exhaustion_corroboration(
    *,
    label: str,
    lineage: str,
    projectils_state: object,
    projectils_quality: str,
) -> str:
    """Describe independent reference support without changing the RNA/ADT call."""
    if label not in {"CD8 exhausted", "CD8 precursor exhausted"}:
        return "not applicable"
    if pd.isna(projectils_state) or projectils_quality == "not projected":
        return "not projected"
    state = str(projectils_state)
    if lineage != "CD8" or not state.startswith("CD8."):
        return "lineage discordant"
    if state == "CD8.TEX":
        if projectils_quality == "unanimous":
            return "unanimous CD8.TEX"
        return "CD8.TEX reference support"
    if state == "CD8.TPEX":
        return "CD8.TPEX reference support"
    return "state discordant"


def _load_inputs(
    donor: str,
    expression_root: Path,
    adt_root: Path,
    projectils_root: Path,
) -> pd.DataFrame:
    expr = pd.read_csv(
        expression_root / f"{donor}_marker_cell_expr.csv",
        usecols=["barcode", *EVIDENCE_MARKERS],
    )
    adt = pd.read_csv(adt_root / f"{donor}_adt_evidence.csv")
    missing_adt_columns = set(ADT_EVIDENCE_COLUMNS) - set(adt.columns)
    if missing_adt_columns:
        raise ValueError(
            f"{donor} ADT evidence is missing required columns: "
            f"{sorted(missing_adt_columns)}"
        )
    for column in OPTIONAL_ADT_EVIDENCE_COLUMNS:
        if column not in adt.columns:
            adt[column] = pd.NA
    adt = adt[["barcode", *ADT_EVIDENCE_COLUMNS, *OPTIONAL_ADT_EVIDENCE_COLUMNS]]
    cc = pd.read_csv(
        CLUSTER_ROOT / f"{donor}_cc_recluster_cells.csv",
        usecols=["barcode", "old_cluster", "Phase"],
    )
    coords = pd.read_csv(
        UMAP_ROOT / f"{donor}_native_umap_coords.csv",
        usecols=["barcode", "seurat_clusters", "day_label"],
    )
    projectils = pd.read_csv(
        projectils_root / donor / "scrna" / "projectils_predictions.csv",
        usecols=[
            "barcode",
            "lineage_gate",
            "functional.cluster",
            "functional.cluster.conf",
        ],
    ).rename(
        columns={
            "lineage_gate": "projectils_lineage",
            "functional.cluster": "projectils_state",
            "functional.cluster.conf": "projectils_confidence",
        }
    )
    for name, frame in (
        ("expression", expr),
        ("ADT", adt),
        ("cell-cycle", cc),
        ("UMAP", coords),
        ("ProjecTILs", projectils),
    ):
        _assert_unique(frame, f"{donor} {name}")

    df = coords.merge(cc, on="barcode", how="inner", validate="one_to_one")
    df = df.merge(expr, on="barcode", how="inner", validate="one_to_one")
    df = df.merge(adt, on="barcode", how="inner", validate="one_to_one")
    df = df.merge(projectils, on="barcode", how="left", validate="one_to_one")
    expected = len(coords)
    if (
        len(df) != expected
        or len(cc) != expected
        or len(expr) != expected
        or len(adt) != expected
        or len(projectils) != expected
    ):
        raise ValueError(
            f"{donor} cell accounting mismatch: UMAP={len(coords)}, "
            f"cell-cycle={len(cc)}, expression={len(expr)}, ADT={len(adt)}, "
            f"ProjecTILs={len(projectils)}, joined={len(df)}"
        )
    if not (df["seurat_clusters"].astype(int) == df["old_cluster"].astype(int)).all():
        raise ValueError(f"{donor} cluster assignments disagree between inputs")
    if not set(df["Phase"]).issubset({"G1", *DIVIDING_PHASES}):
        raise ValueError(f"{donor} has unexpected cell-cycle phases: {sorted(df['Phase'].unique())}")

    df["seurat_cluster"] = df["seurat_clusters"].astype(int)
    df["day"] = df["day_label"].str.extract(r"Day_(\d+)", expand=False).astype(int)
    return df


def _assign(donor: str, df: pd.DataFrame) -> pd.DataFrame:
    context = CLUSTER_CONTEXT[donor]
    observed = set(df["seurat_cluster"])
    if observed != set(context):
        raise ValueError(
            f"{donor} cluster map mismatch: missing={sorted(observed - set(context))}, "
            f"extra={sorted(set(context) - observed)}"
        )

    df = df.copy()
    df["cluster_lineage"] = df["seurat_cluster"].map(
        lambda cluster: context[cluster].lineage
    )
    lineage_calls = df.apply(
        lambda row: classify_lineage(row, row["cluster_lineage"]), axis=1
    )
    df["lineage"] = lineage_calls.map(lambda call: call[0])
    df["lineage_source"] = lineage_calls.map(lambda call: call[1])
    df["label"] = "contaminant"
    is_cd8 = df["lineage"] == "CD8"
    is_cd4 = df["lineage"] == "CD4"
    df.loc[is_cd8, "label"] = df.loc[is_cd8].apply(classify_cd8_state, axis=1)
    df.loc[is_cd4, "label"] = df.loc[is_cd4].apply(classify_cd4_state, axis=1)
    df["proliferation"] = df["Phase"].map(
        lambda phase: "cycling" if phase in DIVIDING_PHASES else "G1-scored"
    )
    tcell = is_cd4 | is_cd8
    tcell_calls = df.loc[tcell].apply(_state_evidence_calls, axis=1)
    cd8_calls = tcell_calls.loc[is_cd8[is_cd8].index]
    df["checkpoint_status"] = "not applicable"
    df["memory_status"] = "not applicable"
    df["cytotoxic_status"] = "not applicable"
    df.loc[is_cd8, "checkpoint_status"] = cd8_calls.map(
        lambda call: "coexpressed" if call[0] else "not coexpressed"
    )
    df.loc[tcell, "memory_status"] = tcell_calls.map(
        lambda call: "memory-like supported" if call[1] else "not supported"
    )
    df.loc[tcell, "cytotoxic_status"] = tcell_calls.map(
        lambda call: "cytotoxic machinery detected" if call[2] else "not detected"
    )
    df["type"] = df["label"].map(ANALYSIS_TYPES)
    df["projectils_quality"] = df.apply(
        lambda row: classify_projectils_quality(
            row["projectils_state"], row["projectils_confidence"]
        ),
        axis=1,
    )
    df["projectils_lineage_agreement"] = "not projected"
    projected = df["projectils_state"].notna()
    df.loc[projected, "projectils_lineage_agreement"] = (
        df.loc[projected, "projectils_lineage"] == df.loc[projected, "lineage"]
    ).map({True: "agreement", False: "discordant"})
    df["exhaustion_corroboration"] = df.apply(
        lambda row: exhaustion_corroboration(
            label=row["label"],
            lineage=row["lineage"],
            projectils_state=row["projectils_state"],
            projectils_quality=row["projectils_quality"],
        ),
        axis=1,
    )

    if df["label"].isna().any():
        raise ValueError(f"{donor} has unlabeled cells")
    if df.loc[df["label"] == "contaminant", "type"].notna().any():
        raise ValueError(f"{donor} contaminants unexpectedly have an Incytr type")
    if df.loc[df["label"] != "contaminant", "type"].isna().any():
        raise ValueError(f"{donor} non-contaminants are missing an Incytr type")

    if (df.loc[df["label"] == "CD4 proliferating", "Phase"] == "G1").any():
        raise ValueError(f"{donor} assigned a G1-scored cell to CD4 proliferating")
    checkpoint_labels = df["label"].isin(
        ["CD8 precursor exhausted", "CD8 exhausted"]
    )
    checkpoint_evidence = (
        df.loc[checkpoint_labels, list(CHECKPOINT_MARKERS)] > 0
    ).sum(axis=1)
    if (checkpoint_evidence < 2).any():
        raise ValueError(f"{donor} assigned exhausted CD8 without checkpoint coexpression")

    rename = {gene: f"{gene}_log_normalized_expression" for gene in EVIDENCE_MARKERS}
    df = df.rename(columns=rename)
    columns = [
        "barcode",
        "donor",
        "seurat_cluster",
        "day",
        "lineage",
        "lineage_source",
        "Phase",
        "proliferation",
        "checkpoint_status",
        "memory_status",
        "cytotoxic_status",
        "projectils_lineage",
        "projectils_state",
        "projectils_confidence",
        "projectils_quality",
        "projectils_lineage_agreement",
        "exhaustion_corroboration",
        *rename.values(),
        *ADT_EVIDENCE_COLUMNS,
        *OPTIONAL_ADT_EVIDENCE_COLUMNS,
        "label",
        "type",
    ]
    df["donor"] = donor
    return df[columns].sort_values("barcode").reset_index(drop=True)


def _write_cluster_context(output_root: Path) -> None:
    rows = []
    for donor, clusters in CLUSTER_CONTEXT.items():
        for cluster, context in clusters.items():
            rows.append(
                {
                    "donor": donor,
                    "cluster": cluster,
                    "lineage": context.lineage,
                    "fixed_label": context.fixed_label,
                    "description": context.description,
                }
            )
    pd.DataFrame(rows).to_csv(output_root / "cluster_context.csv", index=False)


def _print_summary(donor: str, labels: pd.DataFrame) -> None:
    summary = (
        labels.groupby(["label", "proliferation"], dropna=False)
        .size()
        .unstack(fill_value=0)
        .sort_index()
    )
    print(f"[{donor}] wrote {len(labels)} cell labels")
    print(summary.to_string())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--donor", choices=(*DONORS, "all"), default="all")
    parser.add_argument("--expression-root", type=Path, default=EXPRESSION_ROOT)
    parser.add_argument("--adt-root", type=Path, default=ADT_ROOT)
    parser.add_argument("--projectils-root", type=Path, default=PROJECTILS_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    donors = DONORS if args.donor == "all" else (args.donor,)
    for donor in donors:
        labels = _assign(
            donor,
            _load_inputs(
                donor,
                args.expression_root,
                args.adt_root,
                args.projectils_root,
            ),
        )
        labels.to_csv(args.output_root / f"{donor}_state_labels.csv", index=False)
        _print_summary(donor, labels)
    _write_cluster_context(args.output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

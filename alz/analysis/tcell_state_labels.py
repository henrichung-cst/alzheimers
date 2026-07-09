#!/usr/bin/env python3
"""Assign evidence-backed T-cell state labels at the single-cell level.

CD4 and contaminant identities are fixed by their Seurat cluster. CD8 state is
assigned per cell because the native clusters mix proliferation with the
inhibitory program:

* terminal/exhausted cells must be G1 and detect at least two of HAVCR2, LAG3,
  ENTPD1, and PDCD1;
* donor2 TPEX cells detect TCF7 plus at least one inhibitory marker (and may be
  cycling); donor1 has no supported TPEX/TEX split;
* the remaining CD8 cells are cytotoxic/activated effectors.

The output deliberately retains the raw log-normalized expression values
and the categorical cell-cycle call. The internal marker-count gate is not
exported as an analysis score.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


DONORS = ("donor1", "donor2")
INHIBITORY_MARKERS = ("HAVCR2", "LAG3", "ENTPD1", "PDCD1")
PRECURSOR_RESTING_MARKERS = ("TCF7", "LEF1", "SELL", "CCR7", "IL7R")
EVIDENCE_MARKERS = (*INHIBITORY_MARKERS, "TOX", *PRECURSOR_RESTING_MARKERS)
DIVIDING_PHASES = {"S", "G2M"}

REPORT_ROOT = Path("outputs/reports/tcell_labeling")
EXPRESSION_ROOT = REPORT_ROOT / "auroc"
CLUSTER_ROOT = REPORT_ROOT / "clusters"
UMAP_ROOT = REPORT_ROOT / "umap"
OUTPUT_ROOT = REPORT_ROOT / "cells"


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
        0: ClusterContext("CD4", "CD4 resting", "resting CD4 (low TCF7, non-dividing)"),
        1: ClusterContext("CD4", "CD4 proliferating", "dividing CD4 (+ metabolic stress)"),
        2: ClusterContext("CD4", "CD4 proliferating", "dividing CD4"),
        3: ClusterContext("CD4", "CD4 naive", "naive CD4 (TCF7-high, day0 isolated)"),
        4: ClusterContext("CD8", None, "late CD8; state resolved per cell"),
        5: ClusterContext("CD4", "CD4 activated", "activated CD4 (inflammatory)"),
        6: ClusterContext("CD8", None, "late CD8; state resolved per cell"),
        7: ClusterContext("CD4", "CD4 proliferating", "dividing CD4"),
        8: ClusterContext("CD8", None, "day0 baseline cytotoxic CD8"),
        9: ClusterContext("CD8", None, "late CD8; state resolved per cell"),
        10: ClusterContext("CD4", "CD4 proliferating", "dividing CD4 (CD40LG/GZMA)"),
        11: ClusterContext("CD8", None, "day2 activated CD8"),
        12: ClusterContext("contaminant", "contaminant", "myeloid / antigen-presenting (not a T cell)"),
        13: ClusterContext("contaminant", "contaminant", "NK / gamma-delta (not a T cell)"),
    },
    "donor2": {
        0: ClusterContext("CD8", None, "late CD8; state resolved per cell"),
        1: ClusterContext("CD4", "CD4 proliferating", "dividing CD4"),
        2: ClusterContext("CD4", "CD4 activated / stress", "stressed CD4 (TRIB3/ATF3)"),
        3: ClusterContext("CD4", "CD4 activated", "activated CD4 (inflammatory)"),
        4: ClusterContext("CD8", None, "late CD8; state resolved per cell"),
        5: ClusterContext("CD8", None, "late CD8; state resolved per cell"),
        6: ClusterContext("CD4", "CD4 resting", "resting CD4 (IL7R/KLF2/TXNIP)"),
        7: ClusterContext("CD8", None, "day2 activated CD8"),
        8: ClusterContext("CD8", None, "late CD8; state resolved per cell"),
        9: ClusterContext("CD8", None, "dividing CD8"),
        10: ClusterContext("CD4", "CD4 activated", "activated CD4 (inflammatory)"),
        11: ClusterContext("contaminant", "contaminant", "myeloid (not a T cell)"),
        12: ClusterContext("contaminant", "contaminant", "mast (not a T cell)"),
    },
}

TYPE_NAMES = {
    "CD8 cytotoxic": "CD8Cytotoxic",
    "CD8 exhausted": "CD8Exhausted",
    "CD8 TPEX": "CD8Tpex",
    "CD8 TEX": "CD8Tex",
    "CD4 resting": "CD4Resting",
    "CD4 naive": "CD4Naive",
    "CD4 activated": "CD4Activated",
    "CD4 activated / stress": "CD4ActivatedStress",
    "CD4 proliferating": "CD4Proliferating",
}


def _assert_unique(df: pd.DataFrame, name: str) -> None:
    if df["barcode"].duplicated().any():
        examples = df.loc[df["barcode"].duplicated(), "barcode"].head().tolist()
        raise ValueError(f"{name} has duplicate barcodes: {examples}")


def _label_cd8(row: pd.Series, donor: str) -> str:
    inhibitory_detected = [row[gene] > 0 for gene in INHIBITORY_MARKERS]

    # A TCF7+ inhibitory-bearing precursor is not terminal exhaustion and may
    # divide. This split is supported only in donor2.
    if donor == "donor2" and row["TCF7"] > 0 and any(inhibitory_detected):
        return "CD8 TPEX"

    # Proliferation arrest is required for the terminal/exhausted call.
    if row["Phase"] == "G1" and sum(inhibitory_detected) >= 2:
        return "CD8 TEX" if donor == "donor2" else "CD8 exhausted"

    return "CD8 cytotoxic"


def _load_inputs(donor: str, expression_root: Path) -> pd.DataFrame:
    expr = pd.read_csv(
        expression_root / f"{donor}_marker_cell_expr.csv",
        usecols=["barcode", *EVIDENCE_MARKERS],
    )
    cc = pd.read_csv(
        CLUSTER_ROOT / f"{donor}_cc_recluster_cells.csv",
        usecols=["barcode", "old_cluster", "Phase"],
    )
    coords = pd.read_csv(
        UMAP_ROOT / f"{donor}_native_umap_coords.csv",
        usecols=["barcode", "seurat_clusters", "day_label"],
    )
    for name, frame in (("expression", expr), ("cell-cycle", cc), ("UMAP", coords)):
        _assert_unique(frame, f"{donor} {name}")

    df = coords.merge(cc, on="barcode", how="inner", validate="one_to_one")
    df = df.merge(expr, on="barcode", how="inner", validate="one_to_one")
    expected = len(coords)
    if len(df) != expected or len(cc) != expected or len(expr) != expected:
        raise ValueError(
            f"{donor} cell accounting mismatch: UMAP={len(coords)}, "
            f"cell-cycle={len(cc)}, expression={len(expr)}, joined={len(df)}"
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
    df["lineage"] = df["seurat_cluster"].map(lambda cluster: context[cluster].lineage)
    df["label"] = df["seurat_cluster"].map(lambda cluster: context[cluster].fixed_label)
    is_cd8 = df["lineage"] == "CD8"
    df.loc[is_cd8, "label"] = df.loc[is_cd8].apply(_label_cd8, axis=1, donor=donor)
    df["proliferation"] = df["Phase"].map(
        lambda phase: "dividing" if phase in DIVIDING_PHASES else "non-dividing"
    )
    df["type"] = df["label"].map(TYPE_NAMES)

    if df["label"].isna().any():
        raise ValueError(f"{donor} has unlabeled cells")
    if df.loc[df["label"] == "contaminant", "type"].notna().any():
        raise ValueError(f"{donor} contaminants unexpectedly have an Incytr type")
    if df.loc[df["label"] != "contaminant", "type"].isna().any():
        raise ValueError(f"{donor} non-contaminants are missing an Incytr type")

    terminal = df["label"].isin(["CD8 exhausted", "CD8 TEX"])
    if (df.loc[terminal, "proliferation"] != "non-dividing").any():
        raise ValueError(f"{donor} assigned a dividing cell to terminal/exhausted CD8")
    terminal_evidence = (df.loc[terminal, list(INHIBITORY_MARKERS)] > 0).sum(axis=1)
    if (terminal_evidence < 2).any():
        raise ValueError(f"{donor} assigned terminal/exhausted CD8 without two inhibitory markers")

    tpex = df["label"] == "CD8 TPEX"
    if ((df.loc[tpex, "TCF7"] <= 0) | ~((df.loc[tpex, list(INHIBITORY_MARKERS)] > 0).any(axis=1))).any():
        raise ValueError(f"{donor} assigned TPEX without TCF7 and inhibitory evidence")
    if donor == "donor1" and df["label"].isin(["CD8 TPEX", "CD8 TEX"]).any():
        raise ValueError("donor1 must not be split into TPEX/TEX")

    rename = {gene: f"{gene}_log_normalized_expression" for gene in EVIDENCE_MARKERS}
    df = df.rename(columns=rename)
    columns = [
        "barcode",
        "donor",
        "seurat_cluster",
        "day",
        "lineage",
        "Phase",
        "proliferation",
        *rename.values(),
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
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    donors = DONORS if args.donor == "all" else (args.donor,)
    for donor in donors:
        labels = _assign(donor, _load_inputs(donor, args.expression_root))
        labels.to_csv(args.output_root / f"{donor}_state_labels.csv", index=False)
        _print_summary(donor, labels)
    _write_cluster_context(args.output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

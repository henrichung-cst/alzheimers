"""Re-derive the 46-cluster -> WMB-class mapping from per-nucleus Allen CTM
class_name labels.

Inputs (already on disk):
- renamed_sobj_obs.csv  : per-nucleus barcode -> Idents (46 New_ID) +
  originalexp_snn_res.7 (110-cluster ID), extracted from
  yuyu01/.../deconvolution_with_new_clusters_20250721/renamed_sobj.rds
  via code/deconvolution/extract_yuyu_obs.R.
- 170_gex_celltypes_00.h5ad : Allen CTM class_name per nucleus.
- wmb_class_manifest.csv : maps Allen class_name (no prefix) -> WMB class_label
  ('30 Astro-Epen' style).

Outputs (new files; do NOT overwrite existing yuyu_46_to_wmb_class.csv):
- code/deconvolution/yuyu_46_to_wmb_class_v2.csv : hard plurality assignment
  with audit columns (plurality_fraction, n_nuclei, second-best class).
- code/deconvolution/yuyu_46_to_wmb_class_soft.csv : full mass matrix
  (46 clusters x 34 WMB classes), row-normalized fractions.

Usage:  pixi run python code/deconvolution/remap_clusters_via_ctm.py
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.dirname(HERE)
REPO_ROOT = os.path.dirname(CODE_DIR)
sys.path.insert(0, CODE_DIR)
sys.path.insert(0, REPO_ROOT)

import anndata as ad
import numpy as np
import pandas as pd

import config

OBS_CSV = os.path.join(
    REPO_ROOT, "data", "raw", "external", "gdrive_shared", "integrations",
    "yuyu01", "documentation", "incytr", "deconvolution",
    "deconvolution_with_new_clusters_20250721", "renamed_sobj_obs.csv",
)
H5AD = os.path.join(
    REPO_ROOT, "data", "datasets", "song", "transcriptomics",
    "170_gex_celltypes_00.h5ad",
)
MANIFEST = os.path.join(
    REPO_ROOT, "data", "external", "allen_abc", "wmb_class_manifest.csv",
)
HARD_OUT = os.path.join(HERE, "yuyu_46_to_wmb_class_v2.csv")
SOFT_OUT = os.path.join(HERE, "yuyu_46_to_wmb_class_soft.csv")


def main() -> None:
    print(f"Loading Seurat obs from {OBS_CSV}")
    md = pd.read_csv(
        OBS_CSV,
        usecols=["barcode", "Idents", "originalexp_snn_res.7", "sample"],
    )
    md = md.rename(columns={
        "Idents": "cluster_name",
        "originalexp_snn_res.7": "fine_cluster_id",
    })
    print(f"  {len(md):,} nuclei  /  {md['cluster_name'].nunique()} clusters")

    print(f"Loading Allen CTM class_name from {H5AD}")
    adata = ad.read_h5ad(H5AD, backed="r")
    h5 = pd.DataFrame({
        "barcode": adata.obs.index.astype(str),
        "class_name": adata.obs["class_name"].astype(str),
        "class_prob": adata.obs["class_prob"].values,
    })
    print(f"  {len(h5):,} nuclei  /  {h5['class_name'].nunique()} classes")

    print(f"Loading WMB class manifest from {MANIFEST}")
    man = pd.read_csv(MANIFEST)[["class_name", "class_label"]]
    print(f"  {len(man)} classes in published taxonomy")

    df = md.merge(h5, on="barcode", how="inner")
    df = df.merge(man, on="class_name", how="left")
    df["class_label"] = df["class_label"].fillna("Unmapped")
    print(f"Joined: {len(df):,} nuclei "
          f"(seurat {len(md):,} / h5ad {len(h5):,})")
    n_unmapped = (df["class_label"] == "Unmapped").sum()
    if n_unmapped:
        print(f"  WARN: {n_unmapped:,} nuclei have CTM class_name "
              f"not in manifest (treated as Unmapped)")

    # Soft mass matrix: row-normalized fractions per cluster.
    counts = (df.groupby(["cluster_name", "class_label"])
                .size()
                .unstack(fill_value=0)
                .astype(int))
    # Order columns: 34 WMB classes in canonical order, then Unmapped if present.
    ordered_cols = [c for c in config.WMB_CLASSES if c in counts.columns]
    if "Unmapped" in counts.columns:
        ordered_cols += ["Unmapped"]
    extra = [c for c in counts.columns if c not in ordered_cols]
    counts = counts.reindex(columns=ordered_cols + extra, fill_value=0)
    soft = counts.div(counts.sum(axis=1), axis=0).round(6)
    soft.insert(0, "n_nuclei", counts.sum(axis=1).values)

    soft.index.name = "cluster_name"
    soft.to_csv(SOFT_OUT)
    print(f"Wrote {SOFT_OUT}  ({soft.shape[0]} clusters x "
          f"{soft.shape[1] - 1} class columns)")

    # Hard plurality with audit columns. Compute over WMB classes only;
    # exclude Unmapped from the plurality call.
    wmb_cols = [c for c in counts.columns if c in config.WMB_CLASSES]
    cnt_wmb = counts[wmb_cols]
    plur_class = cnt_wmb.idxmax(axis=1)
    plur_count = cnt_wmb.max(axis=1)
    second = (
        cnt_wmb
        .apply(lambda r: r.sort_values(ascending=False), axis=1)
    )
    second_class = second.iloc[:, 1] if second.shape[1] > 1 else np.nan
    second_count = (
        cnt_wmb.apply(lambda r: r.nlargest(2).iloc[-1] if len(r) > 1 else 0,
                      axis=1)
    )
    n_total = counts.sum(axis=1)

    hard = pd.DataFrame({
        "cluster_name": plur_class.index,
        "wmb_class_v2": plur_class.values,
        "plurality_fraction": (plur_count / n_total.replace(0, np.nan)).round(4).values,
        "n_nuclei": n_total.values.astype(int),
        "n_plurality": plur_count.values.astype(int),
        "second_class": [second.columns[1] if second.shape[1] > 1 else ""] * len(plur_class),
        "second_fraction": (second_count / n_total.replace(0, np.nan)).round(4).values,
        "unmapped_fraction": (
            counts.get("Unmapped", pd.Series(0, index=counts.index))
            / n_total.replace(0, np.nan)
        ).round(4).values,
    })

    # Per-row second-best (vectorize properly).
    def _row_second(row: pd.Series) -> pd.Series:
        nz = row.sort_values(ascending=False)
        if len(nz) >= 2:
            return pd.Series({"second_class": nz.index[1],
                              "second_fraction": nz.iloc[1] / row.sum() if row.sum() else 0.0})
        return pd.Series({"second_class": "", "second_fraction": 0.0})

    sec = cnt_wmb.apply(_row_second, axis=1)
    hard["second_class"] = sec["second_class"].values
    hard["second_fraction"] = sec["second_fraction"].round(4).values

    # Merge in old hand mapping for side-by-side audit.
    old_path = os.path.join(HERE, "yuyu_46_to_wmb_class.csv")
    if os.path.exists(old_path):
        old = pd.read_csv(old_path)[
            ["cluster_name", "wmb_class", "confidence_in_mapping", "notes"]
        ].rename(columns={"wmb_class": "wmb_class_v1_hand",
                          "confidence_in_mapping": "v1_confidence",
                          "notes": "v1_notes"})
        hard = hard.merge(old, on="cluster_name", how="left")
        hard["agrees_with_v1"] = (
            hard["wmb_class_v2"] == hard["wmb_class_v1_hand"]
        )

    hard.sort_values(["wmb_class_v2", "n_nuclei"],
                     ascending=[True, False], inplace=True)
    hard.to_csv(HARD_OUT, index=False)
    print(f"Wrote {HARD_OUT}  ({len(hard)} clusters)")

    # Coverage summary.
    n_classes_v2 = hard["wmb_class_v2"].nunique()
    print()
    print(f"Coverage: {n_classes_v2}/{len(config.WMB_CLASSES)} WMB classes "
          f"reachable via CTM-derived plurality assignment")
    if "agrees_with_v1" in hard:
        n_agree = int(hard["agrees_with_v1"].fillna(False).sum())
        print(f"v1-vs-v2 agreement: {n_agree}/{len(hard)} clusters unchanged")


if __name__ == "__main__":
    main()

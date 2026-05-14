"""Build the 19-cluster strict spine + rejection log from the Levy 46-cluster taxonomy.

Inputs (must exist):
  - data/incytr/v2_46clusters/provenance/kr_cluster_id_key.csv
  - data/incytr/v2_46clusters/barcode_to_cluster.csv
  - data/incytr/v2_46clusters/cell_metadata.csv

Outputs:
  - data/incytr/v2_46clusters/cluster_spine.csv      (46 rows, in_spine bool)
  - data/incytr/v2_46clusters/rejected_clusters.csv  (27 rows, with reason)

Gate logic:
  - Drop unnamed `cluster-NN` (Q5).
  - For each named cluster, count animals with >= SONG_MIN_CELLS cells.
  - Build a 10-parameter design matrix over those animals
    (const, App, Tau, Int, time_4mo, time_6mo, App:t4, App:t6, Tau:t4, Tau:t6)
    and compute matrix rank.
  - in_spine == (rank == 10 and n_animals_g20 >= 1).
  - tier:
      full_rank      rank == 10
      partial        7 <= rank < 10
      severe         1 <= rank < 7
      fails_gate     n_animals_g20 == 0
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

SONG_MIN_CELLS = 20

REPO = Path(__file__).resolve().parents[2]
KEY_PATH = REPO / "data/incytr/v2_46clusters/provenance/kr_cluster_id_key.csv"
BC_PATH = REPO / "data/incytr/v2_46clusters/barcode_to_cluster.csv"
META_PATH = REPO / "data/incytr/v2_46clusters/cell_metadata.csv"
SPINE_OUT = REPO / "data/incytr/v2_46clusters/cluster_spine.csv"
REJECT_OUT = REPO / "data/incytr/v2_46clusters/rejected_clusters.csv"

GENO_CODES = {
    "WTyp": (0, 0, 0),
    "AppP": (1, 0, 0),
    "Ttau": (0, 1, 0),
    "ApTt": (1, 1, 1),
}
TIME_CODES = {"2mo": (0, 0), "4mo": (1, 0), "6mo": (0, 1)}


def build_design(rows: pd.DataFrame) -> np.ndarray:
    n = len(rows)
    X = np.zeros((n, 10))
    X[:, 0] = 1.0
    for i, (_, r) in enumerate(rows.iterrows()):
        app, tau, intx = GENO_CODES[r["Genotype"]]
        t4, t6 = TIME_CODES[r["Time"]]
        X[i, 1] = app
        X[i, 2] = tau
        X[i, 3] = intx
        X[i, 4] = t4
        X[i, 5] = t6
        X[i, 6] = app * t4
        X[i, 7] = app * t6
        X[i, 8] = tau * t4
        X[i, 9] = tau * t6
    return X


def main() -> int:
    key = pd.read_csv(KEY_PATH)
    bc = pd.read_csv(BC_PATH)
    meta = pd.read_csv(META_PATH)

    name_for_id = dict(zip(key["Cluster ID"].astype(int), key["New_ID"].astype(str)))
    bc["cluster_name"] = bc["seurat_cluster_id"].map(name_for_id)
    if bc["cluster_name"].isna().any():
        raise RuntimeError("seurat_cluster_id not in key")

    df = bc.merge(meta, on="barcode", how="inner", validate="1:1")
    df["is_unnamed"] = df["cluster_name"].str.startswith("cluster-")

    all_names = sorted(key["New_ID"].astype(str).unique())
    assert len(all_names) == 46, f"expected 46 cluster names, got {len(all_names)}"

    animal_meta = (
        df[["sample", "Genotype", "Time"]].drop_duplicates().reset_index(drop=True)
    )

    pc = (
        df.groupby(["cluster_name", "sample"])
        .size()
        .reset_index(name="n_cells")
    )
    pc_g20 = pc[pc["n_cells"] >= SONG_MIN_CELLS].merge(animal_meta, on="sample")

    rows = []
    for name in all_names:
        total_cells = int((df["cluster_name"] == name).sum())
        is_unnamed = name.startswith("cluster-")
        animals = pc_g20[pc_g20["cluster_name"] == name][
            ["sample", "Genotype", "Time"]
        ].drop_duplicates()
        n_animals_g20 = len(animals)
        if is_unnamed:
            rank = 0
            tier = "unnamed"
        elif n_animals_g20 == 0:
            rank = 0
            tier = "fails_gate"
        else:
            X = build_design(animals)
            rank = int(np.linalg.matrix_rank(X))
            if rank == 10:
                tier = "full_rank"
            elif rank >= 7:
                tier = "partial"
            else:
                tier = "severe"
        in_spine = (tier == "full_rank")
        missing_geno = sorted(set(GENO_CODES) - set(animals["Genotype"])) if n_animals_g20 else sorted(GENO_CODES)
        missing_time = sorted(set(TIME_CODES) - set(animals["Time"])) if n_animals_g20 else sorted(TIME_CODES)
        if is_unnamed:
            reason = "unnamed"
        elif in_spine:
            reason = ""
        else:
            reason = tier
        rows.append(
            {
                "cluster_name": name,
                "in_spine": in_spine,
                "tier": tier,
                "n_cells": total_cells,
                "n_animals_g20": n_animals_g20,
                "rank": rank,
                "missing_geno": ",".join(missing_geno),
                "missing_time": ",".join(missing_time),
                "exclusion_reason": reason,
            }
        )

    spine_df = pd.DataFrame(rows).sort_values(
        ["in_spine", "n_cells"], ascending=[False, False]
    )
    spine_df.to_csv(SPINE_OUT, index=False)
    rejected_df = spine_df[~spine_df["in_spine"]].copy()
    rejected_df.to_csv(REJECT_OUT, index=False)

    n_spine = int(spine_df["in_spine"].sum())
    n_total_cells_spine = int(spine_df.loc[spine_df["in_spine"], "n_cells"].sum())
    n_total_all = int(spine_df["n_cells"].sum())
    print(f"wrote {SPINE_OUT.relative_to(REPO)} ({len(spine_df)} rows)")
    print(f"wrote {REJECT_OUT.relative_to(REPO)} ({len(rejected_df)} rows)")
    print(f"in_spine: {n_spine} clusters, {n_total_cells_spine}/{n_total_all} cells "
          f"({100*n_total_cells_spine/n_total_all:.2f}%)")
    print("tier counts:")
    print(spine_df["tier"].value_counts().to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())

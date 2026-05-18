"""Build a per-cluster spine + rejection log from the Levy 46-cluster taxonomy.

Inputs (must exist):
  - data/incytr_frozen/v2_46clusters/provenance/kr_cluster_id_key.csv
  - data/incytr_frozen/v2_46clusters/barcode_to_cluster.csv
  - data/incytr_frozen/v2_46clusters/cell_metadata.csv

Outputs (under data/incytr_frozen/v2_46clusters/spines/<spine-name>/):
  - cluster_spine.csv         (46 rows, in_spine bool)
  - rejected_clusters.csv     (with reason)
  - spine.scope.json          {name, min_cells, rank_gate, generated_at, n_in_spine}

Gate logic:
  - Drop unnamed `cluster-NN` (Q5).
  - For each named cluster, count animals with >= min_cells cells.
  - If --no-rank-gate (or rank_gate disabled):
      in_spine == (not is_unnamed) and n_animals_ge_min >= 1
    else (legacy):
      Build a 10-parameter design matrix over the qualifying animals
      and require matrix rank == 10.
  - tier annotation continues to record the rank even when the rank gate
    is off, so callers can post-hoc see how rank-deficient a kept cluster is.

Backward compatibility:
  Defaults (--spine-name levy19, --min-cells 20, rank gate ON) reproduce the
  legacy `data/incytr_frozen/v2_46clusters/cluster_spine.csv` outputs. To preserve
  existing readers (config_integration.CLUSTER_SPINE_FILE,
  plot_cluster_spine.py, etc.) the levy19 outputs are also surfaced via a
  top-level symlink:
    data/incytr_frozen/v2_46clusters/cluster_spine.csv -> spines/levy19/cluster_spine.csv
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
KEY_PATH = REPO / "data/incytr_frozen/v2_46clusters/provenance/kr_cluster_id_key.csv"
BC_PATH = REPO / "data/incytr_frozen/v2_46clusters/barcode_to_cluster.csv"
META_PATH = REPO / "data/incytr_frozen/v2_46clusters/cell_metadata.csv"
V2_ROOT = REPO / "data/incytr_frozen/v2_46clusters"
SPINES_ROOT = V2_ROOT / "spines"
LEGACY_SPINE_CSV = V2_ROOT / "cluster_spine.csv"
LEGACY_REJECT_CSV = V2_ROOT / "rejected_clusters.csv"

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
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--spine-name", default="levy19",
                    help="output spine name (default: levy19)")
    ap.add_argument("--min-cells", type=int, default=20,
                    help="per-(cluster, animal) cell-count gate (default: 20)")
    ap.add_argument("--no-rank-gate", action="store_true",
                    help="disable the rank-10 design-matrix gate; any named "
                         "cluster with >=1 qualifying animal becomes in_spine")
    args = ap.parse_args()

    spine_name = args.spine_name
    min_cells = int(args.min_cells)
    rank_gate = not args.no_rank_gate

    out_dir = SPINES_ROOT / spine_name
    out_dir.mkdir(parents=True, exist_ok=True)
    spine_out = out_dir / "cluster_spine.csv"
    reject_out = out_dir / "rejected_clusters.csv"
    scope_out = out_dir / "spine.scope.json"

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
    pc_qual = pc[pc["n_cells"] >= min_cells].merge(animal_meta, on="sample")

    rows = []
    for name in all_names:
        total_cells = int((df["cluster_name"] == name).sum())
        is_unnamed = name.startswith("cluster-")
        animals = pc_qual[pc_qual["cluster_name"] == name][
            ["sample", "Genotype", "Time"]
        ].drop_duplicates()
        n_animals_qual = len(animals)
        if is_unnamed:
            rank = 0
            tier = "unnamed"
        elif n_animals_qual == 0:
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
        if is_unnamed:
            in_spine = False
        elif rank_gate:
            in_spine = (tier == "full_rank")
        else:
            in_spine = (n_animals_qual >= 1)
        missing_geno = sorted(set(GENO_CODES) - set(animals["Genotype"])) if n_animals_qual else sorted(GENO_CODES)
        missing_time = sorted(set(TIME_CODES) - set(animals["Time"])) if n_animals_qual else sorted(TIME_CODES)
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
                "n_animals_qual": n_animals_qual,
                "rank": rank,
                "missing_geno": ",".join(missing_geno),
                "missing_time": ",".join(missing_time),
                "exclusion_reason": reason,
            }
        )

    spine_df = pd.DataFrame(rows).sort_values(
        ["in_spine", "n_cells"], ascending=[False, False]
    )
    # Preserve the legacy column name `n_animals_g20` when the gate matches
    # the historical value, so old readers don't choke.
    if min_cells == 20:
        spine_df = spine_df.rename(columns={"n_animals_qual": "n_animals_g20"})
    spine_df.to_csv(spine_out, index=False)
    rejected_df = spine_df[~spine_df["in_spine"]].copy()
    rejected_df.to_csv(reject_out, index=False)

    n_spine = int(spine_df["in_spine"].sum())
    n_total_cells_spine = int(spine_df.loc[spine_df["in_spine"], "n_cells"].sum())
    n_total_all = int(spine_df["n_cells"].sum())

    scope = {
        "name": spine_name,
        "min_cells": min_cells,
        "rank_gate": rank_gate,
        "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "n_in_spine": n_spine,
        "n_total_clusters": int(len(spine_df)),
    }
    with open(scope_out, "w") as fh:
        json.dump(scope, fh, indent=2)

    # levy19 back-compat: keep the old top-level paths pointing at the new
    # outputs, so config_integration.CLUSTER_SPINE_FILE and friends resolve.
    if spine_name == "levy19":
        for legacy, new in (
            (LEGACY_SPINE_CSV, spine_out),
            (LEGACY_REJECT_CSV, reject_out),
        ):
            try:
                if legacy.is_symlink() or legacy.exists():
                    legacy.unlink()
            except FileNotFoundError:
                pass
            try:
                legacy.symlink_to(new.relative_to(legacy.parent))
            except OSError:
                # Filesystem without symlink support — fall back to a copy.
                import shutil
                shutil.copyfile(new, legacy)

    print(f"wrote {spine_out.relative_to(REPO)} ({len(spine_df)} rows)")
    print(f"wrote {reject_out.relative_to(REPO)} ({len(rejected_df)} rows)")
    print(f"wrote {scope_out.relative_to(REPO)}")
    print(f"spine_name: {spine_name} | min_cells: {min_cells} | rank_gate: {rank_gate}")
    print(f"in_spine: {n_spine} clusters, {n_total_cells_spine}/{n_total_all} cells "
          f"({100*n_total_cells_spine/n_total_all:.2f}%)")
    print("tier counts:")
    print(spine_df["tier"].value_counts().to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())

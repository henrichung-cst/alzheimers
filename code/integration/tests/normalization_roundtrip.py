"""Round-trip checks for normalized Incytr storage.

Usage:
  python code/integration/tests/normalization_roundtrip.py \
      --legacy-all-pairs code/integration/intermediates/factorial/all_pairs \
      --universe outputs/incytr/universes/u_... \
      --scoring outputs/incytr/scoring/s_... \
      --config outputs/incytr/configs/c_...
"""

from __future__ import annotations

import argparse
import os
import sys
from glob import glob

import numpy as np
import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
CODE_DIR = os.path.join(REPO_ROOT, "code")
INTEGRATION_DIR = os.path.join(CODE_DIR, "integration")
ADAPTERS_DIR = os.path.join(CODE_DIR, "integration", "adapters")
for path in (CODE_DIR, INTEGRATION_DIR, ADAPTERS_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

from normalization import CONCORDANCE_ENUM  # noqa: E402


def _load_id_maps(universe_dir):
    cells = pd.read_parquet(os.path.join(universe_dir, "cells.parquet"))
    genes = pd.read_parquet(os.path.join(universe_dir, "genes.parquet"))
    contrasts = pd.read_parquet(os.path.join(universe_dir, "contrasts.parquet"))
    backbones = pd.read_parquet(os.path.join(universe_dir, "backbones.parquet"))
    pathways = pd.read_parquet(os.path.join(universe_dir, "pathways.parquet"))

    cell_name = dict(zip(cells["cell_id"], cells["name"]))
    gene_name = dict(zip(genes["gene_id"], genes["symbol"]))
    contrast_name = dict(zip(contrasts["contrast_id"], contrasts["name"]))

    bb = backbones.copy()
    bb["receiver"] = bb["receiver_id"].map(cell_name)
    bb["Receptor"] = bb["receptor_gene_id"].map(gene_name)
    bb["EM"] = bb["em_gene_id"].map(gene_name)
    bb["Target"] = bb["target_gene_id"].map(gene_name)
    bb = bb[["backbone_id", "receiver", "Receptor", "EM", "Target"]]

    pw = pathways.merge(bb, on="backbone_id", how="left", validate="many_to_one")
    pw["Ligand"] = pw["ligand_gene_id"].map(gene_name)
    pw["reconstructed_path"] = (
        pw["Ligand"] + "*" + pw["Receptor"] + "*" + pw["EM"] + "*" + pw["Target"]
    )
    return {
        "pathways": pw,
        "contrast_name": contrast_name,
    }


def check_paths(legacy_all_pairs, universe_dir, sample_pairs=5):
    maps = _load_id_maps(universe_dir)
    path_map = maps["pathways"][["path_id", "path", "reconstructed_path"]]
    mismatches = path_map[path_map["path"] != path_map["reconstructed_path"]]
    if not mismatches.empty:
        raise AssertionError(f"{len(mismatches)} normalized pathway strings do not round-trip")

    route_files = sorted(glob(os.path.join(legacy_all_pairs, "*", "kinase_routes.parquet")))
    for route_file in route_files[:sample_pairs]:
        legacy_paths = set(pd.read_parquet(route_file, columns=["Path"])["Path"].drop_duplicates())
        missing = legacy_paths - set(path_map["path"])
        if missing:
            raise AssertionError(f"{route_file}: {len(missing)} paths absent from normalized pathways")
    return len(path_map)


def check_scores(legacy_all_pairs, universe_dir, scoring_dir, sample_pairs=5, atol=1e-6):
    pathways = pd.read_parquet(
        os.path.join(universe_dir, "pathways.parquet"),
        columns=["path_id", "path", "sender_id"],
    )
    pair_dim = pd.read_parquet(os.path.join(universe_dir, "pair_dim.parquet"))
    contrast = pd.read_parquet(os.path.join(universe_dir, "contrasts.parquet")).iloc[0]

    checked = 0
    for row in pair_dim.itertuples(index=False):
        legacy_file = os.path.join(legacy_all_pairs, row.name, "kinase_support_scores.csv")
        normalized_file = os.path.join(
            scoring_dir, "pathway_scores.parquet", f"pair_id={int(row.pair_id)}", "part-0.parquet",
        )
        if not os.path.exists(legacy_file) or not os.path.exists(normalized_file):
            continue
        legacy = pd.read_csv(legacy_file)
        path_subset = pathways[pathways["sender_id"] == int(row.sender_id)]
        path_to_id = dict(zip(path_subset["path"], path_subset["path_id"]))
        legacy["path_id"] = legacy["Path"].map(path_to_id)
        legacy["contrast_id"] = int(contrast.contrast_id)
        legacy["mea_concordance_flag_id"] = legacy["mea_concordance_flag"].map(CONCORDANCE_ENUM).astype("uint8")
        norm = pd.read_parquet(normalized_file)
        merged = legacy.merge(norm, on=["path_id", "contrast_id"], how="inner", validate="one_to_one")
        if len(merged) != len(legacy):
            raise AssertionError(f"{row.name}: normalized score row count mismatch")
        checks = [
            ("mea_kinase_support_score", "support_score"),
            ("mea_kinase_support_sum", "support_sum"),
            ("TPDS", "tpds"),
        ]
        for legacy_col, norm_col in checks:
            if not np.allclose(merged[legacy_col], merged[norm_col], atol=atol, rtol=0):
                raise AssertionError(f"{row.name}: {legacy_col} differs from {norm_col}")
        if not (merged["mea_n_distinct_kinases"] == merged["n_distinct_kinases"]).all():
            raise AssertionError(f"{row.name}: n_distinct_kinases mismatch")
        if not (merged["mea_concordance_flag_id"] == merged["concordance_flag"]).all():
            raise AssertionError(f"{row.name}: concordance flag enum mismatch")
        checked += 1
        if checked >= sample_pairs:
            break
    if checked == 0:
        raise AssertionError("No comparable score partitions found")
    return checked


def check_config(config_dir):
    required = [
        "backbones_by_contrast.parquet",
        "backbone_senders.parquet",
        "target_convergence.parquet",
        "target_convergence_senders.parquet",
    ]
    present = [name for name in required if os.path.exists(os.path.join(config_dir, name))]
    if not present:
        raise AssertionError(f"No normalized config tables found in {config_dir}")
    return present


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy-all-pairs", required=True)
    parser.add_argument("--universe", required=True)
    parser.add_argument("--scoring")
    parser.add_argument("--config")
    parser.add_argument("--sample-pairs", type=int, default=5)
    args = parser.parse_args()

    n_paths = check_paths(args.legacy_all_pairs, args.universe, args.sample_pairs)
    print(f"paths: {n_paths:,} round-trip")
    if args.scoring:
        n_score_pairs = check_scores(
            args.legacy_all_pairs, args.universe, args.scoring, args.sample_pairs,
        )
        print(f"scores: {n_score_pairs} pair partitions match")
    if args.config:
        present = check_config(args.config)
        print(f"config tables: {', '.join(present)}")


if __name__ == "__main__":
    main()

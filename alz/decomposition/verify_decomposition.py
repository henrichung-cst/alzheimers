"""Step 13 verification harness for the Levy-19 per-cluster decomposition pivot.

Four contracts (see docs/incytr_deconvolution_pivot.md §Verification):
  1. Mass identity: Σ_c [P_c × (N_c / N_total)] ≈ bulk per (gene, animal)
  2. Coverage: all spine clusters present in Stage 6 outputs
  3. Per-cluster vs bulk MEA agreement under f_c-weighting
  4. Incytr produces |spine|² scored sender × receiver pairs (here 19² = 361)

Writes outputs/reports/decomposition/{spine}/verification.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402
from decomposition.build_celltype_decomposition import (  # noqa: E402
    _bulk_to_long, _load_sample_mapping,
)

sys.path.insert(0, os.path.join(config.REPO_ROOT, "alz", "integration"))
import config_integration as icfg  # noqa: E402

REPO = Path(config.REPO_ROOT)
BULK_DIR = REPO / "outputs/reports/kinase_attribution"
INCYTR_PAIRS = REPO / "outputs/reports/incytr_factorial/pair_metadata.parquet"
CELL_COUNTS_FILE = REPO / "outputs/reports/snrna_integration/pseudobulk_cell_counts.csv"
SAMPLE_MAPPING_FILE = REPO / "outputs/reports/data_ingest/sample_mapping.csv"

CHECKS = ("mass", "coverage", "mea", "incytr")


def _spine_dir(spine: str) -> Path:
    return REPO / "outputs/reports/decomposition" / spine


def _cluster_weights(clusters: set[str]) -> pd.DataFrame:
    """Per-(animal_id, cluster) share = N_c / N_total, snRNA-observed animals only."""
    counts = pd.read_csv(CELL_COUNTS_FILE).rename(
        columns={"sample": "snrna_sample_id", "cell_type": "cluster"}
    )
    mp = pd.read_csv(SAMPLE_MAPPING_FILE)
    mp = mp.loc[mp["has_snrna_seq"].astype(bool), ["animal_id", "snrna_sample_id"]]
    n_cells = counts.merge(mp, on="snrna_sample_id", how="inner")
    n_cells = n_cells[n_cells["cluster"].isin(clusters)]
    totals = n_cells.groupby("animal_id")["n_cells"].transform("sum")
    n_cells["w"] = n_cells["n_cells"] / totals
    return n_cells[["animal_id", "cluster", "w"]]


def _bulk_protein_long() -> pd.DataFrame:
    mp = _load_sample_mapping()
    col_to_animal = dict(zip(mp["column_name"], mp["animal_id"]))
    protein = pd.read_csv(BULK_DIR / "total_proteome_normalized.csv")
    meta_cols = [c for c in ("gene_symbol", "protein_id") if c in protein.columns]
    val_cols = [c for c in protein.columns if c not in meta_cols]
    protein = protein.dropna(subset=["gene_symbol"]).copy()
    protein["gene_symbol"] = protein["gene_symbol"].astype(str)
    protein = protein[protein["gene_symbol"].str.match(r"^[A-Za-z][\w\-\.]*$")]
    protein = protein.drop_duplicates(subset=["gene_symbol"], keep="first")
    return _bulk_to_long(
        protein, val_cols, ["gene_symbol"], col_to_animal, "bulk_value",
    )


def check_mass_identity(spine_dir: Path, weights: pd.DataFrame) -> dict:
    obs_animals = sorted(weights["animal_id"].unique())
    decomp = pd.read_parquet(
        spine_dir / "protein_per_cluster.parquet",
        filters=[("animal_id", "in", obs_animals)],
    )
    j = decomp.merge(weights, on=["animal_id", "cluster"], how="inner")
    j["weighted"] = j["value"] * j["w"]
    re_bulk = (j.groupby(["gene_symbol", "animal_id"], as_index=False)["weighted"]
                 .sum()
                 .rename(columns={"weighted": "reconstructed"}))

    bulk = _bulk_protein_long()
    bulk = bulk[bulk["animal_id"].isin(obs_animals)]
    cmp = re_bulk.merge(bulk, on=["gene_symbol", "animal_id"], how="inner")
    err = (cmp["reconstructed"] - cmp["bulk_value"]).abs()
    rel_err = err / cmp["bulk_value"].abs().clip(lower=1e-12)
    max_rel = float(rel_err.max()) if len(cmp) else None
    return {
        "check": "mass_identity",
        "n_compared": int(len(cmp)),
        "max_rel_err": max_rel,
        "median_rel_err": float(rel_err.median()) if len(cmp) else None,
        "frac_rel_err_below_1e-6": float((rel_err < 1e-6).mean()) if len(cmp) else None,
        "pass": max_rel is not None and max_rel < 1e-6,
    }


def check_coverage(spine_dir: Path) -> dict:
    expected = set(icfg.load_cluster_spine())
    results = {}
    for name in ["protein_per_cluster.parquet", "phospho_per_cluster.parquet"]:
        p = spine_dir / name
        if not p.exists():
            results[name] = {"status": "missing"}
            continue
        got = set(pd.read_parquet(p, columns=["cluster"])["cluster"].unique())
        results[name] = {
            "expected": len(expected),
            "got": len(got),
            "missing": sorted(expected - got),
            "extra": sorted(got - expected),
        }
    all_match = all(
        r.get("missing") == [] and r.get("extra") == []
        for r in results.values() if r.get("status") != "missing"
    )
    return {"check": "coverage", "spine_size": len(expected),
            "results": results, "pass": all_match}


def check_per_cluster_vs_bulk_mea(spine_dir: Path, weights: pd.DataFrame) -> dict:
    pc_path = spine_dir / "mea_per_cluster.parquet"
    bulk_path = BULK_DIR / "mea_stoichiometry.csv"
    if not pc_path.exists() or not bulk_path.exists():
        return {"check": "per_cluster_vs_bulk_mea", "status": "skipped",
                "reason": f"missing: pc={pc_path.exists()}, bulk={bulk_path.exists()}",
                "pass": None}
    pc = pd.read_parquet(pc_path)
    bulk = pd.read_csv(bulk_path)

    cluster_w = (weights[weights["cluster"].isin(pc["cluster"].unique())]
                 .groupby("cluster", as_index=False)["w"].mean())

    j = pc.merge(cluster_w, on="cluster", how="inner")
    j["nes_w"] = j["NES"] * j["w"]
    agg = j.groupby(["contrast", "kinase"], as_index=False).agg(
        nes_recon=("nes_w", "sum"), w_sum=("w", "sum"),
    )
    agg["nes_recon"] = agg["nes_recon"] / agg["w_sum"].clip(lower=1e-12)

    cmp = agg.merge(
        bulk[["contrast", "kinase", "NES"]].rename(columns={"NES": "nes_bulk"}),
        on=["contrast", "kinase"], how="inner",
    )
    per_contrast = []
    for c, sub in cmp.groupby("contrast"):
        rho = (float(sub[["nes_recon", "nes_bulk"]].corr(method="spearman").iloc[0, 1])
               if len(sub) >= 5 else None)
        per_contrast.append({
            "contrast": c,
            "n": int(len(sub)),
            "spearman_rho": rho,
            "median_abs_diff": float((sub["nes_recon"] - sub["nes_bulk"]).abs().median()),
        })
    rhos = [p["spearman_rho"] for p in per_contrast if p["spearman_rho"] is not None]
    median_diffs = [p["median_abs_diff"] for p in per_contrast]
    passed = bool(rhos and min(rhos) >= 0.7 and max(median_diffs) <= 0.5)
    return {"check": "per_cluster_vs_bulk_mea",
            "per_contrast": per_contrast,
            "min_rho": min(rhos) if rhos else None,
            "max_median_abs_diff": max(median_diffs) if median_diffs else None,
            "pass": passed}


def check_incytr_pair_count() -> dict:
    expected = icfg.load_cluster_spine()
    expected_pairs = len(expected) ** 2
    if not INCYTR_PAIRS.exists():
        return {"check": "incytr_pair_count", "status": "missing",
                "expected_pairs": expected_pairs, "pass": None}
    pm = pd.read_parquet(INCYTR_PAIRS, columns=["sender", "receiver"])
    senders = set(pm["sender"].unique())
    receivers = set(pm["receiver"].unique())
    return {"check": "incytr_pair_count",
            "expected_pairs": expected_pairs,
            "n_pairs": int(len(pm)),
            "n_senders": len(senders),
            "n_receivers": len(receivers),
            "spine_senders_missing": sorted(set(expected) - senders),
            "spine_receivers_missing": sorted(set(expected) - receivers),
            "pass": len(pm) == expected_pairs
                    and senders == set(expected)
                    and receivers == set(expected)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--spine", default="levy19")
    ap.add_argument("--checks", nargs="+", choices=CHECKS, default=list(CHECKS),
                    help=f"Subset of checks to run (default: all of {CHECKS})")
    args = ap.parse_args()

    spine_dir = _spine_dir(args.spine)
    spine_clusters = set(icfg.load_cluster_spine())
    weights = (_cluster_weights(spine_clusters)
               if {"mass", "mea"} & set(args.checks) else None)

    results = []
    for name in args.checks:
        print(f"[{name}] running ...")
        if name == "mass":
            r = check_mass_identity(spine_dir, weights)
        elif name == "coverage":
            r = check_coverage(spine_dir)
        elif name == "mea":
            r = check_per_cluster_vs_bulk_mea(spine_dir, weights)
        else:  # incytr
            r = check_incytr_pair_count()
        print(f"      pass={r['pass']}")
        results.append(r)

    out = {"spine": args.spine, "checks": results,
           "all_pass": all(r.get("pass") is True for r in results)}
    out_path = spine_dir / "verification.json"
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nWrote {out_path}")
    print(f"all_pass={out['all_pass']}")
    if not out["all_pass"]:
        sys.exit(1)


if __name__ == "__main__":
    main()

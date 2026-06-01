"""Verification harness for the Levy-t5 per-cluster decomposition.

Contracts:
  1. Mass identity: Σ_c [P_c × (N_c / N_total)] ≈ bulk per (gene, animal)
  2. Coverage: all spine clusters present in Stage 6 outputs
  3. Diagnostic: per-cluster vs bulk MEA agreement under f_c-weighting
  4. Diagnostic: Incytr sender × receiver coverage for a specific artifact

By default, only the hard decomposition gates (mass, coverage) run and write
outputs/reports/decomposition/{spine}/verification.json. Diagnostic checks can
be requested explicitly and write a separate report unless an output path is
provided.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from alz.shared import config  # noqa: E402

sys.path.insert(0, os.path.join(config.REPO_ROOT, "alz", "integration"))
import config_integration as icfg  # noqa: E402

REPO = Path(config.REPO_ROOT)
BULK_DIR = REPO / "outputs/reports/kinase_attribution"
INCYTR_PAIRS = REPO / "outputs/reports/incytr_pair_mode/pair_metadata.parquet"
CELL_COUNTS_FILE = REPO / "outputs/reports/snrna_integration/pseudobulk_cell_counts.csv"
SAMPLE_MAPPING_FILE = REPO / "outputs/reports/data_ingest/sample_mapping.csv"

HARD_CHECKS = ("mass", "coverage")
DIAGNOSTIC_CHECKS = ("mea", "incytr")
CHECKS = HARD_CHECKS + DIAGNOSTIC_CHECKS


def _spine_dir(spine: str) -> Path:
    return REPO / "outputs/reports/decomposition" / spine


def _load_sample_mapping() -> pd.DataFrame:
    mp = pd.read_csv(SAMPLE_MAPPING_FILE)
    if not {"column_name", "animal_id"}.issubset(mp.columns):
        raise KeyError(f"{SAMPLE_MAPPING_FILE}: missing column_name / animal_id")
    return mp[["column_name", "animal_id"]]


def _bulk_to_long(
    df: pd.DataFrame, value_cols: list[str], id_cols: list[str],
    col_to_animal: dict[str, str], value_name: str,
) -> pd.DataFrame:
    """Wide bulk → long (id_cols..., animal_id, value)."""
    long = df.melt(
        id_vars=id_cols,
        value_vars=value_cols,
        var_name="column_name",
        value_name=value_name,
    )
    long["animal_id"] = long["column_name"].map(col_to_animal)
    if long["animal_id"].isna().any():
        miss = sorted(long.loc[long["animal_id"].isna(), "column_name"].unique())
        raise KeyError(f"Sample mapping missing for columns: {miss[:5]} ...")
    long = long.drop(columns=["column_name"]).dropna(subset=[value_name])
    return long


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
        "severity": "hard",
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
    return {"check": "coverage", "severity": "hard", "spine_size": len(expected),
            "results": results, "pass": all_match}


def check_per_cluster_vs_bulk_mea(spine_dir: Path, weights: pd.DataFrame) -> dict:
    pc_path = spine_dir / "mea_per_cluster.parquet"
    bulk_path = BULK_DIR / "mea_stoichiometry.csv"
    if not pc_path.exists() or not bulk_path.exists():
        return {"check": "per_cluster_vs_bulk_mea", "severity": "diagnostic",
                "status": "skipped",
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
            "severity": "diagnostic",
            "per_contrast": per_contrast,
            "min_rho": min(rhos) if rhos else None,
            "max_median_abs_diff": max(median_diffs) if median_diffs else None,
            "pass": passed}


def check_incytr_pair_count() -> dict:
    expected = icfg.load_cluster_spine()
    # Pair-mode (Cal_pairwise_grid) emits the full N×N grid including
    # self-pairs (sender == receiver). The legacy factorial wrapper used to
    # exclude self-pairs (N × (N−1)); factorial was archived 2026-05-18 and
    # the upstream APIs deleted at commit 424119f, so this check no longer
    # subtracts the diagonal.
    expected_pairs = len(expected) * len(expected)
    if not INCYTR_PAIRS.exists():
        return {"check": "incytr_pair_count", "severity": "diagnostic",
                "status": "missing",
                "expected_pairs": expected_pairs, "pass": None}
    pm = pd.read_parquet(INCYTR_PAIRS, columns=["sender", "receiver"])
    senders = set(pm["sender"].unique())
    receivers = set(pm["receiver"].unique())
    self_pairs = int((pm["sender"] == pm["receiver"]).sum())
    return {"check": "incytr_pair_count",
            "severity": "diagnostic",
            "expected_pairs": expected_pairs,
            "n_pairs": int(len(pm)),
            "n_senders": len(senders),
            "n_receivers": len(receivers),
            "n_self_pairs": self_pairs,
            "spine_senders_missing": sorted(set(expected) - senders),
            "spine_receivers_missing": sorted(set(expected) - receivers),
            "pass": len(pm) == expected_pairs
                    and senders == set(expected)
                    and receivers == set(expected)
                    and self_pairs == len(expected)}


def _auto_output_path(spine_dir: Path, checks: list[str], include_diagnostics: bool) -> Path:
    if tuple(checks) == HARD_CHECKS and not include_diagnostics:
        return spine_dir / "verification.json"
    suffix = "_".join(checks)
    if include_diagnostics and tuple(checks) == CHECKS:
        suffix = "diagnostics"
    return spine_dir / f"verification.{suffix}.json"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--spine", default="levy_t5")
    ap.add_argument("--checks", nargs="+", choices=CHECKS, default=list(HARD_CHECKS),
                    help="Subset of checks to run (default: hard gates only: "
                         f"{HARD_CHECKS})")
    ap.add_argument("--include-diagnostics", action="store_true",
                    help="Append diagnostic checks (MEA concordance and Incytr "
                         "artifact pair count). Diagnostics are reported but do "
                         "not affect the default exit code.")
    ap.add_argument("--strict-diagnostics", action="store_true",
                    help="Make diagnostic failures affect the exit code. Use for "
                         "investigations, not the viewer hard gate.")
    ap.add_argument("--output", type=Path, default=None,
                    help="Write report to this path. By default, hard-only runs "
                         "write verification.json; diagnostic runs write a "
                         "separate verification.*.json file.")
    args = ap.parse_args()

    checks = list(dict.fromkeys(args.checks))
    if args.include_diagnostics:
        checks = list(dict.fromkeys(checks + list(DIAGNOSTIC_CHECKS)))

    spine_dir = _spine_dir(args.spine)
    spine_clusters = set(icfg.load_cluster_spine())
    weights = (_cluster_weights(spine_clusters)
               if {"mass", "mea"} & set(checks) else None)

    results = []
    for name in checks:
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

    hard_results = [r for r in results if r.get("severity") == "hard"]
    diagnostic_results = [r for r in results if r.get("severity") == "diagnostic"]
    hard_pass = (
        all(r.get("pass") is True for r in hard_results)
        if hard_results else None
    )
    diagnostics_pass = (
        all(r.get("pass") is True for r in diagnostic_results)
        if diagnostic_results else None
    )
    exit_pass = bool(hard_pass is not False)
    if args.strict_diagnostics and diagnostics_pass is False:
        exit_pass = False

    out = {
        "spine": args.spine,
        "checks_requested": checks,
        "hard_checks": list(HARD_CHECKS),
        "diagnostic_checks": list(DIAGNOSTIC_CHECKS),
        "checks": results,
        "hard_pass": hard_pass,
        "diagnostics_pass": diagnostics_pass,
        "all_checks_pass": all(r.get("pass") is True for r in results),
        "exit_pass": exit_pass,
        # Backward-compatible field consumed by older runners/viewer gates.
        "all_pass": exit_pass,
    }
    out_path = args.output or _auto_output_path(
        spine_dir, checks, args.include_diagnostics,
    )
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nWrote {out_path}")
    print(f"hard_pass={out['hard_pass']} diagnostics_pass={out['diagnostics_pass']}")
    print(f"exit_pass={exit_pass}")
    if not exit_pass:
        sys.exit(1)


if __name__ == "__main__":
    main()

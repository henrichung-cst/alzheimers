#!/usr/bin/env python3
"""Print a cached-results summary of the kinase attribution pipeline.

Reads outputs already produced by the four modular stages
(``normalize`, `enrich`, `attribute`,
`mechanism`). Does not run any analysis.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from alz.shared import config

OUTPUT_DIR = config.KINASE_ATTRIBUTION_OUTPUT_DIR


def print_summary():
    print("\n" + "=" * 72)
    print("Kinase Attribution Pipeline — Summary")
    print("=" * 72)

    norm_path = os.path.join(OUTPUT_DIR, "normalization_summary.json")
    if os.path.exists(norm_path):
        with open(norm_path) as f:
            ns = json.load(f)
        print(f"\n--- Stage 1: Normalization ---")
        print(f"  Method: {ns.get('normalization_method', '?')}")
        print(f"  Sites matched to proteins: {ns.get('n_sites_matched', '?')}"
              f"/{ns.get('n_sites_total', '?')} "
              f"({ns.get('pct_matched', '?')}%)")
        print(f"  Valid stoichiometry values: {ns.get('pct_valid_stoich', '?')}%")
        if ns.get("pca_after"):
            pa = ns["pca_after"]
            print(f"  PCA (after norm): PC1={pa.get('pc1_var', '?')}%, "
                  f"PC2={pa.get('pc2_var', '?')}% "
                  f"({pa.get('n_proteins', '?')} proteins)")
    else:
        print("\n--- Stage 1: Not yet run ---")

    mea_path = os.path.join(OUTPUT_DIR, "mea_stoichiometry.csv")
    if os.path.exists(mea_path):
        mea = pd.read_csv(mea_path)
        print(f"\n--- Stage 2: OLS + MEA Kinase Enrichment ---")
        print(f"  MEA stoichiometry entries: {len(mea)}")
        n_sig = (mea["FDR"] < config.MEA_FDR_THRESH).sum()
        print(f"  Significant (FDR<{config.MEA_FDR_THRESH}): {n_sig}")
        if "contrast" in mea.columns:
            for cn in mea["contrast"].unique():
                sub = mea[mea["contrast"] == cn]
                n_s = (sub["FDR"] < config.MEA_FDR_THRESH).sum()
                print(f"    {cn}: {n_s} significant")
    else:
        print("\n--- Stage 2: Not yet run ---")

    attr_path = os.path.join(OUTPUT_DIR, "unified_attribution.csv")
    summ_path = os.path.join(OUTPUT_DIR, "attribution_summary.json")
    print(f"\n--- Stage 3: Unified Attribution ---")
    if os.path.exists(summ_path):
        with open(summ_path) as f:
            summ = json.load(f)
        print(f"  MEA significant kinases: {summ.get('n_mea_significant', '?')}")
        print(f"  Attributed rows: {summ.get('n_attributed', '?')}")
        by_conf = summ.get("by_confidence", {})
        if by_conf:
            print(f"  By confidence:")
            for conf, cnt in by_conf.items():
                print(f"    {conf}: {cnt}")
        by_ct = summ.get("by_cell_type", {})
        if by_ct:
            print(f"  By cell type (subclass):")
            for ct, cnt in sorted(by_ct.items(), key=lambda x: -x[1]):
                print(f"    {ct}: {cnt}")
    elif os.path.exists(attr_path):
        attr = pd.read_csv(attr_path)
        print(f"  Attributed rows: {len(attr)}")
        if "combined_confidence" in attr.columns:
            for conf, cnt in attr["combined_confidence"].value_counts().items():
                print(f"    {conf}: {cnt}")
    else:
        print("  Not yet run")

    mech_path = os.path.join(OUTPUT_DIR, "mechanism_annotation.csv")
    if os.path.exists(mech_path):
        mech = pd.read_csv(mech_path)
        print(f"\n--- Mechanism Annotation (supplementary) ---")
        print(f"  Total entries: {len(mech)}")
        for m, c in mech["mechanism"].value_counts().items():
            print(f"    {m}: {c}")

    print()


def main():
    argparse.ArgumentParser(
        description="Print cached-results summary of the kinase pipeline."
    ).parse_args()
    print_summary()


if __name__ == "__main__":
    main()

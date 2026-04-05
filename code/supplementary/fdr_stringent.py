#!/usr/bin/env python3
"""Q4: Stringent FDR analysis — compare MEA results at FDR < 0.10 vs FDR < 0.25.

Reads canonical pipeline outputs and produces a side-by-side comparison showing
which kinase-contrast pairs survive the stricter threshold.

Usage:
    python code/supplementary/fdr_stringent.py --run
    python code/supplementary/fdr_stringent.py --summary
"""

import argparse
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

OUTPUT_DIR = os.path.join(config.SUPPLEMENTARY_OUTPUT_DIR, "fdr_stringent")

FDR_STRICT = 0.10
FDR_DEFAULT = config.MEA_FDR_THRESH  # 0.25


def _ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def step_run():
    """Run the stringent FDR comparison."""
    _ensure_output_dir()
    print("\n=== Stringent FDR Analysis (FDR < 0.10 vs FDR < 0.25) ===\n")

    # Load MEA results
    mea_path = os.path.join(config.KINASE_ATTRIBUTION_OUTPUT_DIR, "mea_stoichiometry.csv")
    if not os.path.exists(mea_path):
        raise FileNotFoundError(f"{mea_path} not found. Run kinase_attribution.py --enrich first.")
    mea = pd.read_csv(mea_path)
    print(f"  MEA results loaded: {len(mea)} total rows")

    # Filter at both thresholds
    sig_default = set(zip(mea.loc[mea["FDR"] < FDR_DEFAULT, "kinase"],
                          mea.loc[mea["FDR"] < FDR_DEFAULT, "contrast"]))
    sig_strict = set(zip(mea.loc[mea["FDR"] < FDR_STRICT, "kinase"],
                         mea.loc[mea["FDR"] < FDR_STRICT, "contrast"]))

    mea_strict = mea[mea["FDR"] < FDR_STRICT].copy()
    mea_strict_path = os.path.join(OUTPUT_DIR, "mea_fdr010.csv")
    mea_strict.to_csv(mea_strict_path, index=False)
    print(f"  FDR < {FDR_DEFAULT}: {len(sig_default)} kinase-contrast pairs")
    print(f"  FDR < {FDR_STRICT}: {len(sig_strict)} kinase-contrast pairs")
    print(f"  Saved {mea_strict_path}")

    # Build comparison table
    mea_sig = mea[mea["FDR"] < FDR_DEFAULT].copy()
    mea_sig["survives_strict"] = mea_sig.set_index(
        ["kinase", "contrast"]).index.isin(sig_strict)
    comp_path = os.path.join(OUTPUT_DIR, "fdr_comparison.csv")
    mea_sig.to_csv(comp_path, index=False)
    print(f"  Saved {comp_path}")

    # Attribution table subset (if unified attribution exists)
    attr_path = os.path.join(config.KINASE_ATTRIBUTION_OUTPUT_DIR, "unified_attribution.csv")
    if os.path.exists(attr_path):
        attr = pd.read_csv(attr_path)
        attr_strict = attr[
            attr.set_index(["kinase", "contrast"]).index.isin(sig_strict)
        ].copy()
        attr_strict_path = os.path.join(OUTPUT_DIR, "attribution_fdr010.csv")
        attr_strict.to_csv(attr_strict_path, index=False)
        print(f"  Attribution at FDR < {FDR_STRICT}: {len(attr_strict)} rows "
              f"(from {len(attr)} at FDR < {FDR_DEFAULT})")
        print(f"  Saved {attr_strict_path}")

    # Summary stats
    n_dropped = len(sig_default) - len(sig_strict)
    summary = {
        "fdr_default": FDR_DEFAULT,
        "fdr_strict": FDR_STRICT,
        "n_pairs_default": len(sig_default),
        "n_pairs_strict": len(sig_strict),
        "n_dropped": n_dropped,
        "pct_retained": round(100 * len(sig_strict) / max(len(sig_default), 1), 1),
    }
    summary_path = os.path.join(OUTPUT_DIR, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  {n_dropped} kinase-contrast pairs dropped at stricter threshold "
          f"({summary['pct_retained']}% retained)")
    print(f"  Saved {summary_path}")


def step_summary():
    """Print cached summary."""
    summary_path = os.path.join(OUTPUT_DIR, "summary.json")
    if not os.path.exists(summary_path):
        print("No summary found. Run --run first.")
        return
    with open(summary_path) as f:
        s = json.load(f)
    print(f"\nStringent FDR Analysis:")
    print(f"  FDR < {s['fdr_default']}: {s['n_pairs_default']} kinase-contrast pairs")
    print(f"  FDR < {s['fdr_strict']}: {s['n_pairs_strict']} kinase-contrast pairs")
    print(f"  Dropped: {s['n_dropped']} ({s['pct_retained']}% retained)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="Run analysis")
    parser.add_argument("--summary", action="store_true", help="Print cached summary")
    args = parser.parse_args()

    if not any([args.run, args.summary]):
        parser.print_help()
        sys.exit(1)
    if args.run:
        step_run()
    if args.summary:
        step_summary()

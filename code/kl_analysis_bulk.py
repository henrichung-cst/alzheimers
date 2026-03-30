"""Kinase enrichment analysis on bulk (non-deconvoluted) phosphoproteomics data."""

import argparse

import config
from kl_analysis_clusters import (
    run_enrichment_step,
    run_plot_step,
    run_prepare_step,
    run_summarize_step,
)

# Canonical condition names → bulk column suffixes
COND_MAP = config.BULK_CONDITION_MAP


def get_comparisons(test_mode=False):
    genders = ["M"]
    conditions = ["Ttau", "AppP", "ApTt"]  # canonical names
    timepoints = ["2mo", "4mo", "6mo"]

    if test_mode:
        print("Running in TEST mode: limiting scope to 1 condition, 1 timepoint.")
        conditions = ["Ttau"]
        timepoints = ["2mo"]

    comparisons = []
    for gender in genders:
        for condition in conditions:
            col_suffix = COND_MAP[condition]
            for timepoint in timepoints:
                bg = f"{gender}_{timepoint}_WT"
                fg = f"{gender}_{timepoint}_{col_suffix}"
                comparisons.append({
                    "bg": bg,
                    "fg": fg,
                    "percent_thresh": config.PERCENT_THRESH,
                    "cluster": "bulk",
                    "gender": gender,
                    "condition": condition,
                    "timepoint": timepoint,
                })
    return comparisons


def main():
    parser = argparse.ArgumentParser(description="Bulk Kinase Enrichment Analysis")
    parser.add_argument("--step", choices=["prepare", "enrich", "summarize", "plot", "all"],
                        default="all", help="Execution step to run (default: all)")
    parser.add_argument("--test", action="store_true", help="Run in test mode (limited scope)")
    parser.add_argument("--volcano", action="store_true",
                        help="Generate per-comparison volcano plots")
    parser.add_argument("--kin-type", choices=["ser_thr", "tyrosine"], default="ser_thr",
                        help="Kinase type to analyze (default: ser_thr)")
    parser.add_argument("--percent-thresh", type=float, default=None,
                        help="Override percentile threshold (default: config.PERCENT_THRESH)")
    args = parser.parse_args()

    comparisons = get_comparisons(args.test)
    if args.percent_thresh is not None:
        for comp in comparisons:
            comp["percent_thresh"] = args.percent_thresh

    # Dynamic output directory based on kinase type
    output_dir = "outputs/bulk" if args.kin_type == "ser_thr" else "outputs/bulk_tyrosine"
    input_file = config.get_bulk_input_file(args.kin_type)

    all_kinases = None
    kinase_expression_info = None
    if args.step in ["prepare", "enrich", "summarize", "plot", "all"]:
        all_kinases, kinase_expression_info = run_prepare_step(kin_type=args.kin_type)

    if args.step in ["enrich", "all"]:
        run_enrichment_step(comparisons, kinase_expression_info=kinase_expression_info,
                           input_file=input_file, output_dir=output_dir,
                           kin_type=args.kin_type)

    if args.step in ["summarize", "all"]:
        run_summarize_step(comparisons, kinase_expression_info=kinase_expression_info,
                          output_dir=output_dir, gender_map=config.BULK_GENDER_MAP)

    if args.step in ["plot", "all"]:
        run_plot_step(comparisons, kinase_expression_info=kinase_expression_info,
                     volcano=args.volcano, output_dir=output_dir,
                     gender_map=config.BULK_GENDER_MAP)


if __name__ == "__main__":
    main()

"""Deconvolution feasibility — marker → composition concordance via factorial OLS.

Tests whether bulk total-proteome marker abundance tracks per-animal WMB-class
composition fractions under disease perturbation, after OLS-controlling for
genotype, age, and sex. This closes the deconvolution-feasibility question
(see docs/foundation/analysis_rationale.md): a precondition for any
deconvolution attempt is that bona fide cell-type markers correlate positively
with their cell type's fraction across animals; this diagnostic shows they do
not, even after disease correction, and that the most-specific markers
actively *anti-track* composition (consistent with disease-driven per-cell
expression dynamics dominating over fraction shifts).

For each high-specificity gene × WMB cell type pair, fits the same factorial
OLS used by alz/kinase_enrich.py on (a) IRS-normalized log2 protein abundance
and (b) per-animal composition fraction, then correlates the 9 disease×time
β coefficients across the pair. A specificity-threshold sweep shows that
tightening the marker pool worsens (rather than rescues) the null result.

Inputs (canonical pipeline outputs):
    outputs/reports/wmb_expression/wmb_proteome_expression.csv
    outputs/reports/data_ingest/sample_mapping.csv
Raw inputs:
    data/datasets/song/primary/proteomics/song2024_tmttotal_protein_quant_*.xlsx
    data/datasets/song/transcriptomics/170_gex_celltypes_00.h5ad

Outputs (under outputs/reports/supplementary/deconvolution_feasibility/):
    marker_concordance.csv              one row per (gene × cell type)
    marker_concordance_summary.csv      per-cell-type rollup
    marker_concordance_sweep.csv        specificity threshold sweep
    marker_concordance_distribution.png per-cell-type box plot
    marker_concordance_threshold_sweep.png  3-panel diagnostic
    summary.json                        headline metrics

Usage:
    python alz/supplementary/deconvolution_feasibility.py --run
    python alz/supplementary/deconvolution_feasibility.py --summary
"""

import argparse
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config  # noqa: E402

OUTPUT_DIR = os.path.join(config.SUPPLEMENTARY_OUTPUT_DIR, "deconvolution_feasibility")

TOTAL_PROTEOME_FILE = os.path.join(
    config.SONG_PRIMARY_PROTEOMICS_DIR,
    "song2024_tmttotal_protein_quant_merged_labeled (2).xlsx",
)
SAMPLE_MAPPING_FILE = os.path.join(
    config.DATA_INGEST_OUTPUT_DIR, "sample_mapping.csv",
)

CONTRASTS = [
    ("App",  "2mo", ["App"]),
    ("App",  "4mo", ["App", "App:t4"]),
    ("App",  "6mo", ["App", "App:t6"]),
    ("Tau",  "2mo", ["Tau"]),
    ("Tau",  "4mo", ["Tau", "Tau:t4"]),
    ("Tau",  "6mo", ["Tau", "Tau:t6"]),
    ("ApTt", "2mo", ["App", "Tau", "Int"]),
    ("ApTt", "4mo", ["App", "Tau", "Int", "App:t4", "Tau:t4"]),
    ("ApTt", "6mo", ["App", "Tau", "Int", "App:t6", "Tau:t6"]),
]
CONTRAST_LABELS = [f"{g}_{t}" for g, t, _ in CONTRASTS]
SPEC_THRESHOLDS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]


def _ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def _proteome_ref_col(plex):
    return f"plex{plex}_{config.TMT_REF_CHANNEL}_sn_mean"


def _load_proteome_normalized():
    """IRS-normalize the total proteome and return a log2 + median-centered
    DataFrame indexed by uppercase gene symbol, columns = the 72 sample columns,
    plus the sample mapping for design-matrix construction.
    """
    if not os.path.exists(SAMPLE_MAPPING_FILE):
        raise FileNotFoundError(
            f"sample_mapping.csv not found at {SAMPLE_MAPPING_FILE}. "
            "Run: python alz/data_ingest.py --mapping"
        )
    tp = pd.read_excel(TOTAL_PROTEOME_FILE, header=1)
    mapping = pd.read_csv(SAMPLE_MAPPING_FILE)
    bio_cols = mapping["column_name"].tolist()
    plexes = sorted(mapping["plex"].unique())
    ref_cols = {p: _proteome_ref_col(p) for p in plexes}
    sample_to_plex = dict(zip(mapping["column_name"], mapping["plex"]))

    quant_raw = tp[bio_cols + list(ref_cols.values())].apply(
        pd.to_numeric, errors="coerce"
    )
    ref_mat = pd.DataFrame(
        {p: quant_raw[c] for p, c in ref_cols.items() if c in quant_raw.columns}
    )
    global_ref = ref_mat.mean(axis=1, skipna=True)
    norm = quant_raw[bio_cols].copy()
    for col, plex in sample_to_plex.items():
        rc = ref_cols[plex]
        with np.errstate(divide="ignore", invalid="ignore"):
            norm[col] = (quant_raw[col].values / quant_raw[rc].values) * global_ref.values
    log2 = np.log2(norm.replace([0, np.inf, -np.inf], np.nan))
    log2 = log2.subtract(log2.median(axis=0), axis=1)
    log2["gene"] = tp["Gene Symbol"].astype(str).str.upper()
    log2 = log2.dropna(subset=["gene"]).set_index("gene")
    log2 = log2[~log2.index.duplicated(keep="first")]
    return log2, mapping


def _load_composition_per_animal():
    """Per-animal WMB-class composition fractions from the paired Song h5ad."""
    import anndata as ad

    adata = ad.read_h5ad(config.SONG_H5AD_FILE, backed="r")
    obs = adata.obs[["sample", "class_name", "class_prob"]].copy()
    obs = obs[obs["class_prob"] >= config.SONG_MIN_SUBCLASS_PROB]
    obs = obs.dropna(subset=["class_name"])
    obs["animal"] = obs["sample"].str.split("_").str[0]
    counts = obs.groupby(["animal", "class_name"], observed=True).size().unstack(fill_value=0)
    fracs = counts.div(counts.sum(axis=1), axis=0)

    manifest = pd.read_csv(config.WMB_CLASS_MANIFEST_FILE)
    fracs = fracs.rename(columns=dict(zip(manifest["class_name"], manifest["class_label"])))
    fracs = fracs.T.groupby(level=0).sum().T
    return fracs


def _build_factorial_design(rows):
    """Full-cohort design: const + App + Tau + Int + female + t4 + t6 + interactions."""
    X = pd.DataFrame(index=rows.index)
    X["const"] = 1.0
    X["App"] = rows["genotype"].isin(["AppP", "ApTt"]).astype(float)
    X["Tau"] = rows["genotype"].isin(["Ttau", "ApTt"]).astype(float)
    X["Int"] = (rows["genotype"] == "ApTt").astype(float)
    X["female"] = (rows["sex"] == "F").astype(float)
    X["t4"] = (rows["timepoint"] == "4mo").astype(float)
    X["t6"] = (rows["timepoint"] == "6mo").astype(float)
    X["App:t4"] = X["App"] * X["t4"]
    X["App:t6"] = X["App"] * X["t6"]
    X["Tau:t4"] = X["Tau"] * X["t4"]
    X["Tau:t6"] = X["Tau"] * X["t6"]
    return X


def _fit_contrasts(y, X):
    """Fit OLS y ~ X (NaN-safe), return the 9 contrast β's or None on failure."""
    if np.isnan(y).any():
        mask = ~np.isnan(y)
        if mask.sum() < X.shape[1] + 2:
            return None
        Xf, yf = X.loc[mask], y[mask]
    else:
        Xf, yf = X, y
    try:
        res = sm.OLS(yf, Xf).fit()
    except Exception:
        return None
    return np.array([sum(res.params.get(t, 0.0) for t in terms) for _, _, terms in CONTRASTS])


def step_run():
    _ensure_output_dir()
    print("Deconvolution feasibility — marker → composition concordance (factorial OLS)")

    if not os.path.exists(config.WMB_PROTEOME_EXPRESSION_FILE):
        raise FileNotFoundError(
            f"WMB proteome expression not found at {config.WMB_PROTEOME_EXPRESSION_FILE}. "
            "Run: python alz/wmb_expression.py --proteome"
        )

    print("  Loading & IRS-normalizing total proteome ...")
    proteome, mapping = _load_proteome_normalized()
    X_prot = _build_factorial_design(mapping.set_index("column_name"))
    proteome_aligned = proteome[X_prot.index]

    print(f"  Fitting OLS on {proteome_aligned.shape[0]:,} genes ...")
    prot_betas = {
        gene: b for gene in proteome_aligned.index
        if (b := _fit_contrasts(proteome_aligned.loc[gene].values, X_prot)) is not None
    }
    prot_beta_df = pd.DataFrame(prot_betas, index=CONTRAST_LABELS).T
    print(f"    {len(prot_beta_df):,} gene fits")

    print("  Loading per-animal composition ...")
    fracs = _load_composition_per_animal()
    paired = mapping[mapping["has_snrna_seq"]].copy()
    paired["ac"] = paired["mouse_id"].astype(str)
    paired = paired[paired["ac"].isin(fracs.index)].drop_duplicates("ac")
    X_comp = _build_factorial_design(paired.set_index("ac"))
    fracs_aligned = fracs.loc[X_comp.index]
    print(f"  Fitting OLS on {fracs_aligned.shape[1]} cell types × {len(X_comp)} paired animals ...")
    comp_betas = {
        ct: b for ct in fracs_aligned.columns
        if (b := _fit_contrasts(fracs_aligned[ct].values, X_comp)) is not None
    }
    comp_beta_df = pd.DataFrame(comp_betas, index=CONTRAST_LABELS).T

    print("  Computing β-vs-β concordance (specificity > 0.30 markers) ...")
    wmb = pd.read_csv(config.WMB_PROTEOME_EXPRESSION_FILE)
    wmb["gene_upper"] = wmb["gene_symbol_mouse"].str.upper()

    primary = wmb[(wmb["specificity_score"] > 0.30) & wmb["binary_expressed"]]
    rows = []
    for _, w in primary.iterrows():
        gene, ct, spec = w["gene_upper"], w["cell_type"], w["specificity_score"]
        if gene not in prot_beta_df.index or ct not in comp_beta_df.index:
            continue
        a, b = prot_beta_df.loc[gene].values, comp_beta_df.loc[ct].values
        if np.std(a) == 0 or np.std(b) == 0:
            continue
        r, p = stats.pearsonr(a, b)
        rows.append({
            "gene_symbol": gene,
            "cell_type": ct,
            "specificity_score": float(spec),
            "n_contrasts": len(a),
            "concordance_r": float(r),
            "concordance_p": float(p),
        })
    detail = pd.DataFrame(rows)
    detail_path = os.path.join(OUTPUT_DIR, "marker_concordance.csv")
    detail.to_csv(detail_path, index=False)
    print(f"    {len(detail)} (gene × cell type) pairs → {detail_path}")

    summary = (
        detail.groupby("cell_type")
              .agg(n_markers=("concordance_r", "size"),
                   mean_r=("concordance_r", "mean"),
                   median_r=("concordance_r", "median"),
                   n_pos=("concordance_r", lambda s: int((s > 0).sum())),
                   n_neg=("concordance_r", lambda s: int((s < 0).sum())),
                   n_pos_sig=("concordance_p",
                              lambda s: int(((s < 0.05) &
                                             (detail.loc[s.index, "concordance_r"] > 0)).sum())),
                   n_neg_sig=("concordance_p",
                              lambda s: int(((s < 0.05) &
                                             (detail.loc[s.index, "concordance_r"] < 0)).sum())))
              .reset_index()
              .sort_values("n_markers", ascending=False)
    )
    summary_path = os.path.join(OUTPUT_DIR, "marker_concordance_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"    Per-cell-type rollup → {summary_path}")

    print("  Specificity-threshold sweep ...")
    sweep_rows = []
    for thr in SPEC_THRESHOLDS:
        pool = wmb[(wmb["specificity_score"] > thr) & wmb["binary_expressed"]]
        rs, ps = [], []
        for _, w in pool.iterrows():
            gene, ct = w["gene_upper"], w["cell_type"]
            if gene not in prot_beta_df.index or ct not in comp_beta_df.index:
                continue
            a, b = prot_beta_df.loc[gene].values, comp_beta_df.loc[ct].values
            if np.std(a) == 0 or np.std(b) == 0:
                continue
            r, p = stats.pearsonr(a, b)
            rs.append(r); ps.append(p)
        if not rs:
            continue
        rs_arr, ps_arr = np.array(rs), np.array(ps)
        sweep_rows.append({
            "specificity_threshold": thr,
            "n_pairs": int(len(rs_arr)),
            "mean_r": float(rs_arr.mean()),
            "median_r": float(np.median(rs_arr)),
            "frac_pos": float((rs_arr > 0).mean()),
            "frac_sig_pos": float(((ps_arr < 0.05) & (rs_arr > 0)).mean()),
            "frac_sig_neg": float(((ps_arr < 0.05) & (rs_arr < 0)).mean()),
        })
    sweep = pd.DataFrame(sweep_rows)
    sweep_path = os.path.join(OUTPUT_DIR, "marker_concordance_sweep.csv")
    sweep.to_csv(sweep_path, index=False)
    print(f"    Sweep ({len(sweep)} thresholds) → {sweep_path}")

    cts = summary["cell_type"].tolist()
    fig, ax = plt.subplots(figsize=(min(1.0 + 0.45 * len(cts), 16), 5))
    data = [detail[detail["cell_type"] == ct]["concordance_r"].values for ct in cts]
    bp = ax.boxplot(data, labels=cts, showfliers=False, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("#cce5ff"); patch.set_edgecolor("#1f4e79")
    rng = np.random.default_rng(0)
    for i, ys in enumerate(data):
        xs = rng.normal(i + 1, 0.06, size=len(ys))
        ax.scatter(xs, ys, s=8, alpha=0.5, c="#222")
    ax.axhline(0, color="k", lw=0.5, ls="--")
    ax.set_ylabel("Concordance r:  protein β  vs  composition β  (across 9 disease×time contrasts)")
    ax.set_xlabel("WMB cell type")
    ax.set_title(
        "Marker → composition concordance after disease/age/sex correction\n"
        "If markers track composition under disease perturbation → boxes shifted positive."
    )
    plt.setp(ax.get_xticklabels(), rotation=40, ha="right")
    fig.tight_layout()
    dist_path = os.path.join(OUTPUT_DIR, "marker_concordance_distribution.png")
    fig.savefig(dist_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"    Distribution box plot → {dist_path}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    ax = axes[0]
    ax.plot(sweep["specificity_threshold"], sweep["mean_r"], "o-", label="mean", color="#1f4e79")
    ax.plot(sweep["specificity_threshold"], sweep["median_r"], "s--", label="median", color="#c00000")
    ax.axhline(0, color="k", lw=0.5, ls=":")
    ax.set_xlabel("Specificity threshold (>)")
    ax.set_ylabel("Concordance r")
    ax.set_title("Center of concordance distribution\n(should rise toward +1 if markers work)")
    ax.legend()

    ax = axes[1]
    ax.plot(sweep["specificity_threshold"], sweep["frac_pos"] * 100, "o-", color="#1f4e79")
    ax.axhline(50, color="k", lw=0.5, ls=":", label="null (50%)")
    ax.set_xlabel("Specificity threshold (>)")
    ax.set_ylabel("% pairs with r > 0")
    ax.set_title("Directional agreement\n(should rise toward 100%)")
    ax.legend()

    ax = axes[2]
    ax.plot(sweep["specificity_threshold"], sweep["frac_sig_pos"] * 100,
            "o-", color="#1f4e79", label="sig pos (concordant)")
    ax.plot(sweep["specificity_threshold"], sweep["frac_sig_neg"] * 100,
            "o--", color="#c00000", label="sig neg (anti-marker)")
    ax.axhline(2.5, color="k", lw=0.5, ls=":", label="α = 0.05 baseline")
    ax.set_xlabel("Specificity threshold (>)")
    ax.set_ylabel("% pairs with p < 0.05")
    ax.set_title("Significant hits\n(blue should outpace red if markers work)")
    ax.legend()
    for ax in axes:
        ax.set_xticks(SPEC_THRESHOLDS)
        ax.tick_params(axis="x", rotation=45)
    fig.suptitle(
        "Threshold sweep: does concordance strengthen as we restrict to more-specific markers?",
        fontsize=11, y=1.02,
    )
    fig.tight_layout()
    sweep_fig = os.path.join(OUTPUT_DIR, "marker_concordance_threshold_sweep.png")
    fig.savefig(sweep_fig, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"    Threshold-sweep figure → {sweep_fig}")

    overall = detail["concordance_r"]
    headline = {
        "n_pairs": int(len(detail)),
        "mean_r": float(overall.mean()),
        "median_r": float(overall.median()),
        "frac_pos": float((overall > 0).mean()),
        "frac_sig_pos": float(((detail["concordance_p"] < 0.05) & (overall > 0)).mean()),
        "frac_sig_neg": float(((detail["concordance_p"] < 0.05) & (overall < 0)).mean()),
    }
    summary_json = os.path.join(OUTPUT_DIR, "summary.json")
    with open(summary_json, "w") as f:
        json.dump(headline, f, indent=2)

    print(f"\n  HEADLINE (specificity > 0.30, n = {headline['n_pairs']}):")
    print(f"    mean r        = {headline['mean_r']:+.3f}")
    print(f"    median r      = {headline['median_r']:+.3f}")
    print(f"    fraction r>0  = {headline['frac_pos']:.1%}  (50% = null)")
    print(f"    fraction sig+ = {headline['frac_sig_pos']:.1%}")
    print(f"    fraction sig− = {headline['frac_sig_neg']:.1%}")


def step_summary():
    summary_json = os.path.join(OUTPUT_DIR, "summary.json")
    if not os.path.exists(summary_json):
        print("Deconvolution feasibility: not run yet")
        return
    with open(summary_json) as f:
        h = json.load(f)
    print("Deconvolution feasibility — marker → composition concordance:")
    print(f"  Pairs (specificity > 0.30): {h['n_pairs']}")
    print(f"  Mean r       = {h['mean_r']:+.3f}")
    print(f"  Median r     = {h['median_r']:+.3f}")
    print(f"  Fraction r>0 = {h['frac_pos']:.1%}  (50% = null)")
    print(f"  Sig+ vs sig− = {h['frac_sig_pos']:.1%} vs {h['frac_sig_neg']:.1%}")


def main():
    parser = argparse.ArgumentParser(
        description="Deconvolution feasibility diagnostic (marker → composition concordance)."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run", action="store_true", help="Run the full diagnostic")
    group.add_argument("--summary", action="store_true", help="Print cached headline metrics")
    args = parser.parse_args()
    if args.run:
        step_run()
    elif args.summary:
        step_summary()


if __name__ == "__main__":
    main()

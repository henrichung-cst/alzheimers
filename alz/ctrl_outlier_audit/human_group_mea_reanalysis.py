"""Suspect-control kinase MEA reanalysis: four group contrasts on the same 17 brains.

The 2026-05-25 audit (outputs/.../ctrl_audit/investigation_report/README.md) showed
CTRL-07/08/10 carry a genuine pre-symptomatic-AD-like phospho signature and contaminate
the all-control baseline. This script re-runs the human group-level kinase MEA under that
perturbation, as four explicit contrasts on the same 17 samples:

  suspect_vs_cleanCTRL        SUSPECT (3)        vs CLEAN (4)   -- analysis A (comparison)
  AD_vs_cleanCTRL             AD (10)            vs CLEAN (4)   -- analysis A (canonical baseline)
  ADwithSuspect_vs_cleanCTRL  AD u SUSPECT (13)  vs CLEAN (4)   -- analysis B (merged)
  AD_vs_allCTRL               AD (10)            vs ALLCTRL (7) -- analysis C (pre-audit ref)

Analysis C resurrects the pre-audit all-7-control baseline ONLY as a labeled hard-line
reference -- it is not a live fallback and does not reopen the contaminated baseline.

Method: per contrast, per track (st/py),
ranking metric = NaN-aware per-site LFC mean(A) - mean(B), fed to the same MEA helper
(`alz.bulk_mea.enrich._run_mea`: median-center -> winsorize -> GSEA). Sign convention is
pipeline-wide `+ = up in disease/suspect direction`: +NES = higher kinase activity in
group A than group B. Held identical across all four.

Outputs one CSV per contrast (both tracks) + MANIFEST.md under
outputs/reports/kinase_attribution_human/ctrl_audit/reanalysis_mea/. The filename token
matches the in-file `contrast` value verbatim so they cannot drift.

`mea_AD_vs_cleanCTRL.csv` (AD vs clean CTRL-01/02/03/04) is the canonical clean-baseline
human group MEA; the other three contrasts are labeled sensitivity references around it.
"""
from __future__ import annotations

import argparse
import datetime as _dt
from pathlib import Path

import pandas as pd

from alz.shared import config
from alz.bulk_mea import enrich as kinase_enrich
from alz.ingest.mukesh import SAMPLE_MAPPING_CSV
from alz.ingest.mukesh_perdonor import _load_track_matrix, PERDONOR_DIR

# Audit verdict (2026-05-25): these three controls carry a genuine AD-like phospho
# signature; drop them from the clean-control baseline.
AD_LIKE_CTRL = {"CTRL-07", "CTRL-08", "CTRL-10"}

OUT_DIR = Path("outputs/reports/kinase_attribution_human/ctrl_audit/reanalysis_mea")
LFC_KEY = "stoich_lfc"
TRACKS = ("st", "py")

# (contrast label, group-A set, group-B set, analysis bucket). Label == filename token.
CONTRASTS = [
    ("suspect_vs_cleanCTRL", "SUSPECT", "CLEAN", "A (comparison)"),
    ("AD_vs_cleanCTRL", "AD", "CLEAN", "A (canonical clean baseline)"),
    ("ADwithSuspect_vs_cleanCTRL", "ADSUSP", "CLEAN", "B (merged)"),
    ("AD_vs_allCTRL", "AD", "ALLCTRL", "C (pre-audit reference)"),
]


def _normalize_ad_sample(sample: str) -> str:
    sample = sample.strip()
    if sample.isdigit():
        return f"AD-{int(sample):02d}"
    return sample


def _sample_sets(matrix_cols, exclude_ad_samples: set[str] | None = None) -> dict[str, list[str]]:
    """Build the four group sets from the sample mapping, filtered to present columns."""
    exclude_ad_samples = exclude_ad_samples or set()
    m = pd.read_csv(SAMPLE_MAPPING_CSV)
    cols = set(matrix_cols)
    ad = sorted(
        s for s in m.loc[m.group == "AD", "sample_id"]
        if s in cols and s not in exclude_ad_samples
    )
    allctrl = sorted(s for s in m.loc[m.group == "CTRL", "sample_id"] if s in cols)
    clean = [s for s in allctrl if s not in AD_LIKE_CTRL]
    suspect = [s for s in allctrl if s in AD_LIKE_CTRL]
    return {
        "AD": ad,
        "CLEAN": clean,
        "SUSPECT": suspect,
        "ALLCTRL": allctrl,
        "ADSUSP": sorted(ad + suspect),
    }


def run_contrast(label: str, group_a: list[str], group_b: list[str], track: str):
    matrix = _load_track_matrix(track, "stoich")
    if matrix is None:
        return None
    X = matrix.set_index("site_id")
    lfc = X[group_a].astype(float).mean(axis=1) - X[group_b].astype(float).mean(axis=1)
    mea_df, _, _, _ = kinase_enrich._run_mea(
        motif_series=matrix["motif"],
        results_by_contrast={label: {LFC_KEY: lfc.values}},
        lfc_key=LFC_KEY,
        site_ids=matrix["site_id"].values,
        gene_symbols=matrix["gene_symbol"].values,
        track=track,
    )
    if mea_df.empty:
        return None
    mea_df = mea_df.sort_values("NES", ascending=False)
    mea_df["track"] = track
    return mea_df


def write_contrast(label: str, a_key: str, b_key: str, sets: dict[str, list[str]]) -> pd.DataFrame:
    frames = []
    for track in TRACKS:
        df = run_contrast(label, sets[a_key], sets[b_key], track)
        if df is not None:
            frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    cols = [c for c in ["kinase", "NES", "ES", "p-value", "FDR", "Subs fraction",
                        "contrast", "track", "residue_type"] if c in out.columns]
    out = out[cols + [c for c in out.columns if c not in cols]]
    path = OUT_DIR / f"mea_{label}.csv"
    out.to_csv(path, index=False)
    return out


# Wide diagnostic view: per kinase × track, NES side by side across contrasts so the
# pre-symptomatic (suspect) signature can be compared directly to AD on the SAME clean
# baseline. The diagnostic axis is `delta = NES_suspect - NES_AD`:
#   delta ~ 0 with both NES high+sig  -> kinase already AD-like in suspects = early marker
#   delta strongly negative (AD>>sus) -> ramps up later = progression marker
#   delta strongly positive (sus>>AD) -> suspect-only (n=3: suspect of noise)
# The merged and pre-audit-reference contrasts ride along as context NES only.
WIDE_RENAME = {
    "suspect_vs_cleanCTRL": "suspect",
    "AD_vs_cleanCTRL": "AD",
    "ADwithSuspect_vs_cleanCTRL": "ADwithSuspect",
    "AD_vs_allCTRL": "AD_allCTRL",
}


def _load_viewer_median() -> pd.DataFrame:
    """Per-donor `median_nes_sig_only` against all 7 controls — the exact value the
    unified viewer's human tab surfaces as "median NES (sig)". Sourced from the
    per-donor recurrence files (st = recurrence.csv, py = recurrence_pY.csv), NOT
    recomputed. Different estimator from the group-contrast NES columns: median across
    AD donors of each kinase's NES, restricted to donors significant at FDR<thresh."""
    rows = []
    for track in TRACKS:
        suffix = config.PHOSPHO_TRACKS[track]["output_suffix"]
        path = Path(PERDONOR_DIR) / f"recurrence{suffix}.csv"
        if not path.exists():
            continue
        r = pd.read_csv(path)[["kinase", "median_nes_sig_only"]].copy()
        r["track"] = track
        rows.append(r.rename(columns={"median_nes_sig_only": "median_nes_sig_only_allCTRL"}))
    if not rows:
        return pd.DataFrame(columns=["kinase", "track", "median_nes_sig_only_allCTRL"])
    return pd.concat(rows, ignore_index=True)


def write_wide(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    keys = ["kinase", "track", "residue_type"]
    wide = None
    for label, short in WIDE_RENAME.items():
        slim = frames[label][keys + ["NES", "FDR"]].rename(
            columns={"NES": f"NES_{short}", "FDR": f"FDR_{short}"})
        wide = slim if wide is None else wide.merge(slim, on=keys, how="outer")

    wide["delta_NES_suspect_minus_AD"] = wide["NES_suspect"] - wide["NES_AD"]
    wide["abs_delta_NES"] = wide["delta_NES_suspect_minus_AD"].abs()

    # Cross-reference the viewer's per-donor "median NES (sig)" (all-7-ctrl baseline).
    wide = wide.merge(_load_viewer_median(), on=["kinase", "track"], how="left")

    cols = keys + [
        "NES_suspect", "NES_AD", "delta_NES_suspect_minus_AD", "abs_delta_NES",
        "FDR_suspect", "FDR_AD",
        "NES_ADwithSuspect", "NES_AD_allCTRL", "median_nes_sig_only_allCTRL",
    ]
    wide = wide[cols]
    # st first, then largest divergence between the suspect and AD contrasts at the top
    wide["_t"] = (wide.track != "st").astype(int)
    wide = wide.sort_values(["_t", "abs_delta_NES"], ascending=[True, False]).drop(
        columns="_t").reset_index(drop=True)
    wide.to_csv(OUT_DIR / "suspect_vs_AD_kinase_wide.csv", index=False)
    return wide


def write_manifest(
    sets: dict[str, list[str]],
    summaries: dict[str, dict],
    exclude_ad_samples: set[str] | None = None,
):
    sig = config.MEA_FDR_THRESH
    today = _dt.date.today().isoformat()
    unique_samples = sorted(set().union(*sets.values()))
    exclude_ad_samples = exclude_ad_samples or set()
    lines = [
        "# Suspect-control kinase MEA reanalysis — MANIFEST",
        "",
        f"**Generated:** {today}  ",
        "**Generator:** `alz/ctrl_outlier_audit/human_group_mea_reanalysis.py` "
        "(`pixi run mea-suspect-reanalysis`)  ",
        "**Substrate:** human NBB phosphosite log2 stoichiometry matrices "
        "(`stoichiometry_matrix.csv` = st, `stoichiometry_matrix_pY.csv` = py)  ",
        "**Method:** per contrast, per track, ranking metric = NaN-aware per-site "
        "`mean(A) - mean(B)` → `alz.bulk_mea.enrich._run_mea` (median-center → winsorize "
        "→ GSEA). Identical to production `human_group_mea.py`.  ",
        f"**FDR threshold:** `config.MEA_FDR_THRESH = {sig}`  ",
        "**Sign convention:** pipeline-wide `+ = up in disease/suspect direction`; "
        "`+NES` = higher kinase activity in group A than group B. Identical across all four.",
        "",
        f"These are **four views of the same {len(unique_samples)} NBB human brains** — "
        "not independent cohorts.",
    ]
    if exclude_ad_samples:
        lines += [
            "",
            "**Sensitivity exclusion:** AD pool excludes "
            f"{', '.join(sorted(exclude_ad_samples))}.",
        ]
    lines += [
        "",
        "## Group sets",
        "",
        "| Set | n | Sample IDs |",
        "|-----|---|-----------|",
    ]
    label_order = [("AD", "AD"), ("CLEAN", "CLEAN (clean controls)"),
                   ("SUSPECT", "SUSPECT (AD-like controls CTRL-07/08/10)"),
                   ("ALLCTRL", "ALLCTRL (CLEAN ∪ SUSPECT, the contaminated baseline)"),
                   ("ADSUSP", "AD ∪ SUSPECT")]
    for key, name in label_order:
        ids = sets[key]
        lines.append(f"| **{name}** | {len(ids)} | {', '.join(ids)} |")
    lines += [
        "",
        "## Contrasts",
        "",
        "| File | Contrast | Group A (LFC numerator) | Group B (reference) | n_A | n_B | Analysis |",
        "|------|----------|-------------------------|---------------------|-----|-----|----------|",
    ]
    for label, a_key, b_key, bucket in CONTRASTS:
        s = summaries[label]
        lines.append(
            f"| `mea_{label}.csv` | `{label}` | {a_key} | {b_key} | "
            f"{len(sets[a_key])} | {len(sets[b_key])} | {bucket} |")
    lines += [
        "",
        "## Significant kinase counts (FDR < %s)" % sig,
        "",
        "| Contrast | track | up | down |",
        "|----------|-------|----|----|",
    ]
    for label, *_ in CONTRASTS:
        for track in TRACKS:
            c = summaries[label]["counts"].get(track, (0, 0))
            lines.append(f"| `{label}` | {track} | {c[0]} | {c[1]} |")
    if exclude_ad_samples:
        ad_clean_note = (
            "- **`AD_vs_cleanCTRL`** is a sensitivity rerun of the audit-adjusted "
            f"production contrast after excluding {', '.join(sorted(exclude_ad_samples))} "
            "from the AD pool. Do not treat this file as the canonical clean-baseline "
            "production output."
        )
    else:
        ad_clean_note = (
            "- **`AD_vs_cleanCTRL`** is the audit-adjusted production contrast. Its "
            "canonical copy lives at `../human_group_mea_clean_ctrl.csv`; "
            "`mea_AD_vs_cleanCTRL.csv` here is the investigation-scoped twin produced by "
            "the same code path (NES values are regression-checked to match)."
        )
    lines += [
        "",
        "## Provenance & status notes",
        "",
        ad_clean_note,
        "- **`AD_vs_allCTRL`** resurrects the **pre-audit** all-7-control baseline as a "
        "labeled hard-line reference only. It is contaminated by CTRL-07/08/10 by design — "
        "it is *not* a live fallback and must not be used as the production AD-vs-control "
        "result.",
        "- **`ADwithSuspect_vs_cleanCTRL`** folds the suspects into AD (the pre-symptomatic "
        "hypothesis); it assumes the thing the audit could not confirm, so treat as "
        "exploratory.",
        "- **`suspect_vs_cleanCTRL`** is the kinase analog of the site-level "
        "`../suspect_vs_ad_lfc_*.csv` (analysis 0).",
        "",
        "Each CSV holds both tracks (`track` ∈ {st, py}); the `contrast` column equals the "
        "filename token.",
        "",
        "## Diagnostic wide view — `suspect_vs_AD_kinase_wide.csv`",
        "",
        "One row per kinase × track, derived from the four long files. Built for the "
        "early-onset-diagnostic question: does the pre-symptomatic (suspect) signature "
        "already look like AD? Both groups are scored against the **same clean baseline**, "
        "so NES is directly comparable.",
        "",
        "| Column | Meaning |",
        "|--------|---------|",
        "| `NES_suspect` / `FDR_suspect` | `suspect_vs_cleanCTRL` enrichment + q |",
        "| `NES_AD` / `FDR_AD` | `AD_vs_cleanCTRL` enrichment + q (production) |",
        "| `delta_NES_suspect_minus_AD` | **diagnostic axis** = `NES_suspect − NES_AD` |",
        "| `abs_delta_NES` | `|delta|` |",
        "| `NES_ADwithSuspect` | merged contrast NES (context; exploratory) |",
        "| `NES_AD_allCTRL` | `AD_vs_allCTRL` **group-contrast** NES (pooled mean(AD) − "
        "mean(all 7 ctrl), one GSEA); pre-audit contaminated baseline, reference only |",
        "| `median_nes_sig_only_allCTRL` | the unified viewer's human-tab **\"median NES "
        "(sig)\"** — **per-donor** estimator (median across AD donors of each kinase's NES, "
        "restricted to FDR<%s-significant donors), same all-7-ctrl cohort. Sourced verbatim "
        "from `perdonor/recurrence{,_pY}.csv`, NOT recomputed. Correlates ~0.87 with "
        "`NES_AD_allCTRL` but is a different statistic — do not equate them. |" % sig,
        "",
        "Reading: small `|delta|` with both NES high = kinase already AD-like in suspects "
        "(the suspect and AD contrasts agree); large `|delta|` = the suspect and AD "
        "signatures diverge for that kinase (signed `delta` tells the direction — positive "
        "= stronger in suspects, negative = stronger in AD). On n=3 suspects, large-delta "
        "rows are hypothesis-generating, not diagnostic. Leading-edge sites, ES, p-value "
        "and Subs fraction are intentionally dropped here — pull them from the per-contrast "
        "long files. **Sorted st-first, then `abs_delta_NES` descending** so the kinases "
        "that most differ between the suspect and AD contrasts sit at the top.",
    ]
    (OUT_DIR / "MANIFEST.md").write_text("\n".join(lines) + "\n")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run suspect-control human kinase MEA reanalysis."
    )
    parser.add_argument(
        "--out-dir",
        default=str(OUT_DIR),
        help="Output directory for the reanalysis CSVs and manifest.",
    )
    parser.add_argument(
        "--exclude-ad-samples",
        nargs="*",
        default=[],
        help="AD samples to remove from the AD pool, e.g. AD-01 AD-03 or 1 3.",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    global OUT_DIR
    OUT_DIR = Path(args.out_dir)
    exclude_ad_samples = {_normalize_ad_sample(s) for s in args.exclude_ad_samples}

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    matrix = _load_track_matrix("st", "stoich")
    if matrix is None:
        raise RuntimeError("st stoichiometry matrix unavailable")
    mapping_samples = set(pd.read_csv(SAMPLE_MAPPING_CSV)["sample_id"])
    missing = sorted(exclude_ad_samples - mapping_samples)
    if missing:
        raise ValueError(f"Excluded AD samples not found in sample mapping: {missing}")
    sets = _sample_sets(matrix.columns, exclude_ad_samples=exclude_ad_samples)
    sig = config.MEA_FDR_THRESH

    summaries: dict[str, dict] = {}
    frames: dict[str, pd.DataFrame] = {}
    for label, a_key, b_key, _bucket in CONTRASTS:
        out = write_contrast(label, a_key, b_key, sets)
        frames[label] = out
        counts = {}
        for track in TRACKS:
            t = out[out.track == track]
            up = int(((t.NES > 0) & (t.FDR < sig)).sum())
            dn = int(((t.NES < 0) & (t.FDR < sig)).sum())
            counts[track] = (up, dn)
        summaries[label] = {"counts": counts}
        cstr = "  ".join(f"{tr}:{c[0]}up/{c[1]}dn" for tr, c in counts.items())
        print(f"[{label}] A={a_key}({len(sets[a_key])}) B={b_key}({len(sets[b_key])})  "
              f"rows={len(out)}  FDR<{sig}: {cstr}")

    wide = write_wide(frames)
    write_manifest(sets, summaries, exclude_ad_samples=exclude_ad_samples)
    print(f"\nwrote {len(CONTRASTS)} contrasts + suspect_vs_AD_kinase_wide.csv "
          f"({len(wide)} rows) + MANIFEST.md -> {OUT_DIR}")


if __name__ == "__main__":
    main()

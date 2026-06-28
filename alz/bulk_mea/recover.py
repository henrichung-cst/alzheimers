#!/usr/bin/env python3
"""Kinase hypothesis table assembly: kinase-first and cell-type-first views.

Reads MEA enrichment from alz/bulk_mea/enrich.py and unified attribution from
alz/bulk_mea/attribute.py and produces four output tables suited for hypothesis generation about AD
progression across cell types, timepoints, and conditions.

Design principle: static localization evidence (WMB expression, SEA-AD
concordance) is separated from the dynamic NES signal (varies per contrast).
WMB specificity acts as a gate (must be expressed) rather than a weight.

Inputs:
  outputs/reports/kinase_attribution/mea_stoichiometry.csv
  outputs/reports/kinase_attribution/unified_attribution_full.csv

Outputs (all under outputs/reports/attribution_recovery/):
  kinase_activity_matrix.csv    -- wide NES/FDR + trajectory label (1 row/kinase)
  celltype_evidence_table.csv   -- WMB-gated static evidence (1 row/kinase×celltype)
  kinase_hypothesis_table.csv   -- kinase-first synthesis (1 row/kinase)
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
from alz.bulk_mea.confidence import CONFIDENCE_RANK

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OUTPUT_DIR = config.ATTRIBUTION_RECOVERY_OUTPUT_DIR

CONTRASTS = list(config.CONTRAST_COEFS.keys())
NES_COLS = [f"{c}_NES" for c in CONTRASTS]
FDR_COLS = [f"{c}_FDR" for c in CONTRASTS]

_SUSTAINED_RATIO_THRESH = 1.5


def _load_gene_symbol_map(kinases) -> dict[str, str]:
    """Load the canonical kinase-abbreviation -> gene-symbol map."""
    kinases = sorted({str(k) for k in kinases})
    gene_map = {k: k for k in kinases}
    if not os.path.exists(config.MAPPING_CACHE_FILE):
        raise FileNotFoundError(
            f"canonical kinase mapping cache not found: {config.MAPPING_CACHE_FILE}"
        )

    cache = pd.read_csv(config.MAPPING_CACHE_FILE)
    required = {"kinase_abbreviation", "gene_symbol"}
    missing_cols = required - set(cache.columns)
    if missing_cols:
        raise ValueError(
            f"{config.MAPPING_CACHE_FILE} is missing required columns: "
            f"{sorted(missing_cols)}"
        )
    cache_map = dict(zip(cache["kinase_abbreviation"].astype(str),
                         cache["gene_symbol"].astype(str)))
    for k in kinases:
        gene_map[k] = cache_map.get(k, k)
    return gene_map


def _validate_gene_symbols(df, gene_map, table_name: str) -> None:
    """Fail fast if an upstream table was built with stale kinase-gene labels."""
    if "kinase" not in df.columns or "gene_symbol" not in df.columns:
        return
    pairs = df[["kinase", "gene_symbol"]].drop_duplicates().copy()
    pairs["kinase"] = pairs["kinase"].astype(str)
    pairs["gene_symbol"] = pairs["gene_symbol"].astype(str)
    pairs["expected_gene_symbol"] = pairs["kinase"].map(
        lambda k: gene_map.get(k, k)
    )
    mismatches = pairs[
        pairs["gene_symbol"] != pairs["expected_gene_symbol"]
    ].sort_values("kinase")
    if mismatches.empty:
        return

    preview = mismatches.head(20).to_string(index=False)
    raise ValueError(
        f"{table_name} has {len(mismatches)} kinase-gene mapping mismatch(es) "
        f"against {config.MAPPING_CACHE_FILE}. Re-run "
        f"alz/bulk_mea/attribute.py before recovery.\n{preview}"
    )


# ---------------------------------------------------------------------------
# Shared helper: Table 2 (Song-first static evidence)
# ---------------------------------------------------------------------------

def _build_celltype_evidence(uaf_full):
    """Build Table 2: one Song-first row per (kinase, cell_type).

    Deduplicate by taking the row with the highest canonical confidence tier,
    then explicit Song/decomposition/localization evidence. Keep rows that are
    attributed or where the kinase is detected (expressed) in that cell type per
    the Song within-cohort reference.
    """
    uaf_ranked = uaf_full.copy()
    uaf_ranked["_confidence_rank"] = (
        uaf_ranked["confidence_tier"].map(CONFIDENCE_RANK).fillna(0).astype(int)
    )
    uaf_ranked["_song_lfc_abs"] = uaf_ranked["song_lfc"].abs()
    uaf_ranked["_sea_ad_lfc_abs"] = uaf_ranked["sea_ad_lfc"].abs()
    idx = (uaf_ranked
           .sort_values(
               [
                   "_confidence_rank",
                   "song_concentration",
                   "decomp_agrees_bulk",
                   "wmb_concentration",
                   "human_location_score",
                   "_song_lfc_abs",
                   "_sea_ad_lfc_abs",
               ],
               ascending=[False, False, False, False, False, False, False],
           )
           .groupby(["kinase", "cell_type"], sort=False)
           .head(1).index)
    deduped = uaf_ranked.loc[idx].drop(
        columns=["_confidence_rank", "_song_lfc_abs", "_sea_ad_lfc_abs"],
    ).reset_index(drop=True)

    filtered = deduped[
        (deduped["confidence_tier"].astype(str) != "none")
        | (deduped["song_detected"].fillna(False).astype(bool))
    ].copy()

    filtered["concordance_direction"] = filtered["sea_ad_lfc"].apply(
        lambda x: "up" if x > config.SEA_AD_LFC_MIN
        else ("down" if x < -config.SEA_AD_LFC_MIN else "none"))

    # Song within-cohort concordance (carry forward from unified attribution)
    if "song_lfc" in filtered.columns:
        filtered["song_concordance_direction"] = filtered["song_lfc"].apply(
            lambda x: ("up" if x > config.SONG_LFC_MIN
                       else "down" if x < -config.SONG_LFC_MIN
                       else "none") if pd.notna(x) else "none")
    else:
        filtered["song_lfc"] = float("nan")
        filtered["song_concordance_direction"] = "none"

    cols = [
        "kinase", "gene_symbol", "cell_type",
        "confidence_tier", "confidence_basis",
        "song_direction_support", "human_location_tier", "decomp_agrees_bulk",
        "wmb_detected", "wmb_concentration", "wmb_concentration_tier",
        "wmb_fraction_cells_expressing",
        "song_detected", "song_concentration", "song_concentration_tier",
        "song_fraction_cells_expressing", "song_effective_n",
        "song_top_celltype", "song_top_concentration",
        "sea_ad_lfc", "sea_ad_n_supertypes",
        "seaad_location_score", "hbca_location_score", "human_location_score",
        "concordance_direction",
        "song_lfc", "song_concordance_direction",
        "decomp_nes", "decomp_fdr",
    ]
    for extra in ("contrast",):
        if extra in filtered.columns and extra not in cols:
            cols.append(extra)
    cols = [c for c in cols if c in filtered.columns]
    filtered["_confidence_rank"] = (
        filtered["confidence_tier"].map(CONFIDENCE_RANK).fillna(0).astype(int)
    )
    filtered["_song_lfc_abs"] = filtered["song_lfc"].abs()
    filtered["_sea_ad_lfc_abs"] = filtered["sea_ad_lfc"].abs()
    sorted_filtered = filtered.sort_values(
        ["kinase", "_confidence_rank", "song_concentration", "_song_lfc_abs", "_sea_ad_lfc_abs"],
        ascending=[True, False, False, False, False],
    )
    return sorted_filtered[cols]


# ---------------------------------------------------------------------------
# Trajectory classification
# ---------------------------------------------------------------------------

def _get_sig_conditions(row):
    """Comma-separated conditions (App/Tau/ApTt) with >=1 significant contrast."""
    sig_conds = set()
    for c in CONTRASTS:
        if row[f"{c}_FDR"] < config.MEA_FDR_THRESH:
            sig_conds.add(c.rsplit("_", 1)[0])
    return ",".join(sorted(sig_conds)) if sig_conds else ""


def _classify_trajectory(row, g_contrasts):
    """Classify temporal pattern for a genotype's contrasts.

    Takes ``g_contrasts`` — the list of contrasts belonging to one genotype
    (e.g. ["App_2mo","App_4mo","App_6mo"]).  The genotype prefix is inferred
    from the first element.

    Labels (checked in order):
      none            -- no significant contrasts within this genotype
      single_contrast -- exactly 1 significant contrast within this genotype
      progressive     -- |NES| monotonically increases 2mo->4mo->6mo
      declining       -- |NES| monotonically decreases
      peaked          -- 4mo has highest |NES|
      sustained       -- all 3 timepoints significant, max/min |NES| <= threshold
      early           -- only 2mo significant
      late            -- only 6mo significant
      mixed           -- everything else
    """
    g_fdr = [row[f"{c}_FDR"] for c in g_contrasts]
    n_sig = sum(1 for f in g_fdr if f < config.MEA_FDR_THRESH)
    if n_sig == 0:
        return "none"
    if n_sig == 1:
        return "single_contrast"

    # Genotype prefix is the shared prefix of all g_contrasts.
    peak_cond = g_contrasts[0].rsplit("_", 1)[0]
    tp_keys = [f"{peak_cond}_2mo", f"{peak_cond}_4mo", f"{peak_cond}_6mo"]
    nes = [abs(row[f"{k}_NES"]) for k in tp_keys]
    sig = [row[f"{k}_FDR"] < config.MEA_FDR_THRESH for k in tp_keys]
    n2, n4, n6 = nes
    s2, s4, s6 = sig

    if n2 < n4 < n6:
        return "progressive"
    if n2 > n4 > n6:
        return "declining"
    if n4 > n2 and n4 > n6:
        return "peaked"
    if s2 and s4 and s6:
        if min(nes) > 0 and max(nes) / min(nes) <= _SUSTAINED_RATIO_THRESH:
            return "sustained"
    if s2 and not s4 and not s6:
        return "early"
    if not s2 and not s4 and s6:
        return "late"
    return "mixed"


# ---------------------------------------------------------------------------
# Table 1: Kinase activity matrix
# ---------------------------------------------------------------------------

def _build_kinase_activity_matrix(mea, gene_map):
    """Build Table 1: wide NES/FDR matrix + per-genotype scalars, one row per kinase."""
    nes_wide = mea.pivot(index="kinase", columns="contrast", values="NES")
    nes_wide.columns = [f"{c}_NES" for c in nes_wide.columns]
    fdr_wide = mea.pivot(index="kinase", columns="contrast", values="FDR")
    fdr_wide.columns = [f"{c}_FDR" for c in fdr_wide.columns]

    t1 = nes_wide.join(fdr_wide).reset_index()

    # Reorder to canonical contrast order
    present_nes = [c for c in NES_COLS if c in t1.columns]
    present_fdr = [c for c in FDR_COLS if c in t1.columns]
    t1 = t1[["kinase"] + present_nes + present_fdr]

    t1["sig_conditions"] = t1.apply(_get_sig_conditions, axis=1)

    # Per-genotype scalars: peak_NES_{g}, peak_contrast_{g}, n_sig_{g}, trajectory_{g}
    # Split key: genotype = contrast.split("_")[0] in config.DISEASE_GROUPS
    per_genotype_cols = []
    for g in config.DISEASE_GROUPS:
        g_contrasts = [c for c in CONTRASTS if c.split("_")[0] == g]
        g_nes_cols = [f"{c}_NES" for c in g_contrasts if f"{c}_NES" in t1.columns]
        g_fdr_cols = [f"{c}_FDR" for c in g_contrasts if f"{c}_FDR" in t1.columns]

        if g_nes_cols:
            t1[f"peak_NES_{g}"] = t1[g_nes_cols].apply(
                lambda r: max(r, key=abs), axis=1)
            t1[f"peak_contrast_{g}"] = t1.apply(
                lambda r, gc=g_contrasts: max(
                    gc, key=lambda c: abs(r[f"{c}_NES"]) if f"{c}_NES" in r.index else 0
                ), axis=1)
        else:
            t1[f"peak_NES_{g}"] = float("nan")
            t1[f"peak_contrast_{g}"] = ""

        if g_fdr_cols:
            t1[f"n_sig_{g}"] = (t1[g_fdr_cols] < config.MEA_FDR_THRESH).sum(axis=1)
        else:
            t1[f"n_sig_{g}"] = 0

        g_contrasts_present = [c for c in g_contrasts if f"{c}_FDR" in t1.columns]
        t1[f"trajectory_{g}"] = t1.apply(
            lambda r, gc=g_contrasts_present: _classify_trajectory(r, gc) if gc else "none",
            axis=1)

        per_genotype_cols += [
            f"peak_NES_{g}", f"peak_contrast_{g}", f"n_sig_{g}", f"trajectory_{g}"
        ]

    t1["gene_symbol"] = t1["kinase"].map(gene_map)

    # Carry residue type through so downstream consumers can stratify
    # Tyr vs Ser/Thr biology. Each kinase only appears in one track's MEA
    # output (S/T and Y kinome name spaces are disjoint), so a per-kinase
    # mode lookup is unambiguous.
    if "residue_type" in mea.columns:
        kinase_to_residue = (mea.drop_duplicates("kinase")
                             .set_index("kinase")["residue_type"]
                             .to_dict())
        t1["residue_type"] = t1["kinase"].map(kinase_to_residue).fillna("ST")
    else:
        t1["residue_type"] = "ST"

    return t1[["kinase", "gene_symbol", "residue_type"] +
              present_nes + present_fdr +
              ["sig_conditions"] + per_genotype_cols]


# ---------------------------------------------------------------------------
# Table 3: Kinase hypothesis table
# ---------------------------------------------------------------------------

def _build_kinase_hypothesis_table(t1, t2):
    """Build Table 3: kinase-first synthesis joining activity profile + top cell types.

    Cell types ranked by confidence tier, then weighted concordance and WMB
    enrichment. None-confidence cells are dropped from the ranking when at least one
    high/moderate cell exists for that kinase, so the kinase-list "Top cell"
    reflects an actual attribution claim rather than a WMB plausibility floor.
    """
    t2_work = t2.copy()
    w_song = config.SONG_CONCORDANCE_WEIGHT
    w_sea_ad = config.SEA_AD_CONCORDANCE_WEIGHT
    abs_sea_ad = t2_work["sea_ad_lfc"].abs().fillna(0)
    abs_song = (t2_work["song_lfc"].abs().fillna(0)
                if "song_lfc" in t2_work.columns else 0)
    t2_work["_weighted_concordance"] = (
        (w_song * abs_song + w_sea_ad * abs_sea_ad) / (w_song + w_sea_ad))

    if "confidence_tier" in t2_work.columns:
        attributable = t2_work["confidence_tier"].isin(["very_high", "high", "moderate"])
        kinases_with_attribution = set(
            t2_work.loc[attributable, "kinase"].unique())
        keep_mask = attributable | (~t2_work["kinase"].isin(kinases_with_attribution))
        t2_work = t2_work[keep_mask].copy()

    t2_work["_confidence_rank"] = (
        t2_work["confidence_tier"].map(CONFIDENCE_RANK).fillna(0).astype(int)
    )
    sort_keys = ["kinase", "_confidence_rank",
                 "_weighted_concordance", "wmb_concentration"]
    ascending = [True, False, False, False]
    t2_sorted = t2_work.sort_values(sort_keys, ascending=ascending)

    n_cands = t2.groupby("kinase").size().rename("n_celltype_candidates")

    has_song = "song_lfc" in t2_work.columns

    def _top3(group):
        top = group.head(3).reset_index(drop=True)
        result = {}
        for i, rank in enumerate([1, 2, 3]):
            if i < len(top):
                result[f"top_celltype_{rank}"] = top.loc[i, "cell_type"]
                result[f"top_celltype_{rank}_wmb_tier"] = top.loc[i, "wmb_concentration_tier"]
                result[f"top_celltype_{rank}_sea_ad_lfc"] = top.loc[i, "sea_ad_lfc"]
                result[f"top_celltype_{rank}_evidence"] = top.loc[i, "confidence_basis"]
                if has_song:
                    result[f"top_celltype_{rank}_song_lfc"] = top.loc[i, "song_lfc"]
            else:
                result[f"top_celltype_{rank}"] = None
                result[f"top_celltype_{rank}_wmb_tier"] = None
                result[f"top_celltype_{rank}_sea_ad_lfc"] = None
                result[f"top_celltype_{rank}_evidence"] = None
                if has_song:
                    result[f"top_celltype_{rank}_song_lfc"] = None
        return pd.Series(result)

    top3_df = (t2_sorted.groupby("kinase", group_keys=False)
               .apply(_top3, include_groups=False)
               .reset_index())

    # Single source of truth: the pill mirrors the per-row confidence_tier from
    # unified_attribution_full. Pill = True iff at least one (kinase, cell_type)
    # row reaches high or very_high.
    if "confidence_tier" in t2.columns:
        high_conf = (
            t2[t2["confidence_tier"].isin(["very_high", "high"])]
            .groupby("kinase").size().gt(0).rename("has_high_conf_attribution")
        )
    else:
        high_conf = pd.Series(dtype=bool, name="has_high_conf_attribution")

    # Pull per-genotype scalars from t1 for the hypothesis table header.
    t1_cols = ["kinase", "gene_symbol", "residue_type", "sig_conditions"]
    for g in config.DISEASE_GROUPS:
        for suffix in [f"peak_NES_{g}", f"peak_contrast_{g}", f"n_sig_{g}", f"trajectory_{g}"]:
            if suffix in t1.columns:
                t1_cols.append(suffix)
    t1_cols = [c for c in t1_cols if c in t1.columns]
    t3 = (t1[t1_cols]
          .merge(n_cands.reset_index(), on="kinase", how="left")
          .merge(top3_df, on="kinase", how="left")
          .merge(high_conf.reset_index(), on="kinase", how="left"))

    t3["n_celltype_candidates"] = t3["n_celltype_candidates"].fillna(0).astype(int)
    t3["has_high_conf_attribution"] = (
        t3["has_high_conf_attribution"].where(
            t3["has_high_conf_attribution"].notna(), other=False
        ).astype(bool))

    # Sort by max n_sig across genotypes, then max |peak_NES| across genotypes.
    peak_nes_cols = [f"peak_NES_{g}" for g in config.DISEASE_GROUPS if f"peak_NES_{g}" in t3.columns]
    n_sig_cols = [f"n_sig_{g}" for g in config.DISEASE_GROUPS if f"n_sig_{g}" in t3.columns]
    if n_sig_cols:
        t3["_max_n_sig"] = t3[n_sig_cols].max(axis=1)
    else:
        t3["_max_n_sig"] = 0
    if peak_nes_cols:
        t3["_abs_peak"] = t3[peak_nes_cols].abs().max(axis=1)
    else:
        t3["_abs_peak"] = 0.0
    t3 = t3.sort_values(["_max_n_sig", "_abs_peak"], ascending=[False, False])
    t3 = t3.drop(columns=["_max_n_sig", "_abs_peak"])

    return t3


# ===========================================================================
# Pure compute orchestrator
# ===========================================================================

def step_attribution_recovery(mea, uaf_full):
    """Compute all three recovery tables from in-memory inputs.

    Returns ``(kinase_activity_matrix, celltype_evidence_table,
    kinase_hypothesis_table)``. Side-effect free; callers (CLI shim or
    Kedro nodes) decide where to persist the outputs.
    """
    gene_map = _load_gene_symbol_map(mea["kinase"].unique())
    _validate_gene_symbols(uaf_full, gene_map, "unified_attribution_full.csv")
    t1 = _build_kinase_activity_matrix(mea, gene_map)
    t2 = _build_celltype_evidence(uaf_full)
    t3 = _build_kinase_hypothesis_table(t1, t2)
    return t1, t2, t3


# ===========================================================================
# Summary
# ===========================================================================

def print_summary():
    """Print cached results summary."""
    print("\n" + "=" * 70)
    print("Attribution Recovery — Summary")
    print("=" * 70)

    t1_path = os.path.join(OUTPUT_DIR, "kinase_activity_matrix.csv")
    if os.path.exists(t1_path):
        t1 = pd.read_csv(t1_path)
        print(f"\nS3 (Table 1): Kinase Activity Matrix")
        print(f"  {len(t1)} kinases")
        for g in config.DISEASE_GROUPS:
            col = f"trajectory_{g}"
            if col in t1.columns:
                print(f"  Trajectory distribution ({g}):")
                for label, cnt in t1[col].value_counts().items():
                    print(f"    {label}: {cnt}")
    else:
        print("\nS3 (Table 1): Not yet computed")

    t3_path = os.path.join(OUTPUT_DIR, "kinase_hypothesis_table.csv")
    if os.path.exists(t3_path):
        t3 = pd.read_csv(t3_path)
        print(f"\nS3 (Table 3): Kinase Hypothesis Table")
        print(f"  {len(t3)} kinases")
        print(f"  With >=1 cell-type candidate: "
              f"{(t3['n_celltype_candidates'] > 0).sum()}")
        print(f"  High-confidence attribution: "
              f"{t3['has_high_conf_attribution'].sum()}")
    else:
        print("\nS3 (Table 3): Not yet computed")

    t2_path = os.path.join(OUTPUT_DIR, "celltype_evidence_table.csv")
    if os.path.exists(t2_path):
        t2 = pd.read_csv(t2_path)
        print(f"\nS4 (Table 2): Cell-Type Evidence Table")
        print(f"  {len(t2)} (kinase, cell_type) pairs")
        print(f"  {t2['kinase'].nunique()} unique kinases")
        print(f"  Song detection breakdown:")
        for det, cnt in t2["song_detected"].value_counts().items():
            print(f"    detected={det}: {cnt}")
    else:
        print("\nS4 (Table 2): Not yet computed")

    print()


def main():
    parser = argparse.ArgumentParser(
        description="Attribution Recovery: Kinase and cell-type hypothesis tables",
    )
    parser.add_argument("--summary", action="store_true",
                        help="Print cached results summary (no pipeline run)")
    args = parser.parse_args()

    if args.summary:
        print_summary()
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    kinase_attr_dir = config.KINASE_ATTRIBUTION_OUTPUT_DIR

    # Load per-track MEA stoichiometry, inject residue_type/track if missing
    frames = []
    for track_name, track_cfg in config.PHOSPHO_TRACKS.items():
        suffix = track_cfg["output_suffix"]
        fname = f"mea_stoichiometry{suffix}.csv"
        path = os.path.join(kinase_attr_dir, fname)
        if not os.path.exists(path):
            print(f"  [{track_name}] {path} not found; skipping track.")
            continue
        df = pd.read_csv(path)
        if "residue_type" not in df.columns:
            df["residue_type"] = track_cfg["residue"]
        if "track" not in df.columns:
            df["track"] = track_cfg["name"]
        frames.append(df)
    if not frames:
        raise FileNotFoundError(
            "No MEA stoichiometry files found; run alz/bulk_mea/enrich.py first.")
    mea_combined = pd.concat(frames, ignore_index=True, sort=False)

    uaf_path = os.path.join(kinase_attr_dir, "unified_attribution_full.csv")
    if not os.path.exists(uaf_path):
        raise FileNotFoundError(
            f"unified_attribution_full.csv not found at {uaf_path}; "
            "run alz/bulk_mea/attribute.py first.")
    uaf_full = pd.read_csv(uaf_path)

    t1, t2, t3 = step_attribution_recovery(mea_combined, uaf_full)

    t1.to_csv(os.path.join(OUTPUT_DIR, "kinase_activity_matrix.csv"), index=False)
    t2.to_csv(os.path.join(OUTPUT_DIR, "celltype_evidence_table.csv"), index=False)
    t3.to_csv(os.path.join(OUTPUT_DIR, "kinase_hypothesis_table.csv"), index=False)
    print(f"  Saved kinase_activity_matrix.csv ({len(t1)} kinases)")
    print(f"  Saved celltype_evidence_table.csv ({len(t2)} rows)")
    print(f"  Saved kinase_hypothesis_table.csv ({len(t3)} kinases)")

    summary = {
        "n_kinases_in_activity_matrix": int(len(t1)),
        "n_celltype_evidence_rows": int(len(t2)),
        "n_kinases_in_hypothesis_table": int(len(t3)),
        "n_kinases_with_celltype_candidate": int(
            (t3["n_celltype_candidates"] > 0).sum()
        ) if "n_celltype_candidates" in t3.columns else 0,
        "n_high_conf_attribution": int(t3["has_high_conf_attribution"].sum())
            if "has_high_conf_attribution" in t3.columns else 0,
        **config.provenance_stamp(),
    }
    summary_path = os.path.join(OUTPUT_DIR, "recovery_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved recovery_summary.json")
    print("\nAttribution recovery complete.")


if __name__ == "__main__":
    main()

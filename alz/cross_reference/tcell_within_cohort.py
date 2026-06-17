#!/usr/bin/env python3
"""Within-cohort cell-type attribution for the T-cell exhaustion cohort (donor1).

Localizes **bulk** kinase activity signals (donor1 ST + pY stoichiometry MEA) to
ProjecTILs cell-type **states** using only the cohort's own paired scRNA — the
"Song" within-cohort method (`alz/reference/snrna_integration.py` +
`alz/bulk_mea/attribute.py`), no disease reference required. A kinase attributes
to a state when its transcript is (a) preferentially expressed in that state
(specificity) and (b) moving the same direction as the bulk activity
(concordance, sign-checked against the MEA NES).

Two deliberate departures from the mouse design (see
`docs/plans/tcell_within_cohort_attribution.md`):

1. **No per-cell significance test / no FDR.** Donor1 is a single donor with one
   scRNA library per day — no biological replicates, so a per-cell `FindMarkers`
   would be pseudoreplication (Squair 2021) and any per-(state,day) p-value is
   fabricated. The mouse tier never consumed the Song p-value anyway. Concordance
   is a pseudobulk log2FC (direction + magnitude); credibility comes from
   timecourse consistency across d13/d17/d20.
2. **Binned specificity (1x/2x/5x/10x of uniform = 1/N_states) replaces the
   high/moderate/low confidence tiers** — copying the unified viewer's WMB-tier
   design. Binning is done HERE in Python (not in JS like the mouse) because the
   uniform baseline 1/N_states is donor-dependent.

CORRECTNESS TRAP: `aggexp_data.csv` is `AggregateExpression(slot="data")` — a SUM
of log-normalized expression across cells, NOT a mean. Every value is divided by
the matching `cell_counts.csv` n_cells to recover per-cell mean log-expression
`m[state,day]`; otherwise abundant states look artificially "specific".

Outputs (donor1) under outputs/reports/kinase_attribution_tcells/donor1/:
  tcell_specificity.csv          (gene, state, tcell_specificity, tcell_mean_log2_expression)
  tcell_concordance.csv          (gene, state, day, tcell_lfc)
  unified_attribution_tcells.csv full kinase-track × state × day grid — EVERY row ships
                                 (no gate). `tcell_concordant` is a shown label,
                                 never a filter; concordance is directionally
                                 uninformative for kinases (OR≈1, see docs/plans/
                                 tcell_attribution_degate_2026-06-03.md).

Usage:  python alz/cross_reference/tcell_within_cohort.py [donor1]
"""

from __future__ import annotations

import os
import re
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from alz.shared import config  # noqa: E402

# Binned-specificity tiers: multiples of the uniform baseline 1/N_states.
# Mirrors kinase_explorer.js _WMB_TIER_VALUES = [10, 5, 2, 1].
TIER_MULTIPLES = (10, 5, 2, 1)

# Bulk MEA contrast days that also have scRNA (MEA days {13,15,17,19,20} ∩
# scRNA days {0,2,9,13,17,20}, minus the d2 baseline). d15/d19 have no scRNA →
# no attribution rows (honest empty).
CONTRAST_DAYS = ("d13", "d17", "d20")
BASELINE_DAY = "d2"


def _donor_dir(donor: str) -> str:
    return os.path.join(config.REPO_ROOT, "data", "derived",
                        "tcells_incytr_inputs", donor, "scrna")


def _mea_dir(donor: str) -> str:
    return os.path.join(config.REPO_ROOT, "outputs", "reports",
                        "kinase_attribution_tcells", donor, "mea")


def _out_dir(donor: str) -> str:
    return os.path.join(config.REPO_ROOT, "outputs", "reports",
                        "kinase_attribution_tcells", donor)


def _load_kinase_to_gene() -> dict[str, str]:
    df = pd.read_csv(config.MAPPING_CACHE_FILE)
    return dict(zip(df["kinase_abbreviation"], df["gene_symbol"]))


def _per_cell_mean(donor: str) -> tuple[pd.DataFrame, list[str]]:
    """Recover per-cell mean log-expression m[gene, state, day] from the
    aggexp SUM ÷ cell_counts. Returns long DataFrame (gene, state, day, m) and
    the sorted list of all states."""
    agg = pd.read_csv(os.path.join(_donor_dir(donor), "aggexp_data.csv"))
    counts = pd.read_csv(os.path.join(_donor_dir(donor), "cell_counts.csv"))
    counts["day"] = "d" + counts["day"].astype(int).astype(str)
    n_cells = {(r.state, r.day): int(r.n_cells) for r in counts.itertuples()}

    gene_col = agg.columns[0]
    value_cols = [c for c in agg.columns if c != gene_col]
    # Parse "<state>__<day>" columns.
    parsed = []
    for c in value_cols:
        state, day = c.rsplit("__", 1)
        parsed.append((c, state, day))
    states = sorted({p[1] for p in parsed})

    long = agg.melt(id_vars=[gene_col], value_vars=value_cols,
                    var_name="_col", value_name="_sum")
    col_meta = pd.DataFrame(parsed, columns=["_col", "state", "day"])
    long = long.merge(col_meta, on="_col").drop(columns="_col")
    long = long.rename(columns={gene_col: "gene"})
    long["n_cells"] = long.apply(
        lambda r: n_cells.get((r["state"], r["day"]), np.nan), axis=1)
    # Columns present in aggexp must have a matching count row.
    missing = long[long["n_cells"].isna()][["state", "day"]].drop_duplicates()
    if len(missing):
        raise AssertionError(
            f"aggexp (state,day) columns with no cell_counts entry: "
            f"{missing.to_dict('records')}")
    long["m"] = long["_sum"] / long["n_cells"]
    return long[["gene", "state", "day", "m"]], states


def _compute_specificity(mean_long: pd.DataFrame, states: list[str],
                         n_states: int) -> pd.DataFrame:
    """Song S2 specificity, pooled across all available scRNA days per state.

    For each gene: mean_in_state = mean over that state's available days of
    m[state,day]; tcell_specificity = mean_in_state / Σ_states mean_in_state.
    Tier = first TIER_MULTIPLE t where specificity >= t/N_states, else 0.
    """
    # Pool across days (the static-property analog of Song pooling all animals).
    mean_in_state = (mean_long.groupby(["gene", "state"], as_index=False)["m"]
                     .mean().rename(columns={"m": "tcell_mean_log2_expression"}))
    total = (mean_in_state.groupby("gene")["tcell_mean_log2_expression"]
             .transform("sum"))
    spec = mean_in_state["tcell_mean_log2_expression"] / total.replace(0, np.nan)
    mean_in_state["tcell_specificity"] = spec
    mean_in_state = mean_in_state[
        np.isfinite(mean_in_state["tcell_specificity"])
        & (mean_in_state["tcell_mean_log2_expression"] > 0)].copy()

    uniform = 1.0 / n_states
    s = mean_in_state["tcell_specificity"].to_numpy()
    tier = np.zeros(len(s), dtype=int)
    for t in TIER_MULTIPLES:  # descending; first satisfied wins
        tier = np.where((tier == 0) & (s >= t * uniform), t, tier)
    mean_in_state["tcell_tier"] = tier
    return mean_in_state[["gene", "state", "tcell_specificity",
                          "tcell_mean_log2_expression", "tcell_tier"]]


def _compute_concordance(mean_long: pd.DataFrame) -> pd.DataFrame:
    """Pseudobulk log2FC vs the d2 baseline, per (gene, state, day) for the
    contrast days. tcell_lfc = m[state,day] − m[state,d2] (log-space diff)."""
    wide = mean_long.pivot_table(index=["gene", "state"], columns="day",
                                 values="m")
    rows = []
    for day in CONTRAST_DAYS:
        if day not in wide.columns or BASELINE_DAY not in wide.columns:
            continue
        lfc = wide[day] - wide[BASELINE_DAY]
        sub = lfc.dropna().reset_index()
        sub.columns = ["gene", "state", "tcell_lfc"]
        sub["day"] = day
        rows.append(sub)
    if not rows:
        return pd.DataFrame(columns=["gene", "state", "day", "tcell_lfc"])
    return pd.concat(rows, ignore_index=True)[["gene", "state", "day", "tcell_lfc"]]


def _load_mea(donor: str) -> pd.DataFrame:
    """Long bulk MEA table: kinase, residue_type, day, NES, FDR.

    Mirrors the unified viewer convention: primary stoichiometry MEA rows from
    all residue tracks appear together, with `residue_type` carrying ST/Y.
    Raw phospho remains a sensitivity/audit track, not a separate attribution
    row family.
    """

    def _melt(df, name):
        long = df.melt(id_vars=["kinase"], var_name="_col", value_name=name)
        long["day"] = long["_col"].str.replace(r"^D\d+_", "", regex=True)
        return long.drop(columns="_col")

    frames = []
    for suffix, residue_type in (("", "ST"), ("_pY", "Y")):
        nes_path = os.path.join(_mea_dir(donor), f"kinase_timepoint_nes{suffix}.csv")
        fdr_path = os.path.join(_mea_dir(donor), f"kinase_timepoint_fdr{suffix}.csv")
        if not (os.path.exists(nes_path) and os.path.exists(fdr_path)):
            continue
        nes_l = _melt(pd.read_csv(nes_path), "NES")
        fdr_l = _melt(pd.read_csv(fdr_path), "FDR")
        sub = nes_l.merge(fdr_l, on=["kinase", "day"])
        sub["residue_type"] = residue_type
        frames.append(sub)
    if not frames:
        return pd.DataFrame(
            columns=["kinase", "residue_type", "day", "NES", "FDR"])
    mea = pd.concat(frames, ignore_index=True)
    return mea[mea["day"].isin(CONTRAST_DAYS)].copy()


def build(donor: str = "donor1") -> dict:
    print("=" * 72)
    print(f"T-cell within-cohort attribution — {donor}")
    print("=" * 72)

    mean_long, states = _per_cell_mean(donor)
    n_states = len(states)
    print(f"  states: {n_states}  ({', '.join(states)})")
    print(f"  uniform baseline 1/N = {1.0 / n_states:.4f}; "
          f"tiers {[f'{t}x={t / n_states:.4f}' for t in TIER_MULTIPLES]}")

    spec = _compute_specificity(mean_long, states, n_states)
    conc = _compute_concordance(mean_long)
    mea = _load_mea(donor)
    k2g = _load_kinase_to_gene()

    out_dir = _out_dir(donor)
    os.makedirs(out_dir, exist_ok=True)
    spec.drop(columns="tcell_tier").rename(columns={"state": "state"}).to_csv(
        os.path.join(out_dir, "tcell_specificity.csv"), index=False)
    conc.to_csv(os.path.join(out_dir, "tcell_concordance.csv"), index=False)
    print(f"  tcell_specificity.csv: {len(spec)} (gene × state) rows")
    print(f"  tcell_concordance.csv: {len(conc)} (gene × state × day) rows")

    # --- cross-join: kinase × state × contrast-day -------------------------
    kinases = (mea[["kinase", "residue_type"]].drop_duplicates()
               .sort_values(["residue_type", "kinase"]).reset_index(drop=True))
    state_day = pd.MultiIndex.from_product(
        [states, list(CONTRAST_DAYS)],
        names=["cell_type", "contrast"]).to_frame(index=False)
    kinases["_join_key"] = 1
    state_day["_join_key"] = 1
    grid = (kinases.merge(state_day, on="_join_key")
            .drop(columns="_join_key"))
    grid["gene_symbol"] = grid["kinase"].map(lambda k: k2g.get(k, k))

    # bulk anchor
    grid = grid.merge(mea.rename(columns={"day": "contrast"}),
                      on=["kinase", "residue_type", "contrast"], how="left")
    # specificity (per gene × state, repeated across days)
    grid = grid.merge(
        spec.rename(columns={"gene": "gene_symbol", "state": "cell_type"}),
        on=["gene_symbol", "cell_type"], how="left")
    grid["tcell_tier"] = grid["tcell_tier"].fillna(0).astype(int)
    # concordance lfc (per gene × state × day)
    grid = grid.merge(
        conc.rename(columns={"gene": "gene_symbol", "state": "cell_type",
                             "day": "contrast"}),
        on=["gene_symbol", "cell_type", "contrast"], how="left")

    grid["tcell_concordance"] = np.sign(grid["NES"]) * grid["tcell_lfc"]

    # timecourse consistency: per (kinase, state), # of contrast days where the
    # transcript moves the same direction as that day's bulk NES.
    consist = (grid.assign(_agree=(grid["tcell_concordance"] > 0).astype(int))
               .groupby(["kinase", "residue_type", "cell_type"],
                        as_index=False)["_agree"].sum()
               .rename(columns={"_agree": "tcell_consistency"}))
    grid = grid.merge(consist, on=["kinase", "residue_type", "cell_type"],
                      how="left")

    # `tcell_concordant` is a SHOWN label (sign of concordance), never a gate.
    # Every kinase × state × day row ships; the viewer displays all axes and the
    # human reads them. Concordance is directionally uninformative for kinases
    # (activity is post-translational, decoupled from the kinase's own mRNA;
    # sign-agreement runs at chance, OR≈1 — same in the mouse Song reference), so
    # it must not filter anything. See docs/plans/tcell_attribution_degate_2026-06-03.md.
    grid["tcell_concordant"] = grid["tcell_concordance"] > 0

    cols = ["kinase", "residue_type", "gene_symbol", "contrast", "cell_type",
            "NES", "FDR", "tcell_specificity", "tcell_tier", "tcell_lfc",
            "tcell_concordance", "tcell_concordant", "tcell_consistency"]
    full = grid[cols].copy()

    expected = len(kinases) * n_states * len(CONTRAST_DAYS)
    if len(full) != expected:
        raise AssertionError(
            f"full row count {len(full)} != expected {expected} "
            f"(n_kinase_tracks {len(kinases)} × n_states {n_states} × "
            f"n_contrast_days {len(CONTRAST_DAYS)}) — silent drop in merge")

    # Ship the ENTIRE grid (no concordance/specificity gate). Sorted for
    # readability only (tier desc, then concordance desc).
    shipped = full.sort_values(
        ["tcell_tier", "tcell_concordance"], ascending=False)
    shipped.to_csv(os.path.join(out_dir, "unified_attribution_tcells.csv"),
                   index=False)

    tier_dist = (shipped["tcell_tier"].value_counts().sort_index(ascending=False)
                 .to_dict())
    n_conc = int(shipped["tcell_concordant"].sum())
    print(f"  unified_attribution_tcells.csv: {len(shipped)} rows "
          f"(guard {expected} ✓; all rows shipped, no gate)")
    print(f"  concordance is a LABEL: {n_conc} concordant / "
          f"{len(shipped) - n_conc} discordant")
    print(f"  tier distribution (tier: rows): {tier_dist}")
    n_no_gene = full["tcell_specificity"].isna().groupby(
        [full["kinase"], full["residue_type"]]).all().sum()
    print(f"  kinase tracks with no transcript in scRNA: {n_no_gene}")

    return {"n_states": n_states, "n_kinases": len(kinases),
            "n_full": len(full), "n_shipped": len(shipped),
            "n_concordant": n_conc, "tier_dist": tier_dist}


if __name__ == "__main__":
    donor = sys.argv[1] if len(sys.argv) > 1 else "donor1"
    build(donor)

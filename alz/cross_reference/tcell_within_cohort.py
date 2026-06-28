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
`docs/tcell_exhaustion_analysis_summary.md`):

1. **No per-cell significance test / no FDR.** Donor1 is a single donor with one
   scRNA library per day — no biological replicates, so a per-cell `FindMarkers`
   would be pseudoreplication (Squair 2021) and any per-(state,day) p-value is
   fabricated. The mouse tier never consumed the Song p-value anyway. Concordance
   is a pseudobulk log2FC (direction + magnitude); credibility comes from
   timecourse consistency across d13/d17/d20.
2. **Detection evidence** — `tcell_fraction_expressing >= DETECTION_FRAC_MIN` (0.10;
   a state where the kinase is expressed in <10% of cells cannot carry a
   trustworthy enrichment) is reported as a normalization-free presence column.
   The breadth `tcell_effective_n` is computed over all states.

Two orthogonal specificity axes, reported as two columns:

3. **Cell-type specificity** (`tcell_top_celltype` / `tcell_celltype_concentration_tier`
   / `tcell_celltype_effective_n`). The 14 ProjecTILs states are a grid of cell
   TYPE (CD8 / CD4 / Treg) × state. Collapsing the states onto cell types and
   running the SHARED `specificity.compute` coarse-group aggregation (cell-weighted,
   `group_col`) over DETECTED states only gives the same concentration-tier +
   effective_n math every cohort uses (Song/5xFAD) — answering "is this kinase
   concentrated in one T-cell type?". The detected-basis restriction stops an
   undetected state's trace noise from winning a phantom home; a kinase detected
   in zero states gets no home/tier/effective_n. `tcell_top_celltype` is the
   dominant detected cell TYPE (cell-weighted argmax), NOT a state. The per-STATE
   concentration-tier is retired (the ~14 states are transcriptionally homogeneous,
   so a per-state dominance tier saturates at ≥2×).
4. **State enrichment** (`tcell_state_enrichment`) — the activation-continuum axis,
   T-cell-only (brain cohorts have no "state"). For each state, the fold of that
   state's linear expression over the gene's BASELINE = the MEAN linear expression
   across all adequately-sampled states (n_cells >= MIN_STATE_CELLS), gene-agnostic
   in its state set (the brain cohorts' celltype_mean / global_mean, ported to
   states). The baseline includes states where the gene is undetected, so a kinase
   concentrated in one state divides by a low baseline -> high fold = state-specific.
   GUARDED at DISPLAY time: only states that are detected (frac >= DETECTION_FRAC_MIN)
   AND have >= MIN_STATE_CELLS cells carry a badge; ineligible states get NaN. A
   median-over-detected-states baseline (the prior bug) inverted the metric —
   single-state kinases pinned to 1.0 and breadth drove the fold.

CORRECTNESS TRAP: `aggexp_data.csv` is `AggregateExpression(slot="data")` — a SUM
of log-normalized expression across cells, NOT a mean. Every value is divided by
the matching `cell_counts.csv` n_cells to recover per-cell mean log-expression
`m[state,day]`; otherwise abundant states look artificially abundant. (The
detection input `pct_expressing.csv` is already a fraction, not a sum — it is
pooled across days cell-weighted, no per-state division needed.)

Outputs (donor1) under outputs/reports/kinase_attribution_tcells/donor1/:
  tcell_enrichment.csv           (gene, state, tcell_detected, tcell_fraction_expressing,
                                  tcell_state_enrichment, tcell_mean_log2_expression)
                                  — renamed from tcell_specificity.csv; "enrichment" is
                                  the correct term for this within-cohort activation-state
                                  concentration metric (see common.py vocabulary note).
  tcell_concordance.csv          (gene, state, day, tcell_lfc)
  unified_attribution_tcells.csv full kinase-track × state × day grid — EVERY row ships
                                 (no gate). `tcell_concordant` is a shown label,
                                 never a filter; concordance is directionally
                                 uninformative for kinases (OR≈1; see
                                 docs/tcell_exhaustion_analysis_summary.md).

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
from alz.cross_reference import specificity  # noqa: E402

# Bulk MEA contrast days that also have scRNA (MEA days {13,15,17,19,20} ∩
# scRNA days {0,2,9,13,17,20}, minus the d2 baseline). d15/d19 have no scRNA →
# no attribution rows (honest empty).
CONTRAST_DAYS = ("d13", "d17", "d20")
BASELINE_DAY = "d2"

# A state must have at least this many cells (pooled across scRNA days) to enter
# the state-enrichment baseline mean or to be picked as a kinase's headline state.
# Drops CD8MAIT (n=8 in donor1); every other state clears it.
MIN_STATE_CELLS = 50

# Detection floor: a (gene, state) is "detected" when this fraction of the state's
# cells express the transcript. The SINGLE cross-cohort gate
# (specificity.DETECTION_FRAC_MIN, 10%) — not a local copy — so "detected" and
# "specific to N cell states" mean the same thing across the within-cohort scRNA
# and the NSCLC reference. A state below it cannot carry a trustworthy enrichment
# (the fold would be built on a handful of cells), so it is not eligible.
DETECTION_FRAC_MIN = specificity.DETECTION_FRAC_MIN


def _state_to_celltype(state: str) -> str:
    """Collapse a sanitized ProjecTILs state to its cell TYPE (CD8 / CD4 / Treg).

    The 14 states are a cell-type × activation-state grid; cell-type specificity
    is asked over these three types. ``Treg`` stands alone; everything else folds
    by CD8/CD4 prefix (CD8MAIT → CD8, so the n=8 state can never stand alone)."""
    s = str(state)
    if s == "Treg":
        return "Treg"
    if s.startswith("CD8"):
        return "CD8"
    if s.startswith("CD4"):
        return "CD4"
    return s


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


def _load_cell_counts(donor: str) -> pd.DataFrame:
    """Per-(state, day) cell counts with the day column normalized to ``d<N>``."""
    counts = pd.read_csv(os.path.join(_donor_dir(donor), "cell_counts.csv"))
    counts["day"] = "d" + counts["day"].astype(int).astype(str)
    return counts


def _per_cell_mean(donor: str) -> tuple[pd.DataFrame, list[str]]:
    """Recover per-cell mean log-expression m[gene, state, day] from the
    aggexp SUM ÷ cell_counts. Returns long DataFrame (gene, state, day, m) and
    the sorted list of all states."""
    agg = pd.read_csv(os.path.join(_donor_dir(donor), "aggexp_data.csv"))
    counts = _load_cell_counts(donor)
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


def _per_state_detection(donor: str) -> pd.DataFrame:
    """Pool pct_expressing across days into per-(gene, state) detection.

    `pct_expressing.csv` is the fraction of cells expressing per (state, day);
    pooling over a state's days is CELL-WEIGHTED (Σ pct·n_cells / Σ n_cells =
    fraction of all that state's cells, across days, that express), not a plain
    mean of fractions. Returns (gene, state, tcell_fraction_expressing, n_cells).
    """
    pct = pd.read_csv(os.path.join(_donor_dir(donor), "pct_expressing.csv"))
    counts = _load_cell_counts(donor)

    gene_col = pct.columns[0]
    value_cols = [c for c in pct.columns if c != gene_col]
    parsed = [(c, *c.rsplit("__", 1)) for c in value_cols]
    col_meta = pd.DataFrame(parsed, columns=["_col", "state", "day"])

    long = pct.melt(id_vars=[gene_col], value_vars=value_cols,
                    var_name="_col", value_name="frac")
    long = (long.merge(col_meta, on="_col").drop(columns="_col")
            .rename(columns={gene_col: "gene"})
            .merge(counts[["state", "day", "n_cells"]], on=["state", "day"],
                   how="left"))
    missing = long[long["n_cells"].isna()][["state", "day"]].drop_duplicates()
    if len(missing):
        raise AssertionError(
            f"pct_expressing (state,day) columns with no cell_counts entry: "
            f"{missing.to_dict('records')}")
    long["_n_expr"] = long["frac"] * long["n_cells"]
    pooled = long.groupby(["gene", "state"], as_index=False).agg(
        _n_expr=("_n_expr", "sum"), n_cells=("n_cells", "sum"))
    pooled["tcell_fraction_expressing"] = (
        pooled["_n_expr"] / pooled["n_cells"].clip(lower=1))
    return pooled[["gene", "state", "tcell_fraction_expressing", "n_cells"]]


def _compute_metric(donor: str, mean_long: pd.DataFrame
                    ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Standard detection metric per (gene, state), pooled across scRNA days.

    Expression is pooled as the mean over a state's available days of the
    per-cell mean m[state,day]; detection is pooled cell-weighted from
    pct_expressing. Both feed `specificity.compute` (the one cross-cohort
    definition). Returns (per_label, per_gene) renamed to the tcell_* schema.
    """
    mean_in_state = (mean_long.groupby(["gene", "state"], as_index=False)["m"]
                     .mean().rename(columns={"m": "tcell_mean_log2_expression"}))
    det = _per_state_detection(donor)
    df = mean_in_state.merge(det, on=["gene", "state"], how="left")
    df["tcell_fraction_expressing"] = df["tcell_fraction_expressing"].fillna(0.0)
    df["n_cells"] = df["n_cells"].fillna(0).astype(int)
    # Cell TYPE (CD8 / CD4 / Treg) for the coarse-group aggregation.
    df["cell_type_coarse"] = df["state"].map(_state_to_celltype)

    per_label, _per_group, per_gene = specificity.compute(
        df, gene_col="gene", label_col="state",
        mean_log2_col="tcell_mean_log2_expression",
        frac_col="tcell_fraction_expressing", ncells_col="n_cells",
        detection_frac_min=DETECTION_FRAC_MIN)   # native axes; cell type below
    per_label = per_label.rename(columns={"detected": "tcell_detected"})

    # --- Cell-type specificity (CD8 / CD4 / Treg), DETECTED-BASIS -----------
    # The SHARED coarse-group aggregation (cell-weighted; the same effective_n /
    # concentration-tier math Song/5xFAD use, just with cell types as the unit),
    # with the share basis restricted to DETECTED expression. At shallow scRNA
    # depth an undetected state is noise, not low-real-expression, so letting it
    # into the basis lets a rounding-error fraction "win" a phantom home (CAMK1G/
    # TTBK1 read as CD4 while detected nowhere). We zero the expression of
    # undetected states but KEEP all three cell types present, so the
    # concentration tier stays a fold over the fixed 1/3 even share (filtering the
    # rows out would shrink the denominator and make "concentrated" unreachable).
    # A kinase detected in zero states then has an all-zero basis and is nulled
    # below (no home, no tier, no effective_n → no pill). The per-STATE tier stays
    # retired (the ~14 states are transcriptionally homogeneous, saturating at ≥2×).
    df_ct = df.copy()
    _und = df_ct["tcell_fraction_expressing"].astype(float) < DETECTION_FRAC_MIN
    df_ct.loc[_und, "tcell_mean_log2_expression"] = np.nan   # → linear 0
    _, per_group_ct, per_gene_ct = specificity.compute(
        df_ct, gene_col="gene", label_col="state",
        mean_log2_col="tcell_mean_log2_expression",
        frac_col="tcell_fraction_expressing", ncells_col="n_cells",
        group_col="cell_type_coarse", detection_frac_min=DETECTION_FRAC_MIN)
    top_tier = (
        per_group_ct.merge(per_gene_ct[["gene", "top_group_coarse"]], on="gene")
        .loc[lambda d: d["cell_type_coarse"] == d["top_group_coarse"],
             ["gene", "concentration_tier_coarse"]]
        .rename(columns={
            "concentration_tier_coarse": "tcell_celltype_concentration_tier"}))
    ct_axis = (per_gene_ct[["gene", "effective_n_coarse", "top_group_coarse"]]
               .merge(top_tier, on="gene", how="left"))
    per_gene = per_gene.merge(ct_axis, on="gene", how="left")
    per_gene = per_gene.rename(columns={
        "n_detected_native": "tcell_n_detected",
        "effective_n_native": "tcell_effective_n",        # per-state subtype spread
        "effective_n_coarse": "tcell_celltype_effective_n",
        "top_group_coarse": "tcell_top_celltype"})         # CD8 / CD4 / Treg

    # Gene-level guard: a kinase detected in zero states has an all-zero share
    # basis, so its argmax cell type is arbitrary noise — null the whole axis.
    _no_det = per_gene["tcell_n_detected"].fillna(0).astype(int) == 0
    per_gene.loc[_no_det, ["tcell_top_celltype", "tcell_celltype_effective_n",
                           "tcell_celltype_concentration_tier"]] = np.nan

    # --- State enrichment (activation continuum), GUARDED -------------------
    # For each state: fold of that state's linear expression over the gene's
    # BASELINE = MEAN linear expression across all adequately-sampled states
    # (n_cells >= MIN_STATE_CELLS), gene-AGNOSTIC in its state set. The baseline
    # deliberately includes states where THIS gene is undetected, so a kinase
    # living in one state divides by a baseline pulled down by its many low
    # states (-> high fold = state-specific). This is the brain cohorts'
    # celltype_mean / global_mean specificity, ported to states.
    #   MEAN, not median: a median over the ~13 sampled states is 0 for a
    #   single-state gene and would null the very signal we want. The earlier
    #   median-over-eligible-states baseline INVERTED the metric — it normalized
    #   each gene by its OWN detected states, so a 1-state gene scored exactly 1.0
    #   and breadth (not specificity) drove the fold (Spearman +0.69).
    #   DISPLAY guard unchanged: only ELIGIBLE states (detected AND
    #   >= MIN_STATE_CELLS cells) carry a badge; ineligible states get NaN. A
    #   kinase with no eligible state has no state enrichment.
    sampled = per_label["n_cells"] >= MIN_STATE_CELLS
    baseline = (per_label[sampled].groupby("gene")["linear_expression"]
                .mean().replace(0.0, np.nan))
    per_label["tcell_state_enrichment"] = (
        per_label["linear_expression"] / per_label["gene"].map(baseline))
    elig = per_label["tcell_detected"] & sampled
    per_label.loc[~elig, "tcell_state_enrichment"] = np.nan
    return per_label, per_gene


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
    print(f"  detection evidence marker: fraction_cells_expressing >= "
          f"{DETECTION_FRAC_MIN}")

    spec, gene_breadth = _compute_metric(donor, mean_long)
    conc = _compute_concordance(mean_long)
    mea = _load_mea(donor)
    k2g = _load_kinase_to_gene()

    out_dir = _out_dir(donor)
    os.makedirs(out_dir, exist_ok=True)
    spec_cols = ["gene", "state", "tcell_detected", "tcell_fraction_expressing",
                 "tcell_state_enrichment", "tcell_mean_log2_expression"]
    spec[spec_cols].to_csv(
        os.path.join(out_dir, "tcell_enrichment.csv"), index=False)
    conc.to_csv(os.path.join(out_dir, "tcell_concordance.csv"), index=False)
    print(f"  tcell_enrichment.csv: {len(spec)} (gene × state) rows, "
          f"{int(spec['tcell_detected'].sum())} detected "
          f"(>= {DETECTION_FRAC_MIN:.0%} cells expressing)")
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
    # detection metric (per gene × state, repeated across contrast days)
    grid = grid.merge(
        spec[spec_cols].rename(columns={"gene": "gene_symbol",
                                        "state": "cell_type"}),
        on=["gene_symbol", "cell_type"], how="left")
    grid = grid.merge(
        gene_breadth[["gene", "tcell_effective_n", "tcell_top_celltype",
                      "tcell_celltype_effective_n",
                      "tcell_celltype_concentration_tier"]].rename(
            columns={"gene": "gene_symbol"}),
        on="gene_symbol", how="left")
    grid["tcell_detected"] = (
        grid["tcell_detected"].fillna(False).infer_objects(copy=False).astype(bool))
    grid["tcell_celltype_concentration_tier"] = (
        grid["tcell_celltype_concentration_tier"].fillna(0).astype(int))
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
    # it must not filter anything. See docs/tcell_exhaustion_analysis_summary.md.
    grid["tcell_concordant"] = grid["tcell_concordance"] > 0

    cols = ["kinase", "residue_type", "gene_symbol", "contrast", "cell_type",
            "NES", "FDR",
            "tcell_detected", "tcell_fraction_expressing", "tcell_state_enrichment",
            "tcell_effective_n", "tcell_top_celltype",
            "tcell_celltype_effective_n", "tcell_celltype_concentration_tier",
            "tcell_lfc", "tcell_concordance", "tcell_concordant",
            "tcell_consistency"]
    full = grid[cols].copy()

    expected = len(kinases) * n_states * len(CONTRAST_DAYS)
    if len(full) != expected:
        raise AssertionError(
            f"full row count {len(full)} != expected {expected} "
            f"(n_kinase_tracks {len(kinases)} × n_states {n_states} × "
            f"n_contrast_days {len(CONTRAST_DAYS)}) — silent drop in merge")

    # Ship the ENTIRE grid (no detection/concordance gate — detection is a SHOWN
    # axis, like the mouse). Sorted for readability only (state enrichment desc).
    shipped = full.sort_values("tcell_state_enrichment", ascending=False)
    shipped.to_csv(os.path.join(out_dir, "unified_attribution_tcells.csv"),
                   index=False)

    enr = shipped["tcell_state_enrichment"].dropna()
    enr_dist = {">=3x": int((enr >= 3).sum()), ">=2x": int((enr >= 2).sum()),
                ">=1.5x": int((enr >= 1.5).sum()), "<1.5x": int((enr < 1.5).sum())}
    n_conc = int(shipped["tcell_concordant"].sum())
    n_detected = int(shipped["tcell_detected"].sum())
    print(f"  unified_attribution_tcells.csv: {len(shipped)} rows "
          f"(guard {expected} ✓; all rows shipped, no gate)")
    print(f"  concordance is a LABEL: {n_conc} concordant / "
          f"{len(shipped) - n_conc} discordant")
    print(f"  detected (frac>={DETECTION_FRAC_MIN}): {n_detected}/{len(shipped)} rows; "
          f"state_enrichment dist {enr_dist}")
    # Cell-type axis: one value per kinase track (broadcast across state×day).
    per_kinase = shipped.drop_duplicates(["kinase", "residue_type"])
    ct_dist = per_kinase["tcell_top_celltype"].value_counts(dropna=False).to_dict()
    tier_dist = (per_kinase["tcell_celltype_concentration_tier"]
                 .value_counts().sort_index(ascending=False).to_dict())
    ceff = per_kinase["tcell_celltype_effective_n"].dropna()
    print(f"  cell-type axis (per kinase): top_celltype {ct_dist}; "
          f"tier {tier_dist}; eff[min/median/max] "
          f"{ceff.min():.2f}/{ceff.median():.2f}/{ceff.max():.2f}")
    n_no_gene = grid["tcell_mean_log2_expression"].isna().groupby(
        [grid["kinase"], grid["residue_type"]]).all().sum()
    print(f"  kinase tracks with no transcript in scRNA: {n_no_gene}")

    return {"n_states": n_states, "n_kinases": len(kinases),
            "n_full": len(full), "n_shipped": len(shipped),
            "n_concordant": n_conc, "n_detected": n_detected,
            "enrichment_dist": enr_dist}


if __name__ == "__main__":
    donor = sys.argv[1] if len(sys.argv) > 1 else "donor1"
    build(donor)

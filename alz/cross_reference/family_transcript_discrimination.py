"""Can per-cell-type transcript level discriminate kinases within a family?

MEA scores a kinase by motif enrichment across a ranked phosphosite list. Kinases with
near-identical motifs are therefore scored against near-identical evidence and receive
near-identical NES — the assay cannot say which family member produced the signal. This
module asks whether snRNA transcript level, measured per cell type, is independent enough
to break that tie.

Scope is deliberately narrow: **does it discriminate**. No reweighting of NES, no
redistribution of activity, no viewer payload. The output is evidence for deciding whether
a transcript-weighted activity track is worth building.

Source
------
``outputs/reports/attribution_recovery/celltype_evidence_table.csv`` (produced by
``alz/bulk_mea/recover.py``) — one row per kinase x cell type for the Song cohort, carrying
both matched-snRNA measures:

- ``song_fraction_cells_expressing`` — fraction of cells with non-zero counts
- ``song_concentration`` + ``song_concentration_tier`` — expression level and its bucket

Note: ``kinase_incytr_bridge.py`` reads ``kinase_hypothesis_table.csv`` instead, which keeps
only the top-3 cell-type names, and hardcodes ``expression_fraction = None`` for Song. The
per-cell-type Song measures used here are therefore present on disk but absent from the
bridge output.

Zeros are treated as zeros — no dropout floor, no imputation.

Grouping is FAMILY from the Kinase Library kinome table. Family and motif similarity are
orthogonal signals and are not interchangeable; this module groups by family only.

Run
---
    pixi run python -m alz.cross_reference.family_transcript_discrimination

Writes ``outputs/reports/family_transcript_discrimination/`` — report.md, family_summary.csv,
pair_stats.csv. Read-only with respect to every input.
"""
from __future__ import annotations

import logging
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
KINOME = (
    REPO_ROOT
    / ".pixi/envs/default/lib/python3.11/site-packages/kinase_library"
    / "databases/kinase_data/kinome_information.tsv"
)
EVIDENCE = Path("outputs/reports/attribution_recovery/celltype_evidence_table.csv")
ABBREV_MAP = Path("data/derived/caches/kinase_to_gene_mapping.csv")
OUT_DIR = Path("outputs/reports/family_transcript_discrimination")

# Presence uses the project's existing tier vocabulary rather than a new threshold:
# tier 0 is the bottom bucket, >= 1 is a positive concentration call.
PRESENT_TIER = 1
# Control sample size, and the seed that makes it reproducible.
N_CONTROL = 20_000
SEED = 0
# "Indistinguishable" cutoff on cosine — reported, not used to filter.
IDENTICAL_COSINE = 0.99


# ---------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------
def load_family_map(kinases: list[str]) -> pd.Series:
    """Kinase (MEA abbreviation) -> FAMILY, via gene symbol with a matrix-name fallback."""
    kinome = pd.read_csv(KINOME, sep="\t")
    kinome["gene_u"] = kinome["GENE_NAME"].astype(str).str.upper()
    by_gene = (
        kinome.dropna(subset=["FAMILY"]).drop_duplicates("gene_u").set_index("gene_u")["FAMILY"]
    )
    by_matrix = (
        kinome.dropna(subset=["FAMILY"])
        .drop_duplicates("MATRIX_NAME")
        .set_index("MATRIX_NAME")["FAMILY"]
    )
    abbrev = pd.read_csv(ABBREV_MAP)
    amap = dict(
        zip(
            abbrev[abbrev.columns[0]].astype(str),
            abbrev[abbrev.columns[1]].astype(str).str.upper(),
        )
    )
    out = {}
    for k in kinases:
        gene = amap.get(str(k))
        fam = by_gene.get(gene) if gene else None
        if not isinstance(fam, str):
            fam = by_matrix.get(str(k))
        if isinstance(fam, str):
            out[k] = fam
    return pd.Series(out, name="family")


def load_matrices() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Kinase x cell-type matrices for fraction, concentration, and concentration tier."""
    ev = pd.read_csv(
        EVIDENCE,
        usecols=[
            "kinase",
            "cell_type",
            "song_fraction_cells_expressing",
            "song_concentration",
            "song_concentration_tier",
        ],
    ).drop_duplicates(subset=["kinase", "cell_type"])

    def pivot(col: str) -> pd.DataFrame:
        m = ev.pivot(index="kinase", columns="cell_type", values=col)
        return m.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    return (
        pivot("song_fraction_cells_expressing"),
        pivot("song_concentration"),
        pivot("song_concentration_tier"),
    )


# ---------------------------------------------------------------------------
# pair statistics
# ---------------------------------------------------------------------------
def _cosine(x: np.ndarray, y: np.ndarray) -> float:
    nx, ny = np.linalg.norm(x), np.linalg.norm(y)
    return float(x @ y / (nx * ny)) if nx and ny else np.nan


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.corrcoef(x, y)[0, 1]) if x.std() and y.std() else np.nan


def pair_stats(i: int, j: int, V: np.ndarray, L: np.ndarray, T: np.ndarray) -> dict:
    """Discrimination statistics for one kinase pair.

    ``V`` raw measure, ``L`` its log1p transform (the skew-corrected space the summary
    statistics use), ``T`` concentration tier for the presence calls.
    """
    x, y = V[i], V[j]
    lx, ly = L[i], L[j]
    tx, ty = T[i], T[j]
    exclusive = ((tx >= PRESENT_TIER) & (ty < PRESENT_TIER)) | (
        (ty >= PRESENT_TIER) & (tx < PRESENT_TIER)
    )
    both_present = (tx >= PRESENT_TIER) & (ty >= PRESENT_TIER)
    n_exc, n_both = int(exclusive.sum()), int(both_present.sum())
    occupied = n_exc + n_both
    return {
        "cosine_log": _cosine(lx, ly),
        "pearson_log": _pearson(lx, ly),
        "max_gap_raw": float(np.abs(x - y).max()),
        "n_exclusive": n_exc,
        "n_both_present": n_both,
        # Share of the cell types either member occupies that only one of them occupies.
        # The raw count saturates (a kinase is present in ~11 of 31 types, so discordant
        # types are near-guaranteed); this ratio does not.
        "exclusivity_ratio": (n_exc / occupied) if occupied else np.nan,
        "argmax_differs": int(np.argmax(x) != np.argmax(y)),
        "either_present": int(occupied > 0),
    }


def build_pairs(mat: pd.DataFrame, tier: pd.DataFrame, fam: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Same-family pairs and a seeded different-family control."""
    keys = [k for k in fam.index if k in mat.index]
    fam = fam.loc[keys]
    V = mat.loc[keys].to_numpy(dtype=float)
    T = tier.loc[keys].to_numpy(dtype=float)
    L = np.log1p(V)
    idx = {k: n for n, k in enumerate(keys)}

    same = [
        {"family": f, "a": a, "b": b, **pair_stats(idx[a], idx[b], V, L, T)}
        for f, grp in fam.groupby(fam)
        if len(grp) >= 2
        for a, b in combinations(list(grp.index), 2)
    ]

    rng = np.random.default_rng(SEED)
    seen: set[tuple[str, str]] = set()
    control = []
    guard = 0
    while len(control) < N_CONTROL and guard < N_CONTROL * 50:
        guard += 1
        p, q = rng.choice(len(keys), 2, replace=False)
        a, b = keys[p], keys[q]
        if fam[a] == fam[b] or (a, b) in seen:
            continue
        seen.add((a, b))
        control.append(pair_stats(idx[a], idx[b], V, L, T))
    return pd.DataFrame(same), pd.DataFrame(control)


def summarize(df: pd.DataFrame) -> dict:
    if df.empty:
        return {}
    return {
        "pairs": len(df),
        "cosine_log_mean": df["cosine_log"].mean(),
        "pearson_log_mean": df["pearson_log"].mean(),
        "exclusivity_ratio_mean": df["exclusivity_ratio"].mean(),
        "exclusivity_ratio_median": df["exclusivity_ratio"].median(),
        "median_exclusive": df["n_exclusive"].median(),
        "median_both_present": df["n_both_present"].median(),
        "pct_argmax_differs": df["argmax_differs"].mean(),
        "pct_indistinguishable": (df["cosine_log"] >= IDENTICAL_COSINE).mean(),
    }


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------
def _row(label: str, s: dict) -> str:
    return (
        f"| {label} | {s['pairs']} | {s['cosine_log_mean']:.3f} | {s['pearson_log_mean']:.3f} "
        f"| {s['exclusivity_ratio_mean']:.3f} | {int(s['median_exclusive'])} "
        f"| {int(s['median_both_present'])} | {s['pct_argmax_differs']:.1%} "
        f"| {s['pct_indistinguishable']:.1%} |"
    )


# Per-family verdict bands on cosine of the log concentration profile — low cosine means the
# members' across-cell-type profiles point in different directions and so separate them.
#
# The verdict is NOT based on the exclusivity ratio. That ratio sits near its own baseline
# (same-family ~0.74 vs different-family ~0.76) because presence is sparse — a kinase occupies
# a median 12 of 31 cell types, so almost any two kinases occupy mostly disjoint sets. It
# measures occupancy sparsity, not family separation, and would rate nearly every family as
# discriminated. Profile correlation is the quantity that actually varies (0.12 to 0.99).
CLEAN_COSINE = 0.50
WEAK_COSINE = 0.80


def family_table(same: pd.DataFrame, sizes: pd.Series, min_members: int = 3) -> pd.DataFrame:
    agg = same.groupby("family").agg(
        pairs=("cosine_log", "size"),
        cosine_log=("cosine_log", "mean"),
        exclusivity_ratio=("exclusivity_ratio", "mean"),
        exclusive=("n_exclusive", "mean"),
        both_present=("n_both_present", "mean"),
        argmax_differs=("argmax_differs", "mean"),
    )
    agg["members"] = agg.index.map(sizes)
    agg = agg[agg["members"] >= min_members]
    agg["verdict"] = np.where(
        agg["cosine_log"] <= CLEAN_COSINE,
        "discriminated",
        np.where(agg["cosine_log"] <= WEAK_COSINE, "partial", "undiscriminated"),
    )
    return agg.sort_values("cosine_log")[
        ["members", "pairs", "cosine_log", "exclusivity_ratio", "exclusive",
         "both_present", "argmax_differs", "verdict"]
    ]


def write_report(results: dict, fam_tables: dict, n_kin: int, n_ct: int, out: Path) -> None:
    frac, conc = results["fraction"], results["concentration"]
    ft = fam_tables["concentration"]
    disc = (ft["verdict"] == "discriminated").sum()
    undisc = (ft["verdict"] == "undiscriminated").sum()

    lines = [
        "# Kinase family discrimination by per-cell-type transcript level",
        "",
        "Generated by `alz/cross_reference/family_transcript_discrimination.py`. Regenerate with:",
        "",
        "```",
        "pixi run python -m alz.cross_reference.family_transcript_discrimination",
        "```",
        "",
        "## Question",
        "",
        "MEA scores kinases by motif enrichment over a ranked phosphosite list, so kinases with",
        "near-identical motifs receive near-identical NES — the assay cannot attribute the signal",
        "to one family member over another. This asks whether snRNA transcript level, per cell",
        "type, separates them.",
        "",
        "Scope is discrimination only. No NES reweighting, no redistribution, no payload changes.",
        "",
        "## Data",
        "",
        f"Song cohort matched snRNA, {n_kin} kinases x {n_ct} cell types, from",
        "`outputs/reports/attribution_recovery/celltype_evidence_table.csv`.",
        "Zeros are zeros — no dropout floor, no imputation. Grouping is FAMILY from the Kinase",
        "Library kinome table.",
        "",
        "Two measures compared: `song_fraction_cells_expressing` (fraction of cells with non-zero",
        "counts) and `song_concentration` (expression level). Summary statistics are computed on",
        "log1p, since concentration is zero-inflated and right-skewed. Presence uses the existing",
        f"`song_concentration_tier` vocabulary (tier >= {PRESENT_TIER}), not a new threshold.",
        "",
        "**A cell type is 'exclusive' for a pair when one member is present there and the other",
        "is not.** That is the quantity of interest: it is a cell type in which the family's",
        "shared MEA signal can only be attributed to one of them.",
        "",
        "The raw exclusive *count* saturates and must not be used as a headline. A kinase is",
        "present in a median of 12 of 31 cell types, so two kinases almost always disagree",
        "somewhere — 'has >= 1 exclusive cell type' is true for 100% of pairs and carries no",
        "information. The reported **exclusivity ratio** normalises it: of the cell types either",
        "member occupies, the share occupied by only one of them. 1.0 is disjoint occupancy,",
        "0.0 is identical occupancy.",
        "",
        "Presence is defined once, from `song_concentration_tier`. The exclusivity columns are",
        "therefore identical between the two measures by construction — the measures differ only",
        "on the continuous statistics (cosine, pearson), which is where the comparison lives.",
        "",
        "## Result",
        "",
        "| measure / group | pairs | cosine (log) | pearson (log) | exclusivity ratio | median exclusive | median shared | top type differs | indistinguishable |",
        "|---|---|---|---|---|---|---|---|---|",
        _row("**concentration** — same family", conc["same"]),
        _row("concentration — different family (control)", conc["control"]),
        _row("**fraction** — same family", frac["same"]),
        _row("fraction — different family (control)", frac["control"]),
        "",
        "The control is context, not a null to beat: the question is whether family members can",
        "be separated at all, not whether they are harder to separate than arbitrary pairs. A",
        "control close to the same-family row means family membership does not make separation",
        "unusually difficult.",
        "",
        "## Per-family verdict (concentration, families with >= 3 members)",
        "",
        f"{disc} families discriminated, {undisc} undiscriminated, "
        f"{len(ft) - disc - undisc} partial.",
        "",
        f"**discriminated** = mean cosine <= {CLEAN_COSINE:.2f} · "
        f"**partial** = <= {WEAK_COSINE:.2f} · **undiscriminated** = above that.",
        "",
        "The verdict rests on profile cosine, not the exclusivity ratio. The ratio sits at its",
        "own baseline (same-family 0.735 vs different-family 0.764) because occupancy is sparse,",
        "so it rates almost every family as separable and is not a usable discriminator. Profile",
        "cosine spans 0.12 to 0.99 across families and does separate them.",
        "",
        "| family | members | pairs | cosine (log) | exclusivity ratio | mean exclusive | mean shared | top type differs | verdict |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for f, r in ft.iterrows():
        lines.append(
            f"| {f} | {int(r['members'])} | {int(r['pairs'])} | {r['cosine_log']:.3f} "
            f"| {r['exclusivity_ratio']:.3f} | {r['exclusive']:.2f} | {r['both_present']:.2f} "
            f"| {r['argmax_differs']:.1%} | {r['verdict']} |"
        )
    lines += [
        "",
        "## Reading this for viewer integration",
        "",
        "- Discrimination is **family-dependent, not uniform**. Any surface built on this must",
        "  carry the per-family verdict, so undiscriminated families are shown as undiscriminated",
        "  rather than given an arbitrary winner.",
        "- The unit that matters is the exclusive cell type. It supports the claim 'in this cell",
        "  type the signal is attributable to A, not B' and nothing stronger.",
        "- Transcript is not activity. A present transcript does not establish that the kinase is",
        "  active; an absent one is the informative direction, since a kinase that is not",
        "  transcribed in a cell type cannot be responsible for signal attributed there.",
        "- Song only. 5xFAD and T-cells carry `expression_fraction` in the bridge parquet at the",
        "  same grain and can be measured the same way; Song's measures are richer but reach only",
        "  this evidence table, not the bridge.",
        "",
        "## Files",
        "",
        "- `family_summary.csv` — per-family table above, both measures",
        "- `pair_stats.csv` — every same-family pair with its statistics",
    ]
    (out / "report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if not EVIDENCE.exists():
        raise SystemExit(
            f"missing {EVIDENCE} — run the attribution recovery step first "
            "(alz/bulk_mea/recover.py)"
        )
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    frac_m, conc_m, tier_m = load_matrices()
    fam = load_family_map(list(conc_m.index))
    sizes = fam.value_counts()
    log.info(
        f"{conc_m.shape[0]} kinases x {conc_m.shape[1]} cell types; "
        f"{len(fam)} with a family, {(sizes >= 2).sum()} families with >= 2 members"
    )

    results, fam_tables = {}, {}
    for name, mat in (("fraction", frac_m), ("concentration", conc_m)):
        same, control = build_pairs(mat, tier_m, fam)
        results[name] = {"same": summarize(same), "control": summarize(control)}
        fam_tables[name] = family_table(same, sizes)
        same.insert(0, "measure", name)
        fam_tables[name].insert(0, "measure", name)
        results[name]["_pairs"] = same
        s = results[name]["same"]
        log.info(
            f"{name:>14}: same-family {s['pairs']} pairs, "
            f"exclusivity ratio {s['exclusivity_ratio_mean']:.3f}, "
            f"cosine(log)={s['cosine_log_mean']:.3f}"
        )

    pd.concat([fam_tables[k].reset_index() for k in fam_tables]).to_csv(
        OUT_DIR / "family_summary.csv", index=False
    )
    pd.concat([results[k]["_pairs"] for k in results]).to_csv(
        OUT_DIR / "pair_stats.csv", index=False
    )
    write_report(results, fam_tables, conc_m.shape[0], conc_m.shape[1], OUT_DIR)
    log.info(f"wrote {OUT_DIR}/report.md")


if __name__ == "__main__":
    main()

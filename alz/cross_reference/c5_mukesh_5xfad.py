"""alz/cross_reference/c5_mukesh_5xfad.py

C5 analysis — Mukesh b-donorset substrate profile vs 5xFAD (ST track, 8 contexts).

Orchestrates:
  Stage 1  — Load the frozen 60-kinase pool from overlap_AD8_sus_clean.csv.
  Stage 2  — Build the human per-donor profile over the b-donorset
              (AD8 excl AD-01/AD-03, plus CTRL-08/CTRL-10 as AD) at M=1.
            — Load measured-gene universes (human once; mouse per tissue).
  Stage 3  — Sweep 8 mouse contexts (cortex/hippocampus × 3/6/9/12mo):
              build mouse profile → substrate_overlap(human, mouse, universe_h,
              universe_m) + substrate_similarity(human, mouse).
  Stage 4  — Assemble 60-kinase × 8-context summary matrix + MANIFEST.

Overlap metric is gene-identity-keyed set decomposition:
  shared_genes / human_only_genes / mouse_only_genes, refined by BLOSUM62 site
  matching within shared genes (same-site vs diff-site), and tagged by coverage
  (engaged = gene measurable in the other cohort; unmeasured = coverage gap).
  overlap_frac_gene = shared_gene / (shared + human_only + mouse_only) genes.

Output: outputs/reports/substrate_compare/c5_mukesh_5xfad_<YYYYMMDD_HHMMSS>/
  kinase_summary.csv         — 60 × 8 matrix (one row per kinase × context),
                                overlap_frac_gene / gene counts / coverage splits
                                / n_shared_site / n_diffsite / direction_agree_frac
                                / blosum_similarity / sim_hist
  kinase_pairs_<ctx>.csv     — per-context substrate detail with partition +
                                coverage columns
  manifest.json              — pool metadata, donor set, parameters, honesty notes
  support_distribution.csv   — human-side motif support counts (honesty guard)

Memory: DuckDB-streamed, pool-filtered at scan.  No whole-file pandas reads of
substrate-set CSVs or large matrices.  Run under systemd-run memory cap.

CLI flags
---------
  --context TISSUE_AGE     Limit to one context (e.g. cortex_12mo) — smoke mode.
  --out-dir PATH           Override the auto-generated timestamped output dir.
  --no-manifest            Skip MANIFEST emit (for fast iteration).

Task: pixi run c5-mukesh-5xfad
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import os
import resource
import sys
import traceback
from pathlib import Path
from typing import Optional

import duckdb
import numpy as np

# Ensure project root on sys.path when invoked as a module
_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from alz.cross_reference.substrate_motif_compare import (
    SIM_FLOOR,
    Profile,
    SubstrateOverlap,
    build_profile,
    substrate_overlap,
    substrate_similarity,
    load_human_gene_universe,
    load_fivexfad_gene_universe,
    motif_similarity,
)

# BLOSUM-similarity histogram of per-human-motif best matches (viewer detail
# panel). Fixed bins over the full [0, 1] range so the distribution spans
# low-agreement motifs, not only the shared (>= SIM_FLOOR) subset.
_HIST_N_BINS = 10
_HIST_RANGE = (0.0, 1.0)

# ─── Constants ────────────────────────────────────────────────────────────────

POOL_CSV = (
    "outputs/reports/kinase_attribution_human/ctrl_audit/"
    "concordance_AD8_excl01_03/overlap_AD8_sus_clean.csv"
)
SAMPLE_MAP_CSV = "outputs/reports/data_ingest_human/sample_mapping.csv"

TISSUES = ("cortex", "hippocampus")
AGES = ("3mo", "6mo", "9mo", "12mo")

# Donors excluded from the b-donorset (vote outliers in AD group)
BDONORSET_EXCLUDE = {"AD-01", "AD-03"}

# Suspects treated as AD in the b-donorset
BDONORSET_SUSPECTS = ["CTRL-08", "CTRL-10"]


# ─── Stage 1: pool loader ─────────────────────────────────────────────────────

def load_pool(repo_root: str) -> list[str]:
    """Load the 60-kinase pool from overlap_AD8_sus_clean.csv.

    Returns a sorted list of unique kinase names (all ST).
    """
    csv_path = os.path.join(repo_root, POOL_CSV)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Pool CSV not found: {csv_path}")
    conn = duckdb.connect()
    rows = conn.execute(
        f"SELECT DISTINCT kinase FROM read_csv_auto('{csv_path}') ORDER BY kinase"
    ).fetchall()
    kinases = [r[0] for r in rows]
    if not kinases:
        raise RuntimeError(f"Pool CSV is empty: {csv_path}")
    return kinases


# ─── Stage 2: b-donorset construction ────────────────────────────────────────

def build_b_donorset(repo_root: str) -> list[str]:
    """Construct the b-donorset from sample_mapping.csv.

    b-donorset = (official AD donors) MINUS BDONORSET_EXCLUDE UNION BDONORSET_SUSPECTS.
    Derives sample_ids from data (does not hardcode the AD8 literals).
    Returns sorted list of donor sample_ids.
    """
    smap_path = os.path.join(repo_root, SAMPLE_MAP_CSV)
    if not os.path.exists(smap_path):
        raise FileNotFoundError(f"sample_mapping.csv not found: {smap_path}")
    conn = duckdb.connect()
    rows = conn.execute(
        f'SELECT sample_id, "group" FROM read_csv_auto(\'{smap_path}\')'
    ).fetchall()
    ad_official = {r[0] for r in rows if r[1] == "AD"}
    b_donorset = sorted(
        (ad_official - BDONORSET_EXCLUDE) | set(BDONORSET_SUSPECTS)
    )
    return b_donorset


# ─── Stage 3: mouse context profiles ─────────────────────────────────────────

def _mouse_contrast(tissue: str, age: str) -> str:
    """Format the 5xFAD contrast string for build_profile."""
    return f"{tissue}_TG_vs_WT_{age}"


def all_contexts() -> list[tuple[str, str]]:
    """Return all (tissue, age) pairs."""
    return [(t, a) for t in TISSUES for a in AGES]


# ─── Stage 4: summary matrix ─────────────────────────────────────────────────

def _context_label(tissue: str, age: str) -> str:
    return f"{tissue}_{age}"


def _sim_histogram(best_match: list) -> str:
    """Fixed-bin histogram of per-human-motif best-match similarity, ';'-joined.

    Every human substrate motif contributes its single best BLOSUM match against
    this context's full mouse motif set, spanning [0, 1] — includes low-agreement
    motifs, not only the shared (>= SIM_FLOOR) subset.
    Feeds the viewer detail-panel histogram with no per-pair shard fetch.
    Empty input (kinase absent in this context) yields all-zero bins.
    """
    vals = np.asarray(best_match, dtype=float) if best_match else np.array([])
    counts, _ = np.histogram(vals, bins=_HIST_N_BINS, range=_HIST_RANGE)
    return ";".join(str(int(c)) for c in counts)


def _compute_best_match_hist(
    human_profile: Profile,
    mouse_profile: Profile,
) -> str:
    """Per-human-motif best-match BLOSUM similarity histogram for the detail pane.

    O(H × M) motif_similarity calls; profiles are typically <500 × <200 motifs.
    """
    hkeys = list(human_profile.keys())
    mkeys = list(mouse_profile.keys())
    best_match: list[float] = []
    for ha in hkeys:
        best = 0.0
        for mb in mkeys:
            s = motif_similarity(ha, mb, sim_floor=0.0).score
            if s > best:
                best = s
        best_match.append(best)
    return _sim_histogram(best_match)


def _summary_row(
    kinase: str,
    tissue: str,
    age: str,
    ov: SubstrateOverlap,
    human_profile: Profile,
    mouse_profile: Profile,
    blosum_sim: float,
) -> dict:
    """Build one summary row for the kinase × context matrix.

    Overlap metric: gene-identity partition (shared / human_only / mouse_only),
    refined by BLOSUM62 site matching within shared genes, tagged by coverage
    (engaged vs unmeasured relative to each cohort's measured-gene universe).
    """
    # Support stats from shared-site human side
    supports = [s.support_a for s in ov.shared_sites]
    support_min = int(min(supports)) if supports else 0
    support_med = float(np.median(supports)) if supports else float("nan")

    # Direction correlation over shared sites
    dir_corr: float
    if ov.shared_sites:
        da = np.array([s.direction_a for s in ov.shared_sites], dtype=float)
        db = np.array([s.direction_b for s in ov.shared_sites], dtype=float)
        if np.std(da) > 0 and np.std(db) > 0:
            dir_corr = float(np.corrcoef(da, db)[0, 1])
        else:
            dir_corr = float("nan")
    else:
        dir_corr = float("nan")

    def _num(v):
        return round(float(v), 6) if v is not None and v == v else ""

    daf = ov.direction_agree_frac
    return {
        "kinase": kinase,
        "tissue": tissue,
        "age": age,
        "context": _context_label(tissue, age),
        "n_shared_gene": ov.n_shared_gene,
        "n_human_only_gene": ov.n_human_only_gene,
        "n_mouse_only_gene": ov.n_mouse_only_gene,
        "n_human_only_engaged": ov.n_human_only_engaged,
        "n_human_only_unmeasured": ov.n_human_only_unmeasured,
        "n_mouse_only_engaged": ov.n_mouse_only_engaged,
        "n_mouse_only_unmeasured": ov.n_mouse_only_unmeasured,
        "n_shared_site": ov.n_shared_site,
        "n_diffsite": ov.n_diffsite,
        "overlap_frac_gene": _num(ov.overlap_frac_gene),
        "blosum_similarity": _num(blosum_sim),
        "direction_agree_frac": _num(daf) if daf == daf else "",
        "direction_corr": _num(dir_corr) if dir_corr == dir_corr else "",
        "human_support_min": support_min,
        "human_support_median": round(support_med, 3) if not np.isnan(support_med) else "",
        "sim_hist": _compute_best_match_hist(human_profile, mouse_profile),
    }


# ─── Main orchestration ───────────────────────────────────────────────────────

def run(
    *,
    context_filter: Optional[str] = None,
    out_dir: Optional[str] = None,
    emit_manifest: bool = True,
    verbose: bool = True,
) -> Path:
    """Run C5 analysis.  Returns the output directory path."""
    from alz.shared import config
    repo_root = config.REPO_ROOT

    # ── Stage 1: pool ──────────────────────────────────────────────────────────
    if verbose:
        print("[C5] Stage 1: loading kinase pool...", flush=True)
    kinases = load_pool(repo_root)
    if verbose:
        print(f"[C5]   pool = {len(kinases)} ST kinases", flush=True)

    # ── Stage 2: b-donorset ────────────────────────────────────────────────────
    if verbose:
        print("[C5] Stage 2: building b-donorset and human profiles...", flush=True)
    b_donorset = build_b_donorset(repo_root)
    if verbose:
        print(f"[C5]   b-donorset ({len(b_donorset)} donors): {b_donorset}", flush=True)

    # Build human profiles for all 60 kinases (once — reused across all 8 contexts)
    human_profiles: dict[str, Profile] = {}
    n_empty_human = 0
    for k in kinases:
        try:
            p = build_profile(
                k, "mukesh", "", "st",
                human_mode="perdonor", human_m=1,
                ad_donor_set=b_donorset,
            )
        except Exception as exc:
            if verbose:
                print(f"[C5]   WARN human profile failed for {k}: {exc}", flush=True)
            p = {}
        human_profiles[k] = p
        if not p:
            n_empty_human += 1

    if verbose:
        n_nonempty = len(kinases) - n_empty_human
        print(f"[C5]   human profiles built: {n_nonempty}/{len(kinases)} non-empty", flush=True)

    # Support distribution across all motifs in all human profiles
    all_supports: list[int] = []
    for p in human_profiles.values():
        for e in p.values():
            all_supports.append(e.support)
    if verbose and all_supports:
        print(
            f"[C5]   human motif support — min={min(all_supports)}, "
            f"max={max(all_supports)}, "
            f"median={float(np.median(all_supports)):.1f}, "
            f"n_singleton={sum(1 for s in all_supports if s == 1)}/"
            f"{len(all_supports)} motifs total",
            flush=True,
        )

    # ── Output directory ───────────────────────────────────────────────────────
    if out_dir is None:
        ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.join(
            repo_root, "outputs", "reports", "substrate_compare",
            f"c5_mukesh_5xfad_{ts}"
        )
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    if verbose:
        print(f"[C5]   output → {out_path}", flush=True)

    # ── Load measured-gene universes (DuckDB DISTINCT — never full matrix) ─────
    if verbose:
        print("[C5] Loading measured-gene universes...", flush=True)
    universe_human: set
    try:
        universe_human = load_human_gene_universe("st")
        if verbose:
            print(f"[C5]   human universe: {len(universe_human)} genes", flush=True)
    except FileNotFoundError as exc:
        if verbose:
            print(f"[C5]   WARN human universe load failed: {exc}", flush=True)
        universe_human = set()

    universe_mouse: dict[str, set] = {}
    for tissue in TISSUES:
        try:
            universe_mouse[tissue] = load_fivexfad_gene_universe(tissue, "st")
            if verbose:
                print(f"[C5]   mouse/{tissue} universe: "
                      f"{len(universe_mouse[tissue])} genes", flush=True)
        except FileNotFoundError as exc:
            if verbose:
                print(f"[C5]   WARN mouse/{tissue} universe load failed: {exc}", flush=True)
            universe_mouse[tissue] = set()

    # ── Stage 3 + 4: sweep contexts ───────────────────────────────────────────
    if verbose:
        print("[C5] Stage 3/4: sweeping mouse contexts...", flush=True)

    contexts = all_contexts()
    if context_filter:
        # Accept 'cortex_12mo' or 'cortex 12mo' forms
        cf = context_filter.replace(" ", "_").lower()
        contexts = [(t, a) for t, a in contexts if _context_label(t, a) == cf]
        if not contexts:
            raise ValueError(
                f"--context '{context_filter}' did not match any of: "
                + ", ".join(_context_label(t, a) for t, a in all_contexts())
            )

    summary_rows: list[dict] = []
    missing_kinases_per_ctx: dict[str, list[str]] = {}

    for tissue, age in contexts:
        ctx_label = _context_label(tissue, age)
        mouse_contrast = _mouse_contrast(tissue, age)
        u_mouse = universe_mouse.get(tissue, set())
        if verbose:
            print(f"[C5]   context {ctx_label}...", flush=True)

        mouse_profiles_ctx: dict[str, Profile] = {}
        ctx_overlaps: list = []
        missing_for_ctx: list[str] = []

        for k in kinases:
            human_p = human_profiles[k]
            try:
                mouse_p = build_profile(k, "fivexfad", mouse_contrast, "st")
            except Exception as exc:
                if verbose:
                    print(f"[C5]     WARN mouse profile failed {k}/{ctx_label}: {exc}", flush=True)
                mouse_p = {}

            if not mouse_p and not human_p:
                missing_for_ctx.append(k)

            ov = substrate_overlap(
                human_p, mouse_p,
                universe_a=universe_human,
                universe_b=u_mouse,
            )
            sim = substrate_similarity(human_p, mouse_p)
            ctx_overlaps.append((k, ov, sim, human_p, mouse_p))
            mouse_profiles_ctx[k] = mouse_p

        for k, ov, sim, human_p, mouse_p in ctx_overlaps:
            summary_rows.append(
                _summary_row(k, tissue, age, ov, human_p, mouse_p, sim)
            )

        if verbose:
            n_any_shared = sum(1 for _, ov, _, _, _ in ctx_overlaps if ov.n_shared_gene > 0)
            overlap_vals = [ov.overlap_frac_gene for _, ov, _, _, _ in ctx_overlaps
                           if ov.n_shared_gene + ov.n_human_only_gene + ov.n_mouse_only_gene > 0]
            mean_ov = float(np.mean(overlap_vals)) if overlap_vals else float("nan")
            print(
                f"[C5]     {ctx_label}: {n_any_shared}/{len(kinases)} kinases have "
                f"≥1 shared gene  mean overlap_frac={mean_ov:.3f}",
                flush=True,
            )

        if missing_for_ctx:
            missing_kinases_per_ctx[ctx_label] = missing_for_ctx

        # Emit per-context pairs CSV
        pairs_path = out_path / f"kinase_pairs_{ctx_label}.csv"
        _emit_pairs_csv(ctx_overlaps, pairs_path, mouse_contrast)
        if verbose:
            n_miss = len(missing_for_ctx)
            print(
                f"[C5]     {ctx_label}: pairs written; "
                f"{n_miss}/{len(kinases)} kinases both-empty",
                flush=True,
            )

    # ── Summary CSV ───────────────────────────────────────────────────────────
    summary_path = out_path / "kinase_summary.csv"
    _emit_summary_csv(summary_rows, summary_path)
    if verbose:
        print(f"[C5]   kinase_summary.csv written ({len(summary_rows)} rows)", flush=True)

    # ── Support distribution CSV ───────────────────────────────────────────────
    supp_path = out_path / "support_distribution.csv"
    _emit_support_distribution(human_profiles, supp_path)
    if verbose:
        print(f"[C5]   support_distribution.csv written", flush=True)

    # ── Headline table (cortex-12mo) — ranked by overlap_frac_gene ────────────
    headline_rows = [r for r in summary_rows if r["context"] == "cortex_12mo"]
    if headline_rows:
        headline_rows_sorted = sorted(
            headline_rows,
            key=lambda r: (float(r["overlap_frac_gene"])
                           if r["overlap_frac_gene"] != "" else -1.0),
            reverse=True,
        )
        headline_path = out_path / "headline_cortex_12mo.csv"
        _emit_summary_csv(headline_rows_sorted, headline_path)
        if verbose:
            print(f"[C5]   headline_cortex_12mo.csv written", flush=True)

    # ── MANIFEST ──────────────────────────────────────────────────────────────
    if emit_manifest:
        manifest = {
            "analysis": "C5 — Mukesh b-donorset vs 5xFAD substrate overlap",
            "generated_at": _dt.datetime.now().isoformat(),
            "pool": {
                "source": POOL_CSV,
                "n_kinases": len(kinases),
                "track": "ST only",
                "kinases": kinases,
            },
            "b_donorset": {
                "members": b_donorset,
                "n": len(b_donorset),
                "construction": (
                    "official AD donors (sample_mapping.csv group=='AD') "
                    "MINUS {AD-01, AD-03} (vote outliers) "
                    "UNION {CTRL-08, CTRL-10} (suspects treated as AD, one-off policy). "
                    "To use the official grouping: pass ad_donor_set equal to the "
                    "unfiltered group=='AD' list (one-argument swap, no code change)."
                ),
                "excluded_from_ad": sorted(BDONORSET_EXCLUDE),
                "suspects_added": sorted(BDONORSET_SUSPECTS),
            },
            "overlap_definition": {
                "partition_key": (
                    "Ortholog gene symbol (uppercased); human HGNC and mouse MGI "
                    "symbols are identical for the 1:1 orthologs of these substrates. "
                    "Caveat: no curated ortholog table — uppercase-symbol equality is v1."
                ),
                "site_refinement": (
                    f"Within shared genes: BLOSUM62 motif similarity >= {SIM_FLOOR} "
                    "(center-aligned, central-residue-type gated) distinguishes "
                    "shared_site (matched residue) from shared_gene_diffsite (same "
                    "protein, different engaged residue)."
                ),
                "coverage_aware_uniqueness": (
                    "human_only / mouse_only genes split by the other cohort's "
                    "measured-gene universe (SELECT DISTINCT gene_symbol from ST "
                    "stoichiometry matrix). engaged = gene detectable in the other "
                    "cohort but not engaged by this kinase → real model difference. "
                    "unmeasured = gene not in the other cohort's universe → assay "
                    "coverage gap. Universe is the cohort's full measured-gene set, "
                    "not restricted to the 60-kinase candidate substrates (conservative "
                    "detectable-at-all denominator)."
                ),
                "direction_concordance": (
                    "Fraction of shared_site pairs where disease direction agrees. "
                    "Computed over shared_site pairs only (BLOSUM-matched residue, "
                    "same gene). NaN when no shared sites."
                ),
                "mouse_direction": (
                    "Per-replicate majority vote: wt_mean = nanmean(WT replicates), "
                    "direction = majority sign of (TG_i - wt_mean). Symmetric with "
                    "the human per-donor majority vote. Falls back to 0 if TG or WT "
                    "replicate columns are absent for a tissue × age."
                ),
                "blosum_similarity": (
                    "Symmetric mean best-match: 0.5*(mean_a max_b sim(a,b) + "
                    "mean_b max_a sim(a,b)) over all motif pairs. Descriptive "
                    "secondary metric; not the partition key."
                ),
                "sim_floor": SIM_FLOOR,
                "columns": [
                    "n_shared_gene", "n_human_only_gene", "n_mouse_only_gene",
                    "n_human_only_engaged", "n_human_only_unmeasured",
                    "n_mouse_only_engaged", "n_mouse_only_unmeasured",
                    "n_shared_site", "n_diffsite",
                    "overlap_frac_gene", "blosum_similarity",
                    "direction_agree_frac", "direction_corr",
                ],
            },
            "human_profile": {
                "mode": "perdonor",
                "M": 1,
                "sparsity_caveat": (
                    "M=1 means any motif appearing in even one donor's leading edge "
                    "is included.  The support_distribution.csv and per-kinase "
                    "human_support_min/median columns quantify human-side sparsity."
                ),
                "n_kinases_with_profile": len(kinases) - n_empty_human,
                "n_kinases_empty": n_empty_human,
                "support_stats": {
                    "n_motifs_total": len(all_supports),
                    "n_singleton": int(sum(1 for s in all_supports if s == 1)),
                    "min": int(min(all_supports)) if all_supports else None,
                    "max": int(max(all_supports)) if all_supports else None,
                    "median": float(np.median(all_supports)) if all_supports else None,
                },
            },
            "mouse_contexts": [_context_label(t, a) for t, a in all_contexts()],
            "headline_context": "cortex_12mo",
            "contexts_run": [_context_label(t, a) for t, a in contexts],
            "missing_kinases_per_ctx": missing_kinases_per_ctx,
        }
        manifest_path = out_path / "manifest.json"
        with manifest_path.open("w") as fh:
            json.dump(manifest, fh, indent=2)
        if verbose:
            print(f"[C5]   manifest.json written", flush=True)

    # ── Peak RSS ──────────────────────────────────────────────────────────────
    rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss_mb = rss_kb / 1024
    if verbose:
        print(f"[C5] Done.  Peak RSS: {rss_mb:.1f} MB", flush=True)

    return out_path


# ─── Output helpers ───────────────────────────────────────────────────────────

def _emit_summary_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def _emit_pairs_csv(
    ctx_overlaps: list,
    path: Path,
    mouse_contrast: str,
) -> None:
    """Per-context substrate detail with gene-identity partition.

    Each row carries a substrate from one side (or both for shared sites).
    partition column: shared_site | shared_gene_diffsite |
                      human_only_engaged | human_only_unmeasured |
                      mouse_only_engaged | mouse_only_unmeasured
    coverage column: engaged | unmeasured (unique-side rows only, blank for shared)
    """
    header = [
        "kinase", "cohort_a", "contrast_a", "cohort_b", "contrast_b",
        "gene_a", "site_a", "motif_a", "gene_b", "site_b", "motif_b",
        "similarity", "partition", "coverage",
        "direction_a", "direction_b", "direction_agree",
        "support_a", "support_b",
    ]
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for kinase, ov, _sim, _hp, _mp in ctx_overlaps:
            ca = "b-donorset_perdonor_M1"
            cb = mouse_contrast
            for s in ov.shared_sites:
                w.writerow([
                    kinase, "mukesh", ca, "fivexfad", cb,
                    s.gene, s.site_a, s.motif_a,
                    s.gene, s.site_b, s.motif_b,
                    f"{s.similarity:.6f}", "shared_site", "",
                    s.direction_a, s.direction_b, int(s.direction_agree),
                    s.support_a, s.support_b,
                ])
            for s in ov.diff_site:
                if s.motif_a:
                    # human side entry — mouse motif blank
                    w.writerow([
                        kinase, "mukesh", ca, "fivexfad", cb,
                        s.gene, s.site_a, s.motif_a,
                        s.gene, "", "",
                        "", "shared_gene_diffsite", "",
                        s.direction_a, "", "",
                        s.support_a, "",
                    ])
                else:
                    # mouse side entry — human motif blank
                    w.writerow([
                        kinase, "mukesh", ca, "fivexfad", cb,
                        s.gene, "", "",
                        s.gene, s.site_b, s.motif_b,
                        "", "shared_gene_diffsite", "",
                        "", s.direction_b, "",
                        "", s.support_b,
                    ])
            for u in ov.human_only:
                w.writerow([
                    kinase, "mukesh", ca, "fivexfad", cb,
                    u.gene, u.site, u.motif,
                    "", "", "",
                    "", f"human_only_{u.coverage}", u.coverage,
                    u.direction, "", "",
                    u.support, "",
                ])
            for u in ov.mouse_only:
                w.writerow([
                    kinase, "mukesh", ca, "fivexfad", cb,
                    "", "", "",
                    u.gene, u.site, u.motif,
                    "", f"mouse_only_{u.coverage}", u.coverage,
                    "", u.direction, "",
                    "", u.support,
                ])


def _emit_support_distribution(
    human_profiles: dict[str, Profile],
    path: Path,
) -> None:
    """Emit per-(kinase, motif) support count for the honesty guard."""
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["kinase", "motif", "gene", "support"])
        for kinase in sorted(human_profiles):
            for motif, e in sorted(human_profiles[kinase].items()):
                w.writerow([kinase, motif, e.gene, e.support])


# ─── CLI ─────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="C5 — Mukesh b-donorset substrate profile vs 5xFAD (ST, 8 contexts)"
    )
    ap.add_argument(
        "--context",
        metavar="TISSUE_AGE",
        default=None,
        help=(
            "Limit to one context, e.g. cortex_12mo.  "
            "Omit to run all 8 contexts (gate-compute)."
        ),
    )
    ap.add_argument(
        "--out-dir",
        metavar="PATH",
        default=None,
        help="Override the auto-generated timestamped output directory.",
    )
    ap.add_argument(
        "--no-manifest",
        action="store_true",
        default=False,
        help="Skip manifest.json emit.",
    )
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    try:
        out = run(
            context_filter=args.context,
            out_dir=args.out_dir,
            emit_manifest=not args.no_manifest,
        )
        print(f"[C5] Output directory: {out}", flush=True)
    except Exception:
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

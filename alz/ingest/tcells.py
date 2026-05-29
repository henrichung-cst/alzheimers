"""T-cell exhaustion cohort ingest — ForPerseus → Song-shaped artifacts.

Unlike the Mukesh human cohort (peptide-level Spectronaut reports that need a
UniProt canonical/isoform cache to *assign* site positions), the T-cell tables
are the Spectronaut **ForPerseus / site report** representation: already
site-collapsed, with ``PG.Genes``, ``PTM.SiteAA``, ``PTM.SiteLocation`` and
(where present) the ``PTM.FlankingRegion`` motif pre-parsed. So there is no
peptide→protein mapping step.

Three transforms reconcile the raw report to one row per physical site per
timepoint:

1. **Localization gate (IMAC only).** The site report carries per-run
   ``PTM.SiteProbability``; rows whose best-run localization probability is below
   the class-I cutoff (0.75) are ambiguous (the phosphate cannot be pinned to one
   residue — distinct flanking windows, shared peptide quant) and are dropped.
   pY ForPerseus has no probability column, so the gate does not apply there.

2. **Isoform collapse.** The report enumerates a site once per isoform of its
   protein group (e.g. AAAS S462 in ENSP…437 == S495 in ENSP…438 — same residue,
   same measurement, identical flanking window). We re-key sites by their
   physical identity ``(gene, residue, flanking window)`` and collapse, taking the
   **canonical** isoform's position (the protein listed first in
   ``PG.ProteinGroups``); a site present only on a non-canonical isoform is kept
   at that isoform's position and flagged ``isoform_specific``. This mirrors
   Mukesh's "canonical except isoform-specific" policy. Where no window is
   available (donor2 pY) the key falls back to ``(gene, residue, location)``.

3. **Technical-replicate collapse.** Donor 1 was injected twice per timepoint
   (``_DIA``/``_DIA_2``, ``_pY_1``/``_pY_2``, IMAC ``_2``/``_3``); donor 2 once.
   These are technical re-injections of one biological sample, so they are
   averaged to a single column per (donor, day) after per-run median-centering.

Self-normalization (the collaborator's "Normalized" reports are NOT used so both
donors and all tracks share one basis): linear → log2 (empty/≤0 → NaN; the
ForPerseus ``1`` detection floor kept as log2=0 but counted) → per-run
median-center → rep-average.

Tracks per donor:
  * donor1: Total proteome, pY, IMAC (ST track)   — suffix ""=IMAC/ST, "_pY"=pY
  * donor2: Total proteome, pY    (NO IMAC → no ST track; kinase MEA skipped)

Outputs (mirroring the Mukesh split):
  outputs/reports/kinase_attribution_tcells/donor{1,2}/
      stoichiometry_matrix{,_pY}.csv     log2(phospho_centered) − log2(total_centered[gene])
      raw_phospho_normalized{,_pY}.csv   per-sample median-centered log2 phospho
      total_proteome_normalized.csv      per-sample median-centered log2 protein (per gene)
  outputs/reports/data_ingest_tcells/donor{1,2}/
      sample_mapping.csv                 sample_id, donor, day, baseline
      ingest_manifest.json

Usage:  python alz/ingest/tcells.py --reshape
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from alz.shared import config

TCELLS_DATA_DIR = os.path.join(config.REPO_ROOT, "data", "datasets", "tcells")
KINASE_DIR = os.path.join(config.REPO_ROOT, "outputs", "reports", "kinase_attribution_tcells")
INGEST_DIR = os.path.join(config.REPO_ROOT, "outputs", "reports", "data_ingest_tcells")
INCYTR_INPUT_DIR = os.path.join(config.REPO_ROOT, "data", "derived", "tcells_incytr_inputs")

PHOSPHO_STY = "Phospho (STY)"
SITE_PROB_MIN = 0.75  # class-I localization cutoff (IMAC PTM.SiteProbability)
META_COLS = ["site_id", "protein_id", "gene_symbol", "site_position", "motif"]

SOURCES = {
    "donor1": {
        "total": "donor1/proteomics/10Feb2026_Donor1_TotalProteome_ForPerseus.txt",
        "pY": "donor1/proteomics/10Feb2026_Donor1_pY_ForPerseus.txt",
        "imac": "donor1/proteomics/18May2026_TCellDonor1_Normalized_IMACSiteReporttsv.tsv",
    },
    "donor2": {
        "total": "donor2/proteomics/10Feb2026_Donor2_TotalProteome_ForPerseus.txt",
        "pY": "donor2/proteomics/10Feb2026_Donor2_pY_ForPerseus.txt",
    },
}

# --- column parsing -------------------------------------------------------

_DAY_RE = (
    re.compile(r"[Dd][Aa][Yy]\s*(\d+)"),  # "Day 13", "DAY13"
    re.compile(r"_D(\d+)_"),              # donor2 pY: "..._D2_DIA_pY..."
)
_LEADING_REP_RE = re.compile(r"^\s*(\d+)\.\s")   # donor1: "1. Day 2 Total Quantity"
_DIA_REP_RE = re.compile(r"_DIA_(\d+)")          # IMAC: "..._DIA_2_S.raw.PTM.Quantity"


def _parse_day(col: str) -> int | None:
    for rx in _DAY_RE:
        m = rx.search(col)
        if m:
            return int(m.group(1))
    return None


def _parse_rep_token(col: str) -> int:
    m = _LEADING_REP_RE.search(col)
    if m:
        return int(m.group(1))
    m = _DIA_REP_RE.search(col)
    if m:
        return int(m.group(1))
    return 1


def _run_columns(columns: list[str], keyword: str) -> dict[str, int]:
    """Map each column containing ``keyword`` (with a parseable day) → day int."""
    out: dict[str, int] = {}
    for c in columns:
        if keyword not in c:
            continue
        day = _parse_day(c)
        if day is None:
            continue
        out[c] = day
    return out


def _donor_prefix(donor: str) -> str:
    return "D1" if donor == "donor1" else "D2"


# --- normalization --------------------------------------------------------


def _log2_center_collapse(
    linear: pd.DataFrame, col_day: dict[str, int], donor: str
) -> tuple[pd.DataFrame, int, int]:
    """log2 → per-run median-center → average technical reps within a day.

    ``linear`` columns are individual MS runs; ``col_day`` maps each to its day.
    Returns (per-day centered matrix, n_value_cells, n_floor_cells).
    """
    vals = linear.where(linear > 0)
    n_cells = int(vals.notna().sum().sum())
    n_floor = int((linear == 1.0).sum().sum())
    log2 = np.log2(vals)
    centered = log2.sub(log2.median(axis=0, skipna=True), axis=1)  # per-run center
    centered.columns = [col_day[c] for c in centered.columns]      # → day ints (dup per rep)
    collapsed = centered.T.groupby(level=0).mean().T               # average reps within day
    pref = _donor_prefix(donor)
    collapsed.columns = [f"{pref}_d{int(d)}" for d in collapsed.columns]
    return collapsed, n_cells, n_floor


def _linear_bulk_collapse(
    linear: pd.DataFrame, col_day: dict[str, int], donor: str
) -> pd.DataFrame:
    """Per-day LINEAR bulk for the multiplicative `P_c` deconvolution.

    The D2 log2-centered matrices cannot feed `P_c = (N_total/N_c)×bulk×share`
    (which needs positive linear intensities). Here we apply the SAME run-level
    loading normalization as D2 (per-run median-center in log2) but **re-anchor**
    to the mean of the per-run log2-medians before exponentiating — so the bulk
    stays in natural MS-intensity magnitude (the `pmax(pr,1)` Incytr floor would
    otherwise clobber a median-1 scale) — then averages technical reps within a
    day in linear space. Undetected (gene,day) cells stay NaN (honestly missing).
    """
    vals = linear.where(linear > 0)
    log2 = np.log2(vals)
    run_med = log2.median(axis=0, skipna=True)
    anchor = float(run_med.mean())
    norm = log2.sub(run_med, axis=1) + anchor           # loading-equalized, scale kept
    lin = np.power(2.0, norm)
    lin.columns = [col_day[c] for c in lin.columns]
    bulk = lin.T.groupby(level=0).mean().T              # rep-average per day (skipna)
    pref = _donor_prefix(donor)
    bulk.columns = [f"{pref}_d{int(d)}" for d in bulk.columns]
    return bulk


# --- per-track readers ----------------------------------------------------


def _parse_total(donor: str, path: str) -> tuple[pd.DataFrame, dict[str, int]]:
    """Parse the Total-proteome ForPerseus to a per-run LINEAR matrix indexed by
    gene (duplicate genes mean-collapsed). Shared by the D2 reshape and the
    decomposition linear-bulk emitter — only the downstream collapse differs."""
    df = pd.read_csv(path, sep="\t", dtype=str)
    col_day = _run_columns(list(df.columns), "Quantity")
    gene = df["PG.Genes"].fillna("").str.split(";").str[0].str.strip()
    linear = df[list(col_day)].apply(pd.to_numeric, errors="coerce")
    linear.index = gene.values
    linear = linear[gene.values != ""]
    linear = linear.groupby(level=0).mean()  # collapse duplicate genes (rare)
    return linear, col_day


def _read_total(donor: str, path: str) -> tuple[pd.DataFrame, dict, list[str]]:
    linear, col_day = _parse_total(donor, path)
    collapsed, n_cells, n_floor = _log2_center_collapse(linear, col_day, donor)
    samples = list(collapsed.columns)
    meta = {"n_genes": int(len(collapsed)), "n_cells": n_cells, "n_floor": n_floor,
            "samples": samples}
    return collapsed, meta, samples


def _isoform_collapse(
    df: pd.DataFrame, col_day: dict[str, int], gene: pd.Series, aa: pd.Series,
    pos_int: pd.Series, flank: pd.Series,
) -> tuple[pd.DataFrame, pd.Series, int]:
    """Canonical-except-isoform-specific collapse using PTM.ProteinId (IMAC).

    The report enumerates each physical site once per isoform of its protein
    group (identical measurement, position numbered per isoform; flanking can
    differ near splice junctions). Canonical = the protein listed first in
    ``PG.ProteinGroups``. We keep every canonical-isoform site, then add back a
    non-canonical site only when its measurement (raw value vector) is not
    already present on the canonical isoform for that gene — i.e. it is genuinely
    isoform-specific. This never merges distinct sites of a multiply-phosphorylated
    peptide (they are both canonical rows, kept) and never double-counts the same
    measurement across isoforms.

    Returns (kept df reindexed, isoform_specific bool Series, n_isoform_specific).
    """
    protein_id = df["PTM.ProteinId"].fillna("").str.strip().values
    first_acc = df["PG.ProteinGroups"].fillna("").str.split(";").str[0].str.strip().values
    vhash = df[list(col_day)].fillna("").agg("|".join, axis=1).values
    canon = protein_id == first_acc

    canon_keys = set(zip(gene.values[canon], vhash[canon]))
    keys = list(zip(gene.values, vhash))
    redundant = np.array([(not canon[i]) and (keys[i] in canon_keys) for i in range(len(df))])
    keep = ~redundant

    sub = df[keep].reset_index(drop=True)
    g, fl, vh, cn = gene.values[keep], flank.values[keep], vhash[keep], canon[keep]
    # dedup isoform-specific copies shared across several non-canonical isoforms
    # (same gene+window+measurement) — keep first; window separates multi-phospho.
    dedup = pd.DataFrame({"g": g, "fl": fl, "vh": vh, "isf": ~cn})
    drop = (dedup.duplicated(["g", "fl", "vh"]) & dedup["isf"]).values
    sub = sub[~drop].reset_index(drop=True)
    iso_specific = pd.Series(~cn[~drop], index=sub.index)
    return sub, iso_specific, int(iso_specific.sum())


def _parse_phospho(
    donor: str, track: str, path: str
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int], dict]:
    """Parse a phospho ForPerseus (pY or IMAC) to a per-run LINEAR matrix indexed
    by site_id, plus full per-site meta and parse-stage stats. Applies the
    localization gate and isoform/window collapse. Shared by the D2 reshape and
    the decomposition linear-bulk emitter — only the downstream collapse differs.
    """
    df = pd.read_csv(path, sep="\t", dtype=str)
    cols = list(df.columns)
    col_day = _run_columns(cols, "Quantity")
    has_iso = "PTM.ProteinId" in df and "PG.ProteinGroups" in df
    sp_cols = [c for c in cols if "SiteProbability" in c]

    mod = df.get("PTM.ModificationTitle")
    n_raw = len(df)
    if mod is not None:
        df = df[mod == PHOSPHO_STY].copy()
    n_phospho = len(df)

    # localization gate (IMAC only; the 18May report is already ≥0.75-filtered
    # upstream, so this is a defensive no-op that records the cutoff).
    n_gated = 0
    if sp_cols:
        sp = df[sp_cols].apply(pd.to_numeric, errors="coerce").max(axis=1)
        keep = sp >= SITE_PROB_MIN
        n_gated = int((~keep).sum())
        df = df[keep].copy()

    df = df.reset_index(drop=True)
    gene = df["PG.Genes"].fillna("").str.split(";").str[0].str.strip()
    aa = df["PTM.SiteAA"].fillna("").str.strip()
    loc = df["PTM.SiteLocation"].fillna("").str.split(";").str[0].str.strip()
    flank = (df["PTM.FlankingRegion"].fillna("").str.strip()
             if "PTM.FlankingRegion" in df else pd.Series("", index=df.index))
    pos_int = pd.to_numeric(loc, errors="coerce")
    valid = (gene.values != "") & pos_int.notna().values
    df = df[valid].reset_index(drop=True)
    gene, aa, loc, flank = gene[valid].reset_index(drop=True), aa[valid].reset_index(drop=True), \
        loc[valid].reset_index(drop=True), flank[valid].reset_index(drop=True)
    pos_int = pos_int[valid].astype(int).reset_index(drop=True)
    n_in = len(df)

    if has_iso:
        # canonical-except-isoform-specific: keep distinct rows, no value merge
        df, iso_specific, n_iso = _isoform_collapse(df, col_day, gene, aa, pos_int, flank)
        gene = df["PG.Genes"].fillna("").str.split(";").str[0].str.strip()
        aa = df["PTM.SiteAA"].fillna("").str.strip()
        pos = pd.to_numeric(df["PTM.SiteLocation"].fillna("").str.split(";").str[0], errors="coerce").astype(int)
        flank = df["PTM.FlankingRegion"].fillna("").str.strip()
        linear = df[list(col_day)].apply(pd.to_numeric, errors="coerce")
    else:
        # pY: no isoform info → collapse same-window copies (median); window
        # separates distinct sites of a multiply-phosphorylated peptide.
        win = flank.where(flank != "", loc)
        phys = (gene + "|" + aa + "|" + win).values
        linear = df[list(col_day)].apply(pd.to_numeric, errors="coerce")
        linear.index = phys
        linear = linear.groupby(level=0).median()
        meta_first = pd.DataFrame({"phys": phys, "gene": gene.values, "aa": aa.values,
                                   "motif": flank.values, "pos": pos_int.values}) \
            .groupby("phys").agg({"gene": "first", "aa": "first", "motif": "first", "pos": "min"})
        meta_first = meta_first.reindex(linear.index)
        gene, aa, pos, flank = meta_first["gene"], meta_first["aa"], meta_first["pos"].astype(int), meta_first["motif"]
        iso_specific, n_iso = pd.Series(pd.NA, index=linear.index), None

    site_pos = (aa.astype(str) + pos.astype(str)).values
    site_id = (gene.astype(str) + "_" + site_pos).values
    linear.index = site_id
    meta = pd.DataFrame({
        "site_id": site_id, "protein_id": gene.values, "gene_symbol": gene.values,
        "site_position": site_pos, "motif": np.asarray(flank),
        "isoform_specific": np.asarray(iso_specific),
    })
    # final uniqueness guard on site_id (keeps first metadata + intensity row)
    dup = meta["site_id"].duplicated().values
    meta, linear = meta[~dup].reset_index(drop=True), linear[~dup]
    stats = {
        "rows_raw": n_raw, "rows_phospho_sty": n_phospho,
        "low_localization_rows_dropped": n_gated, "siteprob_gated": bool(sp_cols),
        "rows_in": n_in, "sites_isoform_specific": n_iso,
    }
    return linear, meta, col_day, stats


def _read_phospho(
    donor: str, track: str, path: str
) -> tuple[pd.DataFrame, pd.DataFrame, dict, list[str]]:
    linear, meta, col_day, stats = _parse_phospho(donor, track, path)
    collapsed, n_cells, n_floor = _log2_center_collapse(linear, col_day, donor)
    n_motif = int((meta["motif"].fillna("") != "").sum())
    stats.update({
        "sites": int(len(collapsed)),
        "sites_with_motif": n_motif, "sites_without_motif": int(len(collapsed) - n_motif),
        "cells": n_cells, "floor_cells": n_floor,
        "samples": list(collapsed.columns),
    })
    return collapsed, meta[META_COLS], stats, list(collapsed.columns)


# --- emission -------------------------------------------------------------


def _stoichiometry(phospho: pd.DataFrame, meta: pd.DataFrame, total: pd.DataFrame) -> pd.DataFrame:
    samples = [c for c in phospho.columns if c in total.columns]
    parent = total.reindex(meta["gene_symbol"].values)[samples]
    parent.index = phospho.index
    stoich = phospho[samples] - parent.values
    stoich = stoich.sub(stoich.median(axis=0, skipna=True), axis=1)
    return stoich


def _write_matrix(path: str, meta: pd.DataFrame, values: pd.DataFrame) -> int:
    out = meta[META_COLS].reset_index(drop=True).copy()
    for s in values.columns:
        out[s] = values[s].values
    os.makedirs(os.path.dirname(path), exist_ok=True)
    out.to_csv(path, index=False)
    return len(out)


def _file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _reshape_donor(donor: str) -> dict:
    src = SOURCES[donor]
    kdir = os.path.join(KINASE_DIR, donor)
    idir = os.path.join(INGEST_DIR, donor)
    os.makedirs(kdir, exist_ok=True)
    os.makedirs(idir, exist_ok=True)
    print(f"\n=== reshape {donor} ===")

    total, total_meta, total_samples = _read_total(donor, os.path.join(TCELLS_DATA_DIR, src["total"]))
    total_df = total.reset_index()
    total_df.columns = ["gene_symbol"] + list(total.columns)
    total_df.to_csv(os.path.join(kdir, "total_proteome_normalized.csv"), index=False)
    print(f"  total: {total_meta['n_genes']} genes × {len(total.columns)} samples "
          f"({total_meta['n_floor']} floor cells)")

    track_stats = {"total": total_meta}
    all_samples: set[str] = set(total_samples)

    tracks = [("pY", "_pY", "pY")]
    if "imac" in src:
        tracks.append(("IMAC", "", "imac"))

    for tname, suffix, skey in tracks:
        phospho, meta, stats, samples = _read_phospho(donor, tname, os.path.join(TCELLS_DATA_DIR, src[skey]))
        all_samples |= set(samples)
        n_ph = _write_matrix(os.path.join(kdir, f"raw_phospho_normalized{suffix}.csv"), meta, phospho)
        stoich = _stoichiometry(phospho, meta, total)
        n_st = _write_matrix(os.path.join(kdir, f"stoichiometry_matrix{suffix}.csv"), meta, stoich)
        stats["stoich_sites"] = n_st
        track_stats[tname] = stats
        iso = stats["sites_isoform_specific"]
        print(f"  {tname}: {stats['rows_in']} rows → {stats['sites']} sites "
              f"(gated {stats['low_localization_rows_dropped']}, "
              f"isoform-specific {iso if iso is not None else 'n/a'}, "
              f"{stats['sites_without_motif']} w/o motif), "
              f"phospho={n_ph} stoich={n_st}, samples={len(samples)}")

    rows = []
    for sid in sorted(all_samples, key=lambda s: int(s.split("_d")[1])):
        day = int(sid.split("_d")[1])
        rows.append({"sample_id": sid, "donor": donor, "day": day, "baseline": day == 2})
    pd.DataFrame(rows).to_csv(os.path.join(idir, "sample_mapping.csv"), index=False)

    manifest = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "donor": donor,
        "site_prob_min": SITE_PROB_MIN,
        "inputs": {
            k: {"path": os.path.relpath(os.path.join(TCELLS_DATA_DIR, v), config.REPO_ROOT),
                "sha256": _file_sha256(os.path.join(TCELLS_DATA_DIR, v))}
            for k, v in src.items()
        },
        "has_imac": "imac" in src,
        "n_samples": len(all_samples),
        "tracks": track_stats,
    }
    with open(os.path.join(idir, "ingest_manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    return manifest


def run_reshape() -> None:
    for donor in ("donor1", "donor2"):
        _reshape_donor(donor)
    print("\n[tcells] reshape done.")


# --- linear bulk for decomposition (D4) -----------------------------------


def _export_bulk_donor(donor: str) -> dict:
    """Emit per-day LINEAR bulk per channel for the `P_c` decomposition:
        pr_bulk_linear.csv  (gene_symbol + day cols)        — Total proteome
        py_bulk_linear.csv  (site_id, gene_symbol, motif + day cols) — pY
        ps_bulk_linear.csv  (site_id, gene_symbol, motif + day cols) — IMAC (donor1)
    under data/derived/tcells_incytr_inputs/<donor>/.
    """
    src = SOURCES[donor]
    outdir = os.path.join(INCYTR_INPUT_DIR, donor)
    os.makedirs(outdir, exist_ok=True)
    print(f"\n=== export linear bulk {donor} ===")
    summary: dict = {}

    lin_t, cd_t = _parse_total(donor, os.path.join(TCELLS_DATA_DIR, src["total"]))
    pr = _linear_bulk_collapse(lin_t, cd_t, donor)
    pr_out = pr.reset_index()
    pr_out.columns = ["gene_symbol"] + list(pr.columns)
    pr_out.to_csv(os.path.join(outdir, "pr_bulk_linear.csv"), index=False)
    summary["pr"] = {"genes": int(len(pr)), "days": list(pr.columns)}
    print(f"  pr: {len(pr)} genes × {len(pr.columns)} days")

    channels = [("pY", "pY", "py", "py_bulk_linear.csv")]
    if "imac" in src:
        channels.append(("IMAC", "imac", "ps", "ps_bulk_linear.csv"))
    for tname, skey, ch, fname in channels:
        lin_p, meta, cd_p, _ = _parse_phospho(donor, tname,
                                              os.path.join(TCELLS_DATA_DIR, src[skey]))
        bulk = _linear_bulk_collapse(lin_p, cd_p, donor)
        meta = meta.set_index("site_id").reindex(bulk.index)
        out = pd.DataFrame({"site_id": bulk.index,
                            "gene_symbol": meta["gene_symbol"].values,
                            "motif": meta["motif"].values})
        for c in bulk.columns:
            out[c] = bulk[c].values
        out.to_csv(os.path.join(outdir, fname), index=False)
        summary[ch] = {"sites": int(len(bulk)), "days": list(bulk.columns)}
        print(f"  {ch}: {len(bulk)} sites × {len(bulk.columns)} days")
    return summary


def run_export_bulk() -> None:
    for donor in ("donor1", "donor2"):
        _export_bulk_donor(donor)
    print("\n[tcells] linear bulk export done.")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reshape", action="store_true", help="ForPerseus → Song-shaped artifacts")
    p.add_argument("--export-bulk", action="store_true",
                   help="per-day LINEAR bulk per channel for the P_c decomposition")
    args = p.parse_args(argv)
    if args.reshape:
        run_reshape()
        return 0
    if args.export_bulk:
        run_export_bulk()
        return 0
    p.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())

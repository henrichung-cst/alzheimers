"""Boolean overlap: AD ∩ suspect agreement vs clean opposition, AD per-donor set excl. AD-01/AD-03.

Single, self-contained generator. Reads ONLY inputs (no intermediate report artifacts):
  - human stoichiometry matrices  outputs/reports/kinase_attribution_human/stoichiometry_matrix{,_pY}.csv
  - sample group labels           outputs/reports/data_ingest_human/sample_mapping.csv  (via mukesh)
  - per-donor kinase NES / FDR    outputs/reports/kinase_attribution_human/perdonor/kinase_donor_{nes,fdr}{,_pY}.csv

and writes a single result folder
  outputs/reports/kinase_attribution_human/ctrl_audit/concordance_AD8_excl01_03/
    overlap_AD8_sus_clean.csv     the kinase set + per-group agreement counts
    substrates_leading_edge.csv   the GSEA leading-edge substrate sites behind each overlap kinase
    MANIFEST.md                   exact sample membership of every role

The result is the base-gated boolean overlap (formerly the `ad_all__sus_all__clean_all_opp`
cell of a 4×4×4 sweep — the other 63 cells were never used and are not produced):

  base gate : AD-vs-CLEAN group FDR<0.25 AND suspect-vs-CLEAN group FDR<0.25, sharing direction
  AD        : all 8 AD donors (AD-01, AD-03 excluded) share the group direction
  suspect   : all suspect controls (CTRL-08, CTRL-10) share the group direction
  clean     : all 4 clean controls (CTRL-01..04) are opposite

PROVENANCE RULE (see ../README in the output folder): the group MEA is run over ALL 10 AD
samples — that defines the reference direction and the significance base gate. AD-01/AD-03 are
excluded ONLY from the per-donor agreement vote. The sample membership is stamped into the
MANIFEST and per row, so a result can never be read without knowing which samples produced it.

Suspect pool = CTRL-08, CTRL-10. CTRL-07 is EXCLUDED ENTIRELY. CLEAN is ALWAYS the four clean
controls — it is NOT "all CTRL minus suspect" — so dropping CTRL-07 from the suspect pool does
NOT move it into the clean baseline (the audit verdict is that CTRL-07 is genuinely AD-like);
CTRL-07 lands in neither pool and the MANIFEST stamps it as EXCLUDED.

Shared leading-edge substrates. For each overlap kinase we emit the GSEA leading-edge substrate
sites it shares in BOTH group contrasts — the substrate motifs at/before the running-sum peak in
the AD-vs-clean AND the suspect-vs-clean enrichment, i.e. the substrates concordantly driving the
kinase in AD and in the suspect controls. Each shared site carries its AD and suspect LFC.
Substrate identity is the motif (±7 aa window); one motif can map to several phosphosites (same
context, different proteins), and every matching site is emitted.
"""
from __future__ import annotations

import csv
import datetime as _dt
from pathlib import Path
from statistics import median

import pandas as pd

from alz.shared import config
from alz.bulk_mea import enrich as kinase_enrich
from alz.ingest.mukesh import SAMPLE_MAPPING_CSV
from alz.ingest.mukesh_perdonor import _load_track_matrix, PERDONOR_DIR

# Audit verdict (2026-05-25): CTRL-07/08/10 carry a genuine AD-like phospho signature. The
# suspect pool for THIS analysis is CTRL-08/CTRL-10 only (CTRL-07 is excluded entirely — see
# SUSPECT below). The four clean controls define the baseline and never include any suspect.
CLEAN_CTRL = {"CTRL-01", "CTRL-02", "CTRL-03", "CTRL-04"}
EXCLUDED_AD = {"AD-01", "AD-03"}

# Suspect pool = CTRL-08, CTRL-10. CTRL-07 is excluded ENTIRELY: CLEAN is fixed at CLEAN_CTRL, so
# CTRL-07 enters neither the suspect nor the clean pool (the MANIFEST stamps it as EXCLUDED).
OUT_NAME = "concordance_AD8_excl01_03"
SUSPECT = {"CTRL-08", "CTRL-10"}

CTRL_AUDIT_DIR = Path("outputs/reports/kinase_attribution_human/ctrl_audit")
TRACKS = ("st", "py")
FDR_THRESH = config.MEA_FDR_THRESH  # 0.25
LFC_KEY = "stoich_lfc"

AD_CONTRAST = "AD_vs_cleanCTRL"
SUSPECT_CONTRAST = "suspect_vs_cleanCTRL"

OVERLAP_HEADER = [
    "kinase", "track", "residue", "direction",
    "group_nes_AD", "group_nes_suspect", "clean_median_nes",
    "group_fdr_AD", "group_fdr_suspect",
    "n_AD8_same_direction", "n_suspect_same_direction", "n_clean_opposite_direction",
]

# Per-site substrate table: the leading-edge sites SHARED by a kinase in BOTH group contrasts
# (the substrates concordantly driving it in AD and in the suspect controls). Each shared site
# carries its AD and suspect LFC; `kl_percentile` (motif-vs-kinase match strength) is contrast-
# independent, so it is single. `motif` marks the phospho-residue in lowercase (kinase-library
# convention), so the acceptor site is explicit (e.g. IRANRADsEEEGTVE).
SUBSTRATE_HEADER = [
    "kinase", "track", "residue",
    "site_id", "gene_symbol", "site_position", "motif", "lfc_AD", "lfc_suspect", "kl_percentile",
]


def _sample_sets(matrix_cols, suspect_set: set[str]) -> dict[str, list[str]]:
    """AD (10, group), AD8 (per-donor vote), SUSPECT, CLEAN from the mapping.

    CLEAN = CLEAN_CTRL ∩ cols (fixed clean baseline). SUSPECT = suspect_set ∩ cols. A control
    in neither set (CTRL-07, given the CTRL-08/10 suspect pool) is dropped from the analysis entirely.
    """
    m = pd.read_csv(SAMPLE_MAPPING_CSV)
    cols = set(matrix_cols)
    ad = sorted(s for s in m.loc[m.group == "AD", "sample_id"] if s in cols)
    allctrl = sorted(s for s in m.loc[m.group == "CTRL", "sample_id"] if s in cols)
    return {
        "AD": ad,
        "AD8": [s for s in ad if s not in EXCLUDED_AD],
        "SUSPECT": [s for s in allctrl if s in suspect_set],
        "CLEAN": [s for s in allctrl if s in CLEAN_CTRL],
        "EXCLUDED_CTRL": [s for s in allctrl if s not in suspect_set and s not in CLEAN_CTRL],
    }


def _group_contrast(track: str, group_a: list[str], group_b: list[str], label: str) -> dict | None:
    """Group GSEA on the per-site mean(A)-mean(B) stoichiometry LFC — identical method to
    production human_group_mea. Returns a context with everything overlap + substrate emission
    need, computed in a single _run_mea call:

      mea       kinase / residue_type / NES / FDR / "Leading substrates" (per-kinase leading edge)
      substrate kinase / motif / kl_percentile  (full substrate set; leading edge is a subset)
      lfc       Series  site_id -> contrast LFC
      meta      DataFrame  site_id / gene_symbol / site_position / motif
    """
    matrix = _load_track_matrix(track, "stoich")
    if matrix is None:
        return None
    X = matrix.set_index("site_id")
    lfc = X[group_a].astype(float).mean(axis=1) - X[group_b].astype(float).mean(axis=1)
    mea_df, _shift_df, _outlier_df, substrate_df = kinase_enrich._run_mea(
        motif_series=matrix["motif"],
        results_by_contrast={label: {LFC_KEY: lfc.values}},
        lfc_key=LFC_KEY,
        site_ids=matrix["site_id"].values,
        gene_symbols=matrix["gene_symbol"].values,
        track=track,
    )
    if mea_df.empty:
        return None
    meta = matrix[["site_id", "gene_symbol", "site_position", "motif"]].copy()
    return {
        "label": label,
        "mea": mea_df,
        "substrate": substrate_df,
        "lfc": lfc,
        "meta": meta,
    }


def _per_donor(track: str) -> dict[str, dict[str, float]]:
    """Per-donor {kinase: {sample: nes}}."""
    suffix = "" if track == "st" else "_pY"
    out: dict[str, dict[str, float]] = {}
    with (Path(PERDONOR_DIR) / f"kinase_donor_nes{suffix}.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            kinase = row.pop("kinase")
            out[kinase] = {
                col: (float(v) if v not in ("", "NA") else float("nan"))
                for col, v in row.items()
            }
    return out


def _sign(x: float) -> int:
    return 1 if x > 0 else -1 if x < 0 else 0


def _all_same(nes: dict, kinase: str, samples: list[str], want: int) -> tuple[bool, int]:
    """(all finite donor NES equal `want` sign, count matching)."""
    vals = [nes.get(kinase, {}).get(s, float("nan")) for s in samples]
    finite = [v for v in vals if v == v]  # drop NaN
    n_match = sum(_sign(v) == want for v in finite)
    return (len(finite) == len(samples) and n_match == len(samples)), n_match


def compute_overlap(ctx_by_track: dict, sets: dict[str, list[str]]) -> list[dict]:
    members: list[dict] = []
    for track in TRACKS:
        ad_ctx, sus_ctx = ctx_by_track[track]
        if ad_ctx is None or sus_ctx is None:
            continue
        cols = ["kinase", "residue_type", "NES", "FDR"]
        grp = ad_ctx["mea"][cols].merge(
            sus_ctx["mea"][cols], on=["kinase", "residue_type"], suffixes=("_AD", "_suspect")
        )
        nes_donor = _per_donor(track)

        for _, r in grp.iterrows():
            nes_ad, nes_sus = float(r["NES_AD"]), float(r["NES_suspect"])
            fdr_ad, fdr_sus = float(r["FDR_AD"]), float(r["FDR_suspect"])
            # base gate: both group-significant, sharing direction
            if not (fdr_ad < FDR_THRESH and fdr_sus < FDR_THRESH and nes_ad * nes_sus > 0):
                continue
            d = _sign(nes_ad)
            ad_ok, n_ad = _all_same(nes_donor, r["kinase"], sets["AD8"], d)
            sus_ok, n_sus = _all_same(nes_donor, r["kinase"], sets["SUSPECT"], d)
            clean_ok, n_clean = _all_same(nes_donor, r["kinase"], sets["CLEAN"], -d)
            if not (ad_ok and sus_ok and clean_ok):
                continue
            clean_nes = [
                nes_donor.get(r["kinase"], {}).get(s, float("nan")) for s in sets["CLEAN"]
            ]
            clean_nes = [v for v in clean_nes if v == v]
            members.append({
                "kinase": r["kinase"], "track": track, "residue": r["residue_type"],
                "direction": "up" if d > 0 else "down",
                "group_nes_AD": nes_ad, "group_nes_suspect": nes_sus,
                "clean_median_nes": median(clean_nes) if clean_nes else float("nan"),
                "group_fdr_AD": fdr_ad, "group_fdr_suspect": fdr_sus,
                "n_AD8_same_direction": n_ad,
                "n_suspect_same_direction": n_sus,
                "n_clean_opposite_direction": n_clean,
            })
    members.sort(key=lambda m: -m["group_nes_AD"])
    return members


def _leading_motifs(mea_df: pd.DataFrame, kinase: str, residue: str) -> list[str]:
    """The leading-edge motif list for one kinase from the contrast's `Leading substrates`."""
    sel = mea_df[(mea_df["kinase"] == kinase) & (mea_df["residue_type"] == residue)]
    if sel.empty:
        return []
    raw = str(sel.iloc[0].get("Leading substrates", "") or "")
    return [m for m in raw.split(";") if m]


def _mark_motif(motif: str) -> str:
    """Lowercase the central phospho-residue of the ±7 aa window (kinase-library convention)."""
    if not isinstance(motif, str) or not motif:
        return motif
    c = len(motif) // 2
    return motif[:c] + motif[c].lower() + motif[c + 1:]


def _substrate_rows(ctx: dict, kinase: str, track: str, residue: str) -> list[dict]:
    """Leading-edge substrate SITES for one kinase in one contrast.

    Joins the leading-edge motifs back to the matrix metadata (one motif → possibly several
    sites) and attaches the contrast LFC + the kinase-vs-site kl_percentile. The kinase library
    lowercases the central phospho-residue in its motif strings while the matrix `motif` column
    is all-uppercase, so the join key is the uppercased motif.
    """
    motifs = _leading_motifs(ctx["mea"], kinase, residue)
    if not motifs:
        return []
    motif_set = {m.upper() for m in motifs}
    # kl_percentile per uppercased motif for THIS kinase (substrate_df is keyed by kinase+motif).
    sub = ctx["substrate"]
    pct = {
        row["motif"].upper(): float(row["kl_percentile"])
        for _, row in sub[sub["kinase"] == kinase].iterrows()
        if row["motif"].upper() in motif_set
    }
    lfc = ctx["lfc"]
    rows: list[dict] = []
    hits = ctx["meta"][ctx["meta"]["motif"].str.upper().isin(motif_set)]
    for _, s in hits.iterrows():
        site_id = s["site_id"]
        rows.append({
            "kinase": kinase, "track": track, "residue": residue, "contrast": ctx["label"],
            "site_id": site_id, "gene_symbol": s["gene_symbol"],
            "site_position": s["site_position"], "motif": _mark_motif(s["motif"]),
            "lfc": float(lfc.get(site_id, float("nan"))),
            "kl_percentile": pct.get(s["motif"].upper(), float("nan")),
        })
    return rows


def compute_substrates(ctx_by_track: dict, members: list[dict]) -> list[dict]:
    """For each overlap kinase, the leading-edge sites SHARED by both group contrasts.

    A site is kept only if it is in the kinase's leading edge in BOTH the AD-vs-clean AND the
    suspect-vs-clean enrichment — the substrates concordantly driving the kinase in AD and in the
    suspect controls. The shared site carries both its AD and suspect LFC.
    """
    rows: list[dict] = []
    for m in members:
        ad_ctx, sus_ctx = ctx_by_track[m["track"]]
        if ad_ctx is None or sus_ctx is None:
            continue
        kw = (m["kinase"], m["track"], m["residue"])
        ad = {r["site_id"]: r for r in _substrate_rows(ad_ctx, *kw)}
        sus = {r["site_id"]: r for r in _substrate_rows(sus_ctx, *kw)}
        for site_id in ad.keys() & sus.keys():
            a, s = ad[site_id], sus[site_id]
            rows.append({
                "kinase": m["kinase"], "track": m["track"], "residue": m["residue"],
                "site_id": site_id, "gene_symbol": a["gene_symbol"],
                "site_position": a["site_position"], "motif": a["motif"],
                "lfc_AD": a["lfc"], "lfc_suspect": s["lfc"],
                "kl_percentile": a["kl_percentile"],  # contrast-independent (PSSM match strength)
            })
    rows.sort(key=lambda r: (r["kinase"], r["track"], -abs(r["lfc_AD"])))
    return rows


def _fmt(v) -> str:
    return f"{v:.12g}" if isinstance(v, float) else str(v)


def _write_csv(path: Path, header: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for r in rows:
            writer.writerow([_fmt(r[c]) for c in header])


def write_manifest(out_dir: Path, members: list[dict], sets: dict[str, list[str]],
                   n_substrates: int) -> None:
    up = sum(m["direction"] == "up" for m in members)
    down = sum(m["direction"] == "down" for m in members)
    n_st = sum(m["track"] == "st" for m in members)
    n_py = sum(m["track"] == "py" for m in members)
    excluded_ctrl = sets["EXCLUDED_CTRL"]
    lines = [
        "# Boolean overlap — AD (excl. AD-01, AD-03) ∩ suspect ∩ ¬clean — MANIFEST",
        "",
        f"**Generated:** {_dt.date.today().isoformat()}  ",
        "**Generator:** `alz/ctrl_outlier_audit/concordance_overlap_AD_excl_01_03.py` "
        "(`pixi run concordance-overlap-ad8-excl01-03`) — single, self-contained.  ",
        "**Inputs:** `stoichiometry_matrix{,_pY}.csv`, `data_ingest_human/sample_mapping.csv`, "
        "`perdonor/kinase_donor_{nes,fdr}{,_pY}.csv`  ",
        f"**FDR threshold:** {FDR_THRESH}  ",
        "**Sign convention:** `+ = up in disease/suspect direction`.",
        "",
        "## Group membership",
        "",
        "| Group | n | Sample IDs |",
        "|-------|---|-----------|",
        f"| AD — group MEA / direction / base gate | {len(sets['AD'])} | {', '.join(sets['AD'])} |",
        f"| AD — per-donor agreement vote (AD-01, AD-03 excluded) | {len(sets['AD8'])} | "
        f"{', '.join(sets['AD8'])} |",
        f"| SUSPECT | {len(sets['SUSPECT'])} | {', '.join(sets['SUSPECT'])} |",
        f"| CLEAN | {len(sets['CLEAN'])} | {', '.join(sets['CLEAN'])} |",
    ]
    if excluded_ctrl:
        lines.append(
            f"| EXCLUDED (in neither pool) | {len(excluded_ctrl)} | {', '.join(excluded_ctrl)} |"
        )
    lines += [
        "",
        "## Definition",
        "",
        "- **base gate:** AD-vs-CLEAN group `FDR<0.25` AND suspect-vs-CLEAN group `FDR<0.25`, "
        "sharing the group direction (sign of the 10-AD group `NES_AD`)",
        "- **AD:** all 8 AD donors (AD-01, AD-03 excluded) share the group direction",
        f"- **suspect:** all {len(sets['SUSPECT'])} suspect controls share the group direction",
        "- **clean:** all 4 clean controls are opposite",
    ]
    if excluded_ctrl:
        lines.append(
            f"- **{', '.join(excluded_ctrl)} excluded entirely** — CLEAN is fixed at the four clean "
            f"controls, so dropping it from the suspect pool puts it in NEITHER pool (never "
            f"reclassified to clean, per the audit verdict that it is AD-like)."
        )
    lines += [
        "",
        "## Result",
        "",
        f"**{len(members)} kinases** — {up} up, {down} down (st: {n_st}, py: {n_py}). "
        "Full list with per-group agreement counts in `overlap_AD8_sus_clean.csv`.",
        "",
        f"**Shared leading-edge substrates:** {n_substrates} site rows in "
        "`substrates_leading_edge.csv` — the leading-edge sites each overlap kinase shares in BOTH "
        "the AD-vs-clean AND suspect-vs-clean enrichments (the substrates concordantly driving it "
        "in AD and in the suspect controls); each carries its AD and suspect LFC. Substrate "
        "identity is the motif (±7 aa window, phospho-residue lowercased); one motif may map to "
        "several phosphosites, so each is emitted.",
        "",
        "| kinase | track | dir | group NES AD | group NES suspect | clean median NES |",
        "|--------|-------|-----|-------------:|------------------:|-----------------:|",
    ]
    for m in members:
        lines.append(
            f"| {m['kinase']} | {m['track']} | {m['direction']} | "
            f"{m['group_nes_AD']:.3f} | {m['group_nes_suspect']:.3f} | "
            f"{m['clean_median_nes']:.3f} |"
        )
    (out_dir / "MANIFEST.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    matrix = _load_track_matrix("st", "stoich")
    if matrix is None:
        raise RuntimeError("st stoichiometry matrix unavailable")

    out_dir = CTRL_AUDIT_DIR / OUT_NAME
    out_dir.mkdir(parents=True, exist_ok=True)
    sets = _sample_sets(matrix.columns, SUSPECT)

    ctx_by_track = {
        track: (
            _group_contrast(track, sets["AD"], sets["CLEAN"], AD_CONTRAST),
            _group_contrast(track, sets["SUSPECT"], sets["CLEAN"], SUSPECT_CONTRAST),
        )
        for track in TRACKS
    }

    members = compute_overlap(ctx_by_track, sets)
    substrate_rows = compute_substrates(ctx_by_track, members)

    _write_csv(out_dir / "overlap_AD8_sus_clean.csv", OVERLAP_HEADER, members)
    _write_csv(out_dir / "substrates_leading_edge.csv", SUBSTRATE_HEADER, substrate_rows)
    write_manifest(out_dir, members, sets, len(substrate_rows))

    up = sum(m["direction"] == "up" for m in members)
    down = sum(m["direction"] == "down" for m in members)
    print(f"[{OUT_NAME}]")
    print(f"  suspect ({len(sets['SUSPECT'])}): {', '.join(sets['SUSPECT'])}  "
          f"(excluded: {', '.join(sets['EXCLUDED_CTRL']) or 'none'})")
    print(f"  clean ({len(sets['CLEAN'])}): {', '.join(sets['CLEAN'])}")
    print(f"  AD per-donor vote ({len(sets['AD8'])}): {', '.join(sets['AD8'])}")
    print(f"  overlap: {len(members)} kinases (up={up}, down={down})")
    print(f"  shared leading-edge substrate rows: {len(substrate_rows)}")
    print(f"  wrote {out_dir}/")


if __name__ == "__main__":
    main()

"""Mukesh (human NBB per-donor) unified-viewer slice adapter."""

from __future__ import annotations

import json
import os
import shutil

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from alz.shared import config
from alz.viewer.paths import EDGE_SLICES_HUMAN_PERDONOR_DIR, SCHEMA_VERSION
from alz.viewer.shared.cohort_slice import CohortViewerSlice, EdgeSliceContribution

try:
    from alz.cross_reference.human_celltype_attribution import build_celltype_specificity_payload
    _HAS_HUMAN_CELLTYPE = True
except ImportError:
    _HAS_HUMAN_CELLTYPE = False


HUMAN_PERDONOR_DIR = os.path.join(
    config.REPO_ROOT, "outputs", "reports",
    "kinase_attribution_human", "perdonor",
)
HUMAN_TRACK_SUFFIXES = [("", "ST"), ("_pY", "Y")]
_MECHANISM_COLUMNS = [
    "cohort",
    "donor",
    "track",
    "contrast",
    "kinase",
    "stoich_NES",
    "stoich_FDR",
    "raw_NES",
    "raw_FDR",
    "stoich_significant",
    "raw_significant",
    "sign_relation",
    "mechanism_call",
    "skip_reason",
]


def _write_human_perdonor_substrate_slices(
    perdonor_rows: list[tuple],
) -> dict:
    """Per-kinase shards for `leading_substrates` + `substrate_motifs`.

    Together these two columns added ~50 MB to the inlined PAYLOAD; they are
    only consumed by the human Audit drawer (Trace + Running Enrichment sub-
    tabs) when a specific kinase is selected. Shard format: one parquet per
    kinase id with rows (donor, leading_substrates, substrate_motifs). Rows
    with both fields empty are skipped — the JS treats a missing row as
    "no leading edge" the same way it used to treat an empty string.
    """
    shutil.rmtree(EDGE_SLICES_HUMAN_PERDONOR_DIR, ignore_errors=True)
    os.makedirs(EDGE_SLICES_HUMAN_PERDONOR_DIR, exist_ok=True)

    by_kid: dict[int, list[tuple[str, str, str, str]]] = {}
    for r in perdonor_rows:
        kid = int(r[0])
        donor = str(r[1])
        leading = str(r[4]) if r[4] is not None else ""
        motifs = str(r[11]) if r[11] is not None else ""
        kl_pcts = str(r[12]) if len(r) > 12 and r[12] is not None else ""
        if not leading and not motifs:
            continue
        by_kid.setdefault(kid, []).append((donor, leading, motifs, kl_pcts))

    template = "{kinase_id:03d}.parquet"
    present: list[int] = []
    total_rows = 0
    for kid in sorted(by_kid):
        df = pd.DataFrame(
            by_kid[kid],
            columns=["donor", "leading_substrates", "substrate_motifs", "substrate_kl_percentiles"],
        )
        path = os.path.join(
            EDGE_SLICES_HUMAN_PERDONOR_DIR,
            template.format(kinase_id=kid),
        )
        pq.write_table(
            pa.Table.from_pandas(df, preserve_index=False),
            path, compression="zstd",
        )
        present.append(kid)
        total_rows += len(df)

    index = {
        "schema_version": SCHEMA_VERSION,
        "slice_count": len(present),
        "present_kinase_ids": present,
        "filename_template": template,
        "n_total_rows": total_rows,
    }
    with open(os.path.join(EDGE_SLICES_HUMAN_PERDONOR_DIR, "index.json"), "w") as f:
        json.dump(index, f)
    print(f"  human_perdonor_substrate: wrote {len(present)} shards "
          f"({total_rows:,} total rows)", flush=True)
    return index


def _human_track_load(suffix: str, residue: str) -> dict | None:
    """Load all per-donor CSVs for one track. Returns None if missing.

    Both the stoichiometry and raw-phospho MEA outputs are loaded when
    present. Raw-phospho is the sensitivity check (analogous to mouse
    ``alz/bulk_mea/mechanism.py``); when only stoichiometry is available the
    raw entries are empty DataFrames and the viewer hides the comparison.
    """
    rec_path = os.path.join(HUMAN_PERDONOR_DIR, f"recurrence{suffix}.csv")
    rec_ctrl_path = os.path.join(HUMAN_PERDONOR_DIR, f"recurrence_ctrl{suffix}.csv")
    nes_path = os.path.join(HUMAN_PERDONOR_DIR, f"kinase_donor_nes{suffix}.csv")
    fdr_path = os.path.join(HUMAN_PERDONOR_DIR, f"kinase_donor_fdr{suffix}.csv")
    mea_path = os.path.join(HUMAN_PERDONOR_DIR, f"mea_perdonor{suffix}.csv")
    shift_path = os.path.join(HUMAN_PERDONOR_DIR, f"mea_global_shift{suffix}.csv")
    winsor_path = os.path.join(HUMAN_PERDONOR_DIR, f"winsorized_sites{suffix}.csv")
    # Raw-phospho counterparts (optional; populated when ingest_mukesh_perdonor.py
    # has been re-run with the raw-track export enabled).
    raw_mea_path = os.path.join(HUMAN_PERDONOR_DIR, f"mea_perdonor_raw{suffix}.csv")
    raw_nes_path = os.path.join(HUMAN_PERDONOR_DIR, f"kinase_donor_nes_raw{suffix}.csv")
    raw_fdr_path = os.path.join(HUMAN_PERDONOR_DIR, f"kinase_donor_fdr_raw{suffix}.csv")
    subs_path = os.path.join(HUMAN_PERDONOR_DIR, f"mea_substrate_sets{suffix}.csv")
    # Site-level matrices live one directory up from perdonor/
    parent = os.path.dirname(HUMAN_PERDONOR_DIR)
    stoich_path = os.path.join(parent, f"stoichiometry_matrix{suffix}.csv")
    raw_path = os.path.join(parent, f"raw_phospho_normalized{suffix}.csv")
    for p in (rec_path, nes_path, fdr_path):
        if not os.path.exists(p):
            return None
    rec = pd.read_csv(rec_path)
    rec_ctrl = pd.read_csv(rec_ctrl_path) if os.path.exists(rec_ctrl_path) else pd.DataFrame()
    nes = pd.read_csv(nes_path).set_index("kinase")
    fdr = pd.read_csv(fdr_path).set_index("kinase")
    mea = pd.read_csv(mea_path) if os.path.exists(mea_path) else pd.DataFrame()
    shift = pd.read_csv(shift_path) if os.path.exists(shift_path) else pd.DataFrame()
    winsor = pd.read_csv(winsor_path) if os.path.exists(winsor_path) else pd.DataFrame()
    stoich = pd.read_csv(stoich_path) if os.path.exists(stoich_path) else pd.DataFrame()
    raw = pd.read_csv(raw_path) if os.path.exists(raw_path) else pd.DataFrame()
    raw_mea = pd.read_csv(raw_mea_path) if os.path.exists(raw_mea_path) else pd.DataFrame()
    raw_nes = pd.read_csv(raw_nes_path).set_index("kinase") if os.path.exists(raw_nes_path) else pd.DataFrame()
    raw_fdr = pd.read_csv(raw_fdr_path).set_index("kinase") if os.path.exists(raw_fdr_path) else pd.DataFrame()
    subs = pd.read_csv(subs_path) if os.path.exists(subs_path) else pd.DataFrame()
    return {
        "residue": residue, "rec": rec, "rec_ctrl": rec_ctrl,
        "nes": nes, "fdr": fdr,
        "mea": mea, "shift": shift, "winsor": winsor,
        "stoich": stoich, "raw": raw,
        "raw_mea": raw_mea, "raw_nes": raw_nes, "raw_fdr": raw_fdr,
        "subs": subs,
    }


def _load_human_mechanism_attribution() -> list[dict]:
    rows: list[dict] = []
    for suffix, _residue in HUMAN_TRACK_SUFFIXES:
        path = os.path.join(HUMAN_PERDONOR_DIR, f"mechanism_attribution{suffix}.csv")
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        if df.empty or "mechanism_call" not in df.columns or "mechanism_score" in df.columns:
            continue
        rows.extend(df[[c for c in _MECHANISM_COLUMNS if c in df.columns]].to_dict(orient="records"))
    return rows


def build_human_slice() -> tuple[dict, dict] | tuple[None, None]:
    """Per-donor human (NBB / Mukesh) kinase slice.

    Returns (human_slice, substrate_slice_index). Both are None when the
    perdonor outputs are absent — caller omits the PAYLOAD.human block so
    the mouse-only artifact stays byte-equivalent.
    """
    if not os.path.isdir(HUMAN_PERDONOR_DIR):
        return None, None
    tracks: list[dict] = []
    for suffix, residue in HUMAN_TRACK_SUFFIXES:
        t = _human_track_load(suffix, residue)
        if t is not None:
            tracks.append(t)
    if not tracks:
        return None, None

    # Donor membership — prefer the explicit donor_groups.json sidecar
    # (written by ingest_mukesh_perdonor.py); fall back to name-prefix
    # inference for older perdonor outputs that predate the sidecar.
    donor_groups_path = os.path.join(HUMAN_PERDONOR_DIR, "donor_groups.json")
    if os.path.exists(donor_groups_path):
        with open(donor_groups_path) as fh:
            _dg = json.load(fh)
        ad_set = {str(x) for x in _dg.get("ad", [])}
        ctrl_set = {str(x) for x in _dg.get("ctrl", [])}
    else:
        ad_set = set()
        ctrl_set = set()

    # AD donors drive the rendering axis ("donors"). CTRL donors are
    # rendered side-by-side as a separate column group.
    donors: list[str] = []          # AD only
    ctrl_donors: list[str] = []     # CTRL only
    seen: set[str] = set()
    for t in tracks:
        for d in list(t["nes"].columns):
            ds = str(d)
            if ds in seen:
                continue
            seen.add(ds)
            if ctrl_set:
                if ds in ad_set:
                    donors.append(ds)
                elif ds in ctrl_set:
                    ctrl_donors.append(ds)
                else:
                    # Unknown — treat as AD so we don't silently drop it.
                    donors.append(ds)
            else:
                # No sidecar: legacy prefix inference.
                (ctrl_donors if ds.upper().startswith("CTRL") else donors).append(ds)
    all_donors_axis = donors + ctrl_donors
    contrasts = [f"{d}_vs_CTRLmean" for d in all_donors_axis]

    # Gene-symbol map from the cached mapping CSV; fall back to identity.
    gene_map: dict[str, str] = {}
    gene_csv = os.path.join(
        config.REPO_ROOT, "data", "datasets", "song", "analysis_cache",
        "kinase_to_gene_mapping.csv",
    )
    if os.path.exists(gene_csv):
        gm = pd.read_csv(gene_csv)
        for k, g in zip(gm["kinase_abbreviation"], gm["gene_symbol"]):
            if pd.notna(g):
                gene_map[str(k)] = str(g)

    # Optional: SEA-AD per-kinase agreement (one cohort-level LFC per kinase,
    # collapsed across SEA-AD supertypes; no cell-type bridge because the
    # human AD samples have no per-cell-type resolution).
    seaad_csv = os.path.join(
        config.REPO_ROOT, "outputs", "reports", "kinase_attribution",
        "human_seaad_agreement.csv",
    )
    seaad_lookup: dict[tuple[str, str], dict] = {}
    if os.path.exists(seaad_csv):
        sdf = pd.read_csv(seaad_csv)
        for _, srow in sdf.iterrows():
            key = (str(srow["kinase"]), str(srow["residue_type"]))
            seaad_lookup[key] = {
                "lfc": (float(srow["sea_ad_lfc_median"])
                        if pd.notna(srow["sea_ad_lfc_median"]) else None),
                "n": int(srow["n_supertypes_finite"]) if pd.notna(srow["n_supertypes_finite"]) else 0,
                "agreement": (float(srow["sea_ad_direction_agreement"])
                              if pd.notna(srow["sea_ad_direction_agreement"]) else None),
            }

    # Build (kinase, residue) rows. Same kinase can appear once per track.
    cols: dict[str, list] = {
        "id": [], "name": [], "gene_symbol": [], "residue_type": [],
        "n_donors_sig": [], "n_donors_up": [], "n_donors_down": [],
        "n_donors_tested": [],
        "median_nes": [], "median_nes_sig_only": [],
        "sea_ad_lfc": [], "sea_ad_n_supertypes": [], "sea_ad_direction_agreement": [],
    }
    for d in all_donors_axis:
        cols[f"NES_{d}_vs_CTRLmean"] = []
        cols[f"FDR_{d}_vs_CTRLmean"] = []

    perdonor_rows: list[tuple] = []  # (kid, donor, NES, FDR, lead, ES, subs_fraction, p_value, raw_NES, raw_FDR, raw_p, substrate_motifs, substrate_kl_percentiles)
    next_id = 0
    for t in tracks:
        residue = t["residue"]
        rec = t["rec"].set_index("kinase")
        nes = t["nes"]
        fdr = t["fdr"]
        mea = t["mea"]
        raw_mea = t.get("raw_mea", pd.DataFrame())
        raw_lookup: dict[tuple[str, str], tuple[float, float, float]] = {}
        if not raw_mea.empty:
            for _, rrow in raw_mea.iterrows():
                k = str(rrow.get("kinase", ""))
                contrast = str(rrow.get("contrast", ""))
                donor = contrast.replace("_vs_CTRLmean", "")
                raw_lookup[(k, donor)] = (
                    float(rrow.get("NES")) if pd.notna(rrow.get("NES")) else float("nan"),
                    float(rrow.get("FDR")) if pd.notna(rrow.get("FDR")) else float("nan"),
                    float(rrow.get("p-value")) if pd.notna(rrow.get("p-value")) else float("nan"),
                )
        subs = t.get("subs", pd.DataFrame())
        # Per-(kinase, donor) substrate motif set used by GSEA as the hit set.
        # Mirrors the mouse `mea_substrate_sets.csv` contract; persisted by
        # `ingest_mukesh_perdonor.py`. Used at view time to replay the GSEA walk
        # against the full substrate set (not the leading-edge subset).
        subs_lookup: dict[tuple[str, str], list[str]] = {}
        subs_pct_lookup: dict[tuple[str, str], list[float]] = {}
        if not subs.empty:
            for _, srow in subs.iterrows():
                k = str(srow.get("kinase", ""))
                contrast = str(srow.get("contrast", ""))
                donor = contrast.replace("_vs_CTRLmean", "")
                motif = str(srow.get("motif", ""))
                if motif:
                    subs_lookup.setdefault((k, donor), []).append(motif)
                    pct_val = srow.get("kl_percentile")
                    subs_pct_lookup.setdefault((k, donor), []).append(
                        float(pct_val) if pd.notna(pct_val) else float("nan")
                    )
        mea_lookup: dict[tuple[str, str], tuple[float, float, str, float, str, float]] = {}
        if not mea.empty:
            for _, row in mea.iterrows():
                k = str(row.get("kinase", ""))
                contrast = str(row.get("contrast", ""))
                donor = contrast.replace("_vs_CTRLmean", "")
                lead = str(row.get("Leading substrates", ""))
                mea_lookup[(k, donor)] = (
                    float(row.get("NES")) if pd.notna(row.get("NES")) else float("nan"),
                    float(row.get("FDR")) if pd.notna(row.get("FDR")) else float("nan"),
                    lead,
                    float(row.get("ES")) if pd.notna(row.get("ES")) else float("nan"),
                    str(row.get("Subs fraction", "")),
                    float(row.get("p-value")) if pd.notna(row.get("p-value")) else float("nan"),
                )

        for k in nes.index.astype(str):
            kid = next_id
            next_id += 1
            cols["id"].append(kid)
            cols["name"].append(k)
            cols["gene_symbol"].append(gene_map.get(k, k))
            cols["residue_type"].append(residue)
            rrow = rec.loc[k] if k in rec.index else None

            def _gr(c, default=0):
                if rrow is None or c not in rrow.index:
                    return default
                v = rrow[c]
                return default if pd.isna(v) else v

            cols["n_donors_sig"].append(int(_gr("n_donors_sig", 0)))
            cols["n_donors_up"].append(int(_gr("n_donors_up", 0)))
            cols["n_donors_down"].append(int(_gr("n_donors_down", 0)))
            cols["n_donors_tested"].append(int(_gr("n_donors_tested", 0)))
            cols["median_nes"].append(_gr("median_nes", float("nan")))
            cols["median_nes_sig_only"].append(_gr("median_nes_sig_only", float("nan")))
            sea = seaad_lookup.get((k, residue))
            cols["sea_ad_lfc"].append(round(sea["lfc"], 4) if (sea and sea["lfc"] is not None) else None)
            cols["sea_ad_n_supertypes"].append(sea["n"] if sea else 0)
            cols["sea_ad_direction_agreement"].append(
                round(sea["agreement"], 3) if (sea and sea["agreement"] is not None) else None
            )

            for d in all_donors_axis:
                if d in nes.columns:
                    v_nes = nes.loc[k, d] if k in nes.index else float("nan")
                    v_fdr = fdr.loc[k, d] if k in fdr.index else float("nan")
                else:
                    v_nes = float("nan")
                    v_fdr = float("nan")
                cols[f"NES_{d}_vs_CTRLmean"].append(float(v_nes) if pd.notna(v_nes) else float("nan"))
                cols[f"FDR_{d}_vs_CTRLmean"].append(float(v_fdr) if pd.notna(v_fdr) else float("nan"))
                if (k, d) in mea_lookup:
                    n_, f_, lead, es_, subs_, p_ = mea_lookup[(k, d)]
                    rn_, rf_, rp_ = raw_lookup.get((k, d), (float("nan"), float("nan"), float("nan")))
                    motifs = ";".join(subs_lookup.get((k, d), []))
                    kl_pcts = ";".join(
                        ("" if not (v == v) else f"{v:.2f}")  # NaN check via self-cmp
                        for v in subs_pct_lookup.get((k, d), [])
                    )
                    perdonor_rows.append((kid, d, n_, f_, lead, es_, subs_, p_, rn_, rf_, rp_, motifs, kl_pcts))

    # Per-donor scalar index (one row per (kinase, donor) with an MEA record).
    # `leading_substrates` and `substrate_motifs` are sharded out to
    # edge_slices/human_perdonor/ and loaded on demand via
    # SliceCache.loadHumanPerdonorSubstrate.
    perdonor_index = {
        "kinase_id":   [r[0] for r in perdonor_rows],
        "donor":       [r[1] for r in perdonor_rows],
        "NES":         [round(r[2], 4) if pd.notna(r[2]) else None for r in perdonor_rows],
        "FDR":         [round(r[3], 4) if pd.notna(r[3]) else None for r in perdonor_rows],
        "ES":          [round(r[5], 4) if pd.notna(r[5]) else None for r in perdonor_rows],
        "subs_fraction": [r[6] for r in perdonor_rows],
        "p_value":     [round(r[7], 6) if pd.notna(r[7]) else None for r in perdonor_rows],
        "raw_NES":     [round(r[8], 4) if pd.notna(r[8]) else None for r in perdonor_rows],
        "raw_FDR":     [round(r[9], 4) if pd.notna(r[9]) else None for r in perdonor_rows],
        "raw_p_value": [round(r[10], 6) if pd.notna(r[10]) else None for r in perdonor_rows],
    }
    human_perdonor_substrate_slice_index = (
        _write_human_perdonor_substrate_slices(perdonor_rows)
    )

    # Global-shift diagnostics (track-tagged).
    shift_rows: list[dict] = []
    for t in tracks:
        if t["shift"].empty:
            continue
        for _, row in t["shift"].iterrows():
            contrast = str(row.get("contrast", ""))
            donor = contrast.replace("_vs_CTRLmean", "")
            shift_rows.append({
                "contrast": contrast,
                "donor": donor,
                "residue_type": t["residue"],
                "median_shift": float(row.get("median_shift")) if pd.notna(row.get("median_shift")) else None,
                "mean_before": float(row.get("mean_before")) if pd.notna(row.get("mean_before")) else None,
                "pct_pos_before": float(row.get("pct_pos_before")) if pd.notna(row.get("pct_pos_before")) else None,
                "pct_pos_after": float(row.get("pct_pos_after")) if pd.notna(row.get("pct_pos_after")) else None,
            })

    # Winsorization receipts: per-(donor, track) clipped sites.
    winsor_cols: dict[str, list] = {
        "donor": [], "residue_type": [], "site_id": [], "gene_symbol": [],
        "original_lfc": [], "clipped_lfc": [], "lower_bound": [], "upper_bound": [],
    }
    for t in tracks:
        if t["winsor"].empty:
            continue
        for _, row in t["winsor"].iterrows():
            contrast = str(row.get("contrast", ""))
            donor = contrast.replace("_vs_CTRLmean", "")
            winsor_cols["donor"].append(donor)
            winsor_cols["residue_type"].append(t["residue"])
            winsor_cols["site_id"].append(str(row.get("site_id", "")))
            winsor_cols["gene_symbol"].append(str(row.get("gene_symbol", "")))
            for k in ("original_lfc", "clipped_lfc", "lower_bound", "upper_bound"):
                v = row.get(k)
                winsor_cols[k].append(round(float(v), 4) if pd.notna(v) else None)

    # Site-level measurement-trace matrices (per-site stoichiometry + raw phospho).
    # Donor axis = union of donor + CTRL columns across tracks.
    sites_cols: dict[str, list] = {
        "site_id": [], "motif": [], "gene_symbol": [], "site_position": [],
        "residue_type": [],
    }
    donors_all: list[str] = []
    seen_all: set[str] = set()
    META_COLS = {"site_id", "protein_id", "gene_symbol", "site_position", "motif"}
    for t in tracks:
        for df in (t["stoich"], t["raw"]):
            if df is None or df.empty:
                continue
            for c in df.columns:
                if c in META_COLS:
                    continue
                if c not in seen_all:
                    seen_all.add(c)
                    donors_all.append(str(c))
    # Reorder donors_all so case (AD) donors land before CTRL — matches the
    # rendering axis order and keeps wide loaders that key off column
    # position stable.
    if ctrl_donors:
        ctrl_set_axis = set(ctrl_donors)
        donors_all = (
            [d for d in donors_all if d not in ctrl_set_axis]
            + [d for d in donors_all if d in ctrl_set_axis]
        )
    case_donors = [d for d in donors_all if d not in set(ctrl_donors)]

    stoich_by_site: list[list[float | None]] = []
    raw_by_site: list[list[float | None]] = []
    for t in tracks:
        residue = t["residue"]
        sdf = t["stoich"]
        rdf = t["raw"]
        if sdf is None or sdf.empty:
            continue
        # Index raw by site_id for quick join (raw may differ in coverage).
        raw_idx = rdf.set_index("site_id") if (rdf is not None and not rdf.empty) else None
        for _, srow in sdf.iterrows():
            sid = str(srow.get("site_id", ""))
            sites_cols["site_id"].append(sid)
            sites_cols["motif"].append(str(srow.get("motif", "")))
            sites_cols["gene_symbol"].append(str(srow.get("gene_symbol", "")))
            sites_cols["site_position"].append(str(srow.get("site_position", "")))
            sites_cols["residue_type"].append(residue)
            srow_vals: list[float | None] = []
            rrow_vals: list[float | None] = []
            for d in donors_all:
                v = srow.get(d) if d in srow.index else None
                srow_vals.append(round(float(v), 4) if (v is not None and pd.notna(v)) else None)
                if raw_idx is not None and sid in raw_idx.index and d in raw_idx.columns:
                    rv = raw_idx.loc[sid, d]
                    rrow_vals.append(round(float(rv), 4) if pd.notna(rv) else None)
                else:
                    rrow_vals.append(None)
            stoich_by_site.append(srow_vals)
            raw_by_site.append(rrow_vals)

    # CTRL recurrence sidecar (kinase × CTRL-donor sig counts). Always
    # AD-only on the kinase columns — the CTRL block here is a separate
    # diagnostic surface. Track-tagged.
    ctrl_rec_rows: list[dict] = []
    for t in tracks:
        residue = t["residue"]
        rec_ctrl = t.get("rec_ctrl")
        if rec_ctrl is None or rec_ctrl.empty:
            continue
        for _, row in rec_ctrl.iterrows():
            ctrl_rec_rows.append({
                "kinase": str(row.get("kinase", "")),
                "residue_type": residue,
                "n_donors_sig": int(row.get("n_donors_sig", 0) or 0),
                "n_donors_up": int(row.get("n_donors_up", 0) or 0),
                "n_donors_down": int(row.get("n_donors_down", 0) or 0),
                "n_donors_tested": int(row.get("n_donors_tested", 0) or 0),
                "median_nes": (round(float(row["median_nes"]), 4)
                               if pd.notna(row.get("median_nes")) else None),
            })

    # Cell-type specificity block (CR03). Populated when human_reference_expression.py
    # has been run and the specificity CSVs exist. Gracefully absent in phase-1 runs.
    celltype_specificity: dict | None = None
    if _HAS_HUMAN_CELLTYPE:
        try:
            celltype_specificity = build_celltype_specificity_payload()
        except Exception as exc:
            print(f"  human celltype_specificity: skipped ({exc})", flush=True)

    human_slice: dict = {
        "schema_version": 2,  # bumped: adds ctrl_donors, NES_<ctrl>_vs_CTRLmean cols
        "kinases": cols,
        "donors": donors,
        "ctrl_donors": ctrl_donors,
        "contrasts": contrasts,
        "perdonor_index": perdonor_index,
        "global_shift": shift_rows,
        "winsor": winsor_cols,
        "sites": sites_cols,
        "donors_all": donors_all,
        "case_donors": case_donors,
        "recurrence_ctrl": ctrl_rec_rows,
        "stoich_by_site": stoich_by_site,
        "raw_phospho_by_site": raw_by_site,
    }
    if celltype_specificity is not None:
        human_slice["celltype_specificity"] = celltype_specificity
    mechanism_attribution = _load_human_mechanism_attribution()
    if mechanism_attribution:
        human_slice["mechanism_attribution"] = mechanism_attribution
    return human_slice, human_perdonor_substrate_slice_index


def build_mukesh_viewer_slice() -> CohortViewerSlice | None:
    """Mukesh (human NBB per-donor) contribution to the unified viewer payload.
    Returns None when the perdonor outputs are absent (mouse-only build stays
    byte-equivalent — the caller omits PAYLOAD.human)."""
    human_slice, substrate_index = build_human_slice()
    if human_slice is None:
        return None
    present = (substrate_index or {}).get("present_kinase_ids", [])
    return CohortViewerSlice(
        cohort_id="mukesh",
        context_ids=("song_ad",),
        owned_sections={"human": human_slice},
        capabilities={"human_reference": True},
        edge_slice_ref=(EdgeSliceContribution(
            "human_perdonor",
            {"present_human_perdonor_kinase_ids": present},
        ),),
        kinase_names=tuple(human_slice["kinases"].get("name", [])),
        provenance={"source_dir": HUMAN_PERDONOR_DIR},
    )

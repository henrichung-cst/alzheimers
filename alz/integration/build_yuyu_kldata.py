"""Build a Yuyu/Song-derived kinase library (kldata.csv) for Incytr.

Replaces the 5xFAD demo kldata (`data/datasets/5xFAD/kinase/kldata_pspy.csv`)
that the live factorial integration was incorrectly defaulting to. The kinase
library is study-specific: its row set is the substrate sites actually
phosphoprofiled in this cohort, ranked against PSPA motif specificity to
predict the top-N kinases per site. Using a different study's substrate set
silently biases PDS.

Output: `data/datasets/song/kinase/kldata_pspy.csv` + `PROVENANCE.json`.

Procedure (matches `data/incytr/shared/kinase_library.ipynb`):
  1. Load IMAC + pY sitequant Excel files (substrate gene_symbol is already
     mouse-format).
  2. Replace truncation marker "x" → "_" in motif sequences.
  3. Derive `Type` ∈ {pS, pT, pY} from the central residue.
  4. Run kinase_library.PhosphoProteomics → percentile rank.
     - ser_thr kinases ranked over S/T sites; keep top-5 per site.
     - tyrosine kinases ranked over Y sites; keep top-15 per site.
  5. Map human KINASE → mouse symbol via mygene homologene; cache the mapping.
  6. Emit kldata.csv with the upstream-required schema:
     `gene` (mouse substrate), `site_pos`, `motif.geneName` (mouse kinase),
     `Type`.

Run: `pixi run python alz/integration/build_yuyu_kldata.py`
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
IMAC_XLSX = REPO_ROOT / "data/datasets/song/primary/phospho/song_IMAC_sitequant_merged_labeled (2).xlsx"
PY_XLSX = REPO_ROOT / "data/datasets/song/primary/phospho/song_pY_sitequant_merged_labeled (2).xlsx"
OUT_DIR = REPO_ROOT / "data/datasets/song/kinase"
KLDATA_PATH = OUT_DIR / "kldata_pspy.csv"
PROVENANCE_PATH = OUT_DIR / "PROVENANCE.json"
HOMOLOGENE_CACHE = REPO_ROOT / "data/derived/caches/human_to_mouse_homologene.csv"

# Per the kinase_library notebook: top-5 ser/thr predictions per S/T site,
# top-15 tyr predictions per Y site.
TOP_N_SER_THR = 5
TOP_N_TYR = 15


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_yuyu_sites() -> pd.DataFrame:
    """Concatenate IMAC + pY sitequant; keep substrate identity + motif + type."""
    imac = pd.read_excel(IMAC_XLSX, skiprows=1)
    py = pd.read_excel(PY_XLSX, skiprows=1)
    cols = ["protein_id", "gene_symbol", "site_position", "motif"]
    df = pd.concat([imac[cols], py[cols]], ignore_index=True)

    # Truncation marker → underscore (kinase-library convention).
    df["motif"] = df["motif"].astype(str).str.replace("x", "_", regex=False)

    # Center residue determines pS/pT/pY classification.
    center = df["motif"].str.len() // 2
    df["phos_res"] = df.apply(lambda r: r["motif"][center.loc[r.name]], axis=1)
    df = df[df["phos_res"].isin(["S", "T", "Y"])].copy()
    df["Type"] = "p" + df["phos_res"]

    # Drop site duplicates on (gene, position, motif) — same site can appear
    # in both IMAC and pY (cross-enrichment artifact). Keep first.
    df = df.drop_duplicates(subset=["gene_symbol", "site_position", "motif"])
    return df.reset_index(drop=True)


def rank_kinases(df: pd.DataFrame, kin_type: str, top_n: int) -> pd.DataFrame:
    """Run kinase_library percentile-rank for one kinase family and reshape to long.

    Returns (gene_symbol, site_position, motif, kinase) rows — one row per
    (site, predicted kinase) up to top_n per site.
    """
    import kinase_library as kl

    if kin_type == "ser_thr":
        sub = df[df["phos_res"].isin(["S", "T"])]
    else:
        sub = df[df["phos_res"] == "Y"]
    if sub.empty:
        return pd.DataFrame(columns=["gene_symbol", "site_position", "motif", "kinase"])

    # Pass only the columns kinase_library needs; carrying wider metadata
    # bleeds non-numeric columns into the ranked output and breaks nsmallest.
    sub_min = sub[["gene_symbol", "site_position", "motif"]].copy()
    pps = kl.PhosphoProteomics(sub_min, seq_col="motif")
    ranked = pps.rank(metric="percentile", kin_type=kin_type)
    ranked = ranked.reset_index(drop=True)

    meta_cols = ["gene_symbol", "site_position", "motif"]
    # Score cols are the numeric (per-kinase percentile) columns kinase_library
    # appends. Detect by dtype — robust to any future metadata-column changes.
    score_cols = [c for c in ranked.columns
                  if c not in meta_cols and pd.api.types.is_numeric_dtype(ranked[c])]
    if not score_cols:
        raise RuntimeError(f"kinase_library returned no kinase columns for kin_type={kin_type}")

    top_kinases = ranked[score_cols].apply(
        lambda row: row.nsmallest(top_n).index.tolist(), axis=1
    )
    top_df = pd.DataFrame(top_kinases.tolist(), index=ranked.index,
                          columns=[str(i + 1) for i in range(top_n)])
    wide = pd.concat([ranked[meta_cols], top_df], axis=1)
    long = wide.melt(
        id_vars=meta_cols,
        value_vars=top_df.columns.tolist(),
        value_name="kinase",
    )
    long = long.dropna(subset=["kinase"]).reset_index(drop=True)
    long = long.rename(columns={"site_position": "site_position", "gene_symbol": "gene_symbol"})
    return long[["gene_symbol", "site_position", "motif", "kinase"]]


def map_human_to_mouse(human_symbols: list[str]) -> dict:
    """Map human gene symbols → mouse via mygene homologene. Cache to CSV."""
    HOMOLOGENE_CACHE.parent.mkdir(parents=True, exist_ok=True)
    cache: dict[str, str] = {}
    if HOMOLOGENE_CACHE.exists():
        prior = pd.read_csv(HOMOLOGENE_CACHE)
        cache = dict(zip(prior["human"], prior["mouse"]))

    missing = sorted({s for s in human_symbols if s and s not in cache})
    if missing:
        import mygene
        mg = mygene.MyGeneInfo()
        # homologene field returns [[species_id, gene_id, symbol], ...]; species 10090 = mouse.
        results = mg.querymany(
            missing,
            scopes="symbol",
            species="human",
            fields="homologene",
            returnall=False,
            verbose=False,
        )
        for r in results:
            human = r.get("query")
            mouse_sym = None
            homo = r.get("homologene", {})
            for entry in homo.get("genes", []) or []:
                # Each entry is [tax_id, gene_id]. Need a second call to resolve
                # mouse gene_id → symbol — mygene returns symbol via separate query.
                if entry and entry[0] == 10090:
                    mouse_id = entry[1]
                    try:
                        sym = mg.getgene(mouse_id, fields="symbol").get("symbol")
                    except Exception:
                        sym = None
                    if sym:
                        mouse_sym = sym
                    break
            cache[human] = mouse_sym or ""
        # Persist
        out = pd.DataFrame(
            sorted(cache.items()), columns=["human", "mouse"]
        )
        out.to_csv(HOMOLOGENE_CACHE, index=False)
        print(f"  cached {len(out)} human→mouse mappings to {HOMOLOGENE_CACHE.relative_to(REPO_ROOT)}")
    return cache


def build_kldata() -> pd.DataFrame:
    print("Loading Yuyu IMAC + pY sitequant ...")
    sites = load_yuyu_sites()
    print(f"  {len(sites)} unique sites  (S={sum(sites.phos_res=='S')}, T={sum(sites.phos_res=='T')}, Y={sum(sites.phos_res=='Y')})")

    print(f"Ranking ser_thr kinases (top {TOP_N_SER_THR} per S/T site) ...")
    st = rank_kinases(sites, "ser_thr", TOP_N_SER_THR)
    print(f"  {len(st)} (site × kinase) rows from S/T")

    print(f"Ranking tyrosine kinases (top {TOP_N_TYR} per Y site) ...")
    tyr = rank_kinases(sites, "tyrosine", TOP_N_TYR)
    print(f"  {len(tyr)} (site × kinase) rows from Y")

    combined = pd.concat([st, tyr], ignore_index=True)

    # Join with kinome_info to recover human KINASE → GENE_NAME (canonical)
    import kinase_library as kl
    info = kl.get_kinome_info()[["KINASE", "GENE_NAME"]].drop_duplicates()
    combined = combined.merge(info, how="left", left_on="kinase", right_on="KINASE")
    # GENE_NAME is the canonical human symbol; some kinase MATRIX_NAMEs may not
    # round-trip, fall back to the kinase name.
    combined["human_kinase"] = combined["GENE_NAME"].fillna(combined["kinase"])

    print("Resolving human → mouse kinase symbols (mygene homologene) ...")
    mapping = map_human_to_mouse(combined["human_kinase"].unique().tolist())
    combined["motif.geneName"] = combined["human_kinase"].map(mapping).fillna("")

    # Re-merge with sites to recover Type.
    kldata = combined.merge(
        sites[["gene_symbol", "site_position", "motif", "Type"]],
        on=["gene_symbol", "site_position", "motif"],
        how="left",
    )

    # Final schema match (5xFAD kldata: '', gene, site_pos, motif.geneName, Type).
    out = pd.DataFrame({
        "gene": kldata["gene_symbol"],
        "site_pos": kldata["site_position"].astype(int),
        "motif.geneName": kldata["motif.geneName"],
        "Type": kldata["Type"],
    })
    n_pre = len(out)
    out = out[out["motif.geneName"] != ""].reset_index(drop=True)
    print(f"  {n_pre - len(out)} rows dropped (no mouse homolog); {len(out)} rows in final kldata")
    return out


def write_provenance(out: pd.DataFrame) -> None:
    import kinase_library as kl
    prov = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "kinase_library_version": kl.__version__,
        "top_n_ser_thr": TOP_N_SER_THR,
        "top_n_tyr": TOP_N_TYR,
        "rows": int(len(out)),
        "unique_substrate_genes": int(out["gene"].nunique()),
        "unique_kinases": int(out["motif.geneName"].nunique()),
        "type_counts": out["Type"].value_counts().to_dict(),
        "sources": {
            "imac_xlsx": {
                "path": str(IMAC_XLSX.relative_to(REPO_ROOT)),
                "sha256": _sha256(IMAC_XLSX),
            },
            "py_xlsx": {
                "path": str(PY_XLSX.relative_to(REPO_ROOT)),
                "sha256": _sha256(PY_XLSX),
            },
        },
        "homologene_cache": str(HOMOLOGENE_CACHE.relative_to(REPO_ROOT)),
        "notes": (
            "Replaces data/datasets/5xFAD/kinase/kldata_pspy.csv as the kldata "
            "input to alz/integration/. The kinase library is study-specific; "
            "the substrate row set must come from sites actually "
            "phosphoprofiled in this cohort."
        ),
    }
    PROVENANCE_PATH.write_text(json.dumps(prov, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=KLDATA_PATH,
                    help="Output kldata.csv path")
    ap.add_argument("--force", action="store_true",
                    help="Overwrite existing output")
    args = ap.parse_args()

    if args.out.exists() and not args.force:
        print(f"{args.out} already exists. Use --force to overwrite.")
        sys.exit(0)

    out = build_kldata()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index_label="")
    print(f"\nWrote {args.out.relative_to(REPO_ROOT)}  ({len(out)} rows)")
    write_provenance(out)
    print(f"Wrote {PROVENANCE_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()

"""Build a human kinase library (kldata.csv) for the T-cell exhaustion cohort.

The T-cell cohort is **human** (upper-case HGNC symbols); the Song/Yuyu kldata
is mouse-cased and would silently match nothing against human scRNA. SiK
scoring needs a human kinase→substrate edge set. It is derivable: the kinase
library is study-specific — its row set is the substrate sites actually
phosphoprofiled in this cohort, ranked against PSPA motif specificity to
predict the top-N kinases per site. The PSPA ranking core
(`rank_kinases`) is species-agnostic; the only mouse-specific step in the
Song builder is a homologene conversion, which we skip (substrate + kinase
symbols are already human).

Substrate source = **donor1's** bulk phospho (`ps_bulk_linear.csv` +
`py_bulk_linear.csv`). donor2's pY motif column is empty (no IMAC assay), so
it cannot self-seed; per the cohort decision a single donor1-derived human
kldata is applied to both donors — donor2's SiK Exclusiveness Index is still
computed on donor2's own scRNA, only the candidate kinase→substrate edge set
is borrowed (same human T-cell biology).

Output: `data/datasets/tcells/kinase/kldata_human.csv` + `PROVENANCE.json`.

Procedure (mirrors build_yuyu_kldata.py, minus homologene):
  1. Load donor1 ps + py bulk_linear sites (gene_symbol + motif already human,
     15-char, `_`-padded — already kinase_library format; no `x→_` replace).
  2. Derive `Type` ∈ {pS, pT, pY} from the central residue.
  3. rank_kinases percentile-rank: top-5 ser/thr per S/T, top-15 tyr per Y.
  4. Map kinase MATRIX_NAME → human GENE_NAME via kl.get_kinome_info().
  5. Emit kldata.csv: `gene` (human substrate), `site_pos`,
     `motif.geneName` (human kinase), `Type`.

Run: `pixi run python alz/integration/build_tcells_kldata.py`
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))  # repo root

from alz.integration.build_yuyu_kldata import (
    TOP_N_SER_THR,
    TOP_N_TYR,
    _sha256,
    rank_kinases,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DONOR1_DIR = REPO_ROOT / "data/derived/tcells_incytr_inputs/donor1"
PS_CSV = DONOR1_DIR / "ps_bulk_linear.csv"
PY_CSV = DONOR1_DIR / "py_bulk_linear.csv"
OUT_DIR = REPO_ROOT / "data/datasets/tcells/kinase"
KLDATA_PATH = OUT_DIR / "kldata_human.csv"
PROVENANCE_PATH = OUT_DIR / "PROVENANCE.json"


def load_tcell_sites() -> pd.DataFrame:
    """Concatenate donor1 ps + py bulk_linear; keep substrate identity + motif.

    site_id is `GENE_<res><pos>` (e.g. M6PR_S267); motif is already the 15-char
    `_`-padded kinase_library central-residue window. We carry only the columns
    rank_kinases needs (gene_symbol, site_position, motif) plus phos_res/Type.
    """
    frames = []
    for path in (PS_CSV, PY_CSV):
        df = pd.read_csv(path, usecols=["site_id", "gene_symbol", "motif"])
        frames.append(df)
    sites = pd.concat(frames, ignore_index=True)

    sites["motif"] = sites["motif"].astype(str)
    # Drop rows lacking a usable motif window (donor2-style empties, NaN).
    sites = sites[sites["motif"].str.len() > 0]
    sites = sites[~sites["motif"].isin(["nan", ""])].copy()

    # Central residue determines pS/pT/pY.
    center = sites["motif"].str.len() // 2
    sites["phos_res"] = [m[c] for m, c in zip(sites["motif"], center)]
    sites = sites[sites["phos_res"].isin(["S", "T", "Y"])].copy()
    sites["Type"] = "p" + sites["phos_res"]

    # site_position is parsed from site_id (`..._S267` → 267) as a merge key;
    # the driver drops it. Fall back to 0 if unparseable.
    def _pos(site_id: str) -> int:
        tail = str(site_id).split("_")[-1]
        digits = tail[1:] if tail[:1].isalpha() else tail
        return int(digits) if digits.isdigit() else 0

    sites["site_position"] = sites["site_id"].map(_pos)

    # Same site can recur across ps/py enrichment — dedup on (gene, motif).
    sites = sites.drop_duplicates(subset=["gene_symbol", "motif"])
    return sites.reset_index(drop=True)


def build_human_kldata() -> pd.DataFrame:
    print("Loading donor1 ps + py bulk_linear sites ...")
    sites = load_tcell_sites()
    print(f"  {len(sites)} unique sites  "
          f"(S={sum(sites.phos_res=='S')}, T={sum(sites.phos_res=='T')}, "
          f"Y={sum(sites.phos_res=='Y')})")

    print(f"Ranking ser_thr kinases (top {TOP_N_SER_THR} per S/T site) ...")
    st = rank_kinases(sites, "ser_thr", TOP_N_SER_THR)
    print(f"  {len(st)} (site × kinase) rows from S/T")

    print(f"Ranking tyrosine kinases (top {TOP_N_TYR} per Y site) ...")
    tyr = rank_kinases(sites, "tyrosine", TOP_N_TYR)
    print(f"  {len(tyr)} (site × kinase) rows from Y")

    combined = pd.concat([st, tyr], ignore_index=True)

    # kinase MATRIX_NAME → canonical human GENE_NAME. No homologene: both
    # substrate and kinase symbols are already human HGNC.
    import kinase_library as kl
    info = kl.get_kinome_info()[["KINASE", "GENE_NAME"]].drop_duplicates()
    combined = combined.merge(info, how="left", left_on="kinase", right_on="KINASE")
    combined["motif.geneName"] = combined["GENE_NAME"].fillna(combined["kinase"])

    # Re-merge to recover Type.
    kldata = combined.merge(
        sites[["gene_symbol", "site_position", "motif", "Type"]],
        on=["gene_symbol", "site_position", "motif"],
        how="left",
    )

    out = pd.DataFrame({
        "gene": kldata["gene_symbol"],
        "site_pos": kldata["site_position"].astype(int),
        "motif.geneName": kldata["motif.geneName"],
        "Type": kldata["Type"],
    })
    n_pre = len(out)
    out = out[out["motif.geneName"].fillna("") != ""].reset_index(drop=True)
    print(f"  {n_pre - len(out)} rows dropped (no kinase symbol); "
          f"{len(out)} rows in final kldata")
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
            "ps_bulk_linear": {
                "path": str(PS_CSV.relative_to(REPO_ROOT)),
                "sha256": _sha256(PS_CSV),
            },
            "py_bulk_linear": {
                "path": str(PY_CSV.relative_to(REPO_ROOT)),
                "sha256": _sha256(PY_CSV),
            },
        },
        "notes": (
            "Human kldata for the T-cell exhaustion cohort. Substrate sites "
            "from donor1's ps+py bulk_linear (donor2 lacks IMAC motifs). "
            "Human substrate + kinase symbols throughout; NO homologene "
            "conversion. Applied to both donors per cohort decision."
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

    out = build_human_kldata()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index_label="")
    print(f"\nWrote {args.out.relative_to(REPO_ROOT)}  ({len(out)} rows)")
    write_provenance(out)
    print(f"Wrote {PROVENANCE_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()

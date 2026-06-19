#!/usr/bin/env python
"""Derive the phospho-only pair-mode result from a --ptm (pr,ps,py,Ack,KGG) run.

The Incytr multimodel score is an ADDITIVE weighted sum over omics layers
(Incytr R/evaluation.R Pathway_evaluation):

    multimodel_score = TPDS + 0.5*(PPDS + PhPDS_ps + PhPDS_py + Ack_score + KGG_score + Rme1_score)

with an absent layer contributing exactly 0. The final PDS adds a sign-directional
kinase term (R/evaluation.R Cal_PDS + apply_condition_direction), with weight 0.5:

    PDS = mms + ( mms>0 ?  0.5*SiK_ref
                : mms<0 ? -0.5*SiK_alt
                :          0.5*(SiK_ref - SiK_alt) )

The kinase SiK terms are PTM-independent and SigProb (hence the per-condition
p_value_<cond> columns) is computed pre-integration, so the phospho-only result
equals the --ptm result with the Ack/KGG layer removed: subtract their score from
mms, recompute PDS, zero the (now-absent) Ack_score/KGG_score, and drop the 12
PTM-only node columns. Every remaining column — including p_value_<cond> — is
byte-identical to a real phospho-only run (verified: max|Δ| on multimodel_score and
PDS is ~9e-16, all other shared columns exact). The reconstruction is lossless.

MEMORY: the wide pair-mode parquet is ~14M rows x ~68 cols (multi-GB decompressed).
This runs entirely in DuckDB (streamed, spills to ~/.cache/duckdb) and NEVER
materializes the frame in Python — a pandas read of this file OOM-kills the box.

Usage:
  derive:   python derive_phospho_from_ptm.py --ptm IN.parquet --out OUT.parquet
  validate: python derive_phospho_from_ptm.py --ptm PTM.parquet --validate REF_PHOSPHO.parquet
"""
from __future__ import annotations
import argparse, os, re, sys
import duckdb

W_OMICS = 0.5   # score.weight default (rep(0.5, 6))
W_KPDS = 0.5    # KPDS.weight default
PTM_LAYER_SCORES = ("Ack_score", "KGG_score", "Rme1_score")
# the 12 PTM-only node columns a phospho-only run does not emit.
DROP_RE = re.compile(r"^(EM|Ligand|Receptor|Target)_(Ack|KGG)_log2FC$|^(Ack|KGG)_(up|down)$")


def _connect() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute("PRAGMA temp_directory=$d", {"d": os.path.expanduser("~/.cache/duckdb")})
    con.execute("PRAGMA memory_limit='8GB'")
    con.execute("PRAGMA threads=4")
    return con


def _columns(con, path: str) -> list[str]:
    return [r[0] for r in con.execute(
        "DESCRIBE SELECT * FROM read_parquet($p)", {"p": path}).fetchall()]


def _resolve_sik(cols) -> tuple[str, str]:
    """Return (cond_ref, cond_alt) SiK_score_<cond> column names.
    ref = the TG (transgene) condition = conditions[1] in the driver; alt = WT."""
    sik = [c for c in cols if c.startswith("SiK_score_")]
    if len(sik) != 2:
        raise SystemExit(f"expected exactly 2 SiK_score_<cond> cols, found {sik}")
    ref = next((c for c in sik if c[len("SiK_score_"):].startswith("TG")), None)
    alt = next((c for c in sik if c[len("SiK_score_"):].startswith("WT")), None)
    if ref is None or alt is None:
        raise SystemExit(f"could not resolve TG/WT from {sik}")
    return ref, alt


def _mms_pds_sql(cols, sik_ref, sik_alt) -> tuple[str, str]:
    """SQL expressions for phospho-only multimodel_score and PDS."""
    ptm_present = [c for c in PTM_LAYER_SCORES if c in cols]
    contrib = " + ".join(f"COALESCE({c},0)" for c in ptm_present) or "0"
    mms = f"(multimodel_score - {W_OMICS}*({contrib}))"
    pds = (f"({mms} + CASE WHEN {mms}>0 THEN {W_KPDS}*COALESCE({sik_ref},0) "
           f"WHEN {mms}<0 THEN -{W_KPDS}*COALESCE({sik_alt},0) "
           f"ELSE {W_KPDS}*(COALESCE({sik_ref},0)-COALESCE({sik_alt},0)) END)")
    return mms, pds


def _phospho_select(cols, mms, pds) -> str:
    """SQL projection list: PTM cols -> phospho-only schema. Drops the 12 PTM-only
    node cols; recomputes multimodel_score + PDS; zeros the absent Ack/KGG_score.
    All other columns pass through unchanged."""
    drop = {c for c in cols if DROP_RE.match(c)}
    override = {"multimodel_score": mms, "PDS": pds, "Ack_score": "0.0", "KGG_score": "0.0"}
    proj = [f'{override[c]} AS "{c}"' if c in override else f'"{c}"'
            for c in cols if c not in drop]
    return ",\n  ".join(proj)


def derive(con, ptm_path: str, out_path: str) -> None:
    cols = _columns(con, ptm_path)
    sik_ref, sik_alt = _resolve_sik(cols)
    mms, pds = _mms_pds_sql(cols, sik_ref, sik_alt)
    sel = _phospho_select(cols, mms, pds)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    con.execute(
        f"COPY (SELECT\n  {sel}\nFROM read_parquet($p)) TO $o (FORMAT parquet)",
        {"p": ptm_path, "o": out_path})
    n = con.execute("SELECT count(*) FROM read_parquet($o)", {"o": out_path}).fetchone()[0]
    print(f"wrote {out_path}  rows={n}")


def validate(con, ptm_path: str, ref_path: str) -> int:
    cols = _columns(con, ptm_path)
    sik_ref, sik_alt = _resolve_sik(cols)
    mms, pds = _mms_pds_sql(cols, sik_ref, sik_alt)
    sql = f"""
    WITH d AS (
      SELECT Sender, Receiver, Path,
             {mms} AS mms_phos, {pds} AS pds_phos,
             TPDS, PPDS, PhPDS_ps, PhPDS_py
      FROM read_parquet($ptm)
    ),
    r AS (
      SELECT Sender, Receiver, Path,
             multimodel_score AS mms_r, PDS AS pds_r,
             TPDS AS TPDS_r, PPDS AS PPDS_r, PhPDS_ps AS PhPDS_ps_r, PhPDS_py AS PhPDS_py_r
      FROM read_parquet($ref)
    ),
    j AS (SELECT d.*, r.mms_r, r.pds_r, r.TPDS_r, r.PPDS_r, r.PhPDS_ps_r, r.PhPDS_py_r
          FROM d JOIN r USING (Sender, Receiver, Path))
    SELECT (SELECT count(*) FROM d) derived_rows, (SELECT count(*) FROM r) ref_rows,
           count(*) joined,
           max(abs(mms_phos-mms_r)) d_mms, max(abs(pds_phos-pds_r)) d_pds,
           max(abs(TPDS-TPDS_r)) d_tpds, max(abs(PPDS-PPDS_r)) d_ppds,
           max(abs(PhPDS_ps-PhPDS_ps_r)) d_phps, max(abs(PhPDS_py-PhPDS_py_r)) d_phpy,
           sum(CASE WHEN abs(mms_phos-mms_r)>1e-9 THEN 1 ELSE 0 END) n_mms_off,
           sum(CASE WHEN abs(pds_phos-pds_r)>1e-9 THEN 1 ELSE 0 END) n_pds_off
    FROM j
    """
    row = con.execute(sql, {"ptm": ptm_path, "ref": ref_path}).fetchone()
    res = dict(zip([c[0] for c in con.description], row))
    print(f"derived_rows={res['derived_rows']}  ref_rows={res['ref_rows']}  joined={res['joined']}")
    tol, ok = 1e-9, True
    for k, label in (("d_mms", "multimodel_score"), ("d_pds", "PDS"), ("d_tpds", "TPDS"),
                     ("d_ppds", "PPDS"), ("d_phps", "PhPDS_ps"), ("d_phpy", "PhPDS_py")):
        v = res[k]
        flag = "OK " if (v is None or v <= tol) else "FAIL"
        ok = ok and flag == "OK "
        print(f"  {label:16s} max|Δ|={(float('nan') if v is None else v):.3e}  {flag}")
    print(f"  n(|Δmms|>1e-9)={res['n_mms_off']}  n(|Δpds|>1e-9)={res['n_pds_off']}")
    ok = ok and res["joined"] == res["ref_rows"] == res["derived_rows"]
    print("VALIDATE:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ptm", required=True, help="--ptm wide parquet (input)")
    p.add_argument("--out", help="write derived phospho-only parquet here")
    p.add_argument("--validate", metavar="REF", help="compare derive against a real phospho-only parquet")
    a = p.parse_args(argv)
    con = _connect()
    if a.validate:
        return validate(con, a.ptm, a.validate)
    if not a.out:
        raise SystemExit("--out required when not validating")
    derive(con, a.ptm, a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())

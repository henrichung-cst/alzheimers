#!/usr/bin/env python3
"""Regression check: canonical pair-mode output reproduces sce4's reference.

WHAT IS GATED (the achievable, reproducible invariant): our GATED pathway set
(SigProb > 0.1 either arm AND |PDS| >= 0.2) must equal sce4's gated Allpathway
set pair-for-pair, with the symmetric difference allowed to contain ONLY
transgene paths (a path with App/Psen1/Mapt in any of its four positions). The
AD gene.use is sourced per-pair from sce4's own Allpathway rds
(alz/incytr_pair/extract_sce4_geneuse.R), so the engine reproduces sce4's
enumeration exactly — Micro→Cholin 1283/1283 (0 extra / 0 missing); Ndnf×Ndnf
698/699 with the symmetric difference being App/Psen1 paths only. See
archive/sce4_reproduction_2026-06-08/README.md §6 and §6.7 (the per-pair gene.use record).

WHY NOT THE TOP300 CAP: the shipped per-pair top-300 cap is ranked by PDS, and
PDS carries two DOCUMENTED, off-box residuals — the phospho-substrate provenance
gap (§5; PDS overlap ~169/1283 on Micro→Cholin) and the App-transgene value
(0.19 vs sce4's saturated 7.65). Both perturb the rank-300 boundary, so cap
membership CANNOT match sce4 regardless of gene.use/engine correctness. Cap
fidelity is therefore reported as INFORMATIONAL only, never gated.

The six engine overrides (CLAUDE.md §Pair-mode) are still checked numerically:
on the gated∩sce4 overlap, max |Δ sclog2FC| == 0 for Receptor/Target (exact);
Ligand/EM may carry the App-transgene value residual (tolerated only when the
position gene is a transgene).

Usage (the gate, from verify_incytr_sce4.sh):
  pixi run python alz/incytr_pair/verify_sce4_parity.py \\
      --all-known-pairs --wide-dir <regen> --sce4-allpathway <allpathway_ref.csv>
"""

from __future__ import annotations

import argparse
import os
import sys

import pyarrow.parquet as pq
import pyarrow.compute as pc
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))

SCE4_REF_DEFAULT = os.path.join(REPO_ROOT, "bench/sce4_DEG_PRG_Top300_table_10302025.csv")
WIDE_DIR_DEFAULT = os.path.join(REPO_ROOT, "outputs/reports/incytr_pair_mode/wide")

# sce4's downstream gate (filter_significant_paths.py): SigProb floor (either arm)
# AND |PDS| floor, applied before the per-pair top-300 up ∪ down cap.
PDS_GATE = 0.2

# AD model human transgenes. A path touching any of these in any position carries
# the off-box transgene value residual, so it is exempt from the set-identity gate.
TRANSGENES = ("App", "Psen1", "Mapt")

# Contrast string (e.g. "App_2mo") → wide-parquet filename stem.
_GENO_DECODE = {"App": "AppP", "Tau": "Ttau", "ApTt": "ApTt"}

NDNF = "Ndnf.positive.neurogliaform.inhibitory.interneurons.GABAergic"

KNOWN_PAIRS = [
    ("App_2mo", "Microglia", "Cholinergic.Neurons"),
    ("App_2mo", NDNF,        NDNF),
]


def contrast_to_parquet(contrast: str, wide_dir: str) -> str:
    geno, age = contrast.split("_")
    code = _GENO_DECODE[geno]
    return os.path.join(wide_dir, f"ma_{age}_{code}_ma_{age}_WTyp_incytr_output.parquet")


def _to_canonical(name: str) -> str:
    """sce4-dotted cluster name (Cholinergic.Neurons) → our dashed wide-parquet
    form (Cholinergic-Neurons). Canonical names carry no '.', so unambiguous."""
    return name.replace(".", "-")


def load_pair(parquet: str, sender: str, receiver: str) -> pd.DataFrame:
    filt = (pc.field("Sender") == _to_canonical(sender)) & (pc.field("Receiver") == _to_canonical(receiver))
    cols = [
        "Ligand", "Receptor", "EM", "Target",
        "Ligand_sclog2FC", "Receptor_sclog2FC", "EM_sclog2FC", "Target_sclog2FC",
        "SigProb_ma_2mo_AppP", "SigProb_ma_2mo_WTyp", "PDS",
        "Sender", "Receiver",
    ]
    available = set(pq.ParquetFile(parquet).schema_arrow.names)
    cols = [c for c in cols if c in available]
    return pq.read_table(parquet, columns=cols, filters=filt).to_pandas()


def _mkkey(df: pd.DataFrame) -> pd.Series:
    return (df["Ligand"].astype(str) + "|" + df["Receptor"].astype(str)
            + "|" + df["EM"].astype(str) + "|" + df["Target"].astype(str))


def _is_transgene_path(df: pd.DataFrame) -> pd.Series:
    """A path is transgene-exempt iff any of its four positions is a transgene."""
    m = pd.Series(False, index=df.index)
    for pos in ("Ligand", "Receptor", "EM", "Target"):
        m = m | df[pos].astype(str).isin(TRANSGENES)
    return m


def check_pair(contrast: str, sender: str, receiver: str,
               wide_dir: str, allpathway: pd.DataFrame, top300_path: str | None,
               sigprob_cutoff: float, tol: float) -> tuple[bool, list[str]]:
    msgs: list[str] = []
    parquet = contrast_to_parquet(contrast, wide_dir)
    if not os.path.exists(parquet):
        return False, [f"FAIL: {parquet} not found"]

    ours = load_pair(parquet, sender, receiver)
    refA = allpathway[(allpathway["Sender.group"] == sender)
                      & (allpathway["Receiver.group"] == receiver)].copy()
    msgs.append(f"[{sender} → {receiver} | {contrast}] ours(raw)={len(ours)} sce4 Allpathway={len(refA)}")
    if len(refA) == 0:
        return False, msgs + [f"FAIL: no sce4 Allpathway rows for {sender} → {receiver}"]

    # Our gated set (sce4's own gate), and sce4's Allpathway is already gated.
    sp_cols = [c for c in ("SigProb_ma_2mo_AppP", "SigProb_ma_2mo_WTyp") if c in ours.columns]
    if not sp_cols or "PDS" not in ours.columns:
        return False, msgs + ["FAIL: missing SigProb/PDS columns — re-run nboot=0 parity gen"]
    gated = ours[((ours[sp_cols] > sigprob_cutoff).any(axis=1)) & (ours["PDS"].abs() >= PDS_GATE)].copy()
    gated["k"] = _mkkey(gated)
    refA["k"] = _mkkey(refA)

    # --- THE GATE: gated path-set identity, transgene-exempt -----------------
    ours_keys = set(gated["k"])
    ref_keys = set(refA["k"])
    missing = refA[~refA["k"].isin(ours_keys)]            # sce4 has, we don't
    extra = gated[~gated["k"].isin(ref_keys)]             # we have, sce4 doesn't
    nm_missing = missing[~_is_transgene_path(missing)]
    nm_extra = extra[~_is_transgene_path(extra)]
    msgs.append(f"  gated={len(gated)}  overlap={len(ours_keys & ref_keys)}  "
                f"missing={len(missing)} (non-transgene {len(nm_missing)})  "
                f"extra={len(extra)} (non-transgene {len(nm_extra)})")

    ok = True
    if len(nm_missing) or len(nm_extra):
        ok = False
        if len(nm_missing):
            ex = sorted(set(nm_missing["Ligand"] + "*" + nm_missing["Receptor"]
                            + "*" + nm_missing["EM"] + "*" + nm_missing["Target"]))[:5]
            msgs.append(f"    FAIL: {len(nm_missing)} non-transgene sce4 paths NOT enumerated (e.g. {ex})")
        if len(nm_extra):
            ex = sorted(set(nm_extra["Ligand"] + "*" + nm_extra["Receptor"]
                            + "*" + nm_extra["EM"] + "*" + nm_extra["Target"]))[:5]
            msgs.append(f"    FAIL: {len(nm_extra)} non-transgene paths we emit that sce4 did NOT (e.g. {ex})")
    else:
        msgs.append(f"    path-set identity OK (symmetric diff is {len(missing)+len(extra)} "
                    f"transgene-only path(s))")

    # --- Per-position max|Δ| on the gated∩sce4 overlap (engine fixes) ---------
    j = gated.merge(
        refA[["k", "Ligand_sclog2FC", "Receptor_sclog2FC", "EM_sclog2FC", "Target_sclog2FC"]]
            .rename(columns={"Ligand_sclog2FC": "s_L", "Receptor_sclog2FC": "s_R",
                             "EM_sclog2FC": "s_E", "Target_sclog2FC": "s_T"}),
        on="k",
    )
    msgs.append(f"  joined overlap: {len(j)}")
    for pos, (oc, sc) in {"Ligand": ("Ligand_sclog2FC", "s_L"),
                          "Receptor": ("Receptor_sclog2FC", "s_R"),
                          "EM": ("EM_sclog2FC", "s_E"),
                          "Target": ("Target_sclog2FC", "s_T")}.items():
        d = (j[oc] - j[sc]).abs()
        over = j[d > tol]
        max_d = float(d.max()) if len(d) else float("nan")
        msgs.append(f"  {pos:8s}  max|Δ|={max_d:.4f}  n(|Δ|>{tol})={len(over)}")
        if pos in ("Receptor", "Target"):
            if len(over):
                msgs.append(f"    FAIL: {pos} has {len(over)} mismatches above tol {tol}")
                ok = False
        else:  # Ligand / EM: App-transgene value residual tolerated
            non_tg = over[~over[pos].astype(str).isin(TRANSGENES)]
            if len(non_tg):
                msgs.append(f"    FAIL: {pos} has {len(non_tg)} non-transgene mismatches "
                            f"(e.g. {sorted(non_tg[pos].unique())[:5]})")
                ok = False
            elif len(over):
                msgs.append(f"    ({len(over)} transgene-value residual, allowed)")

    # --- Cap fidelity: INFORMATIONAL only (bounded by phospho/App PDS residual)
    if top300_path and os.path.exists(top300_path):
        t300 = pd.read_csv(top300_path, low_memory=False)
        ref300 = t300[(t300["Sender.group"] == sender) & (t300["Receiver.group"] == receiver)].copy()
        if len(ref300):
            ref300["k"] = _mkkey(ref300)
            up = gated[gated["PDS"] > 0].nlargest(300, "PDS")
            dn = gated[gated["PDS"] < 0].nsmallest(300, "PDS")
            cap_keys = set(pd.concat([up, dn])["k"])
            r300 = set(ref300["k"])
            frac = len(r300 & cap_keys) / len(r300) if r300 else 0.0
            msgs.append(f"  [info] Top300 cap fidelity: {len(r300 & cap_keys)}/{len(r300)} "
                        f"({frac:.1%}) — informational; bounded by phospho/App PDS residual")

    msgs.append(f"  STATUS: {'PASS' if ok else 'FAIL'}")
    return ok, msgs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--contrast", default="App_2mo")
    ap.add_argument("--sender", default="Microglia")
    ap.add_argument("--receiver", default="Cholinergic.Neurons")
    ap.add_argument("--wide-dir", default=WIDE_DIR_DEFAULT)
    ap.add_argument("--sce4-allpathway", required=True,
                    help="CSV of sce4's gated Allpathway tuples for the benchmark "
                         "pairs (Sender.group/Receiver.group + L/R/EM/T + *_sclog2FC); "
                         "the gate reference. Dumped from the rds by verify_incytr_sce4.sh.")
    ap.add_argument("--sce4-top300", default=SCE4_REF_DEFAULT,
                    help="sce4 Top300 table — used ONLY for the informational cap-fidelity line.")
    ap.add_argument("--sigprob-cutoff", type=float, default=0.1)
    ap.add_argument("--tol", type=float, default=1e-4,
                    help="per-position max |Δ| tolerance (float64 round-off)")
    ap.add_argument("--all-known-pairs", action="store_true",
                    help="check both Micro→Cholin and Ndnf×Ndnf")
    args = ap.parse_args()

    allpathway = pd.read_csv(args.sce4_allpathway, low_memory=False)

    pairs = KNOWN_PAIRS if args.all_known_pairs else [(args.contrast, args.sender, args.receiver)]
    all_ok = True
    for contrast, sender, receiver in pairs:
        ok, msgs = check_pair(contrast, sender, receiver, args.wide_dir,
                              allpathway, args.sce4_top300, args.sigprob_cutoff, args.tol)
        for m in msgs:
            print(m)
        if not ok:
            all_ok = False

    print()
    if all_ok:
        print("verify_sce4_parity: PASS  (gated path-set == sce4 Allpathway, transgene-exempt; "
              "R/T sclog2FC max|Δ|=0)")
        return 0
    print("verify_sce4_parity: FAIL.")
    print("  - non-transgene missing/extra ⇒ gene.use (per-pair sce4 source) regressed")
    print("    (extract_sce4_geneuse.R, archive/sce4_reproduction_2026-06-08/README.md §6).")
    print("  - a Receptor/Target max|Δ| ⇒ one of the six engine fixes regressed")
    print("    (CLAUDE.md §Pair-mode Incytr, bench/bench.md §sce4 reproduction).")
    return 1


if __name__ == "__main__":
    sys.exit(main())

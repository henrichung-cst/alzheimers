"""Anchor-calibrated novelty scan of the donor1 kinase-MEA surface.

For every kinase on each track, take whichever sign dominates its 5 day-vs-d2
contrasts and ask the AGREE condition without a predicted direction:

    coherent  = dominant sign holds in >=3/5 contrasts
    significant = FDR < 0.25 in >=1 contrast
    hit       = coherent AND significant   (the AGREE-analog, direction free)

FDR is saturated on the ST track (median min-FDR = 0), so |NES| magnitude is the
real discriminator. We rank every kinase by per-kinase mean |NES| within its
track and mark the a priori known-mechanism kinases (ANCHORS). The novelty read:
non-anchor kinases landing in the same |NES| tail as the anchors are candidate
novel mechanisms -- they share the enrichment level of the known true positives.
"""
import numpy as np
import pandas as pd
from pathlib import Path

D = Path("outputs/reports/kinase_attribution_tcells/donor1/mea")
THRESHOLDS = [0.25, 0.1, 0.05]
CONTRASTS = ["D1_d13", "D1_d15", "D1_d17", "D1_d19", "D1_d20"]

TRACKS = {
    "py": dict(nes="kinase_timepoint_nes_pY.csv", fdr="kinase_timepoint_fdr_pY.csv"),
    "st": dict(nes="kinase_timepoint_nes.csv", fdr="kinase_timepoint_fdr.csv"),
}


# |NES| tail cut: FDR is saturated on ST, so magnitude is the discriminator.
# A kinase is in the tail if its per-kinase mean |NES| sits in the top TAILS
# quantile OF ITS OWN TRACK -- i.e. it stands above the track's global shift.
TAILS = {"top_quartile": 0.75, "top_decile": 0.90}

# A priori known-mechanism kinases (docs/reference/tcell_kinase_mea_apriori.md),
# keyed by track. These are the true-positive anchors we calibrate against.
ANCHORS = {
    "py": {
        "KY1 proximal-TCR": ["LCK", "FYN", "ZAP70", "SYK", "ITK", "TEC", "TXK", "CSK"],
        "KY2 JAK": ["JAK1", "JAK2", "JAK3", "TYK2"],
    },
    "st": {
        "KS1 anabolic": ["AKT1", "AKT2", "AKT3", "PDK1", "MTOR", "P70S6K", "RSK2",
                         "PKCT", "MEK1", "MEK2", "ERK1", "ERK2", "CAMK2A", "CAMK2B",
                         "CAMK2D", "CAMK2G", "CAMK4", "IKKA", "IKKB"],
        "KS2 stress/energy": ["AMPKA1", "AMPKA2", "P38A", "JNK1", "JNK2", "JNK3", "GSK3B"],
        "KS3 proliferation": ["CDK1", "CDK2", "AURA", "AURB", "PLK1"],
        "KSc CK2": ["CK2A1", "CK2A2"],
    },
}
ANCHOR_BLOCK = {k: blk for tr in ANCHORS for blk, ks in ANCHORS[tr].items() for k in ks}


def scan(track):
    t = TRACKS[track]
    nes = pd.read_csv(D / t["nes"]).set_index("kinase")[CONTRASTS]
    fdr = pd.read_csv(D / t["fdr"]).set_index("kinase")[CONTRASTS]
    rows = []
    for k in nes.index:
        v = nes.loc[k].to_numpy(float)
        f = fdr.loc[k].to_numpy(float)
        n_pos = int(np.nansum(v > 0))
        n_neg = int(np.nansum(v < 0))
        row = dict(track=track, kinase=k, dom=("up" if n_pos >= n_neg else "down"),
                   coherent=max(n_pos, n_neg) >= 3, absmean=float(np.nanmean(np.abs(v))),
                   min_fdr=float(np.nanmin(f)) if np.isfinite(f).any() else np.nan,
                   anchor=k in ANCHOR_BLOCK, block=ANCHOR_BLOCK.get(k, ""))
        for thr in THRESHOLDS:
            row[f"hit_{thr}"] = row["coherent"] and int(np.nansum(f < thr)) >= 1
        rows.append(row)
    df = pd.DataFrame(rows)
    df["pct"] = df.absmean.rank(pct=True)  # per-track |NES| percentile
    for name, q in TAILS.items():
        df[name] = df.absmean >= df.absmean.quantile(q)  # per-track cutoff
    return df


all_rows = pd.concat([scan(tr) for tr in TRACKS], ignore_index=True)

for tr in list(TRACKS) + ["POOLED"]:
    sub = all_rows if tr == "POOLED" else all_rows[all_rows.track == tr]
    print(f"\n=== {tr} (n={len(sub)} kinases, all coherent >=3/5) ===")
    for thr in THRESHOLDS:
        h = sub[sub[f"hit_{thr}"]]
        print(f"  HIT @ FDR<{thr:<4}: {len(h):>3}"
              f"   [up {int((h.dom=='up').sum())} / down {int((h.dom=='down').sum())}]")
    for name in TAILS:
        t_ = sub[sub[name]]
        both = int((t_[f"hit_{THRESHOLDS[-1]}"]).sum())
        print(f"  |NES| {name:<12}: {len(t_):>3}"
              f"   [up {int((t_.dom=='up').sum())} / down {int((t_.dom=='down').sum())}]"
              f"   ({both} also FDR<{THRESHOLDS[-1]})")


def fmt(r):
    return (f"    {r.kinase:<9} {r.dom:>4}  |NES|={r.absmean:.2f}  pct={r.pct*100:4.0f}  "
            f"minFDR={r.min_fdr:.3f}" + (f"  [{r.block}]" if r.anchor else ""))


# Anchor landing + novel candidates, per real track (pY / ST).
for tr in TRACKS:
    sub = all_rows[all_rows.track == tr].sort_values("absmean", ascending=False)
    anchors = sub[sub.anchor]
    a_in_tq = anchors[anchors.top_quartile]
    print(f"\n########## {tr}: anchor landing ##########")
    print(f"  {len(anchors)} anchors present; {len(a_in_tq)} in top quartile "
          f"(|NES| pct {anchors.pct.min()*100:.0f}-{anchors.pct.max()*100:.0f})")
    for _, r in anchors.iterrows():
        print(fmt(r))
    # candidates: non-anchor kinases in the anchors' band (top quartile).
    cand = sub[(~sub.anchor) & sub.top_quartile]
    print(f"\n  novel candidates (non-anchor, top-quartile |NES|): {len(cand)}")
    for _, r in cand.iterrows():
        print(fmt(r))

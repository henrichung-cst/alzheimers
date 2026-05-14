"""Generate data/external/sea_ad/cluster_to_seaad_supertype.csv.

For each of the 19 spine clusters, list SEA-AD MTG supertype IDs to average,
with equal weight. Subcortical / hippocampal clusters get an `n/a` row with a
reason. The mapping is hand-curated at the SEA-AD Subclass level, then
expanded to supertypes via the var-table of effect_sizes.h5ad.
"""

from __future__ import annotations
from pathlib import Path
import anndata as ad
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
H5AD = REPO / "data/external/sea_ad/effect_sizes.h5ad"
OUT = REPO / "data/external/sea_ad/cluster_to_seaad_supertype.csv"

# cluster_name -> list of SEA-AD Subclass labels (None == n/a)
SUBCLASS_MAP: dict[str, list[str] | None] = {
    "Erbb4-VIP-inhibitory-neurons": ["Vip"],
    "Astrocytes": ["Astrocyte"],
    "Oligodendrocytes": ["Oligodendrocyte"],
    "Excitatory-Pyramidal-Satb2-Cux2": ["L2/3 IT"],
    "Striatal-medium-spiny-neuron": None,                       # subcortical (striatum)
    "Excitatory-Rorb": ["L4 IT", "L5 IT"],
    "Foxp2-Excitatory-Neurons-layers-6-and-2-3": ["L6 CT"],
    "Excitatory-Pyramidal": ["L5 IT", "L6 IT"],
    "Microglia": ["Microglia-PVM"],
    "glutamatergic-excitatory-neurons": None,                   # too generic to bridge confidently
    "OPC": ["OPC"],
    "Excitatory principal neurons in the hippocampal dentate gyrus": None,  # hippocampal-DG
    "Erbb4-inhibitory-neurons": ["Sncg", "Vip"],                # broad CGE-ish; flagged as low-confidence
    "Excitatory-neurons": None,                                 # too generic
    "Endothelial-cell": ["Endothelial"],
    "VIP-positive-interneuron": ["Vip"],
    "Reln-neurons": ["Lamp5", "Pax6"],
    "Pericyte": ["VLMC"],                                       # closest mural-cell match in MTG taxonomy
    "Ndnf-positive-neurogliaform-inhibitory-interneurons-GABAergic": ["Lamp5"],
}

NA_REASONS = {
    "Striatal-medium-spiny-neuron": "subcortical-striatum",
    "glutamatergic-excitatory-neurons": "too-generic",
    "Excitatory principal neurons in the hippocampal dentate gyrus": "hippocampal-DG",
    "Excitatory-neurons": "too-generic",
}


def main() -> None:
    a = ad.read_h5ad(H5AD, backed="r")
    var = a.var.copy()
    var["Supertype"] = var.index.astype(str)
    sub2supers = var.groupby("Subclass", observed=True)["Supertype"].apply(list).to_dict()

    rows = []
    for cluster, subclasses in SUBCLASS_MAP.items():
        if subclasses is None:
            rows.append({
                "cluster_name": cluster,
                "seaad_supertype": "n/a",
                "weight": 0.0,
                "notes": NA_REASONS.get(cluster, "unmapped"),
            })
            continue
        supertypes = []
        for sc in subclasses:
            if sc not in sub2supers:
                raise KeyError(f"SEA-AD Subclass not found: {sc!r}")
            supertypes.extend(sub2supers[sc])
        w = 1.0 / len(supertypes)
        note = "subclasses=" + "+".join(subclasses)
        for st in supertypes:
            rows.append({
                "cluster_name": cluster,
                "seaad_supertype": st,
                "weight": round(w, 6),
                "notes": note,
            })

    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    print(f"wrote {OUT.relative_to(REPO)} ({len(df)} rows)")
    print(f"distinct clusters: {df['cluster_name'].nunique()}")
    mapped = df[df["seaad_supertype"] != "n/a"]["cluster_name"].nunique()
    na = df[df["seaad_supertype"] == "n/a"]["cluster_name"].nunique()
    print(f"  mapped: {mapped}    n/a: {na}")


if __name__ == "__main__":
    main()

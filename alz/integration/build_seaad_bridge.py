"""Generate data/derived/bridges/cluster_to_seaad_supertype.csv.

For each of the 31 Levy-t5 spine clusters, list SEA-AD MTG supertype IDs to
average, with equal weight. Subcortical / hippocampal / non-MTG clusters get
an `n/a` row with a reason. The mapping is hand-curated at the SEA-AD Subclass
level, then expanded to supertypes via the var-table of effect_sizes.h5ad.
"""

from __future__ import annotations
from pathlib import Path
import sys
import anndata as ad
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from alz.shared import config  # noqa: E402

H5AD = Path(config.SEA_AD_DIR) / "effect_sizes.h5ad"
OUT = Path(config.CLUSTER_TO_SEAAD_SUPERTYPE_FILE)

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
    "Ptprz1-protoplasmic-astrocytes": ["Astrocyte"],
    "Basal-Ganglia-GABAergic-Neurons": None,                              # subcortical (basal ganglia)
    "Vascular-Leptomeningeal-Cells": ["VLMC"],
    "Inhibitory-Neurons": None,                                           # too-generic
    "Ependymal-cell": None,                                               # not present in SEA-AD MTG taxonomy
    "GABAergic-inhibitory-interneurons-Dlx6os1-Erbb4": ["Sncg", "Vip"],   # mirrors Erbb4-inhibitory-neurons
    "GABAergic inhibitory interneurons": None,                            # too-generic
    "Excitatory-neurons-Cajal-Retzius-cells-layer-I-Reelin": None,        # CR cells absent from MTG supertypes; Reln-neurons row covers Reelin+
    "Choroid-Plexus-Epithelial-Cells": None,                              # not present in SEA-AD MTG taxonomy
    "Glutamatergic-excitatory-neurons-Cortical-layer-2-4-pyramidal-neurons": ["L2/3 IT", "L4 IT"],
    "GABAergic-inhibitory-interneurons-VIP-positive": ["Vip"],
    "Cholinergic-Neurons": None,                                          # subcortical / brainstem cholinergic
}

NA_REASONS = {
    "Striatal-medium-spiny-neuron": "subcortical-striatum",
    "glutamatergic-excitatory-neurons": "too-generic",
    "Excitatory principal neurons in the hippocampal dentate gyrus": "hippocampal-DG",
    "Excitatory-neurons": "too-generic",
    "Basal-Ganglia-GABAergic-Neurons": "subcortical-basal-ganglia",
    "Inhibitory-Neurons": "too-generic",
    "Ependymal-cell": "not-in-MTG-taxonomy",
    "GABAergic inhibitory interneurons": "too-generic",
    "Excitatory-neurons-Cajal-Retzius-cells-layer-I-Reelin": "CR-cells-not-in-MTG-supertypes",
    "Choroid-Plexus-Epithelial-Cells": "not-in-MTG-taxonomy",
    "Cholinergic-Neurons": "subcortical-or-brainstem",
}


def main() -> None:
    # Coverage assert: SUBCLASS_MAP must enumerate every cluster in the active
    # spine. Drift between the spine and this hand-curated map would silently
    # drop clusters from SEA-AD evidence; fail loudly instead.
    spine = set(config.CLUSTER_SPINE)
    mapped = set(SUBCLASS_MAP)
    missing = spine - mapped
    extra = mapped - spine
    if missing or extra:
        raise AssertionError(
            f"SUBCLASS_MAP/spine mismatch — missing={sorted(missing)} "
            f"extra={sorted(extra)}"
        )
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

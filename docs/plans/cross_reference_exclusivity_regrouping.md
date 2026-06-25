# Cross-reference tissue-exclusivity regrouping (confidence-pill recalculation)

**Status:** IMPLEMENTED. This section records the final design as built.
**Goal (user's words):** *"Recalculate the confidence pills to account for all attribution
information and cleanly group kinases into those highly and exclusively expressed in one
tissue type across all references and correlated information."* + *"keep the reference data
corroborative and prioritize using within cohort metrics."*

**Final design (as built):**
1. **The confidence pill keeps its vocabulary** — `none / low / moderate / high / very_high`.
   Only its *meaning* is recalculated: the top tiers now mean "highly and exclusively
   expressed in one cell type." (Not a new `exclusive/enriched/...` scale — that was an
   early draft, scrapped.)
2. **Within-cohort Song is primary and sets the tier; reference data only corroborates.**
   WMB (mouse atlas) and human (SEA-AD/HBCA, when its location is strong) can promote a tier
   by one step but are never required and never veto.
3. **Resolution = curated specificity units** (`config.load_specificity_unit_map`), one
   mechanism handling three cases per WMB class:
   - **collapse** (`SPECIFICITY_COLLAPSE_WMB_CLASSES` = `01 IT-ET Glut`, `06 CTX-CGE GABA`,
     `09 CNU-LGE GABA`, `02 NP-CT-L6b Glut`): Song over-splits one cell type into subtypes →
     fold its clusters into the WMB-class parent unit.
   - **combined label**: a WMB class with one Song cluster keeps the cluster, labelled
     `"cluster · class"` (e.g. `Microglia · 34 Immune`).
   - **split**: a WMB class whose clusters are genuinely distinct cell types stays split
     (`33 Vascular`, `30 Astro-Epen`, `31 OPC-Oligo`).
   → 17 units over 31 clusters. Exclusivity `eff = 1/Σ unit_share²`; corroboration at the WMB
   class of the dominant unit's home cluster (the references have no cluster vocabulary). The
   collapse is never silent — the viewer renders a collapsed unit as an expandable parent.
4. The prior disease-**direction** concordance tier is preserved as `direction_tier` /
   `direction_basis` and shown in the pill tooltip ("account for all attribution information").

**History note.** Two earlier resolutions were tried and superseded: (a) a flat 9-tissue-lineage
rollup (`CLASS_TO_TISSUE_CATEGORY`) — too coarse, it merged distinct cell types (vascular);
(b) the raw 31 native clusters — too fine, it over-split one cell type (excitatory → 6 subtypes)
and penalized pan-class kinases. The curated unit map is the principled middle, decided
per WMB class. Neither prior mode is kept (no dual system).

---

## 1. What the pills mean today (the thing being recalculated)

`confidence_tier ∈ {none, low, moderate, high, very_high}` is produced by
`alz/bulk_mea/confidence.py:assign_confidence`. Its axis is **disease-direction
concordance**, not expression specificity:

- `high` = within-cohort Song supports the *bulk MEA direction* (|Song LFC| > 0.1) **and**
  Song detects the kinase at that cell type.
- `very_high` = `high` + the decomposition layer agrees in sign (decomp FDR < 0.25).
- `moderate` / `low` = weaker or SEA-AD-only direction support.

So a kinase scores "high" because its *activity goes the same way in disease* across
bulk + snRNA — it says nothing about whether the kinase is **restricted to one cell type**.
That is the axis the user is asking for. This is a genuine pivot of meaning, not a tweak.

Pills render via two CSS families (no change needed to either — both already have a 4-step
green→amber→grey ladder, `vhi`/`hi`/`mid`/`lo` and `attr-conf-*`):

| Surface | Class family | Reads |
|---|---|---|
| Audit → attribution-verdict table, drawer | `attr-conf-{tier}` | `confidence_tier`, `confidence_basis` |
| Kinase Explorer "Conf" + "Cell types" columns | `badge {vhi/hi/mid/lo}` | `confidence_tier` (max over scope) |
| 5xFAD verdict + drawer | `attr-conf-{tier}` | `confidence_tier` |
| Human tab "Conf" | `badge {hi/mid/lo}` | computed live, no `very_high` |
| tcell viewer | `badge` on `tcell_concentration_tier` | (already specificity, not `confidence_tier`) |

---

## 2. The signals available, and the one shared label space

Per-kinase specificity signals that already exist (✓ = on the unified row;
△ = computed by the producer but **not forwarded** to `unified_attribution_full.csv`):

| Reference | detected | conc_tier | effective_n | top cell type | location score |
|---|---|---|---|---|---|
| **Song** (snRNA, within-cohort, 31 Levy clusters) | ✓ `song_detected` | ✓ `song_concentration_tier` | ✓ `song_effective_n` | ✓ `song_top_celltype` | — |
| **WMB** (mouse atlas, 9 classes) | ✓ `wmb_detected` | ✓ `wmb_concentration_tier` | △ | △ | — |
| **SEA-AD** (human MTG) | — | — | — | — | ✓ `seaad_location_score` |
| **HBCA** (human brain) | — | — | — | — | ✓ `hbca_location_score` |
| (merged human) | — | — | — | — | ✓ `human_location_score = max(seaad,hbca)` |
| **NSCLC** | (T-cell track only — not in the mouse unified table) | | | | |

**Shared "tissue type" space — the load-bearing fact:** there is no single cell-type
vocabulary, but every brain reference collapses to the **9 tissue categories** in
`config.CLASS_TO_TISSUE_CATEGORY` (Excitatory neurons, Interneurons, Subcortical neurons,
Brainstem neurons, Cerebellum, Astrocytes, Oligodendrocytes, Endothelial cells, Microglia):

- Song cluster → (`cluster_to_wmb_class.csv`) → WMB class → tissue category.
- WMB class → tissue category directly.
- SEA-AD / HBCA supertypes → rolled onto Levy clusters → WMB class → tissue category.

So "the same tissue type across all references" is well-defined at the 9-category level.
SEA-AD/HBCA contribute only a *log2-over-brain-mean location score* (no detection gate), so
human references vote on the **dominant** tissue and on whether it is "strong" (≥ 1.0
log2), but they cannot contribute a concentration tier.

**No existing code compares top cell type across references** — this grouping is net-new.

---

## 3. The recalculated confidence tier (as built)

Computed per kinase in `alz/bulk_mea/specificity_class.py:assign_specificity_class`, run
right after `assign_confidence` in `attribute.py`, over the curated specificity units:

- **Song** (within-cohort, primary): fold detected-set `song_concentration` onto the units and
  take the **effective number of units** `eff = 1/Σ unit_share²`. Dominant unit = max share;
  its top detected child cluster = `specificity_celltype`.
- **WMB** (corroborator): agrees when its true top class `wmb_top_celltype` (forwarded from
  `evidence.py`) equals the WMB class of the dominant unit's home cluster.
- **Human** (corroborator): agrees when the cluster with the max `human_location_score` has
  the same WMB class as the dominant unit's home cluster, counted only when that score is
  **strong** (≥ 1.0 log2).
- `corroborated` = WMB **or** human agrees at the WMB-class level.

| Pill | Rule | Example |
|---|---|---|
| **`very_high`** | Song `eff ≤ 1.5` **and** corroborated | very_high · Microglia (BTK) |
| **`high`** | (Song `eff ≤ 3` **and** corroborated) **or** (Song `eff ≤ 1.5`, uncorroborated) | high · Excitatory neurons (IT/ET) (CDK9, collapsed) |
| **`moderate`** | Song `eff ≤ 3`, no reference corroborates | moderate · Cholinergic-Neurons (AKT1) |
| **`low`** | detected but `eff > 3` (broadly expressed) | low (CAMK2A, GRK5) |
| **`none`** | not detected in within-cohort Song | — |

Within-cohort Song alone can reach `high` (`eff ≤ 1.5`); references only ever **promote** by
corroborating the cell class — never required, never a veto. Output columns: `confidence_tier`,
`confidence_basis`, `specificity_unit`, `specificity_unit_label`, `specificity_celltype` (home
child cluster), `specificity_collapsed`, plus the preserved `direction_tier` / `direction_basis`.

Observed per-kinase distribution (389 kinases): **very_high 18 · high 39 · moderate 15 · low
246 · none 71**. The within-class collapse rescues ~12 kinases from `low` (e.g. CDK9/DYRK2,
spread across IT/ET excitatory subtypes → one unit → `high`). Spot-checks: BTK→very_high·Microglia,
ALK1→very_high·Endothelial, SYK→high·Microglia. **CAMK2A stays `low`** — even after collapse it
spans several *distinct* units (IT/ET excitatory, dentate, interneurons, striatal), which the
design correctly does not merge.

---

## 4. "Account for all attribution information" — what happens to direction concordance

The old `confidence_tier` axis (disease-direction concordance + decomp agreement) is **not
discarded** — it is demoted from the headline pill to a **secondary tag** carried in the pill
tooltip and kept as its own column, so both questions are answerable:

- **headline pill** → *where* is the kinase expressed (tissue exclusivity, new).
- **secondary tag / tooltip** → *does its activity move with disease* (direction concordance,
  the old meaning, preserved verbatim from `confidence.py`).

The tooltip lists each reference's dominant tissue + tier + the direction-concordance tag, so
the pill is self-documenting.

---

## 5. What was changed (as built)

**Producer:**
1. `config.py` — `SPECIFICITY_COLLAPSE_WMB_CLASSES` + `WMB_CLASS_DISPLAY` +
   `load_specificity_unit_map()` (cluster → unit{id,label,collapsed,children}).
2. `evidence.py:prepare_wmb_specificity` — forwards `wmb_top_celltype` (WMB's true dominant
   class) onto the unified row.
3. `alz/bulk_mea/specificity_class.py:assign_specificity_class` — recalculates
   `confidence_tier`/`confidence_basis` from §3 (fold `song_concentration` onto curated units,
   `eff = 1/Σ unit_share²`; WMB-class corroboration), emits `specificity_unit`,
   `specificity_unit_label`, `specificity_celltype`, `specificity_collapsed`, and snapshots the
   prior tier into `direction_tier` / `direction_basis`. Called after `assign_confidence`.

**Payload (`alz/viewer/cohorts/song.py`, `compose.py`):** `attribution_index` carries the four
`specificity_*` fields + `direction_*`; a new top-level `specificity_units` section
(`cluster_to_unit` + per-unit `label`/`collapsed`/`children`) is registered in `TOP_LEVEL_ORDER`
for the child reveal. The 9-family `specificity_lineage` column is gone.

**Unified viewer (`alz/viewer/template/`):**
- Explorer "Cell type" column (`_renderCellTypesCell`): shows the dominant unit label; a
  **collapsed** unit renders as an expandable `<details>` parent listing its child Song clusters
  with the kinase's per-cluster detection. New CSS `details.spec-unit` / `ul.spec-unit-children`.
- Audit verdict table: pill on the dominant unit's home cluster (`_attrVerdictConfCell`); the
  exclusivity summary names the unit and, when collapsed, states the child count + that the
  per-cluster rows are below; explainer legend rewritten for the curated-unit criteria.

**Out of scope (separate pipelines, untouched):** 5xFAD tab (own attribution source), Human tab
(computes its own tier live), tcell viewer (uses `tcell_concentration_tier`). Their pills are
unaffected.

**Verify:** Stage 3 regenerated (exit 0); per-kinase distribution + spot-checks confirmed (§3);
payload `specificity_units` (31→17) + per-row fields verified; both edited JS files pass
`node --check`. Unified viewer rebuilt under an 8 G memory cap.

---

## 6. Decisions (resolved during implementation)

1. **Keep the `none/low/moderate/high/very_high` pill vocabulary**; recalculate only its
   meaning. (No new `exclusive/enriched/...` scale.)
2. **Within-cohort Song primary; references corroborative** (promote-only, no veto).
3. **Curated specificity units**, decided per WMB class (collapse over-split cell types,
   keep distinct ones split, combined label for 1:1). Supersedes both the flat 9-tissue-lineage
   rollup (too coarse — merged distinct cell types) and the raw 31 clusters (too fine — over-split
   one cell type). Distribution: very_high 18 / high 39 / moderate 15 / low 246 / none 71. The
   collapse is surfaced as an expandable viewer parent — never silent.
4. **Human corroborates, never vetoes** (cross-species disagreement does not demote).

# C1 Audit — Song genotype-pooling surfaces

Read-only recon (2026-06-24). C1 is **de-aggregation**: every Song scalar that pools across the 3 disease genotypes (App/Tau/ApTt) splits into 3 per-genotype values. Split key = contrast prefix (`contrast.split("_")[0]`). Contract: `_contracts.md §C1`.

**Payload baseline:** `song.py:_build_kinases_slice()` (124–156) emits all 9 per-contrast `NES_{c}`/`FDR_{c}` columns — genotype is recoverable downstream. The collapse happens at TWO layers (the contract assumed only the JS one):

## Layer 1 — JS-only (render-time pooling; payload already has the data)
| Surface | Site | Pools over |
|---|---|---|
| `_mNes` ("M med" column) | `kinase_crosstable.js:825–826` `_kxComputeAgreement` — `for (const c of CONTRASTS) songVals.push(...)` then `_kxSourceFromValues` medians | all 9 contrasts |
| `song.sig` / `song.dir` / `_mouseSig` | derived from same `_kxSourceFromValues(songVals)` | all 9 |
| `_agreeCategory` / `_agreeScore` | `kinase_crosstable.js:845–852`, uses `song.sig`/`song.dir`/`song.nSig` | all 9 |

**Mutation point:** the `for (const c of CONTRASTS)` loop at 825. `_kxMedian` itself is generic — not the problem. Fix = filter the loop by genotype, 3×.

## Layer 2 — Payload-side (pre-collapsed in the pipeline, NOT in payload per-genotype)
| Surface | Site | Pools over | Needs |
|---|---|---|---|
| `peak_NES` / `peak_contrast` | `recover.py:266–270` argmax \|NES\| over all 9 | all 9 | per-genotype emit |
| `n_sig_contrasts` | `recover.py:264` sum(FDR<thresh) over 9 | all 9 | per-genotype emit |
| `trajectory_label` | `recover.py:204–243,271` `_classify_trajectory`, keyed on cross-genotype `peak_contrast` | all 9 (proxy) | per-genotype emit |

Consumed in payload via `song.py:143–145`; consumed in JS by `kinase_explorer.js:256–257,535,864`, `kinase_crosstable.js:164,1291`, `kinase_audit.js:7`.

## Confirmed CORRECT — NOT C1 targets (contract was mistaken on the first two)
- **Audit-verdict "Song NES/LFC"** (`kinase_audit.js` ATTR_VERDICT_COLS `song_lfc`, 584–603, 706) — contrast-scoped via the picker (`_renderAttributionVerdict` filters on `ctx.contrast`; `attribution_index` keyed by `(kinase,contrast)`). User already sees one genotype at a time. **No change.**
- **3×3 mouse glyph** (`_kxMouseGlyphCell`) — already renders all 9 contrasts individually. Correct.
- **`song_specificity` / `song_tau` / `song_top_cluster`** — expression-based, genotype-independent. Unified by design (contract). Untouched.
- **`top_celltype_1_song_lfc`** (`recover.py:340–343`, `song.py:151`) — NOT a genotype pooling; it's the LFC of the highest-confidence attribution row. Out of C1 (revisit under C3 if needed).

## Scope verdict
C1 is **both** a pipeline/payload change (Layer 2: `recover.py` + `song.py`) **and** a viewer change (Layer 1: `kinase_crosstable.js` + the 3-column render). Collision class = bulk_mea pipeline + unified-viewer builder + JS. Gate runs a bulk_mea recompute.

# TODO Masterplan — Roadmap & Sequencing (2026-06-18)

Threads the eight `TODO.md` items into one ordered program. Each item has a detailed
plan already written (linked below); this document is the **sequencing layer**: what to
do first, what depends on what, and where items collide.

Ordering principle (user directive): **easiest/most-feasible first, progressing to larger
changes.** Feasibility = (small blast radius) × (no external blocker) × (no new data
ingest) × (no shared-contract refactor).

Per-item plans:
- #1 `docs/plans/todo1_signed_nes_column.md`
- #2 `docs/plans/todo2_tcell_specificity_reference.md`
- #3 `docs/plans/todo3_standardized_csv_export.md`
- #4 `docs/plans/todo4_kinase_regulatory_network.md`
- #5 `docs/plans/todo5_monotonic_timepoint_exporter.md`
- #6 `docs/plans/todo6_incytr_on_5xfad.md`
- #7 `docs/plans/todo7_incytr_acetyl_ubiquitin.md`
- #8 `docs/plans/todo8_unified_viewer_scaling_audit.md`

---

## Feasibility ranking (summary)

| Rank | Item | Size | External dep | New data ingest | Touches shared contract | Blocker |
|------|------|------|--------------|-----------------|-------------------------|---------|
| 1 | #1 Signed NES | XS | none | no | render-only (3 JS + 3 labels) | none |
| 2 | #5 Monotonic exporter | S | none | no | no (standalone script) | none |
| 3 | #3 CSV export | M | none | no | new shared JS module + wiring | none |
| 4 | #6 Incytr on 5xFAD | M | none | no (inputs built) | no driver change | needs `allmarkers.csv` (1 task) |
| 5 | #7 Incytr AcK/KGG | M | none | derive 5xFAD AcK/KGG CSVs | ~20-line driver param | only 5xFAD has data |
| 6 | #2 T-cell NSCLC ref | L | 10x public DL | yes (~900K cells) | new viewer ref table | Flex panel ≠ all kinases |
| 7 | #4 Kinase network | L | **PhosphoSitePlus gated DL** | no | standalone viz (no payload) | Regulatory_sites.gz absent |
| 8 | #8 Viewer scaling audit | XL | none | no | **the contract itself** | audit-first, approval-gated. **All P-items implemented 2026-06-19** — P1+P2 verified (payload 105→69 MB raw, 10.1→8.49 MB gzip); P3 (crosstable lazy), P5 (gene_node_index sidecar), P6 (5xFAD cache LRU; index uncapped — see §2.10), P8 (T-cell sidecar mode). Full `pixi run viewer`/`tcell-viewer` rebuild + browser pass pending |

---

## Dependency graph

```
#1 ─┐
#3 ─┼─► feed feature list into #8's preservation checklist
     │
#6 ──► #7        (#7 reuses the 5xFAD Incytr runner; run #6 first to exercise it)
     │
#2 ──► #4(t-cell arm)   (#4's "same cell type" gate for t-cells needs #2's NSCLC ref;
                         #4's mouse/human arms can proceed on WMB / SEA-AD without it)
     │
#2, #7 add payload ──► tension with #8 (which shrinks payload). #8 audit must be
                       WRITTEN (done) before these land large new keys; #8
                       IMPLEMENTATION sequences last.
```

Key collisions to respect:
- **#1 & #3 vs #8.** Both edit the viewer. They are small and must NOT wait for the audit —
  but their changes (signed NES display, per-table CSV) become entries in #8's
  feature-preservation checklist. Land #1/#3 first, then snapshot the feature list.
- **#2 & #7 grow the payload #8 is trying to shrink.** #8's audit is already drafted;
  before #2's reference table or #7's PTM tracks inline large keys, apply #8's
  "shard, don't inline" rule (the 5xFAD keys are already the worst offenders). Treat
  any new ≥5 MB payload key as an edge-shard from day one.
- **#6 → #7 share one runner.** `run_pair_mode_5xfad.sh` + driver. Do #6 end-to-end,
  confirm the sce4 byte-identity gate (`pixi run verify-incytr-sce4`) still passes with
  `CHANNELS=pr,ps,py`, THEN add the `Ack`/`KGG` channels in #7.

---

## Wave plan

### Wave 0 — Memory correction (DONE 2026-06-18)
- Verified on disk that the 5xFAD `.sne` blocker is resolved (Cortex-IMAC, Hippo-Total
  re-exported as TSVs; kinase MEA st+py built both regions; Incytr inputs built).
- Replaced `project_5xfad_cohort_on_hold` memory with `project_5xfad_cohort` (unblocked).

### Wave 1 — Quick wins (XS–S, no data, no external dep)
Land these first; each is independently shippable and reversible.

1. **#1 Signed NES** — change 3 JS render sites to emit `r.peak_NES` (signed payload
   field already exists); update 3 template `<th>` labels/titles. Sort stays on `|NES|`.
   Verify in built viewer (hard-refresh). *Smallest, do first.*
2. **#5 Monotonic exporter** — add standalone `alz/cohorts/tcells/monotonic_export.py`
   (weak-monotone default, `--strict`, per-donor, OLS slope). Not pixi-wired. No
   coupling to anything else.

### Wave 2 — Bounded viewer + pipeline work (M)
3. **#3 CSV export** — add shared `alz/viewer_shared/template/js/06_export_csv.js`
   (RFC-4180), wire the 4 tables lacking export, replace Incytr's 3 private helpers.
   Export reflects current filter/sort. **Open question for user:** the kinase tab keeps
   its Markdown "Export view" button (chatbot helper) — confirm that's acceptable, or
   remove it to make CSV the sole export everywhere.
4. **#6 Incytr on 5xFAD** — `pixi run 5xfad-build-incytr-gene-list` (allmarkers, both
   tissues) → smoke (`--smoke cortex`) → full `pixi run 5xfad-incytr`. No driver change
   (derives DEG∪prG, t-cell path). 8 parquets (4 contrasts × 2 tissues), 961 pairs each.
   **Raise before reporting:** no cross-cohort absolute-value comparison (quantile-norm
   makes prG scale-invariant but absolute units differ ~400–1000×); 3 contrasts have
   n_wt=2 (lower-confidence SigProb).

### Wave 3 — PTM extension (M, builds on Wave 2 plumbing)
5. **#7 Incytr AcK/KGG** — extend driver `CHANNELS` + 4 env vars; build AcK/KGG
   deconvoluted CSVs for 5xFAD (analogous to existing ps/py export). New output dir
   `outputs/reports/incytr_pair_mode_5xfad/{cortex,hippocampus}/` — does NOT overwrite
   phospho. **Reality check from the audit:** *only 5xFAD has AcK/KGG data on disk.* Song
   and T-cell have none, so the TODO's "compatible datasets (song, 5xfad, tcell)" reduces
   to **5xFAD only** — surface this to the user rather than fabricating the other two.
   Gate: `pixi run verify-incytr-sce4` must stay byte-identical at `CHANNELS=pr,ps,py`.

### Wave 4 — New reference ingest (L, external download)
6. **#2 T-cell NSCLC reference** — download 10x NSCLC Flex (~900K cells, public `wget`)
   → `nsclc_kinase_expression.csv` crosswalked to **ProjecTILs states** (not Levy-t5).
   Two metrics: `fraction_cells_expressing` + `binary_expressed`. `probe_covered` flag
   separates "not in Flex panel" from "absent." Audit mode lists MEA-predicted kinases
   not expressed in any panel-covered cell type. **Memory-safety:** ~900K cells — inspect
   schema with pyarrow/DuckDB, chunk; do not full-load. Surfaces as a viewer audit table
   (apply #8's shard rule if the key is large).

### Wave 5 — Kinase regulatory network (L, gated external data)
7. **#4 Kinase–kinase network** — kinase-library v1.7.0 already installed (edge
   candidates = scored phosphoproteome). **Blocker:** PhosphoSitePlus `Regulatory_sites.gz`
   (activating/inhibiting sign) is absent and registration-gated. Phase 0 = acquire it
   (manual DL → `data/external/phosphosite/`) or fall back to OmniPath REST. Then edge
   model + CORROBORATED/CONTRADICTED logic across all cohorts; standalone D3/Dagre HTML
   (deliberately NOT in the unified payload — respects #8). T-cell co-expression gate
   depends on #2; mouse/human arms use WMB / SEA-AD and can run earlier. Ground biology
   claims against literature MCPs before finalizing.

### Wave 6 — Viewer scaling refactor (XL, approval-gated)
8. **#8 Unified-viewer scaling** — the audit document is already written (payload 104 MB
   raw / 10 MB gzip; top lever = shard `supporting_5xfad` 48.7 MB out of the inline
   payload → ~10→6 MB gzip; lazy tab init). **Audit-first per workflow rules: get explicit
   approval on the roadmap before any edit.** Sequence the *implementation* last because it
   refactors the shared PAYLOAD/Store/TAB_MANIFEST contract every other viewer change
   depends on — but feed Waves 1–4's new features into its preservation checklist as they
   land so the refactor is provably lossless.

---

## Open questions to resolve with the user before starting

1. **#3** — keep the kinase tab's Markdown "Export view" button, or make CSV the only export?
2. **#7** — confirm "compatible datasets" = **5xFAD only** (Song/T-cell have no AcK/KGG data).
   Acceptable, or should we chase the missing PTM data first?
3. **#4** — OK to register/download PhosphoSitePlus, or prefer the OmniPath REST fallback
   (no registration, slightly different coverage)?
4. **#8** — this is a large, contract-touching refactor. Approve the audit roadmap before
   any viewer edits, and decide whether it runs before or after the payload-growing items
   (#2, #7).
5. **Gating cadence** — confirm you want a human stop at each wave boundary (consistent with
   the project's gated-refactor convention), not a single autonomous run.

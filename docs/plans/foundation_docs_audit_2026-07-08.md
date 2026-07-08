# Foundation docs audit — 2026-07-08

Audit of all 19 files in `docs/foundation/` against the actual code on branch
`feat/incytr-backbone-refactor`. Every finding is grep/ls-verified; data files were not read.
No files edited yet — this is the pre-edit audit.

Three defect classes:
- **BROKEN** — points at a path/function/task/config that no longer exists (hard breakage).
- **STALE** — describes behavior/vocabulary the code has moved past (actively misleading).
- **CRUFT** — dated build logs, phase/packet labels, tombstones, "was previously here" pointers,
  redundancy with CLAUDE.md. No information value; violates no-tombstone / honesty rules.

Two findings are **CODE bugs** surfaced by the audit, not doc edits — called out separately at the end.

---

## Tier 0 — code bugs surfaced (fix in code, independent of doc rewrites)

- **C1 (functional breakage): mukesh policy loader path.** `alz/cohorts/mukesh/ingest.py:871`
  builds `POLICY_FILE = REPO_ROOT/"docs"/"audits"/"mukesh_ingest_policies.yml"` and the reshape
  guard refuses to run unless it loads (`ingest.py:900-901`). `docs/audits/` does not exist — the
  file was relocated to `docs/foundation/`. `conf/human_nbb/parameters.yml:15` has the same stale
  path in a comment. → `mukesh ingest --reshape` cannot find its gating policy. Fix both to
  `docs/foundation/`.
- **C2: pipeline_conventions cites deleted source files** (see D3 below) — the fix is a doc edit,
  but it means any tooling/reader trusting those line refs is pointing at nothing.

---

## Tier 1 — BROKEN references (delete or repoint; highest priority)

- **D1 `backbone_incytr_track.md`** — `pixi run verify-incytr-sce4` (L58,64,69,112,130) is not a
  pixi task; the three backing scripts (`verify_incytr_sce4.sh`, `verify_sce4_full.R`,
  `verify_sce4_parity.py`) were deleted this branch. `alz/viewer/verify_backbone_spine_index.py`
  (L108, "35 passing") does not exist. Overnight runner `run_backbone_overnight.sh` (L124-126)
  replaced by three cohort-split scripts (`run_backbone_overnight_{5xfad,tcells,all}.sh`).
- **D2 `cohort_contract.md`** — §3.3 YAML block (L133-145: `cohort_design`, `contrast_coefs`,
  `genotype_levels`, `animal_id_regex`, …) is fictional; `grep conf/` finds none of these keys.
  §2.1 "nodes branch on `parameters.yml:cohort_design`" (L61) — zero code hits. Onboarding step
  `KEDRO_ENV=mukesh pixi run live` (L256-257) is wrong twice: env is `human_nbb`, and case/control
  cohorts use `run_mukesh_perdonor.sh`, not `pixi run live` (per `conf/human_nbb/parameters.yml:1-9`).
- **D3 `pipeline_conventions.md`** — sign-table cites `kinase_enrich.py:44-62` / `:306-335` and
  `decomposition/factorial_ols.py:71`; neither file exists. Logic moved to `alz/bulk_mea/enrich.py`
  (`_build_design_matrix`, `_run_ols_all_sites`, `_run_mea`, `lfc` at :523). Repoint to function
  names, drop line numbers.
- **D4 `multiple_testing.md`** — "Storey's q in `alz/ingest/song.py`" (L29,69-70): no Storey code
  anywhere (`grep storey` → 0); marker/composition diagnostic moved to
  `alz/supplementary/deconvolution_feasibility.py` (factorial OLS, not Storey). Three table rows
  (L28,30,31) point at `integration/adapters/*` dirs that don't exist. `docs/archive/*pre_remediation*`
  (L28,68) — not found.
- **D5 `specificity_confidence.md`** — L71 attributes `_build_tcell_attribution_index` to
  `build_tcell_viewer.py`; it lives in `alz/tcell_viewer/slices_kinase.py:463`. L170 invents
  `ENRICHED_EFF_MAX` (no such symbol; only `EXCLUSIVE_EFF_MAX`/`BROAD_EFF_MAX`).
- **D6 `standard_attribution_metric.md`** — L5 "cataloged in `attribution_specificity_audit.md`":
  file does not exist.

---

## Tier 2 — STALE behavior/vocabulary (rewrite against current code)

- **D7 `concordance.md` (worst offender).** Predates the detection-gate refactor. §"Evidence basis
  labels" (L148-166): all 7 label strings (`three_way`, `within_cohort`, `song_only`, …) exist
  nowhere in code — `confidence.py` emits different strings (e.g. `seaad_wmb_moderate`). §"Direction
  tiers" + thresholds still describe the removed **share-based** gate (`song_specificity ≥ 1/N`);
  code is detection-gated (`confidence.py:154-155` `song_detected`/`wmb_detected`). "WMB fold/tier"
  → now `wmb_concentration_tier`. Keep the direction-concordance math + verified config keys
  (`SONG_CONCORDANCE_WEIGHT` etc.); rewrite the labels/gates sections. Confirm `SONG_TO_WMB_CLASS_MAP`
  (L244) still exists — likely renamed to the bridge/crosswalk mechanism.
- **D8 `backbone_incytr_track.md` cohort scope.** Doc frames backbone-grain payload as song-only
  (5xFAD/t-cell only lacking Ack/KGG columns). Code is ahead: `write_incytr_backbone_grains` is wired
  into `fivexfad.py:60` (+ backbone plumbing L1724-1823) and `tcell_viewer/slices_incytr.py:34,733`;
  `BACKBONE_OUT_DIR` live in `run_pair_mode_tcells.sh:85`. Update scope to all three builders.
  **NOTE:** the active plan (`incytr_rerun_ksg_ptm_backbone_2026-06-29.md`) calls the viewer port
  "deferred (Phase 2)" — code appears to be ahead of that plan. Flagging the discrepancy rather than
  silently reconciling; confirm intent.
- **D9 `backbone_incytr_track.md` "1-pair fixture" state (L118-131).** False now — runner executes
  the full 31²=961 grid (`run_pair_mode.sh:5`, `incytr_commandline.R:86`). Delete fixture snapshot +
  "Human gates outstanding" checklist.
- **D10 `live_pipeline_contract.md` (F8).** Says live "= ingest→normalize→enrich→attribute→recover"
  and calls mechanism "off the live arc," but `pixi.toml:59` includes `mechanism` in the bundle
  (between attribute and recover). One contradiction; otherwise this doc is accurate.
- **D11 `analysis_rationale.md` §6 + `statistical_constraints.md` §6** — dead "Track A / Track B"
  vocabulary; not in code (only false positive is `_f5_track_assay`, an unrelated st/py residue
  track). Restate in current `confidence_tier`/`direction_tier` + mechanism-class terms.
- **D12 `cohort_contract.md` minor** — §2.2 `parse_animal_id` "returns tuple" — actually returns a
  **dict** (`config.py:807-826`). §2.4 "Phase 4 task: inject as `params:analysis_mode`" — already
  threaded as a function param (`enrich.py:422`); delete the TODO.
- **D13 `projected_state_mea_contract.md` (F10)** — "first target: donor1 only … no other donor
  must be runnable" is superseded; both donors run (`pixi.toml:49-50`, `tcells_decompose.py:15-17`;
  donor1 st+py, donor2 py-only). Rewrite to actual donor/track eligibility.
- **D14 `repo_retention_policy.md` inventory** — the code inventory predates the whole `alz/cohorts/`
  refactor: no `alz/cohorts/{fivexfad,tcells,mukesh}/`, `alz/viewer/`, `alz/cross_reference/`,
  `alz/integration/*`; mukesh cohort unmentioned. Needs a re-inventory, not line edits. The runner
  table (L64-70) and `alz/bulk_mea/*` rows are accurate. `docs/archive/` section (L126-133) documents
  a tree that isn't present — repoint to repo-root `archive/` or delete.

---

## Tier 3 — CRUFT (delete; dated logs, phase/packet labels, tombstones, redundancy)

- **D15 `standard_attribution_metric.md`** — ~180 lines of build history (L149→end): "RESOLVED
  2026-06-20", Phase 1/2/2C/2D/3 "DONE" logs, payload byte-sizes, one-time gene counts, `RETIRED
  2026-06-21` / `Superseded 2026-06-22` annotations. Delete; keep only the normative spec above it.
  (L183 also misattributes the tcell index to `build_tcell_viewer.py` — another reason to cut, not fix.)
- **D16 `backbone_incytr_track.md`** — L1-14 Wave/Theme program preamble; dated workflow-run headers
  (`wf_82154a8b`, "2026-06-29"); L132-145 "Open threads" (references deleted `_contracts.md`);
  L147-156 tombstone narration. Compress to grain table (L24-56, the durable contract) + current
  wiring + one closed-paths line.
- **D17 `viewer_frontend_contract.md`** — L124-145 dated 2026-06-01 saturation audit (Pearson 0.993,
  artifact paths) belongs in an audit doc, not the frontend contract; L265-267 dated smoke-test
  tombstone; L206-231 two "was previously listed here" consolidation notes. Structure otherwise
  verified accurate.
- **D18 `viewer_payload_contract.md`** — L601-629 "Migration Plan" + "status as of 2026-06-01"
  (migration done); legacy flat/`by_donor`/`d=` fallback prose (L25-29,218-231,631-646) — **confirm
  in `00_payload_adapter.js` whether the fallback still exists in code before deleting the prose**;
  if builders emit v2-only, delete both prose and "Deprecation Targets" list. Schema description is
  the durable, accurate part.
- **D19 `mechanism_attribution_contract.md`** — drop "(Packet 0A)" / "Phase 0 Packet 0A" labels
  (no code counterpart). Contract body verified accurate against `mechanism_attribution.py` +
  `validate_cohort.py`.
- **D20 `projected_state_mea_contract.md`** — drop "(Packet 0B)" labels (same as D19).
- **D21 `cohort_contract.md`** — L9-13 "2026-05-21 factorial vocab unification / Phase 3 closeout"
  provenance stamp; §5.3 Incytr invariants duplicate CLAUDE.md verbatim (trim to a pointer).
- **D22 `repo_retention_policy.md`** — L72-77 + L134-144 tombstone paragraphs about deleted runner
  scripts / archived docs.
- **D23 `kinase_explorer_attribution.md`** — L446-455 "Historical Inputs this supersedes" (4
  plan/audit doc pointers, several likely deleted this branch). Verify + delete the dead ones.
  Otherwise the most current of the 19 — use as the reconciliation template.

---

## Clean (no action)

- `tcell_reference.md` — static ProjecTILs glossary, verified consistent.
- `mukesh_ingest_policies.yml` — content correct (the bug is the consumer path, C1).
- `analysis_charter.md` — directionally correct; only nit is it still reads as Song-bulk-only front
  door (5xFAD/t-cell/pair-mode appear as "further layers"). Optional reframe, not a defect.

---

## Unverifiable (data files off-limits on shared box — flag, don't touch)

Numeric "retained facts" in `analysis_charter.md` / `analysis_rationale.md` (72 animals, 91.7%,
Kruskal p=1.4e-42) and worked-example tier counts in `specificity_confidence.md` (L189-203) /
`standard_attribution_metric.md`. Historical provenance; would need data reads to confirm.

---

## Suggested execution order

1. **C1** (code bug — mukesh loader path) — one-line fix, unblocks reshape.
2. **Tier 1 BROKEN** (D1-D6) — mechanical repoint/delete.
3. **D7 `concordance.md`** rewrite (largest stale surface).
4. **Tier 2 remainder** (D8-D14) — behavior/vocabulary rewrites; D8 needs your call on the
   plan-vs-code discrepancy.
5. **Tier 3 CRUFT** (D15-D23) — deletions; D18 needs the `00_payload_adapter.js` check first.

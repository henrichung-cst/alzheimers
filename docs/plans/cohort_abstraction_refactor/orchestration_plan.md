# Cohort Abstraction Refactor — Orchestration Plan

Date: 2026-06-17
Status: SUPERSEDED BY IMPLEMENTATION — kept as historical orchestration context

Implementation note (2026-06-18): the implemented repository differs from this
pre-implementation plan in the places explicitly called out below. Current
source state and compatibility policy win over draft orchestration guidance.

This document defines **how** the control pack (`README.md`, `agent_protocol.md`,
`phase_0..5_*.md`) gets executed by an orchestrator + subagent model. The control
pack says *what* each phase must achieve and *what must not drift*. This plan says
who runs what, in what order, with what parallelism, and where the gates are.

It is grounded against the real repo (three structural maps run 2026-06-17), not
the control pack's assumed paths. Where the control pack and the code disagree,
the code wins and the correction is recorded in §2.

---

## 1. Roles

| Role | Who | Tools | Responsibility |
| --- | --- | --- | --- |
| **Orchestrator** | main session (this conversation) | full | Sequences phases, dispatches packet waves, runs the start/end checklists, **runs the parity comparator itself**, writes the monitoring report, stops at every gate. |
| **Implementer** | `general-purpose` subagent (one per packet) | full, scoped | Implements exactly one work packet inside its declared file allowlist. Writes its own decision-log entries. Returns a structured result (files touched, commands run, scratch outputs produced). Does **not** certify its own parity. |
| **Verifier** | `audit-pipeline` subagent or orchestrator | Read/Grep/Glob/Bash | Runs parity comparators and validators against protected outputs. Separated from the implementer on purpose — the agent that wrote the code does not grade its own drift. |
| **Human reviewer** | the user | — | Approves the protected-file list and parity policy before Phase 0; approves each phase-boundary gate; approves any drift exception or canonical-output switch. |

**Hard rule:** implementer and verifier are never the same agent for a given
packet. Parity is adversarial by construction.

---

## 2. Grounded corrections to the control pack

These are facts from the repo maps that override the control pack's assumed
shapes. Carry them into every packet.

1. **SUPERSEDED: `alz/core/` exists.** The original draft assumed all
   `alz/core/*` modules were future files. Implementation created the package
   and the shared runner/validation/output modules.
2. **The shared MEA engine already exists and is already shared.** Every cohort
   calls `alz/bulk_mea/enrich.py::_run_mea()` (line 206). Phase 3 must **not**
   touch `_run_mea` or any statistics. Phase 3 wraps the *orchestration around*
   it (track loop, skip recording, motif checks, output writing, provenance).
   Non-goal restated with teeth: `enrich.py` is read-only for the entire refactor
   except where Phase 2 extracts duplicated **post-`_run_mea` writer** code.
3. **Protected-file list is per-cohort, not uniform.** NES/FDR matrices exist
   **only** for T-cell (`kinase_timepoint_nes/fdr.csv`) and Mukesh per-donor
   (`kinase_donor_nes/fdr.csv`). Song and 5xFAD produce **OLS/effect-size** tables
   (`site_level_ols.csv`, `*_site_level_ols.csv`), no NES/FDR. A flat protected
   template would emit false "missing" failures. Phase 0 packet 0B must template
   per-cohort.
4. **T-cell donor2 is partial by design** (pY only; manifest, no full MEA long
   table). Record as `absent_by_design`, not a gap. Baseline must not demand its
   `mea_timecourse.csv`.
5. **Phase-2 duplication is exact and located.** `mukesh_perdonor.py:142-218` and
   `tcells_perdonor.py:62-206` carry character-identical write / pivot /
   recurrence logic modulo `donor`↔`timepoint` field names, plus a duplicated
   `_KIND_SPEC` dict. This is the surgical target. The shared helper imports
   thresholds from `alz/shared/config.py` (`MEA_FDR_THRESH=0.25`), never
   hardcodes.
6. **Viewer namespace is `alz/viewer/` (singular) + `alz/tcell_viewer/`**, not the
   control pack's `alz/viewers/`. The unified builder `alz/build_unified_viewer.py`
   is a **5,411-line monolith** with all Song/Mukesh/5xFAD payload construction
   inline; the T-cell builder is a fully independent 2,800-line file. Phase 5 is
   the largest single extraction in the pack and must be sequenced smallest-cohort
   first (Mukesh).
7. **Determinism is on our side.** MEA uses a fixed seed (`MEA_SEED=112123`,
   `MEA_PERMUTATION_NUM=1000`). Scratch-vs-canonical MEA output should be
   bit-reproducible, which makes **exact-identity parity** the default target (see
   §6) rather than loose numeric tolerance.
8. **Memory safety.** The unified payload is 104 MB JSON / 10 MB gz. Parity on it
   is by sha256 + streamed key-set/shard-index diff (ijson / DuckDB), never
   `json.load`. Same rule for any `data/derived/*.parquet` > 500 MB.

---

## 3. Cross-cutting invariants (every packet, every phase)

From `agent_protocol.md`, made executable:

- **Clean start.** Orchestrator records `git status --short --branch` +
  `git rev-parse HEAD` before dispatching a wave. Dirty tree outside the assigned
  scope → stop.
- **Scratch-only writes.** New code writes to
  `outputs/reports/refactor_audit/phase_N/..._new/`. Canonical outputs under
  `outputs/reports/{kinase_attribution*,decomposition,unified_viewer,tcell_viewer,
  incytr_pair_mode*}/` are **never** overwritten until a separate, explicitly
  approved switch.
- **SUPERSEDED: module-only compatibility.** The implemented Phase 4 retired old
  Mukesh/T-cell/5xFAD Python paths without wrappers. Canonical cohort commands
  are `python -m alz.cohorts...`; old `alz/ingest/{mukesh,tcells,fivexfad}*.py`
  paths are historical provenance only.
- **Decision log per packet** at
  `docs/audits/cohort_abstraction_refactor/phase_N_decisions.md` using the
  protocol template. Any behavior/schema/naming/provenance change = an entry.
- **Drift exception** is the only way a non-matching protected output is accepted,
  and only with human approval.
- **End checklist** per packet: `git status --short`, stale-import scan
  (`rg` for old paths), `python -m py_compile` on changed Python, phase
  validators.
- **Monitoring report per phase** at
  `outputs/reports/refactor_audit/phase_N/monitoring_report.md` with the README's
  fixed schema (phase id, agent/packet ids, git base + final commit, commands,
  input/output roots, protected files checked, pass/fail per file, skipped checks,
  drift exceptions, rollback status).
- **Rollback granularity.** Every packet reverts by one commit or by dropping its
  scratch dir. If rollback would need manual reconstruction, the packet is too big
  and gets split before dispatch.

---

## 4. The two-level control loop

```
for phase in [0,1,2,3,4,5]:
    orchestrator: start checklist (git clean, base commit)
    for wave in phase.waves:              # waves = topological layers of the packet DAG
        dispatch implementer subagents in parallel  # only packets with disjoint file scopes
        collect structured results + decision-log entries
        if wave produced scratch outputs:
            orchestrator/verifier: run parity comparator vs Phase-0 baseline
    orchestrator: write monitoring_report.md
    >>> HUMAN GATE <<<  present phase verdict; STOP.
    # do not enter phase+1 until the user approves and all protected parity passes
```

Two non-negotiables encoded here:

- **Waves, not flat fan-out.** Packets parallelize only when their `files_allowed`
  sets are disjoint. Same-file packets are sequenced (protocol §Conflict
  Handling). The per-phase DAGs in §5 mark this explicitly.
- **The phase boundary is a real stop.** The orchestrator does not advance phases
  autonomously. This satisfies both the control pack ("do not start a later phase
  with unresolved parity failures") and the standing user rule to honor gated
  plans rather than collapse them into one run.

---

## 5. Per-phase execution (packet DAGs + file scopes)

Legend: `→` sequence, `∥` parallel-safe (disjoint scope), `⟂` read-only.

### Phase 0 — Inventory & Baseline Lock  *(read-only; no parity risk)*

```
0A output-root discovery  ⟂ ─┐
0B protected-file list    ⟂ ─┴→ 0C inventory generator → 0D baseline summary
```
- 0A + 0B are discovery; much is already known from the 2026-06-17 maps — the
  orchestrator can pre-seed both and have one subagent confirm-and-serialize them
  to `output_roots.json` / `protected_files.json` (per-cohort templating per §2.3).
- 0C builds `alz/core/baseline_inventory.py` (read-only: path, sha256, size, mtime,
  row/col counts, key columns; streams big files). Creates `alz/core/__init__.py`.
- 0D writes `phase_0_baseline_summary.md`.
- **Verify:** run inventory twice; sha256 + row counts identical across runs.
- **Gate output:** the protected-file list + parity policy for human sign-off.

### Phase 1 — Read-Only Validators  *(read-only; no producer changes)*

```
1A schema dataclasses (phospho_schema.py, cohort_manifest.py)
   → 1B cohort validators (validate_cohort.py)  ┐ share validation.py →
   → 1C report writer (validation.py reports)    ┘ sequence, not parallel
   → 1D legacy-deviation register (doc)
```
- 1B and 1C both touch validation infra → **sequenced**, single implementer owns
  `alz/core/validation.py` + `validate_cohort.py`.
- Validators must pass or emit an *accepted legacy deviation* (1D) for all four
  cohorts. No producer code changes.

### Phase 2 — Shared Output / Recurrence Helpers  *(first parity-bearing phase)*

```
[2A recurrence + 2B bundle writer]  ── one implementer owns alz/core/mea_outputs.py
   → 2C mukesh scratch adapter (mukesh_perdonor.py)  ∥  2D tcell scratch adapter (tcells_perdonor.py)
   → run both scratch commands
   → 2E parity comparator  ── VERIFIER (audit-pipeline), not 2C/2D's author
```
- 2A and 2B target the *same new module* → merged into one packet (avoid the
  artificial file-split the control pack implies).
- 2C/2D edit different existing files → parallel-safe. Each adds an **opt-in
  scratch path only**; canonical writers untouched.
- 2E is the gate: scratch-vs-canonical row counts, key sets, numeric fields
  (exact per §6), categorical fields exact. Owned by the verifier.
- **Why start here:** smallest, most localized, exact duplication, deterministic
  outputs. Highest-confidence parity proof → builds trust in the harness before
  the riskier phases.

### Phase 3 — Shared MEA Runner

```
3A runner skeleton (mea_runner.py, contrast.py, provenance.py)  [scratch only]
   → 3B mukesh adapter → parity
   → 3C tcell adapter  → parity
   → 3D 5xFAD bulk adapter → parity
   → 3E 5xFAD celltype adapter (ONLY after 3D parity passes)
3F Song feasibility report  ⟂  (read-only doc; may run anytime)
```
- Mandated order: Mukesh → T-cell → 5xFAD-bulk → 5xFAD-celltype → Song-last.
  Song is **feasibility report only** this phase (richest downstream attribution +
  recovery deps; do not migrate).
- Runner owns orchestration only; `_run_mea` untouched (§2.2). Each cohort's
  contrast logic is a `ContrastAdapter`, not copied pipeline.
- Each adapter gated on its own parity before the next starts.

### Phase 4 — Cohort Directory Migration  *(source moves; zero behavior change)*

```
4A namespace skeleton (alz/cohorts/{song,mukesh,tcells,fivexfad}/ + READMEs)
   → 4B mukesh move → 4C tcell move → 4D 5xFAD move   (sequenced: shared import graph)
   → 4F runner/pixi/docs update
4E Song assessment (decision doc)  ⟂
```
- **SUPERSEDED:** moved Mukesh/T-cell/5xFAD modules do not leave compatibility
  wrappers at old paths. Old imports do not resolve by design.
- Moves are sequenced, not parallel — they perturb the shared import graph and a
  stale-import scan must pass cleanly after each.
- **Verify:** `py_compile` moved modules; new import paths import;
  stale-direct-import scan clean; Phase-1 validators still pass; **no protected
  output changes** (this phase touches no producers).
- This is where the no-wrapper policy was recorded and old paths were retired.

### Phase 5 — Viewer Slice Contract  *(largest extraction)*

```
5A payload field inventory (read-only)
   → 5B CohortViewerSlice schema draft (alz/viewer/shared/cohort_slice.py)
   → 5C mukesh slice adapter  (smallest; payload["human"])
   → 5D tcell slice adapter   (keep dedicated tcell_viewer output)
   → 5E 5xFAD slice adapter   (extract payload["supporting_5xfad"] from monolith)
5F Song slice feasibility (doc)  ⟂
```
- Extract from the 5,411-line `build_unified_viewer.py` cohort by cohort,
  smallest first. Each adapter exposes `build_viewer_slice(...) -> CohortViewerSlice`;
  the builder becomes a composer.
- **Verify:** `verify_payload_contract.py` + `verify_template.py` pass; payload
  key sets identical (streamed diff on the 104 MB JSON); lazy shard indexes match
  per family; frontend `raw(...)` references resolve; file-size deltas explained.
- Song payload logic stays until 5F feasibility is accepted.

---

## 6. Parity policy (RATIFIED 2026-06-17 — declared numeric tolerance)

Structural fields are exact; numeric fields compare within a **declared, logged
tolerance**. Keys, rows, and categoricals never get slack — only continuous
numerics do, and the tolerance is a stated constant, not a per-file negotiation.

| Field class | Rule |
| --- | --- |
| Row count | exact |
| Key set (key columns) | exact set identity |
| Categorical fields (calls, signs, tiers, IDs) | exact string match |
| Numeric fields | `numpy.isclose(rtol=DEFAULT_RTOL, atol=DEFAULT_ATOL)`, NaN-positions must match exactly |
| File-level (binary/parquet/large JSON) | sha256 first; on mismatch fall to streamed structural diff under the rules above |

**Default tolerance:** `DEFAULT_RTOL = 1e-6`, `DEFAULT_ATOL = 1e-9` — tight enough
that any real analytical drift trips it, loose enough to absorb float-formatting
and library round-trip noise. These two constants are the only knobs; they are
recorded in `phase_0_decisions.md` and surfaced at the Phase-0 gate for final
ratification of the numbers. A *wider* tolerance on any specific field requires a
logged drift exception naming the field and the non-deterministic source. Sign
flips, NaN-position changes, and key/row changes are never tolerance-absorbable —
they are always drift.

---

## 7. Tooling: how waves are actually dispatched

- **Default: interactive per-phase dispatch.** The orchestrator (this session)
  spawns implementer subagents via the Agent tool, one wave at a time, and stops
  at each phase gate for the user. This keeps the human gates real and every
  decision visible. Recommended.
- **Optional: `Workflow` for a single phase's internal packet-wave.** A phase's
  DAG (fan-out adapters → barrier → parity verify) maps cleanly onto one Workflow
  script, but **only ever one phase per invocation** so the gate is not bypassed.
  Requires explicit user opt-in (multi-agent orchestration). Not the default.
- Either way: verifier ≠ implementer; parity comparator is orchestrator/verifier
  territory.

---

## 8. What is NOT in scope of this plan

- No analysis math, thresholds, sign conventions, or contrast semantics change in
  any phase. (Anti-shim closed-paths and Incytr/MEA invariants in `CLAUDE.md`
  remain frozen; this refactor is structural only.)
- No canonical-output regeneration. Scratch only until an explicit, separately
  approved switch.
- 5xFAD proteomics ingest is on hold (`.sne` blocker) — its existing on-disk
  outputs are baselined and protected, but no new 5xFAD ingest is attempted here.

---

## 9. Human review gate — decisions needed before Phase 0

The control pack requires these confirmations before implementation. Resolved by
the 2026-06-17 maps where possible; open items flagged.

| Item | Status |
| --- | --- |
| Which dirty/WIP changes are in scope | **Resolved.** Only `docs/plans/cohort_abstraction_refactor/` is untracked (the control pack itself). Tree otherwise clean. |
| Unrelated changes to isolate | **Resolved.** None pending. |
| Where Phase-0 baseline reports live | **Proposed.** `outputs/reports/refactor_audit/phase_0/` (README default). |
| Which outputs available locally | **Resolved.** All cohort output roots present (T-cell donor2 partial by design). |
| Viewer reorg committed before/after Phase 0 | **Resolved.** Already committed (HEAD `eb3387b` et al.); tree clean. Moot. |
| **Parity policy** (§6) | **RATIFIED.** Declared numeric tolerance: structural exact, numerics `isclose(rtol=1e-6, atol=1e-9)`, NaN-positions exact. Numbers confirmable at Phase-0 gate. |
| **Dispatch mechanism** (§7) | **RATIFIED.** Interactive per-phase dispatch (Agent tool, stop at each gate). |

Phases run strictly 0→5. No phase begins while a prior phase has an unresolved
parity failure on a protected output.

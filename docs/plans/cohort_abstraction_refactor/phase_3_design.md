# Phase 3 Design — Shared MEA Runner

Date: 2026-06-17
Status: design SSOT (Phase 3 approved to proceed; this records the concrete runner
interface + parity strategy that the control pack left abstract). Grounded against the
MEA-orchestration map run 2026-06-17 (all five `_run_mea` callers read).

## Prime directive (unchanged)
Zero output drift. `alz/bulk_mea/enrich.py::_run_mea` is FROZEN — not touched, not
wrapped in a way that changes its inputs. No statistic, seed, permutation count,
threshold, or sign convention changes. Scratch-only writes until a separately approved
cutover, exactly as Phase 2.

## What is actually shared (from the map)
The only invariant shell across all 5 callers (Song bulk, 5xFAD bulk, 5xFAD celltype,
Mukesh per-donor, T-cell per-donor), once cohort-specific contrast construction is
stripped:

1. Resolve a **unit** (one `_run_mea` invocation): a `(track, lfc_key, motif_series,
   site_ids, gene_symbols, out_dir, naming)` bundle.
2. **Call `_run_mea`** with those args → `(mea_df, shift_df, wins_df, substrate_df)`.
3. **Write the 4 standard tables** (`mea_*`, `mea_global_shift`, `winsorized_sites`,
   `mea_substrate_sets`) under the unit's naming convention.
4. **Record skip/empty** (matrix absent, 0 motifs, empty mea_df) — structured.
5. **Stamp provenance** (a per-run manifest).

Contrast construction (`results_by_contrast`), input loading, and any per-unit
aggregate step (NES/FDR matrices + recurrence) stay **cohort-specific** and are supplied
by the adapter. The aggregate step for Mukesh/T-cell already lives in
`alz/core/mea_outputs.py` (Phase 2) and is reused unchanged.

## The runner interface (`alz/core/mea_runner.py`, new, scratch-capable)
Thin. Designed against the two clean cohorts (Mukesh, T-cell) first; generalized to
5xFAD only at waves 3D/3E. The adapter is a small object/protocol:

```
load_inputs(unit) -> (motif_series, site_ids, gene_symbols)   # usually a CSV read
iter_units() -> Iterable[unit]        # the track/kind/donor/tissue/celltype loop
build_contrasts(unit, loaded) -> results_by_contrast   # COHORT-SPECIFIC, reuses
                                                       # existing _build_*_deltas etc.
skip_check(unit, loaded) -> (skip: bool, reason: str|None)
output_dir(unit) -> Path
write_standard_tables(mea, shift, wins, subs, unit)   # default impl; cohort may override
write_aggregates(mea_df, unit) -> None                # default no-op; Mukesh/T-cell
                                                      # delegate to mea_outputs
write_provenance(records) -> None                     # manifest/audit json
```

The runner owns steps 2–5 and the loop; the adapter owns load + contrast + naming +
the optional aggregate/provenance shape. **The runner adds NO new behavior** — every
default mirrors the current inline code exactly.

## Divergences and how each is handled (do not force one mold)
- **5xFAD bulk double `_run_mea` (stoich + raw) + concat:** the adapter yields TWO
  sub-units per `(tissue, track)` (one per `lfc_key`) and provides a cohort-specific
  `write_standard_tables` that concatenates stoich+raw shift/wins/subs before writing
  (matching `fivexfad.py` lines 599–636 exactly). The runner's per-unit call is reused;
  only the output assembly is adapter-owned.
- **5xFAD celltype triple loop + reweighted signal + end-accumulation:** `iter_units`
  yields `(tissue, track, cell_type)`; `load_inputs` performs the pseudobulk join +
  weight construction (`y = raw_vals + log_w`); the adapter accumulates unit outputs and
  writes the single concatenated parquet/csv set at the end (matches
  `fivexfad_celltype_mea.py`). Runner supplies only the `_run_mea`+skip primitive.
- **Skip/provenance variance:** the runner records a structured skip list always;
  adapters that historically wrote a manifest (T-cell `mea_manifest.json`, celltype
  audit json) keep that exact shape via `write_provenance`. Cohorts that wrote nothing
  (Song, 5xFAD bulk from `run_mea`) keep writing nothing — preserved, not "upgraded".
- **Song raw_lfc not run through a 2nd `_run_mea`:** preserved as-is. NOT a Phase-3 fix.

## Parity strategy (the key decision)
`_run_mea` is deterministic (`MEA_SEED=112123`, fixed permutations) and frozen. So the
runner is correct iff it feeds `_run_mea` identical inputs and writes identical outputs.

1. **End-to-end re-run (AUTHORITATIVE — revised after 3A).** Actually run `_run_mea`
   through the runner for a small unit per cohort to a scratch dir and diff the written 4
   tables + aggregates against canonical under the ratified policy (exact key set, column
   order, NaN positions; numerics `isclose(rtol=1e-6, atol=1e-9)`; recurrence integer
   counts exact). This is the load-bearing proof. Use the small `_pY` units (cheap) and,
   where determinism makes it safe, treat one passing small unit + frozen `_run_mea` as
   covering the larger units; if ever a full re-run is needed it goes through a shell
   script, not interactive monitoring.
2. **Input-fingerprint harness (FAST PRE-CHECK only — demoted after 3A).** The 3A verifier
   found the harness's "canonical" side reconstructs `_run_mea` args via the same pure
   functions the adapter calls, not via the literal inline `_run_track_kind` body — so it
   proves adapter==reconstruction, not adapter==inline (tautology risk). It is non-vacuous
   (a 1-element perturbation flips the SHA-256) and useful as a fast smoke screen, but it
   is NOT the proof. The verifier MUST NOT rest a PASS on the harness alone; the
   independent end-to-end re-run (on a unit the verifier picks, distinct from the
   implementer's) is required for every adapter wave.
3. Aggregate tables (NES/FDR/recurrence) for Mukesh/T-cell are already Phase-2-proven;
   the runner reuses `mea_outputs` unchanged, so they inherit that parity.
4. **Layout fidelity (added after 3B — content `isclose` is NOT sufficient).** Two
   adapter defects passed content-parity but were deliverable drift: (a) a hardcoded
   long-table stem (`mea_perdonor` written for T-cell instead of `mea_timecourse`), and
   (b) the Mukesh adapter nesting outputs under `<scratch>/<track>/<kind>/` while
   canonical writes them FLAT into one dir (infix/suffix disambiguate). Every wave's
   parity MUST now assert that the SET of relative output paths the runner produces
   equals the set of canonical relative filenames for that cohort — same directory shape,
   same filenames — not merely that each file's contents match once located. Filenames
   and directory layout are output; a difference is drift.

Long full re-runs (if ever needed for a full end-to-end confirmation) go through a
shell script, not interactive monitoring.

## Wave order (gated; matches orchestration_plan.md §5)
```
3A runner skeleton + Mukesh adapter  [scratch] -> parity (input-equiv + spot-check)
3B T-cell adapter                    [scratch] -> parity
3C 5xFAD bulk adapter (double-call)  [scratch] -> parity
3D 5xFAD celltype adapter (ONLY after 3C parity) -> parity
3F Song feasibility report (read-only doc; anytime)
```
Each adapter gated on its own parity before the next. Implementer ≠ verifier per wave.
Mukesh is folded into 3A so the runner interface is designed against a real consumer,
not in the abstract (mirrors how Phase 2's module was co-designed with its first caller).

## Anti-over-abstraction guard
If, at any wave, fitting a cohort into the runner requires an escape hatch that makes the
shared code MORE complex than the duplication it removes, that is a signal to STOP and
leave that cohort partially custom — surfaced at the wave's gate, not forced. The runner
exists to remove genuine duplication (the 4-table write + skip + loop), not to prove an
abstraction.

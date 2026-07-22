# Pathways detail tab — subplan index

Decomposition of `docs/plans/kinase-viewer-pathways-detail-tab.md` (reviewed:
Approved with minor revisions, Separable). Revision #1 already folded into the
parent plan (Explorer columns are NOT re-sourced from the new block — the scan
populates both the existing columnar arrays and the new block).

## Subplans

| # | Title | Files | Risk | Depends on |
|---|---|---|---|---|
| 01 | Backend compute & delivery (Python) | `slices_kinase.py`, `slices_incytr.py`, `slices_audit.py`, `build_tcell_viewer.py` | high | none |
| 02 | Frontend Pathways tab (JS) | `kinase_audit.js` | medium | shared contract only (see below); can build against a fixture |

## Collisions & resolution

- **File-disjoint.** 01 touches only Python files; 02 touches only
  `kinase_audit.js`. No file is edited by both. Per the cut principle
  (file-disjoint + independently verifiable) they are **parallel-safe**.
- The Explorer columns (`kinase_explorer.js:205-208`) are edited by **neither**
  subplan — revision #1 keeps them reading the existing columnar arrays that 01
  continues to populate.
- The only coupling is a **data contract**, not a file: the payload block shape
  and the sidecar/manifest entry that 01 produces and 02 consumes. It is pinned
  below and embedded in both subplans. 02 builds against a fixture matching it,
  so no serialize dependency is required. **Integration check** (real build →
  real tab render) runs after both land.

## Pinned interface contract (do not drift — both subplans depend on this exact shape)

### Inlined payload block (01 writes, 02 reads)

```
payload["kinase_incytr_participation"] = {
  "<kinase name>": {
    "counts":      { "pathways": <int>, "backbones": <int>, "total": <int> },
    "by_role":     { "Receptor": <int>, "EM": <int>, "Target": <int> },   // role-membership (a row at >1 role counts under each); NOT a partition
    "by_contrast": { "<contrast_row>": <int>, ... },                       // partitions pathways; values sum to counts.pathways
    "by_receiver": { "<receiver cluster>": <int>, ... }                    // partitions pathways; values sum to counts.pathways
  },
  ...
}
```
- donor1-only. Absent for donor2 (drives 02's tab-inclusion gate).
- `counts.pathways` = |union of role masks|; `counts.backbones` = |Receptor∪EM|
  (≤ pathways); `counts.total` = pathway total for the %-denominator.
- `contrast_row` is the row-form vocabulary from `_terminal_contrast_to_row`.

### Edge sidecar (01 writes CSV + registers manifest entry, 02 fetches + slices)

- Manifest key: **`kinase_incytr_edges`** (02 calls `AuditDataStore.load("kinase_incytr_edges")`).
- CSV columns, in order:
  `kinase, target_gene, role, contrast, receiver, pathways, signed_nes, best_fdr, n_sites, edge_delta, n_significant_concordant, motif_peers_detected, motif_peers_informative`
  - `contrast` in row form (`_terminal_contrast_to_row`); `pathways` ≥ 1 (count-0 edges dropped).
- Manifest entry built via `_audit_csv_meta(dest, label, key)` (`slices_audit.py:143`)
  so `columns`/`relative_path`/`preview` match every other audit table exactly.
  `preview` stays **empty** (`[]`) — that is what makes `file://` fall through to
  the note (same as the empty-shim at `slices_audit.py:235-244`).

## Model-mapping policy (retune here; no subplan hard-codes a model)

`high → strongest model`, `medium → mid`, `low → mid/cheapest`.
So: **01 → strongest**, **02 → mid**.

## After implementation

Suggested (not wired): run `/code-review` on each returned diff before merge, and
run the parent plan's integration verification (memory-capped `pixi run` build →
serve over HTTP → confirm headline counts == Explorer columns, cross-link lands).

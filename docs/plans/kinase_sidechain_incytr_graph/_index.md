# kinase_sidechain_incytr_graph — subplan dependency index

Aligned + grilled, then decomposed by `/orchestrate` into four **file-disjoint** subplans — no two touch the same file, so
none can merge-conflict. The ordering below is a **data-flow dependency** (backend artifact → payload
→ viewer), not a file collision: a downstream subplan can be *authored* in parallel against the
contract, but its acceptance check only passes once its upstream lands.

## Subplans

| # | File | Unit | Risk | Files touched |
|---|------|------|------|---------------|
| 01 | `01_backend-edge-model.md` | PSP loader + kinase→kinase interactome + terminal-edge map (song, 5xFAD) | **high** | **new** `alz/cross_reference/kinase_kinase_edges.py` |
| 02 | `02_tcell-motif-bridge.md` | T-cell single-hop motif bridge, per-donor + celltype attribution | **medium** | `alz/cross_reference/kinase_incytr_bridge.py` |
| 03 | `03_payload-slice.md` | Per-cohort edge_slice_ref for the sidechain tab | **medium** | `alz/viewer/shared/compose.py`, `alz/viewer/shared/incytr_index.py`, `alz/viewer/cohorts/song.py`, `alz/viewer/cohorts/fivexfad.py`, `alz/tcell_viewer/slices_incytr.py` |
| 04 | `04_viewer-tab.md` | Cytoscape one-pathway tab + registration | **low** | **new** `alz/viewer_shared/template/js/tabs/incytr_sidechains.js`, `alz/viewer/template/js/02_ui_chrome.js`, `alz/tcell_viewer/template/js/02_ui_chrome.js` |

## Model-mapping policy

Risk tier = **silent-failure risk**, not complexity — `(correctness-critical logic) × (how
undetectable a wrong answer is)`. Route accordingly (retune the mapping here; no subplan hard-codes a
model):

- **high → strongest model** (Opus). Wrong output looks plausible and ships silently.
- **medium → mid model** (Sonnet). Mostly loud failures, with a named silent spot to watch.
- **low → mid/cheapest** (Sonnet/Haiku). Fails loudly; the acceptance check catches it.

Note 04 is the largest subplan but **low** risk — a wrong render is visible on inspection. Complexity
would have mis-routed it to the strongest model and 01 (small, silent) to the cheapest, backwards.

## Collisions

**None.** All four subplans are file-disjoint (confirmed in-code, not from the graph). No serialize
order is forced by a shared file; no merge is required.

## Declared data dependencies (prerequisite order, not file locks)

- **01 and 02 are independent producers** — run in parallel. 01 builds the literature backbone + the
  cohort-parameterized terminal-map builder (song/5xFAD motif already exists in
  `kinase_incytr_bridge.py`); 02 produces the t-cell motif edges in the *same schema* as song/5xFAD.
  Contract between them: 01's terminal-map builder must accept a motif-edge input so 02's t-cell output
  can be assembled by 03. Surfaced in both subplans.
- **03 depends on 01 + 02** — the payload slice reads 01's per-cohort backend artifact (all cohorts)
  and, for the t-cell arm, 02's motif edges. Cannot verify without both.
- **04 depends on 03** — the tab renders the real `edge_slice_ref` payload keys 03 emits.

Effective order: **[01 ∥ 02] → 03 → 04.**

## Before merge

These subplans may be handed to different tools (Claude agent, Codex, Copilot, a human) — each carries
its operating core inline. When each returns a diff, running `/code-review` on it before merge is
**suggested** (this is a data pipeline; the backend edge-weight and provenance math is the part most
worth an adversarial pass). Suggested, not wired — the review call is yours.

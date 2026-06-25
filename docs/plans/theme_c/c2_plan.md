# Theme C2 — Cohort display rename

**Contract:** `_contracts.md §C2`. **Audit/worklist:** `c2_audit.md`. **Wave:** 1 (contract producer). **Prerequisite-for:** C1 (genotype split builds on the renamed `MouseC1` label), and the F2 export sweep (consumes the map). **Collision class:** unified-viewer builder + its JS template; serialize against C1/C3 within the builder.

## Decisions (locked, P3 grill 2026-06-24)
- Map governs **programmatic construction sites only**; pervasive prose/tooltip strings are a literal find-replace. No runtime map for sentences (over-engineering).
- **Two coordinated constants** — `COHORT_DISPLAY` (Python, `build_unified_viewer.py`) for context/audit labels; `COHORT_LABELS` (JS) for structural labels. The Python-build/JS-runtime boundary makes one shared object impossible; this is not a shim.
- Mode buttons **keep the grouping word**: `Mouse (MouseC1)`, `Mouse (MouseC2)`, `Human (HumanC1)`.
- Provenance stamps stay the literal model name `5xFAD` (correct provenance; leave unless rendered as a visible label).
- CSV export (headers + filenames) is **NOT in C2** — F2 owns the export boundary and reads C2's map (Wave 4).

## Display name table
| Internal key | Display | Split suffix |
|---|---|---|
| `song` | `MouseC1` | C1 → `MouseC1_App/Tau/ApTt` |
| `fivexfad` | `MouseC2` | `MouseC2 Cortex` / `MouseC2 Hippocampus` |
| `mukesh` | `HumanC1` | — |

---

## Stage 0 — Verify the two ambiguities (no edits)
Decides whether 2 borderline surfaces are in scope.
1. Does the Audit tab JS render the provenance `cohort` field as a visible label? Grep the audit-tab JS for the audit-JSON `cohort` key. If rendered → add to Stage 2; if not → leave provenance as `5xFAD`.
2. Does any JS render `PAYLOAD.supporting_5xfad.cohort` (`fivexfad.py:1590`) as text? Scan `kinase_fivexfad.js`. If not → confirmed internal, no edit.

**Gate:** record both answers inline in this file before proceeding.

## Stage 1 — Establish the two constants
- **Python** `build_unified_viewer.py`: add module-level
  ```python
  COHORT_DISPLAY = {"song": "MouseC1", "fivexfad": "MouseC2", "mukesh": "HumanC1"}
  ```
  Rewrite the inline label sites to derive from it: `"Song AD"` (983) → `COHORT_DISPLAY["song"]`; `_5xfad_incytr_ctx_labels` (1046–1049) → `f'{COHORT_DISPLAY["fivexfad"]} Cortex'` / `… Hippocampus`; audit-table labels (171, 173) → `f'{COHORT_DISPLAY["fivexfad"]} sample manifest'` / `… dataset index`.
- **JS** one constant near the top of `01_state.js`:
  ```js
  const COHORT_LABELS = { song: "MouseC1", fivexfad: "MouseC2", mukesh: "HumanC1" };
  ```
  Used by the structural-label sites in Stage 3 (dropdown options, `_KX_COMPARE_LABELS`, rendered `TH(...)` headers). Prose stays literal.

**Note:** `"Song AD"` → `MouseC1` here is provisional — C1 replaces this context label with the per-genotype split. C2 only de-surnames it.

## Stage 2 — `body.html` (structural labels)
- Mode buttons (5–7): `Mouse (MouseC1)`, `Mouse (MouseC2)`, `Human (HumanC1)`.
- Filter labels/tooltips (51–53, 92, 398): `Song` → `MouseC1`.
- Compare dropdown (431–434): `MouseC1 vs HumanC1`, `MouseC1 vs MouseC2`, `HumanC1 vs MouseC2`, `MouseC2 tissue split`.

## Stage 3 — JS display strings
**Structural (via `COHORT_LABELS` where a lookup is natural):**
- `_KX_COMPARE_LABELS` (crosstable 90–96): `MouseC1 vs HumanC1` etc.
- Rendered `TH(...)` headers (crosstable 1181–1190): `Mouse`→`MouseC1`, `Human`→`HumanC1`, `5xFAD cortex`→`MouseC2 cortex`, `5xFAD hip`→`MouseC2 hip`; tooltips `Song location evidence`→`MouseC1 location evidence`.
- `ATTR_VERDICT_COLS` labels (audit 589–596): `Song`→`MouseC1`, `Song LFC`→`MouseC1 LFC`.
- `F5_ATTR_COLS` (fivexfad 281): `5xFAD snRNA`→`MouseC2 snRNA`.

**Prose / tooltips / messages (literal find-replace of the cohort token):**
- crosstable 572/579/591–592/804–806/1003/1045/1056/1159/1401/1467–1469/1480–1481
- audit 576/792/800–802/849/1052/1059/1066/1076
- explorer 658–659
- fivexfad 1671–1672/1718/1731/1735
- TAB_GUIDE prose (01_state 266/268/272/280/293/296)

Token map for replace: `Song`→`MouseC1`, `Mukesh`→`HumanC1`, `5xFAD`→`MouseC2` (preserve surrounding words, e.g. `5xFAD cortex`→`MouseC2 cortex`, `Mukesh AD`→`HumanC1 AD`).

## Stage 4 — `_tissueLabel()` regex
`02_ui_chrome.js:143`: `/^5xFAD\s+/i` → `/^MouseC2\s+/i` (must match the Stage 1 context-label rename or the tissue dropdown shows the full `MouseC2 Cortex` instead of `Cortex`).

---

## Verification
- `pixi run viewer` → hard-refresh (Ctrl+Shift+R). Check `PAYLOAD.meta.generated_at` is fresh.
- **Browser click-through (human, authoritative):** mode buttons read `Mouse (MouseC1/C2)` / `Human (HumanC1)`; compare dropdown + crossplay badges read the new pair names; crosstable column headers + detail panel read `MouseC1`/`HumanC1`/`MouseC2`; tissue dropdown still shows bare `Cortex`/`Hippocampus` (Stage 4 regex); no stray `Song`/`Mukesh`/`5xFAD` visible in any tab/tooltip/how-to drawer.
- `command grep -rn 'Song\|Mukesh\|5xFAD' alz/viewer/template alz/build_unified_viewer.py` → every remaining hit is an internal key, a payload field, a code comment, or an intentional provenance/atlas string (no visible-label hits).

## Out of scope (do not touch in C2)
CSV export headers/filenames (F2, Wave 4), internal keys / payload keys / `cohort_id` / data paths / module dirs, provenance stamps (unless Stage 0 finds them rendered), T-cell viewer, reference-atlas names.

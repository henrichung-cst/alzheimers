# rclone ingest → reusable SSOT (shared submodule + per-project manifest)

**Date:** 2026-06-08
**Status:** PROPOSED — awaiting approval before any file edits
**Decisions locked (user):** (1) distribution = **shared repo vendored as a git submodule**; (2) scope = **both remotes** (`gdrive_shared:` + `gdrive:`, incl. the bench sce4 fetches).

---

## 1. Goal

Replace the six hardcoded, copy-pasted rclone call sites with **one declarative manifest per project** driven by **one shared engine**. The engine is project-agnostic and lives in its own repo; each project carries only a `conf/data_sources.yaml` describing *its* data sources. Adding a new project = add the submodule + write a manifest. No engine code is duplicated.

Secondary wins this buys:
- **Remote preflight** (the gap the audit flagged): fail fast with a clear message if `gdrive_shared:`/`gdrive:` aren't configured in `rclone.conf`.
- **Folder IDs centralized** — one place, not four.
- **`--dry-run`** to print the exact rclone commands before transferring.

## 2. Current state (from the audit)

| Call site | Remote | What it does |
|---|---|---|
| `pixi.toml:10` ingest-gdrive-shared | `gdrive_shared:` | whole-folder copy |
| `pixi.toml:11` ingest-lucie-proteomics | `gdrive_shared:` | whole-folder copy |
| `pixi.toml:12` ingest-deconvolution-bulk | `gdrive_shared:` | filtered (max-depth 1, 4 includes) |
| `alz/runners/supporting/ingest_tcells.sh` | `gdrive_shared:` | 5 proteomics files + optional `--scrna` group (2 × ~5 GB rds) |
| `bench/perf/fetch_sce4_canonical.sh` | `gdrive:` | 2 `copyto` single-file fetches + 1 bespoke md5-compare-and-delete |
| `bench/perf/download_sce4_source.sh` | `gdrive:` | full 7.5 GiB dir, tuned flags (`--transfers=4 --checkers=8 --drive-acknowledge-abuse`) |

All pull-only, all via the two pre-configured rclone remotes. No Drive API/creds in-repo.

## 3. Target architecture

```
~/Projects/work/rclone-ingest/            # NEW shared repo (the SSOT engine)
  pyproject.toml                          # console_scripts: rclone-ingest = rclone_ingest.cli:main
  rclone_ingest/
    __init__.py
    cli.py                                # arg parsing + dispatch
    manifest.py                           # load + validate YAML against schema
    engine.py                             # build rclone argv, preflight, run
  schema/data_sources.schema.json         # JSON-schema the manifest validates against
  examples/data_sources.yaml              # documented reference manifest
  README.md                               # adoption recipe for new projects

alzheimers/                               # this project (a CONSUMER)
  .gitmodules                             # vendor/rclone-ingest -> shared repo
  vendor/rclone-ingest/                   # submodule (pinned commit)
  conf/data_sources.yaml                  # THIS project's sources (the only per-project file)
  pixi.toml                               # tasks call the `rclone-ingest` CLI
```

**SSOT split:** engine + schema are shared (one copy, versioned by submodule commit). *Data-source definitions are per-project* — they belong to the consuming repo, not the engine. That's the line that keeps the engine reusable.

**How pixi loads it:** install the submodule as an editable path dep so the console-script is on PATH inside the env:

```toml
[pypi-dependencies]
rclone-ingest = { path = "vendor/rclone-ingest", editable = true }
```

Then tasks are thin: `rclone-ingest sync <name>`.

## 4. Manifest schema (`conf/data_sources.yaml`)

```yaml
version: 1

# Remote names MUST match `rclone listremotes` on the box. Engine preflights this.
remotes:
  gdrive_shared: { description: "Shared collaborator drive (Song/yuyu, Lucie, T-cells)" }
  gdrive:        { description: "Personal/project drive (sce4 provenance)" }

# rclone flags applied to every transfer unless a transfer overrides.
defaults:
  flags: ["--progress"]
  retries: 3

sources:
  <name>:
    remote: <remote-name>
    folder_id: <drive-root-folder-id>
    transfers:
      - mode: copy|copyto        # default copy; copyto = single-file rename
        src:  "<path under folder_id>"   # "" = the folder root
        dest: "<path relative to project root>"
        group: <optional tag>    # transfers w/o a group always run; grouped run only when requested
        max_depth: <int?>
        include: [<glob>, ...]
        flags: [<extra rclone flags>]    # appended to defaults.flags
```

Semantics:
- A **source** = one named ingest unit (maps to one pixi task).
- A source has N **transfers**, each = one `rclone` invocation.
- **Groups**: ungrouped transfers always run; `--group scrna` adds the `scrna`-tagged ones. (Models tcells' `--scrna`.)
- Engine auto-`mkdir -p`s each `dest`, resolves `dest` relative to project root (CWD, or `--root`).

## 5. The alzheimers manifest (full, all 6 sources)

```yaml
version: 1
remotes:
  gdrive_shared: { description: "Shared collaborator drive" }
  gdrive:        { description: "Personal drive — sce4 provenance" }
defaults:
  flags: ["--progress"]
  retries: 3

sources:
  gdrive-shared:
    remote: gdrive_shared
    folder_id: 1syiic6d9DUJIc1sPL5wc8uL7Od3Dsq3Z
    transfers:
      - { src: "", dest: "data/external/gdrive_shared/" }

  lucie-proteomics:
    remote: gdrive_shared
    folder_id: 1uWjRb_ZnrDp9HwgijE48zhb8gvJjOr2P
    transfers:
      - { src: "", dest: "data/external/lucie_proteomics/" }

  deconvolution-bulk:
    remote: gdrive_shared
    folder_id: 1syiic6d9DUJIc1sPL5wc8uL7Od3Dsq3Z
    transfers:
      - src: "yuyu01/documentation/incytr/deconvolution/"
        dest: "data/datasets/song/proteomics/source/"
        max_depth: 1
        include: [imac_median.csv, py_median.csv, pr_median.csv, yuyu_samplekey.csv]

  # In scope: Total + pY ForPerseus (both donors), Donor1 IMAC. scrna group = ~10 GB .rds.
  # Out of scope: KGG/AcK/MME, Flow, NotParsed reports, collaborator Log2FC. (2026-05-27 notes)
  tcells:
    remote: gdrive_shared
    folder_id: 1YE_h1jIyBajtm6ArxJqevJ0rt0xLKQgX
    transfers:
      - { src: "T Cell Exhaustion Donor 1/Proteomics Data/Total with Ensembl/10Feb2026_Donor1_TotalProteome_ForPerseus.txt", dest: "data/datasets/tcells/donor1/proteomics/" }
      - { src: "T Cell Exhaustion Donor 1/Proteomics Data/pY with Ensembl/10Feb2026_Donor1_pY_ForPerseus.txt",               dest: "data/datasets/tcells/donor1/proteomics/" }
      - { src: "T Cell Exhaustion Donor 1/Proteomics Data/IMAC with Ensembl/18May2026_TCellDonor1_Normalized_IMACSiteReporttsv.tsv", dest: "data/datasets/tcells/donor1/proteomics/" }
      - { src: "T Cell Exhaustion Donor 2/Proteomics Data/Total/10Feb2026_Donor2_TotalProteome_ForPerseus.txt", dest: "data/datasets/tcells/donor2/proteomics/" }
      - { src: "T Cell Exhaustion Donor 2/Proteomics Data/pY/10Feb2026_Donor2_pY_ForPerseus.txt",               dest: "data/datasets/tcells/donor2/proteomics/" }
      - { group: scrna, src: "T Cell Exhaustion Donor 1/Single Cell Data/Tcells.singlet.rds",       dest: "data/datasets/tcells/donor1/scrna/" }
      - { group: scrna, src: "T Cell Exhaustion Donor 2/Single Cell Data/Tcells_d2.singlet (1).rds", dest: "data/datasets/tcells/donor2/scrna/" }

  sce4-canonical:
    remote: gdrive
    folder_id: 1yAAvtXNlrXDZhp7nwVFMG46HbQAaQfUQ
    transfers:
      - mode: copyto
        src:  "Analysis_new cluster labels_cutoff_0.1/DEG_PRG_ma_2mo_ApTt_WTyp_10302025/sce4_DEG_PRG_Top300_table_10302025.csv"
        dest: "data/incytr_frozen/outputs/Analysis_new cluster labels_cutoff_0.1/DEG_PRG_ma_2mo_ApTt_WTyp_10302025/sce4_DEG_PRG_Top300_table_10302025.csv"
      - mode: copyto
        src:  "deconvolution_with_new_clusters_20250721/limma_corrected_IMAC_yuyu.tsv"
        dest: "data/incytr_frozen/v2_46clusters/provenance/limma_corrected_IMAC_yuyu.tsv"

  sce4-source:
    remote: gdrive
    folder_id: 1yAAvtXNlrXDZhp7nwVFMG46HbQAaQfUQ
    transfers:
      - src: ""
        dest: "data/incytr_frozen/sce4_source/"
        flags: ["--transfers=4", "--checkers=8", "--drive-acknowledge-abuse", "--ignore-existing"]
```

## 6. Engine CLI

```
rclone-ingest sync <name> [--group G ...] [--dry-run] [--root DIR] [--manifest PATH]
rclone-ingest sync --all  [--dry-run]
rclone-ingest list                     # show sources + transfer counts
rclone-ingest check                    # preflight only: rclone present + remotes configured
```

- **Preflight** (every `sync`): `rclone` on PATH; each referenced remote present in `rclone listremotes`. Missing remote → exit non-zero with `remote 'gdrive_shared:' not configured — run 'rclone config'`. This is the audit's missing safety net.
- **Manifest default path:** `conf/data_sources.yaml` resolved from `--root` (default CWD). pixi runs from repo root, so it just works.
- **Build argv** from `defaults.flags + source/transfer flags`, `--drive-root-folder-id <id>`, `--retries`, `--max-depth`, repeated `--include`. `copy` → `rclone copy <remote>:<src> <dest>`; `copyto` → `rclone copyto <remote>:<src> <dest-file>`.
- **`--dry-run`** prints each command, runs nothing.
- Validates manifest against `schema/data_sources.schema.json` on load; clear error on unknown keys / bad types.

## 7. pixi task rewrite (before → after)

```toml
# before (4 lines hardcoded) → after:
ingest-gdrive-shared      = "rclone-ingest sync gdrive-shared"
ingest-lucie-proteomics   = "rclone-ingest sync lucie-proteomics"
ingest-deconvolution-bulk = "rclone-ingest sync deconvolution-bulk"
ingest-tcells             = "rclone-ingest sync tcells"
ingest-tcells-scrna       = "rclone-ingest sync tcells --group scrna"
# new, replacing the bench scripts:
ingest-sce4-canonical     = "rclone-ingest sync sce4-canonical"
ingest-sce4-source        = "rclone-ingest sync sce4-source"
```

## 8. File-by-file change set

**New shared repo `~/Projects/work/rclone-ingest/`** (separate git init): `pyproject.toml`, `rclone_ingest/{__init__,cli,manifest,engine}.py`, `schema/data_sources.schema.json`, `examples/data_sources.yaml`, `README.md`.

**alzheimers — create:**
- `.gitmodules` + `vendor/rclone-ingest` submodule
- `conf/data_sources.yaml` (section 5)

**alzheimers — modify:**
- `pixi.toml`: rewrite the 5 ingest tasks (§7), add the 2 sce4 tasks, add the editable path dep
- `README.md:196-200`: ingest section → `rclone-ingest sync …` + submodule-init note
- `data/README.md`: rclone-target table → point at the manifest
- `docs/foundation/repo_retention_policy.md:83-84`: same

**alzheimers — delete (replaced, per anti-shim):**
- `alz/runners/supporting/ingest_tcells.sh` (→ `tcells` source; scope comments preserved in manifest)
- `bench/perf/download_sce4_source.sh` (→ `sce4-source` source)
- `bench/perf/fetch_sce4_canonical.sh` — **partial:** copies move to `sce4-canonical`; the bespoke `renamed_sobj.rds` md5-compare-and-delete (§3 of that script) does NOT generalize. It becomes a slim `bench/perf/verify_sce4_scoring_object.sh` that calls `rclone-ingest sync` for the fetch, then does only the md5 check. **This is the one place the manifest can't fully absorb** — flagging explicitly.

## 9. Open item (needs your call at implement time)

**Submodule URL.** A git submodule needs a fetchable URL to be portable to other boxes/projects. Options:
- (a) Push `rclone-ingest` to a GitHub repo (cellsignal org or personal) and reference that URL — most portable, but I won't push without explicit go-ahead.
- (b) Local-path submodule (`file://~/Projects/work/rclone-ingest`) — works on this box only; fine for now, re-point to GitHub later.

I'll default to (b) for the first cut unless you say push. Everything else in the plan is independent of this choice.

## 10. Rollout / verification

1. Build engine + `rclone-ingest check` → confirms both remotes resolve.
2. `rclone-ingest sync --all --dry-run` → diff printed commands against the 6 original call sites (must be byte-equivalent rclone argv). This is the correctness gate before deleting anything.
3. `pixi install` (picks up the editable path dep), then `rclone-ingest list`.
4. Spot-run one cheap real sync (`deconvolution-bulk`, 4 small CSVs) end-to-end.
5. Only after 2–4 pass: delete the old scripts.

## 11. Adopting in a new project (the payoff)

```bash
git submodule add <rclone-ingest-url> vendor/rclone-ingest
# add the editable path dep to pixi.toml [pypi-dependencies]
$EDITOR conf/data_sources.yaml        # describe that project's sources
pixi install && rclone-ingest check
```

No engine code copied. Engine fixes propagate by bumping the submodule commit.

# Theme D1 — Cross-cohort substrate phosphosite comparator

**TODO:** §D1. **Audit:** dispatched 2026-06-25 (Explore). **Wave:** 2 (disjoint analysis; greenfield module — parallel-safe). **Not a contract producer.** **Consumer/sibling:** C5 reuses the engine as a one-off. **Collision class:** disjoint — new file `alz/cross_reference/substrate_motif_compare.py` + new output dir; **no viewer/builder/pipeline edits** (analysis→display invariant: the viewer never computes).

## Decisions (locked, P3 grill 2026-06-25, one question at a time)

**Q1 — Substrate composition = GSEA leading-edge substrates, annotated with site-LFC direction.**
The `Leading substrates` column is a uniform stored artifact in every cohort's MEA enrichment table (`mea_stoichiometry.csv` / `mea_perdonor.csv` / `{tissue}_{track}_mea_stoichiometry.csv`) — not human-only. Chosen over (a) the full KL substrate set (`mea_substrate_sets.csv`), which scales with detection breadth → overlap would partly measure assay depth, a coverage artifact; and over (c) FDR-significant sites alone, which aren't kinase-attributed. Leading-edge is intrinsic to the kinase's activity call. Direction layered from the site LFC (option-c data as the *direction source*, not the composition unit). **Lift the motif→site/gene bridge from `alz/ctrl_outlier_audit/concordance_overlap_AD_excl_01_03.py`** (`_leading_motifs`, `_substrate_rows`): leading-edge motif → uppercased → join `stoichiometry_matrix.csv` `motif` column → `(site_id, gene_symbol, site_position)`. `site_id` is cohort-local — never a cross-cohort key.

**Q2 — Comparison is motif-level everywhere, via center-aligned graded similarity.**
The scientific goal is *whether a kinase's contributing motifs are species-specific* — gene-level collapse would answer a different question (same protein?) and hide that signal. Problems with naive string-equality and their resolution:
- Motifs are **center-aligned by construction** (fixed ±7 window on the phospho-acceptor) → position-by-position comparison needs **no alignment step** (dissolves the "no residue map" problem).
- Replace exact-equality with a **graded match** over the center-aligned window: **exact** (identical 15-mer) = conserved; **conserved-with-substitution** (high BLOSUM62/identity over ±7, same central residue type) = drifted flank, *the mismatching positions name the species-specific residues*; **species-unique** (no counterpart above floor) = divergent substrate.
- **Carry the continuous similarity score; bin only for display** (exact/conserved/unique is an interpretation layer, not a hidden cutoff — surface the spectrum numerically).
- **Gene case-flip** (`Rasgrf1↔RASGRF1`) is an **optional disambiguator** (high similarity *at the orthologous gene* = true conserved site; *at a different gene* = convergent/coincidental), **not** the join key. The homologene cache covers only 389 kinases, not the substrate universe → case-flip is approximate → **report unmatched genes explicitly, never silently drop** (a high miss rate flags the need for a MyGene-homologene substrate-ortholog table, noted follow-up).
- **Track (ST/pY) is part of identity** — ST motifs compare only to ST. Within-species pairs simply land at the "exact" end of the same spectrum far more often (no separate code path).

**Q3 — Symmetric overlap-decomposition primitive + `comparison_mode` switch.**
The primitive: *two substrate profiles → `shared`, `A-only`, `B-only`, + direction-agreement on `shared`.* Conservation framing (C5: Jaccard, shared-direction fraction, direction correlation) and discrimination framing (GRF "what separates them": the set-differences) both read off the one decomposition. `comparison_mode`: **`matched`** (pair profiles by kinase across two cohort groups — cross-cohort / species-specificity / C5) | **`all-pairs`** (every pair within one set — within-cohort family discrimination). Cross-cohort and within-cohort GRF are *invocations*, not separate engines. Axis-generic costs nothing structural — the primitive is symmetric and needed regardless.

**Q4 — Python analysis, not a viewer panel (repo invariant: analysis→Python, viewer = display-only over precomputed results, never computes).**
`alz/cross_reference/substrate_motif_compare.py` (the cross-cohort-comparison module — siblings: `evidence.py`, `seaad_human_agreement.py`, `tcell_within_cohort.py`). Pixi task `substrate-compare`, outputs to `outputs/reports/substrate_compare/<run>/`. The TODO's "select a pool / select another pool" is an **analysis parameter** (CLI/config), re-run to change pools. Emit **cleanly-keyed, flat, viewer-consumable CSVs** (one row per compared substrate) as output hygiene so a *future, separate* display theme can read them — but D1 builds **no shards/payload** itself (no speculative viewer plumbing; the motif-matching logic stays canonical in Python, never duplicated in JS).

**Q5 — Explicit kinase lists only; no top-N helper.**
Pool input = an explicit list of kinase names, one per cohort side. Selection is the caller's concern. The audit's "5xFAD has no ranked table" gap is moot (top-N dropped); for the record it was only a *missing summary* — all three cohorts have per-(kinase,contrast) NES/FDR in their MEA tables. **C5 is a separate one-off**, considered apart from this stable infrastructure: its pool ("agreement across mukesh AD / preclinical / controls", where *preclinical* = the **suspect** AD-like controls CTRL-08/10/… from `concordance_AD8_excl01_03`) is computed entirely inside C5 and handed to D1 as a list. D1 bakes in no selection criteria.

**Q6 — Single caller-specified contrast per side; human gets two profile builders, per-donor default.**
Profile identity `(kinase, cohort, contrast)`. **Single contrast, not a union** → leading-edge membership *and* direction are both unambiguous; cross-timecourse/genotype comparisons are **separate invocations**. Disease contrast per side: Song → `{genotype}_{age}mo`; 5xFAD → `(tissue, track, TG_vs_WT_{age})`; Human → see below.
- **Human per-donor (PRIORITY, default):** reads the existing per-donor leading-edge (`mea_perdonor.csv`) and aggregates by **donor recurrence** — motif in profile if leading-edge in ≥ M of N AD donors; direction = per-donor sign votes (n_up/n_down); `support` = donor count (heterogeneity stays visible). **No MEA run, no new artifact.** Faithful to the cohort's deliberate per-donor design (it does NOT group-average). Aggregation = recurrence-consensus, the substrate-level analog of `recurrence.csv`'s `n_donors_sig` (veto on review).
- **Human per-group (READY, secondary):** AD-vs-CTRL **group MEA** leading-edge — the only path needing a materialized `human_group_mea.csv` (lift `_group_contrast` from the concordance script). Built and available, not default.
- The comparator is **profile-mode-agnostic**: every builder emits `{motif: (direction, support, gene, site_position)}`; mouse single-contrast emits it with `support=1`.

## Stages

**Stage 1 — Profile builders** (`build_profile(kinase, cohort, contrast|mode) → {motif: (direction, support, gene, site_position, track)}`):
- Mouse (Song / 5xFAD): leading-edge at the chosen contrast from the MEA table; bridge motif→site/gene via that cohort's `stoichiometry_matrix.csv`; direction = sign of `stoich_lfc` at the contrast; `support=1`.
- Human per-donor: per-donor leading-edge from `mea_perdonor.csv`; recurrence-consensus (threshold M); direction = donor sign votes from per-donor site LFC (donor − CTRL-mean, from the stoichiometry matrix); `support = n_donors`.
- Human per-group: leading-edge from `human_group_mea.csv` (Stage 5 emits it); `support=n_AD`.
- **DuckDB-streamed, filtered to the pool kinases at scan** — never whole-file pandas on the 47–116 MB substrate-set files or the stoichiometry matrices; leading-edge sets are small.

**Stage 2 — Comparison primitive** (`compare(profile_A, profile_B) → decomposition`): center-aligned graded motif match (Stage 3) → `shared` / `A-only` / `B-only`; on `shared`, direction agreement (same/opposite sign). `comparison_mode` (`matched` | `all-pairs`) selects which profile pairs feed it.

**Stage 3 — Motif similarity** (`motif_similarity(m_a, m_b)`): center-aligned BLOSUM62 (or identity) over the ±7 window, same central residue type required; continuous score + bin (exact / conserved / unique). Gene case-flip disambiguator; collect unmatched leading-edge genes for explicit reporting.

**Stage 4 — Outputs** to `outputs/reports/substrate_compare/<run>/`:
- `substrate_pairs.csv` (flat, viewer-consumable): `kinase, cohort_a, contrast_a, cohort_b, contrast_b, motif_a, motif_b, gene_a, gene_b, site_a, site_b, track, match_class, similarity, direction_a, direction_b, direction_agree, support_a, support_b`.
- `kinase_summary.csv`: per kinase — `n_shared, n_a_only, n_b_only, jaccard, direction_agree_frac, direction_corr, conservation_class_counts, n_genes_unmatched`.
- Figures (existing matplotlib/`matplotlib_venn` convention from `ctrl_outlier_audit_report_figs.py`): per-kinase Venn, similarity-spectrum histogram, direction scatter.
- `MANIFEST.md`: run params (pools, cohorts, contrasts, mode, M, similarity cutoff), provenance.

**Stage 5 — Runner + task:** pixi task `substrate-compare = "python -m alz.cross_reference.substrate_motif_compare"` with pool/cohort/contrast/mode args. For per-group human mode only: a small upstream `human_group_mea` emission (lift `_group_contrast`) → `human_group_mea.csv`; skipped when running per-donor (default).

## Verification
- **GRF1-5 runs:** within-cohort `all-pairs` (what separates the family members) + cross-species `matched` (Rasgrf1 mouse vs RASGRF1 human — are its motifs conserved?). Both produce non-empty decompositions.
- A known conserved phosphopeptide lands `match_class=exact`; a known divergent one lands `species-unique`; the mismatching positions in a `conserved` case are reported.
- Direction agreement on a shared substrate matches the underlying site LFC signs in both cohorts; human per-donor direction = the donor sign-vote majority.
- **Honesty:** `n_genes_unmatched` is reported, not hidden; a low-support (few-donor) human substrate is PRESENT with `support` shown, not dropped; the similarity spectrum is surfaced (not collapsed to a boolean).
- **Memory:** runs under the cap; peak RSS reported; no whole-file pandas read of any substrate-set CSV or stoichiometry matrix (DuckDB-streamed, pool-filtered at scan).

## Out of scope
The KL-full-substrate-set composition (Q1), gene-level-only comparison (Q2), top-N pool selection (Q5), any viewer panel/shard/payload (Q4 — display is a separate downstream theme that reads these CSVs), C5's pool-selection logic (one-off, stays in C5), a residue-level cross-species alignment table (motif center-alignment suffices), the MyGene substrate-ortholog table (follow-up only if case-flip miss rate is high).

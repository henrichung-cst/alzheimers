// ---------------------------------------------------------------------------
// OmicsTraceStore — per-cluster protein + phospho raw-value shards backing
// the Incytr Pathways "Evidence" tab.
//
// Shard layout: outputs/reports/unified_viewer/audit_sources/omics_trace/
// <slug>.parquet where slug = sanitize_celltype(cluster).
//
// Schema per shard row (from build_omics_trace.py, schema_version=1):
//   layer        : "protein" | "phospho_ps" | "phospho_py"
//   gene_symbol  : string
//   site_id      : string | null (null for protein rows)
//   animal_id    : string  (e.g. "37_E50(L)_M_4mo_WT")
//   group        : string  (e.g. "ma_2mo_AppP")
//   sex          : "M" | "F"
//   timepoint    : "2mo" | "4mo" | "6mo"
//   genotype     : "AppP" | "Ttau" | "ApTt" | "WTyp"
//   value        : float (raw intensity)
//   log2_value   : float (log2(value), NaN when value==0)
//
// Per-animal (3 males per group arm). Use value column for LFC; log2_value
// is display-only (see Item 3.2 implementation notes).
//
// Cluster routing per incytr/R/evaluation.R:227-230:
//   Ligand → sender cluster
//   Receptor → receiver cluster
//   EM → receiver cluster
//   Target → receiver cluster
// The `omics_trace_normalized` sidecar still ships in the build (used by
// `verify_pathway_round_trip.py` as the build-time gate for stored ↔
// recomputed LFC agreement, strict 1e-4) but is not consumed by the viewer.
// The Evidence tab shows the Incytr-stored LFC chip (from the pair-mode wide
// parquet) alongside per-animal evidence (from OmicsTraceStore); no second
// derived value is displayed, eliminating the prior coverage/value-mismatch
// confusion between the two sidecars.
// ---------------------------------------------------------------------------

const OmicsTraceStore = (() => {
  const cache = new Map();        // cluster -> rows[]
  const inflight = new Map();     // cluster -> Promise<rows>

  // Mirror of alz/incytr_pair/pair_to_receiver_cache.py::_sanitize_celltype.
  function _sanitize(name) {
    return String(name).replaceAll("/", "-").replaceAll(" ", "_");
  }

  function _meta() {
    return (typeof PAYLOAD !== "undefined"
            && PAYLOAD.meta
            && PAYLOAD.meta.omics_trace) || null;
  }

  function isAvailable() {
    const m = _meta();
    return !!(m && Array.isArray(m.clusters) && m.clusters.length);
  }

  function hasCluster(cluster) {
    const m = _meta();
    if (!m || !cluster) return false;
    return (m.clusters || []).includes(cluster);
  }

  async function _fetchParquet(url) {
    let resp;
    try {
      resp = await fetch(url);
    } catch (e) {
      if (window.location.protocol === "file:") {
        throw new Error(
          "Browser blocked local sidecar fetches under file://. " +
          "Serve outputs/reports/unified_viewer over HTTP and open that URL."
        );
      }
      throw e;
    }
    if (!resp.ok) throw new Error(`fetch ${url} → ${resp.status}`);
    const buf = await resp.arrayBuffer();
    if (typeof hyparquet === "undefined") {
      throw new Error("parquet reader not loaded (hyparquet missing)");
    }
    return await hyparquet.parquetReadObjects({
      file: buf, compressors: hyparquet.compressors,
    });
  }

  async function loadCluster(cluster) {
    if (!hasCluster(cluster)) return [];
    if (cache.has(cluster)) return cache.get(cluster);
    if (inflight.has(cluster)) return inflight.get(cluster);
    const m = _meta();
    const base = (m && m.relative_path) ? `${m.relative_path}/` : "audit_sources/omics_trace/";
    const ver = (m && m.omics_schema_version) || 0;
    const url = `${base}${_sanitize(cluster)}.parquet?v=${ver}`;
    const p = _fetchParquet(url).then(rows => {
      cache.set(cluster, rows);
      inflight.delete(cluster);
      return rows;
    }).catch(err => {
      inflight.delete(cluster);
      throw err;
    });
    inflight.set(cluster, p);
    return p;
  }

  // Map contrast string (e.g. "App_2mo") to the two group codes the JS uses to
  // tag arms. The omics_trace shard schema stores sex/timepoint/genotype as
  // separate columns (no joined `group` field) — males-only filtering is
  // applied at row-match time via `rowGroupKey()`.
  const _GENO_DECODE = { App: "AppP", Tau: "Ttau", ApTt: "ApTt" };
  function contrastToArms(contrast) {
    if (!contrast) return null;
    const parts = String(contrast).split("_");
    if (parts.length !== 2) return null;
    const [geno, age] = parts;
    const genoCode = _GENO_DECODE[geno];
    if (!genoCode) return null;
    return [
      { arm: geno, group: `ma_${age}_${genoCode}`, sex: "M", timepoint: age, genotype: genoCode },
      { arm: "WT",  group: `ma_${age}_WTyp`,        sex: "M", timepoint: age, genotype: "WTyp"  },
    ];
  }

  // Derive the synthetic `ma_<tp>_<geno>` group key from a raw shard row,
  // restricted to males (analysis_mode == males_only). Returns null for
  // female rows so they are silently dropped by the filters that consume this.
  function rowGroupKey(r) {
    if (!r || String(r.sex) !== "M") return null;
    return `ma_${String(r.timepoint)}_${String(r.genotype)}`;
  }

  // Return per-animal rows for (cluster, layer, gene_symbol, contrast).
  // Returns { arms: [{arm, group}], rows: [] } where rows carry all columns
  // plus a synthetic `_groupKey` field for downstream filtering.
  async function valuesForGene(cluster, layer, gene, contrast) {
    const arms = contrastToArms(contrast);
    if (!arms) return { arms: null, rows: [] };
    const allRows = await loadCluster(cluster);
    const groupSet = new Set(arms.map(a => a.group));
    const rows = [];
    for (const r of allRows) {
      if (String(r.layer) !== layer) continue;
      if (String(r.gene_symbol) !== String(gene)) continue;
      const gk = rowGroupKey(r);
      if (!gk || !groupSet.has(gk)) continue;
      rows.push(r);
    }
    return { arms, rows };
  }

  // Return per-site rows grouped by site_id. Only meaningful for phospho layers.
  // Returns Map<site_id_str, rows[]>.
  async function siteRowsForGene(cluster, layer, gene, contrast) {
    const { arms, rows } = await valuesForGene(cluster, layer, gene, contrast);
    const bySite = new Map();
    for (const r of rows) {
      const sid = r.site_id == null ? "__protein__" : String(r.site_id);
      if (!bySite.has(sid)) bySite.set(sid, []);
      bySite.get(sid).push(r);
    }
    return { arms, bySite };
  }

  return { isAvailable, hasCluster, loadCluster, contrastToArms, rowGroupKey,
           valuesForGene, siteRowsForGene, _sanitize };
})();
window.OmicsTraceStore = OmicsTraceStore;



// ---------------------------------------------------------------------------
// EvidencePanel — renders a 4-node × 4-layer evidence matrix for a single
// Incytr pathway row. Information-dense redesign (2026-05-21).
//
// Layout (CSS-grid table, 5 cols × 5 rows):
//   header row:  [—]   | Transcript | Protein | Phospho pS | Phospho pY
//   ligand row:  Ligand    | cell      | cell      | cell        | cell
//   receptor row:Receptor  | cell      | cell      | cell        | cell
//   em row:      EM        | cell      | cell      | cell        | cell
//   target row:  Target    | cell      | cell      | cell        | cell
//
// Cluster routing (per incytr/R/evaluation.R:227-230):
//   Ligand → sender; Receptor/EM/Target → receiver.
//
// Each cell renders a micro dot-bar (WT vs Disease) + an LFC chip showing
// only the Incytr-stored value, sign-coloured. The build-time 1e-4
// round-trip assertion in verify_pathway_round_trip.py is the load-bearing
// check that stored matches the limma-normalized recomputation; no per-cell
// recomputed comparator is displayed (that would just surface a redundant
// near-equal number next to the stored one).
//
// Phospho cells are gene-aggregated (mean across site_id per animal × group),
// matching Incytr's upstream `summarise_all(mean)` in incytr_commandline.R.
// Per-site detail is one click away in a popover.
//
// Missing-data states (three, visually distinct):
//   - no gene on node:        hatched empty cell ("—")
//   - gene unmeasured in layer: "n/a" italic
//   - measured, value zero:   dot-bar with bars at zero
// ---------------------------------------------------------------------------

const EvidencePanel = (() => {

  const _LAYERS = ["transcript", "protein", "phospho_ps", "phospho_py"];
  const _LAYER_LABELS = {
    transcript:  "Transcript",
    protein:     "Protein",
    phospho_ps:  "Phospho pS",
    phospho_py:  "Phospho pY",
  };
  // Tooltip unit hints surfaced in column headers.
  const _LAYER_UNITS = {
    transcript:  "log-norm CP10K mean",
    protein:     "log₂(IRS-norm + ε)",
    phospho_ps:  "log₂(IRS-norm + ε), gene-mean",
    phospho_py:  "log₂(IRS-norm + ε), gene-mean",
  };

  // Map layer → stored LFC column suffix in the receiver-cache row.
  const _LAYER_LFC_KEY = {
    transcript:  "sclog2FC",
    protein:     "pr_log2FC",
    phospho_ps:  "ps_log2FC",
    phospho_py:  "py_log2FC",
  };

  const _ARM_WT_COLOR      = "#777";
  const _ARM_DISEASE_COLOR = "#a3203c";

  function _esc(s) {
    return String(s == null ? "" : s)
      .replace(/&/g, "&amp;").replace(/</g, "&lt;")
      .replace(/>/g, "&gt;").replace(/"/g, "&quot;");
  }

  function _isNum(v) { return v != null && isFinite(v); }

  // ---------------------------------------------------------------------------
  // LFC chip — single Incytr-stored value, sign-coloured. No "recomputed"
  // comparator: stored ↔ recomputed agreement is enforced at build time by
  // verify_pathway_round_trip.py (strict 1e-4) and would otherwise just
  // appear as a redundant near-equal number next to the stored one. Drift
  // becomes a build failure, not a per-cell warning.
  // ---------------------------------------------------------------------------
  function _renderLfcChip(stored) {
    if (!_isNum(stored)) {
      return `<div class="ev-lfc-chip ev-lfc-empty"></div>`;
    }
    const sign = stored > 0 ? "pos" : (stored < 0 ? "neg" : "zero");
    const txt  = (stored >= 0 ? "+" : "") + stored.toFixed(2);
    return `<div class="ev-lfc-chip" title="Incytr stored log2 fold-change · disease vs WT">`
      + `<span class="ev-lfc-stored ${sign}">${_esc(txt)}</span>`
      + `</div>`;
  }

  // ---------------------------------------------------------------------------
  // Micro dot-bar plot — 130×44 SVG. Two bars (WT, Disease), per-animal dots
  // in a dedicated 8px band beneath the bars (never overlapping bar labels).
  //
  // perArm = [{arm: "WT"|<disease>, vals: number[]}, ...]
  // unitHint = string for axis-label tooltip
  // animalIdsByArm = Map<arm, string[]> — optional, for hover tooltip
  // ---------------------------------------------------------------------------
  function _renderMicroDotBar(perArm, unitHint, animalIdsByArm) {
    // Layout (top→bottom, never overlapping):
    //   y=4..10   arm label  (WT / ApTt etc.)
    //   y=12..18  mean value (numeric, sign-coloured)
    //   y=20..42  bar area (22px) — bars grow upward from baseline at y=42
    //   y=45..55  dot band (10px)
    const allVals = perArm.flatMap(a => a.vals);
    const hasAny  = allVals.length > 0;
    const allZero = hasAny && allVals.every(v => v === 0);

    const W = 130, H = 58;
    const padL = 4, padR = 4;
    const armLblY  = 8;
    const meanLblY = 17;
    const barTop   = 20;
    const barBot   = 42;
    const barAreaH = barBot - barTop;
    const dotBandY = 45;
    const dotBandH = 11;
    const nArms = perArm.length;
    const colW = (W - padL - padR) / nArms;

    if (!hasAny) {
      return `<svg class="ev-cell-plot" viewBox="0 0 ${W} ${H}" width="${W}" height="${H}">`
        + `<text x="${W/2}" y="${H/2 + 3}" text-anchor="middle" font-size="9" fill="#b0bec5" font-style="italic">no data</text>`
        + `</svg>`;
    }

    const vmax = allZero ? 1 : Math.max(...allVals.map(Math.abs));

    let parts = [];

    perArm.forEach((a, ai) => {
      const cx = padL + ai * colW + colW / 2;
      const isWT = a.arm === "WT";
      const color = isWT ? _ARM_WT_COLOR : _ARM_DISEASE_COLOR;
      const mean = a.vals.length
        ? a.vals.reduce((s, v) => s + v, 0) / a.vals.length
        : null;
      const animals = (animalIdsByArm && animalIdsByArm.get(a.arm)) || [];

      // Arm label (top, never under anything else)
      parts.push(
        `<text x="${cx.toFixed(1)}" y="${armLblY}" `
        + `text-anchor="middle" font-size="8" fill="#37474f" font-weight="600">${_esc(a.arm)}</text>`
      );

      // Bar + mean number
      if (mean != null) {
        const barH = allZero ? 1.5 : Math.max(1.5, (Math.abs(mean) / vmax) * barAreaH);
        const barY = barBot - barH;
        const barW = Math.min(colW * 0.42, 26);
        const barX = cx - barW / 2;
        const animalsTxt = animals.length
          ? a.vals.map((v, i) => `${animals[i] || `n${i+1}`}=${v.toFixed(3)}`).join(", ")
          : a.vals.map(v => v.toFixed(3)).join(", ");
        const tip = `${a.arm} · n=${a.vals.length} · mean=${mean.toFixed(3)} · [${animalsTxt}]`;
        parts.push(
          `<rect x="${barX.toFixed(1)}" y="${barY.toFixed(1)}" `
          + `width="${barW.toFixed(1)}" height="${barH.toFixed(1)}" `
          + `fill="${color}" opacity="0.75" rx="1"><title>${_esc(tip)}</title></rect>`
        );
        // Mean number above bar area, never under bars or dots
        parts.push(
          `<text x="${cx.toFixed(1)}" y="${meanLblY}" `
          + `text-anchor="middle" font-size="8" fill="${color}" font-weight="600">${mean.toFixed(2)}</text>`
        );
      } else {
        parts.push(
          `<text x="${cx.toFixed(1)}" y="${meanLblY}" `
          + `text-anchor="middle" font-size="8" fill="#cfd8dc" font-style="italic">—</text>`
        );
      }

      // Dots (deterministic horizontal spread; vertical position by relative value)
      const n = a.vals.length;
      a.vals.forEach((v, di) => {
        const spreadW = Math.min(colW * 0.55, 32);
        const dx = n <= 1 ? 0 : (di / (n - 1) - 0.5) * spreadW;
        const dotX = cx + dx;
        const norm = vmax === 0 ? 0.5 : Math.abs(v) / vmax;
        const dy = (1 - norm) * (dotBandH - 4) + 2;
        const dotY = dotBandY + dy;
        const animalId = animals[di] || `n${di+1}`;
        const tip = `${a.arm} · ${animalId} · ${v.toFixed(3)}`;
        parts.push(
          `<circle cx="${dotX.toFixed(1)}" cy="${dotY.toFixed(1)}" `
          + `r="1.8" fill="${color}" opacity="0.9">`
          + `<title>${_esc(tip)}</title></circle>`
        );
      });
    });

    // Bar baseline (zero line, at bottom of bar area)
    parts.push(
      `<line x1="${padL}" y1="${barBot}" x2="${W - padR}" y2="${barBot}" `
      + `stroke="#cfd8dc" stroke-width="0.5"/>`
    );

    return `<svg class="ev-cell-plot" viewBox="0 0 ${W} ${H}" width="${W}" height="${H}">`
      + `<title>${_esc(unitHint || "")}</title>`
      + parts.join("")
      + `</svg>`;
  }

  // ---------------------------------------------------------------------------
  // Aggregate phospho rows from site-level → gene-level per animal × group.
  // Mirrors the upstream Incytr aggregation (incytr_commandline.R:
  // `summarise_all(mean)` over gene_symbol). Returns Map<group, perAnimalVals>.
  // ---------------------------------------------------------------------------
  function _aggregatePhosphoToGene(layerRows, arms) {
    // Group by (group, animal_id) → mean across sites
    const perGroup = new Map();   // group -> Map<animal_id, [values]>
    const armGroups = new Set((arms || []).map(a => a.group));
    for (const r of (layerRows || [])) {
      const g = OmicsTraceStore.rowGroupKey(r);
      if (!g || !armGroups.has(g)) continue;
      const a = String(r.animal_id);
      const v = (r.value == null || !isFinite(r.value)) ? null : Number(r.value);
      if (v == null) continue;
      if (!perGroup.has(g)) perGroup.set(g, new Map());
      const am = perGroup.get(g);
      if (!am.has(a)) am.set(a, []);
      am.get(a).push(v);
    }
    // Reduce: per (group, animal) → mean across sites
    const armOut = (arms || []).map(arm => {
      const am = perGroup.get(arm.group) || new Map();
      const animalIds = [...am.keys()].sort();
      const vals = animalIds.map(aid => {
        const xs = am.get(aid);
        return xs.reduce((s, v) => s + v, 0) / xs.length;
      });
      return { arm: arm.arm, group: arm.group, vals, animalIds };
    });
    return armOut;
  }

  // Build perArm from omics-trace rows (animal-level, no site aggregation).
  function _perArmFromRows(layerRows, arms) {
    const perGroup = new Map();
    const armGroups = new Set((arms || []).map(a => a.group));
    for (const r of (layerRows || [])) {
      const g = OmicsTraceStore.rowGroupKey(r);
      if (!g || !armGroups.has(g)) continue;
      const v = (r.value == null || !isFinite(r.value)) ? null : Number(r.value);
      if (v == null) continue;
      if (!perGroup.has(g)) perGroup.set(g, []);
      perGroup.get(g).push({ animal_id: String(r.animal_id), value: v });
    }
    return (arms || []).map(arm => {
      const entries = (perGroup.get(arm.group) || []).slice()
        .sort((a, b) => a.animal_id.localeCompare(b.animal_id));
      return {
        arm: arm.arm,
        group: arm.group,
        vals: entries.map(e => e.value),
        animalIds: entries.map(e => e.animal_id),
      };
    });
  }

  function _animalIdsByArm(perArm) {
    const m = new Map();
    for (const a of perArm) m.set(a.arm, a.animalIds || []);
    return m;
  }

  // ---------------------------------------------------------------------------
  // Build cell HTML for one (node, layer) slot.
  //
  // Returns a string of inner-HTML for an `.ev-cell` div, or a special class
  // that the caller wraps the cell with (returned as { html, klass }).
  // ---------------------------------------------------------------------------
  function _buildCell(opts) {
    const { node, layer, gene, cluster, contrast,
            ttRows, omicsShard, storedLfcRow,
            cellId } = opts;

    // State: no gene on this node
    if (!gene) {
      return { klass: "ev-cell ev-cell-empty-nogene",
               html: `<span>— no gene —</span>` };
    }
    if (!cluster) {
      return { klass: "ev-cell ev-cell-na",
               html: `<span>no cluster</span>` };
    }

    const arms = OmicsTraceStore.contrastToArms(contrast);
    if (!arms) {
      return { klass: "ev-cell ev-cell-na",
               html: `<span>no contrast</span>` };
    }

    const storedKey = `${node}_${_LAYER_LFC_KEY[layer]}`;
    const stored = (storedLfcRow && storedLfcRow[storedKey] != null
                    && isFinite(storedLfcRow[storedKey]))
      ? Number(storedLfcRow[storedKey]) : null;

    // --------- Transcript layer ---------
    if (layer === "transcript") {
      const ttArms = TranscriptTraceStore && TranscriptTraceStore.isAvailable()
        && TranscriptTraceStore.hasCluster(cluster)
        ? TranscriptTraceStore.contrastToArms(contrast) : null;
      if (!ttArms) {
        return { klass: "ev-cell ev-cell-na",
                 html: `<span>n/a</span><span>${_renderLfcChip(stored)}</span>` };
      }
      // Transcript shard has one row per (gene, group) — pseudobulk mean.
      // Build single-value "perArm" for visualization.
      const perArm = ttArms.map(a => {
        const row = (ttRows || []).find(r =>
          String(r.gene) === String(gene) && String(r.group) === a.group);
        const v = (row && _isNum(row.value)) ? Number(row.value) : null;
        return { arm: a.arm, group: a.group,
                 vals: v == null ? [] : [v],
                 animalIds: ["pseudobulk"] };
      });
      const hasAny = perArm.some(a => a.vals.length);
      if (!hasAny) {
        return { klass: "ev-cell ev-cell-na",
                 html: `<span>n/a</span><span>${_renderLfcChip(stored)}</span>` };
      }
      return { klass: "ev-cell",
               html: _renderMicroDotBar(perArm, _LAYER_UNITS[layer], _animalIdsByArm(perArm))
                 + _renderLfcChip(stored) };
    }

    // --------- Omics layers: protein / phospho_ps / phospho_py ---------
    if (!OmicsTraceStore.isAvailable() || !OmicsTraceStore.hasCluster(cluster)) {
      return { klass: "ev-cell ev-cell-na",
               html: `<span>n/a</span><span>${_renderLfcChip(stored)}</span>` };
    }

    const layerRows = (omicsShard || []).filter(r =>
      String(r.layer) === layer && String(r.gene_symbol) === String(gene));

    if (layerRows.length === 0) {
      return { klass: "ev-cell ev-cell-na",
               html: `<span>n/a</span><span>${_renderLfcChip(stored)}</span>` };
    }

    let perArm;
    let footerHtml = "";
    if (layer === "protein") {
      perArm = _perArmFromRows(layerRows, arms);
    } else {
      // Phospho: aggregate sites → gene per animal × group (matches Incytr).
      perArm = _aggregatePhosphoToGene(layerRows, arms);
      // Only count sites that contribute *signal* (≥1 positive value) in this
      // contrast. Sites where the forward projection collapses to zero across
      // every animal (e.g. parent gene not expressed in this cluster) carry no
      // information and would clutter the popover.
      const armGroups = new Set(arms.map(a => a.group));
      const sitesWithSignal = new Set();
      for (const r of layerRows) {
        const gk = OmicsTraceStore.rowGroupKey(r);
        if (!gk || !armGroups.has(gk)) continue;
        if (r.value == null || !isFinite(r.value) || r.value <= 0) continue;
        sitesWithSignal.add(r.site_id == null ? "__protein__" : String(r.site_id));
      }
      if (sitesWithSignal.size === 0) {
        return { klass: "ev-cell ev-cell-na",
                 html: `<span>n/a</span><span>${_renderLfcChip(stored)}</span>` };
      }
      footerHtml = `<button type="button" class="ev-phospho-expand" `
        + `data-ev-popover="${_esc(cellId)}" `
        + `title="View per-site detail (Incytr aggregates to gene-mean before LFC)">`
        + `▾ ${sitesWithSignal.size} site${sitesWithSignal.size === 1 ? "" : "s"}</button>`;
    }

    // If after filtering to males-only and the requested contrast no per-arm
    // values remain, route to the same "n/a" state used when the shard has
    // zero rows for this gene — keeps "no underlying data" visually uniform.
    const anyArmVals = perArm.some(a => a.vals && a.vals.length > 0);
    if (!anyArmVals) {
      return { klass: "ev-cell ev-cell-na",
               html: `<span>n/a</span><span>${_renderLfcChip(stored)}</span>` };
    }
    const plotHtml = _renderMicroDotBar(perArm, _LAYER_UNITS[layer], _animalIdsByArm(perArm));
    return { klass: "ev-cell",
             html: `<div>${plotHtml}${footerHtml}</div>`
                 + _renderLfcChip(stored) };
  }

  // ---------------------------------------------------------------------------
  // Popover (per-site phospho detail). Attached to body to avoid clipping.
  // ---------------------------------------------------------------------------
  let _activePopover = null;
  function _closePopover() {
    if (_activePopover && _activePopover.parentNode) {
      _activePopover.parentNode.removeChild(_activePopover);
    }
    _activePopover = null;
    document.removeEventListener("click", _onDocClick, true);
  }
  function _onDocClick(e) {
    if (!_activePopover) return;
    if (_activePopover.contains(e.target)) return;
    if (e.target.closest && e.target.closest(".ev-phospho-expand")) return;
    _closePopover();
  }
  function _openPopover(anchorEl, contentHtml) {
    _closePopover();
    const pop = document.createElement("div");
    pop.className = "ev-phospho-popover";
    pop.innerHTML = contentHtml;
    document.body.appendChild(pop);
    const rect = anchorEl.getBoundingClientRect();
    const left = Math.min(window.innerWidth - 320,
                          rect.left + window.scrollX);
    const top  = rect.bottom + window.scrollY + 4;
    pop.style.left = `${left}px`;
    pop.style.top  = `${top}px`;
    _activePopover = pop;
    setTimeout(() => document.addEventListener("click", _onDocClick, true), 0);
  }

  function _formatMotif(motif) {
    // Phospho motif is a 13-mer with the modified residue at position 7 (index 6).
    // Render with that residue bolded so the reader can see which S/T/Y was measured.
    if (!motif || typeof motif !== "string" || motif.length < 7) return null;
    const left = motif.slice(0, 6);
    const center = motif.slice(6, 7);
    const right = motif.slice(7);
    return `<span class="ev-pop-motif-flank">${_esc(left)}</span>`
         + `<span class="ev-pop-motif-center">${_esc(center)}</span>`
         + `<span class="ev-pop-motif-flank">${_esc(right)}</span>`;
  }

  function _buildSitePopoverHtml(layer, gene, layerRows, arms) {
    const bySite = new Map();
    const motifBySite = new Map();
    for (const r of layerRows) {
      const sid = r.site_id == null ? "__protein__" : String(r.site_id);
      if (!bySite.has(sid)) bySite.set(sid, []);
      bySite.get(sid).push(r);
      if (!motifBySite.has(sid) && r.motif) motifBySite.set(sid, String(r.motif));
    }
    let rowsHtml = "";
    let kept = 0, skipped = 0;
    for (const [sid, siteRows] of bySite) {
      const perArm = _perArmFromRows(siteRows, arms);
      const hasSignal = perArm.some(a => a.vals && a.vals.some(v => v > 0));
      if (!hasSignal) { skipped += 1; continue; }
      kept += 1;
      const plot = _renderMicroDotBar(perArm, _LAYER_UNITS[layer], _animalIdsByArm(perArm));
      const motif = motifBySite.get(sid);
      const motifHtml = _formatMotif(motif);
      const label = motifHtml
        ? `<span class="ev-pop-site-id" title="motif: ${_esc(motif)} · site_id ${_esc(sid)}">${motifHtml}</span>`
        : `<span class="ev-pop-site-id" title="site_id ${_esc(sid)} (no motif)">${_esc(sid)}</span>`;
      rowsHtml += `<div class="ev-pop-site">`
        + label
        + `<span>${plot}</span>`
        + `</div>`;
    }
    const totalSites = bySite.size;
    const subhead = kept === 0
      ? `<div class="ev-pop-empty">No sites with signal in this contrast for ${_esc(gene)} (${totalSites} site${totalSites === 1 ? "" : "s"} in shard, all zero or missing).</div>`
      : (skipped > 0
          ? `<div class="ev-pop-note">${kept} of ${totalSites} sites contribute signal (${skipped} zero or missing).</div>`
          : `<div class="ev-pop-note">${kept} site${kept === 1 ? "" : "s"} with signal.</div>`);
    return `<div class="ev-pop-head">${_esc(_LAYER_LABELS[layer])} · ${_esc(gene)} — per-site detail</div>`
      + subhead
      + rowsHtml;
  }

  // ---------------------------------------------------------------------------
  // Public entry point.
  // ---------------------------------------------------------------------------
  async function render(host, r, rk) {
    if (!host) return;

    const contrast = r.contrast || "";
    const sender   = r._sender   || "";
    const receiver = r._receiver || "";

    const nodes = [
      { node: "Ligand",   gene: r.Ligand,   cluster: sender   },
      { node: "Receptor", gene: r.Receptor, cluster: receiver },
      { node: "EM",       gene: r.EM,       cluster: receiver },
      { node: "Target",   gene: r.Target,   cluster: receiver },
    ];

    const arms = OmicsTraceStore.contrastToArms(contrast);
    const tp   = contrast.split("_")[1] || "";
    const armsLabel = arms
      ? `${arms[0].arm} vs ${arms[1].arm} @ ${tp}`
      : contrast;

    const safeRk = rk.replace(/[^a-zA-Z0-9]/g, "_");
    const gridId = `ev-grid-${safeRk}`;

    // Header (one-line meta).
    const headerHtml =
      `<div class="ev-note">`
      + `<strong>${_esc(armsLabel)}</strong>`
      + `<span class="ev-meta-sep">·</span>sender ${_esc(sender || "—")}`
      + `<span class="ev-meta-sep">·</span>receiver ${_esc(receiver || "—")}`
      + `<span class="ev-meta-sep">·</span>males-only · n=3 vs n=3`
      + `</div>`;

    // Skeleton: matrix with loading cells.
    const headerRowHtml = `<div class="ev-matrix-corner"></div>`
      + _LAYERS.map(l =>
          `<div class="ev-matrix-head" title="${_esc(_LAYER_UNITS[l])}">`
          + `${_esc(_LAYER_LABELS[l])}<span class="ev-unit">${_esc(_LAYER_UNITS[l])}</span>`
          + `</div>`).join("");

    const bodyHtml = nodes.map(nd => {
      const rh = `<div class="ev-matrix-rowhead">`
        + `<span class="ev-rh-node">${_esc(nd.node)}</span>`
        + `<span class="ev-rh-gene">${_esc(nd.gene || "—")}</span>`
        + `<span class="ev-rh-cluster" title="${_esc(nd.cluster || "")}">${_esc(nd.cluster || "—")}</span>`
        + `</div>`;
      const cells = _LAYERS.map(layer => {
        const cellId = `${gridId}-${nd.node}-${layer}`;
        return `<div class="ev-cell ev-cell-loading" id="${_esc(cellId)}">loading…</div>`;
      }).join("");
      return rh + cells;
    }).join("");

    host.innerHTML = headerHtml
      + `<div class="ev-matrix" id="${_esc(gridId)}">`
      + headerRowHtml + bodyHtml
      + `</div>`;

    // Load shards (only unique clusters).
    const clusterSet = new Set(nodes.map(nd => nd.cluster).filter(Boolean));
    const omicsByCluster = new Map();
    const ttByCluster    = new Map();

    await Promise.all([...clusterSet].map(async cl => {
      const tasks = [];
      if (OmicsTraceStore.isAvailable() && OmicsTraceStore.hasCluster(cl)) {
        tasks.push(OmicsTraceStore.loadCluster(cl).then(rs => omicsByCluster.set(cl, rs))
          .catch(() => omicsByCluster.set(cl, [])));
      } else { omicsByCluster.set(cl, []); }
      if (typeof TranscriptTraceStore !== "undefined"
          && TranscriptTraceStore.isAvailable()
          && TranscriptTraceStore.hasCluster(cl)) {
        tasks.push(TranscriptTraceStore.loadCluster(cl).then(rs => ttByCluster.set(cl, rs))
          .catch(() => ttByCluster.set(cl, [])));
      } else { ttByCluster.set(cl, []); }
      await Promise.all(tasks);
    }));

    const gridEl = document.getElementById(gridId);
    if (!gridEl) return; // panel was replaced before load completed

    // Fill cells.
    for (const nd of nodes) {
      for (const layer of _LAYERS) {
        const cellId = `${gridId}-${nd.node}-${layer}`;
        const el = document.getElementById(cellId);
        if (!el) continue;
        const built = _buildCell({
          node: nd.node, layer, gene: nd.gene, cluster: nd.cluster, contrast,
          ttRows: ttByCluster.get(nd.cluster) || [],
          omicsShard: omicsByCluster.get(nd.cluster) || [],
          storedLfcRow: r,
          cellId,
        });
        el.className = built.klass;
        el.innerHTML = built.html;
      }
    }

    // Wire up phospho-site popovers.
    const popoverButtons = gridEl.querySelectorAll(".ev-phospho-expand");
    popoverButtons.forEach(btn => {
      btn.addEventListener("click", (e) => {
        e.stopPropagation();
        const cellId = btn.getAttribute("data-ev-popover");
        // cellId encodes node + layer; recover layer + gene from nodes array.
        const suffix = cellId.replace(`${gridId}-`, "");
        const [nodeName, ...layerParts] = suffix.split("-");
        const layer = layerParts.join("-");
        const nd = nodes.find(n => n.node === nodeName);
        if (!nd) return;
        const layerRows = (omicsByCluster.get(nd.cluster) || [])
          .filter(rr => String(rr.layer) === layer
                     && String(rr.gene_symbol) === String(nd.gene));
        if (layerRows.length === 0) return;
        const armsLocal = OmicsTraceStore.contrastToArms(contrast);
        const html = _buildSitePopoverHtml(layer, nd.gene, layerRows, armsLocal);
        _openPopover(btn, html);
      });
    });
  }

  return { render };
})();
window.EvidencePanel = EvidencePanel;

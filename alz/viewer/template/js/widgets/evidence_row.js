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
// ---------------------------------------------------------------------------
// NormalizedSubstrateStore — per-cluster limma-normalized condition means
// backing the right-edge LFC recomputation in the Evidence tab.
//
// Shard layout: audit_sources/omics_trace_normalized/<slug>.parquet
//
// Schema per shard row (from build_normalized_substrate.py, schema_version=1):
//   layer               : "protein" | "phospho_ps" | "phospho_py"
//   gene_symbol         : string
//   contrast            : string  (e.g. "ma_2mo_ApTt_ma_2mo_WTyp")
//   group               : string  (e.g. "ma_2mo_ApTt" | "ma_2mo_WTyp")
//   mean_value_normalized : float
//
// Epsilon for LFC: PAYLOAD.meta.omics_trace_normalized.epsilon (= 0.001).
// log2((D_norm + ε) / (W_norm + ε)) agrees with stored *_pr/_ps/_py_log2FC
// to ≤1e-4 (validated by build_normalized_substrate.py round-trip check).
//
// Transcript LFC uses ε = 1e-5 (Cal_scFC default, analysis.R:248).
// ---------------------------------------------------------------------------

const OmicsTraceStore = (() => {
  const cache = new Map();        // cluster -> rows[]
  const inflight = new Map();     // cluster -> Promise<rows>

  // Mirror of alz/integration/pair_to_receiver_cache.py::_sanitize_celltype.
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
    const url = `${base}${_sanitize(cluster)}.parquet`;
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

  // Map contrast string (e.g. "App_2mo") to the two group codes present in the
  // shard. Mirrors TranscriptTraceStore.contrastToArms convention.
  // Returns [{arm, group}, {arm, group}] or null if unrecognised.
  const _GENO_DECODE = { App: "AppP", Tau: "Ttau", ApTt: "ApTt" };
  function contrastToArms(contrast) {
    if (!contrast) return null;
    const parts = String(contrast).split("_");
    if (parts.length !== 2) return null;
    const [geno, age] = parts;
    const genoCode = _GENO_DECODE[geno];
    if (!genoCode) return null;
    return [
      { arm: geno, group: `ma_${age}_${genoCode}` },
      { arm: "WT",  group: `ma_${age}_WTyp` },
    ];
  }

  // Return per-animal rows for (cluster, layer, gene_symbol, contrast).
  // Returns { arms: [{arm, group}], rows: [] } where rows carry all columns.
  // If gene has no rows in this layer, returns rows=[].
  async function valuesForGene(cluster, layer, gene, contrast) {
    const arms = contrastToArms(contrast);
    if (!arms) return { arms: null, rows: [] };
    const allRows = await loadCluster(cluster);
    const groupSet = new Set(arms.map(a => a.group));
    const rows = allRows.filter(r =>
      String(r.layer) === layer
      && String(r.gene_symbol) === String(gene)
      && groupSet.has(String(r.group))
    );
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

  return { isAvailable, hasCluster, loadCluster, contrastToArms,
           valuesForGene, siteRowsForGene, _sanitize };
})();
window.OmicsTraceStore = OmicsTraceStore;


// ---------------------------------------------------------------------------
// NormalizedSubstrateStore — per-cluster limma-normalized condition means.
// Used only for right-edge LFC computation; not for dot/bar display.
// ---------------------------------------------------------------------------
const NormalizedSubstrateStore = (() => {
  const cache = new Map();     // cluster -> rows[]
  const inflight = new Map();  // cluster -> Promise<rows>

  function _sanitize(name) {
    return String(name).replaceAll("/", "-").replaceAll(" ", "_");
  }

  function _meta() {
    return (typeof PAYLOAD !== "undefined"
            && PAYLOAD.meta
            && PAYLOAD.meta.omics_trace_normalized) || null;
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

  // Epsilon from PAYLOAD meta, defaulting to 1e-3 (Item 3.2b validated value).
  function epsilon() {
    const m = _meta();
    return (m && typeof m.epsilon === "number") ? m.epsilon : 0.001;
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
    const base = (m && m.relative_path) ? `${m.relative_path}/` : "audit_sources/omics_trace_normalized/";
    const url = `${base}${_sanitize(cluster)}.parquet`;
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

  // Convert viewer contrast string ("ApTt_2mo") to the normalized-shard contrast
  // key ("ma_2mo_ApTt_ma_2mo_WTyp").
  // Returns null if contrast is unrecognised.
  const _GENO_DECODE = { App: "AppP", Tau: "Ttau", ApTt: "ApTt" };
  function contrastToNormKey(contrast) {
    if (!contrast) return null;
    const parts = String(contrast).split("_");
    if (parts.length !== 2) return null;
    const [geno, age] = parts;
    const genoCode = _GENO_DECODE[geno];
    if (!genoCode) return null;
    return `ma_${age}_${genoCode}_ma_${age}_WTyp`;
  }

  // Compute LFC from normalized shard rows for (layer, gene, contrast).
  // Returns { lfc: number | null, diseaseVal: number | null, wtVal: number | null }.
  function computeLfc(normRows, layer, gene, contrastKey) {
    if (!normRows || !contrastKey) return { lfc: null, diseaseVal: null, wtVal: null };
    const eps = epsilon();
    const layerGeneRows = normRows.filter(r =>
      String(r.layer) === layer
      && String(r.gene_symbol) === String(gene)
      && String(r.contrast) === contrastKey
    );
    // Disease group: the one that doesn't end with _WTyp.
    let D = null, W = null;
    for (const row of layerGeneRows) {
      const g = String(row.group);
      const v = row.mean_value_normalized;
      if (v == null || !isFinite(v)) continue;
      if (g.endsWith("_WTyp")) {
        W = Number(v);
      } else {
        D = Number(v);
      }
    }
    if (D === null || W === null) return { lfc: null, diseaseVal: D, wtVal: W };
    const lfc = Math.log2((D + eps) / (W + eps));
    return { lfc, diseaseVal: D, wtVal: W };
  }

  return { isAvailable, hasCluster, loadCluster, epsilon,
           contrastToNormKey, computeLfc, _sanitize };
})();
window.NormalizedSubstrateStore = NormalizedSubstrateStore;


// ---------------------------------------------------------------------------
// EvidencePanel — renders a 4-node × 4-layer evidence grid for a single
// Incytr pathway row. Replaces the old "Fold Change" and "Measurement Trace"
// tabs from Phase 3 Item 3.3.
//
// Layout:
//   Column: Ligand | Receptor | EM | Target
//   Row: transcript | protein | phospho_ps | phospho_py
//
// Cluster routing (per incytr/R/evaluation.R:227-230):
//   Ligand  → sender cluster
//   Receptor, EM, Target → receiver cluster
//
// Each sub-cell shows:
//   - Per-animal dot strip (one dot per animal, coloured by arm)
//   - Per-group mean bar
//   - Right-edge LFC placeholder (filled by Item 3.4)
//
// For phospho layers, one sub-row per site_id. If a gene has zero sites in a
// layer, renders a single "n/a" row.
//
// LFC computation is deferred to Item 3.4. This item renders raw dots + means
// only, leaving an empty/placeholder LFC slot for 3.4 to fill.
// ---------------------------------------------------------------------------

const EvidencePanel = (() => {

  // Human-readable labels for layers.
  const _LAYER_LABELS = {
    transcript:  "Transcript",
    protein:     "Protein",
    phospho_ps:  "Phospho (pS/pT)",
    phospho_py:  "Phospho (pY)",
  };

  // Map layer → stored LFC column suffix in the receiver-cache row.
  // e.g. "protein" → "pr_log2FC" so node="Ligand" → "Ligand_pr_log2FC".
  const _LAYER_LFC_KEY = {
    transcript:  "sclog2FC",
    protein:     "pr_log2FC",
    phospho_ps:  "ps_log2FC",
    phospho_py:  "py_log2FC",
  };

  // Epsilon for transcript LFC (Cal_scFC default, analysis.R:248).
  const _TRANSCRIPT_EPS = 1e-5;

  // ---------------------------------------------------------------------------
  // Compute transcript LFC from ttRows (pseudobulk per-(gene, group) means).
  // Returns null if gene not found in both arms.
  // ---------------------------------------------------------------------------
  function _computeTranscriptLfc(ttRows, gene, arms) {
    if (!ttRows || !arms || arms.length < 2) return null;
    const [diseaseArm, wtArm] = arms; // arms[0]=disease, arms[1]=WT
    let D = null, W = null;
    for (const row of ttRows) {
      if (String(row.gene) !== String(gene)) continue;
      const g = String(row.group);
      if (g === diseaseArm.group) D = (row.value != null && isFinite(row.value)) ? Number(row.value) : null;
      if (g === wtArm.group)      W = (row.value != null && isFinite(row.value)) ? Number(row.value) : null;
    }
    if (D === null || W === null) return null;
    return Math.log2((D + _TRANSCRIPT_EPS) / (W + _TRANSCRIPT_EPS));
  }

  // ---------------------------------------------------------------------------
  // Render the right-edge LFC slot content.
  // recomputed : number | null
  // stored     : number | null (from receiver-cache shard column)
  // Returns an HTML string to replace the ev-lfc-slot placeholder.
  // ---------------------------------------------------------------------------
  function _renderLfcSlot(recomputed, stored) {
    if (recomputed === null) {
      return `<div class="ev-lfc-slot ev-lfc-na" title="LFC not available (gene absent from substrate)">LFC —</div>`;
    }
    const recomputedStr = recomputed.toFixed(3);
    if (stored === null || stored === undefined || !isFinite(stored)) {
      // No stored value to compare (gene absent from Incytr output).
      return `<div class="ev-lfc-slot ev-lfc-value" title="Recomputed LFC (no stored value to compare)">LFC ${recomputedStr}</div>`;
    }
    const diff = Math.abs(recomputed - stored);
    const storedStr = stored.toFixed(3);
    if (diff <= 1e-4) {
      return `<div class="ev-lfc-slot ev-lfc-ok" title="Recomputed: ${recomputedStr} | Stored: ${storedStr} | Δ: ${diff.toExponential(1)}">`
        + `LFC ${recomputedStr} <span class="ev-lfc-check" style="color:#2a7a2a;font-size:9px;">✓</span></div>`;
    }
    return `<div class="ev-lfc-slot ev-lfc-fail" style="color:#cc0000;font-weight:bold;" `
      + `title="LFC mismatch — recomputed: ${recomputedStr} stored: ${storedStr} Δ: ${diff.toExponential(2)}">`
      + `FAIL stored=${storedStr} recomputed=${recomputedStr} Δ=${diff.toExponential(2)}</div>`;
  }

  // ---------------------------------------------------------------------------
  // Fill all ev-lfc-slot placeholders inside a container element.
  // Finds [data-lfc-placeholder] divs and replaces their innerHTML.
  // ---------------------------------------------------------------------------
  function _fillLfcSlot(containerEl, recomputed, stored) {
    const slots = containerEl.querySelectorAll("[data-lfc-placeholder]");
    const html = _renderLfcSlot(recomputed, stored);
    for (const slot of slots) {
      slot.outerHTML = html;
    }
  }

  // Arm colours: disease arm red, WT arm grey.
  const _ARM_COLOR = {
    disease: "#a3203c",
    WT:      "#777777",
  };

  function _armColor(arm) {
    return arm === "WT" ? _ARM_COLOR.WT : _ARM_COLOR.disease;
  }

  function _esc(s) {
    return String(s == null ? "" : s)
      .replace(/&/g, "&amp;").replace(/</g, "&lt;")
      .replace(/>/g, "&gt;").replace(/"/g, "&quot;");
  }

  // ---------------------------------------------------------------------------
  // SVG strip-plot + mean bars for one sub-cell (one layer, one gene, one site).
  //
  // arms    = [{arm, group}, {arm, group}] from contrastToArms
  // armRows = Map<group_string, rows[]>   (rows from shard for this site)
  // label   = display string for the sub-cell header (e.g. gene or site_id)
  // isLog   = whether to use log2_value (true) or raw value (false) for dots
  // ---------------------------------------------------------------------------
  function _renderDotBarSvg(arms, armRows, label, isLog) {
    if (!arms) {
      return `<div class="ev-cell-label muted">${_esc(label)}</div>`
           + `<div class="ev-cell-empty muted" style="font-size:10px;">no contrast</div>`;
    }

    // Collect values per arm.
    const perArm = arms.map(a => {
      const rows = armRows.get(a.group) || [];
      const vals = rows.map(r => {
        const v = isLog ? r.log2_value : r.value;
        return (v == null || !isFinite(v)) ? null : Number(v);
      }).filter(v => v != null);
      return { arm: a.arm, group: a.group, vals };
    });

    const allVals = perArm.flatMap(a => a.vals);
    const hasAny = allVals.length > 0;

    if (!hasAny) {
      return `<div class="ev-cell-label">${_esc(label)}</div>`
           + `<div class="ev-cell-empty muted" style="font-size:10px;">no data</div>`
           + `<div class="ev-lfc-slot" data-lfc-placeholder="1" style="font-size:10px;color:#999;">LFC —</div>`;
    }

    const vmax = Math.max(...allVals.map(Math.abs), 0.001);
    const yScale = vmax * 1.2;

    // SVG dimensions.
    const W = 130, H = 88;
    const padTop = 14, padBot = 22, padL = 6, padR = 6;
    const barAreaH = H - padTop - padBot;
    const nArms = perArm.length;
    const colW = (W - padL - padR) / nArms;
    const dotR = 3.5;

    let svgParts = [];

    perArm.forEach((a, ai) => {
      const cx = padL + ai * colW + colW / 2;
      const color = _armColor(a.arm);

      // Mean bar.
      const mean = a.vals.length > 0
        ? a.vals.reduce((s, v) => s + v, 0) / a.vals.length
        : null;

      if (mean != null) {
        const barH = Math.max(2, (Math.abs(mean) / yScale) * barAreaH);
        const barY = padTop + barAreaH - barH;
        const barW = colW * 0.45;
        const barX = cx - barW / 2;
        svgParts.push(
          `<rect x="${barX.toFixed(1)}" y="${barY.toFixed(1)}" `
          + `width="${barW.toFixed(1)}" height="${barH.toFixed(1)}" `
          + `fill="${color}" opacity="0.25" rx="1"/>`
        );
        // Mean value label above bar.
        svgParts.push(
          `<text x="${cx.toFixed(1)}" y="${(barY - 2).toFixed(1)}" `
          + `text-anchor="middle" font-size="8" fill="${color}">${mean.toFixed(2)}</text>`
        );
      }

      // Per-animal dots (jittered horizontally within column).
      const n = a.vals.length;
      a.vals.forEach((v, di) => {
        const jitter = n <= 1 ? 0 : (di / (n - 1) - 0.5) * colW * 0.5;
        const dotX = cx + jitter;
        const dotY = padTop + barAreaH - (v / yScale) * barAreaH;
        svgParts.push(
          `<circle cx="${dotX.toFixed(1)}" cy="${dotY.toFixed(1)}" `
          + `r="${dotR}" fill="${color}" opacity="0.75" `
          + `title="${_esc(a.arm)}: ${v.toFixed(3)}"/>`
        );
      });

      // X-axis label.
      svgParts.push(
        `<text x="${cx.toFixed(1)}" y="${(H - 6).toFixed(1)}" `
        + `text-anchor="middle" font-size="9" fill="#444">${_esc(a.arm)}</text>`
      );
    });

    // Zero line.
    const zeroY = padTop + barAreaH;
    svgParts.push(
      `<line x1="${padL}" y1="${zeroY.toFixed(1)}" `
      + `x2="${(W - padR).toFixed(1)}" y2="${zeroY.toFixed(1)}" `
      + `stroke="#ccc" stroke-width="0.5"/>`
    );

    const valueLabel = isLog ? "log₂" : "raw";
    return `<div class="ev-cell-label" title="${_esc(label)}">${_esc(label)}</div>`
         + `<svg class="ev-dot-svg" viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" `
         + `title="${_esc(valueLabel)} · dots=animals, bars=mean">${svgParts.join("")}</svg>`
         + `<div class="ev-lfc-slot" data-lfc-placeholder="1" style="font-size:10px;color:#999;">LFC —</div>`;
  }

  // Build the arms-keyed row lookup for transcript data.
  // TranscriptTraceStore rows have {gene, group, value} (one row per group).
  function _transcriptArmRows(ttRows, gene, arms) {
    // Build Map<group, rows[]> matching the dot-bar renderer's expectation.
    // Transcript has exactly one value per group (pseudobulk mean), so each
    // group maps to a single row.
    const m = new Map();
    for (const a of (arms || [])) m.set(a.group, []);
    for (const row of (ttRows || [])) {
      if (String(row.gene) !== String(gene)) continue;
      const g = String(row.group);
      if (m.has(g)) m.get(g).push(row);
    }
    return m;
  }

  // ---------------------------------------------------------------------------
  // Render one node column (Ligand/Receptor/EM/Target) into a DOM element.
  // Handles all 4 layers sequentially.
  //
  // normShard    : rows from NormalizedSubstrateStore for this cluster
  // storedLfcRow : the receiver-cache shard row (r) for this pathway,
  //                used to look up stored Ligand/Receptor/EM/Target_*_log2FC
  // ---------------------------------------------------------------------------
  async function _renderNodeColumn(
    colEl, nodeLabel, gene, cluster,
    contrast, ttRows, omicsShard,
    normShard, storedLfcRow
  ) {
    if (!gene || !cluster) {
      colEl.innerHTML = `<div class="ev-node-head">${_esc(nodeLabel)} · <em>${_esc(gene || "—")}</em></div>`
        + `<div class="ev-na muted" style="font-size:10px;">no gene on this node</div>`;
      return;
    }

    const arms = OmicsTraceStore.contrastToArms(contrast);

    // ---- Transcript sub-row ------------------------------------------------
    const ttDiv = document.createElement("div");
    ttDiv.className = "ev-layer-block";
    ttDiv.innerHTML = `<div class="ev-layer-label">${_esc(_LAYER_LABELS.transcript)}</div>`
      + `<div class="ev-cell-loading muted" style="font-size:10px;">loading…</div>`;
    colEl.innerHTML = `<div class="ev-node-head">${_esc(nodeLabel)} · <em>${_esc(gene)}</em></div>`;
    colEl.appendChild(ttDiv);

    // Fill transcript from TranscriptTraceStore.
    const hasTranscript = (typeof TranscriptTraceStore !== "undefined")
      && TranscriptTraceStore.isAvailable()
      && TranscriptTraceStore.hasCluster(cluster);

    if (!hasTranscript) {
      ttDiv.innerHTML = `<div class="ev-layer-label">${_esc(_LAYER_LABELS.transcript)}</div>`
        + `<div class="ev-na muted" style="font-size:10px;">transcript not available</div>`;
    } else {
      const ttArms = TranscriptTraceStore.contrastToArms(contrast);
      if (!ttArms) {
        ttDiv.innerHTML = `<div class="ev-layer-label">${_esc(_LAYER_LABELS.transcript)}</div>`
          + `<div class="ev-na muted" style="font-size:10px;">no contrast mapping</div>`;
      } else {
        // ttRows is already-loaded per cluster; use it directly.
        const ttArmRows = _transcriptArmRows(ttRows, gene, ttArms);
        // Convert {gene, group, value} row structure to the armRows map.
        // value is pseudobulk log-normalized expression (not raw), use as-is.
        const rawMap = new Map();
        for (const a of ttArms) {
          const srcRows = ttArmRows.get(a.group) || [];
          // Map to {value: v} objects for _renderDotBarSvg.
          rawMap.set(a.group, srcRows.map(r => ({ value: r.value, log2_value: r.value })));
        }
        ttDiv.innerHTML = `<div class="ev-layer-label">${_esc(_LAYER_LABELS.transcript)}</div>`
          + _renderDotBarSvg(ttArms, rawMap, gene, false);

        // Fill transcript LFC slot.
        // Cal_scFC (analysis.R:246) calls Cal_foldchange directly — no
        // normalizeBetweenArrays — so naive log2((D+ε)/(W+ε)) with ε=1e-5
        // (Cal_scFC default correction, analysis.R:248) agrees to ≤1e-4.
        const ttLfc = _computeTranscriptLfc(ttRows, gene, ttArms);
        const storedScKey = `${nodeLabel}_sclog2FC`;
        const storedScVal = (storedLfcRow && storedLfcRow[storedScKey] != null)
          ? Number(storedLfcRow[storedScKey]) : null;
        _fillLfcSlot(ttDiv, ttLfc, storedScVal);
      }
    }

    // ---- Omics layers (protein, phospho_ps, phospho_py) --------------------
    const omicsLayers = ["protein", "phospho_ps", "phospho_py"];
    for (const layer of omicsLayers) {
      const layerDiv = document.createElement("div");
      layerDiv.className = "ev-layer-block";
      colEl.appendChild(layerDiv);

      if (!OmicsTraceStore.isAvailable()) {
        layerDiv.innerHTML = `<div class="ev-layer-label">${_esc(_LAYER_LABELS[layer])}</div>`
          + `<div class="ev-na muted" style="font-size:10px;">omics trace not available</div>`;
        continue;
      }

      // Filter shard rows to this layer + gene.
      const layerRows = (omicsShard || []).filter(r =>
        String(r.layer) === layer && String(r.gene_symbol) === String(gene)
      );

      // Compute normalized-substrate LFC and stored value for this (node, layer).
      // Both protein and phospho use NormalizedSubstrateStore (ε from meta).
      const normContrastKey = NormalizedSubstrateStore.contrastToNormKey(contrast);
      const { lfc: normLfc } = NormalizedSubstrateStore.computeLfc(
        normShard, layer, gene, normContrastKey
      );
      const lkKey = _LAYER_LFC_KEY[layer];
      const storedNormKey = lkKey ? `${nodeLabel}_${lkKey}` : null;
      const storedNormVal = (storedLfcRow && storedNormKey && storedLfcRow[storedNormKey] != null)
        ? Number(storedLfcRow[storedNormKey]) : null;

      if (layer === "protein") {
        // Single sub-row (no site_id dimension).
        if (layerRows.length === 0) {
          layerDiv.innerHTML = `<div class="ev-layer-label">${_esc(_LAYER_LABELS[layer])}</div>`
            + `<div class="ev-na muted" style="font-size:10px;">n/a</div>`;
          // Still render LFC if normalized substrate has the gene.
          if (normLfc !== null) {
            layerDiv.innerHTML += _renderLfcSlot(normLfc, storedNormVal);
          }
          continue;
        }
        const armRows = _groupByArm(layerRows, arms);
        layerDiv.innerHTML = `<div class="ev-layer-label">${_esc(_LAYER_LABELS[layer])}</div>`
          + _renderDotBarSvg(arms, armRows, gene, false);
        _fillLfcSlot(layerDiv, normLfc, storedNormVal);
      } else {
        // Phospho: one sub-row per site_id.
        // The right-edge LFC is per-gene aggregated (same as Incytr's stored value),
        // not per-site. Individual site sub-rows each show the per-gene LFC.
        const bySite = new Map();
        for (const r of layerRows) {
          const sid = r.site_id == null ? "__null__" : String(r.site_id);
          if (!bySite.has(sid)) bySite.set(sid, []);
          bySite.get(sid).push(r);
        }
        if (bySite.size === 0) {
          layerDiv.innerHTML = `<div class="ev-layer-label">${_esc(_LAYER_LABELS[layer])}</div>`
            + `<div class="ev-na muted" style="font-size:10px;">n/a</div>`;
          continue;
        }
        let layerHtml = `<div class="ev-layer-label">${_esc(_LAYER_LABELS[layer])}</div>`;
        for (const [sid, siteRows] of bySite) {
          const armRows = _groupByArm(siteRows, arms);
          layerHtml += `<div class="ev-site-block">`
            + _renderDotBarSvg(arms, armRows, sid, false)
            + `</div>`;
        }
        layerDiv.innerHTML = layerHtml;
        // Fill per-gene LFC into all placeholder slots in this layer block.
        _fillLfcSlot(layerDiv, normLfc, storedNormVal);
      }
    }
  }

  // Group shard rows by arm's group code → Map<group, rows[]>.
  function _groupByArm(rows, arms) {
    const m = new Map();
    if (arms) for (const a of arms) m.set(a.group, []);
    for (const r of (rows || [])) {
      const g = String(r.group);
      if (m.has(g)) m.get(g).push(r);
    }
    return m;
  }

  // ---------------------------------------------------------------------------
  // Public entry point: render the Evidence panel for one pathway row.
  //
  // host     : DOM element to write into
  // r        : shard row object (keys: Ligand, Receptor, EM, Target, _sender,
  //            _receiver, contrast)
  // rk       : row key string (for id namespacing)
  // ---------------------------------------------------------------------------
  async function render(host, r, rk) {
    if (!host) return;

    const contrast = r.contrast || "";
    const sender   = r._sender   || "";
    const receiver = r._receiver || "";

    // Per evaluation.R:227-230: Ligand→sender; Receptor/EM/Target→receiver.
    const nodes = [
      { node: "Ligand",   gene: r.Ligand,   cluster: sender   },
      { node: "Receptor", gene: r.Receptor, cluster: receiver },
      { node: "EM",       gene: r.EM,       cluster: receiver },
      { node: "Target",   gene: r.Target,   cluster: receiver },
    ];

    const arms = OmicsTraceStore.contrastToArms(contrast);
    const armsLabel = arms
      ? `${arms[0].arm} vs ${arms[1].arm} @ ${contrast.split("_")[1] || ""}`
      : contrast;

    // Render skeleton with loading placeholders.
    const safeRk = rk.replace(/[^a-zA-Z0-9]/g, "_");
    const gridId = `ev-grid-${safeRk}`;
    host.innerHTML =
      `<div class="ev-note muted" style="font-size:11px;margin-bottom:6px;">`
      + `Evidence · ${_esc(armsLabel)} · dots=animals · bars=mean · males-only`
      + `</div>`
      + `<div class="ev-grid" id="${_esc(gridId)}">`
      + nodes.map((nd, i) =>
          `<div class="ev-col" id="${_esc(gridId)}-col-${i}">`
          + `<div class="ev-node-head">${_esc(nd.node)} · <em>${_esc(nd.gene || "—")}</em></div>`
          + `<div class="ev-col-loading muted" style="font-size:10px;">loading…</div>`
          + `</div>`
        ).join("")
      + `</div>`;

    // Load per-cluster omics shards (only unique clusters).
    const clusterSet = new Set(nodes.map(nd => nd.cluster).filter(Boolean));
    const shardMap = new Map(); // cluster -> rows[]
    await Promise.all([...clusterSet].map(async cl => {
      try {
        const rows = OmicsTraceStore.isAvailable()
          ? await OmicsTraceStore.loadCluster(cl) : [];
        shardMap.set(cl, rows);
      } catch (e) {
        shardMap.set(cl, []);
      }
    }));

    // Load transcript shards (only unique clusters present in TranscriptTraceStore).
    const ttShardMap = new Map(); // cluster -> rows[]
    const hasTT = (typeof TranscriptTraceStore !== "undefined")
      && TranscriptTraceStore.isAvailable();
    if (hasTT) {
      await Promise.all([...clusterSet].map(async cl => {
        if (!TranscriptTraceStore.hasCluster(cl)) { ttShardMap.set(cl, []); return; }
        try {
          ttShardMap.set(cl, await TranscriptTraceStore.loadCluster(cl));
        } catch (e) {
          ttShardMap.set(cl, []);
        }
      }));
    }

    // Load normalized substrate shards for LFC recomputation (Item 3.4).
    const normShardMap = new Map(); // cluster -> rows[]
    const hasNorm = (typeof NormalizedSubstrateStore !== "undefined")
      && NormalizedSubstrateStore.isAvailable();
    if (hasNorm) {
      await Promise.all([...clusterSet].map(async cl => {
        if (!NormalizedSubstrateStore.hasCluster(cl)) { normShardMap.set(cl, []); return; }
        try {
          normShardMap.set(cl, await NormalizedSubstrateStore.loadCluster(cl));
        } catch (e) {
          normShardMap.set(cl, []);
        }
      }));
    }

    // Render each node column.
    const gridEl = document.getElementById(gridId);
    if (!gridEl) return;   // panel was replaced before load completed

    await Promise.all(nodes.map(async (nd, i) => {
      const colEl = document.getElementById(`${gridId}-col-${i}`);
      if (!colEl) return;
      const omicsShard = shardMap.get(nd.cluster) || [];
      const ttRows     = ttShardMap.get(nd.cluster) || [];
      const normShard  = normShardMap.get(nd.cluster) || [];
      await _renderNodeColumn(
        colEl, nd.node, nd.gene, nd.cluster,
        contrast, ttRows, omicsShard,
        normShard, r   // r is the pathway row with all stored LFC columns
      );
    }));
  }

  return { render };
})();
window.EvidencePanel = EvidencePanel;

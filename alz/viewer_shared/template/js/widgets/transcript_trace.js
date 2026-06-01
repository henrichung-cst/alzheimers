// ---------------------------------------------------------------------------
// TranscriptTraceStore — per-cluster transcript pseudobulk shards backing the
// Incytr Pathways "Measurement Trace" panel. Symmetric with MeasurementTraceStore
// for the kinase-audit drawer.
//
// Shard layout: outputs/reports/unified_viewer/audit_sources/transcript_trace/
// <slug>.parquet where slug = sanitize_celltype(cluster).
//
// N=1 per arm — the Song snRNA-seq pseudobulk has one library per (sex × tp ×
// genotype) group, so each arm shows a single value. The panel must label
// this prominently so the bars aren't misread as a distribution.
// ---------------------------------------------------------------------------

const TranscriptTraceStore = (() => {
  const MAX = 8;                  // LRU cap — matches SliceCache discipline
  const cache = new Map();        // cluster -> rows[] (insertion-order LRU)
  const inflight = new Map();     // cluster -> Promise<rows>

  function _lruTouch(key, value) {
    if (cache.has(key)) cache.delete(key);
    cache.set(key, value);
    while (cache.size > MAX) cache.delete(cache.keys().next().value);
  }

  // Mirror of alz/integration/load.R::sanitize_celltype and
  // alz/incytr_pair/pair_to_receiver_cache.py::_sanitize_celltype.
  // Single helper — do NOT scatter the replace calls across this file.
  function _sanitize(name) {
    return String(name).replaceAll("/", "-").replaceAll(" ", "_");
  }

  function _activeContext() {
    return ViewerPayload.activeContext();
  }

  // Context-aware block resolution. Single-context payloads can still use a
  // top-level {clusters, relative_path}; multi-context payloads can provide
  // by_context[<context>] with distinct shard directories.
  function _meta() {
    const m = (typeof PAYLOAD !== "undefined"
               && PAYLOAD.meta
               && PAYLOAD.meta.transcript_trace) || null;
    if (!m) return null;
    if (m.by_context) return m.by_context[_activeContext()] || null;
    return m;
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
    const m = _meta();
    const base = (m && m.relative_path) ? `${m.relative_path}/` : "audit_sources/transcript_trace/";
    // Donor-scoped relative_path → unique cache key per donor (otherwise the
    // same cluster name in two donors would collide).
    const cacheKey = `${base}${cluster}`;
    if (cache.has(cacheKey)) {
      const v = cache.get(cacheKey); _lruTouch(cacheKey, v); return v;
    }
    if (inflight.has(cacheKey)) return inflight.get(cacheKey);
    const url = `${base}${_sanitize(cluster)}.parquet`;
    const p = _fetchParquet(url).then(rows => {
      _lruTouch(cacheKey, rows);
      inflight.delete(cacheKey);
      return rows;
    }).catch(err => {
      inflight.delete(cacheKey);
      throw err;
    });
    inflight.set(cacheKey, p);
    return p;
  }

  // Pair-mode inverse of _GENO_MAP (alz/incytr_pair/pair_to_receiver_cache.py):
  //   AppP ← App, Ttau ← Tau, ApTt ← ApTt.
  // Hardcoded `ma_` prefix encodes the males-only pair-mode assumption (see
  // analysis_mode in conf/base/parameters.yml). Revisit if the contrast schema
  // gains a sex dimension or non-males-only pair runs land.
  const _GENO_INV = { App: "AppP", Tau: "Ttau", ApTt: "ApTt" };

  function _cohort() {
    return (typeof PAYLOAD !== "undefined" && PAYLOAD.meta && PAYLOAD.meta.cohort) || "mouse";
  }

  function contrastToArms(contrast) {
    if (!contrast) return null;
    const parts = String(contrast).split("_");
    if (parts.length !== 2) return null;
    // T-cell contrasts are `<day>_<baseline>` (e.g. d13_d2). Shard `group`
    // column is the day token itself.
    if (_cohort() === "tcell") {
      const [day, baseline] = parts;
      return [
        { arm: day,      group: day },
        { arm: baseline, group: baseline },
      ];
    }
    const [geno, age] = parts;
    const genoCode = _GENO_INV[geno];
    if (!genoCode) return null;
    return [
      { arm: geno,  group: `ma_${age}_${genoCode}` },
      { arm: "WT",  group: `ma_${age}_WTyp` },
    ];
  }

  // Return [{arm, group, value}, {arm, group, value}] for (cluster, gene,
  // contrast). Missing gene → value null.
  async function values(cluster, gene, contrast) {
    const arms = contrastToArms(contrast);
    if (!arms) return null;
    const rows = await loadCluster(cluster);
    if (!rows || !rows.length) {
      return arms.map(a => ({ ...a, value: null }));
    }
    const out = [];
    for (const a of arms) {
      const hit = rows.find(r =>
        String(r.gene) === String(gene) && String(r.group) === a.group);
      out.push({
        ...a,
        value: hit && hit.value != null ? Number(hit.value) : null,
      });
    }
    return out;
  }

  // Render a 2-bar SVG into `host` for one (gene, cluster, contrast) panel.
  // arms = [{arm, group, value}, ...] from values().
  function renderTwoBarPanel(host, label, arms) {
    if (!host) return;
    if (!arms) {
      host.innerHTML = `<div class="tt-panel-empty muted">no contrast mapping</div>`;
      return;
    }
    const allNull = arms.every(a => a.value == null);
    if (allNull) {
      host.innerHTML =
        `<div class="tt-panel-head">${_escapeHtml(label)}</div>` +
        `<div class="tt-panel-empty muted">no transcript pseudobulk for this gene in this cluster</div>`;
      return;
    }
    const vmax = Math.max(0.001, ...arms.map(a => a.value == null ? 0 : Math.abs(a.value)));
    const yScale = vmax * 1.15;
    const W = 140, H = 90, padTop = 18, padBot = 28, padLR = 14;
    const barW = (W - 2 * padLR) / arms.length - 8;
    const barAreaH = H - padTop - padBot;
    const bars = arms.map((a, i) => {
      const cx = padLR + i * ((W - 2 * padLR) / arms.length) + ((W - 2 * padLR) / arms.length - barW) / 2;
      const v = a.value;
      if (v == null) {
        return `<text x="${cx + barW / 2}" y="${padTop + barAreaH - 4}" `
             + `text-anchor="middle" font-size="9" fill="#888">—</text>`
             + `<text x="${cx + barW / 2}" y="${H - 10}" text-anchor="middle" `
             + `font-size="10" fill="#444">${_escapeHtml(a.arm)}</text>`;
      }
      const hpx = Math.max(1, (Math.abs(v) / yScale) * barAreaH);
      const y = padTop + barAreaH - hpx;
      const color = a.arm === "WT" ? "#777" : "#a3203c";
      const valLabel = v.toFixed(2);
      return `<rect x="${cx}" y="${y}" width="${barW}" height="${hpx}" fill="${color}" rx="2"/>`
           + `<text x="${cx + barW / 2}" y="${y - 3}" text-anchor="middle" `
           + `font-size="10" fill="#222">${valLabel}</text>`
           + `<text x="${cx + barW / 2}" y="${H - 10}" text-anchor="middle" `
           + `font-size="10" fill="#444">${_escapeHtml(a.arm)}</text>`;
    }).join("");
    host.innerHTML =
      `<div class="tt-panel-head">${_escapeHtml(label)}</div>` +
      `<svg class="tt-panel-svg" viewBox="0 0 ${W} ${H}" width="${W}" height="${H}">${bars}</svg>`;
  }

  return {
    isAvailable, hasCluster, loadCluster, contrastToArms, values,
    renderTwoBarPanel, _sanitize,
  };
})();
window.TranscriptTraceStore = TranscriptTraceStore;

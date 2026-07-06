// ---------------------------------------------------------------------------
// Early AD Kinases tab — D1 cross-cohort substrate overlap.
//
// Each row is one kinase from the 60-kinase pool. Overlap is gene-identity-keyed
// (shared / human_only / mouse_only genes, split by BLOSUM62 site matching and
// coverage awareness) across 8 contexts (tissue × age). The 5xFAD age axis
// (3→6→9→12 mo) is the disease-progression timeline, so each tissue renders a
// 1×4 age strip whose tiles are shaded by the count of cross-species-conserved
// substrate genes (n_shared_gene), normalized within the kinase. Bright-early =
// the kinase's human-validated conserved program is already engaged at 3/6 mo
// = an early-AD kinase; bright-late = engages only as pathology matures.
// Human vs mouse kinase-NES direction glyphs are joined by name against
// PAYLOAD.human / supporting_5xfad. The detail pane shows per-context overlap
// counts (including overlap_frac_gene and coverage splits) and a substrate table
// (fetched lazily as a per-kinase parquet shard with partition/coverage columns).
// ---------------------------------------------------------------------------

let _subRows = null;              // folded per-kinase row model
let _subVisible = [];             // last filtered+sorted subset (export scope)
let _subHumanDir = null;          // Map<NAME, +1/-1/0>
let _subMouseDir = null;          // Map<NAME, +1/-1/0>
let _subPairsCache = new Map();   // NAME -> shard rows[] (detail table)
let _subDetailTab = "overlap";    // detail sub-tab: "overlap" | "pairs"

const _subFilter = { search: "", minShared: 0,
                     sortCol: "peakShared", sortAsc: false };

function _subEsc(s) {
  return String(s == null ? "" : s).replace(/[&<>"]/g,
    c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
}
function _subSanitizeKinase(name) {
  return String(name || "").replace(/[^A-Za-z0-9]+/g, "_");
}
function _subFmt(v, d) {
  return (v == null || Number.isNaN(v)) ? "—" : Number(v).toFixed(d == null ? 2 : d);
}

// ---- direction lookups (kinase NES sign in each cohort) --------------------
function _subBuildDirLookups() {
  _subHumanDir = new Map();
  const H = (typeof PAYLOAD !== "undefined" && PAYLOAD && PAYLOAD.human) || null;
  if (H && H.kinases && H.kinases.name) {
    const HK = H.kinases;
    const nes = HK.median_nes_sig_only || [];
    for (let i = 0; i < HK.name.length; i++) {
      const v = nes[i];
      if (v != null && !Number.isNaN(v) && v !== 0) {
        _subHumanDir.set(HK.name[i], v > 0 ? 1 : -1);
      } else if (!_subHumanDir.has(HK.name[i])) {
        _subHumanDir.set(HK.name[i], 0);
      }
    }
  }
  // Mouse: mean ST-stoichiometry NES across contexts, sign per kinase.
  _subMouseDir = new Map();
  const F5 = (typeof PAYLOAD !== "undefined" && PAYLOAD && PAYLOAD.supporting_5xfad) || null;
  if (F5 && F5.rows) {
    const acc = new Map();          // NAME -> {sum, n}
    for (const r of F5.rows) {
      if (r.analysis_track && r.analysis_track !== "stoichiometry") continue;
      const residue = (r.residue_type || r.track || "").toUpperCase();
      if (residue && residue !== "ST") continue;   // pool is ST-only
      const v = Number(r.NES);
      if (!Number.isFinite(v)) continue;
      const a = acc.get(r.kinase) || { sum: 0, n: 0 };
      a.sum += v; a.n += 1; acc.set(r.kinase, a);
    }
    for (const [name, a] of acc) {
      const mean = a.n ? a.sum / a.n : 0;
      _subMouseDir.set(name, mean > 0 ? 1 : (mean < 0 ? -1 : 0));
    }
  }
}

// ---- row model -------------------------------------------------------------
function _subBuildRows() {
  const SC = (typeof PAYLOAD !== "undefined" && PAYLOAD && PAYLOAD.substrate_compare) || null;
  if (!SC || !SC.kinase) { _subRows = []; return; }
  if (_subHumanDir === null) _subBuildDirLookups();

  const famMap = (typeof META !== "undefined" && META && META.familyMap) || {};
  const byName = new Map();
  const n = SC.kinase.length;
  for (let i = 0; i < n; i++) {
    const name = SC.kinase[i];
    let row = byName.get(name);
    if (!row) {
      const tier = (SC.tiers && SC.tiers[name]) || null;
      row = {
        name,
        family: famMap[name] || "",
        perCtx: {},
        humanDir: _subHumanDir.get(name) ?? null,
        mouseDir: _subMouseDir.get(name) ?? null,
        tier,
        tierRank: tier === "green" ? 2 : (tier === "yellow" ? 1 : 0),
      };
      byName.set(name, row);
    }
    row.perCtx[SC.context[i]] = {
      overlapFrac: SC.overlap_frac_gene[i],
      nSharedGene: SC.n_shared_gene[i],
      nHumanOnlyGene: SC.n_human_only_gene[i],
      nMouseOnlyGene: SC.n_mouse_only_gene[i],
      nHumanOnlyEngaged: SC.n_human_only_engaged[i],
      nHumanOnlyUnmeasured: SC.n_human_only_unmeasured[i],
      nMouseOnlyEngaged: SC.n_mouse_only_engaged[i],
      nMouseOnlyUnmeasured: SC.n_mouse_only_unmeasured[i],
      nSharedSite: SC.n_shared_site[i],
      nDiffsite: SC.n_diffsite[i],
      blosumSim: SC.blosum_similarity[i],
      dagree: SC.direction_agree_frac[i],
      hist: SC.sim_hist[i] || [],
    };
  }
  // Derived scalars per kinase. Headline metric is conserved-substrate breadth
  // (n_shared_gene); overlap_frac_gene is retained for the detail table only.
  const rows = [];
  for (const row of byName.values()) {
    let peakShared = null, peakCtx = null;   // peak # conserved substrate genes
    let peakOverlap = null;                  // peak overlap_frac (detail/export)
    let agreeSites = 0, totSites = 0;        // pooled cross-species site concordance
    const sharedByTissue = {};
    for (const t of SC.tissues) sharedByTissue[t] = null;
    for (const ctx of SC.contexts) {
      const c = row.perCtx[ctx];
      if (!c) continue;
      if (c.nSharedGene != null && (peakShared == null || c.nSharedGene > peakShared)) {
        peakShared = c.nSharedGene; peakCtx = ctx;
      }
      if (c.overlapFrac != null && (peakOverlap == null || c.overlapFrac > peakOverlap)) {
        peakOverlap = c.overlapFrac;
      }
      if (c.nSharedGene != null) {
        const tissue = ctx.slice(0, ctx.lastIndexOf("_"));
        if (sharedByTissue[tissue] == null || c.nSharedGene > sharedByTissue[tissue])
          sharedByTissue[tissue] = c.nSharedGene;
      }
      // Pool agreeing / total shared sites across contexts — stable per-kinase
      // concordance, unlike the per-context fraction (small, noisy denominators).
      if (c.dagree != null && !Number.isNaN(c.dagree) && c.nSharedSite) {
        totSites += c.nSharedSite;
        agreeSites += c.dagree * c.nSharedSite;
      }
    }
    // Per-kinase normalizer for tile shading: max conserved-gene count over all
    // 8 contexts, so each kinase's own age trajectory is legible and comparable
    // across its two tissue strips.
    row.maxSharedGene = peakShared;
    row.peakShared = peakShared;
    row.peakOverlap = peakOverlap;
    row.peakCtx = peakCtx;
    row.peakSharedCortex = sharedByTissue["cortex"] ?? null;
    row.peakSharedHippocampus = sharedByTissue["hippocampus"] ?? null;
    row.dirAgree = totSites > 0 ? agreeSites / totSites : null;
    row.dirAgreeSites = totSites;   // pooled shared-site denominator (tooltip)
    rows.push(row);
  }
  _subRows = rows;
}

// ---- 1×4 conserved-breadth strip (one tissue, four ages) -------------------
// Tile shade = n_shared_gene at that age, normalized to the kinase's peak
// conserved-gene count over all 8 contexts. The age axis is the 5xFAD disease
// timeline, so bright-early tiles flag a kinase engaged before pathology matures.
function _subOverlapRow(row, tissue) {
  const SC = PAYLOAD.substrate_compare;
  const denom = row.maxSharedGene || 0;
  let cells = "";
  for (const age of SC.ages) {
    const ctx = `${tissue}_${age}`;
    const c = row.perCtx[ctx];
    if (!c || c.nSharedGene == null) {
      cells += `<div class="sub-zc na" title="${_subEsc(ctx)}: no data"></div>`;
      continue;
    }
    const frac = denom > 0 ? c.nSharedGene / denom : 0;
    const op = (0.12 + 0.88 * frac).toFixed(3);
    const tip = `${ctx}: shared substrates=${c.nSharedGene} `
      + `(${denom > 0 ? (frac * 100).toFixed(0) : "0"}% of this kinase's peak) · `
      + `overlap=${c.overlapFrac == null ? "—" : (c.overlapFrac * 100).toFixed(1) + "%"} · `
      + `human-only=${c.nHumanOnlyGene} · mouse-only=${c.nMouseOnlyGene}`;
    cells += `<div class="sub-zc" style="--op:${op}" title="${_subEsc(tip)}"></div>`;
  }
  return `<div class="sub-profile" role="img" aria-label="${_subEsc(tissue)} shared substrates by age">${cells}</div>`;
}

// Age ticks aligned over a tissue column's four cells (rendered in the header).
function _subAgeTicks() {
  const SC = PAYLOAD.substrate_compare;
  const ticks = SC.ages.map(a => `<span>${_subEsc(String(a).replace(/mo$/, ""))}</span>`).join("");
  return `<div class="sub-age-ticks">${ticks}</div>`;
}

// Curated effector-tier pill. tier ∈ {green, yellow} → label from SC.tier_labels.
function _subTierPill(tier) {
  if (tier !== "green" && tier !== "yellow") return `<span class="muted">—</span>`;
  const SC = PAYLOAD.substrate_compare;
  const label = (SC.tier_labels && SC.tier_labels[tier]) || tier;
  return `<span class="sub-tier sub-tier-${tier}">${_subEsc(label)}</span>`;
}

function _subDirGlyph(d) {
  if (d == null) return `<span class="muted">—</span>`;
  if (d > 0) return `<span class="sub-dir up" title="up in disease">▲</span>`;
  if (d < 0) return `<span class="sub-dir down" title="down in disease">▼</span>`;
  return `<span class="muted" title="no net direction">·</span>`;
}

// ---- master table ----------------------------------------------------------
const _SUB_COLS = [
  { key: "name", label: "Kinase", num: false, sort: "name", str: true },
  { key: "tier", label: "Effector tier", num: false, sort: "tierRank" },
  { key: "family", label: "Family", num: false, sort: "family", str: true },
  { key: "humanDir", label: "Human", num: false, sort: "humanDir" },
  { key: "mouseDir", label: "Mouse", num: false, sort: "mouseDir" },
  { key: "peakShared", label: "Shared substrates", num: true, sort: "peakShared" },
  { key: "dirAgree", label: "Direction match", num: true, sort: "dirAgree" },
  { key: "profileCortex", label: "Cortex shared", num: false, sort: "peakSharedCortex", ticks: true },
  { key: "profileHippocampus", label: "Hippocampus shared", num: false, sort: "peakSharedHippocampus", ticks: true },
];
const _SUB_STR_COLS = new Set(["name", "family"]);

function _subCompare(a, b) {
  const col = _subFilter.sortCol;
  const av = a[col], bv = b[col];
  if (_SUB_STR_COLS.has(col)) {
    const sa = String(av == null ? "" : av), sb = String(bv == null ? "" : bv);
    return _subFilter.sortAsc ? sa.localeCompare(sb) : sb.localeCompare(sa);
  }
  return numCmp(av, bv, _subFilter.sortAsc ? -1 : 1);
}

function renderSubstrateCompare() {
  if (_subRows === null) _subBuildRows();
  const wrap = document.getElementById("sub-table-wrap");
  if (!wrap) return;
  const kf = _subFilter;
  const q = kf.search.trim().toLowerCase();

  let visible = _subRows.filter(r => {
    if (q && !(r.name.toLowerCase().includes(q) || r.family.toLowerCase().includes(q))) return false;
    if (kf.minShared > 0 && !(r.peakShared != null && r.peakShared >= kf.minShared)) return false;
    return true;
  });
  visible.sort(_subCompare);
  _subVisible = visible;

  const sel = (typeof Store !== "undefined" && Store.state.selection.substrate) || null;
  const arrow = c => (kf.sortCol === c ? (kf.sortAsc ? " ▲" : " ▼") : "");
  let thead = "<thead><tr>";
  for (const c of _SUB_COLS) {
    const sortKey = c.sort || (c.num ? c.key : null);
    const cls = c.num ? ' class="attr-num"' : "";
    const ticks = c.ticks ? _subAgeTicks() : "";
    if (sortKey) {
      thead += `<th${cls} data-sort="${sortKey}" role="button" tabindex="0" title="Sort">${_subEsc(c.label)}${arrow(sortKey)}${ticks}</th>`;
    } else {
      thead += `<th${cls}>${_subEsc(c.label)}${ticks}</th>`;
    }
  }
  thead += "</tr></thead>";

  let body = "<tbody>";
  for (const r of visible) {
    const selCls = (sel === r.name) ? " selected" : "";
    body += `<tr class="sub-row${selCls}" data-name="${_subEsc(r.name)}">`;
    body += `<td>${_subEsc(r.name)}</td>`;
    body += `<td>${_subTierPill(r.tier)}</td>`;
    body += `<td>${_subEsc(r.family) || '<span class="muted">—</span>'}</td>`;
    body += `<td>${_subDirGlyph(r.humanDir)}</td>`;
    body += `<td>${_subDirGlyph(r.mouseDir)}</td>`;
    body += `<td class="attr-num" title="peak # substrate genes hit in both human AD and 5xFAD, across the 8 contexts @ ${_subEsc(r.peakCtx || "—")}">`
      + `${r.peakShared == null ? '<span class="muted">—</span>' : r.peakShared.toLocaleString()}</td>`;
    body += `<td class="attr-num" title="${r.dirAgree == null ? "no shared sites" : `share of ${r.dirAgreeSites.toLocaleString()} shared substrate sites that move the same disease direction in human and mouse (pooled across contexts)`}">`
      + `${r.dirAgree == null ? '<span class="muted">—</span>' : (r.dirAgree * 100).toFixed(0) + "%"}</td>`;
    body += `<td>${_subOverlapRow(r, "cortex")}</td>`;
    body += `<td>${_subOverlapRow(r, "hippocampus")}</td>`;
    body += `</tr>`;
  }
  body += "</tbody>";

  wrap.innerHTML = `<table class="data-table sub-table">${thead}${body}</table>`;
  const cnt = document.getElementById("sub-count");
  if (cnt) cnt.textContent = `${visible.length} / ${_subRows.length} kinases`;

  if (sel != null) {
    renderSubstrateDetail(sel);
  } else {
    const el = document.getElementById("sub-detail");
    if (el) el.innerHTML = `<div class="muted" style="padding:12px;">Select a kinase to see its shared substrate pairs and BLOSUM-similarity histogram.</div>`;
  }
}

// ---- detail pane -----------------------------------------------------------
function _subHistogram(bins, range) {
  const total = bins.reduce((s, v) => s + v, 0);
  const max = Math.max(1, ...bins);
  const lo = range ? range[0] : 0.5, hi = range ? range[1] : 1.0;
  let bars = "";
  for (let i = 0; i < bins.length; i++) {
    const h = (bins[i] / max * 100).toFixed(1);
    const a = (lo + (hi - lo) * i / bins.length).toFixed(2);
    const b = (lo + (hi - lo) * (i + 1) / bins.length).toFixed(2);
    bars += `<div class="sub-bar" style="height:${h}%" title="${a}–${b}: ${bins[i]}"></div>`;
  }
  return `<div class="sub-hist-wrap">
      <div class="sub-hist">${bars}</div>
      <div class="sub-hist-axis"><span>${lo.toFixed(2)}</span><span>BLOSUM similarity</span><span>${hi.toFixed(2)}</span></div>
      <div class="muted sub-hist-total">${total} human substrate motifs (best BLOSUM match to mouse)</div>
    </div>`;
}

// All-8-contexts overlap table (detail "Overlap" tab).
function _subOverlapTable(row, SC) {
  let body = "";
  for (const ctx of SC.contexts) {
    const c = row.perCtx[ctx];
    const peakTag = ctx === row.peakCtx
      ? ' <span class="sub-z-peaktag">peak</span>' : "";
    if (!c || c.overlapFrac == null) {
      body += `<tr${ctx === row.peakCtx ? ' class="sub-z-peak"' : ""}>` +
              `<td>${_subEsc(ctx)}${peakTag}</td>` +
              `<td class="attr-num muted" colspan="9">no data</td></tr>`;
      continue;
    }
    body += `<tr${ctx === row.peakCtx ? ' class="sub-z-peak"' : ""}>`;
    body += `<td>${_subEsc(ctx)}${peakTag}</td>`;
    body += `<td class="attr-num">${(c.overlapFrac * 100).toFixed(1)}%</td>`;
    body += `<td class="attr-num">${c.nSharedGene == null ? "—" : c.nSharedGene.toLocaleString()}</td>`;
    body += `<td class="attr-num" title="engaged=${c.nHumanOnlyEngaged} unmeasured=${c.nHumanOnlyUnmeasured}">`
      + `${c.nHumanOnlyGene == null ? "—" : c.nHumanOnlyGene.toLocaleString()}</td>`;
    body += `<td class="attr-num" title="engaged=${c.nMouseOnlyEngaged} unmeasured=${c.nMouseOnlyUnmeasured}">`
      + `${c.nMouseOnlyGene == null ? "—" : c.nMouseOnlyGene.toLocaleString()}</td>`;
    body += `<td class="attr-num">${c.nSharedSite == null ? "—" : c.nSharedSite.toLocaleString()}</td>`;
    body += `<td class="attr-num">${c.nDiffsite == null ? "—" : c.nDiffsite.toLocaleString()}</td>`;
    body += `<td class="attr-num">${c.dagree == null ? "—" : (c.dagree * 100).toFixed(0) + "%"}</td>`;
    body += `<td class="attr-num">${c.blosumSim == null ? "—" : c.blosumSim.toFixed(3)}</td>`;
    body += `</tr>`;
  }
  return `<table class="data-table sub-z-table">
      <thead><tr>
        <th>Context</th>
        <th class="attr-num">overlap%</th>
        <th class="attr-num">shared</th>
        <th class="attr-num">human-only</th>
        <th class="attr-num">mouse-only</th>
        <th class="attr-num">same-site</th>
        <th class="attr-num">diff-site</th>
        <th class="attr-num">direction match</th>
        <th class="attr-num">BLOSUM sim</th>
      </tr></thead>
      <tbody>${body}</tbody>
    </table>
    <div class="muted sub-hist-total">Gene-identity overlap per tissue × age context.
      human-only / mouse-only hover shows engaged (detectable but not engaged) vs unmeasured (coverage gap).</div>`;
}

// Per-context histogram + substrate tables split by partition (detail "Substrate
// motifs" tab): shared_site, shared_gene_diffsite, human-only, mouse-only.
function _subPairsPanel(SC, ctx, opts, c) {
  const nHumEng = c.nHumanOnlyEngaged == null ? "—" : c.nHumanOnlyEngaged;
  const nHumUnm = c.nHumanOnlyUnmeasured == null ? "—" : c.nHumanOnlyUnmeasured;
  const nMouEng = c.nMouseOnlyEngaged == null ? "—" : c.nMouseOnlyEngaged;
  const nMouUnm = c.nMouseOnlyUnmeasured == null ? "—" : c.nMouseOnlyUnmeasured;
  return `
    <div class="sub-detail-ctxbar">
      <label class="ke-filter-label">Context
        <select id="sub-ctx-select" aria-label="Comparison context">${opts}</select>
      </label>
      <span class="sub-ctx-stats muted">
        overlap=${c.overlapFrac == null ? "—" : (c.overlapFrac * 100).toFixed(1) + "%"} ·
        shared ${c.nSharedGene == null ? "—" : c.nSharedGene} ·
        human-only ${c.nHumanOnlyGene == null ? "—" : c.nHumanOnlyGene} (eng ${nHumEng} / unm ${nHumUnm}) ·
        mouse-only ${c.nMouseOnlyGene == null ? "—" : c.nMouseOnlyGene} (eng ${nMouEng} / unm ${nMouUnm}) ·
        direction match ${c.dagree == null ? "—" : (c.dagree * 100).toFixed(0) + "%"}
      </span>
    </div>
    ${_subHistogram(c.hist || [], SC.hist_range)}
    <div class="sub-pairs-section">
      <h4 class="sub-pairs-h">Shared — same site <span class="muted">(same gene, BLOSUM-matched residue ≥ 0.50)</span></h4>
      <div id="sub-pairs-shared" class="sub-pairs-host"></div>
    </div>
    <div class="sub-pairs-section">
      <h4 class="sub-pairs-h">Shared — different residue <span class="muted">(same gene, no BLOSUM-matched residue)</span></h4>
      <div id="sub-pairs-diffsite" class="sub-pairs-host"></div>
    </div>
    <div class="sub-pairs-section">
      <h4 class="sub-pairs-h">Human-only — engaged in mouse <span class="muted">(gene detectable in 5xFAD but not engaged by this kinase)</span></h4>
      <div id="sub-pairs-human-engaged" class="sub-pairs-host"></div>
    </div>
    <div class="sub-pairs-section">
      <h4 class="sub-pairs-h">Human-only — unmeasured in mouse <span class="muted">(gene not in 5xFAD ST universe — coverage gap)</span></h4>
      <div id="sub-pairs-human-unmeasured" class="sub-pairs-host"></div>
    </div>
    <div class="sub-pairs-section">
      <h4 class="sub-pairs-h">Mouse-only — engaged in human <span class="muted">(gene detectable in human NBB but not engaged by this kinase)</span></h4>
      <div id="sub-pairs-mouse-engaged" class="sub-pairs-host"></div>
    </div>
    <div class="sub-pairs-section">
      <h4 class="sub-pairs-h">Mouse-only — unmeasured in human <span class="muted">(gene not in human NBB ST universe — coverage gap)</span></h4>
      <div id="sub-pairs-mouse-unmeasured" class="sub-pairs-host"></div>
    </div>`;
}

const _SUB_SHARED_COLS = ["gene_a", "site_a", "motif_a", "site_b", "motif_b",
                          "similarity", "direction_a", "direction_b", "direction_agree"];
const _SUB_DIFFSITE_COLS = ["gene_a", "site_a", "motif_a", "site_b", "motif_b", "direction_a", "direction_b"];
const _SUB_HUMAN_COLS = ["gene_a", "site_a", "motif_a", "direction_a", "support_a"];
const _SUB_MOUSE_COLS = ["gene_b", "site_b", "motif_b", "direction_b", "support_b"];

function _subRenderSection(hostId, columns, rows, title) {
  const host = document.getElementById(hostId);
  if (!host) return;
  if (!rows.length) {
    host.innerHTML = `<div class="muted" style="padding:6px 8px;">None in this context.</div>`;
    return;
  }
  new AuditTable(hostId, {
    tableKey: "substrate_pairs",
    columns,
    rows,
    pageSize: 10,
    title,
    fullSourceKey: false,
  }).render();
}

async function _subFillPairTable(name, ctx) {
  const anchor = document.getElementById("sub-pairs-shared");
  if (!anchor) return;
  let rows = _subPairsCache.get(name);
  if (rows === undefined) {
    anchor.innerHTML = `<div class="muted" style="padding:8px;">Loading substrate motifs…</div>`;
    try {
      rows = await SliceCache.loadSubstratePairs(name);
    } catch (e) {
      anchor.innerHTML = `<div class="muted" style="padding:8px;">Could not load substrate motifs: ${_subEsc(e.message)}</div>`;
      return;
    }
    _subPairsCache.set(name, rows);
  }
  // Guard against a race: only render if this kinase is still selected.
  if (Store.state.selection.substrate !== name) return;
  const inCtx = rows.filter(r => r.context === ctx);

  const sharedRows = inCtx
    .filter(r => r.partition === "shared_site")
    .map(r => ({
      gene_a: r.gene_a, site_a: r.site_a, motif_a: r.motif_a,
      site_b: r.site_b, motif_b: r.motif_b,
      similarity: (r.similarity == null || r.similarity === "") ? "" : Number(r.similarity).toFixed(3),
      direction_a: _subDirWord(r.direction_a),
      direction_b: _subDirWord(r.direction_b),
      direction_agree: (r.direction_agree == null || r.direction_agree === "") ? "" : (Number(r.direction_agree) ? "yes" : "no"),
    }));
  const diffSiteRows = inCtx
    .filter(r => r.partition === "shared_gene_diffsite")
    .map(r => ({
      gene_a: r.gene_a || r.gene_b,
      site_a: r.site_a, motif_a: r.motif_a,
      site_b: r.site_b, motif_b: r.motif_b,
      direction_a: _subDirWord(r.direction_a),
      direction_b: _subDirWord(r.direction_b),
    }));
  const humanEngagedRows = inCtx
    .filter(r => r.partition === "human_only_engaged")
    .map(r => ({
      gene_a: r.gene_a, site_a: r.site_a, motif_a: r.motif_a,
      direction_a: _subDirWord(r.direction_a), support_a: r.support_a,
    }));
  const humanUnmeasuredRows = inCtx
    .filter(r => r.partition === "human_only_unmeasured")
    .map(r => ({
      gene_a: r.gene_a, site_a: r.site_a, motif_a: r.motif_a,
      direction_a: _subDirWord(r.direction_a), support_a: r.support_a,
    }));
  const mouseEngagedRows = inCtx
    .filter(r => r.partition === "mouse_only_engaged")
    .map(r => ({
      gene_b: r.gene_b, site_b: r.site_b, motif_b: r.motif_b,
      direction_b: _subDirWord(r.direction_b), support_b: r.support_b,
    }));
  const mouseUnmeasuredRows = inCtx
    .filter(r => r.partition === "mouse_only_unmeasured")
    .map(r => ({
      gene_b: r.gene_b, site_b: r.site_b, motif_b: r.motif_b,
      direction_b: _subDirWord(r.direction_b), support_b: r.support_b,
    }));

  _subRenderSection("sub-pairs-shared", _SUB_SHARED_COLS, sharedRows, `Shared same-site — ${ctx}`);
  _subRenderSection("sub-pairs-diffsite", _SUB_DIFFSITE_COLS, diffSiteRows, `Shared diff-residue — ${ctx}`);
  _subRenderSection("sub-pairs-human-engaged", _SUB_HUMAN_COLS, humanEngagedRows, `Human-only engaged — ${ctx}`);
  _subRenderSection("sub-pairs-human-unmeasured", _SUB_HUMAN_COLS, humanUnmeasuredRows, `Human-only unmeasured — ${ctx}`);
  _subRenderSection("sub-pairs-mouse-engaged", _SUB_MOUSE_COLS, mouseEngagedRows, `Mouse-only engaged — ${ctx}`);
  _subRenderSection("sub-pairs-mouse-unmeasured", _SUB_MOUSE_COLS, mouseUnmeasuredRows, `Mouse-only unmeasured — ${ctx}`);
}

// signed disease direction (+1/-1/0) → word; blank when absent.
function _subDirWord(v) {
  if (v == null || v === "") return "";
  const n = Number(v);
  if (!Number.isFinite(n)) return "";
  return n > 0 ? "up" : (n < 0 ? "down" : "flat");
}

function renderSubstrateDetail(name) {
  const el = document.getElementById("sub-detail");
  if (!el) return;
  if (_subRows === null) _subBuildRows();
  const row = _subRows.find(r => r.name === name);
  if (!row) { el.innerHTML = `<div class="muted" style="padding:12px;">Select a kinase.</div>`; return; }
  const SC = PAYLOAD.substrate_compare;
  const ctx = (el.dataset.ctx && row.perCtx[el.dataset.ctx]) ? el.dataset.ctx
              : (row.peakCtx || SC.contexts[0]);
  el.dataset.ctx = ctx;
  const c = row.perCtx[ctx] || {};

  const opts = SC.contexts.map(x =>
    `<option value="${x}"${x === ctx ? " selected" : ""}>${_subEsc(x)}</option>`).join("");
  const tab = _subDetailTab;

  el.innerHTML = `
    <div class="sub-detail-head">
      <h3>${_subEsc(row.name)} <span class="muted" style="font-weight:400;">${_subEsc(row.family)}</span> ${_subTierPill(row.tier)}</h3>
      <div class="sub-detail-dir">
        <span>Human ${_subDirGlyph(row.humanDir)}</span>
        <span>Mouse ${_subDirGlyph(row.mouseDir)}</span>
        <span class="muted">peak ${row.peakShared == null ? "—" : row.peakShared.toLocaleString()} shared substrates @ ${_subEsc(row.peakCtx || "—")}</span>
      </div>
    </div>
    <div class="sub-detail-tabs" role="tablist">
      <button type="button" class="sub-tab${tab === "overlap" ? " active" : ""}" data-subtab="overlap"
        role="tab" aria-selected="${tab === "overlap"}">Overlap (8 contexts)</button>
      <button type="button" class="sub-tab${tab === "pairs" ? " active" : ""}" data-subtab="pairs"
        role="tab" aria-selected="${tab === "pairs"}">Substrate motifs</button>
    </div>
    <div class="sub-tab-body">
      ${tab === "overlap" ? _subOverlapTable(row, SC) : _subPairsPanel(SC, ctx, opts, c)}
    </div>
  `;
  if (tab === "pairs") _subFillPairTable(name, ctx);
}

// ---- wiring ----------------------------------------------------------------
function wireSubstrateCompare() {
  const wrap = document.getElementById("sub-table-wrap");
  if (wrap) {
    wrap.addEventListener("click", (e) => {
      const th = e.target.closest("th[data-sort]");
      if (th) {
        const col = th.getAttribute("data-sort");
        if (_subFilter.sortCol === col) _subFilter.sortAsc = !_subFilter.sortAsc;
        else { _subFilter.sortCol = col; _subFilter.sortAsc = false; }
        renderSubstrateCompare();
        return;
      }
      const tr = e.target.closest("tr.sub-row");
      if (tr) {
        Store.dispatch({ type: "SET_SELECTION", key: "substrate", value: tr.getAttribute("data-name") });
      }
    });
    wrap.addEventListener("keydown", (e) => {
      if (e.key !== "Enter" && e.key !== " ") return;
      const th = e.target.closest("th[data-sort]");
      if (th) { e.preventDefault(); th.click(); }
    });
  }
  const search = document.getElementById("sub-search");
  if (search) search.addEventListener("input", () => {
    _subFilter.search = search.value; renderSubstrateCompare();
  });
  const minSharedEl = document.getElementById("sub-min-shared");
  if (minSharedEl) minSharedEl.addEventListener("change", () => {
    _subFilter.minShared = Number(minSharedEl.value) || 0; renderSubstrateCompare();
  });
  const reset = document.getElementById("sub-reset");
  if (reset) reset.addEventListener("click", () => {
    _subFilter.search = ""; _subFilter.minShared = 0;
    _subFilter.sortCol = "peakShared"; _subFilter.sortAsc = false;
    if (search) search.value = "";
    if (minSharedEl) minSharedEl.value = "0";
    Store.dispatch({ type: "SET_SELECTION", key: "substrate", value: null });
    renderSubstrateCompare();
  });
  const exp = document.getElementById("sub-export");
  if (exp) exp.addEventListener("click", exportSubstrateCsv);

  // Delegated: detail-pane sub-tab buttons + context <select> (both re-rendered).
  const detail = document.getElementById("sub-detail");
  if (detail) {
    detail.addEventListener("click", (e) => {
      const tabBtn = e.target.closest(".sub-tab");
      if (!tabBtn) return;
      const next = tabBtn.getAttribute("data-subtab");
      if (next === _subDetailTab) return;
      _subDetailTab = next;
      const name = Store.state.selection.substrate;
      if (name != null) renderSubstrateDetail(name);
    });
    detail.addEventListener("change", (e) => {
      if (e.target && e.target.id === "sub-ctx-select") {
        detail.dataset.ctx = e.target.value;
        const name = Store.state.selection.substrate;
        if (name != null) renderSubstrateDetail(name);
      }
    });
  }
}

function exportSubstrateCsv() {
  const headers = ["Kinase", "Family", "Human_dir", "Mouse_dir",
                   "peak_shared_substrates", "peak_context",
                   "direction_match", "shared_sites", "peak_overlap_frac"];
  const rows = _subVisible.map(r => ({
    Kinase: r.name, Family: r.family,
    Human_dir: r.humanDir == null ? "" : (r.humanDir > 0 ? "up" : r.humanDir < 0 ? "down" : "flat"),
    Mouse_dir: r.mouseDir == null ? "" : (r.mouseDir > 0 ? "up" : r.mouseDir < 0 ? "down" : "flat"),
    peak_shared_substrates: r.peakShared == null ? "" : r.peakShared,
    peak_context: r.peakCtx || "",
    direction_match: r.dirAgree == null ? "" : r.dirAgree.toFixed(4),
    shared_sites: r.dirAgreeSites || 0,
    peak_overlap_frac: r.peakOverlap == null ? "" : r.peakOverlap.toFixed(4),
  }));
  const keys = ["Kinase", "Family", "Human_dir", "Mouse_dir",
                "peak_shared_substrates", "peak_context",
                "direction_match", "shared_sites", "peak_overlap_frac"];
  csvDownload(csvSerialize(headers, keys, rows), exportFilename(null, "early_ad_kinases"));
}

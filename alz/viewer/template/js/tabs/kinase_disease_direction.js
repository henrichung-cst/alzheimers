"use strict";
// ---------------------------------------------------------------------------
// Disease Direction tab — C3
//
// Two stacked panels:
//   Top:    Kinase directional ranking — one row per kinase; signed peak-NES
//           for each genotype (App / Tau / ApTt); trajectory badge; n_sig.
//           Default sort: songOverallPeak().nes (signed, largest-|NES| on top).
//           Header-click cycles asc/desc on a chosen column.
//   Bottom: Candidate-biomarker table — row entity = substrate gene symbol.
//           Columns: gene, Secreted (human, HPA), LFC (top_celltype_1_song_lfc
//           for kinase-row genes; omitted where unavailable), h_spec.
//           Gene-list textarea → filter table; report unmatched count.
//           Secondary: "Open matched in Kinase Explorer" → setWhitelist.
// ---------------------------------------------------------------------------

// ---- Module state ----------------------------------------------------------
let _ddKinaseRows   = null;   // built once from payload; reset on payload reload
let _ddBioRows      = null;   // built once from payload; reset on payload reload
let _ddState = {
  // Top panel
  sortKey:  "overall",        // "overall" | "App" | "Tau" | "ApTt" | "n_sig_App" | "n_sig_Tau" | "n_sig_ApTt"
  sortDir:  1,                // +1 = desc (largest on top), -1 = asc

  // Bottom panel
  geneFilter: "",             // raw textarea content
};

// ---- Payload builders ------------------------------------------------------

function _ddBuildKinaseRows() {
  const K = ViewerPayload.kinases();
  const rows = [];
  for (let i = 0; i < K.id.length; i++) {
    rows.push({
      id:              K.id[i],
      name:            K.name[i],
      gene_symbol:     (K.gene_symbol    && K.gene_symbol[i])    || "",
      residue_type:    (K.residue_type   && K.residue_type[i])   || "ST",
      peak_NES_App:    (K.peak_NES_App   && K.peak_NES_App[i]   != null) ? K.peak_NES_App[i]   : null,
      peak_NES_Tau:    (K.peak_NES_Tau   && K.peak_NES_Tau[i]   != null) ? K.peak_NES_Tau[i]   : null,
      peak_NES_ApTt:   (K.peak_NES_ApTt  && K.peak_NES_ApTt[i]  != null) ? K.peak_NES_ApTt[i]  : null,
      peak_contrast_App:  (K.peak_contrast_App  && K.peak_contrast_App[i])  || "",
      peak_contrast_Tau:  (K.peak_contrast_Tau  && K.peak_contrast_Tau[i])  || "",
      peak_contrast_ApTt: (K.peak_contrast_ApTt && K.peak_contrast_ApTt[i]) || "",
      n_sig_App:       (K.n_sig_App  && K.n_sig_App[i])  || 0,
      n_sig_Tau:       (K.n_sig_Tau  && K.n_sig_Tau[i])  || 0,
      n_sig_ApTt:      (K.n_sig_ApTt && K.n_sig_ApTt[i]) || 0,
      trajectory_App:  (K.trajectory_App  && K.trajectory_App[i])  || "",
      trajectory_Tau:  (K.trajectory_Tau  && K.trajectory_Tau[i])  || "",
      trajectory_ApTt: (K.trajectory_ApTt && K.trajectory_ApTt[i]) || "",
      // LFC for the biomarker panel (kinase-level scalar)
      top_celltype_1_song_lfc: (K.top_celltype_1_song_lfc && K.top_celltype_1_song_lfc[i] != null)
        ? K.top_celltype_1_song_lfc[i] : null,
      // Secretome annotation (present when secretome was ingested at build time)
      secretome_location: (K.secretome_location && K.secretome_location[i]) || "",
    });
  }
  return rows;
}

// Build biomarker rows: one row per unique gene symbol across kinase payload.
// LFC = top_celltype_1_song_lfc for kinase genes; blank otherwise.
function _ddBuildBioRows() {
  const K = ViewerPayload.kinases();
  // Gene → first kinase id seen (for setWhitelist), LFC, secretome_location, h_spec
  const byGene = new Map();
  const AI = PAYLOAD.attribution_index || {};
  // h_spec: seaad_location_score max per kinase_id
  const seaadMax = new Map();
  if (AI.kinase_id && AI.seaad_location_score) {
    for (let j = 0; j < AI.kinase_id.length; j++) {
      const kid = AI.kinase_id[j];
      const s = AI.seaad_location_score ? AI.seaad_location_score[j] : null;
      if (s != null && isFinite(s)) {
        const cur = seaadMax.get(kid);
        if (cur == null || s > cur) seaadMax.set(kid, s);
      }
    }
  }
  for (let i = 0; i < K.id.length; i++) {
    const gene = (K.gene_symbol && K.gene_symbol[i]) || "";
    if (!gene) continue;
    const geneUp = gene.toUpperCase();
    if (byGene.has(geneUp)) continue;
    byGene.set(geneUp, {
      gene:               gene,
      kinase_id:          K.id[i],
      kinase_name:        K.name[i],
      secretome_location: (K.secretome_location && K.secretome_location[i]) || "",
      lfc:                (K.top_celltype_1_song_lfc && K.top_celltype_1_song_lfc[i] != null)
                            ? K.top_celltype_1_song_lfc[i] : null,
      h_spec:             seaadMax.get(K.id[i]) || null,
    });
  }
  return Array.from(byGene.values());
}

function _ddEnsureRows() {
  if (!_ddKinaseRows) _ddKinaseRows = _ddBuildKinaseRows();
  if (!_ddBioRows)    _ddBioRows    = _ddBuildBioRows();
}

// ---- Sort helpers ----------------------------------------------------------

function _ddSortKinase(rows, key, dir) {
  return rows.slice().sort((a, b) => {
    let av, bv;
    if (key === "App")  { av = a.peak_NES_App;  bv = b.peak_NES_App; }
    else if (key === "Tau")   { av = a.peak_NES_Tau;  bv = b.peak_NES_Tau; }
    else if (key === "ApTt")  { av = a.peak_NES_ApTt; bv = b.peak_NES_ApTt; }
    else if (key === "n_sig_App")  { av = a.n_sig_App;  bv = b.n_sig_App; }
    else if (key === "n_sig_Tau")  { av = a.n_sig_Tau;  bv = b.n_sig_Tau; }
    else if (key === "n_sig_ApTt") { av = a.n_sig_ApTt; bv = b.n_sig_ApTt; }
    else {
      // Default: songOverallPeak — signed NES of max-|NES| genotype
      const pa = songOverallPeak(a), pb = songOverallPeak(b);
      av = pa.nes; bv = pb.nes;
    }
    return numCmp(av, bv, dir);
  });
}

function _ddSortBio(rows, key, dir) {
  return rows.slice().sort((a, b) => {
    let av, bv;
    if (key === "lfc")    { av = a.lfc;    bv = b.lfc; }
    else if (key === "h_spec") { av = a.h_spec; bv = b.h_spec; }
    else if (key === "gene")   { return dir > 0 ? a.gene.localeCompare(b.gene) : b.gene.localeCompare(a.gene); }
    else                  { av = a.lfc;    bv = b.lfc; }
    return numCmp(av, bv, dir);
  });
}

// ---- Trajectory badge ------------------------------------------------------

// Returns badge HTML for a per-genotype trajectory string.
// "early" = sig at 2mo only (from C1's _classify_trajectory).
function _ddTrajBadge(traj) {
  if (!traj) return "";
  const lc = traj.toLowerCase();
  let cls = "dd-traj";
  let label = traj;
  if (lc === "early")    { cls += " dd-traj-early";    label = "early"; }
  else if (lc === "sustained") { cls += " dd-traj-sustained"; label = "sust"; }
  else if (lc === "peak")      { cls += " dd-traj-peak";      label = "peak"; }
  else if (lc === "trough")    { cls += " dd-traj-trough";    label = "trgh"; }
  return `<span class="${cls}" title="${_escapeHtml(traj)}">${_escapeHtml(label)}</span>`;
}

// ---- Signed NES cell (reuse style from _kxMedNesCell) ----------------------

function _ddNesCell(val) {
  if (val == null || !isFinite(val)) return `<td class="dd-nes-num muted">—</td>`;
  const col = val >= 0 ? "#c53030" : "#2b6cb0";
  return `<td class="dd-nes-num" style="color:${col};" title="signed peak NES">${val >= 0 ? "+" : ""}${val.toFixed(2)}</td>`;
}

// ---- Column header sort toggle ---------------------------------------------

function _ddHeaderClick(key) {
  const s = _ddState;
  if (s.sortKey === key) {
    s.sortDir = s.sortDir > 0 ? -1 : 1;
  } else {
    s.sortKey = key;
    s.sortDir = 1;
  }
  _ddRenderKinaseTable();
}

function _ddBioHeaderClick(key) {
  const s = _ddState;
  if (s.sortKey === "bio_" + key) {
    s.sortDir = s.sortDir > 0 ? -1 : 1;
  } else {
    s.sortKey = "bio_" + key;
    s.sortDir = 1;
  }
  _ddRenderBioTable();
}

// ---- Kinase table rendering -----------------------------------------------

function _ddRenderKinaseTable() {
  _ddEnsureRows();
  const s = _ddState;
  const sorted = _ddSortKinase(_ddKinaseRows, s.sortKey, s.sortDir);

  const dir = s.sortDir;
  function thSort(key, label, title) {
    const active = s.sortKey === key;
    const arrow = active ? (dir > 0 ? " ▾" : " ▴") : "";
    return `<th data-dd-sort="${key}" class="dd-th-sort${active ? " dd-th-active" : ""}" title="${_escapeHtml(title)}">${_escapeHtml(label)}${arrow}</th>`;
  }

  let thead = `<thead><tr>
    ${thSort("overall",    "Kinase",        "Default sort: max |NES| across genotypes (signed). Click to sort.")}
    <th>Gene</th>
    ${thSort("App",        "MouseC1 App",   "Signed peak NES for App genotype. Click to sort.")}
    <th title="Trajectory badge for App genotype (early=2mo only, peak, sustained, …)">Traj App</th>
    ${thSort("n_sig_App",  "n_sig App",     "Count of significant App contrasts at active FDR. Click to sort.")}
    ${thSort("Tau",        "MouseC1 Tau",   "Signed peak NES for Tau genotype. Click to sort.")}
    <th title="Trajectory badge for Tau genotype">Traj Tau</th>
    ${thSort("n_sig_Tau",  "n_sig Tau",     "Count of significant Tau contrasts. Click to sort.")}
    ${thSort("ApTt",       "MouseC1 ApTt",  "Signed peak NES for ApTt (double knock-in) genotype. Click to sort.")}
    <th title="Trajectory badge for ApTt genotype">Traj ApTt</th>
    ${thSort("n_sig_ApTt", "n_sig ApTt",    "Count of significant ApTt contrasts. Click to sort.")}
  </tr></thead>`;

  let html = `<table class="data-table dd-table" id="dd-kinase-table">${thead}<tbody>`;
  for (const r of sorted) {
    html += `<tr>
      <td class="dd-name"><span class="kinase-label">${_escapeHtml(r.name)}</span></td>
      <td class="dd-gene">${_escapeHtml(r.gene_symbol)}</td>
      ${_ddNesCell(r.peak_NES_App)}
      <td class="dd-traj-cell">${_ddTrajBadge(r.trajectory_App)}</td>
      <td class="dd-nsig">${r.n_sig_App}</td>
      ${_ddNesCell(r.peak_NES_Tau)}
      <td class="dd-traj-cell">${_ddTrajBadge(r.trajectory_Tau)}</td>
      <td class="dd-nsig">${r.n_sig_Tau}</td>
      ${_ddNesCell(r.peak_NES_ApTt)}
      <td class="dd-traj-cell">${_ddTrajBadge(r.trajectory_ApTt)}</td>
      <td class="dd-nsig">${r.n_sig_ApTt}</td>
    </tr>`;
  }
  html += `</tbody></table>`;

  const wrap = document.getElementById("dd-kinase-wrap");
  if (wrap) {
    wrap.innerHTML = html;
    // Wire sort headers
    wrap.querySelectorAll("th[data-dd-sort]").forEach(th => {
      th.style.cursor = "pointer";
      th.addEventListener("click", () => _ddHeaderClick(th.dataset.ddSort));
    });
  }
}

// ---- Biomarker table rendering --------------------------------------------

function _ddParsedGenes() {
  const raw = _ddState.geneFilter || "";
  if (!raw.trim()) return null;
  return raw.split(/[\n,;]+/).map(s => s.trim().toUpperCase()).filter(Boolean);
}

function _ddRenderBioTable() {
  _ddEnsureRows();
  const s = _ddState;

  const filterGenes = _ddParsedGenes();
  let rows = _ddBioRows;
  let unmatchedCount = 0;
  let matchedKinaseIds = [];

  if (filterGenes && filterGenes.length > 0) {
    const matchedSet = new Set();
    rows = rows.filter(r => {
      if (filterGenes.includes(r.gene.toUpperCase())) {
        matchedSet.add(r.gene.toUpperCase());
        return true;
      }
      return false;
    });
    unmatchedCount = filterGenes.filter(g => !matchedSet.has(g)).length;
    matchedKinaseIds = rows.map(r => r.kinase_id);
  }

  // Sort by bio key if active, else by lfc desc by default
  const sortKey = (typeof s.sortKey === "string" && s.sortKey.startsWith("bio_"))
    ? s.sortKey.replace("bio_", "") : "lfc";
  const sorted = _ddSortBio(rows, sortKey, s.sortDir);

  const dir = s.sortDir;
  function thSort(key, label, title) {
    const stateKey = "bio_" + key;
    const active = s.sortKey === stateKey;
    const arrow = active ? (dir > 0 ? " ▾" : " ▴") : "";
    return `<th data-dd-biosort="${key}" class="dd-th-sort${active ? " dd-th-active" : ""}" title="${_escapeHtml(title)}">${_escapeHtml(label)}${arrow}</th>`;
  }

  let thead = `<thead><tr>
    ${thSort("gene",   "Gene",                     "Gene symbol. Click to sort alphabetically.")}
    <th title="Human Protein Atlas secretome location category. Blank where HPA has no entry for this gene.">Secreted (human, HPA)</th>
    ${thSort("lfc",    "LFC (top cell type)",       "Kinase-level log2 fold change in the top attributed cell type (top_celltype_1_song_lfc). Blank where unavailable.")}
    ${thSort("h_spec", "SEA-AD expr",               "Human SEA-AD MTG expression location score (h_spec). Higher = more cell-type specific expression. Blank where unavailable.")}
  </tr></thead>`;

  let html = `<table class="data-table dd-table" id="dd-bio-table">${thead}<tbody>`;
  for (const r of sorted) {
    const lfcCell = r.lfc != null && isFinite(r.lfc)
      ? `<td class="dd-nes-num" style="color:${r.lfc >= 0 ? "#c53030" : "#2b6cb0"};">${r.lfc >= 0 ? "+" : ""}${r.lfc.toFixed(3)}</td>`
      : `<td class="muted dd-nes-num"></td>`;
    const hspecCell = r.h_spec != null && isFinite(r.h_spec)
      ? `<td class="dd-num">${r.h_spec.toFixed(2)}</td>`
      : `<td class="muted dd-num"></td>`;
    html += `<tr>
      <td class="dd-gene">${_escapeHtml(r.gene)}</td>
      <td class="dd-secretome">${_escapeHtml(r.secretome_location)}</td>
      ${lfcCell}
      ${hspecCell}
    </tr>`;
  }
  html += `</tbody></table>`;

  const wrap = document.getElementById("dd-bio-wrap");
  if (wrap) {
    wrap.innerHTML = html;
    wrap.querySelectorAll("th[data-dd-biosort]").forEach(th => {
      th.style.cursor = "pointer";
      th.addEventListener("click", () => _ddBioHeaderClick(th.dataset.ddBiosort));
    });
  }

  // Unmatched count feedback
  const unEl = document.getElementById("dd-bio-unmatched");
  if (unEl) {
    if (filterGenes && filterGenes.length > 0 && unmatchedCount > 0) {
      unEl.textContent = `${unmatchedCount} symbol${unmatchedCount === 1 ? "" : "s"} not found`;
      unEl.hidden = false;
    } else {
      unEl.textContent = "";
      unEl.hidden = true;
    }
  }

  // Count label
  const countEl = document.getElementById("dd-bio-count");
  if (countEl) {
    const total = _ddBioRows ? _ddBioRows.length : 0;
    const shown = sorted.length;
    countEl.textContent = filterGenes && filterGenes.length > 0
      ? `${shown} of ${total} genes`
      : `${total} genes`;
  }

  // Wire "Open in Kinase Explorer" button
  const openBtn = document.getElementById("dd-open-explorer");
  if (openBtn) {
    if (matchedKinaseIds.length > 0) {
      openBtn.disabled = false;
      openBtn._ddIds = matchedKinaseIds;
    } else {
      openBtn.disabled = true;
      openBtn._ddIds = [];
    }
  }
}

// ---- Gene filter textarea --------------------------------------------------

function _ddOnGeneFilter() {
  const el = document.getElementById("dd-gene-filter");
  _ddState.geneFilter = el ? el.value : "";
  _ddRenderBioTable();
}

// ---- "Open matched in Kinase Explorer" ------------------------------------

function _ddOpenInExplorer(kinaseIds) {
  if (!kinaseIds || !kinaseIds.length) return;
  if (typeof KinaseFilter === "undefined" || !KinaseFilter.setWhitelist) {
    console.warn("KinaseFilter.setWhitelist not available");
    return;
  }
  KinaseFilter.setWhitelist(kinaseIds.slice(), "Disease Direction matched genes");
  KinaseFilter.setWhitelistStack(false);
  Store.dispatch({type: "SET_VIEW", key: "activeTab", value: "kinase"});
  if (typeof _syncKinaseFilterUI === "function") setTimeout(_syncKinaseFilterUI, 0);
}

// ---- Wire + Render (TAB_MANIFEST hooks) -----------------------------------

function wireDiseasedirection() {
  const textarea = document.getElementById("dd-gene-filter");
  if (textarea) textarea.addEventListener("input", _ddOnGeneFilter);

  const clearBtn = document.getElementById("dd-gene-clear");
  if (clearBtn) {
    clearBtn.addEventListener("click", () => {
      const el = document.getElementById("dd-gene-filter");
      if (el) el.value = "";
      _ddState.geneFilter = "";
      _ddRenderBioTable();
    });
  }

  const openBtn = document.getElementById("dd-open-explorer");
  if (openBtn) {
    openBtn.addEventListener("click", () => _ddOpenInExplorer(openBtn._ddIds || []));
  }
}

function renderDiseasedirection() {
  _ddKinaseRows = null;  // rebuild on every render (payload may have reloaded)
  _ddBioRows    = null;
  // Reset sort to default on full render
  _ddState.sortKey = "overall";
  _ddState.sortDir = 1;
  _ddRenderKinaseTable();
  _ddRenderBioTable();
}

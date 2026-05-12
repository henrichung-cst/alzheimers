// ---------------------------------------------------------------------------
// Incytr Pathways tab — table over a single (sender, receiver) shard or the
// union of multiple selected pairs. Filter UI mirrors the Kinase tab:
// multiselect popovers for Sender / Receiver / Disease / Timepoint; ordinal
// <select>s for the 4 label columns; <select> tier preset + 3 numeric inputs
// for metric thresholds; Reset button + live count.
//
// Filter state lives in IncytrFilter (shared with the heatmap tab). The
// heatmap click handler writes pair + senderIn + receiverIn + disease +
// timepoint into IncytrFilter and switches tabs.
// ---------------------------------------------------------------------------

const _IP_LABEL_COLS = ["Ligand.label", "Receptor.label", "EM.label", "Target.label"];
const _IP_LABEL_KEYS = {
  "Ligand.label":   "ligandLabel",
  "Receptor.label": "receptorLabel",
  "EM.label":       "emLabel",
  "Target.label":   "targetLabel",
};
const _IP_DISEASES = ["App", "Tau", "ApTt"];
const _IP_TIMEPOINTS = ["2mo", "4mo", "6mo"];
const _IP_ROW_CAP = 1000;

const _ipRuntime = {
  rows:        null,         // concatenated rows from currently-loaded shards
  loadedKey:   null,         // sig string of pairs currently loaded
  loading:     false,
  loadError:   null,
};

function _ipBlock() {
  return (typeof PAYLOAD !== "undefined" && PAYLOAD.incytr_pathways) || null;
}

function _ipPairsInScope(block) {
  // Returns [{sender, receiver}] to load. Honors the multiselect filters; if
  // both are empty, defaults to "no pairs" (user must pick) so we don't hammer
  // 349 fetches by accident.
  const f = IncytrFilter.get();
  if (f.pair) return [f.pair];
  const sIn = new Set(f.senderIn || []);
  const rIn = new Set(f.receiverIn || []);
  if (!sIn.size && !rIn.size) return [];
  const out = [];
  for (const [s, r] of block.slice_index.present) {
    if (sIn.size && !sIn.has(s)) continue;
    if (rIn.size && !rIn.has(r)) continue;
    out.push({ sender: s, receiver: r });
  }
  return out;
}

function _ipScopeSig(pairs) {
  return pairs.map(p => p.sender + "||" + p.receiver).sort().join(";");
}

// ---- toolbar builders ----

function _ipMountMultiselect(hostId, label, options, key) {
  const host = document.getElementById(hostId);
  if (!host) return;
  mountMultiselect(host, {
    label, options,
    current: IncytrFilter.get(key) || [],
    onChange: (next) => {
      // Picking sender/receiver clears any pinned pair from the heatmap.
      const patch = { [key]: next };
      if (key === "senderIn" || key === "receiverIn") patch.pair = null;
      IncytrFilter.set(patch);
      _ipMountMultiselect(hostId, label, options, key);   // re-render badge
      _ipInvalidateScope();
      _ipEnsureShards();
    },
  });
}

function _ipSyncControls(block) {
  const f = IncytrFilter.get();

  _ipMountMultiselect("ip-ms-sender",   "Sender",    block.senders,   "senderIn");
  _ipMountMultiselect("ip-ms-receiver", "Receiver",  block.receivers, "receiverIn");
  _ipMountMultiselect("ip-ms-disease",  "Disease",   _IP_DISEASES,    "disease");
  _ipMountMultiselect("ip-ms-time",     "Timepoint", _IP_TIMEPOINTS,  "timepoint");

  // Label ordinal selects.
  for (const col of _IP_LABEL_COLS) {
    const sel = document.getElementById("ip-lbl-" + col.replace(".", "-"));
    if (sel) sel.value = f[_IP_LABEL_KEYS[col]] || "";
  }
  // Tier preset.
  const tierSel = document.getElementById("ip-tier");
  if (tierSel) tierSel.value = f.tier || "paper";
  // Numeric sliders.
  const set = (id, v) => {
    const el = document.getElementById(id);
    if (el) el.value = (v == null || !isFinite(v)) ? "" : v;
  };
  set("ip-slider-p",   f.sliderP);
  set("ip-slider-pds", f.sliderPds);
  set("ip-slider-sp",  f.sliderSp);
}

function _ipInvalidateScope() {
  _ipRuntime.rows = null;
  _ipRuntime.loadedKey = null;
}

// ---- shard loading ----

async function _ipEnsureShards() {
  const block = _ipBlock();
  if (!block) return;
  const pairs = _ipPairsInScope(block);
  const sig = _ipScopeSig(pairs);
  if (_ipRuntime.loadedKey === sig) {
    _ipRenderTable();
    return;
  }
  _ipRuntime.loading = true;
  _ipRuntime.loadError = null;
  _ipRuntime.rows = null;
  _ipRenderTable();
  try {
    const all = [];
    for (const p of pairs) {
      const rows = await SliceCache.loadIncytrShard(p.sender, p.receiver);
      // Stamp sender/receiver onto each row so multi-pair queries still
      // identify the originating cell-type pair in the table.
      for (const r of rows) { r._sender = p.sender; r._receiver = p.receiver; }
      all.push(...rows);
    }
    // Resolve race: only commit if the scope hasn't changed mid-fetch.
    const newSig = _ipScopeSig(_ipPairsInScope(block));
    if (newSig !== sig) return;
    _ipRuntime.rows = all;
    _ipRuntime.loadedKey = sig;
  } catch (e) {
    _ipRuntime.loadError = String(e.message || e);
    console.error("incytr shards load failed", e);
  } finally {
    _ipRuntime.loading = false;
    _ipRenderTable();
  }
}

// ---- row filtering + sort ----

function _ipFilterRows() {
  if (!_ipRuntime.rows) return [];
  const f = IncytrFilter.get();
  const diseaseSet   = new Set(f.disease   || []);
  const timeSet      = new Set(f.timepoint || []);
  const out = [];
  for (const r of _ipRuntime.rows) {
    if (diseaseSet.size || timeSet.size) {
      const [d, t] = (r.contrast || "").split("_");
      if (diseaseSet.size && !diseaseSet.has(d)) continue;
      if (timeSet.size    && !timeSet.has(t))    continue;
    }
    if (f.sliderP   != null && !(r.pvalue       <  f.sliderP))   continue;
    if (f.sliderPds != null && !(Math.abs(r.PDS) >= f.sliderPds)) continue;
    if (f.sliderSp  != null && !(r.sigprob_max  >= f.sliderSp))  continue;
    let ok = true;
    for (const col of _IP_LABEL_COLS) {
      const want = f[_IP_LABEL_KEYS[col]];
      if (want && r[col] !== want) { ok = false; break; }
    }
    if (ok) out.push(r);
  }
  const key = f.sortKey, dir = f.sortDir;
  const numericKeys = new Set(["pvalue", "PDS", "log2FC", "sigprob_max"]);
  out.sort((a, b) => {
    const av = a[key], bv = b[key];
    if (av == null && bv == null) return 0;
    if (av == null) return 1;
    if (bv == null) return -1;
    if (numericKeys.has(key)) return dir * (av - bv);
    return dir * (String(av).localeCompare(String(bv)));
  });
  return out;
}

function _ipFmtNum(v, digits) {
  if (v == null || !isFinite(v)) return "—";
  if (digits === "sci" && Math.abs(v) < 0.01 && v !== 0) return v.toExponential(2);
  return Number(v).toFixed(digits == null ? 3 : digits);
}

function _ipRenderTable() {
  const countEl = document.getElementById("ip-count");
  const wrap = document.getElementById("ip-table-wrap");
  const block = _ipBlock();
  if (!wrap || !countEl || !block) return;

  const pairs = _ipPairsInScope(block);
  if (!pairs.length) {
    countEl.textContent = "Select sender(s) or receiver(s) — or click a heatmap cell.";
    wrap.innerHTML = '<div class="muted" style="padding:16px;">No pair selected.</div>';
    return;
  }
  if (_ipRuntime.loading) {
    countEl.textContent = `Loading ${pairs.length} shard${pairs.length === 1 ? "" : "s"}…`;
    wrap.innerHTML = '<div class="muted" style="padding:16px;">Fetching shards…</div>';
    return;
  }
  if (_ipRuntime.loadError) {
    countEl.textContent = "Shard load failed.";
    wrap.innerHTML = `<div class="muted" style="padding:16px;">${_escapeHtml(_ipRuntime.loadError)}</div>`;
    return;
  }
  if (!_ipRuntime.rows) {
    countEl.textContent = "";
    wrap.innerHTML = "";
    return;
  }
  if (!_ipRuntime.rows.length) {
    countEl.textContent = "No rows in the selected shard(s).";
    wrap.innerHTML = '<div class="muted" style="padding:16px;">Empty (likely an empty-DEG cell type).</div>';
    return;
  }
  const filtered = _ipFilterRows();
  const total = _ipRuntime.rows.length;
  const shown = Math.min(filtered.length, _IP_ROW_CAP);
  const f = IncytrFilter.get();
  countEl.textContent =
    `${filtered.length.toLocaleString()} rows pass filters `
    + `(of ${total.toLocaleString()} loaded from ${pairs.length} pair${pairs.length === 1 ? "" : "s"}) · `
    + `tier preset: ${f.tier}.`
    + (filtered.length > _IP_ROW_CAP
        ? ` Showing top ${shown.toLocaleString()} by ${f.sortKey}.`
        : "");

  const cols = [
    { key: "_sender",       label: "Sender" },
    { key: "_receiver",     label: "Receiver" },
    { key: "Path",          label: "Path" },
    { key: "Ligand",        label: "Ligand" },
    { key: "Receptor",      label: "Receptor" },
    { key: "EM",            label: "EM" },
    { key: "Target",        label: "Target" },
    { key: "Ligand.label",  label: "L.lbl" },
    { key: "Receptor.label",label: "R.lbl" },
    { key: "EM.label",      label: "EM.lbl" },
    { key: "Target.label",  label: "T.lbl" },
    { key: "contrast",      label: "contrast" },
    { key: "pvalue",        label: "pvalue", numeric: true, digits: "sci" },
    { key: "PDS",           label: "PDS",    numeric: true, digits: 3 },
    { key: "log2FC",        label: "log2FC", numeric: true, digits: 3 },
    { key: "sigprob_max",   label: "sigprob", numeric: true, digits: 3 },
  ];
  const thead = cols.map(c => {
    const on = (f.sortKey === c.key);
    const arrow = on ? (f.sortDir > 0 ? " ▲" : " ▼") : "";
    return `<th data-ip-sort="${c.key}">${_escapeHtml(c.label)}${arrow}</th>`;
  }).join("");
  const visible = filtered.slice(0, _IP_ROW_CAP);
  const tbody = visible.map(r => {
    const cells = cols.map(c => {
      const v = r[c.key];
      if (c.numeric) return `<td style="text-align:right;">${_ipFmtNum(v, c.digits)}</td>`;
      return `<td>${_escapeHtml(v == null ? "" : v)}</td>`;
    }).join("");
    return `<tr>${cells}</tr>`;
  }).join("");
  wrap.innerHTML = `<div class="ke-table-wrap"><table class="data-table" id="ip-table">`
    + `<thead><tr>${thead}</tr></thead><tbody>${tbody}</tbody></table></div>`;
  const head = wrap.querySelector("#ip-table thead");
  if (head) head.addEventListener("click", ev => {
    const th = ev.target.closest("th[data-ip-sort]");
    if (!th) return;
    const k = th.dataset.ipSort;
    if (f.sortKey === k) IncytrFilter.set({ sortDir: -1 * f.sortDir });
    else IncytrFilter.set({ sortKey: k, sortDir: (k === "pvalue" ? 1 : -1) });
    _ipRenderTable();
  });
}

function wireIncytrPathways() {
  // Label-column ordinal selects.
  for (const col of _IP_LABEL_COLS) {
    const id = "ip-lbl-" + col.replace(".", "-");
    const sel = document.getElementById(id);
    if (!sel) continue;
    sel.addEventListener("change", () => {
      IncytrFilter.set({ [_IP_LABEL_KEYS[col]]: sel.value });
      _ipRenderTable();
    });
  }
  // Tier preset → seeds the sliders.
  const tierSel = document.getElementById("ip-tier");
  if (tierSel) tierSel.addEventListener("change", () => {
    IncytrFilter.applyTier(tierSel.value);
    const f = IncytrFilter.get();
    const set = (id, v) => {
      const el = document.getElementById(id);
      if (el) el.value = (v == null || !isFinite(v)) ? "" : v;
    };
    set("ip-slider-p",   f.sliderP);
    set("ip-slider-pds", f.sliderPds);
    set("ip-slider-sp",  f.sliderSp);
    _ipRenderTable();
  });
  // Numeric sliders.
  const wireSlider = (id, key) => {
    const el = document.getElementById(id);
    if (!el) return;
    el.addEventListener("input", () => {
      const raw = el.value === "" ? null : parseFloat(el.value);
      IncytrFilter.set({ [key]: (raw != null && isFinite(raw)) ? raw : null });
      _ipRenderTable();
    });
  };
  wireSlider("ip-slider-p",   "sliderP");
  wireSlider("ip-slider-pds", "sliderPds");
  wireSlider("ip-slider-sp",  "sliderSp");

  // Reset.
  const resetBtn = document.getElementById("ip-reset");
  if (resetBtn) resetBtn.addEventListener("click", () => {
    IncytrFilter.reset();
    const block = _ipBlock();
    if (block) _ipSyncControls(block);
    _ipInvalidateScope();
    _ipEnsureShards();
  });
}

function renderIncytrPathways() {
  const block = _ipBlock();
  const countEl = document.getElementById("ip-count");
  if (!block) {
    if (countEl) countEl.textContent = "No incytr_pathways block in payload.";
    return;
  }
  _ipSyncControls(block);
  _ipEnsureShards();
}

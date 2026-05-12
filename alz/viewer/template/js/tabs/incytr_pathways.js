// ---------------------------------------------------------------------------
// Incytr Pathways tab — virtualized table over a single (sender, receiver)
// shard from edge_slices/incytr_pathways/. Filter chips for sender/receiver/
// contrast/label values; metric sliders for pvalue / |PDS| / sigprob_max.
//
// Activated by heatmap click (Store.view.pendingIncytrFilter) or by directly
// selecting from the sender/receiver/contrast pickers.
// ---------------------------------------------------------------------------

const _IP_LABEL_COLS = ["Ligand.label", "Receptor.label", "EM.label", "Target.label"];
const _IP_LABEL_VALS = ["DEG", "prG"];
const _IP_TIER_DEFAULTS = {
  // Mirrors _INCYTR_HEATMAP_TIERS in alz/build_unified_viewer.py.
  all:    { p: null, pds: null, sp: null },
  p05:    { p: 0.05, pds: null, sp: null },
  paper:  { p: 0.05, pds: 0.76, sp: 0.10 },
  strict: { p: 0.01, pds: 0.76, sp: 0.10 },
};
const _IP_ROW_CAP = 1000;   // table display cap; user sees "showing N of M".

const _ipState = {
  pair: null,           // {sender, receiver}
  contrast: null,
  allContrasts: false,
  labelFilters: { "Ligand.label": null, "Receptor.label": null,
                  "EM.label": null, "Target.label": null },
  sliders: { p: 0.05, pds: 0.76, sp: 0.10 },  // start at paper tier
  sortKey: "pvalue",
  sortDir: 1,           // 1 asc, -1 desc
  rows: null,           // loaded shard (immutable)
  loading: false,
  loadError: null,
  consumedPendingSig: null,
};

function _ipBlock() {
  return (typeof PAYLOAD !== "undefined" && PAYLOAD.incytr_pathways) || null;
}

function _ipPendingSig(p) {
  return p ? `${p.sender}|${p.receiver}|${p.contrast || ""}|${p.tier || ""}` : null;
}

// ---- chip rows ----

function _ipPairChips(block, hostId, key, items, current, onPick) {
  const host = document.getElementById(hostId);
  if (!host) return;
  host.innerHTML = items.map(v => {
    const on = (v === current);
    return `<button class="chip${on ? " active" : ""}" data-ip-${key}="${_escapeHtml(v)}">`
      + _escapeHtml(v) + `</button>`;
  }).join("");
  if (host.dataset.ipWired === "1") return;
  host.dataset.ipWired = "1";
  host.addEventListener("click", ev => {
    const btn = ev.target.closest(`[data-ip-${key}]`);
    if (!btn) return;
    onPick(btn.dataset[`ip${key.charAt(0).toUpperCase()}${key.slice(1)}`]);
  });
}

function _ipLabelChips() {
  const host = document.getElementById("ip-label-chips");
  if (!host) return;
  const parts = [];
  for (const col of _IP_LABEL_COLS) {
    const cur = _ipState.labelFilters[col];
    parts.push(`<span class="muted" style="margin:0 4px 0 8px;">${_escapeHtml(col)}:</span>`);
    parts.push(`<button class="chip${cur == null ? " active" : ""}" data-ip-lbl="${_escapeHtml(col)}" data-ip-lbl-val="">any</button>`);
    for (const v of _IP_LABEL_VALS) {
      const on = (cur === v);
      parts.push(`<button class="chip${on ? " active" : ""}" data-ip-lbl="${_escapeHtml(col)}" data-ip-lbl-val="${_escapeHtml(v)}">${_escapeHtml(v)}</button>`);
    }
  }
  host.innerHTML = parts.join("");
  if (host.dataset.ipWired === "1") return;
  host.dataset.ipWired = "1";
  host.addEventListener("click", ev => {
    const btn = ev.target.closest("[data-ip-lbl]");
    if (!btn) return;
    const col = btn.dataset.ipLbl;
    const val = btn.dataset.ipLblVal || null;
    _ipState.labelFilters[col] = val;
    renderIncytrPathways();
  });
}

// ---- sliders ----

function _ipBuildSliders() {
  const host = document.getElementById("ip-slider-row");
  if (!host || host.dataset.ipWired === "1") return;
  host.dataset.ipWired = "1";
  host.addEventListener("input", ev => {
    const inp = ev.target.closest("input[data-ip-slider]");
    if (!inp) return;
    const key = inp.dataset.ipSlider;
    const raw = inp.value === "" ? null : parseFloat(inp.value);
    _ipState.sliders[key] = isFinite(raw) ? raw : null;
    _ipSyncSliderLabels();
    _ipRenderTable();
  });
  document.getElementById("ip-slider-preset").addEventListener("change", ev => {
    const t = ev.target.value;
    if (!_IP_TIER_DEFAULTS[t]) return;
    const d = _IP_TIER_DEFAULTS[t];
    _ipState.sliders = { p: d.p, pds: d.pds, sp: d.sp };
    _ipSyncSliderInputs();
    _ipRenderTable();
  });
  document.getElementById("ip-all-contrasts").addEventListener("change", ev => {
    _ipState.allContrasts = !!ev.target.checked;
    _ipRenderTable();
  });
}

function _ipSyncSliderInputs() {
  const s = _ipState.sliders;
  const ip = (id, v) => {
    const el = document.getElementById(id);
    if (el) el.value = v == null ? "" : v;
  };
  ip("ip-slider-p", s.p);
  ip("ip-slider-pds", s.pds);
  ip("ip-slider-sp", s.sp);
  _ipSyncSliderLabels();
}

function _ipSyncSliderLabels() {
  const s = _ipState.sliders;
  const fmt = (v) => (v == null || !isFinite(v)) ? "any" : v;
  const set = (id, v) => { const el = document.getElementById(id); if (el) el.textContent = v; };
  set("ip-slider-p-val",   fmt(s.p));
  set("ip-slider-pds-val", fmt(s.pds));
  set("ip-slider-sp-val",  fmt(s.sp));
}

// ---- shard loading ----

async function _ipEnsureShard() {
  const pair = _ipState.pair;
  if (!pair) return;
  const sig = pair.sender + "||" + pair.receiver;
  if (_ipState._loadedSig === sig) return;
  _ipState.loading = true;
  _ipState.loadError = null;
  _ipState.rows = null;
  _ipRenderTable();
  try {
    const rows = await SliceCache.loadIncytrShard(pair.sender, pair.receiver);
    if (_ipState.pair !== pair) return;   // user switched mid-fetch
    _ipState.rows = rows;
    _ipState._loadedSig = sig;
  } catch (e) {
    _ipState.loadError = String(e.message || e);
    console.error("incytr shard load failed", e);
  } finally {
    _ipState.loading = false;
    _ipRenderTable();
  }
}

// ---- filter + sort ----

function _ipFilterRows() {
  if (!_ipState.rows) return [];
  const s = _ipState.sliders;
  const lf = _ipState.labelFilters;
  const wantContrast = _ipState.allContrasts ? null : _ipState.contrast;
  const out = [];
  for (const r of _ipState.rows) {
    if (wantContrast && r.contrast !== wantContrast) continue;
    if (s.p   != null && !(r.pvalue       <  s.p))   continue;
    if (s.pds != null && !(Math.abs(r.PDS) >= s.pds)) continue;
    if (s.sp  != null && !(r.sigprob_max  >= s.sp))  continue;
    let ok = true;
    for (const col of _IP_LABEL_COLS) {
      if (lf[col] != null && r[col] !== lf[col]) { ok = false; break; }
    }
    if (ok) out.push(r);
  }
  const key = _ipState.sortKey, dir = _ipState.sortDir;
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
  const sub = document.getElementById("ip-subtitle");
  const wrap = document.getElementById("ip-table-wrap");
  if (!sub || !wrap) return;
  if (!_ipState.pair) {
    sub.textContent = "Click a heatmap cell in the Incytr Heatmap tab to load candidate paths for a sender × receiver pair.";
    wrap.innerHTML = "";
    return;
  }
  const block = _ipBlock();
  if (!block) {
    sub.textContent = "No incytr_pathways block in the payload.";
    wrap.innerHTML = "";
    return;
  }
  if (_ipState.loading) {
    sub.textContent = `Loading ${_ipState.pair.sender} → ${_ipState.pair.receiver} …`;
    wrap.innerHTML = '<div class="muted" style="padding:16px;">Fetching shard…</div>';
    return;
  }
  if (_ipState.loadError) {
    sub.textContent = `Failed to load ${_ipState.pair.sender} → ${_ipState.pair.receiver}.`;
    wrap.innerHTML = `<div class="muted" style="padding:16px;">${_escapeHtml(_ipState.loadError)}</div>`;
    return;
  }
  if (!_ipState.rows || !_ipState.rows.length) {
    sub.textContent = `No candidate paths for ${_ipState.pair.sender} → ${_ipState.pair.receiver}.`;
    wrap.innerHTML = '<div class="muted" style="padding:16px;">Empty shard (likely involves an empty-DEG cell type).</div>';
    return;
  }
  const filtered = _ipFilterRows();
  const total = _ipState.rows.length;
  const shown = Math.min(filtered.length, _IP_ROW_CAP);
  const scope = _ipState.allContrasts
    ? "all 9 contrasts"
    : `contrast ${_ipState.contrast || "(none)"}`;
  sub.textContent =
    `${_ipState.pair.sender} → ${_ipState.pair.receiver} · ${scope} · `
    + `${filtered.length.toLocaleString()} rows pass filters `
    + `(of ${total.toLocaleString()} in shard). `
    + (filtered.length > _IP_ROW_CAP
        ? `Showing top ${shown.toLocaleString()} by ${_ipState.sortKey}.`
        : "");

  const cols = [
    { key: "Path",          label: "Path",      align: "left",  fmt: (v) => _escapeHtml(v) },
    { key: "Ligand",        label: "Ligand" },
    { key: "Receptor",      label: "Receptor" },
    { key: "EM",            label: "EM" },
    { key: "Target",        label: "Target" },
    { key: "Ligand.label",  label: "L.lbl",   short: true },
    { key: "Receptor.label",label: "R.lbl",   short: true },
    { key: "EM.label",      label: "EM.lbl",  short: true },
    { key: "Target.label",  label: "T.lbl",   short: true },
    { key: "contrast",      label: "contrast" },
    { key: "pvalue",        label: "pvalue", numeric: true, digits: "sci" },
    { key: "PDS",           label: "PDS",    numeric: true, digits: 3 },
    { key: "log2FC",        label: "log2FC", numeric: true, digits: 3 },
    { key: "sigprob_max",   label: "sigprob", numeric: true, digits: 3 },
  ];
  const thead = cols.map(c => {
    const on = (_ipState.sortKey === c.key);
    const arrow = on ? (_ipState.sortDir > 0 ? " ▲" : " ▼") : "";
    return `<th data-ip-sort="${c.key}" title="Click to sort">${_escapeHtml(c.label)}${arrow}</th>`;
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
  const tbl = wrap.querySelector("#ip-table thead");
  if (tbl) tbl.addEventListener("click", ev => {
    const th = ev.target.closest("th[data-ip-sort]");
    if (!th) return;
    const k = th.dataset.ipSort;
    if (_ipState.sortKey === k) _ipState.sortDir *= -1;
    else { _ipState.sortKey = k; _ipState.sortDir = (["pvalue"].includes(k) ? 1 : -1); }
    _ipRenderTable();
  });
}

function wireIncytrPathways() {
  _ipBuildSliders();
}

function renderIncytrPathways() {
  const block = _ipBlock();
  if (!block) {
    const sub = document.getElementById("ip-subtitle");
    if (sub) sub.textContent = "No incytr_pathways block in payload.";
    return;
  }

  // Consume any pending filter handed off from the heatmap tab. The filter is
  // consumed once and tracked locally so the user can override picker state
  // afterward without it snapping back on the next re-render.
  const pending = Store.state.view.pendingIncytrFilter || null;
  const psig = _ipPendingSig(pending);
  if (pending && psig !== _ipState.consumedPendingSig) {
    _ipState.consumedPendingSig = psig;
    _ipState.pair = { sender: pending.sender, receiver: pending.receiver };
    _ipState.contrast = pending.contrast || _ipState.contrast || block.contrasts[0];
    if (pending.tier && _IP_TIER_DEFAULTS[pending.tier]) {
      const d = _IP_TIER_DEFAULTS[pending.tier];
      _ipState.sliders = { p: d.p, pds: d.pds, sp: d.sp };
      const sel = document.getElementById("ip-slider-preset");
      if (sel) sel.value = pending.tier;
      _ipSyncSliderInputs();
    }
    _ipState.allContrasts = false;
    _ipState._loadedSig = null;   // force reload
  }

  // Build pickers each render so they reflect current selection.
  _ipPairChips(block, "ip-sender-chips", "sender", block.senders,
    _ipState.pair && _ipState.pair.sender,
    (v) => {
      _ipState.pair = { sender: v, receiver: (_ipState.pair && _ipState.pair.receiver) || block.receivers[0] };
      _ipState._loadedSig = null;
      renderIncytrPathways();
    });
  _ipPairChips(block, "ip-receiver-chips", "receiver", block.receivers,
    _ipState.pair && _ipState.pair.receiver,
    (v) => {
      _ipState.pair = { sender: (_ipState.pair && _ipState.pair.sender) || block.senders[0], receiver: v };
      _ipState._loadedSig = null;
      renderIncytrPathways();
    });
  _ipPairChips(block, "ip-contrast-chips", "contrast", block.contrasts,
    _ipState.contrast,
    (v) => { _ipState.contrast = v; _ipRenderTable(); });

  _ipLabelChips();
  _ipSyncSliderInputs();
  const allCb = document.getElementById("ip-all-contrasts");
  if (allCb) allCb.checked = !!_ipState.allContrasts;

  if (_ipState.pair) _ipEnsureShard();
  else _ipRenderTable();
}

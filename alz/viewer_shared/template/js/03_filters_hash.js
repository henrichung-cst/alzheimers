function _checkRequirement(req) {
  const f = Store.state.filters;
  const sel = Store.state.selection;
  if (req.type === "filter") {
    if (req.notEqual !== undefined) return f[req.key] !== req.notEqual;
    if (req.equal !== undefined) return f[req.key] === req.equal;
    return f[req.key] != null;
  }
  if (req.type === "selection") {
    if (req.equal !== undefined) return sel[req.key] === req.equal;
    return sel[req.key] != null;
  }
  if (req.type === "payload") {
    return !!(typeof PAYLOAD !== "undefined" && PAYLOAD && PAYLOAD[req.key]);
  }
  return true;
}

function renderUnmetPrerequisite(panelEl, tab) {
  const manifest = TAB_MANIFEST[tab];
  if (!manifest || !manifest.requires || manifest.requires.length === 0)
    return false;
  const unmet = manifest.requires.find(r => !_checkRequirement(r));
  if (!unmet) return false;
  const card = document.createElement("div");
  card.className = "prereq-card";
  card.innerHTML =
    '<span class="prereq-icon" aria-hidden="true">&#9888;</span>' +
    '<div class="prereq-msg"></div>' +
    '<button type="button" class="prereq-action"></button>';
  card.querySelector(".prereq-msg").textContent = unmet.message;
  const btn = card.querySelector(".prereq-action");
  btn.textContent = unmet.cta;
  if (!unmet.goTo && !unmet.setSelection && !unmet.focus) btn.hidden = true;
  btn.addEventListener("click", () => {
    if (unmet.goTo) {
      Store.dispatch({type:"SET_VIEW", key:"activeTab", value:unmet.goTo});
    } else if (unmet.setSelection) {
      Store.dispatch({type:"SET_SELECTION",
        key:unmet.setSelection.key, value:unmet.setSelection.value});
    } else if (unmet.focus) {
      const el = document.getElementById(unmet.focus);
      if (el) { el.focus(); if (el.click) try { el.click(); } catch(_){} }
    }
  });
  panelEl.innerHTML = "";
  panelEl.appendChild(card);
  return true;
}

// ---------------------------------------------------------------------------
// URL-hash sync — serialize tab/filters/selection into location.hash so
// reload and back/forward restore state. Only non-default keys are emitted
// to keep the URL short. Suppresses re-broadcast while applying inbound.
// ---------------------------------------------------------------------------
function _hashDefaults() {
  return {
    t: "kinase",
    fdr: 0.25,
    k: null, b: null, ct: null,
    m: "mouse", kh: null,
    ctx: ViewerPayload.defaultContext(),
  };
}
let _hashApplying = false;

function _serializeHash() {
  const v = Store.state.view, f = Store.state.filters, s = Store.state.selection;
  const cur = {
    t: v.activeTab,
    fdr: f.fdr,
    k: s.kinase, b: s.backbone, ct: s.celltype,
    m: v.mode, kh: s.kinaseHuman,
    ctx: s.context,
  };
  const defaults = _hashDefaults();
  const parts = [];
  for (const k in cur) {
    const val = cur[k];
    if (val == null) continue;
    if (val === defaults[k]) continue;
    parts.push(encodeURIComponent(k) + "=" + encodeURIComponent(String(val)));
  }
  return parts.length ? "#" + parts.join("&") : "";
}

function pushHash() {
  if (_hashApplying) return;
  const h = _serializeHash();
  if (h === window.location.hash) return;
  // Use replaceState so each filter twiddle doesn't pollute history; only
  // tab changes create a new history entry.
  history.replaceState(null, "", h || window.location.pathname + window.location.search);
}

function applyHash() {
  const raw = (window.location.hash || "").replace(/^#/, "");
  if (!raw) return;
  const parts = raw.split("&");
  const map = {};
  parts.forEach(p => {
    const [k, v] = p.split("=").map(decodeURIComponent);
    if (k) map[k] = v;
  });
  _hashApplying = true;
  try {
    if (map.fdr != null) Store.dispatch({type:"SET_FILTER", key:"fdr", value:parseFloat(map.fdr)});
    if (map.k != null) Store.dispatch({type:"SET_SELECTION", key:"kinase", value:parseInt(map.k,10)});
    if (map.b != null) Store.dispatch({type:"SET_SELECTION", key:"backbone", value:parseInt(map.b,10)});
    if (map.ct != null) Store.dispatch({type:"SET_SELECTION", key:"celltype", value:parseInt(map.ct,10)});
    if (map.kh != null) Store.dispatch({type:"SET_SELECTION", key:"kinaseHuman", value:parseInt(map.kh,10)});
    if (map.m != null) {
      const modeOk = (typeof _modeAvailable === "function")
        ? _modeAvailable(map.m)
        : (map.m === "mouse" || (map.m === "human" && HAS_HUMAN));
      if (modeOk) Store.dispatch({type:"SET_VIEW", key:"mode", value:map.m});
    }
    const ctx = map.ctx || map.d;
    if (ctx != null) {
      Store.dispatch({type:"SET_SELECTION", key:"context", value:ctx});
    }
    if (map.t != null && TAB_MANIFEST[map.t]) {
      Store.dispatch({type:"SET_VIEW", key:"activeTab", value:map.t});
    }
  } finally {
    _hashApplying = false;
  }
}

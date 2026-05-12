// ---------------------------------------------------------------------------
// IncytrFilter — shared filter state for the Incytr Heatmap + Pathways tabs.
// Mirrors the KinaseFilter contract: localStorage-backed, get(k) / set(patch)
// / reset() / subscribe(fn). Persistence key: incytrFilter.v1.
//
// Tier defaults mirror _INCYTR_HEATMAP_TIERS in alz/build_unified_viewer.py.
// ---------------------------------------------------------------------------

window.INCYTR_TIER_DEFAULTS = {
  all:    { p: null, pds: null, sp: null },
  p05:    { p: 0.05, pds: null, sp: null },
  paper:  { p: 0.05, pds: 0.76, sp: 0.10 },
  strict: { p: 0.01, pds: 0.76, sp: 0.10 },
};

window.IncytrFilter = (function() {
  const _KEY = "incytrFilter.v1";
  const _defaults = {
    // Heatmap projection — single contrast picker, two ordinal selects.
    hmTier:         "paper",
    hmDisease:      "App",
    hmTimepoint:    "2mo",

    // Pathway table — multiselect filters (empty = any).
    pair:           null,          // {sender, receiver} or null
    disease:        [],            // [] = any
    timepoint:      [],            // [] = any
    senderIn:       [],            // [] = any
    receiverIn:     [],            // [] = any
    ligandLabel:    "",            // "" = any | "DEG" | "prG"
    receptorLabel:  "",
    emLabel:        "",
    targetLabel:    "",
    tier:           "paper",       // preset that seeds the sliders
    sliderP:        0.05,
    sliderPds:      0.76,
    sliderSp:       0.10,
    sortKey:        "pvalue",
    sortDir:        1,
  };
  const _arrKeys = new Set(["disease","timepoint","senderIn","receiverIn"]);
  let _state = Object.assign({}, _defaults);
  try {
    const saved = JSON.parse(localStorage.getItem(_KEY) || "null");
    if (saved && typeof saved === "object") {
      for (const k of Object.keys(_defaults)) {
        if (!(k in saved)) continue;
        if (_arrKeys.has(k)) _state[k] = Array.isArray(saved[k]) ? saved[k].slice() : [];
        else _state[k] = saved[k];
      }
    }
  } catch(e) {}
  const _subs = [];
  function _save() {
    try { localStorage.setItem(_KEY, JSON.stringify(_state)); } catch(e) {}
  }
  return {
    get: function(k) { return k ? _state[k] : Object.assign({}, _state); },
    set: function(patch) {
      let changed = false;
      for (const k of Object.keys(patch)) {
        const nv = patch[k];
        if (_arrKeys.has(k)) {
          const cur = _state[k] || [];
          const a = Array.isArray(nv) ? nv.slice() : [];
          if (cur.length !== a.length || cur.some((v,i) => v !== a[i])) {
            _state[k] = a; changed = true;
          }
        } else {
          // Deep-equal for object values (pair = {sender, receiver}).
          const same = (typeof nv === "object" && nv !== null && typeof _state[k] === "object" && _state[k] !== null)
            ? JSON.stringify(nv) === JSON.stringify(_state[k])
            : _state[k] === nv;
          if (!same) { _state[k] = nv; changed = true; }
        }
      }
      if (changed) { _save(); for (const fn of _subs) fn(); }
    },
    applyTier: function(tier) {
      const d = INCYTR_TIER_DEFAULTS[tier];
      if (!d) return;
      this.set({
        tier: tier,
        sliderP:   d.p,
        sliderPds: d.pds,
        sliderSp:  d.sp,
      });
    },
    reset: function() {
      _state = JSON.parse(JSON.stringify(_defaults));
      _save();
      for (const fn of _subs) fn();
    },
    subscribe: function(fn) {
      _subs.push(fn);
      return () => { const i = _subs.indexOf(fn); if (i >= 0) _subs.splice(i, 1); };
    },
  };
})();

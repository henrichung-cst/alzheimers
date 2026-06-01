// ---------------------------------------------------------------------------
// IncytrFilter — shared filter state for the Incytr Heatmap + Pathways tabs.
// Mirrors the KinaseFilter contract: localStorage-backed, get(k) / set(patch)
// / reset() / subscribe(fn). Persistence key: incytrFilter.v3.
//
// v3 (per-group trajectory chips):
//   trajLabels     — object keyed by the active context's contrast group, each value an
//                    array of selected labels. AND within a disease (path
//                    must carry every selected label) AND across diseases.
//                    {App: [], Tau: [], ApTt: []} = no gate.
//   recurContrasts — diseases that must each have a complete trajectory
//                    (AND logic). [] = no recur gate. e.g. ["App","Tau"].
//   detailRowKey   — row key whose expanded panel is currently showing the
//                    Trajectory sub-tab (null = none).
// ---------------------------------------------------------------------------

window.IncytrFilter = (function() {
  const _KEY = "incytrFilter.v3";
  const _defaults = {
    // Heatmap projection — single contrast picker, two ordinal selects, and
    // pvalue + |PDS| gates (snapped to heatmap_counts.thresholds /
    // .abs_pds_thresholds). pvalue defaults to null (no gate) since the
    // per-animal SigProb Wald-t is unreliable in this cohort; |PDS| is the
    // recommended primary filter.
    hmDisease:      "App",
    hmTimepoint:    "2mo",
    hmPvalue:       null,
    hmAbsPds:       0.01,

    // Pathway table — multiselect filters (empty = any). sliderPds is the
    // |PDS| effect-size floor (primary). sliderP is the pvalue gate (opt-in;
    // legacy sliderSp (sigprob) was retired 2026-05-12).
    pair:           null,          // {sender, receiver} or null
    disease:        [],            // [] = any
    timepoint:      [],            // [] = any
    senderIn:       [],            // [] = any
    receiverIn:     [],            // [] = any
    sliderP:        null,
    sliderPds:      0.5,
    searchText:     "",
    pairPage:       0,
    sortKey:        "PDS",
    sortDir:        -1,

    // CR-04 trajectory / recurrence filters.
    trajLabels:     { App: [], Tau: [], ApTt: [] },   // per-disease chip sets
    recurContrasts: [],            // [] = no gate; ["App","Tau"] = AND both
    detailRowKey:   null,          // expanded detail row key (ephemeral)
  };
  const _arrKeys = new Set(["disease","timepoint","senderIn","receiverIn",
                             "recurContrasts"]);
  // Per-disease object — needs special merging in set() and on load.
  const _objKeys = new Set(["trajLabels"]);
  let _state = Object.assign({}, _defaults);
  try {
    const saved = JSON.parse(localStorage.getItem(_KEY) || "null");
    if (saved && typeof saved === "object") {
      for (const k of Object.keys(_defaults)) {
        if (!(k in saved)) continue;
        if (_arrKeys.has(k)) _state[k] = Array.isArray(saved[k]) ? saved[k].slice() : [];
        else if (_objKeys.has(k)) {
          // Merge dynamic per-context keys, sanitising each value to an array.
          const cur = Object.assign({}, _defaults[k]);
          if (saved[k] && typeof saved[k] === "object") {
            for (const d of Object.keys(saved[k])) {
              cur[d] = Array.isArray(saved[k][d]) ? saved[k][d].slice() : [];
            }
          }
          _state[k] = cur;
        }
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
        } else if (_objKeys.has(k)) {
          // Deep-equal compare via JSON; replace wholesale on diff.
          const cur = _state[k] || {};
          const merged = Object.assign({}, cur);
          if (nv && typeof nv === "object") {
            for (const d of Object.keys(nv)) {
              merged[d] = Array.isArray(nv[d]) ? nv[d].slice() : [];
            }
          }
          if (JSON.stringify(merged) !== JSON.stringify(cur)) {
            _state[k] = merged; changed = true;
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

// ---------------------------------------------------------------------------
// IncytrFilter — shared filter state for the Incytr Heatmap + Pathways tabs.
// Mirrors the KinaseFilter contract: localStorage-backed, get(k) / set(patch)
// / reset() / subscribe(fn). Persistence key: incytrFilter.v11.
//
// v11 (J-3):
//   grain               — "Full" | "L-R-EM" | "R-EM-T" | "R-EM" node grain.
//   timepointCombine    — timepoints that must be covered (entity-level, per-disease).
//   timepointCombineMode — "all" (AND) | "any" (OR).
//
// v10:
//   pdsSign        — "both" | "up" | "down" pathway PDS direction filter.
//   scoreMinAbs    — per-score-column absolute-value floors, e.g. {TPDS: 0.2}.
//
// v9:
//   trend          — canonical TrendFilter value applied to Incytr PDS
//                    trajectory labels for the row's disease group.
//   recurContrasts — diseases that must each have a complete trajectory
//                    (AND logic). [] = no recur gate. e.g. ["App","Tau"].
//   detailRowKey   — row key whose expanded panel is currently showing the
//                    Trajectory sub-tab (null = none).
// ---------------------------------------------------------------------------

window.IncytrFilter = (function() {
  const _KEY = "incytrFilter.v11";
  const _defaults = {
    // Heatmap projection — timeline contrast scrubber, ordinal selects, and
    // pvalue + |PDS| gates (snapped to heatmap_counts.thresholds /
    // .abs_pds_thresholds). pvalue defaults to null (no gate) since the
    // per-animal SigProb Wald-t is unreliable in this cohort; |PDS| is the
    // recommended primary filter.
    hmView:         "timeline",
    hmTimelineIndex: 0,
    hmDisease:      "App",
    hmTimepoint:    "2mo",
    hmPvalue:       null,
    hmAbsPds:       0.01,
    hmAxisLimit:    "all",
    hmScale:        "linear",
    hmPdsSign:      "both",
    excludeLowSignalCelltypes: false,

    // Pathway table — multiselect filters (empty = any). sliderPds is the
    // |PDS| effect-size floor (primary). sliderP is the pvalue gate (opt-in;
    // legacy sliderSp (sigprob) was retired 2026-05-12).
    ipMode:         "top",        // "top" = ranked across all pairs; "pair" = one sender/receiver shard
    topLimit:       500,
    pair:           null,          // {sender, receiver} or null
    disease:        [],            // [] = any
    timepoint:      [],            // [] = any
    senderIn:       [],            // [] = any
    receiverIn:     [],            // [] = any
    sliderP:        null,
    sliderPds:      0.5,
    pdsSign:        "both",
    scoreMinAbs:    {},
    searchText:     "",
    pairPage:       0,
    sortKey:        "rank",
    sortDir:        1,
    trend:          "",

    // J-3: Node-grain selector.
    // "Full" = L-R-EM-T (default, existing behavior).
    // "L-R-EM" / "R-EM-T" / "R-EM" = backbone grain; entity is defined by
    // the surviving nodes; dropped nodes render as "—".
    grain:               "Full",

    // J-3: Timepoint-combination filter.
    // timepointCombine: which timepoints must be covered ([] = no constraint).
    // timepointCombineMode: "all" (AND, every selected tp in the same disease)
    //   or "any" (OR, at least one selected tp in some disease).
    // Predicate is entity-level and evaluated per-disease, not across diseases.
    timepointCombine:    [],
    timepointCombineMode: "all",

    // CR-04 trajectory / recurrence filters.
    recurContrasts: [],            // [] = no gate; ["App","Tau"] = AND both
    detailRowKey:   null,          // expanded detail row key (ephemeral)
  };
  const _arrKeys = new Set(["disease","timepoint","senderIn","receiverIn",
                             "recurContrasts","timepointCombine"]);
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

window.IncytrCelltypeQc = (function() {
  let _cacheBlock = null;
  let _cacheSet = null;

  function lowSignalSet(block) {
    const b = block || (window.ViewerPayload && ViewerPayload.incytr && ViewerPayload.incytr());
    if (!b) return new Set();
    if (_cacheBlock === b && _cacheSet) return _cacheSet;
    const names = b.low_signal_celltypes
      || (b.celltype_qc && b.celltype_qc.low_signal_celltypes)
      || [];
    _cacheBlock = b;
    _cacheSet = new Set(Array.isArray(names) ? names : []);
    return _cacheSet;
  }

  function hasLowSignal(block) {
    return lowSignalSet(block).size > 0;
  }

  function enabled(block) {
    return !!(window.IncytrFilter
      && IncytrFilter.get("excludeLowSignalCelltypes")
      && hasLowSignal(block));
  }

  function endpointExcluded(name, block) {
    return enabled(block) && lowSignalSet(block).has(name);
  }

  function pairExcluded(sender, receiver, block) {
    return enabled(block)
      && (lowSignalSet(block).has(sender) || lowSignalSet(block).has(receiver));
  }

  function controlText(block) {
    const set = lowSignalSet(block);
    if (!set.size) return "";
    return `excluding ${set.size} sparse cell type${set.size === 1 ? "" : "s"}`;
  }

  return { lowSignalSet, hasLowSignal, enabled, endpointExcluded, pairExcluded, controlText };
})();

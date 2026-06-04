// ---------------------------------------------------------------------------
// Shared trend / trajectory filter helpers.
// Values are canonicalized with underscores for UI state. Incytr payload rows
// store trajectory labels with hyphens, so helpers expose both forms.
// ---------------------------------------------------------------------------

window.TrendFilter = (function() {
  const KINASE_VALUES = ["always_up", "always_down", "monotonic_up", "monotonic_down", "mixed", "peak", "trough"];
  const INCYTR_VALUES = ["always_up", "always_down", "monotonic_up", "monotonic_down", "mixed"];
  const LABELS = {
    "": "Any",
    always_up: "always up",
    always_down: "always down",
    monotonic_up: "monotonic up",
    monotonic_down: "monotonic down",
    mixed: "mixed",
    peak: "single peak",
    trough: "single trough",
  };

  function normalize(value) {
    return String(value || "").trim().replace(/-/g, "_");
  }

  function payloadLabel(value) {
    return normalize(value).replace(/_/g, "-");
  }

  function label(value) {
    const v = normalize(value);
    return LABELS[v] || v || LABELS[""];
  }

  function options(kind) {
    const vals = kind === "incytr" ? INCYTR_VALUES : KINASE_VALUES;
    return ["", ...vals];
  }

  function optionsHtml(kind) {
    return options(kind).map(v =>
      `<option value="${_escapeHtml(v)}">${_escapeHtml(label(v))}</option>`
    ).join("");
  }

  function vectorMatches(values, pattern) {
    const p = normalize(pattern);
    if (!p) return true;
    const v = (values || []).map(Number);
    if (v.length < 2) return false;
    for (let i = 0; i < v.length; i++) {
      if (!isFinite(v[i])) return false;
    }
    if (p === "always_up") return v.every(x => x > 0);
    if (p === "always_down") return v.every(x => x < 0);
    if (p === "mixed") {
      let hasUp = false, hasDown = false;
      for (const x of v) {
        if (x > 0) hasUp = true;
        if (x < 0) hasDown = true;
      }
      return hasUp && hasDown;
    }
    if (p === "monotonic_up") {
      let strict = false;
      for (let i = 1; i < v.length; i++) {
        if (v[i] < v[i - 1]) return false;
        if (v[i] > v[i - 1]) strict = true;
      }
      return strict;
    }
    if (p === "monotonic_down") {
      let strict = false;
      for (let i = 1; i < v.length; i++) {
        if (v[i] > v[i - 1]) return false;
        if (v[i] < v[i - 1]) strict = true;
      }
      return strict;
    }
    if (p === "peak") {
      let k = 0;
      for (let i = 1; i < v.length; i++) if (v[i] > v[k]) k = i;
      if (k === 0 || k === v.length - 1) return false;
      for (let i = 1; i <= k; i++) if (v[i] <= v[i - 1]) return false;
      for (let i = k + 1; i < v.length; i++) if (v[i] >= v[i - 1]) return false;
      return true;
    }
    if (p === "trough") {
      let k = 0;
      for (let i = 1; i < v.length; i++) if (v[i] < v[k]) k = i;
      if (k === 0 || k === v.length - 1) return false;
      for (let i = 1; i <= k; i++) if (v[i] >= v[i - 1]) return false;
      for (let i = k + 1; i < v.length; i++) if (v[i] <= v[i - 1]) return false;
      return true;
    }
    return true;
  }

  return { normalize, payloadLabel, label, options, optionsHtml, vectorMatches };
})();

// ---------------------------------------------------------------------------
// Shared trend / trajectory filter helpers.
// Values are canonicalized with underscores for UI state. Incytr payload rows
// store trajectory labels with hyphens, so helpers expose both forms.
// ---------------------------------------------------------------------------

window.TrendFilter = (function() {
  // One trend vocabulary repo-wide (kinase per-genotype pills + incytr paths):
  // direction × monotonicity, plus "mixed" for a sign reversal across timepoints.
  const TREND_VALUES = ["always_up", "always_down", "monotonic_up", "monotonic_down", "mixed"];
  const LABELS = {
    "": "Any",
    always_up: "always up",
    always_down: "always down",
    monotonic_up: "monotonic up",
    monotonic_down: "monotonic down",
    mixed: "mixed",
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

  function options() {
    return ["", ...TREND_VALUES];
  }

  function optionsHtml() {
    return options().map(v =>
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
    return true;
  }

  // Classify an ordered NES vector into a single, mutually-exclusive trend label
  // (the pill state, and what the kinase filter matches against). Sign-consistency
  // is the primary axis: a vector that never changes sign is always_up/always_down
  // regardless of monotonicity; a sign-crossing vector that is strictly ordered is
  // monotonic_up/down; any other sign-crossing vector is "mixed". Returns null when
  // there are < 2 finite values (rendered as a muted "—", never a fabricated label).
  function classify(values) {
    const v = (values || []).map(Number).filter(Number.isFinite);
    if (v.length < 2) return null;
    if (v.every(x => x > 0)) return "always_up";
    if (v.every(x => x < 0)) return "always_down";
    if (vectorMatches(v, "monotonic_up")) return "monotonic_up";
    if (vectorMatches(v, "monotonic_down")) return "monotonic_down";
    return "mixed";
  }

  return { normalize, payloadLabel, label, options, optionsHtml, vectorMatches, classify };
})();

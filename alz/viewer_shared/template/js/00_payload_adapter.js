"use strict";

// Shared v2 payload adapter. Context-aware blocks are canonical; flat blocks
// remain accepted for single-context AD payloads.
const ViewerPayload = (function(){
  function _payload() {
    return (typeof PAYLOAD !== "undefined" && PAYLOAD) || {};
  }
  function _meta() {
    return _payload().meta || {};
  }
  function schemaVersion() {
    return Number(_meta().viewer_payload_schema_version || 1);
  }
  function contexts() {
    const ctx = _meta().contexts;
    if (Array.isArray(ctx) && ctx.length) return ctx;
    return [{
      id: _meta().default_context || "song_ad",
      label: "Song AD",
      cohort: _meta().cohort || "song_ad",
      axis_kind: "cohort",
      capabilities: {},
      notes: [],
    }];
  }
  function defaultContext() {
    return _meta().default_context || (contexts()[0] && contexts()[0].id) || "song_ad";
  }
  function activeContext() {
    const sel = (window.Store && Store.state && Store.state.selection) || {};
    return sel.context || defaultContext();
  }
  function contextRecord(contextId) {
    const id = contextId || activeContext();
    return contexts().find(c => c.id === id) || contexts()[0] || null;
  }
  function contextCapabilities(contextId) {
    const g = _meta().capabilities || {};
    const c = (contextRecord(contextId) || {}).capabilities || {};
    return Object.assign({}, g, c);
  }
  function contrastAxis(contextId) {
    const rec = contextRecord(contextId) || {};
    const axis = rec.contrast_axis || {};
    return {
      primary: axis.primary || "disease_timepoint",
      groups: axis.groups || _meta().diseaseGroups || [],
      timepoints: axis.timepoints || _meta().timepoints || [],
      baseline: axis.baseline || null,
    };
  }
  function _contextBlock(block, contextId) {
    if (!block) return null;
    const id = contextId || activeContext();
    // Context-keyed blocks resolve strictly: a missing id means "no data for
    // this context" (caller's default applies), NOT "return the whole wrapper".
    // Returning the {by_context:{…}} envelope here would hand kinases()/
    // celltypes() a malformed shape under a context that only carries incytr
    // (e.g. fivexfad_cortex). Flat legacy blocks (no by_context) pass through.
    if (block.by_context) return block.by_context[id] || null;
    return block;
  }
  function kinases(contextId) {
    return _contextBlock(_payload().kinases, contextId) || { id: [], name: [] };
  }
  function celltypes(contextId) {
    return _contextBlock(_payload().celltypes, contextId) || { id: [], name: [] };
  }
  function incytr(contextId) {
    return _contextBlock(_payload().incytr_pathways, contextId);
  }
  function incytrSliceIndex(contextId) {
    const block = incytr(contextId);
    return (block && block.slice_index) || {};
  }
  function edgeUrl(kind) {
    const ref = _payload().edge_slice_ref || {};
    return ref[kind + "_url"] || ref[kind] || "";
  }
  function _sanitizePathToken(value, index) {
    let out = String(value).replace(/\//g, "-").replace(/ /g, "_");
    if (String((index && index.sanitize_rule) || "").includes("'.'"))
      out = out.replace(/\./g, "");
    return out;
  }
  function incytrShardFilename(sender, receiver, contextId) {
    const ctx = contextId || activeContext();
    const idx = incytrSliceIndex(ctx);
    const tmpl = idx.filename_template || "{sender}__{receiver}.parquet";
    return tmpl
      .replace("{context}", _sanitizePathToken(ctx, idx))
      .replace("{donor}", _sanitizePathToken(ctx, idx))
      .replace("{sender}", _sanitizePathToken(sender, idx))
      .replace("{receiver}", _sanitizePathToken(receiver, idx));
  }
  return {
    schemaVersion,
    contexts,
    defaultContext,
    activeContext,
    contextRecord,
    contextCapabilities,
    contrastAxis,
    kinases,
    celltypes,
    incytr,
    incytrSliceIndex,
    edgeUrl,
    incytrShardFilename,
  };
})();
window.ViewerPayload = ViewerPayload;

/**
 * Shared CSV export utilities.
 * Three pure functions — no globals, no DOM side-effects.
 */

/**
 * Escape one cell value for RFC-4180 CSV.
 * null/undefined → ""; non-finite numbers → "".
 * Wraps in "" when value contains , " \r \n; doubles internal ".
 */
function csvEscape(v) {
  if (v == null || (typeof v === "number" && !isFinite(v))) return "";
  const s = String(v);
  return /[",\r\n]/.test(s) ? '"' + s.replace(/"/g, '""') + '"' : s;
}

/**
 * Serialize rows to a CSV string.
 * @param {string[]} headers  Column display names for the header row.
 * @param {string[]} keys     Parallel array of keys to read from each row object.
 * @param {Object[]} rows     Data rows (plain JS objects).
 * @returns {string}          Complete CSV text, UTF-8, trailing newline.
 */
function csvSerialize(headers, keys, rows) {
  const lines = [headers.map(csvEscape).join(",")];
  for (const r of rows) {
    lines.push(keys.map(k => csvEscape(r[k])).join(","));
  }
  return lines.join("\n") + "\n";
}

/**
 * Trigger a browser file download of a CSV blob.
 * @param {string} csv       Output of csvSerialize().
 * @param {string} filename  Filename including .csv extension.
 */
function csvDownload(csv, filename) {
  const blob = new Blob([csv], { type: "text/csv;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 0);
}

/**
 * Signed numeric comparator for table sort (dir: +1 desc, -1 asc).
 * Nulls and NaN are always last regardless of dir.
 */
function numCmp(av, bv, dir) {            // dir: +1 desc (default), -1 asc
  const an = (av == null || av !== av), bn = (bv == null || bv !== bv);
  if (an && bn) return 0;
  if (an) return 1;                        // nulls/NaN ALWAYS last
  if (bn) return -1;
  return dir > 0 ? (bv - av) : (av - bv);
}

/**
 * Rewrite a leading cohort token in a CSV column header to the display name.
 * Token map: Mouse/Song/song→MouseC1, 5xFAD/fivexfad→MouseC2, Human/Mukesh→HumanC1.
 * Atlas tokens (WMB, SEAAD) pass through unchanged.
 * @param {string} header    Original column header string.
 * @param {Object} displayMap  COHORT_LABELS (or equivalent) mapping cohort keys to display names.
 * @returns {string}           Header with leading cohort token replaced, or original if no match.
 */
function cohortHeader(header, displayMap) {
  const mc1 = displayMap.song || "MouseC1";
  const mc2 = displayMap.fivexfad || "MouseC2";
  const hc1 = displayMap.mukesh || "HumanC1";
  return header
    .replace(/^(Mouse|Song|song)(_|$)/, mc1 + "$2")
    .replace(/^(5xFAD|fivexfad)(_|$)/, mc2 + "$2")
    .replace(/^(Human|Mukesh)(_|$)/, hc1 + "$2");
}

/**
 * Build a normalized CSV download filename.
 * @param {string|null} cohortDisplay  Cohort display name (e.g. "MouseC1"), or null for cross-cohort tables.
 * @param {string}      table          Table identifier (e.g. "kinase", "attribution").
 * @returns {string}                   "<cohortdisplay>__<table>.csv" lowercased, or "<table>.csv" when cohort is null.
 */
function exportFilename(cohortDisplay, table) {
  const t = String(table).toLowerCase().replace(/[^a-z0-9_.-]+/g, "_");
  if (!cohortDisplay) return `${t}.csv`;
  const c = String(cohortDisplay).toLowerCase().replace(/[^a-z0-9_.-]+/g, "_");
  return `${c}__${t}.csv`;
}

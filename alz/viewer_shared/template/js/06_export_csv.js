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

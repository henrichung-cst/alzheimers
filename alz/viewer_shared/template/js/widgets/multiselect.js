// ---------------------------------------------------------------------------
// Reusable multiselect popover. Mirrors the Disease/Timepoint/Cell type
// dropdowns on the Kinase tab: button summarizes selection ("Any" / single
// value / "N selected"); panel has a Clear action + checkbox list; closes on
// outside click.
//
// Usage:
//   mountMultiselect(hostEl, {
//     label:    "Disease",                    // optional inline prefix label
//     options:  ["App", "Tau", "ApTt"],
//     current:  ["App"],                      // [] = any
//     onChange: (nextArray) => { ... },
//   });
//
// Re-mount with new `current` to re-render after external state changes.
// ---------------------------------------------------------------------------

function mountMultiselect(host, opts) {
  if (!host) return;
  const label   = opts.label || "";
  const options = opts.options || [];
  const cur     = (opts.current || []).slice();
  const curSet  = new Set(cur);
  const onChange = opts.onChange || (() => {});
  const summary =
      cur.length === 0     ? "Any"
    : cur.length <= 2      ? cur.join(", ")
    : `${cur.length} selected`;
  const optsHtml = options.map(v => {
    const checked = curSet.has(v) ? " checked" : "";
    return `<label class="ms-opt"><input type="checkbox" data-val="${_escapeHtml(v)}"${checked}/>${_escapeHtml(v)}</label>`;
  }).join("");
  host.innerHTML =
    (label ? `<span style="margin-right:4px;">${_escapeHtml(label)}</span>` : "") +
    `<span class="ms-wrap">` +
      `<button type="button" class="ms-button" data-active="${cur.length ? 1 : 0}" ` +
        `aria-haspopup="true" aria-expanded="false">${_escapeHtml(summary)}</button>` +
      `<div class="ms-panel" role="listbox" aria-multiselectable="true">` +
        `<div class="ms-action" data-action="clear">Clear</div>` +
        `<div class="ms-divider"></div>` +
        optsHtml +
      `</div>` +
    `</span>`;
  const wrap = host.querySelector(".ms-wrap");
  const btn  = wrap.querySelector(".ms-button");
  const panel = wrap.querySelector(".ms-panel");
  btn.addEventListener("click", ev => {
    ev.stopPropagation();
    const open = panel.classList.toggle("open");
    btn.setAttribute("aria-expanded", open ? "true" : "false");
    document.querySelectorAll(".ms-panel.open").forEach(p => {
      if (p !== panel) {
        p.classList.remove("open");
        const b = p.parentElement && p.parentElement.querySelector(".ms-button");
        if (b) b.setAttribute("aria-expanded", "false");
      }
    });
  });
  panel.addEventListener("click", ev => ev.stopPropagation());
  panel.querySelectorAll('input[type="checkbox"]').forEach(cb => {
    cb.addEventListener("change", () => {
      const v = cb.dataset.val;
      const i = cur.indexOf(v);
      if (cb.checked && i < 0) cur.push(v);
      else if (!cb.checked && i >= 0) cur.splice(i, 1);
      onChange(cur.slice());
    });
  });
  const clearBtn = panel.querySelector('[data-action="clear"]');
  if (clearBtn) clearBtn.addEventListener("click", () => onChange([]));
}

// One-time outside-click handler — closes any open ms-panel. Idempotent across
// tab wires.
if (!window._msOutsideWired) {
  document.addEventListener("click", () => {
    document.querySelectorAll(".ms-panel.open").forEach(p => {
      p.classList.remove("open");
      const b = p.parentElement && p.parentElement.querySelector(".ms-button");
      if (b) b.setAttribute("aria-expanded", "false");
    });
  });
  window._msOutsideWired = true;
}
window.mountMultiselect = mountMultiselect;

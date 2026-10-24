// GPU Shepherd UI helpers. Loaded as a file so the Content-Security-Policy can forbid inline scripts.
(function () {
  "use strict";
  const modal = () => document.getElementById("modal");
  // the update progress keeps asking while GPU Shepherd restarts: don't report each failed try
  // (registered before the toast below, so it can stop it)
  document.addEventListener("htmx:sendError", (e) => {
    if (e.detail.elt && e.detail.elt.closest("[data-quiet-offline]")) e.stopImmediatePropagation();
  });
  // a GitHub token handed to the server: don't leave it in the form
  document.addEventListener("credentials-sent", () => document.querySelectorAll("[name=token]").forEach((el) => { el.value = ""; }));

  function toast(msg, kind) {
    const box = document.getElementById("toasts");
    if (!box || !msg) return;
    const el = document.createElement("div");
    el.className = "toast " + (kind || "");
    el.textContent = msg;
    box.appendChild(el);
    setTimeout(() => el.remove(), kind === "error" ? 9000 : 6000);
  }
  function openModal() { const m = modal(); if (m && !m.open) m.showModal(); }
  function closeModal() { const m = modal(); if (m && m.open) m.close(); }

  document.addEventListener("toast", (e) => toast(e.detail.msg, e.detail.kind));
  document.addEventListener("open-modal", openModal);
  document.addEventListener("close-modal", closeModal);
  document.addEventListener("htmx:afterSwap", (e) => { if (e.detail.target.id === "modal-body") openModal(); });
  document.addEventListener("htmx:responseError", (e) => toast("Request failed (HTTP " + e.detail.xhr.status + ")", "error"));
  document.addEventListener("htmx:sendError", () => toast("GPU Shepherd is not reachable", "error"));
  // the dashboard re-renders every few seconds: keep the "Advanced details" boxes the user opened open
  const openDetails = new Set();
  document.addEventListener("toggle", (e) => {
    const d = e.target;
    if (d.tagName !== "DETAILS" || !d.closest("#dash")) return;
    const card = d.closest("[id]");
    if (card) (d.open ? openDetails.add(card.id) : openDetails.delete(card.id));
  }, true);
  document.addEventListener("htmx:afterSwap", (e) => {
    if (e.detail.target.id !== "dash") return;
    openDetails.forEach((id) => { const d = document.querySelector("#" + CSS.escape(id) + " details"); if (d) d.open = true; });
  });
  // tooltips: hover and keyboard focus are pure CSS; a click or tap pins one open until you click elsewhere or press Esc
  const tipOf = (el) => el && el.closest(".has-tip");
  let pinned = null, focusedTip = null;
  function closeTips(except) {
    document.querySelectorAll(".has-tip.open").forEach((t) => {
      if (t === except) return;
      t.classList.remove("open");
      const b = t.querySelector(".tip-btn"); if (b) b.setAttribute("aria-expanded", "false");
    });
    if (!except) pinned = null;
  }
  function pin(t) {
    closeTips(t);
    t.classList.add("open");
    t.classList.remove("hush");
    const b = t.querySelector(".tip-btn"); if (b) b.setAttribute("aria-expanded", "true");
    pinned = t.dataset.tip;
  }
  document.addEventListener("click", (e) => {
    const btn = e.target.closest(".tip-btn");
    if (btn) { const t = tipOf(btn); t.classList.contains("open") ? closeTips() : pin(t); return; }
    if (!e.target.closest(".has-tip.open .tip")) closeTips();
  });
  document.addEventListener("keydown", (e) => {
    if (e.key !== "Escape") return;
    const t = tipOf(document.activeElement);
    if (document.querySelector(".has-tip.open") || t) { closeTips(); if (t) t.classList.add("hush"); }
  });
  document.addEventListener("focusout", (e) => { const t = tipOf(e.target); if (t) t.classList.remove("hush"); });
  document.addEventListener("mouseleave", (e) => { if (e.target.classList && e.target.classList.contains("has-tip")) e.target.classList.remove("hush"); }, true);
  // the dashboard refresh replaces the cards: bring back a pinned tooltip and the keyboard focus
  document.addEventListener("htmx:beforeSwap", (e) => {
    if (e.detail.target.id !== "dash") return;
    const t = tipOf(document.activeElement);
    focusedTip = t && e.detail.target.contains(t) ? t.dataset.tip : null;
  });
  document.addEventListener("htmx:afterSwap", (e) => {
    if (e.detail.target.id !== "dash") return;
    const find = (key) => key && document.querySelector('.has-tip[data-tip="' + CSS.escape(key) + '"]');
    const p = find(pinned); if (p) pin(p); else pinned = null;
    const f = find(focusedTip); if (f) { const b = f.querySelector(".tip-btn"); if (b) b.focus({ preventScroll: true }); }
  });
  document.addEventListener("refresh", () => { const d = document.getElementById("dash"); if (d && window.htmx) htmx.trigger(d, "refresh"); });

  document.addEventListener("click", (e) => {
    const t = e.target;
    if (t.closest("[data-close-modal]")) { closeModal(); return; }
    const pick = t.closest("[data-pick-file]");
    if (pick) {
      e.preventDefault();
      const input = document.querySelector("[name=file]");
      if (input) { input.value = pick.dataset.pickFile; input.focus(); }
      const id = document.querySelector("[name=id]");
      if (id && !id.value && pick.dataset.suggestId) id.value = pick.dataset.suggestId;
      return;
    }
    const add = t.closest("[data-add-row]");
    if (add) {
      e.preventDefault();
      const tpl = document.getElementById(add.dataset.addRow);
      tpl.parentElement.insertBefore(tpl.content.cloneNode(true), tpl);
    }
    const del = t.closest("[data-remove-row]");
    if (del) { e.preventDefault(); del.closest(".autorow").remove(); }
  });

  // show only the fields that fit the chosen GPU mode and busy check
  const toggles = { gpu_mode: "gpuMode", drain_check: "drainCheck" };
  function sync() {
    Object.entries(toggles).forEach(([name, key]) => {
      const sel = document.querySelector("[name=" + name + "]");
      if (!sel) return;
      document.querySelectorAll("[data-" + key.replace(/[A-Z]/g, (c) => "-" + c.toLowerCase()) + "]").forEach((el) => {
        el.hidden = !el.dataset[key].split(" ").includes(sel.value);
      });
    });
  }
  document.addEventListener("change", (e) => { if (e.target.name in toggles) sync(); });
  document.addEventListener("DOMContentLoaded", sync);
  document.addEventListener("htmx:afterSettle", sync);
})();

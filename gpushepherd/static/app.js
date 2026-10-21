// GPU Shepherd UI helpers. Loaded as a file so the Content-Security-Policy can forbid inline scripts.
(function () {
  "use strict";
  const modal = () => document.getElementById("modal");

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
  document.addEventListener("refresh", () => { const d = document.getElementById("dash"); if (d && window.htmx) htmx.trigger(d, "refresh"); });

  document.addEventListener("click", (e) => {
    const t = e.target;
    if (t.closest("[data-close-modal]")) { closeModal(); return; }
    const open = t.closest("[data-open-port]");
    if (open) {
      e.preventDefault();
      window.open("http://" + location.hostname + ":" + open.dataset.openPort + (open.dataset.openPath || "/"), "_blank", "noopener");
      return;
    }
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

  // show only the GPU fields that fit the chosen mode
  function syncGpu() {
    const sel = document.querySelector("[name=gpu_mode]");
    if (!sel) return;
    document.querySelectorAll("[data-gpu-mode]").forEach((el) => {
      el.hidden = !el.dataset.gpuMode.split(" ").includes(sel.value);
    });
  }
  document.addEventListener("change", (e) => { if (e.target.name === "gpu_mode") syncGpu(); });
  document.addEventListener("DOMContentLoaded", syncGpu);
  document.addEventListener("htmx:afterSettle", syncGpu);
})();

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
      const input = document.querySelector("[name=" + (pick.dataset.pickTarget || "file") + "]");
      if (input) { input.value = pick.dataset.pickFile; input.focus(); }
      if (pick.hasAttribute("data-pick-close")) closeModal();
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

  // App Center: the category chips show one category at a time (the links still jump there without JS)
  function filterApps(group) {
    document.querySelectorAll("#apps [data-group]").forEach((el) => { el.hidden = !!group && el.dataset.group !== group; });
    document.querySelectorAll("[data-filter]").forEach((c) => c.setAttribute("aria-pressed", String(c.dataset.filter === group)));
  }
  document.addEventListener("click", (e) => {
    const chip = e.target.closest("[data-filter]");
    if (!chip) return;
    e.preventDefault();
    filterApps(chip.dataset.filter);
  });
  document.addEventListener("DOMContentLoaded", () => { if (document.getElementById("apps")) filterApps(""); });

  // theme: System (no cookie), Light or Dark. A cookie, so the server draws every page in it without a flash.
  const THEMES = { "": "System", light: "Light", dark: "Dark" };
  const ORDER = ["", "light", "dark"];
  function setTheme(t) {
    const root = document.documentElement;
    if (t) root.dataset.theme = t; else delete root.dataset.theme;
    document.cookie = "theme=" + t + "; path=/; SameSite=Strict; max-age=" + (t ? 400 * 24 * 3600 : 0);
    document.querySelectorAll("[data-theme-toggle]").forEach((b) => {
      b.title = "Theme: " + THEMES[t] + " (click to change)";
      b.setAttribute("aria-label", "Theme: " + THEMES[t]);
    });
    document.querySelectorAll("[data-theme-set]").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.themeSet === t)));
  }
  document.addEventListener("click", (e) => {
    const set = e.target.closest("[data-theme-set]");
    if (set) return setTheme(set.dataset.themeSet);
    if (!e.target.closest("[data-theme-toggle]")) return;
    const now = document.documentElement.dataset.theme || "";
    setTheme(ORDER[(ORDER.indexOf(now) + 1) % ORDER.length]);
  });

  // show only the fields that fit the chosen GPU mode, busy check and model server
  const toggles = { gpu_mode: "gpuMode", drain_check: "drainCheck", server: "server" };
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

// Books page: upload many books, whole folders or zips from the browser, one file at a time
// (folders kept, books already there skipped, so adding the same folder again carries on)
(function () {
  "use strict";
  const BOOK = /\.(epub|pdf|txt|md|markdown|mobi|azw3|prc|zip)$/i;
  let busy = false, stopped = false, xhr = null;
  const $ = (id) => document.getElementById(id);

  // a dropped folder: walk it (readEntries hands back at most 100 entries a call)
  function walk(entry, prefix, out) {
    if (entry.isFile) return new Promise((ok) => entry.file((f) => { out.push({ file: f, path: prefix + f.name }); ok(); }, () => ok()));
    if (!entry.isDirectory) return Promise.resolve();
    const reader = entry.createReader();
    return new Promise((ok) => {
      const next = () => reader.readEntries(async (list) => {
        if (!list.length) return ok();
        for (const e of list) await walk(e, prefix + entry.name + "/", out);
        next();
      }, () => ok());
      next();
    });
  }

  function put(item) {
    return new Promise((ok) => {
      xhr = new XMLHttpRequest();
      xhr.open("PUT", "/books/files?path=" + encodeURIComponent(item.path));
      xhr.setRequestHeader("HX-Request", "true");
      xhr.upload.onprogress = (e) => item.onprogress(e.loaded);
      xhr.onload = () => {
        let r = {};
        try { r = JSON.parse(xhr.responseText); } catch (err) { r = { error: "HTTP " + xhr.status }; }
        if (xhr.status === 401) r.error = "signed out: sign in again, then add the same books to carry on";
        ok(xhr.status < 300 ? r : { error: r.error || "HTTP " + xhr.status, status: xhr.status });
      };
      xhr.onerror = () => ok({ error: "GPU Shepherd is not reachable", net: true });
      xhr.onabort = () => ok({ error: "stopped", aborted: true });
      xhr.send(item.file);
    });
  }

  async function upload(items) {
    const box = $("book-drop");
    if (busy || !box) return;
    const text = $("book-up-text"), bar = $("book-up-bar"), fails = $("book-up-fails");
    const maxBook = Number(box.dataset.uploadMb) * 1048576, maxZip = Number(box.dataset.zipMb) * 1048576;
    const failed = [];
    const fail = (path, why) => {
      failed.push(path + ": " + why);
      const li = document.createElement("li"); li.textContent = path + ": " + why;
      fails.querySelector("ul").appendChild(li);
      fails.querySelector("summary").textContent = failed.length + " not added";
      fails.hidden = false;
    };
    busy = true; stopped = false;
    $("book-up").hidden = false; fails.hidden = true; fails.querySelector("ul").textContent = "";
    text.textContent = "Checking what is there already…";
    let have = {};
    try { have = (await (await fetch("/books/files", { headers: { "HX-Request": "true" } })).json()).files || {}; } catch (err) { have = {}; }
    const todo = [];
    let skipped = 0, other = 0;
    for (const it of items) {
      it.path = it.path.replace(/^\/+/, "");
      const name = it.path.split("/").pop();
      if (name.startsWith(".") || !BOOK.test(name)) { other++; continue; }
      const zip = /\.zip$/i.test(name);
      if (!zip && have[it.path] === it.file.size) { skipped++; continue; }
      if (zip && Object.keys(have).some((p) => p.startsWith(it.path.slice(0, -4) + "/"))) { skipped++; continue; }
      if (it.file.size > (zip ? maxZip : maxBook)) { fail(it.path, "too big to upload here"); continue; }
      todo.push(it);
    }
    const total = todo.reduce((n, it) => n + it.file.size, 0) || 1;
    let sent = 0, added = 0;
    window.addEventListener("beforeunload", warn);
    for (let i = 0; i < todo.length && !stopped; i++) {
      const it = todo[i];
      it.onprogress = (n) => {
        bar.style.width = Math.min(100, Math.round(100 * (sent + n) / total)) + "%";
        text.textContent = "Uploading " + (i + 1) + " of " + todo.length + ": " + it.path +
          " (" + (Math.round((sent + n) / 1048576)) + " of " + Math.round(total / 1048576) + " MB)";
      };
      it.onprogress(0);
      let r = await put(it);
      if (r.net && !stopped) { await new Promise((ok) => setTimeout(ok, 3000)); r = await put(it); }  // one more try
      if (r.aborted) break;
      if (r.error) fail(it.path, r.error); else added += (r.added || []).length;
      if (r.status === 401) break;
      sent += it.file.size;
    }
    window.removeEventListener("beforeunload", warn);
    busy = false; xhr = null;
    bar.style.width = "100%";
    let msg = (stopped ? "Stopped. " : "") + "Added " + added + " book" + (added === 1 ? "" : "s");
    if (skipped) msg += ", skipped " + skipped + " already there";
    if (other) msg += ", left out " + other + " other file" + (other === 1 ? "" : "s");
    text.textContent = msg + ".";
    if (added && !stopped) {
      text.textContent = msg + ". Scanning…";
      try {
        const r = await fetch("/books/scan", { method: "POST", headers: { "HX-Request": "true" } });
        const go = r.headers.get("HX-Redirect");
        if (go && !failed.length) window.location = go;
        else if (go) text.textContent = msg + ". The scan has started: reload the page to follow it.";
      } catch (err) { text.textContent = msg + ". Scan the folder to read them."; }
    }
  }
  function warn(e) { e.preventDefault(); e.returnValue = ""; }

  const fromInput = (input) => Array.from(input.files).map((f) => ({ file: f, path: f.webkitRelativePath || f.name }));
  document.addEventListener("change", (e) => {
    if (e.target.id === "book-files" || e.target.id === "book-folder") { upload(fromInput(e.target)); e.target.value = ""; }
  });
  document.addEventListener("click", (e) => {
    if (e.target.id === "book-up-stop") { stopped = true; if (xhr) xhr.abort(); }
  });
  ["dragenter", "dragover"].forEach((t) => document.addEventListener(t, (e) => {
    const box = e.target.closest && e.target.closest("#book-drop");
    if (box) { e.preventDefault(); box.classList.add("over"); }
  }));
  document.addEventListener("dragleave", (e) => { const box = e.target.closest && e.target.closest("#book-drop"); if (box && !box.contains(e.relatedTarget)) box.classList.remove("over"); });
  document.addEventListener("drop", async (e) => {
    const box = e.target.closest && e.target.closest("#book-drop");
    if (!box) return;
    e.preventDefault(); box.classList.remove("over");
    const entries = Array.from(e.dataTransfer.items || []).map((i) => i.webkitGetAsEntry && i.webkitGetAsEntry()).filter(Boolean);
    const out = [];
    if (entries.length) { for (const en of entries) await walk(en, "", out); }
    else Array.from(e.dataTransfer.files).forEach((f) => out.push({ file: f, path: f.name }));
    upload(out);
  });
})();

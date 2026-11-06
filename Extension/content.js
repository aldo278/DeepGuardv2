// Content script: remembers which image was right-clicked and draws the result card next to it.
// It never talks to the backend (see background.js for why); it only shows what the worker sends.
(() => {
  if (window.__deepguardLoaded) return; // injected again after an extension reload
  window.__deepguardLoaded = true;

  const GAP = 8;
  const MARGIN = 8;

  // ---- which image did the user click? ----
  // The context-menu click only gives the worker a URL, not the element, so we note the element on
  // the contextmenu event (capture phase, so pages that stop propagation can't hide it from us).
  let lastImage = null;
  document.addEventListener("contextmenu", (e) => {
    lastImage = e.target instanceof HTMLImageElement ? e.target : null;
  }, true);

  function findImage(srcUrl) {
    if (lastImage && lastImage.isConnected) return lastImage;
    return [...document.images].find((img) => img.currentSrc === srcUrl || img.src === srcUrl) || null;
  }

  // ---- the card ----
  let host = null;   // element in the page that owns the shadow root
  let root = null;   // the shadow root, so page CSS can't touch the card and ours can't leak out
  let anchor = null; // the image the card sits next to

  const CSS = `
    :host { all: initial; }
    .card { position: absolute; z-index: 2147483647; width: 280px; box-sizing: border-box; padding: 12px 14px;
      font: 14px/1.4 system-ui, sans-serif; color: #1c1f24; background: #fff; border: 1px solid #dfe3e8;
      border-radius: 12px; box-shadow: 0 6px 24px rgba(0,0,0,.22); }
    @media (prefers-color-scheme: dark) { .card { color: #e8eaed; background: #1d2026; border-color: #30343c; } }
    .top { display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px; }
    .brand { font-weight: 600; font-size: 12px; opacity: .7; letter-spacing: .02em; }
    .close { all: unset; cursor: pointer; padding: 0 4px; font-size: 16px; line-height: 1; opacity: .6; }
    .close:hover { opacity: 1; }
    .verdict { display: flex; align-items: center; gap: 10px; }
    .badge { color: #fff; font-weight: 700; font-size: 13px; padding: 3px 10px; border-radius: 6px; }
    .FAKE { background: #c0392b; } .REAL { background: #1e8e5a; } .UNKNOWN { background: #8a94a3; } .ERROR { background: #b7791f; }
    .headline { font-weight: 600; }
    .sub { margin-top: 6px; font-size: 12.5px; opacity: .75; }
    .note { margin-top: 6px; font-size: 12px; color: #b7791f; }
    .toggle { all: unset; cursor: pointer; margin-top: 8px; font-size: 12.5px; color: #3b6df0; }
    .rows { margin-top: 6px; border-top: 1px solid #dfe3e8; }
    @media (prefers-color-scheme: dark) { .rows { border-color: #30343c; } .toggle { color: #7c9cff; } }
    .row { display: grid; grid-template-columns: 64px 1fr auto; gap: 8px; align-items: center; padding: 5px 0; font-size: 12.5px; }
    .row .v { font-weight: 600; }
    .row .v.FAKE { color: #c0392b; background: none; } .row .v.REAL { color: #1e8e5a; background: none; } .row .v.UNKNOWN, .row .v.ERROR { color: #8a94a3; background: none; }
    .row .d { opacity: .7; grid-column: 1 / -1; margin-top: -3px; }
    .spin { width: 18px; height: 18px; border: 3px solid #dfe3e8; border-top-color: #3b6df0; border-radius: 50%; animation: s .8s linear infinite; }
    @keyframes s { to { transform: rotate(360deg); } }
    .err { color: #c0392b; font-weight: 600; }
  `;

  function el(tag, cls, text) {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text !== undefined) e.textContent = text; // textContent only: nothing from the page or backend is ever parsed as HTML
    return e;
  }

  function close() {
    if (host) host.remove();
    host = root = anchor = null;
  }

  function open(image) {
    close();
    anchor = image;
    host = document.createElement("div");
    root = host.attachShadow({ mode: "closed" });
    const style = document.createElement("style");
    style.textContent = CSS;
    root.append(style);
    document.documentElement.append(host);
  }

  // Below the image if it fits, otherwise over its lower edge; always kept inside the viewport.
  function place(card) {
    const vw = document.documentElement.clientWidth;
    const vh = document.documentElement.clientHeight;
    const rect = anchor ? anchor.getBoundingClientRect() : { left: vw / 2 - 140, bottom: 60, top: 60 };
    const h = card.offsetHeight;
    let left = Math.min(Math.max(rect.left, MARGIN), Math.max(vw - card.offsetWidth - MARGIN, MARGIN));
    let top = rect.bottom + GAP;
    if (top + h > vh - MARGIN) top = Math.max(rect.bottom - h - GAP, MARGIN);
    card.style.left = left + window.scrollX + "px"; // page coordinates, so the card scrolls with the page
    card.style.top = top + window.scrollY + "px";
  }

  function render(body) {
    const card = el("div", "card");
    const top = el("div", "top");
    top.append(el("span", "brand", "DeepGuard"));
    const x = el("button", "close", "✕");
    x.setAttribute("aria-label", "Close");
    x.addEventListener("click", close);
    top.append(x);
    card.append(top, body);
    root.querySelector(".card")?.remove();
    root.append(card);
    place(card);
  }

  // Which result speaks for the image: a found watermark is near-certain proof; otherwise the
  // ensemble once it exists, then the CNN, then whichever model committed to an answer.
  function headline(results) {
    const by = Object.fromEntries(results.map((r) => [r.model, r]));
    if (by.watermark && by.watermark.verdict === "FAKE") return by.watermark;
    return by.ensemble || by.cnn || results.find((r) => r.verdict === "REAL" || r.verdict === "FAKE") || results[0];
  }

  const WORDS = { FAKE: "Likely AI-generated", REAL: "Likely a real photo", UNKNOWN: "Can't tell" };

  function showLoading() {
    const body = el("div", "verdict");
    body.append(el("div", "spin"), el("span", "headline", "Analyzing…"));
    render(body);
  }

  function showError(error) {
    render(el("div", "err", error.message));
  }

  function showResult(outcome) {
    const main = headline(outcome.results);
    const body = el("div");
    const verdict = el("div", "verdict");
    verdict.append(el("span", "badge " + main.verdict, main.verdict), el("span", "headline", WORDS[main.verdict] || main.verdict));
    body.append(verdict);

    const sub = [];
    if (typeof main.prob_fake === "number") sub.push(`${Math.round(main.prob_fake * 100)}% chance it's fake`);
    sub.push("Based on " + main.model + ". The models can be wrong.");
    body.append(el("div", "sub", sub.join(" · ")));
    if (outcome.downscaled) body.append(el("div", "note", "Image was over 10 MB, so it was shrunk and the watermark check was skipped."));

    const rows = el("div", "rows");
    rows.hidden = true;
    for (const r of outcome.results) {
      const row = el("div", "row");
      row.append(el("span", "", r.model), el("span", "v " + r.verdict, r.verdict));
      row.append(el("span", "", typeof r.prob_fake === "number" ? Math.round(r.prob_fake * 100) + "%" : "–"));
      if (r.detail) row.append(el("span", "d", r.detail));
      rows.append(row);
    }
    const toggle = el("button", "toggle", "Show details");
    toggle.addEventListener("click", () => {
      rows.hidden = !rows.hidden;
      toggle.textContent = rows.hidden ? "Show details" : "Hide details";
      place(root.querySelector(".card"));
    });
    body.append(toggle, rows);
    render(body);
  }

  // ---- messages from background.js ----
  chrome.runtime.onMessage.addListener((msg) => {
    if (!msg || msg.type !== "deepguard") return;
    if (msg.state === "loading") {
      open(findImage(msg.srcUrl));
      showLoading();
    } else if (!root) {
      open(findImage(msg.srcUrl)); // result arrived with no card open, e.g. the page re-rendered
    }
    if (msg.state === "result") showResult(msg.outcome);
    else if (msg.state === "error") showError(msg.outcome.error);
  });

  // ---- closing ----
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") close(); }, true);
  document.addEventListener("mousedown", (e) => {
    if (host && !e.composedPath().includes(host)) close();
  }, true);
})();

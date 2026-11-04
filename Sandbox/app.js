// Same origin as the backend by default; open the page with ?api=http://127.0.0.1:8000 to point elsewhere
const API = new URLSearchParams(location.search).get("api") || "";

const drop = document.getElementById("drop");
const picker = document.getElementById("picker");
const results = document.getElementById("results");
const status = document.getElementById("status");
const offline = document.getElementById("offline");
const summary = document.getElementById("summary");
const evalBtn = document.getElementById("evalBtn");
const evalNote = document.getElementById("evalNote");
const evalOut = document.getElementById("evalOut");

function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text;
  return e;
}

async function checkHealth() {
  try {
    const r = await fetch(API + "/health");
    const j = await r.json();
    status.textContent = "backend ok · " + j.models.join(", ");
    status.className = "pill ok";
    offline.hidden = true;
  } catch {
    status.textContent = "backend offline";
    status.className = "pill bad";
    offline.hidden = false;
  }
}

// every analyzed image: { truth: "REAL" | "FAKE" | null, results: [...], render: fn }
const entries = [];

// a model is only scored when it commits to REAL or FAKE; UNKNOWN and ERROR count as abstained
function mark(r, truth) {
  if (!truth || (r.verdict !== "REAL" && r.verdict !== "FAKE")) return null;
  return r.verdict === truth;
}

function resultRows(r, truth) {
  const frag = document.createDocumentFragment();
  const row = el("div", "row");
  const ok = mark(r, truth);
  row.append(el("span", "", r.model + (ok === null ? "" : ok ? " ✓" : " ✗")));
  if (ok !== null) row.firstChild.classList.add(ok ? "right" : "wrong");
  row.append(el("span", "badge " + r.verdict, r.verdict));
  const bar = el("div", "bar");
  if (r.prob_fake !== null && r.prob_fake !== undefined) {
    const fill = el("i");
    fill.style.width = Math.round(r.prob_fake * 100) + "%";
    bar.append(fill);
    bar.title = "P(fake) = " + r.prob_fake.toFixed(3);
  }
  row.append(bar);
  row.append(el("span", "ms", r.ms + " ms"));
  frag.append(row);
  if (r.detail) frag.append(el("div", "detail", r.detail));
  return frag;
}

// footer: per model, correct / scored for the images that have a ground truth
function renderSummary() {
  const stats = {};
  let labeled = 0;
  for (const e of entries) {
    if (!e.truth) continue;
    labeled++;
    for (const r of e.results) {
      const s = (stats[r.model] ||= { right: 0, scored: 0, abstained: 0 });
      const ok = mark(r, e.truth);
      if (ok === null) s.abstained++;
      else { s.scored++; if (ok) s.right++; }
    }
  }
  summary.replaceChildren();
  if (!labeled) { summary.hidden = true; return; }
  summary.hidden = false;
  summary.append(el("strong", "", "Running accuracy on " + labeled + " labeled image" + (labeled === 1 ? "" : "s")));
  for (const [model, s] of Object.entries(stats)) {
    const pct = s.scored ? Math.round(100 * s.right / s.scored) + "%" : "n/a";
    summary.append(el("div", "", model + ": " + pct + " (" + s.right + "/" + s.scored + (s.abstained ? ", " + s.abstained + " abstained" : "") + ")"));
  }
}

function truthButtons(entry) {
  const wrap = el("div", "truth");
  wrap.append(el("span", "", "Actually:"));
  for (const t of ["REAL", "FAKE"]) {
    const b = el("button", "", t === "REAL" ? "Real" : "Fake");
    b.type = "button";
    b.classList.toggle("on", entry.truth === t);
    b.addEventListener("click", () => {
      entry.truth = entry.truth === t ? null : t; // click again to clear
      entry.render();
      renderSummary();
    });
    wrap.append(b);
  }
  return wrap;
}

async function addImage(file) {
  const card = el("div", "card");
  const img = el("img");
  img.src = URL.createObjectURL(file);
  const body = el("div");
  body.append(el("div", "name", file.name || "pasted image"));
  const size = (file.size / 1024).toFixed(0) + " KB";
  const meta = el("div", "meta", size + " · analyzing…");
  const detailBox = el("div");
  body.append(meta);
  card.append(img, body);
  results.prepend(card);

  const entry = { truth: null, results: [] };
  entry.render = () => {
    detailBox.replaceChildren(truthButtons(entry), ...entry.results.map(r => resultRows(r, entry.truth)));
  };

  const form = new FormData();
  form.append("file", file);
  try {
    const r = await fetch(API + "/predict", { method: "POST", body: form });
    const j = await r.json();
    if (!r.ok) throw new Error(j.detail || r.statusText);
    const dims = img.naturalWidth ? img.naturalWidth + "x" + img.naturalHeight + " · " : "";
    meta.textContent = dims + size;
    entry.results = j.results;
    entries.push(entry);
    body.append(detailBox);
    entry.render();
  } catch (e) {
    meta.textContent = "failed: " + e.message;
    meta.style.color = "var(--fake)";
  }
}

// one table per model: rows are the true label, columns the verdicts it got. Counts are shown next to accuracy because small sets are noisy
function renderEval(data) {
  evalOut.replaceChildren();
  const { REAL, FAKE } = data.totals;
  evalNote.textContent = REAL + " real + " + FAKE + " fake images" + (data.skipped.length ? " · " + data.skipped.length + " unreadable skipped" : "");
  for (const [name, m] of Object.entries(data.models)) {
    const box = el("div", "evalmodel");
    const acc = m.accuracy === null ? "n/a" : Math.round(m.accuracy * 100) + "% (" + m.scored + " scored)";
    box.append(el("h3", "", name + " · accuracy " + acc + " · avg " + m.avg_ms + " ms"));
    const verdicts = ["REAL", "FAKE", "UNKNOWN", "ERROR"].filter(v => v === "REAL" || v === "FAKE" || m.counts.REAL[v] || m.counts.FAKE[v]);
    const table = el("table");
    const head = el("tr");
    head.append(el("th", "", "true \ said"), ...verdicts.map(v => el("th", "", v)));
    table.append(head);
    for (const truth of ["REAL", "FAKE"]) {
      const tr = el("tr");
      tr.append(el("td", "", truth), ...verdicts.map(v => el("td", "", String(m.counts[truth][v] || 0))));
      table.append(tr);
    }
    box.append(table);
    evalOut.append(box);
  }
}

evalBtn.addEventListener("click", async () => {
  evalBtn.disabled = true;
  evalNote.textContent = "running, this can take a while…";
  try {
    const r = await fetch(API + "/evaluate", { method: "POST", body: new FormData() });
    const j = await r.json();
    if (!r.ok) throw new Error(j.detail || r.statusText);
    renderEval(j);
  } catch (e) {
    evalNote.textContent = "failed: " + e.message;
  } finally {
    evalBtn.disabled = false;
  }
});

function addFiles(files) {
  [...files].filter(f => f.type.startsWith("image/")).forEach(addImage);
}

drop.addEventListener("keydown", e => { if (e.key === "Enter" || e.key === " ") picker.click(); });
picker.addEventListener("change", () => { addFiles(picker.files); picker.value = ""; });
["dragenter", "dragover"].forEach(t => drop.addEventListener(t, e => { e.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach(t => drop.addEventListener(t, e => { e.preventDefault(); drop.classList.remove("over"); }));
drop.addEventListener("drop", e => addFiles(e.dataTransfer.files));
document.addEventListener("paste", e => addFiles(e.clipboardData.files));

checkHealth();
setInterval(checkHealth, 10000);

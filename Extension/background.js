// Service worker: adds the "Detect image" menu entry, gets the image bytes, and asks the backend.
// All backend calls live here, never in a content script, because a page on https can't call
// http://127.0.0.1 (CORS / mixed content) but the service worker can (it has host permission).
// The card on the page is drawn by content.js; this file only sends it messages.

const MENU_ID = "detect-image";
const DEFAULT_BACKEND = "http://127.0.0.1:8000";
const MAX_BYTES = 10 * 1024 * 1024; // same limit as the backend

// MV3 service workers sleep and restart, so the menu is created once when the extension is
// installed or updated, not every time the worker starts. Chrome keeps the entry after that.
chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.removeAll(() => {
    chrome.contextMenus.create({
      id: MENU_ID,
      title: "Detect image",
      contexts: ["image"],
    });
  });
});

chrome.contextMenus.onClicked.addListener(async (info, tab) => {
  if (info.menuItemId !== MENU_ID) return;
  const frameId = info.frameId || 0; // the card is drawn in the frame that holds the image
  await sendToPage(tab, frameId, { state: "loading", srcUrl: info.srcUrl });
  const outcome = await detect(info, tab);
  console.log("DeepGuard result", outcome);
  await sendToPage(tab, frameId, { state: outcome.ok ? "result" : "error", srcUrl: info.srcUrl, outcome });
});

// Pages that were already open when the extension was installed or reloaded don't have content.js yet,
// so if the message has no receiver we inject it once and try again. Pages that can't be scripted
// (chrome://, the web store) stay silent; the result is still in this worker's console.
async function sendToPage(tab, frameId, message) {
  if (!tab || tab.id === undefined) return;
  const msg = { type: "deepguard", ...message };
  try {
    await chrome.tabs.sendMessage(tab.id, msg, { frameId });
  } catch {
    try {
      await chrome.scripting.executeScript({ target: { tabId: tab.id, frameIds: [frameId] }, files: ["content.js"] });
      await chrome.tabs.sendMessage(tab.id, msg, { frameId });
    } catch {
      // nowhere to show it
    }
  }
}

// Every failure becomes { ok: false, error: { code, message } } so the card in step 11 can show a
// clear message per case instead of a stack trace.
function fail(code, message) {
  return { ok: false, error: { code, message } };
}

async function backendUrl() {
  const { backendUrl } = await chrome.storage.sync.get("backendUrl");
  return (backendUrl || DEFAULT_BACKEND).replace(/\/+$/, "");
}

async function detect(info, tab) {
  const image = await getImageBlob(info, tab);
  if (!image.ok) return image;

  let blob = image.blob;
  let downscaled = false;
  if (blob.size > MAX_BYTES) {
    try {
      blob = await shrinkToFit(blob);
      downscaled = true;
    } catch {
      return fail("too_large", "This image is over 10 MB and couldn't be shrunk.");
    }
  }

  let response;
  try {
    const form = new FormData();
    form.append("file", blob, "image");
    response = await fetch((await backendUrl()) + "/predict", { method: "POST", body: form });
  } catch {
    return fail("backend_offline", "DeepGuard backend isn't running. Start it with `uvicorn server:app`.");
  }

  let body = null;
  try {
    body = await response.json();
  } catch {
    // not JSON, handled below
  }
  if (!response.ok) {
    if (response.status === 400) return fail("unreadable", (body && body.detail) || "The backend couldn't read this image.");
    if (response.status === 413) return fail("too_large", "The backend rejected the image as too large.");
    return fail("backend_error", `Backend error ${response.status}` + (body && body.detail ? `: ${body.detail}` : ""));
  }
  if (!body || !Array.isArray(body.results)) return fail("bad_response", "The backend sent back something unexpected.");

  return { ok: true, srcUrl: info.srcUrl, results: body.results, downscaled };
}

// 1. fetch from the service worker (plain http/https and data: URLs).
// 2. if that fails (blob: URLs, images behind a login, hotlink protection), fetch from inside the
//    page, where the cookies and origin are right.
// The original file bytes are sent, never a canvas copy: re-encoding would destroy the invisible watermark.
async function getImageBlob(info, tab) {
  try {
    const r = await fetch(info.srcUrl);
    if (r.ok) return { ok: true, blob: await r.blob() };
  } catch {
    // fall through to the in-page fetch
  }

  if (tab && tab.id !== undefined) {
    try {
      const [injection] = await chrome.scripting.executeScript({
        target: { tabId: tab.id, frameIds: [info.frameId || 0] },
        func: fetchInPage,
        args: [info.srcUrl],
      });
      const result = injection && injection.result;
      if (result && result.dataUrl) {
        const blob = await (await fetch(result.dataUrl)).blob();
        return { ok: true, blob };
      }
    } catch {
      // restricted pages (chrome://, the web store) can't be scripted
    }
  }
  return fail("fetch_failed", "Couldn't read this image.");
}

// Runs inside the page, so it can't use anything from this file. Returns the bytes as a data: URL
// because executeScript results have to be plain serializable data.
async function fetchInPage(url) {
  try {
    const r = await fetch(url, { credentials: "include" });
    if (!r.ok) return null;
    const blob = await r.blob();
    return await new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve({ dataUrl: reader.result });
      reader.onerror = reject;
      reader.readAsDataURL(blob);
    });
  } catch {
    return null;
  }
}

// Only for images over the backend's size limit. Draws a smaller copy as JPEG, which loses the
// invisible watermark, so the card should say the watermark check was skipped (result.downscaled).
async function shrinkToFit(blob) {
  const bitmap = await createImageBitmap(blob);
  let scale = Math.sqrt(MAX_BYTES / blob.size) * 0.9;
  for (let attempt = 0; attempt < 5; attempt++) {
    const w = Math.max(1, Math.floor(bitmap.width * scale));
    const h = Math.max(1, Math.floor(bitmap.height * scale));
    const canvas = new OffscreenCanvas(w, h);
    canvas.getContext("2d").drawImage(bitmap, 0, 0, w, h);
    const out = await canvas.convertToBlob({ type: "image/jpeg", quality: 0.9 });
    if (out.size <= MAX_BYTES) return out;
    scale *= 0.7;
  }
  throw new Error("could not shrink under the limit");
}

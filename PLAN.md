# Sandbox + Backend — Plan

Goal: a local web page where I can drop in images, run them through the detection models, and
see each model's verdict side by side. The same HTTP API will later be called by the browser
extension, so the sandbox is the first client of the final product, not a throwaway.

```
                 +--------------------+
  Sandbox page ->|                    |-> watermark precheck
  (Sandbox/)     |  Backend (FastAPI) |-> logistic regression
                 |  Detector/         |-> CNN
  Extension   -> |                    |-> ensemble learning for both models
  (Extension/)   +--------------------+
```

---

## 1. Decisions

| Piece | Choice | Why |
|---|---|---|
| Backend | **FastAPI + uvicorn** | file uploads, JSON, auto-generated `/docs` page for poking the API, CORS is one line |
| Sandbox UI | **Plain HTML + JS, served by the backend** | no build step; same-origin so no CORS issues while testing, easy build with Claude |
| Model storage | **`joblib`** (sklearn) and **`.keras`** (CNN) in `Detector/models/` | train once, load at server start |
| Transport | `multipart` image upload now; `image_url` option later for the extension | extensions usually have a URL, not a file |

New dependencies: `pip install fastapi uvicorn python-multipart`.

## 2. Current state that blocks this

- `logisticRegression.py` downloads the dataset, trains, and prints at **import time**. It
  can't be imported by a server without retraining (~5 min).
- No model is saved to disk. `CNN.py` is 11 lines and doesn't train yet.
- `watermark.py` runs a directory loop at import time and refers to a `check_c2pa` function that
  doesn't exist yet.
- `predict_one` hard-resizes to 32x32. Fine for CIFAKE-trained models, but my own `Data/`
  images are full-size, so that resize is a deliberate, documented step in each model wrapper.

So step 1 is making the Detector importable. The sandbox comes after.

## 3. Folder layout

```
DeepGuardv2/
├── Detector/
│   ├── models/                 # saved models (gitignored)
│   ├── watermark.py            # functions only, demo loop under __main__
│   ├── logisticRegression.py   # train() + predict(); training under __main__
│   ├── CNN.py                  # same shape
│   ├── registry.py             # name -> predict function; the one place models are added
│   └── server.py               # FastAPI app
├── Sandbox/
│   ├── index.html
│   ├── app.js
│   └── style.css
└── Extension/                  # Chrome/Edge extension, calls the same API (section 7)
    ├── manifest.json
    ├── background.js           # service worker: context menu, fetches the image, calls the backend
    ├── content.js              # shows the result on the page, fallback image fetch
    ├── popup.html / popup.js   # toolbar popup: backend status, last results
    ├── options.html / options.js  # backend URL
    └── icons/                  # 16, 48, 128 px
```

## 4. Model interface

Every detector exposes the same function, so the server and the sandbox never special-case a model:

```python
predict(image_bytes: bytes) -> {
    "model":      "logreg",
    "verdict":    "FAKE" | "REAL" | "UNKNOWN",
    "prob_fake":  0.71 | None,       # None when the model doesn't give a probability
    "detail":     "free text",       # e.g. "too small for watermark check"
    "ms":         12
}
```

`registry.py` holds `MODELS = {"watermark": ..., "logreg": ..., "cnn": ...}`. Adding a model
later = write one function, add one line.

Taking `bytes` (not a file path) matches what the API and the extension will have in hand.

## 5. API

| Route | Does |
|---|---|
| `GET /health` | `{ok: true, models: [...]}` — sandbox and extension use it to detect the backend |
| `GET /models` | list of available model names |
| `POST /predict` | form: `file` (image), optional `models` (comma list, default all) → `{results: [...]}` |
| `POST /predict` (later) | JSON `{image_url}` — the server fetches the image |
| `POST /evaluate` | walks `Data/Real` and `Data/Fake` server-side, returns per-model accuracy + confusion matrix |
| `GET /` | serves `Sandbox/index.html` |

CORS: allow `*` for local dev now; before the extension ships, restrict to the extension's origin.
Reject files over ~10 MB and non-images with a clear 4xx message.

## 6. Sandbox page, built by Claude

- Drop zone / file picker / paste, multiple images.
- Per image: thumbnail + one row per model with verdict badge, probability bar, ms, detail text.
- **Ground-truth toggle** (Real / Fake) per image, optional. When set, each model's result shows
  correct/wrong, and a footer shows running accuracy per model. This is the main reason the
  sandbox exists: comparing models on my own images.
- **"Evaluate Data/"** button: calls `/evaluate` and shows a per-model accuracy table and confusion
  matrix (labels come from the folder), so the browser doesn't upload hundreds of files.
- Backend-offline banner driven by `/health`.

## 7. Browser extension

Goal: right-click an image on any web page → **"Detect image"** → a small card appears on the page
with the verdict. Same backend and API as the sandbox, nothing model-specific in the extension.

**Target:** Chrome and Edge (both Chromium, Manifest V3). Firefox later if wanted; the code is
nearly the same but its manifest differs.

### 7.1 How the user triggers it

The menu entry comes from the `contextMenus` API with `contexts: ["image"]`, which shows
**"Detect image"** when right-clicking an image. Two limits worth knowing:

- Chrome has no "selected image" context. Highlighting an image (dragging a selection over it) and
  then right-clicking *on the image* works, but the entry only appears if the right-click lands on the
  image itself. Right-clicking empty space inside a selection won't show it.
- Images set with CSS `background-image` aren't `<img>` elements, so Chrome doesn't treat them as
  images and the entry won't appear. Not supported in v1.

Optional extras for later: a keyboard shortcut (`commands`) for the image under the cursor, and a
"Detect all images on this page" button in the toolbar popup.

### 7.2 Flow

```
right-click image -> "Detect image"
  background.js: onClicked gives info.srcUrl and the tab
  1. tell content.js to show "Analyzing..." next to the image
  2. get the image bytes (7.3)
  3. POST /predict (multipart, field "file") to the backend
  4. send the results to content.js, which draws the card (7.4)
```

**All backend calls go through `background.js`, never `content.js`.** A content script runs inside the
page, so a request from an https site to `http://127.0.0.1` would be blocked (CORS, mixed content, and
Chrome's local-network rules). The service worker has host permission and isn't subject to that.

### 7.3 Getting the image bytes

Send the **original file bytes**, not a screenshot or canvas copy. Re-encoding destroys the invisible
watermark `watermark.py` looks for and changes the pixels the models see.

1. `fetch(srcUrl)` from the service worker (works for ordinary http/https and `data:` URLs).
2. If that fails (`blob:` URLs, images behind a login, hotlink protection), ask `content.js` to
   fetch it from inside the page, where the cookies and origin are right, and send the bytes back.
3. If both fail, show "Couldn't read this image".

This means the extension uploads the file itself, so the backend's `image_url` route (old step 8) is **not
needed for the extension**, and the server never fetches arbitrary URLs. That removes the biggest
security item in section 9. The route can stay as a separate nice-to-have.

Over 10 MB (the backend limit): downscale to fit and tell the user the watermark check was skipped.

### 7.4 What the user sees

`content.js` draws a small card next to the image inside a **Shadow DOM**, so the page's CSS can't
break it and ours can't leak into the page.

- **Loading:** "Analyzing…" with a spinner.
- **Result:** a big verdict badge (FAKE / REAL / UNKNOWN) with the probability. The headline comes from the
  `ensemble` result once it exists, otherwise from `cnn`. A "details" toggle lists every model's
  verdict, probability and ms, same as the sandbox.
- **Error:** backend offline → "DeepGuard backend isn't running. Start it with `uvicorn server:app`" and a
  link to the options page. Unreadable image, too large and bad response each get their own message.
- Closes on ✕, on Esc, or when the page is clicked elsewhere. Wording is "likely fake", never "is
  fake": the models are not certain and the card should say so.

### 7.5 Other pieces needed

| Piece | What | Why |
|---|---|---|
| `manifest.json` | MV3; permissions `contextMenus`, `storage`, `activeTab`, `scripting`; `host_permissions` for `<all_urls>` (to fetch any image) | the extension can't read other sites' images or call the backend without them |
| Toolbar popup | backend status (calls `/health`), the last few results, link to options | quick way to see why nothing works |
| Options page | backend URL, default `http://127.0.0.1:8000`, saved in `chrome.storage.sync`, with a "Test connection" button | the URL changes if the backend is hosted later |
| Icons | 16, 48, 128 px | required for the store and the toolbar |
| Menu setup | create the context menu in `runtime.onInstalled` | MV3 service workers sleep, so setup must not depend on the worker staying alive |
| Backend CORS | allow only `chrome-extension://<id>`, with a fixed id (set `"key"` in the manifest) | replaces `*` before shipping; the extension id is otherwise random per install |
| Privacy note | one short text in the options page and README: images are sent to the backend | the user should know where their images go |

### 7.6 Testing it

Load it with `chrome://extensions` → Developer mode → **Load unpacked** → pick `Extension/`. Reload
the extension after each edit. Test list (put these on one local test page):

- normal `<img>` jpg / png / webp, a lazy-loaded image, an image inside `<picture>`
- an image on a site that blocks hotlinking, and one behind a login
- a `data:` URL image and a `blob:` URL image
- tiny (under 256 px, watermark skipped) and huge (over 10 MB) images, an SVG, an animated GIF
- backend stopped, backend slow, backend returns an error
- an https page calling the local backend (the real situation the service worker has to handle)

### 7.7 Not for v1

Video (frame sampling later), CSS background images, Firefox, caching results per image (cheap to add:
key by a hash of the bytes), and a hosted backend. A localhost backend only works on my own machine; for
other people to use the extension the backend must be hosted over https and the extension needs a
Chrome Web Store listing with a privacy policy.

## 8. Build order

| Step | What | Done when |
|---|---|---|
| 1 | Refactor `watermark.py`, `logisticRegression.py`: functions + `__main__` guard, remove import-time work | `import logisticRegression` returns instantly |
| 2 | `train()` saves to `Detector/models/`; `predict(bytes)` loads from there | predict works on a fresh Python shell |
| 3 | `registry.py` + `server.py` with `/health`, `/models`, `/predict` | `/docs` page can classify an image |
| 4 | Sandbox: upload, thumbnails, per-model result rows | drop an image, see all verdicts |
| 5 | Ground-truth toggle and running accuracy | I can compare models by hand on ~20 images |
| 6 | `/evaluate` over `Data/` with confusion matrices | one click gives per-model accuracy on my own set |
| 7 | CNN trained, saved, and added to the registry | appears in the sandbox with no UI changes |
| 8 | CORS limited to the extension's origin; `image_url` route only if wanted (not needed by the extension, see 7.3) | backend only answers the extension and the sandbox |
| 9 | Extension skeleton: `manifest.json`, icons, loads unpacked, "Detect image" appears on images | right-click an image and see the entry |
| 10 | `background.js`: get bytes (with the content-script fallback), `POST /predict`, log the result | result in the service worker console |
| 11 | `content.js`: Shadow DOM card with loading, result and error states | verdict shows next to the image on a real site |
| 12 | Popup and options page: backend URL, health status, last results | I can point it at another backend without editing code |
| 13 | Run the test list in 7.6 and fix what breaks | every case either works or shows a clear message |
| 14 | README section and a short demo: start backend, load extension, detect an image | someone else can set it up from the README |

## 9. Open questions / risks

- **Resolution mismatch.** Models trained on 32x32 CIFAKE will likely do poorly on full-size images
  after downscaling. The sandbox is how I find out; expect to retrain on `Data/` later.
- **`Data/` is small and uneven** (README: still need real samples). Accuracy numbers from it are
  noisy until there are more real images, so show counts next to every accuracy figure.
- **Latency for the extension.** Uploading full-size images on every hover is slow; the extension
  should probably downscale or send URLs. Worth measuring in the sandbox (`ms` field) first.
- **Video** is out of scope for now (images only, per README); the API shape leaves room for
  frame-sampling later.
- **Security.** Because the extension uploads the bytes itself (7.3), the server doesn't need to fetch URLs.
  If an `image_url` route is added anyway, restrict it to http(s), cap size and timeout, and block private
  IP ranges. Bind the server to `127.0.0.1` only until it is hosted.
- **Extension permissions.** `<all_urls>` is a broad permission and store review looks at it
  closely. It is needed to fetch images from any site; keep the justification written down.
- **Only works while the backend is running.** Until the backend is hosted, the extension is a local
  demo. The offline message must make that obvious.
- **Wrong answers shown with confidence.** The models are weak on full-size images (see the first
  risk). The card wording and the probability display must not overstate certainty.

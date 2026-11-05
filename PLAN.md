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
  (later)        +--------------------+
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
└── Extension/                  # later, calls the same API
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

## 7. Build order

| Step | What | Done when |
|---|---|---|
| 1 | Refactor `watermark.py`, `logisticRegression.py`: functions + `__main__` guard, remove import-time work | `import logisticRegression` returns instantly |
| 2 | `train()` saves to `Detector/models/`; `predict(bytes)` loads from there | predict works on a fresh Python shell |
| 3 | `registry.py` + `server.py` with `/health`, `/models`, `/predict` | `/docs` page can classify an image |
| 4 | Sandbox: upload, thumbnails, per-model result rows | drop an image, see all verdicts |
| 5 | Ground-truth toggle and running accuracy | I can compare models by hand on ~20 images |
| 6 | `/evaluate` over `Data/` with confusion matrices | one click gives per-model accuracy on my own set |
| 7 | CNN trained, saved, and added to the registry | appears in the sandbox with no UI changes |
| 8 | `image_url` route and CORS tightening | ready for the extension to start |

## 8. Open questions / risks

- **Resolution mismatch.** Models trained on 32x32 CIFAKE will likely do poorly on full-size images
  after downscaling. The sandbox is how I find out; expect to retrain on `Data/` later.
- **`Data/` is small and uneven** (README: still need real samples). Accuracy numbers from it are
  noisy until there are more real images, so show counts next to every accuracy figure.
- **Latency for the extension.** Uploading full-size images on every hover is slow; the extension
  should probably downscale or send URLs. Worth measuring in the sandbox (`ms` field) first.
- **Video** is out of scope for now (images only, per README); the API shape leaves room for
  frame-sampling later.
- **Security.** The server will fetch arbitrary URLs in step 8; restrict to http(s), cap size and
  timeout, and block private IP ranges. Bind to `127.0.0.1` only until then.

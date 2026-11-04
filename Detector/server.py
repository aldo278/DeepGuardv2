# run from the Detector folder: uvicorn server:app --host 127.0.0.1 --port 8000   (docs at http://127.0.0.1:8000/docs)
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path

import registry

MAX_BYTES = 10 * 1024 * 1024

app = FastAPI(title="DeepGuard")

# wide open for local development, restrict to the extension's origin before shipping
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/health")
def health():
    return {"ok": True, "models": list(registry.MODELS)}


@app.get("/models")
def models():
    return {"models": list(registry.MODELS)}


# models is an optional comma separated list like "logreg,watermark", default is all of them
@app.post("/predict")
async def predict(file: UploadFile = File(...), models: str = Form("")):
    names = [m.strip() for m in models.split(",") if m.strip()]
    unknown = [m for m in names if m not in registry.MODELS]
    if unknown:
        raise HTTPException(400, f"unknown model(s): {', '.join(unknown)}")

    data = await file.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "image is over 10 MB")
    try:
        results = registry.run(data, names or None)
    except ValueError:
        raise HTTPException(400, "file is not a readable image")
    return {"filename": file.filename, "results": results}


DATA_DIR = Path(__file__).parent.parent / "Data"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


# runs every image in Data/Real and Data/Fake through the models (labels come from the folder name) and returns, per model, a confusion matrix.
# counts[truth][verdict] holds how many images of that true label got that verdict. UNKNOWN and ERROR are kept as their own columns and left out of accuracy
@app.post("/evaluate")
def evaluate(models: str = Form("")):
    names = [m.strip() for m in models.split(",") if m.strip()]
    unknown = [m for m in names if m not in registry.MODELS]
    if unknown:
        raise HTTPException(400, f"unknown model(s): {', '.join(unknown)}")

    report = {m: {"counts": {t: {} for t in ("REAL", "FAKE")}, "ms": 0} for m in names or registry.MODELS}
    totals = {"REAL": 0, "FAKE": 0}
    skipped = []
    for truth in ("REAL", "FAKE"):
        for path in sorted((DATA_DIR / truth.capitalize()).glob("*")):
            if path.suffix.lower() not in IMAGE_EXTS:
                continue
            try:
                results = registry.run(path.read_bytes(), names or None)
            except ValueError:
                skipped.append(path.name)
                continue
            totals[truth] += 1
            for r in results:
                m = report[r["model"]]
                m["counts"][truth][r["verdict"]] = m["counts"][truth].get(r["verdict"], 0) + 1
                m["ms"] += r["ms"]

    for m in report.values():
        c = m["counts"]
        right = c["REAL"].get("REAL", 0) + c["FAKE"].get("FAKE", 0)
        wrong = c["REAL"].get("FAKE", 0) + c["FAKE"].get("REAL", 0)
        m["scored"] = right + wrong
        m["accuracy"] = right / (right + wrong) if right + wrong else None
        m["avg_ms"] = round(m.pop("ms") / max(sum(totals.values()), 1))
    return {"totals": totals, "skipped": skipped, "models": report}


# the sandbox page. Mounted last so it only answers paths the API routes above do not
app.mount("/", StaticFiles(directory=Path(__file__).parent.parent / "Sandbox", html=True), name="sandbox")

"""File + manifest handling for the annotator. No GUI code here."""
import csv
import hashlib
import os
import shutil
import uuid
from datetime import datetime
from pathlib import Path

DATA_DIR = Path(__file__).parent / "Data"
MANIFEST = DATA_DIR / "annotations.csv"

COLUMNS = [
    "id", "file", "label", "source", "notes", "tags", "origin_url", "original_name",
    "width", "height", "format", "size_bytes", "has_exif", "watermark", "sha256", "added_at",
]


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_manifest():
    if not MANIFEST.exists():
        return []
    with open(MANIFEST, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_manifest(rows):
    """Write to a temp file then rename, so a crash can't leave a half-written CSV."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    tmp = MANIFEST.with_suffix(".csv.tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, MANIFEST)


def find_duplicate(sha):
    for row in read_manifest():
        if row["sha256"] == sha:
            return row
    return None


def counts():
    result = {"REAL": 0, "FAKE": 0}
    for row in read_manifest():
        if row["label"] in result:
            result[row["label"]] += 1
    return result


def sources():
    return sorted({r["source"] for r in read_manifest() if r["source"]})


def save_image(src_path, label, info, source="", notes="", tags="", origin_url=""):
    """Copy src_path into Data/<Label>/ under a uuid name and append a manifest row.

    `info` is the dict from checks.analyze(). Returns the manifest row.
    """
    label = label.upper()
    if label not in ("REAL", "FAKE"):
        raise ValueError("label must be REAL or FAKE")
    src_path = Path(src_path)
    image_id = str(uuid.uuid4())
    folder = DATA_DIR / label.capitalize()
    folder.mkdir(parents=True, exist_ok=True)
    ext = src_path.suffix.lower() or ".png"
    dest = folder / f"{image_id}{ext}"
    shutil.copy2(src_path, dest)

    row = {
        "id": image_id,
        "file": f"{label.capitalize()}/{dest.name}",
        "label": label,
        "source": source,
        "notes": notes,
        "tags": tags,
        "origin_url": origin_url,
        "original_name": src_path.name,
        "width": info.get("width"),
        "height": info.get("height"),
        "format": info.get("format"),
        "size_bytes": info.get("size_bytes"),
        "has_exif": info.get("has_exif"),
        "watermark": info.get("watermark"),
        "sha256": info.get("sha256") or sha256_of(src_path),
        "added_at": datetime.now().isoformat(timespec="seconds"),
    }
    rows = read_manifest()
    rows.append(row)
    try:
        write_manifest(rows)
    except Exception:
        dest.unlink(missing_ok=True)
        raise
    return row


def undo_last():
    """Delete the most recently added file and its manifest row. Returns the row or None."""
    rows = read_manifest()
    if not rows:
        return None
    row = rows.pop()
    write_manifest(rows)
    (DATA_DIR / row["file"]).unlink(missing_ok=True)
    return row

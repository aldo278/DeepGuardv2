"""Build a CIFAKE-shaped train/test folder from the annotated set.

    python Annotation/export.py [--size 32] [--test-frac 0.2] [--seed 42] [--skip-tag unsure]
"""
import argparse
import random
import shutil
from pathlib import Path

from PIL import Image

import storage

EXPORT_DIR = Path(__file__).parent / "Export"


def export(size=None, test_frac=0.2, seed=42, skip_tag="unsure", out_dir=EXPORT_DIR):
    """Stratified split by image. Returns {"train": n, "test": n}."""
    rows = storage.read_manifest()
    seen, unique = set(), []
    for row in rows:  # one row per hash, so duplicates never cross train/test
        tags = [t.strip().lower() for t in row["tags"].split(",")]
        if row["sha256"] in seen or (skip_tag and skip_tag in tags):
            continue
        seen.add(row["sha256"])
        unique.append(row)

    if out_dir.exists():
        shutil.rmtree(out_dir)
    rng = random.Random(seed)
    written = {"train": 0, "test": 0}
    for label in ("REAL", "FAKE"):
        group = [r for r in unique if r["label"] == label]
        rng.shuffle(group)
        n_test = round(len(group) * test_frac)
        for i, row in enumerate(group):
            split = "test" if i < n_test else "train"
            dest_dir = out_dir / split / label
            dest_dir.mkdir(parents=True, exist_ok=True)
            src = storage.DATA_DIR / row["file"]
            if size:
                with Image.open(src) as img:
                    img.convert("RGB").resize((size, size), Image.LANCZOS).save(
                        dest_dir / f"{row['id']}.png")
            else:
                shutil.copy2(src, dest_dir / src.name)
            written[split] += 1
    return written


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=None, help="resize to NxN (e.g. 32)")
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-tag", default="unsure")
    a = ap.parse_args()
    print(export(a.size, a.test_frac, a.seed, a.skip_tag))

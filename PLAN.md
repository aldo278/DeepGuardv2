# Annotation App — Plan

A small desktop app for building my own Real/Fake image dataset. I drop images in, label them
Real or Fake, add a short optional annotation, and the app files them into `Annotation/Data/Real` or
`Annotation/Data/Fake` and records everything in a manifest the Detector models can read later.

Why: CIFAKE images are 32x32, too small for the watermark detector and limited for the models.
A hand-labeled set of full-size images fixes both problems (see README → Detector).

---

## 1. Tech choices

| Piece | Choice | Why |
|---|---|---|
| GUI | **tkinter** (ships with Python) + **tkinterdnd2** | tkinter has no native drag-and-drop on Windows; `tkinterdnd2` adds it with one `pip install` |
| Image loading / preview | **Pillow** (already in `.venv`) | thumbnails, format/size info, EXIF, clipboard paste via `ImageGrab` |
| Watermark check | **imwatermark** (already in `.venv`) | reuse the idea from `Detector/watermark.py`, now on full-size images |
| Manifest | **CSV** (`Annotation/Data/annotations.csv`) | opens in Excel, loads in one line with pandas/csv, easy to diff |
| Duplicate detection | **SHA-256** of file bytes (`hashlib`, stdlib) | stops the same image being added twice or under both labels |

New dependency: `pip install tkinterdnd2` (that's it).

---

## 2. Folder layout

```
Annotation/
├── PLAN.md
├── app.py               # entry point: python Annotation/app.py
├── storage.py           # copy/rename files, hashing, manifest read/write (no GUI code)
├── checks.py            # image validation, metadata, watermark check
├── export.py            # build a CIFAKE-style train/test set for the Detector
└── Data/
    ├── Real/            # <uuid>.<ext>
    ├── Fake/            # <uuid>.<ext>
    └── annotations.csv  # one row per image
```

Keeping file logic in `storage.py` (separate from the GUI) means it can be tested on its own
and reused by the Detector scripts.

---

## 3. Window layout

```
+---------------------------------------------------------------+
|  DeepGuard Annotator                    Real: 42   Fake: 37    |
+----------------------+----------------------------------------+
|  QUEUE (3)           |                                        |
|  > img_001.png       |          [ large image preview ]       |
|    photo.jpg         |                                        |
|    gen_17.webp       |   1024x1024 · PNG · 1.2 MB · no EXIF    |
|                      |   watermark: none found                 |
|  +----------------+  |   ⚠ duplicate of Fake/3f2a...png        |
|  |  Drop images   |  +----------------------------------------+
|  |  or folders    |  |  Label:   ( ) Real   ( ) Fake           |
|  |  here / Ctrl+V |  |  Source:  [Midjourney        v]         |
|  +----------------+  |  Notes:   [extra fingers, smooth skin ] |
|  [Browse...]         |  Tags:    [face, portrait          ]    |
|                      |                                        |
|                      |  [Save (Enter)] [Skip (S)] [Undo (Ctrl+Z)] |
+----------------------+----------------------------------------+
|  Saved photo.jpg → Data/Real/9c1e...jpg                        |
+---------------------------------------------------------------+
```

---

## 4. Core features (MVP)

### 4.1 Getting images in
- **Drag and drop** files *or whole folders* onto the drop zone (folders are scanned for images).
- **Browse…** button as a fallback file picker.
- **Ctrl+V paste** from the clipboard (screenshots, "Copy image" from a browser) — saved as PNG.
- Dropped images go into a **queue**; the app shows them one at a time.

### 4.2 Labeling and annotation
Per image:
- **Label** — Real / Fake (required).
- **Source** — editable dropdown remembering past values: e.g. `Camera`, `Phone`, `Midjourney`,
  `DALL·E`, `Stable Diffusion`, `Gemini`, `Unknown`. For Fake images this is the generator.
- **Notes** — one line of free text (what gives it away, where it came from).
- **Tags** — optional comma-separated (`face`, `landscape`, `text`, `CGI`…). The `CGI` tag helps
  answer the open README question about separating CGI from AI images.
- **Origin URL** — optional, for citing where an image was found.

### 4.3 Filing the image
On **Save**:
1. Validate the file opens as an image (reject corrupt/non-image files with a message).
2. Hash it; if the hash is already in the manifest, warn and show where it is.
3. **Copy** (never move) it to `Data/Real/` or `Data/Fake/` as `<uuid4>.<ext>` — same naming as
   `Detector/TestImages/`. The original filename is kept in the manifest, not the filename,
   so the name can't leak the label to a model.
4. Append a row to `annotations.csv`.
5. Advance to the next image in the queue.

Writes are done by writing to a temp file then renaming, so a crash can't leave a half-written CSV.

### 4.4 Manifest columns (`annotations.csv`)
```
id, file, label, source, notes, tags, origin_url, original_name,
width, height, format, size_bytes, has_exif, watermark, sha256, added_at
```
- `file` is relative to `Annotation/Data` (e.g. `Fake/3f2a….png`) so the folder can move.
- `label` is `REAL`/`FAKE` to match CIFAKE and `load_split()` in `logisticRegression.py`.

### 4.5 Auto-filled info (shown in the preview panel, saved in the manifest)
- Width × height, format, file size.
- **EXIF present?** and camera make/model if any — real phone/camera photos usually have it,
  generated images usually don't. Useful as a hint, not proof.
- **Watermark check** with `imwatermark` — the thing that failed on 32x32 CIFAKE images.
- **PNG text chunks** — some generators (e.g. Stable Diffusion web UIs) write the prompt into
  PNG metadata; if found, show it and suggest `Fake`.

### 4.6 Keyboard shortcuts
| Key | Action |
|---|---|
| `R` / `F` | set label Real / Fake |
| `Enter` | save and go to next |
| `S` | skip (leave in queue end / drop from queue) |
| `Ctrl+Z` | undo last save (deletes the copied file + its manifest row) |
| `Ctrl+V` | paste image from clipboard |

### 4.7 Status
- Running **Real / Fake counts** in the header so the dataset stays balanced.
- Status bar with the last action and where the file went.

---

## 5. Extra features (after MVP, in priority order)

1. **Browse / edit tab** — a table of the manifest with thumbnails; filter by label/source/tag;
   change a label (moves the file between folders and updates the row) or delete an entry.
2. **Export for the Detector** (`export.py`, also a button) — builds a CIFAKE-shaped folder:
   ```
   Annotation/Export/train/{REAL,FAKE}/   Annotation/Export/test/{REAL,FAKE}/
   ```
   with a stratified train/test split (e.g. 80/20, fixed seed), optionally resized to 32x32
   so `load_split()` and the CNN can use it with **no code changes**, or kept full-size for
   future models. Split by image, so duplicates never cross train/test (the hash check helps here).
3. **Model hint** — after a model is saved to disk, show "logistic regression says: 71% fake"
   next to the preview. Shown *after* I pick a label (or hidden by default) so it doesn't bias
   my labeling. Also lets me collect the images the model gets wrong — the most useful ones.
4. **Near-duplicate warning** — perceptual hash (`imagehash` package) to catch resized/re-saved
   copies the SHA-256 check misses.
5. **Integrity check** button — find files on disk missing from the CSV and CSV rows whose
   file is gone.
6. **Stats view** — counts per source/generator and per tag, image size distribution.

---

## 6. Build order

| Step | What | Done when |
|---|---|---|
| 1 | `storage.py`: hash, copy-with-uuid, manifest append/read/undo | can file an image from a Python shell |
| 2 | `checks.py`: validate, size/format, EXIF, PNG text, watermark | returns a dict for any image path |
| 3 | `app.py`: window, drop zone, queue, preview, label/notes form, Save | can label 10 images end-to-end |
| 4 | Shortcuts, Ctrl+V paste, folder drops, counters, undo | labeling one image takes a couple of seconds |
| 5 | Duplicate warning + error messages for bad files | dropping junk or repeats never breaks anything |
| 6 | `export.py` + button | `load_split(Path("Annotation/Export/train"))` runs |
| 7 | Extras from §5 as time allows | — |

---

## 7. Notes / decisions

- **Copy, don't move** originals — safer; I can delete the source myself afterwards.
- **Git:** images can get large. Add `Annotation/Data/Real/`, `Annotation/Data/Fake/` and
  `Annotation/Export/` to `.gitignore` but **commit `annotations.csv`** so the labels are versioned.
  (Folders need a `.gitkeep` to stay in the repo.)
- **Label honesty:** if I'm not sure where an image came from, don't guess — skip it or set
  source `Unknown` and tag `unsure`, so it can be filtered out of training.
- **Folder name case:** `Data/Real` vs CIFAKE's `REAL` doesn't matter on Windows, but the export
  step uses uppercase so the Detector code works on any OS.


The app had some restrictions so I may just place the images in the data directory with 2 subfolder, Fake, and Real manually. I have added some data to the folder, delete in lab.

claude session with...
Resume this session with:
claude --resume da2b3c65-3449-4841-8f22-7fbb9f236377

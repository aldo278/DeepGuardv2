"""DeepGuard Annotator: label images Real/Fake and file them into Annotation/Data.

    python Annotation/app.py
"""
import sys
import tempfile
import tkinter as tk
import uuid
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageGrab, ImageTk

sys.path.insert(0, str(Path(__file__).parent))
import checks  # noqa: E402
import export  # noqa: E402
import storage  # noqa: E402

try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
    Root = TkinterDnD.Tk
except ImportError:  # app still works with Browse / paste
    DND_FILES, Root = None, tk.Tk

DEFAULT_SOURCES = ["Camera", "Phone", "Midjourney", "DALL·E", "Stable Diffusion", "Gemini", "Unknown"]
PASTE_DIR = Path(tempfile.gettempdir()) / "deepguard_paste"


class App(Root):
    def __init__(self):
        super().__init__()
        self.title("DeepGuard Annotator")
        self.geometry("1100x720")
        self.queue = []          # list of Paths still to label
        self.info = None         # analysis of the current image
        self.photo = None        # keep a reference so Tk doesn't drop the image
        self.label_var = tk.StringVar()
        self.source_var = tk.StringVar()
        self.notes_var = tk.StringVar()
        self.tags_var = tk.StringVar()
        self.url_var = tk.StringVar()
        self._build()
        self._bind_keys()
        self.refresh_counts()
        self.show_current()

    # ---------- layout ----------
    def _build(self):
        header = ttk.Frame(self, padding=8)
        header.pack(fill="x")
        ttk.Label(header, text="DeepGuard Annotator", font=("Segoe UI", 14, "bold")).pack(side="left")
        self.counts_lbl = ttk.Label(header, font=("Segoe UI", 11))
        self.counts_lbl.pack(side="right")
        ttk.Button(header, text="Export…", command=self.do_export).pack(side="right", padx=12)

        body = ttk.Frame(self)
        body.pack(fill="both", expand=True)

        left = ttk.Frame(body, padding=8, width=260)
        left.pack(side="left", fill="y")
        left.pack_propagate(False)
        self.queue_lbl = ttk.Label(left, text="QUEUE (0)", font=("Segoe UI", 10, "bold"))
        self.queue_lbl.pack(anchor="w")
        self.listbox = tk.Listbox(left, activestyle="none", exportselection=False)
        self.listbox.pack(fill="both", expand=True, pady=4)
        self.listbox.bind("<<ListboxSelect>>", self.on_select)
        self.drop = tk.Label(left, text="Drop images or folders here\nor press Ctrl+V",
                             relief="groove", height=5, bg="#f2f2f2")
        self.drop.pack(fill="x", pady=4)
        if DND_FILES:
            self.drop.drop_target_register(DND_FILES)
            self.drop.dnd_bind("<<Drop>>", self.on_drop)
            self.listbox.drop_target_register(DND_FILES)
            self.listbox.dnd_bind("<<Drop>>", self.on_drop)
        else:
            self.drop.config(text="Drag-drop unavailable\n(pip install tkinterdnd2)\nuse Browse or Ctrl+V")
        ttk.Button(left, text="Browse…", command=self.browse).pack(fill="x")

        right = ttk.Frame(body, padding=8)
        right.pack(side="left", fill="both", expand=True)
        self.preview = tk.Label(right, bg="#222", text="No image", fg="#aaa")
        self.preview.pack(fill="both", expand=True)
        self.meta_lbl = ttk.Label(right, text="")
        self.meta_lbl.pack(anchor="w", pady=(6, 0))
        self.wm_lbl = ttk.Label(right, text="")
        self.wm_lbl.pack(anchor="w")
        self.warn_lbl = ttk.Label(right, text="", foreground="#b45309", wraplength=700)
        self.warn_lbl.pack(anchor="w")

        form = ttk.Frame(right, padding=(0, 8))
        form.pack(fill="x")
        form.columnconfigure(1, weight=1)
        ttk.Label(form, text="Label:").grid(row=0, column=0, sticky="w")
        radios = ttk.Frame(form)
        radios.grid(row=0, column=1, sticky="w")
        ttk.Radiobutton(radios, text="Real (R)", value="REAL", variable=self.label_var).pack(side="left")
        ttk.Radiobutton(radios, text="Fake (F)", value="FAKE", variable=self.label_var).pack(side="left", padx=12)
        ttk.Label(form, text="Source:").grid(row=1, column=0, sticky="w")
        self.source_box = ttk.Combobox(form, textvariable=self.source_var)
        self.source_box.grid(row=1, column=1, sticky="ew", pady=2)
        for r, (text, var) in enumerate([("Notes:", self.notes_var), ("Tags:", self.tags_var),
                                         ("Origin URL:", self.url_var)], start=2):
            ttk.Label(form, text=text).grid(row=r, column=0, sticky="w")
            ttk.Entry(form, textvariable=var).grid(row=r, column=1, sticky="ew", pady=2)

        buttons = ttk.Frame(right)
        buttons.pack(fill="x")
        ttk.Button(buttons, text="Save (Enter)", command=self.save).pack(side="left")
        ttk.Button(buttons, text="Skip (Ctrl+S)", command=self.skip).pack(side="left", padx=6)
        ttk.Button(buttons, text="Undo (Ctrl+Z)", command=self.undo).pack(side="left")

        self.status = tk.StringVar(value="Ready")
        ttk.Label(self, textvariable=self.status, relief="sunken", anchor="w", padding=4).pack(fill="x", side="bottom")

    def _bind_keys(self):
        self.bind("<Return>", lambda e: self.save())
        self.bind("<Control-z>", lambda e: self.undo())
        self.bind("<Control-s>", lambda e: self.skip())
        self.bind("<Control-v>", lambda e: self.paste())
        # R / F / S would be swallowed while typing, so ignore them when a text field has focus
        for key, action in (("r", lambda: self.label_var.set("REAL")),
                            ("f", lambda: self.label_var.set("FAKE")),
                            ("s", self.skip)):
            self.bind(f"<KeyPress-{key}>", lambda e, a=action: self._typing(e) or a())

    @staticmethod
    def _typing(event):
        return isinstance(event.widget, (tk.Entry, ttk.Entry, ttk.Combobox))

    # ---------- queue ----------
    def add_paths(self, paths):
        added, bad = 0, 0
        for p in paths:
            p = Path(p)
            found = checks.scan_folder(p) if p.is_dir() else [p]
            for f in found:
                if checks.is_image_file(f) and f not in self.queue:
                    self.queue.append(f)
                    added += 1
                else:
                    bad += 1
        self.status.set(f"Added {added} image(s)" + (f", ignored {bad} non-image/duplicate file(s)" if bad else ""))
        self.refresh_queue()
        if self.info is None:
            self.show_current()

    def on_drop(self, event):
        self.add_paths(self.tk.splitlist(event.data))

    def browse(self):
        files = filedialog.askopenfilenames(
            filetypes=[("Images", " ".join(f"*{e}" for e in sorted(checks.IMAGE_EXTS)))])
        if files:
            self.add_paths(files)

    def paste(self):
        try:
            data = ImageGrab.grabclipboard()
        except Exception as e:
            self.status.set(f"Clipboard error: {e}")
            return
        if isinstance(data, Image.Image):
            PASTE_DIR.mkdir(exist_ok=True)
            path = PASTE_DIR / f"pasted_{uuid.uuid4().hex[:8]}.png"
            data.save(path, "PNG")
            self.add_paths([path])
        elif isinstance(data, list):  # files copied in Explorer
            self.add_paths(data)
        else:
            self.status.set("Clipboard has no image")

    def refresh_queue(self):
        self.listbox.delete(0, "end")
        for p in self.queue:
            self.listbox.insert("end", p.name)
        if self.queue:
            self.listbox.selection_set(0)
        self.queue_lbl.config(text=f"QUEUE ({len(self.queue)})")

    def on_select(self, _event):
        sel = self.listbox.curselection()
        if sel and sel[0] != 0:  # move the chosen image to the front
            self.queue.insert(0, self.queue.pop(sel[0]))
            self.refresh_queue()
            self.show_current()

    # ---------- display ----------
    def show_current(self):
        self.info = None
        self.warn_lbl.config(text="")
        while self.queue:
            path = self.queue[0]
            try:
                self.info = checks.analyze(path)
                break
            except ValueError as e:
                messagebox.showwarning("Bad file", str(e))
                self.queue.pop(0)
                self.refresh_queue()
        if self.info is None:
            self.preview.config(image="", text="No image — drop some in")
            self.photo = None
            self.meta_lbl.config(text="")
            self.wm_lbl.config(text="")
            return

        with Image.open(path) as img:
            img = img.convert("RGB")
            img.thumbnail((max(self.preview.winfo_width(), 400), max(self.preview.winfo_height(), 300)))
            self.photo = ImageTk.PhotoImage(img)
        self.preview.config(image=self.photo, text="")
        self.meta_lbl.config(text=checks.describe(self.info))
        self.wm_lbl.config(text=f"watermark: {self.info['watermark']}")

        warnings = []
        dup = storage.find_duplicate(self.info["sha256"])
        if dup:
            warnings.append(f"⚠ duplicate of {dup['file']} (labelled {dup['label']})")
        if self.info["prompt_text"]:
            warnings.append(f"Generator metadata found, probably Fake — {self.info['prompt_text']}")
        self.warn_lbl.config(text="\n".join(warnings))

        self.label_var.set("")
        self.notes_var.set("")
        self.tags_var.set("")
        self.url_var.set("")
        self.source_box["values"] = sorted(set(DEFAULT_SOURCES) | set(storage.sources()))
        self.focus_set()

    def refresh_counts(self):
        c = storage.counts()
        self.counts_lbl.config(text=f"Real: {c['REAL']}   Fake: {c['FAKE']}")

    # ---------- actions ----------
    def save(self):
        if self.info is None:
            return
        if not self.label_var.get():
            self.status.set("Pick Real or Fake first (R / F)")
            return
        dup = storage.find_duplicate(self.info["sha256"])
        if dup and not messagebox.askyesno(
                "Duplicate", f"This image is already saved as {dup['file']} ({dup['label']}).\nSave it again?"):
            return
        path = self.queue[0]
        try:
            row = storage.save_image(
                path, self.label_var.get(), self.info, self.source_var.get().strip(),
                self.notes_var.get().strip(), self.tags_var.get().strip(), self.url_var.get().strip())
        except Exception as e:
            messagebox.showerror("Save failed", str(e))
            return
        self.status.set(f"Saved {path.name} → Data/{row['file']}")
        self.queue.pop(0)
        self.refresh_queue()
        self.refresh_counts()
        self.show_current()

    def skip(self):
        if len(self.queue) > 1:
            self.queue.append(self.queue.pop(0))
            self.status.set("Skipped (moved to end of queue)")
        elif self.queue:
            self.status.set("Only one image left in the queue")
        self.refresh_queue()
        self.show_current()

    def undo(self):
        if not messagebox.askyesno("Undo", "Delete the last saved image and its manifest row?"):
            return
        row = storage.undo_last()
        self.status.set(f"Undid {row['original_name']}" if row else "Nothing to undo")
        self.refresh_counts()

    def do_export(self):
        try:
            n = export.export(size=32 if messagebox.askyesno(
                "Export", "Resize to 32x32 for the existing Detector models?\n(No keeps full size.)") else None)
        except Exception as e:
            messagebox.showerror("Export failed", str(e))
            return
        self.status.set(f"Exported {n['train']} train / {n['test']} test to {export.EXPORT_DIR}")


if __name__ == "__main__":
    App().mainloop()

"""Image validation and metadata for the annotator. No GUI code here."""
from pathlib import Path

from PIL import ExifTags, Image

import storage

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff"}

# PNG text chunks that generators are known to write prompts/params into
PROMPT_KEYS = {"parameters", "prompt", "workflow", "comment", "software", "description"}


def is_image_file(path):
    return Path(path).suffix.lower() in IMAGE_EXTS


def scan_folder(folder):
    return sorted(p for p in Path(folder).rglob("*") if p.is_file() and is_image_file(p))


def watermark_check(path):
    """Try the Stable Diffusion invisible watermark. Returns a short string."""
    try:
        import cv2
        from imwatermark import WatermarkDecoder

        img = cv2.imread(str(path))
        if img is None:
            return "unreadable"
        h, w = img.shape[:2]
        if h < 256 or w < 256:
            return "too small to check"
        decoded = WatermarkDecoder("bytes", 32).decode(img, "dwtDct")
        text = decoded.decode("utf-8", errors="ignore").strip("\x00")
        if text and text.isprintable():
            return f"found: {text}"
        return "none found"
    except Exception as e:  # watermark check is a hint, never block labeling
        return f"check failed ({type(e).__name__})"


def analyze(path):
    """Validate the image and return a dict of info. Raises ValueError for bad files."""
    path = Path(path)
    try:
        with Image.open(path) as img:
            img.verify()
        with Image.open(path) as img:
            img.load()
            info = {
                "width": img.width,
                "height": img.height,
                "format": img.format,
                "size_bytes": path.stat().st_size,
                "has_exif": False,
                "camera": "",
                "prompt_text": "",
            }
            exif = img.getexif()
            if exif:
                info["has_exif"] = True
                make = exif.get(ExifTags.Base.Make, "")
                model = exif.get(ExifTags.Base.Model, "")
                info["camera"] = f"{make} {model}".strip()
            for key, value in (getattr(img, "text", None) or {}).items():
                if key.lower() in PROMPT_KEYS:
                    info["prompt_text"] = f"{key}: {str(value)[:200]}"
                    break
    except Exception as e:
        raise ValueError(f"{path.name} is not a readable image ({e})") from e

    info["watermark"] = watermark_check(path)
    info["sha256"] = storage.sha256_of(path)
    return info


def describe(info):
    """One-line summary for the preview panel."""
    size_mb = info["size_bytes"] / (1024 * 1024)
    size = f"{size_mb:.1f} MB" if size_mb >= 0.1 else f"{info['size_bytes'] / 1024:.0f} KB"
    exif = f"EXIF: {info['camera'] or 'yes'}" if info["has_exif"] else "no EXIF"
    return f"{info['width']}x{info['height']} · {info['format']} · {size} · {exif}"

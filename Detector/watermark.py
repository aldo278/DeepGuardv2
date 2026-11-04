from imwatermark import WatermarkDecoder
import cv2 
import numpy as np
import time

# before adding any ML models, we check for a hidden watermark, if none we do LR and CNN


# this function will be specifically for SD, add list for different watermark strings
def check_sd_watermark(image):
    img = cv2.imread(image)
    if img is None:
        return None 
    
    h, w = img.shape[:2]
    if h < 256 or w < 256:
        return None

    decoder = WatermarkDecoder('bytes', 32) # SD uses 32. other models like gemini might be different
    watermark = decoder.decode(img, 'dwtDct')

    # SD's watermark string, add other later
    watermark_strings = []

    detected = watermark.decode('utf-8', errors='ignore')
    return detected


def watermark_precheck(img):
    c2pa_result = check_c2pa(img)

    if c2pa_result:
        return {"source": "c2pa", "verdict": "FAKE", "detail": c2pa_result}

    sd_result = check_sd_watermark(img)
    if sd_result and "StableDiffusion" in sd_result:
        return {"source": "sd_watermark", "verdict": "FAKE"}

    return {"source": None, "verdict": "unknown"}


# the shared detector interface: image file bytes in, result dict out (see PLAN.md section 4)
# SD hides "StableDiffusionV1" (17 bytes = 136 bits) with dwtDct. check_sd_watermark above decodes only 32 bits, which can never contain that string,
# so this version decodes 136 bits. It can only say FAKE (watermark found) or UNKNOWN, a missing watermark doesn't make an image real.
# c2pa isn't included yet (check_c2pa doesn't exist)
def predict(image_bytes):
    start = time.perf_counter()
    img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("not a readable image")
    h, w = img.shape[:2]
    if h < 256 or w < 256:
        verdict, detail = "UNKNOWN", f"{w}x{h} is too small for the watermark check (needs 256x256)"
    else:
        text = WatermarkDecoder("bytes", 136).decode(img, "dwtDct").decode("utf-8", errors="ignore")
        if "StableDiffusion" in text:
            verdict, detail = "FAKE", "Stable Diffusion watermark found"
        else:
            verdict, detail = "UNKNOWN", "no Stable Diffusion watermark found"
    return {
        "model": "watermark",
        "verdict": verdict,
        "prob_fake": 1.0 if verdict == "FAKE" else None,
        "detail": detail,
        "ms": round((time.perf_counter() - start) * 1000),
    }


if __name__ == "__main__":
    import os
    import kagglehub

    # demo: run the SD watermark check on a few CIFAKE fake images
    image_dir = "C:\\Users\\aantanolopez\\.cache\\kagglehub\\datasets\\birdy654\\cifake-real-and-ai-generated-synthetic-images\\versions\\3\\test\\FAKE"
    for fname in os.listdir(image_dir)[:20]:
        result = check_sd_watermark(os.path.join(image_dir, fname))
        print(f"{fname} -----> {result}")
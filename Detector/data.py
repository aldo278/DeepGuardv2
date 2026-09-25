import kagglehub

path = kagglehub.dataset_download("birdy654/cifake-real-and-ai-generated-synthetic-images")



from imwatermark import WatermarkDecoder
import cv2 
import numpy as np

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


import os 
image_dir = "C:\\Users\\aantanolopez\\.cache\\kagglehub\\datasets\\birdy654\\cifake-real-and-ai-generated-synthetic-images\\versions\\3\\test\\FAKE"
for fname in os.listdir(image_dir)[:20]:
    result = check_sd_watermark(os.path.join(image_dir, fname))
    print(f"{fname} -----> {result}")
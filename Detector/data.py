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
    
    decoder = WatermarkDecoder('bytes', 32) # SD uses 32. other models like gemini might be different
    watermark = decoder.decode(img, 'dwtDct')

    # SD's watermark string
    watermark_strings = []

    detected = watermark.decode('utf-8', errors='ignore')
    return detected

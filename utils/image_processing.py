import numpy as np
import cv2
from PIL import Image
import io
import base64

IMG_WIDTH = 256
IMG_HEIGHT = 256

def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((IMG_WIDTH, IMG_HEIGHT))
    img_array = np.array(img) / 255.0
    img_input = np.expand_dims(img_array, axis=0)
    return img_input

def create_binary_mask(pred_mask):
    return (pred_mask > 0.5).astype(np.uint8) * 255

def create_heatmap(pred_mask):
    heatmap_norm = (pred_mask * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)
    return cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

def encode_image_to_base64(numpy_image):
    if numpy_image.dtype != np.uint8:
        numpy_image = (numpy_image * 255).astype(np.uint8)
        
    is_success, buffer = cv2.imencode(".png", numpy_image)
    if is_success:
        base64_str = base64.b64encode(buffer).decode("utf-8")
        return f"data:image/png;base64,{base64_str}"
    return None
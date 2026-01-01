import numpy as np
import cv2
from PIL import Image
import io
import base64

# Ukuran input Model
IMG_WIDTH = 256
IMG_HEIGHT = 256

def preprocess_image(image_bytes):
    """
    Ubah bytes upload -> Numpy Array siap prediksi [1, 256, 256, 3]
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((IMG_WIDTH, IMG_HEIGHT))
    img_array = np.array(img) / 255.0
    img_input = np.expand_dims(img_array, axis=0)
    return img_input

def create_binary_mask(pred_mask):
    """
    Ubah prediksi (0.0-1.0) -> Masker Hitam Putih (0-255)
    """
    return (pred_mask > 0.5).astype(np.uint8) * 255

def create_heatmap(pred_mask):
    """
    Ubah prediksi -> Heatmap Warna (JET Colormap)
    """
    heatmap_norm = (pred_mask * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)
    # OpenCV pakai BGR, kita ubah ke RGB biar warna benar di web
    return cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

def encode_image_to_base64(numpy_image):
    """
    Ubah Numpy Array -> String Base64 untuk dikirim ke JSON
    """
    if numpy_image.dtype != np.uint8:
        numpy_image = (numpy_image * 255).astype(np.uint8)
        
    is_success, buffer = cv2.imencode(".png", numpy_image)
    if is_success:
        base64_str = base64.b64encode(buffer).decode("utf-8")
        return f"data:image/png;base64,{base64_str}"
    return None
import os
import tensorflow as tf

MODEL_PATH = os.path.join("models", "saliency_unet_model.keras")

print(f"🔍 Memeriksa model di: {MODEL_PATH}")

try:
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    print("\n✅ Model BERHASIL dimuat!")
    
    input_shape = model.input_shape
    print(f"📏 Model meminta Input Shape: {input_shape}")
    
except Exception as e:
    print("\n❌ Model GAGAL dimuat.")
    print(f"Error detail: {e}")
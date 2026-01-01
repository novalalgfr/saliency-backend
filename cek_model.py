import os
import tensorflow as tf

# Pastikan path ini benar sesuai lokasi file Anda
MODEL_PATH = os.path.join("models", "saliency_unet_model.keras")

print(f"🔍 Memeriksa model di: {MODEL_PATH}")

try:
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    print("\n✅ Model BERHASIL dimuat!")
    
    # Cek ukuran input yang diminta model
    input_shape = model.input_shape
    print(f"📏 Model meminta Input Shape: {input_shape}")
    
    # Biasanya outputnya: (None, 256, 256, 3) atau (None, 128, 128, 3)
    # Angka di tengah itulah ukurannya.
    
except Exception as e:
    print("\n❌ Model GAGAL dimuat.")
    print(f"Error detail: {e}")
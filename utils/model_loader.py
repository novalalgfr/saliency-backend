import os
import tensorflow as tf

def load_model(model_path):
    """
    Memuat model Keras dari path yang diberikan.
    """
    if os.path.exists(model_path):
        print(f"🔄 Sedang memuat model dari: {model_path} ...")
        try:
            # compile=False biar lebih cepat (kita cuma butuh predict, bukan training)
            model = tf.keras.models.load_model(model_path, compile=False)
            print("✅ Model Berhasil Dimuat!")
            return model
        except Exception as e:
            print(f"❌ Gagal memuat model: {e}")
            return None
    else:
        print(f"⚠️ Error: File model tidak ditemukan di {model_path}")
        return None
# SalientVision - Backend (API)

Backend service untuk sistem **Salient Object Detection** yang dibangun menggunakan **Flask** dan **TensorFlow**. Layanan ini bertugas menerima citra dari client, memprosesnya menggunakan model Deep Learning (U-Net), dan mengembalikan hasil berupa Binary Mask serta Visual Attention Heatmap.

## 🧠 Tech Stack

- **Language:** Python 3.10+
- **Framework:** Flask & Flask-CORS
- **Deep Learning:** TensorFlow / Keras 3
- **Image Processing:** OpenCV, Pillow, NumPy
- **Architecture:** U-Net (Custom Trained on DUTS Dataset)

## 📂 Struktur Folder

```text
backend/
├── app.py                   # Entry point server Flask
├── requirements.txt         # Daftar library dependencies
├── models/                  # Folder penyimpanan file model (.keras)
│   └── saliency_unet_model.keras
└── utils/                   # Modul bantuan
    ├── image_processing.py  # Preprocessing & Heatmap generation
    └── model_loader.py      # Load model TensorFlow
```

## 🚀 Cara Menjalankan (Local)

Pastikan Python sudah terinstal di komputer Anda.

### 1. Clone Repository

```bash
git clone https://github.com/novalalgfr/saliency-backend.git
cd frontend
```

### 2. Setup Virtual Environment

Disarankan menggunakan virtual environment agar library tidak konflik.

**Windows:**

```bash
python -m venv venv
venv\Scripts\activate
```

**Mac/Linux:**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Setup Model

Pastikan file model `saliency_unet_model.keras` sudah diletakkan di dalam folder `models/`.

> **Catatan:** File model tidak disertakan di repository ini karena ukurannya yang besar dan batasan GitHub. Silakan hubungi pengembang untuk mendapatkan file model.

### 5. Jalankan Server

```bash
python app.py
```

Server akan berjalan di `http://127.0.0.1:5000`.

## 📡 API Endpoints

### `POST /predict`

Menerima upload gambar dan mengembalikan hasil deteksi.

- **URL:** `http://127.0.0.1:5000/predict`
- **Method:** `POST`
- **Content-Type:** `multipart/form-data`
- **Body:**
  - `file`: File gambar (JPG/PNG)

**Response (JSON):**

```json
{
  "status": "success",
  "mask_url": "data:image/png;base64,.....",
  "heatmap_url": "data:image/png;base64,....."
}
```

## 📝 Lisensi

Project ini dikembangkan untuk keperluan edukasi dan penelitian.

---

**© 2026 SalientVision**

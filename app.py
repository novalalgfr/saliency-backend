import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np

from utils.model_loader import load_model
from utils.image_processing import (
    preprocess_image, 
    create_binary_mask, 
    create_heatmap, 
    encode_image_to_base64
)

app = Flask(__name__)
CORS(app)

MODEL_PATH = os.path.join("models", "saliency_unet_model.keras")

model = load_model(MODEL_PATH)

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "Tidak ada file"}), 400
    
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Nama file kosong"}), 400

    if model is None:
        return jsonify({"error": "Model gagal dimuat."}), 500

    try:
        img_bytes = file.read()
        processed_input = preprocess_image(img_bytes)

        prediction = model.predict(processed_input)
        pred_mask = prediction[0]

        binary_mask = create_binary_mask(pred_mask)
        heatmap_color = create_heatmap(pred_mask)

        mask_b64 = encode_image_to_base64(binary_mask)
        heatmap_b64 = encode_image_to_base64(heatmap_color)

        return jsonify({
            "status": "success",
            "mask_url": mask_b64,
            "heatmap_url": heatmap_b64
        })

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "SalientVision Backend Ready!",
        "model_status": "Loaded" if model else "Not Loaded"
    })

if __name__ == "__main__":
    app.run(debug=True, port=5000)
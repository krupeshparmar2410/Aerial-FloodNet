"""
app.py — FloodNet Flask Backend (Production Ready - Safe Version)
"""

import os
import cv2
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# Safe import (prevents crash if predict.py fails)
try:
    from predict import predict_classification, predict_segmentation
    MODEL_LOADED = True
except Exception as e:
    print("❌ Error loading models:", e)
    MODEL_LOADED = False

# Initialize app
app = Flask(__name__, static_folder="static")
CORS(app)

# Base directory (important for Render)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# -------------------------
# Helper: Read Image
# -------------------------
def read_img():
    if "image" not in request.files:
        return None, {"error": "No image provided"}

    f = request.files["image"]
    npimg = np.frombuffer(f.read(), np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    if img is None:
        return None, {"error": "Could not read image"}

    return img, None


# -------------------------
# Routes
# -------------------------

# Home (Frontend)
@app.route("/")
def index():
    return send_from_directory(app.static_folder, "floodnet_frontend.html")


# Task 1: Classification
@app.route("/api/classify", methods=["POST"])
def classify():
    if not MODEL_LOADED:
        return jsonify({"error": "Model not loaded"}), 500

    img, err = read_img()
    if err:
        return jsonify(err), 400

    try:
        result = predict_classification(img)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Task 2: Segmentation
@app.route("/api/segment", methods=["POST"])
def segment():
    if not MODEL_LOADED:
        return jsonify({"error": "Model not loaded"}), 500

    img, err = read_img()
    if err:
        return jsonify(err), 400

    try:
        result = predict_segmentation(img)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Health Check
@app.route("/api/health")
def health():
    md = os.path.join(BASE_DIR, "..", "models")

    models = {
        "task1_model": os.path.exists(os.path.join(md, "task1_model.pkl")),
        "task2_model": os.path.exists(os.path.join(md, "task2_model.pkl")),
    }

    return jsonify({
        "status": "ready" if all(models.values()) else "models missing",
        "models": models,
        "all_ready": all(models.values())
    })


# -------------------------
# Run App
# -------------------------
if __name__ == "__main__":
    print("=" * 50)
    print("🚀 FloodNet Flask Backend Running")
    print("=" * 50)

    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)
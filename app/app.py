"""
app.py — FloodNet Flask Backend
Run:  cd D:\\Projects\\ML\\app && python app.py
Open: http://localhost:5000
"""

import os, cv2, numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from predict import predict_classification, predict_segmentation
app = Flask(__name__, static_folder="static")
CORS(app)

def read_img():
    if "image" not in request.files:
        return None, {"error": "No image provided"}
    f = request.files["image"]
    npimg = np.frombuffer(f.read(), np.uint8)
    img   = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    if img is None:
        return None, {"error": "Could not read image"}
    return img, None

@app.route("/")
def index():
    return send_from_directory("static", "floodnet_frontend.html")

@app.route("/api/classify", methods=["POST"])
def classify():
    img, err = read_img()
    if err: return jsonify(err), 400
    return jsonify(predict_classification(img))

@app.route("/api/segment", methods=["POST"])
def segment():
    img, err = read_img()
    if err: return jsonify(err), 400
    return jsonify(predict_segmentation(img))


@app.route("/api/health")
def health():
    import os
    md = os.path.join(os.path.dirname(__file__),"..","models")
    models = {
        "task1_model":         os.path.exists(os.path.join(md,"task1_model.pkl")),
        "task2_model":         os.path.exists(os.path.join(md,"task2_model.pkl")),
        "task3_model":         os.path.exists(os.path.join(md,"task3_model.pkl")),
        "task3_tfidf":         os.path.exists(os.path.join(md,"task3_tfidf.pkl")),
        "task3_label_encoder": os.path.exists(os.path.join(md,"task3_label_encoder.pkl")),
    }
    return jsonify({"status":"ready" if all(models.values()) else "models missing",
                    "models":models, "all_ready":all(models.values())})

if __name__ == "__main__":
    print("="*50)
    print("  FloodNet Flask Backend")
    print("  http://localhost:5000")
    print("="*50)
    app.run(debug=True, port=5000)

from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import cv2
import numpy as np
from yolov5 import YOLOv5  # YOLOv5 package import

app = Flask(__name__)
CORS(app)

# Load your YOLOv5 model (CPU or GPU if available)
yolo = YOLOv5("best.pt", device="cpu")  # use "cuda" if GPU available

@app.route("/")
def index():
    return "YOLOv5 API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files["image"]
    img_bytes = file.read()
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Run YOLOv5 prediction
    results = yolo.predict(img)

    # Convert results to JSON
    detections = []
    for r in results.xyxy[0]:
        detections.append({
            "xmin": float(r[0]),
            "ymin": float(r[1]),
            "xmax": float(r[2]),
            "ymax": float(r[3]),
            "confidence": float(r[4]),
            "class": int(r[5])
        })

    return jsonify({"detections": detections})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)

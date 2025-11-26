from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
import os
from PIL import Image
import io

app = Flask(__name__)
CORS(app)

# Load YOLO model
model = YOLO("best.pt")  # make sure best.pt is in the same folder

@app.route("/")
def home():
    return "YOLO API is running!"

@app.route("/detect", methods=["POST"])
def detect():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    img = Image.open(io.BytesIO(file.read()))

    results = model.predict(img)

    detections = []
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            name = model.names[cls]
            detections.append({
                "label": name,
                "confidence": round(conf, 3)
            })

    return jsonify({"detections": detections})

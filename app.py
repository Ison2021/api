from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
from ultralytics import YOLO

app = Flask(__name__)
CORS(app)

# Load YOLOv5 model converted to YOLOv8 format (pt still works)
model = YOLO("best.pt")

@app.route("/")
def home():
    return "YOLO API running!"

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    img_bytes = file.read()
    npimg = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    # Run prediction
    results = model.predict(img, conf=0.25)

    detections = []
    for r in results:
        for box in r.boxes:
            detections.append({
                "xmin": float(box.xyxy[0][0]),
                "ymin": float(box.xyxy[0][1]),
                "xmax": float(box.xyxy[0][2]),
                "ymax": float(box.xyxy[0][3]),
                "confidence": float(box.conf[0]),
                "class_id": int(box.cls[0]),
                "class_name": model.names[int(box.cls[0])]
            })

    return jsonify({"detections": detections})


if __name__ == "__main__":
    import os
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))

from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import cv2
import numpy as np
import sys
import os

# Add local yolov5 repo to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), "yolov5"))

from models.common import DetectMultiBackend
from utils.torch_utils import select_device
from utils.general import non_max_suppression, scale_coords

app = Flask(__name__)
CORS(app)

# Device (CPU or GPU if available)
device = select_device("cpu")  # change to "0" or "cuda:0" if GPU

# Load YOLOv5 model
model = DetectMultiBackend("best.pt", device=device)
stride, names, pt = model.stride, model.names, model.pt

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

    # Preprocess image
    img_tensor = torch.from_numpy(img).permute(2, 0, 1).float()  # HWC to CHW
    img_tensor /= 255.0  # normalize to 0-1
    if img_tensor.ndim == 3:
        img_tensor = img_tensor.unsqueeze(0)

    # Inference
    pred = model(img_tensor)
    pred = non_max_suppression(pred, 0.25, 0.45, classes=None, agnostic=False)

    # Convert results to JSON
    detections = []
    for det in pred[0]:
        if len(det):
            det[:, :4] = scale_coords(img_tensor.shape[2:], det[:, :4], img.shape).round()
            for *xyxy, conf, cls in det:
                detections.append({
                    "xmin": float(xyxy[0]),
                    "ymin": float(xyxy[1]),
                    "xmax": float(xyxy[2]),
                    "ymax": float(xyxy[3]),
                    "confidence": float(conf),
                    "class": int(cls)
                })

    return jsonify({"detections": detections})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))

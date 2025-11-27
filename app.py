from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import cv2
import numpy as np
import sys
import os

# Add local YOLOv5 repo (must exist in your GitHub project!)
sys.path.append(os.path.join(os.path.dirname(__file__), "yolov5"))

from models.common import DetectMultiBackend
from utils.torch_utils import select_device
from utils.general import non_max_suppression, scale_coords, letterbox

app = Flask(__name__)
CORS(app)

# CPU only for Render
device = select_device("cpu")

# Load YOLOv5 model
model = DetectMultiBackend("best.pt", device=device)
stride = model.stride
names = model.names

@app.route("/")
def home():
    return "YOLOv5 API running on Render!"

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    img_bytes = file.read()

    # Convert image
    npimg = np.frombuffer(img_bytes, np.uint8)
    img0 = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    # Preprocess (important!)
    img = letterbox(img0, 640, stride=stride)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR→RGB, HWC→CHW
    img = np.ascontiguousarray(img)

    img_tensor = torch.from_numpy(img).to(device)
    img_tensor = img_tensor.float() / 255.0
    img_tensor = img_tensor.unsqueeze(0)

    # Inference
    pred = model(img_tensor, augment=False, visualize=False)
    pred = non_max_suppression(pred, 0.25, 0.45)

    results = []

    for det in pred:
        if len(det):
            det[:, :4] = scale_coords(img_tensor.shape[2:], det[:, :4], img0.shape).round()

            for *xyxy, conf, cls in det:
                results.append({
                    "xmin": float(xyxy[0]),
                    "ymin": float(xyxy[1]),
                    "xmax": float(xyxy[2]),
                    "ymax": float(xyxy[3]),
                    "confidence": float(conf),
                    "class_id": int(cls),
                    "class_name": names[int(cls)]
                })

    return jsonify({"detections": results})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))

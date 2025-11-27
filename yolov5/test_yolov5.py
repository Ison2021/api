import torch
from pathlib import Path
from models.common import DetectMultiBackend
from utils.general import non_max_suppression
from utils.ops import scale_coords
from utils.plots import save_one_box
import cv2
import os
import sys
# Add YOLOv5 root folder to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# ----- SETTINGS -----
MODEL_PATH = 'last.pt'          # Your trained model
IMAGE_FOLDER = 'test_images'    # Folder with images to test
CONFIDENCE_THRESHOLD = 0.25
IMG_SIZE = 640
DEVICE = 'cpu'                  # 'cpu' or 'cuda:0'
SAVE_RESULTS = True             # Save images with detections
RESULTS_FOLDER = 'runs/detect/exp'  # Where to save results

# Create results folder
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# ----- LOAD MODEL -----
device = torch.device(DEVICE)
model = DetectMultiBackend(MODEL_PATH, device=device)
stride, names = model.stride, model.names
imgsz = (IMG_SIZE, IMG_SIZE)

# ----- LOAD IMAGES -----
image_folder = Path(IMAGE_FOLDER)
image_paths = [p for p in image_folder.iterdir() if p.suffix.lower() in ['.jpg', '.jpeg', '.png']]

if not image_paths:
    raise FileNotFoundError(f"No images found in folder '{IMAGE_FOLDER}'!")

# ----- RUN DETECTION -----
for img_path in image_paths:
    print(f"Processing {img_path.name}...")

    # Load image
    img0 = cv2.imread(str(img_path))  # BGR
    img = torch.from_numpy(img0).to(device)
    img = img.permute(2, 0, 1).float()  # HWC to CHW
    img /= 255.0
    if img.ndim == 3:
        img = img.unsqueeze(0)

    # Inference
    pred = model(img, augment=False, visualize=False)
    pred = non_max_suppression(pred, CONFIDENCE_THRESHOLD, 0.45, None, False, max_det=1000)

    # Process predictions
    for i, det in enumerate(pred):
        if len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
            for *xyxy, conf, cls in reversed(det):
                label = f'{names[int(cls)]} {conf:.2f}'
                save_one_box(xyxy, img0, file=os.path.join(RESULTS_FOLDER, img_path.name), label=label, color=(0,255,0), line_thickness=3)

    print(f"Saved result for {img_path.name}")

print(f"Detection finished! All results saved in {RESULTS_FOLDER}")

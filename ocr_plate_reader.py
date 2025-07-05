import os
import cv2
import torch
import easyocr
import numpy as np
from pathlib import Path
from models.common import DetectMultiBackend
from utils.general import non_max_suppression
from utils.torch_utils import select_device

# ========== CONFIG ==========
weights = 'runs/train/plate_detector2/weights/best.pt'
image_path = r'C:\Users\tapas\ds assigment\yolov5\test_plate.jpg.jpg'  # Use raw string (r'') for Windows paths
output_image_path = 'ocr_result.jpg'

# ========== DEVICE SETUP ==========
device = select_device('')
model = DetectMultiBackend(weights, device=device)
model.model.float().eval()

# ========== OCR INIT ==========
reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())

# ========== IMAGE LOAD ==========
img0 = cv2.imread(image_path)
assert img0 is not None, f"⚠️ Image not found: {image_path}"

# ========== PREPROCESS ==========
img = cv2.resize(img0, (640, 640))
img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB and HWC to CHW
img = np.ascontiguousarray(img)
img = torch.from_numpy(img).to(device).float() / 255.0
img = img.unsqueeze(0)

# ========== INFERENCE ==========
with torch.no_grad():
    pred = model(img, augment=False, visualize=False)
    pred = non_max_suppression(pred, 0.25, 0.45)

# ========== SCALE COORDS ==========
def rescale_coords(img_shape, coords, original_shape):
    gain = min(img_shape[0] / original_shape[0], img_shape[1] / original_shape[1])
    pad_x = (img_shape[1] - original_shape[1] * gain) / 2
    pad_y = (img_shape[0] - original_shape[0] * gain) / 2

    coords[:, [0, 2]] -= pad_x
    coords[:, [1, 3]] -= pad_y
    coords[:, :4] /= gain
    coords[:, :4] = coords[:, :4].round()
    return coords

# ========== DETECTION & OCR ==========
for det in pred:
    if len(det):
        det[:, :4] = rescale_coords((640, 640), det[:, :4], img0.shape[:2])
        for *xyxy, conf, cls in det:
            x1, y1, x2, y2 = map(int, xyxy)
            crop = img0[y1:y2, x1:x2]

            if crop.size == 0:
                print("⚠️ Skipped empty crop!")
                continue

            result = reader.readtext(crop)
            for (bbox, text, prob) in result:
                print(f"🔠 Plate: {text} (Confidence: {prob:.2f})")
                cv2.rectangle(img0, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img0, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

# ========== SAVE ==========
cv2.imwrite(output_image_path, img0)
print(f"✅ OCR complete! Check → {output_image_path}")

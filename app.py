import streamlit as st
import cv2
import torch
import easyocr
import numpy as np
from PIL import Image
from models.common import DetectMultiBackend
from utils.general import non_max_suppression
from utils.torch_utils import select_device

# ========== CONFIG ==========
weights = 'runs/train/plate_detector2/weights/best.pt'
device = select_device('')
model = DetectMultiBackend(weights, device=device)
model.model.float().eval()
reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())

st.set_page_config(page_title="License Plate Detector", layout="centered")
st.title("🔍 License Plate Recognition System")
st.markdown("Upload a vehicle image to detect the number plate and extract the text.")

uploaded_file = st.file_uploader("Upload Image", type=['jpg', 'jpeg', 'png'])

if uploaded_file:
    # Read image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img0 = cv2.imdecode(file_bytes, 1)
    img_display = img0.copy()

    # Preprocess
    img = cv2.resize(img0, (640, 640))
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device).float() / 255.0
    img = img.unsqueeze(0)

    # Inference
    with torch.no_grad():
        pred = model(img, augment=False, visualize=False)
        pred = non_max_suppression(pred, 0.25, 0.45)

    def rescale_coords(img_shape, coords, original_shape):
        gain = min(img_shape[0] / original_shape[0], img_shape[1] / original_shape[1])
        pad_x = (img_shape[1] - original_shape[1] * gain) / 2
        pad_y = (img_shape[0] - original_shape[0] * gain) / 2
        coords[:, [0, 2]] -= pad_x
        coords[:, [1, 3]] -= pad_y
        coords[:, :4] /= gain
        coords[:, :4] = coords[:, :4].round()
        return coords

    detected_text = []
    for det in pred:
        if len(det):
            det[:, :4] = rescale_coords((640, 640), det[:, :4], img0.shape[:2])
            for *xyxy, conf, cls in det:
                x1, y1, x2, y2 = map(int, xyxy)
                crop = img0[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                result = reader.readtext(crop)
                for (_, text, prob) in result:
                    detected_text.append(f"{text} (Confidence: {prob:.2f})")
                    cv2.rectangle(img_display, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(img_display, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    st.subheader("📸 Detected Image")
    st.image(cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB), channels="RGB", use_column_width=True)

    if detected_text:
        st.subheader("🔠 Detected Plates")
        for text in detected_text:
            st.success(text)
    else:
        st.warning("No plate text detected.")
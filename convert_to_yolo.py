import os
import pandas as pd
from PIL import Image
import shutil

# 📂 Input paths
csv_path = r"C:\Users\tapas\ds assigment\Dectection\Licplatesdetection_train.csv"
images_dir = r"C:\Users\tapas\ds assigment\Dectection\license_plates_detection_train"

# 📂 Output paths
output_img_dir = r"C:\Users\tapas\yolo_dataset\images\train"
output_lbl_dir = r"C:\Users\tapas\yolo_dataset\labels\train"

# ✅ Create output folders
os.makedirs(output_img_dir, exist_ok=True)
os.makedirs(output_lbl_dir, exist_ok=True)

# 📄 Read and clean CSV
df = pd.read_csv(csv_path)
df.columns = [c.strip().lower() for c in df.columns]
df.rename(columns={df.columns[0]: "image_id"}, inplace=True)

# 🔁 Loop through and create label files
for _, row in df.iterrows():
    filename = row["image_id"]
    img_path = os.path.join(images_dir, filename)
    out_lbl = os.path.join(output_lbl_dir, filename.replace(".jpg", ".txt"))
    out_img = os.path.join(output_img_dir, filename)

    if not os.path.exists(img_path):
        print(f"Missing image: {filename}")
        continue

    # Get image size
    with Image.open(img_path) as img:
        w, h = img.size

    # Convert to YOLO format
    x_center = ((row["xmin"] + row["xmax"]) / 2) / w
    y_center = ((row["ymin"] + row["ymax"]) / 2) / h
    bbox_w = (row["xmax"] - row["xmin"]) / w
    bbox_h = (row["ymax"] - row["ymin"]) / h

    # Save label
    with open(out_lbl, "w") as f:
        f.write(f"0 {x_center:.6f} {y_center:.6f} {bbox_w:.6f} {bbox_h:.6f}\n")

    # Copy image
    shutil.copy(img_path, out_img)

print("✅ YOLO formatted labels and images ready!")

from ultralytics import YOLO
import os

# ----------------------------
# Model Path
# ----------------------------

MODEL_PATH = r"C:\Users\HP\runs\detect\runs\silkworm_detector-2\weights\best.pt"

# ----------------------------
# Image Path
# ----------------------------

IMAGE_PATH = r"C:\Users\HP\Downloads\MST-100\Datasets\img1\000001.jpg"

# ----------------------------
# Load Model
# ----------------------------

model = YOLO(MODEL_PATH)

# ----------------------------
# Predict
# ----------------------------

results = model.predict(
    source=IMAGE_PATH,
    conf=0.25,
    save=True,
    save_txt=False,
    show=False
)

print("\nPrediction completed successfully!")

print("\nOutput saved in:")
print("runs/detect/predict/")
from ultralytics import YOLO
import os

# Absolute path to data.yaml
DATA_YAML = r"C:\Users\HP\Downloads\MST-100\Datasets_yolo\data.yaml"

print("=" * 60)
print("Training Silkworm Detector")
print("=" * 60)
print(f"Dataset YAML: {DATA_YAML}")
print(f"Exists? {os.path.exists(DATA_YAML)}")
print("=" * 60)

# Load pretrained YOLO11 Nano model
model = YOLO("yolo11n.pt")

# Train
model.train(
    data=DATA_YAML,
    epochs=50,
    patience=10,
    imgsz=640,
    batch=4,
    workers=0,
    device="cpu",
    project="runs",
    name="silkworm_detector",
)

print("\nTraining Complete!")
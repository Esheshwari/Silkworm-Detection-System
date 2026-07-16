from ultralytics import YOLO
import os
import numpy as np

MODEL_PATH = r"C:\Users\HP\runs\detect\runs\silkworm_detector-2\weights\best.pt"

TEST_IMAGES = r"C:\Users\HP\Downloads\MST-100\Datasets_yolo\images\test"

TEST_LABELS = r"C:\Users\HP\Downloads\MST-100\Datasets_yolo\labels\test"

model = YOLO(MODEL_PATH)

true_counts = []
pred_counts = []

images = sorted(os.listdir(TEST_IMAGES))

for img_name in images:

    img_path = os.path.join(TEST_IMAGES,img_name)

    results = model.predict(
        img_path,
        conf=0.25,
        verbose=False
    )

    pred = len(results[0].boxes)

    label_file = os.path.join(
        TEST_LABELS,
        img_name.replace(".jpg",".txt")
    )

    with open(label_file) as f:
        gt = len(f.readlines())

    true_counts.append(gt)
    pred_counts.append(pred)

true_counts=np.array(true_counts)
pred_counts=np.array(pred_counts)

mae=np.mean(np.abs(true_counts-pred_counts))
rmse=np.sqrt(np.mean((true_counts-pred_counts)**2))
mce=np.mean(pred_counts-true_counts)

print("="*50)
print("Evaluation Results")
print("="*50)
print("Mean Absolute Error :",round(mae,2))
print("RMSE :",round(rmse,2))
print("Mean Count Error :",round(mce,2))
print("="*50)
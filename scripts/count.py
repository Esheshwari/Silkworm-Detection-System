from ultralytics import YOLO
import cv2
import time

MODEL_PATH = r"C:\Users\HP\runs\detect\runs\silkworm_detector-2\weights\best.pt"

IMAGE_PATH = r"C:\Users\HP\Downloads\MST-100\Datasets\img1\000001.jpg"

model = YOLO(MODEL_PATH)

start = time.time()

results = model.predict(
    source=IMAGE_PATH,
    conf=0.25,
    save=False,
    verbose=False
)

end = time.time()

boxes = results[0].boxes

count = len(boxes)

print("="*40)
print("Silkworm Count :", count)
print("Inference Time :", round(end-start,3),"seconds")
print("="*40)

# Draw detections

annotated = results[0].plot()

cv2.imwrite("count_result.jpg", annotated)

print("\nAnnotated image saved as count_result.jpg")
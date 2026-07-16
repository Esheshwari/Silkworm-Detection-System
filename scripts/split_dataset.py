import os
import shutil

# =====================================================
# Project Paths
# =====================================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

IMAGE_DIR = os.path.join(PROJECT_ROOT, "Datasets", "img1")
LABEL_DIR = os.path.join(PROJECT_ROOT, "Datasets_yolo", "labels")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "Datasets_yolo")

# =====================================================
# Create/Clear Output Folders
# =====================================================

folders = [
    "images/train",
    "images/val",
    "images/test",
    "labels/train",
    "labels/val",
    "labels/test",
]

for folder in folders:
    folder_path = os.path.join(OUTPUT_DIR, folder)

    os.makedirs(folder_path, exist_ok=True)

    # Clear existing files
    for file in os.listdir(folder_path):
        os.remove(os.path.join(folder_path, file))

# =====================================================
# Copy Function
# =====================================================

def copy_range(start_frame, end_frame, image_dest, label_dest):

    for frame in range(start_frame, end_frame + 1):

        image = f"{frame:06d}.jpg"
        label = f"{frame:06d}.txt"

        shutil.copy(
            os.path.join(IMAGE_DIR, image),
            os.path.join(image_dest, image)
        )

        shutil.copy(
            os.path.join(LABEL_DIR, label),
            os.path.join(label_dest, label)
        )

# =====================================================
# Train
# =====================================================

copy_range(
    1,
    840,
    os.path.join(OUTPUT_DIR, "images/train"),
    os.path.join(OUTPUT_DIR, "labels/train")
)

# =====================================================
# Validation
# =====================================================

copy_range(
    841,
    1080,
    os.path.join(OUTPUT_DIR, "images/val"),
    os.path.join(OUTPUT_DIR, "labels/val")
)

# =====================================================
# Test
# =====================================================

copy_range(
    1081,
    1200,
    os.path.join(OUTPUT_DIR, "images/test"),
    os.path.join(OUTPUT_DIR, "labels/test")
)

print("=" * 50)
print("Dataset Split Completed Successfully!")
print("=" * 50)
print("Train Images      :", len(os.listdir(os.path.join(OUTPUT_DIR, "images/train"))))
print("Validation Images :", len(os.listdir(os.path.join(OUTPUT_DIR, "images/val"))))
print("Test Images       :", len(os.listdir(os.path.join(OUTPUT_DIR, "images/test"))))
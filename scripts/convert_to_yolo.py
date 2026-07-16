import os
import pandas as pd

# -----------------------------
# Paths
# -----------------------------
GT_FILE = "../Datasets/gt/gt.txt"

IMAGE_FOLDER = "../Datasets/img1"

OUTPUT_LABELS = "../Datasets_yolo/labels"

# -----------------------------
# Image Size
# -----------------------------
IMG_WIDTH = 1660
IMG_HEIGHT = 1080

# -----------------------------
# Create output folder
# -----------------------------
os.makedirs(OUTPUT_LABELS, exist_ok=True)

# -----------------------------
# Read GT file
# -----------------------------
columns = [
    "frame",
    "id",
    "x",
    "y",
    "w",
    "h",
    "conf",
    "class",
    "visibility"
]

df = pd.read_csv(
    GT_FILE,
    header=None,
    names=columns
)

print("Annotations Loaded:", len(df))

# -----------------------------
# Group by frame
# -----------------------------
grouped = df.groupby("frame")

# -----------------------------
# Create one txt per image
# -----------------------------
for frame, rows in grouped:

    label_path = os.path.join(
        OUTPUT_LABELS,
        f"{frame:06d}.txt"
    )

    with open(label_path, "w") as f:

        for _, row in rows.iterrows():

            x = row["x"]
            y = row["y"]
            w = row["w"]
            h = row["h"]

            # Convert to YOLO format

            x_center = (x + w / 2) / IMG_WIDTH
            y_center = (y + h / 2) / IMG_HEIGHT

            width = w / IMG_WIDTH
            height = h / IMG_HEIGHT

            f.write(
                f"0 "
                f"{x_center:.6f} "
                f"{y_center:.6f} "
                f"{width:.6f} "
                f"{height:.6f}\n"
            )

print("YOLO labels created successfully!")

print("Total label files:",
      len(os.listdir(OUTPUT_LABELS)))
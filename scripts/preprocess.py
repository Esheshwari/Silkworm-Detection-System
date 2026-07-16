import os
from PIL import Image

img_dir = "img1"

images = sorted(os.listdir(img_dir))

print(f"Total Images: {len(images)}")

print("First 10 images:")
print(images[:10])

# Check image sizes
sizes = set()

for img_name in images:
    img = Image.open(os.path.join(img_dir, img_name))
    sizes.add(img.size)

print("Unique image sizes:")
print(sizes)

import pandas as pd

cols = [
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
    "gt/gt.txt",
    header=None,
    names=cols
)

print("\nFirst 5 annotations:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

import cv2
import matplotlib.pyplot as plt

img = cv2.imread("img1/000001.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

row = df.iloc[0]

x = int(row["x"])
y = int(row["y"])
w = int(row["w"])
h = int(row["h"])

cv2.rectangle(
    img,
    (x, y),
    (x + w, y + h),
    (255, 0, 0),
    3
)

plt.figure(figsize=(12,8))
plt.imshow(img)
plt.title("Image with Bounding Box")
plt.axis("off")
plt.show()

from PIL import Image
import os

bad_images = []

for img_name in images:
    try:
        Image.open(os.path.join(img_dir, img_name)).verify()
    except:
        bad_images.append(img_name)

print(f"Corrupt images: {len(bad_images)}")

if bad_images:
    print(bad_images)
    
img_width = 1660
img_height = 1080

invalid = df[
    (df["x"] < 0) |
    (df["y"] < 0) |
    (df["x"] + df["w"] > img_width) |
    (df["y"] + df["h"] > img_height)
]

print("Invalid Bounding Boxes:", len(invalid))

print(df.describe())
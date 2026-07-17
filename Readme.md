# Silkworm Detection & Counting System

> A computer vision application that estimates and counts the number of silkworms present in a basket or tray using YOLO11 object detection.

![Python](https://img.shields.io/badge/Python-3.14-blue?logo=python)
![YOLO11](https://img.shields.io/badge/YOLO11-Ultralytics-green)
![Streamlit](https://img.shields.io/badge/Streamlit-Web_App-red?logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## Problem Statement

Manual counting of silkworms in trays or baskets is time-consuming, labor-intensive, and susceptible to human error, especially in large-scale sericulture operations.

The objective of this project is to develop an intelligent computer vision system capable of:

- Detecting individual silkworms in an image
- Estimating the total number of silkworms
- Providing accurate object localization using bounding boxes
- Delivering an interactive web application for real-time counting

This solution automates the counting process, reducing manual effort while improving consistency and scalability.

---

## Features

- AI-powered silkworm detection using YOLO11
- Automatic silkworm counting
- Interactive Streamlit dashboard
- Adjustable confidence threshold
- Original vs detected image comparison
- Detection analytics dashboard
- Confidence histogram
- Detection gauge visualization
- Detection history tracking
- Download annotated image
- Export detection results as CSV
- Sample image gallery
- Modern glassmorphism UI

---

## Demo

### Dashboard

https://github.com/user-attachments/assets/3ae3498a-d443-4900-b2af-9a299695f0aa


https://silkworm-detection-system.streamlit.app/


---

# System Architecture

```
                Input Image
                     │
                     ▼
            Image Preprocessing
                     │
                     ▼
              YOLO11 Detection
                     │
      ┌──────────────┴──────────────┐
      ▼                             ▼
Bounding Boxes              Object Count
      │                             │
      └──────────────┬──────────────┘
                     ▼
            Analytics Dashboard
                     │
                     ▼
          Streamlit Web Application
```

---

# Methodology

1. Load the trained YOLO11 model.
2. Upload an image containing silkworms.
3. Perform object detection.
4. Draw bounding boxes around detected silkworms.
5. Count detected instances.
6. Display analytics and visualizations.
7. Allow users to download results.

---

# Tech Stack

### Programming Language

- Python

### Machine Learning

- Ultralytics YOLO11
- PyTorch

### Computer Vision

- OpenCV
- Pillow

### Data Processing

- NumPy
- Pandas

### Visualization

- Plotly

### Web Framework

- Streamlit

### Dataset

- MST-100 Dataset

---

# Project Structure

```
Silkworm-Detection/
│
├── app.py
├── utils.py
├── style.css
├── requirements.txt
├── README.md
│
├── model/
│   └── best.pt
│
├── assets/
│   ├── samples/
│   └── demo/
│
├── scripts/
│   ├── preprocess.py
│   ├── convert_to_yolo.py
│   ├── split_dataset.py
│   ├── train.py
│   ├── predict.py
│   ├── count.py
│   └── evaluate.py
│
├── Datasets/
│
└── .streamlit/
    └── config.toml
```

---

# Dataset

The project uses the **MST-100** silkworm object detection dataset.

Dataset characteristics include:

- 1200 annotated images
- Bounding box annotations
- Single object class (Silkworm)
- YOLO annotation format

---

# Installation

Clone the repository

```bash
git clone https://github.com/yourusername/Silkworm-Detection.git

cd Silkworm-Detection
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
streamlit run app.py
```

---

# Training

Train the YOLO model

```bash
python scripts/train.py
```

Convert annotations

```bash
python scripts/convert_to_yolo.py
```

Split dataset

```bash
python scripts/split_dataset.py
```

---

# Evaluation Metrics

The model is evaluated using:

- Precision
- Recall
- mAP@50
- mAP@50-95
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Mean Count Error

---

# Application Workflow

```
Upload Image
      │
      ▼
YOLO11 Detection
      │
      ▼
Bounding Boxes
      │
      ▼
Count Silkworms
      │
      ▼
Analytics Dashboard
      │
      ▼
Download Results
```

---

# Dependencies

- streamlit
- ultralytics
- torch
- torchvision
- opencv-python-headless
- pillow
- numpy
- pandas
- plotly

---

# Future Improvements

- Video-based silkworm counting
- Real-time webcam inference
- Batch image processing
- PDF report generation
- Multi-class silkworm lifecycle detection
- Mobile application
- Cloud deployment
- Model optimization for edge devices

---

# Deployment

The project can be deployed on:

- Hugging Face Spaces
- Streamlit Community Cloud
- Render

---

# 👨‍💻 Author

**Esheshwari Kumari**

Founder of Zyonix

GitHub: https://github.com/Esheshwari

LinkedIn: https://linkedin.com/in/esheshwari

---

# License

This project is released under the MIT License.

---

## ⭐ If you found this project useful, consider giving it a star.

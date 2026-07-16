# ==========================================================
# IMPORTS
# ==========================================================

import os
import time

import numpy as np
import cv2
import tempfile

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import streamlit as st
from pathlib import Path
from PIL import Image

from utils import *


# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="Silkworm Detection Dashboard",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==========================================================
# LOAD CUSTOM CSS
# ==========================================================

with open("style.css") as css:
    st.markdown(
        f"<style>{css.read()}</style>",
        unsafe_allow_html=True
    )


# ==========================================================
# LOAD MODEL
# ==========================================================

MODEL_PATH = Path("model") / "best.pt"

model = YOLO(str(MODEL_PATH))


# ==========================================================
# SESSION STATE
# ==========================================================

if "history" not in st.session_state:
    st.session_state.history = []

if "selected_sample" not in st.session_state:
    st.session_state.selected_sample = None

if "uploaded_name" not in st.session_state:
    st.session_state.uploaded_name = ""

if "count_history" not in st.session_state:
    st.session_state.count_history = []

if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None


# ==========================================================
# GLOBAL CONSTANTS
# ==========================================================

APP_NAME = "Silkworm Detection & Counting"

MODEL_NAME = "YOLO11n"

DATASET_NAME = "MST-100"

CLASS_NAME = "Silkworm"

VERSION = "v1.0"

MAX_EXPECTED_COUNT = 100


# ==========================================================
# PAGE HEADER SPACING
# ==========================================================

st.markdown("<br>", unsafe_allow_html=True)

# ==========================================================
# HERO SECTION
# ==========================================================

st.markdown(
    f"""
<div class="hero">

<div class="hero-badge">
AI • Computer Vision • {MODEL_NAME}
</div>

<h1>{APP_NAME}</h1>

<p>

Estimate and count silkworms from tray or basket images using a
custom-trained YOLO11 object detection model.

This dashboard performs automatic object detection,
confidence analysis, visualization, and silkworm counting
through an interactive AI interface.

</p>

</div>
""",
    unsafe_allow_html=True,
)

# ==========================================================
# FEATURE CARDS
# ==========================================================

c1, c2, c3 = st.columns(3)

with c1:

    st.markdown(
        """
<div class="feature-card">

<h3>Object Detection</h3>

<p>

Detect every silkworm
using a custom YOLO11 model
trained on MST-100.

</p>

</div>
""",
        unsafe_allow_html=True,
    )

with c2:

    st.markdown(
        """
<div class="feature-card">

<h3>Automatic Counting</h3>

<p>

Instantly estimate the
number of silkworms from
high-resolution tray images.

</p>

</div>
""",
        unsafe_allow_html=True,
    )

with c3:

    st.markdown(
        """
<div class="feature-card">

<h3>Analytics Dashboard</h3>

<p>

Interactive statistics,
confidence analysis,
visualizations and exports.

</p>

</div>
""",
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# ==========================================================
# SIDEBAR
# ==========================================================

st.sidebar.markdown(
    """
# Detection Settings
Adjust the inference parameters.
"""
)

confidence = st.sidebar.slider(
    "Confidence Threshold",
    0.10,
    1.00,
    0.50,
    0.05,
)

iou = st.sidebar.slider(
    "IoU Threshold",
    0.10,
    1.00,
    0.45,
    0.05,
)

max_det = st.sidebar.slider(
    "Maximum Detections",
    10,
    300,
    150,
)

st.sidebar.divider()

# ==========================================================
# MODEL INFORMATION
# ==========================================================

info = model_information()

st.sidebar.markdown(
    f"""
### Model Information

**Model**

{info['Model']}

---

**Dataset**

{info['Dataset']}

---

**Framework**

{info['Framework']}

---

**Detection Class**

{info['Class']}

---

**Task**

{info['Task']}

"""
)

# ==========================================================
# SYSTEM STATUS
# ==========================================================

st.sidebar.divider()

st.sidebar.success("Model Loaded Successfully")

st.sidebar.caption(
    f"""
Version : {VERSION}

Status : Ready

Inference Device : CPU
"""
)

# ==========================================================
# QUICK STATS
# ==========================================================

st.sidebar.divider()

st.sidebar.metric(
    "Model",
    MODEL_NAME,
)

st.sidebar.metric(
    "Dataset",
    DATASET_NAME,
)

st.sidebar.metric(
    "Class",
    CLASS_NAME,
)

st.markdown("---")

# ==========================================================
# IMAGE INPUT
# ==========================================================

st.markdown("## Upload or Try Sample Images")

uploaded_file = st.file_uploader(
    "",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed"
)

# ==========================================================
# SAMPLE GALLERY
# ==========================================================

st.markdown("### Try Sample Images")

sample_folder = Path("assets") / "samples"

sample_files = [
    sample_folder / f"sample{i}.jpg"
    for i in range(1, 6)
]

cols = st.columns(5)

for i, sample in enumerate(sample_files):

    with cols[i]:

        if sample.exists():

            st.image(sample, use_container_width=True)

            if st.button(
                "Use Image",
                key=f"sample_{i}"
            ):

                st.session_state.selected_sample = sample

# ==========================================================
# SELECT IMAGE SOURCE
# ==========================================================

selected_image = None

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    selected_image = image

    st.session_state.selected_sample = None

elif st.session_state.selected_sample is not None:

    image = Image.open(
        st.session_state.selected_sample
    ).convert("RGB")

    selected_image = image
    
# ==========================================================
# EMPTY STATE
# ==========================================================

if selected_image is None:

    st.markdown("<br><br>", unsafe_allow_html=True)

    st.markdown(
        """
<div class="hero">

<h2 style="text-align:center;">
Upload an Image or Choose a Sample
</h2>

<p style="text-align:center;">

Start by uploading a basket image or
click one of the sample images above.

The AI model will automatically detect,
count and analyse silkworms.

</p>

</div>
""",
        unsafe_allow_html=True
    )

    st.stop()
    
# ==========================================================
# PREVIEW
# ==========================================================

st.markdown("---")

st.success("Image Loaded Successfully")

st.caption(
    f"Resolution : {selected_image.size[0]} × {selected_image.size[1]}"
)

# ==========================================================
# RUN YOLO DETECTION
# ==========================================================

with tempfile.NamedTemporaryFile(
    suffix=".jpg",
    delete=False
) as tmp:

    selected_image.save(tmp.name)

    image_path = tmp.name


start = time.time()

with st.spinner("Running AI Detection..."):

    results = model.predict(
        image_path,
        conf=confidence,
        iou=iou,
        max_det=max_det,
        verbose=False
    )

end = time.time()

inference_time = end - start

result = results[0]

# ==========================================================
# PREPARE IMAGES
# ==========================================================

original = np.array(selected_image)

annotated = original.copy()

boxes = result.boxes

count = len(boxes)

st.session_state.last_prediction = result

st.session_state.count_history.append(count)

# ==========================================================
# DRAW CUSTOM BOUNDING BOXES
# ==========================================================

for box in boxes:

    x1, y1, x2, y2 = map(
        int,
        box.xyxy[0]
    )

    conf = float(box.conf[0])

    if conf >= 0.75:

        color = (0,255,0)

    elif conf >= 0.50:

        color = (0,255,255)

    else:

        color = (0,0,255)

    cv2.rectangle(

        annotated,

        (x1,y1),

        (x2,y2),

        color,

        2

    )
    
# ==========================================================
# CONVERT TO RGB
# ==========================================================

annotated = cv2.cvtColor(
    annotated,
    cv2.COLOR_BGR2RGB
)

# ==========================================================
# IMAGE DISPLAY
# ==========================================================

st.markdown(
    '<div class="glass-card">',
    unsafe_allow_html=True
)

left,right = st.columns(2)

with left:

    st.subheader("Original Image")

    st.image(
        original,
        use_container_width=True
    )
    
with right:

    st.subheader("Detection Result")

    st.image(
        annotated,
        use_container_width=True
    )
    
st.markdown(
    "</div>",
    unsafe_allow_html=True
)

# ==========================================================
# DETECTION STATS
# ==========================================================

confidences = []

areas = []

for box in boxes:

    confidences.append(
        float(box.conf[0])
    )

    x1,y1,x2,y2 = box.xyxy[0]

    areas.append(
        (x2-x1)*(y2-y1)
    )

df = pd.DataFrame({

    "Confidence":confidences,

    "Bounding Box Area":areas

})

st.success(
    f"{count} silkworms detected successfully."
)

# ==========================================================
# ANALYTICS DASHBOARD
# ==========================================================

st.markdown("---")

st.markdown(
"""
## Detection Analytics

Explore the detection performance,
confidence distribution and statistics
generated by the AI model.
"""
)

# ==========================================================
# KPI METRICS
# ==========================================================

avg_conf = (
    sum(confidences) / len(confidences)
    if confidences else 0
)

max_conf = (
    max(confidences)
    if confidences else 0
)

m1, m2, m3, m4 = st.columns(4)

m1.metric(
    "Detected",
    f"{count}"
)

m2.metric(
    "Average Confidence",
    f"{avg_conf:.2f}"
)

m3.metric(
    "Inference Time",
    f"{inference_time:.2f}s"
)

m4.metric(
    "Highest Confidence",
    f"{max_conf:.2f}"
)

# ==========================================================
# DETECTION GAUGE
# ==========================================================

st.markdown("### Detection Gauge")

max_expected = MAX_EXPECTED_COUNT

gauge = go.Figure(

    go.Indicator(

        mode="gauge+number",

        value=count,

        title={
            "text":"Detected Silkworms"
        },

        gauge={

            "axis":{
                "range":[0,max_expected]
            },

            "bar":{
                "color":"#00E5A8"
            },

            "steps":[

                {
                    "range":[0,max_expected*0.40],
                    "color":"#1E293B"
                },

                {
                    "range":[max_expected*0.40,max_expected*0.70],
                    "color":"#334155"
                },

                {
                    "range":[max_expected*0.70,max_expected],
                    "color":"#475569"
                }

            ]

        }

    )

)

gauge.update_layout(

    template="plotly",

    paper_bgcolor="rgba(0,0,0,0)",

    plot_bgcolor="rgba(0,0,0,0)",

    font=dict(
        color="white",
        family="Inter"
    ),

    height=360

)

st.plotly_chart(
    gauge,
    use_container_width=True
)

st.markdown("---")

left, right = st.columns(2)

with left:

    st.subheader("Confidence Distribution")

    fig = px.histogram(

        df,

        x="Confidence",

        nbins=15,

        title="Confidence Histogram"

    )

    fig.update_layout(

        template="plotly",

        paper_bgcolor="rgba(0,0,0,0)",

        plot_bgcolor="rgba(0,0,0,0)",

        font=dict(
            color="white",
            family="Inter"
        )

    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )
    
with right:

    st.subheader("Confidence Categories")

    high = len(df[df["Confidence"] >= 0.75])

    medium = len(
        df[
            (df["Confidence"] >= 0.50)
            &
            (df["Confidence"] < 0.75)
        ]
    )

    low = len(df[df["Confidence"] < 0.50])

    pie = px.pie(

        names=[
            "High",
            "Medium",
            "Low"
        ],

        values=[
            high,
            medium,
            low
        ],

        hole=0.55

    )

    pie.update_layout(

        template="plotly",

        paper_bgcolor="rgba(0,0,0,0)",

        plot_bgcolor="rgba(0,0,0,0)",

        font=dict(
            color="white",
            family="Inter"
        )

    )

    st.plotly_chart(
        pie,
        use_container_width=True
    )
    
st.markdown("---")

st.subheader("Bounding Box Area Distribution")

area = px.histogram(

    df,

    x="Bounding Box Area",

    nbins=25,

    title="Object Size Distribution"

)

area.update_layout(

    template="plotly",

    paper_bgcolor="rgba(0,0,0,0)",

    plot_bgcolor="rgba(0,0,0,0)",

    font=dict(
        color="white",
        family="Inter"
    )

)

st.plotly_chart(
    area,
    use_container_width=True
)

# ==========================================================
# DETECTION TABLE
# ==========================================================

st.markdown("---")

st.subheader("Detection Details")

table = []

for i, box in enumerate(boxes):

    x1, y1, x2, y2 = map(int, box.xyxy[0])

    table.append({

        "ID": i + 1,

        "Confidence": round(float(box.conf[0]), 3),

        "X1": x1,

        "Y1": y1,

        "X2": x2,

        "Y2": y2,

        "Width": x2 - x1,

        "Height": y2 - y1

    })

table_df = pd.DataFrame(table)

st.dataframe(
    table_df,
    use_container_width=True,
    hide_index=True
)

# ==========================================================
# DOWNLOAD CSV
# ==========================================================

csv = table_df.to_csv(index=False).encode("utf-8")

st.download_button(

    "Download Detection CSV",

    csv,

    "detections.csv",

    "text/csv"

)

# ==========================================================
# DOWNLOAD IMAGE
# ==========================================================

annotated_bgr = cv2.cvtColor(

    annotated,

    cv2.COLOR_RGB2BGR

)

_, buffer = cv2.imencode(

    ".jpg",

    annotated_bgr

)

st.download_button(

    "Download Annotated Image",

    buffer.tobytes(),

    "silkworm_detection.jpg",

    "image/jpeg"

)

# ==========================================================
# DETECTION HISTORY
# ==========================================================

st.markdown("---")

st.subheader("Detection History")

history = pd.DataFrame({

    "Run": list(

        range(

            1,

            len(st.session_state.count_history) + 1

        )

    ),

    "Count": st.session_state.count_history

})

fig = px.line(

    history,

    x="Run",

    y="Count",

    markers=True,

    title="Detected Silkworm Count"

)

fig.update_layout(

    template="plotly",

    paper_bgcolor="rgba(0,0,0,0)",

    plot_bgcolor="rgba(0,0,0,0)",

    font=dict(

        color="white",

        family="Inter"

    )

)

st.plotly_chart(

    fig,

    use_container_width=True

)

# ==========================================================
# AI RECOMMENDATION
# ==========================================================

st.markdown("---")

st.subheader("AI Recommendation")

if avg_conf >= 0.80:

    st.success("""

Detection quality is excellent.

The model is highly confident in its predictions.

""")

elif avg_conf >= 0.60:

    st.warning("""

Detection quality is acceptable.

Increasing image quality or lighting may improve confidence.

""")

else:

    st.error("""

Detection confidence is relatively low.

Consider using a clearer image or lowering the camera angle.

""")
    
# ==========================================================
# SUMMARY
# ==========================================================

st.markdown("---")

st.info(f"""

### Detection Summary

**Detected Silkworms:** {count}

**Average Confidence:** {avg_conf:.2f}

**Inference Time:** {inference_time:.2f} sec

**Model:** {MODEL_NAME}

**Dataset:** {DATASET_NAME}

""")

# ==========================================================
# FOOTER
# ==========================================================

st.markdown("---")

st.markdown("""

<div class="footer">

Silkworm Detection & Counting System

Built using Streamlit, YOLO11, OpenCV, Plotly and Python

</div>

""", unsafe_allow_html=True)

from ultralytics import YOLO
import streamlit as st
import pandas as pd
import cv2
import numpy as np
import tempfile
import time

# ==========================================================
# LOAD MODEL (Cached)
# ==========================================================

@st.cache_resource
def load_model(model_path):

    model = YOLO(model_path)

    return model


# ==========================================================
# RUN DETECTION
# ==========================================================

def run_detection(
    model,
    image,
    confidence,
    iou,
    max_det
):

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".jpg"
    ) as tmp:

        image.save(tmp.name)

        start = time.time()

        results = model.predict(

            tmp.name,

            conf=confidence,

            iou=iou,

            max_det=max_det,

            verbose=False

        )

        end = time.time()

    return results[0], end-start


# ==========================================================
# GET COUNT
# ==========================================================

def get_count(result):

    return len(result.boxes)


# ==========================================================
# CREATE DETECTION DATAFRAME
# ==========================================================

def create_dataframe(result):

    rows = []

    for idx, box in enumerate(result.boxes):

        xyxy = box.xyxy[0].tolist()

        rows.append({

            "ID": idx+1,

            "Confidence": round(float(box.conf),3),

            "X1": int(xyxy[0]),

            "Y1": int(xyxy[1]),

            "X2": int(xyxy[2]),

            "Y2": int(xyxy[3]),

            "Width": int(xyxy[2]-xyxy[0]),

            "Height": int(xyxy[3]-xyxy[1])

        })

    return pd.DataFrame(rows)


# ==========================================================
# CONFIDENCE VALUES
# ==========================================================

def confidence_scores(result):

    scores = []

    for box in result.boxes:

        scores.append(float(box.conf))

    return scores


# ==========================================================
# ANNOTATED IMAGE
# ==========================================================

def annotated_image(result):

    return result.plot(

        labels=False,

        conf=False

    )


# ==========================================================
# IMAGE DOWNLOAD
# ==========================================================

def image_bytes(image):

    image = cv2.cvtColor(

        image,

        cv2.COLOR_RGB2BGR

    )

    _, buffer = cv2.imencode(

        ".jpg",

        image

    )

    return buffer.tobytes()


# ==========================================================
# CSV EXPORT
# ==========================================================

def dataframe_to_csv(df):

    return df.to_csv(index=False)


# ==========================================================
# CONFIDENCE STATISTICS
# ==========================================================

def confidence_statistics(conf_list):

    if len(conf_list)==0:

        return {

            "Average":0,

            "Maximum":0,

            "Minimum":0

        }

    return {

        "Average":round(np.mean(conf_list),3),

        "Maximum":round(np.max(conf_list),3),

        "Minimum":round(np.min(conf_list),3)

    }


# ==========================================================
# DETECTION SUMMARY
# ==========================================================

def detection_summary(result):

    scores = confidence_scores(result)

    stats = confidence_statistics(scores)

    summary = {

        "Count":len(result.boxes),

        "Average Confidence":stats["Average"],

        "Highest Confidence":stats["Maximum"],

        "Lowest Confidence":stats["Minimum"]

    }

    return summary


# ==========================================================
# HISTOGRAM DATA
# ==========================================================

def histogram_dataframe(result):

    return pd.DataFrame(

        {

            "Confidence":confidence_scores(result)

        }

    )


# ==========================================================
# BOUNDING BOX AREA
# ==========================================================

def box_area_dataframe(result):

    area=[]

    for box in result.boxes:

        x1,y1,x2,y2=box.xyxy[0].tolist()

        area.append(

            (x2-x1)*(y2-y1)

        )

    return pd.DataFrame(

        {

            "Bounding Box Area":area

        }

    )


# ==========================================================
# COUNT STATUS
# ==========================================================

def detection_status(count):

    if count==0:

        return "No Silkworms Detected"

    elif count<25:

        return "Low Population"

    elif count<60:

        return "Moderate Population"

    else:

        return "High Population"


# ==========================================================
# MODEL INFORMATION
# ==========================================================

def model_information():

    return {

        "Model":"YOLO11n",

        "Dataset":"MST-100",

        "Framework":"Ultralytics",

        "Task":"Object Detection",

        "Class":"Silkworm"

    }
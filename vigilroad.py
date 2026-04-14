import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
import random

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="VigilRoad Smart Monitoring",
    layout="wide"
)

st.title("🚧 VigilRoad – Smart Road Damage Monitoring System")
st.markdown("AI-powered pothole detection dashboard for municipal authorities")

# ---------------- SIDEBAR ----------------

st.sidebar.title("Control Panel")

uploaded_file = st.sidebar.file_uploader(
    "Upload Road Inspection Image",
    type=["jpg", "png", "jpeg"]
)

confidence_threshold = st.sidebar.slider(
    "Detection Sensitivity",
    0.1, 1.0, 0.5
)

st.sidebar.write("System Status: 🟢 Active")

# ---------------- LOAD MODEL ----------------

model = YOLO("best.pt")

# ---------------- PROCESS IMAGE ----------------

if uploaded_file is not None:

    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    results = model(image)

    annotated = results[0].plot()

    pothole_count = 0
    minor_count = 0
    moderate_count = 0
    severe_count = 0

    for box in results[0].boxes.xyxy:

        x1, y1, x2, y2 = box

        width = x2 - x1
        height = y2 - y1

        area = width * height

        pothole_count += 1

        if area < 2000:
            minor_count += 1
        elif area < 8000:
            moderate_count += 1
        else:
            severe_count += 1

    # ---------------- DASHBOARD LAYOUT ----------------

    col1, col2 = st.columns([2,1])

    with col1:
        st.subheader("Road Inspection Result")
        st.image(annotated, use_container_width=True)

    with col2:
        st.subheader("Incident Report")

        report_id = f"VR-{random.randint(1000,9999)}"

        st.write("Report ID:", report_id)
        st.write("Inspection Time:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

        st.metric("Total Potholes Detected", pothole_count)
        st.metric("Severe Damage", severe_count)
        st.metric("Moderate Damage", moderate_count)
        st.metric("Minor Damage", minor_count)

        st.markdown("---")

        if severe_count > 0:
            st.error("🔴 Priority Level: Immediate Repair Required")
        elif moderate_count > 0:
            st.warning("🟠 Priority Level: Schedule Maintenance")
        else:
            st.success("🟢 Priority Level: Monitor")

    # ---------------- DAMAGE CHART ----------------

    st.markdown("---")

    st.subheader("Damage Severity Distribution")

    data = {
        "Severity": ["Minor", "Moderate", "Severe"],
        "Count": [minor_count, moderate_count, severe_count]
    }

    df = pd.DataFrame(data)

    st.bar_chart(df.set_index("Severity"))
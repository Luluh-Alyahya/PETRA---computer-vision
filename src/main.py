# app.py
import os
import io
import time
import base64
import requests
import streamlit as st
from PIL import Image

# ---------------------------
# CONFIG
# ---------------------------
st.set_page_config(page_title="Oil Spill Detection", page_icon="🛰️", layout="wide")

import pathlib

# ---------------------------
# CONFIG
# ---------------------------
BASE_DIR = pathlib.Path(__file__).parent.resolve()

FASTAPI_URL = os.getenv("FASTAPI_URL", "https://fastapipetra-production.up.railway.app")  
BACKGROUND_IMAGE_PATH = str(BASE_DIR / "background.png")
INTRO_VIDEO_PATH = str(BASE_DIR / "earth_zoom.mp4")


# ---------------------------
# HELPERS
# ---------------------------
def _as_data_uri(path_or_url: str, mime: str) -> str:
    if path_or_url.lower().startswith("http"):
        return path_or_url
    with open(path_or_url, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"

def set_background_image(path_or_url: str):
    uri = _as_data_uri(path_or_url, "image/png")
    st.markdown(
        f"""
        <style>
        [data-testid="stAppViewContainer"] {{
            background: url("{uri}") no-repeat center center fixed !important;
            background-size: cover !important;
            background-color: transparent !important;
        }}
        .stApp, .main, .block-container {{
            background-color: rgba(0,0,0,0) !important;
        }}
        header[data-testid="stHeader"] {{
            background: rgba(0,0,0,0) !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

def show_brandbar():
    st.markdown(
        """
        <div style="
            display:flex;
            justify-content:center;
            align-items:center;
            gap:12px;
            margin-bottom: 25px;
        ">
          <div style="font-size:2.8rem;">🛰️ <b>Petra</b></div>
          <div style="opacity:.7; font-size:2.8rem;">| Oil Spill Detection From World Sea</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def call_fastapi_predict_with_visual(file_bytes: bytes, filename: str):
    try:
        resp = requests.post(
            f"{FASTAPI_URL}/predict",
            files={"file": (filename, file_bytes, "image/jpeg")},
            timeout=60,
        )
        if resp.ok:
            return True, resp.json()
        else:
            return False, {"error": f"{resp.status_code}: {resp.text}"}
    except Exception as e:
        return False, {"error": str(e)}

def call_fastapi_predict_url_with_visual(image_url: str):
    try:
        resp = requests.post(
            f"{FASTAPI_URL}/predict-url",
            json={"url": image_url},
            timeout=60,
        )
        if resp.ok:
            return True, resp.json()
        else:
            return False, {"error": f"{resp.status_code}: {resp.text}"}
    except Exception as e:
        return False, {"error": str(e)}

def display_detection_results(data):
    col1, col2 = st.columns([1.3, 1])
    with col1:
        st.subheader("📸 Detection Results")
        if "annotated_image" in data and data["annotated_image"]:
            image_data = base64.b64decode(data["annotated_image"])
            image = Image.open(io.BytesIO(image_data))
            st.image(image, caption="🎯 Detected Oil Spills", use_container_width=True)
        else:
            st.info("No annotated image available")
    with col2:
        st.subheader("📊 Detection Summary")
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("🔍 Total Detections", data.get("total_detections", 0))
        with col_b:
            st.metric("⚡ Processing Time", f"{data.get('processing_time', 0)}s")
        if data.get("detections"):
            st.subheader("🔍 Individual Detections")
            for i, detection in enumerate(data["detections"], 1):
                with st.expander(f"🛢️ Detection #{i}: {detection['class'].title()}", expanded=True):
                    col_x, col_y = st.columns(2)
                    with col_x:
                        st.metric("Confidence", f"{detection['confidence']}")
                    with col_y:
                        st.metric("Area Coverage", f"{detection['area_percentage']}%")
                    bbox = detection['bbox']
                    st.write(f"**📍 Location:** ({bbox['x1']}, {bbox['y1']}) → ({bbox['x2']}, {bbox['y2']})")
        else:
            st.success("✅ No oil spills detected in this image!")


# ---------------------------
# STATE: Splash -> App
# ---------------------------
if "intro_done" not in st.session_state:
    st.session_state.intro_done = False

if not st.session_state.intro_done:
    st.markdown(
        """
        <style>
        [data-testid="stAppViewContainer"] { background: black !important; }
        .intro-card { text-align:center; padding-top: 6vh; color: #ddd; }
        .intro-title { font-size: 2rem; margin: 12px 0 4px 0; }
        .intro-sub { opacity:.8; margin-bottom: 14px; }
        </style>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    """
    <div style="text-align:center; padding-top: 6vh; color: #ddd;">
        <div style="font-size: 2.5rem; margin-bottom: 8px;">🛰️ <b>Oil Spill Detection</b></div>
        <div style="font-size: 1.2rem; opacity: 0.8;">Cinematic journey from orbit to ocean</div>
    </div>
    """,
    unsafe_allow_html=True
)


# ---------------------------
# HERO VIDEO INTRO (AUTOPLAY)
# ---------------------------
with open(INTRO_VIDEO_PATH, "rb") as f:
    video_bytes = f.read()
video_base64 = base64.b64encode(video_bytes).decode()

st.markdown(f"""
<style>
.hero-container {{
    position: relative;
    width: 100%;
    height: 90vh;
    overflow: hidden;
    border-radius: 18px;
    box-shadow: 0 8px 30px rgba(0,0,0,0.5);
    margin-bottom: 2.5rem;
    margin-top: 2.5rem;
}}
.hero-container video {{
    position: absolute;
    top: 50%;
    left: 50%;
    min-width: 100%;
    min-height: 100%;
    transform: translate(-50%, -50%);
    object-fit: cover;
    filter: brightness(0.65) contrast(1.2);
}}
</style>
<div class="hero-container">
  <video autoplay muted loop playsinline>
    <source src="data:video/mp4;base64,{video_base64}" type="video/mp4">
  </video>
</div>
""", unsafe_allow_html=True)

# ---------------------------
# BACKGROUND + HEADER
# ---------------------------
set_background_image(BACKGROUND_IMAGE_PATH)
show_brandbar()

st.markdown("""
<style>
.stTabs [role="tablist"] {
    gap: 24px;
    justify-content: center;
    border-bottom: 2px solid rgba(255,255,255,0.2);
    margin-bottom: 30px;
}
.stTabs [role="tab"] {
    font-size: 1.2rem !important;
    font-weight: 600 !important;
    padding: 20px 35px !important;
    border-radius: 12px 12px 0 0 !important;
    background: rgba(255,255,255,0.05);
    color: #e0e0e0 !important;
    transition: all 0.3s ease;
    border: 2px solid rgba(255,255,255,0.1);
    border-bottom: none !important;
}
.stTabs [role="tab"][aria-selected="true"] {
    background: rgba(0, 0, 0, 0.55) !important;
    color: #ffffff !important;
    box-shadow: 0 4px 15px rgba(0,0,0,0.4);
    border: 2px solid rgba(255,255,255,0.25);
    border-bottom: none !important;
}
.stTabs [role="tab"]:hover {
    background: rgba(255,255,255,0.1);
    color: #fff !important;
    transform: translateY(-2px);
}
</style>
""", unsafe_allow_html=True)

# 🚨 عدلنا هنا: شلنا Satellite
tabs = st.tabs(["Intro", "Evaluation", "Test Model"])

# ---------------------------
# TAB 1: INTRO
# ---------------------------
with tabs[0]:
    st.subheader("AI-Powered Oil Spill Detection from Satellite Imagery")

    st.markdown("""
    ### 🌍 Overview
    **Petra** (a blend of **Petroleum** and **Terra**, meaning **Oil + Earth**) is a Computer Vision project 
    designed to **detect oil spills in our oceans using satellite imagery**.  
    Our mission is to support environmental protection and marine safety through 
    advanced deep learning techniques and real-time monitoring.

    ### 📡 Core Idea
    - Petra uses **YOLO** trained on real **SAR and optical satellite images**.
    - The model can **detect oil slick patterns** across large water surfaces.
    - It can integrate with **FastAPI** as a backend and a **Streamlit dashboard** for real-time monitoring.

    ### 🧠 Technical Highlights
    - **Image Preprocessing:** Images are enhanced, normalized, and resized for YOLO input.
    - **Data Augmentation:** Improves generalization across different lighting, angles, and resolutions.
    - **Model Architecture:**  YOLO Architecture. with 125 layers
    - **Deployment:** FastAPI backend for prediction + Streamlit front-end.

    ### 🌊 Why It Matters
    - Oil spills threaten marine ecosystems and coastal economies.
    - Early detection can help authorities react faster and reduce damage.
    - Petra provides a scalable and automated solution for **continuous satellite monitoring**.

    ### 📈 Future Goals
    - Expand dataset with real-time Sentinel-1 SAR imagery
    - Add geolocation-based detection on global map
    - Provide API endpoints for integration with maritime authorities
    """)

    st.info("Scroll to the next tabs to explore the evaluation and run your own predictions.")


# ---------------------------
# TAB 2: EVALUATION
# ---------------------------
with tabs[1]:
    st.markdown("## 🧠 Evaluation of Petra YOLO Model")
    st.caption("Architecture • Labels • Performance Metrics")

    # --- Section 1: YOLO Architecture ---
    st.markdown("### 📐 YOLO Architecture")
    st.markdown("""
### **A. Backbone (Feature Extraction):**
- Purpose: Extract hierarchical features from input images
- Structure: conv layers + residual connections
- Features: CSP + SPP

### **B. Neck (Feature Fusion):**
- Purpose: Combine features from different scales
- Structure: FPN + PAN
- Why: Oil spills appear in different sizes

### **C. Head (Detection Output):**
- Purpose: Generate final predictions
- Outputs: bounding boxes, objectness, class probs
""")

    st.info("This model was trained on real satellite images labeled manually as **Oil Spill** or **Gas Spill**.")

    st.markdown("### 📊 Evaluation Metrics on Validation Set")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1: st.metric("Accuracy", "93.7%")
    with col2: st.metric("Precision", "85.7%")
    with col3: st.metric("Recall", "91.9%")
    with col4: st.metric("F1-Score", "88.7")
    with col5: st.metric("Loss", "0.847")

    st.success("✅ Petra achieved over **93.7% accuracy** distinguishing oil spills from clean sea surfaces.")


# ---------------------------
# TAB 3: TEST MODEL
# ---------------------------
with tabs[2]:
    st.markdown("### Run Inference (FastAPI)")
    st.caption(f"Endpoint: `{FASTAPI_URL}`  •  Update with env var FASTAPI_URL")

    left, right = st.columns([1.2, 1])
    with left:
        st.subheader("Upload Image")
        file = st.file_uploader("Satellite image (JPG/PNG)", type=["jpg", "jpeg", "png"])
        run = st.button("🚀 Predict with Visualization", type="primary", use_container_width=True, disabled=(file is None))
        if run and file is not None:
            with st.spinner("🔄 Analyzing satellite image..."):
                ok, resp = call_fastapi_predict_with_visual(file.read(), file.name)
            if ok:
                st.success("✅ Analysis complete!")
                display_detection_results(resp)
            else:
                st.error("❌ Analysis failed")
                st.json(resp)

        st.divider()
        st.subheader("Or Predict by Image URL")
        url_in = st.text_input("Public image URL", placeholder="https://…/satellite.jpg")
        run_url = st.button("🌐 Predict URL with Visualization", use_container_width=True, disabled=(not url_in))
        if run_url and url_in:
            with st.spinner("🔄 Downloading and analyzing image..."):
                ok, resp = call_fastapi_predict_url_with_visual(url_in)
            if ok:
                st.success("✅ Analysis complete!")
                display_detection_results(resp)
            else:
                st.error("❌ Analysis failed")
                st.json(resp)

    with right:
        st.subheader("Preview")
        if file is not None:
            try:
                img = Image.open(file).convert("RGB")
                st.image(img, caption=file.name, use_container_width=True)
            except Exception:
                st.info("Preview not available. The file will still be sent to the API.")

    st.caption("FastAPI returns annotated images with bounding boxes around detected oil spills.")

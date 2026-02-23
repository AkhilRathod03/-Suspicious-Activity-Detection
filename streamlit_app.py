import streamlit as st
import cv2
import os
import shutil
from imageai.Prediction.Custom import CustomImagePrediction
import pandas as pd
from PIL import Image

st.set_page_config(page_title="Suspicious Activity Detection", layout="wide")

st.title("📹 Human Suspicious Activity Detection")
st.markdown("""
This application detects suspicious activities from CCTV footage using **ResNet** Deep Learning model.
It extracts frames from the video and analyzes them for potential threats.
""")

# Setup Sidebar
st.sidebar.header("Configuration")
uploaded_video = st.sidebar.file_uploader("Upload CCTV Footage", type=["mp4", "webm", "avi"])
frame_limit = st.sidebar.slider("Number of frames to analyze", 10, 500, 100)

# Paths
MODEL_PATH = "model.h5"
JSON_PATH = "model_class.json"
FRAME_DIR = "frames_st"

def load_model():
    prediction = CustomImagePrediction()
    prediction.setModelTypeAsResNet()
    prediction.setModelPath(MODEL_PATH)
    prediction.setJsonPath(JSON_PATH)
    prediction.loadModel(num_objects=2)
    return prediction

if uploaded_video is not None:
    # Save video locally to process with OpenCV
    with open("temp_video.mp4", "wb") as f:
        f.write(uploaded_video.getbuffer())
    
    st.video("temp_video.mp4")

    if st.button("Extract Frames & Detect Activity"):
        # Prepare Frames Directory
        if os.path.exists(FRAME_DIR):
            shutil.rmtree(FRAME_DIR)
        os.makedirs(FRAME_DIR)

        # 1. Frame Extraction
        st.subheader("🛠️ Extracting Frames...")
        vidObj = cv2.VideoCapture("temp_video.mp4")
        count = 0
        success = True
        progress_bar = st.progress(0)
        
        while success and count < frame_limit:
            success, image = vidObj.read()
            if success:
                cv2.imwrite(f"{FRAME_DIR}/frame{count}.jpg", image)
                count += 1
                progress_bar.progress(count / frame_limit)
        
        st.success(f"Extracted {count} frames successfully.")

        # 2. Activity Detection
        st.subheader("🔍 Analyzing Activity...")
        prediction_obj = load_model()
        
        results = []
        suspicious_found = False
        
        image_files = [f for f in os.listdir(FRAME_DIR) if f.endswith(".jpg")]
        image_files.sort(key=lambda x: int(x.replace("frame", "").replace(".jpg", "")))

        # Prediction Loop
        detection_progress = st.progress(0)
        for i, img_file in enumerate(image_files):
            img_path = os.path.join(FRAME_DIR, img_file)
            predictions, probabilities = prediction_obj.predictImage(img_path, result_count=1)
            
            label = predictions[0]
            confidence = probabilities[0]
            
            if label == "suspicious" and confidence > 80:
                results.append({"Frame": img_file, "Status": label, "Confidence": f"{confidence:.2f}%", "Action": "⚠️ Alert!"})
                suspicious_found = True
            else:
                results.append({"Frame": img_file, "Status": label, "Confidence": f"{confidence:.2f}%", "Action": "✅ Normal"})
            
            detection_progress.progress((i + 1) / len(image_files))

        # 3. Display Results
        df_results = pd.DataFrame(results)
        
        if suspicious_found:
            st.error("🚨 Suspicious Activity Detected in the footage!")
            # Show the first few suspicious frames
            suspicious_frames = df_results[df_results["Status"] == "suspicious"].head(5)
            cols = st.columns(len(suspicious_frames))
            for idx, row in enumerate(suspicious_frames.itertuples()):
                with cols[idx]:
                    img = Image.open(os.path.join(FRAME_DIR, row.Frame))
                    st.image(img, caption=f"{row.Frame} ({row.Confidence})")
        else:
            st.success("✅ No suspicious activity detected.")

        st.table(df_results)

else:
    st.info("Please upload a video file to start detection.")

st.markdown("---")
st.caption("Powered by Streamlit, OpenCV, and ImageAI")

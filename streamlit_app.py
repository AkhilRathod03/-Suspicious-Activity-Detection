import streamlit as st
import cv2
import os
import shutil
import tensorflow as tf
import numpy as np
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

@st.cache_resource
def load_keras_model():
    """Load the model directly using Keras to avoid ImageAI dependency issues."""
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def preprocess_image(img_path):
    """Preprocess image for ResNet (Standard 224x224)."""
    img = Image.open(img_path).convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    # ResNet normalization (ImageAI typically uses 1/255.0)
    img_array = img_array.astype('float32') / 255.0
    return img_array

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
        model = load_keras_model()
        
        if model:
            results = []
            suspicious_found = False
            
            image_files = [f for f in os.listdir(FRAME_DIR) if f.endswith(".jpg")]
            image_files.sort(key=lambda x: int(x.replace("frame", "").replace(".jpg", "")))

            # Classes based on model_class.json: 0=normal, 1=suspicious
            labels = ["normal", "suspicious"]

            detection_progress = st.progress(0)
            for i, img_file in enumerate(image_files):
                img_path = os.path.join(FRAME_DIR, img_file)
                img_input = preprocess_image(img_path)
                
                prediction = model.predict(img_input, verbose=0)
                # Keras returns an array of probabilities
                class_idx = np.argmax(prediction[0])
                confidence = float(prediction[0][class_idx]) * 100
                label = labels[class_idx]
                
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
            st.error("Could not load the model. Please check if model.h5 is correct.")

else:
    st.info("Please upload a video file to start detection.")

st.markdown("---")
st.caption("Powered by Streamlit, OpenCV, and TensorFlow")

# Human Suspicious Activity Detection

This project uses Deep Learning (ResNet via ImageAI) to detect suspicious human activities from CCTV footage. It processes video files, extracts frames, and analyzes them to identify potential threats.

## Features
- Upload and process CCTV video footage (.mp4, .webm).
- Automatic frame extraction using OpenCV.
- Suspicious activity detection using a pre-trained ResNet model.
- High-probability alerts for detected threats.

## Prerequisites
- Python 3.6 - 3.8 (recommended for ImageAI/TensorFlow compatibility)
- OpenCV
- ImageAI
- TensorFlow

## Installation

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd Human-Suspicious-Activity-Detection
   ```

2. **Install dependencies:**
   ```bash
   pip install opencv-python imageai==2.1.5 tensorflow==1.15.0 keras==2.3.1 imutils
   ```

3. **Download Model:**
   Ensure `model.h5` and `model_class.json` are in the project root.

## Execution

### Option 1: Desktop App (Tkinter)
```bash
python SuspiciousDetection.py
```

### Option 2: Web App (Streamlit)
```bash
streamlit run streamlit_app.py
```

## Project Structure
- `SuspiciousDetection.py`: Tkinter-based desktop application.
- `streamlit_app.py`: Streamlit-based web application.
- `model.h5`: Pre-trained ResNet model.
- `model_class.json`: Model class configuration.
- `videos/`: Sample footage for testing.

## Authors
- **Akhil Rathod** - [GitHub Profile](https://github.com/AkhilRathod03)

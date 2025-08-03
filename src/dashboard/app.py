import os
import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import torch
import time

st.set_page_config(layout="wide", page_title="Smart Office Detection Dashboard", initial_sidebar_state="expanded")

# Load the trained model
@st.cache_resource
def load_model():
    model_path = '../model/best_model.pt'
    return YOLO(model_path)

model = load_model()

# css styling
st.markdown(
    """
    <style>
    .sidebar .sidebar-content {
        background-color: #2e2e2e;
        color: white;
        width: 300px !important;
    }
    .main .block-container {
        background-color: #1e1e1e;
        color: white;
    }
    .stImage > img {
        max-width: 100%;
        height: auto;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar
with st.sidebar:
    st.title("Smart Office Detection")
    st.write("Upload an image, video, or GIF to detect objects (Person, Chair, Monitor, Keyboard, Laptop, Phone).")
    input_type = st.selectbox("Select Input Type", ["Image", "Video", "GIF"], index=0)
    uploaded_file = st.file_uploader("Upload File", type=["jpg", "jpeg", "png", "mp4", "gif"], key="media_uploader")

    if uploaded_file is not None:
        # Save uploaded file temporarily
        file_path = os.path.join("temp", f"uploaded_{input_type.lower()}.{uploaded_file.name.split('.')[-1]}")
        os.makedirs("temp", exist_ok=True)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Display original file in sidebar
        if input_type == "Image":
            image = cv2.imread(file_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            st.image(image, caption="Original Image", use_container_width=True)
        elif input_type in ["Video", "GIF"]:
            st.video(file_path)

# Ouput area
if uploaded_file is not None:
    with st.container():
        st.write("### Processing Media...")

        if input_type == "Image":
            results = model.predict(source=file_path, conf=0.5, device='cuda' if torch.cuda.is_available() else 'cpu')
            result = results[0]
            annotated_image = result.plot()
            annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
            st.image(annotated_image, caption="Detected Objects", use_container_width=True)
        else:  # Video or GIF
            video_placeholder = st.empty()
            cap = cv2.VideoCapture(file_path)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            current_frame = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break  # End of video

                current_frame += 1
                # Perform inference on the current frame
                results = model.predict(source=frame, conf=0.5, device='cuda' if torch.cuda.is_available() else 'cpu')
                result = results[0]
                annotated_frame = result.plot()
                annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

                # Display the annotated frame
                video_placeholder.image(annotated_frame, caption=f"Detected Objects (Frame {current_frame}/{frame_count})", use_container_width=True)
                time.sleep(0.03)  # Approx. 30 FPS

            # Release the video capture
            cap.release()
            video_placeholder.empty()

            # Display detection results from the last frame
            st.write("### Detection Results")
            if hasattr(result, 'boxes'):
                boxes = result.boxes.xyxy.cpu().numpy()
                scores = result.boxes.conf.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy()
                class_names = result.names

                if len(boxes) > 0:
                    for i in range(len(boxes)):
                        st.write(f"Object: {class_names[int(classes[i])]}, Confidence: {scores[i]:.2f}, "
                                 f"Box: [{boxes[i][0]:.0f}, {boxes[i][1]:.0f}, {boxes[i][2]:.0f}, {boxes[i][3]:.0f}]")
                else:
                    st.write("No objects detected with confidence > 0.5.")
            else:
                st.write("Detection results not available for video/GIF stream.")

        # Clean up temporary files after processing is complete
        def remove_temp_file(file_path):
            max_attempts = 5
            for attempt in range(max_attempts):
                try:
                    if os.path.exists(file_path):
                        os.remove(file_path)
                    break
                except PermissionError:
                    if attempt < max_attempts - 1:
                        time.sleep(1)  # Wait 1 second before retrying
                    else:
                        st.warning(f"Failed to delete {file_path} after {max_attempts} attempts.")

        remove_temp_file(file_path)

import streamlit as st
from streamlit_webrtc import webrtc_streamer
from ultralytics import YOLO
import av
import cv2

# Load YOLO model
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

st.title("🎥 Live Object Detection & Tracing")
st.write("Point your camera at objects to identify them in real-time.")

# Sidebar Enhancement Options
st.sidebar.header("Enhancement Options")

# Option 1: Object for Alert
alert_object = st.sidebar.text_input(
    "Enter object name for alert:",
    value="person"
)

# Option 2: Show Object Count
show_count = st.sidebar.checkbox(
    "Show Object Count",
    value=True
)

# Video callback
def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")

    results = model.track(
        img,
        persist=True,
        conf=0.5,
        verbose=False
    )

    annotated_frame = results[0].plot()

    # Get detected objects
    names = results[0].names
    classes = results[0].boxes.cls.tolist() if results[0].boxes else []
    detected = [names[int(i)] for i in classes]

    # Option 1: Alert
    if alert_object.lower() in [x.lower() for x in detected]:
        st.sidebar.error(f"⚠ {alert_object} detected!")

    # Option 2: Count Objects
    if show_count:
        count_dict = {}
        for obj in detected:
            count_dict[obj] = count_dict.get(obj, 0) + 1
        st.sidebar.write("📊 Count:")
        st.sidebar.write(count_dict)

    return av.VideoFrame.from_ndarray(
        annotated_frame,
        format="bgr24"
    )

# Start webcam
webrtc_streamer(
    key="object-detection",
    video_frame_callback=video_frame_callback,
    async_processing=True,
    rtc_configuration={
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]}
        ]
    },
    media_stream_constraints={
        "video": True,
        "audio": False
    },
)
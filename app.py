import streamlit as st
from ultralytics import YOLO
from PIL import Image

st.set_page_config(page_title="Food Detector", layout="centered")
st.title("Food Detection with YOLOv8")

@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

conf_threshold = st.slider("Confidence Threshold", 0.05, 0.95, 0.25, 0.05)
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    results = model.predict(source=image, conf=conf_threshold)
    annotated_img = results[0].plot()[:, :, ::-1]  # BGR -> RGB
    st.image(annotated_img, caption="Detected Objects", use_container_width=True)

import streamlit as st
import tensorflow as tf
from pathlib import Path
import numpy as np

from src.model import build_model
from src.gradcam import save_gradcam

MODEL_PATH = Path("models/final_model.h5")
model = build_model()
if MODEL_PATH.exists():
    model.load_weights(MODEL_PATH)

st.title("Pneumonia Detection from Chest X-rays")
uploaded_file = st.file_uploader("Upload Chest X-ray", type=["png", "jpg", "jpeg"])

if uploaded_file:
    img = tf.keras.preprocessing.image.load_img(uploaded_file, target_size=model.input_shape[1:3])
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    pred = model.predict(np.expand_dims(img_array / 255.0, axis=0))[0][0]
    label = "PNEUMONIA" if pred > 0.5 else "NORMAL"
    st.write(f"Prediction: **{label}** ({pred:.2f})")

    gradcam_path = Path("data/app_gradcam.png")
    save_gradcam(model, uploaded_file, gradcam_path)
    st.image(str(gradcam_path), caption="Grad-CAM")

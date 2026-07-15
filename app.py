# pip install opencv-python
# pip install streamlit-drawable-canvas

import streamlit as st
import numpy as np
import cv2
from streamlit_drawable_canvas import st_canvas
from tensorflow.keras.models import load_model

# Load model
model = load_model("digit_recognition_model.keras")

st.set_page_config(
    page_title="Handwritten Digit Recognition",
    page_icon="✍️"
)

st.title("✍️ Handwritten Digit Recognition")
st.write("Draw a digit (0-9) and click Predict.")

# Canvas
canvas_result = st_canvas(
    fill_color="rgba(0,0,0,0)",
    stroke_width=15,
    stroke_color="#FFFFFF",
    background_color="#000000",
    width=280,
    height=280,
    drawing_mode="freedraw",
    key="canvas",
)

if st.button("Predict"):

    if canvas_result.image_data is not None:

        # Convert canvas to image
        img = canvas_result.image_data.astype(np.uint8)

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)

        # Resize to MNIST size
        gray = cv2.resize(gray, (28, 28))

        # Normalize
        gray = gray.astype("float32") / 255.0

        # Show processed image
        st.subheader("Processed Image")
        st.image(gray, width=150)

        # Dense model input
        gray_input = gray.reshape(1, 784)

        # Prediction
        prediction = model.predict(gray_input, verbose=0)

        digit = np.argmax(prediction)
        confidence = np.max(prediction) * 100

        st.success(f"Predicted Digit : {digit}")
        st.info(f"Confidence : {confidence:.2f}%")

        st.subheader("Prediction Probabilities")
        st.write(prediction)

    else:
        st.warning("Please draw a digit first.")
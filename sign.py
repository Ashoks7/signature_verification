import streamlit as st
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Function to load and process image
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

# Function to predict class and confidence
def predict_class(model, img):
    predictions = model.predict(img)
    # Threshold for class decision
    threshold = 0.5
    # Get class label and confidence percentage
    if predictions[0][0] > threshold:
        class_label = 'Real'
        confidence = predictions[0][0] * 100
    else:
        class_label = 'Fake'
        confidence = (1 - predictions[0][0]) * 100
    return class_label, confidence

# Load the saved models
model_paths = {
    "Customer id 1": "C:/Users/ADMIN/sign_for_id1_new_model_2.h5",
    "Customer id 2": "C:/Users/ADMIN/sign_for_id2_new_model_1.h5",
    "Customer id 3": "C:/Users/ADMIN/sign_for_id3_new_model_5.h5"
}

models = {}
for name, path in model_paths.items():
    models[name] = load_model(path)

# Streamlit app
# Load the banner image
banner_image = "https://sqnbankingsystems.com/wp-content/uploads/2021/02/Forged-Signatures-300x154.jpg"

# Display the banner image
st.image(banner_image, use_column_width=True)



st.title("Signature Verification")
model_choice = st.selectbox("Select Customer", list(models.keys()))

uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    if st.button("Check"):
            # Display processing spinner
        with st.spinner('Processing...'):
            img = preprocess_image(uploaded_file)
            class_label, confidence = predict_class(models[model_choice], img)
            # Display the uploaded image
            st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
            st.write(f"Class: {class_label}")
            st.write(f"Confidence: {confidence:.2f}%")

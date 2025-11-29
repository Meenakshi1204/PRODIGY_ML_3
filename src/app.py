import streamlit as st
import numpy as np
from PIL import Image
import joblib

# Import your feature extractor
from src.feature_extractor import extract_features

# Load trained model
model = joblib.load("models/svm_model.pkl")

st.title("🐶🐱 Dog vs Cat Classifier")
st.write("Upload an image to classify it as Dog or Cat.")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Extract features
    features = extract_features(uploaded_file)

    # Predict
    prediction = model.predict([features])[0]
    proba = model.predict_proba([features])[0]

    label = "🐱 Cat" if prediction == 0 else "🐶 Dog"

    st.subheader(f"Prediction: **{label}**")
    st.write("Confidence:")
    st.write(f"Cat: {proba[0]:.4f}  |  Dog: {proba[1]:.4f}")

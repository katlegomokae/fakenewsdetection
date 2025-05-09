import streamlit as st
import joblib
import numpy as np

# Load model and vectorizer
vectorizer = joblib.load("vectorizer.jb")
model = joblib.load("lr_model.jb")

st.set_page_config(page_title="Fake News Detector", layout="centered")
st.markdown("<h1 style='text-align: center; color: black;'>📰 Fake News Detector</h1>", unsafe_allow_html=True)
st.markdown("---")

# Input selection
option = st.radio("Choose input type:", ["Paste Text", "Upload File"])
input_text = ""

if option == "Paste Text":
    input_text = st.text_area("Paste a news article or headline below:")
elif option == "Upload File":
    uploaded_file = st.file_uploader("Upload a .txt file", type=["txt"])
    if uploaded_file is not None:
        input_text = uploaded_file.read().decode("utf-8")

# Prediction and flagging
if st.button("🕵️‍♀️ CHECK"):
    if input_text.strip() != "":
        # Transform the input text for prediction
        transform_input = vectorizer.transform([input_text])
        
        # Get model prediction and probability estimates
        prediction = model.predict(transform_input)[0]
        probabilities = model.predict_proba(transform_input)[0]
        
        # Calculate overall confidence (maximum probability over classes)
        confidence = np.max(probabilities)
        confidence_percent = round(confidence * 100, 2)
        
        # Determine label assuming class 1 = 'REAL' and class 0 = 'FAKE'
        label = "REAL" if prediction == 1 else "FAKE"
        
        # Flag based on confidence:
        # - If confidence is between 40% and 60% (0.4 <= confidence < 0.6): Uncertain.
        # - Additionally, if confidence is less than 50%: mark as Very Uncertain.
        if confidence >= 0.6:
            flag = "✅ Confident Prediction"
        elif 0.5 <= confidence < 0.6:
            flag = "⚠️ Uncertain — Please Review"
        else:  # confidence < 0.5
            flag = "❗ Very Uncertain — Immediate Review Recommended"
        
        # Display results
        if prediction == 1:
            st.markdown(f"<h2 style='color: green;'>{label}</h2>", unsafe_allow_html=True)
        else:
            st.markdown(f"<h2 style='color: red;'>{label}</h2>", unsafe_allow_html=True)
            
        st.markdown(f"**Confidence:** {confidence_percent}%")
        st.markdown(f"**Status:** {flag}")
    else:
        st.warning("Please enter or upload some text to check.")

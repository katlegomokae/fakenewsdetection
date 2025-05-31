import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os
import csv
from sklearn.metrics import accuracy_score, precision_score, recall_score

# === PAGE CONFIGURATION ===
st.set_page_config(page_title="Fake News Detector", layout="wide")
st.markdown("<h1 style='text-align: center; color: black;'>📰 Fake News Detector</h1>", unsafe_allow_html=True)
st.markdown("---")

# === LOAD MODEL & DATA ===
try:
    model = joblib.load("lr_model.jb")
    vectorizer = joblib.load("vectorizer.jb")
except Exception as e:
    st.error(f"🚨 Failed to load model or vectorizer: {e}")
    st.stop()

@st.cache_data
def load_test_data():
    return pd.read_csv("test_data1.csv")

test_data = load_test_data()
X_test = vectorizer.transform(test_data['text'])
y_true = test_data['label']
y_pred = model.predict(X_test)

# === METRICS ===
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)

# === LAYOUT: INPUT AND METRICS SIDE-BY-SIDE ===
col1, col2 = st.columns(2)

# === COLUMN 1: USER INPUT ===
with col1:
    st.subheader("📝 Enter or Upload News Article")
    option = st.radio("Choose input type:", ["Paste Text", "Upload File"])
    input_text = ""

    if option == "Paste Text":
        input_text = st.text_area("Paste the news text here:", height=250)
    elif option == "Upload File":
        uploaded_file = st.file_uploader("Upload a .txt file", type=["txt"])
        if uploaded_file is not None:
            input_text = uploaded_file.read().decode("utf-8")

    if st.button("🕵️‍♀️ CHECK"):
        if input_text.strip() != "":
            vect_input = vectorizer.transform([input_text])
            prediction = model.predict(vect_input)[0]
            probabilities = model.predict_proba(vect_input)[0]
            confidence = np.max(probabilities)
            confidence_percent = round(confidence * 100, 2)
            label = "REAL" if prediction == 1 else "FAKE"

            # Confidence flagging
            if confidence >= 0.6:
                flag = "✅ Confident Prediction"
            elif 0.4 <= confidence < 0.6:
                flag = "⚠️ REVIEW — Model is Uncertain"
            else:
                flag = "❗ REVIEW — Very Low Confidence (Likely Unfamiliar News)"

            # Display results
            st.markdown(f"<h2 style='color: {'green' if prediction == 1 else 'red'};'>{label}</h2>", unsafe_allow_html=True)
            st.markdown(f"**Confidence:** {confidence_percent}%")
            st.markdown(f"**Status:** {flag}")

            # Save flagged result
            if confidence < 0.6:
                flagged_data = {
                    "text": input_text.strip(),
                    "predicted_label": label,
                    "confidence": confidence_percent,
                    "flag": flag
                }
                file_exists = os.path.isfile("flagged_reviews.csv")
                with open("flagged_reviews.csv", mode='a', newline='', encoding='utf-8') as file:
                    writer = csv.DictWriter(file, fieldnames=flagged_data.keys())
                    if not file_exists:
                        writer.writeheader()
                    writer.writerow(flagged_data)
        else:
            st.warning("⚠️ Please enter or upload some text.")

# === COLUMN 2: MODEL PERFORMANCE ===
with col2:
    st.subheader("📊 Model Performance Metrics")
    st.metric("Accuracy", f"{accuracy:.2f}")
    st.metric("Precision", f"{precision:.2f}")
    st.metric("Recall", f"{recall:.2f}")
    
    st.write("**Test Data Class Distribution:**")
    st.bar_chart(test_data["label"].value_counts())

# === REVIEW FLAGGED PREDICTIONS ===
st.markdown("---")
st.subheader("🔎 Review Flagged Predictions")

if os.path.exists("flagged_reviews.csv"):
    flagged_df = pd.read_csv("flagged_reviews.csv")
    if st.checkbox("Show Flagged Predictions Table"):
        st.dataframe(flagged_df, use_container_width=True)

    csv_data = flagged_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Flagged Reviews",
        data=csv_data,
        file_name='flagged_reviews.csv',
        mime='text/csv'
    )
else:
    st.info("ℹ️ No flagged predictions yet. Once low-confidence news is detected, it will appear here.")

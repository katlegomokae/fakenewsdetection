import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from wordcloud import WordCloud

# Set page config
st.set_page_config(page_title="Fake News Detector", layout="wide")
st.markdown("<h1 style='text-align: center; color: black;'>📰 Fake News Detector</h1>", unsafe_allow_html=True)
st.markdown("---")

# Load model and vectorizer
model = joblib.load("lr_model.jb")
vectorizer = joblib.load("vectorizer.jb")

# Load test data for evaluation
test_data = pd.read_csv("test_data1.csv")
X_test = vectorizer.transform(test_data['text'])
y_true = test_data['label']
y_pred = model.predict(X_test)

# Compute metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)

# === Layout Columns ===
col1, col2 = st.columns(2)

# === Column 1: User Input ===
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
        # Transform and predict
        vect_input = vectorizer.transform([input_text])
        prediction = model.predict(vect_input)[0]
        probabilities = model.predict_proba(vect_input)[0]
        confidence = np.max(probabilities)
        confidence_percent = round(confidence * 100, 2)
        label = "REAL" if prediction == 1 else "FAKE"

        # Updated Confidence Flagging
        if confidence >= 0.6:
            flag = "✅ Confident Prediction"
        elif 0.4 <= confidence < 0.6:
            flag = "⚠️ REVIEW — Model is Uncertain"
        else:
            flag = "❗ REVIEW — Very Low Confidence (Likely Unfamiliar News)"

        # Display prediction
        if prediction == 1:
            st.markdown(f"<h2 style='color: green;'>{label}</h2>", unsafe_allow_html=True)
        else:
            st.markdown(f"<h2 style='color: red;'>{label}</h2>", unsafe_allow_html=True)

        st.markdown(f"**Confidence:** {confidence_percent}%")
        st.markdown(f"**Status:** {flag}")

        # Save uncertain predictions for future review
        if confidence < 0.6:
            import csv
            import os

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



# === Review Flagged Predictions ===
st.markdown("---")
st.subheader("🔎 Review Flagged Predictions")


if os.path.exists("flagged_reviews.csv"):
    flagged_df = pd.read_csv("flagged_reviews.csv")
    
    if st.checkbox("Show Flagged Predictions Table"):
        st.dataframe(flagged_df, use_container_width=True)

    # Convert DataFrame to CSV for download
    csv_data = flagged_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Flagged Reviews",
        data=csv_data,
        file_name='flagged_reviews.csv',
        mime='text/csv'
    )
else:
    st.info("ℹ️ No flagged predictions yet. Once low-confidence news is detected, it will appear here.")

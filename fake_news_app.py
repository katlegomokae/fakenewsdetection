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
test_data = pd.read_csv("test_data.csv")
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
            label = "REAL" if prediction == 0 else "FAKE"

            # Confidence-based flag
            if confidence >= 0.6:
                flag = "✅ Confident Prediction"
            elif 0.5 <= confidence < 0.6:
                flag = "⚠️ Uncertain — Please Review"
            else:
                flag = "❗ Very Uncertain — Immediate Review Recommended"

            # Show results
            if prediction == 1:
                st.markdown(f"<h2 style='color: green;'>{label}</h2>", unsafe_allow_html=True)
            else:
                st.markdown(f"<h2 style='color: red;'>{label}</h2>", unsafe_allow_html=True)

            st.markdown(f"**Confidence:** {confidence_percent}%")
            st.markdown(f"**Status:** {flag}")
        else:
            st.warning("⚠️ Please enter or upload some text.")

# === Column 2: Model Performance ===
with col2:
    st.subheader("📊 Model Performance Metrics")
    st.metric("Accuracy", f"{accuracy:.2f}")
    st.metric("Precision", f"{precision:.2f}")
    st.metric("Recall", f"{recall:.2f}")

    if st.checkbox("Show Confusion Matrix"):
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=["Fake", "Real"], yticklabels=["Fake", "Real"])
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        st.pyplot(fig)

# === Word Cloud Section ===
st.markdown("---")
st.subheader("☁️ Word Clouds")

real_news = test_data[test_data["label"] == 1]["text"].tolist()
fake_news = test_data[test_data["label"] == 0]["text"].tolist()

real_text = " ".join(real_news)
fake_text = " ".join(fake_news)

if st.checkbox("Show Word Clouds"):
    col3, col4 = st.columns(2)
    with col3:
        st.markdown("### 🔵 Real News Word Cloud")
        real_wc = WordCloud(width=400, height=200, background_color='white').generate(real_text)
        st.image(real_wc.to_array())

    with col4:
        st.markdown("### 🔴 Fake News Word Cloud")
        fake_wc = WordCloud(width=400, height=200, background_color='white').generate(fake_text)
        st.image(fake_wc.to_array())

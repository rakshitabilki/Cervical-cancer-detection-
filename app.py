import streamlit as st
import numpy as np
import cv2
import joblib
import json
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="Cervical Cancer Detection", layout="wide")

# -------------------------------
# CUSTOM CSS (UI DESIGN)
# -------------------------------
st.markdown("""
<style>
.main {
    background-color: #0e1117;
}
h1 {
    color: #00e6ac;
}
.stButton>button {
    background-color: #00e6ac;
    color: black;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# LOAD MODELS
# -------------------------------
lr = joblib.load("lr_model.pkl")
svm = joblib.load("svm_model.pkl")
knn = joblib.load("knn_model.pkl")
rf = joblib.load("rf_model.pkl")

# -------------------------------
# LOAD METRICS
# -------------------------------
with open("metrics.json", "r") as f:
    metrics = json.load(f)

df = pd.DataFrame(list(metrics.items()), columns=["Model", "Accuracy"])

IMG_SIZE = 128

# -------------------------------
# SIDEBAR
# -------------------------------
st.sidebar.title("🧬 Settings")
model_choice = st.sidebar.selectbox("Select Model", df["Model"])

st.sidebar.markdown("""
### 📌 About
AI-based cervical cancer detection using ML & Deep Learning.

### 👩‍⚕️ Use Case
Helps in early diagnosis and screening.
""")

# -------------------------------
# HEADER
# -------------------------------
st.markdown("""
# 🧬 Cervical Cancer Detection System
### 🔍 AI-powered Medical Image Classification
""")

# -------------------------------
# LAYOUT (2 COLUMNS)
# -------------------------------
col1, col2 = st.columns([1, 1])

# -------------------------------
# IMAGE UPLOAD
# -------------------------------
with col1:
    st.subheader("📤 Upload Image")

    uploaded_file = st.file_uploader(
        "Upload cervical cell image",
        type=["jpg", "png", "jpeg", "bmp"]
    )

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)

        st.image(image, caption="Uploaded Image", use_column_width=True)

# -------------------------------
# PREPROCESS FUNCTION
# -------------------------------
def preprocess(image):
    img = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    flat = img.reshape(1, -1)
    cnn_img = img.reshape(1, IMG_SIZE, IMG_SIZE, 3)
    return flat, cnn_img

# -------------------------------
# PREDICTION
# -------------------------------
with col2:
    st.subheader("🧠 Prediction Result")

    if uploaded_file is not None:
        flat, cnn_img = preprocess(image)

        if model_choice == "Logistic Regression":
            result = lr.predict(flat)[0]
            confidence = None

        elif model_choice == "SVM":
            result = svm.predict(flat)[0]
            confidence = None

        elif model_choice == "KNN":
            result = knn.predict(flat)[0]
            confidence = None

        elif model_choice == "Random Forest":
            result = rf.predict(flat)[0]
            confidence = None

        # DISPLAY RESULT
        if result == 0:
            st.success("✅ Normal Cell Detected")
        else:
            st.error("⚠️ Cancer Detected")

        # CONFIDENCE
        if confidence is not None:
            st.info(f"Confidence Score: {confidence:.2f}")

# -------------------------------
# DASHBOARD SECTION
# -------------------------------
st.markdown("---")
st.subheader("📊 Model Performance Dashboard")

col3, col4 = st.columns([1, 1])

with col3:
    st.dataframe(df)

with col4:
    fig, ax = plt.subplots()
    ax.bar(df["Model"], df["Accuracy"])
    ax.set_title("Model Comparison")
    plt.xticks(rotation=30)
    st.pyplot(fig)

# -------------------------------
# BEST MODEL
# -------------------------------
best_model = df.loc[df["Accuracy"].idxmax()]
st.success(f"🏆 Best Model: {best_model['Model']} ({best_model['Accuracy']:.2f})")

# -------------------------------
# FOOTER
# -------------------------------
st.markdown("""
---
💡 *Developed using Machine Learning & Deep Learning techniques for medical diagnosis.*
""")
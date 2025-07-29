import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from pymongo import MongoClient
import numpy as np
import hashlib
import os
from datetime import datetime

# -------------------- CONFIG --------------------
IMG_HEIGHT = 128
IMG_WIDTH = 128
MODEL_PATH = 'cnn_monkeypox_model.h5'
model = load_model(MODEL_PATH)

# -------------------- MONGODB DATABASE --------------------
def get_mongo_connection():
    client = MongoClient("mongodb://localhost:27017/")
    db = client["monkeypox_db"]
    return db

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def register_user(username, password, email, phone):
    db = get_mongo_connection()
    users = db["users"]
    if users.find_one({"username": username}):
        return False
    users.insert_one({
        "username": username,
        "password": hash_password(password),
        "email": email,
        "phone": phone
    })
    return True

def login_user(username, password):
    db = get_mongo_connection()
    users = db["users"]
    hashed_pwd = hash_password(password)
    user = users.find_one({"username": username, "password": hashed_pwd})
    return user is not None

# -------------------- IMAGE PREPROCESS --------------------
def preprocess_image(image):
    img = image.resize((IMG_HEIGHT, IMG_WIDTH))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# -------------------- SAVE PREDICTIONS TO DATABASE --------------------
def save_prediction(username, result, confidence):
    db = get_mongo_connection()
    predictions = db["predictions"]
    predictions.insert_one({
        "username": username,
        "result": result,
        "confidence": confidence,
        "timestamp": st.session_state.get("timestamp", None)
    })

# -------------------- UI STYLING --------------------
st.set_page_config(page_title="Monkeypox Detector", layout="centered")
st.markdown("""
    <style>
    .stApp {
        background-image: url("https://images.unsplash.com/photo-1617839620652-2d5da27f38de?ixlib=rb-4.0.3&auto=format&fit=crop&w=1400&q=80");
        background-size: cover;
        background-attachment: fixed;
        color: white;
    }
    h1, h2, h3, h4, h5 {
        color: #fdfdfd;
    }
    </style>
""", unsafe_allow_html=True)

# -------------------- LOGIN PAGE --------------------
def show_login_page():
    st.title("Login Page")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if login_user(username, password):
            st.success("✅ Login successful!")
            st.session_state.logged_in = True
            st.session_state.username = username
        else:
            st.error("❌ Invalid credentials!")

# -------------------- REGISTER PAGE --------------------
def show_register_page():
    st.title("Register Page")
    username = st.text_input("Enter a username")
    password = st.text_input("Enter a password", type="password")
    email = st.text_input("Email ID")
    phone = st.text_input("Phone Number")

    if st.button("Register"):
        if register_user(username, password, email, phone):
            st.success("🎉 Registration successful! Please login.")
        else:
            st.error("⚠️ Username already exists.")

# -------------------- DETECTION PAGE --------------------
def show_detection_page():
    st.title("Monkeypox Detection System")
    st.write("Upload an image to check if it is **Monkeypox** or **Normal**.")
    uploaded_image = st.file_uploader("Upload an image file", type=["jpg", "jpeg", "png"])

    if uploaded_image:
        st.image(uploaded_image, caption="Uploaded Image", use_container_width=True)
        with st.spinner("🔍 Processing..."):
            image = load_img(uploaded_image)
            preprocessed_image = preprocess_image(image)
            prediction = model.predict(preprocessed_image)[0][0]

            if prediction > 0.5:
                result = "Normal"
                confidence = prediction
            else:
                result = "Monkeypox"
                confidence = 1 - prediction

            # Save timestamp
            st.session_state.timestamp = datetime.now()

            # Save to MongoDB
            save_prediction(st.session_state.username, result, float(confidence))

            # Show results to user
            st.success(f"✅ Prediction: **{result}**")
            st.info(f"📊 Confidence: **{confidence * 100:.2f}%**")

            cm_image_path = r"D:\mPOX\output.png"
            if os.path.exists(cm_image_path):
                st.write("### 📉 Confusion Matrix")
                st.image(cm_image_path, caption="Confusion Matrix", use_container_width=True)

# -------------------- NAVIGATION --------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    option = st.selectbox("Choose an option", ["Login", "Register"])
    if option == "Login":
        show_login_page()
    else:
        show_register_page()
else:
    show_detection_page()

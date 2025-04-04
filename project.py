import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import time  # For animation effect
import requests
from io import BytesIO

# Set page configuration
st.set_page_config(page_title="Waste Classifier using CNN", layout="wide")

# Load the trained CNN model (Updated to use local file)
@st.cache_resource
def load_cnn_model():
    return load_model("waste_classifier.h5")  # Ensure this file is in the same directory

model = load_cnn_model()

# Updated Class Labels (9 categories)
class_labels = {
    0: "Cardboard",
    1: "Compost",
    2: "Glass",
    3: "Metal",
    4: "Paper",
    5: "Plastic",
    6: "Trash",
    7: "Organic",
    8: "Recyclable"
}

# Sidebar - About Section
with st.sidebar:
    st.title("ℹ️ About")
    st.write("""
    **Waste Classification Using CNN Model**
    
    - **Detects 9 waste categories**
    - Model trained using a **Convolutional Neural Network (CNN)**
    - **Future Improvements:** Adding hazardous & mixed waste detection
    
    **Thanks for visiting!**
    """)

# Main Title
st.title("♻️ Waste Classification using CNN")

# Upload Image Section
uploaded_file = st.file_uploader("📤 Upload an image for classification", type=["jpg", "png", "jpeg"])

# Option to classify using an Image URL
image_url = st.text_input("🔗 Or enter an Image URL for Classification:")

# Two-Column Layout
col1, col2 = st.columns([1, 1])  # Two equal-width columns

# Function to process and predict waste type
def classify_image(img):
    """Preprocess and classify the uploaded image"""
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize

    # Make prediction
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]  # Get highest probability class
    predicted_label = class_labels[predicted_class]

    return predicted_label, predictions

# Icons for Waste Categories
icon_map = {
    "Cardboard": "📦",
    "Compost": "🌿",
    "Glass": "🍾",
    "Metal": "🔩",
    "Paper": "📄",
    "Plastic": "♳",
    "Trash": "🚮",
    "Organic": "🌱",
    "Recyclable": "🔄"
}

# If an image is uploaded
if uploaded_file is not None:
    with col1:
        st.image(uploaded_file, caption="🖼️ Uploaded Image", use_column_width=True)

    with col2:
        st.write("⏳ **Processing Image...**")
        time.sleep(1)

        img = image.load_img(uploaded_file, target_size=(224, 224))
        predicted_label, predictions = classify_image(img)

        # Show result
        st.success(f"✅ **Prediction: {predicted_label} Waste**")
        st.markdown(f"{icon_map[predicted_label]} **This waste is {predicted_label}!**")

        # Confidence Scores
        st.write("🧠 **Confidence Scores:**")
        for i, label in class_labels.items():
            st.write(f"- {label}: **{predictions[0][i] * 100:.2f}%**")

# If an image URL is entered
elif image_url:
    try:
        response = requests.get(image_url)
        img = image.load_img(BytesIO(response.content), target_size=(224, 224))

        with col1:
            st.image(img, caption="🖼️ Image from URL", use_column_width=True)

        with col2:
            st.write("⏳ **Processing Image...**")
            time.sleep(1)

            predicted_label, predictions = classify_image(img)

            # Show result
            st.success(f"✅ **Prediction: {predicted_label} Waste**")
            st.markdown(f"{icon_map[predicted_label]} **This waste is {predicted_label}!**")

            # Confidence Scores
            st.write("🧠 **Confidence Scores:**")
            for i, label in class_labels.items():
                st.write(f"- {label}: **{predictions[0][i] * 100:.2f}%**")

    except Exception as e:
        st.error(f"❌ Error loading image: {e}")

# Bottom Credits
st.markdown("---")
st.markdown("⚠️ **Note:** The model is still learning and may not be 100% accurate!")

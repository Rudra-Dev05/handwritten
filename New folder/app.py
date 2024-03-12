import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('C:\\Users\\myada\\Desktop\\New folder\\model.h5')

# Function to preprocess the image
def preprocess_image(image):
    img = image.resize((28, 28))
    img = img.convert('L')  # Convert to grayscale
    img_array = img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Streamlit UI
st.title("Handwritten Text Recognition")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Load and preprocess the image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    img_array = preprocess_image(image)

    # Make predictions
    predictions = model.predict(img_array)
    digit = np.argmax(predictions)

    st.write(f"Predicted Digit: {digit}")

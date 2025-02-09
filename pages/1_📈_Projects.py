import streamlit as st
from keras.models import load_model  # TensorFlow is required for Keras to work
from PIL import Image, ImageOps
import numpy as np

st.set_page_config(page_title="Projects", page_icon="📈", layout="wide")

# Custom HTML/CSS for the banner
custom_html = r"""
<div class="banner">
    <img src="https://img.pikbest.com/backgrounds/20200504/technology-blue-minimalist-banner-background_2754199.jpg!bwr800" alt="Banner Image">
</div>
<style>
    .banner {
        width: 160%;
        height: 200px;
        overflow: hidden;
    }
    .banner img {
        width: 100%;
        object-fit: cover;
    }
</style>
"""

# Display the custom HTML
st.components.v1.html(custom_html)


st.sidebar.header("Applications of Image Segmentation")
st.sidebar.text("Image Segmentation is used in a variety of industries, including healthcare, retail, automotive, agriculture and space.")

st.title("Image Segmentation Projects")

import streamlit as st

with st.expander("Pneumonia Classification"):
    st.write("Upload a chest X-ray image, and the model will predict whether it shows signs of pneumonia.")

# Load model and labels
model = load_model("C:/Users/selva/OneDrive/Desktop/openproject_CV_workshop/model/keras_model.h5", compile=False)
class_names = open("C:/Users/selva/OneDrive/Desktop/openproject_CV_workshop/model/labels.txt", "r").readlines()

# User uploads image
uploaded_file = st.file_uploader("Upload a Chest X-ray (JPEG, PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load the image
    image = Image.open(uploaded_file).convert("RGB")
    
    # Display uploaded image
    st.image(image, caption="Uploaded Chest X-ray", use_column_width=True)

    # Preprocess the image
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)  # Resize image
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1  # Normalize

    # Prepare input tensor
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

    # Run prediction
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index].strip()  # Remove whitespace
    confidence_score = prediction[0][index]

    # Display results
    st.write("## Prediction: ", class_name)
    st.write("### Confidence Score: {:.2f}%".format(confidence_score * 100))
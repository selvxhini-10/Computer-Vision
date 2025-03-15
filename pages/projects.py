import streamlit as st
import os
from keras.models import load_model  # TensorFlow is required for Keras to work
from PIL import Image, ImageOps  # Install pillow instead of PIL
import numpy as np

st.set_page_config(page_title="Projects", page_icon="📈", layout="wide")

# Custom HTML/CSS for the banner
custom_html = """
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
import streamlit as st
from streamlit_option_menu import option_menu

# Inject CSS to change the font
st.markdown(
    """
    <style>
        /* Change font for sidebar */
        .css-1d391kg, .css-1v3fvcr {  /* Adjusts sidebar text */
            font-family: 'Source Sans Pro', sans-serif !important;
        }
        [data-testid="stSidebarNav"] {display: none;}
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar Menu 
with st.sidebar:
    st.title("Main Menu")  

    with st.expander("📑 APPS", True):
        # Define checkbox options
        item1 = st.checkbox("Home")
        item2 = st.checkbox("About Me")
        item3 = st.checkbox("Goals")

        # Handle navigation when checkbox is selected
        if item1:
            st.switch_page("🏠_Home.py")  # Replace with actual page path
        if item2:
            st.switch_page("pages/computer_vision.py")
        if item3:
            st.switch_page("pages/nlp.py")

    with st.expander("🤖 PROJECTS", True):
        # Define checkbox options
        item4 = st.checkbox("Pneumonia Classifier")
        item5 = st.checkbox("Item 5 - how are yoou")
        item6 = st.checkbox("Item 6 - testing")

        # Handle navigation when checkbox is selected
        if item4:
            st.switch_page("pages/projects.py") 

        if item5:
            st.switch_page("pages/computer_vision.py")
        if item6:
            st.switch_page("pages/nlp.py")

    # Custom CSS for Indentation
    st.markdown("""
        <style>
        [data-testid="stMarkdownContainer"] ul{
             padding-left: 20px !important;
        }
        .stCheckbox > label { 
            margin-left: 15px !important;
        }
        </style>
    """, unsafe_allow_html=True)

st.sidebar.header("Applications of Image Segmentation")
st.sidebar.text("Image Segmentation is used in a variety of industries, including healthcare, retail, automotive, agriculture and astronomy.")

st.title("Pneumonia Classifier")

st.write("This AI-powered pneumonia classifier analyzes chest X-ray images to detect signs of pneumonia. Using deep learning, the model distinguishes between normal and pneumonia-affected lungs, providing quick and reliable results to support medical diagnosis. Upload a chest X-ray image, and the model will predict whether it shows signs of pneumonia.")

import glob
# Load images from directory
def load_images():
    image_files = glob.glob(r"C:/Users/selva/OneDrive/Desktop/openproject_CV_workshop/images/*/*.jpeg")
    manuscripts = set()

    for image_file in image_files:
        # Extract manuscript folder name (parent folder of image)
        manuscript_name = os.path.basename(os.path.dirname(image_file))
        manuscripts.add(manuscript_name)

    return image_files, sorted(manuscripts)

# Load available images
image_files, manuscripts = load_images()

# Dropdown for users to select manuscripts
view_manuscripts = st.multiselect("Try testing the model! Feel free to select and download normal and pneumonia X-ray images.", manuscripts)
n = 4;
# Display images from the selected manuscript folders
view_images = []
for image_file in image_files:
    if any(manuscript in image_file for manuscript in view_manuscripts):
        view_images.append(image_file)
groups = []
for i in range(0, len(view_images), n):
    groups.append(view_images[i:i+n])

for group in groups:
    cols = st.columns(n)
    for i, image_file in enumerate(group):
        cols[i].image(image_file)

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

import h5py

f = h5py.File(r"C:/Users/selva/OneDrive/Desktop/openproject_CV_workshop/model/keras_model.h5", mode="r+")
model_config_string = f.attrs.get("model_config")

if model_config_string.find('"groups": 1,') != -1:
    model_config_string = model_config_string.replace('"groups": 1,', '')
f.attrs.modify('model_config', model_config_string)
f.flush()

model_config_string = f.attrs.get("model_config")

assert model_config_string.find('"groups": 1,') == -1

# Load the pre-trained model
model = load_model(r"C:/Users/selva/OneDrive/Desktop/openproject_CV_workshop/model/keras_model.h5", compile=False)

class_names = open(r"C:/Users/selva/OneDrive/Desktop/openproject_CV_workshop/model/labels.txt", "r").readlines()

data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)
else:
    st.warning("Please upload an image first.")

# resizing the image to be at least 224x224 and then cropping from the center
size = (224, 224)
image = ImageOps.fit(Image, size, Image.Resampling.LANCZOS)

# turn the image into a numpy array
image_array = np.asarray(image)

# Normalize the image
normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

# Load the image into the array
data[0] = normalized_image_array

# Predicts the model
prediction = model.predict(data)
index = np.argmax(prediction)
class_name = class_names[index].strip()  # Remove whitespace
confidence_score = round(prediction[0][index] * 100, 2)  # Convert to percentage

# Print prediction and confidence score
st.subheader(f"Results")
st.write(f"Prediction: {class_name}")
st.write(f"Confidence Score: {confidence_score}%")

# Results Survey
import streamlit as st

with st.expander("Was the AI model's prediction correct?"):
    survey = st.radio(
        "Select an option below.",
        ["***Yes :white_check_mark:***", "***No :x:***"],
        captions=[
            "All hail AI.",
            "Nah, it sucks."
        ],
    )

    if survey == "***Yes :white_check_mark:***":
        st.write("Your response is noted :)")
    else:
        st.write("We will work on improving the accuracy of the model.")

# Waste Classification System

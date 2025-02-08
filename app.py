import streamlit as st
from PIL import Image
import numpy as np
import cv2
from PIL import Image
import mediapipe as mp
import io
import os
import urllib.request
import base64

import streamlit as st
st.set_page_config(layout="wide")
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

# Sidebar content
st.sidebar.header("Sidebar Title")
st.sidebar.subheader("Subheading")
st.sidebar.text("Sidebar content goes here.")

st.title("Image Segmenter with MediaPipe")

st.markdown("""
This application allows you to upload an image, performs image segmentation using MediaPipe's DeepLab v3 model (or any other model), and displays the segmentation masks and background-blurred images.
""")

# Download the segmentation model if not present
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/image_segmenter/deeplab_v3/float32/1/deeplab_v3.tflite"
MODEL_PATH = "models/deeplabv3.tflite"


@st.cache_resource
def download_model():
    if not os.path.exists("models"):
        os.makedirs("models")
    if not os.path.exists(MODEL_PATH):
        st.info("Downloading Model...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        st.success("Model downloaded successfully!")
    else:
        st.info("Model already exists.")

download_model()

def load_image(image_file):
    image = Image.open(image_file)
    return np.array(image)

def resize_image(image, desired_height=1000, desired_width=1000):
    h, w = image.shape[:2]
    if h < w:
        img = cv2.resize(image, (desired_width, int(h / (w / desired_width))))
    else:
        img = cv2.resize(image, (int(w / (h / desired_height)), desired_height))
    return img

def segment_image(image_bgr):
    mp_image_segmenter = mp.tasks.vision.ImageSegmenter
    mp_base_options = mp.tasks.BaseOptions
    mp_image_segmenter_options = mp.tasks.vision.ImageSegmenterOptions

    # Initialize ImageSegmenter
    base_options = mp_base_options(model_asset_path=MODEL_PATH)
    options = mp_image_segmenter_options(base_options=base_options, output_category_mask=True)
    segmenter = mp_image_segmenter.create_from_options(options)

    # Convert image to MediaPipe Image format
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_bgr)

    # Perform segmentation
    segmentation_result = segmenter.segment(mp_image)
    category_mask = segmentation_result.category_mask

    # Cleanup
    segmenter.close()

    return category_mask

def apply_mask(image_bgr, category_mask, threshold=0.2, mask_color=(255, 255, 255), bg_color=(192, 192, 192)):
    condition = np.stack((category_mask.numpy_view(),) * 3, axis=-1) > threshold
    fg_image = np.full(image_bgr.shape, mask_color, dtype=np.uint8)
    bg_image = np.full(image_bgr.shape, bg_color, dtype=np.uint8)
    output_image = np.where(condition, fg_image, bg_image)
    return output_image

def apply_background_blur(image_rgb, category_mask, threshold=0.1):
    blurred_image = cv2.GaussianBlur(image_rgb, (55, 55), 0)
    condition_blur = np.stack((category_mask.numpy_view(),) * 3, axis=-1) > threshold
    output_blur = np.where(condition_blur, image_rgb, blurred_image)
    return output_blur

def get_image_download_link(img_array, filename):
    img = Image.fromarray(img_array)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    byte_im = buf.getvalue()
    encoded = base64.b64encode(byte_im).decode()
    href = f'<a href="data:image/png;base64,{encoded}" download="{filename}">Download {filename}</a>'
    return href

# 📷 **Camera Input**
captured_image = st.camera_input("Take a picture")

# 📤 **File Upload**
uploaded_file = st.file_uploader("Or upload an image...", type=["jpg", "jpeg", "png"])

# **Ensure only one source is used**
if captured_image:
    image_source = captured_image
elif uploaded_file:
    image_source = uploaded_file
else:
    image_source = None

if image_source is not None:
    try:
        # Load and display original image
        image = load_image(image_source)
        resized_image = resize_image(image)
        st.image(resized_image, caption='Uploaded Image', use_container_width=True)

        # Perform segmentation
        with st.spinner('Performing image segmentation...'):
            category_mask = segment_image(image)

        # Display segmentation mask
        mask_image = apply_mask(image, category_mask, threshold=0.2)
        mask_image_rgb = cv2.cvtColor(mask_image, cv2.COLOR_BGR2RGB)
        resized_mask = resize_image(mask_image_rgb)
        st.image(resized_mask, caption='Segmentation Mask', use_container_width=True)

        # Apply and display background blur
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        blurred_image = apply_background_blur(image_rgb, category_mask, threshold=0.1)
        blurred_image_bgr = cv2.cvtColor(blurred_image, cv2.COLOR_RGB2BGR)
        resized_blurred = resize_image(blurred_image_bgr)
        st.image(resized_blurred, caption='Blurred Background Image', use_container_width=True)

        # Provide download links
        st.markdown(get_image_download_link(mask_image_rgb, "segmentation_mask.png"), unsafe_allow_html=True)
        st.markdown(get_image_download_link(cv2.cvtColor(resized_blurred, cv2.COLOR_BGR2RGB), "blurred_background.png"), unsafe_allow_html=True)

    except Exception as e:
        st.error(f"An error occurred: {e}")



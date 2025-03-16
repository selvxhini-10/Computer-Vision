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

st.set_page_config(page_title="Home", page_icon="🏠", layout="wide")

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

# Sidebar Menu ----------------------------------------------------------------------------
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
        item4 = st.checkbox("Image Segmenter")
        item5 = st.checkbox("Pneumonia Classifier")
        item6 = st.checkbox("Traffic Flow Optimization")

        # Handle navigation when checkbox is selected
        if item4:
            st.switch_page("pages/segmenter.py") 

        if item5:
            st.switch_page("pages/pneumonia.py") 
        if item6:
            st.switch_page("pages/traffic.py")

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

#-------------------------------------------------------------------------------------------
         
# Sidebar content
st.title("Exploring the World Through AI & Computer Vision")
st.write("A collection of innovative projects applying machine learning to real-world problems—from object detection to medical imaging.")

st.header("About This Website")
st.subheader("What is Computer Vision?")
st.write("Short description of CV and why it matters.")

st.subheader("What is Image Segmentation?")
st.write("Image Segmentation is a computer vision technique that seperates a digital image into discrete groups of pixels or image segments for object detection and processing. It aims to simplify an image to make it more meaningful for further analysis.")
st.write("Image Segmentation is used in a variety of industries, including healthcare, retail, automotive, agriculture and astronomy.")
st.image("https://mindy-support.com/wp-content/uploads/2022/10/types-of-image-segmentation-1.jpg", caption=None, width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto", use_container_width=False)

st.header("Applications of AI & CV")
st.write("Object Detection & Recognition")
st.markdown("- Object Detection & Recognition")
st.markdown("- Medical Imaging (Example: Assisting in diagnosing diseases using AI-powered analysis.)")
st.markdown("- Autonomous Systems (Example: Using AI for traffic flow optimization or robotics.)")
st.markdown("Sustainability & Environmental Monitoring (Example: Tracking ocean plastic pollution or wildfire impact.)")

st.markdown('''
<style>
[data-testid="stMarkdownContainer"] ul{
    list-style-position: inside;
}
</style>
''', unsafe_allow_html=True)

# Featured Projects (Showcase) with buttons

# Blog / Research Insights

# Contact / Collaborate 
'''
“Let’s Connect” with links to GitHub, LinkedIn, or email.
'''



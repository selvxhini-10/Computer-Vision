import shap
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import pandas as pd
import numpy as np
import streamlit.components.v1 as components

st.set_page_config(page_title="Traffic Optimization", page_icon="🚦", layout="wide")

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

st.title("Intelligent Traffic Flow Optimization")

url = "https://medium.com/@devarshpatel15062001/building-an-end-to-end-traffic-flow-prediction-system-a-step-by-step-guide-3f201c7a9c9f"
st.header("Displaying SHAP analysis for AI-based traffic flow prediction.")

st.write("I learned to build an end-to-end traffic flow prediction system using neural networks with the help of this [comprehensive blog.](%s)" % url)

st.subheader("Step-by-Step Process")
st.write("The METR-LA traffic dataset was utilized for data collection as it contains valuable information from over 200 loop detectors in the Los Angeles traffic network. The dataset is also widely used in traffic forecasting and graphing neural networks. Jupyter Notebook was used to load and explore the dataset to better understand the structure, features and format of the data. ")
url = "https://www.kaggle.com/datasets/annnnguyen/metr-la-dataset"
st.write("[METR-LA Traffic Dataset](%s)" % url)

# Data Collection 

import pandas as pd

# Load the dataset
data = pd.read_csv("METR-LA.csv")

data = pd.read_csv("METR-LA.csv", usecols=range(2))  # Reads only the first 2 columns

# Convert timestamp column if necessary
data["timestamp"] = pd.to_datetime(data["timestamp"], errors='coerce')

# Streamlit App
st.subheader("Traffic Volume Over Time Plot")

# Create the plot
fig, ax = plt.subplots(figsize=(12, 6))
sns.lineplot(x="timestamp", y="volume", data=data, ax=ax)
ax.set_title("Traffic Volume Over Time")
ax.set_xlabel("Timestamp")
ax.set_ylabel("Traffic Volume")
ax.tick_params(axis="x", rotation=45)

# Display the plot in Streamlit
st.pyplot(fig)

# Check for missing values
missing_values = data.isnull().sum()
print(missing_values)

data["timestamp"] = pd.to_datetime(data["timestamp"])
data["hour_of_day"] = data["timestamp"].dt.hour
data["day_of_week"] = data["timestamp"].dt.dayofweek
data["is_weekend"] = data["timestamp"].dt.weekday 

st.write("Matplotlib and Seaborn were used to visualize traffic volume trends over time and explore data patterns. A basic linear regression model was used in the project to effectively describe the relationship between a dependent variable, y, for traffic volume, and an independent variable, x, for the timestamps (e.g. 'hour_of_day', 'day_of_week', 'is_weekend'). "
"Once a basic model was chosen, a simple neural network was built using TensorFlow and Keras.")

# Model Development
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Prepare the data
X = data[["hour_of_day", "day_of_week", "is_weekend"]]
y = data["volume"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
predictions = model.predict(X_test)

# Evaluate the model
st.subheader("Model Evaluation")
st.write("The model's performance is evaluated using metrics like Mean Squared Error (MSE).")
mse = mean_squared_error(y_test, predictions)
print(f"Mean Squared Error: {mse}")
st.write(f"Mean Squared Error: {mse}")

from sklearn.ensemble import RandomForestRegressor

# Initialize and train the random forest model
model_rf = RandomForestRegressor()
model_rf.fit(X_train, y_train)

# Make predictions on the test set
predictions_rf = model_rf.predict(X_test)

# Evaluate the model
mse_rf = mean_squared_error(y_test, predictions_rf)
print(f"Mean Squared Error (Random Forest): {mse_rf}")
st.write(f"Mean Squared Error (Random Forest): {mse_rf}")

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Build a simple neural network
model_nn = Sequential()
model_nn.add(Dense(units=10, activation="relu", input_dim=3))
model_nn.add(Dense(units=1, activation="linear"))
model_nn.compile(optimizer="adam", loss="mean_squared_error")

# Train the neural network
model_nn.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

import shap

shap.initjs()

# Assume model_nn is already trained
explainer = shap.Explainer(model_nn, X_test)  # Ensure X_test is provided
shap_values = explainer(X_test)

# Summary plot (works fine in Streamlit)
st.subheader("SHAP Summary Plot")
fig, ax = plt.subplots()
shap.summary_plot(shap_values, X_test, show=False)  # Prevents SHAP from auto-displaying
st.pyplot(fig)  # Show in Streamlit

# Force plot workaround (convert to HTML)
st.subheader("SHAP Force Plot")

# Convert SHAP force plot to HTML
force_plot_html = shap.force_plot(explainer.expected_value, shap_values.values[0,:], X_test.iloc[0,:])._repr_html_()

# Display using Streamlit components
import streamlit.components.v1 as components
components.html(force_plot_html, height=400)

# Use KernelExplainer for neural networks
'''
explainer = shap.Explainer(model_nn, X_train)
shap_values = explainer(X_test)

# SHAP Summary Plot
st.subheader("SHAP Summary Plot")
fig, ax = plt.subplots()
shap.summary_plot(shap_values.values, X_test, show=False)  # Prevent auto-display
st.pyplot(fig)  # Display in Streamlit

# SHAP Force Plot (Workaround using HTML)
st.subheader("SHAP Force Plot")

# Ensure base_values exist
base_value = shap_values.base_values[0] if hasattr(shap_values, 'base_values') else None

# Generate the force plot (matplotlib=False for HTML output)
force_plot_html = shap.plots.force(base_value, shap_values.values[0, :], X_test.iloc[0, :], matplotlib=False)

# Render SHAP Force Plot in Streamlit
components.html(force_plot_html, height=400)
'''

'''
The "masker cannot be None" error occurs because shap.Explainer() requires a masker for some models (especially deep learning models like neural networks). 
Here's how you can fix this and properly display SHAP force plots in Streamlit. What does a masker mean?
'''
# Check Feature Importance 
'''
shap.summary_plot(shap_values, X_test)
shap.force_plot(explainer.expected_value, shap_values[0, :], X_test.iloc[0, :])

'''

st.subheader("Understanding This Neural Network and SHAP for Traffic Flow Prediction")
st.write("Based on historical data, this neural network is trained to predict traffic flow (volume) at a given time. The SHAP (SHapley Additive exPlanations) values help explain why the neural network makes certain predictions by measuring the impact of each input feature on the model’s output.")

st.subheader("What does the Neural Network Do?")
st.write("The neural network takes input features like timestamp (e.g., time of day, day of the week) and historical traffic volume (e.g., past congestion levels). It then learns patterns from this data and predicts the future traffic volume at a given time and location.")

st.subheader("What Are SHAP Values?")
st.write("SHAP values explain how each feature contributes to a prediction. SHAP values work by comparing a model’s predictions with and without a particular feature. Then, SHAP values assign an importance value to each feature based on how much it contributes to the prediction. This means SHAP values show how each feature affects the prediction, and whether it has an overall positive or negative impact. If the SHAP value is positive, the feature increases traffic congestion, and if it’s negative, the feature reduces congestion. SHAP values are useful for understanding and analyzing how complex models like deep neural networks make predictions based on the interactions between different features. It uses Shapley values from game theory to measure the contributions of each feature to the final prediction or outcome. ")

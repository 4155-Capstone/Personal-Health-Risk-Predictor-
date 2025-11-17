import streamlit as st
from heart_disease_app import heart_ui
from diabetes_app import diabetes_ui

st.set_page_config(page_title="Multi-disease Risk Predictor", layout="centered")
st.title("🩺 Multi-disease Risk Predictor")
st.markdown("**Educational tool only — not a medical diagnosis.**")

# Select model
model_choice = st.radio(
    "Select which disease risk to check:",
    ("Heart Disease", "Diabetes"),
    horizontal=True  # This makes the radio buttons appear side by side
)


# Call the corresponding UI function
if model_choice == "Heart Disease":
    heart_ui()
else:
    diabetes_ui()

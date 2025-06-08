import streamlit as st
import pandas as pd
import numpy as np
from dataprediction import InsurancePremiumPredictor
import os
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Insurance Premium Predictor",
    page_icon="💰",
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- Load the Model ---
MODEL_PATH = 'models/best_insurance_premium_model.pkl'

if not os.path.exists(MODEL_PATH):
    st.error(f"Model file not found at: `{MODEL_PATH}`. "
             f"Please ensure `model_training.py` has been run "
             f"and the best model is saved to this path.")
    st.stop()

predictor = InsurancePremiumPredictor(model_path=MODEL_PATH)

if predictor.model is None:
    st.error("Failed to load the prediction model. Please check the logs.")
    st.stop()

# --- Streamlit UI ---
st.title("💰 Insurance Premium Predictor")
st.markdown("Enter customer details below to get an estimated insurance premium.")

# --- Input Fields ---
st.header("Customer Details")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.slider("Age", 18, 90, 30)
    gender = st.selectbox("Gender", ['Male', 'Female'])
    marital_status = st.selectbox("Marital Status", ['Single', 'Married', 'Divorced'])
    education_level = st.selectbox("Education Level", ['High School', 'Bachelor\'s', 'Master\'s', 'PhD'])


with col2:
    annual_income = st.number_input("Annual Income ($)", 10000, 500000, 50000, step=1000)
    num_dependents = st.slider("Number of Dependents", 0, 10, 0)
    occupation = st.selectbox("Occupation", ['Employed', 'Self-Employed', 'Unemployed'])
    health_score = st.slider("Health Score (1-100)", 1, 100, 70)


with col3:
    location = st.selectbox("Location Type", ['Urban', 'Suburban', 'Rural'])
    policy_type = st.selectbox("Policy Type", ['Basic', 'Comprehensive', 'Premium'])
    previous_claims = st.number_input("Previous Claims", 0, 20, 0)
    vehicle_age = st.slider("Vehicle Age (years)", 0, 30, 5)


credit_score = st.slider("Credit Score", 300, 850, 700)
insurance_duration = st.slider("Insurance Duration (years)", 1, 30, 5)
smoking_status = st.selectbox("Smoking Status", ['Yes', 'No'])
exercise_frequency = st.selectbox("Exercise Frequency", ['Daily', 'Weekly', 'Monthly', 'Occasionally', 'Rarely'])
property_type = st.selectbox("Property Type", ['House', 'Apartment', 'Condo'])


# --- Prediction Button ---
if st.button("Predict Premium"):
    # Creating a DataFrame from the input data
    input_data = pd.DataFrame([{
        'Age': age,
        'Gender': gender,
        'Annual Income': annual_income,
        'Marital Status': marital_status,
        'Number of Dependents': num_dependents,
        'Education Level': education_level,
        'Occupation': occupation,
        'Health Score': health_score,
        'Location': location,
        'Policy Type': policy_type,
        'Previous Claims': previous_claims,
        'Vehicle Age': vehicle_age,
        'Credit Score': credit_score,
        'Insurance Duration': insurance_duration,
        'Smoking Status': smoking_status,
        'Exercise Frequency': exercise_frequency,
        'Property Type': property_type
    }])

    # Make prediction
    with st.spinner("Predicting premium..."):
        predicted_premium = predictor.predict_premium(input_data)

    if predicted_premium is not None:
        st.success(f"**Predicted Insurance Premium: ${predicted_premium[0]:.2f}**")
        st.balloons()
    else:
        st.error("Prediction failed. Please check the input values and model status.")

st.markdown("---")
st.markdown("Streamlit for smart premium prediction signing off until next time :) ")
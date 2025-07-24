import streamlit as st
import numpy as np
import joblib

# Load trained model
model = joblib.load("ml_model/rent_model.pkl")

st.set_page_config(page_title="NYC Rent Estimator", layout="centered")
st.title("🏙️ NYC Rent Estimator")
st.markdown("Estimate monthly rent based on apartment features.")

# Numeric Inputs
bedrooms = st.number_input("Bedrooms", min_value=0, step=1)
bathrooms = st.number_input("Bathrooms", min_value=0.0, step=0.5)
size_sqft = st.number_input("Size (sqft)", min_value=0, step=10)
min_to_subway = st.number_input("Minutes to Subway", min_value=0, step=1)
floor = st.number_input("Floor Number", min_value=0, step=1)
building_age_yrs = st.number_input("Building Age (years)", min_value=0, step=1)

# Binary Inputs
no_fee = st.checkbox("No Broker Fee")
has_roofdeck = st.checkbox("Roof Deck")
has_washer_dryer = st.checkbox("Washer/Dryer")
has_doorman = st.checkbox("Doorman")
has_elevator = st.checkbox("Elevator")
has_dishwasher = st.checkbox("Dishwasher")
has_patio = st.checkbox("Patio")
has_gym = st.checkbox("Gym")

# Combine inputs into array
input_features = np.array([[
    bedrooms, bathrooms, size_sqft, min_to_subway, floor, building_age_yrs,
    int(no_fee), int(has_roofdeck), int(has_washer_dryer), int(has_doorman),
    int(has_elevator), int(has_dishwasher), int(has_patio), int(has_gym)
]])

if st.button("Estimate Rent"):
    prediction = model.predict(input_features)[0]
    st.success(f"💰 Estimated Monthly Rent: **${np.round(prediction, 2)}**")

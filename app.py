# app.py

import streamlit as st
import joblib
import numpy as np

# Load model and weather category map
model = joblib.load('accident_model.pkl')
weather_map = joblib.load('weather_map.pkl')
weather_inv_map = {v: k for k, v in weather_map.items()}

st.set_page_config(page_title="Traffic Accident Predictor", page_icon="🚦")

st.title("🚦 Traffic Accident Prediction App")
st.markdown("""
Predict the **likelihood of a traffic accident** based on:
- Time of day
- Weather condition
- Traffic volume
""")

# Sidebar input form
st.sidebar.header("Enter Conditions")

time = st.sidebar.slider("Time of Day (24h format)", 0, 23, 8)
weather = st.sidebar.selectbox("Weather Condition", list(weather_inv_map.keys()))
traffic_volume = st.sidebar.slider("Traffic Volume (number of cars)", 50, 600, 300)

# Main button
if st.sidebar.button("Predict Accident Risk"):
    weather_encoded = weather_inv_map[weather]
    input_data = np.array([[time, weather_encoded, traffic_volume]])

    prediction = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0][1] if hasattr(model, "predict_proba") else None

    if prediction == 1:
        st.error("⚠️ High risk of accident!")
    else:
        st.success("✅ Low risk of accident.")

    if proba is not None:
        st.info(f"Predicted accident probability: **{proba:.2%}**")

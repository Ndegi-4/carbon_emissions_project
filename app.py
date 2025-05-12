# app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# === Load model + feature scaler ===
model = joblib.load('kenya3_co2_model.pkl')
scaler = joblib.load('feature_scaler.pkl')

# --- Pretty Title + Intro ---
st.markdown("<h1 style='text-align: center; color: darkblue;'>Kenya CO₂ Emissions Predictor</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center;'>Predict CO₂ emissions based on GDP, electricity consumption, and population</h4>", unsafe_allow_html=True)

# --- Sidebar Inputs ---
st.sidebar.header('Enter Input Values')
electricity = st.sidebar.number_input('Electricity Consumption (kWh per capita)', min_value=0.0, value=100.0)
gdp = st.sidebar.number_input('GDP per Capita (USD)', min_value=0.0, value=1000.0)
population_millions = st.sidebar.number_input('Population (millions)', min_value=0.0, value=50.0)

# Convert population from millions to actual count
population = population_millions * 1_000_000

# === Standardize user inputs using SAME scaler ===
user_input_df = pd.DataFrame({
    'electric_consumption_kwh_per_capita': [electricity],
    'gdp_per_capita_ppp': [gdp],
    'population': [population]
})

user_input_scaled = scaler.transform(user_input_df)

# Sidebar Footer
st.sidebar.markdown("---")
st.sidebar.info("Developed by Victoria Ndegi | KPSK-DeKUT-2-13")

# --- Predict Button ---
if st.sidebar.button('Predict CO2 Emissions'):
    prediction_real = model.predict(user_input_scaled)[0]  # Direct real CO2 value

    st.success(f'Predicted CO₂ Emissions: {prediction_real:.2f} metric tons (approx)')

# Footer caption
st.markdown("<hr style='border:1px solid lightgray'>", unsafe_allow_html=True)
st.caption('This is a machine learning-powered tool to forecast Kenya’s carbon emissions.')
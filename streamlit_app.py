# ============================================================
# FITNESS CLUB ATTENDANCE PREDICTION APP
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(page_title="Fitness Attendance Predictor", page_icon="💪")

st.title("🏋️‍♀️ Fitness Club Attendance Prediction")
st.markdown("Predict whether a member will **attend** or **miss** their fitness class!")

# ------------------------------------------------------------
# Load trained model
# ------------------------------------------------------------
MODEL_PATH = r"C:\Users\Machine learning Projects\Fitness\best_fitness_rf_model.joblib"
if not os.path.exists(MODEL_PATH):
    st.error("❌ Model file not found! Train the model first.")
    st.stop()

model = joblib.load(MODEL_PATH)
st.success("✅ Model loaded successfully!")

# ------------------------------------------------------------
# Collect user input
# ------------------------------------------------------------
st.header("📋 Member & Class Details")

col1, col2 = st.columns(2)
with col1:
    months = st.number_input("Months as Member", 1, 60, 5)
    weight = st.number_input("Weight (kg)", 30, 150, 70)
    days_before = st.number_input("Days Before Booking", 0, 10, 1)
with col2:
    day_of_week = st.selectbox("Day of Week", ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"])
    class_time = st.selectbox("Class Time", ["Morning","Afternoon","Evening"])
    class_category = st.selectbox("Class Category", ["Cardio","Strength","Yoga"])

# ------------------------------------------------------------
# Prepare input DataFrame
# ------------------------------------------------------------
input_data = pd.DataFrame({
    "booking_id": [0],
    "months_as_member": [months],
    "weight": [weight],
    "days_before": [days_before],
    "day_of_week": [day_of_week],
    "time": [class_time],
    "category": [class_category]
})

# ------------------------------------------------------------
# Predict
# ------------------------------------------------------------
if st.button("🔍 Predict Attendance"):
    prediction = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0]

    st.subheader("🎯 Prediction Result")
    st.write(f"🟢 Probability of Attending: {proba[1]*100:.2f}%")
    st.write(f"🔴 Probability of Missing: {proba[0]*100:.2f}%")

    

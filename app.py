import streamlit as st
import pandas as pd
import joblib
import numpy as np

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")
time_encoder = joblib.load("time_encoder.pkl")

st.title("🎵 Mood-Based Music Recommendation System")

st.write("Enter physiological signals to predict mood and get music recommendations.")

# Inputs
hr = st.number_input("Heart Rate", min_value=40, max_value=150, value=70)
temp = st.number_input("Skin Temperature", min_value=30.0, max_value=40.0, value=36.5)
blink = st.number_input("Blink Rate", min_value=0.0, max_value=60.0, value=15.0)
time_day = st.selectbox("Time of Day", ["Morning","Afternoon","Evening","Night"])
score = st.number_input("Score", min_value=0.0, max_value=100.0, value=50.0)

# Encode time
time_val = time_encoder.transform([time_day])[0]

if st.button("Predict Mood"):

    X = np.array([[hr, temp, blink, time_val, score]])
    X_scaled = scaler.transform(X)

    pred = model.predict(X_scaled)[0]
    mood = label_encoder.inverse_transform([pred])[0]

    st.success(f"Predicted Mood: {mood}")

    # Simple recommendation mapping
    recommendations = {
        "Happy": "Pop / Dance",
        "Sad": "Acoustic / Soft",
        "Stressed": "Classical / Chill",
        "Relaxed": "Jazz / Ambient"
    }

    music = recommendations.get(mood, "Instrumental")

    st.subheader("🎧 Recommended Music Genre")
    st.write(music)
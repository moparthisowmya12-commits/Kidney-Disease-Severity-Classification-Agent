import streamlit as st
import joblib
import pandas as pd

model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Kidney Disease Prediction")
Age = st.number_input("Age")
SerumCreatinine = st.number_input("SerumCreatinine")
GFR = st.number_input("GFR")
ProteinInUrine = st.number_input("ProteinInUrine")
Itching = st.number_input("Itching")
BUNLevels = st.number_input("BUNLevels")

if st.button("Predict"):
    input_df = pd.DataFrame([[
        SerumCreatinine, GFR, ProteinInUrine, Itching, BUNLevels
    ]], columns=[
        'SerumCreatinine','GFR','ProteinInUrine','Itching','BUNLevels'
    ])

    scaled = scaler.transform(input_df)
    prediction = model.predict(scaled)

    if prediction[0] == 1:
        st.error("Kidney Disease Detected")
    else:
        st.success("No Kidney Disease")

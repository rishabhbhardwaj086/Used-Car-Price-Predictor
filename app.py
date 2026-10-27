import streamlit as st
import pandas as pd
import numpy as np
import joblib

model = joblib.load("model.pkl")
columns = joblib.load("columns.pkl")
models_list = joblib.load("models_list.pkl")

st.title("🚗 Used Car Price Predictor")
st.caption("Trained on CarDekho used-car listings (data up to 2020)")

col1, col2 = st.columns(2)
with col1:
    car_model = st.selectbox("Car model", models_list)
    year = st.number_input("Manufacturing year", 1995, 2020, 2015)
    km = st.number_input("Kilometres driven", 500, 300000, 50000, step=1000)
with col2:
    fuel = st.selectbox("Fuel", ["Petrol", "Diesel", "CNG", "LPG", "Electric"])
    seller = st.selectbox("Seller type", ["Individual", "Dealer", "Trustmark Dealer"])
    trans = st.selectbox("Transmission", ["Manual", "Automatic"])
    owner = st.selectbox("Owner", ["First Owner", "Second Owner", "Third Owner",
                                   "Fourth & Above Owner", "Test Drive Car"])

if st.button("Predict price"):
    row = pd.DataFrame([{
        "km_driven": km, "car_age": 2020 - year, "fuel": fuel,
        "seller_type": seller, "transmission": trans,
        "owner": owner, "model": car_model,
    }])
    row = pd.get_dummies(row)
    row = row.reindex(columns=columns, fill_value=0)
    price = np.expm1(model.predict(row)[0])
    st.success(f"Estimated price: ₹{int(price):,}")
    st.caption("This is an estimate; the actual price depends on condition and city.")

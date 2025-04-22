import streamlit as st
import joblib
import pandas as pd
from model.predictor import predict_price
from utils.visualizer import plot_price_range

# Load model
from model.price_model_dummy import DummyModel
model = DummyModel()

# Page Config
st.set_page_config(page_title="Used Electronics Price Estimator", layout="centered")

st.title("📱 Used Electronics Auction Price Estimator")

# Input form
with st.form("product_form"):
    brand = st.selectbox("Brand", ["Apple", "Samsung", "Xiaomi", "OnePlus", "Other"])
    age = st.slider("Device Age (in years)", 0, 5, 1)
    battery_health = st.slider("Battery Health (%)", 60, 100, 90)
    storage = st.selectbox("Storage Capacity", ["32GB", "64GB", "128GB", "256GB", "512GB"])
    condition = st.selectbox("Physical Condition", ["Like New", "Good", "Fair", "Poor"])
    accessories = st.multiselect("Accessories Included", ["Charger", "Earphones", "Box", "None"])
    submitted = st.form_submit_button("Predict Price")

# On submission
if submitted:
    features = {
        "brand": brand,
        "age": age,
        "battery_health": battery_health,
        "storage": storage,
        "condition": condition,
        "accessories": accessories,
    }

    # Predict price
    price, confidence = predict_price(model, features)

    st.success(f"Estimated Price: ₹{price:,.0f}")
    st.info(f"Confidence Score: {confidence*100:.1f}%")

    # Optional visualization
    plot_price_range(st, brand, storage, condition)

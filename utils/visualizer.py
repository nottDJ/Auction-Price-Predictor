import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# Simulated price ranges for example purposes
SIMULATED_PRICE_RANGES = {
    ("Apple", "128GB", "Like New"): [18000, 22000, 26000],
    ("Samsung", "128GB", "Good"): [12000, 15000, 18000],
    ("Xiaomi", "64GB", "Fair"): [5000, 7000, 9000],
    ("OnePlus", "256GB", "Like New"): [15000, 19000, 23000],
    # Default fallback
    ("Other", "64GB", "Good"): [6000, 8000, 10000],
}

def plot_price_range(st, brand, storage, condition):
    """
    Plots a bar chart for estimated low, average, and high prices of similar products.

    Args:
        st: Streamlit instance
        brand: Brand of the device
        storage: Storage capacity
        condition: Physical condition
    """
    # Get price range
    key = (brand, storage, condition)
    prices = SIMULATED_PRICE_RANGES.get(key, SIMULATED_PRICE_RANGES[("Other", "64GB", "Good")])

    labels = ["Low", "Average", "High"]
    colors = ["#FFDDC1", "#FFABAB", "#FF6363"]

    fig, ax = plt.subplots()
    ax.bar(labels, prices, color=colors)
    ax.set_title("Price Range for Similar Devices")
    ax.set_ylabel("Price (₹)")
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    st.pyplot(fig)

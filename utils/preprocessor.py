import pandas as pd

# Encoding maps
BRAND_MAP = {
    "Apple": 0, "Samsung": 1, "Xiaomi": 2, "OnePlus": 3, "Other": 4
}

STORAGE_MAP = {
    "32GB": 0, "64GB": 1, "128GB": 2, "256GB": 3, "512GB": 4
}

CONDITION_MAP = {
    "Like New": 3, "Good": 2, "Fair": 1, "Poor": 0
}

ACCESSORY_LIST = ["Charger", "Earphones", "Box"]


def transform_input(features: dict) -> pd.DataFrame:
    """
    Transforms raw form input into a format compatible with the ML model.

    Args:
        features: Raw input dict from Streamlit form.

    Returns:
        pd.DataFrame with single row of encoded features.
    """
    # Encode categorical values
    brand = BRAND_MAP.get(features["brand"], 4)
    storage = STORAGE_MAP.get(features["storage"], 1)
    condition = CONDITION_MAP.get(features["condition"], 1)

    # Encode accessories as binary flags
    accessories = features.get("accessories", [])
    accessory_flags = {f"has_{acc.lower()}": int(acc in accessories) for acc in ACCESSORY_LIST}

    # Assemble all features into a DataFrame
    data = {
        "brand": brand,
        "age": features["age"],
        "battery_health": features["battery_health"],
        "storage": storage,
        "condition": condition,
        **accessory_flags
    }

    # Ensure all accessory columns exist
    for acc in ACCESSORY_LIST:
        col_name = f"has_{acc.lower()}"
        data.setdefault(col_name, 0)

    return pd.DataFrame([data])

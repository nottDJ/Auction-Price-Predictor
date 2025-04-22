import numpy as np
from utils.preprocessor import transform_input

def predict_price(model, features: dict) -> tuple:
    """
    Predicts the price based on product features and returns the price and confidence score.

    Args:
        model: Pretrained machine learning model (loaded via joblib).
        features: Dictionary of product attributes.

    Returns:
        Tuple containing (predicted price as float, confidence score as float).
    """
    try:
        # Transform input into model-friendly format
        X = transform_input(features)
        
        # Predict using the model
        predicted_price = model.predict(X)[0]

        # Simulate confidence score (can be replaced with model's predict_proba if supported)
        confidence = np.clip(np.random.normal(loc=0.85, scale=0.05), 0.7, 0.99)

        return predicted_price, confidence

    except Exception as e:
        raise RuntimeError(f"Prediction failed: {e}")

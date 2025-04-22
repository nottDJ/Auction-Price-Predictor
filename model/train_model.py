import pandas as pd
import joblib
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from utils.preprocessor import transform_input

# Load your labeled dataset (replace with real path)
df = pd.read_csv("data/sample_data.csv")

# Manually preprocess (you can also use transform_input in bulk)
def encode_dataframe(df):
    from utils.preprocessor import BRAND_MAP, STORAGE_MAP, CONDITION_MAP, ACCESSORY_LIST

    df['brand'] = df['brand'].map(BRAND_MAP)
    df['storage'] = df['storage'].map(STORAGE_MAP)
    df['condition'] = df['condition'].map(CONDITION_MAP)
    df['battery_health'] = df['battery_health'].fillna(90)

    # Accessory one-hot
    for acc in ACCESSORY_LIST:
        df[f'has_{acc.lower()}'] = df['accessories'].apply(lambda x: int(acc in x if isinstance(x, list) else []))

    return df.drop(columns=['accessories'])

# Preprocess
df_encoded = encode_dataframe(df)

# Split features/target
X = df_encoded.drop(columns=['price'])
y = df_encoded['price']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training (you can swap XGBRegressor for RandomForestRegressor)
model = XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.1)
model.fit(X_train, y_train)

# Evaluate
preds = model.predict(X_test)
rmse = mean_squared_error(y_test, preds, squared=False)
print(f"Model trained. RMSE: {rmse:.2f}")

# Export model
joblib.dump(model, "model/price_model.pkl")
print("Model saved to model/price_model.pkl")

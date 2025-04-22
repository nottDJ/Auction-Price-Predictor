import numpy as np

class DummyModel:
    def predict(self, X):
        # Dummy logic: base + brand + storage + randomness
        base_price = 8000
        brand_adj = X['brand'].values[0] * 500
        storage_adj = X['storage'].values[0] * 1000
        condition_adj = X['condition'].values[0] * 1000
        noise = np.random.randint(-1000, 1000)
        return [base_price + brand_adj + storage_adj + condition_adj + noise]

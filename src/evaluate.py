from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
from sklearn.model_selection import cross_val_score

def evaluate_models(models, x_train, y_train):
    results = {}

    for name, model in models.items():
        scores = cross_val_score(
            model,
            x_train,
            y_train,
            cv=2,
            scoring="neg_mean_absolute_error"
        )

        mae = -scores.mean()

        results[name] = {
            "mae": round(float(mae), 4)
        }

        print(f"{name} evaluated")

    return results
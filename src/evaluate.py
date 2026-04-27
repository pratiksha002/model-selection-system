from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

def evaluate_models(models, x_test, y_test):
    results = {}

    for name, model in models.items():
        y_pred = model.predict(x_test)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        results[name] = {
            "mae": round(mae, 4),
            "rmse": round(float(rmse), 4),
            "r2": round(r2, 4)
        }

        print(f"{name} evaluated")

    return results
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

def get_models_with_params():
    models = {
        "Linear Regression": {
            "model": LinearRegression(),
            "params":{}
        },

        "Decision Tree": {
            "model": DecisionTreeRegressor(),
            "params": {
                "max_depth": [3, 5, 10],
                "min_samples_split": [2, 5]
            }
        },

        "Random Forest": {
            "model": RandomForestRegressor(),
            "params": {
                "n_estimators": [50, 100],
                "max_depth": [3, 5]
            }
        }
    }

    return models


def train_models(x_train, y_train):
    models = get_models_with_params()
    trained_models = {}

    for name, config in models.items():
        model = config["model"]
        params = config["params"]

        grid = GridSearchCV(
            model,
            params,
            cv=5,
            scoring="neg_mean_absolute_error",
            n_jobs=-1
        )

        grid.fit(x_train, y_train)
        trained_models[name] = grid.best_estimator_
        print(f"{name} trained with best params: {grid.best_params_}")

    return trained_models
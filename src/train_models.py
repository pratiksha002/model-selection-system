from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

def train_model():
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
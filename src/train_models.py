from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

def train_model():
    models = {
        "Linear Regression": LogisticRegression(),
        "Decision Tree": DecisionTreeRegressor(max_depth=5, min_samples_split=4),
        "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=5, random_state=45)
    }

    return models
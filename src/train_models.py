from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

def train_model(x_train, y_train):
    models = {
        "Linear Regression": LogisticRegression(),
        "Decision Tree": DecisionTreeRegressor(),
        "Random Forest": RandomForestRegressor()
    }


    trained_models = {}

    for name,model in models.items():
        model.fit(x_train, y_train)
        trained_models[name] = model
        print(f"{name} trained")

    return trained_models
import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split
from src.pipeline_builder import build_pipeline
from src.train_models import train_models
from src.evaluate import evaluate_models
from src.selector import select_best_model

def main():
    print("Starting Automated Model Selection System...")

    data = {
        "age": [25, 30, None, 35, 40, None],
        "city": ["Mumbai", "Delhi", "Mumbai", None, "Delhi", "Mumbai"],
        "salary": [50000, 60000, 55000, 65000, 70000, 62000]
    }

    df = pd.DataFrame(data)

    x = df[["age", "city"]]
    y = df["salary"]

    numerical_features = ["age"]
    categorical_features = ["city"]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    pipeline = build_pipeline(numerical_features, categorical_features)

    x_train_transformed = pipeline.fit_transform(x_train)
    x_test_transformed = pipeline.transform(x_test)

    models = train_models(x_train_transformed, y_train)

    results = evaluate_models(models, x_train_transformed, y_train)

    print("\nModel Performance")
    for model, metrics in results.items():
        print(model, "=", metrics)

    best_model_name, best_metrics = select_best_model(results)
    best_model = models[best_model_name]


    os.makedirs("models", exist_ok=True)
    os.makedirs("pipelines", exist_ok=True)

    with open("models/best_model.pkl", "wb") as f:
        pickle.dump(best_model, f)

    with open("pipelines/preprocess_pipeline.pkl", "wb") as f:
        pickle.dump(pipeline, f)

    print("\nBest Model:", best_model_name)
    print("Model and pipeline saved successfully!")


if __name__ == "__main__":
    main()
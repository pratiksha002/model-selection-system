from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from src.preprocess import get_preprocessing_components

def build_pipeline(numerical_features, categorical_features):
    num_imputer, scaler, cat_imputer, encoder = get_preprocessing_components()

    num_pipeline = Pipeline(steps=[
        ("imputer", num_imputer),
        ("scaler", scaler)
    ])

    cat_pipeline = Pipeline(steps=[
        ("imputer", cat_imputer),
        ("encoder", encoder)
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", num_pipeline, numerical_features),
        ("cat", cat_pipeline, categorical_features)
    ])

    return preprocessor
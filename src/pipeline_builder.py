from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

def get_preprocessing_components():
    #numerical pipeline

    num_imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    #categorical pipeline
    cat_imputer = SimpleImputer(strategy="most_frequent")
    encoder = OneHotEncoder(handle_unknown="ignore")

    return num_imputer, scaler, cat_imputer, encoder

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
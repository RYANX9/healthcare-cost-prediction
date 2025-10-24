"""
data_processing.py
------------------
Handles data loading, preprocessing, encoding, and scaling for the healthcare cost prediction model.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def load_data(path: str) -> pd.DataFrame:
    """Load the insurance dataset."""
    return pd.read_csv(path)

def preprocess_data(df: pd.DataFrame):
    """
    Preprocess the insurance dataset.
    Encodes categorical variables, scales numeric features, and splits into train/test sets.
    """
    X = df.drop("charges", axis=1)
    y = df["charges"]

    numeric_features = ["age", "bmi", "children"]
    categorical_features = ["sex", "smoker", "region"]

    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])
    categorical_transformer = Pipeline(steps=[("encoder", OneHotEncoder(drop="first"))])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    X_processed = preprocessor.fit_transform(X)
    feature_names = list(
        numeric_features + list(preprocessor.named_transformers_["cat"]["encoder"].get_feature_names_out(categorical_features))
    )

    X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, preprocessor, feature_names

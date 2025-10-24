"""
model.py
--------
Defines, trains, and evaluates the Conv1D neural network model for healthcare cost prediction.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

def build_conv1d_model(input_shape: tuple) -> keras.Model:
    """Build Conv1D neural network architecture."""
    model = keras.Sequential([
        layers.Conv1D(64, kernel_size=2, activation="relu", input_shape=input_shape),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(64, activation="relu"),
        layers.Dense(32, activation="relu"),
        layers.Dense(1, activation="linear"),
    ])

    model.compile(optimizer="adam", loss="mse", metrics=["mae", "mse"])
    return model

def train_model(model, X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
    """Train the Conv1D model."""
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )
    return history

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance and return metrics."""
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test))

    return {"R2": r2, "RMSE": rmse, "MAE": mae, "MAPE": mape}

def predict_healthcare_cost(age, bmi, children, smoker, region, sex, preprocessor, model):
    """Generate a single healthcare cost prediction."""
    import pandas as pd

    input_df = pd.DataFrame([{
        "age": age,
        "bmi": bmi,
        "children": children,
        "smoker": smoker,
        "region": region,
        "sex": sex
    }])

    X_processed = preprocessor.transform(input_df)
    X_reshaped = np.expand_dims(X_processed, axis=2)
    return float(model.predict(X_reshaped)[0][0])

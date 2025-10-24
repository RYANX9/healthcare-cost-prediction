"""
utils.py
--------
Utility functions for visualization, metrics display, and model persistence.
"""

import matplotlib.pyplot as plt
import joblib
import tensorflow as tf

def save_model(model: tf.keras.Model, path: str):
    """Save trained TensorFlow model."""
    model.save(path)

def load_model(path: str) -> tf.keras.Model:
    """Load a saved TensorFlow model."""
    return tf.keras.models.load_model(path)

def save_preprocessor(preprocessor, path: str):
    """Save preprocessing pipeline."""
    joblib.dump(preprocessor, path)

def load_preprocessor(path: str):
    """Load preprocessing pipeline."""
    return joblib.load(path)

def plot_training_history(history):
    """Plot training and validation loss curves."""
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("Model Training History")
    plt.xlabel("Epochs")
    plt.ylabel("Loss (MSE)")
    plt.legend()
    plt.show()

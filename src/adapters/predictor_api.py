# File: src/adaptors/predictor_api.py

# Predictor Implementation

# Import Libraries

# Add this BEFORE importing tensorflow
import os
import sys


# Suppress all types messages tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Add project root to sys.path to resolve module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))


import numpy as np
import tensorflow as tf

# Import Custom App Libraries

from src.ports.predictor import PredictorPort

class PredictorAPI(PredictorPort):
    """Handles model inference"""

    def load_model(self, model_path: str) -> tf.keras.Model:
        """Loads a trained model"""

        return tf.keras.models.load_model(model_path)

    def predict(self, model: tf.keras.Model, image: np.ndarray) -> np.ndarray:
        """Runs inference"""

        return model.predict(image[np.newaxis, ...])[0]

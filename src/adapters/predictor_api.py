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

    def load_model(self, model_path: str, custom_objects=None) -> tf.keras.Model:
        """Loads a trained model"""

        return tf.keras.models.load_model(model_path, custom_objects=custom_objects)

    # def predict(self, model: tf.keras.Model, image: np.ndarray) -> np.ndarray:
    #     """Runs inference"""

    #     return model.predict(image[np.newaxis, ...])[0]

    def predict(self,model, image):
        """Runs inference"""
        # image is expected to be (1, 128, 128, 1)
        prediction = model.predict(image)
        # prediction shape: (1, 128, 128, 1)
        denoised_img = prediction[0]  # (128, 128, 1)
        if denoised_img.shape[-1] == 1:
            denoised_img = denoised_img[..., 0]  # Convert (128, 128, 1) -> (128, 128)
        return denoised_img

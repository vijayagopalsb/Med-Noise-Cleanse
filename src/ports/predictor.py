# File: src/ports/predictor.py

# Model Prediction Interface

# Import Libraries

# Add this BEFORE importing tensorflow
import os
import sys


# Suppress all types messages tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Add project root to sys.path to resolve module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from abc import ABC, abstractmethod
import tensorflow as tf
import numpy as np

class PredictorPort(ABC):
    """Interface for Model Prediction."""

    @abstractmethod
    def predict(self, model: tf.keras.Model, image: np.ndarray) -> np.ndarray:
        """Run interface on the input image."""
        pass

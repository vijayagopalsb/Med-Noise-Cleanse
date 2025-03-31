# File: src/ports/trainer.py

# Model Training Interface

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

class TrainerPort(ABC):
    """Interface for Model Training."""

    @abstractmethod
    def train(self, model: tf.keras.Model, train_data, val_data) -> None:
        """Train the model using the provided data."""
        pass

    @abstractmethod
    def save_model(self, model: tf.keras.Model, save_path: str) -> None:
        """Save the trained model to disk."""
        pass


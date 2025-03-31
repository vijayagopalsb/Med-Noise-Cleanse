# File: src/adaptors/model_trainer.py

# Model Trainer Implementation

# Import Libraries

# Add this BEFORE importing tensorflow
import os
import sys


# Suppress all types messages tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Add project root to sys.path to resolve module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import tensorflow as tf

# Import Custom App Libraries

from src.ports.trainer import TrainerPort

import tensorflow as tf
from src.ports.trainer import TrainerPort

class ModelTrainer(TrainerPort):
    """Trains and saves TensorFlow model"""

    def train(self, model: tf.keras.Model, train_data, val_data) -> None:
        """Train the model"""

        model.compile(optimizer='adam', loss='mse')
        model.fit(train_data, train_data, epochs=20, validation_data=(val_data, val_data))

    def save_model(self, model: tf.keras.Model, save_path: str) -> None:
        """Save trained model"""

        model.save(save_path)


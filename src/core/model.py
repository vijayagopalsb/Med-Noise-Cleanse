# File: src/core/model.py

# Implement the Denoising Autoencoder Model

# Import Libraries

# Add this BEFORE importing tensorflow
import os
import sys
import absl.logging

# Suppress all types messages tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Suppress absl warnings
absl.logging.set_verbosity(absl.logging.ERROR)

# Add project root to sys.path to resolve module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import tensorflow as tf
from tensorflow.keras import layers, models # type: ignore
from config.logging_config import logger

# Disable GPU if not needed
tf.config.set_visible_devices([], 'GPU')
logger.info("Disableing GPU !!!")
logger.info("Tensorflow Warnings suppressed successfully!")

class DenoisingAutoencoder:

    def __init__(self, input_shape=(128, 128, 1)):
        self.input_shape = input_shape
        self.model = self._create_model()

    def _create_model(self):
        """Create the denoising autoencoder model."""
        inputs = layers.Input(shape=self.input_shape)

        # Encoder
        x = layers.Conv2D(32, (3, 3), activation="relu", padding="same") (inputs)
        x = layers.MaxPooling2D((2, 2), padding = "same")(x)
        x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
        x = layers.MaxPooling2D((2, 2), padding= "same")(x)

        # Bottleneck
        x = layers.Conv2D(128, (3, 3), activation="relu", padding = "same")(x)

        # Decoder
        x = layers.Conv2DTranspose(64, (3, 3), strides=2, activation="relu", padding="same")(x)
        x = layers.Conv2DTranspose(32, (3, 3), strides=2, activation="relu", padding="same")(x)
        outputs = layers.Conv2D(1, (3,3), activation="sigmoid", padding="same")(x)

        model = models.Model(inputs, outputs)
        model.compile(optimizer="adam", loss="mse")
        return model

    def summary(self):
        """Prints model summary."""
        self.model.summary()

    def get_model(self):
        """Return the compiled model."""
        return self.model

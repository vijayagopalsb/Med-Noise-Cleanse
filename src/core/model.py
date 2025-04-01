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
from tensorflow.keras import layers, models, optimizers # type: ignore
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

        # Encoder with skip connections
        e1 = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(inputs)
        e1 = layers.BatchNormalization()(e1)
        p1 = layers.MaxPooling2D((2, 2))(e1)

        e2 = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(p1)
        e2 = layers.BatchNormalization()(e2)
        p2 = layers.MaxPooling2D((2, 2))(e2)

        # Bottleneck and dropout
        b = layers.Conv2D(256, (3,3), activation="relu", padding="same")(p2)
        b = layers.Dropout(0.5)(b)

        # Decoder with skip connections
        d1 = layers.Conv2DTranspose(128, (3, 3), strides =2, activation="relu", padding="same")(b)
        d1 = layers.concatenate([d1, e2])
        d1 = layers.BatchNormalization()(d1)

        d2 = layers.Conv2DTranspose(64, (3, 3), strides=2, activation='relu', padding='same')(d1)
        d2 = layers.concatenate([d2, e1])
        outputs = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(d2)

        #x = layers.Conv2D(32, (3, 3), activation="relu", padding="same") (inputs)
        #x = layers.MaxPooling2D((2, 2), padding = "same")(x)
        #x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
        #x = layers.MaxPooling2D((2, 2), padding= "same")(x)

        # Bottleneck with dropout
        #x = layers.Conv2D(128, (3, 3), activation="relu", padding = "same")(x)

        # Decoder
        #x = layers.Conv2DTranspose(64, (3, 3), strides=2, activation="relu", padding="same")(x)
        #x = layers.Conv2DTranspose(32, (3, 3), strides=2, activation="relu", padding="same")(x)
        #outputs = layers.Conv2D(1, (3,3), activation="sigmoid", padding="same")(x)

        model = models.Model(inputs, outputs)
        model.compile(optimizer="adam", loss="mse")
        return model

    def summary(self):
        """Prints model summary."""
        self.model.summary()

    def get_model(self):
        """Return the compiled model."""
        return self.model

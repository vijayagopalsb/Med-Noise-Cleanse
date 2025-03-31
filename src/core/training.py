# File: src/core/training.py

# Implement training logic

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
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img # type: ignore
from tensorflow.keras.callbacks import Callback # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
import keras.saving # type: ignore

# Import Custom App Libraries
from config .settings import IMAGE_SIZE, BATCH_SIZE, EPOCHS, NOISE_FACTOR, MODEL_SAVE_PATH
from src.core.model import DenoisingAutoencoder
from config.logging_config import logger
# Disable GPU if not needed
tf.config.set_visible_devices([], 'GPU')

logger.info("Disableing GPU !!!")
logger.info("Tensorflow Warnings suppressed successfully!")


class Trainer:

    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.model = DenoisingAutoencoder(input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 1)).get_model()

    def load_images(self):
        """Loads images from dataset and applies noise."""
        images = []
        for filename in os.listdir(self.dataset_path):
            img_path =  os.path.join(self.dataset_path, filename)
            img = load_img(img_path, color_mode="grayscale", target_size=IMAGE_SIZE)
            img = img_to_array(img)
            images.append(img)

        images = np.array(images)
        noisy_images = images + NOISE_FACTOR * np.random.normal(loc=0.0, scale=1.0, size=images.shape)
        noisy_images = np.clip(noisy_images, 0.0, 1.0)  # Keep values in [0,1]
        return train_test_split(noisy_images, images, test_size=0.2, random_state=42)

    def train(self):
        """Trains the denoising autoencoder."""
        x_train_noisy, x_test_noisy, x_train, x_test = self.load_images()

        # Attach the logging callback
        log_callback = LoggingCallback(logger)

        self.model.fit(
            x_train_noisy, x_train,
            validation_data=(x_test_noisy, x_test),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=[log_callback],  # <-- Add custom callback here
            verbose=0  # Disable default progress bar

        )

        # self.model.save(MODEL_SAVE_PATH)
        keras.saving.save_model(self.model, MODEL_SAVE_PATH)
        logger.info(f"-->> Model saved to {MODEL_SAVE_PATH}")




class LoggingCallback(Callback):
    """Logs loss and validation loss after each epoch."""

    def __init__(self, logger):
        super().__init__()
        self.logger = logger

    def on_epoch_end(self, epoch, logs=None):
        if logs is not None:
            loss = logs.get('loss', 'N/A')
            val_loss = logs.get('val_loss', 'N/A')
            self.logger.info(f"Epoch {epoch + 1}/{self.params['epochs']} -> Loss: {loss:.4f}, Val_Loss: {val_loss:.4f}")



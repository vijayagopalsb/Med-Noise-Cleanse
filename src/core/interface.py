# File: src/core/interface.py

# Define interface for the Denoising 

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


import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img # type: ignore

# Import Custom App Libraries
from config.settings import IMAGE_SIZE, MODEL_SAVE_PATH
from config.logging_config import logger

# Disable GPU if not needed
tf.config.set_visible_devices([], 'GPU')

logger.info("Disableing GPU !!!")
logger.info("Warnings suppressed successfully!")

class Inference:
    def __init__(self):
        self.model = tf.keras.models.load_model(MODEL_SAVE_PATH)

    def denoise_image(self, image_path):
        """Denoises a single medical image."""
        img = load_img(image_path, color_mode='grayscale', target_size=IMAGE_SIZE)
        img = img_to_array(img) / 255.0  # Normalize
        img = np.expand_dims(img, axis=0)  # Add batch dimension

        denoised_img = self.model.predict(img)[0]
        return np.squeeze(denoised_img, axis=-1)  # Remove extra dimension

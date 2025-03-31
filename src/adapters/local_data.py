# File: src/adaptors/local_data.py

# Data Loader Implementatiomn

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
from tensorflow.keras.preprocessing.image import img_to_array, load_img # type: ignore
from src.ports.data_loader import DataLoaderPort

class  LocalDataLoader(DataLoaderPort):
    """Loads image data from local directory."""

    def __init__(self, dataset_path: str, img_size: tuple = (128, 128)):
        self.dataset_path = dataset_path
        self.img_size = img_size

    def load_data(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Loads images and splits into training & validation sets."""

        images = []

        for img_name in os.listdir(self.dataset_path):
            img_path = os.path.join(self.dataset_path, img_name)
            img = load_img(img_path, target_size=self.img_size, color_mode='grayscale')
            images.append(img_to_array(img) / 255.0)

        images = np.array(images)
        split_idx = int(len(images) * 0.8)
        return images[:split_idx], images[:split_idx], images[split_idx:], images[split_idx:]


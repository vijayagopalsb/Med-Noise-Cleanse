# File: src/core/training.py

# Implement training logic

# Import Libraries

# Add this BEFORE importing tensorflow
import os
import sys
import absl.logging
from PIL import Image

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
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import array_to_img # type: ignore

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
        logger.info("Loading image from dataset...")
        images = []
        for filename in os.listdir(self.dataset_path):
            img_path =  os.path.join(self.dataset_path, filename)
            img = load_img(img_path, color_mode="grayscale", target_size=IMAGE_SIZE)
            img = img_to_array(img)
            images.append(img)

        # images = np.array(images)
        img_array = np.array(images).astype("float32") / 255.0

        # Add Gaussian noise
        noisy_images = img_array + NOISE_FACTOR * np.random.normal(loc=0.0, scale=1.0, size=img_array.shape)
        noisy_images = np.clip(noisy_images, 0.0, 1.0)  # Keep values in [0,1]
        X_train, X_test, y_train, y_test = train_test_split(noisy_images, img_array, test_size=0.2, random_state=42)
        # Save 10 samples from the test set for review
        self.save_noisy_clean_samples(X_test, y_test, output_folder="noise_data", num_samples=10)
        return  X_train, X_test, y_train, y_test

    def save_noisy_clean_samples(self, noisy_images, clean_images, output_folder="noise_data", num_samples=10):
        """
        Saves noisy and clean image pairs side-by-side for manual inspection.

        Args:
            noisy_images (numpy.ndarray): Noisy input images.
            clean_images (numpy.ndarray): Clean ground truth images.
            output_folder (str): Directory where images will be saved.
            num_samples (int): Number of samples to save.
        """
        os.makedirs(output_folder, exist_ok=True)
        # for i in range(min(num_samples, len(noisy_images))):
        #     noisy = array_to_img(noisy_images[i])
        #     clean = array_to_img(clean_images[i])
        #     fig, axes = plt.subplots(1, 2, figsize=(4, 2))
        #     axes[0].imshow(noisy, cmap='gray')
        #     axes[0].set_title("Noisy")
        #     axes[0].axis('off')
        #     axes[1].imshow(clean, cmap='gray')
        #     axes[1].set_title("Clean")
        #     axes[1].axis('off')
        #     save_path = os.path.join(output_folder, f"sample_{i+1}.png")
        #     plt.savefig(save_path, bbox_inches='tight')
        #     plt.close(fig)

        # for i in range(min(num_samples, len(noisy_images))):
        #     noisy = array_to_img(noisy_images[i])
        #     fig, ax = plt.subplots(figsize=(2, 2))
        #     ax.imshow(noisy, cmap='gray')
        #     ax.set_title("Noisy")
        #     ax.axis('off')
        #     save_path = os.path.join(output_folder, f"noisy_sample_{i+1}.png")
        #     plt.savefig(save_path, bbox_inches='tight')
        #     plt.close(fig)

        for i in range(min(num_samples, len(noisy_images))):
            # Compute noise: (assumes both are float32 arrays in [0,1])
            # noise = noisy_images[i] - clean_images[i]  # shape: (h, w, c)
            # Rescale noise to [0, 1] for visualization (min-max normalization)
            # noise_vis = (noise - noise.min()) / (noise.max() - noise.min() + 1e-8)
            # noise_img = array_to_img(noise_vis)

            noisy = array_to_img(noisy_images[i])
            noisy_resized = noisy.resize((192, 192), Image.LANCZOS)
            save_path = os.path.join(output_folder, f"noise_only_{i+1}.png")
            noisy_resized.save(save_path)  # This will always save at 256x256
            print(f"Saved: {save_path}")

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



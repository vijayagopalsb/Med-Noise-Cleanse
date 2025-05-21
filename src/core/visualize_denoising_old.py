import os
import sys


# Ensure module import paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model  # type: ignore
from tensorflow.keras.preprocessing.image import load_img, img_to_array  # type: ignore
from PIL import Image
from config.settings import NOISE_FACTOR, TEST_CLEAN_DIR, MODEL_SAVE_PATH, IMAGE_SIZE

import random

# Add project root to sys.path to resolve module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# --- Load model (no custom_objects needed for MSE) ---
model = load_model(MODEL_SAVE_PATH)

# --- Load a random test image ---
filenames = os.listdir(TEST_CLEAN_DIR)
sample_file = random.choice(filenames)
clean_path = os.path.join(TEST_CLEAN_DIR, sample_file)

# --- Preprocess ---
clean_img = load_img(clean_path, color_mode="grayscale", target_size=IMAGE_SIZE)
clean_array = img_to_array(clean_img).astype("float32") / 255.0

# Add synthetic noise
noisy_array = clean_array + NOISE_FACTOR * np.random.normal(loc=0.0, scale=1.0, size=clean_array.shape)
noisy_array = np.clip(noisy_array, 0.0, 1.0)

# --- Predict ---
input_tensor = np.expand_dims(noisy_array, axis=0)
denoised_array = model.predict(input_tensor)[0]

# --- Optional: Inspect dynamic range ---
print(f"[INFO] Clean range: ({clean_array.min():.4f}, {clean_array.max():.4f})")
print(f"[INFO] Noisy range: ({noisy_array.min():.4f}, {noisy_array.max():.4f})")
print(f"[INFO] Denoised range: ({denoised_array.min():.4f}, {denoised_array.max():.4f})")

# --- Plot results ---
fig, axs = plt.subplots(1, 3, figsize=(18, 6))

axs[0].imshow(clean_array.squeeze(), cmap='gray')
axs[0].set_title("Clean Image")
axs[0].axis('off')

axs[1].imshow(noisy_array.squeeze(), cmap='gray')
axs[1].set_title("Noisy Image")
axs[1].axis('off')

axs[2].imshow(denoised_array.squeeze(), cmap='gray')
axs[2].set_title("Denoised Output")
axs[2].axis('off')

# Adjust subplot spacing (margins)
plt.subplots_adjust(
    left=0.05,    # margin left side
    right=0.95,   # margin right side
    top=0.9,      # margin top
    bottom=0.1,   # margin bottom
    wspace=0.1,   # horizontal spacing between subplots
    hspace=0.1    # vertical spacing (irrelevant here but good practice)
)

plt.tight_layout()
plt.savefig("vishualize.png")
print("Saved result image to: vishualize.png")

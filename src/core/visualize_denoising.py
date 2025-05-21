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
import tensorflow as tf
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
import random

# --- Load model ---
model = load_model(MODEL_SAVE_PATH)

# --- Load a random image ---
filenames = os.listdir(TEST_CLEAN_DIR)
sample_file = random.choice(filenames)
clean_path = os.path.join(TEST_CLEAN_DIR, sample_file)

# --- Preprocess ---
clean_img = load_img(clean_path, color_mode="grayscale", target_size=IMAGE_SIZE)
clean_array = img_to_array(clean_img).astype("float32") / 255.0
noisy_array = clean_array + NOISE_FACTOR * np.random.normal(loc=0.0, scale=1.0, size=clean_array.shape)
noisy_array = np.clip(noisy_array, 0.0, 1.0)

# --- Predict ---
input_tensor = np.expand_dims(noisy_array, axis=0)
denoised_array = model.predict(input_tensor)[0]

# --- Calculate Metrics ---
psnr_value = psnr(clean_array, denoised_array, data_range=1.0)
ssim_value = ssim(clean_array.squeeze(), denoised_array.squeeze(), data_range=1.0)

print(f"[Metrics] PSNR: {psnr_value:.4f} dB")
print(f"[Metrics] SSIM: {ssim_value:.4f}")

# --- Plot results ---
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

axs[0].imshow(clean_array.squeeze(), cmap='gray')
axs[0].set_title("Clean Image")
axs[0].axis('off')

axs[1].imshow(noisy_array.squeeze(), cmap='gray')
axs[1].set_title("Noisy Image")
axs[1].axis('off')

axs[2].imshow(denoised_array.squeeze(), cmap='gray')
axs[2].set_title(f"Denoised Output\nPSNR: {psnr_value:.2f} dB | SSIM: {ssim_value:.4f}")
axs[2].axis('off')

# Adjust subplot margins nicely
plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.1, wspace=0.1)

plt.tight_layout()
plt.savefig("visualize_with_metrics.png", bbox_inches='tight', pad_inches=0.2)
print("Saved result image to: visualize_with_metrics.png")

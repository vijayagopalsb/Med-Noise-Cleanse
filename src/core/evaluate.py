import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.metrics import MeanSquaredError # type: ignore
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from PIL import Image
from config.settings import MODEL_SAVE_PATH

# Paths (Update as needed)
MODEL_PATH = "models/autoencoder.h5"
TEST_CLEAN_DIR = "data/test/clean"
TEST_NOISY_DIR = "data/test/noisy"

# Load the trained model
model = load_model(MODEL_SAVE_PATH)

# Metric placeholders
psnr_scores = []
ssim_scores = []
mse_scores = []

# Helper: load and preprocess image
def load_image(path, size=(256, 256)):
    img = Image.open(path).convert("L")
    img = img.resize(size)
    img_array = np.array(img).astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=(0, -1))  # shape: (1, H, W, 1)
    return img_array

# Evaluate each image pair
for filename in os.listdir(TEST_CLEAN_DIR):
    clean_path = os.path.join(TEST_CLEAN_DIR, filename)
    noisy_path = os.path.join(TEST_NOISY_DIR, filename)

    clean_img = load_image(clean_path)
    noisy_img = load_image(noisy_path)

    denoised_img = model.predict(noisy_img)

    # Remove batch and channel dimensions
    clean_np = clean_img.squeeze()
    denoised_np = denoised_img.squeeze()

    # Metrics
    psnr = peak_signal_noise_ratio(clean_np, denoised_np, data_range=1.0)
    ssim = structural_similarity(clean_np, denoised_np, data_range=1.0)
    mse = tf.keras.losses.MeanSquaredError()(clean_img, denoised_img).numpy()

    psnr_scores.append(psnr)
    ssim_scores.append(ssim)
    mse_scores.append(mse)

# Print final results
print(f"Average PSNR: {np.mean(psnr_scores):.2f} dB")
print(f"Average SSIM: {np.mean(ssim_scores):.4f}")
print(f"Average MSE : {np.mean(mse_scores):.6f}")

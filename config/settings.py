# File: config/settings.py

# Configuration setting for project

# Input image size
IMAGE_SIZE = (192, 192)

BATCH_SIZE = 32

EPOCHS = 20

# Noise level for training
NOISE_FACTOR = 0.05

DATASET_PATH =  "/home/vijay/my_github_projects/Med-Noise-Cleanse/images/data"

DATASET_PATH_CLEAN =  "/home/vijay/my_github_projects/Med-Noise-Cleanse/images/clean"

MODEL_SAVE_PATH = "models/denoising_autoencoder.keras"

TEST_CLEAN_DIR = "/home/vijay/my_github_projects/Med-Noise-Cleanse/images/test/clean"


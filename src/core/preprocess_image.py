import cv2
import os
import sys
# Add project root to sys.path to resolve module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import numpy as np
from config.settings import DATASET_PATH, DATASET_PATH_CLEAN 


from PIL import Image

# Input and output directories
input_dir = DATASET_PATH   # Replace with your dataset folder path
output_dir = DATASET_PATH_CLEAN     # Output folder for resized images

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Loop through all jpg/JPG files
for fname in os.listdir(input_dir):
    if fname.lower().endswith('.jpg'):
        img_path = os.path.join(input_dir, fname)
        try:
            with Image.open(img_path) as img:
                # Convert to grayscale if needed: img = img.convert('L')
                img_resized = img.resize((192, 192), Image.LANCZOS)  # High quality downsampling
                # Save with same filename in output directory, always .jpg
                base = os.path.splitext(fname)[0]
                out_path = os.path.join(output_dir, f"{base}.jpg")
                img_resized.save(out_path, "JPEG", quality=95)
                print(f"Saved: {out_path}")
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

print("All images resized and saved to:", output_dir)
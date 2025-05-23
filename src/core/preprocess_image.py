import cv2
import os
import sys
# Add project root to sys.path to resolve module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import numpy as np
import shutil
from config.settings import DATASET_PATH, DATASET_PATH_CLEAN 


from PIL import Image

# Input and output directories
input_dir = DATASET_PATH   # Replace with your dataset folder path
output_dir = DATASET_PATH_CLEAN     # Output folder for resized images

# ---- Step 1: Clean the output directory ----
if os.path.exists(output_dir):
    # Remove all files and subfolders in output_dir
    for filename in os.listdir(output_dir):
        file_path = os.path.join(output_dir, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")
else:
    os.makedirs(output_dir)

print(f"Output directory '{output_dir}' cleaned.")


# ---- Step 2: Resize and save images ----
for fname in os.listdir(input_dir):
    if fname.lower().endswith('.jpg'):
        img_path = os.path.join(input_dir, fname)
        try:
            with Image.open(img_path) as img:
                # img = img.convert('L')  # Uncomment if you want grayscale
                img_resized = img.resize((192, 192), Image.LANCZOS)
                base = os.path.splitext(fname)[0]
                out_path = os.path.join(output_dir, f"{base}.jpg")
                img_resized.save(out_path, "JPEG", quality=100)  # Max quality
                print(f"Saved: {out_path}")
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

print("All images resized and saved to:", output_dir)

# # Create output directory if it doesn't exist
# os.makedirs(output_dir, exist_ok=True)

# # Loop through all jpg/JPG files
# for fname in os.listdir(input_dir):
#     if fname.lower().endswith('.jpg'):
#         img_path = os.path.join(input_dir, fname)
#         try:
#             with Image.open(img_path) as img:
#                 # Convert to grayscale if needed: img = img.convert('L')
#                 img = img.convert('L')  # Ensures grayscale
#                 img_resized = img.resize((256, 256), Image.LANCZOS)  # High quality downsampling
#                 # Save with same filename in output directory, always .jpg
#                 base = os.path.splitext(fname)[0]
#                 out_path = os.path.join(output_dir, f"{base}.jpg")
#                 img_resized.save(out_path, "JPEG", quality=100)
#                 print(f"Saved: {out_path}")
#         except Exception as e:
#             print(f"Error processing {img_path}: {e}")

# print("All images resized and saved to:", output_dir)
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Load images
img_orig = cv2.imread("images/test/compare/Y13.jpg", cv2.IMREAD_GRAYSCALE)
img_denoised = cv2.imread("images/test/compare/output.png", cv2.IMREAD_GRAYSCALE)

# Resize original to match denoised (128x128)
img_orig_resized = cv2.resize(img_orig, (img_denoised.shape[1], img_denoised.shape[0]), interpolation=cv2.INTER_AREA)

# Normalize to [0, 1]
# img_orig_resized = img_orig_resized.astype(np.float32) / 255.0
# img_denoised = img_denoised.astype(np.float32) / 255.0

# Expand dims for channels
img_orig_resized = np.expand_dims(img_orig_resized, axis=-1)  # Shape: (128,128,1)
img_denoised = np.expand_dims(img_denoised, axis=-1)          # Shape: (128,128,1)

# Compute PSNR and SSIM
psnr = tf.image.psnr(img_orig_resized, img_denoised, max_val=1.0).numpy()
ssim = tf.image.ssim(img_orig_resized, img_denoised, max_val=1.0).numpy()

print(f"PSNR: {psnr:.2f} dB")
print(f"SSIM: {ssim:.4f}")

# Optional: visual comparison
plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
plt.imshow(img_orig_resized, cmap='gray')
plt.title("Original")
plt.axis('off')
plt.subplot(1,2,2)
plt.imshow(img_denoised, cmap='gray')
plt.title("Denoised")
plt.axis('off')
plt.suptitle(f"PSNR: {psnr:.2f} dB   SSIM: {ssim:.4f}")
plt.tight_layout()
plt.show()

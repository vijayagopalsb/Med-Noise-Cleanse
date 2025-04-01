# File: app/app.py


import streamlit as st
import requests
import numpy as np
import cv2
from PIL import Image
import io

# Flask API URL (Ensure Flask backend is running before using)
API_URL = "http://127.0.0.1:5000/predict"

# Streamlit UI
st.title("MRI Image Denoising")
st.write("Upload an MRI scan to remove noise using the trained Autoencoder.")

# File uploader
uploaded_file = st.file_uploader("Choose an MRI image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Convert image to display
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded MRI Image", use_column_width=True)

    # Convert image to bytes for API
    img_bytes = io.BytesIO()
    image.save(img_bytes, format="PNG")
    img_bytes = img_bytes.getvalue()

    # Button to send request
    if st.button("Denoise Image"):
        with st.spinner("Processing..."):
            # Send request to Flask API
            response = requests.post(API_URL, files={"image": img_bytes})

            if response.status_code == 200:
                # Convert received data to image
                denoised_img_array = np.array(response.json()["prediction"]) * 255  # De-normalize
                denoised_img = Image.fromarray(denoised_img_array.astype("uint8"))

                # Display processed image
                st.image(denoised_img, caption="Denoised MRI Image", use_column_width=True)

                # Download option
                denoised_img.save("denoised_output.png")
                with open("denoised_output.png", "rb") as file:
                    st.download_button("Download Denoised Image", file, file_name="denoised_output.png")
            else:
                st.error("Error processing image. Please try again.")

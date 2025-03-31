# File: app/api.py

# Predictor Implementation

# Import Libraries

# Add this BEFORE importing tensorflow
import os
import sys
from io import BytesIO
from PIL import Image
import io
from flask import send_file


# Suppress all types messages tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Add project root to sys.path to resolve module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import os
import numpy as np
from flask import Flask, request, jsonify
from tensorflow.keras.preprocessing.image import img_to_array, load_img # type: ignore

# Import Custom App libraries

from src.adapters.local_data import LocalDataLoader
from src.adapters.model_trainer import ModelTrainer
from src.adapters.predictor_api import PredictorAPI
from src.core.model import DenoisingAutoencoder  # Assuming this is your model definition

# Initialize Flask
app = Flask(__name__)

# Define paths
DATASET_PATH = "/home/vijay/my_github_projects/Med-Noise-Cleanse/images"
MODEL_PATH = "models/denoising_autoencoder.keras"

# Initialize components
data_loader = LocalDataLoader(DATASET_PATH)
trainer = ModelTrainer()
predictor = PredictorAPI()

@app.route("/train", methods=["POST"])
def train_model():
    """Trains the model and saves it"""

    print(f"Loading data from: {data_loader.dataset_path}")  # Debugging

    X_train, _, X_val, _ = data_loader.load_data()
    autoencoder = DenoisingAutoencoder(input_shape=(128, 128, 1))  # Assuming grayscale images
    model = autoencoder.get_model()
    trainer.train(model, X_train, X_val)
    trainer.save_model(model, MODEL_PATH)
    return jsonify({"message": "Model trained and saved successfully"}), 200

@app.route("/predict", methods=["POST"])
def predict():
    """Runs inference on an uploaded image"""
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    img = load_img(BytesIO(file.read()), target_size=(128, 128), color_mode="grayscale")
    img_array = img_to_array(img) / 255.0  # Normalize

    model = predictor.load_model(MODEL_PATH)
    denoised_img = predictor.predict(model, img_array)

    pil_img = Image.fromarray(denoised_img.squeeze(), mode="L")  # Convert to PIL Image

    # Save image to a BytesIO stream
    img_io = io.BytesIO()
    pil_img.save(img_io, format="PNG")
    img_io.seek(0)

    return send_file(img_io, mimetype="image/png")
    #return jsonify({"prediction": denoised_img.tolist()}), 200  # Returning as list for JSON serialization

@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "API is running"}), 200

if __name__ == "__main__":
    app.run(debug=True)

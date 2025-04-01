# Med-Noise-Cleanse (MRI Denoising with Hexagonal Architecture)
The Medical Image Denoising using Denoising Autoencoders in TensorFlow and Hexagonal Architecture project enhances medical image quality by removing noise using a Denoising Autoencoder (DAE) built with TensorFlow. Designed using Hexagonal Architecture (Ports &amp; Adapters), it ensures modularity, scalability, and flexibility.

## Hexagonal Architecture

Hexagonal Architecture (also called Ports and Adapters) ensures that the core logic of your application remains independent of external frameworks (e.g., TensorFlow, Flask, or specific data sources).

This means:

✅ Core Logic (Domain) contains the ML model, training, and inference logic.

✅ Ports (Interfaces) define how external systems interact with the core.

✅ Adapters implement those interfaces, connecting the model to various data sources, APIs, or frameworks.

### Directory Structure of the Project

Important Folders and Files.

<pre>
Med-Noise-Cleanse/
│── src/
│   ├── core/          # Core ML Logic (Domain)
│   │   ├── model.py   # Autoencoder model
│   │   ├── training.py  # Training logic
│   │   ├── inference.py  # Inference logic
│   ├── ports/         # Interfaces (Ports)
│   │   ├── data_loader.py   # Data handling interface
│   │   ├── trainer.py       # Training interface
│   │   ├── predictor.py     # Inference interface
│   ├── adapters/       # Concrete implementations (Adapters)
│   │   ├── local_data.py    # Implements data_loader (Local Images)
│   │   ├── model_trainer.py # Implements trainer (TensorFlow)
│   │   ├── predictor_api.py # Implements predictor (Flask API)
│── app/
│   ├── api.py               # Flask/FastAPI for Model Serving
│   ├── frontend.py          # Streamlit UI for real-time image denoising
│── config/
│   ├── settings.py          # Configurations (Paths, Hyperparameters)
│   ├── logging_config.py    # Centralised Logger
│── tests/
│   ├── test_training.py     # Unit tests for training
│   ├── test_inference.py    # Unit tests for inference
│── requirements.txt         # Dependencies
│── README.md
</pre>

### How Hexagonal Architecture Helps

- **Flexibility**: You can switch between different storage systems, APIs, or models without modifying core logic.

- **Testability**: The domain logic is decoupled, making unit testing easier.

- **Maintainability**: The system remains scalable with well-defined boundaries between components.

---

## Features

✅ **Autoencoder-based Model** for noise reduction

✅ **Hexagonal Architecture** for modular design

✅ **REST API** for inference using **Flask**

✅ **Configurable Hyperparameters**

✅ **Scalable & Extendable**

---

## Installation & Setup

###  **Clone the Repository**

```bash
git clone https://github.com/vijayagopalsb/Med-Noise-Cleanse.git

cd Med-Noise-Cleanse
```

### Set Up a Virtual Environment

```python
python3 -m venv venv

source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Install Dependencies

```python
pip install -r requirements.txt
```

### Download MRI Dataset

- Download a publicly available Brain MRI Dataset (e.g., from Kaggle).

- Place images inside dataset_path/ folder.

## Usage

### Train the Model

```python
python src/adapters/model_trainer.py
```

### Run Inference via API

```python
python app/api.py
```

**Send an image for prediction using cURL

```python
curl -X POST -F "image=@/path/to/mri_image.png" http://127.0.0.1:5000/predict --output output.png
```

## API Endpoints

| Method | Endpoint  | Description            |
|--------|----------|-------------------------|
| POST   | /predict | Denoise an MRI image    |

## Testing

Run unit tests using

```python
pytest tests/
```

## Contribution

- Fork the repo

- Create a new branch

- Make improvements

- Submit a Pull Request (PR)

## License

This project is open-source and available under the MIT License.

## Contact

Email: vijayagopal.sb@gmail.com

GitHub: vijayagopalsb
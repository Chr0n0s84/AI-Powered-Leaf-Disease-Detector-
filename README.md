# AI-Powered Leaf Disease Detector

An end-to-end plant health classifier that combines a TensorFlow/Keras CNN with
a FastAPI inference service and a lightweight browser interface. Upload a leaf
image to receive a predicted disease class, confidence score, and per-class
scores.

> **Disclaimer:** This project is intended for educational and decision-support
> purposes. It is not a substitute for professional agricultural diagnosis.

## Features

- MobileNetV2-based image classifier
- Six configurable leaf-health classes
- Dataset structure and file-extension validation
- FastAPI `POST /predict` inference endpoint
- Interactive Swagger documentation at `/docs`
- Simple browser UI for local demonstrations
- Upload size and image-type validation

## Project structure

```text
.
├── app/
│   ├── config.py          # Model path, image size, and class labels
│   ├── main.py            # FastAPI application and routes
│   ├── model.py           # Model loading and image inference
│   └── static/index.html  # Browser interface
├── check_dataset.py       # Dataset validation utility
├── train_model.py         # CNN training script
├── requirements.txt       # Python dependencies
└── models/                # Local trained model (not committed by default)
```

## Requirements

- Python 3.9 or newer
- A trained Keras model, or a dataset for training
- TensorFlow-compatible hardware (CPU works, but training is faster with a GPU)

## Installation

```bash
git clone <your-repository-url>
cd <your-repository-directory>
python3 -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Dataset layout

Place images in the following folders. Folder names must match `CLASS_NAMES` in
[`app/config.py`](app/config.py).

```text
dataset/
├── train/
│   ├── healthy/
│   ├── bacterial_spot/
│   ├── early_blight/
│   ├── late_blight/
│   ├── leaf_mold/
│   └── powdery_mildew/
└── val/
    ├── healthy/
    ├── bacterial_spot/
    ├── early_blight/
    ├── late_blight/
    ├── leaf_mold/
    └── powdery_mildew/
```

Validate the folders before training:

```bash
python check_dataset.py
```

## Train a model

```bash
python train_model.py
```

The script saves the model to:

```text
models/leaf_disease_model.keras
```

Generated model files and datasets are ignored by Git because they can be
large. To distribute a model, use Git LFS or attach it to a release, then
place it at the path above after cloning.

## Run the application

```bash
uvicorn app.main:app --reload
```

Open <http://127.0.0.1:8000> for the web interface.

Useful endpoints:

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Check service status and model availability |
| `POST` | `/predict` | Classify an uploaded image |
| `GET` | `/docs` | Explore and test the API |

Example API request:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -F "file=@path/to/leaf.jpg"
```

Example response:

```json
{
  "filename": "leaf.jpg",
  "label": "late_blight",
  "confidence": 0.97,
  "scores": {
    "healthy": 0.01,
    "bacterial_spot": 0.00,
    "early_blight": 0.01,
    "late_blight": 0.97,
    "leaf_mold": 0.00,
    "powdery_mildew": 0.01
  }
}
```

## Model and preprocessing

The model expects RGB images resized to `224x224`. Preprocessing is embedded
in the training model using MobileNetV2 preprocessing. Keep the class order in
[`app/config.py`](app/config.py) synchronized with the model output order.

## License

This project is licensed under the [MIT License](LICENSE).

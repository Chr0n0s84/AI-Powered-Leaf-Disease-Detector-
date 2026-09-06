from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "leaf_disease_model.keras"
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
IMAGE_SIZE = (224, 224)

# Replace these with the labels used while training the model, in the same order.
CLASS_NAMES = [
    "healthy",
    "bacterial_spot",
    "early_blight",
    "late_blight",
    "leaf_mold",
    "powdery_mildew",
]

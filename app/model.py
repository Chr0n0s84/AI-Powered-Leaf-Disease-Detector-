import io
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

from .config import CLASS_NAMES, IMAGE_SIZE, MODEL_PATH


class ModelUnavailableError(RuntimeError):
    """Raised when the trained model has not been installed."""


class InvalidImageError(ValueError):
    """Raised when an upload cannot be decoded as an image."""


@lru_cache(maxsize=1)
def load_model(model_path: Path = MODEL_PATH):
    if not model_path.is_file():
        raise ModelUnavailableError(
            f"TensorFlow model not found at {model_path}. "
            "Place a trained .keras model at this path."
        )

    import tensorflow as tf

    return tf.keras.models.load_model(model_path)


def predict_image(image_bytes: bytes) -> dict:
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise InvalidImageError("The uploaded file is not a valid image.") from exc

    image = image.resize(IMAGE_SIZE)
    # The trained model contains MobileNetV2 preprocessing, so pass raw
    # pixel values here to keep inference consistent with training.
    input_array = np.asarray(image, dtype=np.float32)
    predictions = np.asarray(load_model().predict(input_array[None, ...], verbose=0))[0]

    if predictions.size == 0:
        raise RuntimeError("The model returned no predictions.")

    class_index = int(np.argmax(predictions))
    label = (
        CLASS_NAMES[class_index]
        if class_index < len(CLASS_NAMES)
        else f"class_{class_index}"
    )
    return {
        "label": label,
        "confidence": round(float(predictions[class_index]), 4),
        "scores": {
            (
                CLASS_NAMES[index]
                if index < len(CLASS_NAMES)
                else f"class_{index}"
            ): round(float(score), 4)
            for index, score in enumerate(predictions)
        },
    }

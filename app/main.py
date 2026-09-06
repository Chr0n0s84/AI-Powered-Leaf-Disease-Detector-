from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

from .config import MAX_UPLOAD_BYTES, MODEL_PATH
from .model import InvalidImageError, ModelUnavailableError, predict_image

app = FastAPI(
    title="Leaf Disease Detector",
    description="Diagnose plant health from a leaf image using a TensorFlow CNN.",
    version="1.0.0",
)
STATIC_DIR = Path(__file__).resolve().parent / "static"


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
async def health() -> dict:
    return {
        "status": "ok",
        "model_configured": MODEL_PATH.is_file(),
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Upload a PNG, JPEG, or other image.")

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="The uploaded image is empty.")
    if len(image_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="The image must be 10 MB or smaller.")

    try:
        result = predict_image(image_bytes)
    except InvalidImageError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ModelUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return {"filename": file.filename, **result}

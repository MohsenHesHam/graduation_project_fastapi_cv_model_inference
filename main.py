import json
import os
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from huggingface_hub import hf_hub_download

from detection import detect_defect, load_yolo_model
from detection_functions import decode_image

app = FastAPI(title="Surface Defect Detection API", version="2.0.0")

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
DEFAULT_MODEL_REPO_ID = os.getenv("HF_MODEL_REPO_ID", "").strip()
MODEL_FILENAME = os.getenv("HF_MODEL_FILENAME", "best.pt")
CLASS_NAMES_FILENAME = os.getenv("HF_CLASS_NAMES_FILENAME", "class_names.json")
LOCAL_MODEL_PATH = MODEL_DIR / MODEL_FILENAME
LOCAL_CLASS_NAMES_PATH = BASE_DIR / CLASS_NAMES_FILENAME

model = None
class_names = None
startup_error = None
resolved_model_path = None
resolved_class_names_path = None


def resolve_artifact_path(local_path: Path, repo_id: str, filename: str, repo_type: str = "model") -> Path:
    if local_path.exists():
        return local_path

    if not repo_id:
        raise FileNotFoundError(
            f"Missing required local artifact: {local_path.name}. "
            "Set HF_MODEL_REPO_ID or place the file locally."
        )

    local_path.parent.mkdir(parents=True, exist_ok=True)
    downloaded_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type=repo_type,
        local_dir=str(local_path.parent),
    )
    return Path(downloaded_path)


@app.on_event("startup")
async def load_model() -> None:
    global model, class_names, startup_error, resolved_model_path, resolved_class_names_path

    try:
        resolved_model_path = resolve_artifact_path(
            LOCAL_MODEL_PATH, DEFAULT_MODEL_REPO_ID, MODEL_FILENAME
        )
        resolved_class_names_path = resolve_artifact_path(
            LOCAL_CLASS_NAMES_PATH, DEFAULT_MODEL_REPO_ID, CLASS_NAMES_FILENAME
        )

        model = load_yolo_model(resolved_model_path)

        with open(resolved_class_names_path, "r", encoding="utf-8") as file:
            class_names = json.load(file)

        startup_error = None
        print(f"YOLO model loaded successfully from {resolved_model_path}")
    except Exception as exc:
        model = None
        class_names = None
        resolved_model_path = None
        resolved_class_names_path = None
        startup_error = str(exc)
        print(f"Error loading YOLO model or class names: {exc}")


@app.get("/")
async def root() -> dict:
    return {
        "message": "Surface Defect Detection API",
        "status": "ready" if model is not None else "degraded",
        "model_loaded": model is not None,
        "model_type": "yolo11",
        "model_repo": DEFAULT_MODEL_REPO_ID or None,
    }


@app.get("/health")
async def health() -> dict:
    return {
        "status": "ok" if model is not None else "error",
        "model_loaded": model is not None,
        "model_type": "yolo11",
        "model_path": str(resolved_model_path) if resolved_model_path else None,
        "class_names_path": str(resolved_class_names_path) if resolved_class_names_path else None,
        "details": startup_error,
    }


@app.post("/predict")
async def predict_defect(
    file: UploadFile = File(...),
    confidence_threshold: float = 0.4,
) -> JSONResponse:
    if model is None or class_names is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Detection model is unavailable. "
                f"{startup_error or 'Startup did not complete successfully.'}"
            ),
        )

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    if confidence_threshold < 0 or confidence_threshold > 1:
        raise HTTPException(status_code=400, detail="confidence_threshold must be between 0 and 1")

    try:
        image_bytes = await file.read()
        image_bgr = decode_image(image_bytes)
        result = detect_defect(
            model=model,
            image_bgr=image_bgr,
            class_names=class_names,
            confidence_threshold=confidence_threshold,
        )
        result["filename"] = file.filename
        return JSONResponse(content=result)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Internal server error: {exc}") from exc

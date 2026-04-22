import base64
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO


def load_yolo_model(model_path: str | Path) -> YOLO:
    return YOLO(str(model_path))


def decode_image(image_bytes: bytes) -> np.ndarray:
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Invalid image format")
    return image


def image_to_base64(image_bgr: np.ndarray) -> str:
    rgb_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_image)
    buffer = BytesIO()
    pil_image.save(buffer, format="JPEG", quality=90)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _bbox_area_percentage(box: list[int], image_shape: tuple[int, int, int]) -> float:
    height, width = image_shape[:2]
    total_area = max(width * height, 1)
    x1, y1, x2, y2 = box
    box_area = max(x2 - x1, 0) * max(y2 - y1, 0)
    return round((box_area / total_area) * 100, 4)


def detect_defect(
    model: YOLO,
    image_bgr: np.ndarray,
    class_names: list[str],
    confidence_threshold: float = 0.4,
) -> dict:
    try:
        results = model.predict(
            source=image_bgr,
            conf=confidence_threshold,
            verbose=False,
        )
        result = results[0]
        annotated_image = result.plot()
        detections: list[dict] = []

        if result.boxes is not None and len(result.boxes) > 0:
            for box in result.boxes:
                class_id = int(box.cls[0])
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                bbox = [x1, y1, x2, y2]
                confidence = round(float(box.conf[0]) * 100, 2)
                class_name = class_names[class_id] if class_id < len(class_names) else str(class_id)

                detections.append(
                    {
                        "class_id": class_id,
                        "class": class_name,
                        "defect_type": class_name,
                        "confidence": confidence,
                        "bbox": bbox,
                        "bbox_xywh": [x1, y1, x2 - x1, y2 - y1],
                        "area_percentage": _bbox_area_percentage(bbox, image_bgr.shape),
                    }
                )

        detections.sort(key=lambda item: item["confidence"], reverse=True)
        primary_detection = detections[0] if detections else None

        return {
            "class": primary_detection["class"] if primary_detection else None,
            "confidence": primary_detection["confidence"] if primary_detection else 0.0,
            "defect_percentage": primary_detection["area_percentage"] if primary_detection else 0.0,
            "bbox": primary_detection["bbox"] if primary_detection else None,
            "detections": detections,
            "detections_count": len(detections),
            "annotated_image": image_to_base64(annotated_image),
            "image_width": int(image_bgr.shape[1]),
            "image_height": int(image_bgr.shape[0]),
            "model_type": "yolo11",
        }
    except Exception as exc:
        raise ValueError(f"Error in defect detection: {exc}") from exc

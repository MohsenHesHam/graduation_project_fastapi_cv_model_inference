---
title: Surface Defect Detection API
emoji: 🚀
colorFrom: yellow
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# Surface Defect Detection FastAPI Service

This service exposes a YOLO11-based FastAPI API for steel surface defect detection.
It keeps the old deployment shape for Laravel integration while switching inference to Ultralytics YOLO.

## Expected artifacts

The API looks for:

- `models/best.pt`
- `class_names.json`

If `models/best.pt` is not present locally, the service can download it from a Hugging Face model repo when `HF_MODEL_REPO_ID` is set.

Default filenames:

- `HF_MODEL_FILENAME=best.pt`
- `HF_CLASS_NAMES_FILENAME=class_names.json`

## Local run

```powershell
py -3.12 -m venv .grad_project
.\.grad_project\Scripts\Activate.ps1
pip install -r requirements.txt
python -m uvicorn main:app --host 0.0.0.0 --port 8001
```

## API usage

### `POST /predict`

- Method: `POST`
- Content-Type: `multipart/form-data`
- Body field: `file`
- Optional query param: `confidence_threshold`

Example response:

```json
{
  "class": "scratches",
  "confidence": 95.2,
  "defect_percentage": 12.4,
  "bbox": [10, 20, 120, 150],
  "detections_count": 1,
  "detections": [
    {
      "class_id": 5,
      "class": "scratches",
      "defect_type": "scratches",
      "confidence": 95.2,
      "bbox": [10, 20, 120, 150],
      "bbox_xywh": [10, 20, 110, 130],
      "area_percentage": 12.4
    }
  ],
  "annotated_image": "base64..."
}
```

## Laravel integration

```php
use Illuminate\Http\Request;
use Illuminate\Support\Facades\Http;

public function detectDefect(Request $request)
{
    $image = $request->file('image');

    $response = Http::attach(
        'file',
        file_get_contents($image->getRealPath()),
        $image->getClientOriginalName()
    )->post('https://your-space-url.hf.space/predict?confidence_threshold=0.4');

    return response()->json($response->json(), $response->status());
}
```

## Notes

- `POST /predict` keeps the same multipart upload style used by the old service.
- If the model file is missing, `/health` will report the startup error and `/predict` returns `503`.

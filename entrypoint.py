import os
import sys
from pathlib import Path

MODEL_DIR = os.environ.get("MODEL_DIR", "/app/models")
PORT = os.environ.get("PORT", "8001")
MODELS = [
    "detection/detect-20241225.ckpt",
    "inpainting/lama_large_512px.ckpt",
    "ocr/ocr_ar_48px.ckpt"
]

print("[startup] Verifying models...")
for model in MODELS:
    if not (Path(MODEL_DIR) / model).exists():
        print(f"[startup] MISSING: {model}")
        sys.exit(1)

print("[startup] All models verified. Starting API...")
os.execvp("gunicorn", [
    "gunicorn",
    "--workers", "1",
    "--worker-class", "uvicorn.workers.UvicornWorker",
    "--bind", f"0.0.0.0:{PORT}",
    "--timeout", "0",
    "snipshot_engine.server:app"
])
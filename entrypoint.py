import os
import sys
from pathlib import Path
from azure.storage.blob import BlobServiceClient
from azure.identity import ManagedIdentityCredential

MODEL_DIR = os.environ.get("MODEL_DIR", "/app/models")
ACCOUNT = os.environ.get("AZURE_STORAGE_ACCOUNT", "")
CONTAINER = os.environ.get("MODEL_CONTAINER", "snipshot-models")
MODELS = ["detect-20241225.ckpt", "lama_large_512px.ckpt", "ocr_ar_48px.ckpt"]

Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)

if ACCOUNT:
    print(f"[startup] Connecting to {ACCOUNT}/{CONTAINER}")
    credential = ManagedIdentityCredential()
    account_url = f"https://{ACCOUNT}.blob.core.windows.net"
    service_client = BlobServiceClient(account_url=account_url, credential=credential)
    container_client = service_client.get_container_client(CONTAINER)

    for model in MODELS:
        dest = Path(MODEL_DIR) / model
        if dest.exists():
            print(f"[startup] Skipping {model} (already exists)")
            continue
        print(f"[startup] Downloading {model}...")
        blob_client = container_client.get_blob_client(model)
        with open(dest, "wb") as f:
            f.write(blob_client.download_blob().readall())
        size = dest.stat().st_size / 1024 / 1024
        print(f"[startup] Done: {model} ({size:.1f} MB)")
else:
    print("[startup] AZURE_STORAGE_ACCOUNT not set, skipping download")

print("[startup] Verifying models...")
for model in MODELS:
    if not (Path(MODEL_DIR) / model).exists():
        print(f"[startup] MISSING: {model}")
        sys.exit(1)

print("[startup] All models verified. Starting API...")
os.execvp("gunicorn", [
    "gunicorn",
    "--workers", "2",
    "--worker-class", "uvicorn.workers.UvicornWorker",
    "--bind", "0.0.0.0:8001",
    "snipshot_engine.server:app"
])
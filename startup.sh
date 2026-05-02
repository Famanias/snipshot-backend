$content = @'
#!/bin/bash
set -e

MODEL_DIR="${MODEL_DIR:-/app/models}"
AZURE_STORAGE_ACCOUNT="${AZURE_STORAGE_ACCOUNT:-}"
MODEL_CONTAINER="${MODEL_CONTAINER:-snipshot-models}"

echo "[startup] Model directory: $MODEL_DIR"
echo "[startup] Starting model download check..."

mkdir -p "$MODEL_DIR"

MODELS=("detect-20241225.ckpt" "lama_large_512px.ckpt" "ocr_ar_48px.ckpt")

if [ -n "$AZURE_STORAGE_ACCOUNT" ]; then
    echo "[startup] Downloading models from Azure Blob Storage..."
    echo "[startup] Account: $AZURE_STORAGE_ACCOUNT, Container: $MODEL_CONTAINER"

    for model in "${MODELS[@]}"; do
        model_path="$MODEL_DIR/$model"

        if [ -f "$model_path" ]; then
            echo "[startup] already exists, skipping: $model"
            continue
        fi

        echo "[startup] Downloading $model..."

        python3 -c "
import os, sys
from pathlib import Path
from azure.storage.blob import BlobClient
from azure.identity import ManagedIdentityCredential

blob_name = '$model'
account = '$AZURE_STORAGE_ACCOUNT'
container = '$MODEL_CONTAINER'
output_path = '$model_path'

try:
    blob_url = f'https://{account}.blob.core.windows.net/{container}/{blob_name}'
    credential = ManagedIdentityCredential()
    blob_client = BlobClient(blob_url, credential=credential)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'wb') as f:
        f.write(blob_client.download_blob().readall())
    size = os.path.getsize(output_path)
    print(f'[startup] Done: {blob_name} ({size/1024/1024:.1f} MB)')
except Exception as e:
    print(f'[startup] Failed: {e}', file=sys.stderr)
    sys.exit(1)
"
    done
else
    echo "[startup] AZURE_STORAGE_ACCOUNT not set, skipping download"
fi

echo "[startup] Verifying required models..."
for model in "${MODELS[@]}"; do
    if [ ! -f "$MODEL_DIR/$model" ]; then
        echo "[startup] Missing: $model"
        exit 1
    fi
done

echo "[startup] All models verified"
echo "[startup] Starting SnipShot Backend API on 0.0.0.0:8001..."

exec gunicorn \
    --workers 4 \
    --worker-class uvicorn.workers.UvicornWorker \
    --bind 0.0.0.0:8001 \
    --access-logfile - \
    --error-logfile - \
    snipshot_engine.server:app
'@

# Save WITHOUT BOM using UTF8NoBOM encoding
[System.IO.File]::WriteAllText(
    (Resolve-Path "startup.sh").Path,
    $content,
    [System.Text.UTF8Encoding]::new($false)
)
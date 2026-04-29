#!/bin/bash
set -e

# SnipShot Backend Container Startup Script
# Downloads models from Azure Blob Storage using Managed Identity

MODEL_DIR="${MODEL_DIR:-/app/models}"
AZURE_STORAGE_ACCOUNT="${AZURE_STORAGE_ACCOUNT:-}"
MODEL_CONTAINER="${MODEL_CONTAINER:-snipshot-models}"

echo "[startup] Model directory: $MODEL_DIR"
echo "[startup] Starting model download check..."

# Create models directory if it doesn't exist
mkdir -p "$MODEL_DIR"

# List of model files to download
declare -a MODELS=(
    "detect-20241225.ckpt"
    "lama_large_512px.ckpt"
    "ocr_ar_48px.ckpt"
)

# Function to download a single blob using Managed Identity
download_blob() {
    local blob_name="$1"
    local account="$2"
    local container="$3"
    local output_path="$4"
    
    python3 << 'PYTHON_EOF'
import os
import sys
from pathlib import Path

blob_name = sys.argv[1]
account = sys.argv[2]
container = sys.argv[3]
output_path = sys.argv[4]

try:
    from azure.storage.blob import BlobClient
    from azure.identity import ManagedIdentityCredential
except ImportError:
    print(f"[startup] ✗ Azure SDK not available, cannot download {blob_name}")
    sys.exit(1)

try:
    # Create blob URL
    blob_url = f"https://{account}.blob.core.windows.net/{container}/{blob_name}"
    
    # Use Managed Identity credential (works in Container Apps)
    credential = ManagedIdentityCredential()
    blob_client = BlobClient(blob_url, credential=credential)
    
    # Create parent directory
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Download blob
    with open(output_path, "wb") as file_stream:
        download_stream = blob_client.download_blob()
        file_stream.write(download_stream.readall())
    
    file_size = os.path.getsize(output_path)
    print(f"[startup] ✓ Downloaded {blob_name} ({file_size / 1024 / 1024:.1f} MB)")
    
except Exception as e:
    print(f"[startup] ✗ Failed to download {blob_name}: {e}", file=sys.stderr)
    sys.exit(1)
PYTHON_EOF
    
    python3 -c "
import os
import sys
from pathlib import Path

blob_name = '$blob_name'
account = '$account'
container = '$container'
output_path = '$output_path'

try:
    from azure.storage.blob import BlobClient
    from azure.identity import ManagedIdentityCredential
except ImportError:
    print(f'[startup] ✗ Azure SDK not available, cannot download {blob_name}')
    sys.exit(1)

try:
    blob_url = f'https://{account}.blob.core.windows.net/{container}/{blob_name}'
    credential = ManagedIdentityCredential()
    blob_client = BlobClient(blob_url, credential=credential)
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'wb') as f:
        download_stream = blob_client.download_blob()
        f.write(download_stream.readall())
    
    file_size = os.path.getsize(output_path)
    print(f'[startup] ✓ Downloaded {blob_name} ({file_size / 1024 / 1024:.1f} MB)')
    
except Exception as e:
    print(f'[startup] ✗ Failed to download {blob_name}: {e}', file=sys.stderr)
    sys.exit(1)
"
}

# Download models from Azure Blob Storage if configured
if [ -n "$AZURE_STORAGE_ACCOUNT" ]; then
    echo "[startup] Downloading models from Azure Blob Storage..."
    echo "[startup] Account: $AZURE_STORAGE_ACCOUNT, Container: $MODEL_CONTAINER"
    
    for model in "${MODELS[@]}"; do
        model_path="$MODEL_DIR/$model"
        
        if [ -f "$model_path" ]; then
            file_size=$(stat -f%z "$model_path" 2>/dev/null || stat -c%s "$model_path" 2>/dev/null || echo "?")
            echo "[startup] ✓ $model already exists ($(numfmt --to=iec $file_size 2>/dev/null || echo $file_size bytes))"
            continue
        fi
        
        echo "[startup] → Downloading $model..."
        download_blob "$model" "$AZURE_STORAGE_ACCOUNT" "$MODEL_CONTAINER" "$model_path"
    done
else
    echo "[startup] ⚠ AZURE_STORAGE_ACCOUNT not set, skipping model download from Azure"
    echo "[startup] Ensure models are available at $MODEL_DIR or mounted into the container"
fi

# Verify required models exist
echo "[startup] Verifying required models..."
declare -a REQUIRED=(
    "detect-20241225.ckpt"
    "lama_large_512px.ckpt"
    "ocr_ar_48px.ckpt"
)

for model in "${REQUIRED[@]}"; do
    if [ ! -f "$MODEL_DIR/$model" ]; then
        echo "[startup] ✗ Required model missing: $model"
        echo "[startup] Please ensure all models are in $MODEL_DIR or configure AZURE_STORAGE_ACCOUNT"
        exit 1
    fi
done

echo "[startup] ✓ All required models verified"
echo "[startup] Starting SnipShot Backend API on 0.0.0.0:8001..."
echo ""

# Start the FastAPI application with gunicorn + uvicorn workers
exec gunicorn \
    --workers 4 \
    --worker-class uvicorn.workers.UvicornWorker \
    --bind 0.0.0.0:8001 \
    --access-logfile - \
    --error-logfile - \
    snipshot_engine.server:app

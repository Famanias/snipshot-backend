# Backend Migration Plan: Moving off Azure to Hugging Face Spaces (Free Tier)

This document outlines the strategy for migrating the SnipShot backend off Azure to a completely free hosting stack using **Hugging Face Spaces (Docker SDK)** and **Supabase**.

---

## Background

### Current Hosting Architecture (Azure & Docker)

Before migrating, it is important to document the existing architecture:
* **Host Environment:** Azure App Service (Web App for Containers) or an Azure VM, running a custom Docker container based on the project's standard `Dockerfile`.
* **Container Startup:** The container executes `entrypoint.py` upon launching.
* **Model Storage & Provisioning:** If `AZURE_STORAGE_ACCOUNT` is set, `entrypoint.py` uses Azure Blob Storage and Managed Identity (`ManagedIdentityCredential`) to download the three ML models (`detect-20241225.ckpt`, `lama_large_512px.ckpt`, and `ocr_ar_48px.ckpt`) into `/app/models/` at startup. If they are missing and Azure credentials are not provided, startup fails.
* **Port Bindings:** The current Docker setup exposes and binds the Gunicorn/FastAPI app to port `8001`.

### Why Hugging Face Spaces?

Standard free hosting tiers (Render Free, Koyeb Free, Zeabur Free) limit container memory to **512 MB RAM**.

The SnipShot backend runs deep learning models using PyTorch on CPU:
* **Detection:** `detect-20241225.ckpt` (~308 MB)
* **Inpainting (LaMa):** `lama_large_512px.ckpt` (~204 MB)
* **OCR:** `ocr_ar_48px.ckpt` (~204 MB)

Loading PyTorch plus these three checkpoints simultaneously requires at least **1.2 GB - 1.5 GB of RAM**. Running this on Render's free tier would cause instant container crashes due to Out-Of-Memory (OOM) errors.

**Hugging Face Spaces (Docker SDK)** provides:
* **16 GB RAM** (Free)
* **2 vCPU** (Free)
* **50 GB Disk Space** (Free)
* Runs any custom `Dockerfile` on port `7860`.
* Spins down after 48 hours of inactivity, but wakes up automatically on the first incoming HTTP request.

### Infrastructure Replacements

| Expiring Azure Component | Free Alternative | Configuration in `.env` |
| :--- | :--- | :--- |
| **Azure Web App (Docker Container)** | **Hugging Face Space (Docker SDK)** | Expose API on port `7860` |
| **Azure Blob Storage (Models)** | **Baked into Docker Image** | Pre-downloaded from GitHub/Hugging Face during Docker build |
| **Azure Blob Storage (Translated Images)** | **Supabase Storage** | `SUPABASE_URL`, `SUPABASE_SERVICE_KEY`, `SUPABASE_STORAGE_BUCKET` |
| **Azure SQL Database** | **Supabase PostgreSQL** (if needed) | `DATABASE_URL` |

---

## Phase 1: Set Up Supabase

1. Create a free project on [Supabase](https://supabase.com/).
2. In your project settings, find the API keys and URL.
3. Go to **Storage** and create a public bucket named `images`.

---

## Phase 2: Set Up Hugging Face Space

1. Go to [Hugging Face Spaces](https://huggingface.co/spaces) and click **Create new Space**.
2. Name your space and select **Docker** as the SDK, then choose the **Blank** template.

---

## Phase 3: Configure Environment Secrets

In your HF Space settings under **Variables and secrets**, add:
* `SUPABASE_URL`: `<your-supabase-url>`
* `SUPABASE_JWT_SECRET`: `<your-supabase-jwt-secret>`
* `SUPABASE_SERVICE_KEY`: `<your-supabase-service-role-key>`
* `SUPABASE_STORAGE_BUCKET`: `images`
* `GROQ_API_KEY`: `<your-groq-api-key>`

---

## Phase 4: Update the Codebase

Hugging Face runs containers as a non-root user (UID `1000`) and routes incoming traffic to port `7860`. The following changes are required to make the codebase compatible with Hugging Face's Docker runtime.

### 4.1. Dockerfile Update

Pre-downloading the models during the build stage prevents slow startup times.

```dockerfile
FROM python:3.10

# Create a non-root user for Hugging Face compatibility
RUN useradd -m -u 1000 user
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libssl-dev libffi-dev libsm6 libxext6 libxrender-dev \
    libgl1 libglib2.0-0 wget \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download models during the build process
RUN mkdir -p /app/models/detection /app/models/inpainting /app/models/ocr && \
    wget -q -O /app/models/detection/detect-20241225.ckpt https://github.com/zyddnys/manga-image-translator/releases/download/beta-0.3/detect-20241225.ckpt && \
    wget -q -O /app/models/inpainting/lama_large_512px.ckpt https://huggingface.co/dreMaz/AnimeMangaInpainting/resolve/main/lama_large_512px.ckpt && \
    wget -q -O /app/models/ocr/ocr_ar_48px.ckpt https://github.com/zyddnys/manga-image-translator/releases/download/beta-0.3/ocr_ar_48px.ckpt && \
    wget -q -O /app/models/ocr/alphabet-all-v7.txt https://github.com/zyddnys/manga-image-translator/releases/download/beta-0.3/alphabet-all-v7.txt

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 MODEL_DIR=/app/models PORT=7860

COPY snipshot_engine/ ./snipshot_engine/
COPY fonts/ ./fonts/
COPY main.py .
COPY entrypoint.py .

RUN chown -R user:user /app
USER user

EXPOSE 7860

CMD ["python3", "entrypoint.py"]
```

### 4.2. entrypoint.py Update

Simplify the entrypoint by removing the strict Azure Blob Storage check and letting Gunicorn bind to the custom port.

```python
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
```

---

## Phase 5: Deploy to Hugging Face

Clone the Hugging Face Space Git repository, copy your codebase including the updated `Dockerfile` and `entrypoint.py`, commit, and push. HF will automatically build the Docker image and deploy the API.

Your public API endpoint will be:
```
https://<your-username>-<your-space-name>.hf.space
```
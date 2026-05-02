# SnipShot Backend Deployment Guide
## Azure Container Apps with Managed Identity

This guide walks you through deploying your SnipShot FastAPI backend to Azure Container Apps using Docker, with models stored in Azure Blob Storage and accessed via Managed Identity.

---

## Prerequisites

- Azure subscription with $200 student credits
- Docker installed locally
- Azure CLI installed (`az` command)
- Your repository with the generated `Dockerfile`, `startup.sh`, updated `requirements.txt`, and modified `inference.py`

---

## Step 1: Build Docker Image Locally

### 1.1 Navigate to your repository
```bash
cd d:\repos\snipshot-backend
```

### 1.2 Build the image
```bash
docker build -t snipshot-backend:latest .
```

**What this does:**
- Uses `python:3.10-slim` as base image
- Installs system dependencies
- Installs Python packages from `requirements.txt`
- Copies code (excluding models/)
- Sets up the startup script

### 1.3 Verify the build succeeded
```bash
docker images | grep snipshot-backend
```

---

## Step 2: Create Azure Container Registry (ACR)

Container Registry is where you push your Docker image. Azure Container Apps will pull from it.

### 2.1 Via Azure Portal
1. Go to [portal.azure.com](https://portal.azure.com)
2. Search for "Container Registries"
3. Click "Create"
4. Fill in:
   - **Name:** `snipshotregistry` (must be globally unique, use a unique prefix)
   - **Resource Group:** Use the same "Test" group from before
   - **Location:** East Asia (same as your blob storage region)
   - **SKU:** Basic (cheapest, fine for this use case)
5. Click "Create"

### 2.2 Wait for deployment, then get the login server
1. Go to your new Container Registry
2. Under "Settings" → "Access Keys"
3. Note down:
   - **Login server:** e.g., `snipshotregistry.azurecr.io`
   - **Username:** e.g., `snipshotregistry`
   - **Password:** (copy this, you'll need it)

### 2.3 Alternative: Via Azure CLI
```bash
az acr create \
  --resource-group Test \
  --name snipshotregistry \
  --sku Basic \
  --location eastasia
```

---

## Step 3: Push Docker Image to ACR

### 3.1 Login to your container registry
```bash
docker login snipshotregistry.azurecr.io
```

When prompted:
- **Username:** `snipshotregistry` (from Access Keys)
- **Password:** Paste the password from Access Keys

### 3.2 Tag the image
```bash
docker tag snipshot-backend:latest snipshotregistry.azurecr.io/snipshot-backend:latest
```

### 3.3 Push the image
```bash
docker push snipshotregistry.azurecr.io/snipshot-backend:latest
```

**Expected output:**
```
The push refers to repository [snipshotregistry.azurecr.io/snipshot-backend]
Pushed
```

### 3.4 Verify in Azure Portal
1. Go to your Container Registry
2. Click "Repositories"
3. You should see `snipshot-backend` with tag `latest`

---

## Step 4: Create Azure Storage Account

This is where you'll upload your model checkpoint files.

### 4.1 Via Azure Portal
1. Go to [portal.azure.com](https://portal.azure.com)
2. Search for "Storage Accounts"
3. Click "Create"
4. Fill in:
   - **Storage account name:** `snipshotmodels` (must be globally unique, use lowercase + numbers)
   - **Resource Group:** Test
   - **Location:** East Asia
   - **Performance:** Standard
   - **Redundancy:** Locally-redundant storage (LRS) — cheapest
5. Click "Create"

### 4.2 Alternative: Via Azure CLI
```bash
az storage account create \
  --name snipshotmodels \
  --resource-group Test \
  --location eastasia \
  --sku Standard_LRS
```

---

## Step 5: Create Blob Container & Upload Models

### 5.1 Via Azure Portal
1. Go to your Storage Account
2. Click "Containers" under "Data storage"
3. Click "+ Container"
4. Name: `snipshot-models`
5. Click "Create"

### 5.2 Upload your model files
1. Click on the `snipshot-models` container
2. Click "Upload"
3. Upload these 3 files from your local `models/` folder:
   - `models/detection/detect-20241225.ckpt`
   - `models/inpainting/lama_large_512px.ckpt`
   - `models/ocr/ocr_ar_48px.ckpt`

**Each upload may take several minutes.**

### 5.3 Verify uploads
After all 3 files are uploaded, you should see them listed in the `snipshot-models` container.

### 5.4 Alternative: Via Azure CLI
```bash
# Get storage account connection string
az storage account show-connection-string \
  --name snipshotmodels \
  --resource-group Test

# Create container
az storage container create \
  --account-name snipshotmodels \
  --name snipshot-models

# Upload blobs (from your local models folder)
az storage blob upload \
  --account-name snipshotmodels \
  --container-name snipshot-models \
  --name detect-20241225.ckpt \
  --file "d:\repos\snipshot-backend\models\detection\detect-20241225.ckpt"

az storage blob upload \
  --account-name snipshotmodels \
  --container-name snipshot-models \
  --name lama_large_512px.ckpt \
  --file "d:\repos\snipshot-backend\models\inpainting\lama_large_512px.ckpt"

az storage blob upload \
  --account-name snipshotmodels \
  --container-name snipshot-models \
  --name ocr_ar_48px.ckpt \
  --file "d:\repos\snipshot-backend\models\ocr\ocr_ar_48px.ckpt"
```

---

## Step 6: Create Container Apps Environment

Container Apps need an environment to run in.

### 6.1 Via Azure Portal
1. Go to [portal.azure.com](https://portal.azure.com)
2. Search for "Container Apps Environments"
3. Click "Create"
4. Fill in:
   - **Name:** `snipshot-env`
   - **Resource Group:** Test
   - **Location:** East Asia
5. Click "Create"

### 6.2 Alternative: Via Azure CLI
```bash
az containerapp env create \
  --name snipshot-env \
  --resource-group Test \
  --location eastasia
```

---

## Step 7: Create Container App

This is the actual instance that will run your FastAPI server.

### 7.1 Via Azure Portal
1. Go to [portal.azure.com](https://portal.azure.com)
2. Search for "Container Apps"
3. Click "Create"
4. **Basics tab:**
   - **Container App name:** `snipshot-backend`
   - **Resource Group:** Test
   - **Container Apps Environment:** Select `snipshot-env`
5. Click "Next: Container"
6. **Container tab:**
   - **Image source:** Azure Container Registry
   - **Registry:** `snipshotregistry.azurecr.io`
   - **Image:** `snipshot-backend`
   - **Image tag:** `latest`
   - **CPU and Memory:** 0.5 CPU, 1 Gi memory (sufficient for light load)
7. Click "Next: Environment variables"
8. **Environment variables tab:**
   - Add these three variables:
     - `AZURE_STORAGE_ACCOUNT` = `snipshotmodels`
     - `MODEL_CONTAINER` = `snipshot-models`
     - `MODEL_DIR` = `/app/models`
9. Click "Next: Ingress"
10. **Ingress tab:**
    - **Ingress:** Enabled
    - **Ingress traffic:** Accepting traffic from anywhere
    - **Transport:** HTTP
    - **Target port:** `8001`
11. Click "Review + create"
12. Click "Create"

**Wait for deployment (2-3 minutes)**

### 7.2 Alternative: Via Azure CLI
```bash
az containerapp create \
  --name snipshot-backend \
  --resource-group Test \
  --environment snipshot-env \
  --image snipshotregistry.azurecr.io/snipshot-backend:latest \
  --target-port 8001 \
  --ingress external \
  --cpu 0.5 \
  --memory 1.0Gi \
  --env-vars \
    AZURE_STORAGE_ACCOUNT=snipshotmodels \
    MODEL_CONTAINER=snipshot-models \
    MODEL_DIR=/app/models
```

---

## Step 8: Assign Managed Identity & Grant Permissions

The Container App needs to read blobs from your Storage Account without any secrets.

### 8.1 Enable Managed Identity on Container App
1. Go to your Container App in the Portal
2. Click "Identity" in the left menu
3. Under "System assigned" tab, toggle **Status** to "On"
4. Click "Save"
5. Wait for it to be enabled, then copy the **Object ID** (you'll need it)

### 8.2 Grant Storage Blob Data Reader role
1. Go to your Storage Account
2. Click "Access Control (IAM)"
3. Click "+ Add" → "Add role assignment"
4. **Role:** Search for "Storage Blob Data Reader"
5. **Assign access to:** Managed identity
6. Click "Select members"
7. Search for your Container App name (`snipshot-backend`)
8. Select it and click "Select"
9. Click "Review + assign"
10. Click "Assign"

### 8.3 Alternative: Via Azure CLI
```bash
# Get the Container App's principal ID
PRINCIPAL_ID=$(az containerapp identity show \
  --name snipshot-backend \
  --resource-group Test \
  --query principalId -o tsv)

# Get the Storage Account's resource ID
STORAGE_ID=$(az storage account show \
  --name snipshotmodels \
  --resource-group Test \
  --query id -o tsv)

# Assign the role
az role assignment create \
  --assignee $PRINCIPAL_ID \
  --role "Storage Blob Data Reader" \
  --scope $STORAGE_ID
```

---

## Step 9: Test Your Deployment

### 9.1 Get the Container App URL
1. Go to your Container App in the Portal
2. Click "Overview"
3. Copy the **Application URL** (e.g., `https://snipshot-backend.xxx.azurecontainerapps.io`)

### 9.2 Test the health endpoint
```bash
curl https://snipshot-backend.xxx.azurecontainerapps.io/health
```

**Expected response:**
```json
{"ok": true, "service": "snipshot-engine", "storage": "not configured"}
```

The `storage: "not configured"` is expected because Supabase is optional.

### 9.3 Check logs
1. Go to your Container App
2. Click "Logs" in the left menu
3. Look for:
   - `[startup] Model directory: /app/models`
   - `[startup] Downloading models from Azure Blob Storage...`
   - `[startup] ✓ Downloaded detect-20241225.ckpt`
   - `[startup] ✓ All required models verified`
   - `[startup] Starting SnipShot Backend API on 0.0.0.0:8001...`

If you see these messages, your deployment is working!

---

## Step 10: Test the Translation Endpoint

Once the models are downloaded and the API is running:

```bash
# Test with a simple translate request
curl -X POST https://snipshot-backend.xxx.azurecontainerapps.io/translate/raw \
  -F "image=@/path/to/test/image.png" \
  -F "config={}"
```

The API should return a translated image as PNG bytes.

---

## Troubleshooting

### Container not starting
**Check logs:**
1. Go to Container App → Logs
2. Look for errors like "Model not found" or "Failed to download"
3. Ensure all 3 .ckpt files are in the blob container
4. Ensure Managed Identity has `Storage Blob Data Reader` role

### Models not downloading
**Possible causes:**
- `AZURE_STORAGE_ACCOUNT` or `MODEL_CONTAINER` env var is wrong
- Managed Identity doesn't have permission on the storage account
- Blob files don't exist in the container

**Fix:**
1. Verify env vars in Container App settings
2. Verify Managed Identity has the correct role assignment
3. Verify files exist in storage container

### Slow startup
The first start may take 2-5 minutes as models are being downloaded. Subsequent restarts (if models persist) will be faster.

### High costs
If you're concerned about costs:
- Container Apps charges ~$0.000061/second (very cheap for light usage)
- With $200 credit, you can run continuously for ~100 days
- Stop the container if not in use: `az containerapp stop --name snipshot-backend --resource-group Test`
- Resume later: `az containerapp start --name snipshot-backend --resource-group Test`

---

## Next Steps

1. **Test the API** with real manga/manhwa images
2. **Configure Supabase** (optional) if you want to store results
3. **Set up CI/CD** to auto-deploy when you push to GitHub:
   - GitHub Actions can rebuild and push to ACR automatically
   - Container Apps can pull the latest image automatically

---

## References

- [Azure Container Apps docs](https://learn.microsoft.com/en-us/azure/container-apps/)
- [Azure Managed Identity](https://learn.microsoft.com/en-us/azure/active-directory/managed-identities-azure-resources/)
- [Azure Storage authentication with Managed Identity](https://learn.microsoft.com/en-us/azure/storage/blobs/storage-auth-aad)

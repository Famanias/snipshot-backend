"""
SnipShot Translator API - VM Backend (Google Cloud)

This is a STATELESS translation service:
1. Receives image + config
2. Translates using manga_translator
3. Uploads result to Supabase Storage
4. Returns Supabase Storage URL

NO user authentication here - that's handled by the Database API.
"""

import os
import pickle
import json
import io
import time
from typing import Optional

import aiohttp
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import Response, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from dotenv import load_dotenv
from supabase import create_client, Client

from manga_translator import Config

# Load environment variables
load_dotenv()

# Configure Supabase
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
SUPABASE_STORAGE_BUCKET = os.getenv("SUPABASE_STORAGE_BUCKET", "images")

supabase: Optional[Client] = None
if SUPABASE_URL and SUPABASE_SERVICE_KEY:
    supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

# Configuration
BACKEND_PORT = os.getenv("BACKEND_PORT", "8001")
BACKEND_URL = f"http://127.0.0.1:{BACKEND_PORT}/simple_execute/translate"

# Create FastAPI app
app = FastAPI(
    title="SnipShot Translator API",
    description="Stateless image translation service",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def upload_to_supabase(image: Image.Image, folder: str = "translated") -> dict:
    """Upload PIL Image to Supabase Storage and return URL + path"""
    if not supabase:
        raise Exception("Supabase not configured")
    
    # Convert PIL to bytes
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    image_bytes = buffer.getvalue()
    
    # Generate unique path
    timestamp = int(time.time() * 1000)
    storage_path = f"{folder}/{timestamp}.png"
    
    # Upload to Supabase Storage
    result = supabase.storage.from_(SUPABASE_STORAGE_BUCKET).upload(
        path=storage_path,
        file=image_bytes,
        file_options={"content-type": "image/png"}
    )
    
    # Get public URL
    public_url = supabase.storage.from_(SUPABASE_STORAGE_BUCKET).get_public_url(storage_path)
    
    return {
        "url": public_url,
        "storage_path": storage_path
    }


@app.get("/")
def root():
    """Root endpoint"""
    return {
        "service": "SnipShot Translator API",
        "version": "2.0.0",
        "storage": "Supabase",
        "endpoints": {
            "/translate": "POST - Translate image and get Supabase Storage URL",
            "/translate/raw": "POST - Translate image and get raw PNG",
            "/health": "GET - Health check"
        }
    }


@app.get("/health")
def health():
    """Health check"""
    supabase_status = "connected" if supabase else "not configured"
    return JSONResponse({"ok": True, "service": "translator-api", "storage": supabase_status})


@app.post("/translate")
async def translate(
    image: UploadFile = File(...),
    config: str = Form(...)
):
    """
    Translate image and upload to Supabase Storage.
    
    Returns:
        {
            "success": true,
            "image_url": "https://xxx.supabase.co/storage/v1/object/public/...",
            "storage_path": "translated/1234567890.png"
        }
    """
    try:
        # 1. Load image into PIL
        img_bytes = await image.read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        # 2. Parse config JSON
        try:
            config_json = json.loads(config)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON in config")
        
        config_obj = Config(**config_json)

        # 3. Serialize for manga_translator backend
        payload = pickle.dumps({
            "image": img,
            "config": config_obj
        })

        # 4. Send to internal manga_translator
        async with aiohttp.ClientSession() as session:
            async with session.post(BACKEND_URL, data=payload) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    raise HTTPException(
                        status_code=502,
                        detail=f"Translation backend error: {error_text}"
                    )
                ctx = pickle.loads(await resp.read())

        # 5. Upload to Supabase Storage
        result = upload_to_supabase(ctx.result)

        # 6. Return URL
        return {
            "success": True,
            "image_url": result["url"],
            "storage_path": result["storage_path"]
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}")


@app.post("/translate/raw")
async def translate_raw(
    image: UploadFile = File(...),
    config: str = Form(...)
):
    """
    Translate image and return raw PNG bytes.
    No Cloudinary upload - for anonymous/quick translations.
    """
    try:
        # 1. Load image
        img_bytes = await image.read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        # 2. Parse config
        try:
            config_json = json.loads(config)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON in config")
        
        config_obj = Config(**config_json)

        # 3. Serialize
        payload = pickle.dumps({
            "image": img,
            "config": config_obj
        })

        # 4. Translate
        async with aiohttp.ClientSession() as session:
            async with session.post(BACKEND_URL, data=payload) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    raise HTTPException(
                        status_code=502,
                        detail=f"Translation backend error: {error_text}"
                    )
                ctx = pickle.loads(await resp.read())

        # 5. Return raw image
        output_buffer = io.BytesIO()
        ctx.result.save(output_buffer, format="PNG")
        output_buffer.seek(0)

        return Response(
            content=output_buffer.getvalue(),
            media_type="image/png"
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}")

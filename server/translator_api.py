"""
SnipShot Translator API - VM Backend (Google Cloud)

This is a STATELESS translation service:
1. Receives image + config
2. Translates using manga_translator
3. Uploads result to Cloudinary
4. Returns Cloudinary URL

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
import cloudinary
import cloudinary.uploader

from manga_translator import Config

# Load environment variables
load_dotenv()

# Configure Cloudinary
cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET"),
    secure=True
)

# Configuration
BACKEND_PORT = os.getenv("BACKEND_PORT", "8001")
BACKEND_URL = f"http://127.0.0.1:{BACKEND_PORT}/simple_execute/translate"

# Create FastAPI app
app = FastAPI(
    title="SnipShot Translator API",
    description="Stateless image translation service",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def upload_to_cloudinary(image: Image.Image, folder: str = "snipshot/translated") -> dict:
    """Upload PIL Image to Cloudinary and return URL + public_id"""
    # Convert PIL to bytes
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    
    # Generate unique public_id
    public_id = f"{folder}/{int(time.time() * 1000)}"
    
    # Upload
    result = cloudinary.uploader.upload(
        buffer.getvalue(),
        public_id=public_id,
        resource_type="image",
        format="png",
        overwrite=True
    )
    
    return {
        "url": result["secure_url"],
        "public_id": result["public_id"]
    }


@app.get("/")
def root():
    """Root endpoint"""
    return {
        "service": "SnipShot Translator API",
        "version": "1.0.0",
        "endpoints": {
            "/translate": "POST - Translate image and get Cloudinary URL",
            "/translate/raw": "POST - Translate image and get raw PNG",
            "/health": "GET - Health check"
        }
    }


@app.get("/health")
def health():
    """Health check"""
    return JSONResponse({"ok": True, "service": "translator-api"})


@app.post("/translate")
async def translate(
    image: UploadFile = File(...),
    config: str = Form(...)
):
    """
    Translate image and upload to Cloudinary.
    
    Returns:
        {
            "success": true,
            "image_url": "https://res.cloudinary.com/...",
            "public_id": "snipshot/translated/..."
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

        # 5. Upload to Cloudinary
        result = upload_to_cloudinary(ctx.result)

        # 6. Return URL
        return {
            "success": True,
            "image_url": result["url"],
            "public_id": result["public_id"]
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

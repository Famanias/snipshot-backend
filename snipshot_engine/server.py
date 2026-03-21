"""
SnipShot Engine — Single-process FastAPI server.

Replaces the two-process pickle-over-HTTP architecture with a direct
in-process call to SnipshotTranslator.

Endpoints:
    POST /translate       → Translate and upload to Supabase Storage
    POST /translate/raw   → Translate and return raw PNG
    GET  /health          → Health check
"""

import io
import json
import os
import time
import logging

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import Response, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from dotenv import load_dotenv

from .config import Config
from .translator import SnipshotTranslator

load_dotenv()

logger = logging.getLogger("snipshot_engine.server")

# ── Supabase ─────────────────────────────────────────────────────────────

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
SUPABASE_STORAGE_BUCKET = os.getenv("SUPABASE_STORAGE_BUCKET", "images")

_supabase = None


def _get_supabase():
    global _supabase
    if _supabase is None and SUPABASE_URL and SUPABASE_SERVICE_KEY:
        from supabase import create_client
        _supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
    return _supabase


def _upload_to_supabase(image: Image.Image, folder: str = "translated") -> dict:
    client = _get_supabase()
    if not client:
        raise RuntimeError("Supabase not configured (SUPABASE_URL / SUPABASE_SERVICE_KEY)")

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    buf.seek(0)

    storage_path = f"{folder}/{int(time.time() * 1000)}.png"
    client.storage.from_(SUPABASE_STORAGE_BUCKET).upload(
        path=storage_path,
        file=buf.getvalue(),
        file_options={"content-type": "image/png"},
    )
    public_url = client.storage.from_(SUPABASE_STORAGE_BUCKET).get_public_url(storage_path)
    return {"url": public_url, "storage_path": storage_path}


# ── FastAPI app ──────────────────────────────────────────────────────────

app = FastAPI(
    title="SnipShot Translator API",
    description="Single-process manga/manhwa/manhua translation service",
    version="3.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Translator instance — created lazily on first request
_translator: SnipshotTranslator | None = None


async def _get_translator() -> SnipshotTranslator:
    global _translator
    if _translator is None:
        _translator = SnipshotTranslator(Config(), device="cpu")
        logger.info("Loading models for the first time...")
        await _translator.load_models()
        logger.info("Models loaded.")
    return _translator


# ── Endpoints ────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "service": "SnipShot Translator API",
        "version": "3.0.0",
        "endpoints": {
            "/translate": "POST - Translate and upload to Supabase",
            "/translate/raw": "POST - Translate and return raw PNG",
            "/health": "GET - Health check",
        },
    }


@app.get("/health")
def health():
    supabase_ok = "connected" if _get_supabase() else "not configured"
    return JSONResponse({"ok": True, "service": "snipshot-engine", "storage": supabase_ok})


@app.post("/translate")
async def translate(
    image: UploadFile = File(...),
    config: str = Form("{}"),
):
    """Translate image and upload result to Supabase Storage."""
    try:
        img = Image.open(io.BytesIO(await image.read())).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    try:
        cfg = Config(**json.loads(config))
    except (json.JSONDecodeError, Exception) as exc:
        raise HTTPException(status_code=400, detail=f"Invalid config JSON: {exc}")

    translator = await _get_translator()
    translator.config = cfg

    try:
        result = await translator.translate(img)
    except Exception as exc:
        logger.exception("Translation failed")
        raise HTTPException(status_code=500, detail=f"Translation failed: {exc}")

    try:
        upload = _upload_to_supabase(result)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Supabase upload failed: {exc}")

    return {"success": True, "image_url": upload["url"], "storage_path": upload["storage_path"]}


@app.post("/translate/raw")
async def translate_raw(
    image: UploadFile = File(...),
    config: str = Form("{}"),
):
    """Translate image and return raw PNG bytes."""
    try:
        img = Image.open(io.BytesIO(await image.read())).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    try:
        cfg = Config(**json.loads(config))
    except (json.JSONDecodeError, Exception) as exc:
        raise HTTPException(status_code=400, detail=f"Invalid config JSON: {exc}")

    translator = await _get_translator()
    translator.config = cfg

    try:
        result = await translator.translate(img)
    except Exception as exc:
        logger.exception("Translation failed")
        raise HTTPException(status_code=500, detail=f"Translation failed: {exc}")

    buf = io.BytesIO()
    result.save(buf, format="PNG")
    buf.seek(0)
    return Response(content=buf.getvalue(), media_type="image/png")


# uvicorn snipshot_engine.server:app --host 0.0.0.0 --port 8000  --reload
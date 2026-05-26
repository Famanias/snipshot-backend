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
import jwt
from jwt import PyJWKClient

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends, status, Request
from fastapi.responses import Response, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from PIL import Image
from dotenv import load_dotenv

from .config import Config
from .translator import SnipshotTranslator

load_dotenv()

logger = logging.getLogger("snipshot_engine.server")

# Read environment variables and fail fast
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_JWT_SECRET = os.getenv("SUPABASE_JWT_SECRET")
if not SUPABASE_JWT_SECRET:
    logger.critical("Startup aborted: SUPABASE_JWT_SECRET is missing.")
    raise RuntimeError("SUPABASE_JWT_SECRET is missing from the environment.")

# Set up JWK Client for asymmetric key verification (e.g. ES256)
jwks_client = None
if SUPABASE_URL:
    jwks_url = f"{SUPABASE_URL.rstrip('/')}/auth/v1/.well-known/jwks.json"
    jwks_client = PyJWKClient(jwks_url)

ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
TRANSLATOR_URL = os.getenv("TRANSLATOR_URL", "")

if ENVIRONMENT != "development" and not TRANSLATOR_URL.startswith("https://"):
    logger.critical(
        "Startup aborted: TRANSLATOR_URL must use HTTPS in non-development environments. "
        "Got %r. Set ENVIRONMENT=development to bypass this check locally.",
        TRANSLATOR_URL,
    )
    raise RuntimeError(
        f"TRANSLATOR_URL must start with 'https://' in environment '{ENVIRONMENT}'. "
        f"Got: {TRANSLATOR_URL!r}"
    )

# HTTPBearer automatically checks for the Authorization header
security = HTTPBearer()

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> dict:
    """Verifies the incoming Supabase JWT token."""
    token = credentials.credentials
    try:
        # Determine signature type from JWT header
        unverified_header = jwt.get_unverified_header(token)
        alg = unverified_header.get("alg")
        
        if alg in ["ES256", "RS256"] and jwks_client:
            signing_key = jwks_client.get_signing_key_from_jwt(token)
            payload = jwt.decode(
                token,
                signing_key.key,
                algorithms=[alg],
                audience="authenticated"
            )
        else:
            payload = jwt.decode(
                token,
                SUPABASE_JWT_SECRET,
                algorithms=["HS256"],
                audience="authenticated"
            )
        return payload
    except jwt.ExpiredSignatureError as e:
        logger.warning("JWT verification failed – token expired: %s", e)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token."
        )
    except jwt.PyJWTError as e:
        try:
            header = jwt.get_unverified_header(token)
            logger.warning("JWT verification failed: %s. Unverified Header: %s", e, header)
        except Exception as ex:
            logger.warning("JWT verification failed: %s. Failed to parse header: %s", e, ex)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token."
        )

# ── Supabase ─────────────────────────────────────────────────────────────

# SUPABASE_URL is already defined at startup
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

# Explicit rate-limit toggle — overrides environment default if set
isRateLimited = os.getenv("RATE_LIMIT_ENABLED", str(ENVIRONMENT != "development")).lower() == "true"

limiter = Limiter(key_func=get_remote_address, enabled=isRateLimited)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

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
@limiter.limit("60/minute")
async def translate(
    request: Request,
    image: UploadFile = File(...),
    config: str = Form("{}"),
    user: dict = Depends(get_current_user)
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
@limiter.limit("60/minute")
async def translate_raw(
    request: Request,
    image: UploadFile = File(...),
    config: str = Form("{}"),
    user: dict = Depends(get_current_user)
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
from fastapi import FastAPI, File, Form, UploadFile, Request, Depends, status, HTTPException  # type: ignore
from fastapi.middleware.cors import CORSMiddleware   # type: ignore
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials  # type: ignore
from groq import Groq                                # type: ignore
from dotenv import load_dotenv                       # type: ignore
from langdetect import detect                        # type: ignore
from PIL import Image                                # type: ignore
from slowapi import Limiter, _rate_limit_exceeded_handler  # type: ignore
from slowapi.util import get_remote_address          # type: ignore
from slowapi.errors import RateLimitExceeded         # type: ignore

import os
import base64
import io
import jwt
from jwt import PyJWKClient
import logging

app = FastAPI()
load_dotenv()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Explicit rate-limit toggle — overrides environment default if set
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
isRateLimited = os.getenv("RATE_LIMIT_ENABLED", str(ENVIRONMENT != "development")).lower() == "true"

limiter = Limiter(key_func=get_remote_address, enabled=isRateLimited)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

logger = logging.getLogger("snipshot_engine.main")

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

# HTTPBearer automatically checks for the Authorization header
security = HTTPBearer(auto_error=False)

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> dict:
    """Verifies the incoming Supabase JWT token."""
    # In development environment, bypass token verification if missing/invalid
    if ENVIRONMENT == "development":
        if not credentials:
            return {"role": "authenticated", "email": "dev@local.dev", "sub": "dev-user"}
        try:
            token = credentials.credentials
            unverified_header = jwt.get_unverified_header(token)
            alg = unverified_header.get("alg")
            if alg in ["ES256", "RS256"] and jwks_client:
                signing_key = jwks_client.get_signing_key_from_jwt(token)
                return jwt.decode(token, signing_key.key, algorithms=[alg], audience="authenticated")
            else:
                return jwt.decode(token, SUPABASE_JWT_SECRET, algorithms=["HS256"], audience="authenticated")
        except Exception:
            return {"role": "authenticated", "email": "dev@local.dev", "sub": "dev-user"}

    # In production/non-development, enforce token presence and valid signature
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated"
        )
        
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

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

SUPPORTED_MIME_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_IMAGE_BYTES = 10 * 1024 * 1024  # 10MB
MAX_DIMENSION = 2048                 # Resize if larger


def validate_and_compress_image(raw_bytes: bytes) -> tuple[bytes, str]:
    """
    Validates the image, resizes if too large,
    and returns (compressed_bytes, mime_type).
    """
    try:
        image = Image.open(io.BytesIO(raw_bytes))
    except Exception:
        raise ValueError("Uploaded file is not a valid image.")

    # Detect format
    fmt = (image.format or "PNG").upper()
    mime_map = {"JPEG": "image/jpeg", "PNG": "image/png", "WEBP": "image/webp"}
    mime_type = mime_map.get(fmt, "image/png")

    # Convert RGBA/P to RGB for JPEG compatibility
    if image.mode in ("RGBA", "P"):
        image = image.convert("RGB")
        mime_type = "image/jpeg"
        fmt = "JPEG"

    # Resize if exceeds max dimension
    if max(image.size) > MAX_DIMENSION:
        image.thumbnail(
            (MAX_DIMENSION, MAX_DIMENSION),
            Image.LANCZOS
        )

    # Re-encode
    output = io.BytesIO()
    save_fmt = fmt if fmt in ("JPEG", "PNG", "WEBP") else "PNG"
    image.save(output, format=save_fmt, quality=85)
    output.seek(0)

    return output.read(), mime_type


@app.post("/translate-image")
@limiter.limit("30/minute")
async def translate_image(
    request: Request,
    image: UploadFile = File(...),
    target_lang: str = Form(default="en"),
    user: dict = Depends(get_current_user),
):
    try:
        # --- Validate file type ---
        content_type = image.content_type or ""
        if content_type not in SUPPORTED_MIME_TYPES:
            return {
                "error": (
                    f"Unsupported file type: {content_type}. "
                    f"Accepted: JPEG, PNG, WEBP."
                )
            }

        # --- Read and size-check ---
        raw_bytes = await image.read()

        if len(raw_bytes) == 0:
            return {"error": "Empty image file received."}

        if len(raw_bytes) > MAX_IMAGE_BYTES:
            return {
                "error": (
                    f"Image too large "
                    f"({len(raw_bytes) // (1024*1024)}MB). "
                    f"Maximum allowed: 10MB."
                )
            }

        # --- Validate, resize, compress ---
        try:
            processed_bytes, mime_type = validate_and_compress_image(raw_bytes)
        except ValueError as e:
            return {"error": str(e)}

        # --- Encode to base64 for Groq ---
        clean_base64 = base64.b64encode(processed_bytes).decode("utf-8")

        # --- Language mapping ---
        lang_mapping = {
            "en":    "English",
            "ja":    "Japanese",
            "ko":    "Korean",
            "zh_cn": "Simplified Chinese",
            "zh_tw": "Traditional Chinese",
        }
        target_language = lang_mapping.get(target_lang, "English")

        # --- Groq API call ---
        chat_completion = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            temperature=0.1,
            top_p=0.3,
            max_completion_tokens=1024,
            messages=[
                {
                    "role": "system",
                    "content": f"""
You are a precise multilingual OCR and translation assistant specialized in visual text extraction across East Asian languages.

## ROLE
Detect, extract, reconstruct, and translate all readable text from images with high fidelity, applying language-specific reading and translation rules.

---

## STEP 1 — IMAGE TYPE DETECTION
Identify the image type before doing anything else:
- manga (Japanese comic)
- manhwa (Korean comic)
- manhua (Chinese comic)
- street sign / poster / advertisement
- product label
- UI screenshot / app interface
- menu
- meme
- chat log / message thread
- document / article
- other

This determines which extraction and reading rules apply in Step 2.

---

## STEP 2 — TEXT EXTRACTION

### General Rules (ALL image types):
- Extract ALL visible text. Do NOT skip any text, however small or peripheral.
- Preserve the original script and language exactly as written.
- If multiple languages appear, label each segment: [JA], [KO], [ZH-CN], [ZH-TW], [EN], etc.
- If no readable text exists, return exactly: NO_TEXT_FOUND

---

### JAPANESE TEXT EXTRACTION

**Script identification:**
- Japanese uses three scripts mixed together: Kanji (漢字), Hiragana (ひらがな), Katakana (カタカナ).
- Do NOT confuse Kanji with Chinese — context and kana will confirm it is Japanese.
- Furigana (small kana printed above/beside kanji) should be extracted and noted in parentheses.
  Example: 漢字(かんじ)

**Text direction:**
- Manga and traditional documents: VERTICAL (tategumi) — top-to-bottom, right column before left.
- Modern UI, signs, some manga captions: HORIZONTAL (yokogumi) — left-to-right.
- Identify direction per bubble/block before extracting.

**CRITICAL — Vertical column reading (manga bubbles):**
- Each bubble contains text in vertical columns. A column = one top-to-bottom strand of characters.
- Read the RIGHTMOST column top-to-bottom FIRST, then move LEFT to the next column.
- Do NOT scan horizontally across rows — this produces garbled output.
- Do NOT split characters that belong to the same column.

  Correct example:
    Columns (right → left):  Col1     Col2     Col3
                              医       言       ２
                              学       え       １
                              的       ば       番
                              に                染
    Read as: 医学的に → 言えば → ２１番染  ✓
    NOT:     医言２学え１的ばに染          ✗

**Self-validation after each bubble:**
- Ask: "Does this form a grammatically valid Japanese sentence?"
- If scrambled → you read horizontally by mistake. Re-read the bubble correctly.

**Manga-specific elements:**
- [SPEECH]: spoken dialogue in rounded bubbles.
- [THOUGHT]: inner monologue in cloud-shaped bubbles.
- [NARRATION]: rectangular caption boxes, usually at panel edges.
- [SFX]: sound effects — often large, stylized, scattered. NEVER skip these.
  Format: [SFX]: <original> → <translated meaning>
  Example: [SFX]: ドキドキ → *thump thump / heart pounding*

**Panel reading order:**
- Multi-column page: TOP-RIGHT panel first → TOP-LEFT → next row right → next row left.
- Single-column vertical strip: top panel first → downward.

---

### KOREAN TEXT EXTRACTION

**Script identification:**
- Korean uses Hangul (한글) syllable blocks. Each block = one syllable unit.
- Do NOT confuse Hangul with other scripts.
- Hanja (Chinese characters) may appear occasionally — extract as-is and note [HANJA].

**Text direction:**
- Manhwa and modern text: HORIZONTAL, left-to-right.
- Traditional or stylized documents: may be VERTICAL — apply same column rules as Japanese if so.

**Manhwa-specific elements:**
- [SPEECH], [THOUGHT], [NARRATION], [SFX] — same labeling conventions.
- [SFX] example: [SFX]: 쿵 → *thud / heavy thump*

**Panel reading order:**
- Manhwa: LEFT-TO-RIGHT — TOP-LEFT → TOP-RIGHT → next row left → next row right.

**Line break handling:**
- Hangul syllables may wrap mid-word — reconstruct into complete words before outputting.

---

### CHINESE TEXT EXTRACTION

**Script identification:**
- Simplified Chinese (简体字): Mainland China. Fewer strokes.
- Traditional Chinese (繁體字): Taiwan, Hong Kong, Macau. More complex strokes.
- Label accordingly: [ZH-CN] or [ZH-TW]. Do NOT mix the two.

**Text direction:**
- Simplified Chinese: mostly HORIZONTAL, left-to-right.
- Traditional Chinese: may be VERTICAL, right-to-left columns.
- Apply same vertical column rules as Japanese if vertical layout detected.

**Manhua-specific elements:**
- [SPEECH], [THOUGHT], [NARRATION], [SFX] — same conventions.
- [SFX] example: [SFX]: 碰 → *bang / collision impact*

**Panel reading order:**
- Simplified manhua: LEFT-TO-RIGHT.
- Traditional manhua: RIGHT-TO-LEFT.

---

## STEP 3 — TRANSLATION

### CRITICAL — Reconstruct before translating (ALL languages):
- Line breaks inside a bubble are layout artifacts — NOT sentence boundaries.
- Join all lines in a bubble into one continuous sentence before translating.
- Translate the full reconstructed sentence as a single unit.
- Do NOT translate line by line or fragment by fragment.

**Japanese reconstruction example:**
  Extracted:   医学的に言えば / ２１番染色体が / ３本ある異常 / のことです
  Reconstruct: "医学的に言えば２１番染色体が３本ある異常のことです"
  Translate:   "Medically speaking, it refers to an abnormality where chromosome 21 has 3 copies."

**Korean reconstruction example:**
  Extracted:   성낙숙 쪽에서 / 했다기엔 군이... / 이 이름으로 할까?
  Reconstruct: "성낙숙 쪽에서 했다기엔 군이... 이 이름으로 할까?"
  Translate:   "For it to have been done on Seongnaksuk's side... shall we go with this name?"

**Chinese reconstruction example:**
  Extracted:   你说的那个人 / 真的存在吗？
  Reconstruct: "你说的那个人真的存在吗？"
  Translate:   "Does the person you mentioned really exist?"

---

### Language-Specific Translation Rules

**Japanese:**
- Subjects are frequently omitted — infer from context.
- Preserve sentence-final particle nuance:
    か → question ("...right?"), ね → agreement ("...you know"),
    よ → assertive, な → reflection, わ → soft feminine,
    ぞ／ぜ → strong masculine, って → casual quoting
- Polite (です／ます) vs casual (だ／る) — preserve tone in translation.
- Ellipses (……) = trailing speech — keep in translation.
- Furigana informs translation when kanji reading is ambiguous.

**Korean:**
- Preserve speech level:
    합쇼체／해요체 → formal/polite | 해체／반말 → casual
- Sentence-final endings:
    ~요／~습니다 → polite | ~야／~아／~다 → casual
    ~네 → mild surprise | ~겠 → intention/conjecture
- Honorifics (씨, 님, 형, 오빠, 언니, 누나) — keep original + note role.
- Ellipses and dashes signal trailing speech — preserve.

**Chinese:**
- Simplified vs Traditional vocabulary must not be mixed in output.
- Measure words — translate naturally, not literally.
- Classical patterns (也、乃、曰、之) → use formal register.
- Preserve modal particles: 吧 (suggestion), 啊 (exclamation), 嘛 (obviousness), 呢 (continuation).

### General Translation Rules (ALL languages):
- Translate into {target_language}.
- Preserve original meaning, tone, nuance — do NOT sanitize or soften.
- Maintain all labels: [SPEECH], [THOUGHT], [NARRATION], [SFX], panel markers.
- Proper nouns: keep original + add romanization if helpful.
- Already in {target_language}: keep as-is, note [already in {target_language}].

---

## STEP 4 — VISUAL CONTEXT SUMMARY
- Describe only what is clearly and directly visible.
- State detected image type and language(s).
- For comics: describe panel layout, visible characters, scene — visual evidence only.
- Do NOT infer emotions, plot, or off-screen events.
- 2–4 sentences max.

---

## HARD RULES
- Do NOT hallucinate or assume text not visible in the image.
- Do NOT skip any visible text.
- Do NOT alter profanity or sensitive language — translate faithfully.
- Do NOT add commentary or disclaimers.
- Respond ONLY in the exact format below.

---

## OUTPUT FORMAT (strict — no deviations)

EXTRACTED_TEXT:
<original extracted text with labels, or NO_TEXT_FOUND>

TRANSLATION:
<translated text in {target_language} preserving all labels, or N/A if NO_TEXT_FOUND>

SUMMARY:
<2–4 sentence factual description including image type and language>
"""
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{clean_base64}"
                            },
                        }
                    ],
                }
            ],
        )

        response_text = (
            chat_completion
            .choices[0]
            .message
            .content
            .strip()
        )

        extracted_text = ""
        translated_text = ""
        summary = ""

        try:
            if "TRANSLATION:" in response_text:
                extracted_part = response_text.split("TRANSLATION:")[0]
                extracted_text = (
                    extracted_part
                    .replace("EXTRACTED_TEXT:", "")
                    .strip()
                )

                remaining = response_text.split("TRANSLATION:")[1]

                if "SUMMARY:" in remaining:
                    translation_part, summary_part = remaining.split(
                        "SUMMARY:", 1
                    )
                    translated_text = translation_part.strip()
                    summary = summary_part.strip()
                else:
                    translated_text = remaining.strip()

        except Exception:
            translated_text = response_text

        # Safe language detection
        detected_language = "unknown"

        try:
            if (
                extracted_text
                and extracted_text != "NO_TEXT_FOUND"
                and len(extracted_text.strip()) > 3
            ):
                detected_language = detect(extracted_text)
        except Exception:
            detected_language = "unknown"

        print("===== IMAGE TRANSLATION =====")
        print(f"Detected language: {detected_language}")
        print(f"Extracted text:    {extracted_text}")
        print(f"Translated text:   {translated_text}")
        print(f"Summary:           {summary}")

        return {
            "detected_language": detected_language,
            "extracted_text":    extracted_text,
            "translated_text":   translated_text,
            "summary":           summary,
            "raw_response":      response_text,
        }

    except Exception as e:
        print(f"Translation error: {str(e)}")
        return {"error": f"Translation failed: {str(e)}"}


if __name__ == "__main__":
    import uvicorn  # type: ignore
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
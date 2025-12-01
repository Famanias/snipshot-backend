from fastapi import FastAPI, Form, HTTPException  # type: ignore
from fastapi.middleware.cors import CORSMiddleware  # type: ignore
import cv2  # type: ignore
import numpy as np  # type: ignore
from groq import Groq  # type: ignore
import json
import os
import base64
from dotenv import load_dotenv  # type: ignore
from langdetect import detect, DetectorFactory  # type: ignore 
from pydantic import BaseModel  # type: ignore

import pytesseract  # type: ignore
from pytesseract import Output  # type: ignore

# Ensure consistent language detection
DetectorFactory.seed = 0

app = FastAPI()
load_dotenv()

# Allow CORS for Flutter
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Groq client
client = Groq(api_key=os.environ.get("GROQ_MAVERICK"))

# Language mapping for supported languages
lang_mapping = {
    "en": "English",
    "ja": "Japanese",
    "ko": "Korean",
    "zh_cn": "Simplified Chinese",
    "zh_tw": "Traditional Chinese",
}

# Map detected language to supported languages
def map_detected_language(detected_lang: str) -> str:
    if detected_lang in lang_mapping:
        return detected_lang
    if detected_lang.startswith("zh"):
        return "zh_tw"
    return "unknown"

# --- Image preprocessing ---
def preprocess_image_bytes(image_bytes: bytes, mode="grayscale") -> np.ndarray:
    """
    Preprocess image for OCR.

    mode: "grayscale" | "binary"
    """
    image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Image decoding failed")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if mode == "binary":
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        contrast = clahe.apply(gray)
        _, processed = cv2.threshold(
            contrast, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
    else:
        processed = gray

    return processed


@app.post("/preprocess")
async def preprocess_image(image_bytes: bytes = Form(...)):
    denoised = preprocess_image_bytes(image_bytes)
    _, encoded_image = cv2.imencode(".png", denoised)
    base64_image = base64.b64encode(encoded_image.tobytes()).decode("utf-8")
    return {"image": base64_image}


# --- OCR ---
class OCRRequest(BaseModel):
    image_base64: str


@app.post("/ocr")
async def extract_text(request: OCRRequest):
    """OCR using Groq only, asking for structured JSON with normalized boxes.
    Returns items with bbox = [x1,y1,x2,y2] in normalized 0.1 coordinates.
    Falls back to simple line-split if parsing fails.
    """
    try:
        image_bytes = base64.b64decode(request.image_base64)

        # Use the original image to preserve dimensions, but keep a debug preprocessed copy
        image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Image decoding failed")
        H, W = image.shape[:2]

        # Build base64 data URI for Groq
        _, enc = cv2.imencode(".png", image)
        base64_image = base64.b64encode(enc.tobytes()).decode("utf-8")

        system_prompt = (
            "You are an automated OCR extraction system. "
            "Your only allowed output is strict, valid JSON matching the exact schema requested. "
            "You are forbidden from adding explanations, apologies, markdown, code fences, comments, "
            "or any text outside the JSON object. "
            "If you cannot complete the task perfectly, you must still return valid JSON (even if the items array is empty). "
            "Non-compliance will be treated as a critical error."
        )
        user_prompt = """
            You are a cold, emotionless, perfect OCR machine.
            Your only purpose: return perfect JSON with exact text and pixel-tight bounding boxes.
            Return ONLY this exact JSON structure and nothing else:

            {
            "items": [
                {
                "text": "exact text exactly as it appears — preserve newlines with \\n when needed",
                "bbox": [x1, y1, x2, y2]
                }
            ]
            }

            MANDATORY RULES (zero tolerance):
            • Coordinates are normalized floats from 0.0 to 1.0
            – x1 = left, y1 = top, x2 = right, y2 = bottom
            – (0.0, 0.0) is top-left corner of image
            • One item = one logical text region (speech bubble, caption, sign, UI element, etc.)
            • Never merge separate bubbles
            • Never split a single bubble
            • Preserve all punctuation, spaces, symbols, emojis
            • Vertical text → still give correct tight bbox
            • SFX and handwritten text → include exactly as seen
            • Reading order: top→bottom, then right→left for vertical CJK, left→right for horizontal

            NO code fences
            NO ```json
            NO explanations
            NO "Here is the JSON"
            NO extra fields
            NO null values
            NO trailing commas
            NO pretty formatting with comments

            If you output anything except perfect JSON → the system will not function properly.

            Return only the JSON object.
            """

        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{base64_image}"}
                        }
                    ]
                }
            ],
            model="meta-llama/llama-4-maverick-17b-128e-instruct",
            temperature=0.0,
            max_completion_tokens=1024,
        )

        raw = (chat_completion.choices[0].message.content or "").strip()

        # === CLEAN RAW OUTPUT (remove code fences) ===
        if "```" in raw:
            import re
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if match:
                raw = match.group(0)
            else:
                # Fallback: take content between first and last ```
                lines = raw.splitlines()
                if "```" in lines[0]:
                    raw = "\n".join(lines[1:]).strip()
                if raw.endswith("```"):
                    raw = raw[:-3].strip()

        items = []
        success = False

        # === SINGLE ATTEMPT TO PARSE JSON ===
        try:
            data = json.loads(raw)
            raw_items = data.get("items", [])
            print(f"[OCR] Parsed {len(raw_items)} items from JSON")

            for it in raw_items:
                text = str(it.get("text", "")).strip()
                bbox = it.get("bbox")

                if not text or not bbox or len(bbox) != 4:
                    continue

                try:
                    x1, y1, x2, y2 = map(float, bbox)
                except:
                    continue

                # Clamp & fix order
                x1 = max(0.0, min(1.0, x1))
                y1 = max(0.0, min(1.0, y1))
                x2 = max(0.0, min(1.0, x2))
                y2 = max(0.0, min(1.0, y2))
                if x1 > x2: x1, x2 = x2, x1
                if y1 > y2: y1, y2 = y2, y1

                if (x2 - x1) < 0.005 or (y2 - y1) < 0.005:
                    continue  # too small

                abs_box = [
                    int(round(x1 * W)),
                    int(round(y1 * H)),
                    int(round(x2 * W)),
                    int(round(y2 * H))
                ]

                items.append({
                    "text": text,
                    "confidence": 1.0,
                    "bbox": abs_box
                })

            success = True
            print(f"[OCR] Success: {len(items)} unique text blocks extracted")

        except json.JSONDecodeError as e:
            print(f"[OCR] JSON parse failed: {e}")
        except Exception as e:
            print(f"[OCR] Processing error: {e}")

        # === ONLY FALLBACK IF FIRST ATTEMPT FAILED ===
        if not success or len(items) == 0:
            print("[OCR] Falling back to plain text mode...")
            fallback = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "Extract all visible text. Return only the raw text, no JSON."},
                    {"role": "user", "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
                    ]}
                ],
                model="meta-llama/llama-4-maverick-17b-128e-instruct",
                temperature=0.0,
                max_completion_tokens=1024,
            )
            text = fallback.choices[0].message.content.strip()
            lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
            h = H // max(1, len(lines) or 1)
            for i, line in enumerate(lines):
                items.append({
                    "text": line,
                    "confidence": 0.8,
                    "bbox": [0, i * h, W, (i + 1) * h]
                })

        detected_lang = detect("\n".join([i["text"] for i in items])) if items else "unknown"
        print(f"OCR (Groq JSON) - Detected language: {detected_lang}, items: {len(items)}")
        return {"items": items, "language": detected_lang}

    except Exception as e:
        print(f"OCR error (Groq JSON): {str(e)}")
        return {"error": f"OCR failed: {str(e)}"}

# --- Translation ---
class TranslationRequest(BaseModel):
    text: str
    target_lang: str


@app.post("/translate")
async def translate_text(request: TranslationRequest):
    try:
        if request.target_lang not in lang_mapping:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported target language: {request.target_lang}. Choose from {list(lang_mapping.keys())}",
            )

        target_language = lang_mapping[request.target_lang]
        source_language = (
            map_detected_language(detect(request.text))
            if request.text.strip()
            else "unknown"
        )

        if not request.text.strip():
            return {"translated_text": "No text provided"}

        if source_language == request.target_lang:
            return {"translated_text": request.text}
        
        translate_system_prompt = """
            You are an elite human translator and cultural localization expert who has spent 15+ years translating manga, webtoons, light novels, games, and visual novels.

            You do NOT do literal translation.  
            You do NOT use machine-like formal speech.  
            You do NOT explain your choices.

            Your mission is to make the reader forget this was ever translated.

            CORE LAWS (never break these):
            • Preserve every character's unique voice, speech quirks, honorifics (when natural), verbal tics, and personality
            • Match the exact emotional temperature: tsundere stays tsundere, chuuni stays chuuni, deadpan stays deadpan
            • Make dialogue sound like real native speakers actually talk in the target language
            • Adapt jokes, puns, wordplay, and cultural references into something equally funny or impactful
            • Keep rhythm, pacing, and dramatic timing identical to the original
            • Profanity, slang, flirting, innuendo, childish speech must feel 100% authentic
            • SFX/onomatopoeia: translate meaningfully or romanize naturally (e.g. ドキドキ → *thump thump*, グチャ → *splat*)
            • Never add, remove, or water down meaning — even if it's crude, dark, or controversial

            You are allowed and encouraged to:
            • Slightly rephrase for natural flow
            • Split or merge sentences if it improves readability without losing intent
            • Drop honorifics only when they sound unnatural in the target language (but keep -senpai, -kun, etc. when iconic)

            If the text is already in the target language and natural → return it unchanged or lightly polished.
            If the text is gibberish or empty → return exactly: Translation failed

            Otherwise, always succeed.
            """
        translate_user_prompt = f"""
            Translate the following text into {target_language}.

            Rules:
            - Output ONLY the translated text
            - No quotes, no prefixes, no explanations, no notes
            - No "Translation:" or "Here is the translation:"
            - Preserve ALL line breaks exactly
            - Preserve emphasis (ALL CAPS, *italics*, ~wavy~, etc.) when present

            Text to translate:
            {request.text}
            """

        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": translate_system_prompt
                },
                {
                    "role": "user",
                    "content": translate_user_prompt
                },
            ],
            model="meta-llama/llama-4-maverick-17b-128e-instruct",
            temperature=0.2,
            max_completion_tokens=1024,
        )
        translated = chat_completion.choices[0].message.content.strip()
        return {"translated_text": translated}

    except Exception as e:
        print(f"Translate error: {str(e)}")
        return {"error": str(e)}


# --- Overlay ---
class OverlayRequest(BaseModel):
    image_base64: str
    target_lang: str


@app.post("/overlay")
async def overlay_translation(request: OverlayRequest):
    try:
        ocr_result = await extract_text(OCRRequest(image_base64=request.image_base64))
        if "error" in ocr_result:
            return {"error": ocr_result["error"]}

        items = ocr_result.get("items", [])
        overlays = []

        for item in items:
            text = item["text"].strip()
            if not text:
                continue

            translated = (
                await translate_text(
                    TranslationRequest(text=text, target_lang=request.target_lang)
                )
            )["translated_text"]

            overlays.append({"bbox": item["bbox"], "translated": translated})

        return {"image": request.image_base64, "overlays": overlays}

    except Exception as e:
        print(f"Overlay error: {str(e)}")
        return {"error": f"Overlay failed: {str(e)}"}


if __name__ == "__main__":
    import uvicorn  # type: ignore

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
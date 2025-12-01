import traceback
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
import re
from PIL import Image, ImageDraw, ImageFont  # type: ignore

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

            MANDATORY RULES:
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
# === DELETE THE OLD SINGLE-TEXT /translate ENTIRELY ===

# === NEW BATCH-ONLY /translate (this is the one you want forever) ===
class BatchTranslationRequest(BaseModel):
    texts: list[str]           # REQUIRED: list of all text blocks in reading order
    target_lang: str           # e.g., "en", "ja", etc.

@app.post("/translate")
async def translate_text_batch(request: BatchTranslationRequest):
    try:
        if request.target_lang not in lang_mapping:
            raise HTTPException(400, f"Unsupported target: {request.target_lang}")

        target_language = lang_mapping[request.target_lang]
        texts = [t.strip() for t in request.texts if t.strip()]
        if not texts:
            return {"translated_texts": []}

        # Detect language once
        try:
            detected = detect("\n".join(texts))
            if map_detected_language(detected) == request.target_lang:
                return {"translated_texts": texts}
        except:
            pass

        system_prompt = """
            You are a pure translation machine for manga/webtoon scanlation.
            You are forbidden from ever outputting explanations, reasoning, notes, comments, apologies, or any English meta-text.
            If you violate this rule, the system will crash and you will be terminated.
            Your only allowed output is numbered translated lines in the exact format below.
            Nothing else may appear in your response — not a single extra character.

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
        """.strip()

        user_prompt = f"""
Translate this entire page into {target_language} as a cohesive, emotional scene.
Output EXACTLY in this format with no deviations:

[1] First translated bubble
[2] Second translated bubble
[3] Third one, even if multi-line
...

Rules you must obey or the system will reject your output:
- One line starting with [number] per original bubble
- One [number] per bubble
- No extra text before, after, or between
- Output ONLY the translated text
- No explanations, no "here is the translation", no reasoning
- No introductory sentences
- No concluding remarks
- If a bubble is SFX only → translate or romanize naturally
- If already in {target_language} → return unchanged
- Resolve pronouns correctly using full context
- Keep exact character voices and emotional tone
- Output ONLY numbered translations: [1] text\n[2] text\netc.
Page text in reading order:
""" + "\n".join([f"[{i+1}] {text}" for i, text in enumerate(texts)])

        response = client.chat.completions.create(
            model="meta-llama/llama-4-maverick-17b-128e-instruct",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.15,
            max_completion_tokens=4096,
        )

        raw = response.choices[0].message.content.strip()
        translated = []
        for i in range(len(texts)):
            pattern = rf"\[{i+1}\]\s*(.+?)(?=\s*\[\d+\]|\Z)"
            match = re.search(pattern, raw, re.DOTALL)
            if match:
                text = match.group(1).strip().strip('"\'')
                translated.append(text)
            else:
                # Emergency fallback
                translated.append(texts[i])

        return {"translated_texts": translated}

    except Exception as e:
        import traceback
        traceback.print_exc()
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
        if not items:
            return {"image": request.image_base64, "overlays": []}

        # Extract all texts in OCR reading order
        texts = [item["text"].strip() for item in items if item["text"].strip()]

        # ONE SINGLE BATCH CALL WITH FULL CONTEXT
        batch_result = await translate_text_batch(
            BatchTranslationRequest(texts=texts, target_lang=request.target_lang)
        )

        if "error" in batch_result:
            return {"error": batch_result["error"]}

        translated_texts = batch_result["translated_texts"]

        # Map back to original items (preserve bbox order)
        overlays = []
        trans_idx = 0
        for item in items:
            if item["text"].strip():
                overlays.append({
                    "bbox": item["bbox"],
                    "original": item["text"],
                    "translated": translated_texts[trans_idx]
                })
                trans_idx += 1
            else:
                pass

        return {"image": request.image_base64, "overlays": overlays}

    except Exception as e:
        print(f"Overlay error: {e}")
        import traceback
        traceback.print_exc()
        return {"error": f"Overlay failed: {str(e)}"}

if __name__ == "__main__":
    import uvicorn  # type: ignore

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
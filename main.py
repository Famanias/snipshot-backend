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
    Returns items with bbox = [x1,y1,x2,y2] in normalized 0..1 coordinates.
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
            "You are an OCR engine. Return structured JSON only. "
            "Detect all visible text as separate items with tight line-level bounding boxes. "
            "Coordinates MUST be normalized in [0,1] relative to image width/height, format: [x1,y1,x2,y2]. "
            "(0,0) is top-left. Keep reading order top-to-bottom, left-to-right. "
            "Output format:\n"
            "{\n  \"items\": [\n    {\n      \"text\": string,\n      \"bbox\": [number, number, number, number]\n    }\n  ]\n}"
        )
        user_content = [
            {"type": "text", "text": "Extract text with normalized bounding boxes in strict JSON as specified."},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}", "detail": "high"}},
        ]

        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            model="meta-llama/llama-4-maverick-17b-128e-instruct",
            temperature=0.4,
            max_completion_tokens=1024,
        )

        raw = (chat_completion.choices[0].message.content or "").strip()

        # Some models wrap JSON in code fences; strip if present
        if raw.startswith("```"):
            # Remove the first fence line and the last ``` if present
            parts = raw.split("\n")
            if parts[0].startswith("```"):
                parts = parts[1:]
            if parts and parts[-1].strip().startswith("```"):
                parts = parts[:-1]
            raw = "\n".join(parts).strip()

        items = []
        try:
            payload = json.loads(raw)
            for it in payload.get("items", []):
                text = (it.get("text") or "").strip()
                bbox = it.get("bbox") or []
                if not text or not isinstance(bbox, list) or len(bbox) != 4:
                    continue
                x1, y1, x2, y2 = bbox
                # Validate and clip normalized coords
                def clamp(v):
                    try:
                        return float(max(0.0, min(1.0, v)))
                    except Exception:
                        return 0.0
                x1, y1, x2, y2 = clamp(x1), clamp(y1), clamp(x2), clamp(y2)
                if x2 < x1:
                    x1, x2 = x2, x1
                if y2 < y1:
                    y1, y2 = y2, y1
                # Skip degenerate boxes
                if (x2 - x1) < 1e-4 or (y2 - y1) < 1e-4:
                    continue
                # Convert normalized [0..1] to absolute pixel [x1,y1,x2,y2]
                abs_box = [int(round(x1 * W)), int(round(y1 * H)), int(round(x2 * W)), int(round(y2 * H))]
                items.append({"text": text, "confidence": 1.0, "bbox": abs_box})
        except Exception as e:
            print(f"[OCR] JSON parse failed, falling back to line-split: {e}")

        if not items:
            # Fallback: ask Groq for plain text and split lines, full-width boxes per line
            fallback_completion = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "Extract all visible text from the image. Return only the text."},
                    {"role": "user", "content": user_content},
                ],
                model="meta-llama/llama-4-maverick-17b-128e-instruct",
                temperature=0.4,
                max_completion_tokens=1024,
            )
            extracted_text = (fallback_completion.choices[0].message.content or "").strip()
            lines = [ln.strip() for ln in extracted_text.splitlines() if ln.strip()]
            h_step = int(H / max(1, len(lines)))
            for idx, line in enumerate(lines):
                y1, y2 = idx * h_step, (idx + 1) * h_step
                items.append({"text": line, "confidence": 1.0, "bbox": [0, y1, W, y2]})

        detected_lang = detect("\n".join([i["text"] for i in items])) if items else "unknown"
        print(f"OCR (Groq JSON) - Detected language: {detected_lang}, items: {len(items)}")
        return {"items": items, "language": detected_lang}

    except Exception as e:
        print(f"OCR error (Groq JSON): {str(e)}")
        return {"error": f"OCR failed: {str(e)}"}

@app.post("/ocr")
async def extract_text(request: OCRRequest):
    try:
        image_bytes = base64.b64decode(request.image_base64)
        tess_langs = "eng+jpn+jpn_vert+kor+chi_sim+chi_sim_vert+chi_tra+chi_tra_vert"

        # --- 1st pass: grayscale preprocessing ---
        denoised = preprocess_image_bytes(image_bytes, mode="grayscale")
        data = pytesseract.image_to_data(denoised, lang=tess_langs, output_type=Output.DICT)

        # --- If no text, retry with binary preprocessing ---
        if all((t or "").strip() == "" for t in data.get("text", [])):
            print("OCR grayscale failed → retrying with binary mode")
            denoised = preprocess_image_bytes(image_bytes, mode="binary")
            data = pytesseract.image_to_data(denoised, lang=tess_langs, output_type=Output.DICT)

        # --- Group words into lines with bounding boxes ---
        n = len(data.get("text", []))
        lines = {}
        for i in range(n):
            word = (data["text"][i] or "").strip()
            try:
                conf = float(data["conf"][i])
            except Exception:
                conf = -1.0
            if not word or conf < 30:
                continue

            key = (
                int(data.get("block_num", [0]*n)[i]),
                int(data.get("par_num", [0]*n)[i]),
                int(data.get("line_num", [0]*n)[i]),
            )
            x, y, w, h = (
                int(data.get("left", [0]*n)[i]),
                int(data.get("top", [0]*n)[i]),
                int(data.get("width", [0]*n)[i]),
                int(data.get("height", [0]*n)[i]),
            )
            if key not in lines:
                lines[key] = {"texts": [word], "confs": [conf], "bbox": [x, y, x+w, y+h]}
            else:
                lines[key]["texts"].append(word)
                lines[key]["confs"].append(conf)
                bx = lines[key]["bbox"]
                bx[0] = min(bx[0], x)
                bx[1] = min(bx[1], y)
                bx[2] = max(bx[2], x+w)
                bx[3] = max(bx[3], y+h)

        # --- Convert lines to items ---
        items, full_text_parts = [], []
        for v in lines.values():
            line_text = " ".join(v["texts"]).strip()
            avg_conf = sum(v["confs"]) / max(1, len(v["confs"]))
            items.append({"text": line_text, "confidence": round(avg_conf, 2), "bbox": v["bbox"]})
            full_text_parts.append(line_text)

        full_text = "\n".join(full_text_parts).strip()
        detected_lang = detect(full_text) if full_text else "unknown"

        # --- Fallback to Groq OCR if nothing extracted ---
        if len(items) == 0:
            print("Tesseract found nothing → falling back to Groq OCR")
            _, enc = cv2.imencode(".png", denoised)
            base64_image = base64.b64encode(enc.tobytes()).decode("utf-8")
            # ... [your existing Groq fallback code] ...

        print(f"OCR - Detected language: {detected_lang}, items: {len(items)}")
        return {"items": items, "language": detected_lang}

    except Exception as e:
        print(f"OCR error: {str(e)}")
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

        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert translator with proficiency in multiple languages. Translate the provided text accurately into the specified target language, preserving the tone, context, and intent, including profanity for educational purposes. If the text cannot be translated due to invalid input, unsupported language, or other issues, return only 'Translation failed' without additional explanation. Handle idiomatic expressions, cultural nuances, and special characters appropriately. Return only the translated text unless specified otherwise."
                },
                {
                    "role": "user",
                    "content": f"Translate the following text: '{request.text}' into {target_language}. "
                            f"Return only the translated text, with no explanations or additional output."
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

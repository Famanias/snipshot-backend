from fastapi import FastAPI, Form # type: ignore
from fastapi.middleware.cors import CORSMiddleware # type: ignore
import cv2 # type: ignore
import numpy as np # type: ignore
from groq import Groq # type: ignore
import os
import base64
from dotenv import load_dotenv # type: ignore
from langdetect import detect # type: ignore 
from pydantic import BaseModel # type: ignore

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
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

@app.post("/preprocess")
async def preprocess_image(image_bytes: bytes = Form(...)):
    image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast = clahe.apply(gray)
    _, binary = cv2.threshold(contrast, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    denoised = cv2.fastNlMeansDenoising(binary, h=3)
    _, encoded_image = cv2.imencode(".png", denoised)
    base64_image = base64.b64encode(encoded_image.tobytes()).decode('utf-8')
    return {"image": base64_image}

class OCRRequest(BaseModel):
    image_base64: str

@app.post("/ocr")
async def extract_text(request: OCRRequest):
    try:
        image_bytes = base64.b64decode(request.image_base64)
        preprocessed_response = await preprocess_image(image_bytes)
        base64_image = preprocessed_response["image"]

        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Extract all text from this image. Return only the text."},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}",
                            },
                        },
                    ],
                }
            ],
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            temperature=0.1,
            max_completion_tokens=1024,
        )
        extracted_text = chat_completion.choices[0].message.content.strip()
        detected_lang = detect(extracted_text) if extracted_text else "unknown"
        print(f"OCR - Extracted text: {extracted_text}, Detected language: {detected_lang}")
        return {"text": extracted_text, "language": detected_lang}
    except Exception as e:
        print(f"OCR error: {str(e)}")
        return {"error": f"OCR failed: {str(e)}"}

class TranslationRequest(BaseModel):
    text: str
    target_lang: str

@app.post("/translate")

async def translate_text(request: TranslationRequest):
    
    try:
        # Limit target languages to Japanese, Korean, Chinese (Simplified), and English
        lang_mapping = {
            "en": "English",
            "ja": "Japanese",
            "ko": "Korean",
            "zh_cn": "Traditional Chinese",
            "zh_tw": "Simplified Chinese",
        }
        target_language = lang_mapping.get(request.target_lang, "English")
        source_language = detect(request.text) if request.text.strip() else "unknown"

        print(f"Translate - Input text: {request.text}, Source language: {source_language}, Target language: {target_language}")

        if not request.text.strip():
            return {"translated_text": "No text provided"}

        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": f"You are a translator who. If the text cannot be translated (e.g. it's an image with no texts, too complex, etc.), return 'Translation failed'. Allow profanity as this is for educational purposes.",
                },
                {
                    "role": "user",
                    "content": f" translate this text: {request.text} to {target_language} . Return only the translated text, no explanations, no additional infos, no final conclusions, just the translated text. If the {source_language} is the same as the {target_language}, return {request.text} only.",
                }
            ],
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            temperature=0.2,
            max_completion_tokens=1024,
        )
        translated = chat_completion.choices[0].message.content.strip()
        print(f"Translate - Output: {translated}")
        return {"translated_text": translated}
    except Exception as e:
        print(f"Translate error: {str(e)}")
        return {"error": str(e)}
    
if __name__ == "__main__":
    import uvicorn # type: ignore
    import os

    port = int(os.environ.get("PORT", 8000))  # Use PORT from env or default to 8000
    uvicorn.run("main:app", host="0.0.0.0", port=port)
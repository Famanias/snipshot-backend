from fastapi import FastAPI  # type: ignore
from fastapi.middleware.cors import CORSMiddleware  # type: ignore
from groq import Groq  # type: ignore
from dotenv import load_dotenv  # type: ignore
from pydantic import BaseModel  # type: ignore
from langdetect import detect  # type: ignore

import os
import base64

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


class ImageTranslationRequest(BaseModel):
    image_base64: str
    target_lang: str


@app.post("/translate-image")
async def translate_image(request: ImageTranslationRequest):
    try:
        # Supported target languages
        lang_mapping = {
            "en": "English",
            "ja": "Japanese",
            "ko": "Korean",
            "zh_cn": "Simplified Chinese",
            "zh_tw": "Traditional Chinese",
        }

        target_language = lang_mapping.get(
            request.target_lang,
            "English"
        )

        if not request.image_base64.strip():
            return {
                "error": "No image provided"
            }

        # Remove possible data URL prefix
        clean_base64 = request.image_base64

        if "," in clean_base64:
            clean_base64 = clean_base64.split(",")[1]

        # Validate base64
        try:
            base64.b64decode(clean_base64)
        except Exception:
            return {
                "error": "Invalid base64 image"
            }

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
            - Do NOT confuse Hangul with other scripts. It is always composed of consonant + vowel block shapes.
            - Hanja (Chinese characters) may appear occasionally — extract as-is and note [HANJA].

            **Text direction:**
            - Manhwa (Korean comics) and modern text: HORIZONTAL, left-to-right.
            - Traditional or stylized documents: may be VERTICAL — apply same column rules as Japanese if so.
            - Identify direction per bubble/block before extracting.

            **Manhwa-specific elements:**
            - [SPEECH]: standard dialogue bubbles.
            - [THOUGHT]: thought bubbles (often cloud-shaped or dashed outline).
            - [NARRATION]: caption boxes.
            - [SFX]: Korean onomatopoeia, often bold/stylized.
            Example: [SFX]: 쿵 → *thud / heavy thump*

            **Panel reading order:**
            - Manhwa is typically read LEFT-TO-RIGHT (opposite of manga).
            - Panel order: TOP-LEFT → TOP-RIGHT → next row left → next row right.
            - Confirm direction from panel layout before extracting.

            **Line break handling:**
            - Hangul syllables may wrap mid-word due to bubble width.
            - Reconstruct broken syllable sequences into complete words before outputting.

            ---

            ### CHINESE TEXT EXTRACTION

            **Script identification — distinguish first:**
            - Simplified Chinese (简体字): used in Mainland China. Fewer strokes, streamlined characters.
            - Traditional Chinese (繁體字): used in Taiwan, Hong Kong, Macau. More complex strokes.
            - Do NOT mix the two. Identify which variant is present and label accordingly: [ZH-CN] or [ZH-TW].

            **Text direction:**
            - Simplified Chinese (manhua, modern): mostly HORIZONTAL, left-to-right.
            - Traditional Chinese (classic publications, some Taiwanese manhua): may be VERTICAL, right-to-left columns.
            - Apply same vertical column rules as Japanese if vertical layout is detected.

            **Script mixing:**
            - Chinese text may include Bopomofo (ㄅㄆㄇㄈ, used in Traditional Chinese) beside characters — extract and note in parentheses.
            - Punctuation in vertical text is rotated — normalize to standard punctuation in output.

            **Manhua-specific elements:**
            - [SPEECH], [THOUGHT], [NARRATION], [SFX] — same labeling conventions as above.
            - [SFX] example: [SFX]: 碰 → *bang / collision impact*

            **Panel reading order:**
            - Simplified Chinese manhua: LEFT-TO-RIGHT (same as manhwa).
            - Traditional Chinese manhua: RIGHT-TO-LEFT (same as manga).
            - Confirm from panel layout.

            ---

            ## STEP 3 — TRANSLATION

            ### CRITICAL — Reconstruct before translating (ALL languages):
            - Line breaks inside a bubble are layout artifacts — NOT sentence boundaries.
            - Before translating, join all lines in a bubble into one continuous sentence.
            - Translate the full reconstructed sentence as a single unit.
            - Do NOT translate line by line or fragment by fragment.

            **Japanese reconstruction example:**
            Extracted (line-broken from vertical columns):
                医学的に言えば
                ２１番染色体が
                ３本ある異常
                のことです

            Reconstruct: "医学的に言えば２１番染色体が３本ある異常のことです"
            Translate:   "Medically speaking, it refers to an abnormality where chromosome 21 has 3 copies."

            NOT (line-by-line):
                "Medically speaking / Chromosome 21 / Three abnormalities / That is the thing." ✗

            **Korean reconstruction example:**
            Extracted (line-broken):
                성낙숙 쪽에서
                했다기엔 군이...
                이 이름으로 할까?

            Reconstruct: "성낙숙 쪽에서 했다기엔 군이... 이 이름으로 할까?"
            Translate:   "For it to have been done on Seongnaksuk's side... shall we go with this name?"

            **Chinese reconstruction example:**
            Extracted (line-broken):
                你说的那个人
                真的存在吗？

            Reconstruct: "你说的那个人真的存在吗？"
            Translate:   "Does the person you mentioned really exist?"

            ---

            ### Language-Specific Translation Rules

            **Japanese:**
            - Subjects are frequently omitted — infer from context. Do NOT insert an explicit pronoun unless unambiguous.
            - Preserve sentence-final particle nuance:
                か   → question tone ("...right?", "is it?")
                ね   → seeking agreement ("...you know", "isn't it")
                よ   → assertive declaration ("I'm telling you", "it is")
                な   → personal reflection ("...I suppose", "...huh")
                わ   → soft feminine assertion
                ぞ／ぜ → strong masculine assertion
                って → casual quoting or trailing ("they said...", "like...")
            - Polite register (です／ます) vs casual (だ／る) — preserve the tone in translation.
            - Ellipses (……) at bubble end = trailing or interrupted speech — keep in translation.
            - Furigana readings should inform translation if the kanji reading is ambiguous.

            **Korean:**
            - Preserve speech level in translation:
                합쇼체 / 해요체 → formal/polite ("I will...", "Please...")
                해체 / 반말      → casual/informal ("I'm gonna...", "Hey...")
            - Sentence-final endings carry tone:
                ~요 / ~습니다 → polite
                ~야 / ~아 / ~다 → casual/direct
                ~네 → mild surprise or realization
                ~겠 → intention or conjecture ("I think...", "I will...")
            - Honorifics embedded in speech (씨, 님, 선생님, 형, 오빠, 언니, 누나) — keep original + note role in parentheses.
            - Subjects and objects are frequently dropped — infer from context.
            - Ellipses (...) and dashes (—) signal trailing or hesitant speech — preserve.

            **Chinese:**
            - Distinguish Simplified vs Traditional in translation output — do not mix vocabulary.
                Simplified: 电脑, 软件, 网络
                Traditional: 電腦, 軟體, 網路
            - Measure words (量词 / 量詞) are grammatically required — translate naturally without over-literalizing.
                一本书 → "a book" (not "one volume-of book")
            - Classical Chinese patterns (也、乃、曰、之) in older manhua — translate with appropriate formal register.
            - Regional expressions:
                Mainland: 打的 (take a taxi), 不客气 (you're welcome)
                Taiwan/HK: 計程車, 不客氣 — preserve regional flavor in translation.
            - Preserve rhetorical 吧 (suggestion/uncertainty), 啊 (exclamation), 嘛 (obviousness), 呢 (continuation question).

            ### General Translation Rules (ALL languages):
            - Translate all extracted text into {target_language}.
            - Preserve original meaning, tone, and nuance — do NOT sanitize, censor, or soften.
            - Maintain all structural labels: [SPEECH], [THOUGHT], [NARRATION], [SFX], panel markers.
            - For untranslatable proper nouns (names, places, brands): keep original + add romanization if helpful.
            - If a segment is already in {target_language}: keep as-is and note [already in {target_language}].

            ---

            ## STEP 4 — VISUAL CONTEXT SUMMARY
            - Describe only what is clearly and directly visible in the image.
            - State the detected image type from Step 1 and the detected language(s).
            - For manga/manhwa/manhua: describe panel layout, visible characters, and scene — based only on visual evidence.
            - Do NOT infer emotions, plot details, or off-screen events not visually supported.
            - Keep it factual and concise (2–4 sentences max).

            ---

            ## HARD RULES
            - Do NOT hallucinate, invent, or assume any text not visible in the image.
            - Do NOT skip or omit any visible text, however small or peripheral.
            - Do NOT alter profanity, slurs, or sensitive language — translate faithfully.
            - Do NOT add commentary, disclaimers, or unsolicited opinions.
            - Respond ONLY in the exact output format below — no preamble, no extra sections.

            ---

            ## OUTPUT FORMAT (strict — no deviations)

            EXTRACTED_TEXT:
            <original extracted text with panel/bubble/type labels if manga/manhwa/manhua, or NO_TEXT_FOUND>

            TRANSLATION:
            <translated text in {target_language}, preserving all labels and structure, or N/A if NO_TEXT_FOUND>

            SUMMARY:
            <2–4 sentence factual image and scene description including detected image type and language>
            """
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{clean_base64}"
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
                extracted_part = response_text.split(
                    "TRANSLATION:"
                )[0]

                extracted_text = (
                    extracted_part
                    .replace("EXTRACTED_TEXT:", "")
                    .strip()
                )

                remaining = response_text.split(
                    "TRANSLATION:"
                )[1]

                if "SUMMARY:" in remaining:
                    translation_part, summary_part = remaining.split(
                        "SUMMARY:",
                        1
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
        print(f"Extracted text: {extracted_text}")
        print(f"Translated text: {translated_text}")
        print(f"Summary: {summary}")

        return {
            "detected_language": detected_language,
            "extracted_text": extracted_text,
            "translated_text": translated_text,
            "summary": summary,
            "raw_response": response_text
        }

    except Exception as e:
        print(f"Translation error: {str(e)}")

        return {
            "error": f"Translation failed: {str(e)}"
        }


if __name__ == "__main__":
    import uvicorn  # type: ignore

    port = int(os.environ.get("PORT", 8000))

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port
    )
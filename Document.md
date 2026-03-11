# SnipShot Backend — Full System Documentation

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [System Architecture](#2-system-architecture)
3. [End-to-End Request Flow](#3-end-to-end-request-flow)
4. [Image Translation Pipeline](#4-image-translation-pipeline)
   - 4.1 [Image Input & Pre-Processing](#41-image-input--pre-processing)
   - 4.2 [Text Detection](#42-text-detection)
   - 4.3 [OCR (Optical Character Recognition)](#43-ocr-optical-character-recognition)
   - 4.4 [Textline Merging](#44-textline-merging)
   - 4.5 [Translation](#45-translation)
   - 4.6 [Mask Refinement](#46-mask-refinement)
   - 4.7 [Inpainting (Text Removal)](#47-inpainting-text-removal)
   - 4.8 [Text Rendering](#48-text-rendering)
5. [Models Used in the Project](#5-models-used-in-the-project)
   - 5.1 [Detection Models](#51-detection-models)
   - 5.2 [OCR Models](#52-ocr-models)
   - 5.3 [Inpainting Models](#53-inpainting-models)
   - 5.4 [Upscaling Models](#54-upscaling-models)
   - 5.5 [Colorization Model](#55-colorization-model)
   - 5.6 [Translation Model (Groq LLM)](#56-translation-model-groq-llm)
6. [Server APIs](#6-server-apis)
   - 6.1 [VM Translator API](#61-vm-translator-api)
   - 6.2 [Database API](#62-database-api)
7. [Storage](#7-storage)
8. [Configuration System](#8-configuration-system)
9. [Key Libraries & Dependencies](#9-key-libraries--dependencies)

---

## 1. Project Overview

**SnipShot Backend** is a full-stack image translation service designed primarily for manga, manhwa, and manhua (East Asian comics). It accepts comic images, automatically detects Japanese/Korean/Chinese text regions, reads those texts using OCR, translates them to English (or another target language) using an LLM, removes the original text from the image by inpainting, and renders the translated text back onto the image.

The system is split into two independent backend services:

| Service | Purpose | Deployment |
|---|---|---|
| **VM Translator API** | Stateless image translation engine | Google Cloud VM |
| **Database API** | User auth, image metadata, folder management | Render (Supabase) |

---

## 2. System Architecture

```
┌───────────────────────────────────────────────┐
│             FRONTEND (Desktop / Mobile)        │
└──────────────┬──────────────────┬─────────────┘
               │                  │
               ▼                  ▼
┌──────────────────────┐  ┌─────────────────────┐
│  VM TRANSLATOR API   │  │    DATABASE API      │
│  (Google Cloud VM)   │  │  (Render + Supabase) │
│                      │  │                      │
│  POST /translate     │  │  POST /users/login   │
│  POST /translate/raw │  │  POST /users/register│
│  GET  /health        │  │  GET  /images        │
│                      │  │  POST /images        │
│  Port: 8000          │  │  GET  /folders       │
│  (Public IP)         │  │  Port: 10000         │
└──────────┬───────────┘  └──────────────────────┘
           │
           │ Internal (127.0.0.1:8001)
           ▼
┌──────────────────────┐
│  manga_translator    │
│  (shared/API mode)   │
│  POST /simple_execute│
│  /translate          │
└──────────────────────┘
           │
           ▼
┌──────────────────────┐
│   Supabase Storage   │
│   (Translated PNGs)  │
└──────────────────────┘
```

### Process Startup (`main.py`)

The orchestrator (`main.py`) at the root of the project starts two subprocesses:

1. **Manga Translator Backend** — runs `python -m manga_translator shared` on port `8001` (internal only).
2. **FastAPI Server** — runs `uvicorn server.translator_api:app` on port `8000` (public-facing).

The API server acts as a gateway: it receives image uploads, forwards them to the internal translation engine, and returns the translated image URL from Supabase Storage.

---

## 3. End-to-End Request Flow

```
1. Frontend sends POST /translate (image + config JSON)
        │
2. translator_api.py receives the request
   - Reads image bytes → PIL Image (RGB)
   - Parses JSON config → Config object
   - Pickles {image, config} → sends to internal backend (port 8001)
        │
3. manga_translator (shared mode) processes the image
   - Runs the full translation pipeline (see Section 4)
   - Returns a Context object (containing the result PIL Image)
        │
4. translator_api.py uploads result to Supabase Storage
   - Converts PIL Image → PNG bytes
   - Uploads to bucket path: translated/{timestamp}.png
   - Returns { image_url, storage_path }
        │
5. Frontend receives the Supabase public URL
6. Frontend calls Database API to save the image metadata (URL, user, folder)
```

---

## 4. Image Translation Pipeline

The core translation pipeline lives in `manga_translator/manga_translator.py`, implemented as the `MangaTranslator` class. The `_translate()` method orchestrates all stages sequentially.

### 4.1 Image Input & Pre-Processing

**Entry point:** `MangaTranslator.translate(image, config)`

1. The input `PIL.Image` is stored in a `Context` object.
2. **Colorization (optional):** If `config.colorizer.colorizer != none`, the image is colorized first using `MangaColorizationV2` (a GAN model). This is useful for black-and-white manga so that downstream processing has better visual contrast.
3. **Upscaling (optional):** If `config.upscale.upscale_ratio > 1`, the image is upscaled using Waifu2x, ESRGAN, or 4xUltraSharp. This improves detection and OCR accuracy on small images.
4. **Image conversion:** The (possibly upscaled) image is converted from PIL to a NumPy array in RGB format (`ctx.img_rgb`). Any alpha channel is separated into `ctx.img_alpha`.

### 4.2 Text Detection

**Function:** `_run_detection()` → calls `dispatch_detection()`  
**Output:** `ctx.textlines` (list of `Quadrilateral` regions), `ctx.mask_raw`, `ctx.mask`

The detector scans the full image and returns **bounding polygons** (quadrilaterals) around each text region, along with a **binary mask** that marks text pixels.

Detection is configurable and supports multiple backends (see Section 5.1). The default model is a **DBNet with ResNet34** backbone. The detector takes parameters:
- `detection_size`: Resolution at which to run detection (e.g., 1024px)
- `text_threshold`: Minimum confidence for a pixel to be considered text
- `box_threshold`: Minimum score for a candidate box to be kept
- `unclip_ratio`: How much to expand the detected boxes outward

If no text regions are found, the pipeline short-circuits and returns the upscaled image as-is.

If `verbose` mode is enabled, the raw mask and unfiltered bounding boxes are saved as debug images (`mask_raw.png`, `bboxes_unfiltered.png`).

### 4.3 OCR (Optical Character Recognition)

**Function:** `_run_ocr()` → calls `dispatch_ocr()`  
**Input:** Full image + list of detected `Quadrilateral` regions  
**Output:** Updated `ctx.textlines`, each now carrying recognized text

Each text region quadrilateral is **rectified** (geometrically transformed) to a standard upright rectangle matching the detected text direction (horizontal or vertical). The OCR model processes these crops and returns:
- The **recognized Unicode string** for each region
- The **language** of the text (auto-detected, e.g., Japanese, Korean, Chinese)
- **Character-level probability** scores
- The **text direction** (horizontal / vertical)

Multiple OCR backends are available (see Section 5.2). The `Model48pxOCR` and `ModelMangaOCR` are the primary models.

If no text strings are successfully recognized, the pipeline short-circuits.

### 4.4 Textline Merging

**Function:** `_run_textline_merge()` → `dispatch_textline_merge()`  
**Output:** `ctx.text_regions` (list of `TextBlock` objects)

Individual OCR results (`Quadrilateral`) are grouped together into logical **speech bubbles or text blocks** (`TextBlock`). The merging algorithm uses graph connectivity:

1. A graph is built where nodes are individual text lines.
2. Two lines are connected if they are close enough, have similar font sizes, and similar angles.
3. Connected components of the graph form candidate text regions.
4. Each region may be further **split** if internal line distances are too large (using a minimum spanning tree analysis).

The merged result respects reading order — lines within a block are sorted for left-to-right or right-to-left (RTL) reading.

**Pre-dictionary substitution** is applied after merging: a user-defined dictionary file can map patterns (regex) in the recognized source text to custom strings before translation.

### 4.5 Translation

**Function:** `_run_text_translation()` → `dispatch_translation()`  
**Input:** List of `TextBlock` objects with recognized text  
**Output:** Updated `TextBlock` objects with `.translation` field set

The system uses **Groq's LLM API** (see Section 5.6). The translation process:

1. All text regions on the page are collected.
2. Previous page translations are appended as **context** (configurable via `CONTEXT_RETENTION` and `CONTEXT_LENGTH` env vars) to help the model maintain consistency.
3. Texts are sent to the Groq API with a specialized system prompt tailored for manga/manhwa/manhua translation.
4. The model returns a JSON object `{"translated": "..."}`.
5. The translated string is assigned back to each `TextBlock`.

**Post-dictionary substitution** is applied after translation: patterns can be replaced in the output text before rendering.

### 4.6 Mask Refinement

**Function:** `_run_mask_refinement()`

If the detector did not produce a fine enough text mask, this step refines the binary mask. It uses morphological operations and pixel-level analysis to produce a precise mask that tightly covers only the text pixels — this matters because a too-large mask leaves visible artifacts after inpainting.

### 4.7 Inpainting (Text Removal)

**Function:** `_run_inpainting()` → `dispatch_inpainting()`  
**Input:** `ctx.img_rgb` + `ctx.mask`  
**Output:** `ctx.img_inpainted` — image with original text erased

The inpainter fills in the regions covered by the text mask, reconstructing the underlying artwork (background, screen tones, colors) as if the text was never there. This creates the clean image onto which translated text will be drawn.

The default inpainter is **AOT-GAN** (`AotInpainter`). Other options include LaMa and Stable Diffusion (see Section 5.3).

If `verbose` mode is on, the inpainted result is saved as `inpainted.png`.

### 4.8 Text Rendering

**Function:** `_run_text_rendering()` → `dispatch_rendering()`  
**Input:** Inpainted image + translated `TextBlock` list  
**Output:** `ctx.img_rendered` — final image with translated text drawn in

Translated text is rendered back into the original text regions. The renderer:

1. Determines font size based on the original text region size and a minimum font size formula (`(image_h + image_w) / 200`).
2. Handles **horizontal and vertical** text layout.
3. Adjusts region bounding boxes if translated text is longer than original.
4. Applies **foreground/background color detection** from the original text for matching text colors.
5. Uses bundled CJK fonts (`fonts/` directory) for rendering:
   - `msgothic.ttc` (MS Gothic — Japanese)
   - `msyh.ttc` (Microsoft YaHei — Chinese)
   - `NotoSansMonoCJK-VF.ttf.ttc` (multilingual CJK)

Three renderer implementations are available:
- `default`: Standard renderer
- `manga2eng`: Optimized for English text in manga layouts
- `manga2eng_pillow`: Pillow-based variant of the manga2eng renderer

The final result is stored as `ctx.result` (a `PIL.Image`), which is then returned through the API chain and uploaded to Supabase Storage.

---

## 5. Models Used in the Project

### 5.1 Detection Models

| Key | Class | Architecture | Checkpoint |
|---|---|---|---|
| `default` | `DefaultDetector` | **DBNet + ResNet-34** | `detect-20241225.ckpt` |
| `dbconvnext` | `DBConvNextDetector` | **DBNet + ConvNext** backbone | Custom weights |
| `ctd` | `ComicTextDetector` | **YOLOv5 + DBNet** (dual-head) | `comictextdetector.pt` (PyTorch) / `comictextdetector.pt.onnx` (ONNX for CPU) |
| `craft` | `CRAFTDetector` | **VGG16-BN U-Net** (CRAFT) | CRAFT weights |
| `paddle` | `PaddleDetector` | **PaddleOCR** detector | PaddleOCR runtime |
| `none` | `NoneDetector` | No detection | — |

**Default (DBNet + ResNet-34):**  
The default detector uses a Differentiable Binarization Network (DBNet). The input image is preprocessed with bilateral filtering, then resized to `detect_size × detect_size`. The model outputs a probability map (`db`) and a text mask (`mask`). The `SegDetectorRepresenter` post-processes these outputs into bounding polygons using a contour-based approach with unclipping (expanding detected regions by `unclip_ratio`).

**CTD (Comic Text Detector):**  
A dual-purpose model combining YOLOv5 (for box detection) with a DBNet-style segmentation head (for pixel masks). On CUDA/MPS devices it loads a PyTorch `.pt` file; on CPU it falls back to an ONNX model processed by OpenCV's DNN module.

**CRAFT:**  
Character Region Awareness for Text Detection. Uses a VGG16-BN encoder as backbone followed by a U-Net style decoder (`upconv1`–`upconv4`). It predicts per-character region scores and affinity scores between characters, which are then grouped into word/line bounding boxes.

### 5.2 OCR Models

| Key | Class | Architecture | Checkpoint |
|---|---|---|---|
| `32px` | `Model32pxOCR` | Autoregressive sequence model | `ocr_ar_32px.ckpt` |
| `48px` | `Model48pxOCR` | **Roformer + XPos + Local Attention ViT** | `ocr_ar_48px.ckpt` |
| `48px_ctc` | `Model48pxCTCOCR` | CTC sequence model (48px) | `ocr_ctc_48px.ckpt` |
| `mocr` | `ModelMangaOCR` | **ViT-based Transformer** (HuggingFace `manga-ocr`) | HuggingFace Hub |

**Model48pxOCR (`48px`) — primary model:**  
The recognized text regions are first cropped and normalized to a fixed height of **48 pixels**. Each crop is optionally padded to the maximum width in a batch. The model `OCR` uses a Rotary Position Embedding Transformer (`Roformer`) with XPos relative position encoding and local windowed attention (ViT-style). Inference uses **beam search** (`beams_k=5`, `max_seq_length=255`) against a large Unicode dictionary (`alphabet-all-v7.txt`) covering Japanese, Chinese, Korean, and other characters.

**ModelMangaOCR (`mocr`):**  
This wraps the [`manga-ocr`](https://github.com/kha-white/manga-ocr) library, which is a fine-tuned Vision Transformer (ViT) model from HuggingFace. It specializes in Japanese manga text and uses the `TrOCR`-style encode-decode architecture.

### 5.3 Inpainting Models

| Key | Class | Architecture | Checkpoint |
|---|---|---|---|
| `default` | `AotInpainter` | **AOT-GAN** | `inpainting.ckpt` |
| `lama_large` | `LamaLargeInpainter` | **LaMa (large)** with FFC + MPE | Large LaMa weights |
| `lama_mpe` | `LamaMPEInpainter` | **LaMa + Mask Pooling Embedding** | LaMa MPE weights |
| `sd` | `StableDiffusionInpainter` | **Stable Diffusion** inpainting | SD inpaint model |
| `none` | `NoneInpainter` | No inpainting | — |
| `original` | `OriginalInpainter` | Restore original pixels | — |

**AOT-GAN (default):**  
The `AOTGenerator` uses **Aggregated Contextual Transformations** — a GAN that uses multi-dilation convolutions to capture context at different scales. The generator processes the masked image and predicts pixel values for the masked regions. It uses `ScaledWSConv2d` (Weight-Standardized convolutions) throughout for training stability.

**LaMa (lama_mpe):**  
Large Mask Model using **Fast Fourier Convolutions (FFC)** — convolutions in the frequency domain that have a theoretically global receptive field. The `LamaMPEInpainter` extends LaMa with Mask Pooling Embedding, giving the model explicit information about the shape and location of the mask.

### 5.4 Upscaling Models

| Key | Class | Description |
|---|---|---|
| `waifu2x` | `Waifu2xUpscaler` | ML-based upscaler optimized for anime/manga art |
| `esrgan` | `ESRGANUpscaler` | Enhanced Super-Resolution GAN |
| `4xultrasharp` | `ESRGANUpscalerPytorch` | 4× UltraSharp ESRGAN (PyTorch) |

Upscaling is applied **before** detection. The default pipeline skips upscaling (`upscale_ratio = 0`), but it can be enabled to improve detection and OCR quality on low-resolution source images.

### 5.5 Colorization Model

| Key | Class | Architecture |
|---|---|---|
| `mc2` | `MangaColorizationV2` | **I2I GAN** + FFDNet denoiser |

**MangaColorizationV2:**  
An image-to-image GAN designed to colorize grayscale manga. The pipeline:
1. Optionally denoises the input using **FFDNet** (Fast and Flexible Denoising Network) with a configurable sigma.
2. Runs the GAN generator on the image at a size that is a multiple of 32 (max 576px for best quality).
3. An empty hint tensor (all zeros) is used, meaning the model colorizes autonomously without user input.

Colorization is applied **before** detection so the detector works on a color image.

### 5.6 Translation Model (Groq LLM)

| Key | Class | Backend |
|---|---|---|
| `groq` | `GroqTranslator` | Groq Cloud API (LLM inference) |

**GroqTranslator** is the only active translator in this build. It sends recognized text to the **Groq API** using an `AsyncGroq` client. The model used is specified by the `GROQ_MODEL` environment variable.

**System Prompt Design:**  
The prompt is specially engineered for East Asian comic translation:
- Identifies the source material as manga (Japanese), manhwa/webtoon (Korean), or manhua (Chinese).
- Preserves cultural terms: Japanese honorifics (`-chan`, `-kun`, `-sama`, `Senpai`), Korean honorifics (`Oppa`, `Hyung`, `Sunbae`), Chinese cultivation terms and titles.
- Retains onomatopoeia and SFX appropriate to each source language.
- Never translates proper names, technique names, or organization names unless an official English equivalent exists.
- Maintains natural, expressive dialogue style.
- Returns output strictly as JSON: `{"translated": "..."}`.

**Rate limiting:** Max 200 requests/minute. Retry attempts: 5. Timeout: 40s. Max tokens: 8192.

**Context retention:** Previous page translations can optionally be prepended to the prompt (via `CONTEXT_RETENTION=true` env var) so the LLM maintains character name and speech pattern consistency across pages.

---

## 6. Server APIs

### 6.1 VM Translator API

**File:** `server/translator_api.py`  
**Framework:** FastAPI  
**Storage:** Supabase Storage

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Service info |
| `GET` | `/health` | Health check (returns Supabase connection status) |
| `POST` | `/translate` | Translate image → upload to Supabase → return URL |
| `POST` | `/translate/raw` | Translate image → return raw PNG bytes |

**`POST /translate` flow:**
1. Accept `multipart/form-data` with `image` (file) and `config` (JSON string).
2. Parse the `Config` pydantic object from the JSON.
3. Pickle `{image, config}` and POST it to the internal manga_translator at `http://127.0.0.1:8001/simple_execute/translate`.
4. Unpickle the returned `Context` object and extract `ctx.result` (PIL Image).
5. Convert to PNG bytes and upload to Supabase Storage under `images/translated/{timestamp}.png`.
6. Return the public Supabase URL and storage path.

### 6.2 Database API

**Directory:** `database_api/`  
**Framework:** FastAPI  
**Backend:** Supabase (Auth + PostgreSQL + Storage)

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/users/register` | Register new user (via Supabase Auth) |
| `POST` | `/api/users/login` | Login → returns JWT token |
| `GET` | `/api/users/me` | Get current user profile |
| `GET` / `POST` | `/api/folders` | List or create folders |
| `GET` / `POST` | `/api/images` | List or save image metadata |
| `DELETE` | `/api/images/{id}` | Delete image record + Supabase Storage object |

Authentication is handled by **Supabase Auth** — the client receives a JWT on login and presents it in the `Authorization: Bearer` header for all subsequent requests. The Database API verifies this token using Supabase.

---

## 7. Storage

All translated images are stored in **Supabase Storage** (a Supabase-managed S3-compatible object store):

- **Bucket:** Configured via `SUPABASE_STORAGE_BUCKET` (default: `"images"`)
- **Path format:** `translated/{unix_timestamp_ms}.png`
- **Access:** Public URLs are generated and returned to the client.
- **Deletion:** The Database API handles deleting images from Supabase Storage when a user deletes an image record.

---

## 8. Configuration System

Translation behavior is fully controlled by a `Config` pydantic object (defined in `manga_translator/config.py`). The client sends this as a JSON string in the `config` form field.

Key configuration groups:

| Group | Key Config Fields | Description |
|---|---|---|
| `detector` | `detector`, `detection_size`, `text_threshold`, `box_threshold`, `unclip_ratio` | Which detection model and its parameters |
| `ocr` | `ocr`, `prob` | Which OCR model and minimum character probability |
| `translator` | `translator`, `target_lang` | Which translator and target language |
| `inpainter` | `inpainter`, `inpainting_size` | Which inpainting model and resolution |
| `render` | `renderer`, `alignment`, `direction`, `font_size`, `font_color`, `rtl` | Text rendering settings |
| `upscale` | `upscaler`, `upscale_ratio`, `revert_upscaling` | Optional pre-processing upscale |
| `colorizer` | `colorizer`, `colorization_size`, `denoise_sigma` | Optional colorization |

**Supported target languages:** English (`ENG`), Japanese (`JPN`), Korean (`KOR`), Simplified Chinese (`CHS`), Traditional Chinese (`CHT`), French (`FRA`), German (`DEU`), Spanish (`ESP`), Russian (`RUS`), and many more.

---

## 9. Key Libraries & Dependencies

| Library | Role |
|---|---|
| `torch` / `torchvision` | Neural network runtime (CPU build in production) |
| `opencv-python` (`cv2`) | Image pre/post-processing, contour detection, bilateral filter |
| `PIL` / `Pillow` | Image loading, saving, format conversion |
| `numpy` | Array operations throughout the pipeline |
| `einops` | Tensor dimension rearrangement in model inference |
| `shapely` | Geometric operations on text polygons (union, distance, area) |
| `networkx` | Graph-based textline merging algorithm |
| `groq` | Async client for Groq LLM API |
| `manga-ocr` | Pre-trained ViT OCR model for Japanese manga |
| `fastapi` + `uvicorn` | API server framework |
| `supabase` | Supabase client (Auth + Storage + PostgreSQL) |
| `aiohttp` | Async HTTP communication between API layers |
| `pydantic` | Configuration and schema validation |
| `omegaconf` | Config file management |
| `timm` | Pre-trained model backbones (used in some detectors) |
| `onnxruntime` | ONNX model inference (CTD CPU mode) |
| `kornia` | Differentiable geometry ops (used in inpainting) |
| `py3langid` | Automatic language identification |
| `langcodes` | Language code utilities |
| `freetype-py` | Font rendering |
| `pyclipper` | Polygon clipping (used in detection post-processing) |

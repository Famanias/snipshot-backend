# SnipShot Engine — Usage Guide

This guide explains how to use the new `snipshot_engine/` package in place of the old `manga_translator/` setup.

---

## What Changed

| | Old (`manga_translator`) | New (`snipshot_engine`) |
|---|---|---|
| **Architecture** | Two processes: `main.py` spawns a `manga_translator shared` backend on port 8001, then a FastAPI wrapper on port 8000 that pickles images over HTTP | Single process: one FastAPI server with the translator loaded in-process |
| **Entry point** | `python main.py` (orchestrator) | `uvicorn snipshot_engine.server:app` |
| **Module size** | ~150 files, 12+ model backends, colorization, upscaling, web UI, WebSocket streaming | ~33 files, 1 model per stage, no extras |
| **Config** | CLI `argparse` flags scattered across files | Single pydantic `Config` object |
| **Models** | Dynamically selects from many options at runtime | Fixed stack: DBNet → 48px OCR → Groq LLM → LaMa Large → FreeType |

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

Key packages the engine needs:
- `torch`, `torchvision` — neural net inference
- `fastapi`, `uvicorn`, `python-multipart` — API server
- `opencv-python`, `numpy`, `Pillow` — image processing
- `freetype-py` — text rendering
- `groq` — LLM translation API
- `networkx`, `shapely`, `pyclipper` — geometry/graph ops
- `einops` — tensor manipulation
- `pydantic` — configuration
- `python-dotenv` — env var loading
- `supabase` — storage uploads (optional)
- `pydensecrf` — mask refinement (optional, graceful fallback)
- `pyhyphen`, `langcodes` — text hyphenation (optional)

### 2. Set environment variables

Create a `.env` file in the project root:

```env
# Required — Groq API for translation
GROQ_API_KEY=gsk_your_key_here
GROQ_MODEL=meta-llama/llama-4-maverick-17b-128e-instruct

# Required for /translate endpoint (Supabase Storage upload)
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_KEY=your-service-role-key
SUPABASE_STORAGE_BUCKET=images

# Optional
PORT=8000
```

> **Note:** The `/translate/raw` endpoint works without Supabase — it returns raw PNG bytes directly.

### 3. Run the server

```bash
uvicorn snipshot_engine.server:app --host 0.0.0.0 --port 8001
```

Or with auto-reload during development:

```bash
uvicorn snipshot_engine.server:app --host 0.0.0.0 --port 8001 --reload
```

Models are loaded **lazily** on the first translation request (expect a delay on the first call).

---

## API Endpoints

### `GET /health`

Health check.

```json
{"ok": true, "service": "snipshot-engine", "storage": "connected"}
```

### `POST /translate`

Translate an image and upload the result to Supabase Storage.

**Request:** `multipart/form-data`
- `image` — image file (PNG, JPG, WebP)
- `config` — optional JSON string with config overrides (see below)

**Response:**
```json
{
  "success": true,
  "image_url": "https://your-project.supabase.co/storage/v1/object/public/images/translated/1234567890.png",
  "storage_path": "translated/1234567890.png"
}
```

### `POST /translate/raw`

Same as above, but returns the translated image as raw PNG bytes instead of uploading to Supabase. No Supabase credentials needed.

**Response:** `image/png` binary

### Example: cURL

```bash
# Upload to Supabase
curl -X POST http://localhost:8000/translate \
  -F "image=@manga_page.png" \
  -F 'config={"translator": {"target_lang": "ENG"}}'

# Get raw PNG back
curl -X POST http://localhost:8000/translate/raw \
  -F "image=@manga_page.png" \
  -o translated.png
```

### Example: Python

```python
import requests

# Raw PNG
with open("manga_page.png", "rb") as f:
    resp = requests.post(
        "http://localhost:8000/translate/raw",
        files={"image": f},
        data={"config": '{"translator": {"target_lang": "ENG"}}'},
    )

with open("translated.png", "wb") as out:
    out.write(resp.content)
```

---

## Config Options

Pass a JSON string as the `config` form field. All fields are optional — defaults are shown below.

```json
{
  "detector": {
    "detector": "default",
    "detection_size": 1536,
    "text_threshold": 0.5,
    "box_threshold": 0.7,
    "unclip_ratio": 2.3
  },
  "ocr": {
    "ocr": "48px",
    "min_text_length": 0,
    "ignore_bubble": 0
  },
  "translator": {
    "translator": "groq",
    "target_lang": "ENG"
  },
  "inpainter": {
    "inpainter": "lama_large",
    "inpainting_size": 2048,
    "inpainting_precision": "bf16"
  },
  "render": {
    "renderer": "default",
    "alignment": "auto",
    "direction": "auto",
    "font_size_offset": 0,
    "disable_font_border": false,
    "no_hyphenation": false,
    "uppercase": false,
    "font_color": null
  },
  "kernel_size": 3,
  "mask_dilation_offset": 30
}
```

### Notable config fields

| Field | Effect |
|-------|--------|
| `translator.target_lang` | Target language code: `"ENG"`, `"CHS"`, `"KOR"`, `"JPN"`, etc. |
| `inpainter.inpainter` | `"lama_large"` (neural inpainting) or `"none"` (skip inpainting) |
| `inpainter.inpainting_precision` | `"bf16"` (default, CUDA only), `"fp32"`, or `"fp16"` |
| `render.font_color` | Hex color like `"FF0000"` for foreground, or `"FF0000:FFFFFF"` for fg:bg |
| `render.direction` | `"auto"`, `"horizontal"`, or `"vertical"` |
| `detector.detection_size` | Longer dimension to resize input to before detection. Smaller = faster. |

---

## Using the Engine as a Library

You don't have to run the FastAPI server. You can use `SnipshotTranslator` directly in your own code:

```python
import asyncio
from PIL import Image
from snipshot_engine import SnipshotTranslator, Config

async def main():
    config = Config()
    config.translator.target_lang = "ENG"

    translator = SnipshotTranslator(config, device="cpu")
    await translator.load_models()

    img = Image.open("manga_page.png")
    result = await translator.translate(img)
    result.save("translated.png")

asyncio.run(main())
```

### With GPU (CUDA)

```python
translator = SnipshotTranslator(config, device="cuda")
```

Make sure you install the CUDA version of PyTorch:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

---

## Model Files

Models are auto-downloaded on first use to the `models/` directory at the project root.

| Stage | Checkpoint | Source |
|-------|-----------|--------|
| Detection | `models/detection/detect-20241225.ckpt` | HuggingFace `zyddnys/manga-image-translator` |
| OCR | `models/ocr/ocr_ar_48px.ckpt` + `alphabet-all-v7.txt` | HuggingFace `zyddnys/manga-image-translator` |
| Inpainting | `models/inpainting/lama_large_512px.ckpt` | HuggingFace `dreMaz/AnimeMangaInpainting` |
| Translation | *(no local model — Groq API)* | — |
| Rendering | *(no model — algorithmic FreeType)* | — |

---

## Migrating from the Old Setup

### Old `main.py` (two-process)

```bash
# OLD — don't use this anymore
python main.py
```

This spawned two processes: `manga_translator shared` on port 8001 and `server.translator_api:app` on port 8000, communicating via pickled HTTP.

### New (single process)

```bash
# NEW — use this
uvicorn snipshot_engine.server:app --host 0.0.0.0 --port 8000
```

### Old `render.yaml` start command

```yaml
# OLD
startCommand: python main.py
# or
startCommand: uvicorn server.translator_api:app --host 0.0.0.0 --port $PORT
```

```yaml
# NEW
startCommand: uvicorn snipshot_engine.server:app --host 0.0.0.0 --port $PORT
```

### Old `API_MODULE` env var

The old setup used `API_MODULE=server.translator_api:app` or `server.main:app`. This is no longer needed — the engine has one entry point: `snipshot_engine.server:app`.

### Old `BACKEND_PORT` / `BACKEND_URL` env vars

These controlled the internal pickle-over-HTTP link between the two processes. **No longer needed** — translation runs in-process.

---

## Folder Structure

```
snipshot_engine/
├── __init__.py              # Exports: Config, SnipshotTranslator
├── config.py                # All enums + pydantic config classes
├── translator.py            # Pipeline orchestrator
├── server.py                # FastAPI server
├── utils/                   # Shared utilities
│   ├── generic.py           # Context, BBox, Quadrilateral, image I/O
│   ├── generic2.py          # Character classification helpers
│   ├── textblock.py         # TextBlock data class
│   ├── inference.py         # ModelWrapper base class
│   ├── log.py, sort.py, bubble.py
│   └── __init__.py
├── detection/               # DBNet + ResNet-34
│   ├── detector.py
│   ├── dbnet_utils/
│   └── __init__.py
├── ocr/                     # 48px ConvNext + XPos ViT
│   ├── model_48px.py
│   ├── xpos_relative_position.py
│   └── __init__.py
├── textline_merge/          # Graph-based line merging
│   └── __init__.py
├── translation/             # Groq LLM API
│   └── __init__.py
├── mask_refinement/         # CRF mask refinement
│   └── __init__.py
├── inpainting/              # LaMa Large (FFC architecture)
│   ├── lama.py
│   └── __init__.py
└── rendering/               # FreeType text renderer
    ├── text_render.py
    └── __init__.py
```

---

## Pipeline Flow

```
Input Image (PIL)
    │
    ▼
 1. Detection  ─── DBNet + ResNet-34 ──→ list of text region quads + raw mask
    │
    ▼
 2. OCR  ────────── 48px ConvNext+ViT ──→ recognized text per region
    │
    ▼
 3. Textline Merge ─ NetworkX graph ────→ merged TextBlock list
    │
    ▼
 4. Translation ──── Groq API (Llama) ──→ translated text per block
    │
    ▼
 5. Mask Refinement ─ CRF + dilation ──→ refined text mask
    │
    ▼
 6. Inpainting ───── LaMa Large FFC ───→ clean background image
    │
    ▼
 7. Rendering ────── FreeType ─────────→ text composited onto image
    │
    ▼
Output Image (PIL)
```

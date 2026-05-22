# SnipShot Backend Architecture (Current)

This repository currently hosts the translator service built on `snipshot_engine`.

It no longer uses the old `manga_translator` two-process bridge in this codebase.

## 1. High-Level Architecture

```
[ Frontend (mobile/desktop/web) ]
              |
              | HTTP multipart/form-data
              v
[ SnipShot Translator API ]  (FastAPI, single process)
  - Entry: main.py -> uvicorn snipshot_engine.server:app
  - Default port: 8001
  - Endpoints: /translate, /translate/raw, /health
              |
              +--> [Optional Supabase Storage]
                      - Used by /translate for URL output
              |
              +--> [Local model checkpoints in models/]
                      - detection/
                      - ocr/
                      - inpainting/
```

Notes:
- `snipshot_engine` runs in-process, so there is no internal pickle-over-HTTP hop.
- If you use a separate auth/metadata backend, that service is external to this repository.

## 2. Runtime Flow

### A. `/translate/raw` (no storage required)

1. Client uploads image + optional JSON config.
2. API loads/uses singleton `SnipshotTranslator`.
3. Pipeline runs fully in-process.
4. API returns translated PNG bytes.

### B. `/translate` (storage upload)

1. Client uploads image + optional JSON config.
2. Pipeline runs fully in-process.
3. Output image is uploaded to Supabase Storage.
4. API returns JSON with `image_url` and `storage_path`.

## 3. Translation Pipeline (snipshot_engine)

Main orchestrator: `snipshot_engine/translator.py`

Stages:
1. Detection (DBNet)
2. OCR (48px model)
3. Textline merge
4. Translation (Groq API)
5. Mask refinement
6. Inpainting (LaMa)
7. Rendering translated text
8. Reassemble output image

## 4. API Surface

Implemented in `snipshot_engine/server.py`.

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Service info |
| GET | `/health` | Health and storage status |
| POST | `/translate/raw` | Return translated PNG bytes |
| POST | `/translate` | Upload translated image to Supabase and return URL |

## 5. Config and Environment

### Core runtime

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `HOST` | No | `0.0.0.0` | API bind host |
| `PORT` | No | `8001` | API port |
| `RELOAD` | No | `false` | Uvicorn auto-reload |
| `GROQ_API_KEY` | Yes | - | Translation API key |
| `GROQ_MODEL` | No | service default | Model selection |

### Storage for `/translate`

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `SUPABASE_URL` | Yes (for `/translate`) | - | Supabase project URL |
| `SUPABASE_SERVICE_KEY` | Yes (for `/translate`) | - | Service key |
| `SUPABASE_STORAGE_BUCKET` | No | `images` | Storage bucket |

Notes:
- `/translate/raw` works without Supabase.
- If Supabase is missing, `/translate` returns upload failure.

## 6. Local Development

### Start translator API

```bash
pip install -r requirements.txt
python main.py
```

Service will run at:
- `http://localhost:8001`

### Test locally

```bash
# Full local translate
python test.py --local --image sample_pages/test-image-medium.png

# Inpaint-only preview (no translated text rendering)
python test.py --local --inpaint-only --image sample_pages/test-image-medium.png
```

## 7. Request Examples

### `POST /translate/raw`

```bash
curl -X POST http://localhost:8001/translate/raw \
  -F "image=@sample_pages/test-image-medium.png" \
  -F 'config={"translator":{"target_lang":"ENG"}}' \
  --output translated.png
```

### `POST /translate`

```bash
curl -X POST http://localhost:8001/translate \
  -F "image=@sample_pages/test-image-medium.png" \
  -F 'config={"translator":{"target_lang":"ENG"}}'
```

Expected JSON (when storage is configured):

```json
{
  "success": true,
  "image_url": "https://...",
  "storage_path": "translated/1234567890000.png"
}
```

## 8. Repository Structure (Current)

```
snipshot-backend/
|- main.py
|- requirements.txt
|- test.py
|- sample_pages/
|- models/
|  |- detection/
|  |- ocr/
|  `- inpainting/
`- snipshot_engine/
   |- server.py
   |- translator.py
   |- config.py
   |- detection/
   |- ocr/
   |- translation/
   |- mask_refinement/
   |- inpainting/
   `- rendering/
```

## 9. Deprecated References

The following are legacy and should not be used for current architecture decisions in this repo:
- `manga_translator` runtime bridge architecture
- Internal translator subprocess assumptions
- `server/translator_api.py` paths from older layouts

Use `main.py` and `snipshot_engine/server.py` as the source of truth.

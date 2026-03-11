# Migration Plan: `manga_translator` → `snipshot_engine`

## Why

The `manga_translator` folder is a fork of [zyddnys/manga-image-translator](https://github.com/zyddnys/manga-image-translator) — a feature-rich CLI tool that supports **20+ translators**, **6 detectors**, **5 inpainters**, **4 OCR models**, **3 upscalers**, colorization, WebSocket mode, multi-file batch processing, GIMP/PSD/PDF export, and an entire embedded Latent Diffusion Model library for Stable Diffusion inpainting.

**SnipShot uses exactly one of each:**

| Stage | What we use | What `manga_translator` ships |
|---|---|---|
| Detection | `default` (DBNet + ResNet-34) | 6 detectors (default, dbconvnext, ctd, craft, paddle, none) |
| OCR | `48px` (Roformer + XPos ViT) | 5 models (32px, 48px, 48px_ctc, mocr, large) |
| Translation | `groq` (Groq LLM API) | 20+ backends (ChatGPT, DeepL, Baidu, Papago, Sakura, NLLB, Sugoi, M2M100, etc.) |
| Inpainting | `default` / AOT-GAN | 6 inpainters (AOT, LaMa, LaMa-MPE, SD, none, original) + full LDM library |
| Rendering | `default` | default + manga2eng + gimp export |
| Upscaling | **Not used** | Waifu2x, ESRGAN, 4xUltraSharp |
| Colorization | **Not used** | MangaColorization v2 GAN |
| Modes | `shared` (HTTP server) | shared + local CLI batch + WebSocket protobuf |

This means we carry a large amount of dead code, unused model classes, unused dependencies (e.g., `manga-ocr`, DeepSeek tokenizers, protobuf for WebSocket), and the entire `ldm/` Stable Diffusion library — all adding to install size, complexity, and startup import time.

---

## Goal

Create a new `snipshot_engine/` folder that contains **only what SnipShot needs** — the same pipeline logic, same model weights, same output quality — but with a clean, minimal codebase we fully own and understand.

---

## What We Actually Need (from `test_translate.py`)

```python
config = {
    "detector":   { "detector": "default", "detection_size": 1536, "box_threshold": 0.7, "unclip_ratio": 2.3 },
    "translator":  { "translator": "groq", "target_lang": "ENG" },
    "inpainter":  { "inpainter": "default", "inpainting_size": 2048 },
    "render":     { "direction": "auto" },
    "mask_dilation_offset": 30
}
```

### Model weights (3 files, already in `models/`)

| File | Size | Used by |
|---|---|---|
| `models/detection/detect-20241225.ckpt` | ~85 MB | DefaultDetector (DBNet + ResNet-34) |
| `models/inpainting/inpainting.ckpt` | ~15 MB | AotInpainter (AOT-GAN) |
| `models/ocr/ocr_ar_48px.ckpt` | ~120 MB | Model48pxOCR (Roformer + XPos) |
| `models/ocr/alphabet-all-v7.txt` | ~200 KB | OCR character dictionary |

---

## Proposed Folder Structure

```
snipshot_engine/
│
├── __init__.py              # Exports: SnipshotTranslator, Config
├── config.py                # Minimal Config (only detector/ocr/translator/inpainter/render settings)
├── translator.py            # Main pipeline class (equivalent to MangaTranslator._translate)
├── server.py                # FastAPI shared-mode server (simplified from mode/share.py)
│
├── detection/
│   ├── __init__.py          # Single dispatch function
│   ├── detector.py          # DefaultDetector (DBNet + ResNet-34) — from default.py
│   └── dbnet_utils/         # Minimal utils from default_utils/
│       ├── __init__.py
│       ├── model.py         # DBNet_resnet34 TextDetection module
│       ├── imgproc.py       # Image preprocessing (resize_aspect_ratio, etc.)
│       └── postprocess.py   # SegDetectorRepresenter, adjustResultCoordinates
│
├── ocr/
│   ├── __init__.py          # Single dispatch function
│   ├── model_48px.py        # Roformer + XPos OCR model
│   └── xpos.py              # XPos relative position encoding
│
├── inpainting/
│   ├── __init__.py          # Single dispatch function
│   └── aot.py               # AOT-GAN generator + LaMa-MPE base inpaint logic
│
├── translation/
│   ├── __init__.py          # Single dispatch function
│   └── groq.py              # GroqTranslator (Groq LLM API)
│
├── rendering/
│   ├── __init__.py          # Dispatch + font sizing + region adjustment
│   ├── text_render.py       # CJK text rendering
│   └── text_render_eng.py   # English text rendering
│
├── textline_merge/
│   ├── __init__.py          # Graph-based textline merge algorithm
│
├── mask_refinement/
│   ├── __init__.py          # Mask refinement dispatch
│   └── text_mask_utils.py   # Morphological mask refinement
│
└── utils/
    ├── __init__.py           # Re-exports
    ├── textblock.py          # TextBlock, Quadrilateral classes
    ├── inference.py          # det_rearrange_forward, ModelWrapper, model download
    ├── generic.py            # load_image, dump_image, get_image_md5, etc.
    ├── sort.py               # sort_regions, reading-order sorting
    ├── bubble.py             # is_ignore, bubble detection
    └── log.py                # Logger setup
```

---

## What Gets Removed

### Entire modules dropped

| Module | Reason |
|---|---|
| `colorization/` (+ `manga_colorization_v2_utils/`) | Not used — colorizer is always `none` |
| `upscaling/` | Not used — no upscale_ratio set |
| `inpainting/ldm/` | Stable Diffusion library — only for SD inpainter |
| `inpainting/inpainting_sd.py`, `sd_hack.py`, `booru_tagger.py` | SD inpainting support |
| `inpainting/inpainting_lama.py`, `inpainting_attn.py` | Unused inpainter variants |
| `inpainting/guided_ldm_*.yaml` | LDM configs |
| `detection/craft.py` + `craft_utils/` | CRAFT detector |
| `detection/ctd.py` + `ctd_utils/` | Comic Text Detector (YOLOv5) |
| `detection/dbnet_convnext.py` | DBConvNext variant |
| `detection/paddle_rust.py`, `common_rust.py` | PaddleOCR Rust detector |
| `ocr/model_32px.py` | Older OCR model |
| `ocr/model_48px_ctc.py` | CTC variant |
| `ocr/model_manga_ocr.py` | HuggingFace manga-ocr (separate model) |
| `ocr/model_ocr_large.py` | Large OCR model |
| `translators/tokenizers/` | DeepSeek tokenizers |
| `mode/local.py` | CLI batch mode |
| `mode/ws.py` | WebSocket protobuf mode |
| `rendering/gimp_render.py` | GIMP/PSD/PDF export |
| `rendering/ballon_extractor.py` | Unused balloon extraction |
| `save.py` | Output format registry (GIMP, etc.) |
| `args.py` | CLI argument parser (200+ args) — replaced by Config-only approach |

### Dependencies we can potentially drop

| Package | Was needed by |
|---|---|
| `manga-ocr` | `model_manga_ocr.py` (HuggingFace model) |
| `protobuf` | WebSocket `ws_pb2` mode |
| `deepl` | DeepL translator |
| `openai` | ChatGPT translator |
| `tiktoken` | ChatGPT token counting |
| `rusty-manga-image-translator` | Rust-based detection/paddle |
| `safetensors` | SD inpainting model loading |
| `pandas` | Used somewhere in LDM |

---

## Migration Strategy: File-by-File Mapping

| `manga_translator/` source | → `snipshot_engine/` target | Changes |
|---|---|---|
| `manga_translator.py` (class `MangaTranslator`) | `translator.py` (class `SnipshotTranslator`) | Remove colorization/upscaling steps, remove batch logic, remove multi-page context complexity, remove verbose debug image saving, simplify to single-image pipeline |
| `config.py` | `config.py` | Keep only: `Detector.default`, `Ocr.ocr48px`, `Translator.groq`, `Inpainter.default`, `Renderer.default/manga2Eng`. Remove 15+ unused enum values. Simplify config models. |
| `mode/share.py` | `server.py` | Keep `/simple_execute/translate` endpoint. Drop streaming `/execute`, drop `/is_locked`, drop WebSocket, drop nonce auth. |
| `detection/default.py` | `detection/detector.py` | Direct copy, remove cache/registry overhead — just one detector. |
| `detection/default_utils/*` | `detection/dbnet_utils/*` | Copy `DBNet_resnet34.py`, `imgproc.py`, `dbnet_utils.py`. Take `adjustResultCoordinates` from `craft_utils.py`. |
| `ocr/model_48px.py` | `ocr/model_48px.py` | Direct copy, clean up verbose debug image saving. |
| `ocr/xpos_relative_position.py` | `ocr/xpos.py` | Direct copy (needed by model_48px). |
| `translators/groq.py` | `translation/groq.py` | Direct copy, inline the key loading (remove `keys.py`, just use `os.getenv`). |
| `translators/common.py` | `translation/__init__.py` | Extract only `CommonTranslator` base + `VALID_LANGUAGES` + `ISO_639_1_TO_VALID_LANGUAGES`. |
| `inpainting/inpainting_aot.py` | `inpainting/aot.py` | Copy AOTGenerator. |
| `inpainting/inpainting_lama_mpe.py` | `inpainting/aot.py` (merged) | AOT inherits from LamaMPE's `_infer` method for the shared `inpaint()` logic. Merge the needed base inpaint method into one file. |
| `inpainting/common.py` | `inpainting/__init__.py` | Extract `OfflineInpainter` base class. |
| `inpainting/none.py` | `inpainting/__init__.py` | Keep `NoneInpainter` (used for debug `inpaint_input.png`). |
| `rendering/__init__.py` | `rendering/__init__.py` | Copy dispatch logic, font sizing, region adjustment. |
| `rendering/text_render.py` | `rendering/text_render.py` | Direct copy. |
| `rendering/text_render_eng.py` | `rendering/text_render_eng.py` | Direct copy. |
| `rendering/text_render_pillow_eng.py` | (Drop or keep) | Only needed if `manga2eng_pillow` renderer is used. |
| `textline_merge/` | `textline_merge/` | Direct copy (pure algorithm, no models). |
| `mask_refinement/` | `mask_refinement/` | Direct copy (pure algorithm, no models). |
| `utils/*` | `utils/*` | Copy needed files: `textblock.py`, `inference.py`, `generic.py`, `sort.py`, `bubble.py`, `log.py`. Drop `threading.py` if only used by batch mode. |

---

## Pipeline Comparison

### Current (`manga_translator`)

```
translate(image, config)
  → colorization?     (skipped)
  → upscaling?        (skipped)
  → load_image()
  → detection          (DefaultDetector)
  → OCR                (Model48pxOCR)
  → textline_merge     (graph algorithm)
  → pre-dictionary     (regex substitution)
  → translation        (GroqTranslator)
  → post-dictionary    (regex substitution)
  → mask_refinement    (morphological)
  → inpainting         (AotInpainter)
  → rendering          (default renderer)
  → dump_image()
```

### New (`snipshot_engine`)

```
translate(image, config)
  → load_image()
  → detection          (DBNet + ResNet-34)
  → OCR                (48px Roformer)
  → textline_merge     (graph algorithm)
  → translation        (Groq LLM)
  → mask_refinement    (morphological)
  → inpainting         (AOT-GAN)
  → rendering          (default renderer)
  → dump_image()
```

Same effective pipeline — minus the optional colorization/upscaling steps and pre/post dictionary substitution that aren't being used.

---

## Integration Plan

### Phase 1: Create `snipshot_engine/` alongside `manga_translator/`
- Build the new module file by file.
- Both folders coexist — nothing breaks.

### Phase 2: Update server to use `snipshot_engine`
- Change `main.py` to launch `snipshot_engine.server` instead of `manga_translator shared`.
- Change `server/translator_api.py` to import `Config` from `snipshot_engine`.
- Run tests against the new engine.

### Phase 3: Remove `manga_translator/`
- Once all tests pass, delete the entire `manga_translator/` folder.
- Clean up `requirements.txt` to drop unused dependencies.

### Phase 4: Trim `requirements.txt`
- Remove: `manga-ocr`, `protobuf`, `deepl`, `openai`, `tiktoken`, `rusty-manga-image-translator`, `safetensors`, `pandas`, `cython`.
- Keep everything else (torch, torchvision, cv2, numpy, einops, shapely, networkx, groq, fastapi, etc.).

---

## Risk & Compatibility Notes

1. **Model weights stay the same.** The `.ckpt` files in `models/` don't change. Same neural network architectures loading the same checkpoints = identical output quality.

2. **The `Config` JSON format from the frontend stays the same.** We'll keep the same field names so no frontend changes are needed.

3. **The `/translate` and `/translate/raw` API endpoints stay the same.** The internal `/simple_execute/translate` pickle protocol changes (our new server can use a cleaner interface), but the public API contract is unchanged.

4. **Rollback is easy.** Since both folders coexist during migration, we can switch back to `manga_translator` at any time by reverting `main.py`.

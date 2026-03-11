# snipshot_engine Migration Progress

## Overview
Migrating from the bloated `manga_translator/` fork to a lean `snipshot_engine/` folder
containing only the models and code actually used by SnipShot.

**Chosen stack:**
- Detection: DBNet + ResNet-34 (`detect-20241225.ckpt`)
- OCR: Roformer + XPos ViT 48px (`ocr_ar_48px.ckpt` + `alphabet-all-v7.txt`)
- Translation: Groq LLM API (`meta-llama/llama-4-maverick-17b-128e-instruct`)
- Inpainting: LaMa Large (`lama_large_512px.ckpt`)
- Rendering: Default FreeType + OpenCV (no neural net)
- Server: Single-process in-process FastAPI (no subprocess pickle relay)

---

## ✅ Completed

### `snipshot_engine/` root
| File | Description |
|------|-------------|
| `__init__.py` | Package entry — exports `Config`, `SnipshotTranslator` |
| `config.py` | All enums (`Detector`, `Ocr`, `Translator`, `Inpainter`, `Renderer`) and pydantic sub-configs (`DetectorConfig`, `OcrConfig`, `TranslatorConfig`, `InpainterConfig`, `RenderConfig`, `Config`) |

### `snipshot_engine/utils/`
| File | Description |
|------|-------------|
| `__init__.py` | Re-exports everything from all util submodules |
| `generic.py` | `Context`, `BBox`, `Quadrilateral`, `load_image`, `dump_image`, `resize_keep_aspect`, `quadrilateral_can_merge_region`, `download_url_with_progressbar`, `get_digest`, `BASE_PATH` |
| `generic2.py` | `color_difference`, `is_punctuation`, `is_whitespace`, `is_control`, `is_valuable_char`, `is_valuable_text`, `is_right_to_left_char`, `dist`, `rect_distance` |
| `textblock.py` | `TextBlock` class (full 480-line port) with all cached properties, `LANGUAGE_ORIENTATION_PRESETS`, `rotate_polygons` |
| `inference.py` | `ModelWrapper` ABC — download / verify / load / unload / infer lifecycle |
| `log.py` | `get_logger(name)` — returns `logging.getLogger("snipshot_engine.<name>")` |
| `sort.py` | `sort_regions()` — panel-aware + simple fallback sorting |
| `bubble.py` | `is_ignore()` — bubble edge pixel ratio check |

### `snipshot_engine/detection/`
| File | Description |
|------|-------------|
| `__init__.py` | `prepare()` downloads model, `dispatch()` runs detection, exports `TextDetector` |
| `detector.py` | `DefaultDetector` — DBNet + ResNet-34 inference, `SegDetectorRepresenter` post-processing |
| `dbnet_utils/` | Subpackage with model architecture files (DBNet_resnet34.py, DBHead.py, imgproc.py, dbnet_utils.py, craft_utils.py) |
| `default_utils/` | Duplicate of source files (same as dbnet_utils), kept for import compatibility |

### `snipshot_engine/ocr/`
| File | Description |
|------|-------------|
| `__init__.py` | `prepare()`, `dispatch()`, `unload()` — maps `Ocr.ocr48px` → `Model48pxOCR` |
| `model_48px.py` | Full OCR model: ConvNext feature extractor, XposMultiheadAttention transformer decoder, beam search (k=5, max_seq=255), `Model48pxOCR` wrapper with `_MODEL_MAPPING` for `ocr_ar_48px.ckpt` + `alphabet-all-v7.txt` |
| `xpos_relative_position.py` | XPOS positional encoding class |

### `snipshot_engine/textline_merge/`
| File | Description |
|------|-------------|
| `__init__.py` | Graph-based merge with `_split_text_region` (MST splitting), `_merge_bboxes_text_region` (NetworkX connected components), `dispatch()` → `List[TextBlock]` |

### `snipshot_engine/translation/`
| File | Description |
|------|-------------|
| `__init__.py` | Single file: `CommonTranslator` base (rate-limit, translate, _clean), `GroqTranslator` (Groq async API, manga system prompt), module-level `prepare()`, `dispatch()`, `get_translator()` |

### `snipshot_engine/mask_refinement/`
| File | Description |
|------|-------------|
| `__init__.py` | CRF mask refinement (`pydensecrf` with `HAS_CRF` fallback), `_complete_mask` (connected components, polygon overlap, bilateral filter, dilation), `dispatch()` with scale/bubble filtering |

### `snipshot_engine/inpainting/`
| File | Description |
|------|-------------|
| `__init__.py` | `prepare()`, `dispatch()`, `unload()` — maps `Inpainter.lama_large` → `LamaLargeInpainter` |
| `lama.py` | Full FFC architecture: `FourierUnit`, `SpectralTransform`, `FFCSE_block`, `FFC`, `FFC_BN_ACT`, `FFCResnetBlock`, `ConcatTupleLayer`, `FFCResNetGenerator` (18 blocks for large arch), `LamaFourier` wrapper, `LamaLargeInpainter` (ModelWrapper-based, resizes/pads/normalizes, autocast bf16 on CUDA) |

### `snipshot_engine/rendering/`
| File | Description |
|------|-------------|
| `__init__.py` | `dispatch()` entry point, `_render_region()` per text block (homography warp, aspect-ratio padding, alpha compositing) |
| `text_render.py` | Core FreeType rendering: font management, CJK H↔V mapping, `put_text_horizontal()`, `put_text_vertical()`, `calc_horizontal()` with hyphenation, `put_char_horizontal/vertical()`, `add_color()` |

### `snipshot_engine/translator.py`
Simplified pipeline orchestrator (~160 lines). `SnipshotTranslator` class with:
- `load_models()` — pre-download all checkpoints
- `translate(image)` → PIL Image: detect → OCR → merge → translate → mask_refine → inpaint → render

### `snipshot_engine/server.py`
Single-process in-process FastAPI v3.0.0. Replaces two-process pickle-over-HTTP:
- `POST /translate` — multipart image + JSON config → Supabase Storage URL
- `POST /translate/raw` — multipart image + JSON config → raw PNG bytes
- `GET /health` — health check with Supabase status

---

## ✅ Migration Complete

All modules have been ported. The `snipshot_engine/` folder is fully self-contained.

### File Summary

```
snipshot_engine/
├── __init__.py              # Package entry — exports Config, SnipshotTranslator
├── config.py                # Enums + pydantic configs
├── translator.py            # Pipeline orchestrator
├── server.py                # FastAPI server
├── utils/
│   ├── __init__.py          # Re-exports
│   ├── generic.py           # Context, BBox, Quadrilateral, load_image, dump_image
│   ├── generic2.py          # color_difference, is_punctuation, etc.
│   ├── textblock.py         # TextBlock class (480 lines)
│   ├── inference.py         # ModelWrapper ABC
│   ├── log.py               # get_logger()
│   ├── sort.py              # sort_regions()
│   └── bubble.py            # is_ignore()
├── detection/
│   ├── __init__.py
│   ├── detector.py          # DefaultDetector (DBNet + ResNet-34)
│   ├── dbnet_utils/         # Model architecture files
│   └── default_utils/       # Import compat duplicate
├── ocr/
│   ├── __init__.py
│   ├── model_48px.py        # ConvNext + XPos ViT OCR
│   └── xpos_relative_position.py
├── textline_merge/
│   └── __init__.py          # Graph-based merge
├── translation/
│   └── __init__.py          # Groq LLM translator
├── mask_refinement/
│   └── __init__.py          # CRF mask refinement
├── inpainting/
│   ├── __init__.py
│   └── lama.py              # LaMa Large FFC inpainter
└── rendering/
    ├── __init__.py           # Dispatch + region renderer
    └── text_render.py        # FreeType text rendering
```

### Next Steps
1. **Test imports** — `python -c "from snipshot_engine import SnipshotTranslator, Config"`
2. **Integration test** — Run `server.py` with `uvicorn snipshot_engine.server:app`
3. **Cleanup** — Remove `default_utils/` duplicate under detection, point imports to `dbnet_utils/`
4. **Requirements** — Create `snipshot_engine/requirements.txt` with exact pinned deps

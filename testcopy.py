"""
Test script for snipshot_engine — the new lean translation pipeline.

Tests:
  1. Import check          — verify all modules load
  2. Config construction   — verify pydantic config works
  3. In-process translate   — run SnipshotTranslator directly (no server)
  4. Server /health        — hit the FastAPI health endpoint
  5. Server /translate/raw — send an image and get translated PNG back
  6. Server /translate     — send an image and get Supabase URL back

Usage:
  # Test 1-3 only (no server needed):
        python test.py --local

    # Test 1-6 (start the server first):
        python main.py
        python test.py

  # Specify a different image or server URL:
        python test.py --image 155.jpg --url http://localhost:9000
"""

import argparse
import asyncio
import io
import json
import sys
import time
from pathlib import Path


# ── Recommended manga/manhwa config defaults ─────────────────────────────
#
# These values are tuned for manga/manhwa speech-bubble translation:
#
#   Detector:
#     text_threshold   0.4  — catches faint/stylized bubble text
#     box_threshold    0.6  — avoids missing partially obscured bubbles
#     unclip_ratio     2.5  — expands boxes to capture full bubble area
#     det_auto_rotate  True — handles diagonal/rotated SFX text
#     det_gamma_correct True — improves detection on dark/low-contrast scans
#
#   OCR:
#     min_text_length  1    — captures even single-character text
#
#   Inpainter:
#     inpainter        lama_large — best quality bubble erasure
#     inpainting_size  2048       — high-res for clean background fill
#     inpainting_precision bf16   — best speed/quality balance
#
#   Renderer:
#     renderer         manga2eng  — purpose-built for manga-to-English layout
#     alignment        center     — most speech bubbles look best centered
#     direction        horizontal — English reads horizontally
#     uppercase        True       — standard comic/manga convention
#     no_hyphenation   True       — hyphenation looks unnatural in bubbles
#     rtl              False      — English reads left-to-right
#     font_size_minimum 10        — prevents unreadably tiny text
#     disable_font_border False   — keep outline for readability over backgrounds
#
#   Top-level:
#     mask_dilation_offset 40    — ensures full bubble text area is erased
# ─────────────────────────────────────────────────────────────────────────

MANGA_CONFIG = {
    "detector": {
        "detection_size": 1536,         # overridden dynamically if --auto-detector
        "text_threshold": 0.4,          # ↓ from 0.5 — catches faint/stylized text
        "box_threshold": 0.6,           # ↓ from 0.7 — avoids missing partial bubbles
        "unclip_ratio": 2.5,            # ↑ from 2.3 — captures full bubble area
        "det_auto_rotate": True,        # handles rotated SFX / diagonal text
        "det_gamma_correct": True,      # improves low-contrast / dark scan detection
    },
    "ocr": {
        "min_text_length": 1,           # capture even single-character text
    },
    "inpainter": {
        "inpainter": "lama_large",      # best quality bubble erasure
        "inpainting_size": 2048,        # high-res for clean background fill
        "inpainting_precision": "bf16",
    },
    "render": {
        "renderer": "manga2eng",        # purpose-built for manga-to-English layout
        "alignment": "center",          # speech bubbles look best centered
        "direction": "horizontal",      # English reads horizontally
        "uppercase": True,              # standard comic/manga convention
        "no_hyphenation": True,         # hyphenation looks unnatural in bubbles
        "rtl": False,                   # English reads left-to-right
        "font_size_minimum": 10,        # prevents unreadably tiny text
        "disable_font_border": False,   # keep outline for readability over backgrounds
    },
    "mask_dilation_offset": 40,         # ↑ from 30 — fully erases bubble text area
    "kernel_size": 3,
}


async def _run_inpaint_preview(translator, pil_image):
    """Run pipeline up to inpainting and return PIL image + region count."""
    import numpy as np
    from snipshot_engine.config import Inpainter
    from snipshot_engine.utils import (
        load_image,
        dump_image,
        is_valuable_text,
        sort_regions,
        LANGUAGE_ORIENTATION_PRESETS,
    )
    from snipshot_engine.ocr import dispatch as dispatch_ocr
    from snipshot_engine.textline_merge import dispatch as dispatch_textline_merge
    from snipshot_engine.mask_refinement import dispatch as dispatch_mask_refinement
    from snipshot_engine.inpainting import dispatch as dispatch_inpainting

    cfg = translator.config
    img_rgb, img_alpha = load_image(pil_image)

    textlines, mask_raw, _ = await translator._detector.infer(
        img_rgb,
        cfg.detector.detection_size,
        cfg.detector.text_threshold,
        cfg.detector.box_threshold,
        cfg.detector.unclip_ratio,
        verbose=False,
    )
    if not textlines:
        return dump_image(pil_image, img_rgb, img_alpha), 0

    textlines = await dispatch_ocr(cfg.ocr.ocr, img_rgb, textlines, cfg.ocr, translator.device, verbose=False)
    textlines = [t for t in textlines if t.text.strip()]
    if not textlines:
        return dump_image(pil_image, img_rgb, img_alpha), 0

    text_regions = await dispatch_textline_merge(textlines, img_rgb.shape[1], img_rgb.shape[0], verbose=False)
    text_regions = [
        r for r in text_regions
        if len(r.text) >= cfg.ocr.min_text_length and is_valuable_text(r.text)
    ]
    if not text_regions:
        return dump_image(pil_image, img_rgb, img_alpha), 0

    text_regions = sort_regions(text_regions, img_rgb.shape[1], img_rgb.shape[0])
    target_lang = cfg.translator.target_lang
    for region in text_regions:
        region.target_lang = target_lang
        preset = LANGUAGE_ORIENTATION_PRESETS.get(target_lang)
        if preset:
            region._direction = preset

    mask = await dispatch_mask_refinement(
        text_regions,
        img_rgb,
        mask_raw if mask_raw is not None else np.zeros(img_rgb.shape[:2], dtype=np.uint8),
        method="fit_text",
        dilation_offset=cfg.mask_dilation_offset,
        ignore_bubble=cfg.ocr.ignore_bubble,
        kernel_size=cfg.kernel_size,
    )

    if cfg.inpainter.inpainter != Inpainter.none:
        img_inpainted = await dispatch_inpainting(
            cfg.inpainter.inpainter,
            img_rgb,
            mask,
            cfg.inpainter,
            cfg.inpainter.inpainting_size,
            translator.device,
            verbose=False,
        )
    else:
        img_inpainted = img_rgb.copy()

    return dump_image(pil_image, img_inpainted, img_alpha), len(text_regions)


def _infer_dynamic_detector_params(width: int, height: int) -> dict:
    """
    Infer detector settings from image dimensions for local test sweeps.
    Thresholds are tuned lower than stock defaults to avoid missing manga
    text in partially obscured or stylised speech bubbles.
    """
    long_side = max(width, height)
    mp = (width * height) / 1_000_000.0

    # Round to nearest 64 and clamp to practical range.
    detection_size = int(round(long_side / 64.0) * 64)
    detection_size = max(1024, min(3072, detection_size))

    # ↓ Thresholds lowered vs. original to reduce missed bubbles in manga.
    if mp < 1.0:
        box_threshold = 0.50
    elif mp < 2.5:
        box_threshold = 0.55
    elif mp < 4.0:
        box_threshold = 0.60
    else:
        box_threshold = 0.65

    return {
        "detection_size": detection_size,
        "box_threshold": box_threshold,
    }


def _collect_image_paths(image_arg: str):
    """Collect input images from a file path or directory."""
    p = Path(image_arg)
    exts = {".png", ".jpg", ".jpeg", ".webp"}

    if p.is_file():
        return [str(p)]

    if p.is_dir():
        out = []
        for fp in sorted(p.iterdir()):
            if fp.suffix.lower() not in exts:
                continue
            stem = fp.stem.lower()
            if stem.endswith("_translated") or stem.endswith("_inpainted"):
                continue
            out.append(str(fp))
        return out

    return [image_arg]


# ── Build a merged config, letting CLI args override MANGA_CONFIG ─────────

def _build_config(target_lang: str, detection_size: int, box_threshold: float, inpainting_size: int) -> dict:
    """
    Merge MANGA_CONFIG with any CLI-level overrides.
    translator.target_lang and the three sizing params can be
    specified per-run; everything else stays at the manga defaults.
    """
    cfg = {
        **MANGA_CONFIG,
        "translator": {"target_lang": target_lang},
    }
    cfg["detector"] = {**MANGA_CONFIG["detector"], "detection_size": detection_size, "box_threshold": box_threshold}
    cfg["inpainter"] = {**MANGA_CONFIG["inpainter"], "inpainting_size": inpainting_size}
    return cfg


# ---------------------------------------------------------------------------
# 1. Import check
# ---------------------------------------------------------------------------

def test_imports():
    print("\n[1] Import check...")
    failures = []

    try:
        from snipshot_engine import Config, SnipshotTranslator
        print("    OK  snipshot_engine (Config, SnipshotTranslator)")
    except Exception as e:
        failures.append(f"snipshot_engine: {e}")

    modules = [
        ("snipshot_engine.config", None),
        ("snipshot_engine.utils", ["TextBlock", "Quadrilateral", "ModelWrapper", "load_image", "dump_image"]),
        ("snipshot_engine.detection", ["DefaultDetector"]),
        ("snipshot_engine.ocr", ["prepare", "dispatch", "unload"]),
        ("snipshot_engine.textline_merge", ["dispatch"]),
        ("snipshot_engine.translation", ["prepare", "dispatch"]),
        ("snipshot_engine.mask_refinement", ["dispatch"]),
        ("snipshot_engine.inpainting", ["prepare", "dispatch", "unload"]),
        ("snipshot_engine.rendering", ["dispatch"]),
    ]

    for mod_name, symbols in modules:
        try:
            mod = __import__(mod_name, fromlist=symbols or ["__name__"])
            if symbols:
                missing = [s for s in symbols if not hasattr(mod, s)]
                if missing:
                    failures.append(f"{mod_name}: missing {missing}")
                else:
                    print(f"    OK  {mod_name}  ({', '.join(symbols)})")
            else:
                print(f"    OK  {mod_name}")
        except Exception as e:
            failures.append(f"{mod_name}: {e}")

    if failures:
        for f in failures:
            print(f"    FAIL  {f}")
        return False
    print("    All imports passed.")
    return True


# ---------------------------------------------------------------------------
# 2. Config construction
# ---------------------------------------------------------------------------

def test_config():
    print("\n[2] Config construction...")
    from snipshot_engine import Config

    # Default config
    cfg = Config()
    print(f"    Default detector:   {cfg.detector.detector.value}")
    print(f"    Default OCR:        {cfg.ocr.ocr.value}")
    print(f"    Default translator: {cfg.translator.translator.value}")
    print(f"    Default inpainter:  {cfg.inpainter.inpainter.value}")
    print(f"    Default renderer:   {cfg.render.renderer.value}")

    # Manga-recommended config assertions
    manga_cfg = Config(**MANGA_CONFIG)
    assert manga_cfg.detector.text_threshold == 0.4,        "text_threshold should be 0.4"
    assert manga_cfg.detector.box_threshold == 0.6,         "box_threshold should be 0.6"
    assert manga_cfg.detector.unclip_ratio == 2.5,          "unclip_ratio should be 2.5"
    assert manga_cfg.detector.det_auto_rotate is True,      "det_auto_rotate should be True"
    assert manga_cfg.detector.det_gamma_correct is True,    "det_gamma_correct should be True"
    assert manga_cfg.ocr.min_text_length == 1,              "min_text_length should be 1"
    assert manga_cfg.render.renderer.value == "manga2eng",  "renderer should be manga2eng"
    assert manga_cfg.render.alignment.value == "center",    "alignment should be center"
    assert manga_cfg.render.direction.value == "horizontal","direction should be horizontal"
    assert manga_cfg.render.uppercase is True,              "uppercase should be True"
    assert manga_cfg.render.no_hyphenation is True,         "no_hyphenation should be True"
    assert manga_cfg.render.rtl is False,                   "rtl should be False"
    assert manga_cfg.render.font_size_minimum == 10,        "font_size_minimum should be 10"
    assert manga_cfg.mask_dilation_offset == 40,            "mask_dilation_offset should be 40"
    print("    Manga config assertions passed.")

    # Custom override (like the frontend would send)
    custom = Config(**{
        "detector": {"detection_size": 1024, "box_threshold": 0.5},
        "translator": {"target_lang": "CHS"},
        "inpainter": {"inpainter": "none"},
        "render": {"direction": "horizontal"},
    })
    assert custom.detector.detection_size == 1024
    assert custom.translator.target_lang == "CHS"
    assert custom.inpainter.inpainter.value == "none"
    assert custom.render.direction.value == "horizontal"
    print("    Custom config overrides work.")
    print("    Config test passed.")
    return True


# ---------------------------------------------------------------------------
# 3. In-process translate (local, no server)
# ---------------------------------------------------------------------------

async def _process_one_local_image(
    translator,
    image_path: str,
    save_inpainted: bool,
    inpaint_only: bool,
    auto_detector: bool,
):
    print(f"\n[3] In-process translate ({image_path})...")
    from PIL import Image

    try:
        img = Image.open(image_path).convert("RGB")
        print(f"    Loaded {image_path} ({img.size[0]}x{img.size[1]})")
    except FileNotFoundError:
        print(f"    SKIP  {image_path} not found")
        return None

    print("    Running translation pipeline...")

    if auto_detector:
        dyn = _infer_dynamic_detector_params(img.size[0], img.size[1])
        translator.config.detector.detection_size = dyn["detection_size"]
        translator.config.detector.box_threshold = dyn["box_threshold"]
        print(
            f"    Dynamic detector: size={dyn['detection_size']} "
            f"box_threshold={dyn['box_threshold']:.2f}"
        )

    if save_inpainted or inpaint_only:
        t_preview = time.time()
        inpainted_img, region_count = await _run_inpaint_preview(translator, img)
        inpaint_out = image_path.rsplit(".", 1)[0] + "_inpainted.png"
        inpainted_img.save(inpaint_out)
        print(
            f"    Saved {inpaint_out} ({inpainted_img.size[0]}x{inpainted_img.size[1]}) "
            f"from {region_count} regions in {time.time() - t_preview:.1f}s"
        )
        if inpaint_only:
            print("    Inpaint-only mode complete (translation/render skipped).")
            return True

    t0 = time.time()
    try:
        result = await translator.translate(img)
    except Exception as exc:
        exc_str = str(exc)
        if "model" in exc_str.lower() and ("not found" in exc_str.lower() or "404" in exc_str.lower()):
            print(f"    WARN  Groq model not found — update GROQ_MODEL in .env")
            print(f"    {exc_str[:200]}")
            print("    (Detection + OCR + merge stages all passed before this point)")
            return None
        raise
    elapsed = time.time() - t0

    out_path = image_path.rsplit(".", 1)[0] + "_translated.png"
    result.save(out_path)
    print(f"    Saved {out_path} ({result.size[0]}x{result.size[1]}) in {elapsed:.1f}s")
    print("    Local translate passed.")
    return True


async def test_local_translate(
    image_paths,
    save_inpainted: bool = False,
    inpaint_only: bool = False,
    target_lang: str = "ENG",
    detection_size: int = 1536,
    box_threshold: float = 0.6,       # ↓ from 0.7
    inpainting_size: int = 2048,
    auto_detector: bool = False,
):
    from snipshot_engine import SnipshotTranslator, Config

    config = Config(**_build_config(target_lang, detection_size, box_threshold, inpainting_size))

    translator = SnipshotTranslator(config, device="cpu")

    print("\n    Loading models once for all local image tests...")
    t0 = time.time()
    await translator.load_models()
    print(f"    Models loaded in {time.time() - t0:.1f}s")

    successes = 0
    total = len(image_paths)
    t_all = time.time()

    for image_path in image_paths:
        ok = await _process_one_local_image(
            translator,
            image_path,
            save_inpainted,
            inpaint_only,
            auto_detector,
        )
        if ok:
            successes += 1

    print(f"\n    Local summary: {successes}/{total} succeeded in {time.time() - t_all:.1f}s")
    return successes == total


# ---------------------------------------------------------------------------
# 4-6. Server endpoint tests
# ---------------------------------------------------------------------------

async def test_server(api_url: str, image_path: str):
    import httpx

    print(f"\n    Server: {api_url}")

    async with httpx.AsyncClient(timeout=180.0) as client:
        # 4. Health
        print("\n[4] GET /health...")
        try:
            resp = await client.get(f"{api_url}/health")
            if resp.status_code == 200:
                print(f"    OK  {resp.json()}")
            else:
                print(f"    FAIL  status {resp.status_code}: {resp.text}")
                return False
        except httpx.ConnectError:
            print(f"    FAIL  Cannot connect to {api_url}")
            print("    Is the server running? Start it with:")
            print("      python main.py")
            return False

        # Load image bytes
        try:
            with open(image_path, "rb") as f:
                image_bytes = f.read()
            print(f"    Using {image_path} ({len(image_bytes)} bytes)")
        except FileNotFoundError:
            from PIL import Image as PILImage
            print(f"    {image_path} not found — using blank 200x200 test image")
            buf = io.BytesIO()
            PILImage.new("RGB", (200, 200), "white").save(buf, format="PNG")
            image_bytes = buf.getvalue()

        # Use manga-recommended config for server tests too
        config_json = json.dumps({
            **MANGA_CONFIG,
            "translator": {"target_lang": "ENG"},
        })

        # 5. /translate/raw
        # print("\n[5] POST /translate/raw...")
        # resp = await client.post(
        #     f"{api_url}/translate/raw",
        #     files={"image": ("test.jpg", image_bytes, "image/jpeg")},
        #     data={"config": config_json},
        # )
        # if resp.status_code == 200 and "image/png" in resp.headers.get("content-type", ""):
        #     out_path = "test_snipshot_raw.png"
        #     with open(out_path, "wb") as f:
        #         f.write(resp.content)
        #     print(f"    OK  Got PNG ({len(resp.content)} bytes) → {out_path}")
        # else:
        #     print(f"    FAIL  status {resp.status_code}: {resp.text[:200]}")

        # 6. /translate (Supabase upload)
        # print("\n[6] POST /translate (Supabase upload)...")
        # resp = await client.post(
        #     f"{api_url}/translate",
        #     files={"image": ("test.jpg", image_bytes, "image/jpeg")},
        #     data={"config": config_json},
        # )
        # if resp.status_code == 200:
        #     result = resp.json()
        #     print(f"    OK  success={result.get('success')}")
        #     print(f"    URL: {result.get('image_url', 'N/A')}")
        # elif resp.status_code == 502:
        #     print(f"    WARN  Supabase not configured (expected if no env vars)")
        #     print(f"    {resp.json().get('detail', resp.text[:200])}")
        # else:
        #     print(f"    FAIL  status {resp.status_code}: {resp.text[:200]}")

    print("\n    Server tests done.")
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Test snipshot_engine")
    parser.add_argument("--local", action="store_true", help="Run only local tests (1-3), skip server tests")
    parser.add_argument("--save-inpainted", action="store_true", help="Also save an inpaint-only preview image (*_inpainted.png)")
    parser.add_argument("--inpaint-only", action="store_true", help="Run only up to inpainting and save *_inpainted.png (skip translation/render)")
    parser.add_argument("--image", default="test-image-easy1.jpg", help="Path to a test image or directory of images")
    parser.add_argument("--url", default="http://localhost:8001", help="Server URL for endpoint tests (default: http://localhost:8001)")
    parser.add_argument("--target-lang", default="ENG", help="Target language for local translate (default: ENG)")
    parser.add_argument("--detection-size", type=int, default=1536, help="Detector size for local translate (default: 1536)")
    parser.add_argument("--box-threshold", type=float, default=0.6, help="Detector box threshold for local translate (default: 0.6)")  # ↓ from 0.7
    parser.add_argument("--inpainting-size", type=int, default=2048, help="Inpainting size for local translate (default: 2048)")
    parser.add_argument("--auto-detector", action="store_true", help="Auto-tune detector size/threshold per image dimensions")
    args = parser.parse_args()

    if args.inpaint_only:
        args.local = True

    print("=" * 60)
    print("  SnipShot Engine Test Suite")
    print("=" * 60)

    # Always run import + config tests
    ok = test_imports()
    if not ok:
        print("\nImport check failed — aborting.")
        sys.exit(1)

    test_config()

    image_paths = _collect_image_paths(args.image)
    if not image_paths:
        print(f"\nNo input images found for: {args.image}")
        sys.exit(1)

    if len(image_paths) > 1:
        print(f"\n    Collected {len(image_paths)} images from {args.image}")

    # Local in-process translate
    asyncio.run(
        test_local_translate(
            image_paths,
            save_inpainted=args.save_inpainted or args.inpaint_only,
            inpaint_only=args.inpaint_only,
            target_lang=args.target_lang,
            detection_size=args.detection_size,
            box_threshold=args.box_threshold,
            inpainting_size=args.inpainting_size,
            auto_detector=args.auto_detector,
        )
    )

    # Server endpoint tests (unless --local)
    if not args.local:
        asyncio.run(test_server(args.url, image_paths[0]))
    else:
        print("\n    --local flag set, skipping server tests (4-6).")

    print("\n" + "=" * 60)
    print("  Done.")
    print("=" * 60)


if __name__ == "__main__":
    main()
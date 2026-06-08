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
    """Infer detector settings from image dimensions for local test sweeps."""
    long_side = max(width, height)
    mp = (width * height) / 1_000_000.0

    # Round to nearest 64 and clamp to practical range.
    detection_size = int(round(long_side / 64.0) * 64)
    detection_size = max(1024, min(3072, detection_size))

    if mp < 1.0:
        box_threshold = 0.55
    elif mp < 2.5:
        box_threshold = 0.60
    elif mp < 4.0:
        box_threshold = 0.65
    else:
        box_threshold = 0.70

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

    # Custom config (like the frontend would send)
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
# 2.5 LIR Geometry Validation
# ---------------------------------------------------------------------------

def test_lir_geometry():
    print("\n[2.5] LIR Geometry Validation...")
    import numpy as np
    from snipshot_engine.rendering.bubble import find_largest_inscribed_rectangle

    # Test case 1: Perfect rectangle
    mask_rect = np.zeros((20, 20), dtype=np.uint8)
    mask_rect[4:16, 5:15] = 255
    x, y, w, h = find_largest_inscribed_rectangle(mask_rect)
    print(f"    Rectangle LIR: got ({x}, {y}, {w}, {h}), expected (5, 4, 10, 12)")
    assert (x, y, w, h) == (5, 4, 10, 12), "Rectangle LIR coordinates mismatch"
    assert np.all(mask_rect[y:y+h, x:x+w] == 255), "Rectangle LIR subset validation failed"

    # Test case 2: Perfect circle
    mask_circle = np.zeros((30, 30), dtype=np.uint8)
    cy, cx = 15, 15
    r = 10
    for i in range(30):
        for j in range(30):
            if (j - cx)**2 + (i - cy)**2 <= r**2:
                mask_circle[i, j] = 255
    cx_lir, cy_lir, cw_lir, ch_lir = find_largest_inscribed_rectangle(mask_circle)
    print(f"    Circle LIR: got ({cx_lir}, {cy_lir}, {cw_lir}, {ch_lir})")
    assert cw_lir > 0 and ch_lir > 0, "Circle LIR returned empty rectangle"
    assert np.all(mask_circle[cy_lir:cy_lir+ch_lir, cx_lir:cx_lir+cw_lir] == 255), "Circle LIR subset validation failed"

    # Test case 3: Perfect ellipse
    mask_ellipse = np.zeros((40, 40), dtype=np.uint8)
    ey, ex = 20, 20
    a, b = 15, 8
    for i in range(40):
        for j in range(40):
            if ((j - ex) / a)**2 + ((i - ey) / b)**2 <= 1.0:
                mask_ellipse[i, j] = 255
    ex_lir, ey_lir, ew_lir, eh_lir = find_largest_inscribed_rectangle(mask_ellipse)
    print(f"    Ellipse LIR: got ({ex_lir}, {ey_lir}, {ew_lir}, {eh_lir})")
    assert ew_lir > 0 and eh_lir > 0, "Ellipse LIR returned empty rectangle"
    assert np.all(mask_ellipse[ey_lir:ey_lir+eh_lir, ex_lir:ex_lir+ew_lir] == 255), "Ellipse LIR subset validation failed"

    # Test case 4: Complex irregular polygon / jagged shape
    mask_irregular = np.zeros((30, 30), dtype=np.uint8)
    mask_irregular[5:25, 5:25] = 255
    mask_irregular[20:28, 20:22] = 255 # tail
    ix_lir, iy_lir, iw_lir, ih_lir = find_largest_inscribed_rectangle(mask_irregular)
    print(f"    Irregular LIR: got ({ix_lir}, {iy_lir}, {iw_lir}, {ih_lir})")
    assert iw_lir > 0 and ih_lir > 0, "Irregular LIR returned empty rectangle"
    assert np.all(mask_irregular[iy_lir:iy_lir+ih_lir, ix_lir:ix_lir+iw_lir] == 255), "Irregular LIR subset validation failed"

    print("    LIR geometry validation tests passed.")
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
        from snipshot_engine.utils import load_image, dump_image, is_valuable_text, sort_regions, LANGUAGE_ORIENTATION_PRESETS
        from snipshot_engine.ocr import dispatch as dispatch_ocr
        from snipshot_engine.textline_merge import dispatch as dispatch_textline_merge
        from snipshot_engine.translation import dispatch as dispatch_translation
        from snipshot_engine.mask_refinement import dispatch as dispatch_mask_refinement
        from snipshot_engine.inpainting import dispatch as dispatch_inpainting
        from snipshot_engine.rendering import dispatch as dispatch_rendering
        from snipshot_engine.rendering.bubble import detect_bubbles
        from snipshot_engine.config import Inpainter
        import numpy as np

        cfg = translator.config
        img_rgb, img_alpha = load_image(img)

        # ── 1. Detection ─────────────────────────────────────────────
        textlines, mask_raw, _ = await translator._detector.infer(
            img_rgb,
            cfg.detector.detection_size,
            cfg.detector.text_threshold,
            cfg.detector.box_threshold,
            cfg.detector.unclip_ratio,
            verbose=False,
        )

        if not textlines:
            print("    [Diagnostics] 0 regions detected.")
            result = img
        else:
            # ── 2. OCR ───────────────────────────────────────────────────
            textlines = await dispatch_ocr(
                cfg.ocr.ocr, img_rgb, textlines, cfg.ocr, translator.device, verbose=False
            )
            textlines = [t for t in textlines if t.text.strip()]

            if not textlines:
                print("    [Diagnostics] 0 regions recognized after OCR.")
                result = img
            else:
                # ── 3. Textline merge ────────────────────────────────────────
                text_regions = await dispatch_textline_merge(
                    textlines, img_rgb.shape[1], img_rgb.shape[0], verbose=False
                )

                # Filter short / non-valuable text
                text_regions = [
                    r for r in text_regions
                    if len(r.text) >= cfg.ocr.min_text_length and is_valuable_text(r.text)
                ]

                if not text_regions:
                    print("    [Diagnostics] 0 regions after merge/filtering.")
                    result = img
                else:
                    text_regions = sort_regions(text_regions, img_rgb.shape[1], img_rgb.shape[0])
                    
                    target_lang = cfg.translator.target_lang
                    for region in text_regions:
                        region.target_lang = target_lang
                        preset = LANGUAGE_ORIENTATION_PRESETS.get(target_lang)
                        if preset:
                            region._direction = preset

                    # ── 4. Translation ───────────────────────────────────────────
                    queries = [r.text for r in text_regions]
                    from_lang = "auto"
                    translations = await dispatch_translation(from_lang, target_lang, queries)
                    for region, trans in zip(text_regions, translations):
                        region.translation = trans

                    # ── 5. Mask refinement ───────────────────────────────────────
                    mask = await dispatch_mask_refinement(
                        text_regions,
                        img_rgb,
                        mask_raw if mask_raw is not None else np.zeros(img_rgb.shape[:2], dtype=np.uint8),
                        method="fit_text",
                        dilation_offset=cfg.mask_dilation_offset,
                        ignore_bubble=cfg.ocr.ignore_bubble,
                        kernel_size=cfg.kernel_size,
                    )

                    # ── 6. Inpainting ───────────────────────────────────────────
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

                    # ── 7. Bubble Detection Analysis ────────────────────────────
                    bubble_rects = detect_bubbles(img_inpainted, text_regions)
                    
                    print(f"\n    [DIAGNOSTICS] Pipeline Status for {image_path}:")
                    print(f"    - Total Text Regions: {len(text_regions)}")
                    
                    bubble_count = 0
                    for idx, (region, rect) in enumerate(zip(text_regions, bubble_rects)):
                        ocr_txt = region.text.replace("\n", " ").strip()
                        trans_txt = region.translation.replace("\n", " ").strip()
                        if rect is not None:
                            bubble_count += 1
                            bw = int(rect[0, 1, 0] - rect[0, 0, 0])
                            bh = int(rect[0, 2, 1] - rect[0, 0, 1])
                            status = f"Bubble Detected ({bw}x{bh})"
                        else:
                            status = "No Bubble (Fallback to bounding box)"
                        
                        print(f"      Region #{idx + 1}:")
                        print(f"        Status:      {status}")
                        print(f"        Original:    \"{ocr_txt}\"")
                        print(f"        Translation: \"{trans_txt}\"")
                    
                    print(f"    - Total Bubbles Successfully Detected: {bubble_count}/{len(text_regions)}\n")

                    # ── 8. Rendering ────────────────────────────────────────────
                    import snipshot_engine.rendering as rendering_mod
                    original_render_region = rendering_mod._render_region

                    def wrapped_render_region(img, region, dst_points, hyphenate, line_spacing, disable_font_border, font_size_minimum=0):
                        print(f"\n      [RENDER DIAGNOSTICS] Rendering Region:")
                        print(f"        Original Text:  \"{region.text.strip()}\"")
                        print(f"        Translation:    \"{region.translation.strip()}\"")
                        print(f"        Font Size:      {region.font_size}")
                        print(f"        Raw Dst Points: {dst_points.tolist() if hasattr(dst_points, 'tolist') else dst_points}")
                        
                        # Let's inspect inner bounds
                        middle_pts = (dst_points[:, [1, 2, 3, 0]] + dst_points) / 2
                        norm_h = np.linalg.norm(middle_pts[:, 1] - middle_pts[:, 3], axis=1)
                        norm_v = np.linalg.norm(middle_pts[:, 2] - middle_pts[:, 0], axis=1)
                        target_w = max(1, int(round(norm_h[0])))
                        target_h = max(1, int(round(norm_v[0])))
                        print(f"        Target Box:     {target_w}x{target_h}")

                        res_img = original_render_region(img, region, dst_points, hyphenate, line_spacing, disable_font_border, font_size_minimum)
                        return res_img

                    rendering_mod._render_region = wrapped_render_region
                    try:
                        img_rendered = await dispatch_rendering(
                            img_inpainted,
                            text_regions,
                            font_path="",
                            font_size_offset=cfg.render.font_size_offset,
                            font_size_minimum=cfg.render.font_size_minimum,
                            hyphenate=not cfg.render.no_hyphenation,
                            line_spacing=cfg.render.line_spacing,
                            disable_font_border=cfg.render.disable_font_border,
                        )
                    finally:
                        rendering_mod._render_region = original_render_region

                    result = dump_image(img, img_rendered, img_alpha)
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
    box_threshold: float = 0.7,
    inpainting_size: int = 2048,
    auto_detector: bool = False,
):
    from snipshot_engine import SnipshotTranslator, Config

    config = Config(**{
        "translator": {"target_lang": target_lang},
        "inpainter": {"inpainter": "lama_large", "inpainting_size": inpainting_size},
        "detector": {"detection_size": detection_size, "box_threshold": box_threshold},
    })

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

        config_json = json.dumps({
            "translator": {"target_lang": "ENG"},
            "detector": {"detection_size": 1536, "box_threshold": 0.7},
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
    parser.add_argument("--image", default="test-image-medium.png", help="Path to a test image or directory of images")
    parser.add_argument("--url", default="http://localhost:8001", help="Server URL for endpoint tests (default: http://localhost:8001)")
    parser.add_argument("--target-lang", default="ENG", help="Target language for local translate (default: ENG)")
    parser.add_argument("--detection-size", type=int, default=1536, help="Detector size for local translate (default: 1536)")
    parser.add_argument("--box-threshold", type=float, default=0.7, help="Detector box threshold for local translate (default: 0.7)")
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
    test_lir_geometry()

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

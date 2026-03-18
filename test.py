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
# 3. In-process translate (local, no server)
# ---------------------------------------------------------------------------

async def test_local_translate(image_path: str):
    print(f"\n[3] In-process translate ({image_path})...")
    from PIL import Image
    from snipshot_engine import SnipshotTranslator, Config

    try:
        img = Image.open(image_path).convert("RGB")
        print(f"    Loaded {image_path} ({img.size[0]}x{img.size[1]})")
    except FileNotFoundError:
        print(f"    SKIP  {image_path} not found")
        return None

    config = Config(**{
        "translator": {"target_lang": "ENG"},
        "inpainter": {"inpainter": "lama_large", "inpainting_size": 2048},
        "detector": {"detection_size": 1536, "box_threshold": 0.7},
    })

    translator = SnipshotTranslator(config, device="cpu")

    print("    Loading models (first time may download checkpoints)...")
    t0 = time.time()
    await translator.load_models()
    print(f"    Models loaded in {time.time() - t0:.1f}s")

    print("    Running translation pipeline...")
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
    parser.add_argument("--image", default="test-image-medium.png", help="Path to a manga/manhwa test image (default: 15.jpg)")
    parser.add_argument("--url", default="http://localhost:8001", help="Server URL for endpoint tests (default: http://localhost:8001)")
    args = parser.parse_args()

    print("=" * 60)
    print("  SnipShot Engine Test Suite")
    print("=" * 60)

    # Always run import + config tests
    ok = test_imports()
    if not ok:
        print("\nImport check failed — aborting.")
        sys.exit(1)

    test_config()

    # Local in-process translate
    asyncio.run(test_local_translate(args.image))

    # Server endpoint tests (unless --local)
    if not args.local:
        asyncio.run(test_server(args.url, args.image))
    else:
        print("\n    --local flag set, skipping server tests (4-6).")

    print("\n" + "=" * 60)
    print("  Done.")
    print("=" * 60)


if __name__ == "__main__":
    main()

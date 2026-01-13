"""
Test script for translator_api.py
Tests the Supabase Storage upload flow.
"""

import httpx
import asyncio
import json

# Configuration
API_URL = "http://localhost:8000"


async def test_translator_api():
    print("=" * 60)
    print("Testing Translator API (Supabase Storage Upload)")
    print("=" * 60)
    
    async with httpx.AsyncClient(timeout=120.0) as client:
        
        # 1. Health check
        print("\n[1] Health check...")
        resp = await client.get(f"{API_URL}/health")
        if resp.status_code == 200:
            print(f"    ✓ API is healthy: {resp.json()}")
        else:
            print(f"    ✗ Health check failed: {resp.status_code}")
            return
        
        # 2. Test translation with Supabase upload
        print("\n[2] Testing /translate (Supabase Storage upload)...")
        
        config = {
            "detector": {
                "detector": "default",
                "detection_size": 1536,
                "box_threshold": 0.7
            },
            "translator": {
                "translator": "groq",
                "target_lang": "ENG"
            },
            "inpainter": {
                "inpainter": "default"
            }
        }
        
        # Use the test image
        try:
            with open("15.jpg", "rb") as f:
                image_bytes = f.read()
            print(f"    → Using 15.jpg ({len(image_bytes)} bytes)")
        except FileNotFoundError:
            print("    ✗ 15.jpg not found, using placeholder")
            # Create a simple test image
            from PIL import Image
            import io
            img = Image.new("RGB", (100, 100), color="white")
            buffer = io.BytesIO()
            img.save(buffer, format="PNG")
            image_bytes = buffer.getvalue()
        
        files = {"image": ("test.jpg", image_bytes, "image/jpeg")}
        data = {"config": json.dumps(config)}
        
        print("    → Sending translation request...")
        print("    → This may take a minute...")
        
        resp = await client.post(
            f"{API_URL}/translate",
            files=files,
            data=data
        )
        
        if resp.status_code == 200:
            result = resp.json()
            print(f"    ✓ Translation successful!")
            print(f"    → Success: {result.get('success')}")
            print(f"    → Image URL: {result.get('image_url')}")
            print(f"    → Storage Path: {result.get('storage_path')}")
        else:
            print(f"    ✗ Translation failed: {resp.status_code}")
            print(f"    → Response: {resp.text}")
            return
        
        # 3. Test /translate/raw (no storage upload)
        print("\n[3] Testing /translate/raw (raw PNG)...")
        
        resp = await client.post(
            f"{API_URL}/translate/raw",
            files={"image": ("test.jpg", image_bytes, "image/jpeg")},
            data={"config": json.dumps(config)}
        )
        
        if resp.status_code == 200:
            content_type = resp.headers.get("content-type", "")
            if "image/png" in content_type:
                print(f"    ✓ Got raw PNG ({len(resp.content)} bytes)")
                # Save it
                with open("test_output_raw.png", "wb") as f:
                    f.write(resp.content)
                print(f"    → Saved to test_output_raw.png")
            else:
                print(f"    ✗ Unexpected content type: {content_type}")
        else:
            print(f"    ✗ Raw translation failed: {resp.status_code}")
            print(f"    → Response: {resp.text}")
        
        # Summary
        print("\n" + "=" * 60)
        print("✓ Translator API Test Complete!")
        print("=" * 60)
        print("\nYour VM is ready to:")
        print("  1. Receive images from frontend")
        print("  2. Translate them")
        print("  3. Upload to Supabase Storage")
        print("  4. Return {image_url, storage_path}")


if __name__ == "__main__":
    asyncio.run(test_translator_api())

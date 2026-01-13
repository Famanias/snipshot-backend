"""
Test script for VM backend with authentication and Cloudinary storage.

Flow:
1. Login to get JWT token
2. Send translation request with auth
3. VM uploads to Cloudinary automatically
4. Returns Cloudinary URL (no files stored on VM)
"""

import aiohttp
import asyncio
import json
from pathlib import Path

# VM API URL
VM_BASE_URL = "http://34.87.58.21:8000"

# Test user credentials (register first if needed)
TEST_USER = {
    "email": "test@example.com",
    "username": "testuser",
    "password": "testpass123"
}


async def register_user():
    """Register a test user (skip if already exists)"""
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{VM_BASE_URL}/api/users/register",
            json=TEST_USER
        ) as resp:
            if resp.status == 201:
                print("✓ User registered")
            elif resp.status == 400:
                print("✓ User already exists")
            else:
                print(f"Registration error: {await resp.text()}")


async def login() -> str:
    """Login and get JWT token"""
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{VM_BASE_URL}/api/users/login",
            json={
                "email": TEST_USER["email"],
                "password": TEST_USER["password"]
            }
        ) as resp:
            if resp.status == 200:
                data = await resp.json()
                print(f"✓ Login successful")
                return data["access_token"]
            else:
                print(f"Login failed: {await resp.text()}")
                return None


async def translate_with_storage(token: str, image_path: str):
    """
    Translate image with authentication.
    
    - Image is uploaded to VM
    - VM processes translation
    - VM uploads result to Cloudinary (no local storage on VM)
    - Returns Cloudinary URL
    """
    path = Path(image_path)
    assert path.exists(), f"{image_path} not found"

    config_json = {
        "detector": {
            "detector": "default",
            "detection_size": 1536,
            "box_threshold": 0.7,
            "unclip_ratio": 2.3
        },
        "render": {
            "direction": "auto"
        },
        "translator": {
            "translator": "groq",
            "target_lang": "ENG"
        },
        "inpainter": {
            "inpainter": "default",
            "inpainting_size": 2048
        },
        "mask_dilation_offset": 30
    }

    form = aiohttp.FormData()
    form.add_field(
        "image",
        path.open("rb"),
        filename=path.name,
        content_type="image/jpeg"
    )
    form.add_field("config", json.dumps(config_json))
    form.add_field("save", "true")  # Tell VM to save to Cloudinary

    headers = {"Authorization": f"Bearer {token}"}

    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{VM_BASE_URL}/translate",
            data=form,
            headers=headers
        ) as resp:
            if resp.status != 200:
                print(f"❌ Translation error: {resp.status}")
                print(await resp.text())
                return None

            content_type = resp.headers.get("content-type", "")
            
            if "application/json" in content_type:
                # Success! Got Cloudinary URL
                data = await resp.json()
                print(f"✓ Translation saved to Cloudinary!")
                print(f"  Image ID: {data.get('image_id')}")
                print(f"  URL: {data.get('image_url')}")
                return data
            else:
                # Fallback: got raw image (storage failed)
                print("⚠ Got raw image instead of URL (storage may have failed)")
                with open("output_fallback.png", "wb") as f:
                    f.write(await resp.read())
                print("  Saved to: output_fallback.png")
                return None


async def translate_anonymous(image_path: str):
    """
    Translate without authentication.
    
    - Returns raw image directly
    - Nothing stored on Cloudinary
    - Nothing stored on VM
    """
    path = Path(image_path)
    assert path.exists(), f"{image_path} not found"

    config_json = {
        "detector": {
            "detector": "default",
            "detection_size": 1536,
            "box_threshold": 0.7,
            "unclip_ratio": 2.3
        },
        "render": {
            "direction": "auto"
        },
        "translator": {
            "translator": "groq",
            "target_lang": "ENG"
        },
        "inpainter": {
            "inpainter": "default",
            "inpainting_size": 2048
        },
        "mask_dilation_offset": 30
    }

    form = aiohttp.FormData()
    form.add_field(
        "image",
        path.open("rb"),
        filename=path.name,
        content_type="image/jpeg"
    )
    form.add_field("config", json.dumps(config_json))

    async with aiohttp.ClientSession() as session:
        async with session.post(f"{VM_BASE_URL}/translate", data=form) as resp:
            if resp.status != 200:
                print(f"❌ Error: {resp.status}")
                print(await resp.text())
                return

            img_bytes = await resp.read()
            with open("output.png", "wb") as f:
                f.write(img_bytes)
            print("✓ Anonymous translation saved to output.png")


async def list_my_images(token: str):
    """List all images saved to user's account"""
    headers = {"Authorization": f"Bearer {token}"}
    
    async with aiohttp.ClientSession() as session:
        async with session.get(
            f"{VM_BASE_URL}/api/images",
            headers=headers
        ) as resp:
            if resp.status == 200:
                data = await resp.json()
                print(f"\n✓ Found {data['total']} saved images:")
                for img in data['images']:
                    print(f"  - ID {img['id']}: {img['translated_url']}")
                return data['images']
            else:
                print(f"Error listing images: {await resp.text()}")
                return []


async def main():
    print("=" * 60)
    print("SnipShot VM Test - Authenticated Translation with Cloudinary")
    print("=" * 60)
    
    # Register user (first time only)
    await register_user()
    
    # Login
    token = await login()
    if not token:
        print("❌ Cannot continue without login")
        return
    
    # Translate with authentication (saves to Cloudinary, no VM storage)
    print("\n--- Authenticated Translation (saves to Cloudinary) ---")
    result = await translate_with_storage(token, "13.jpg")
    
    # List saved images
    if result:
        await list_my_images(token)
    
    # Also test anonymous (returns raw image, no storage)
    print("\n--- Anonymous Translation (returns image directly) ---")
    await translate_anonymous("13.jpg")
    
    print("\n" + "=" * 60)
    print("Test completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())

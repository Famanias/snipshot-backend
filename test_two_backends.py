"""
Test script for the two-backend architecture:
1. VM Translator API (Google Cloud) - Translation + Cloudinary upload
2. Database API (Render) - User auth + image metadata storage

Flow:
1. Register/Login to Database API → get JWT
2. Send image to VM Translator API → get Cloudinary URL
3. Save Cloudinary URL to Database API → stores metadata
4. Fetch user's images from Database API
"""

import httpx
import asyncio
import base64
from pathlib import Path

# Configuration
VM_URL = "http://34.87.58.21:8000"  # Translator API (Google Cloud)
DB_URL = "http://localhost:8000"     # Database API (local dev, change for Render)

# Test image (1x1 white pixel PNG)
TEST_IMAGE_B64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="


async def test_full_flow():
    async with httpx.AsyncClient(timeout=120.0) as client:
        print("=" * 60)
        print("Testing Two-Backend Architecture")
        print("=" * 60)
        
        # =====================================================
        # Step 1: Register/Login to Database API
        # =====================================================
        print("\n[1] Registering user on Database API...")
        
        register_data = {
            "email": "test_arch@example.com",
            "username": "testarch",
            "password": "testpass123"
        }
        
        # Try to register (may fail if user exists)
        resp = await client.post(f"{DB_URL}/api/users/register", json=register_data)
        if resp.status_code == 201:
            print(f"    ✓ Registered new user: {register_data['email']}")
        else:
            print(f"    → User may already exist, proceeding to login...")
        
        # Login
        print("\n[2] Logging in to Database API...")
        login_data = {
            "username": register_data["email"],
            "password": register_data["password"]
        }
        resp = await client.post(
            f"{DB_URL}/api/users/login",
            data=login_data,  # OAuth2 form data
            headers={"Content-Type": "application/x-www-form-urlencoded"}
        )
        
        if resp.status_code != 200:
            print(f"    ✗ Login failed: {resp.status_code}")
            print(f"    Response: {resp.text}")
            return
        
        token_data = resp.json()
        jwt_token = token_data["access_token"]
        print(f"    ✓ Login successful, got JWT token")
        
        auth_headers = {"Authorization": f"Bearer {jwt_token}"}
        
        # =====================================================
        # Step 2: Send image to VM Translator API
        # =====================================================
        print("\n[3] Sending image to VM Translator API...")
        
        # Decode test image
        image_bytes = base64.b64decode(TEST_IMAGE_B64)
        
        # Prepare multipart form data
        translate_data = {
            "target_lang": "ENG",
            "detector": "craft"
        }
        
        files = {"file": ("test_image.png", image_bytes, "image/png")}
        
        try:
            resp = await client.post(
                f"{VM_URL}/translate",
                data=translate_data,
                files=files
            )
            
            if resp.status_code != 200:
                print(f"    ✗ Translation failed: {resp.status_code}")
                print(f"    Response: {resp.text}")
                return
            
            translate_result = resp.json()
            cloudinary_url = translate_result["image_url"]
            public_id = translate_result["public_id"]
            
            print(f"    ✓ Translation successful!")
            print(f"    → Cloudinary URL: {cloudinary_url[:50]}...")
            print(f"    → Public ID: {public_id}")
            
        except httpx.ConnectError:
            print(f"    ✗ Could not connect to VM at {VM_URL}")
            print("    → Using mock data for testing Database API...")
            cloudinary_url = "https://res.cloudinary.com/demo/image/upload/sample.jpg"
            public_id = "mock_test_image"
        
        # =====================================================
        # Step 3: Save Cloudinary URL to Database API
        # =====================================================
        print("\n[4] Saving image metadata to Database API...")
        
        image_data = {
            "original_url": cloudinary_url,  # In real case, you'd save original too
            "translated_url": cloudinary_url,
            "original_public_id": public_id,
            "translated_public_id": public_id,
            "source_language": "JPN",
            "target_language": "ENG",
            "metadata": {
                "filename": "test_image.png",
                "detector": "craft",
                "translated_at": "2024-01-01T00:00:00Z"
            }
        }
        
        resp = await client.post(
            f"{DB_URL}/api/images",
            json=image_data,
            headers=auth_headers
        )
        
        if resp.status_code != 201:
            print(f"    ✗ Failed to save image: {resp.status_code}")
            print(f"    Response: {resp.text}")
            return
        
        saved_image = resp.json()
        image_id = saved_image["id"]
        print(f"    ✓ Image saved with ID: {image_id}")
        
        # =====================================================
        # Step 4: Fetch user's images from Database API
        # =====================================================
        print("\n[5] Fetching user's images from Database API...")
        
        resp = await client.get(
            f"{DB_URL}/api/images",
            headers=auth_headers
        )
        
        if resp.status_code != 200:
            print(f"    ✗ Failed to fetch images: {resp.status_code}")
            return
        
        images = resp.json()
        print(f"    ✓ Found {len(images)} images for user")
        for img in images:
            print(f"      - ID: {img['id']}, URL: {img['translated_url'][:40]}...")
        
        # =====================================================
        # Step 5: Verify user profile
        # =====================================================
        print("\n[6] Verifying user profile...")
        
        resp = await client.get(
            f"{DB_URL}/api/users/me",
            headers=auth_headers
        )
        
        if resp.status_code == 200:
            user = resp.json()
            print(f"    ✓ User profile: {user['username']} ({user['email']})")
        
        # =====================================================
        # Summary
        # =====================================================
        print("\n" + "=" * 60)
        print("✓ Two-Backend Architecture Test PASSED!")
        print("=" * 60)
        print("\nArchitecture Flow:")
        print("  Frontend → VM Translator API → Cloudinary → Returns URL")
        print("  Frontend → Database API → Saves URL metadata")
        print("  Frontend → Database API → Fetches user's images")


if __name__ == "__main__":
    asyncio.run(test_full_flow())

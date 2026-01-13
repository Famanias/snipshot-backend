"""
End-to-End Test: Register → Login → Translate → Save to Database

This script demonstrates the complete SnipShot workflow:
1. Register a new user (Supabase Auth)
2. Login to get JWT token
3. Send image to VM for translation
4. Save translated image to user's account (Supabase Storage + DB)

Prerequisites:
- VM translator running: python main.py (in snipshot-backend)
- Database API running: cd database_api && uvicorn main:app --port 8002
- 15.jpg exists in the current directory
"""

import httpx
import asyncio
import json
import time

# API URLs
VM_TRANSLATOR_URL = "http://localhost:8000"  # Local VM translator
DATABASE_API_URL = "http://localhost:8002/api"  # Local database API

# Test user credentials
# Option 1: Use your real email for testing
# Option 2: Disable "Confirm email" in Supabase Auth settings
TEST_EMAIL = "haratayo@gmail.com"  # <-- CHANGE THIS to your email!
TEST_PASSWORD = "ediwow123"


async def main():
    print("=" * 70)
    print("SnipShot End-to-End Test")
    print("Register → Login → Translate → Save to Database")
    print("=" * 70)
    
    async with httpx.AsyncClient(timeout=180.0) as client:
        
        # ========================================
        # STEP 1: Register User
        # ========================================
        print("\n[1] REGISTER USER")
        print("-" * 50)
        print(f"    Email: {TEST_EMAIL}")
        
        resp = await client.post(f"{DATABASE_API_URL}/users/register", json={
            "email": TEST_EMAIL,
            "password": TEST_PASSWORD
        })
        
        if resp.status_code == 201:
            data = resp.json()
            print(f"    ✓ Registration successful!")
            if data.get("access_token"):
                print(f"    → Got access token immediately")
        elif resp.status_code == 200:
            print(f"    ✓ User registered (email confirmation may be required)")
        else:
            print(f"    ✗ Registration failed: {resp.status_code}")
            print(f"    → {resp.text}")
            # Try to continue with login anyway
        
        # ========================================
        # STEP 2: Login
        # ========================================
        print("\n[2] LOGIN")
        print("-" * 50)
        
        resp = await client.post(f"{DATABASE_API_URL}/users/login", json={
            "email": TEST_EMAIL,
            "password": TEST_PASSWORD
        })
        
        if resp.status_code != 200:
            print(f"    ✗ Login failed: {resp.status_code}")
            print(f"    → {resp.text}")
            print("\n    Note: If using Supabase, you may need to confirm email first.")
            print("    Check your Supabase Auth settings or use an existing user.")
            return
        
        auth_data = resp.json()
        access_token = auth_data.get("access_token")
        print(f"    ✓ Login successful!")
        print(f"    → Token: {access_token[:50]}...")
        
        headers = {"Authorization": f"Bearer {access_token}"}
        
        # Get user profile
        resp = await client.get(f"{DATABASE_API_URL}/users/me", headers=headers)
        if resp.status_code == 200:
            profile = resp.json()
            print(f"    → User ID: {profile.get('id', 'N/A')}")
        
        # ========================================
        # STEP 3: Create a Folder
        # ========================================
        print("\n[3] CREATE FOLDER")
        print("-" * 50)
        
        resp = await client.post(f"{DATABASE_API_URL}/folders", headers=headers, json={
            "name": "My Translations",
            "description": "Translated manga images"
        })
        
        folder_id = None
        if resp.status_code == 201:
            folder = resp.json()
            folder_id = folder["id"]
            print(f"    ✓ Created folder: {folder['name']} (id={folder_id})")
        elif resp.status_code == 400:
            # Folder might already exist
            print(f"    → Folder may already exist, fetching...")
            resp = await client.get(f"{DATABASE_API_URL}/folders", headers=headers)
            if resp.status_code == 200:
                folders = resp.json().get("folders", [])
                if folders:
                    folder_id = folders[0]["id"]
                    print(f"    ✓ Using existing folder: {folders[0]['name']} (id={folder_id})")
        else:
            print(f"    ✗ Failed to create folder: {resp.status_code}")
        
        # ========================================
        # STEP 4: Translate Image via VM
        # ========================================
        print("\n[4] TRANSLATE IMAGE (VM)")
        print("-" * 50)
        
        # Check if 15.jpg exists
        try:
            with open("15.jpg", "rb") as f:
                image_bytes = f.read()
            print(f"    → Loaded 15.jpg ({len(image_bytes):,} bytes)")
        except FileNotFoundError:
            print("    ✗ 15.jpg not found!")
            print("    → Please ensure 15.jpg exists in the current directory")
            return
        
        # Translation config
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
        
        print(f"    → Sending to VM translator...")
        print(f"    → Target language: English")
        print(f"    → This may take 1-2 minutes...")
        
        resp = await client.post(
            f"{VM_TRANSLATOR_URL}/translate",
            files={"image": ("15.jpg", image_bytes, "image/jpeg")},
            data={"config": json.dumps(config)}
        )
        
        if resp.status_code != 200:
            print(f"    ✗ Translation failed: {resp.status_code}")
            print(f"    → {resp.text[:200]}")
            return
        
        translation_result = resp.json()
        translated_url = translation_result.get("image_url")
        print(f"    ✓ Translation complete!")
        print(f"    → Supabase URL: {translated_url[:70]}...")
        
        # ========================================
        # STEP 5: Save to User's Database
        # ========================================
        print("\n[5] SAVE TO DATABASE")
        print("-" * 50)
        
        save_data = {
            "image_url": translated_url,
            "filename": "15_translated.png",
            "source_language": "JPN",
            "target_language": "ENG"
        }
        
        if folder_id:
            save_data["folder_id"] = str(folder_id)
        
        print(f"    → Saving to folder: {folder_id or 'unfiled'}")
        
        resp = await client.post(
            f"{DATABASE_API_URL}/images/from-url",
            headers=headers,
            data=save_data
        )
        
        if resp.status_code == 201:
            saved = resp.json()
            print(f"    ✓ Image saved to database!")
            print(f"    → Image ID: {saved['id']}")
            print(f"    → Storage Path: {saved['storage_path']}")
            print(f"    → Public URL: {saved['public_url'][:70]}...")
        else:
            print(f"    ✗ Save failed: {resp.status_code}")
            print(f"    → {resp.text}")
            return
        
        # ========================================
        # STEP 6: Verify - List User's Images
        # ========================================
        print("\n[6] VERIFY - LIST IMAGES")
        print("-" * 50)
        
        resp = await client.get(f"{DATABASE_API_URL}/images", headers=headers)
        
        if resp.status_code == 200:
            data = resp.json()
            print(f"    ✓ User has {data['total']} image(s)")
            for img in data["images"]:
                folder_info = f"folder={img.get('folder_id')}" if img.get('folder_id') else "unfiled"
                print(f"      - {img['filename']} ({folder_info})")
        
        # ========================================
        # SUMMARY
        # ========================================
        print("\n" + "=" * 70)
        print("✓ END-TO-END TEST COMPLETE!")
        print("=" * 70)
        print(f"""
Summary:
  • User: {TEST_EMAIL}
  • Folder: {folder_id or 'None'}
  • Translated Image: {saved['id']}
  • URL: {saved['public_url']}

What happened:
  1. Registered new user in Supabase Auth
  2. Logged in and got JWT token
  3. Created a folder to organize translations
  4. Sent image to VM for translation (manga_translator)
  5. VM uploaded translated image to Supabase Storage
  6. Saved image metadata to user's account in PostgreSQL
  7. Image is now linked to user's folder

Next: Build frontend to display user's folders and images!
""")


if __name__ == "__main__":
    asyncio.run(main())

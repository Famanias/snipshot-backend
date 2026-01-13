"""
Test script for Database API - Folders & Images CRUD

This tests:
1. User registration/login
2. Folder CRUD (create, list, update, delete)
3. Image CRUD (upload, list, update, delete)
4. Image organization (moving to folders)

Run the database_api server first:
    cd database_api
    python -m uvicorn main:app --reload --port 8002
"""

import httpx
import asyncio
import json
from pathlib import Path

API_URL = "http://localhost:8002/api"

# Test credentials
TEST_EMAIL = "test@example.com"
TEST_PASSWORD = "testpass123"


async def main():
    print("=" * 60)
    print("Testing Database API - Folders & Images CRUD")
    print("=" * 60)
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        token = None
        
        # ===== 1. USER AUTH =====
        print("\n[1] User Authentication")
        print("-" * 40)
        
        # Try to register (might fail if user exists)
        print("    → Registering user...")
        resp = await client.post(f"{API_URL}/users/register", json={
            "email": TEST_EMAIL,
            "password": TEST_PASSWORD
        })
        if resp.status_code == 201:
            print(f"    ✓ Registered: {TEST_EMAIL}")
        else:
            print(f"    → User may already exist: {resp.status_code}")
        
        # Login
        print("    → Logging in...")
        resp = await client.post(f"{API_URL}/users/login", json={
            "email": TEST_EMAIL,
            "password": TEST_PASSWORD
        })
        if resp.status_code == 200:
            data = resp.json()
            token = data.get("access_token")
            print(f"    ✓ Login successful!")
            print(f"    → Token: {token[:50]}...")
        else:
            print(f"    ✗ Login failed: {resp.status_code} - {resp.text}")
            return
        
        headers = {"Authorization": f"Bearer {token}"}
        
        # Get profile
        print("    → Getting profile...")
        resp = await client.get(f"{API_URL}/users/me", headers=headers)
        if resp.status_code == 200:
            profile = resp.json()
            print(f"    ✓ Profile: {profile}")
        else:
            print(f"    ✗ Profile failed: {resp.status_code}")
        
        # ===== 2. FOLDER CRUD =====
        print("\n[2] Folder CRUD Operations")
        print("-" * 40)
        
        # Create folders
        print("    → Creating folders...")
        folders = []
        for name, desc in [("Manga", "Japanese manga translations"), ("Manhwa", "Korean manhwa")]:
            resp = await client.post(f"{API_URL}/folders", headers=headers, json={
                "name": name,
                "description": desc
            })
            if resp.status_code == 201:
                folder = resp.json()
                folders.append(folder)
                print(f"    ✓ Created folder: {folder['name']} (id={folder['id']})")
            else:
                print(f"    → Folder '{name}' may exist: {resp.status_code}")
        
        # List folders
        print("    → Listing folders...")
        resp = await client.get(f"{API_URL}/folders", headers=headers)
        if resp.status_code == 200:
            data = resp.json()
            print(f"    ✓ Found {data['total']} folders:")
            for f in data['folders']:
                print(f"      - {f['name']}: {f['image_count']} images")
            if data['folders']:
                folders = data['folders']  # Use existing folders
        
        # Update folder
        if folders:
            folder_id = folders[0]['id']
            print(f"    → Updating folder {folder_id}...")
            resp = await client.put(f"{API_URL}/folders/{folder_id}", headers=headers, json={
                "description": "Updated description!"
            })
            if resp.status_code == 200:
                print(f"    ✓ Updated folder description")
        
        # ===== 3. IMAGE CRUD =====
        print("\n[3] Image CRUD Operations")
        print("-" * 40)
        
        # Find a test image
        test_images = list(Path(".").glob("*.jpg")) + list(Path(".").glob("*.png"))
        test_images += list(Path("..").glob("*.jpg")) + list(Path("..").glob("*.png"))
        
        image_id = None
        
        if test_images:
            test_image = test_images[0]
            print(f"    → Uploading image: {test_image.name}...")
            
            with open(test_image, "rb") as f:
                files = {"file": (test_image.name, f, "image/png")}
                data = {"filename": "My Test Image"}
                if folders:
                    data["folder_id"] = str(folders[0]['id'])
                
                resp = await client.post(
                    f"{API_URL}/images",
                    headers=headers,
                    files=files,
                    data=data
                )
            
            if resp.status_code == 201:
                result = resp.json()
                image_id = result['id']
                print(f"    ✓ Uploaded image (id={image_id})")
                print(f"      URL: {result['public_url'][:60]}...")
            else:
                print(f"    ✗ Upload failed: {resp.status_code} - {resp.text}")
        else:
            print("    → No test images found, skipping upload")
        
        # List images
        print("    → Listing all images...")
        resp = await client.get(f"{API_URL}/images", headers=headers)
        if resp.status_code == 200:
            data = resp.json()
            print(f"    ✓ Found {data['total']} images")
            for img in data['images'][:3]:
                print(f"      - {img['filename']} (folder_id={img.get('folder_id')})")
            if data['images']:
                image_id = image_id or data['images'][0]['id']
        
        # List images in a specific folder
        if folders:
            folder_id = folders[0]['id']
            print(f"    → Listing images in folder {folder_id}...")
            resp = await client.get(f"{API_URL}/images?folder_id={folder_id}", headers=headers)
            if resp.status_code == 200:
                data = resp.json()
                print(f"    ✓ Found {data['total']} images in folder")
        
        # Update image (rename & move to folder)
        if image_id:
            print(f"    → Updating image {image_id}...")
            resp = await client.put(f"{API_URL}/images/{image_id}", headers=headers, json={
                "filename": "Renamed Image.png"
            })
            if resp.status_code == 200:
                print(f"    ✓ Renamed image")
            
            # Move to different folder (or unfiled)
            if len(folders) > 1:
                print(f"    → Moving image to folder {folders[1]['id']}...")
                resp = await client.put(f"{API_URL}/images/{image_id}", headers=headers, json={
                    "folder_id": folders[1]['id']
                })
                if resp.status_code == 200:
                    print(f"    ✓ Moved image to folder")
        
        # ===== 4. GET FOLDER WITH IMAGES =====
        print("\n[4] Get Folder Details")
        print("-" * 40)
        
        if folders:
            folder_id = folders[0]['id']
            print(f"    → Getting folder {folder_id} with images...")
            resp = await client.get(f"{API_URL}/folders/{folder_id}", headers=headers)
            if resp.status_code == 200:
                data = resp.json()
                print(f"    ✓ Folder: {data['name']}")
                print(f"      Description: {data['description']}")
                print(f"      Images: {len(data['images'])}")
        
        # ===== 5. CLEANUP (Optional) =====
        print("\n[5] Cleanup (Optional)")
        print("-" * 40)
        print("    → Skipping cleanup to preserve data")
        print("    → To delete, uncomment the code below")
        
        # Uncomment to delete test data:
        # if image_id:
        #     await client.delete(f"{API_URL}/images/{image_id}", headers=headers)
        #     print(f"    ✓ Deleted image {image_id}")
        # 
        # for folder in folders:
        #     await client.delete(f"{API_URL}/folders/{folder['id']}", headers=headers)
        #     print(f"    ✓ Deleted folder {folder['id']}")
        
        print("\n" + "=" * 60)
        print("✓ Database API Test Complete!")
        print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())

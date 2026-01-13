"""
Test script for SnipShot Backend - User Auth & Image Storage Flow

Tests:
1. User registration
2. User login
3. Get user profile
4. Translate image (authenticated - saves to Cloudinary)
5. List saved images
6. Delete image
7. Anonymous translation (no auth - returns image directly)

Run this AFTER starting the server with: python main.py
"""

import requests
import json
import os
from pathlib import Path

# Configuration
BASE_URL = "http://localhost:8000"
TEST_IMAGE_PATH = "test_image.png"  # Will create a simple test image if not exists

# Test user credentials
TEST_USER = {
    "email": "test@example.com",
    "username": "testuser",
    "password": "testpass123"
}


def create_test_image():
    """Create a simple test image if none exists"""
    if os.path.exists(TEST_IMAGE_PATH):
        print(f"✓ Using existing test image: {TEST_IMAGE_PATH}")
        return True
    
    try:
        from PIL import Image, ImageDraw, ImageFont
        
        # Create a simple image with text
        img = Image.new('RGB', (400, 200), color='white')
        draw = ImageDraw.Draw(img)
        
        # Add some Japanese text for translation testing
        draw.text((50, 50), "こんにちは", fill='black')
        draw.text((50, 100), "Hello World", fill='gray')
        
        img.save(TEST_IMAGE_PATH)
        print(f"✓ Created test image: {TEST_IMAGE_PATH}")
        return True
    except Exception as e:
        print(f"✗ Could not create test image: {e}")
        print("  Please provide a test image manually")
        return False


def test_health():
    """Test health endpoint"""
    print("\n--- Testing Health Endpoint ---")
    try:
        resp = requests.get(f"{BASE_URL}/health")
        if resp.status_code == 200:
            print(f"✓ Health check passed: {resp.json()}")
            return True
        else:
            print(f"✗ Health check failed: {resp.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print(f"✗ Cannot connect to {BASE_URL}")
        print("  Make sure the server is running: python main.py")
        return False


def test_register():
    """Test user registration"""
    print("\n--- Testing User Registration ---")
    resp = requests.post(
        f"{BASE_URL}/api/users/register",
        json=TEST_USER
    )
    
    if resp.status_code == 201:
        user = resp.json()
        print(f"✓ User registered: {user['username']} ({user['email']})")
        return True
    elif resp.status_code == 400:
        error = resp.json()
        if "already" in error.get("detail", "").lower():
            print(f"✓ User already exists (OK for re-runs)")
            return True
        print(f"✗ Registration failed: {error}")
        return False
    else:
        print(f"✗ Registration failed: {resp.status_code} - {resp.text}")
        return False


def test_login():
    """Test user login and get JWT token"""
    print("\n--- Testing User Login ---")
    resp = requests.post(
        f"{BASE_URL}/api/users/login",
        json={
            "email": TEST_USER["email"],
            "password": TEST_USER["password"]
        }
    )
    
    if resp.status_code == 200:
        data = resp.json()
        token = data["access_token"]
        print(f"✓ Login successful!")
        print(f"  Token: {token[:50]}...")
        return token
    else:
        print(f"✗ Login failed: {resp.status_code} - {resp.text}")
        return None


def test_get_profile(token):
    """Test getting user profile"""
    print("\n--- Testing Get Profile ---")
    resp = requests.get(
        f"{BASE_URL}/api/users/me",
        headers={"Authorization": f"Bearer {token}"}
    )
    
    if resp.status_code == 200:
        user = resp.json()
        print(f"✓ Profile retrieved: {user['username']} ({user['email']})")
        print(f"  User ID: {user['id']}")
        print(f"  Created: {user['created_at']}")
        return True
    else:
        print(f"✗ Get profile failed: {resp.status_code} - {resp.text}")
        return False


def test_translate_authenticated(token):
    """Test translation with authentication (saves to Cloudinary)"""
    print("\n--- Testing Authenticated Translation ---")
    
    if not os.path.exists(TEST_IMAGE_PATH):
        print(f"✗ Test image not found: {TEST_IMAGE_PATH}")
        return None
    
    with open(TEST_IMAGE_PATH, "rb") as f:
        resp = requests.post(
            f"{BASE_URL}/translate",
            headers={"Authorization": f"Bearer {token}"},
            files={"image": ("test.png", f, "image/png")},
            data={
                "config": json.dumps({
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
                }),
                "save": "true"
            }
        )
    
    if resp.status_code == 200:
        # Check if it's JSON (saved) or image (fallback)
        content_type = resp.headers.get("content-type", "")
        
        if "application/json" in content_type:
            data = resp.json()
            print(f"✓ Translation saved to Cloudinary!")
            print(f"  Image ID: {data.get('image_id')}")
            print(f"  URL: {data.get('image_url')}")
            return data.get("image_id")
        else:
            print(f"✓ Translation returned as image (storage may have failed)")
            # Save the image
            with open("output_auth.png", "wb") as f:
                f.write(resp.content)
            print(f"  Saved to: output_auth.png")
            return None
    else:
        print(f"✗ Translation failed: {resp.status_code} - {resp.text}")
        return None


def test_list_images(token):
    """Test listing user's saved images"""
    print("\n--- Testing List Images ---")
    resp = requests.get(
        f"{BASE_URL}/api/images",
        headers={"Authorization": f"Bearer {token}"}
    )
    
    if resp.status_code == 200:
        data = resp.json()
        print(f"✓ Found {data['total']} saved images")
        for img in data['images'][:5]:  # Show first 5
            print(f"  - ID {img['id']}: {img['translated_url'][:60]}...")
        return data['images']
    else:
        print(f"✗ List images failed: {resp.status_code} - {resp.text}")
        return []


def test_translate_anonymous():
    """Test translation without authentication (returns image directly)"""
    print("\n--- Testing Anonymous Translation ---")
    
    if not os.path.exists(TEST_IMAGE_PATH):
        print(f"✗ Test image not found: {TEST_IMAGE_PATH}")
        return False
    
    with open(TEST_IMAGE_PATH, "rb") as f:
        resp = requests.post(
            f"{BASE_URL}/translate",
            files={"image": ("test.png", f, "image/png")},
            data={"config": json.dumps({
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
            })}
        )
    
    if resp.status_code == 200:
        content_type = resp.headers.get("content-type", "")
        
        if "image" in content_type:
            with open("output_anon.png", "wb") as f:
                f.write(resp.content)
            print(f"✓ Anonymous translation successful!")
            print(f"  Saved to: output_anon.png")
            return True
        else:
            print(f"✓ Response received (unexpected type: {content_type})")
            return True
    else:
        print(f"✗ Anonymous translation failed: {resp.status_code} - {resp.text}")
        return False


def test_delete_image(token, image_id):
    """Test deleting an image"""
    print(f"\n--- Testing Delete Image (ID: {image_id}) ---")
    resp = requests.delete(
        f"{BASE_URL}/api/images/{image_id}",
        headers={"Authorization": f"Bearer {token}"}
    )
    
    if resp.status_code == 200:
        print(f"✓ Image deleted successfully")
        return True
    else:
        print(f"✗ Delete failed: {resp.status_code} - {resp.text}")
        return False


def main():
    print("=" * 50)
    print("SnipShot Backend - Auth & Storage Test")
    print("=" * 50)
    
    # Create test image
    create_test_image()
    
    # Test health
    if not test_health():
        print("\n❌ Server not running. Start with: python main.py")
        return
    
    # Test user registration
    test_register()
    
    # Test login
    token = test_login()
    if not token:
        print("\n❌ Login failed, cannot continue")
        return
    
    # Test get profile
    test_get_profile(token)
    
    # Test authenticated translation (saves to Cloudinary)
    image_id = test_translate_authenticated(token)
    
    # Test list images
    images = test_list_images(token)
    
    # Test anonymous translation
    test_translate_anonymous()
    
    # Test delete image (if one was created)
    if image_id:
        test_delete_image(token, image_id)
    
    print("\n" + "=" * 50)
    print("Tests completed!")
    print("=" * 50)


if __name__ == "__main__":
    main()

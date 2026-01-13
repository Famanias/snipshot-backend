"""
Test script for VM Translator API
Sends image from local PC to VM, gets Cloudinary URL back.
"""

import httpx
import asyncio
import json

# VM Configuration
VM_IP = "34.87.58.21"
VM_PORT = "8000"
VM_URL = f"http://{VM_IP}:{VM_PORT}"


async def test_vm_translator():
    print("=" * 60)
    print("Testing VM Translator API")
    print(f"Target: {VM_URL}")
    print("=" * 60)
    
    async with httpx.AsyncClient(timeout=180.0) as client:
        
        # 1. Health check
        print("\n[1] Health check...")
        try:
            resp = await client.get(f"{VM_URL}/health")
            if resp.status_code == 200:
                print(f"    ✓ VM is healthy: {resp.json()}")
            else:
                print(f"    ✗ Health check failed: {resp.status_code}")
                return
        except httpx.ConnectError as e:
            print(f"    ✗ Cannot connect to VM: {e}")
            print(f"    → Make sure VM is running and port 8000 is open")
            return
        
        # 2. Test translation with Cloudinary upload
        print("\n[2] Testing /translate (Cloudinary upload)...")
        
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
        
        # Load test image
        try:
            with open("15.jpg", "rb") as f:
                image_bytes = f.read()
            print(f"    → Using 15.jpg ({len(image_bytes)} bytes)")
        except FileNotFoundError:
            print("    ✗ 15.jpg not found!")
            print("    → Please make sure 15.jpg exists in the current directory")
            return
        
        files = {"image": ("test.jpg", image_bytes, "image/jpeg")}
        data = {"config": json.dumps(config)}
        
        print("    → Sending image to VM...")
        print("    → This may take 1-2 minutes (translation + Cloudinary upload)...")
        
        try:
            resp = await client.post(
                f"{VM_URL}/translate",
                files=files,
                data=data
            )
            
            if resp.status_code == 200:
                result = resp.json()
                print(f"\n    ✓ Translation successful!")
                print(f"    ┌─────────────────────────────────────")
                print(f"    │ Success: {result.get('success')}")
                print(f"    │ Image URL: {result.get('image_url')}")
                print(f"    │ Public ID: {result.get('public_id')}")
                print(f"    └─────────────────────────────────────")
                
                # Save the URL for reference
                with open("vm_test_result.json", "w") as f:
                    json.dump(result, f, indent=2)
                print(f"\n    → Result saved to vm_test_result.json")
                
            else:
                print(f"    ✗ Translation failed: {resp.status_code}")
                print(f"    → Response: {resp.text}")
                return
                
        except httpx.ReadTimeout:
            print(f"    ✗ Request timed out (>180s)")
            print(f"    → The translation might still be running on the VM")
            return
        
        # 3. Test /translate/raw endpoint
        print("\n[3] Testing /translate/raw (raw PNG)...")
        
        try:
            resp = await client.post(
                f"{VM_URL}/translate/raw",
                files={"image": ("test.jpg", image_bytes, "image/jpeg")},
                data={"config": json.dumps(config)}
            )
            
            if resp.status_code == 200:
                content_type = resp.headers.get("content-type", "")
                if "image/png" in content_type:
                    print(f"    ✓ Got raw PNG ({len(resp.content)} bytes)")
                    with open("vm_test_output.png", "wb") as f:
                        f.write(resp.content)
                    print(f"    → Saved to vm_test_output.png")
                else:
                    print(f"    ✗ Unexpected content type: {content_type}")
            else:
                print(f"    ✗ Raw translation failed: {resp.status_code}")
                
        except httpx.ReadTimeout:
            print(f"    ✗ Request timed out")
        
        # Summary
        print("\n" + "=" * 60)
        print("✓ VM Translator API Test Complete!")
        print("=" * 60)
        print("\nYour architecture is working:")
        print("  Local PC → VM (34.87.58.21:8000) → Cloudinary")
        print("\nNext step:")
        print("  Frontend can now save the image_url to Database API")


if __name__ == "__main__":
    asyncio.run(test_vm_translator())

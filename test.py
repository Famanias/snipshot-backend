import aiohttp
import asyncio
import json
from pathlib import Path

VM_API_URL = "http://34.87.58.21:8000/translate"

async def test_vm_translate():
    image_path = Path("13.jpg")
    assert image_path.exists(), "15.jpg not found"

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
        image_path.open("rb"),
        filename="15.jpg",
        content_type="image/jpeg"
    )
    form.add_field(
        "config",
        json.dumps(config_json),
        content_type="application/json"
    )

    async with aiohttp.ClientSession() as session:
        async with session.post(VM_API_URL, data=form) as resp:
            if resp.status != 200:
                print("❌ Error:", resp.status)
                print(await resp.text())
                return

            img_bytes = await resp.read()

    with open("output.png", "wb") as f:
        f.write(img_bytes)

    print("✅ Saved output.png (from VM backend)")

if __name__ == "__main__":
    asyncio.run(test_vm_translate())

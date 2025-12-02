import aiohttp
import asyncio
import pickle
from PIL import Image
from manga_translator import Config

async def test_local_translate():
    # 1. Load an image
    img = Image.open("15.jpg")

    # 2. JSON-style config (same as your web UI)
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

    # Convert JSON → Config object
    config = Config(**config_json)

    # 3. Serialize for internal API
    payload = pickle.dumps({"image": img, "config": config})

    url = "http://127.0.0.1:8001/simple_execute/translate"

    async with aiohttp.ClientSession() as session:
        async with session.post(url, data=payload) as resp:
            if resp.status != 200:
                print("Error:", await resp.text())
                return

            # 4. Unpickle returned Context() object
            translated = pickle.loads(await resp.read())

            # 5. Save result
            translated.result.save("output.png")
            print("Saved output.png")

asyncio.run(test_local_translate())
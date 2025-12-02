# import os
# from fastapi import FastAPI
# from fastapi.responses import JSONResponse

# app = FastAPI()

# PORT = os.getenv("PORT")  # Render injects this
# BACKEND_PORT = os.getenv("BACKEND_PORT", "8001")

# @app.get("/")
# def read_root():
#     return {
#         "status": "ok",
#         "message": "Port binding test successful",
#         "PORT": PORT,
#         "BACKEND_PORT": BACKEND_PORT
#     }

# @app.get("/health")
# def health():
#     return JSONResponse({"ok": True})


import os
import pickle
import aiohttp
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import Response
from manga_translator import Config
from PIL import Image
import io
import json

app = FastAPI()

# The internal backend URL (shared mode server)
BACKEND_PORT = os.getenv("BACKEND_PORT", "8001")
BACKEND_URL = f"http://127.0.0.1:{BACKEND_PORT}/simple_execute/translate"


@app.post("/translate")
async def translate(
    image: UploadFile = File(...),
    config: str = Form(...)
):
    """
    - image: multipart/form-data file
    - config: JSON string
    """

    # 1. Load the uploaded image into PIL
    img_bytes = await image.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    # 2. Convert JSON string into Config object
    config_json = json.loads(config)
    config_obj = Config(**config_json)

    # 3. Serialize payload for the translator backend
    payload = pickle.dumps({
        "image": img,
        "config": config_obj
    })

    # 4. Send request to manga_translator backend
    async with aiohttp.ClientSession() as session:
        async with session.post(BACKEND_URL, data=payload) as resp:
            if resp.status != 200:
                return {"error": await resp.text()}

            # 5. Response is pickled Context
            ctx = pickle.loads(await resp.read())

    # 6. Convert output image to bytes
    output_buffer = io.BytesIO()
    ctx.result.save(output_buffer, format="PNG")
    output_buffer.seek(0)

    # 7. Return as image/png
    return Response(content=output_buffer.getvalue(), media_type="image/png")
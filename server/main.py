# server/main.py
import os
from fastapi import FastAPI
from fastapi.responses import JSONResponse

app = FastAPI()

PORT = os.getenv("PORT")  # Render injects this
BACKEND_PORT = os.getenv("BACKEND_PORT", "8001")

@app.get("/")
def read_root():
    return {
        "status": "ok",
        "message": "Render port binding test successful",
        "PORT": PORT,
        "BACKEND_PORT": BACKEND_PORT
    }

@app.get("/health")
def health():
    return JSONResponse({"ok": True})
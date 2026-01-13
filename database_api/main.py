"""
SnipShot Database API - User Management & Image Storage

This service handles:
- User registration/login
- JWT authentication
- Saving image URLs (from Cloudinary) to user accounts
- Listing user's images
- Deleting images

Deployed on Render with PostgreSQL.
"""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from .database import init_db
from .routes import users_router, images_router

load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown"""
    await init_db()
    print("✅ Database initialized")
    yield
    print("👋 Shutting down...")


app = FastAPI(
    title="SnipShot Database API",
    description="User management and image storage service",
    version="1.0.0",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routes
app.include_router(users_router, prefix="/api")
app.include_router(images_router, prefix="/api")


@app.get("/")
def root():
    return {
        "service": "SnipShot Database API",
        "version": "1.0.0",
        "endpoints": {
            "/api/users/register": "POST - Register new user",
            "/api/users/login": "POST - Login",
            "/api/users/me": "GET - Get profile",
            "/api/images": "GET - List user's images",
            "/api/images": "POST - Save image URL to account",
            "/api/images/{id}": "DELETE - Delete image"
        }
    }


@app.get("/health")
def health():
    return {"ok": True, "service": "database-api"}

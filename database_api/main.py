"""
SnipShot Database API - Supabase Edition

This service handles:
- User registration/login (via Supabase Auth)
- JWT authentication (Supabase tokens)
- Folder management (organize translations)
- Image storage (Supabase Storage)
- Image metadata CRUD (Supabase PostgreSQL)

All-in-one Supabase backend.
"""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from supabase import create_client, Client

from database import init_db
from routes import users_router, images_router, folders_router

load_dotenv()

# Initialize Supabase client
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_ANON_KEY")
supabase: Client = create_client(supabase_url, supabase_key) if supabase_url and supabase_key else None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown"""
    await init_db()
    if supabase:
        print("✅ Supabase connected")
    else:
        print("⚠️ Supabase not configured (check SUPABASE_URL and SUPABASE_ANON_KEY)")
    print("✅ Database initialized")
    yield
    print("👋 Shutting down...")


app = FastAPI(
    title="SnipShot Database API",
    description="User management, folder organization, and image storage via Supabase",
    version="2.1.0",
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
app.include_router(folders_router, prefix="/api")
app.include_router(images_router, prefix="/api")


@app.get("/")
def root():
    return {
        "service": "SnipShot Database API",
        "version": "2.1.0",
        "backend": "Supabase",
        "endpoints": {
            "users": {
                "POST /api/users/register": "Register new user",
                "POST /api/users/login": "Login → JWT",
                "GET /api/users/me": "Get profile"
            },
            "folders": {
                "POST /api/folders": "Create folder",
                "GET /api/folders": "List folders",
                "GET /api/folders/{id}": "Get folder with images",
                "PUT /api/folders/{id}": "Update folder",
                "DELETE /api/folders/{id}": "Delete folder"
            },
            "images": {
                "POST /api/images": "Upload image",
                "POST /api/images/from-url": "Save from URL",
                "GET /api/images": "List images",
                "GET /api/images/{id}": "Get image details",
                "PUT /api/images/{id}": "Update image",
                "DELETE /api/images/{id}": "Delete image"
            }
        }
    }


@app.get("/health")
def health():
    return {"ok": True, "service": "database-api", "backend": "supabase"}

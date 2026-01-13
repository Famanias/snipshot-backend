# SnipShot Database API Documentation

> **Version:** 2.1.0  
> **Backend:** Supabase (Auth + Storage + PostgreSQL)  
> **Created:** January 2026

---

## Table of Contents

1. [Overview](#1-overview)
2. [Architecture](#2-architecture)
3. [Supabase Configuration](#3-supabase-configuration)
4. [Database Schema](#4-database-schema)
5. [Authentication Flow](#5-authentication-flow)
6. [API Endpoints](#6-api-endpoints)
7. [Folder Structure](#7-folder-structure)
8. [End-to-End Test](#8-end-to-end-test)
9. [Environment Variables](#9-environment-variables)
10. [Security Considerations](#10-security-considerations)

---

## 1. Overview

SnipShot is a manga/image translation application. The Database API handles:

- **User Authentication** - Registration, login, JWT tokens via Supabase Auth
- **Folder Management** - Create, organize, and manage folders for translations
- **Image Storage** - Store translated images in Supabase Storage
- **Image Metadata** - Track image metadata in PostgreSQL

### Complete Workflow

```
User → Register → Login → Create Folder → Upload Image → VM Translation 
    → Supabase Storage → Save Metadata to PostgreSQL → User's Folder
```

---

## 2. Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         SnipShot System                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────┐     ┌─────────────────────────────────────────┐  │
│  │   Frontend   │────▶│           Database API (:8002)          │  │
│  │   (Coming)   │     │  ┌────────┐ ┌────────┐ ┌─────────────┐  │  │
│  └──────────────┘     │  │ Users  │ │Folders │ │   Images    │  │  │
│         │             │  │ Router │ │ Router │ │   Router    │  │  │
│         │             │  └────────┘ └────────┘ └─────────────┘  │  │
│         │             │        │         │            │         │  │
│         │             │        └─────────┴────────────┘         │  │
│         │             │                  │                      │  │
│         │             │        ┌─────────┴─────────┐           │  │
│         │             │        ▼                   ▼           │  │
│         │             │  ┌──────────┐      ┌────────────┐      │  │
│         │             │  │Supabase  │      │ SQLAlchemy │      │  │
│         │             │  │ Client   │      │  (asyncpg) │      │  │
│         │             │  └──────────┘      └────────────┘      │  │
│         │             └─────────────────────────────────────────┘  │
│         │                       │                   │              │
│         │                       ▼                   ▼              │
│         │             ┌─────────────────────────────────────────┐  │
│         │             │              SUPABASE                   │  │
│         │             │  ┌────────┐ ┌─────────┐ ┌────────────┐  │  │
│         │             │  │  Auth  │ │ Storage │ │ PostgreSQL │  │  │
│         │             │  │(ES256) │ │(images) │ │  (tables)  │  │  │
│         │             │  └────────┘ └─────────┘ └────────────┘  │  │
│         │             └─────────────────────────────────────────┘  │
│         │                                                          │
│         ▼                                                          │
│  ┌──────────────┐     ┌─────────────────────────────────────────┐  │
│  │ VM Translator│────▶│  manga_translator + translator_api.py  │  │
│  │    (:8000)   │     │     (Uploads to Supabase Storage)       │  │
│  └──────────────┘     └─────────────────────────────────────────┘  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Services

| Service | Port | Description |
|---------|------|-------------|
| Database API | 8002 | User management, folders, images |
| VM Translator | 8000 | Image translation endpoint |
| manga_translator | 8001 | Core translation engine |

---

## 3. Supabase Configuration

### Required Supabase Setup

1. **Create a Supabase Project** at [supabase.com](https://supabase.com)

2. **Authentication Settings**
   - Go to **Authentication > Email Templates**
   - (Optional) Disable "Confirm email" for testing

3. **Create Storage Bucket**
   ```
   Storage > New Bucket > "images"
   ├── Public: Yes (for public URLs)
   └── File size limit: 50MB
   ```

4. **Run SQL Migration**
   - Go to **SQL Editor**
   - Run `migrations/001_create_tables.sql`

5. **Get API Keys** (Settings > API)
   - Project URL
   - anon/public key
   - service_role key (keep secret!)
   - JWT Secret

### Storage Policies

The `images` bucket uses public access with authenticated uploads:

```sql
-- Allow authenticated users to upload
CREATE POLICY "Authenticated users can upload"
ON storage.objects FOR INSERT TO authenticated
WITH CHECK (bucket_id = 'images');

-- Allow authenticated users to update their files
CREATE POLICY "Users can update own files"
ON storage.objects FOR UPDATE TO authenticated
USING (bucket_id = 'images' AND (storage.foldername(name))[1] = auth.uid()::text);

-- Allow authenticated users to delete their files
CREATE POLICY "Users can delete own files"
ON storage.objects FOR DELETE TO authenticated
USING (bucket_id = 'images' AND (storage.foldername(name))[1] = auth.uid()::text);
```

---

## 4. Database Schema

### Entity Relationship Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    SUPABASE AUTH (managed)                      │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ auth.users                                                │  │
│  │  - id (UUID, PK)     ← Referenced by folders & images     │  │
│  │  - email                                                  │  │
│  │  - encrypted_password                                     │  │
│  │  - created_at                                             │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ user_id (VARCHAR 36)
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                        PUBLIC SCHEMA                            │
│                                                                 │
│  ┌────────────────────────┐     ┌────────────────────────────┐  │
│  │       folders          │     │          images            │  │
│  ├────────────────────────┤     ├────────────────────────────┤  │
│  │ id (SERIAL, PK)        │◀────│ folder_id (FK, nullable)   │  │
│  │ user_id (VARCHAR 36)   │     │ id (SERIAL, PK)            │  │
│  │ name (VARCHAR 100)     │     │ user_id (VARCHAR 36)       │  │
│  │ description (TEXT)     │     │ storage_path (TEXT)        │  │
│  │ created_at (TIMESTAMP) │     │ public_url (TEXT)          │  │
│  │ updated_at (TIMESTAMP) │     │ filename (VARCHAR 255)     │  │
│  └────────────────────────┘     │ original_filename          │  │
│           │                     │ source_language (10)       │  │
│           │ 1:N                 │ target_language (10)       │  │
│           ▼                     │ file_size (INTEGER)        │  │
│      ┌─────────┐               │ created_at (TIMESTAMP)     │  │
│      │ images  │               │ updated_at (TIMESTAMP)     │  │
│      └─────────┘               └────────────────────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Folders Table

```sql
CREATE TABLE folders (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(36) NOT NULL,         -- Supabase Auth UUID
    name VARCHAR(100) NOT NULL,           -- Folder display name
    description TEXT,                     -- Optional description
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Indexes
CREATE INDEX idx_folders_user_id ON folders(user_id);
CREATE UNIQUE INDEX idx_folders_user_name ON folders(user_id, name);  -- No duplicate names per user
```

### Images Table

```sql
CREATE TABLE images (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(36) NOT NULL,                         -- Supabase Auth UUID
    folder_id INTEGER REFERENCES folders(id) ON DELETE SET NULL,  -- Optional folder
    storage_path TEXT NOT NULL,                           -- Path in Supabase Storage
    public_url TEXT NOT NULL,                             -- Public URL for access
    filename VARCHAR(255) NOT NULL,                       -- User-editable name
    original_filename VARCHAR(255),                       -- Original from upload
    source_language VARCHAR(10),                          -- e.g., 'JPN'
    target_language VARCHAR(10),                          -- e.g., 'ENG'
    file_size INTEGER,                                    -- Size in bytes
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Indexes
CREATE INDEX idx_images_user_id ON images(user_id);
CREATE INDEX idx_images_folder_id ON images(folder_id);
CREATE INDEX idx_images_created_at ON images(created_at DESC);
```

### SQLAlchemy Models

Located in `database/models.py`:

```python
class Folder(Base):
    __tablename__ = "folders"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String(36), nullable=False, index=True)
    name = Column(String(100), nullable=False)
    description = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    images = relationship("Image", back_populates="folder", cascade="all, delete-orphan")


class Image(Base):
    __tablename__ = "images"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String(36), nullable=False, index=True)
    folder_id = Column(Integer, ForeignKey("folders.id"), nullable=True, index=True)
    storage_path = Column(Text, nullable=False)
    public_url = Column(Text, nullable=False)
    filename = Column(String(255), nullable=False)
    original_filename = Column(String(255), nullable=True)
    source_language = Column(String(10), nullable=True)
    target_language = Column(String(10), nullable=True)
    file_size = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    folder = relationship("Folder", back_populates="images")
```

---

## 5. Authentication Flow

### JWT Token Flow

```
┌────────────┐                ┌──────────────┐                ┌───────────────┐
│   Client   │                │ Database API │                │ Supabase Auth │
└─────┬──────┘                └──────┬───────┘                └───────┬───────┘
      │                              │                                │
      │  POST /api/users/register    │                                │
      │ ───────────────────────────▶ │                                │
      │                              │   supabase.auth.sign_up()      │
      │                              │ ─────────────────────────────▶ │
      │                              │                                │
      │                              │ ◀───────── user + session ──── │
      │ ◀─── access_token (ES256) ── │                                │
      │                              │                                │
      │  POST /api/users/login       │                                │
      │ ───────────────────────────▶ │                                │
      │                              │ sign_in_with_password()        │
      │                              │ ─────────────────────────────▶ │
      │                              │                                │
      │                              │ ◀───────── session ─────────── │
      │ ◀─── access_token (ES256) ── │                                │
      │                              │                                │
      │  GET /api/folders            │                                │
      │  Authorization: Bearer <tok> │                                │
      │ ───────────────────────────▶ │                                │
      │                              │  supabase.auth.get_user(token) │
      │                              │ ─────────────────────────────▶ │
      │                              │                                │
      │                              │ ◀──── user_id (validated) ──── │
      │                              │                                │
      │ ◀─── folders data ────────── │                                │
      │                              │                                │
```

### Token Verification (auth/security.py)

```python
def decode_supabase_token(token: str) -> Optional[dict]:
    """
    Decode and verify a Supabase access token.
    
    Supabase uses ES256 (ECC P-256) for JWT signing.
    We use supabase.auth.get_user(token) for verification.
    """
    # Check algorithm from token header
    header = jwt.get_unverified_header(token)
    algorithm = header.get("alg", "HS256")
    
    # For HS256, try JWT secret (legacy)
    if algorithm == "HS256" and SUPABASE_JWT_SECRET:
        try:
            payload = jwt.decode(token, SUPABASE_JWT_SECRET, algorithms=["HS256"])
            return payload
        except jwt.InvalidTokenError:
            pass
    
    # For ES256, use Supabase client to verify
    supabase = get_supabase()
    user_response = supabase.auth.get_user(token)
    
    if user_response and user_response.user:
        return {
            "sub": user_response.user.id,  # User UUID
            "email": user_response.user.email,
            "aud": "authenticated"
        }
    
    return None
```

### Auth Dependencies (auth/dependencies.py)

```python
async def get_current_user_id(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> str:
    """Extract and validate user ID from JWT token"""
    
    payload = decode_supabase_token(credentials.credentials)
    if payload is None:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    
    return payload.get("sub")  # Supabase uses 'sub' for user ID
```

---

## 6. API Endpoints

### Users API

| Method | Endpoint | Description | Auth |
|--------|----------|-------------|------|
| POST | `/api/users/register` | Register new user | No |
| POST | `/api/users/login` | Login, get JWT | No |
| GET | `/api/users/me` | Get profile | Yes |
| POST | `/api/users/logout` | Logout | Yes |

#### Register User
```http
POST /api/users/register
Content-Type: application/json

{
  "email": "user@example.com",
  "password": "securepassword123"
}
```

Response:
```json
{
  "access_token": "eyJhbGciOiJFUzI1NiIs...",
  "token_type": "bearer",
  "expires_in": 3600,
  "refresh_token": "...",
  "user": {
    "id": "ac48b591-ce23-4914-803d-d79088fe1a82",
    "email": "user@example.com"
  }
}
```

#### Login
```http
POST /api/users/login
Content-Type: application/json

{
  "email": "user@example.com",
  "password": "securepassword123"
}
```

---

### Folders API

| Method | Endpoint | Description | Auth |
|--------|----------|-------------|------|
| POST | `/api/folders` | Create folder | Yes |
| GET | `/api/folders` | List folders | Yes |
| GET | `/api/folders/{id}` | Get folder with images | Yes |
| PUT | `/api/folders/{id}` | Update folder | Yes |
| DELETE | `/api/folders/{id}` | Delete folder | Yes |

#### Create Folder
```http
POST /api/folders
Authorization: Bearer <token>
Content-Type: application/json

{
  "name": "My Translations",
  "description": "Translated manga chapters"
}
```

Response:
```json
{
  "id": 1,
  "name": "My Translations",
  "description": "Translated manga chapters",
  "image_count": 0,
  "created_at": "2026-01-13T10:00:00Z",
  "updated_at": "2026-01-13T10:00:00Z"
}
```

#### Delete Folder
```http
DELETE /api/folders/1?delete_images=false
Authorization: Bearer <token>
```

- `delete_images=false` (default): Images moved to "unfiled"
- `delete_images=true`: Images permanently deleted from storage

---

### Images API

| Method | Endpoint | Description | Auth |
|--------|----------|-------------|------|
| POST | `/api/images` | Upload image file | Yes |
| POST | `/api/images/from-url` | Save from URL | Yes |
| GET | `/api/images` | List images | Yes |
| GET | `/api/images/{id}` | Get image details | Yes |
| PUT | `/api/images/{id}` | Update image | Yes |
| DELETE | `/api/images/{id}` | Delete image | Yes |

#### Upload Image
```http
POST /api/images
Authorization: Bearer <token>
Content-Type: multipart/form-data

file: <binary>
folder_id: 1
filename: "chapter1_page1.png"
source_language: "JPN"
target_language: "ENG"
```

#### Save from URL (after VM translation)
```http
POST /api/images/from-url
Authorization: Bearer <token>
Content-Type: application/x-www-form-urlencoded

image_url=https://supabase.co/storage/v1/.../image.png
folder_id=1
filename=translated_page.png
source_language=JPN
target_language=ENG
```

#### List Images with Filtering
```http
GET /api/images?page=1&per_page=20&folder_id=1
Authorization: Bearer <token>
```

Query Parameters:
- `folder_id=N` - Filter by folder
- `folder_id=0` - Unfiled images only
- `folder_id` (omit) - All images

#### Update Image (rename/move)
```http
PUT /api/images/1
Authorization: Bearer <token>
Content-Type: application/json

{
  "filename": "new_name.png",
  "folder_id": 2
}
```

---

## 7. Folder Structure

```
database_api/
├── main.py                 # FastAPI app entry point
├── config.py               # Supabase client configuration
├── schemas.py              # Pydantic request/response schemas
├── requirements.txt        # Python dependencies
├── .env                    # Environment variables (not in git)
├── .env.example            # Example environment template
│
├── auth/
│   ├── __init__.py         # Auth module exports
│   ├── security.py         # JWT token verification
│   └── dependencies.py     # FastAPI auth dependencies
│
├── database/
│   ├── __init__.py         # Database module exports
│   ├── models.py           # SQLAlchemy ORM models
│   └── connection.py       # Async database connection
│
├── routes/
│   ├── __init__.py         # Router exports
│   ├── users.py            # User registration/login
│   ├── folders.py          # Folder CRUD operations
│   └── images.py           # Image upload/management
│
└── migrations/
    └── 001_create_tables.sql   # Database schema SQL
```

### Key Files Explained

#### main.py
The FastAPI application entry point:
- Initializes Supabase client
- Sets up CORS middleware
- Includes routers (users, folders, images)
- Defines health check endpoints

#### config.py
Supabase configuration:
```python
@lru_cache()
def get_supabase() -> Client:
    """Cached Supabase client"""
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_KEY")
    return create_client(url, key)

STORAGE_BUCKET = os.getenv("SUPABASE_STORAGE_BUCKET", "images")
```

#### database/connection.py
Async PostgreSQL connection via SQLAlchemy:
```python
# Handle Supabase PostgreSQL URL format
DATABASE_URL = os.getenv("DATABASE_URL")
if DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://", 1)

engine = create_async_engine(DATABASE_URL, echo=False)
AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession)
```

---

## 8. End-to-End Test

The `test_end_to_end.py` script demonstrates the complete workflow:

### Test Flow

```
Step 1: Register User
    ↓
Step 2: Login → Get JWT Token
    ↓
Step 3: Create Folder ("My Translations")
    ↓
Step 4: Send Image to VM Translator
    ↓
Step 5: Save Translated Image to Database
    ↓
Step 6: Verify → List User's Images
```

### Running the Test

```powershell
# Prerequisites
# Terminal 1: Start VM translator
python main.py

# Terminal 2: Start translator API
python translator_api.py

# Terminal 3: Start database API
cd database_api
python -m uvicorn main:app --reload --port 8002

# Terminal 4: Run test
python test_end_to_end.py
```

### Expected Output

```
======================================================================
SnipShot End-to-End Test
Register → Login → Translate → Save to Database
======================================================================

[1] REGISTER USER
--------------------------------------------------
    Email: user@example.com
    ✓ Registration successful!

[2] LOGIN
--------------------------------------------------
    ✓ Login successful!
    → Token: eyJhbGciOiJFUzI1NiIs...

[3] CREATE FOLDER
--------------------------------------------------
    ✓ Created folder: My Translations (id=1)

[4] TRANSLATE IMAGE (VM)
--------------------------------------------------
    → Loaded 15.jpg (144,066 bytes)
    → Sending to VM translator...
    → This may take 1-2 minutes...
    ✓ Translation complete!
    → Supabase URL: https://xxx.supabase.co/storage/v1/...

[5] SAVE TO DATABASE
--------------------------------------------------
    → Saving to folder: 1
    ✓ Image saved to database!
    → Image ID: 1
    → Storage Path: user-id/timestamp_filename.png

[6] VERIFY - LIST IMAGES
--------------------------------------------------
    ✓ User has 1 image(s)
      - 15_translated.png (folder=1)

======================================================================
✓ END-TO-END TEST COMPLETE!
======================================================================
```

### Test Configuration

```python
# API URLs
VM_TRANSLATOR_URL = "http://localhost:8000"
DATABASE_API_URL = "http://localhost:8002/api"

# Test credentials
TEST_EMAIL = "your-email@example.com"
TEST_PASSWORD = "your-password"
```

---

## 9. Environment Variables

Create `.env` from `.env.example`:

```bash
# ===========================================
# SUPABASE SETTINGS (Required)
# ===========================================

# Project URL (Settings > API > Project URL)
SUPABASE_URL=https://your-project-id.supabase.co

# Anon/Public Key (Settings > API > anon public)
SUPABASE_ANON_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...

# Service Role Key (Settings > API > service_role) - KEEP SECRET!
SUPABASE_SERVICE_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...

# JWT Secret (Settings > API > JWT Settings)
SUPABASE_JWT_SECRET=your-jwt-secret

# Storage bucket name
SUPABASE_STORAGE_BUCKET=images

# ===========================================
# DATABASE (Session Pooler)
# ===========================================

# PostgreSQL connection string
# Settings > Database > Connection string > Session pooler
DATABASE_URL=postgresql://postgres.xxx:password@xxx.pooler.supabase.com:5432/postgres
```

### Where to Find Values in Supabase Dashboard

| Variable | Location |
|----------|----------|
| SUPABASE_URL | Settings > API > Project URL |
| SUPABASE_ANON_KEY | Settings > API > Project API keys > anon public |
| SUPABASE_SERVICE_KEY | Settings > API > Project API keys > service_role |
| SUPABASE_JWT_SECRET | Settings > API > JWT Settings > JWT Secret |
| DATABASE_URL | Settings > Database > Connection string > Session |

---

## 10. Security Considerations

### RLS (Row Level Security)

RLS is **disabled** on `folders` and `images` tables because:

1. **Direct PostgreSQL connections** via SQLAlchemy/asyncpg don't support `auth.uid()`
2. Supabase's `auth.uid()` function only works with Supabase client connections
3. Our API handles security at the application layer

### How Security is Enforced

```python
# Every protected route uses this dependency
user_id: str = Depends(get_current_user_id)

# All queries filter by authenticated user_id
result = await db.execute(
    select(Image).where(
        Image.id == image_id,
        Image.user_id == user_id  # ← Security filter
    )
)
```

### Security Checklist

- ✅ JWT tokens verified on every request via `supabase.auth.get_user()`
- ✅ All database queries filtered by `user_id`
- ✅ Users can only access their own folders and images
- ✅ Service role key used only on backend (never exposed to client)
- ✅ Storage paths include user_id prefix for isolation
- ⚠️ CORS configured as `*` for development (restrict in production)

### Production Recommendations

1. **Restrict CORS origins** to your frontend domain
2. **Enable HTTPS** for all API endpoints
3. **Use environment-specific secrets** (don't share between dev/prod)
4. **Monitor failed auth attempts** via Supabase Auth logs
5. **Set up rate limiting** on registration/login endpoints

---

## Quick Reference

### Start All Services

```powershell
# Terminal 1: VM Translator (port 8001)
python main.py

# Terminal 2: Translator API (port 8000)
python translator_api.py

# Terminal 3: Database API (port 8002)
cd database_api
python -m uvicorn main:app --reload --port 8002
```

### Common SQL Commands

```sql
-- View all folders for a user
SELECT * FROM folders WHERE user_id = 'uuid-here';

-- View all images in a folder
SELECT * FROM images WHERE folder_id = 1;

-- Count images per folder
SELECT f.name, COUNT(i.id) as image_count 
FROM folders f 
LEFT JOIN images i ON f.id = i.folder_id 
GROUP BY f.id;

-- Rename a folder
UPDATE folders SET name = 'New Name', updated_at = NOW() WHERE id = 1;

-- Rename an image
UPDATE images SET filename = 'new_name.png', updated_at = NOW() WHERE id = 1;

-- Move image to different folder
UPDATE images SET folder_id = 2, updated_at = NOW() WHERE id = 1;
```

---

*Documentation generated for SnipShot Database API v2.1.0*

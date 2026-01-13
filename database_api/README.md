# SnipShot Database API - Supabase Edition

User management, folder organization, and image storage service powered by Supabase.

## Overview

This service handles:
- **User Auth** - Registration, login via Supabase Auth
- **Folder Management** - Create folders to organize translations
- **Image Storage** - Upload/delete via Supabase Storage
- **Image Metadata** - CRUD via Supabase PostgreSQL

## Architecture

```
Frontend (Desktop/Mobile)
    │
    ├── VM Translator API (Google Cloud)
    │         └── Translates images → Returns translated PNG
    │
    └── Database API (this service)
              ├── Supabase Auth (users)
              ├── Supabase Storage (images)
              └── Supabase PostgreSQL (folders + metadata)
```

## Endpoints

### Users
| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| POST | `/api/users/register` | ❌ | Create account |
| POST | `/api/users/login` | ❌ | Login → JWT |
| GET | `/api/users/me` | ✅ | Get profile |

### Folders
| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| POST | `/api/folders` | ✅ | Create folder |
| GET | `/api/folders` | ✅ | List all folders |
| GET | `/api/folders/{id}` | ✅ | Get folder with images |
| PUT | `/api/folders/{id}` | ✅ | Update folder name/description |
| DELETE | `/api/folders/{id}` | ✅ | Delete folder |

### Images
| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| POST | `/api/images` | ✅ | Upload image (optional folder_id) |
| POST | `/api/images/from-url` | ✅ | Save image from URL |
| GET | `/api/images` | ✅ | List images (filter by folder_id) |
| GET | `/api/images/{id}` | ✅ | Get image details |
| PUT | `/api/images/{id}` | ✅ | Update filename or move to folder |
| DELETE | `/api/images/{id}` | ✅ | Delete image |

## Supabase Setup

### 1. Create Supabase Project
1. Go to [supabase.com](https://supabase.com) and create a project
2. Note down your project URL and API keys

### 2. Create Storage Bucket
1. Go to Storage in Supabase Dashboard
2. Create a new bucket called `images`
3. Set it to **Public** (for serving images)

### 3. Run Database Migration
1. Go to SQL Editor in Supabase Dashboard
2. Copy contents of `migrations/001_create_tables.sql`
3. Run the SQL to create `folders` and `images` tables

### 4. Get Credentials
From Settings > API:
- Project URL → `SUPABASE_URL`
- anon public key → `SUPABASE_ANON_KEY`
- service_role key → `SUPABASE_SERVICE_KEY`
- JWT Secret → `SUPABASE_JWT_SECRET`

From Settings > Database:
- Connection string (Session Pooler) → `DATABASE_URL`

## Local Development

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Create .env file
cp .env.example .env
# Edit .env with your Supabase credentials

# Run
uvicorn main:app --reload --port 8000
```

## Environment Variables

```bash
# Supabase
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_ANON_KEY=eyJ...
SUPABASE_SERVICE_KEY=eyJ...
SUPABASE_JWT_SECRET=your-jwt-secret
SUPABASE_STORAGE_BUCKET=images

# Database
DATABASE_URL=postgresql+asyncpg://...
```

## Deployment

This is a standalone FastAPI service. Deploy to any platform:

### Option 1: Fly.io (Recommended)
```bash
fly launch
fly secrets set SUPABASE_URL=... SUPABASE_SERVICE_KEY=... DATABASE_URL=...
fly deploy
```

### Option 2: Railway
1. Connect GitHub repo
2. Add environment variables
3. Deploy automatically

### Option 3: Any VPS/Cloud
```bash
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```

### Required Environment Variables
All from your Supabase Dashboard:
- `SUPABASE_URL`
- `SUPABASE_ANON_KEY`
- `SUPABASE_SERVICE_KEY`
- `SUPABASE_JWT_SECRET`
- `SUPABASE_STORAGE_BUCKET`
- `DATABASE_URL` (PostgreSQL pooler connection string)

## Project Structure

```
database_api/
├── main.py              # FastAPI app
├── config.py            # Supabase client
├── schemas.py           # Pydantic models
├── requirements.txt     # Dependencies
├── auth/
│   ├── security.py      # JWT verification
│   └── dependencies.py  # Auth middleware
├── database/
│   ├── connection.py    # PostgreSQL connection
│   └── models.py        # SQLAlchemy Image model
└── routes/
    ├── users.py         # Auth endpoints
    └── images.py        # Image CRUD
```

## API Usage

### Register
```bash
curl -X POST http://localhost:8000/api/users/register \
  -H "Content-Type: application/json" \
  -d '{"email":"user@example.com","password":"pass123"}'
```

### Login
```bash
curl -X POST http://localhost:8000/api/users/login \
  -H "Content-Type: application/json" \
  -d '{"email":"user@example.com","password":"pass123"}'

# Response: {"access_token": "eyJ...", ...}
```

### Upload Image
```bash
curl -X POST http://localhost:8000/api/images \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -F "file=@image.png" \
  -F "original_filename=my_image.png"
```

### Save Image from URL
```bash
curl -X POST http://localhost:8000/api/images/from-url \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -F "image_url=https://..." \
  -F "original_filename=translated.png"
```

### List Images
```bash
curl http://localhost:8000/api/images \
  -H "Authorization: Bearer YOUR_TOKEN"
```

## Frontend Integration

```javascript
// 1. Register/Login with Supabase (or via this API)
const { access_token } = await fetch('/api/users/login', {
  method: 'POST',
  body: JSON.stringify({ email, password })
}).then(r => r.json());

// 2. Translate image via VM
const translated = await fetch('http://VM_IP:8000/translate', {
  method: 'POST',
  body: formData
}).then(r => r.json());

// 3. Save to user's account
await fetch('/api/images/from-url', {
  method: 'POST',
  headers: { 'Authorization': `Bearer ${access_token}` },
  body: new URLSearchParams({
    image_url: translated.image_url,
    original_filename: 'translated.png'
  })
});

// 4. List saved images
const { images } = await fetch('/api/images', {
  headers: { 'Authorization': `Bearer ${access_token}` }
}).then(r => r.json());
```

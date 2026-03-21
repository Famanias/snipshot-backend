# SnipShot Backend

Image translation service with user accounts and cloud storage.

## Architecture

```
[ Mobile / Desktop Client ]
           |
           | HTTP Request (with optional JWT)
           v
[ Google Cloud VM ]
  ├── FastAPI Wrapper (0.0.0.0:8000)     ←── Public API
  │     ├── User Authentication (JWT)
  │     ├── Cloudinary Upload
  │     └── PostgreSQL (metadata)
  │
  └── Manga Translator Backend (127.0.0.1:8001)  ←── Internal only
        ├── Detection
        ├── OCR
        ├── Translation (Groq)
        ├── Inpainting
        └── Rendering
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements-2.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env with your values
```

### 3. Run Both Servers

```bash
python main.py
```

This starts:
- Manga Translator Backend on `127.0.0.1:8001`
- FastAPI Wrapper on `0.0.0.0:8000`

## API Endpoints

### Authentication

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/users/register` | POST | Create new account |
| `/api/users/login` | POST | Login, get JWT token |
| `/api/users/me` | GET | Get current user profile |
| `/api/users/me` | PUT | Update profile |
| `/api/users/change-password` | POST | Change password |
| `/api/users/me` | DELETE | Delete account |

### Translation

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/translate` | POST | Translate image (auth optional) |
| `/translate/url` | POST | Translate and get URL (auth required) |

### Images

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/images` | GET | List user's saved images |
| `/api/images/{id}` | GET | Get specific image |
| `/api/images/{id}` | DELETE | Delete specific image |
| `/api/images` | DELETE | Delete all images |

### Health

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API info |
| `/health` | GET | Health check |
| `/docs` | GET | Swagger UI |

## Usage Examples

### Register a New User

```bash
curl -X POST http://34.87.58.21:8000/api/users/register \
  -H "Content-Type: application/json" \
  -d '{"email": "user@example.com", "username": "myuser", "password": "secret123"}'
```

### Login

```bash
curl -X POST http://34.87.58.21:8000/api/users/login \
  -H "Content-Type: application/json" \
  -d '{"email": "user@example.com", "password": "secret123"}'

# Response: {"access_token": "eyJ...", "token_type": "bearer"}
```

### Translate (Anonymous - Returns Image)

```bash
curl -X POST http://34.87.58.21:8000/translate \
  -F "image=@manga.png" \
  -F 'config={"target_lang": "ENG"}' \
  --output translated.png
```

### Translate (Authenticated - Returns URL)

```bash
curl -X POST http://34.87.58.21:8000/translate \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -F "image=@manga.png" \
  -F 'config={"target_lang": "ENG"}' \
  -F "save=true"

# Response: {"success": true, "image_url": "https://res.cloudinary.com/...", "image_id": 1}
```

### List Saved Images

```bash
curl http://34.87.58.21:8000/api/images \
  -H "Authorization: Bearer YOUR_JWT_TOKEN"
```

### Delete an Image

```bash
curl -X DELETE http://34.87.58.21:8000/api/images/1 \
  -H "Authorization: Bearer YOUR_JWT_TOKEN"
```

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `DATABASE_URL` | Yes | PostgreSQL connection string |
| `SECRET_KEY` | Yes | JWT signing key (generate with `python -c "import secrets; print(secrets.token_hex(32))"`) |
| `CLOUDINARY_CLOUD_NAME` | Yes | Cloudinary cloud name |
| `CLOUDINARY_API_KEY` | Yes | Cloudinary API key |
| `CLOUDINARY_API_SECRET` | Yes | Cloudinary API secret |
| `BACKEND_PORT` | No | Translator backend port (default: 8001) |
| `PORT` | No | FastAPI wrapper port (default: 8000) |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | No | JWT expiry (default: 1440 = 24h) |
| `GROQ_API_KEY` | Yes | Groq API key for translation |

## Database Setup

### Local Development (SQLite)

```env
DATABASE_URL=sqlite+aiosqlite:///./snipshot.db
```

### Production (PostgreSQL on Render)

1. Create a PostgreSQL database on Render
2. Copy the connection string
3. Set in environment:

```env
DATABASE_URL=postgres://user:pass@host:5432/dbname
```

The app automatically converts `postgres://` to `postgresql+asyncpg://`.

## Cloudinary Setup

1. Sign up at [cloudinary.com](https://cloudinary.com/)
2. Go to Dashboard
3. Copy your Cloud Name, API Key, and API Secret
4. Add to `.env`

## Render Deployment

### Option 1: Blueprint (Recommended)

```bash
# Install Render CLI
npm install -g render

# Deploy
render blueprint sync
```

### Option 2: Manual

1. Create a new Web Service
2. Connect your GitHub repo
3. Set:
   - Build Command: `pip install -r requirements-2.txt`
   - Start Command: `python main.py`
4. Add environment variables

## Project Structure

```
snipshot-backend/
├── main.py                  # Orchestrator (starts both servers)
├── requirements-2.txt       # Dependencies
├── .env.example            # Environment template
├── render.yaml             # Render blueprint
├── server/
│   ├── __init__.py
│   ├── main.py             # FastAPI app
│   ├── schemas.py          # Pydantic models
│   ├── auth/
│   │   ├── __init__.py
│   │   ├── security.py     # Password hashing, JWT
│   │   └── dependencies.py # Auth dependencies
│   ├── database/
│   │   ├── __init__.py
│   │   ├── models.py       # SQLAlchemy models
│   │   └── connection.py   # DB connection
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── users.py        # User endpoints
│   │   └── images.py       # Image endpoints
│   └── storage/
│       ├── __init__.py
│       └── cloudinary_storage.py  # Cloudinary service
└── manga_translator/        # AI translation engine
```

## Translation Flow

### Anonymous User
```
POST /translate → FastAPI → Manga Backend → Return PNG directly
```

### Authenticated User (save=true)
```
POST /translate
  → FastAPI receives image + config
  → Forwards to Manga Backend (127.0.0.1:8001)
  → Receives translated image
  → Uploads to Cloudinary
  → Saves metadata to PostgreSQL
  → Returns JSON with URL + image_id
```

## Mobile/Desktop Integration

### iOS/Android

```swift
// Swift example
let url = URL(string: "http://34.87.58.21:8000/translate")!
var request = URLRequest(url: url)
request.httpMethod = "POST"
request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
// Add multipart form data with image and config
```

### Desktop (Python)

```python
import requests

response = requests.post(
    "http://34.87.58.21:8000/translate",
    headers={"Authorization": f"Bearer {token}"},
    files={"image": open("manga.png", "rb")},
    data={"config": '{"target_lang": "ENG"}', "save": "true"}
)

data = response.json()
print(f"Translated image URL: {data['image_url']}")
```

## License

MIT

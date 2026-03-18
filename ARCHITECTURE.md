# SnipShot Backend - Two-Backend Architecture

This project uses a **two-backend architecture** for clean separation of concerns:

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         FRONTENDS                                │
│                  (Desktop / Mobile Apps)                         │
└───────────────────────┬─────────────────────────────────────────┘
                        │
         ┌──────────────┴──────────────┐
         │                             │
         ▼                             ▼
┌─────────────────────┐      ┌─────────────────────┐
│  VM TRANSLATOR API  │      │    DATABASE API     │
│   (Google Cloud)    │      │      (Render)       │
│                     │      │                     │
│  • Translation      │      │  • User Auth (JWT)  │
│  • OCR Detection    │      │  • Image Metadata   │
│  • Text Rendering   │      │  • User Accounts    │
│  • Cloudinary Upload│      │  • PostgreSQL       │
│                     │      │                     │
│  Port: 8000         │      │  Port: 10000        │
│  IP: 34.87.58.21    │      │  (Render assigns)   │
└─────────────────────┘      └─────────────────────┘
         │                             │
         ▼                             │
┌─────────────────────┐                │
│     CLOUDINARY      │◄───────────────┘
│   (Image Storage)   │   (Deletion only)
└─────────────────────┘
```

## Request Flow

1. **User Authentication**
   ```
   Frontend → Database API (/api/users/login) → JWT Token
   ```

2. **Image Translation**
   ```
   Frontend → VM Translator API (/translate) → Cloudinary URL + Public ID
   ```

3. **Save Image Metadata**
   ```
   Frontend → Database API (/api/images) + JWT → Stores URL in DB
   ```

4. **Fetch User's Images**
   ```
   Frontend → Database API (/api/images) + JWT → List of image URLs
   ```

---

## 1. VM Translator API (Google Cloud)

**Purpose:** Stateless translation service. No auth, no database.

### Files
- `main.py` - Orchestrator (starts manga_translator + FastAPI)
- `server/translator_api.py` - FastAPI translation endpoints

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| POST | `/translate` | Translate image → Cloudinary URL |
| POST | `/translate/raw` | Translate image → Raw PNG bytes |

### Environment Variables
```bash
# Required
CLOUDINARY_CLOUD_NAME=djbjkbz9n
CLOUDINARY_API_KEY=482817726829321
CLOUDINARY_API_SECRET=ChgVAyY3vuoJo73P7MbKWLd1Bn4
GROQ_API_KEY=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

# Optional
PORT=8000                          # API port
TRANSLATOR_PORT=8001               # Internal manga_translator port
API_MODULE=server.translator_api:app  # FastAPI module
```

### Deployment (Google Cloud VM)

```bash
# SSH into VM
gcloud compute ssh snipshot-vm

# Navigate to project
cd /path/to/snipshot-backend

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export CLOUDINARY_CLOUD_NAME=djbjkbz9n
export CLOUDINARY_API_KEY=482817726829321
export CLOUDINARY_API_SECRET=ChgVAyY3vuoJo73P7MbKWLd1Bn4
export GROQ_API_KEY=your_key

# Run with orchestrator
python main.py
```

### Test Translation API

```bash
curl -X POST http://34.87.58.21:8000/translate \
  -F "file=@image.png" \
  -F "target_lang=ENG" \
  -F "detector=craft"

# Response:
# {
#   "image_url": "https://res.cloudinary.com/.../translated_xxx.png",
#   "public_id": "snipshot/translated_xxx"
# }
```

---

## 2. Database API (Render)

**Purpose:** User authentication and image metadata storage.

### Files
- `database_api/main.py` - FastAPI app entry
- `database_api/database/` - SQLAlchemy models
- `database_api/auth/` - JWT authentication
- `database_api/routes/` - API routes

### Endpoints

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| POST | `/api/users/register` | ❌ | Create account |
| POST | `/api/users/login` | ❌ | Login → JWT |
| GET | `/api/users/me` | ✅ | Get profile |
| GET | `/api/images` | ✅ | List user's images |
| POST | `/api/images` | ✅ | Save image metadata |
| GET | `/api/images/{id}` | ✅ | Get image details |
| DELETE | `/api/images/{id}` | ✅ | Delete image |

### Environment Variables
```bash
# Required
DATABASE_URL=postgresql+asyncpg://user:pass@host/db
SECRET_KEY=your_jwt_secret_key

# For image deletion from Cloudinary
CLOUDINARY_CLOUD_NAME=djbjkbz9n
CLOUDINARY_API_KEY=482817726829321
CLOUDINARY_API_SECRET=ChgVAyY3vuoJo73P7MbKWLd1Bn4
```

### Deployment (Render)

1. **Create PostgreSQL Database**
   - Go to Render Dashboard → New → PostgreSQL
   - Copy the Internal Database URL

2. **Create Web Service**
   - Go to Render Dashboard → New → Web Service
   - Connect GitHub repo
   - Settings:
     - **Root Directory:** `database_api`
     - **Build Command:** `pip install -r requirements.txt`
     - **Start Command:** `uvicorn main:app --host 0.0.0.0 --port $PORT`
   
3. **Add Environment Variables**
   ```
   DATABASE_URL=postgresql+asyncpg://...  (from step 1)
   SECRET_KEY=your_random_secret_key
   CLOUDINARY_CLOUD_NAME=djbjkbz9n
   CLOUDINARY_API_KEY=482817726829321
   CLOUDINARY_API_SECRET=ChgVAyY3vuoJo73P7MbKWLd1Bn4
   ```

4. **Deploy**
   - Push to GitHub → Render auto-deploys

### Test Database API

```bash
# Register
curl -X POST https://your-app.onrender.com/api/users/register \
  -H "Content-Type: application/json" \
  -d '{"email":"test@example.com","username":"test","password":"pass123"}'

# Login
curl -X POST https://your-app.onrender.com/api/users/login \
  -d "username=test@example.com&password=pass123"

# Save image (with JWT)
curl -X POST https://your-app.onrender.com/api/images \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"translated_url":"https://cloudinary.com/...", "translated_public_id":"snipshot/xxx"}'
```

---

## Local Development

### Run Both Services Locally

**Terminal 1 - Translator API (simulated):**
```bash
cd snipshot-backend
# Set API_MODULE to use translator_api
set API_MODULE=server.translator_api:app
python main.py
# Runs on http://localhost:8000
```

**Terminal 2 - Database API:**
```bash
cd snipshot-backend/database_api
pip install -r requirements.txt
uvicorn main:app --reload --port 8001
# Runs on http://localhost:8001
```

### Run Tests

```bash
# Test both backends together
python test_two_backends.py

# Test auth flow only (uses server/main.py)
python test_auth_flow.py
```

---

## Frontend Integration

### JavaScript Example

```javascript
const VM_URL = "http://34.87.58.21:8000";
const DB_URL = "https://your-app.onrender.com";

// 1. Login
const loginResp = await fetch(`${DB_URL}/api/users/login`, {
  method: "POST",
  headers: { "Content-Type": "application/x-www-form-urlencoded" },
  body: "username=user@example.com&password=mypassword"
});
const { access_token } = await loginResp.json();

// 2. Translate image
const formData = new FormData();
formData.append("file", imageFile);
formData.append("target_lang", "ENG");
formData.append("detector", "craft");

const translateResp = await fetch(`${VM_URL}/translate`, {
  method: "POST",
  body: formData
});
const { image_url, public_id } = await translateResp.json();

// 3. Save to user's account
await fetch(`${DB_URL}/api/images`, {
  method: "POST",
  headers: {
    "Authorization": `Bearer ${access_token}`,
    "Content-Type": "application/json"
  },
  body: JSON.stringify({
    translated_url: image_url,
    translated_public_id: public_id,
    source_language: "JPN",
    target_language: "ENG"
  })
});

// 4. Get user's images
const imagesResp = await fetch(`${DB_URL}/api/images`, {
  headers: { "Authorization": `Bearer ${access_token}` }
});
const images = await imagesResp.json();
```

---

## Project Structure

```
snipshot-backend/
├── main.py                    # Orchestrator (VM)
├── requirements.txt           # VM dependencies
├── .env                       # Environment variables
│
├── server/
│   ├── translator_api.py      # VM: Stateless translation API
│   ├── main.py                # VM: Full API with auth (legacy)
│   ├── database/              # SQLAlchemy models
│   ├── auth/                  # JWT auth
│   ├── storage/               # Cloudinary service
│   └── routes/                # API routes
│
├── database_api/              # Render: Database API
│   ├── main.py                # FastAPI entry
│   ├── requirements.txt       # Dependencies
│   ├── database/              # SQLAlchemy models
│   ├── auth/                  # JWT auth
│   └── routes/                # API routes
│
├── manga_translator/          # Core translation engine
│
└── models/                    # ML models
    ├── detection/
    ├── inpainting/
    └── ocr/
```

---

## Security Notes

- **JWT tokens** expire after 7 days by default
- **Passwords** are hashed with bcrypt
- **CORS** is configured to allow frontend origins
- **Cloudinary** credentials should be kept secret
- **DATABASE_URL** should use SSL in production

---

## Troubleshooting

### VM not responding
```bash
# Check if services are running
curl http://34.87.58.21:8000/health
```

### Database connection failed
- Ensure `DATABASE_URL` uses `postgresql+asyncpg://` prefix
- Check Render's PostgreSQL dashboard for connection details

### Cloudinary upload failed
- Verify credentials in environment variables
- Check Cloudinary dashboard for usage limits

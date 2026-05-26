# Authentication Plan: JWT Verification for SnipShot Services

This document outlines the implementation plan for securing the SnipShot translation endpoint (`TRANSLATOR_URL`) across the backend and the two frontend repositories (`snipshot-desktop` and `snipshot-android`).

---

## 1. Architectural Overview

Both frontends currently authenticate directly with **Supabase Auth**. This yields a standard JWT access token signed by Supabase. 

We will verify this token **locally** on the FastAPI backend using the existing `SUPABASE_JWT_SECRET` key. This prevents any external API round-trips to Supabase for authentication check, ensuring low latency.

```
[ Desktop App ] ──(Includes Supabase JWT in Headers)──► [ FastAPI Translator Backend ]
                                                                 │
[ Android App ] ──(Includes Supabase JWT in Headers)──►          ▼
                                                       [ Decodes & verifies locally ]
                                                       [ using SUPABASE_JWT_SECRET ]
```

---

## 2. Backend Implementation (`snipshot-backend`)

### A. Dependencies
Add `PyJWT` to `requirements.txt` (or install it manually):
```bash
pip install PyJWT
```

### B. Add JWT Verification helper
In [snipshot_engine/server.py](file:///c:/Users/neilc/OneDrive/Documents/GitHub/snipshot-backend/snipshot_engine/server.py), implement the token validation dependency using `PyJWT`:

```python
import jwt
import logging
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

logger = logging.getLogger(__name__)

# Read environment variables and fail fast
SUPABASE_JWT_SECRET = os.getenv("SUPABASE_JWT_SECRET")
if not SUPABASE_JWT_SECRET:
    logger.critical("Startup aborted: SUPABASE_JWT_SECRET is missing.")
    raise RuntimeError("SUPABASE_JWT_SECRET is missing from the environment.")

# HTTPBearer automatically checks for the Authorization header
security = HTTPBearer()

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> dict:
    """Verifies the incoming Supabase JWT token."""
    token = credentials.credentials
    try:
        # Decode and verify token signature, expiry, and audience
        payload = jwt.decode(
            token,
            SUPABASE_JWT_SECRET,
            algorithms=["HS256"],
            audience="authenticated"  # Enforce Supabase audience claim
        )
        return payload
    except jwt.ExpiredSignatureError as e:
        logger.warning("JWT verification failed – token expired: %s", e)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token."
        )
    except jwt.PyJWTError as e:
        logger.warning("JWT verification failed: %s", e)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token."
        )
```

> **Security note**: Raw PyJWT exception messages are never forwarded to the caller. All JWT errors are logged server-side and the client receives only the generic `"Invalid token."` detail string.

### C. HTTPS Enforcement Startup Check
Add a startup check in [snipshot_engine/server.py](file:///c:/Users/neilc/OneDrive/Documents/GitHub/snipshot-backend/snipshot_engine/server.py) to ensure `TRANSLATOR_URL` uses HTTPS in non-development environments:

```python
import logging
import os

logger = logging.getLogger(__name__)

ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
TRANSLATOR_URL = os.getenv("TRANSLATOR_URL", "")

if ENVIRONMENT != "development" and not TRANSLATOR_URL.startswith("https://"):
    logger.critical(
        "Startup aborted: TRANSLATOR_URL must use HTTPS in non-development environments. "
        "Got %r. Set ENVIRONMENT=development to bypass this check locally.",
        TRANSLATOR_URL,
    )
    raise RuntimeError(
        f"TRANSLATOR_URL must start with 'https://' in environment '{ENVIRONMENT}'. "
        f"Got: {TRANSLATOR_URL!r}"
    )
```

This check runs at module import time so the server will refuse to start rather than silently serving traffic over an insecure URL.

### D. Environment-Aware Rate Limiting
Rate limiting is enforced **only in non-development environments**. When running locally (desktop dev or the simpler Android URL), all requests pass through without hitting any rate limit.

Configure `slowapi` using the `enabled` flag on the `Limiter` constructor. A dedicated `RATE_LIMIT_ENABLED` environment variable can override the environment default in either direction:

```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
IS_PRODUCTION = ENVIRONMENT != "development"

# Explicit rate-limit toggle — overrides environment default if set
isRateLimited = os.getenv("RATE_LIMIT_ENABLED", str(IS_PRODUCTION)).lower() == "true"

limiter = Limiter(key_func=get_remote_address, enabled=isRateLimited)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
```

This gives four meaningful states:

| `ENVIRONMENT` | `RATE_LIMIT_ENABLED` | `isRateLimited` | Notes |
|---|---|---|---|
| `production` | *(not set)* | `True` | Defaults **on** in prod |
| `development` | *(not set)* | `False` | Defaults **off** locally |
| `production` | `false` | `False` | Force off even in prod (no deploy needed) |
| `development` | `true` | `True` | Force on locally to test without changing environment |

Apply both the rate limiter and authentication decorators on each endpoint. Note that `request: Request` is a required positional parameter for `slowapi` to inject rate-limit context — it must appear before the other parameters:

```python
@app.post("/translate")
@limiter.limit("60/minute")
async def translate(
    request: Request,
    image: UploadFile = File(...),
    config: str = Form("{}"),
    user: dict = Depends(get_current_user)
):
    ...

@app.post("/translate/raw")
@limiter.limit("60/minute")
async def translate_raw(
    request: Request,
    image: UploadFile = File(...),
    config: str = Form("{}"),
    user: dict = Depends(get_current_user)
):
    ...
```

> **Note**: The `enabled` flag is the single toggle that governs all rate-limit enforcement. The endpoint decorators stay identical regardless of environment.

### E. Rate Limiting Nuance (Future Consideration)
IP-based rate limiting (`get_remote_address`) works for basic cases, but has some production limitations:
- **CGNAT / Corporate Proxies**: Multiple desktop users behind the same corporate proxy or CGNAT share a single public IP address, which may lead to accidental rate-limiting of legitimate users.
- **Cellular Networks**: Mobile users on cellular connections frequently switch IPs as they move between towers, making IP-based tracking less reliable.

**Proposed Mitigation**:
For production stability, consider implementing a **user-ID based limiter** as a secondary or primary strategy. This can be done by extracting the user's `sub` (subject UUID) claim from the validated Supabase JWT payload, and passing it to the custom key function of the limiter.

---

## 3. Desktop Frontend Implementation (`snipshot-desktop`)

### A. Update `TranslatorClient`
Modify `translate_image` in [api/translator_client.py](file:///c:/Users/neilc/OneDrive/Documents/GitHub/snipshot-desktop/api/translator_client.py) to accept and forward the token, and to handle 401 responses with a one-time refresh-and-retry:

```python
def translate_image(
    self, 
    image_bytes: bytes, 
    config: Optional[Dict[str, Any]] = None,
    token: Optional[str] = None,          # 1. Accept token parameter
    _retry: bool = True                    # 2. Guard against infinite retry
) -> dict:
    if config is None:
        config = DEFAULT_TRANSLATION_CONFIG

    # 3. Build Authorization header
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    with httpx.Client(timeout=180.0) as client:
        files = {
            "image": ("snip.png", image_bytes, "image/png"),
        }
        data = {
            "config": json.dumps(config)
        }

        translate_url = f"{self.translator_url}/translate/raw"

        try:
            response = client.post(
                translate_url,
                files=files,
                data=data,
                headers=headers          # 4. Include headers in POST
            )

            # 5. On 401, attempt a one-time token refresh and retry
            if response.status_code == 401 and _retry:
                raise _TokenExpired()

            response.raise_for_status()
            return response.json()
        except _TokenExpired:
            raise  # re-raised so SupabaseAPIClient can handle it
```

### B. Define Exception Classes
Define the necessary exception classes to handle token expiry routing and authentication errors:

In [api/translator_client.py](file:///c:/Users/neilc/OneDrive/Documents/GitHub/snipshot-desktop/api/translator_client.py), at the top of the file:
```python
class _TokenExpired(Exception):
    """Sentinel raised when the backend returns 401, triggering a refresh-and-retry."""
```

In a new file [api/exceptions.py](file:///c:/Users/neilc/OneDrive/Documents/GitHub/snipshot-desktop/api/exceptions.py) (create if missing):
```python
class AuthenticationError(Exception):
    """Raised when authentication fails and the user must log in again."""
```

Import `AuthenticationError` in [api/supabase_client.py](file:///c:/Users/neilc/OneDrive/Documents/GitHub/snipshot-desktop/api/supabase_client.py):
```python
from api.exceptions import AuthenticationError
```

### C. Update `SupabaseAPIClient`
Forward the access token **from the live session object** (not a cached value) in [api/supabase_client.py](file:///c:/Users/neilc/OneDrive/Documents/GitHub/snipshot-desktop/api/supabase_client.py), and implement refresh-and-retry:

```python
class SupabaseAPIClient:
    # ...

    @property
    def access_token(self) -> Optional[str]:
        """Always read the token from the live Supabase session, never a cached copy."""
        session = self._supabase.auth.get_session()
        return session.access_token if session else None

    def translate_image(self, image_bytes: bytes, config: dict = None) -> dict:
        """Delegate image translation to the dedicated TranslatorClient."""
        try:
            return self._translator.translate_image(
                image_bytes, config, token=self.access_token
            )
        except _TokenExpired:
            # Refresh the Supabase session and retry once
            self._supabase.auth.refresh_session()
            new_token = self.access_token
            if new_token is None:
                raise AuthenticationError("Session could not be refreshed. Please log in again.")
            try:
                return self._translator.translate_image(
                    image_bytes, config, token=new_token, _retry=False
                )
            except (_TokenExpired, httpx.HTTPStatusError):
                raise AuthenticationError("Session expired. Please log in again.")
```

> **Important**: `self.access_token` is now a `@property` that reads from the live Supabase session on every call, so stale cached tokens are never used.

---

## 4. Android Frontend Implementation (`snipshot-android`)

In the Android app, `SnipOverlayActivity` makes the API request directly.

### A. Update `SnipOverlayActivity`
Modify `performMode1Manga` in [SnipOverlayActivity.kt](file:///c:/Users/neilc/OneDrive/Documents/GitHub/snipshot-android/app/src/main/java/com/example/snipshot/SnipOverlayActivity.kt) to handle 401 responses with a one-time session refresh and retry:

```kotlin
private fun performMode1Manga(bitmap: Bitmap) {
    CoroutineScope(Dispatchers.Main).launch {
        progressBar.visibility = View.VISIBLE
        try {
            // 1. Helper to build a fresh RequestBody on demand
            fun buildRequestBody(): RequestBody {
                val stream = ByteArrayOutputStream()
                bitmap.compress(Bitmap.CompressFormat.PNG, 100, stream)
                val imageBytes = stream.toByteArray()
                return MultipartBody.Builder()
                    .setType(MultipartBody.FORM)
                    .addFormDataPart(
                        "image", "snip.png",
                        imageBytes.toRequestBody("image/png".toMediaType())
                    )
                    .build()
            }

            // 2. Helper to build a request with a fresh body and the current token
            fun buildRequest(): Request {
                val builder = Request.Builder()
                    .url("$backendUrl/translate/raw")
                    .post(buildRequestBody())   // fresh body every time
                ApiClient.accessToken?.let { token ->
                    builder.addHeader("Authorization", "Bearer $token")
                }
                return builder.build()
            }

            // 3. First attempt
            var response = withContext(Dispatchers.IO) {
                client.newCall(buildRequest()).execute()
            }

            // 4. On 401, refresh session and retry once
            if (response.code == 401) {
                val refreshResult = supabase.auth.refreshSession()
                ApiClient.accessToken = refreshResult.accessToken

                response = withContext(Dispatchers.IO) {
                    client.newCall(buildRequest()).execute()
                }

                // 5. If retry also fails, redirect to login
                if (!response.isSuccessful) {
                    navigateToLogin()
                    return@launch
                }
            }

            // ... [Handle Response] ...
        } catch (e: Exception) {
            // ... [Error handling] ...
        } finally {
            progressBar.visibility = View.GONE
        }
    }
}
```

### B. Clear `ApiClient.accessToken` on Logout
In the logout handler (wherever `supabase.auth.signOut()` is called), immediately clear the cached token so subsequent calls cannot reuse it:

```kotlin
// In logout handler
supabase.auth.signOut()
ApiClient.accessToken = null
```

> **Important**: Do not store the access token in `ApiClient` without invalidating it on logout. Clearing it ensures that any pending or subsequent translation calls will correctly fail authentication.

---

## 5. Verification & Testing Checklist

- [ ] **No Authorization header** → expect `401`
- [ ] **Expired token** → expect `401`
- [ ] **Malformed / tampered token** → expect `401`
- [ ] **Token signed by a different Supabase project** → expect `401`
- [ ] **Valid token** → expect `200`
- [ ] **`RATE_LIMIT_ENABLED=true`, >60 req/min** → expect `429`
- [ ] **`RATE_LIMIT_ENABLED=false`, >60 req/min** → expect no `429` (pass through)
- [ ] **Backend started with a non-HTTPS `TRANSLATOR_URL` (non-dev env)** → expect `RuntimeError` at startup
- [ ] **Desktop: trigger a 401 mid-session** → verify token refresh and automatic retry succeed
- [ ] **Android: trigger a 401 mid-session** → verify token refresh and automatic retry succeed
- [ ] **Logout on desktop** → verify subsequent translation attempts fail and prompt login
- [ ] **Logout on Android** → verify `ApiClient.accessToken` is cleared and subsequent calls fail

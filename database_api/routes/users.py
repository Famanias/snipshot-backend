"""
User routes - Supabase Auth

User registration and login are handled directly by Supabase Auth.
Frontend should use Supabase client SDK for auth.

This backend only verifies tokens and provides user profile info.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, EmailStr

from config import get_supabase
from auth import get_current_user_id
from schemas import UserProfile, AuthResponse, MessageResponse

router = APIRouter(prefix="/users", tags=["users"])


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


@router.post("/register", response_model=AuthResponse, status_code=status.HTTP_201_CREATED)
async def register(request: RegisterRequest):
    """
    Register a new user via Supabase Auth.
    
    Note: For production, consider using Supabase client SDK directly from frontend.
    This endpoint is provided for convenience.
    """
    try:
        supabase = get_supabase()
        response = supabase.auth.sign_up({
            "email": request.email,
            "password": request.password
        })
        
        if response.user is None:
            raise HTTPException(status_code=400, detail="Registration failed")
        
        session = response.session
        if session is None:
            # Email confirmation might be required
            return AuthResponse(
                access_token="",
                expires_in=0,
                user={"id": response.user.id, "email": response.user.email}
            )
        
        return AuthResponse(
            access_token=session.access_token,
            expires_in=session.expires_in or 3600,
            refresh_token=session.refresh_token,
            user={"id": response.user.id, "email": response.user.email}
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/login", response_model=AuthResponse)
async def login(request: LoginRequest):
    """
    Login via Supabase Auth.
    
    Returns access token for API authentication.
    """
    try:
        supabase = get_supabase()
        response = supabase.auth.sign_in_with_password({
            "email": request.email,
            "password": request.password
        })
        
        if response.user is None or response.session is None:
            raise HTTPException(status_code=401, detail="Invalid credentials")
        
        return AuthResponse(
            access_token=response.session.access_token,
            expires_in=response.session.expires_in or 3600,
            refresh_token=response.session.refresh_token,
            user={"id": response.user.id, "email": response.user.email}
        )
        
    except Exception as e:
        if "Invalid login credentials" in str(e):
            raise HTTPException(status_code=401, detail="Invalid email or password")
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/me", response_model=UserProfile)
async def get_profile(user_id: str = Depends(get_current_user_id)):
    """
    Get current user profile.
    
    Requires valid Supabase access token in Authorization header.
    """
    try:
        supabase = get_supabase()
        response = supabase.auth.admin.get_user_by_id(user_id)
        
        if response.user is None:
            raise HTTPException(status_code=404, detail="User not found")
        
        return UserProfile(
            id=response.user.id,
            email=response.user.email,
            created_at=str(response.user.created_at) if response.user.created_at else None
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/logout", response_model=MessageResponse)
async def logout(user_id: str = Depends(get_current_user_id)):
    """
    Logout current user (invalidate token on Supabase side).
    """
    # Note: Token invalidation happens on frontend with Supabase client
    # Backend just acknowledges the logout
    return MessageResponse(message="Logged out successfully")

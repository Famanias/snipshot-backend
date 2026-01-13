"""
Pydantic schemas for request/response validation
"""

from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, EmailStr, Field


# ============ User Schemas ============

class UserCreate(BaseModel):
    """Schema for user registration"""
    email: EmailStr
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=6)


class UserLogin(BaseModel):
    """Schema for user login"""
    email: EmailStr
    password: str


class UserResponse(BaseModel):
    """Schema for user in responses"""
    id: int
    email: str
    username: str
    is_active: bool
    created_at: datetime

    class Config:
        from_attributes = True


class UserUpdate(BaseModel):
    """Schema for updating user profile"""
    username: Optional[str] = Field(None, min_length=3, max_length=50)
    email: Optional[EmailStr] = None


class PasswordChange(BaseModel):
    """Schema for password change"""
    current_password: str
    new_password: str = Field(..., min_length=6)


# ============ Auth Schemas ============

class Token(BaseModel):
    """Schema for auth token response"""
    access_token: str
    token_type: str = "bearer"


class TokenData(BaseModel):
    """Schema for decoded token data"""
    user_id: Optional[int] = None


# ============ Image Schemas ============

class ImageResponse(BaseModel):
    """Schema for image in responses"""
    id: int
    original_filename: Optional[str]
    original_url: Optional[str]
    translated_url: str
    source_language: Optional[str]
    target_language: Optional[str]
    created_at: datetime

    class Config:
        from_attributes = True


class ImageListResponse(BaseModel):
    """Schema for paginated image list"""
    images: List[ImageResponse]
    total: int
    page: int
    per_page: int
    pages: int


class TranslateResponse(BaseModel):
    """Schema for translate endpoint response"""
    success: bool
    image_url: str
    image_id: Optional[int] = None  # Only if user is authenticated
    message: Optional[str] = None


class TranslateConfig(BaseModel):
    """Schema for translation configuration"""
    target_lang: str = "ENG"
    detector: str = "default"
    ocr: str = "48px"
    inpainter: str = "lama_large"
    translator: str = "groq"
    direction: str = "auto"
    
    # Optional settings
    upscale_ratio: Optional[int] = 1
    font_size: Optional[int] = None
    font_color: Optional[str] = None


# ============ Generic Schemas ============

class MessageResponse(BaseModel):
    """Generic message response"""
    message: str


class ErrorResponse(BaseModel):
    """Error response"""
    detail: str

"""
Pydantic schemas
"""

from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, EmailStr, Field


# User schemas
class UserCreate(BaseModel):
    email: EmailStr
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=6)


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class UserResponse(BaseModel):
    id: int
    email: str
    username: str
    is_active: bool
    created_at: datetime

    class Config:
        from_attributes = True


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


# Image schemas
class ImageCreate(BaseModel):
    """Schema for saving an image URL from the Translator API"""
    image_url: str
    public_id: str
    original_filename: Optional[str] = None
    source_language: Optional[str] = None
    target_language: Optional[str] = None


class ImageResponse(BaseModel):
    id: int
    image_url: str
    public_id: str
    original_filename: Optional[str]
    source_language: Optional[str]
    target_language: Optional[str]
    created_at: datetime

    class Config:
        from_attributes = True


class ImageListResponse(BaseModel):
    images: List[ImageResponse]
    total: int
    page: int
    per_page: int
    pages: int


class MessageResponse(BaseModel):
    message: str

"""
Database models for SnipShot Database API (Supabase Edition)

User management is handled by Supabase Auth.
We store folder and image metadata here.
"""

from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime, Text, ForeignKey
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class Folder(Base):
    """
    Folder model for organizing translated images.
    
    Users can create folders to organize their translations.
    """
    __tablename__ = "folders"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String(36), nullable=False, index=True)  # Supabase UUID
    
    name = Column(String(100), nullable=False)
    description = Column(Text, nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationship to images
    images = relationship("Image", back_populates="folder", cascade="all, delete-orphan")


class Image(Base):
    """
    Image metadata model.
    
    user_id is the Supabase Auth user UUID (string, not integer).
    Actual image files are stored in Supabase Storage.
    """
    __tablename__ = "images"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String(36), nullable=False, index=True)  # Supabase UUID
    folder_id = Column(Integer, ForeignKey("folders.id"), nullable=True, index=True)  # Optional folder
    
    # Supabase Storage info
    storage_path = Column(Text, nullable=False)  # Path in Supabase Storage
    public_url = Column(Text, nullable=False)    # Public URL for the image
    
    # Metadata
    filename = Column(String(255), nullable=False)  # User-editable filename
    original_filename = Column(String(255), nullable=True)  # Original from translation
    source_language = Column(String(10), nullable=True)
    target_language = Column(String(10), nullable=True)
    file_size = Column(Integer, nullable=True)  # Size in bytes
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationship to folder
    folder = relationship("Folder", back_populates="images")

"""
Database models for SnipShot Backend
- User: stores user accounts
- Image: stores translated image metadata
"""

from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Text, Boolean
from sqlalchemy.orm import relationship, declarative_base

Base = declarative_base()


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    username = Column(String(100), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationship to images
    images = relationship("Image", back_populates="owner", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<User(id={self.id}, username={self.username}, email={self.email})>"


class Image(Base):
    __tablename__ = "images"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    
    # Original image info
    original_filename = Column(String(255), nullable=True)
    
    # Cloudinary URLs
    original_url = Column(Text, nullable=True)  # Optional: store original
    translated_url = Column(Text, nullable=False)  # Required: the result
    
    # Cloudinary public IDs (for deletion)
    original_public_id = Column(String(255), nullable=True)
    translated_public_id = Column(String(255), nullable=False)
    
    # Translation config (stored as JSON string)
    translation_config = Column(Text, nullable=True)
    
    # Metadata
    source_language = Column(String(10), nullable=True)
    target_language = Column(String(10), nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationship to user
    owner = relationship("User", back_populates="images")

    def __repr__(self):
        return f"<Image(id={self.id}, user_id={self.user_id}, translated_url={self.translated_url[:50]}...)>"

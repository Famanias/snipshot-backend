"""
Image routes - Save and retrieve Cloudinary URLs
"""

import os
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
import cloudinary
import cloudinary.uploader
from dotenv import load_dotenv

from ..database import get_db, User, Image
from ..auth import get_current_user
from ..schemas import ImageCreate, ImageResponse, ImageListResponse, MessageResponse

load_dotenv()

# Configure Cloudinary (for deletion)
cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET"),
    secure=True
)

router = APIRouter(prefix="/images", tags=["images"])


@router.post("", response_model=ImageResponse, status_code=status.HTTP_201_CREATED)
async def save_image(
    image_data: ImageCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Save a translated image URL to user's account.
    
    This is called by the frontend AFTER receiving the URL from the Translator API.
    """
    new_image = Image(
        user_id=current_user.id,
        image_url=image_data.image_url,
        public_id=image_data.public_id,
        original_filename=image_data.original_filename,
        source_language=image_data.source_language,
        target_language=image_data.target_language
    )
    db.add(new_image)
    await db.commit()
    await db.refresh(new_image)
    return new_image


@router.get("", response_model=ImageListResponse)
async def list_images(
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """List all images for current user"""
    # Count
    count_result = await db.execute(
        select(func.count(Image.id)).where(Image.user_id == current_user.id)
    )
    total = count_result.scalar()
    
    # Paginate
    offset = (page - 1) * per_page
    pages = (total + per_page - 1) // per_page if total > 0 else 1
    
    # Get images
    result = await db.execute(
        select(Image)
        .where(Image.user_id == current_user.id)
        .order_by(Image.created_at.desc())
        .offset(offset)
        .limit(per_page)
    )
    images = result.scalars().all()
    
    return ImageListResponse(
        images=[ImageResponse.model_validate(img) for img in images],
        total=total,
        page=page,
        per_page=per_page,
        pages=pages
    )


@router.get("/{image_id}", response_model=ImageResponse)
async def get_image(
    image_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get a specific image"""
    result = await db.execute(
        select(Image).where(Image.id == image_id, Image.user_id == current_user.id)
    )
    image = result.scalar_one_or_none()
    
    if not image:
        raise HTTPException(status_code=404, detail="Image not found")
    
    return image


@router.delete("/{image_id}", response_model=MessageResponse)
async def delete_image(
    image_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Delete an image (from Cloudinary and database)"""
    result = await db.execute(
        select(Image).where(Image.id == image_id, Image.user_id == current_user.id)
    )
    image = result.scalar_one_or_none()
    
    if not image:
        raise HTTPException(status_code=404, detail="Image not found")
    
    # Delete from Cloudinary
    try:
        cloudinary.uploader.destroy(image.public_id)
    except Exception as e:
        print(f"Warning: Could not delete from Cloudinary: {e}")
    
    # Delete from database
    await db.delete(image)
    await db.commit()
    
    return MessageResponse(message="Image deleted successfully")


@router.delete("", response_model=MessageResponse)
async def delete_all_images(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Delete all images for current user"""
    result = await db.execute(
        select(Image).where(Image.user_id == current_user.id)
    )
    images = result.scalars().all()
    
    if not images:
        return MessageResponse(message="No images to delete")
    
    # Delete from Cloudinary
    public_ids = [img.public_id for img in images]
    try:
        cloudinary.api.delete_resources(public_ids)
    except Exception as e:
        print(f"Warning: Could not delete from Cloudinary: {e}")
    
    # Delete from database
    for image in images:
        await db.delete(image)
    await db.commit()
    
    return MessageResponse(message=f"Deleted {len(images)} images")

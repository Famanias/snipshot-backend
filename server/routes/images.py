"""
Image-related API routes
- List user's images
- Get single image
- Delete image
"""

from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from ..database import get_db, User, Image
from ..auth import get_current_user
from ..storage import CloudinaryStorage
from ..schemas import ImageResponse, ImageListResponse, MessageResponse

router = APIRouter(prefix="/images", tags=["images"])


@router.get("", response_model=ImageListResponse)
async def list_images(
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    List all translated images for the current user.
    Supports pagination.
    Requires authentication.
    """
    # Count total images
    count_result = await db.execute(
        select(func.count(Image.id)).where(Image.user_id == current_user.id)
    )
    total = count_result.scalar()
    
    # Calculate pagination
    offset = (page - 1) * per_page
    pages = (total + per_page - 1) // per_page if total > 0 else 1
    
    # Get images with pagination
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
    """
    Get a specific image by ID.
    User can only access their own images.
    Requires authentication.
    """
    result = await db.execute(
        select(Image).where(
            Image.id == image_id,
            Image.user_id == current_user.id
        )
    )
    image = result.scalar_one_or_none()
    
    if not image:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Image not found"
        )
    
    return image


@router.delete("/{image_id}", response_model=MessageResponse)
async def delete_image(
    image_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Delete a specific image by ID.
    Removes from both Cloudinary and database.
    User can only delete their own images.
    Requires authentication.
    """
    result = await db.execute(
        select(Image).where(
            Image.id == image_id,
            Image.user_id == current_user.id
        )
    )
    image = result.scalar_one_or_none()
    
    if not image:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Image not found"
        )
    
    # Delete from Cloudinary
    public_ids_to_delete = [image.translated_public_id]
    if image.original_public_id:
        public_ids_to_delete.append(image.original_public_id)
    
    for public_id in public_ids_to_delete:
        await CloudinaryStorage.delete_image(public_id)
    
    # Delete from database
    await db.delete(image)
    await db.commit()
    
    return MessageResponse(message="Image deleted successfully")


@router.delete("", response_model=MessageResponse)
async def delete_all_images(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Delete ALL images for the current user.
    WARNING: This action is irreversible!
    Requires authentication.
    """
    # Get all images
    result = await db.execute(
        select(Image).where(Image.user_id == current_user.id)
    )
    images = result.scalars().all()
    
    if not images:
        return MessageResponse(message="No images to delete")
    
    # Collect all public IDs
    public_ids = []
    for image in images:
        public_ids.append(image.translated_public_id)
        if image.original_public_id:
            public_ids.append(image.original_public_id)
    
    # Delete from Cloudinary (batch)
    if public_ids:
        await CloudinaryStorage.delete_images(public_ids)
    
    # Delete all from database
    for image in images:
        await db.delete(image)
    
    await db.commit()
    
    return MessageResponse(message=f"Deleted {len(images)} images successfully")

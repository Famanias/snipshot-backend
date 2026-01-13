"""
Folder routes - CRUD operations for organizing images

Users can create folders to organize their translated images.
"""

from typing import Optional, List
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from pydantic import BaseModel

from database import get_db, Folder, Image
from auth import get_current_user_id

router = APIRouter(prefix="/folders", tags=["folders"])


# ============== Schemas ==============

class FolderCreate(BaseModel):
    name: str
    description: Optional[str] = None


class FolderUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None


class FolderResponse(BaseModel):
    id: int
    name: str
    description: Optional[str]
    image_count: int
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class FolderListResponse(BaseModel):
    folders: List[FolderResponse]
    total: int


class FolderDetailResponse(BaseModel):
    id: int
    name: str
    description: Optional[str]
    images: List[dict]  # Will contain image summaries
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


# ============== Routes ==============

@router.post("", response_model=FolderResponse, status_code=status.HTTP_201_CREATED)
async def create_folder(
    folder_data: FolderCreate,
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db)
):
    """
    Create a new folder.
    """
    # Check if folder with same name exists for this user
    existing = await db.execute(
        select(Folder).where(
            Folder.user_id == user_id,
            Folder.name == folder_data.name
        )
    )
    if existing.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Folder '{folder_data.name}' already exists"
        )
    
    folder = Folder(
        user_id=user_id,
        name=folder_data.name,
        description=folder_data.description
    )
    
    db.add(folder)
    await db.commit()
    await db.refresh(folder)
    
    return FolderResponse(
        id=folder.id,
        name=folder.name,
        description=folder.description,
        image_count=0,
        created_at=folder.created_at,
        updated_at=folder.updated_at
    )


@router.get("", response_model=FolderListResponse)
async def list_folders(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db)
):
    """
    List all folders for the current user.
    """
    # Get total count
    count_result = await db.execute(
        select(func.count(Folder.id)).where(Folder.user_id == user_id)
    )
    total = count_result.scalar()
    
    # Get folders with image counts
    result = await db.execute(
        select(Folder)
        .where(Folder.user_id == user_id)
        .order_by(Folder.updated_at.desc())
        .offset(skip)
        .limit(limit)
    )
    folders = result.scalars().all()
    
    # Get image counts for each folder
    folder_responses = []
    for folder in folders:
        count_result = await db.execute(
            select(func.count(Image.id)).where(Image.folder_id == folder.id)
        )
        image_count = count_result.scalar()
        
        folder_responses.append(FolderResponse(
            id=folder.id,
            name=folder.name,
            description=folder.description,
            image_count=image_count,
            created_at=folder.created_at,
            updated_at=folder.updated_at
        ))
    
    return FolderListResponse(folders=folder_responses, total=total)


@router.get("/{folder_id}", response_model=FolderDetailResponse)
async def get_folder(
    folder_id: int,
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db)
):
    """
    Get a folder with its images.
    """
    result = await db.execute(
        select(Folder).where(
            Folder.id == folder_id,
            Folder.user_id == user_id
        )
    )
    folder = result.scalar_one_or_none()
    
    if not folder:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Folder not found"
        )
    
    # Get images in this folder
    images_result = await db.execute(
        select(Image)
        .where(Image.folder_id == folder_id)
        .order_by(Image.created_at.desc())
    )
    images = images_result.scalars().all()
    
    image_list = [
        {
            "id": img.id,
            "filename": img.filename,
            "public_url": img.public_url,
            "file_size": img.file_size,
            "created_at": img.created_at.isoformat()
        }
        for img in images
    ]
    
    return FolderDetailResponse(
        id=folder.id,
        name=folder.name,
        description=folder.description,
        images=image_list,
        created_at=folder.created_at,
        updated_at=folder.updated_at
    )


@router.put("/{folder_id}", response_model=FolderResponse)
async def update_folder(
    folder_id: int,
    folder_data: FolderUpdate,
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db)
):
    """
    Update a folder's name or description.
    """
    result = await db.execute(
        select(Folder).where(
            Folder.id == folder_id,
            Folder.user_id == user_id
        )
    )
    folder = result.scalar_one_or_none()
    
    if not folder:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Folder not found"
        )
    
    # Check if new name conflicts with existing folder
    if folder_data.name and folder_data.name != folder.name:
        existing = await db.execute(
            select(Folder).where(
                Folder.user_id == user_id,
                Folder.name == folder_data.name,
                Folder.id != folder_id
            )
        )
        if existing.scalar_one_or_none():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Folder '{folder_data.name}' already exists"
            )
        folder.name = folder_data.name
    
    if folder_data.description is not None:
        folder.description = folder_data.description
    
    folder.updated_at = datetime.utcnow()
    
    await db.commit()
    await db.refresh(folder)
    
    # Get image count
    count_result = await db.execute(
        select(func.count(Image.id)).where(Image.folder_id == folder.id)
    )
    image_count = count_result.scalar()
    
    return FolderResponse(
        id=folder.id,
        name=folder.name,
        description=folder.description,
        image_count=image_count,
        created_at=folder.created_at,
        updated_at=folder.updated_at
    )


@router.delete("/{folder_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_folder(
    folder_id: int,
    delete_images: bool = Query(False, description="Also delete images in the folder from storage"),
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db)
):
    """
    Delete a folder.
    
    If delete_images=False (default), images are moved to "unfiled" (folder_id=None).
    If delete_images=True, images are permanently deleted from storage too.
    """
    from config import get_supabase, STORAGE_BUCKET
    
    result = await db.execute(
        select(Folder).where(
            Folder.id == folder_id,
            Folder.user_id == user_id
        )
    )
    folder = result.scalar_one_or_none()
    
    if not folder:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Folder not found"
        )
    
    if delete_images:
        # Delete images from Supabase Storage
        supabase = get_supabase()
        images_result = await db.execute(
            select(Image).where(Image.folder_id == folder_id)
        )
        images = images_result.scalars().all()
        
        for img in images:
            try:
                supabase.storage.from_(STORAGE_BUCKET).remove([img.storage_path])
            except Exception:
                pass  # Continue even if storage delete fails
        
        # Images will be cascade deleted with the folder
    else:
        # Move images to unfiled (set folder_id to None)
        await db.execute(
            Image.__table__.update()
            .where(Image.folder_id == folder_id)
            .values(folder_id=None)
        )
    
    await db.delete(folder)
    await db.commit()
    
    return None

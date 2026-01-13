"""
Cloudinary storage service for image uploads
Works with both desktop and mobile clients
"""

import os
import io
from typing import Optional, Tuple
from PIL import Image
import cloudinary
import cloudinary.uploader
from dotenv import load_dotenv

load_dotenv()

# Configure Cloudinary from environment variables
cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET"),
    secure=True
)


class CloudinaryStorage:
    """
    Cloudinary storage service for SnipShot images.
    Handles upload, deletion, and URL generation.
    """
    
    FOLDER_TRANSLATED = "snipshot/translated"
    FOLDER_ORIGINAL = "snipshot/original"
    
    @staticmethod
    def _pil_to_bytes(image: Image.Image, format: str = "PNG") -> bytes:
        """Convert PIL Image to bytes"""
        buffer = io.BytesIO()
        image.save(buffer, format=format)
        buffer.seek(0)
        return buffer.getvalue()
    
    @classmethod
    async def upload_translated_image(
        cls,
        image: Image.Image,
        user_id: int,
        filename: Optional[str] = None
    ) -> Tuple[str, str]:
        """
        Upload a translated image to Cloudinary.
        
        Args:
            image: PIL Image object
            user_id: Owner's user ID
            filename: Optional original filename for reference
            
        Returns:
            Tuple of (public_url, public_id)
        """
        image_bytes = cls._pil_to_bytes(image)
        
        # Generate a unique public_id
        import time
        public_id = f"{cls.FOLDER_TRANSLATED}/user_{user_id}/{int(time.time() * 1000)}"
        
        # Upload to Cloudinary
        result = cloudinary.uploader.upload(
            image_bytes,
            public_id=public_id,
            resource_type="image",
            format="png",
            overwrite=True,
            # Optimize for web/mobile delivery
            transformation={
                "quality": "auto",
                "fetch_format": "auto"
            }
        )
        
        return result["secure_url"], result["public_id"]
    
    @classmethod
    async def upload_original_image(
        cls,
        image: Image.Image,
        user_id: int,
        filename: Optional[str] = None
    ) -> Tuple[str, str]:
        """
        Upload an original (pre-translation) image to Cloudinary.
        
        Args:
            image: PIL Image object  
            user_id: Owner's user ID
            filename: Optional original filename
            
        Returns:
            Tuple of (public_url, public_id)
        """
        image_bytes = cls._pil_to_bytes(image)
        
        import time
        public_id = f"{cls.FOLDER_ORIGINAL}/user_{user_id}/{int(time.time() * 1000)}"
        
        result = cloudinary.uploader.upload(
            image_bytes,
            public_id=public_id,
            resource_type="image",
            format="png",
            overwrite=True
        )
        
        return result["secure_url"], result["public_id"]
    
    @classmethod
    async def delete_image(cls, public_id: str) -> bool:
        """
        Delete an image from Cloudinary by public_id.
        
        Args:
            public_id: Cloudinary public ID
            
        Returns:
            True if deleted successfully
        """
        try:
            result = cloudinary.uploader.destroy(public_id)
            return result.get("result") == "ok"
        except Exception:
            return False
    
    @classmethod
    async def delete_images(cls, public_ids: list) -> dict:
        """
        Delete multiple images from Cloudinary.
        
        Args:
            public_ids: List of Cloudinary public IDs
            
        Returns:
            Dict with deletion results
        """
        try:
            result = cloudinary.api.delete_resources(public_ids)
            return result
        except Exception as e:
            return {"error": str(e)}

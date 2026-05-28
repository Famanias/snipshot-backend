"""
SnipShot Engine — Lean manga/manhwa/manhua translation pipeline.

Models:
  - Detection:   DBNet + ResNet-34
  - OCR:         Roformer + XPos ViT (48px)
  - Translation: Groq LLM API
  - Inpainting:  LaMa Large (FFC)
  - Rendering:   Default FreeType algorithmic renderer

Usage:
    from snipshot_engine import SnipshotTranslator, Config
"""

from .config import Config
from .translator import SnipshotTranslator

__all__ = ["Config", "SnipshotTranslator"]

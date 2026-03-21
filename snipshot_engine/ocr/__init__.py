"""OCR module — 48px ConvNext + Roformer + XPos beam-search OCR."""

from typing import List, Optional
import numpy as np

from ..config import Ocr, OcrConfig
from ..utils import Quadrilateral
from .model_48px import Model48pxOCR

OCRS = {
    Ocr.ocr48px: Model48pxOCR,
}

_cache = {}


def _get_ocr(key: Ocr):
    if key not in OCRS:
        raise ValueError(f"Unknown OCR: {key}")
    if key not in _cache:
        _cache[key] = OCRS[key]()
    return _cache[key]


async def prepare(ocr_key: Ocr = Ocr.ocr48px, device: str = "cpu"):
    ocr = _get_ocr(ocr_key)
    await ocr.download()
    await ocr.load(device)


async def dispatch(
    ocr_key: Ocr,
    image: np.ndarray,
    regions: List[Quadrilateral],
    config: Optional[OcrConfig] = None,
    device: str = "cpu",
    verbose: bool = False,
) -> List[Quadrilateral]:
    ocr = _get_ocr(ocr_key)
    await ocr.load(device)
    config = config or OcrConfig()
    return await ocr.infer(image, regions, config, verbose)


async def unload(ocr_key: Ocr):
    _cache.pop(ocr_key, None)

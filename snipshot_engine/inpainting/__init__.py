from typing import Optional

import numpy as np

from .lama import LamaLargeInpainter
from ..config import Inpainter, InpainterConfig

INPAINTERS = {
    Inpainter.lama_large: LamaLargeInpainter,
}

_cache = {}


def _get_inpainter(key: Inpainter):
    if key == Inpainter.none:
        return None
    if key not in INPAINTERS:
        raise ValueError(f'Unknown inpainter: "{key}". Choose from: {list(INPAINTERS)}')
    if key not in _cache:
        _cache[key] = INPAINTERS[key]()
    return _cache[key]


async def prepare(inpainter_key: Inpainter, device: str = "cpu"):
    inpainter = _get_inpainter(inpainter_key)
    if inpainter is not None:
        await inpainter.download()
        await inpainter.load(device)


async def dispatch(
    inpainter_key: Inpainter,
    image: np.ndarray,
    mask: np.ndarray,
    config: Optional[InpainterConfig] = None,
    inpainting_size: int = 2048,
    device: str = "cpu",
    verbose: bool = False,
) -> np.ndarray:
    if inpainter_key == Inpainter.none:
        return image
    inpainter = _get_inpainter(inpainter_key)
    await inpainter.load(device)
    config = config or InpainterConfig()
    return await inpainter.infer(image, mask, config, inpainting_size, verbose)


async def unload(inpainter_key: Inpainter):
    _cache.pop(inpainter_key, None)

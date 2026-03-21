"""
Utility functions and classes ported from manga_translator/utils.

Only the pieces actually needed by the snipshot_engine pipeline are included.
"""

from .generic import (
    Context,
    AvgMeter,
    BBox,
    Quadrilateral,
    quadrilateral_can_merge_region,
    sort_pnts,
    load_image,
    dump_image,
    resize_keep_aspect,
    image_resize,
    chunks,
    repeating_sequence,
    is_valuable_text,
    download_url_with_progressbar,
    get_digest,
    get_filename_from_url,
    BASE_PATH,
    MODULE_PATH,
)
from .generic2 import (
    color_difference,
    is_punctuation,
    is_whitespace,
    is_control,
    is_valuable_char,
    is_right_to_left_char,
)
from .textblock import TextBlock, LANGUAGE_ORIENTATION_PRESETS, rotate_polygons
from .inference import ModelWrapper
from .log import get_logger
from .sort import sort_regions
from .bubble import is_ignore

__all__ = [
    "Context",
    "AvgMeter",
    "BBox",
    "Quadrilateral",
    "quadrilateral_can_merge_region",
    "sort_pnts",
    "load_image",
    "dump_image",
    "resize_keep_aspect",
    "image_resize",
    "chunks",
    "repeating_sequence",
    "is_valuable_text",
    "download_url_with_progressbar",
    "get_digest",
    "get_filename_from_url",
    "BASE_PATH",
    "MODULE_PATH",
    "color_difference",
    "is_punctuation",
    "is_whitespace",
    "is_control",
    "is_valuable_char",
    "is_right_to_left_char",
    "TextBlock",
    "LANGUAGE_ORIENTATION_PRESETS",
    "rotate_polygons",
    "ModelWrapper",
    "get_logger",
    "sort_regions",
    "is_ignore",
]

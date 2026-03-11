"""
Minimal configuration for SnipShot Engine.

Only the models/options actually used are defined here.
"""

import re
from enum import Enum
from typing import Optional, List

from pydantic import BaseModel, Field


# ── Enums (single-choice where we only support one model) ────────────────

class Detector(str, Enum):
    default = "default"  # DBNet + ResNet-34


class Ocr(str, Enum):
    ocr48px = "48px"  # Roformer + XPos ViT


class Translator(str, Enum):
    groq = "groq"


class Inpainter(str, Enum):
    lama_large = "lama_large"
    none = "none"


class Renderer(str, Enum):
    default = "default"
    manga2Eng = "manga2eng"
    none = "none"


class Alignment(str, Enum):
    auto = "auto"
    left = "left"
    center = "center"
    right = "right"


class Direction(str, Enum):
    auto = "auto"
    h = "horizontal"
    v = "vertical"


class InpaintPrecision(str, Enum):
    fp32 = "fp32"
    fp16 = "fp16"
    bf16 = "bf16"


# ── Helper ───────────────────────────────────────────────────────────────

def hex2rgb(h: str):
    h = h.lstrip("#")
    return tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))


# ── Sub-configs ──────────────────────────────────────────────────────────

class RenderConfig(BaseModel):
    renderer: Renderer = Renderer.default
    alignment: Alignment = Alignment.auto
    disable_font_border: bool = False
    font_size_offset: int = 0
    font_size_minimum: int = -1
    direction: Direction = Direction.auto
    uppercase: bool = False
    lowercase: bool = False
    no_hyphenation: bool = False
    font_color: Optional[str] = None
    line_spacing: Optional[int] = None
    font_size: Optional[int] = None
    rtl: bool = True

    _font_color_fg = None
    _font_color_bg = None

    @property
    def font_color_fg(self):
        if self.font_color and not self._font_color_fg:
            colors = self.font_color.split(":")
            try:
                self._font_color_fg = hex2rgb(colors[0]) if colors[0] else None
                self._font_color_bg = (
                    hex2rgb(colors[1]) if len(colors) > 1 and colors[1] else None
                )
            except Exception:
                raise ValueError(
                    f"Invalid --font-color value: {self.font_color}. Use a hex value such as FF0000"
                )
        return self._font_color_fg

    @property
    def font_color_bg(self):
        if self.font_color and not self._font_color_bg:
            self.font_color_fg  # trigger parsing
        return self._font_color_bg


class TranslatorConfig(BaseModel):
    translator: Translator = Translator.groq
    target_lang: str = "ENG"
    no_text_lang_skip: bool = False
    skip_lang: Optional[str] = None


class DetectorConfig(BaseModel):
    detector: Detector = Detector.default
    detection_size: int = 1536
    text_threshold: float = 0.5
    box_threshold: float = 0.7
    unclip_ratio: float = 2.3
    det_rotate: bool = False
    det_auto_rotate: bool = False
    det_invert: bool = False
    det_gamma_correct: bool = False


class InpainterConfig(BaseModel):
    inpainter: Inpainter = Inpainter.lama_large
    inpainting_size: int = 2048
    inpainting_precision: InpaintPrecision = InpaintPrecision.bf16


class OcrConfig(BaseModel):
    ocr: Ocr = Ocr.ocr48px
    min_text_length: int = 0
    ignore_bubble: int = 0
    prob: Optional[float] = None


# ── Top-level config ─────────────────────────────────────────────────────

class Config(BaseModel):
    render: RenderConfig = RenderConfig()
    translator: TranslatorConfig = TranslatorConfig()
    detector: DetectorConfig = DetectorConfig()
    inpainter: InpainterConfig = InpainterConfig()
    ocr: OcrConfig = OcrConfig()

    kernel_size: int = 3
    mask_dilation_offset: int = 30

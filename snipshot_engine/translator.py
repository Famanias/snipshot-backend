"""
SnipShot translation pipeline — single-process orchestrator.

Usage:
    from snipshot_engine import SnipshotTranslator, Config
    translator = SnipshotTranslator(Config())
    result_image = await translator.translate(pil_image)
"""

import cv2
import numpy as np
from PIL import Image
from typing import Optional, List

from .config import Config, Inpainter
from .utils import (
    Context,
    load_image,
    dump_image,
    is_valuable_text,
    sort_regions,
    get_logger,
    LANGUAGE_ORIENTATION_PRESETS,
)

from .detection import DefaultDetector
from .ocr import prepare as prepare_ocr, dispatch as dispatch_ocr, unload as unload_ocr
from .textline_merge import dispatch as dispatch_textline_merge
from .translation import prepare as prepare_translation, dispatch as dispatch_translation
from .mask_refinement import dispatch as dispatch_mask_refinement
from .inpainting import prepare as prepare_inpainting, dispatch as dispatch_inpainting
from .rendering import dispatch as dispatch_rendering

logger = get_logger("translator")


class SnipshotTranslator:
    """Lean single-process manga/manhwa/manhua translation pipeline."""

    def __init__(self, config: Optional[Config] = None, device: str = "cpu"):
        self.config = config or Config()
        self.device = device
        self._detector: Optional[DefaultDetector] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def load_models(self):
        """Pre-download and load all models into memory."""
        # Detection
        self._detector = DefaultDetector()
        await self._detector.download()
        await self._detector.load(self.device)
        # OCR
        await prepare_ocr(self.config.ocr.ocr, self.device)
        # Translation (no-op for API-based translators, but keeps interface consistent)
        await prepare_translation(self.config.translator.translator)
        # Inpainting
        if self.config.inpainter.inpainter != Inpainter.none:
            await prepare_inpainting(self.config.inpainter.inpainter, self.device)

    async def translate(self, image: Image.Image) -> Image.Image:
        """
        Run the full pipeline on a single PIL image.
        Returns translated PIL image.
        """
        cfg = self.config
        ctx = Context()
        ctx.input = image

        # Ensure detector is loaded
        if self._detector is None:
            self._detector = DefaultDetector()
            await self._detector.download()
            await self._detector.load(self.device)

        img_rgb, img_alpha = load_image(image)

        # ── 1. Detection ─────────────────────────────────────────────
        logger.info("Running detection...")
        textlines, mask_raw, _ = await self._detector.infer(
            img_rgb,
            cfg.detector.detection_size,
            cfg.detector.text_threshold,
            cfg.detector.box_threshold,
            cfg.detector.unclip_ratio,
            verbose=False,
        )

        if not textlines:
            logger.info("No text regions detected.")
            return image

        # ── 2. OCR ───────────────────────────────────────────────────
        logger.info("Running OCR on %d regions...", len(textlines))
        textlines = await dispatch_ocr(
            cfg.ocr.ocr, img_rgb, textlines, cfg.ocr, self.device, verbose=False
        )
        textlines = [t for t in textlines if t.text.strip()]
        if not textlines:
            logger.info("No text recognized.")
            return image

        # ── 3. Textline merge ────────────────────────────────────────
        logger.info("Merging textlines...")
        text_regions = await dispatch_textline_merge(
            textlines, img_rgb.shape[1], img_rgb.shape[0], verbose=False
        )

        # Filter short / non-valuable text
        text_regions = [
            r for r in text_regions
            if len(r.text) >= cfg.ocr.min_text_length and is_valuable_text(r.text)
        ]
        if not text_regions:
            logger.info("No valuable text after merge.")
            return image

        # Sort reading order
        text_regions = sort_regions(text_regions, img_rgb.shape[1], img_rgb.shape[0])

        # Set target language on every region
        target_lang = cfg.translator.target_lang
        for region in text_regions:
            region.target_lang = target_lang
            # Detect orientation preset
            preset = LANGUAGE_ORIENTATION_PRESETS.get(target_lang)
            if preset:
                region._direction = preset.get("direction", "auto")
                region.alignment = preset.get("alignment", "auto")

        # ── 4. Translation ───────────────────────────────────────────
        logger.info("Translating %d regions...", len(text_regions))
        queries = [r.text for r in text_regions]
        # Detect source language from first region (simplified)
        from_lang = "auto"
        translations = await dispatch_translation(from_lang, target_lang, queries)
        for region, trans in zip(text_regions, translations):
            region.translation = trans

        text_regions = [r for r in text_regions if r.translation.strip()]
        if not text_regions:
            logger.info("No translations produced.")
            return image

        # ── 5. Mask refinement ───────────────────────────────────────
        logger.info("Refining mask...")
        mask = await dispatch_mask_refinement(
            text_regions,
            img_rgb,
            mask_raw if mask_raw is not None else np.zeros(img_rgb.shape[:2], dtype=np.uint8),
            method="fit_text",
            dilation_offset=cfg.mask_dilation_offset,
            ignore_bubble=cfg.ocr.ignore_bubble,
            kernel_size=cfg.kernel_size,
        )

        # ── 6. Inpainting ───────────────────────────────────────────
        if cfg.inpainter.inpainter != Inpainter.none:
            logger.info("Inpainting...")
            img_inpainted = await dispatch_inpainting(
                cfg.inpainter.inpainter,
                img_rgb,
                mask,
                cfg.inpainter,
                cfg.inpainter.inpainting_size,
                self.device,
                verbose=False,
            )
        else:
            img_inpainted = img_rgb.copy()

        # ── 7. Rendering ────────────────────────────────────────────
        logger.info("Rendering translated text...")
        img_rendered = await dispatch_rendering(
            img_inpainted,
            text_regions,
            font_path="",
            font_size_offset=cfg.render.font_size_offset,
            font_size_minimum=cfg.render.font_size_minimum,
            hyphenate=not cfg.render.no_hyphenation,
            line_spacing=cfg.render.line_spacing,
            disable_font_border=cfg.render.disable_font_border,
        )

        # ── 8. Reassemble ───────────────────────────────────────────
        result = dump_image(image, img_rendered, img_alpha)
        logger.info("Translation complete.")
        return result

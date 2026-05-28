import asyncio
import copy
import json
import traceback
from pathlib import Path
from statistics import mean, median

import cv2
import numpy as np
from PIL import Image

from snipshot_engine import Config
from snipshot_engine.translator import SnipshotTranslator
from snipshot_engine.utils import (
    load_image,
    is_valuable_text,
    sort_regions,
    LANGUAGE_ORIENTATION_PRESETS,
)
from snipshot_engine.ocr import dispatch as dispatch_ocr
from snipshot_engine.textline_merge import dispatch as dispatch_textline_merge
from snipshot_engine.mask_refinement import dispatch as dispatch_mask_refinement
from snipshot_engine.inpainting import dispatch as dispatch_inpainting
from snipshot_engine.rendering import dispatch as dispatch_rendering
import snipshot_engine.rendering as rendering_mod


SAMPLE_DIR = Path("sample_pages")
OUT_JSON = SAMPLE_DIR / "diagnostics_baseline.json"
OUT_MD = SAMPLE_DIR / "diagnostics_baseline.md"


def complexity_from_name(name: str) -> str:
    n = name.lower()
    if "easy" in n:
        return "easy"
    if "medium" in n:
        return "medium"
    if "complex" in n:
        return "complex"
    return "unknown"


def bbox_dims_from_dst_points(dst_points: np.ndarray) -> tuple[int, int]:
    pts = dst_points[0].astype(np.int32)
    x, y, w, h = cv2.boundingRect(pts)
    return max(1, int(w)), max(1, int(h))


def safe_mean(vals):
    return float(mean(vals)) if vals else 0.0


def safe_median(vals):
    return float(median(vals)) if vals else 0.0


def pct(vals, predicate):
    if not vals:
        return 0.0
    return 100.0 * sum(1 for v in vals if predicate(v)) / len(vals)


async def process_one_image(translator: SnipshotTranslator, image_path: Path, cfg: Config):
    pil = Image.open(image_path).convert("RGB")
    img_rgb, _ = load_image(pil)

    # 1) Detection
    textlines, mask_raw, _ = await translator._detector.infer(
        img_rgb,
        cfg.detector.detection_size,
        cfg.detector.text_threshold,
        cfg.detector.box_threshold,
        cfg.detector.unclip_ratio,
        verbose=False,
    )

    if not textlines:
        return {
            "image": image_path.name,
            "complexity": complexity_from_name(image_path.name),
            "size": [pil.size[0], pil.size[1]],
            "regions": 0,
            "note": "no_textlines_detected",
            "metrics": {},
            "region_metrics": [],
        }

    # 2) OCR
    textlines = await dispatch_ocr(cfg.ocr.ocr, img_rgb, textlines, cfg.ocr, translator.device, verbose=False)
    textlines = [t for t in textlines if t.text.strip()]
    if not textlines:
        return {
            "image": image_path.name,
            "complexity": complexity_from_name(image_path.name),
            "size": [pil.size[0], pil.size[1]],
            "regions": 0,
            "note": "no_text_after_ocr",
            "metrics": {},
            "region_metrics": [],
        }

    # 3) Merge
    text_regions = await dispatch_textline_merge(textlines, img_rgb.shape[1], img_rgb.shape[0], verbose=False)
    text_regions = [r for r in text_regions if is_valuable_text(r.text)]
    if not text_regions:
        return {
            "image": image_path.name,
            "complexity": complexity_from_name(image_path.name),
            "size": [pil.size[0], pil.size[1]],
            "regions": 0,
            "note": "no_regions_after_merge",
            "metrics": {},
            "region_metrics": [],
        }

    text_regions = sort_regions(text_regions, img_rgb.shape[1], img_rgb.shape[0])

    # For diagnostics we keep translation=source text to isolate rendering mechanics.
    # This baseline intentionally avoids external translation variability.
    target_lang = cfg.translator.target_lang
    for region in text_regions:
        region.target_lang = target_lang
        region.translation = region.text
        preset = LANGUAGE_ORIENTATION_PRESETS.get(target_lang)
        if preset:
            region._direction = preset

    # 4) Mask refinement + inpainting (for realistic bubble detection input)
    mask = await dispatch_mask_refinement(
        text_regions,
        img_rgb,
        mask_raw if mask_raw is not None else np.zeros(img_rgb.shape[:2], dtype=np.uint8),
        method="fit_text",
        dilation_offset=cfg.mask_dilation_offset,
        ignore_bubble=cfg.ocr.ignore_bubble,
        kernel_size=cfg.kernel_size,
    )

    img_inpainted = await dispatch_inpainting(
        cfg.inpainter.inpainter,
        img_rgb,
        mask,
        cfg.inpainter,
        cfg.inpainter.inpainting_size,
        translator.device,
        verbose=False,
    )

    # Prepare instrumentation around rendering internals (runtime only)
    region_metrics = []
    idx_by_obj = {id(r): i for i, r in enumerate(text_regions)}
    orig_font_sizes = {id(r): int(max(1, getattr(r, "font_size", 1))) for r in text_regions}

    original_detect_bubbles = rendering_mod.detect_bubbles
    original_render_region = rendering_mod._render_region

    bubble_flags = {}

    def wrapped_detect_bubbles(inpainted_img, regions, min_bubble_area=800, max_bubble_ratio=0.3, padding=12):
        rects = original_detect_bubbles(inpainted_img, regions, min_bubble_area, max_bubble_ratio, padding)
        for rg, rc in zip(regions, rects):
            bubble_flags[id(rg)] = rc is not None
        return rects

    def wrapped_render_region(img, region, dst_points, hyphenate, line_spacing, disable_font_border):
        rid = id(region)
        idx = idx_by_obj.get(rid, -1)

        # Reconstruct target box dimensions used by renderer
        target_w, target_h = bbox_dims_from_dst_points(dst_points)

        forced_dir = region._direction if hasattr(region, "_direction") else region.direction
        render_h = (forced_dir in ("horizontal", "h")) if forced_dir != "auto" else region.horizontal

        # Emulate layout geometry to compute overflow/spacing metrics
        overflow_w_ratio = 0.0
        overflow_h_ratio = 0.0
        shrink_factor = 1.0
        line_count = 1
        spacing_px = 0

        text_for_render = region.get_translation_for_rendering()
        margin = rendering_mod._compute_inner_margin(target_w, target_h, len(text_for_render))
        inner_w = max(1, target_w - margin * 2)
        inner_h = max(1, target_h - margin * 2)

        fg, bg = region.get_font_colors()
        fg, bg = rendering_mod._fg_bg_compare(fg, bg)
        if disable_font_border:
            bg = None

        if render_h:
            lines, _ = rendering_mod.text_render.calc_horizontal(
                region.font_size,
                text_for_render,
                target_w,
                target_h,
                getattr(region, "target_lang", "en_US"),
                hyphenate,
            )
            line_count = max(1, len(lines))
            spacing_ratio = line_spacing if line_spacing else 0.15
            spacing_px = max(int(region.font_size * spacing_ratio), 3)
            bg_size = int(max(region.font_size * 0.07, 1)) if bg is not None else 0
            max_line_w = 0
            for ln in lines:
                max_line_w = max(max_line_w, rendering_mod.text_render.get_string_width(region.font_size, ln))
            temp_w = int(max_line_w + (region.font_size + bg_size) * 2)
            temp_h = int(region.font_size * line_count + spacing_px * max(0, line_count - 1) + (region.font_size + bg_size) * 2)
        else:
            cols, _ = rendering_mod.text_render.calc_vertical(region.font_size, text_for_render, target_h)
            line_count = max(1, len(cols))
            spacing_ratio = line_spacing if line_spacing else 0.10
            spacing_px = max(int(region.font_size * spacing_ratio), 2)
            bg_size = int(max(region.font_size * 0.07, 1)) if bg is not None else 0
            temp_w = int(region.font_size * line_count + spacing_px * max(0, line_count - 1) + (region.font_size + bg_size) * 2)
            temp_h = int(target_h + (region.font_size + bg_size) * 2)

        if temp_w > inner_w:
            overflow_w_ratio = (temp_w - inner_w) / inner_w
        if temp_h > inner_h:
            overflow_h_ratio = (temp_h - inner_h) / inner_h

        if temp_w > 0 and temp_h > 0 and (temp_w > inner_w or temp_h > inner_h):
            shrink_factor = min(inner_w / temp_w, inner_h / temp_h)

        # Integer centering asymmetry due to floor division
        x0 = max(0, (inner_w - min(temp_w, inner_w)) // 2)
        y0 = max(0, (inner_h - min(temp_h, inner_h)) // 2)
        left = x0
        right = max(0, inner_w - min(temp_w, inner_w) - x0)
        top = y0
        bottom = max(0, inner_h - min(temp_h, inner_h) - y0)
        centering_offset_px = abs(left - right) + abs(top - bottom)

        region_metrics.append(
            {
                "idx": idx,
                "text_len": len(text_for_render),
                "horizontal": bool(render_h),
                "bubble_detected": bool(bubble_flags.get(rid, False)),
                "font_size_original": int(orig_font_sizes.get(rid, region.font_size)),
                "font_size_render": int(region.font_size),
                "font_delta": int(region.font_size - orig_font_sizes.get(rid, region.font_size)),
                "target_w": int(target_w),
                "target_h": int(target_h),
                "inner_w": int(inner_w),
                "inner_h": int(inner_h),
                "margin_px": int(margin),
                "line_count": int(line_count),
                "line_spacing_px": int(spacing_px),
                "temp_w": int(temp_w),
                "temp_h": int(temp_h),
                "overflow_w_ratio": float(overflow_w_ratio),
                "overflow_h_ratio": float(overflow_h_ratio),
                "shrink_factor": float(shrink_factor),
                "centering_offset_px": int(centering_offset_px),
            }
        )

        return original_render_region(img, region, dst_points, hyphenate, line_spacing, disable_font_border)

    rendering_mod.detect_bubbles = wrapped_detect_bubbles
    rendering_mod._render_region = wrapped_render_region

    try:
        # Run actual rendering so metrics reflect real dispatch flow.
        _ = await dispatch_rendering(
            img_inpainted.copy(),
            text_regions,
            font_path="",
            font_size_offset=cfg.render.font_size_offset,
            font_size_minimum=cfg.render.font_size_minimum,
            hyphenate=not cfg.render.no_hyphenation,
            line_spacing=cfg.render.line_spacing,
            disable_font_border=cfg.render.disable_font_border,
        )
    finally:
        rendering_mod.detect_bubbles = original_detect_bubbles
        rendering_mod._render_region = original_render_region

    # Aggregate per-image metrics
    bubble_detected = [m["bubble_detected"] for m in region_metrics]
    font_sizes = [m["font_size_render"] for m in region_metrics]
    font_delta = [m["font_delta"] for m in region_metrics]
    spacing = [m["line_spacing_px"] for m in region_metrics]
    overflow = [max(m["overflow_w_ratio"], m["overflow_h_ratio"]) for m in region_metrics]
    shrink = [m["shrink_factor"] for m in region_metrics]
    centering = [m["centering_offset_px"] for m in region_metrics]

    metrics = {
        "regions": len(region_metrics),
        "bubble_detection_rate_pct": pct(bubble_detected, lambda x: x),
        "font_size_mean": safe_mean(font_sizes),
        "font_size_median": safe_median(font_sizes),
        "font_small_lt16_pct": pct(font_sizes, lambda x: x < 16),
        "font_large_gt48_pct": pct(font_sizes, lambda x: x > 48),
        "font_delta_mean": safe_mean(font_delta),
        "line_spacing_px_mean": safe_mean(spacing),
        "multiline_pct": pct([m["line_count"] for m in region_metrics], lambda x: x > 1),
        "overflow_regions_pct": pct(overflow, lambda x: x > 0.0),
        "overflow_mean_ratio": safe_mean(overflow),
        "shrink_applied_pct": pct(shrink, lambda x: x < 0.999),
        "shrink_mean": safe_mean(shrink),
        "centering_offset_px_mean": safe_mean(centering),
        "centering_offset_gt1px_pct": pct(centering, lambda x: x > 1),
    }

    return {
        "image": image_path.name,
        "complexity": complexity_from_name(image_path.name),
        "size": [pil.size[0], pil.size[1]],
        "regions": len(region_metrics),
        "note": "translation_identity_baseline",
        "metrics": metrics,
        "region_metrics": region_metrics,
    }


def aggregate_by_group(results):
    groups = {}
    for r in results:
        grp = r["complexity"]
        groups.setdefault(grp, []).append(r)

    out = {}
    for grp, rows in groups.items():
        all_region_metrics = []
        for row in rows:
            all_region_metrics.extend(row.get("region_metrics", []))

        bubble_detected = [m["bubble_detected"] for m in all_region_metrics]
        font_sizes = [m["font_size_render"] for m in all_region_metrics]
        spacing = [m["line_spacing_px"] for m in all_region_metrics]
        overflow = [max(m["overflow_w_ratio"], m["overflow_h_ratio"]) for m in all_region_metrics]
        shrink = [m["shrink_factor"] for m in all_region_metrics]

        out[grp] = {
            "images": len(rows),
            "regions": len(all_region_metrics),
            "bubble_detection_rate_pct": pct(bubble_detected, lambda x: x),
            "font_size_mean": safe_mean(font_sizes),
            "font_small_lt16_pct": pct(font_sizes, lambda x: x < 16),
            "line_spacing_px_mean": safe_mean(spacing),
            "overflow_regions_pct": pct(overflow, lambda x: x > 0.0),
            "shrink_applied_pct": pct(shrink, lambda x: x < 0.999),
        }

    return out


def to_markdown(results, grouped):
    lines = []
    lines.append("# Baseline Diagnostics (Sample Pages)")
    lines.append("")
    lines.append("Method: detection + OCR + merge + mask refinement + inpainting + rendering dispatch.")
    lines.append("Translation mode: identity baseline (`translation = OCR text`) to isolate rendering behavior.")
    lines.append("")

    lines.append("## Group Summary")
    lines.append("")
    lines.append("| Complexity | Images | Regions | Bubble Detect % | Font Mean | Small Font <16% | Spacing px Mean | Overflow % | Shrink % |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")

    for grp in sorted(grouped.keys()):
        g = grouped[grp]
        lines.append(
            f"| {grp} | {g['images']} | {g['regions']} | {g['bubble_detection_rate_pct']:.1f} | {g['font_size_mean']:.1f} | {g['font_small_lt16_pct']:.1f} | {g['line_spacing_px_mean']:.1f} | {g['overflow_regions_pct']:.1f} | {g['shrink_applied_pct']:.1f} |"
        )

    lines.append("")
    lines.append("## Per Image")
    lines.append("")
    lines.append("| Image | Complexity | Regions | Bubble % | Font Mean | Small<16% | Overflow % | Shrink % |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")

    for row in sorted(results, key=lambda x: x["image"]):
        m = row.get("metrics", {})
        lines.append(
            f"| {row['image']} | {row['complexity']} | {row['regions']} | {m.get('bubble_detection_rate_pct', 0):.1f} | {m.get('font_size_mean', 0):.1f} | {m.get('font_small_lt16_pct', 0):.1f} | {m.get('overflow_regions_pct', 0):.1f} | {m.get('shrink_applied_pct', 0):.1f} |"
        )

    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- Overflow% counts regions where rendered temp text exceeds inner render box before centering/shrink.")
    lines.append("- Shrink% counts regions where `_center_text_in_box` must downscale text (shrink factor < 1.0).")
    lines.append("- Bubble Detect% is based on `detect_bubbles` returning a bubble rect for the region.")

    return "\n".join(lines)


async def main():
    paths = sorted(
        [
            p
            for p in SAMPLE_DIR.iterdir()
            if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
            and "_inpainted" not in p.stem
            and "_translated" not in p.stem
        ]
    )
    if not paths:
        print(f"No image files found in {SAMPLE_DIR}")
        return

    cfg = Config(
        translator={"target_lang": "ENG"},
        detector={"detection_size": 1536, "box_threshold": 0.7},
        inpainter={"inpainter": "lama_large", "inpainting_size": 2048},
    )

    translator = SnipshotTranslator(cfg, device="cpu")
    await translator.load_models()

    results = []
    for p in paths:
        print(f"Processing {p.name}...")
        try:
            row = await process_one_image(translator, p, cfg)
            results.append(row)
            print(f"  regions={row['regions']}")
        except Exception as exc:
            tb = traceback.format_exc()
            results.append(
                {
                    "image": p.name,
                    "complexity": complexity_from_name(p.name),
                    "size": [0, 0],
                    "regions": 0,
                    "note": f"error: {exc}",
                    "traceback": tb,
                    "metrics": {},
                    "region_metrics": [],
                }
            )
            print(f"  ERROR: {exc}")

    grouped = aggregate_by_group(results)

    payload = {
        "config": {
            "detection_size": cfg.detector.detection_size,
            "box_threshold": cfg.detector.box_threshold,
            "inpainting_size": cfg.inpainter.inpainting_size,
            "target_lang": cfg.translator.target_lang,
            "translation_mode": "identity_baseline",
        },
        "summary_by_complexity": grouped,
        "images": results,
    }

    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    OUT_MD.write_text(to_markdown(results, grouped), encoding="utf-8")

    print("\nDone.")
    print(f"JSON: {OUT_JSON}")
    print(f"MD:   {OUT_MD}")


if __name__ == "__main__":
    asyncio.run(main())

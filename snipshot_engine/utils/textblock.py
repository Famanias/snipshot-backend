"""TextBlock class — stores a merged text region for the pipeline."""

import copy
import re
from functools import cached_property
from typing import List, Tuple

import cv2
import numpy as np
import py3langid as langid
from shapely.geometry import Polygon, MultiPoint

from .generic2 import color_difference, is_right_to_left_char, is_valuable_char

LANGUAGE_ORIENTATION_PRESETS = {
    "CHS": "auto", "CHT": "auto", "CSY": "h", "NLD": "h", "ENG": "h",
    "FRA": "h", "DEU": "h", "HUN": "h", "ITA": "h", "JPN": "auto",
    "KOR": "h", "POL": "h", "PTB": "h", "ROM": "h", "RUS": "h",
    "ESP": "h", "TRK": "h", "UKR": "h", "VIN": "h",
    "ARA": "hr", "FIL": "h",
}


class TextBlock:
    """Stores a block of text made up of textlines."""

    def __init__(
        self,
        lines,
        texts=None,
        language="unknown",
        font_size=-1,
        angle=0,
        translation="",
        fg_color=(0, 0, 0),
        bg_color=(0, 0, 0),
        line_spacing=1.0,
        letter_spacing=1.0,
        font_family="",
        bold=False,
        underline=False,
        italic=False,
        direction="auto",
        alignment="auto",
        rich_text="",
        _bounding_rect=None,
        default_stroke_width=0.2,
        font_weight=50,
        source_lang="",
        target_lang="",
        opacity=1.0,
        shadow_radius=0.0,
        shadow_strength=1.0,
        shadow_color=(0, 0, 0),
        shadow_offset=None,
        prob=1,
        **kwargs,
    ):
        self.lines = np.array(lines, dtype=np.int32)
        self.language = language
        self.font_size = round(font_size)
        self.angle = angle
        self._direction = direction

        self.texts = texts if texts is not None else []
        self.text = texts[0] if texts else ""
        if self.text and texts and len(texts) > 1:
            for txt in texts[1:]:
                first_cjk = "\u3000" <= self.text[-1] <= "\u9fff"
                second_cjk = txt and ("\u3000" <= txt[0] <= "\u9fff")
                if first_cjk or second_cjk:
                    self.text += txt
                else:
                    self.text += " " + txt
        self.prob = prob
        self.translation = translation
        self.fg_colors = fg_color
        self.bg_colors = bg_color
        self.font_family = font_family
        self.bold = bold
        self.underline = underline
        self.italic = italic
        self.rich_text = rich_text
        self.line_spacing = line_spacing
        self.letter_spacing = letter_spacing
        self._alignment = alignment
        self._source_lang = source_lang
        self.target_lang = target_lang
        self._bounding_rect = _bounding_rect
        self.default_stroke_width = default_stroke_width
        self.font_weight = font_weight
        self.adjust_bg_color = True
        self.opacity = opacity
        self.shadow_radius = shadow_radius
        self.shadow_strength = shadow_strength
        self.shadow_color = shadow_color
        self.shadow_offset = shadow_offset or [0, 0]

    # ── geometry properties ──────────────────────────────────────────

    @cached_property
    def xyxy(self):
        x1 = self.lines[..., 0].min()
        y1 = self.lines[..., 1].min()
        x2 = self.lines[..., 0].max()
        y2 = self.lines[..., 1].max()
        return np.array([x1, y1, x2, y2]).astype(np.int32)

    @cached_property
    def xywh(self):
        x1, y1, x2, y2 = self.xyxy
        return np.array([x1, y1, x2 - x1, y2 - y1]).astype(np.int32)

    @cached_property
    def center(self) -> np.ndarray:
        xyxy = np.array(self.xyxy)
        return (xyxy[:2] + xyxy[2:]) / 2

    @cached_property
    def unrotated_polygons(self) -> np.ndarray:
        polygons = self.lines.reshape(-1, 8)
        if self.angle != 0:
            polygons = rotate_polygons(self.center, polygons, self.angle)
        return polygons

    @cached_property
    def unrotated_min_rect(self) -> np.ndarray:
        polygons = self.unrotated_polygons
        min_x = polygons[:, ::2].min()
        min_y = polygons[:, 1::2].min()
        max_x = polygons[:, ::2].max()
        max_y = polygons[:, 1::2].max()
        return np.array([[min_x, min_y, max_x, min_y, max_x, max_y, min_x, max_y]]).reshape(-1, 4, 2).astype(np.int64)

    @cached_property
    def min_rect(self) -> np.ndarray:
        polygons = self.unrotated_polygons
        min_x = polygons[:, ::2].min()
        min_y = polygons[:, 1::2].min()
        max_x = polygons[:, ::2].max()
        max_y = polygons[:, 1::2].max()
        min_bbox = np.array([[min_x, min_y, max_x, min_y, max_x, max_y, min_x, max_y]])
        if self.angle != 0:
            min_bbox = rotate_polygons(self.center, min_bbox, -self.angle)
        return min_bbox.clip(0).reshape(-1, 4, 2).astype(np.int64)

    @cached_property
    def unrotated_size(self) -> Tuple[int, int]:
        middle_pts = (self.min_rect[:, [1, 2, 3, 0]] + self.min_rect) / 2
        norm_h = np.linalg.norm(middle_pts[:, 1] - middle_pts[:, 3])
        norm_v = np.linalg.norm(middle_pts[:, 2] - middle_pts[:, 0])
        return norm_h, norm_v

    @cached_property
    def aspect_ratio(self) -> float:
        return self.unrotated_size[0] / self.unrotated_size[1]

    @property
    def polygon_object(self) -> Polygon:
        mr = self.min_rect[0]
        return MultiPoint([tuple(mr[0]), tuple(mr[1]), tuple(mr[2]), tuple(mr[3])]).convex_hull

    @property
    def area(self) -> float:
        return self.polygon_object.area

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        return self.lines[idx]

    # ── font & rendering helpers ─────────────────────────────────────

    def set_font_colors(self, fg, bg):
        self.fg_colors = np.array(fg)
        self.bg_colors = np.array(bg)

    def update_font_colors(self, fg: np.ndarray, bg: np.ndarray):
        n = len(self)
        if n > 0:
            self.fg_colors += fg / n
            self.bg_colors += bg / n

    def get_font_colors(self, bgr=False):
        frgb = np.array(self.fg_colors).astype(np.int32)
        brgb = np.array(self.bg_colors).astype(np.int32)
        if bgr:
            frgb, brgb = frgb[::-1], brgb[::-1]
        if self.adjust_bg_color:
            fg_avg = np.mean(frgb)
            if color_difference(frgb, brgb) < 30:
                brgb = np.array([255, 255, 255] if fg_avg <= 127 else [0, 0, 0])
        return frgb, brgb

    def get_transformed_region(self, img, line_idx, textheight, maxwidth=None):
        im_h, im_w = img.shape[:2]
        line = np.round(np.array(self.lines[line_idx])).astype(np.int64)
        x1 = np.clip(line[:, 0].min(), 0, im_w)
        y1 = np.clip(line[:, 1].min(), 0, im_h)
        x2 = np.clip(line[:, 0].max(), 0, im_w)
        y2 = np.clip(line[:, 1].max(), 0, im_h)
        img_crop = img[y1:y2, x1:x2]

        direction = "v" if getattr(self, "src_is_vertical", False) else "h"
        src_pts = line.copy()
        src_pts[:, 0] -= x1
        src_pts[:, 1] -= y1
        mid = (src_pts[[1, 2, 3, 0]] + src_pts) / 2
        norm_v = np.linalg.norm(mid[2] - mid[0])
        norm_h = np.linalg.norm(mid[1] - mid[3])

        if textheight is None:
            textheight = int(norm_v) if direction == "h" else int(norm_h)
        if norm_v <= 0 or norm_h <= 0:
            return np.zeros((textheight, textheight, 3), dtype=np.uint8)
        ratio = norm_v / norm_h

        if direction == "h":
            h, w = int(textheight), int(round(textheight / ratio))
            dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
            M, _ = cv2.findHomography(src_pts, dst, cv2.RANSAC, 5.0)
            if M is None:
                return np.zeros((textheight, textheight, 3), dtype=np.uint8)
            region = cv2.warpPerspective(img_crop, M, (w, h))
        else:
            w, h = int(textheight), int(round(textheight * ratio))
            dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
            M, _ = cv2.findHomography(src_pts, dst, cv2.RANSAC, 5.0)
            if M is None:
                return np.zeros((textheight, textheight, 3), dtype=np.uint8)
            region = cv2.warpPerspective(img_crop, M, (w, h))
            region = cv2.rotate(region, cv2.ROTATE_90_COUNTERCLOCKWISE)

        if maxwidth is not None:
            rh, rw = region.shape[:2]
            if rw > maxwidth:
                region = cv2.resize(region, (maxwidth, rh))
        return region

    # ── language / direction / alignment ─────────────────────────────

    @property
    def source_lang(self):
        if not self._source_lang:
            self._source_lang = langid.classify(self.text)[0]
        return self._source_lang

    @property
    def direction(self):
        if self._direction not in ("h", "v", "hr", "vr"):
            d = LANGUAGE_ORIENTATION_PRESETS.get(self.target_lang)
            if d in ("h", "v", "hr", "vr"):
                return d
            if len(self.lines) > 0:
                max_area = 0
                largest_ar = 1
                for line in self.lines:
                    a = Polygon(line).area
                    if a > max_area:
                        max_area = a
                        xs, ys = line[:, 0], line[:, 1]
                        w = np.max(xs) - np.min(xs)
                        h = np.max(ys) - np.min(ys)
                        largest_ar = w / h if h > 0 else 1
                return "v" if largest_ar < 1 else "h"
            return "v" if self.aspect_ratio < 1 else "h"
        return self._direction

    @property
    def vertical(self):
        return self.direction.startswith("v")

    @property
    def horizontal(self):
        return self.direction.startswith("h")

    @property
    def alignment(self):
        if self._alignment in ("left", "center", "right"):
            return self._alignment
        if len(self.lines) == 1:
            return "center"
        if self.direction == "h":
            return "center"
        if self.direction == "hr":
            return "right"
        return "left"

    @property
    def stroke_width(self):
        diff = color_difference(*self.get_font_colors())
        return self.default_stroke_width if diff > 15 else 0

    def get_translation_for_rendering(self):
        text = self.translation
        if self.direction.endswith("r"):
            text_list = list(text)
            l2r_idx = -1

            def _reverse(l, i1, i2):
                for j1 in range(i1, i2 - (i2 - i1) // 2):
                    j2 = i2 - (j1 - i1) - 1
                    l[j1], l[j2] = l[j2], l[j1]

            for i, c in enumerate(text):
                if not is_right_to_left_char(c) and is_valuable_char(c):
                    if l2r_idx < 0:
                        l2r_idx = i
                elif l2r_idx >= 0 and i - l2r_idx > 1:
                    _reverse(text_list, l2r_idx, i)
                    l2r_idx = -1
            if l2r_idx >= 0 and len(text) - l2r_idx > 1:
                _reverse(text_list, l2r_idx, len(text_list))
            text = "".join(text_list)
        return text


def rotate_polygons(center, polygons, rotation, new_center=None, to_int=True):
    if rotation == 0:
        return polygons
    if new_center is None:
        new_center = center
    rotation = np.deg2rad(rotation)
    s, c = np.sin(rotation), np.cos(rotation)
    polygons = polygons.astype(np.float32)
    polygons[:, 1::2] -= center[1]
    polygons[:, ::2] -= center[0]
    rotated = np.copy(polygons)
    rotated[:, 1::2] = polygons[:, 1::2] * c - polygons[:, ::2] * s
    rotated[:, ::2] = polygons[:, 1::2] * s + polygons[:, ::2] * c
    rotated[:, 1::2] += new_center[1]
    rotated[:, ::2] += new_center[0]
    return rotated.astype(np.int64) if to_int else rotated

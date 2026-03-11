"""
Core data structures and helpers ported from manga_translator/utils/generic.py.

Includes: Context, BBox, Quadrilateral, load_image, dump_image, resize helpers,
download with progress‑bar, and miscellaneous utilities.
"""

import os
import re
import sys
import hashlib
import functools
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np
import requests
import tqdm
from PIL import Image
from shapely import affinity
from shapely.geometry import Polygon, MultiPoint

from .generic2 import is_valuable_char, is_valuable_text as _is_valuable_text

MODULE_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
BASE_PATH = os.path.dirname(MODULE_PATH)

# ── Context ──────────────────────────────────────────────────────────────

class Context(dict):
    """Namespace-like dict used to pass data through the pipeline."""

    def __init__(self, **kwargs):
        for name in kwargs:
            setattr(self, name, kwargs[name])

    def __getattr__(self, item):
        return self.get(item)

    def __delattr__(self, key):
        return self.__delitem__(key)

    def __setattr__(self, key, value):
        return self.__setitem__(key, value)

    def __getstate__(self):
        return self.copy()

    def __setstate__(self, state):
        self.update(state)

    def __eq__(self, other):
        if not isinstance(other, Context):
            return NotImplemented
        return dict(self) == dict(other)

    def __contains__(self, key):
        return key in self.keys()


# ── Small helpers ────────────────────────────────────────────────────────

def repeating_sequence(s: str):
    """Extracts repeating sequence from string. 'abcabca' -> 'abc'."""
    for i in range(1, len(s) // 2 + 1):
        seq = s[:i]
        if seq * (len(s) // len(seq)) + seq[: len(s) % len(seq)] == s:
            return seq
    return s


def is_valuable_text(text: str) -> bool:
    return _is_valuable_text(text)


def count_valuable_text(text: str) -> int:
    return sum(1 for ch in text if is_valuable_char(ch))


def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


# ── File / network helpers ───────────────────────────────────────────────

def get_digest(file_path: str) -> str:
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while True:
            chunk = f.read(65536)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def get_filename_from_url(url: str, default: str = "") -> str:
    m = re.search(r"/([^/?]+)[^/]*$", url)
    return m.group(1) if m else default


def download_url_with_progressbar(url: str, path: str):
    if os.path.basename(path) in (".", "") or os.path.isdir(path):
        new_filename = get_filename_from_url(url)
        if not new_filename:
            raise Exception("Could not determine filename")
        path = os.path.join(path, new_filename)

    headers = {}
    downloaded_size = 0
    if os.path.isfile(path):
        downloaded_size = os.path.getsize(path)
        headers["Range"] = "bytes=%d-" % downloaded_size
        headers["Accept-Encoding"] = "deflate"

    r = requests.get(url, stream=True, allow_redirects=True, headers=headers, timeout=120)
    if downloaded_size and r.headers.get("Accept-Ranges") != "bytes":
        r = requests.get(url, stream=True, allow_redirects=True, timeout=120)
        downloaded_size = 0
    total = int(r.headers.get("content-length", 0))

    if r.ok:
        with tqdm.tqdm(
            desc=os.path.basename(path),
            initial=downloaded_size,
            total=total + downloaded_size,
            unit="iB",
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            with open(path, "ab" if downloaded_size else "wb") as f:
                for data in r.iter_content(chunk_size=1024):
                    bar.update(f.write(data))
    else:
        raise Exception(f'Couldn\'t resolve url: "{url}" (Error: {r.status_code})')


def prompt_yes_no(query: str, default: bool = None) -> bool:
    s = "%s (%s/%s): " % (query, "Y" if default is True else "y", "N" if default is False else "n")
    while True:
        inp = input(s).lower()
        if inp in ("yes", "y"):
            return True
        if inp in ("no", "n"):
            return False
        if default is not None:
            return default


# ── AvgMeter ─────────────────────────────────────────────────────────────

class AvgMeter:
    def __init__(self):
        self.sum = 0
        self.count = 0

    def reset(self):
        self.sum = 0
        self.count = 0

    def __call__(self, val=None):
        if val is not None:
            self.sum += val
            self.count += 1
        return self.sum / self.count if self.count else 0


# ── Image I/O ────────────────────────────────────────────────────────────

def load_image(img: Image.Image) -> Tuple[np.ndarray, Optional[Image.Image]]:
    if img.mode == "RGBA":
        img.load()
        bg = Image.new("RGB", img.size, (255, 255, 255))
        alpha_ch = img.split()[3]
        bg.paste(img, mask=alpha_ch)
        return np.array(bg), alpha_ch
    if img.mode == "P":
        img = img.convert("RGBA")
        img.load()
        bg = Image.new("RGB", img.size, (255, 255, 255))
        alpha_ch = img.split()[3]
        bg.paste(img, mask=alpha_ch)
        return np.array(bg), alpha_ch
    return np.array(img.convert("RGB")), None


def dump_image(
    img_pil: Image.Image,
    img: np.ndarray,
    alpha_ch: Optional[Image.Image] = None,
) -> Image.Image:
    if alpha_ch is not None:
        if img.shape[2] != 4:
            img = np.concatenate(
                [img.astype(np.uint8), np.array(alpha_ch).astype(np.uint8)[..., None]],
                axis=2,
            )
    else:
        img = img.astype(np.uint8)
    result = img_pil.convert("RGBA").resize((img.shape[1], img.shape[0]))
    result.paste(Image.fromarray(img), mask=alpha_ch)
    return result


# ── Resize helpers ───────────────────────────────────────────────────────

def resize_keep_aspect(img: np.ndarray, size: int) -> np.ndarray:
    ratio = float(size) / max(img.shape[0], img.shape[1])
    new_w = round(img.shape[1] * ratio)
    new_h = round(img.shape[0] * ratio)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR_EXACT)


def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    h, w = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    return cv2.resize(image, dim, interpolation=inter)


# ── BBox ─────────────────────────────────────────────────────────────────

class BBox:
    def __init__(self, x, y, w, h, text="", prob=0.0, fg_r=0, fg_g=0, fg_b=0, bg_r=0, bg_g=0, bg_b=0):
        self.x, self.y, self.w, self.h = int(x), int(y), int(w), int(h)
        self.text = text
        self.prob = prob
        self.fg_r, self.fg_g, self.fg_b = fg_r, fg_g, fg_b
        self.bg_r, self.bg_g, self.bg_b = bg_r, bg_g, bg_b

    @property
    def xywh(self):
        return np.array([self.x, self.y, self.w, self.h], dtype=np.int32)

    def to_points(self):
        tl = np.array([self.x, self.y])
        tr = np.array([self.x + self.w, self.y])
        br = np.array([self.x + self.w, self.y + self.h])
        bl = np.array([self.x, self.y + self.h])
        return tl, tr, br, bl


# ── sort_pnts ────────────────────────────────────────────────────────────

def sort_pnts(pts: np.ndarray):
    if isinstance(pts, list):
        pts = np.array(pts)
    assert isinstance(pts, np.ndarray) and pts.shape == (4, 2)
    pairwise_vec = (pts[:, None] - pts[None]).reshape((16, -1))
    pairwise_vec_norm = np.linalg.norm(pairwise_vec, axis=1)
    long_side_ids = np.argsort(pairwise_vec_norm)[[8, 10]]
    long_side_vecs = pairwise_vec[long_side_ids]
    if (long_side_vecs[0] * long_side_vecs[1]).sum() < 0:
        long_side_vecs[0] = -long_side_vecs[0]
    struc_vec = np.abs(long_side_vecs.mean(axis=0))
    is_vertical = struc_vec[0] <= struc_vec[1]

    if is_vertical:
        pts = pts[np.argsort(pts[:, 1])]
        pts = pts[[*np.argsort(pts[:2, 0]), *np.argsort(pts[2:, 0])[::-1] + 2]]
        return pts, is_vertical
    else:
        pts = pts[np.argsort(pts[:, 0])]
        pts_sorted = np.zeros_like(pts)
        pts_sorted[[0, 3]] = sorted(pts[[0, 1]], key=lambda x: x[1])
        pts_sorted[[1, 2]] = sorted(pts[[2, 3]], key=lambda x: x[1])
        return pts_sorted, is_vertical


# ── Quadrilateral ────────────────────────────────────────────────────────

class Quadrilateral:
    """Stores a single detected textline with geometry helpers."""

    def __init__(self, pts: np.ndarray, text: str, prob: float,
                 fg_r=0, fg_g=0, fg_b=0, bg_r=0, bg_g=0, bg_b=0):
        self.pts, is_vertical = sort_pnts(pts)
        self.direction = "v" if is_vertical else "h"
        self.text = text
        self.prob = prob
        self.fg_r, self.fg_g, self.fg_b = fg_r, fg_g, fg_b
        self.bg_r, self.bg_g, self.bg_b = bg_r, bg_g, bg_b
        self.assigned_direction: Optional[str] = None

    @functools.cached_property
    def structure(self):
        p1 = ((self.pts[0] + self.pts[1]) / 2).astype(int)
        p2 = ((self.pts[2] + self.pts[3]) / 2).astype(int)
        p3 = ((self.pts[1] + self.pts[2]) / 2).astype(int)
        p4 = ((self.pts[3] + self.pts[0]) / 2).astype(int)
        return [p1, p2, p3, p4]

    @property
    def fg_colors(self):
        return np.array([self.fg_r, self.fg_g, self.fg_b])

    @property
    def bg_colors(self):
        return np.array([self.bg_r, self.bg_g, self.bg_b])

    @functools.cached_property
    def aspect_ratio(self) -> float:
        [l1a, l1b, l2a, l2b] = [a.astype(np.float32) for a in self.structure]
        v1 = l1b - l1a
        v2 = l2b - l2a
        return np.linalg.norm(v2) / np.linalg.norm(v1)

    @functools.cached_property
    def font_size(self) -> float:
        [l1a, l1b, l2a, l2b] = [a.astype(np.float32) for a in self.structure]
        v1 = l1b - l1a
        v2 = l2b - l2a
        return min(np.linalg.norm(v2), np.linalg.norm(v1))

    @functools.cached_property
    def aabb(self) -> BBox:
        mx = np.max(self.pts, axis=0)
        mn = np.min(self.pts, axis=0)
        return BBox(mn[0], mn[1], mx[0] - mn[0], mx[1] - mn[1],
                     self.text, self.prob, self.fg_r, self.fg_g, self.fg_b,
                     self.bg_r, self.bg_g, self.bg_b)

    @functools.cached_property
    def xyxy(self):
        return self.aabb.x, self.aabb.y, self.aabb.x + self.aabb.w, self.aabb.y + self.aabb.h

    @functools.cached_property
    def area(self) -> float:
        return Polygon(self.pts).area

    @functools.cached_property
    def centroid(self) -> np.ndarray:
        return np.mean(self.pts, axis=0)

    @functools.cached_property
    def angle(self) -> float:
        [l1a, l1b, l2a, l2b] = [a.astype(np.float32) for a in self.structure]
        v = l1b - l1a
        return np.arctan2(v[0], v[1])

    @functools.cached_property
    def is_approximate_axis_aligned(self) -> bool:
        return abs(self.angle) < 15 * np.pi / 180

    def distance(self, other: "Quadrilateral") -> float:
        return Polygon(self.pts).distance(Polygon(other.pts))

    def poly_distance(self, other: "Quadrilateral") -> float:
        return self.distance(other)

    def get_transformed_region(self, img, direction, textheight) -> np.ndarray:
        [l1a, l1b, l2a, l2b] = [a.astype(np.float32) for a in self.structure]
        v_vec = l1b - l1a
        h_vec = l2b - l2a
        ratio = np.linalg.norm(v_vec) / np.linalg.norm(h_vec)

        src_pts = self.pts.astype(np.int64).copy()
        im_h, im_w = img.shape[:2]
        x1, y1 = np.clip(src_pts[:, 0].min(), 0, im_w), np.clip(src_pts[:, 1].min(), 0, im_h)
        x2, y2 = np.clip(src_pts[:, 0].max(), 0, im_w), np.clip(src_pts[:, 1].max(), 0, im_h)
        img_crop = img[y1:y2, x1:x2]
        src_pts[:, 0] -= x1
        src_pts[:, 1] -= y1

        self.assigned_direction = direction
        if direction == "h":
            h = max(int(textheight), 2)
            w = max(int(round(textheight / ratio)), 2)
            dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
            M, _ = cv2.findHomography(src_pts, dst, cv2.RANSAC, 5.0)
            return cv2.warpPerspective(img_crop, M, (w, h))
        else:  # 'v'
            w = max(int(textheight), 2)
            h = max(int(round(textheight * ratio)), 2)
            dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
            M, _ = cv2.findHomography(src_pts, dst, cv2.RANSAC, 5.0)
            region = cv2.warpPerspective(img_crop, M, (w, h))
            return cv2.rotate(region, cv2.ROTATE_90_COUNTERCLOCKWISE)


# ── quadrilateral_can_merge_region ───────────────────────────────────────

def quadrilateral_can_merge_region(
    a: Quadrilateral,
    b: Quadrilateral,
    ratio=1.9,
    discard_connection_gap=2,
    char_gap_tolerance=0.6,
    char_gap_tolerance2=1.5,
    font_size_ratio_tol=1.5,
    aspect_ratio_tol=2,
) -> bool:
    b1, b2 = a.aabb, b.aabb
    char_size = min(a.font_size, b.font_size)
    x1, y1, w1, h1 = b1.x, b1.y, b1.w, b1.h
    x2, y2, w2, h2 = b2.x, b2.y, b2.w, b2.h

    p1 = Polygon(a.pts)
    p2 = Polygon(b.pts)
    d = p1.distance(p2)
    if d > discard_connection_gap * char_size:
        return False
    if max(a.font_size, b.font_size) / char_size > font_size_ratio_tol:
        return False
    if a.aspect_ratio > aspect_ratio_tol and b.aspect_ratio < 1.0 / aspect_ratio_tol:
        return False
    if b.aspect_ratio > aspect_ratio_tol and a.aspect_ratio < 1.0 / aspect_ratio_tol:
        return False

    a_aa = a.is_approximate_axis_aligned
    b_aa = b.is_approximate_axis_aligned

    if a_aa and b_aa:
        if d < char_size * char_gap_tolerance:
            if abs(x1 + w1 // 2 - (x2 + w2 // 2)) < char_gap_tolerance2:
                return True
            if w1 > h1 * ratio and h2 > w2 * ratio:
                return False
            if w2 > h2 * ratio and h1 > w1 * ratio:
                return False
            if w1 > h1 * ratio or w2 > h2 * ratio:
                return (
                    abs(x1 - x2) < char_size * char_gap_tolerance2
                    or abs(x1 + w1 - (x2 + w2)) < char_size * char_gap_tolerance2
                )
            if h1 > w1 * ratio or h2 > w2 * ratio:
                return (
                    abs(y1 - y2) < char_size * char_gap_tolerance2
                    or abs(y1 + h1 - (y2 + h2)) < char_size * char_gap_tolerance2
                )
            return False
        return False

    if abs(a.angle - b.angle) < 15 * np.pi / 180:
        fs = min(a.font_size, b.font_size)
        if a.poly_distance(b) > fs * char_gap_tolerance2:
            return False
        if abs(a.font_size - b.font_size) / fs > 0.25:
            return False
        return True

    return False

# Core FreeType-based text rendering
# Ported from manga_translator/rendering/text_render.py

import os
import re
import cv2
import numpy as np
import freetype
import functools
from pathlib import Path
from typing import Tuple, Optional, List

from ..utils import BASE_PATH, is_punctuation, get_logger

logger = get_logger("text_render")

try:
    from hyphen import Hyphenator
    from hyphen.dictools import LANGUAGES as HYPHENATOR_LANGUAGES
    from langcodes import standardize_tag
    HAS_HYPHEN = True
    try:
        HYPHENATOR_LANGUAGES.remove('fr')
        HYPHENATOR_LANGUAGES.append('fr_FR')
    except Exception:
        pass
except ImportError:
    HAS_HYPHEN = False

# ---------------------------------------------------------------------------
# CJK horizontal-to-vertical character mapping
# ---------------------------------------------------------------------------

CJK_H2V = {
    "‥": "︰", "—": "︱", "―": "|", "–": "︲", "_": "︳",
    "(": "︵", ")": "︶", "（": "︵", "）": "︶",
    "{": "︷", "}": "︸", "〔": "︹", "〕": "︺",
    "【": "︻", "】": "︼", "《": "︽", "》": "︾",
    "〈": "︿", "〉": "﹀", "「": "﹁", "」": "﹂",
    "『": "﹃", "』": "﹄", "[": "﹇", "]": "﹈",
    "…": "⋮", "⋯": "︙",
    "\u201c": "﹁", "\u201d": "﹂",  # " "
    "\u2018": "﹁", "\u2019": "﹂",  # ' '
    "~": "︴", "〜": "︴", "～": "︴",
    "!": "︕", "?": "︖", ".": "︒", "。": "︒",
    ";": "︔", "；": "︔", ":": "︓", "：": "︓",
    ",": "︐", "，": "︐", "-": "︲", "−": "︲", "・": "·",
}

CJK_V2H = {v: k for k, v in CJK_H2V.items()}


def CJK_Compatibility_Forms_translate(cdpt: str, direction: int):
    if cdpt == 'ー' and direction == 1:
        return 'ー', 90
    if cdpt in CJK_V2H and direction == 0:
        return CJK_V2H[cdpt], 0
    if cdpt in CJK_H2V and direction == 1:
        return CJK_H2V[cdpt], 0
    return cdpt, 0


def compact_special_symbols(text: str) -> str:
    text = text.replace('...', '…').replace('..', '…')
    text = re.sub(r'([^\w\s])[ \u3000]+', r'\1', text)
    return text


# ---------------------------------------------------------------------------
# Font management
# ---------------------------------------------------------------------------

FALLBACK_FONTS = [
    os.path.join(BASE_PATH, 'fonts/Arial-Unicode-Regular.ttf'),
    os.path.join(BASE_PATH, 'fonts/msyh.ttc'),
    os.path.join(BASE_PATH, 'fonts/msgothic.ttc'),
]
FONT_SELECTION: List[freetype.Face] = []
_font_cache = {}


def _get_cached_font(path: str) -> freetype.Face:
    path = path.replace('\\', '/')
    if path not in _font_cache:
        _font_cache[path] = freetype.Face(Path(path).open('rb'))
    return _font_cache[path]


def set_font(font_path: str):
    global FONT_SELECTION
    selection = ([font_path] + FALLBACK_FONTS) if font_path else FALLBACK_FONTS
    FONT_SELECTION = [_get_cached_font(p) for p in selection if os.path.isfile(p)]


class _NS:
    """Minimal namespace for glyph caching."""
    pass


class Glyph:
    def __init__(self, glyph):
        self.bitmap = _NS()
        self.bitmap.buffer = glyph.bitmap.buffer
        self.bitmap.rows = glyph.bitmap.rows
        self.bitmap.width = glyph.bitmap.width
        self.advance = _NS()
        self.advance.x = glyph.advance.x
        self.advance.y = glyph.advance.y
        self.bitmap_left = glyph.bitmap_left
        self.bitmap_top = glyph.bitmap_top
        self.metrics = _NS()
        self.metrics.vertBearingX = glyph.metrics.vertBearingX
        self.metrics.vertBearingY = glyph.metrics.vertBearingY
        self.metrics.horiBearingX = glyph.metrics.horiBearingX
        self.metrics.horiBearingY = glyph.metrics.horiBearingY
        self.metrics.horiAdvance = glyph.metrics.horiAdvance
        self.metrics.vertAdvance = glyph.metrics.vertAdvance


@functools.lru_cache(maxsize=1024, typed=True)
def get_char_glyph(cdpt: str, font_size: int, direction: int) -> Glyph:
    for i, face in enumerate(FONT_SELECTION):
        if face.get_char_index(cdpt) == 0 and i != len(FONT_SELECTION) - 1:
            continue
        if direction == 0:
            face.set_pixel_sizes(0, font_size)
        else:
            face.set_pixel_sizes(font_size, 0)
        face.load_char(cdpt)
        return Glyph(face.glyph)


def get_char_border(cdpt: str, font_size: int, direction: int):
    for i, face in enumerate(FONT_SELECTION):
        if face.get_char_index(cdpt) == 0 and i != len(FONT_SELECTION) - 1:
            continue
        if direction == 0:
            face.set_pixel_sizes(0, font_size)
        else:
            face.set_pixel_sizes(font_size, 0)
        face.load_char(cdpt, freetype.FT_LOAD_DEFAULT | freetype.FT_LOAD_NO_BITMAP)
        return face.glyph.get_glyph()


# ---------------------------------------------------------------------------
# Colorization
# ---------------------------------------------------------------------------

def add_color(bw_char_map, color, stroke_char_map, stroke_color):
    if bw_char_map.size == 0:
        return np.zeros((bw_char_map.shape[0], bw_char_map.shape[1], 4), dtype=np.uint8)

    if stroke_color is None:
        x, y, w, h = cv2.boundingRect(bw_char_map)
    else:
        x, y, w, h = cv2.boundingRect(stroke_char_map)

    fg = np.zeros((h, w, 4), dtype=np.uint8)
    fg[:, :, 0] = color[0]
    fg[:, :, 1] = color[1]
    fg[:, :, 2] = color[2]
    fg[:, :, 3] = bw_char_map[y:y + h, x:x + w]

    if stroke_color is None:
        stroke_color = color
    bg = np.zeros((stroke_char_map.shape[0], stroke_char_map.shape[1], 4), dtype=np.uint8)
    bg[:, :, 0] = stroke_color[0]
    bg[:, :, 1] = stroke_color[1]
    bg[:, :, 2] = stroke_color[2]
    bg[:, :, 3] = stroke_char_map

    fg_alpha = fg[:, :, 3:4] / 255.0
    bg_alpha = 1.0 - fg_alpha
    bg[y:y + h, x:x + w] = (fg_alpha * fg + bg_alpha * bg[y:y + h, x:x + w]).astype(np.uint8)
    return bg


# ---------------------------------------------------------------------------
# Character offset helpers
# ---------------------------------------------------------------------------

def get_char_offset_x(font_size: int, cdpt: str):
    c, _ = CJK_Compatibility_Forms_translate(cdpt, 0)
    glyph = get_char_glyph(c, font_size, 0)
    bm = glyph.bitmap
    if bm.rows * bm.width == 0 or len(bm.buffer) != bm.rows * bm.width:
        return glyph.advance.x >> 6
    return glyph.metrics.horiAdvance >> 6


def get_string_width(font_size: int, text: str):
    return sum(get_char_offset_x(font_size, c) for c in text)


# ---------------------------------------------------------------------------
# Horizontal character rendering
# ---------------------------------------------------------------------------

def put_char_horizontal(font_size: int, cdpt: str, pen_l, canvas_text, canvas_border, border_size: int):
    pen = list(pen_l)
    cdpt, _ = CJK_Compatibility_Forms_translate(cdpt, 0)
    slot = get_char_glyph(cdpt, font_size, 0)
    bitmap = slot.bitmap

    if slot.metrics.horiAdvance:
        char_offset_x = slot.metrics.horiAdvance >> 6
    elif slot.advance.x:
        char_offset_x = slot.advance.x >> 6
    else:
        char_offset_x = font_size // 2

    if bitmap.rows * bitmap.width == 0 or len(bitmap.buffer) != bitmap.rows * bitmap.width:
        return char_offset_x

    bm_char = np.array(bitmap.buffer, dtype=np.uint8).reshape((bitmap.rows, bitmap.width))
    cx = pen[0] + slot.bitmap_left
    cy = pen[1] - slot.bitmap_top

    y0 = max(0, cy); x0 = max(0, cx)
    y1 = min(canvas_text.shape[0], cy + bitmap.rows)
    x1 = min(canvas_text.shape[1], cx + bitmap.width)
    if y0 < y1 and x0 < x1:
        sl = bm_char[y0 - cy:y1 - cy, x0 - cx:x1 - cx]
        if sl.size > 0:
            canvas_text[y0:y1, x0:x1] = sl

    if border_size > 0:
        _render_border_horizontal(cdpt, font_size, bitmap, cx, cy, canvas_border)

    return char_offset_x


def _render_border_horizontal(cdpt, font_size, bitmap, cx, cy, canvas_border):
    glyph_b = get_char_border(cdpt, font_size, 0)
    stroker = freetype.Stroker()
    stroker.set(64 * max(int(0.07 * font_size), 1),
                freetype.FT_STROKER_LINEJOIN_ROUND, freetype.FT_STROKER_LINECAP_ROUND, 0)
    glyph_b.stroke(stroker, destroy=True)
    blyph = glyph_b.to_bitmap(freetype.FT_RENDER_MODE_NORMAL, freetype.Vector(0, 0), True)
    bm_b = blyph.bitmap
    if bm_b.rows * bm_b.width == 0 or len(bm_b.buffer) != bm_b.rows * bm_b.width:
        return
    arr = np.array(bm_b.buffer, dtype=np.uint8).reshape((bm_b.rows, bm_b.width))
    # Center-align border with character
    bx = int(round(cx + bitmap.width / 2 - bm_b.width / 2))
    by = int(round(cy + bitmap.rows / 2 - bm_b.rows / 2))
    y0a = max(0, by); x0a = max(0, bx)
    y1a = min(canvas_border.shape[0], by + bm_b.rows)
    x1a = min(canvas_border.shape[1], bx + bm_b.width)
    if y0a < y1a and x0a < x1a:
        sl = arr[y0a - by:y1a - by, x0a - bx:x1a - bx]
        tgt = canvas_border[y0a:y1a, x0a:x1a]
        if sl.shape == tgt.shape:
            canvas_border[y0a:y1a, x0a:x1a] = cv2.add(tgt, sl)


# ---------------------------------------------------------------------------
# Vertical character rendering
# ---------------------------------------------------------------------------

def put_char_vertical(font_size: int, cdpt: str, pen_l, canvas_text, canvas_border, border_size: int):
    pen = list(pen_l)
    cdpt, _ = CJK_Compatibility_Forms_translate(cdpt, 1)
    slot = get_char_glyph(cdpt, font_size, 1)
    bitmap = slot.bitmap

    if bitmap.rows * bitmap.width == 0 or len(bitmap.buffer) != bitmap.rows * bitmap.width:
        if slot.metrics.vertAdvance:
            return slot.metrics.vertAdvance >> 6
        return font_size

    char_offset_y = slot.metrics.vertAdvance >> 6
    bm_char = np.array(bitmap.buffer, dtype=np.uint8).reshape((bitmap.rows, bitmap.width))
    cx = pen[0] + (slot.metrics.vertBearingX >> 6)
    cy = pen[1] + (slot.metrics.vertBearingY >> 6)

    y0 = max(0, cy); x0 = max(0, cx)
    y1 = min(canvas_text.shape[0], cy + bitmap.rows)
    x1 = min(canvas_text.shape[1], cx + bitmap.width)
    if y0 < y1 and x0 < x1:
        sl = bm_char[y0 - cy:y1 - cy, x0 - cx:x1 - cx]
        if sl.size > 0:
            canvas_text[y0:y1, x0:x1] = sl

    if border_size > 0:
        _render_border_vertical(cdpt, font_size, bitmap, cx, cy, canvas_border)

    return char_offset_y


def _render_border_vertical(cdpt, font_size, bitmap, cx, cy, canvas_border):
    glyph_b = get_char_border(cdpt, font_size, 1)
    stroker = freetype.Stroker()
    stroker.set(64 * max(int(0.07 * font_size), 1),
                freetype.FT_STROKER_LINEJOIN_ROUND, freetype.FT_STROKER_LINECAP_ROUND, 0)
    glyph_b.stroke(stroker, destroy=True)
    blyph = glyph_b.to_bitmap(freetype.FT_RENDER_MODE_NORMAL, freetype.Vector(0, 0), True)
    bm_b = blyph.bitmap
    if bm_b.rows * bm_b.width == 0 or len(bm_b.buffer) != bm_b.rows * bm_b.width:
        return
    arr = np.array(bm_b.buffer, dtype=np.uint8).reshape((bm_b.rows, bm_b.width))
    bx = int(round(cx + bitmap.width / 2 - bm_b.width / 2))
    by = int(round(cy + bitmap.rows / 2 - bm_b.rows / 2))
    y0a = max(0, by); x0a = max(0, bx)
    y1a = min(canvas_border.shape[0], by + bm_b.rows)
    x1a = min(canvas_border.shape[1], bx + bm_b.width)
    if y0a < y1a and x0a < x1a:
        sl = arr[y0a - by:y1a - by, x0a - bx:x1a - bx]
        tgt = canvas_border[y0a:y1a, x0a:x1a]
        if sl.shape == tgt.shape:
            canvas_border[y0a:y1a, x0a:x1a] = cv2.add(tgt, sl)


# ---------------------------------------------------------------------------
# Layout calculation
# ---------------------------------------------------------------------------

def calc_vertical(font_size: int, text: str, max_height: int):
    line_text_list = []
    line_height_list = []
    line_str = ""
    line_height = 0
    for cdpt in text:
        if line_height == 0 and cdpt == ' ':
            continue
        cdpt, _ = CJK_Compatibility_Forms_translate(cdpt, 1)
        ckpt = get_char_glyph(cdpt, font_size, 1)
        bm = ckpt.bitmap
        if bm.rows * bm.width == 0 or len(bm.buffer) != bm.rows * bm.width:
            char_offset_y = ckpt.metrics.vertBearingY >> 6
        else:
            char_offset_y = ckpt.metrics.vertAdvance >> 6
        if line_height + char_offset_y > max_height:
            line_text_list.append(line_str)
            line_height_list.append(line_height)
            line_str = ""
            line_height = 0
        line_height += char_offset_y
        line_str += cdpt
    line_text_list.append(line_str)
    line_height_list.append(line_height)
    return line_text_list, line_height_list


def select_hyphenator(lang: str):
    if not HAS_HYPHEN:
        return None
    lang = standardize_tag(lang)
    if lang not in HYPHENATOR_LANGUAGES:
        for avail in reversed(HYPHENATOR_LANGUAGES):
            if avail.startswith(lang):
                lang = avail
                break
        else:
            return None
    try:
        return Hyphenator(lang)
    except Exception:
        return None


def calc_horizontal(font_size: int, text: str, max_width: int, max_height: int,
                    language: str = 'en_US', hyphenate: bool = True):
    max_width = max(max_width, 2 * font_size)
    whitespace_offset_x = get_char_offset_x(font_size, ' ')
    hyphen_offset_x = get_char_offset_x(font_size, '-')

    words = re.split(r'\s+', text)
    word_widths = [get_string_width(font_size, w) for w in words]

    # Expand width if overflow is unavoidable
    while True:
        max_lines = max_height // font_size + 1
        expected = sum(word_widths) + max((len(word_widths) - 1) * whitespace_offset_x - (max_lines - 1) * hyphen_offset_x, 0)
        if max_width * max_lines < expected:
            m = max(np.sqrt(expected / (max_width * max_lines)), 1.05)
            max_width *= m
            max_height *= m
        else:
            break

    # Split into syllables
    syllables = []
    hyphenator = select_hyphenator(language) if hyphenate else None
    for word in words:
        syls = []
        if hyphenator and len(word) <= 100:
            try:
                syls = hyphenator.syllables(word)
            except Exception:
                syls = []
        if not syls:
            syls = [word] if len(word) <= 3 else list(word)
        normed = []
        for s in syls:
            if get_string_width(font_size, s) > max_width:
                normed.extend(list(s))
            else:
                normed.append(s)
        syllables.append(normed)

    line_words_list: list[list[int]] = []
    line_width_list: list[int] = []
    hyphenation_idx_list: list[int] = []
    line_words: list[int] = []
    line_width = 0
    hyph_idx = 0

    def break_line():
        nonlocal line_words, line_width, hyph_idx
        line_words_list.append(line_words)
        line_width_list.append(line_width)
        hyphenation_idx_list.append(hyph_idx)
        line_words = []
        line_width = 0
        hyph_idx = 0

    # Step 1: arrange without hyphenation
    i = 0
    while True:
        if i >= len(words):
            if line_width > 0:
                break_line()
            break
        cw = whitespace_offset_x if line_width > 0 else 0
        if line_width + cw + word_widths[i] <= max_width + hyphen_offset_x:
            line_words.append(i)
            line_width += cw + word_widths[i]
            i += 1
        elif word_widths[i] > max_width:
            j = 0
            hyph_idx = 0
            while j < len(syllables[i]):
                sw = get_string_width(font_size, syllables[i][j])
                if line_width + cw + sw <= max_width:
                    cw += sw
                    j += 1
                    hyph_idx = j
                else:
                    if hyph_idx > 0:
                        line_words.append(i)
                        line_width += cw
                    cw = 0
                    break_line()
            line_words.append(i)
            line_width += cw
            i += 1
        else:
            break_line()

    # Step 2: Backwards hyphenation to fill lines
    def get_present_syllables_range(li, wp):
        while wp < 0:
            wp += len(line_words_list[li])
        wi = line_words_list[li][wp]
        ss = 0
        se = len(syllables[wi])
        if li > 0 and wp == 0 and line_words_list[li - 1][-1] == wi:
            ss = hyphenation_idx_list[li - 1]
        if li < len(line_words_list) - 1 and wp == len(line_words_list[li]) - 1 \
                and line_words_list[li + 1][0] == wi:
            se = hyphenation_idx_list[li]
        return ss, se

    max_lines = max_height // font_size + 1
    if hyphenate and hyphenator and len(line_words_list) > max_lines:
        li = 0
        while li < len(line_words_list) - 1:
            lw1 = line_words_list[li]
            lw2 = line_words_list[li + 1]
            left = max_width - line_width_list[li]
            first_word = True
            while lw2:
                wi = lw2[0]
                if first_word and wi == lw1[-1]:
                    ss = hyphenation_idx_list[li]
                    se = hyphenation_idx_list[li + 1] if (li < len(line_width_list) - 2 and wi == line_words_list[li + 2][0]) else len(syllables[wi])
                else:
                    left -= whitespace_offset_x
                    ss = 0
                    se = len(syllables[wi]) if len(lw2) > 1 else hyphenation_idx_list[li + 1]
                first_word = False
                cw = 0
                for si in range(ss, se):
                    sw = get_string_width(font_size, syllables[wi][si])
                    if left > cw + sw:
                        cw += sw
                    else:
                        if cw > 0:
                            left -= cw
                            line_width_list[li] = max_width - left
                            hyphenation_idx_list[li] = si
                            lw1.append(wi)
                        break
                else:
                    left -= cw
                    line_width_list[li] = max_width - left
                    lw1.append(wi)
                    lw2.pop(0)
                    continue
                break
            if not lw2:
                line_words_list.pop(li + 1)
                line_width_list.pop(li + 1)
                hyphenation_idx_list.pop(li)
            else:
                li += 1

    # Step 3: Move tiny fragments between lines
    li = 0
    while li < len(line_words_list) - 1:
        lw1 = line_words_list[li]
        lw2 = line_words_list[li + 1]
        if lw1[-1] == lw2[0]:
            ss1, se1 = get_present_syllables_range(li, -1)
            t1 = ''.join(syllables[lw1[-1]][ss1:se1])
            ss2, se2 = get_present_syllables_range(li + 1, 0)
            t2 = ''.join(syllables[lw2[0]][ss2:se2])
            w1 = get_string_width(font_size, t1)
            w2 = get_string_width(font_size, t2)
            if len(t2) == 1 or w2 < font_size:
                lw2.pop(0)
                line_width_list[li] += w2
                line_width_list[li + 1] -= w2 + whitespace_offset_x
            elif len(t1) == 1 or w1 < font_size:
                lw1.pop(-1)
                line_width_list[li] -= w1 + whitespace_offset_x
                line_width_list[li + 1] += w1
        if not lw1:
            line_words_list.pop(li); line_width_list.pop(li); hyphenation_idx_list.pop(li)
        elif not lw2:
            line_words_list.pop(li + 1); line_width_list.pop(li + 1); hyphenation_idx_list.pop(li)
        else:
            li += 1

    # Step 4: Assemble
    use_hyphens = hyphenate and hyphenator and max_width > 1.5 * font_size and len(words) > 1
    line_text_list = []
    for i, line in enumerate(line_words_list):
        lt = ''
        for j, wi in enumerate(line):
            ss, se = get_present_syllables_range(i, j)
            lt += ''.join(syllables[wi][ss:se])
            if not lt:
                continue
            if j == 0 and i > 0 and line_text_list[-1][-1] == '-' and lt[0] == '-':
                lt = lt[1:]
                line_width_list[i] -= hyphen_offset_x
            if j < len(line) - 1 and lt:
                lt += ' '
            elif use_hyphens and se != len(syllables[wi]) and len(words[wi]) > 3 and lt[-1] != '-' \
                    and not (se < len(syllables[wi]) and not re.search(r'\w', syllables[wi][se][0])):
                lt += '-'
                line_width_list[i] += hyphen_offset_x
        line_width_list[i] = get_string_width(font_size, lt)
        line_text_list.append(lt)
    return line_text_list, line_width_list


# ---------------------------------------------------------------------------
# High-level text rendering
# ---------------------------------------------------------------------------

def put_text_vertical(font_size: int, text: str, h: int, alignment: str,
                      fg, bg, line_spacing: int):
    text = compact_special_symbols(text)
    if not text:
        return None
    bg_size = int(max(font_size * 0.07, 1)) if bg is not None else 0
    spacing_x = int(font_size * (line_spacing or 0.2))

    num_char_y = max(h // font_size, 1)
    num_char_x = len(text) // num_char_y + 1
    canvas_x = font_size * num_char_x + spacing_x * (num_char_x - 1) + (font_size + bg_size) * 2
    canvas_y = font_size * num_char_y + (font_size + bg_size) * 2
    line_text_list, line_height_list = calc_vertical(font_size, text, h)

    canvas_text = np.zeros((canvas_y, canvas_x), dtype=np.uint8)
    canvas_border = canvas_text.copy()
    pen_orig = [canvas_text.shape[1] - (font_size + bg_size), font_size + bg_size]

    for lt, lh in zip(line_text_list, line_height_list):
        pen_line = pen_orig.copy()
        if alignment == 'center':
            pen_line[1] += (max(line_height_list) - lh) // 2
        elif alignment == 'right':
            pen_line[1] += max(line_height_list) - lh
        for c in lt:
            pen_line[1] += put_char_vertical(font_size, c, pen_line, canvas_text, canvas_border, border_size=bg_size)
        pen_orig[0] -= spacing_x + font_size

    canvas_border = np.clip(canvas_border, 0, 255)
    box = add_color(canvas_text, fg, canvas_border, bg)
    x, y, w, h = cv2.boundingRect(canvas_border)
    return box[y:y + h, x:x + w]


def put_text_horizontal(font_size: int, text: str, width: int, height: int, alignment: str,
                        reversed_direction: bool, fg, bg,
                        lang: str = 'en_US', hyphenate: bool = True, line_spacing: int = 0):
    text = compact_special_symbols(text)
    if not text:
        return None
    bg_size = int(max(font_size * 0.07, 1)) if bg is not None else 0
    spacing_y = int(font_size * (line_spacing or 0.01))

    line_text_list, line_width_list = calc_horizontal(font_size, text, width, height, lang, hyphenate)

    canvas_w = max(line_width_list) + (font_size + bg_size) * 2
    canvas_h = font_size * len(line_width_list) + spacing_y * (len(line_width_list) - 1) + (font_size + bg_size) * 2
    canvas_text = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
    canvas_border = canvas_text.copy()

    pen_orig = [font_size + bg_size, font_size + bg_size]
    if reversed_direction:
        pen_orig[0] = canvas_w - bg_size - 10

    for lt, lw in zip(line_text_list, line_width_list):
        pen_line = pen_orig.copy()
        if alignment == 'center':
            pen_line[0] += (max(line_width_list) - lw) // 2 * (-1 if reversed_direction else 1)
        elif alignment == 'right' and not reversed_direction:
            pen_line[0] += max(line_width_list) - lw
        elif alignment == 'left' and reversed_direction:
            pen_line[0] -= max(line_width_list) - lw
            pen_line[0] = max(lw, pen_line[0])

        for c in lt:
            if reversed_direction:
                cdpt, _ = CJK_Compatibility_Forms_translate(c, 0)
                g = get_char_glyph(cdpt, font_size, 0)
                pen_line[0] -= g.metrics.horiAdvance >> 6
            offset_x = put_char_horizontal(font_size, c, pen_line, canvas_text, canvas_border, border_size=bg_size)
            if not reversed_direction:
                pen_line[0] += offset_x
        pen_orig[1] += spacing_y + font_size

    canvas_border = np.clip(canvas_border, 0, 255)
    box = add_color(canvas_text, fg, canvas_border, bg)
    x, y, w, h = cv2.boundingRect(canvas_border)
    return box[y:y + h, x:x + w]

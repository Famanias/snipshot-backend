"""Mask refinement — CRF-based text mask refinement using pydensecrf."""

import math
from typing import Tuple, List

import cv2
import numpy as np
from tqdm import tqdm
from shapely.geometry import Polygon

from ..utils import Quadrilateral, TextBlock
from ..utils.bubble import is_ignore

try:
    from pydensecrf.utils import unary_from_softmax
    import pydensecrf.densecrf as dcrf
    HAS_CRF = True
except ImportError:
    HAS_CRF = False


# ── helpers ──────────────────────────────────────────────────────────

def _extend_rect(x, y, w, h, max_x, max_y, extend_size):
    x1 = max(x - extend_size, 0)
    y1 = max(y - extend_size, 0)
    w1 = min(w + extend_size * 2, max_x - x1 - 1)
    h1 = min(h + extend_size * 2, max_y - y1 - 1)
    return x1, y1, w1, h1


def _refine_mask(rgbimg, rawmask):
    if not HAS_CRF:
        return rawmask
    if len(rawmask.shape) == 2:
        rawmask = rawmask[:, :, None]
    mask_softmax = np.concatenate([cv2.bitwise_not(rawmask)[:, :, None], rawmask], axis=2)
    mask_softmax = mask_softmax.astype(np.float32) / 255.0
    n_classes = 2
    feat_first = mask_softmax.transpose((2, 0, 1)).reshape((n_classes, -1))
    unary = unary_from_softmax(feat_first)
    unary = np.ascontiguousarray(unary)

    d = dcrf.DenseCRF2D(rgbimg.shape[1], rgbimg.shape[0], n_classes)
    d.setUnaryEnergy(unary)
    d.addPairwiseGaussian(sxy=1, compat=3, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NO_NORMALIZATION)
    d.addPairwiseBilateral(sxy=23, srgb=7, rgbim=rgbimg, compat=20,
                           kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NO_NORMALIZATION)
    Q = d.inference(5)
    res = np.argmax(Q, axis=0).reshape((rgbimg.shape[0], rgbimg.shape[1]))
    return np.array(res * 255, dtype=np.uint8)


def _complete_mask(img, mask, textlines, keep_threshold=1e-2, dilation_offset=0, kernel_size=3):
    bboxes = [txtln.aabb.xywh for txtln in textlines]
    polys = [Polygon(txtln.pts) for txtln in textlines]
    for (x, y, w, h) in bboxes:
        cv2.rectangle(mask, (x, y), (x + w, y + h), 0, 1)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
    M = len(textlines)
    textline_ccs = [np.zeros_like(mask) for _ in range(M)]
    iinfo = np.iinfo(labels.dtype)
    textline_rects = np.full((M, 4), [iinfo.max, iinfo.max, iinfo.min, iinfo.min], dtype=labels.dtype)
    ratio_mat = np.zeros((num_labels, M), dtype=np.float32)
    dist_mat = np.zeros((num_labels, M), dtype=np.float32)
    valid = False

    for label in range(1, num_labels):
        if stats[label, cv2.CC_STAT_AREA] <= 9:
            continue
        x1 = stats[label, cv2.CC_STAT_LEFT]
        y1 = stats[label, cv2.CC_STAT_TOP]
        w1 = stats[label, cv2.CC_STAT_WIDTH]
        h1 = stats[label, cv2.CC_STAT_HEIGHT]
        area1 = stats[label, cv2.CC_STAT_AREA]
        cc_poly = Polygon([[x1, y1], [x1 + w1, y1], [x1 + w1, y1 + h1], [x1, y1 + h1]])

        for tl_idx in range(M):
            area2 = polys[tl_idx].area
            overlapping = polys[tl_idx].intersection(cc_poly).area
            ratio_mat[label, tl_idx] = overlapping / min(area1, area2) if min(area1, area2) > 0 else 0
            dist_mat[label, tl_idx] = polys[tl_idx].distance(cc_poly.centroid)

        avg = np.argmax(ratio_mat[label])
        area2 = polys[avg].area
        if area1 >= area2:
            continue
        if ratio_mat[label, avg] <= keep_threshold:
            avg = np.argmin(dist_mat[label])
            unit = max(min(textlines[avg].font_size, w1, h1), 10)
            if dist_mat[label, avg] >= 0.5 * unit:
                continue

        textline_ccs[avg][y1:y1 + h1, x1:x1 + w1][labels[y1:y1 + h1, x1:x1 + w1] == label] = 255
        textline_rects[avg, 0] = min(textline_rects[avg, 0], x1)
        textline_rects[avg, 1] = min(textline_rects[avg, 1], y1)
        textline_rects[avg, 2] = max(textline_rects[avg, 2], x1 + w1)
        textline_rects[avg, 3] = max(textline_rects[avg, 3], y1 + h1)
        valid = True

    if not valid:
        return None

    textline_rects[:, 2] -= textline_rects[:, 0]
    textline_rects[:, 3] -= textline_rects[:, 1]

    final_mask = np.zeros_like(mask)
    img = cv2.bilateralFilter(img, 17, 80, 80)

    for i, cc in enumerate(tqdm(textline_ccs, "[mask]")):
        x1, y1, w1, h1 = textline_rects[i]
        text_size = min(w1, h1, textlines[i].font_size)
        x1, y1, w1, h1 = _extend_rect(x1, y1, w1, h1, img.shape[1], img.shape[0], int(text_size * 0.1))
        dilate_size = max((int((text_size + dilation_offset) * 0.3) // 2) * 2 + 1, 3)
        kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_size, dilate_size))
        cc_region = np.ascontiguousarray(cc[y1:y1 + h1, x1:x1 + w1])
        if cc_region.size == 0:
            continue
        img_region = np.ascontiguousarray(img[y1:y1 + h1, x1:x1 + w1])
        cc_region = _refine_mask(img_region, cc_region)
        cc[y1:y1 + h1, x1:x1 + w1] = cc_region
        x2, y2, w2, h2 = _extend_rect(x1, y1, w1, h1, img.shape[1], img.shape[0], -(-dilate_size // 2))
        cc[y2:y2 + h2, x2:x2 + w2] = cv2.dilate(cc[y2:y2 + h2, x2:x2 + w2], kern)
        final_mask[y2:y2 + h2, x2:x2 + w2] = cv2.bitwise_or(final_mask[y2:y2 + h2, x2:x2 + w2], cc[y2:y2 + h2, x2:x2 + w2])

    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return cv2.dilate(final_mask, kern)


# ── Module dispatch ──────────────────────────────────────────────────

async def dispatch(
    text_regions: List[TextBlock],
    raw_image: np.ndarray,
    raw_mask: np.ndarray,
    method: str = "fit_text",
    dilation_offset: int = 0,
    ignore_bubble: int = 0,
    kernel_size: int = 3,
) -> np.ndarray:
    scale_factor = max(min((raw_mask.shape[0] - raw_image.shape[0] / 3) / raw_mask.shape[0], 1), 0.5)

    img_resized = cv2.resize(raw_image, (int(raw_image.shape[1] * scale_factor), int(raw_image.shape[0] * scale_factor)), interpolation=cv2.INTER_LINEAR)
    mask_resized = cv2.resize(raw_mask, (int(raw_image.shape[1] * scale_factor), int(raw_image.shape[0] * scale_factor)), interpolation=cv2.INTER_LINEAR)
    mask_resized[mask_resized > 0] = 255

    textlines = []
    for region in text_regions:
        for line in region.lines:
            textlines.append(Quadrilateral(line * scale_factor, "", 0))

    if method == "fit_text":
        final_mask = _complete_mask(img_resized, mask_resized, textlines,
                                    dilation_offset=dilation_offset, kernel_size=kernel_size)
    else:
        final_mask = np.zeros_like(mask_resized)
        for txtln in textlines:
            x, y, w, h = txtln.aabb.xywh
            cv2.rectangle(final_mask, (x, y), (x + w, y + h), 255, -1)

    if final_mask is None:
        final_mask = np.zeros((raw_image.shape[0], raw_image.shape[1]), dtype=np.uint8)
    else:
        final_mask = cv2.resize(final_mask, (raw_image.shape[1], raw_image.shape[0]), interpolation=cv2.INTER_LINEAR)
        final_mask[final_mask > 0] = 255

    if ignore_bubble < 1 or ignore_bubble > 50:
        return final_mask

    # Bubble filtering
    k_size = int(max(final_mask.shape) * 0.025)
    kernel = np.ones((k_size, k_size), np.uint8)
    final_mask = cv2.dilate(final_mask, kernel, iterations=1)
    contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        temp_mask = np.zeros_like(final_mask)
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(temp_mask, (x, y), (x + w, y + h), 255, -1)
        textblock = cv2.bitwise_and(raw_image, raw_image, mask=temp_mask)
        if is_ignore(textblock, ignore_bubble):
            cv2.drawContours(final_mask, [cnt], -1, 0, -1)

    return final_mask

"""CommonDetector base and DefaultDetector implementation."""

import os
import shutil
from abc import abstractmethod
from collections import Counter
from typing import List, Tuple

import cv2
import einops
import numpy as np
import torch

from ..utils import Quadrilateral, ModelWrapper
from ..utils.log import get_logger
from .dbnet_utils import DBNet_resnet34, DBHead, imgproc, dbnet_utils, craft_utils


# ── det_rearrange_forward (ported from manga_translator/utils/generic.py) ──


def _square_pad_resize(img: np.ndarray, tgt_size: int):
    """Pad image to square and resize to tgt_size."""
    h, w = img.shape[:2]
    pad_h = pad_w = 0
    if h > w:
        pad_w = h - w
    elif w > h:
        pad_h = w - h
    if pad_h or pad_w:
        img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    cur = max(h + pad_h, w + pad_w)
    ratio = cur / tgt_size if cur > tgt_size else 1
    if ratio > 1:
        img = cv2.resize(img, (tgt_size, tgt_size), interpolation=cv2.INTER_LINEAR)
    pad_pixel = int(round(max(pad_h, pad_w) / ratio))
    return img, ratio, pad_pixel, pad_pixel


def det_rearrange_forward(
    img: np.ndarray,
    dbnet_batch_forward,
    tgt_size: int = 1280,
    max_batch_size: int = 4,
    device="cpu",
    verbose=False,
):
    h, w = img.shape[:2]
    transpose = False
    if h < w:
        transpose = True
        h, w = img.shape[1], img.shape[0]

    asp_ratio = h / w
    down_scale_ratio = h / tgt_size

    if not (down_scale_ratio > 2.5 and asp_ratio > 3):
        return None, None

    if transpose:
        img = einops.rearrange(img, "h w c -> w h c")

    pw_num = max(int(np.floor(2 * tgt_size / w)), 2)
    patch_size = ph = pw_num * w

    ph_num = int(np.ceil(h / ph))
    ph_step = int((h - ph) / (ph_num - 1)) if ph_num > 1 else 0
    rel_step_list = []
    patch_list = []
    for ii in range(ph_num):
        t = ii * ph_step
        b = t + ph
        rel_step_list.append(t / h)
        patch_list.append(img[t:b])

    p_num = int(np.ceil(ph_num / pw_num))
    pad_num = p_num * pw_num - ph_num
    for _ in range(pad_num):
        patch_list.append(np.zeros_like(patch_list[0]))

    # patch2batches
    patch_arr = np.array(patch_list)
    if transpose:
        patch_arr = einops.rearrange(
            patch_arr, "(p_num pw_num) ph pw c -> p_num (pw_num pw) ph c", p_num=p_num
        )
    else:
        patch_arr = einops.rearrange(
            patch_arr, "(p_num pw_num) ph pw c -> p_num ph (pw_num pw) c", p_num=p_num
        )

    batches = [[]]
    pad_size = 0
    for patch in patch_arr:
        if len(batches[-1]) >= max_batch_size:
            batches.append([])
        p, _ratio, pad_size, _ = _square_pad_resize(patch, tgt_size=tgt_size)
        batches[-1].append(p)

    db_lst, mask_lst = [], []
    for batch in batches:
        batch_np = np.array(batch)
        db, mask = dbnet_batch_forward(batch_np, device=device)
        for d, m in zip(db, mask):
            if pad_size > 0:
                paddb = int(db.shape[-1] / tgt_size * pad_size)
                padmsk = int(mask.shape[-1] / tgt_size * pad_size)
                d = d[..., :-paddb, :-paddb]
                m = m[..., :-padmsk, :-padmsk]
            db_lst.append(d)
            mask_lst.append(m)

    def _unrearrange(patch_lst, trans, channel=1, pad_n=0):
        _psize = patch_lst[0].shape[-1]
        _step = int(ph_step * _psize / patch_size)
        _pw = int(_psize / pw_num)
        _h = int(_pw / w * h)
        tgtmap = np.zeros((channel, _h, _pw), dtype=np.float32)
        num_patches = len(patch_lst) * pw_num - pad_n
        for ii_p, p in enumerate(patch_lst):
            if trans:
                p = einops.rearrange(p, "c h w -> c w h")
            for jj in range(pw_num):
                pidx = ii_p * pw_num + jj
                rel_t = rel_step_list[pidx]
                t_pos = int(round(rel_t * _h))
                b_pos = min(t_pos + _psize, _h)
                l_pos = jj * _pw
                r_pos = l_pos + _pw
                tgtmap[..., t_pos:b_pos, :] += p[..., : b_pos - t_pos, l_pos:r_pos]
                if pidx > 0:
                    interleave = _psize - _step
                    tgtmap[..., t_pos : t_pos + interleave, :] /= 2.0
                if pidx >= num_patches - 1:
                    break
        if trans:
            tgtmap = einops.rearrange(tgtmap, "c h w -> c w h")
        return tgtmap[None, ...]

    db_out = _unrearrange(db_lst, transpose, channel=2, pad_n=pad_num)
    mask_out = _unrearrange(mask_lst, transpose, channel=1, pad_n=pad_num)
    return db_out, mask_out


# ── CommonDetector ───────────────────────────────────────────────────


class CommonDetector:
    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)

    async def detect(
        self,
        image: np.ndarray,
        detect_size: int,
        text_threshold: float,
        box_threshold: float,
        unclip_ratio: float,
        invert: bool,
        gamma_correct: bool,
        rotate: bool,
        auto_rotate: bool = False,
        verbose: bool = False,
    ):
        img_h, img_w = image.shape[:2]
        orig_image = image.copy()
        minimum_image_size = 400
        add_border = min(img_w, img_h) < minimum_image_size

        if rotate:
            image = np.rot90(image, k=-1)
        if add_border:
            image = self._add_border(image, minimum_image_size)
        if invert:
            image = cv2.bitwise_not(image)
        if gamma_correct:
            image = self._add_gamma_correction(image)

        textlines, raw_mask, mask = await self._detect(
            image, detect_size, text_threshold, box_threshold, unclip_ratio, verbose
        )
        textlines = [t for t in textlines if t.area > 1]

        if add_border:
            textlines, raw_mask, mask = self._remove_border(image, img_w, img_h, textlines, raw_mask, mask)
        if auto_rotate:
            if textlines:
                orientations = ["h" if t.aspect_ratio > 1 else "v" for t in textlines]
                majority = Counter(orientations).most_common(1)[0][0]
            else:
                majority = "h"
            if majority == "h":
                return await self.detect(
                    orig_image, detect_size, text_threshold, box_threshold, unclip_ratio,
                    invert, gamma_correct, rotate=(not rotate), auto_rotate=False, verbose=verbose,
                )
        if rotate:
            textlines, raw_mask, mask = self._remove_rotation(textlines, raw_mask, mask, img_w, img_h)

        return textlines, raw_mask, mask

    @abstractmethod
    async def _detect(self, image, detect_size, text_threshold, box_threshold, unclip_ratio, verbose=False):
        pass

    def _add_border(self, image, target):
        old_h, old_w = image.shape[:2]
        new_sz = max(old_w, old_h, target)
        new_image = np.zeros([new_sz, new_sz, 3], dtype=np.uint8)
        new_image[:old_h, :old_w] = image
        return new_image

    def _remove_border(self, image, old_w, old_h, textlines, raw_mask, mask):
        new_h, new_w = image.shape[:2]
        raw_mask = cv2.resize(raw_mask, (new_w, new_h), interpolation=cv2.INTER_LINEAR)[:old_h, :old_w]
        if mask is not None:
            mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_LINEAR)[:old_h, :old_w]
        new_tl = []
        for t in textlines:
            if t.xyxy[0] >= old_w and t.xyxy[1] >= old_h:
                continue
            pts = t.pts.copy()
            pts[:, 0] = np.clip(pts[:, 0], 0, old_w)
            pts[:, 1] = np.clip(pts[:, 1], 0, old_h)
            new_tl.append(Quadrilateral(pts, t.text, t.prob))
        return new_tl, raw_mask, mask

    def _remove_rotation(self, textlines, raw_mask, mask, img_w, img_h):
        raw_mask = np.ascontiguousarray(np.rot90(raw_mask))
        if mask is not None:
            mask = np.ascontiguousarray(np.rot90(mask).astype(np.uint8))
        for i, t in enumerate(textlines):
            rp = t.pts[:, [1, 0]].copy()
            rp[:, 1] = -rp[:, 1] + img_h
            textlines[i] = Quadrilateral(rp, t.text, t.prob)
        return textlines, raw_mask, mask

    def _add_gamma_correction(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mean = np.mean(gray)
        gamma = np.log(0.5 * 255) / np.log(mean + 1e-6)
        return np.power(image, gamma).clip(0, 255).astype(np.uint8)


# ── DefaultDetector ──────────────────────────────────────────────────

MODEL = None


def _det_batch_forward(batch: np.ndarray, device: str):
    global MODEL
    if isinstance(batch, list):
        batch = np.array(batch)
    batch = einops.rearrange(batch.astype(np.float32) / 127.5 - 1.0, "n h w c -> n c h w")
    batch = torch.from_numpy(batch).to(device)
    with torch.no_grad():
        db, mask = MODEL(batch)
        db = db.sigmoid().cpu().numpy()
        mask = mask.cpu().numpy()
    return db, mask


class DefaultDetector(CommonDetector, ModelWrapper):
    _MODEL_SUB_DIR = "detection"
    _MODEL_MAPPING = {
        "model": {
            "url": "https://github.com/zyddnys/manga-image-translator/releases/download/beta-0.3/detect-20241225.ckpt",
            "hash": "67ce1c4ed4793860f038c71189ba9630a7756f7683b1ee5afb69ca0687dc502e",
            "file": ".",
        }
    }

    def __init__(self, *args, **kwargs):
        CommonDetector.__init__(self)
        os.makedirs(self.model_dir, exist_ok=True)
        if os.path.exists("detect-20241225.ckpt"):
            shutil.move("detect-20241225.ckpt", self._get_file_path("detect-20241225.ckpt"))
        ModelWrapper.__init__(self)

    async def _load(self, device: str, *args, **kwargs):
        self.model = DBNet_resnet34.TextDetection()
        sd = torch.load(self._get_file_path("detect-20241225.ckpt"), map_location="cpu")
        self.model.load_state_dict(sd.get("model", sd))
        self.model.eval()
        self.device = device
        if device in ("cuda", "mps"):
            self.model = self.model.to(self.device)
        global MODEL
        MODEL = self.model

    async def _unload(self):
        global MODEL
        MODEL = None
        del self.model

    async def _infer(
        self,
        image: np.ndarray,
        detect_size: int,
        text_threshold: float,
        box_threshold: float,
        unclip_ratio: float,
        verbose: bool = False,
    ):
        db, mask = det_rearrange_forward(image, _det_batch_forward, detect_size, 4, device=self.device, verbose=verbose)

        if db is None:
            img_resized, target_ratio, _, pad_w, pad_h = imgproc.resize_aspect_ratio(
                cv2.bilateralFilter(image, 17, 80, 80), detect_size, cv2.INTER_LINEAR, mag_ratio=1
            )
            img_resized_h, img_resized_w = img_resized.shape[:2]
            ratio_h = ratio_w = 1 / target_ratio
            db, mask = _det_batch_forward([img_resized], self.device)
        else:
            img_resized_h, img_resized_w = image.shape[:2]
            ratio_w = ratio_h = 1
            pad_h = pad_w = 0

        self.logger.info("Detection resolution: %dx%d", img_resized_w, img_resized_h)

        mask = mask[0, 0, :, :]
        det = dbnet_utils.SegDetectorRepresenter(text_threshold, box_threshold, unclip_ratio=unclip_ratio)
        boxes, scores = det({"shape": [(img_resized_h, img_resized_w)]}, db)
        boxes, scores = boxes[0], scores[0]

        if boxes.size == 0:
            polys = []
        else:
            idx = boxes.reshape(boxes.shape[0], -1).sum(axis=1) > 0
            polys, _ = boxes[idx], scores[idx]
            polys = polys.astype(np.float64)
            polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h, ratio_net=1)
            polys = polys.astype(np.int64)

        textlines = [Quadrilateral(pts.astype(int), "", score) for pts, score in zip(polys, scores)]
        textlines = [q for q in textlines if q.area > 16]

        mask_resized = cv2.resize(mask, (mask.shape[1] * 2, mask.shape[0] * 2), interpolation=cv2.INTER_LINEAR)
        if pad_h > 0:
            mask_resized = mask_resized[:-pad_h, :]
        elif pad_w > 0:
            mask_resized = mask_resized[:, :-pad_w]
        raw_mask = np.clip(mask_resized * 255, 0, 255).astype(np.uint8)

        return textlines, raw_mask, None

    async def _detect(self, image, detect_size, text_threshold, box_threshold, unclip_ratio, verbose=False):
        return await self._infer(image, detect_size, text_threshold, box_threshold, unclip_ratio, verbose)

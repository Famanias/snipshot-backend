"""OCR model — ConvNext + Roformer + XPos ViT (48px), beam search decoding."""

import math
import os
import shutil
from typing import List, Optional
from collections import defaultdict

import cv2
import numpy as np
import einops
import torch
import torch.nn as nn
import torch.nn.functional as F

from .xpos_relative_position import XPOS
from ..utils import TextBlock, Quadrilateral, chunks
from ..utils.generic import AvgMeter
from ..utils.inference import ModelWrapper
from ..utils.log import get_logger

logger = get_logger("ocr")


# ── ConvNext feature extractor ───────────────────────────────────────

class ConvNeXtBlock(nn.Module):
    def __init__(self, dim, layer_scale_init_value=1e-6, ks=7, padding=3):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=ks, padding=padding, groups=dim)
        self.norm = nn.BatchNorm2d(dim, eps=1e-6)
        self.pwconv1 = nn.Conv2d(dim, 4 * dim, 1, 1, 0)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(4 * dim, dim, 1, 1, 0)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones(1, dim, 1, 1), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )

    def forward(self, x):
        inp = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        return inp + x


class ConvNext_FeatureExtractor(nn.Module):
    def __init__(self, img_height=48, in_dim=3, dim=512, n_layers=12):
        super().__init__()
        base = dim // 8
        self.stem = nn.Sequential(
            nn.Conv2d(in_dim, base, 7, 1, 3), nn.BatchNorm2d(base), nn.ReLU(),
            nn.Conv2d(base, base * 2, 2, 2, 0), nn.BatchNorm2d(base * 2), nn.ReLU(),
            nn.Conv2d(base * 2, base * 2, 3, 1, 1), nn.BatchNorm2d(base * 2), nn.ReLU(),
        )
        self.block1 = self._make_layers(base * 2, 4)
        self.down1 = nn.Sequential(nn.Conv2d(base * 2, base * 4, 2, 2, 0), nn.BatchNorm2d(base * 4), nn.ReLU())
        self.block2 = self._make_layers(base * 4, 12)
        self.down2 = nn.Sequential(nn.Conv2d(base * 4, base * 8, (2, 1), (2, 1), (0, 0)), nn.BatchNorm2d(base * 8), nn.ReLU())
        self.block3 = self._make_layers(base * 8, 10, ks=5, padding=2)
        self.down3 = nn.Sequential(nn.Conv2d(base * 8, base * 8, (2, 1), (2, 1), (0, 0)), nn.BatchNorm2d(base * 8), nn.ReLU())
        self.block4 = self._make_layers(base * 8, 8, ks=3, padding=1)
        self.down4 = nn.Sequential(nn.Conv2d(base * 8, base * 8, (3, 1), (1, 1), (0, 0)), nn.BatchNorm2d(base * 8), nn.ReLU())

    @staticmethod
    def _make_layers(dim, n, ks=7, padding=3):
        return nn.Sequential(*[ConvNeXtBlock(dim, ks=ks, padding=padding) for _ in range(n)])

    def forward(self, x):
        x = self.stem(x)
        x = self.down1(self.block1(x))
        x = self.down2(self.block2(x))
        x = self.down3(self.block3(x))
        x = self.down4(self.block4(x))
        return x


# ── Transformer components ───────────────────────────────────────────

def transformer_encoder_forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal=False):
    x = src
    if self.norm_first:
        x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
        x = x + self._ff_block(self.norm2(x))
    else:
        x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask))
        x = self.norm2(x + self._ff_block(x))
    return x


class XposMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, self_attention=False, encoder_decoder_attention=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim ** -0.5
        self.self_attention = self_attention
        self.encoder_decoder_attention = encoder_decoder_attention
        assert self.self_attention ^ self.encoder_decoder_attention

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.xpos = XPOS(self.head_dim, embed_dim)
        self.batch_first = True
        self._qkv_same_embed_dim = True

    def forward(self, query, key, value, key_padding_mask=None, attn_mask=None,
                need_weights=False, is_causal=False, k_offset=0, q_offset=0):
        bsz, tgt_len, embed_dim = query.size()
        src_len = key.size(1)

        q = self.q_proj(query) * self.scaling
        k = self.k_proj(key)
        v = self.v_proj(value)

        q = q.view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2).reshape(bsz * self.num_heads, tgt_len, self.head_dim)
        k = k.view(bsz, src_len, self.num_heads, self.head_dim).transpose(1, 2).reshape(bsz * self.num_heads, src_len, self.head_dim)
        v = v.view(bsz, src_len, self.num_heads, self.head_dim).transpose(1, 2).reshape(bsz * self.num_heads, src_len, self.head_dim)

        if self.xpos is not None:
            k = self.xpos(k, offset=k_offset, downscale=True)
            q = self.xpos(q, offset=q_offset, downscale=False)

        attn_weights = torch.bmm(q, k.transpose(1, 2))

        if attn_mask is not None:
            attn_weights = torch.nan_to_num(attn_weights)
            attn_weights += attn_mask.unsqueeze(0)

        if key_padding_mask is not None:
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool), float("-inf"))
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).type_as(attn_weights)
        attn = torch.bmm(attn_weights, v)
        attn = attn.transpose(0, 1).reshape(tgt_len, bsz, embed_dim).transpose(0, 1)
        attn = self.out_proj(attn)

        if need_weights:
            return attn, attn_weights.view(bsz, self.num_heads, tgt_len, src_len).transpose(1, 0)
        return attn, None


def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    return mask.float().masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, 0.0)


# ── Hypothesis for beam search ───────────────────────────────────────

class Hypothesis:
    def __init__(self, device, start_tok, end_tok, padding_tok, memory_idx, num_layers, embd_dim):
        self.device = device
        self.start_tok = start_tok
        self.end_tok = end_tok
        self.padding_tok = padding_tok
        self.memory_idx = memory_idx
        self.embd_size = embd_dim
        self.num_layers = num_layers
        self.cached_activations = [torch.zeros(1, 0, self.embd_size).to(self.device)] * (num_layers + 1)
        self.out_idx = torch.LongTensor([start_tok]).to(self.device)
        self.out_logprobs = torch.FloatTensor([0]).to(self.device)
        self.length = 0

    def seq_end(self):
        return self.out_idx.view(-1)[-1] == self.end_tok

    def logprob(self):
        return self.out_logprobs.mean().item()

    def sort_key(self):
        return -self.logprob()

    def prob(self):
        return self.out_logprobs.mean().exp().item()

    def __len__(self):
        return self.length

    def extend(self, idx, logprob):
        ret = Hypothesis(self.device, self.start_tok, self.end_tok, self.padding_tok,
                         self.memory_idx, self.num_layers, self.embd_size)
        ret.cached_activations = [item.clone() for item in self.cached_activations]
        ret.length = self.length + 1
        ret.out_idx = torch.cat([self.out_idx, torch.LongTensor([idx]).to(self.device)], dim=0)
        ret.out_logprobs = torch.cat([self.out_logprobs, torch.FloatTensor([logprob]).to(self.device)], dim=0)
        return ret

    def output(self):
        return self.cached_activations[-1]


def next_token_batch(hyps, memory, memory_mask, decoders, embd):
    N = len(hyps)
    offset = len(hyps[0])
    last_toks = torch.stack([item.out_idx[-1] for item in hyps])
    tgt = embd(last_toks).unsqueeze_(1)
    memory = torch.stack([memory[item.memory_idx] for item in hyps], dim=0)

    for l, layer in enumerate(decoders):
        combined = torch.cat([item.cached_activations[l] for item in hyps], dim=0)
        combined = torch.cat([combined, tgt], dim=1)
        for i in range(N):
            hyps[i].cached_activations[l] = combined[i:i + 1]
        tgt = tgt + layer.self_attn(layer.norm1(tgt), layer.norm1(combined), layer.norm1(combined), q_offset=offset)[0]
        tgt = tgt + layer.multihead_attn(layer.norm2(tgt), memory, memory, key_padding_mask=memory_mask, q_offset=offset)[0]
        tgt = tgt + layer._ff_block(layer.norm3(tgt))

    for i in range(N):
        hyps[i].cached_activations[len(decoders)] = torch.cat(
            [hyps[i].cached_activations[len(decoders)], tgt[i:i + 1]], dim=1
        )
    return tgt.squeeze_(1)


# ── OCR network ──────────────────────────────────────────────────────

class OCR(nn.Module):
    def __init__(self, dictionary, max_len):
        super().__init__()
        self.max_len = max_len
        self.dictionary = dictionary
        self.dict_size = len(dictionary)
        embd_dim = 320
        nhead = 4

        self.backbone = ConvNext_FeatureExtractor(48, 3, embd_dim)

        self.encoders = nn.ModuleList()
        for _ in range(4):
            encoder = nn.TransformerEncoderLayer(embd_dim, nhead, dropout=0, batch_first=True, norm_first=True)
            encoder.self_attn = XposMultiheadAttention(embd_dim, nhead, self_attention=True)
            encoder.forward = transformer_encoder_forward
            self.encoders.append(encoder)
        self.encoders.forward = self.encoder_forward

        self.decoders = nn.ModuleList()
        for _ in range(5):
            decoder = nn.TransformerDecoderLayer(embd_dim, nhead, dropout=0, batch_first=True, norm_first=True)
            decoder.self_attn = XposMultiheadAttention(embd_dim, nhead, self_attention=True)
            decoder.multihead_attn = XposMultiheadAttention(embd_dim, nhead, encoder_decoder_attention=True)
            self.decoders.append(decoder)
        self.decoders.forward = self.decoder_forward

        self.embd = nn.Embedding(self.dict_size, embd_dim)
        self.pred1 = nn.Sequential(nn.Linear(embd_dim, embd_dim), nn.GELU(), nn.Dropout(0.15))
        self.pred = nn.Linear(embd_dim, self.dict_size)
        self.pred.weight = self.embd.weight
        self.color_pred1 = nn.Sequential(nn.Linear(embd_dim, 64), nn.ReLU())
        self.color_pred_fg = nn.Linear(64, 3)
        self.color_pred_bg = nn.Linear(64, 3)
        self.color_pred_fg_ind = nn.Linear(64, 2)
        self.color_pred_bg_ind = nn.Linear(64, 2)

    def encoder_forward(self, memory, encoder_mask):
        for layer in self.encoders:
            memory = layer(layer, src=memory, src_key_padding_mask=encoder_mask)
        return memory

    def decoder_forward(self, embd, cached_activations, memory, memory_mask, step):
        tgt = embd
        for l, layer in enumerate(self.decoders):
            combined = cached_activations[:, l, :step, :]
            combined = torch.cat([combined, tgt], dim=1)
            cached_activations[:, l, step, :] = tgt.squeeze(1)
            tgt = tgt + layer.self_attn(layer.norm1(tgt), layer.norm1(combined), layer.norm1(combined), q_offset=step)[0]
            tgt = tgt + layer.multihead_attn(layer.norm2(tgt), memory, memory, key_padding_mask=memory_mask, q_offset=step)[0]
            tgt = tgt + layer._ff_block(layer.norm3(tgt))
        cached_activations[:, l + 1, step, :] = tgt.squeeze(1)
        return tgt.squeeze_(1), cached_activations

    def forward(self, img, char_idx, decoder_mask, encoder_mask):
        memory = self.backbone(img)
        memory = einops.rearrange(memory, "N C 1 W -> N W C")
        for layer in self.encoders:
            memory = layer(memory, src_key_padding_mask=encoder_mask)
        char_embd = self.embd(char_idx)
        L = char_idx.shape[1]
        casual_mask = generate_square_subsequent_mask(L).to(img.device)
        decoded = char_embd
        for layer in self.decoders:
            decoded = layer(decoded, memory, tgt_mask=casual_mask,
                            tgt_key_padding_mask=decoder_mask, memory_key_padding_mask=encoder_mask)
        pred_char_logits = self.pred(self.pred1(decoded))
        color_feats = self.color_pred1(decoded)
        return (
            pred_char_logits,
            self.color_pred_fg(color_feats),
            self.color_pred_bg(color_feats),
            self.color_pred_fg_ind(color_feats),
            self.color_pred_bg_ind(color_feats),
        )

    def infer_beam_batch_tensor(self, img, img_widths, beams_k=5,
                                start_tok=1, end_tok=2, pad_tok=0,
                                max_finished_hypos=2, max_seq_length=384):
        N, C, H, W = img.shape
        assert H == 48 and C == 3

        memory = self.backbone(img)
        memory = einops.rearrange(memory, "N C 1 W -> N W C")
        valid_feats_length = [(x + 3) // 4 + 2 for x in img_widths]
        input_mask = torch.zeros(N, memory.size(1), dtype=torch.bool).to(img.device)
        for i, l in enumerate(valid_feats_length):
            input_mask[i, l:] = True
        memory = self.encoders(memory, input_mask)

        out_idx = torch.full((N, 1), start_tok, dtype=torch.long, device=img.device)
        cached_activations = torch.zeros(N, len(self.decoders) + 1, max_seq_length, 320, device=img.device)
        log_probs = torch.zeros(N, 1, device=img.device)

        idx_embedded = self.embd(out_idx[:, -1:])
        decoded, cached_activations = self.decoders(idx_embedded, cached_activations, memory, input_mask, 0)
        pred_char_logprob = self.pred(self.pred1(decoded)).log_softmax(-1)
        pred_chars_values, pred_chars_index = torch.topk(pred_char_logprob, beams_k, dim=1)

        out_idx = torch.cat([out_idx.unsqueeze(1).expand(-1, beams_k, -1), pred_chars_index.unsqueeze(-1)], dim=-1).reshape(-1, 2)
        log_probs = pred_chars_values.view(-1, 1)
        memory = memory.repeat_interleave(beams_k, dim=0)
        input_mask = input_mask.repeat_interleave(beams_k, dim=0)
        cached_activations = cached_activations.repeat_interleave(beams_k, dim=0)
        batch_index = torch.arange(N).repeat_interleave(beams_k, dim=0).to(img.device)

        finished_hypos = defaultdict(list)
        N_remaining = N

        for step in range(1, max_seq_length):
            idx_embedded = self.embd(out_idx[:, -1:])
            decoded, cached_activations = self.decoders(idx_embedded, cached_activations, memory, input_mask, step)
            pred_char_logprob = self.pred(self.pred1(decoded)).log_softmax(-1)
            pred_chars_values, pred_chars_index = torch.topk(pred_char_logprob, beams_k, dim=1)

            finished = out_idx[:, -1] == end_tok
            pred_chars_values[finished] = 0
            pred_chars_index[finished] = end_tok

            new_out_idx = out_idx.unsqueeze(1).expand(-1, beams_k, -1)
            new_out_idx = torch.cat([new_out_idx, pred_chars_index.unsqueeze(-1)], dim=-1)
            new_out_idx = new_out_idx.view(-1, step + 2)
            new_log_probs = log_probs.unsqueeze(1).expand(-1, beams_k, -1) + pred_chars_values.unsqueeze(-1)
            new_log_probs = new_log_probs.view(-1, 1)

            new_out_idx = new_out_idx.view(N_remaining, -1, step + 2)
            new_log_probs = new_log_probs.view(N_remaining, -1)
            batch_topk_log_probs, batch_topk_indices = new_log_probs.topk(beams_k, dim=1)

            expanded_topk_indices = batch_topk_indices.unsqueeze(-1).expand(-1, -1, new_out_idx.shape[-1])
            out_idx = torch.gather(new_out_idx, 1, expanded_topk_indices).reshape(-1, step + 2)
            log_probs = batch_topk_log_probs.view(-1, 1)

            finished = (out_idx[:, -1] == end_tok).view(N_remaining, beams_k)
            finished_counts = finished.sum(dim=1)
            finished_batch_indices = (finished_counts >= max_finished_hypos).nonzero(as_tuple=False).squeeze()

            if finished_batch_indices.numel() == 0:
                continue
            if finished_batch_indices.dim() == 0:
                finished_batch_indices = finished_batch_indices.unsqueeze(0)

            for idx in finished_batch_indices:
                batch_lp = batch_topk_log_probs[idx]
                best = batch_lp.argmax()
                finished_hypos[batch_index[beams_k * idx].item()] = (
                    out_idx[idx * beams_k + best],
                    torch.exp(batch_lp[best]).item(),
                    cached_activations[idx * beams_k + best],
                )

            remaining = []
            for i in range(N_remaining):
                if i not in finished_batch_indices:
                    for j in range(beams_k):
                        remaining.append(i * beams_k + j)

            if not remaining:
                break

            N_remaining = len(remaining) // beams_k
            sel = torch.tensor(remaining, device=img.device)
            out_idx = out_idx.index_select(0, sel)
            log_probs = log_probs.index_select(0, sel)
            memory = memory.index_select(0, sel)
            cached_activations = cached_activations.index_select(0, sel)
            input_mask = input_mask.index_select(0, sel)
            batch_index = batch_index.index_select(0, sel)

        # Fallback for unfinished samples
        if len(finished_hypos) < N:
            for i in range(N):
                if i not in finished_hypos:
                    sample_indices = (batch_index == i).nonzero(as_tuple=True)[0]
                    if sample_indices.numel() > 0:
                        best = sample_indices[0]
                        finished_hypos[i] = (out_idx[best], torch.exp(log_probs[best]).item(), cached_activations[best])
                    else:
                        finished_hypos[i] = (torch.tensor([end_tok], device=img.device), 0.0,
                                             torch.zeros(cached_activations.shape[1:], device=img.device))

        result = []
        for i in range(N):
            final_idx, prob, decoded = finished_hypos[i]
            color_feats = self.color_pred1(decoded[-1].unsqueeze(0))
            result.append((
                final_idx[1:], prob,
                self.color_pred_fg(color_feats),
                self.color_pred_bg(color_feats),
                self.color_pred_fg_ind(color_feats),
                self.color_pred_bg_ind(color_feats),
            ))
        return result


# ── Inferer wrapper ──────────────────────────────────────────────────

class Model48pxOCR(ModelWrapper):
    _MODEL_SUB_DIR = "ocr"
    _MODEL_MAPPING = {
        "model": {
            "url": "https://github.com/zyddnys/manga-image-translator/releases/download/beta-0.3/ocr_ar_48px.ckpt",
            "hash": "29daa46d080818bb4ab239a518a88338cbccff8f901bef8c9db191a7cb97671d",
        },
        "dict": {
            "url": "https://github.com/zyddnys/manga-image-translator/releases/download/beta-0.3/alphabet-all-v7.txt",
            "hash": "f5722368146aa0fbcc9f4726866e4efc3203318ebb66c811d8cbbe915576538a",
        },
    }

    def __init__(self, *args, **kwargs):
        os.makedirs(self._get_file_path(), exist_ok=True)
        for name in ("ocr_ar_48px.ckpt", "alphabet-all-v7.txt"):
            if os.path.exists(name):
                shutil.move(name, self._get_file_path(name))
        super().__init__(*args, **kwargs)

    async def _load(self, device: str):
        with open(self._get_file_path("alphabet-all-v7.txt"), "r", encoding="utf-8") as fp:
            dictionary = [s.rstrip("\n") for s in fp.readlines()]
        self.model = OCR(dictionary, 768)
        sd = torch.load(self._get_file_path("ocr_ar_48px.ckpt"), map_location=device)
        self.model.load_state_dict(sd)
        self.model.eval()
        self.device = device
        self.use_gpu = device in ("cuda", "mps")
        if self.use_gpu:
            self.model = self.model.to(device)

    async def _unload(self):
        del self.model

    async def _infer(self, image: np.ndarray, textlines: List[Quadrilateral],
                     config=None, verbose=False) -> List:
        from ..config import OcrConfig
        import itertools
        from collections import Counter
        import networkx as nx
        from ..utils import quadrilateral_can_merge_region

        config = config or OcrConfig()
        text_height = 48
        max_chunk_size = 16
        threshold = 0.2 if config.prob is None else config.prob

        # Generate text direction pairs
        quadrilaterals = list(self._generate_text_direction(textlines))
        region_imgs = [q.get_transformed_region(image, d, text_height) for q, d in quadrilaterals]
        out_regions = []

        perm = range(len(region_imgs))
        is_quadrilaterals = False
        if quadrilaterals and isinstance(quadrilaterals[0][0], Quadrilateral):
            perm = sorted(range(len(region_imgs)), key=lambda x: region_imgs[x].shape[1])
            is_quadrilaterals = True

        for indices in chunks(perm, max_chunk_size):
            N = len(indices)
            widths = [region_imgs[i].shape[1] for i in indices]
            max_width = 4 * (max(widths) + 7) // 4
            region = np.zeros((N, text_height, max_width, 3), dtype=np.uint8)
            for i, idx in enumerate(indices):
                W = region_imgs[idx].shape[1]
                region[i, :, :W, :] = region_imgs[idx]

            image_tensor = (torch.from_numpy(region).float() - 127.5) / 127.5
            image_tensor = einops.rearrange(image_tensor, "N H W C -> N C H W")
            if self.use_gpu:
                image_tensor = image_tensor.to(self.device)

            with torch.no_grad():
                ret = self.model.infer_beam_batch_tensor(image_tensor, widths, beams_k=5, max_seq_length=255)

            for i, (pred_chars_index, prob, fg_pred, bg_pred, fg_ind_pred, bg_ind_pred) in enumerate(ret):
                if prob < threshold:
                    continue
                has_fg = fg_ind_pred[:, 1] > fg_ind_pred[:, 0]
                has_bg = bg_ind_pred[:, 1] > bg_ind_pred[:, 0]
                seq = []
                fr, fg, fb = AvgMeter(), AvgMeter(), AvgMeter()
                br, bg_m, bb = AvgMeter(), AvgMeter(), AvgMeter()
                for chid, c_fg, c_bg, h_fg, h_bg in zip(pred_chars_index, fg_pred, bg_pred, has_fg, has_bg):
                    ch = self.model.dictionary[chid]
                    if ch == "<S>":
                        continue
                    if ch == "</S>":
                        break
                    if ch == "<SP>":
                        ch = " "
                    seq.append(ch)
                    if h_fg.item():
                        fr(int(c_fg[0] * 255)); fg(int(c_fg[1] * 255)); fb(int(c_fg[2] * 255))
                    if h_bg.item():
                        br(int(c_bg[0] * 255)); bg_m(int(c_bg[1] * 255)); bb(int(c_bg[2] * 255))
                    else:
                        br(int(c_fg[0] * 255)); bg_m(int(c_fg[1] * 255)); bb(int(c_fg[2] * 255))

                txt = "".join(seq)
                frc = min(max(int(fr()), 0), 255)
                fgc = min(max(int(fg()), 0), 255)
                fbc = min(max(int(fb()), 0), 255)
                brc = min(max(int(br()), 0), 255)
                bgc = min(max(int(bg_m()), 0), 255)
                bbc = min(max(int(bb()), 0), 255)

                logger.info("prob: %s %s fg: (%s,%s,%s) bg: (%s,%s,%s)", prob, txt, frc, fgc, fbc, brc, bgc, bbc)

                cur_region = quadrilaterals[indices[i]][0]
                if isinstance(cur_region, Quadrilateral):
                    cur_region.text = txt
                    cur_region.prob = prob
                    cur_region.fg_r = frc
                    cur_region.fg_g = fgc
                    cur_region.fg_b = fbc
                    cur_region.bg_r = brc
                    cur_region.bg_g = bgc
                    cur_region.bg_b = bbc
                else:
                    cur_region.text.append(txt)
                    cur_region.update_font_colors(np.array([frc, fgc, fbc]), np.array([brc, bgc, bbc]))
                out_regions.append(cur_region)

        return out_regions if is_quadrilaterals else textlines

    def _generate_text_direction(self, bboxes):
        """Generate (region, direction) pairs with majority-vote grouping."""
        import itertools
        from collections import Counter
        import networkx as nx
        from ..utils import quadrilateral_can_merge_region

        if not bboxes:
            return
        if isinstance(bboxes[0], TextBlock):
            for blk in bboxes:
                for line_idx in range(len(blk.lines)):
                    yield blk, line_idx
        else:
            G = nx.Graph()
            for i, box in enumerate(bboxes):
                G.add_node(i, box=box)
            for (u, ubox), (v, vbox) in itertools.combinations(enumerate(bboxes), 2):
                if quadrilateral_can_merge_region(ubox, vbox, aspect_ratio_tol=1):
                    G.add_edge(u, v)
            for node_set in nx.algorithms.components.connected_components(G):
                nodes = list(node_set)
                dirs = [bboxes[i].direction for i in nodes]
                majority_dir = Counter(dirs).most_common(1)[0][0]
                if majority_dir == "h":
                    nodes = sorted(nodes, key=lambda x: bboxes[x].aabb.y + bboxes[x].aabb.h // 2)
                elif majority_dir == "v":
                    nodes = sorted(nodes, key=lambda x: -(bboxes[x].aabb.x + bboxes[x].aabb.w))
                for node in nodes:
                    yield bboxes[node], majority_dir

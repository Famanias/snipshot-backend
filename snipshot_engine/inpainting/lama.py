# LaMa Large Inpainter — FFC-based architecture
# Original: https://github.com/DQiaole/ZITS_inpainting.git
# Paper: https://arxiv.org/pdf/2203.00867.pdf

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..config import InpainterConfig
from ..utils import ModelWrapper, resize_keep_aspect, get_logger

logger = get_logger("inpainting")

TORCH_DTYPE_MAP = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


# ---------------------------------------------------------------------------
# FFC building blocks
# ---------------------------------------------------------------------------

def get_activation(kind="tanh"):
    if kind == "tanh":
        return nn.Tanh()
    if kind == "sigmoid":
        return nn.Sigmoid()
    if kind is False:
        return nn.Identity()
    raise ValueError(f"Unknown activation kind {kind}")


class FFCSE_block(nn.Module):
    def __init__(self, channels, ratio_g):
        super().__init__()
        in_cg = int(channels * ratio_g)
        in_cl = channels - in_cg
        r = 16
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv1 = nn.Conv2d(channels, channels // r, kernel_size=1, bias=True)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv_a2l = None if in_cl == 0 else nn.Conv2d(channels // r, in_cl, kernel_size=1, bias=True)
        self.conv_a2g = None if in_cg == 0 else nn.Conv2d(channels // r, in_cg, kernel_size=1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x if type(x) is tuple else (x, 0)
        id_l, id_g = x
        x = id_l if type(id_g) is int else torch.cat([id_l, id_g], dim=1)
        x = self.avgpool(x)
        x = self.relu1(self.conv1(x))
        x_l = 0 if self.conv_a2l is None else id_l * self.sigmoid(self.conv_a2l(x))
        x_g = 0 if self.conv_a2g is None else id_g * self.sigmoid(self.conv_a2g(x))
        return x_l, x_g


class FourierUnit(nn.Module):
    def __init__(self, in_channels, out_channels, groups=1, spatial_scale_factor=None,
                 spatial_scale_mode="bilinear", spectral_pos_encoding=False,
                 use_se=False, se_kwargs=None, ffc3d=False, fft_norm="ortho"):
        super().__init__()
        self.groups = groups
        self.conv_layer = nn.Conv2d(
            in_channels=in_channels * 2 + (2 if spectral_pos_encoding else 0),
            out_channels=out_channels * 2,
            kernel_size=1, stride=1, padding=0, groups=self.groups, bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels * 2)
        self.relu = nn.ReLU(inplace=True)
        self.use_se = use_se
        self.spatial_scale_factor = spatial_scale_factor
        self.spatial_scale_mode = spatial_scale_mode
        self.spectral_pos_encoding = spectral_pos_encoding
        self.ffc3d = ffc3d
        self.fft_norm = fft_norm

    def forward(self, x):
        batch = x.shape[0]
        if self.spatial_scale_factor is not None:
            orig_size = x.shape[-2:]
            x = F.interpolate(x, scale_factor=self.spatial_scale_factor,
                              mode=self.spatial_scale_mode, align_corners=False)
        r_size = x.size()
        fft_dim = (-3, -2, -1) if self.ffc3d else (-2, -1)

        if x.dtype in (torch.float16, torch.bfloat16):
            x = x.type(torch.float32)

        ffted = torch.fft.rfftn(x, dim=fft_dim, norm=self.fft_norm)
        ffted = torch.stack((ffted.real, ffted.imag), dim=-1)
        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()
        ffted = ffted.view((batch, -1,) + ffted.size()[3:])

        if self.spectral_pos_encoding:
            height, width = ffted.shape[-2:]
            coords_vert = torch.linspace(0, 1, height)[None, None, :, None].expand(batch, 1, height, width).to(ffted)
            coords_hor = torch.linspace(0, 1, width)[None, None, None, :].expand(batch, 1, height, width).to(ffted)
            ffted = torch.cat((coords_vert, coords_hor, ffted), dim=1)

        if self.use_se:
            ffted = self.se(ffted)

        ffted = self.conv_layer(ffted)
        ffted = self.relu(self.bn(ffted))

        ffted = ffted.view((batch, -1, 2,) + ffted.size()[2:]).permute(0, 1, 3, 4, 2).contiguous()
        if ffted.dtype in (torch.float16, torch.bfloat16):
            ffted = ffted.type(torch.float32)
        ffted = torch.complex(ffted[..., 0], ffted[..., 1])

        ifft_shape_slice = x.shape[-3:] if self.ffc3d else x.shape[-2:]
        output = torch.fft.irfftn(ffted, s=ifft_shape_slice, dim=fft_dim, norm=self.fft_norm)

        if self.spatial_scale_factor is not None:
            output = F.interpolate(output, size=orig_size, mode=self.spatial_scale_mode, align_corners=False)
        return output


class SpectralTransform(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, groups=1, enable_lfu=True, **fu_kwargs):
        super().__init__()
        self.enable_lfu = enable_lfu
        self.downsample = nn.AvgPool2d(kernel_size=(2, 2), stride=2) if stride == 2 else nn.Identity()
        self.stride = stride
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 2, kernel_size=1, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU(inplace=True),
        )
        self.fu = FourierUnit(out_channels // 2, out_channels // 2, groups, **fu_kwargs)
        if self.enable_lfu:
            self.lfu = FourierUnit(out_channels // 2, out_channels // 2, groups)
        self.conv2 = nn.Conv2d(out_channels // 2, out_channels, kernel_size=1, groups=groups, bias=False)

    def forward(self, x):
        x = self.downsample(x)
        x = self.conv1(x)
        output = self.fu(x)
        if self.enable_lfu:
            n, c, h, w = x.shape
            split_no = 2
            split_s = h // split_no
            xs = torch.cat(torch.split(x[:, :c // 4], split_s, dim=-2), dim=1).contiguous()
            xs = torch.cat(torch.split(xs, split_s, dim=-1), dim=1).contiguous()
            xs = self.lfu(xs)
            xs = xs.repeat(1, 1, split_no, split_no).contiguous()
        else:
            xs = 0
        output = self.conv2(x + output + xs)
        return output


class FFC(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 ratio_gin, ratio_gout, stride=1, padding=0,
                 dilation=1, groups=1, bias=False, enable_lfu=True,
                 padding_type="reflect", gated=False, **spectral_kwargs):
        super().__init__()
        assert stride in (1, 2)
        self.stride = stride
        in_cg = int(in_channels * ratio_gin)
        in_cl = in_channels - in_cg
        out_cg = int(out_channels * ratio_gout)
        out_cl = out_channels - out_cg
        self.ratio_gin = ratio_gin
        self.ratio_gout = ratio_gout
        self.global_in_num = in_cg

        module = nn.Identity if in_cl == 0 or out_cl == 0 else nn.Conv2d
        self.convl2l = module(in_cl, out_cl, kernel_size, stride, padding, dilation, groups, bias, padding_mode=padding_type)
        module = nn.Identity if in_cl == 0 or out_cg == 0 else nn.Conv2d
        self.convl2g = module(in_cl, out_cg, kernel_size, stride, padding, dilation, groups, bias, padding_mode=padding_type)
        module = nn.Identity if in_cg == 0 or out_cl == 0 else nn.Conv2d
        self.convg2l = module(in_cg, out_cl, kernel_size, stride, padding, dilation, groups, bias, padding_mode=padding_type)
        module = nn.Identity if in_cg == 0 or out_cg == 0 else SpectralTransform
        self.convg2g = module(in_cg, out_cg, stride, 1 if groups == 1 else groups // 2, enable_lfu, **spectral_kwargs)

        self.gated = gated
        module = nn.Identity if in_cg == 0 or out_cl == 0 or not self.gated else nn.Conv2d
        self.gate = module(in_channels, 2, 1)

    def forward(self, x):
        x_l, x_g = x if type(x) is tuple else (x, 0)
        out_xl, out_xg = 0, 0
        if self.gated:
            total_input_parts = [x_l]
            if torch.is_tensor(x_g):
                total_input_parts.append(x_g)
            total_input = torch.cat(total_input_parts, dim=1)
            gates = torch.sigmoid(self.gate(total_input))
            g2l_gate, l2g_gate = gates.chunk(2, dim=1)
        else:
            g2l_gate, l2g_gate = 1, 1
        if self.ratio_gout != 1:
            out_xl = self.convl2l(x_l) + self.convg2l(x_g) * g2l_gate
        if self.ratio_gout != 0:
            out_xg = self.convl2g(x_l) * l2g_gate + self.convg2g(x_g)
        return out_xl, out_xg


class FFC_BN_ACT(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, ratio_gin, ratio_gout,
                 stride=1, padding=0, dilation=1, groups=1, bias=False,
                 norm_layer=nn.BatchNorm2d, activation_layer=nn.Identity,
                 padding_type="reflect", enable_lfu=True, **kwargs):
        super().__init__()
        self.ffc = FFC(in_channels, out_channels, kernel_size,
                       ratio_gin, ratio_gout, stride, padding, dilation,
                       groups, bias, enable_lfu, padding_type=padding_type, **kwargs)
        lnorm = nn.Identity if ratio_gout == 1 else norm_layer
        gnorm = nn.Identity if ratio_gout == 0 else norm_layer
        global_channels = int(out_channels * ratio_gout)
        self.bn_l = lnorm(out_channels - global_channels)
        self.bn_g = gnorm(global_channels)
        lact = nn.Identity if ratio_gout == 1 else activation_layer
        gact = nn.Identity if ratio_gout == 0 else activation_layer
        self.act_l = lact(inplace=True)
        self.act_g = gact(inplace=True)

    def forward(self, x):
        x_l, x_g = self.ffc(x)
        x_l = self.act_l(self.bn_l(x_l))
        x_g = self.act_g(self.bn_g(x_g))
        return x_l, x_g


class FFCResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, activation_layer=nn.ReLU,
                 dilation=1, inline=False, **conv_kwargs):
        super().__init__()
        self.conv1 = FFC_BN_ACT(dim, dim, kernel_size=3, padding=dilation, dilation=dilation,
                                norm_layer=norm_layer, activation_layer=activation_layer,
                                padding_type=padding_type, **conv_kwargs)
        self.conv2 = FFC_BN_ACT(dim, dim, kernel_size=3, padding=dilation, dilation=dilation,
                                norm_layer=norm_layer, activation_layer=activation_layer,
                                padding_type=padding_type, **conv_kwargs)
        self.inline = inline

    def forward(self, x):
        if self.inline:
            x_l, x_g = x[:, :-self.conv1.ffc.global_in_num], x[:, -self.conv1.ffc.global_in_num:]
        else:
            x_l, x_g = x if type(x) is tuple else (x, 0)
        id_l, id_g = x_l, x_g
        x_l, x_g = self.conv1((x_l, x_g))
        x_l, x_g = self.conv2((x_l, x_g))
        x_l, x_g = id_l + x_l, id_g + x_g
        out = x_l, x_g
        if self.inline:
            out = torch.cat(out, dim=1)
        return out


class ConcatTupleLayer(nn.Module):
    def forward(self, x):
        assert isinstance(x, tuple)
        x_l, x_g = x
        assert torch.is_tensor(x_l) or torch.is_tensor(x_g)
        if not torch.is_tensor(x_g):
            return x_l
        return torch.cat(x, dim=1)


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class FFCResNetGenerator(nn.Module):
    def __init__(self, input_nc=4, output_nc=3, ngf=64, n_downsampling=3,
                 n_blocks=9, norm_layer=nn.BatchNorm2d, padding_type="reflect",
                 activation_layer=nn.ReLU, up_norm_layer=nn.BatchNorm2d,
                 up_activation=nn.ReLU(True), init_conv_kwargs=None,
                 downsample_conv_kwargs=None, resnet_conv_kwargs=None,
                 add_out_act=True, max_features=1024,
                 out_ffc=False, out_ffc_kwargs=None):
        assert n_blocks >= 0
        super().__init__()
        if init_conv_kwargs is None:
            init_conv_kwargs = {}
        if downsample_conv_kwargs is None:
            downsample_conv_kwargs = {}
        if resnet_conv_kwargs is None:
            resnet_conv_kwargs = {}
        if out_ffc_kwargs is None:
            out_ffc_kwargs = {}

        model = [
            nn.ReflectionPad2d(3),
            FFC_BN_ACT(input_nc, ngf, kernel_size=7, padding=0,
                       norm_layer=norm_layer, activation_layer=activation_layer,
                       **init_conv_kwargs),
        ]

        # downsample
        for i in range(n_downsampling):
            mult = 2 ** i
            if i == n_downsampling - 1:
                cur_conv_kwargs = dict(downsample_conv_kwargs)
                cur_conv_kwargs["ratio_gout"] = resnet_conv_kwargs.get("ratio_gin", 0)
            else:
                cur_conv_kwargs = downsample_conv_kwargs
            model += [FFC_BN_ACT(min(max_features, ngf * mult),
                                 min(max_features, ngf * mult * 2),
                                 kernel_size=3, stride=2, padding=1,
                                 norm_layer=norm_layer,
                                 activation_layer=activation_layer,
                                 **cur_conv_kwargs)]

        mult = 2 ** n_downsampling
        feats_num_bottleneck = min(max_features, ngf * mult)

        # resnet blocks
        for _ in range(n_blocks):
            model += [FFCResnetBlock(feats_num_bottleneck, padding_type=padding_type,
                                     activation_layer=activation_layer,
                                     norm_layer=norm_layer, **resnet_conv_kwargs)]
        model += [ConcatTupleLayer()]

        # upsample
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [
                nn.ConvTranspose2d(min(max_features, ngf * mult),
                                   min(max_features, int(ngf * mult / 2)),
                                   kernel_size=3, stride=2, padding=1, output_padding=1),
                up_norm_layer(min(max_features, int(ngf * mult / 2))),
                up_activation,
            ]

        if out_ffc:
            model += [FFCResnetBlock(ngf, padding_type=padding_type,
                                     activation_layer=activation_layer,
                                     norm_layer=norm_layer, inline=True,
                                     **out_ffc_kwargs)]

        model += [nn.ReflectionPad2d(3),
                  nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        if add_out_act:
            model.append(get_activation("tanh" if add_out_act is True else add_out_act))
        self.model = nn.Sequential(*model)

    def forward(self, img, mask, rel_pos=None, direct=None) -> Tensor:
        masked_img = torch.cat([img * (1 - mask), mask], dim=1)
        if rel_pos is None:
            return self.model(masked_img)
        else:
            x_l, x_g = self.model[:2](masked_img)
            x_l = x_l.to(torch.float32)
            x_l += rel_pos
            x_l += direct
            return self.model[2:]((x_l, x_g))


# ---------------------------------------------------------------------------
# LamaFourier wrapper (not nn.Module)
# ---------------------------------------------------------------------------

class LamaFourier:
    def __init__(self, large_arch: bool = False):
        n_blocks = 18 if large_arch else 9
        self.generator = FFCResNetGenerator(
            4, 3,
            add_out_act="sigmoid",
            n_blocks=n_blocks,
            init_conv_kwargs={"ratio_gin": 0, "ratio_gout": 0, "enable_lfu": False},
            downsample_conv_kwargs={"ratio_gin": 0, "ratio_gout": 0, "enable_lfu": False},
            resnet_conv_kwargs={"ratio_gin": 0.75, "ratio_gout": 0.75, "enable_lfu": False},
        )

    def to(self, device):
        self.generator.to(device)
        return self

    def eval(self):
        self.generator.eval()
        return self

    def __call__(self, img: Tensor, mask: Tensor) -> Tensor:
        predicted_img = self.generator(img, mask)
        return predicted_img * mask + (1 - mask) * img


# ---------------------------------------------------------------------------
# Load helper
# ---------------------------------------------------------------------------

def load_lama_large(model_path: str, device: str = "cpu") -> LamaFourier:
    model = LamaFourier(large_arch=True)
    sd = torch.load(model_path, map_location="cpu", weights_only=False)
    model.generator.load_state_dict(sd["gen_state_dict"])
    model.eval().to(device)
    return model


# ---------------------------------------------------------------------------
# LamaLargeInpainter — ModelWrapper-based inpainter
# ---------------------------------------------------------------------------

class LamaLargeInpainter(ModelWrapper):
    _MODEL_SUB_DIR = "inpainting"
    _MODEL_MAPPING = {
        "model": {
            "url": "https://huggingface.co/dreMaz/AnimeMangaInpainting/resolve/main/lama_large_512px.ckpt",
            "hash": "11d30fbb3000fb2eceae318b75d9ced9229d99ae990a7f8b3ac35c8d31f2c935",
            "file": ".",
        },
    }

    async def _load(self, device: str):
        self.model = load_lama_large(
            self._get_file_path("lama_large_512px.ckpt"), device="cpu"
        )
        self.device = device
        if device.startswith("cuda") or device == "mps":
            self.model.to(device)

    async def _unload(self):
        del self.model

    async def _infer(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        config: InpainterConfig,
        inpainting_size: int = 2048,
        verbose: bool = False,
    ) -> np.ndarray:
        img_original = np.copy(image)
        mask_original = np.copy(mask)
        mask_original[mask_original < 127] = 0
        mask_original[mask_original >= 127] = 1
        mask_original = mask_original[:, :, None]

        height, width = image.shape[:2]
        if max(height, width) > inpainting_size:
            image = resize_keep_aspect(image, inpainting_size)
            mask = resize_keep_aspect(mask, inpainting_size)

        # Pad to multiples of 8
        pad = 8
        h, w = image.shape[:2]
        new_h = h if h % pad == 0 else h + (pad - h % pad)
        new_w = w if w % pad == 0 else w + (pad - w % pad)
        if new_h != h or new_w != w:
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        logger.info(f"Inpainting resolution: {new_w}x{new_h}")

        # Normalize to [0,1] for LamaFourier (sigmoid output)
        img_torch = torch.from_numpy(image).permute(2, 0, 1).unsqueeze_(0).float() / 255.0
        mask_torch = torch.from_numpy(mask).unsqueeze_(0).unsqueeze_(0).float() / 255.0
        mask_torch[mask_torch < 0.5] = 0
        mask_torch[mask_torch >= 0.5] = 1

        if self.device.startswith("cuda") or self.device == "mps":
            img_torch = img_torch.to(self.device)
            mask_torch = mask_torch.to(self.device)

        with torch.no_grad():
            img_torch *= 1 - mask_torch
            if not self.device.startswith("cuda"):
                img_inpainted_torch = self.model(img_torch, mask_torch)
            else:
                precision = TORCH_DTYPE_MAP[str(config.inpainting_precision)]
                if precision == torch.float16:
                    precision = torch.bfloat16
                    logger.warning("Switched to bf16 — LaMa only supports bf16 and fp32.")
                with torch.autocast(device_type="cuda", dtype=precision):
                    img_inpainted_torch = self.model(img_torch, mask_torch)

        img_inpainted_torch = img_inpainted_torch.to(torch.float32)
        img_inpainted = (img_inpainted_torch.cpu().squeeze_(0).permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)

        if new_h != height or new_w != width:
            img_inpainted = cv2.resize(img_inpainted, (width, height), interpolation=cv2.INTER_LINEAR)

        return img_inpainted * mask_original + img_original * (1 - mask_original)

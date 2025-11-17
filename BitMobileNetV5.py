from __future__ import annotations
import argparse
from functools import partial
from typing import List, Sequence, Union, Optional

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from common_utils import CIFAR100DataModule, ImageNetDataModule, LitBit, TinyImageNetDataModule, add_common_args, setup_trainer, summ
from layers import ConvNormAct, RmsNorm2d
from layers.bit import Bit
from layers.efficientnet_blocks import SqueezeExcite, UniversalInvertedResidual
from layers.efficientnet_builder import EfficientNetBuilder, decode_arch_def, round_channels


_GELU = partial(nn.GELU, approximate="tanh")


# --- small helper: simplified version of timm's feature_take_indices ----------------
def _take_indices(num: int, indices: Optional[Union[int, Sequence[int]]]):
    """
    Normalise indices into [0, num-1].

    - indices=None -> all indices
    - indices=int  -> last `indices` entries
    - indices=seq  -> supports negative indices like Python lists
    """
    if indices is None:
        return list(range(num))

    if isinstance(indices, int):
        if indices >= 0:
            start = max(0, num - indices)
            return list(range(start, num))
        else:  # negative single int (rare)
            idx = num + indices
            return [idx]

    # sequence
    out = []
    for idx in indices:
        if idx < 0:
            idx = num + idx
        if idx < 0 or idx >= num:
            raise IndexError(f"index {idx} out of range for length {num}")
        out.append(idx)
    return out


class MobileNetV5MultiScaleFusionAdapter(nn.Module):
    def __init__(
        self,
        in_chs: Union[int, List[int]],
        out_chs: int,
        output_resolution: int,
        expansion_ratio: float = 2.0,
        interpolation_mode: str = "nearest",
        layer_scale_init_value: Optional[float] = None,
        noskip: bool = True,
        act_layer: Optional[nn.Module] = None,
        norm_layer: Optional[nn.Module] = None,
        device=None,
        dtype=None,
    ):
        super().__init__()
        dd = {"device": device, "dtype": dtype}

        self.in_channels = sum(in_chs) if isinstance(in_chs, Sequence) else in_chs
        self.out_channels = out_chs
        self.output_resolution = (output_resolution, output_resolution)
        self.expansion_ratio = expansion_ratio
        self.interpolation_mode = interpolation_mode
        self.layer_scale_init_value = layer_scale_init_value
        self.noskip = noskip

        act_layer = act_layer or _GELU
        norm_layer = norm_layer or RmsNorm2d

        self.ffn = UniversalInvertedResidual(
            in_chs=self.in_channels,
            out_chs=self.out_channels,
            dw_kernel_size_mid=0,
            exp_ratio=self.expansion_ratio,
            act_layer=act_layer,
            norm_layer=norm_layer,
            noskip=self.noskip,
            layer_scale_init_value=self.layer_scale_init_value,
        ).to(**dd)

        self.norm = norm_layer(self.out_channels, **dd)

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        assert len(inputs) > 0, "Need at least one feature map"

        # 1) upsample all to the highest resolution
        high_resolution = inputs[0].shape[-2:]
        resized = []
        for img in inputs:
            h, w = img.shape[-2:]
            if h < high_resolution[0] or w < high_resolution[1]:
                img = F.interpolate(
                    img, size=high_resolution, mode=self.interpolation_mode
                )
            resized.append(img)

        # 2) concat + FFN
        x = torch.cat(resized, dim=1)  # [B, sum(C_i), H, W]
        x = self.ffn(x)

        # 3) downsample / resize to fixed output_resolution if needed
        if high_resolution != self.output_resolution:
            if (
                high_resolution[0] % self.output_resolution[0] != 0
                or high_resolution[1] % self.output_resolution[1] != 0
            ):
                x = F.interpolate(x, size=self.output_resolution, mode="bilinear")
            else:
                h_stride = high_resolution[0] // self.output_resolution[0]
                w_stride = high_resolution[1] // self.output_resolution[1]
                x = F.avg_pool2d(
                    x,
                    kernel_size=(h_stride, w_stride),
                    stride=(h_stride, w_stride),
                )

        x = self.norm(x)
        return x


class MobileNetV5Backbone(nn.Module):
    """
    MobileNetV5-style encoder built on top of EfficientNetBuilder
    + MultiScaleFusionAdapter.
    """

    def __init__(
        self,
        arch_def: Sequence[Sequence[str]],
        in_chans: int = 3,
        stem_size: int = 64,
        channel_multiplier: float = 1.0,
        msfa_indices: Sequence[int] = None,  # which features to fuse (in feature_info space) (-2, -1)
        msfa_output_resolution: int = 16,
        out_channels: int = 2048,
        pad_type: str = "same",
        drop_path_rate: float = 0.0,
        layer_scale_init_value: float = 1e-5,
    ):
        super().__init__()
        dd = {}
        act_layer = _GELU
        norm_layer = RmsNorm2d
        se_layer = SqueezeExcite

        # ---- stem ----
        round_chs_fn = partial(round_channels, multiplier=channel_multiplier)
        if channel_multiplier < 1.0:
            stem_size = round_chs_fn(stem_size)

        self.conv_stem = ConvNormAct(
            in_chans,
            stem_size,
            kernel_size=3,
            stride=2,
            padding=pad_type,
            bias=True,
            norm_layer=norm_layer,
            act_layer=act_layer,
            **dd,
        )

        # ---- blocks via EfficientNetBuilder ----
        block_args = decode_arch_def(arch_def)

        builder = EfficientNetBuilder(
            output_stride=32,
            pad_type=pad_type,
            round_chs_fn=round_chs_fn,
            se_from_exp=True,
            act_layer=act_layer,
            norm_layer=norm_layer,
            aa_layer=None,
            se_layer=se_layer,
            drop_path_rate=drop_path_rate,
            layer_scale_init_value=layer_scale_init_value,
            **dd,
        )

        # builder(...) returns a list of stage-sequentials; wrap in nn.Sequential
        self.blocks = nn.Sequential(*builder(stem_size, block_args))
        self.feature_info = builder.features  # list of dicts with 'stage', 'num_chs', ...

        # 1) choose which entries in feature_info we want to fuse
        if msfa_indices is None:
            msfa_indices = list(range(len(self.feature_info)))
        feat_indices = _take_indices(len(self.feature_info), msfa_indices)
        self.msfa_feat_indices = feat_indices

        # 2) convert those to stage IDs (0 for stem, 1..N for stages)
        self.msfa_stage_ids = [self.feature_info[i]["stage"] for i in feat_indices]

        # 3) compute input channels to the fusion adapter by summing num_chs
        msfa_in_chs = sum(self.feature_info[i]["num_chs"] for i in feat_indices)

        self.msfa_output_resolution = msfa_output_resolution
        self.num_features = out_channels

        self.msfa = MobileNetV5MultiScaleFusionAdapter(
            in_chs=msfa_in_chs,
            out_chs=self.num_features,
            output_resolution=self.msfa_output_resolution,
            norm_layer=norm_layer,
            act_layer=act_layer,
            layer_scale_init_value=layer_scale_init_value,
            **dd,
        )

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        - Run stem + stages
        - Collect the stage outputs whose stage IDs are in self.msfa_stage_ids
        - Fuse them via MSFA
        """
        intermediates: List[torch.Tensor] = []

        # stage_id 0 = stem output
        stage_id = 0
        x = self.conv_stem(x)
        if stage_id in self.msfa_stage_ids:
            intermediates.append(x)

        # each child of self.blocks is a *stage* (nn.Sequential of blocks)
        for blk in self.blocks:
            stage_id += 1
            x = blk(x)
            if stage_id in self.msfa_stage_ids:
                intermediates.append(x)

        # (optional sanity check)
        # print(len(intermediates), "features for MSFA")
        # assert len(intermediates) == len(self.msfa_stage_ids)

        x = self.msfa(intermediates)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_features(x)


class MobileNetV5Classifier(nn.Module):
    """
    MobileNetV5-style classifier:
      - backbone: MobileNetV5Backbone
      - global pool + linear head
    """

    def __init__(
        self,
        num_classes: int,
        arch_def: Sequence[Sequence[str]],
        in_chans: int = 3,
        stem_size: int = 64,
        channel_multiplier: float = 1.0,
        msfa_indices: Sequence[int] = None,
        msfa_output_resolution: int = 16,
        out_channels: int = 2048,
        drop_rate: float = 0.0,
    ):
        super().__init__()

        
        self.num_classes=num_classes
        self.arch_def=arch_def
        self.in_chans=in_chans
        self.stem_size=stem_size
        self.channel_multiplier=channel_multiplier
        self.msfa_indices=msfa_indices
        self.msfa_output_resolution=msfa_output_resolution
        self.out_channels=out_channels
        self.drop_rate=drop_rate

        self.backbone = MobileNetV5Backbone(
            arch_def=arch_def,
            in_chans=in_chans,
            stem_size=stem_size,
            channel_multiplier=channel_multiplier,
            msfa_indices=msfa_indices,
            msfa_output_resolution=msfa_output_resolution,
            out_channels=out_channels,
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
            nn.Dropout(drop_rate) if drop_rate > 0 else nn.Identity(),
            Bit.Linear(out_channels, num_classes)
        )
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone.forward_features(x)  # [B, C, H, W]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encode(x))


    def clone(self) -> "MobileNetV5Classifier":
        return self.__class__(
            num_classes=self.num_classes,
            arch_def=self.arch_def,
            in_chans=self.in_chans,
            stem_size=self.stem_size,
            channel_multiplier=self.channel_multiplier,
            msfa_indices=self.msfa_indices,
            msfa_output_resolution=self.msfa_output_resolution,
            out_channels=self.out_channels,
            drop_rate=self.drop_rate,
        )

MNv5_TINY_ARCH_DEF: list[list[str]] = [
    # Stage 0: 128x128 in
    # Keep only the initial downsampling block
    [
        'er_r1_k3_s2_e4_c128',
    ],

    # Stage 1: 256x256 in
    # Keep downsample + 1 follow-up block
    [
        'uir_r1_a3_k5_s2_e6_c256',
        'uir_r1_a5_k0_s1_e4_c256',
    ],

    # Stage 2: 640x640 in
    # Keep first conv, a light conv, and a single (MQA + UIR) pair
    [
        "uir_r1_a5_k5_s2_e6_c512",   # main downsample
        "uir_r1_a0_k0_s1_e1_c512",   # lightweight conv

        'mqa_r1_k3_h8_s2_d64_c512',  # one attention block
        "uir_r1_a0_k0_s1_e2_c512",   # follow-up conv
        "uir_r1_a0_k0_s1_e2_c512",   # extra conv depth (still cheap)
    ],

    # Stage 3: 1280x1280 in
    # Keep downsample + 1 MQA + 2 light convs
    [
        "uir_r1_a5_k5_s2_e6_c1024",      # downsample

        'mqa_r1_k3_h16_s1_d64_c1024',    # single MQA at top
        "uir_r1_a0_k0_s1_e2_c1024",      # conv
        "uir_r1_a0_k0_s1_e2_c1024",      # one more conv for a bit of depth
    ],
]

MNv5_SMALL_ARCH_DEF: list[list[str]] = [
    # Stage 0: 128x128 in
    # Keep first s2 block + 1 repeat
    [
        'er_r1_k3_s2_e4_c128',
        'er_r1_k3_s1_e4_c128',
    ],

    # Stage 1: 256x256 in
    # Keep downsample + two follow-up blocks (drop 2 repeats)
    [
        'uir_r1_a3_k5_s2_e6_c256',
        'uir_r1_a5_k0_s1_e4_c256',
        'uir_r1_a3_k0_s1_e4_c256',
    ],

    # Stage 2: 640x640 in
    # Keep first conv sequence and only 3 (MQA + UIR) pairs instead of 6
    [
        "uir_r1_a5_k5_s2_e6_c512",
        "uir_r1_a5_k0_s1_e4_c512",
        "uir_r1_a0_k0_s1_e1_c512",

        'mqa_r1_k3_h8_s2_d64_c512',
        "uir_r1_a0_k0_s1_e2_c512",

        'mqa_r1_k3_h8_s2_d64_c512',
        "uir_r1_a0_k0_s1_e2_c512",

        'mqa_r1_k3_h8_s2_d64_c512',
        "uir_r1_a0_k0_s1_e2_c512",
    ],

    # Stage 3: 1280x1280 in
    # Keep the first downsample + 3 (MQA + UIR) pairs instead of 7
    [
        "uir_r1_a5_k5_s2_e6_c1024",

        'mqa_r1_k3_h16_s1_d64_c1024',
        "uir_r1_a0_k0_s1_e2_c1024",

        'mqa_r1_k3_h16_s1_d64_c1024',
        "uir_r1_a0_k0_s1_e2_c1024",

        'mqa_r1_k3_h16_s1_d64_c1024',
        "uir_r1_a0_k0_s1_e2_c1024",
    ],
]

MNv5_BASE_ARCH_DEF: list[list[str]] = [
    # Stage 0: 128x128 in
    [
        'er_r1_k3_s2_e4_c128',
        'er_r1_k3_s1_e4_c128',
        'er_r1_k3_s1_e4_c128',
    ],
    # Stage 1: 256x256 in
    [
        'uir_r1_a3_k5_s2_e6_c256',
        'uir_r1_a5_k0_s1_e4_c256',
        'uir_r1_a3_k0_s1_e4_c256',
        'uir_r1_a5_k0_s1_e4_c256',
        'uir_r1_a3_k0_s1_e4_c256',
    ],
    # Stage 2: 640x640 in
    [
        "uir_r1_a5_k5_s2_e6_c512",
        "uir_r1_a5_k0_s1_e4_c512",
        "uir_r1_a5_k0_s1_e4_c512",
        "uir_r1_a0_k0_s1_e1_c512",
        'mqa_r1_k3_h8_s2_d64_c512',
        "uir_r1_a0_k0_s1_e2_c512",
        'mqa_r1_k3_h8_s2_d64_c512',
        "uir_r1_a0_k0_s1_e2_c512",
        'mqa_r1_k3_h8_s2_d64_c512',
        "uir_r1_a0_k0_s1_e2_c512",
        'mqa_r1_k3_h8_s2_d64_c512',
        "uir_r1_a0_k0_s1_e2_c512",
        'mqa_r1_k3_h8_s2_d64_c512',
        "uir_r1_a0_k0_s1_e2_c512",
        'mqa_r1_k3_h8_s2_d64_c512',
        "uir_r1_a0_k0_s1_e2_c512",
    ],
    # Stage 3: 1280x1280 in
    [
        "uir_r1_a5_k5_s2_e6_c1024",
        'mqa_r1_k3_h16_s1_d64_c1024',
        "uir_r1_a0_k0_s1_e2_c1024",
        'mqa_r1_k3_h16_s1_d64_c1024',
        "uir_r1_a0_k0_s1_e2_c1024",
        'mqa_r1_k3_h16_s1_d64_c1024',
        "uir_r1_a0_k0_s1_e2_c1024",
        'mqa_r1_k3_h16_s1_d64_c1024',
        "uir_r1_a0_k0_s1_e2_c1024",
        'mqa_r1_k3_h16_s1_d64_c1024',
        "uir_r1_a0_k0_s1_e2_c1024",
        'mqa_r1_k3_h16_s1_d64_c1024',
        "uir_r1_a0_k0_s1_e2_c1024",
        'mqa_r1_k3_h16_s1_d64_c1024',
        "uir_r1_a0_k0_s1_e2_c1024",
    ],
]

MNv5_LARGE_ARCH_DEF: list[list[str]] = [
    # Stage 0: 128x128 in
    [
        'er_r1_k3_s2_e4_c128',
        'er_r1_k3_s1_e4_c128',
        'er_r1_k3_s1_e4_c128',
    ],
    # Stage 1: 256x256 in
    [
        'uir_r1_a3_k5_s2_e6_c256',
        'uir_r1_a5_k0_s1_e4_c256',
        'uir_r1_a3_k0_s1_e4_c256',
        'uir_r1_a5_k0_s1_e4_c256',
        'uir_r1_a3_k0_s1_e4_c256',
    ],
    # Stage 2: 640x640 in (reduced)
    [
        "uir_r1_a5_k5_s2_e6_c640",       # downsample, keep
        "uir_r1_a5_k0_s1_e4_c640",
        "uir_r1_a5_k0_s1_e4_c640",
        "uir_r1_a5_k0_s1_e4_c640",       # 3 conv blocks instead of 7

        "uir_r1_a0_k0_s1_e1_c640",       # keep transition block

        # 6x (MQA + UIR) pairs instead of 13
        "mqa_r1_k3_h12_v2_s1_d64_c640",
        "uir_r1_a0_k0_s1_e2_c640",
        "mqa_r1_k3_h12_v2_s1_d64_c640",
        "uir_r1_a0_k0_s1_e2_c640",
        "mqa_r1_k3_h12_v2_s1_d64_c640",
        "uir_r1_a0_k0_s1_e2_c640",
        "mqa_r1_k3_h12_v2_s1_d64_c640",
        "uir_r1_a0_k0_s1_e2_c640",
        "mqa_r1_k3_h12_v2_s1_d64_c640",
        "uir_r1_a0_k0_s1_e2_c640",
        "mqa_r1_k3_h12_v2_s1_d64_c640",
        "uir_r1_a0_k0_s1_e2_c640",
    ],
    # Stage 3: 1280x1280 in (reduced)
    [
        "uir_r1_a5_k5_s2_e6_c1280",      # downsample, keep

        # 8x (MQA + UIR) pairs instead of 19
        "mqa_r1_k3_h16_s1_d96_c1280",
        "uir_r1_a0_k0_s1_e2_c1280",
        "mqa_r1_k3_h16_s1_d96_c1280",
        "uir_r1_a0_k0_s1_e2_c1280",
        "mqa_r1_k3_h16_s1_d96_c1280",
        "uir_r1_a0_k0_s1_e2_c1280",
        "mqa_r1_k3_h16_s1_d96_c1280",
        "uir_r1_a0_k0_s1_e2_c1280",
        "mqa_r1_k3_h16_s1_d96_c1280",
        "uir_r1_a0_k0_s1_e2_c1280",
        "mqa_r1_k3_h16_s1_d96_c1280",
        "uir_r1_a0_k0_s1_e2_c1280",
        "mqa_r1_k3_h16_s1_d96_c1280",
        "uir_r1_a0_k0_s1_e2_c1280",
        "mqa_r1_k3_h16_s1_d96_c1280",
        "uir_r1_a0_k0_s1_e2_c1280",
    ],
]

MNv5_300M_ARCH_DEF: list[list[str]] = [
            # Stage 0: 128x128 in
            [
                'er_r1_k3_s2_e4_c128',
                'er_r1_k3_s1_e4_c128',
                'er_r1_k3_s1_e4_c128',
            ],
            # Stage 1: 256x256 in
            [
                'uir_r1_a3_k5_s2_e6_c256',
                'uir_r1_a5_k0_s1_e4_c256',
                'uir_r1_a3_k0_s1_e4_c256',
                'uir_r1_a5_k0_s1_e4_c256',
                'uir_r1_a3_k0_s1_e4_c256',
            ],
            # Stage 2: 640x640 in
            [
                "uir_r1_a5_k5_s2_e6_c640",
                "uir_r1_a5_k0_s1_e4_c640",
                "uir_r1_a5_k0_s1_e4_c640",
                "uir_r1_a5_k0_s1_e4_c640",
                "uir_r1_a5_k0_s1_e4_c640",
                "uir_r1_a5_k0_s1_e4_c640",
                "uir_r1_a5_k0_s1_e4_c640",
                "uir_r1_a5_k0_s1_e4_c640",
                "uir_r1_a0_k0_s1_e1_c640",
                "mqa_r1_k3_h12_v2_s1_d64_c640",
                "uir_r1_a0_k0_s1_e2_c640",
                "mqa_r1_k3_h12_v2_s1_d64_c640",
                "uir_r1_a0_k0_s1_e2_c640",
                "mqa_r1_k3_h12_v2_s1_d64_c640",
                "uir_r1_a0_k0_s1_e2_c640",
                "mqa_r1_k3_h12_v2_s1_d64_c640",
                "uir_r1_a0_k0_s1_e2_c640",
                "mqa_r1_k3_h12_v2_s1_d64_c640",
                "uir_r1_a0_k0_s1_e2_c640",
                "mqa_r1_k3_h12_v2_s1_d64_c640",
                "uir_r1_a0_k0_s1_e2_c640",
                "mqa_r1_k3_h12_v2_s1_d64_c640",
                "uir_r1_a0_k0_s1_e2_c640",
                "mqa_r1_k3_h12_v2_s1_d64_c640",
                "uir_r1_a0_k0_s1_e2_c640",
                "mqa_r1_k3_h12_v2_s1_d64_c640",
                "uir_r1_a0_k0_s1_e2_c640",
                "mqa_r1_k3_h12_v2_s1_d64_c640",
                "uir_r1_a0_k0_s1_e2_c640",
                "mqa_r1_k3_h12_v2_s1_d64_c640",
                "uir_r1_a0_k0_s1_e2_c640",
                "mqa_r1_k3_h12_v2_s1_d64_c640",
                "uir_r1_a0_k0_s1_e2_c640",
                "mqa_r1_k3_h12_v2_s1_d64_c640",
                "uir_r1_a0_k0_s1_e2_c640",
                "mqa_r1_k3_h12_v2_s1_d64_c640",
                "uir_r1_a0_k0_s1_e2_c640",
            ],
            # Stage 3: 1280x1280 in
            [
                "uir_r1_a5_k5_s2_e6_c1280",
                "mqa_r1_k3_h16_s1_d96_c1280",
                "uir_r1_a0_k0_s1_e2_c1280",
                "mqa_r1_k3_h16_s1_d96_c1280",
                "uir_r1_a0_k0_s1_e2_c1280",
                "mqa_r1_k3_h16_s1_d96_c1280",
                "uir_r1_a0_k0_s1_e2_c1280",
                "mqa_r1_k3_h16_s1_d96_c1280",
                "uir_r1_a0_k0_s1_e2_c1280",
                "mqa_r1_k3_h16_s1_d96_c1280",
                "uir_r1_a0_k0_s1_e2_c1280",
                "mqa_r1_k3_h16_s1_d96_c1280",
                "uir_r1_a0_k0_s1_e2_c1280",
                "mqa_r1_k3_h16_s1_d96_c1280",
                "uir_r1_a0_k0_s1_e2_c1280",
                "mqa_r1_k3_h16_s1_d96_c1280",
                "uir_r1_a0_k0_s1_e2_c1280",
                "mqa_r1_k3_h16_s1_d96_c1280",
                "uir_r1_a0_k0_s1_e2_c1280",
                "mqa_r1_k3_h16_s1_d96_c1280",
                "uir_r1_a0_k0_s1_e2_c1280",
                "mqa_r1_k3_h16_s1_d96_c1280",
                "uir_r1_a0_k0_s1_e2_c1280",
                "mqa_r1_k3_h16_s1_d96_c1280",
                "uir_r1_a0_k0_s1_e2_c1280",
                "mqa_r1_k3_h16_s1_d96_c1280",
                "uir_r1_a0_k0_s1_e2_c1280",
                "mqa_r1_k3_h16_s1_d96_c1280",
                "uir_r1_a0_k0_s1_e2_c1280",
                "mqa_r1_k3_h16_s1_d96_c1280",
                "uir_r1_a0_k0_s1_e2_c1280",
                "mqa_r1_k3_h16_s1_d96_c1280",
                "uir_r1_a0_k0_s1_e2_c1280",
                "mqa_r1_k3_h16_s1_d96_c1280",
                "uir_r1_a0_k0_s1_e2_c1280",
                "mqa_r1_k3_h16_s1_d96_c1280",
                "uir_r1_a0_k0_s1_e2_c1280",
                "mqa_r1_k3_h16_s1_d96_c1280",
                "uir_r1_a0_k0_s1_e2_c1280",
            ],
        ]

# model = MobileNetV5Backbone(
#     arch_def=MNv5_BASE_ARCH_DEF,
#     in_chans=3,
#     stem_size=64,
#     channel_multiplier=1.0,
#     msfa_output_resolution=16,    # final H=W=16
# )

# x = torch.randn(1, 3, 256, 256)
# feat = model(x)
# print(feat.shape)  # -> [1, 256, 16, 16]
# info = summ(model,False)


def make_mobilenetv5_teacher(size="300m",dataset=None,num_classes=None,device="cpu",pretrained=True):
    model = timm.create_model('mobilenetv5_300m.gemma3n', pretrained=True)
    model = model.eval().to(device=device)
    return model


# ----------------------------
# LightningModule: KD + hints
# ----------------------------
class LitMobileNetV5KD(LitBit):
    def __init__(
        self,
        lr, wd, epochs,
        dataset_name='c100',
        model_size="small",            # 'small'|'medium'|'large' or 'hybrid_medium'|'hybrid_large' (alias: hybrid_medium)
        label_smoothing=0.1, alpha_kd=0.0, alpha_hint=0.0, T=4.0,
        amp=True, export_dir="./ckpt_mnv5",
        drop_path_rate=0.0,
        teacher_pretrained=True
    ):
        # dataset -> classes
        ds = dataset_name.lower()
        if ds in ['c10', 'cifar10']:
            num_classes = 10
        elif ds in ['c100', 'cifar100']:
            num_classes = 100
        elif ds in ['timnet', 'tiny', 'tinyimagenet', 'tiny-imagenet']:
            num_classes = 200
        elif ds in ['imnet', 'imagenet', 'in1k', 'imagenet1k']:
            num_classes = 1000
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
        
        def get_mnv5_arch_def(model_size: str) -> list[list[str]]:
            size = model_size.lower()
            if size == "tiny":
                return MNv5_TINY_ARCH_DEF
            elif size == "small":
                return MNv5_SMALL_ARCH_DEF
            elif size == "base":
                return MNv5_BASE_ARCH_DEF
            elif size == "large":
                return MNv5_LARGE_ARCH_DEF
            elif size == "300m":
                return MNv5_300M_ARCH_DEF
            else:
                raise ValueError(f"Unknown MobileNetV5 model_size: {model_size}")

        # student & teacher
        student = MobileNetV5Classifier(arch_def=get_mnv5_arch_def(model_size),
                                        num_classes=num_classes,
                                        drop_rate=drop_path_rate)

        teacher = make_mobilenetv5_teacher(
            size="300m",
            dataset=ds,
            num_classes=num_classes,
            device="cpu",
            pretrained=teacher_pretrained
        )

        summ(student)
        # summ(teacher)

        # pick robust hint tap points via timm feature_info
        hint_points = [("backbone.blocks.0","blocks.0"), ("backbone.blocks.1","blocks.1"),
                       ("backbone.blocks.2","blocks.2"), ("backbone.blocks.3","blocks.3"),
                       ("backbone.msfa","msfa")]

        super().__init__(
            lr, wd, epochs, label_smoothing,
            alpha_kd, alpha_hint, T,
            amp,
            export_dir,
            dataset_name=ds,
            model_name='mobilenetv5',
            model_size=model_size,
            hint_points=hint_points,
            student=student,
            teacher=teacher,
            num_classes=num_classes
        )

# ----------------------------
# CLI / main (MobileNetV5)
# ----------------------------
def parse_args_mnv5():
    p = argparse.ArgumentParser()
    p = add_common_args(p)

    p.add_argument("--dataset", type=str, default="timnet",
                   choices=["c10", "cifar10", "c100", "cifar100", "timnet", "tiny",
                            "tinyimagenet", "tiny-imagenet", "imnet", "imagenet", "in1k", "imagenet1k"],
                   help="Target dataset (affects datamodule, num_classes, transforms).")

    # For MobileNetV5 we accept conv + hybrid tags in one flag
    p.add_argument("--model-size", type=str, default="small",
                   choices=["tiny", "small", "base", "large", "300m"],
                   help="MobileNetV5 variant.")

    p.add_argument("--drop-path", type=float, default=0.0)
    p.add_argument("--teacher-pretrained", type=lambda x: str(x).lower() in ["1","true","yes","y"], default=True,
                   help="Use ImageNet-pretrained teacher backbone when classes != 1000 (head is replaced).")

    p.set_defaults(out=None,batch_size=16,lr=0.2,alpha_kd=0.0,alpha_hint=0.05)

    args = p.parse_args()
    if args.out is None:
        args.out = f"./ckpt_{args.dataset}_mnv5_{args.model_size}"
    return args


def _pick_datamodule_mnv5(dataset_name: str, dmargs: dict):
    # reuse your existing modules; same as before
    ds = dataset_name.lower()
    if ds in ['c100', 'cifar100']:
        if 'CIFAR100DataModule' in globals():
            return CIFAR100DataModule(**dmargs)
        else:
            raise RuntimeError("CIFAR100DataModule not found in common_utils.")
    elif ds in ['timnet', 'tiny', 'tinyimagenet', 'tiny-imagenet']:
        if 'TinyImageNetDataModule' in globals():
            return TinyImageNetDataModule(**dmargs)
        else:
            raise RuntimeError("TinyImageNetDataModule not found in common_utils.")
    elif ds in ['imnet', 'imagenet', 'in1k', 'imagenet1k']:
        if 'ImageNetDataModule' in globals():
            return ImageNetDataModule(**dmargs)
        else:
            raise RuntimeError("ImageNetDataModule not found in common_utils.")
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")


def main_mnv5():
    args = parse_args_mnv5()

    # Derive num_classes for export dir naming (same as your convnext main)
    ds = args.dataset.lower()
    if ds in ['c10', 'cifar10']:
        ncls = 10
    elif ds in ['c100', 'cifar100']:
        ncls = 100
    elif ds in ['timnet', 'tiny', 'tinyimagenet', 'tiny-imagenet']:
        ncls = 200
    elif ds in ['imnet', 'imagenet', 'in1k', 'imagenet1k']:
        ncls = 1000
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    out_dir = f"{args.out}_{ds}_{args.model_size}_{ncls}c"

    lit = LitMobileNetV5KD(
        lr=args.lr, wd=args.wd, epochs=args.epochs,
        dataset_name=args.dataset,
        model_size=args.model_size,
        label_smoothing=args.label_smoothing,
        alpha_kd=args.alpha_kd, alpha_hint=args.alpha_hint, T=args.T,
        amp=args.amp, export_dir=out_dir, drop_path_rate=args.drop_path,
        teacher_pretrained=args.teacher_pretrained
    )

    dmargs = dict(
        data_dir=args.data,
        batch_size=args.batch_size,
        num_workers=4,
        aug_mixup=args.mixup,
        aug_cutmix=args.cutmix,
        alpha=args.mix_alpha
    )
    dm = _pick_datamodule_mnv5(args.dataset, dmargs)

    trainer, dm = setup_trainer(args, lit, dm)
    trainer.fit(lit, datamodule=dm)
    trainer.validate(lit, datamodule=dm)


if __name__ == "__main__":
    main_mnv5()

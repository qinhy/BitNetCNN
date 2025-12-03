MNv5_TINY_ARCH_DEF = [
    [
        "er_r1_k3_s2_e4_c128",
    ],
    [
        "uir_r1_a3_k5_s2_e6_c256",
        "uir_r1_a5_k0_s1_e4_c256",
    ],
    [
        "uir_r1_a5_k5_s2_e6_c512",
        "uir_r1_a0_k0_s1_e1_c512",
        "mqa_r1_k3_h8_s2_d64_c512",
        "uir_r1_a0_k0_s1_e2_c512",
        "uir_r1_a0_k0_s1_e2_c512",
    ],
    [
        "uir_r1_a5_k5_s2_e6_c1024",
        "mqa_r1_k3_h16_s1_d64_c1024",
        "uir_r1_a0_k0_s1_e2_c1024",
        "uir_r1_a0_k0_s1_e2_c1024",
    ],
]

MNv5_SMALL_ARCH_DEF = [
    [
        "er_r1_k3_s2_e4_c128",
        "er_r1_k3_s1_e4_c128",
    ],
    [
        "uir_r1_a3_k5_s2_e6_c256",
        "uir_r1_a5_k0_s1_e4_c256",
        "uir_r1_a3_k0_s1_e4_c256",
    ],
    [
        "uir_r1_a5_k5_s2_e6_c512",
        "uir_r1_a5_k0_s1_e4_c512",
        "uir_r1_a0_k0_s1_e1_c512",
        "mqa_r1_k3_h8_s2_d64_c512",
        "uir_r1_a0_k0_s1_e2_c512",
        "mqa_r1_k3_h8_s2_d64_c512",
        "uir_r1_a0_k0_s1_e2_c512",
        "mqa_r1_k3_h8_s2_d64_c512",
        "uir_r1_a0_k0_s1_e2_c512",
    ],
    [
        "uir_r1_a5_k5_s2_e6_c1024",
        "mqa_r1_k3_h16_s1_d64_c1024",
        "uir_r1_a0_k0_s1_e2_c1024",
        "mqa_r1_k3_h16_s1_d64_c1024",
        "uir_r1_a0_k0_s1_e2_c1024",
        "mqa_r1_k3_h16_s1_d64_c1024",
        "uir_r1_a0_k0_s1_e2_c1024",
    ],
]

MNv5_BASE_ARCH_DEF = [
    # Stage 0: 128x128 in
    [
        "er_r1_k3_s2_e4_c128",
        "er_r1_k3_s1_e4_c128",
        "er_r1_k3_s1_e4_c128",
    ],
    # Stage 1: 256x256 in
    [
        "uir_r1_a3_k5_s2_e6_c256",
        "uir_r1_a5_k0_s1_e4_c256",
        "uir_r1_a3_k0_s1_e4_c256",
        "uir_r1_a5_k0_s1_e4_c256",
        "uir_r1_a3_k0_s1_e4_c256",
    ],
    # Stage 2: 640x640 in
    [
        "uir_r1_a5_k5_s2_e6_c512",
        "uir_r1_a5_k0_s1_e4_c512",
        "uir_r1_a5_k0_s1_e4_c512",
        "uir_r1_a0_k0_s1_e1_c512",
        "mqa_r1_k3_h8_s2_d64_c512",
        "uir_r1_a0_k0_s1_e2_c512",
        "mqa_r1_k3_h8_s2_d64_c512",
        "uir_r1_a0_k0_s1_e2_c512",
        "mqa_r1_k3_h8_s2_d64_c512",
        "uir_r1_a0_k0_s1_e2_c512",
        "mqa_r1_k3_h8_s2_d64_c512",
        "uir_r1_a0_k0_s1_e2_c512",
        "mqa_r1_k3_h8_s2_d64_c512",
        "uir_r1_a0_k0_s1_e2_c512",
        "mqa_r1_k3_h8_s2_d64_c512",
        "uir_r1_a0_k0_s1_e2_c512",
        "mqa_r1_k3_h8_s2_d64_c512",
        "uir_r1_a0_k0_s1_e2_c512",
    ],
    # Stage 3: 1280x1280 in
    [
        "uir_r1_a5_k5_s2_e6_c1024",
        "mqa_r1_k3_h16_s1_d64_c1024",
        "uir_r1_a0_k0_s1_e2_c1024",
        "mqa_r1_k3_h16_s1_d64_c1024",
        "uir_r1_a0_k0_s1_e2_c1024",
        "mqa_r1_k3_h16_s1_d64_c1024",
        "uir_r1_a0_k0_s1_e2_c1024",
        "mqa_r1_k3_h16_s1_d64_c1024",
        "uir_r1_a0_k0_s1_e2_c1024",
        "mqa_r1_k3_h16_s1_d64_c1024",
        "uir_r1_a0_k0_s1_e2_c1024",
        "mqa_r1_k3_h16_s1_d64_c1024",
        "uir_r1_a0_k0_s1_e2_c1024",
        "mqa_r1_k3_h16_s1_d64_c1024",
        "uir_r1_a0_k0_s1_e2_c1024",
    ],
]

MNv5_LARGE_ARCH_DEF = [
    # Stage 0: 128x128 in
    [
        "er_r1_k3_s2_e4_c128",
        "er_r1_k3_s1_e4_c128",
        "er_r1_k3_s1_e4_c128",
    ],
    # Stage 1: 256x256 in
    [
        "uir_r1_a3_k5_s2_e6_c256",
        "uir_r1_a5_k0_s1_e4_c256",
        "uir_r1_a3_k0_s1_e4_c256",
        "uir_r1_a5_k0_s1_e4_c256",
        "uir_r1_a3_k0_s1_e4_c256",
    ],
    # Stage 2: 640x640 in (reduced)
    [
        "uir_r1_a5_k5_s2_e6_c640",  # downsample, keep
        "uir_r1_a5_k0_s1_e4_c640",
        "uir_r1_a5_k0_s1_e4_c640",
        "uir_r1_a5_k0_s1_e4_c640",  # 3 conv blocks instead of 7
        "uir_r1_a0_k0_s1_e1_c640",  # keep transition block
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
        "uir_r1_a5_k5_s2_e6_c1280",  # downsample, keep
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

MNv5_300M_ARCH_DEF = [
    # Stage 0: 128x128 in
    [
        "er_r1_k3_s2_e4_c128",
        "er_r1_k3_s1_e4_c128",
        "er_r1_k3_s1_e4_c128",
    ],
    # Stage 1: 256x256 in
    [
        "uir_r1_a3_k5_s2_e6_c256",
        "uir_r1_a5_k0_s1_e4_c256",
        "uir_r1_a3_k0_s1_e4_c256",
        "uir_r1_a5_k0_s1_e4_c256",
        "uir_r1_a3_k0_s1_e4_c256",
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
    ],
]

import argparse
import re
from functools import partial
from typing import List, Optional, Sequence, Union

from pydanticV2_argparse import ArgumentParser
import torch
import torch.nn as nn
import torch.nn.functional as F

from bitlayers.acts import ActModels
from bitlayers.attn2d import Attention2dModels
from bitlayers.bit import Bit
from bitlayers.convs import Conv2dModels
from bitlayers.norms import NormModels
from bitlayers.uir import UniversalInvertedResidual
from common_utils import *


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

        def act():
            return ActModels.GELU()

        def norm(num_features=-1):
            return NormModels.RmsNorm2d(num_features=num_features)

        self.ffn = (
            UniversalInvertedResidual(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                dw_kernel_size_mid=0,
                exp_ratio=self.expansion_ratio,
                noskip=self.noskip,
                layer_scale_init_value=self.layer_scale_init_value,
                conv_dw_start_layer=Conv2dModels.Conv2dDepthwiseNorm(
                    in_channels=-1,
                    norm=norm(),
                ),
                conv_pw_exp_layer=Conv2dModels.Conv2dPointwiseNormAct(
                    in_channels=-1,
                    norm=norm(),
                    act=act(),
                ),
                conv_dw_mid_layer=Conv2dModels.Conv2dDepthwiseNormAct(
                    in_channels=-1,
                    norm=norm(),
                    act=act(),
                ),
                conv_pw_proj_layer=Conv2dModels.Conv2dPointwiseNorm(
                    in_channels=-1,
                    norm=norm(),
                ),
                conv_dw_end_layer=Conv2dModels.Conv2dDepthwiseNorm(
                    in_channels=-1,
                    norm=norm(),
                ),
            )
            .build()
            .to(**dd)
        )

        self.norm = norm(num_features=self.out_channels).build()

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        assert len(inputs) > 0, "Need at least one feature map"

        # 1) upsample all to the highest resolution
        high_resolution = inputs[0].shape[-2:]
        resized = []
        for img in inputs:
            h, w = img.shape[-2:]
            if h < high_resolution[0] or w < high_resolution[1]:
                img = F.interpolate(
                    img,
                    size=high_resolution,
                    mode=self.interpolation_mode,
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
                x = F.interpolate(
                    x,
                    size=self.output_resolution,
                    mode="bilinear",
                )
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

        # ---- stem ----
        def round_channels(
            channels,
            multiplier=1.0,
            divisor=8,
            channel_min=None,
            round_limit=0.9,
        ):
            """Round number of filters based on depth multiplier."""
            if not multiplier:
                return channels
            return make_divisible(
                channels * multiplier,
                divisor,
                channel_min,
                round_limit=round_limit,
            )

        def make_divisible(v, divisor=8, min_value=None, round_limit=0.9):
            min_value = min_value or divisor
            new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
            # Make sure that round down does not go down by more than 10%.
            if new_v < round_limit * v:
                new_v += divisor
            return new_v

        round_chs_fn = partial(round_channels, multiplier=channel_multiplier)
        if channel_multiplier < 1.0:
            stem_size = round_chs_fn(stem_size)

        self.conv_stem = Conv2dModels.Conv2dNormAct(
            in_channels=in_chans,
            out_channels=stem_size,
            kernel_size=3,
            stride=2,
            padding=pad_type,
            bias=True,
            norm=NormModels.RmsNorm2d(num_features=-1),
            act=ActModels.GELU(),
        ).build()

        self.blocks, info = parse_arch_def(arch_def)
        # builder.features  # list of dicts with 'stage', 'num_chs', ...
        self.feature_info = info

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

        self.num_classes = num_classes
        self.arch_def = arch_def
        self.in_chans = in_chans
        self.stem_size = stem_size
        self.channel_multiplier = channel_multiplier
        self.msfa_indices = msfa_indices
        self.msfa_output_resolution = msfa_output_resolution
        self.out_channels = out_channels
        self.drop_rate = drop_rate

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
            Bit.Linear(out_channels, num_classes),
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

def parse_block_def(block_str):
    """Decode block definition string.

    Gets a list of block arg (dicts) through a string notation of arguments.
    E.g. ir_r2_k3_s2_e1_i32_o16_se0.25_noskip

    All args can exist in any order with the exception of the leading string which
    is assumed to indicate the block type.

    leading string - block type (
      ir = InvertedResidual, ds = DepthwiseSep, dsa = DeptwhiseSep with pw act, cn = ConvBnAct)
    r - number of repeat blocks,
    k - kernel size,
    s - strides (1-9),
    e - expansion ratio,
    c - output channels,
    se - squeeze/excitation ratio
    n - activation fn ('re', 'r6', 'hs', or 'sw')
    Args:
        block_str: a string representation of block arguments.
    Returns:
        A list of block args (dicts)
    Raises:
        ValueError: if the string def not properly specified (TODO)
    """
    assert isinstance(block_str, str)

    def _parse_ksize(ss):
        if ss.isdigit():
            return int(ss)
        else:
            return [int(k) for k in ss.split(".")]

    ops = block_str.split("_")
    block_type = ops[0]  # take the block type off the front
    ops = ops[1:]
    options = {}
    skip = None
    for op in ops:
        # string options being checked on individual basis, combine if they grow
        if op == "noskip":
            skip = False  # force no skip connection
        elif op == "skip":
            skip = True  # force a skip connection
        elif op.startswith("n"):
            # activation fn
            key = op[0]
            v = op[1:]
            if v == "re":
                value = "relu"
            elif v == "r6":
                value = "relu6"
            elif v == "hs":
                value = "hard_swish"
            elif v == "sw":
                value = "swish"
            elif v == "mi":
                value = "mish"
            else:
                continue
            options[key] = value
        else:
            # all numeric options
            splits = re.split(r"(\d.*)", op)
            if len(splits) >= 2:
                key, value = splits[:2]
                options[key] = value

    # if act_layer is None, the model default (passed to model init) will be used
    act_layer = options["n"] if "n" in options else None
    start_kernel_size = _parse_ksize(options["a"]) if "a" in options else 1
    end_kernel_size = _parse_ksize(options["p"]) if "p" in options else 1
    # FIXME hack to deal with in_chs issue in TPU def
    force_in_chs = int(options["fc"]) if "fc" in options else 0
    num_repeat = int(options["r"])

    # each type of block has different valid arguments, fill accordingly
    block_args = dict(
        block_type=block_type,
        out_channels=int(options["c"]),
        stride=int(options["s"]),
        act_layer=act_layer,
    )
    if block_type == "ir":
        block_args.update(
            dict(
                dw_kernel_size=_parse_ksize(options["k"]),
                exp_kernel_size=start_kernel_size,
                pw_kernel_size=end_kernel_size,
                exp_ratio=float(options["e"]),
                se_ratio=float(options.get("se", 0.0)),
                noskip=skip is False,
                s2d=int(options.get("d", 0)) > 0,
            )
        )
        if "cc" in options:
            block_args["num_experts"] = int(options["cc"])
    elif block_type == "ds" or block_type == "dsa":
        block_args.update(
            dict(
                dw_kernel_size=_parse_ksize(options["k"]),
                pw_kernel_size=end_kernel_size,
                se_ratio=float(options.get("se", 0.0)),
                pw_act=block_type == "dsa",
                noskip=block_type == "dsa" or skip is False,
                s2d=int(options.get("d", 0)) > 0,
            )
        )
    elif block_type == "er":
        block_args.update(
            dict(
                exp_kernel_size=_parse_ksize(options["k"]),
                pw_kernel_size=end_kernel_size,
                exp_ratio=float(options["e"]),
                force_in_chs=force_in_chs,
                se_ratio=float(options.get("se", 0.0)),
                noskip=skip is False,
            )
        )
    elif block_type == "cn":
        block_args.update(
            dict(
                kernel_size=int(options["k"]),
                skip=skip is True,
            )
        )
    elif block_type == "uir":
        # override exp / proj kernels for start/end in uir block
        start_kernel_size = _parse_ksize(options["a"]) if "a" in options else 0
        end_kernel_size = _parse_ksize(options["p"]) if "p" in options else 0
        block_args.update(
            dict(
                dw_kernel_size_start=start_kernel_size,  # overload exp ks arg for dw start
                dw_kernel_size_mid=_parse_ksize(options["k"]),
                dw_kernel_size_end=end_kernel_size,  # overload pw ks arg for dw end
                exp_ratio=float(options["e"]),
                se_ratio=float(options.get("se", 0.0)),
                noskip=skip is False,
            )
        )
    elif block_type == "mha":
        kv_dim = int(options["d"])
        block_args.update(
            dict(
                dw_kernel_size=_parse_ksize(options["k"]),
                num_heads=int(options["h"]),
                key_dim=kv_dim,
                value_dim=kv_dim,
                kv_stride=int(options.get("v", 1)),
                noskip=skip is False,
            )
        )
    elif block_type == "mqa":
        kv_dim = int(options["d"])
        block_args.update(
            dict(
                dw_kernel_size=_parse_ksize(options["k"]),
                num_heads=int(options["h"]),
                key_dim=kv_dim,
                value_dim=kv_dim,
                kv_stride=int(options.get("v", 1)),
                noskip=skip is False,
            )
        )
    else:
        assert False, "Unknown block type (%s)" % block_type

    if "gs" in options:
        block_args["group_size"] = int(options["gs"])

    if block_args["act_layer"] is not None:
        raise ValueError("not support act_layer config")
    else:
        del block_args["act_layer"]

    if "force_in_chs" in block_args:
        if block_args["force_in_chs"] > 0:
            raise ValueError("not support force_in_chs config")
        else:
            del block_args["force_in_chs"]

    if "se_ratio" in block_args:
        if block_args["se_ratio"] > 0:
            raise ValueError("not support se_ratio config")
        else:
            del block_args["se_ratio"]

    return [block_args for _ in range(num_repeat)]


def parse_arch_def(
    arch_def: List[List[str]],
    in_channels=3,
    stem_size=64,
):
    """
    Parse the full MNv5_BASE_ARCH_DEF into a nested list
    of block configs per stage.
    """

    def act():
        return ActModels.GELU()

    def norm():
        return NormModels.RmsNorm2d(num_features=-1)

    def to_model(parsed, in_channels_):
        res = []
        in_chs = in_channels_
        for block in parsed:
            block_type = block["block_type"]
            del block["block_type"]
            block["in_channels"] = in_chs
            if block_type == "er":
                b = Conv2dModels.EdgeResidual(
                    **block,
                    conv_pw_exp_layer=Conv2dModels.Conv2dPointwiseNormAct(
                        in_channels=-1,
                        norm=norm(),
                        act=act(),
                    ),
                    conv_pw_layer=Conv2dModels.Conv2dPointwiseNormAct(
                        in_channels=-1,
                        norm=norm(),
                        act=act(),
                    ),
                )
            elif block_type == "uir":
                b = UniversalInvertedResidual(
                    **block,
                    conv_dw_start_layer=Conv2dModels.Conv2dDepthwiseNorm(
                        in_channels=-1,
                        norm=norm(),
                    ),
                    conv_pw_exp_layer=Conv2dModels.Conv2dPointwiseNormAct(
                        in_channels=-1,
                        norm=norm(),
                        act=act(),
                    ),
                    conv_dw_mid_layer=Conv2dModels.Conv2dDepthwiseNormAct(
                        in_channels=-1,
                        norm=norm(),
                        act=act(),
                    ),
                    conv_pw_proj_layer=Conv2dModels.Conv2dPointwiseNorm(
                        in_channels=-1,
                        norm=norm(),
                    ),
                    conv_dw_end_layer=Conv2dModels.Conv2dDepthwiseNorm(
                        in_channels=-1,
                        norm=norm(),
                    ),
                )
            elif block_type == "mqa":
                b = Attention2dModels.MobileAttention(
                    **block,
                    use_multi_query=True,
                    norm_layer=norm(),
                    conv_cpe_layer=Conv2dModels.Conv2dDepthwise(in_channels=-1),
                )
            else:
                raise ValueError(f"Unsupported block_type: {block_type}")
            in_chs = b.out_channels
            res.append(b.build())
        return res, in_chs

    model = nn.Sequential()
    info = []
    in_channels = stem_size
    for i, stage in enumerate(arch_def):
        info.append(dict(stage=i, num_chs=in_channels))
        parsed_stage = []
        for block in [parse_block_def(b) for b in stage]:
            parsed_stage += block
        m, in_channels = to_model(parsed_stage, in_channels)
        blocks = nn.Sequential(*m)
        model.add_module(f"{i}", blocks)
    return model, info


# Example usage:
# parsed_arch = parse_arch_def(MNv5_300M_ARCH_DEF)
# print(parsed_arch)

# Now `parsed_arch` is a list[stage][block_cfg_dict]

model = MobileNetV5Backbone(
    arch_def=MNv5_TINY_ARCH_DEF,
    in_chans=3,
    stem_size=64,
    channel_multiplier=1.0,
    msfa_output_resolution=16,    # final H=W=16
)

# summ(model)
# summ(parse_arch_def(MNv5_TINY_ARCH_DEF)[0])
# x = torch.randn(1, 3, 64, 64)
# feat = model(x)
# print(feat.shape)
# feat = convert_to_ternary(model)(x)
# print(feat.shape)
# # info = summ(model,False)
# exit()


def make_mobilenetv5_teacher(size="300m",dataset=None,num_classes=None,device="cpu",pretrained=True):
    import timm
    model = timm.create_model('mobilenetv5_300m.gemma3n', pretrained=True)
    model = model.eval().to(device=device)
    return model


# ----------------------------
# LightningModule: KD + hints
# ----------------------------

class Config(CommonTrainConfig):
    dataset_name: Literal[
        "c10", "cifar10",
        "c100", "cifar100",
        "timnet", "tiny",
        "tinyimagenet", "tiny-imagenet",
        "imnet", "imagenet", "in1k", "imagenet1k",
    ] = Field(
        default="timnet",
        description="Target dataset (affects datamodule, num_classes, transforms).",
    )

    # For MobileNetV4 we accept conv + hybrid tags in one flag
    model_size: Literal[
        "tiny",
        "small",
        "medium",
        "large",
        "hybrid_medium",
        "hybrid_large",
    ] = Field(
        default="tiny",
        description="MobileNetV4 variant.",
    )

    drop_path: float = Field(
        default=0.0,
        description="Stochastic depth drop-path rate.",
    )

    teacher_pretrained: bool = Field(
        default=True,
        description=(
            "Use ImageNet-pretrained teacher backbone when classes != 1000 "
            "(head is replaced)."
        ),
    )
    
    batch_size:int=4
    lr:float=0.2
    alpha_kd:float=0.0
    alpha_hint:float=0.05

class LitMobileNetV5KD(LitBit):
    def __init__(
        self, config:LitBitConfig,
        drop_path_rate=0.0,
        teacher_pretrained=True
    ):
        # dataset -> classes
        ds = config.dataset_name.lower()
        if ds in ['c10', 'cifar10']:
            config.num_classes = 10
        elif ds in ['c100', 'cifar100']:
            config.num_classes = 100
        elif ds in ['timnet', 'tiny', 'tinyimagenet', 'tiny-imagenet']:
            config.num_classes = 200
        elif ds in ['imnet', 'imagenet', 'in1k', 'imagenet1k']:
            config.num_classes = 1000
        else:
            raise ValueError(f"Unsupported dataset: {config.dataset_name}")
        
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
        config.student = MobileNetV5Classifier(arch_def=get_mnv5_arch_def(config.model_size),
                                        num_classes=config.num_classes,
                                        drop_rate=drop_path_rate)

        config.teacher = make_mobilenetv5_teacher(
            size="300m",
            dataset=ds,
            num_classes=config.num_classes,
            device="cpu",
            pretrained=teacher_pretrained
        )

        # summ(student)
        # summ(teacher)
        # x = torch.randn(1, 3, 64, 64)
        # print(student(x).shape)
        # print(teacher(x).shape)

        # pick robust hint tap points via timm feature_info
        config.hint_points = [("blocks.0","blocks.0"), ("blocks.1","blocks.1"),
                       ("blocks.2","blocks.2"), ("blocks.3","blocks.3"),
                       ("msfa","msfa")]
        config.dataset_name=ds
        config.model_name='mobilenetv5'
        config.model_size=config.model_size
        config.num_classes=config.num_classes
        super().__init__(config)

# ----------------------------
# CLI / main (MobileNetV5)
# ----------------------------

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
    parser = ArgumentParser(model=Config)
    args:Config = parser.parse_typed_args()
    args.export_dir = f"./ckpt_{args.dataset_name}_mnv5_{args.model_size}"
    
    config = LitBitConfig.model_validate(args.model_dump())
    lit = LitMobileNetV5KD(config,
        drop_path_rate=args.drop_path,
        teacher_pretrained=args.teacher_pretrained
    )

    dmargs = DataModuleConfig.model_validate(args.model_dump())

    dm = _pick_datamodule_mnv5(args.dataset_name, dmargs.model_dump())

    trainer, dm = setup_trainer(args, lit, dm)
    trainer.fit(lit, datamodule=dm)
    trainer.validate(lit, datamodule=dm)


if __name__ == "__main__":
    main_mnv5()

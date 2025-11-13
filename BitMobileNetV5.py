from __future__ import annotations
from functools import partial
from typing import List, Sequence, Union, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.layers import ConvNormAct, RmsNorm2d
from timm.models._efficientnet_blocks import SqueezeExcite, UniversalInvertedResidual
from timm.models._efficientnet_builder import EfficientNetBuilder, decode_arch_def, round_channels

from common_utils import summ

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
        msfa_indices: Sequence[int] = (-2, -1),  # which features to fuse (in feature_info space)
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

        # ---- MSFA index + channel calc (THIS WAS THE BUGGY PART) ----
        # 1) choose which entries in feature_info we want to fuse
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
        msfa_indices: Sequence[int] = (-2, -1),
        msfa_output_resolution: int = 16,
        out_channels: int = 2048,
        drop_rate: float = 0.0,
    ):
        super().__init__()

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
            nn.Linear(out_channels, num_classes)
        )
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone.forward_features(x)  # [B, C, H, W]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encode(x))


MNv5_TINY_ARCH_DEF: list[list[str]] = [
    # Stage 0: 128x128 in
    [
        'er_r1_k3_s2_e4_c128',
        'er_r1_k3_s1_e4_c128',
    ],
    # Stage 1: 256x256 in
    [
        'uir_r1_a3_k5_s2_e6_c256',
        'uir_r1_a5_k0_s1_e4_c256',
    ],
    # Stage 2: 640x640 in
    [
        "uir_r1_a5_k5_s2_e6_c512",
        'mqa_r1_k3_h8_s2_d64_c512',
        "uir_r1_a0_k0_s1_e2_c512",
    ],
    # Stage 3: 1280x1280 in
    [
        "uir_r1_a5_k5_s2_e6_c1024",
        'mqa_r1_k3_h16_s1_d64_c1024',
        "uir_r1_a0_k0_s1_e2_c1024",
    ],
]

MNv5_SMALL_ARCH_DEF: list[list[str]] = [
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
    ],
    # Stage 2: 640x640 in
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

model = MobileNetV5Backbone(
    arch_def=MNv5_SMALL_ARCH_DEF,
    in_chans=3,
    stem_size=64,
    channel_multiplier=1.0,
    msfa_indices=(-2, -1),        # fuse last 2 stages
    msfa_output_resolution=16,    # final H=W=16
)

x = torch.randn(1, 3, 256, 256)
feat = model(x)
print(feat.shape)  # -> [1, 256, 16, 16]
summ(model)
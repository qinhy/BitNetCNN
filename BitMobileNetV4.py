# from https://raw.githubusercontent.com/jaiwei98/MobileNetV4-pytorch

import argparse
from typing import Literal, Optional
import warnings

from pydanticV2_argparse import ArgumentParser
import torch
import torch.nn as nn
import torch.nn.functional as F

from common_utils import *

def make_divisible(
        value: float,
        divisor: int,
        min_value: Optional[float] = None,
        round_down_protect: bool = True,
    ) -> int:
    """
    This function is copied from here 
    "https://github.com/tensorflow/models/blob/master/official/vision/modeling/layers/nn_layers.py"
    
    This is to ensure that all layers have channels that are divisible by 8.

    Args:
        value: A `float` of original value.
        divisor: An `int` of the divisor that need to be checked upon.
        min_value: A `float` of  minimum value threshold.
        round_down_protect: A `bool` indicating whether round down more than 10%
        will be allowed.

    Returns:
        The adjusted value in `int` that is divisible against divisor.
    """
    if min_value is None:
        min_value = divisor
    new_value = max(min_value, int(value + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if round_down_protect and new_value < 0.9 * value:
        new_value += divisor
    return int(new_value)

def conv_2d(inp, oup, kernel_size=3, stride=1, groups=1, bias=False, norm=True, act=True):
    conv = nn.Sequential()
    padding = (kernel_size - 1) // 2
    conv.add_module('conv', Bit.Conv2d(inp, oup, kernel_size, stride, padding, bias=bias, groups=groups))
    if norm:
        conv.add_module('BatchNorm2d', nn.BatchNorm2d(oup))
    if act:
        conv.add_module('Activation', nn.ReLU())
    return conv

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, act=False, squeeze_excitation=False):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]
        hidden_dim = int(round(inp * expand_ratio))
        self.block = nn.Sequential()
        if expand_ratio != 1:
            self.block.add_module('exp_1x1', conv_2d(inp, hidden_dim, kernel_size=3, stride=stride))
        if squeeze_excitation:
            self.block.add_module('conv_3x3', conv_2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, groups=hidden_dim))
        self.block.add_module('red_1x1', conv_2d(hidden_dim, oup, kernel_size=1, stride=1, act=act))
        self.use_res_connect = self.stride == 1 and inp == oup

    def forward(self, x):
        if self.use_res_connect:
            return x + self.block(x)
        else:
            return self.block(x)

class UniversalInvertedBottleneckBlock(nn.Module):
    def __init__(self,
            inp,
            oup,
            start_dw_kernel_size,
            middle_dw_kernel_size,
            middle_dw_downsample,
            stride,
            expand_ratio,
            use_layer_scale=False
        ):
        """An inverted bottleneck block with optional depthwises.
        Referenced from here https://github.com/tensorflow/models/blob/master/official/vision/modeling/layers/nn_blocks.py
        """
        super().__init__()
        self.use_res_connect = (inp == oup and stride == 1)

        # Starting depthwise conv.
        self.start_dw_kernel_size = start_dw_kernel_size
        if self.start_dw_kernel_size:            
            stride_ = stride if not middle_dw_downsample else 1
            self._start_dw_ = conv_2d(inp, inp, kernel_size=start_dw_kernel_size, stride=stride_, groups=inp, act=False)
        # Expansion with 1x1 convs.
        expand_filters = make_divisible(inp * expand_ratio, 8)
        self._expand_conv = conv_2d(inp, expand_filters, kernel_size=1)
        # Middle depthwise conv.
        self.middle_dw_kernel_size = middle_dw_kernel_size
        if self.middle_dw_kernel_size:
            stride_ = stride if middle_dw_downsample else 1
            self._middle_dw = conv_2d(expand_filters, expand_filters, kernel_size=middle_dw_kernel_size, stride=stride_, groups=expand_filters)
        # Projection with 1x1 convs.
        self._proj_conv = conv_2d(expand_filters, oup, kernel_size=1, stride=1, act=False)
        
        # Ending depthwise conv.
        # this not used
        # _end_dw_kernel_size = 0
        # self._end_dw = conv_2d(oup, oup, kernel_size=_end_dw_kernel_size, stride=stride, groups=inp, act=False)
       
        self._use_layer_scale = use_layer_scale
        if self._use_layer_scale:
            self.layer_scale_init_value = 1e-5
            self.layer_scale = MNV4LayerScale(oup, self.layer_scale_init_value)

    def forward(self, x):
        shortcut = x
        if self.start_dw_kernel_size:
            x = self._start_dw_(x)
            # print("_start_dw_", x.shape)
        x = self._expand_conv(x)
        # print("_expand_conv", x.shape)
        if self.middle_dw_kernel_size:
            x = self._middle_dw(x)
            # print("_middle_dw", x.shape)
        x = self._proj_conv(x)
        # print("_proj_conv", x.shape)
        if self._use_layer_scale:
            x = self.layer_scale(x)
        if self.use_res_connect:
            x = x + shortcut
        return x

class MultiQueryAttentionLayerWithDownSampling(nn.Module):
    """
    Multi-Query Attention (shared K,V across heads) with optional spatial downsampling.
    """
    def __init__(
        self,
        inp: int,
        num_heads: int,
        key_dim: int,
        value_dim: int,
        query_h_strides: int = 1,
        query_w_strides: int = 1,
        kv_strides: int = 1,
        dw_kernel_size: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert key_dim > 0 and value_dim > 0
        assert num_heads > 0

        self.inp = inp
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.query_h_strides = query_h_strides
        self.query_w_strides = query_w_strides
        self.kv_strides = kv_strides

        self.query_downsample_bn = (
            nn.BatchNorm2d(inp)
            if (query_h_strides > 1 or query_w_strides > 1)
            else None
        )

        # Q: 1x1 projection to (H * key_dim)
        self.q_proj = Bit.Conv2d(inp, num_heads * key_dim, kernel_size=1, bias=False)

        def make_kv_path(out_dim: int) -> nn.Sequential:
            layers = []
            if kv_strides > 1:
                layers.extend(
                    [
                        Bit.Conv2d(
                            inp,
                            inp,
                            kernel_size=dw_kernel_size,
                            stride=kv_strides,
                            padding=dw_kernel_size // 2,
                            groups=inp,
                            bias=False,
                        ),
                        nn.BatchNorm2d(inp),
                    ]
                )
            layers.append(Bit.Conv2d(inp, out_dim, kernel_size=1, bias=False))
            return nn.Sequential(*layers)

        self.k_proj = make_kv_path(key_dim)
        self.v_proj = make_kv_path(value_dim)

        self.out_proj = Bit.Conv2d(num_heads * value_dim, inp, kernel_size=1, bias=False)
        self.attn_drop = nn.Dropout(dropout)

    def _reshape_q(self, q: torch.Tensor):
        # q: [B, H*Kd, Hq, Wq] -> [B, Nq, H, Kd]
        B, HC, Hq, Wq = q.shape
        H = self.num_heads
        Kd = self.key_dim
        q = q.view(B, H, Kd, Hq * Wq).transpose(-1, -2).contiguous()
        return q, Hq, Wq

    @staticmethod
    def _reshape_kv(t: torch.Tensor):
        # t: [B, C, Hk, Wk] -> [B, 1, P, C]
        B, C, Hk, Wk = t.shape
        return t.view(B, C, Hk * Wk).transpose(1, 2).unsqueeze(1).contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        # Query (optional spatial downsampling)
        if self.query_h_strides > 1 or self.query_w_strides > 1:
            q_in = F.avg_pool2d(
                x,
                kernel_size=(self.query_h_strides, self.query_w_strides),
                stride=(self.query_h_strides, self.query_w_strides),
            )
            if self.query_downsample_bn is not None:
                q_in = self.query_downsample_bn(q_in)
        else:
            q_in = x
        q = self.q_proj(q_in)
        q, Hq, Wq = self._reshape_q(q)  # [B, Nq, H, Kd]

        # Keys/Values
        k = self._reshape_kv(self.k_proj(x))  # [B, 1, P, Kd]
        v = self._reshape_kv(self.v_proj(x))  # [B, 1, P, Vd]

        # Attention
        q = q * (self.key_dim ** -0.5)
        attn = q @ k.transpose(-1, -2)  # [B, Nq, H, P]
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        out = attn @ v  # [B, Nq, H, Vd]

        # Merge heads and restore spatial
        out = out.transpose(1, 2).contiguous()  # [B, H, Nq, Vd]
        out = out.view(B, self.num_heads * self.value_dim, Hq, Wq)
        out = self.out_proj(out)  # [B, C, Hq, Wq]
        return out

class MNV4LayerScale(nn.Module):
    def __init__(self, dim: int, init_values: float = 1e-5):
        """LayerScale as introduced in CaiT: https://arxiv.org/abs/2103.17239
        Referenced from here https://github.com/tensorflow/models/blob/master/official/vision/modeling/layers/nn_blocks.py
       
        As used in MobileNetV4.

        Attributes:
            init_value (float): value to initialize the diagonal matrix of LayerScale.
        """
        super().__init__()
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        gamma = self.gamma.view(1, -1, 1, 1)
        return x * gamma

class MultiHeadSelfAttentionBlock(nn.Module):
    def __init__(
            self,
            inp,
            num_heads,
            key_dim,  
            value_dim,
            query_h_strides,
            query_w_strides,
            kv_strides,
            use_layer_scale,
            use_multi_query,
            use_residual = True
        ):
        super().__init__()
        self.query_h_strides = query_h_strides
        self.query_w_strides = query_w_strides
        self.kv_strides = kv_strides
        self.use_layer_scale = use_layer_scale
        self.use_multi_query = use_multi_query
        self.use_residual = use_residual

        self._input_norm = nn.BatchNorm2d(inp)
        if self.use_multi_query:
            self.multi_query_attention = MultiQueryAttentionLayerWithDownSampling(
                inp, num_heads, key_dim, value_dim, query_h_strides, query_w_strides, kv_strides
            )
        else:
            self.multi_head_attention = nn.MultiheadAttention(inp, num_heads, kdim=key_dim)
       
        if self.use_layer_scale:
            self.layer_scale_init_value = 1e-5
            self.layer_scale = MNV4LayerScale(inp, self.layer_scale_init_value)
   
    def forward(self, x):
        # Not using CPE, skipped
        # input norm
        shortcut = x
        x = self._input_norm(x)
        # multi query
        if self.use_multi_query:
            x = self.multi_query_attention(x)
        else:
            x = self.multi_head_attention(x, x)
        # layer scale
        if self.use_layer_scale:
            x = self.layer_scale(x)
        # use residual
        if self.use_residual:
            x = x + shortcut
        return x
    
class MobileNetV4Head(nn.Module):
    """
    Flexible head for MobileNet-V4.
    - pool: 'avg' (default), 'max', or 'avgmax' (concat avg & max)
    - use_bn: add BatchNorm1d before linear
    - act: 'relu' | 'gelu' | 'hswish' | None
    """
    def __init__(
        self,
        in_ch: int,
        num_classes: int,
        pool: Literal['avg','max','avgmax'] = 'avg',
        dropout: float = 0.0,
        use_bn: bool = False,
        act: Optional[str] = None
    ):
        super().__init__()
        self.pool_kind = pool
        if pool == 'avg':
            self.pool = nn.AdaptiveAvgPool2d(1)
            out_ch = in_ch
        elif pool == 'max':
            self.pool = nn.AdaptiveMaxPool2d(1)
            out_ch = in_ch
        elif pool == 'avgmax':
            self.pool_avg = nn.AdaptiveAvgPool2d(1)
            self.pool_max = nn.AdaptiveMaxPool2d(1)
            self.pool = None
            out_ch = in_ch * 2
        else:
            raise ValueError(f"Unknown pool: {pool}")

        self.flatten = nn.Flatten(1)

        self.bn = nn.BatchNorm1d(out_ch) if use_bn else nn.Identity()

        if act is None:
            self.act = nn.Identity()
        elif act.lower() == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif act.lower() == 'gelu':
            self.act = nn.GELU()
        elif act.lower() == 'hswish':
            self.act = nn.Hardswish()
        else:
            raise ValueError(f"Unsupported act: {act}")

        self.drop = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        self.fc = Bit.Linear(out_ch, num_classes)

    @torch.no_grad()
    def feature_dim(self) -> int:
        return self.fc.in_features

    def forward(self, x, return_features: bool = False):
        # x: (N, C, H, W) features from backbone
        if self.pool_kind == 'avgmax':
            xa = self.pool_avg(x)
            xm = self.pool_max(x)
            x = torch.cat([xa, xm], dim=1)
        else:
            x = self.pool(x)
        x = self.flatten(x)      # (N, F)
        f = self.bn(x)
        f = self.act(f)
        f = self.drop(f)
        logits = self.fc(f)
        return (logits, f) if return_features else logits

def build_blocks(layer_spec):
    if not layer_spec.get('block_name'):
        return nn.Sequential()
    block_names = layer_spec['block_name']
    layers = nn.Sequential()
    if block_names == "convbn":
        schema_ = ['inp', 'oup', 'kernel_size', 'stride']
        for i in range(layer_spec['num_blocks']):
            specs_ = layer_spec['block_specs'][i]
            if isinstance(specs_, list):
                args = dict(zip(schema_, layer_spec['block_specs'][i]))
                layers.add_module(f"convbn_{i}", conv_2d(**args))
            elif specs_ == "AdaptiveAvgPool2d":
                layers.add_module(f"AdaptiveAvgPool2d", nn.AdaptiveAvgPool2d(1))
            else:
                raise NotImplementedError
    elif block_names == "uib":
        schema_ =  ['inp', 'oup', 'start_dw_kernel_size', 'middle_dw_kernel_size', 'middle_dw_downsample', 'stride', 'expand_ratio', 'use_layer_scale', 'mhsa']
        for i in range(layer_spec['num_blocks']):
            args = dict(zip(schema_, layer_spec['block_specs'][i]))
            mhsa = args.pop("mhsa") if "mhsa" in args else 0
            layers.add_module(f"uib_{i}", UniversalInvertedBottleneckBlock(**args))
            if mhsa:
                mhsa_schema_ = [
                    "inp", "num_heads", "key_dim", "value_dim", "query_h_strides", "query_w_strides", "kv_strides", 
                    "use_layer_scale", "use_multi_query", "use_residual"
                ]
                args = dict(zip(mhsa_schema_, [args['oup']] + (mhsa)))
                layers.add_module(f"mhsa_{i}", MultiHeadSelfAttentionBlock(**args))
    elif block_names == "fused_ib":
        schema_ = ['inp', 'oup', 'stride', 'expand_ratio', 'act']
        for i in range(layer_spec['num_blocks']):
            args = dict(zip(schema_, layer_spec['block_specs'][i]))
            layers.add_module(f"fused_ib_{i}", InvertedResidual(**args))
    else:
        raise NotImplementedError
    return layers

class MobileNetV4(nn.Module):
    
    POOL = "AdaptiveAvgPool2d"

    def conv(in_c, out_c, k, s):
        return [in_c, out_c, k, s]

    def fused(in_c, out_c, stride, expand, flag=False, extra=None):
        # 5-field (…flag) or 6-field (…flag, extra)
        return [in_c, out_c, stride, expand, flag] if extra is None \
            else [in_c, out_c, stride, expand, flag, extra]

    def MHSA(num_heads, key_dim, value_dim, px):
        kv_strides = 2 if px == 24 else 1 if px == 12 else 1
        # [heads, kdim, vdim, q_h_s, q_w_s, kv_s, use_layer_scale, use_multi_query, use_residual]
        return [num_heads, key_dim, value_dim, 1, 1, kv_strides, True, True, True]

    def uib(in_c, out_c, k1, k2, se, stride, e, shortcut=False, mhsa=None):
        base = [in_c, out_c, k1, k2, se, stride, e, shortcut]
        return base if mhsa is None else base + [mhsa]

    def stage(block_name, *blocks):
        flat = []
        for b in blocks:
            if isinstance(b, list) and b and isinstance(b[0], list):
                flat.extend(b)     # already a list of specs
            else:
                flat.append(b)     # single spec
        return {"block_name": block_name, "num_blocks": len(flat), "block_specs": flat}

    def repeat(n, spec):
        # Deep-ish copy not required for immutable atoms; list() to avoid alias surprises.
        return [list(spec) for _ in range(n)]

    # ---- Specs (DRY) ------------------------------------------------------------

    MNV4ConvSmall = {
        "conv0":  stage("convbn", conv(3, 32, 3, 2)),
        "layer1": stage("convbn",
                        conv(32, 32, 3, 2),
                        conv(32, 32, 1, 1)),
        "layer2": stage("convbn",
                        conv(32, 96, 3, 2),
                        conv(96,  64, 1, 1)),
        "layer3": stage("uib",
                        uib(64, 96, 5, 5, True, 2, 3, False),
                        repeat(4, uib(96, 96, 0, 3, True, 1, 2, False)),
                        uib(96, 96, 3, 0, True, 1, 4, False)),
        "layer4": stage("uib",
                        uib(96,  128, 3, 3, True, 2, 6, False),
                        uib(128, 128, 5, 5, True, 1, 4, False),
                        uib(128, 128, 0, 5, True, 1, 4, False),
                        uib(128, 128, 0, 5, True, 1, 3, False),
                        repeat(2, uib(128, 128, 0, 3, True, 1, 4, False))),
        "layer5": stage("convbn",
                        conv(128, 960, 1, 1),
                        POOL,
                        conv(960, 1280, 1, 1)),
    }

    MNV4ConvMedium = {
        "conv0":  stage("convbn", conv(3, 32, 3, 2)),
        "layer1": stage("fused_ib", fused(32, 48, 2, 4.0, False)),
        "layer2": stage("uib",
                        uib(48, 80, 3, 5, True, 2, 4, False),
                        uib(80, 80, 3, 3, True, 1, 2, False)),
        "layer3": stage("uib",
                        uib(80, 160, 3, 5, True, 2, 6, False),
                        repeat(2, uib(160, 160, 3, 3, True, 1, 4, False)),
                        uib(160, 160, 3, 5, True, 1, 4, False),
                        uib(160, 160, 3, 3, True, 1, 4, False),
                        uib(160, 160, 3, 0, True, 1, 4, False),
                        uib(160, 160, 0, 0, True, 1, 2, False),
                        uib(160, 160, 3, 0, True, 1, 4, False)),
        "layer4": stage("uib",
                        uib(160, 256, 5, 5, True, 2, 6, False),
                        uib(256, 256, 5, 5, True, 1, 4, False),
                        repeat(2, uib(256, 256, 3, 5, True, 1, 4, False)),
                        uib(256, 256, 0, 0, True, 1, 4, False),
                        uib(256, 256, 3, 0, True, 1, 4, False),
                        uib(256, 256, 3, 5, True, 1, 2, False),
                        uib(256, 256, 5, 5, True, 1, 4, False),
                        repeat(2, uib(256, 256, 0, 0, True, 1, 4, False)),
                        uib(256, 256, 5, 0, True, 1, 2, False)),
        "layer5": stage("convbn",
                        conv(256, 960, 1, 1),
                        POOL,
                        conv(960, 1280, 1, 1)),
    }

    MNV4ConvLarge = {
        "conv0":  stage("convbn", conv(3, 24, 3, 2)),
        "layer1": stage("fused_ib", fused(24, 48, 2, 4.0, False)),
        "layer2": stage("uib",   # FIX: ensure 8 fields (…shortcut=False)
                        uib(48, 96, 3, 5, True, 2, 4, False),
                        uib(96, 96, 3, 3, True, 1, 4, False)),
        "layer3": stage("uib",
                        uib(96, 192, 3, 5, True, 2, 4, False),
                        repeat(3, uib(192, 192, 3, 3, True, 1, 4, False)),
                        uib(192, 192, 3, 5, True, 1, 4, False),
                        repeat(5, uib(192, 192, 5, 3, True, 1, 4, False)),
                        uib(192, 192, 3, 0, True, 1, 4, False)),
        "layer4": stage("uib",
                        uib(192, 512, 5, 5, True, 2, 4, False),
                        repeat(3, uib(512, 512, 5, 5, True, 1, 4, False)),
                        uib(512, 512, 5, 0, True, 1, 4, False),
                        uib(512, 512, 5, 3, True, 1, 4, False),
                        repeat(2, uib(512, 512, 5, 0, True, 1, 4, False)),
                        uib(512, 512, 5, 3, True, 1, 4, False),
                        uib(512, 512, 5, 5, True, 1, 4, False),
                        repeat(3, uib(512, 512, 5, 0, True, 1, 4, False))),
        "layer5": stage("convbn",
                        conv(512, 960, 1, 1),
                        POOL,
                        conv(960, 1280, 1, 1)),
    }

    MNV4HybridConvMedium = {
        "conv0":  stage("convbn", conv(3, 32, 3, 2)),
        "layer1": stage("fused_ib", fused(32, 48, 2, 4.0, False)),
        "layer2": stage("uib",
                        uib(48, 80, 3, 5, True, 2, 4, True),
                        uib(80, 80, 3, 3, True, 1, 2, True)),
        "layer3": stage("uib",
                        uib(80, 160, 3, 5, True, 2, 6, True),
                        uib(160, 160, 0, 0, True, 1, 2, True),
                        uib(160, 160, 3, 3, True, 1, 4, True),
                        uib(160, 160, 3, 5, True, 1, 4, True, MHSA(4, 64, 64, 24)),
                        uib(160, 160, 3, 3, True, 1, 4, True, MHSA(4, 64, 64, 24)),
                        uib(160, 160, 3, 0, True, 1, 4, True, MHSA(4, 64, 64, 24)),
                        uib(160, 160, 3, 3, True, 1, 4, True, MHSA(4, 64, 64, 24)),
                        uib(160, 160, 3, 0, True, 1, 4, True)),
        "layer4": stage("uib",
                        uib(160, 256, 5, 5, True, 2, 6, True),
                        uib(256, 256, 5, 5, True, 1, 4, True),
                        uib(256, 256, 3, 5, True, 1, 4, True),
                        uib(256, 256, 3, 5, True, 1, 4, True),
                        uib(256, 256, 0, 0, True, 1, 2, True),
                        uib(256, 256, 3, 5, True, 1, 2, True),
                        uib(256, 256, 0, 0, True, 1, 2, True),
                        uib(256, 256, 0, 0, True, 1, 4, True, MHSA(4, 64, 64, 12)),
                        uib(256, 256, 3, 0, True, 1, 4, True, MHSA(4, 64, 64, 12)),
                        uib(256, 256, 5, 5, True, 1, 4, True, MHSA(4, 64, 64, 12)),
                        uib(256, 256, 5, 0, True, 1, 4, True, MHSA(4, 64, 64, 12)),
                        uib(256, 256, 5, 0, True, 1, 4, True)),
        "layer5": stage("convbn",
                        conv(256, 960, 1, 1),
                        POOL,
                        conv(960, 1280, 1, 1)),
    }

    MNV4HybridConvLarge = {
        "conv0":  stage("convbn", conv(3, 24, 3, 2)),
        "layer1": stage("fused_ib", fused(24, 48, 2, 4.0, False, True)),
        "layer2": stage("uib",
                        uib(48, 96, 3, 5, True, 2, 4, True),
                        uib(96, 96, 3, 3, True, 1, 4, True)),
        "layer3": stage("uib",
                        uib(96, 192, 3, 5, True, 2, 4, True),
                        repeat(3, uib(192, 192, 3, 3, True, 1, 4, True)),
                        uib(192, 192, 3, 5, True, 1, 4, True),
                        repeat(2, uib(192, 192, 5, 3, True, 1, 4, True)),
                        repeat(4, uib(192, 192, 5, 3, True, 1, 4, True, MHSA(8, 48, 48, 24))),
                        uib(192, 192, 3, 0, True, 1, 4, True)),
        "layer4": stage("uib",
                        uib(192, 512, 5, 5, True, 2, 4, True),
                        repeat(3, uib(512, 512, 5, 5, True, 1, 4, True)),
                        uib(512, 512, 5, 0, True, 1, 4, True),
                        uib(512, 512, 5, 3, True, 1, 4, True),
                        repeat(2, uib(512, 512, 5, 0, True, 1, 4, True)),
                        uib(512, 512, 5, 3, True, 1, 4, True),
                        uib(512, 512, 5, 5, True, 1, 4, True, MHSA(8, 64, 64, 12)),
                        repeat(3, uib(512, 512, 5, 0, True, 1, 4, True, MHSA(8, 64, 64, 12))),
                        uib(512, 512, 5, 0, True, 1, 4, True)),
        "layer5": stage("convbn",
                        conv(512, 960, 1, 1),
                        POOL,
                        conv(960, 1280, 1, 1)),
    }

    MODEL_SPECS = {
        "MobileNetV4ConvSmall": MNV4ConvSmall,
        "MobileNetV4ConvMedium": MNV4ConvMedium,
        "MobileNetV4ConvLarge": MNV4ConvLarge,
        "MobileNetV4HybridMedium": MNV4HybridConvMedium,
        "MobileNetV4HybridLarge": MNV4HybridConvLarge
    }

    def __init__(self, model_name, num_classes):
        # MobileNetV4ConvSmall  MobileNetV4ConvMedium  MobileNetV4ConvLarge
        # MobileNetV4HybridMedium  MobileNetV4HybridLarge
        """Params to initiate MobilenNetV4
        Args:
            model : support 5 types of models as indicated in 
            "https://github.com/tensorflow/models/blob/master/official/vision/modeling/backbones/mobilenet.py"        
        """
        super().__init__()
        model_name = {
            "MobileNetV4ConvSmall":"MobileNetV4ConvSmall",
            "small":"MobileNetV4ConvSmall",
            "MobileNetV4ConvMedium":"MobileNetV4ConvMedium",
            "medium":"MobileNetV4ConvMedium",
            "MobileNetV4ConvLarge":"MobileNetV4ConvLarge",
            "large":"MobileNetV4ConvLarge",
            "MobileNetV4HybridMedium":"MobileNetV4HybridMedium",
            "hybrid_medium":"MobileNetV4HybridMedium",
            "MobileNetV4HybridLarge":"MobileNetV4HybridLarge",
            "hybrid_large":"MobileNetV4HybridLarge",
        }[model_name]
        assert model_name in MobileNetV4.MODEL_SPECS.keys()
        self.model_name = model_name
        self.num_classes = num_classes
        self.spec = MobileNetV4.MODEL_SPECS[self.model_name]
       
        # conv0
        self.conv0 = build_blocks(self.spec['conv0'])
        # layer1
        self.layer1 = build_blocks(self.spec['layer1'])
        # layer2
        self.layer2 = build_blocks(self.spec['layer2'])
        # layer3
        self.layer3 = build_blocks(self.spec['layer3'])
        # layer4
        self.layer4 = build_blocks(self.spec['layer4'])
        # layer5   
        self.layer5 = build_blocks(self.spec['layer5'])

        # print("Check output shape ...")
        x = torch.rand(2, 3, 224, 224)
        y = self.feature(x)[-1]
        self.head = MobileNetV4Head(y.shape[1],num_classes=num_classes)
               
    def feature(self, x):
        x0 = self.conv0(x)
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x5 = self.layer5(x4)
        return [x1, x2, x3, x4, x5]
        # return [x0, x1, x2, x3, x5]

    def forward(self, x):
        return self.head(self.feature(x)[-1])

    def clone(self) -> "MobileNetV4":
        return self.__class__(
                self.model_name,
                self.num_classes)

# model = MobileNetV4("MobileNetV4HybridMedium",200)
# # Check the trainable params
# total_params = sum(p.numel() for p in model.parameters())
# print(f"Number of parameters: {total_params}")
# # Check the model's output shape
# print("Check output shape ...")
# x = torch.rand(2, 3, 224, 224)
# y = model.feature(x)
# for i in y: print(i.shape)

# ----------------------------
# MobileNetV4: builders + KD
# ----------------------------
def _parse_mnv4_tag(model_size: str):
    """
    Accepts: 'small', 'medium', 'large', 'hybrid_medium', 'hybrid_large'
             (and typo alias: 'hybrid_medium')
    Returns: family ('conv'|'hybrid'), size ('small'|'medium'|'large'), arch_tag string
    """
    s = (model_size or "").lower().strip()
    if s.startswith("hybrid_"):
        fam, sz = "hybrid", s.split("_", 1)[1]
    elif s in {"hybrid_medium", "hybrid_med", "hybrid_mid"}:
        fam, sz = "hybrid", "medium"
    else:
        fam, sz = ("hybrid", "medium") if s in {"midum", "med", "mid"} else ("conv", s)

    # normalize canonical sizes
    if sz not in {"small", "medium", "large"}:
        raise ValueError(f"Unsupported MobileNetV4 size '{sz}'. Use small|medium|large or hybrid_* variants.")
    arch_tag = f"mobilenetv4_{fam}_{sz}"
    return fam, sz, arch_tag


# ---------- timm model builders (student + teacher) ----------
def make_mobilenetv4_from_timm(
    model_size: str = "small",
    device: str = "cuda",
    pretrained: bool = True,
    model_name: str | None = None
):
    """
    Build a timm MobileNetV4 model. If model_name is provided, it wins.
    Otherwise we map (conv|hybrid, size) -> a reasonable default HF weight;
    fall back to bare arch when needed.
    """
    import timm
    fam, sz, arch_tag = _parse_mnv4_tag(model_size)

    if model_name is None:
        default_name_map = {
            ("conv", "small"):   "mobilenetv4_conv_small.e1200_r224_in1k",
            ("conv", "medium"):  "mobilenetv4_conv_medium.e500_r256_in1k",
            ("conv", "large"):   "mobilenetv4_conv_large.e600_r384_in1k",
            ("hybrid", "medium"): "mobilenetv4_hybrid_medium.ix_e550_r256_in1k",
            ("hybrid", "large"):  "mobilenetv4_hybrid_large.ix_e600_r384_in1k",
        }
        model_name = default_name_map.get((fam, sz), arch_tag)
        if pretrained and model_name == arch_tag:
            warnings.warn(
                f"No default pretrained weights mapped for '{arch_tag}'. "
                "Building architecture without pretrained weights."
            )

    try:
        print("create teacher model", model_name, pretrained)
        m = timm.create_model(model_name, pretrained=pretrained)
    except Exception:
        # try explicit HF hub path
        try:
            m = timm.create_model(f"hf-hub:timm/{model_name}", pretrained=pretrained)
        except Exception as e:
            if pretrained and "." in model_name:
                warnings.warn(
                    f"Could not load pretrained weights '{model_name}'. "
                    f"Falling back to bare arch '{arch_tag}'. Error: {e}"
                )
            m = timm.create_model(arch_tag, pretrained=False)

    return m.eval().to(device)


def make_mobilenetv4_teacher_for_dataset(
    size: str,
    dataset: str,
    num_classes: int,
    device: str = "cpu",
    pretrained: bool = True,
    model_name: str | None = None,
):
    """
    Teacher = MobileNetV4 with IN1K weights if available. If dataset classes != head,
    we replace the classifier via timm's reset_classifier (or manual).
    """
    import timm

    t = make_mobilenetv4_from_timm(
        model_size=size, device=device, pretrained=pretrained
    )

    head_out = getattr(t, "num_classes", None)
    if head_out is None:
        getc = getattr(t, "get_classifier", None)
        if callable(getc):
            cls = getc()
            head_out = getattr(cls, "out_features", None)
    head_out = head_out or 1000

    if head_out != num_classes:     
        mapping = load_tiny200_to_in1k_map("timnet_to_imagenet1k_indices.txt")
        adapter = IN1kToTiny200Adapter(mapping, temperature=2.0, renormalize=True).to("cuda")
        tt = nn.Sequential(
            t,
            adapter,
        )
        tt.num_classes = num_classes
        return tt.eval().to(device)
    
    return t.eval().to(device)

# ----------------------------
# LightningModule: KD + hints
# ----------------------------
class LitMobileNetV4KD(LitBit):
    def __init__(
        self,
        lr, wd, epochs,
        dataset_name='c100',
        model_size="small",            # 'small'|'medium'|'large' or 'hybrid_medium'|'hybrid_large' (alias: hybrid_medium)
        label_smoothing=0.1, alpha_kd=0.0, alpha_hint=0.0, T=4.0,
        amp=True, export_dir="./ckpt_mnv4",
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

        # student & teacher
        student = MobileNetV4(model_size, num_classes=num_classes)

        teacher = make_mobilenetv4_teacher_for_dataset(
            size=model_size,
            dataset=ds,
            num_classes=num_classes,
            device="cpu",
            pretrained=teacher_pretrained
        )

        # summ(student)
        # summ(teacher)

        # pick robust hint tap points via timm feature_info
        hint_points = [("layer1","blocks.0"), ("layer2","blocks.1"), ("layer3","blocks.2"), ("layer4","blocks.3")]

        super().__init__(
            lr, wd, epochs, label_smoothing,
            alpha_kd, alpha_hint, T,
            amp,
            export_dir,
            dataset_name=ds,
            model_name='mobilenetv4',
            model_size=model_size,
            hint_points=hint_points,
            student=student,
            teacher=teacher,
            num_classes=num_classes
        )

# ----------------------------
# CLI / main (MobileNetV4)
# ----------------------------
class Config(CommonTrainConfig):
    dataset_nam: str = "timnet"
    # For MobileNetV4 we accept conv + hybrid tags in one flag
    model_size: Literal[
        "small",
        "medium",
        "large",
        "hybrid_medium",
        "hybrid_large",
    ] = Field(
        default="hybrid_medium",
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
    
    out:Optional[str]=None
    batch_size:int=512
    lr:float=0.2
    alpha_kd:float=0.0
    alpha_hint:float=0.0005
    
def _pick_datamodule_mnv4(dataset_name: str, dmargs: dict):
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


def main_mnv4():
    parser = ArgumentParser(model=Config)
    args = parser.parse_typed_args()
    args.export_dir = f"./ckpt_{args.dataset_nam}_mnv4_{args.model_size}"

    # Derive num_classes for export dir naming (same as your convnext main)
    ds = args.dataset_nam.lower()
    if ds in ['c10', 'cifar10']:
        ncls = 10
    elif ds in ['c100', 'cifar100']:
        ncls = 100
    elif ds in ['timnet', 'tiny', 'tinyimagenet', 'tiny-imagenet']:
        ncls = 200
    elif ds in ['imnet', 'imagenet', 'in1k', 'imagenet1k']:
        ncls = 1000
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset_nam}")

    out_dir = f"{args.out}_{ds}_{args.model_size}_{ncls}c"

    lit = LitMobileNetV4KD(
        **args.model_dump(),
        drop_path_rate=args.drop_path,
        teacher_pretrained=args.teacher_pretrained
    )

    dmargs = dict(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=4,
        mixup=args.mixup,
        cutmix=args.cutmix,
        mix_alpha=args.mix_alpha
    )
    dm = _pick_datamodule_mnv4(args.dataset_nam, dmargs)

    trainer, dm = setup_trainer(args, lit, dm)
    trainer.fit(lit, datamodule=dm)
    trainer.validate(lit, datamodule=dm)


if __name__ == "__main__":
    main_mnv4()

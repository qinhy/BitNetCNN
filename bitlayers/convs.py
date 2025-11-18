from __future__ import annotations

import json
from typing import Any, Dict, Optional, Sequence, Tuple, Union

from pydantic import BaseModel, ConfigDict
from torch import nn
import torch

from .bit import Bit
from .padding import PadSame
from timmlayers.mixed_conv2d import MixedConv2d as _MixedConv2d
from timmlayers.separable_conv import SeparableConv2d as _SeparableConv2d
from timmlayers.split_batchnorm import SplitBatchNorm2d as _SplitBatchNorm2d
from timmlayers.std_conv import (
    ScaledStdConv2d as _ScaledStdConv2d,
    ScaledStdConv2dSame as _ScaledStdConv2dSame,
    StdConv2d as _StdConv2d,
    StdConv2dSame as _StdConv2dSame,
)

IntOrPair = Union[int, Tuple[int, int]]
KernelSizeArg = Union[int, Tuple[int, int], Sequence[int]]
PadArg = Union[str, int, Tuple[int, int]]

class Conv2dModels:
    """Lightweight Pydantic wrappers for key convolutional primitives."""

    class BasicModel(BaseModel):

        def build(self):
            mod = Conv2dControllers
            return mod.__dict__[f'{self.__class__.__name__}Controller'](self)
        
        @classmethod
        def shortcut_prefix_fields(cls,exclude:Dict[str,str]={},le=1,alow=['integer','boolean']):
            ts = cls.model_json_schema()['properties'].items()
            ts = [([v.get('type')] or [i['type'] for i in v['anyOf']]) for k,v in ts]
            ts = [set(i).intersection(set(alow)) for i in ts]

            ms = [i for i,j in zip(cls.model_fields.keys(),ts) if len(i)>0]
            ks = {i:i[:le] for i in ms if i not in exclude}
            if len(ks)!=len(ms):
                raise ValueError(f'short keys is duplicat, {ks} <=> {ms}')
            ks.update(exclude)
            return {v:k for k,v in ks.items() if v is not None}

        @classmethod
        def parse_shortcut_kwargs(cls, cmd: str, prefix: str):
            if not cmd.startswith(prefix): return None
            shortcut_prefix_fields = cls.shortcut_prefix_fields()
            
    class Conv2d(BasicModel):
        in_channels: int
        out_channels: int
        kernel_size: IntOrPair = 3
        stride: IntOrPair = 1
        padding: PadArg = 0
        dilation: IntOrPair = 1
        groups: int = 1
        bias: bool = True
        bit: bool = True
        padding_mode: str = 'zeros'
        scale_op: str ="median"

        # ---------- main entry point users call ----------
        @classmethod
        def shortcut_prefix_fields(cls,exclude:Dict[str,str]={
            'bias':'bs',
            'bit':'bit',
            'padding_mode':None,
            'scale_op':None
            }):
            return super().shortcut_prefix_fields(exclude)
        
        @classmethod
        def shortcut(cls,
            cmd: str = "conv_i3_o32_k3_s1_p1_d1_bs_bit",
            prefix: str = "conv"
        ):
            if not cmd.startswith(prefix): return None
            kwargs = cls.parse_shortcut_kwargs(cmd)
            if not kwargs: return None
            return cls(**kwargs)

        # ---------- parsing logic, easy to override/extend ----------
        @classmethod
        def parse_shortcut_kwargs(cls, cmd: str,
                                    # tokens like i3, o32, k3, s1, p1, d1, g1
                                    shortcut_prefix_fields: Dict[str, str] = {
                                        "i": "in_channels",
                                        "o": "out_channels",
                                        "k": "kernel_size",
                                        "s": "stride",
                                        "p": "padding",
                                        "d": "dilation",
                                        "g": "groups",
                                    },

                                    # flag tokens like bs, nobs, bit, nobit
                                    shortcut_flag_tokens: Dict[str, tuple[str, Any]] = {
                                        "bs": ("bias", True),
                                        "nobs": ("bias", False),
                                        "bit": ("bit", True),
                                        "nobit": ("bit", False),
                                    }
                                ) -> Dict[str, Any]:
            """
            Parse a shortcut command into kwargs for the model constructor.
            Subclasses can override this to add new tokens but still call super().
            """
            parts = cmd.split("_")[1:]  # drop the 'conv' prefix or whatever prefix is used
            kwargs: Dict[str, Any] = {}

            for part in parts:
                # 1) flags like bs, nobs, bit, nobit
                if part in shortcut_flag_tokens:
                    field, value = shortcut_flag_tokens[part]
                    kwargs[field] = value
                    continue

                # 2) padding_mode: e.g. pmreflect -> padding_mode="reflect"
                if part.startswith("pm"):
                    # e.g. "pmzeros", "pmreflect", etc.
                    mode = part[2:] or "zeros"
                    kwargs["padding_mode"] = mode
                    continue

                # 3) scale_op: e.g. scale-median, scale_mean, scale_max
                if part.startswith("scale"):
                    # allow "scale-median" or "scale_median" or just "scale"
                    if "-" in part:
                        op = part.split("-", 1)[1]
                    elif "_" in part:
                        op = part.split("_", 1)[1]
                    else:
                        op = "median"
                    kwargs["scale_op"] = op or "median"
                    continue

                # 4) numeric prefix tokens: i3, o32, k3, s1, p1, d1, g1
                field = shortcut_prefix_fields.get(part[0])
                if field is not None:
                    value_str = part[1:]
                    if value_str:  # guard against malformed tokens like "i"
                        kwargs[field] = int(value_str)
                    continue

                # 5) unknown tokens -> ignore, or you might choose to raise here

            return kwargs


    class Conv2dSame(Conv2d):
        in_channels: int
        out_channels: int
        kernel_size: IntOrPair
        stride: IntOrPair = 1
        padding: PadArg = 0
        dilation: IntOrPair = 1
        groups: int = 1
        bias: bool = True


    class MixedConv2d(Conv2d):
        in_channels: int
        out_channels: int
        kernel_size: KernelSizeArg
        stride: int = 1
        padding: PadArg = ''
        dilation: int = 1
        depthwise: bool = False
        bias: bool = True

        def build(self) -> nn.Module:
            return _MixedConv2d(**self.model_dump())

    class SeparableConv2d(Conv2d):
        in_channels: int
        out_channels: int
        kernel_size: IntOrPair = 3
        stride: int = 1
        dilation: int = 1
        padding: PadArg = ''
        bias: bool = False
        channel_multiplier: float = 1.0
        pw_kernel_size: int = 1

        def build(self) -> nn.Module:
            return _SeparableConv2d(**self.model_dump())

    class SplitBatchNorm2d(Conv2d):
        num_features: int
        eps: float = 1e-5
        momentum: float = 0.1
        affine: bool = True
        track_running_stats: bool = True
        num_splits: int = 2

        def build(self) -> nn.Module:
            return _SplitBatchNorm2d(**self.model_dump())

    class StdConv2d(Conv2d):
        in_channel: int
        out_channels: int
        kernel_size: IntOrPair
        stride: int = 1
        padding: Optional[PadArg] = None
        dilation: int = 1
        groups: int = 1
        bias: bool = False
        eps: float = 1e-6

        def build(self) -> nn.Module:
            return _StdConv2d(**self.model_dump())

    class StdConv2dSame(Conv2d):
        in_channel: int
        out_channels: int
        kernel_size: IntOrPair
        stride: int = 1
        padding: PadArg = 'SAME'
        dilation: int = 1
        groups: int = 1
        bias: bool = False
        eps: float = 1e-6

        def build(self) -> nn.Module:
            return _StdConv2dSame(**self.model_dump())

    class ScaledStdConv2d(Conv2d):
        in_channels: int
        out_channels: int
        kernel_size: IntOrPair
        stride: int = 1
        padding: Optional[PadArg] = None
        dilation: int = 1
        groups: int = 1
        bias: bool = True
        gamma: float = 1.0
        eps: float = 1e-6
        gain_init: float = 1.0

        def build(self) -> nn.Module:
            return _ScaledStdConv2d(**self.model_dump())

    class ScaledStdConv2dSame(Conv2d):
        in_channels: int
        out_channels: int
        kernel_size: IntOrPair
        stride: int = 1
        padding: PadArg = 'SAME'
        dilation: int = 1
        groups: int = 1
        bias: bool = True
        gamma: float = 1.0
        eps: float = 1e-6
        gain_init: float = 1.0

        def build(self) -> nn.Module:
            return _ScaledStdConv2dSame(**self.model_dump())

class Conv2dControllers:
    class Conv2dController(nn.Module):
        def __init__(self,para:Conv2dModels.Conv2d,para_cls=Conv2dModels.Conv2d):
            if type(para) is dict: para = para_cls(**para)
            self.para = json.loads(para.model_dump_json())

            super().__init__()
            if para.bit:
                self.conv = Bit.Conv2d(**para.model_dump())
            else:                
                self.conv = nn.Conv2d(**para.model_dump(exclude=['scale_op']))

            self.para:Conv2dModels.Conv2d = self.para

        def forward(self,x):
            return self.conv(x)

        @torch.no_grad()
        def to_ternary(self):
            if hasattr(self.conv,'to_ternary'):return self.conv.to_ternary()
            print('to_ternary is no support!')
            return self.conv.to_ternary()

    class Conv2dSameController(Conv2dController):
        """ Tensorflow like 'SAME' convolution wrapper for 2D convolutions
        """
        def __init__(self,para:Conv2dModels.Conv2dSame,para_cls=Conv2dModels.Conv2dSame):
            super().__init__(para,para_cls)
            w,s,d = self.conv.weight.shape[-2:], self.conv.stride, self.conv.dilation
            self.pad = PadSame(w,s,d)
            self.para:Conv2dModels.Conv2dSame = self.para

        def forward(self, x):
            return super().forward(self.pad(x))
        
        @torch.no_grad()
        def to_ternary(self):
            return nn.Sequential(
                    self.pad,
                    super().to_ternary())

    #...
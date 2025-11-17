from __future__ import annotations

import json
from typing import Optional, Sequence, Tuple, Union

from pydantic import BaseModel, ConfigDict
from torch import nn
import torch

from bitlayers.bit import Bit
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

    class Conv2d(BaseModel):
        in_channels: int
        out_channels: int
        kernel_size: IntOrPair
        stride: IntOrPair = 1
        padding: PadArg = 0
        dilation: IntOrPair = 1
        groups: int = 1
        bias: bool = True
        bit: bool = True
        padding_mode: str = 'zeros'
        scale_op: str ="median"
        
        def build(self) -> nn.Module:
            return Conv2dControllers.Conv2dController(self)

    class Conv2dSame(Conv2d):
        in_channels: int
        out_channels: int
        kernel_size: IntOrPair
        stride: IntOrPair = 1
        padding: PadArg = 0
        dilation: IntOrPair = 1
        groups: int = 1
        bias: bool = True

        def build(self) -> nn.Module:
            return Conv2dControllers.Conv2dSameController(self)

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

        def forward(self, x):
            return super().forward(self.pad(x))
        
        @torch.no_grad()
        def to_ternary(self):
            return nn.Sequential(
                    self.pad,
                    super().to_ternary())

    #...
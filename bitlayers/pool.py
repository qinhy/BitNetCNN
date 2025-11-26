from __future__ import annotations

import json
import math
from typing import Optional, Sequence, Tuple, Union

from pydantic import BaseModel
from torch import nn
import torch
import torch.nn.functional as F

from .helpers import to_2tuple
from .padding import get_padding_value, pad_same

IntOrPair = Union[int, Tuple[int, int]]
PadArg = Union[str, int, Tuple[int, int]]

class PoolModels:
    class BasicModel(BaseModel):
        def build(self):
            mod = PoolModules
            return mod.__dict__[f'{self.__class__.__name__}'](self)

    class AvgPool2d(BasicModel):
        kernel_size: IntOrPair = 2
        stride: Optional[IntOrPair] = None
        padding: PadArg = 0
        ceil_mode: bool = False
        count_include_pad: bool = True
        divisor_override: Optional[int] = None

    class AvgPool2dSame(AvgPool2d):
        padding: PadArg = "same"

    class BlurPool2d(BasicModel):
        in_channels: int
        kernel_size: IntOrPair = 3
        stride: IntOrPair = 2
        padding: PadArg = "same"
        normalized: bool = True

    type = Union[AvgPool2d,AvgPool2dSame,BlurPool2d]

class PoolModules:
    class Module(nn.Module):
        def __init__(self, para: BaseModel, para_cls):
            if isinstance(para, dict):
                para = para_cls(**para)
            self.para = json.loads(para.model_dump_json())
            super().__init__()

    class AvgPool2d(Module):
        def __init__(self, para: PoolModels.AvgPool2d, para_cls=PoolModels.AvgPool2d):
            super().__init__(para, para_cls)
            self.para: PoolModels.AvgPool2d = self.para

            stride = para.stride if para.stride is not None else para.kernel_size
            kernel_size = para.kernel_size
            padding, dynamic_pad = get_padding_value(para.padding, kernel_size, stride=stride)

            pool_kwargs = dict(
                kernel_size=kernel_size,
                stride=stride,
                padding=self._normalize_padding(padding),
                ceil_mode=para.ceil_mode,
                count_include_pad=para.count_include_pad,
                divisor_override=para.divisor_override,
            )

            self.pool = nn.AvgPool2d(**pool_kwargs)
            self.dynamic_padding = dynamic_pad
            self.kernel_size = to_2tuple(kernel_size)
            self.stride = to_2tuple(stride)

        @staticmethod
        def _normalize_padding(padding: Union[int, Tuple[int, int], Sequence[int]]):
            if isinstance(padding, list):
                return tuple(padding)
            return padding

        def forward(self, x: torch.Tensor):
            if self.dynamic_padding:
                x = pad_same(x, self.kernel_size, self.stride)
            return self.pool(x)

    class AvgPool2dSame(AvgPool2d):
        def __init__(self, para: PoolModels.AvgPool2dSame, para_cls=PoolModels.AvgPool2dSame):
            super().__init__(para, para_cls)

    class BlurPool2d(Module):
        def __init__(self, para: PoolModels.BlurPool2d, para_cls=PoolModels.BlurPool2d):
            super().__init__(para, para_cls)
            self.para: PoolModels.BlurPool2d = self.para

            self.kernel_size = to_2tuple(para.kernel_size)
            self.stride = to_2tuple(para.stride)
            padding, dynamic_pad = get_padding_value(para.padding, self.kernel_size, stride=self.stride)
            self.padding = self._normalize_padding(padding)
            self.dynamic_padding = dynamic_pad

            kernel = self._build_kernel(para.in_channels, self.kernel_size, para.normalized)
            self.register_buffer("kernel", kernel)
            self.groups = para.in_channels

        @staticmethod
        def _normalize_padding(padding: Union[int, Tuple[int, int], Sequence[int]]):
            if isinstance(padding, list):
                return tuple(padding)
            return padding

        @staticmethod
        def _binomial_coeffs(size: int) -> torch.Tensor:
            if size <= 0:
                raise ValueError("BlurPool2d kernel dimensions must be positive.")
            coeffs = torch.tensor([math.comb(size - 1, k) for k in range(size)], dtype=torch.float32)
            return coeffs

        def _build_kernel(self, channels: int, kernel_size: Tuple[int, int], normalized: bool) -> torch.Tensor:
            kh, kw = kernel_size
            kernel_h = self._binomial_coeffs(kh)
            kernel_w = self._binomial_coeffs(kw)
            kernel_2d = torch.outer(kernel_h, kernel_w)
            if normalized:
                kernel_2d = kernel_2d / kernel_2d.sum()
            kernel_2d = kernel_2d.view(1, 1, kh, kw)
            kernel_2d = kernel_2d.repeat(channels, 1, 1, 1)
            return kernel_2d

        def forward(self, x: torch.Tensor):
            if self.dynamic_padding:
                x = pad_same(x, self.kernel_size, self.stride)
                padding = 0
            else:
                padding = self.padding
            return F.conv2d(x, self.kernel, stride=self.stride, padding=padding, groups=self.groups)

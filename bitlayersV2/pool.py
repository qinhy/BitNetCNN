from __future__ import annotations

import json
import math
from typing import Optional, Sequence, Tuple, Union

from pydantic import BaseModel, Field, model_validator
from . import nn
import torch
import torch.nn.functional as F

from .helpers import to_2tuple
from .padding import get_padding_value, pad_same

IntOrPair = Union[int, Tuple[int, int]]
PadArg = Union[str, int, Tuple[int, int]]

class Pools:
    class AvgPool2d(nn.Module, torch.nn.AvgPool2d):
        kernel_size: IntOrPair = 2
        stride: Optional[IntOrPair] = None
        padding: PadArg = 0
        ceil_mode: bool = False
        count_include_pad: bool = True
        divisor_override: Optional[int] = None
        dynamic_padding: bool = False

        def model_post_init(self, __context):
            super().model_post_init(__context)

            stride = self.stride if self.stride is not None else self.kernel_size
            kernel_size = self.kernel_size
            padding, dynamic_pad = get_padding_value(self.padding, kernel_size, stride=stride)
            pool_kwargs = dict(
                kernel_size=kernel_size,
                stride=stride,
                padding=self._normalize_padding(padding),
                ceil_mode=self.ceil_mode,
                count_include_pad=self.count_include_pad,
                divisor_override=self.divisor_override,
            )
            torch.nn.AvgPool2d.__init__(self, **pool_kwargs)
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
            return super().forward(x)

    class AvgPool2dSame(AvgPool2d):
        pass

    class BlurPool2d(nn.Module):
        in_channels: int
        kernel_size: IntOrPair = 3
        stride: IntOrPair = 2
        padding: PadArg = "same"
        normalized: bool = True
        kernel: torch.Tensor = Field(default=None, exclude=True)

        def model_post_init(self, __context):
            super().model_post_init(__context)
            self.kernel_size = to_2tuple(self.kernel_size)
            self.stride = to_2tuple(self.stride)
            padding, dynamic_pad = get_padding_value(self.padding, self.kernel_size, stride=self.stride)
            self.padding = self._normalize_padding(padding)
            self.dynamic_padding = dynamic_pad

            kernel = self._build_kernel(self.in_channels, self.kernel_size, self.normalized)
            self.register_buffer("kernel", kernel)
            self.groups = self.in_channels

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


    type = Union[AvgPool2d,AvgPool2dSame,BlurPool2d]
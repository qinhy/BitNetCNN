"""
Common utilities for BitNetCNN implementations.
This module contains shared components used across different BitNet model implementations.
"""
import collections
from itertools import repeat
import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from bitlayers.padding import PadSame, get_padding_value, pad_same
def _ntuple(n, name="parse"):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return tuple(x)
        return tuple(repeat(x, n))

    parse.__name__ = name
    return parse


_single = _ntuple(1, "_single")
to_2tuple = _pair = _ntuple(2, "_pair")
_triple = _ntuple(3, "_triple")
_quadruple = _ntuple(4, "_quadruple")

# Constants
EPS = 1e-12

# ----------------------------
# Quantization Utilities
# ----------------------------
def _reduce_abs(x: torch.Tensor, keep_dim: int, op: str = "mean") -> torch.Tensor:
    """
    Reduce absolute values along all dimensions except keep_dim.
    Supports mean and median operations.
    Returns a tensor broadcastable to x with singleton dims in reduced axes.
    """
    if keep_dim < 0 or keep_dim >= x.dim():
        raise ValueError(f"keep_dim={keep_dim} out of range for tensor of dim {x.dim()}")
    dims = [d for d in range(x.dim()) if d != keep_dim]
    a = x.abs()
    if op == "mean":
        s = a.mean(dim=dims, keepdim=True)
    elif op == "median":
        # median over flattened other dims
        perm = (keep_dim,) + tuple(d for d in range(x.dim()) if d != keep_dim)
        flat = a.permute(perm).contiguous().view(a.size(keep_dim), -1)
        s = flat.median(dim=1).values.view([a.size(keep_dim)] + [1] * (x.dim() - 1))
        inv = [0] * x.dim()
        for i, p in enumerate(perm):
            inv[p] = i
        s = s.permute(*inv).contiguous()
    else:
        raise ValueError("op must be 'mean' or 'median'")
    return s.clamp_min(EPS)


# ----------------------------
# Model-wide conversion helpers
# ----------------------------
@torch.no_grad()
def convert_to_ternary(module: nn.Module) -> nn.Module:
    """
    Recursively replace Bit.Conv2d/Bit.Linear with Ternary*Infer modules **in-place**.
    Returns the same nn.Module for convenience.

    If you want to keep the original network, call this on a deepcopy:
        from copy import deepcopy
        ternary = convert_to_ternary(deepcopy(model))
    """
    for name, child in list(module.named_children()):
        if hasattr(child, "to_ternary"):
            setattr(module, name, child.to_ternary())
        else:
            convert_to_ternary(child)
    return module


# ----------------------------
# Bit Quantization Classes
# ----------------------------
class Bit:
    """
    Collection of classes for bit-level quantization of neural networks.
    Includes fake-quant building blocks, inference modules, and training modules.
    """

    class functional:
        @staticmethod
        def bit1p58_weight(weight: torch.Tensor, dim: int = 0, scale_op: str = "median") -> torch.Tensor:
            s = _reduce_abs(weight, keep_dim=dim, op=scale_op)
            w_bar = (weight / s).detach()
            w_q = torch.round(w_bar).clamp_(-1, 1)
            return weight + (w_q * s - weight).detach()

        @staticmethod
        def conv2d(
            input: torch.Tensor,
            weight: torch.Tensor,
            bias=None,
            stride=1,
            padding=0,
            padding_mode: str = "zeros",
            dilation=1,
            groups=1,
            dim: int = 0,
            scale_op: str = "median",
        ) -> torch.Tensor:
            """
            Conv2d with fake-quant ternary weights (STE). Supports padding_mode similarly to nn.Conv2d.
            """
            weight = Bit.functional.bit1p58_weight(weight, dim, scale_op)

            # Emulate nn.Conv2d padding_mode behavior when needed
            if padding_mode != "zeros" and padding != 0:
                if isinstance(padding, int):
                    pad = (padding, padding, padding, padding)  # left, right, top, bottom
                elif isinstance(padding, tuple) and len(padding) == 2:
                    pad_h, pad_w = padding
                    pad = (pad_w, pad_w, pad_h, pad_h)
                else:
                    raise ValueError(f"Unsupported padding={padding} for padding_mode='{padding_mode}'")
                input = F.pad(input, pad, mode=padding_mode)
                padding_eff = 0
            else:
                padding_eff = padding

            return F.conv2d(
                input,
                weight,
                bias,
                stride=stride,
                padding=padding_eff,
                dilation=dilation,
                groups=groups,
            )

        @staticmethod
        def linear(
            input: torch.Tensor,
            weight: torch.Tensor,
            bias=None,
            dim: int = 0,
            scale_op: str = "median",
        ) -> torch.Tensor:
            weight = Bit.functional.bit1p58_weight(weight, dim, scale_op)
            return F.linear(input, weight, bias)

    class CommonConv2d(nn.Module):        
        def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,          # can be int, tuple, 'same', 'valid'
            padding_mode="zeros",
            dilation=1,
            groups=1,
            bias=True,
            scale_op="median",
        ):
            super().__init__()
            kh, kw = to_2tuple(kernel_size)
            self.in_channels = in_channels
            self.out_channels = out_channels
            self.kernel_size = (kh, kw)
            self.stride = to_2tuple(stride)
            self.dilation = to_2tuple(dilation)
            self.padding = padding
            self.padding_mode = padding_mode
            self.scale_op = scale_op

            # resolve padding (static vs dynamic 'same')
            self.padding_value, self.dynamic_pad = get_padding_value(
                padding,
                kernel_size=self.kernel_size,
                stride=self.stride,
                dilation=self.dilation,
            )

            # Optional: set up PadSame if needed
            if self.dynamic_pad:
                self.pad_layer = PadSame(self.kernel_size, self.stride, self.dilation)
            else:
                self.pad_layer = None

            self.groups = groups

            self.init_weights(bias)

        def get_weights(self)->torch.Tensor:
            pass
        def init_weights(self,bias):
            self.bias = bias
            pass
            # # weights
            # self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, kh, kw))
            # nn.init.kaiming_normal_(self.weight, nonlinearity="relu")
            # self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None
            # self.w_q = Bit.Bit1p58Weight(dim=0, scale_op=scale_op)

        def forward(self, x:torch.Tensor, w=None):
            if w is None:
                w = self.get_weights()

            # dynamic SAME padding_value if required
            if self.dynamic_pad:
                x = self.pad_layer(x)
                padding_value = 0
            else:
                padding_value = self.padding_value
                
            return F.conv2d(x, w, self.bias, self.stride, padding_value, self.dilation, self.groups)
    # ----------------------------
    # Fake-quant building blocks (QAT) — activation quantization removed
    # ----------------------------
    class Bit1p58Weight(nn.Module):
        """Ternary STE quantizer (≈1.58 bits) with per-channel scale from |w| reduction."""
        def __init__(self, dim: int = 0, scale_op: str = "median"):
            super().__init__()
            self.dim = dim
            self.scale_op = scale_op

        def forward(self, w: torch.Tensor) -> torch.Tensor:
            s = _reduce_abs(w, keep_dim=self.dim, op=self.scale_op)
            w_bar = (w / s).detach()
            w_q = torch.round(w_bar).clamp_(-1, 1)
            return w + (w_q * s - w).detach()

    # ----------------------------
    # Inference (frozen) ternary modules — no activation quantization
    # ----------------------------
    class Conv2dInfer(nn.Module):
        """
        Frozen ternary conv:
        y = (Conv(x, Wq) * s_per_out) + b
        Wq in {-1,0,+1} stored as int8. s is float per output channel.
        """
        def __init__(
            self,
            w_q: torch.Tensor,
            s: torch.Tensor,
            bias,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            padding_mode: str = "zeros",
        ):
            super().__init__()
            self.kernel_size = kernel_size
            self.stride = to_2tuple(stride)
            self.dilation = to_2tuple(dilation)
            self.padding = padding
            self.padding_mode = padding_mode

            # Make them Parameters so param counters include them (but keep frozen)
            self.w_q = nn.Parameter(w_q.to(torch.int8), requires_grad=False)  # [out,in,kh,kw]
            self.s = nn.Parameter(s, requires_grad=False)  # [out,1,1]
            if bias is None:
                self.register_parameter("bias", None)
            else:
                self.bias = nn.Parameter(bias, requires_grad=False)

            # Optional cache for the float view (not in state_dict, not counted as param)
            self.register_buffer("_w_q_float", None, persistent=False)

            # resolve padding (static vs dynamic 'same')
            self.padding_value, self.dynamic_pad = get_padding_value(
                padding,
                kernel_size=self.kernel_size,
                stride=self.stride,
                dilation=self.dilation,
            )
            
            # Optional: set up PadSame if needed
            if self.dynamic_pad:
                self.pad_layer = PadSame(self.kernel_size, self.stride, self.dilation)
            else:
                self.pad_layer = None

            self.groups = groups

        def _weight(self, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
            if self._w_q_float is None or self._w_q_float.dtype != dtype or self._w_q_float.device != device:
                self._w_q_float = self.w_q.to(device=device, dtype=dtype)
            return self._w_q_float

        def forward(self, x:torch.Tensor):
            wq = self._weight(x.dtype, x.device)

            # dynamic SAME padding_value if required
            if self.dynamic_pad:
                x = self.pad_layer(x)
                padding_value = 0
            else:
                padding_value = self.padding_value
                
            return F.conv2d(x, wq, self.bias, self.stride, padding_value, self.dilation, self.groups,)
        
    class LinearInfer(nn.Module):
        """Frozen ternary linear: y = (x @ Wq^T) * s + b"""
        def __init__(self, w_q: torch.Tensor, s: torch.Tensor, bias):
            super().__init__()
            self.w_q = nn.Parameter(w_q.to(torch.int8), requires_grad=False)  # [out,in]
            self.s = nn.Parameter(s, requires_grad=False)  # [out]
            if bias is None:
                self.register_parameter("bias", None)
            else:
                self.bias = nn.Parameter(bias, requires_grad=False)

            self.register_buffer("_w_q_float", None, persistent=False)

        def _weight(self, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
            if self._w_q_float is None or self._w_q_float.dtype != dtype or self._w_q_float.device != device:
                self._w_q_float = self.w_q.to(device=device, dtype=dtype)
            return self._w_q_float

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            w = self._weight(x.dtype, x.device)
            y = F.linear(x, w, bias=None)
            y = y * self.s.to(dtype=y.dtype, device=y.device)
            if self.bias is not None:
                y = y + self.bias.to(dtype=y.dtype, device=y.device)
            return y

    # ----------------------------
    # Train-time modules (no BatchNorm), activation quantization removed
    # ----------------------------
    class Conv2d(CommonConv2d):

        def get_weights(self)->torch.Tensor:
            wq = self.w_q(self.weight)
            return wq
        
        def init_weights(self,bias):
            # weights
            kh, kw = self.kernel_size
            self.weight = nn.Parameter(torch.empty(self.out_channels, self.in_channels // self.groups, kh, kw))
            nn.init.kaiming_normal_(self.weight, nonlinearity="relu")
            self.bias = nn.Parameter(torch.zeros(self.out_channels)) if bias else None
            self.w_q = Bit.Bit1p58Weight(dim=0, scale_op=self.scale_op)
        
        @torch.no_grad()
        def to_ternary(self):
            """
            Convert this layer into a frozen Bit.Conv2dInfer, carrying over:
            - per-out-channel weight scale s and Wq in {-1,0,+1}
            """
            w = self.weight.data
            s_vec = _reduce_abs(w, keep_dim=0, op=self.scale_op).squeeze()  # [out]
            s = s_vec.view(-1, 1, 1)                                        # [out,1,1] for conv broadcast
            w_bar = w / s_vec.view(-1, 1, 1, 1)
            w_q = torch.round(w_bar).clamp_(-1, 1).to(w.dtype)

            return Bit.Conv2dInfer(
                w_q=w_q,
                s=s,
                bias=(None if self.bias is None else self.bias.data.clone()),
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
                kernel_size=self.kernel_size,
                padding_mode=self.padding_mode,
            )
            
    class Linear(nn.Module):
        def __init__(self, in_f: int, out_f: int, bias: bool = True, scale_op: str = "median"):
            super().__init__()
            self.weight = nn.Parameter(torch.empty(out_f, in_f))
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
            self.bias = nn.Parameter(torch.zeros(out_f)) if bias else None
            self.w_q = Bit.Bit1p58Weight(dim=0, scale_op=scale_op)
            self.scale_op = scale_op

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            wq = self.w_q(self.weight)
            return F.linear(x, wq, self.bias)

        @torch.no_grad()
        def to_ternary(self):
            w = self.weight.data
            s = _reduce_abs(w, keep_dim=0, op=self.scale_op).squeeze()  # [out]
            w_q = torch.round(w / s.view(-1, 1)).clamp_(-1, 1).to(w.dtype)
            return Bit.LinearInfer(
                w_q=w_q,
                s=s,
                bias=(None if self.bias is None else self.bias.data.clone()),
            )

    # for debug at normal one
    # class Conv2d(nn.Conv2d): pass
    # class Linear(nn.Linear): pass

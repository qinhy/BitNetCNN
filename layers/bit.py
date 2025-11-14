"""
Common utilities for BitNetCNN implementations.
This module contains shared components used across different BitNet model implementations.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
# Constants
EPS = 1e-12

# ----------------------------
# Quantization Utilities
# ----------------------------
def summ(model,verbose=True):
    info = []
    for name, module in model.named_modules():
        info.append( (name, sum(param.numel() for param in module.parameters())) )
        if verbose: print(*info[-1])
    return info

@torch.no_grad()
def _pow2_quantize_scale(s: torch.Tensor, min_exp: int = -32, max_exp: int = 31):
    """
    s > 0 tensor -> (s_q, k) where s_q = 2^k and k is int8 clamped to [min_exp, max_exp].
    """
    s = s.clamp_min(EPS).float()
    k = torch.round(torch.log2(s)).clamp(min_exp, max_exp).to(torch.int8)
    s_q = torch.pow(2.0, k.to(torch.float32))
    return s_q, k

def _reduce_abs(x, keep_dim, op="mean"):
    """
    Reduce absolute values along all dimensions except keep_dim.
    Supports mean and median operations.
    """
    dims = [d for d in range(x.dim()) if d != keep_dim]
    a = x.abs()
    if op == "mean":
        s = a.mean(dim=dims, keepdim=True)
    elif op == "median":
        # median over flattened other dims
        perm = (keep_dim,) + tuple(d for d in range(x.dim()) if d != keep_dim)
        flat = a.permute(perm).contiguous().view(a.size(keep_dim), -1)
        s = flat.median(dim=1).values.view([a.size(keep_dim)] + [1]*(x.dim()-1))
        inv = [0]*x.dim()
        for i,p in enumerate(perm): inv[p] = i
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
    Recursively replace Bit.Conv2d/Bit.Linear with Ternary*Infer modules.
    Returns a new nn.Module (original left untouched if you deepcopy before).
    """
    for name, child in list(module.named_children()):
        if hasattr(child, 'to_ternary'):
            setattr(module, name, child.to_ternary())
        else:
            convert_to_ternary(child)
    return module

@torch.no_grad()
def convert_to_ternary_p2(module: nn.Module) -> nn.Module:
    """
    Recursively replace Bit.Conv2d/Bit.Linear with their PoT inference counterparts.
    """
    for name, child in list(module.named_children()):
        if hasattr(child, 'to_ternary_p2'):
            setattr(module, name, child.to_ternary_p2())
        else:
            convert_to_ternary_p2(child)
    return module

# ----------------------------
# Bit Quantization Classes
# ----------------------------
class Bit:
    """
    Collection of classes for bit-level quantization of neural networks.
    Includes fake-quant building blocks, inference modules, and training modules.
    """
    # ----------------------------
    # Fake-quant building blocks (QAT) — activation quantization removed
    # ----------------------------
    class Bit1p58Weight(nn.Module):
        """1.58-bit (ternary) weight quantizer with per-out-channel scaling."""
        def __init__(self, dim=0, scale_op="median"):
            super().__init__()
            self.dim = dim
            self.scale_op = scale_op

        def forward(self, w):
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
        def __init__(self, w_q, s, bias, stride, padding, dilation, groups):
            super().__init__()
            # Make them Parameters so param counters include them (but keep frozen)
            self.w_q  = nn.Parameter(w_q.to(torch.int8), requires_grad=False)   # [out,in,kh,kw]
            self.s    = nn.Parameter(s,                     requires_grad=False) # [out,1,1]
            if bias is None:
                self.register_parameter("bias", None)
            else:
                self.bias = nn.Parameter(bias, requires_grad=False)

            self.stride, self.padding, self.dilation, self.groups = stride, padding, dilation, groups
            # Optional cache for the float view (not in state_dict, not counted as param)
            self.register_buffer("_w_q_float", None, persistent=False)

        def _weight(self, dtype, device):
            if self._w_q_float is None or self._w_q_float.dtype != dtype or self._w_q_float.device != device:
                self._w_q_float = self.w_q.to(device=device, dtype=dtype)
            return self._w_q_float

        def forward(self, x):
            w = self._weight(x.dtype, x.device)
            y = F.conv2d(x, w, None, self.stride, self.padding, self.dilation, self.groups)
            y = y * self.s.to(dtype=y.dtype, device=y.device)
            if self.bias is not None:
                y = y + self.bias.to(dtype=y.dtype, device=y.device).view(1, -1, 1, 1)
            return y

    class LinearInfer(nn.Module):
        """Frozen ternary linear: y = (x @ Wq^T) * s + b"""
        def __init__(self, w_q, s, bias):
            super().__init__()
            self.w_q = nn.Parameter(w_q.to(torch.int8), requires_grad=False)   # [out,in]
            self.s   = nn.Parameter(s,                    requires_grad=False)  # [out]
            if bias is None:
                self.register_parameter("bias", None)
            else:
                self.bias = nn.Parameter(bias, requires_grad=False)

            self.register_buffer("_w_q_float", None, persistent=False)

        def _weight(self, dtype, device):
            if self._w_q_float is None or self._w_q_float.dtype != dtype or self._w_q_float.device != device:
                self._w_q_float = self.w_q.to(device=device, dtype=dtype)
            return self._w_q_float

        def forward(self, x):
            w = self._weight(x.dtype, x.device)
            y = F.linear(x, w, bias=None)
            y = y * self.s.to(dtype=y.dtype, device=y.device)
            if self.bias is not None:
                y = y + self.bias.to(dtype=y.dtype, device=y.device)
            return y
        
    class Conv2dInferP2(nn.Module):
        """
        Ternary conv with power-of-two scales:
        y = Conv(x, Wq) * 2^{s_exp} + b
        Wq in {-1,0,+1} as int8. s_exp is per-out exponent [out,1,1].
        """
        def __init__(self, w_q, s_exp, bias, stride, padding, dilation, groups):
            super().__init__()
            # Counted as params but frozen
            self.w_q  = nn.Parameter(w_q.to(torch.int8), requires_grad=False)      # [out,in,kh,kw]
            self.s_exp = nn.Parameter(s_exp.to(torch.int8), requires_grad=False)   # [out,1,1]
            if bias is None:
                self.register_parameter("bias", None)
            else:
                self.bias = nn.Parameter(bias, requires_grad=False)                # [out]

            self.stride, self.padding, self.dilation, self.groups = stride, padding, dilation, groups
            # Cache float weights per (device,dtype); not saved, not counted
            self.register_buffer("_w_q_float", None, persistent=False)

        def _weight(self, dtype, device):
            if self._w_q_float is None or self._w_q_float.dtype != dtype or self._w_q_float.device != device:
                self._w_q_float = self.w_q.to(device=device, dtype=dtype)
            return self._w_q_float

        def forward(self, x):
            w = self._weight(x.dtype, x.device)
            y = F.conv2d(x, w, None, self.stride, self.padding, self.dilation, self.groups)
            y = torch.ldexp(y, self.s_exp.to(torch.int32, device=y.device))  # broadcast [out,1,1]
            if self.bias is not None:
                y = y + self.bias.to(dtype=y.dtype, device=y.device).view(1, -1, 1, 1)
            return y

    class LinearInferP2(nn.Module):
        """Ternary linear with power-of-two output scales: y = (x @ Wq^T) * 2^{s_exp} + b"""
        def __init__(self, w_q, s_exp, bias):
            super().__init__()
            self.w_q   = nn.Parameter(w_q.to(torch.int8), requires_grad=False)     # [out,in]
            self.s_exp = nn.Parameter(s_exp.to(torch.int8), requires_grad=False)   # [out]
            if bias is None:
                self.register_parameter("bias", None)
            else:
                self.bias = nn.Parameter(bias, requires_grad=False)                # [out]
            self.register_buffer("_w_q_float", None, persistent=False)

        def _weight(self, dtype, device):
            if self._w_q_float is None or self._w_q_float.dtype != dtype or self._w_q_float.device != device:
                self._w_q_float = self.w_q.to(device=device, dtype=dtype)
            return self._w_q_float

        def forward(self, x):
            w = self._weight(x.dtype, x.device)
            y = F.linear(x, w, bias=None)
            y = torch.ldexp(y, self.s_exp.to(torch.int32, device=y.device))  # broadcast [out]
            if self.bias is not None:
                y = y + self.bias.to(dtype=y.dtype, device=y.device)
            return y

    # ----------------------------
    # Train-time modules (no BatchNorm), activation quantization removed
    # ----------------------------
    class Conv2d(nn.Module):
        """
        Conv2d with ternary weights (fake-quant for training).
        No BatchNorm inside. Add your own nonlinearity outside if desired.
        """
        def __init__(self, in_c, out_c, kernel_size, stride=1, padding=0, dilation=1, groups=1,
                    bias=True, scale_op="median"):
            super().__init__()
            if isinstance(kernel_size, int):
                kh = kw = kernel_size
            else:
                kh, kw = kernel_size
            self.weight = nn.Parameter(torch.empty(out_c, in_c // groups, kh, kw))
            nn.init.kaiming_normal_(self.weight, nonlinearity="relu")
            self.bias = nn.Parameter(torch.zeros(out_c)) if bias else None
            self.w_q = Bit.Bit1p58Weight(dim=0, scale_op=scale_op)
            self.stride, self.padding, self.dilation, self.groups = stride, padding, dilation, groups
            self.scale_op = scale_op

        def forward(self, x):
            wq = self.w_q(self.weight)
            return F.conv2d(x, wq, self.bias, self.stride, self.padding, self.dilation, self.groups)

        @torch.no_grad()
        def to_ternary(self):
            """
            Convert this layer into a frozen Bit.Conv2dInfer, carrying over:
            - per-out-channel weight scale s and Wq in {-1,0,+1}
            """
            w = self.weight.data
            s_vec = _reduce_abs(w, keep_dim=0, op=self.scale_op).squeeze()   # [out]
            s = s_vec.view(-1, 1, 1)                                         # [out,1,1] for conv broadcast
            w_bar = w / s_vec.view(-1, 1, 1, 1)
            w_q = torch.round(w_bar).clamp_(-1, 1).to(w.dtype)

            return Bit.Conv2dInfer(
                w_q=w_q, s=s,
                bias=(None if self.bias is None else self.bias.data.clone()),
                stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups
            )

        @torch.no_grad()
        def to_ternary_p2(self):
            # Per-out-channel scale from your chosen op
            w = self.weight.data
            s_vec = _reduce_abs(w, keep_dim=0, op=self.scale_op).squeeze()  # [out]
            w_bar = w / s_vec.view(-1,1,1,1)
            w_q = torch.round(w_bar).clamp_(-1, 1).to(w.dtype)

            # Quantize weight scale to power-of-two (save exponents)
            _, s_exp = _pow2_quantize_scale(s_vec)           # int8 exponents
            s_exp = s_exp.view(-1, 1, 1)

            return Bit.Conv2dInferP2(
                w_q=w_q,
                s_exp=s_exp,
                bias=(None if self.bias is None else self.bias.data.clone()),
                stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups
            )

    class Linear(nn.Module):
        def __init__(self, in_f, out_f, bias=True, scale_op="median"):
            super().__init__()
            self.weight = nn.Parameter(torch.empty(out_f, in_f))
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
            self.bias = nn.Parameter(torch.zeros(out_f)) if bias else None
            self.w_q = Bit.Bit1p58Weight(dim=0, scale_op=scale_op)
            self.scale_op = scale_op

        def forward(self, x):
            wq = self.w_q(self.weight)
            return F.linear(x, wq, self.bias)

        @torch.no_grad()
        def to_ternary(self):
            w = self.weight.data
            s = _reduce_abs(w, keep_dim=0, op=self.scale_op).squeeze()  # [out]
            w_q = torch.round(w / s.view(-1,1)).clamp_(-1, 1).to(w.dtype)
            return Bit.LinearInfer(
                w_q=w_q, s=s, bias=(None if self.bias is None else self.bias.data.clone())
            )

        @torch.no_grad()
        def to_ternary_p2(self):
            w = self.weight.data
            s = _reduce_abs(w, keep_dim=0, op=self.scale_op).squeeze()   # [out]
            w_q = torch.round((w / s.view(-1,1))).clamp_(-1, 1).to(w.dtype)

            # Quantize to power-of-two
            _, s_exp = _pow2_quantize_scale(s)   # [out] int8
            return Bit.LinearInferP2(
                w_q=w_q, s_exp=s_exp, bias=(None if self.bias is None else self.bias.data.clone())
            )

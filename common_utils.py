"""
Common utilities for BitNetCNN implementations.
This module contains shared components used across different BitNet model implementations.
"""

from functools import partial
import os
import math
import copy

import re
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import warnings
from PIL import Image, ImageTk

import random
from typing import Any, Callable, List, Literal, Optional, Sequence, Tuple, Type, Union
from pydantic import BaseModel, Field
import torch

torch.set_float32_matmul_precision('high')
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision import transforms as T
import pytorch_lightning as pl
from pytorch_lightning import Callback
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from torchvision.transforms import InterpolationMode
from torchmetrics.classification import MulticlassAccuracy
from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import extract_archive, check_integrity, download_url, verify_str_arg

# Constants
EPS = 1e-12

# ----------------------------
# Quantization Utilities
# ----------------------------
def summ(model, verbose=True, include_buffers=True):
    info = []
    for name, module in model.named_modules():
        # parameters for this module only (no children)
        params = list(module.parameters(recurse=False))
        nparams = sum(p.numel() for p in params)

        # collect dtypes from params (and optionally buffers)
        tensors = params
        if include_buffers:
            tensors += list(module.buffers(recurse=False))

        dtypes = {t.dtype for t in tensors}
        if dtypes:
            # e.g. "float32", "float16, int8"
            dtype_str = ", ".join(
                sorted(str(dt).replace("torch.", "") for dt in dtypes)
            )
        else:
            dtype_str = "-"

        row = (name, module.__class__.__name__, nparams, dtype_str)
        info.append(row)

        if verbose:
            print(f"{name:40} {module.__class__.__name__:20} "
                  f"params={nparams:10d}  dtypes={dtype_str}")
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
            self.s    = nn.Parameter(s,                  requires_grad=False) # [out,1,1]
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


def replace_all2Bit(model: nn.Module, scale_op: str = "median", wrap_same: bool = True) -> Tuple[int, int]:
    convs, linears = 0, 0

    for name, child in list(model.named_children()):
        # Recurse first
        c_cnt, l_cnt = replace_all2Bit(child, scale_op, wrap_same)
        convs += c_cnt
        linears += l_cnt

        # Determine if child is already a Bit layer (if Bit exists)
        try:
            is_bit_conv = isinstance(child, Bit.Conv2d)
            is_bit_linear = isinstance(child, Bit.Linear)
        except Exception:
            is_bit_conv = False
            is_bit_linear = False

        new_child = None

        # Replace Conv2d with Bit.Conv2d (respect wrap_same)
        if isinstance(child, nn.Conv2d) and (wrap_same or not is_bit_conv):
            dev = child.weight.device
            dt = child.weight.dtype

            new_child = Bit.Conv2d(
                in_c=child.in_channels,
                out_c=child.out_channels,
                kernel_size=child.kernel_size,
                stride=child.stride,
                padding=child.padding,
                dilation=child.dilation,
                groups=child.groups,
                bias=(child.bias is not None),
                scale_op=scale_op,
            ).to(device=dev, dtype=dt)

            # Best-effort: copy weights/bias if attributes are compatible
            # with torch.no_grad():
            #     if hasattr(new_child, "weight") and new_child.weight.shape == child.weight.shape:
            #         new_child.weight.copy_(child.weight)
            #     if child.bias is not None and hasattr(new_child, "bias") and new_child.bias is not None:
            #         new_child.bias.copy_(child.bias)

            convs += 1

        # Replace Linear with Bit.Linear (respect wrap_same)
        elif isinstance(child, nn.Linear) and (wrap_same or not is_bit_linear):
            dev = child.weight.device
            dt = child.weight.dtype

            new_child = Bit.Linear(
                in_f=child.in_features,
                out_f=child.out_features,
                bias=(child.bias is not None),
                scale_op=scale_op,
            ).to(device=dev, dtype=dt)

            # Best-effort: copy weights/bias if attributes are compatible
            # with torch.no_grad():
            #     if hasattr(new_child, "weight") and new_child.weight.shape == child.weight.shape:
            #         new_child.weight.copy_(child.weight)
            #     if child.bias is not None and hasattr(new_child, "bias") and new_child.bias is not None:
            #         new_child.bias.copy_(child.bias)

            linears += 1

        if new_child is not None:
            setattr(model, name, new_child)

    return convs, linears
# ----------------------------
# KD losses & feature hints (unchanged from your pattern)
# ----------------------------
class KDLoss(nn.Module):
    def __init__(self, T=4.0): super().__init__(); self.T=T
    def forward(self, z_s, z_t):
        T = self.T
        return F.kl_div(F.log_softmax(z_s/T,1), F.softmax(z_t/T,1), reduction="batchmean") * (T*T)

class AdaptiveHintLoss(nn.Module):
    """Learnable 1x1 per hint; auto matches spatial size then SmoothL1."""
    def __init__(self):
        super().__init__()
        self.proj = nn.ModuleDict()
    
    @staticmethod
    def _k(name: str) -> str:
        # bijective mapping so we never collide with real underscores etc.
        # U+2027 (Hyphenation Point) is printable and allowed in names.
        return name.replace('.', '\u2027')

    def forward(self, name, f_s, f_t):
        f_s = F.adaptive_avg_pool2d(f_s, f_t.shape[-2:])
        c_s, c_t = f_s.shape[1], f_t.shape[1]
        k = self._k(name)
        if (k not in self.proj or
            self.proj[k].in_channels != c_s or
            self.proj[k].out_channels != c_t):
            self.proj[k] = nn.Conv2d(c_s, c_t, kernel_size=1, bias=True).to(f_s.device)
        f_s = self.proj[k](f_s)
        return F.smooth_l1_loss(f_s, f_t.detach())


class SaveOutputHook:
    """Picklable forward hook that stores outputs into a dict under a given key."""
    __slots__ = ("store", "key")
    def __init__(self, store: dict, key: str):
        self.store = store
        self.key = key
    def __call__(self, module, module_in, module_out):
        self.store[self.key] = module_out

def make_feature_hooks(module: nn.Module, names, feats: dict, idx=0):
    """Register picklable forward hooks; returns list of handles."""
    handles = []
    if type(names[0]) == tuple:
        names = [n[idx] for n in names]
    name_set = set(names)
    for n, sub in module.named_modules():
        for i,ii in enumerate(name_set):
            if n.endswith(ii):break
        if n.endswith(ii):
            handles.append(sub.register_forward_hook(SaveOutputHook(feats, ii)))
    return handles

class DataModuleConfig(BaseModel):    
    data_dir: str
    batch_size: int
    num_workers: int = 4,
    mixup: bool = False
    cutmix: bool = False
    mix_alpha: float = 1.0

# ----------------------------
# CIFAR-100 DataModule (mixup/cutmix optional)
# ----------------------------
def mix_collate(batch, *, cutmix: bool, mixup: bool, mix_alpha: float):
    xs, ys = zip(*batch)
    x = torch.stack(xs); y = torch.tensor(ys)
    if not (cutmix or mixup):
        return x, y

    import random
    lam = 1.0
    if cutmix and random.random() < 0.5:
        lam = torch.distributions.Beta(mix_alpha, mix_alpha).sample().item()
        idx = torch.randperm(x.size(0))
        h, w = x.size(2), x.size(3)
        rx, ry = torch.randint(w, (1,)).item(), torch.randint(h, (1,)).item()
        rw = int(w * math.sqrt(1 - lam)); rh = int(h * math.sqrt(1 - lam))
        x1, y1 = max(rx - rw//2, 0), max(ry - rh//2, 0)
        x2, y2 = min(rx + rw//2, w), min(ry + rh//2, h)
        x[:, :, y1:y2, x1:x2] = x[idx, :, y1:y2, x1:x2]
        lam = 1 - ((x2 - x1) * (y2 - y1) / (w * h))
        return x, (y, y[idx], lam)
    else:
        lam = torch.distributions.Beta(mix_alpha, mix_alpha).sample().item()
        idx = torch.randperm(x.size(0))
        x = lam * x + (1 - lam) * x[idx]
        return x, (y, y[idx], lam)
    
class CIFAR100DataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int, num_workers: int = 4,
                 mixup: bool = False, cutmix: bool = False, mix_alpha: float = 1.0):
        super().__init__()
        self.data_dir, self.batch_size, self.num_workers = data_dir, batch_size, num_workers
        self.mixup, self.cutmix, self.mix_alpha = mixup, cutmix, mix_alpha

    def setup(self, stage=None):
        mean = (0.5071,0.4867,0.4408); std=(0.2675,0.2565,0.2761)
        train_tf = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        val_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
        self.train_ds = datasets.CIFAR100(root=self.data_dir, train=True,  download=True, transform=train_tf)
        self.val_ds   = datasets.CIFAR100(root=self.data_dir, train=False, download=True, transform=val_tf)

    def train_dataloader(self):
        collate = partial(
            mix_collate,
            cutmix=self.cutmix,
            mixup=self.mixup,
            mix_alpha=self.mix_alpha,
        )
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=True if self.num_workers > 0 else False,
            collate_fn=collate,
        )


    def val_dataloader(self,batch_size=256):
        return DataLoader(self.val_ds, batch_size=batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=True,
                          persistent_workers=True if self.num_workers > 0 else False)

# ----------------------------
# Mixup / CutMix Collate Function
# ----------------------------
def mix_collate(batch, *, cutmix: bool, mixup: bool, mix_alpha: float):
    xs, ys = zip(*batch)
    x = torch.stack(xs)
    y = torch.tensor(ys)
    
    if not (cutmix or mixup):
        return x, y

    lam = 1.0
    idx = torch.randperm(x.size(0))

    if cutmix and random.random() < 0.5:
        lam = torch.distributions.Beta(mix_alpha, mix_alpha).sample().item()
        h, w = x.size(2), x.size(3)
        rx, ry = torch.randint(w, (1,)).item(), torch.randint(h, (1,)).item()
        rw = int(w * math.sqrt(1 - lam))
        rh = int(h * math.sqrt(1 - lam))
        x1, y1 = max(rx - rw // 2, 0), max(ry - rh // 2, 0)
        x2, y2 = min(rx + rw // 2, w), min(ry + rh // 2, h)

        x[:, :, y1:y2, x1:x2] = x[idx, :, y1:y2, x1:x2]
        lam = 1 - ((x2 - x1) * (y2 - y1) / (w * h))
        return x, (y, y[idx], lam)
    else:
        lam = torch.distributions.Beta(mix_alpha, mix_alpha).sample().item()
        x = lam * x + (1 - lam) * x[idx]
        return x, (y, y[idx], lam)

# ----------------------------
# Tiny ImageNet Dataset Helper
# ----------------------------
# Copied from: https://github.com/lvyilin/pytorch-fgvc-dataset/blob/master/tiny_imagenet.py
class TinyImageNetDataset(VisionDataset):
    """`tiny-imageNet <http://cs231n.stanford.edu/tiny-imagenet-200.zip>`_ Dataset.

        Args:
            root (string): Root directory of the dataset.
            split (string, optional): The dataset split, supports ``train``, or ``val``.
            transform (callable, optional): A function/transform that  takes in an PIL image
               and returns a transformed version. E.g, ``transforms.RandomCrop``
            target_transform (callable, optional): A function/transform that takes in the
               target and transforms it.
            download (bool, optional): If true, downloads the dataset from the internet and
               puts it in root directory. If dataset is already downloaded, it is not
               downloaded again.
    """
    base_folder = 'tiny-imagenet-200/'
    url = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'
    filename = 'tiny-imagenet-200.zip'
    md5 = '90528d7ca1a48142e341f4ef8d21d0de'

    def __init__(self, root, split='train', transform=None, target_transform=None, download=False):
        super(TinyImageNetDataset, self).__init__(root, transform=transform, target_transform=target_transform)

        self.dataset_path = os.path.join(root, self.base_folder)
        self.loader = default_loader
        self.split = verify_str_arg(split, "split", ("train", "val",))

        if self._check_integrity():
            print('Files already downloaded and verified.')
        elif download:
            self._download()
        else:
            raise RuntimeError(
                'Dataset not found. You can use download=True to download it.')
        if not os.path.isdir(self.dataset_path):
            print('Extracting...')
            extract_archive(os.path.join(root, self.filename))

        _, class_to_idx = TinyImageNetDataset.find_classes(os.path.join(self.dataset_path, 'wnids.txt'))

        self.data = TinyImageNetDataset.make_dataset(self.root, self.base_folder, self.split, class_to_idx)


        self.targets = [s[1] for s in self.data]

    def _download(self):
        print('Downloading...')
        download_url(self.url, root=self.root, filename=self.filename)
        print('Extracting...')
        extract_archive(os.path.join(self.root, self.filename))

    def _check_integrity(self):
        return check_integrity(os.path.join(self.root, self.filename), self.md5)

    def __getitem__(self, index):
        img_path, target = self.data[index]
        image = self.loader(img_path)

        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return image, target

    def __len__(self):
        return len(self.data)

    @staticmethod
    def find_classes(class_file):
        with open(class_file) as r:
            classes = list(map(lambda s: s.strip(), r.readlines()))

        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}

        return classes, class_to_idx

    @staticmethod
    def make_dataset(root, base_folder, dirname, class_to_idx):
        images = []
        dir_path = os.path.join(root, base_folder, dirname)

        if dirname == 'train':
            for fname in sorted(os.listdir(dir_path)):
                cls_fpath = os.path.join(dir_path, fname)
                if os.path.isdir(cls_fpath):
                    cls_imgs_path = os.path.join(cls_fpath, 'images')
                    for imgname in sorted(os.listdir(cls_imgs_path)):
                        path = os.path.join(cls_imgs_path, imgname)
                        item = (path, class_to_idx[fname])
                        images.append(item)
        else:
            imgs_path = os.path.join(dir_path, 'images')
            imgs_annotations = os.path.join(dir_path, 'val_annotations.txt')

            with open(imgs_annotations) as r:
                data_info = map(lambda s: s.split('\t'), r.readlines())

            cls_map = {line_data[0]: line_data[1] for line_data in data_info}

            for imgname in sorted(os.listdir(imgs_path)):
                path = os.path.join(imgs_path, imgname)
                item = (path, class_to_idx[cls_map[imgname]])
                images.append(item)

        return images

# ----------------------------
# TinyImageNet DataModule
# ----------------------------
class TinyImageNetDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int, num_workers: int = 4,
                 mixup: bool = False, cutmix: bool = False, mix_alpha: float = 1.0):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.mixup = mixup
        self.cutmix = cutmix
        self.mix_alpha = mix_alpha

    def setup(self, stage=None):
        mean = (0.4802, 0.4481, 0.3975)
        std  = (0.2302, 0.2265, 0.2262)

        train_tfm = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(64, padding=4),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        val_tfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        # If needed, restructure val set (must be done once outside the module)
        self.train_ds = TinyImageNetDataset(self.data_dir, split='train', transform=train_tfm, download=True)
        self.val_ds   = TinyImageNetDataset(self.data_dir, split='val', transform=val_tfm, download=True)

    def train_dataloader(self):
        collate = partial(
            mix_collate,
            cutmix=self.cutmix,
            mixup=self.mixup,
            mix_alpha=self.mix_alpha,
        )
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=True if self.num_workers > 0 else False,
            collate_fn=collate,
        )

    def val_dataloader(self, batch_size: int = None):
        return DataLoader(
            self.val_ds,
            batch_size=batch_size or self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
        )
    
# ----------------------------
# MNIST DataModule
# ----------------------------
class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int, num_workers: int = 4,
                 mixup: bool = False, cutmix: bool = False, mix_alpha: float = 1.0):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        train_tfm = transforms.Compose([
            transforms.RandomCrop(28, padding=2),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        test_tfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.train_ds = datasets.MNIST(root=self.data_dir, train=True, download=True, transform=train_tfm)
        self.val_ds   = datasets.MNIST(root=self.data_dir, train=False, download=True, transform=test_tfm)
        self.test_ds  = self.val_ds  # same for MNIST

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers, pin_memory=True,persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=True,persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=True,persistent_workers=True)

# ----------------------------
# ImageNet DataModule
# ----------------------------
class ImageNetDataModule(pl.LightningDataModule):
    """
    <data_dir>/
    train/
        class_0/ *.jpeg
        class_1/ *.jpeg
        ...
    val/
        class_0/ *.jpeg
        class_1/ *.jpeg
        ...

    ImageNet (1k) DataModule that matches your CIFAR style:
    - train: RandomResizedCrop(224) + HFlip + Normalize
    - val:   Resize(256) -> CenterCrop(224) + Normalize
    - optional mixup/cutmix via your `mix_collate` and (mixup, cutmix, mix_alpha)
    """
    def __init__(self,
                 data_dir: str,
                 batch_size: int,
                 num_workers: int = 8,
                 mixup: bool = False,
                 cutmix: bool = False,
                 mix_alpha: float = 0.2,
                 image_size: int = 224,
                 val_resize: int = 256):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.mixup = mixup
        self.cutmix = cutmix
        self.mix_alpha = mix_alpha
        self.image_size = image_size
        self.val_resize = val_resize

        # Standard ImageNet stats
        self.mean = (0.485, 0.456, 0.406)
        self.std  = (0.229, 0.224, 0.225)

        # Will be set in setup()
        self.train_ds = None
        self.val_ds = None

    def setup(self, stage=None):
        train_tf = transforms.Compose([
            transforms.RandomResizedCrop(self.image_size, interpolation=InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std),
        ])
        val_tf = transforms.Compose([
            transforms.Resize(self.val_resize, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std),
        ])

        # ImageFolder expects subfolders per class
        self.train_ds = datasets.ImageFolder(root=f"{self.data_dir}/train", transform=train_tf)
        self.val_ds   = datasets.ImageFolder(root=f"{self.data_dir}/val",   transform=val_tf)

        # (Optional) you can inspect class count if needed:
        # self.num_classes = len(self.train_ds.classes)

    def train_dataloader(self):
        # Reuse your mix_collate just like CIFAR100DataModule
        collate = partial(
            mix_collate,
            cutmix=self.cutmix,
            mixup=self.mixup,
            mix_alpha=self.mix_alpha,
        )
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=True if self.num_workers > 0 else False,
            collate_fn=collate,
        )

    def val_dataloader(self, batch_size: int = None):
        bs = batch_size if batch_size is not None else self.batch_size
        return DataLoader(
            self.val_ds,
            batch_size=bs,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False
        )

# ----------------------------
# Export callback (save best FP & ternary)
# ----------------------------
class ExportBestTernary(Callback):
    def __init__(self, out_dir: str, monitor: str = "val/acc_tern", mode: str = "max"):
        super().__init__()
        self.out_dir, self.monitor, self.mode = out_dir, monitor, mode
        self.best = None
        os.makedirs(out_dir, exist_ok=True)

    def _is_better(self, current, best):
        if best is None: return True
        return (current > best) if self.mode == "max" else (current < best)

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module):
        metrics = trainer.callback_metrics
        if self.monitor not in metrics: return
        current = metrics[self.monitor].item()
        dataset_name = pl_module.dataset_name if hasattr(pl_module,'dataset_name') else ''
        model_name = pl_module.model_name if hasattr(pl_module,'model_name') else ''
        model_size = pl_module.model_size if hasattr(pl_module,'model_size') else ''
        if self._is_better(current, self.best):
            self.best = current
            # save FP student
            best_fp = copy.deepcopy(pl_module.student).cpu().eval()
            fp_path = os.path.join(self.out_dir, f"bit_{model_name}_{model_size}_{dataset_name}_best_fp.pt")
            torch.save({"model": best_fp.state_dict(), "acc_tern": current}, fp_path)
            pl_module.print(f"[OK] saved {fp_path} (val/acc_tern={current*100:.2f}%)")
            # save ternary PoT export
            tern = convert_to_ternary(copy.deepcopy(best_fp)).cpu().eval()
            tern_path = os.path.join(self.out_dir,
                                     f"bit_{model_name}_{model_size}_{dataset_name}_ternary_val_acc@{current*100:.2f}.pt")
            torch.save({"model": tern.state_dict(), "acc_tern": current}, tern_path)
            pl_module.print(f"[OK] exported ternary PoT -> {tern_path}")

# ----------------------------
# LightningModule: KD + hints + ternary eval/export
# ----------------------------
class CommonTrainConfig(BaseModel):
    data_dir: str = "./data"
    export_dir: str = "./ckpt_c100"

    epochs: int = Field(200, ge=1)
    batch_size: int = Field(512, ge=1)

    lr: float = Field(2e-1, gt=0)
    wd: float = Field(5e-4, ge=0)

    label_smoothing: float = Field(0.1, ge=0.0, le=1.0)
    alpha_kd: float = 0.3
    alpha_hint: float = 0.05
    T: float = Field(4.0, gt=0)

    scale_op: Literal["mean", "median"] = "median"

    amp: bool = False
    cpu: bool = False
    mixup: bool = False
    cutmix: bool = False

    mix_alpha: float = Field(1.0, ge=0.0)

    seed: int = 42
    gpus: int = Field(
        1,
        description="Number of GPUs to use (1 = default, -1 = all available)",
    )
    strategy: Literal["auto", "ddp", "ddp_spawn", "fsdp"] = "auto"

class LitBitConfig(BaseModel):
    lr: float
    wd: float
    epochs: int

    label_smoothing: float = 0.1
    alpha_kd: float = 0.7
    alpha_hint: float = 0.05
    T: float = 4.0
    scale_op: str = "median"

    width_mult: float = 1.0
    amp: bool = True
    export_dir: str = "./ckpt_c100_mbv2"

    student: Optional[Any] = None
    teacher: Optional[Any] = None

    dataset_name: str = ""
    model_name: str = ""
    model_size: str = ""

    hint_points: List[str|Tuple] = Field(default_factory=list)
    num_classes: int = -1

class LitBit(pl.LightningModule):
    def __init__(self, config:LitBitConfig):
        super().__init__()
        if type(config) is not dict:
            config = config.model_dump()
        config = LitBitConfig.model_validate(config)
        self.save_hyperparameters(ignore=['student','teacher','_t_feats','_s_feats',
                                          '_t_handles','_s_handles','_ternary_snapshot'])
        self.scale_op = config.scale_op
        self.student:nn.Module = config.student
        self.teacher = config.teacher
        self.dataset_name = config.dataset_name
        self.model_name = config.model_name
        self.model_size = config.model_size
        self.alpha_kd = config.alpha_kd
        self.alpha_hint = config.alpha_hint
        if self.alpha_kd<=0 and self.alpha_hint<=0:
            self.teacher=None
        if self.teacher:
            for p in self.teacher.parameters(): p.requires_grad_(False)
        self.ce = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing).eval()
        self.kd = KDLoss(T=T).eval()
        self.hint = AdaptiveHintLoss().eval()
        self.hint_points = config.hint_points
        self.acc_fp = MulticlassAccuracy(num_classes=config.num_classes).eval()
                        # average='micro', multidim_average='global', top_k=1).eval()

        self._ternary_snapshot = None
        self._t_feats = {}
        self._s_feats = {}
        self._t_handles = []
        self._s_handles = []
        self.t_acc_fp = None
        self.lr = config.lr
        self.wd = config.wd
        self.epochs = config.epochs

    def setup(self, stage=None):
        if self.teacher:
            self.teacher = self.teacher.to(self.device).eval()
            if self.alpha_hint>0:
                self._s_handles = make_feature_hooks(self.student, self.hint_points, self._s_feats, 0)
                self._t_handles = make_feature_hooks(self.teacher, self.hint_points, self._t_feats, 1)

    def teardown(self, stage=None):
        for h in getattr(self, "_t_handles", []):
            try: h.remove()
            except: pass
        for h in getattr(self, "_s_handles", []):
            try: h.remove()
            except: pass

    def forward(self, x):
        return self.student(x)

    def on_fit_start(self):
        n_params = sum(p.numel() for p in self.student.parameters())
        acc_name = self.trainer.accelerator.__class__.__name__
        strategy_name = self.trainer.strategy.__class__.__name__
        num_devices = getattr(self.trainer, "num_devices", None) or len(self.trainer.devices or [])
        dev_str = str(self.device)
        cuda_name = ""
        if torch.cuda.is_available() and "cuda" in dev_str:
            try: cuda_name = f" | CUDA: {torch.cuda.get_device_name(self.device.index or 0)}"
            except: pass
        self.print(f"Model params: {n_params/1e6:.2f}M | Accelerator: {acc_name} | Devices: {num_devices} | Strategy: {strategy_name} | Device: {dev_str}{cuda_name}")

    @torch.no_grad()
    def _clone_student(self):
        clone:nn.Module = self.student.clone()
        clone.load_state_dict(self.student.state_dict(), strict=True)
        clone = convert_to_ternary(clone)
        return clone.eval().to(self.device)

    def on_validation_epoch_start(self):
        print()
        self._ternary_snapshot = self._clone_student()

    def training_step(self, batch, batch_idx):
        x, y = batch
        is_mix = isinstance(y, tuple)
        if is_mix:
            y_a, y_b, lam = y

        # use_amp = bool(self.amp and "cuda" in str(self.device))

        z_s = self.student(x)
        z_t = None
        
        if is_mix:
            loss_ce = lam * self.ce(z_s, y_a) + (1 - lam) * self.ce(z_s, y_b)
        else:
            loss_ce = self.ce(z_s, y)

        if self.teacher:
            z_t = self.teacher(x)

        loss_kd = 0.0
        if z_t and self.alpha_kd > 0:
            loss_kd = self.kd(z_s.float(), z_t.float())

        # Hints
        loss_hint = 0.0
        if self.alpha_hint>0 and len(self.hint_points)>0:
            for n in self.hint_points:                
                sn = tn = n
                if type(n)==tuple:
                    sn,tn = n
                if sn not in self._s_feats:
                    raise ValueError(f"Hint point {sn} not found in student features of {self._s_feats}.")
                if tn not in self._t_feats:
                    raise ValueError(f"Hint point {tn} not found in teacher features of {self._t_feats}.")
                loss_hint = loss_hint + self.hint(sn, self._s_feats[sn].float(), self._t_feats[tn].float())

        loss = (1.0 - self.alpha_kd) * loss_ce + self.alpha_kd * loss_kd + self.alpha_hint * loss_hint

        logd = {}
        logd["train/loss"] = loss
        if loss_ce>0.0 : logd["train/ce"] = loss_ce
        if loss_kd>0.0 : logd["train/kd"] = loss_kd
        if loss_hint>0.0 : logd["train/hint"] = loss_hint
        logd["lr"] = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log_dict(logd, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=x.size(0))
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        def log_val(n,model=None,acc=None,x=x,y=y):
            if acc is None and model is not None:
                # acc = (model(x).argmax(1)==y).sum()/x.size(0)
                acc = self.acc_fp(model(x), y)
            if acc is not None:
                self.log(f"val/{n}", acc, on_step=False, on_epoch=True, prog_bar=True, batch_size=x.size(0))
            return acc
        
        with torch.no_grad():
            acc_fp = log_val("acc_fp", self.student)
            acc_t = log_val("acc_tern", self._ternary_snapshot)
            if self.alpha_kd>0:
                if self.t_acc_fp is None and self.teacher:
                    self.t_acc_fp = log_val("t_acc_fp", self.teacher)
                else:
                    log_val("t_acc_fp", acc=self.t_acc_fp)

    def configure_optimizers(self):
        opt = torch.optim.SGD(
            list(self.student.parameters()) + list(self.hint.parameters()),
            lr=self.lr, momentum=0.9, weight_decay=self.wd, nesterov=True
        )
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.epochs)
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sched, "interval": "epoch", "monitor": "val/acc_tern"},
        }

# ----------------------------
# Common CLI utilities
# ----------------------------
# def add_common_args(parser):
#     """Add common training arguments to an argument parser."""
#     parser.add_argument("--data", type=str, default="./data")
#     parser.add_argument("--out",  type=str, default="./ckpt_c100")
#     parser.add_argument("--epochs", type=int, default=200)
#     parser.add_argument("--batch-size", type=int, default=512)
#     parser.add_argument("--lr", type=float, default=2e-1)
#     parser.add_argument("--wd", type=float, default=5e-4)
#     parser.add_argument("--label-smoothing", type=float, default=0.1)
#     parser.add_argument("--mix_alpha-kd", type=float, default=0.3)
#     parser.add_argument("--mix_alpha-hint", type=float, default=0.05)
#     parser.add_argument("--T", type=float, default=4.0)
#     parser.add_argument("--scale-op", type=str, default="median", choices=["mean","median"])
#     parser.add_argument("--amp", action="store_true")
#     parser.add_argument("--cpu", action="store_true")
#     parser.add_argument("--mixup", action="store_true")
#     parser.add_argument("--cutmix", action="store_true")
#     parser.add_argument("--mix-mix_alpha", type=float, default=1.0)
#     parser.add_argument("--seed", type=int, default=42)
#     parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs to use (default: 1, use -1 for all available)")
#     parser.add_argument("--strategy", type=str, default="auto", choices=["auto", "ddp", "ddp_spawn", "fsdp"],
#                         help="Distributed training strategy (default: auto)")
#     return parser

def setup_trainer(args:CommonTrainConfig, lit_module, dm = None):
    """
    Setup common PyTorch Lightning training components.

    Args:
        args: Parsed command-line arguments
        lit_module: Lightning module to train

    Returns:
        tuple: (trainer, datamodule)
    """
    pl.seed_everything(args.seed, workers=True)
    if dm is None:
        dm = CIFAR100DataModule(
            data_dir=args.data_dir, batch_size=args.batch_size, num_workers=4,
            mixup=args.mixup, cutmix=args.cutmix, mix_alpha=args.mix_alpha
        )

    os.makedirs(args.export_dir, exist_ok=True)
    logger = CSVLogger(save_dir=args.export_dir, name="logs")
    ckpt_cb = ModelCheckpoint(monitor="val/acc_tern", mode="max", save_top_k=1, save_last=True)
    lr_cb = LearningRateMonitor(logging_interval="epoch")
    export_cb = ExportBestTernary(args.export_dir, monitor="val/acc_tern", mode="max")
    callbacks = [ckpt_cb, lr_cb, export_cb]

    accelerator = "cpu" if args.cpu else "auto"
    precision = "16-mixed" if args.amp else "32-true"

    # Multi-GPU setup
    devices = args.gpus if hasattr(args, 'gpus') else 1
    strategy = args.strategy if hasattr(args, 'strategy') else "auto"

    # Use appropriate strategy for multi-GPU training
    import sys
    if devices > 1 or devices == -1:
        if strategy == "auto":
            # Check if NCCL is available (for CUDA GPUs)
            if sys.platform == "win32":
                # Windows doesn't support NCCL, must use gloo backend
                from pytorch_lightning.strategies import DDPStrategy
                strategy = DDPStrategy(process_group_backend="gloo")
                print(f"[Multi-GPU] Windows detected, using DDP with gloo backend")
            else:
                try:
                    import torch.distributed
                    if torch.cuda.is_available() and torch.distributed.is_nccl_available():
                        strategy = "ddp"
                    else:
                        from pytorch_lightning.strategies import DDPStrategy
                        strategy = DDPStrategy(process_group_backend="gloo")
                        print(f"[Multi-GPU] NCCL not available, using DDP with gloo backend")
                except:
                    from pytorch_lightning.strategies import DDPStrategy
                    strategy = DDPStrategy(process_group_backend="gloo")
                    print(f"[Multi-GPU] Using DDP with gloo backend")

        print(f"[Multi-GPU] Training on {devices if devices > 0 else 'all'} GPUs")

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator=accelerator,
        devices=devices,
        strategy=strategy,
        precision=precision,
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=50,
        deterministic=False,
        sync_batchnorm=True if (devices > 1 or devices == -1) else False,
    )

    return trainer, dm

def GUI_tool(model,
             resize=None,
             class_names=None,
             preprocess=None,
             device=None,
             topk=5,
             logits_fn=None,
             title="Classifier GUI (no resize to model)"):
    """
    model:      torch.nn.Module (put in eval mode automatically)
    class_names:list[str] or None
    preprocess: callable(PIL.Image) -> torch.Tensor (C,H,W); default = ToTensor() only
    device:     torch.device or str; default = model's first parameter device (else cpu)
    topk:       int
    logits_fn:  callable(tensor[N,C,H,W]) -> logits[N,num_classes]; default = model(x)
                (useful to inject an adapter, e.g. lambda x: adapter(model(x)))
    title:      window title

    IMPORTANT: The image fed to the model is NOT resized here.
               The preview image may be scaled ONLY for display.
    """
    # --- Setup model/device ---
    model.eval()
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
    if isinstance(device, str):
        device = torch.device(device)

    # Default preprocess: just ToTensor (0..1), no resize
    if preprocess is None:
        preprocess = T.ToTensor()

    # Helpers
    def load_image(path,resize=resize):
        img = Image.open(path)
        if img.mode != "RGB":
            img = img.convert("RGB")
        if resize:
            img = img.resize(resize)
        return img

    def tensor_from_pil(img):
        # preprocess returns CxHxW
        t = preprocess(img)
        if t.ndim == 3:
            t = t.unsqueeze(0)  # NxCxHxW
        return t

    # --- Tkinter UI ---
    root = tk.Tk()
    root.title(title)

    # Main frames
    frm_top = ttk.Frame(root, padding=8)
    frm_top.pack(side=tk.TOP, fill=tk.X)

    frm_mid = ttk.Frame(root, padding=8)
    frm_mid.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    frm_right = ttk.Frame(frm_mid, padding=(8, 0))
    frm_right.pack(side=tk.RIGHT, fill=tk.Y)

    # Canvas to show image (scaled only for display)
    canvas_size = 448  # display only, not used for model input
    canvas = tk.Canvas(frm_mid, width=canvas_size, height=canvas_size, bg="#222")
    canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    # Controls
    btn_open = ttk.Button(frm_top, text="Open image…")
    btn_predict = ttk.Button(frm_top, text="Predict", state=tk.DISABLED)
    lbl_path = ttk.Label(frm_top, text="No file selected", width=60)

    btn_open.pack(side=tk.LEFT)
    btn_predict.pack(side=tk.LEFT, padx=(8, 0))
    lbl_path.pack(side=tk.LEFT, padx=(12, 0))

    # Results box
    lbl_res = ttk.Label(frm_right, text="Top-K", font=("TkDefaultFont", 10, "bold"))
    lbl_res.pack(anchor="nw")
    txt = tk.Text(frm_right, width=40, height=24, wrap="word")
    txt.pack(fill=tk.Y, expand=False)

    # State
    state = {"pil": None, "path": None, "photo": None}

    def show_image_on_canvas(pil_img):
        # scale to fit canvas while keeping aspect ratio (DISPLAY ONLY)
        w, h = pil_img.size
        scale = min(canvas_size / max(w, 1), canvas_size / max(h, 1), 1.0)
        disp_w, disp_h = int(w * scale), int(h * scale)
        preview = pil_img if (disp_w == w and disp_h == h) else pil_img.resize((disp_w, disp_h))
        photo = ImageTk.PhotoImage(preview)
        canvas.delete("all")
        canvas.create_image(canvas_size // 2, canvas_size // 2, image=photo, anchor="center")
        state["photo"] = photo  # keep ref

    def on_open():
        path = filedialog.askopenfilename(
            title="Choose an image",
            filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp;*.gif;*.webp;*.tif;*.tiff"), ("All files", "*.*")]
        )
        if not path:
            return
        try:
            img = load_image(path)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image:\n{e}")
            return
        state["pil"] = img
        state["path"] = path
        lbl_path.config(text=os.path.basename(path))
        show_image_on_canvas(img)
        btn_predict.config(state=tk.NORMAL)

    def safe_class_name(i):
        if class_names is None:
            return str(i)
        try:
            return str(class_names[i])
        except Exception:
            return str(i)

    @torch.no_grad()
    def on_predict():
        if state["pil"] is None:
            return
        img = state["pil"]

        try:
            x = tensor_from_pil(img).to(device)  # NxCxHxW, original size
        except Exception as e:
            messagebox.showerror("Preprocess error", f"Failed to preprocess image:\n{e}")
            return

        try:
            logits = logits_fn(x) if callable(logits_fn) else model(x)
        except Exception as e:
            messagebox.showerror(
                "Model error",
                f"Forward pass failed.\n\nIf your model requires a fixed size, "
                f"you must pass a preprocess that resizes/crops accordingly.\n\nError:\n{e}"
            )
            return

        # top-K
        probs = F.softmax(logits, dim=-1)
        k = min(topk, probs.shape[-1])
        vals, idxs = probs.topk(k, dim=-1)
        vals, idxs = vals[0].tolist(), idxs[0].tolist()

        # Render results
        txt.delete("1.0", tk.END)
        for rank, (p, i) in enumerate(zip(vals, idxs), start=1):
            txt.insert(tk.END, f"{rank:>2}. {safe_class_name(i)}  —  {p*100:.2f}%\n")

    btn_open.config(command=on_open)
    btn_predict.config(command=on_predict)

    root.mainloop()

def load_tiny200_to_in1k_map(path: str, expected_out: int = 200) -> List[List[int]]:
    """
    Reads a mapping text file where each *line* corresponds to one Tiny-ImageNet class (0..199),
    and contains one or more ImageNet-1k indices (0..999) to aggregate from.
    Lines can use commas/spaces/colons, and may include comments after '#'.

    Example lines:
      8
      65, 67
      42 417 901   # comment
      12: 12

    Returns: list of length 200; each item is a sorted list of unique int indices.
    """
    mapping: List[List[int]] = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            # extract all integers on the line
            ids = [int(s) for s in re.findall(r"\d+", line)]
            ids = sorted(set(ids))
            mapping.append(ids)

    if len(mapping) != expected_out:
        warnings.warn(
            f"Expected {expected_out} mapping rows, found {len(mapping)}. "
            "Ensure the file has one (non-empty) line per Tiny-ImageNet class."
        )

    # sanity checks
    for i, ids in enumerate(mapping):
        if not ids:
            raise ValueError(f"Mapping row {i} is empty (no ImageNet-1k indices).")
        for k in ids:
            if not (0 <= k < 1000):
                raise ValueError(f"Invalid IN1k index {k} in row {i}; expected 0..999.")
    return mapping

def build_projection_matrix(mapping: Sequence[Sequence[int]],
                            in_dim: int = 1000,
                            dtype: torch.dtype = torch.float32,
                            device: Optional[torch.device] = None) -> torch.Tensor:
    """
    Builds a (out_dim x in_dim) binary matrix M where M[j, i] = 1 if IN1k idx i maps to Tiny200 cls j.
    Use with probability vectors: p200 = p1k @ M.T  (or p1k.matmul(M.T))
    """
    out_dim = len(mapping)
    M = torch.zeros((out_dim, in_dim), dtype=dtype, device=device)
    for j, ids in enumerate(mapping):
        M[j, ids] = 1.0
    # Optional: warn if any IN1k index is assigned to multiple Tiny classes (overlap)
    overlaps = (M.sum(0) > 1).nonzero(as_tuple=False).flatten()
    if overlaps.numel() > 0:
        warnings.warn(f"{overlaps.numel()} ImageNet-1k indices map to multiple Tiny classes; "
                      "their probability mass will be counted multiple times before renormalization.")
    return M

class IN1kToTiny200Adapter(nn.Module):
    """
    Adapts teacher outputs from 1000 classes to 200 using a mapping.
    Default behavior:
      - apply temperature (T) to 1k logits,
      - softmax -> p1k,
      - aggregate: p200_raw = p1k @ M.T,
      - renormalize to sum to 1 (per sample),
      - return either probabilities (p200) or logits (log p200).
    """
    def __init__(self,
                 mapping: Sequence[Sequence[int]],
                 temperature: float = 1.0,
                 renormalize: bool = True,
                 in_dim: int = 1000,
                 device: Optional[torch.device] = None):
        super().__init__()
        self.temperature = float(temperature)
        self.renormalize = bool(renormalize)
        M = build_projection_matrix(mapping, in_dim=in_dim, device=device)
        # register as buffer so it moves with .to(device) and gets saved in state_dict
        self.register_buffer("proj", M)  # shape: (200, 1000)

    @torch.no_grad()
    def probs(self, logits_1k: torch.Tensor) -> torch.Tensor:
        """Return 200-way probabilities (with temperature applied at 1k-level)."""
        t = self.temperature
        if t != 1.0:
            logits_1k = logits_1k / t
        p1k = logits_1k.softmax(dim=-1)                  # (N, 1000)
        p200 = p1k.matmul(self.proj.t())                 # (N, 200), sum probabilities per mapping
        if self.renormalize:
            denom = p200.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            p200 = p200 / denom
        return p200

    @torch.no_grad()
    def forward(self, logits_1k: torch.Tensor, return_logits: bool = True) -> torch.Tensor:
        """
        If return_logits=True: returns log(p200 + eps), which are valid "logits"
        (since softmax(log p) = p). Else returns probabilities p200.
        """
        p200 = self.probs(logits_1k)
        if return_logits:
            return (p200 + 1e-12).log()
        return p200
# if __name__ == '__main__':
#     freeze_support()
#     dm = TinyImageNetDataModule("./data",4)
#     dm.setup()
#     for data,label in dm.train_dataloader():
#         break
#     print(data)
#     print(label)
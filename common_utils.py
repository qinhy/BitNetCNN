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
from matplotlib import pyplot as plt
from pydantic import BaseModel, Field
import torch

torch.set_float32_matmul_precision('high')
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import v2
from torch.utils.data import default_collate
import pytorch_lightning as pl
from pytorch_lightning import Callback
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from torchvision.transforms import InterpolationMode
from torchmetrics.classification import MulticlassAccuracy
from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import extract_archive, check_integrity, download_url, verify_str_arg

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from accelerate import Accelerator
from tqdm.auto import tqdm

def snap_model(model):
    def snapshot_params(m):
        return {k: v.detach().clone().cpu() for k, v in m.named_parameters()}
    param = snapshot_params(model)
    def snapshot_buffers(m):
        return {k: v.detach().clone().cpu() for k, v in m.named_buffers()}
    buf = snapshot_buffers(model)
    return param,buf

@torch.no_grad()
def recover_snap(model: nn.Module, params_snap, bufs_snap):
    # restore parameters
    for name, p in model.named_parameters():
        if name in params_snap:
            p.copy_(params_snap[name].to(p.device))

    # restore buffers (e.g. BN running_mean/var)
    for name, b in model.named_buffers():
        if name in bufs_snap:
            b.copy_(bufs_snap[name].to(b.device))

class Cutout(nn.Module):
    """Simple Cutout transform for tensors [C, H, W].

    size:
        - int  -> square hole (size x size)
        - (h, w) -> rectangular hole (height x width)
    """
    def __init__(self, size: Union[int, Tuple[int, int]] = 16):
        super().__init__()
        if isinstance(size, int):
            self.size = (size, size)
        else:
            if len(size) != 2:
                raise ValueError(f"Cutout size must be int or (h, w) tuple, got: {size}")
            self.size = tuple(size)
            
    def forward(self, img: torch.Tensor) -> torch.Tensor:
        # img is expected to be a Tensor after ToTensor()
        if not torch.is_tensor(img):
            return img

        _, h, w = img.shape
        if h == 0 or w == 0:
            return img

        mask_h, mask_w = self.size
        mask_h_half = mask_h // 2
        mask_w_half = mask_w // 2

        cy = torch.randint(0, h, (1,)).item()
        cx = torch.randint(0, w, (1,)).item()

        y1 = max(0, cy - mask_h_half)
        y2 = min(h, cy + mask_h_half)
        x1 = max(0, cx - mask_w_half)
        x2 = min(w, cx + mask_w_half)

        img[:, y1:y2, x1:x2] = 0.0
        return img
    
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
# class Bit:
#     """
#     Collection of classes for bit-level quantization of neural networks.
#     Includes fake-quant building blocks, inference modules, and training modules.
#     """
#     # ----------------------------
#     # Fake-quant building blocks (QAT) — activation quantization removed
#     # ----------------------------
#     class Bit1p58Weight(nn.Module):
#         """1.58-bit (ternary) weight quantizer with per-out-channel scaling."""
#         def __init__(self, dim=0, scale_op="median"):
#             super().__init__()
#             self.dim = dim
#             self.scale_op = scale_op

#         def forward(self, w):
#             s = _reduce_abs(w, keep_dim=self.dim, op=self.scale_op)
#             w_bar = (w / s).detach()
#             w_q = torch.round(w_bar).clamp_(-1, 1)
#             dw = (w_q * s - w).detach()
#             return w + dw

#     # ----------------------------
#     # Inference (frozen) ternary modules — no activation quantization
#     # ----------------------------
#     class Conv2dInfer(nn.Module):
#         """
#         Frozen ternary conv:
#         y = (Conv(x, Wq) * s_per_out) + b
#         Wq in {-1,0,+1} stored as int8. s is float per output channel.
#         """
#         def __init__(self, w_q, s, bias, stride, padding, dilation, groups):
#             super().__init__()
#             # Make them Parameters so param counters include them (but keep frozen)
#             self.w_q  = nn.Parameter(w_q.to(torch.int8), requires_grad=False)   # [out,in,kh,kw]
#             self.s    = nn.Parameter(s,                  requires_grad=False) # [out,1,1]
#             if bias is None:
#                 self.register_parameter("bias", None)
#             else:
#                 self.bias = nn.Parameter(bias, requires_grad=False)

#             self.stride, self.padding, self.dilation, self.groups = stride, padding, dilation, groups
#             # Optional cache for the float view (not in state_dict, not counted as param)
#             self.register_buffer("_w_q_float", None, persistent=False)

#         def _weight(self, dtype, device):
#             if self._w_q_float is None or self._w_q_float.dtype != dtype or self._w_q_float.device != device:
#                 self._w_q_float = self.w_q.to(device=device, dtype=dtype)
#             return self._w_q_float

#         def forward(self, x):
#             w = self._weight(x.dtype, x.device)
#             y = F.conv2d(x, w, None, self.stride, self.padding, self.dilation, self.groups)
#             y = y * self.s.to(dtype=y.dtype, device=y.device)
#             if self.bias is not None:
#                 y = y + self.bias.to(dtype=y.dtype, device=y.device).view(1, -1, 1, 1)
#             return y

#     class LinearInfer(nn.Module):
#         """Frozen ternary linear: y = (x @ Wq^T) * s + b"""
#         def __init__(self, w_q, s, bias):
#             super().__init__()
#             self.w_q = nn.Parameter(w_q.to(torch.int8), requires_grad=False)   # [out,in]
#             self.s   = nn.Parameter(s,                    requires_grad=False)  # [out]
#             if bias is None:
#                 self.register_parameter("bias", None)
#             else:
#                 self.bias = nn.Parameter(bias, requires_grad=False)

#             self.register_buffer("_w_q_float", None, persistent=False)

#         def _weight(self, dtype, device):
#             if self._w_q_float is None or self._w_q_float.dtype != dtype or self._w_q_float.device != device:
#                 self._w_q_float = self.w_q.to(device=device, dtype=dtype)
#             return self._w_q_float

#         def forward(self, x):
#             w = self._weight(x.dtype, x.device)
#             y = F.linear(x, w, bias=None)
#             y = y * self.s.to(dtype=y.dtype, device=y.device)
#             if self.bias is not None:
#                 y = y + self.bias.to(dtype=y.dtype, device=y.device)
#             return y
        
#     class Conv2dInferP2(nn.Module):
#         """
#         Ternary conv with power-of-two scales:
#         y = Conv(x, Wq) * 2^{s_exp} + b
#         Wq in {-1,0,+1} as int8. s_exp is per-out exponent [out,1,1].
#         """
#         def __init__(self, w_q, s_exp, bias, stride, padding, dilation, groups):
#             super().__init__()
#             # Counted as params but frozen
#             self.w_q  = nn.Parameter(w_q.to(torch.int8), requires_grad=False)      # [out,in,kh,kw]
#             self.s_exp = nn.Parameter(s_exp.to(torch.int8), requires_grad=False)   # [out,1,1]
#             if bias is None:
#                 self.register_parameter("bias", None)
#             else:
#                 self.bias = nn.Parameter(bias, requires_grad=False)                # [out]

#             self.stride, self.padding, self.dilation, self.groups = stride, padding, dilation, groups
#             # Cache float weights per (device,dtype); not saved, not counted
#             self.register_buffer("_w_q_float", None, persistent=False)

#         def _weight(self, dtype, device):
#             if self._w_q_float is None or self._w_q_float.dtype != dtype or self._w_q_float.device != device:
#                 self._w_q_float = self.w_q.to(device=device, dtype=dtype)
#             return self._w_q_float

#         def forward(self, x):
#             w = self._weight(x.dtype, x.device)
#             y = F.conv2d(x, w, None, self.stride, self.padding, self.dilation, self.groups)
#             y = torch.ldexp(y, self.s_exp.to(torch.int32, device=y.device))  # broadcast [out,1,1]
#             if self.bias is not None:
#                 y = y + self.bias.to(dtype=y.dtype, device=y.device).view(1, -1, 1, 1)
#             return y

#     class LinearInferP2(nn.Module):
#         """Ternary linear with power-of-two output scales: y = (x @ Wq^T) * 2^{s_exp} + b"""
#         def __init__(self, w_q, s_exp, bias):
#             super().__init__()
#             self.w_q   = nn.Parameter(w_q.to(torch.int8), requires_grad=False)     # [out,in]
#             self.s_exp = nn.Parameter(s_exp.to(torch.int8), requires_grad=False)   # [out]
#             if bias is None:
#                 self.register_parameter("bias", None)
#             else:
#                 self.bias = nn.Parameter(bias, requires_grad=False)                # [out]
#             self.register_buffer("_w_q_float", None, persistent=False)

#         def _weight(self, dtype, device):
#             if self._w_q_float is None or self._w_q_float.dtype != dtype or self._w_q_float.device != device:
#                 self._w_q_float = self.w_q.to(device=device, dtype=dtype)
#             return self._w_q_float

#         def forward(self, x):
#             w = self._weight(x.dtype, x.device)
#             y = F.linear(x, w, bias=None)
#             y = torch.ldexp(y, self.s_exp.to(torch.int32, device=y.device))  # broadcast [out]
#             if self.bias is not None:
#                 y = y + self.bias.to(dtype=y.dtype, device=y.device)
#             return y

#     # ----------------------------
#     # Train-time modules (no BatchNorm), activation quantization removed
#     # ----------------------------
#     class Conv2d(nn.Module):
#         """
#         Conv2d with ternary weights (fake-quant for training).
#         No BatchNorm inside. Add your own nonlinearity outside if desired.
#         """
#         def __init__(self, in_c, out_c, kernel_size, stride=1, padding=0, dilation=1, groups=1,
#                     bias=True, scale_op="median"):
#             super().__init__()
#             if isinstance(kernel_size, int):
#                 kh = kw = kernel_size
#             else:
#                 kh, kw = kernel_size
#             self.weight = nn.Parameter(torch.empty(out_c, in_c // groups, kh, kw))
#             nn.init.kaiming_normal_(self.weight, nonlinearity="relu")
#             self.bias = nn.Parameter(torch.zeros(out_c)) if bias else None
#             self.w_q = Bit.Bit1p58Weight(dim=0, scale_op=scale_op)
#             self.stride, self.padding, self.dilation, self.groups = stride, padding, dilation, groups
#             self.scale_op = scale_op

#         def forward(self, x):
#             wq = self.w_q(self.weight)
#             return F.conv2d(x, wq, self.bias, self.stride, self.padding, self.dilation, self.groups)

#         @torch.no_grad()
#         def to_ternary(self):
#             """
#             Convert this layer into a frozen Bit.Conv2dInfer, carrying over:
#             - per-out-channel weight scale s and Wq in {-1,0,+1}
#             """
#             w = self.weight.data
#             s_vec = _reduce_abs(w, keep_dim=0, op=self.scale_op).squeeze()   # [out]
#             s = s_vec.view(-1, 1, 1)                                         # [out,1,1] for conv broadcast
#             w_bar = w / s_vec.view(-1, 1, 1, 1)
#             w_q = torch.round(w_bar).clamp_(-1, 1).to(w.dtype)

#             return Bit.Conv2dInfer(
#                 w_q=w_q, s=s,
#                 bias=(None if self.bias is None else self.bias.data.clone()),
#                 stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups
#             )

#         @torch.no_grad()
#         def to_ternary_p2(self):
#             # Per-out-channel scale from your chosen op
#             w = self.weight.data
#             s_vec = _reduce_abs(w, keep_dim=0, op=self.scale_op).squeeze()  # [out]
#             w_bar = w / s_vec.view(-1,1,1,1)
#             w_q = torch.round(w_bar).clamp_(-1, 1).to(w.dtype)

#             # Quantize weight scale to power-of-two (save exponents)
#             _, s_exp = _pow2_quantize_scale(s_vec)           # int8 exponents
#             s_exp = s_exp.view(-1, 1, 1)

#             return Bit.Conv2dInferP2(
#                 w_q=w_q,
#                 s_exp=s_exp,
#                 bias=(None if self.bias is None else self.bias.data.clone()),
#                 stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups
#             )

#     class Linear(nn.Module):
#         def __init__(self, in_f, out_f, bias=True, scale_op="median"):
#             super().__init__()
#             self.weight = nn.Parameter(torch.empty(out_f, in_f))
#             nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
#             self.bias = nn.Parameter(torch.zeros(out_f)) if bias else None
#             self.w_q = Bit.Bit1p58Weight(dim=0, scale_op=scale_op)
#             self.scale_op = scale_op

#         def forward(self, x):
#             wq = self.w_q(self.weight)
#             return F.linear(x, wq, self.bias)

#         @torch.no_grad()
#         def to_ternary(self):
#             w = self.weight.data
#             s = _reduce_abs(w, keep_dim=0, op=self.scale_op).squeeze()  # [out]
#             w_q = torch.round(w / s.view(-1,1)).clamp_(-1, 1).to(w.dtype)
#             return Bit.LinearInfer(
#                 w_q=w_q, s=s, bias=(None if self.bias is None else self.bias.data.clone())
#             )

#         @torch.no_grad()
#         def to_ternary_p2(self):
#             w = self.weight.data
#             s = _reduce_abs(w, keep_dim=0, op=self.scale_op).squeeze()   # [out]
#             w_q = torch.round((w / s.view(-1,1))).clamp_(-1, 1).to(w.dtype)

#             # Quantize to power-of-two
#             _, s_exp = _pow2_quantize_scale(s)   # [out] int8
#             return Bit.LinearInferP2(
#                 w_q=w_q, s_exp=s_exp, bias=(None if self.bias is None else self.bias.data.clone())
#             )


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

class DataModuleConfig(BaseModel):    
    data_dir: str
    dataset_name: str = ""
    num_classes:int = -1
    batch_size: int
    num_workers: int = 1
    mixup: bool = False
    cutmix: bool = False
    mix_alpha: float = 1.0

    _datasets = {}

    def model_post_init(self, context):
        self._datasets = {
            'c100':CIFAR100DataModule,
            'timnet':TinyImageNetDataModule,
            'imnet':ImageNetDataModule,
            'mnist':MNISTDataModule,
        }
        ds = self.dataset_name.lower()        
        if ds in ['c100', 'cifar100']:
            self.num_classes = 100
            self.dataset_name = 'c100'
        elif ds in ['timnet', 'tiny', 'tinyimagenet', 'tiny-imagenet']:
            self.num_classes = 200
            self.dataset_name = 'timnet'
        elif ds in ['imnet', 'imagenet', 'in1k', 'imagenet1k']:
            self.num_classes = 1000
            self.dataset_name = 'imnet'
        elif ds in ['mnist']:
            self.num_classes = 10
            self.dataset_name = 'mnist'
        else:
            raise ValueError(f"Unsupported dataset: {ds}")
        return super().model_post_init(context)

    def build(self):
        print(f"[Dataset]: use {self.dataset_name}, {self.num_classes} classes.")
        return self._datasets[self.dataset_name](
            **self.model_dump(exclude=['dataset_name','num_classes'])
        )
        

class LightningDataModule(pl.LightningDataModule):

    @torch.no_grad()
    def show_examples(
        self,
        n: int = 16,
        split: str = "train",
        cols: int = 8,
        seed: Optional[int] = None,
        figsize=(12, 6),
    ):
        """
        Randomly show examples from train/val.
        - split: "train" or "val"
        - with_mix: if True (train only), samples a batch from train_dataloader()
                    so MixUp/CutMix is applied (soft labels shown).
        """
        assert split in {"train", "val"}, "split must be 'train' or 'val'"
        if seed is not None:
            torch.manual_seed(seed)

        ds = self.train_ds if split == "train" else self.val_ds
        assert ds is not None, "Call setup() before show_random_examples()"

        # --- get a batch of images/targets ---
        if split == "train":
            # Use the real train dataloader so MixUp/CutMix happens in collate_fn
            loader = self.train_dataloader()
        else:
            loader = self.val_dataloader()
        
        x, y = next(iter(loader))
        x = x[:n].cpu()
        y = y[:n].cpu()

        # --- denormalize for display ---
        mean = torch.tensor(self.mean, dtype=x.dtype, device=x.device).view(1, 3, 1, 1)
        std = torch.tensor(self.std, dtype=x.dtype, device=x.device).view(1, 3, 1, 1)
        x_vis = (x * std + mean).clamp(0, 1).cpu()

        # class names
        class_names = getattr(ds, "classes", None)

        def label_to_text(target):
            # target can be int tensor, or soft one-hot (MixUp/CutMix)
            if torch.is_tensor(target):
                if target.ndim == 0:
                    idx = int(target.item())
                    return class_names[idx] if class_names else str(idx)
                if target.ndim == 1:  # soft labels
                    topv, topi = torch.topk(target, k=min(2, target.numel()))
                    parts = []
                    for v, i in zip(topv, topi):
                        if float(v) <= 1e-3:
                            continue
                        name = class_names[int(i)] if class_names else str(int(i))
                        parts.append(f"{name}:{float(v):.2f}")
                    return " | ".join(parts) if parts else "mixed"
            return str(target)

        # --- plot grid ---
        cols = max(1, min(cols, n))
        rows = math.ceil(n / cols)
        plt.figure(figsize=figsize)
        for i in range(n):
            plt.subplot(rows, cols, i + 1)
            img = x_vis[i].permute(1, 2, 0).numpy()
            plt.imshow(img)
            plt.axis("off")
            plt.title(label_to_text(y[i]), fontsize=8)
        plt.tight_layout()
        plt.show()
        return x,y
# ----------------------------
# CIFAR-100 DataModule (mixup/cutmix optional)
# ----------------------------
class CIFAR100DataModule(LightningDataModule):
    """
    CIFAR-100 DataModule with:
      - RandAugment + Cutout in train transforms
      - Optional MixUp / CutMix at collate-time (v2.MixUp / v2.CutMix)
    """
    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        num_workers: int = 1,
        mixup: bool = False,
        cutmix: bool = False,
        mix_alpha: float = 1.0,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.mixup = mixup
        self.cutmix = cutmix
        self.mix_alpha = mix_alpha

        self.train_ds = None
        self.val_ds = None

        self.num_classes = 100
        
    def setup(self, stage: Optional[str] = None) -> None:
        self.mean = mean = (0.5071, 0.4867, 0.4408)
        self.std = std = (0.2675, 0.2565, 0.2761)

        train_tf = v2.Compose([
            v2.RandomCrop(32, padding=4),
            v2.RandomHorizontalFlip(),
            v2.RandAugment(num_ops=2, magnitude=9),
            transforms.ToTensor(),
            Cutout(size=(8,8)),          # Cutout after ToTensor
            v2.Normalize(mean, std),
        ])

        val_tf = v2.Compose([
            transforms.ToTensor(),
            v2.Normalize(mean, std),
        ])

        self.train_ds = datasets.CIFAR100(
            root=self.data_dir,
            train=True,
            download=True,
            transform=train_tf,
        )
        self.val_ds = datasets.CIFAR100(
            root=self.data_dir,
            train=False,
            download=True,
            transform=val_tf,
        )

        # Build collate-time MixUp/CutMix transform (no nested function)
        self._build_collate_transform()

    def _build_collate_transform(self):
        """Create a v2 MixUp/CutMix/RandomChoice transform, stored on self."""
        if not self.mixup and not self.cutmix:
            self._collate_transform = None
            return

        transforms_list = []
        if self.mixup:
            transforms_list.append(
                v2.MixUp(num_classes=self.num_classes, alpha=self.mix_alpha)
            )
        if self.cutmix:
            transforms_list.append(
                v2.CutMix(num_classes=self.num_classes, alpha=self.mix_alpha)
            )

        if len(transforms_list) == 1:
            self._collate_transform = transforms_list[0]
        else:
            self._collate_transform = v2.RandomChoice(transforms_list)

    def train_collate_fn(self, batch):
        """
        This method is picklable (no nested closure).
        DataLoader will call this in worker processes.
        """
        x, y = default_collate(batch)
        if self._collate_transform is not None:
            x, y = self._collate_transform(x, y)
        return x, y

    def train_dataloader(self):
        # If no MixUp/CutMix, let DataLoader use its own default collate_fn.
        collate_fn = self.train_collate_fn if self._collate_transform is not None else None

        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=True if self.num_workers > 0 else False,
            collate_fn=collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
            collate_fn=None,
        )

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
class TinyImageNetDataModule(LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int, num_workers: int = 1,
                 mixup: bool = False, cutmix: bool = False, mix_alpha: float = 1.0):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.mixup = mixup
        self.cutmix = cutmix
        self.mix_alpha = mix_alpha

    def setup(self, stage=None):
        self.mean = mean = (0.4802, 0.4481, 0.3975)
        self.std  = std = (0.2302, 0.2265, 0.2262)

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
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=True if self.num_workers > 0 else False,
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
class MNISTDataModule(LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int, num_workers: int = 1,
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
class ImageNetDataModule(LightningDataModule):
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
                 num_workers: int = 1,
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
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=True if self.num_workers > 0 else False,
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
            best_fp = copy.deepcopy(pl_module.student).cpu()
            fp_path = os.path.join(self.out_dir, f"bit_{model_name}_{model_size}_{dataset_name}_best_fp.pt")
            torch.save({"model": best_fp.state_dict(), "acc_tern": current}, fp_path)
            pl_module.print(f"[OK] saved {fp_path} (val/acc_tern={current*100:.2f}%)")
            # save ternary PoT export
            tern = convert_to_ternary(copy.deepcopy(best_fp)).cpu()
            tern_path = os.path.join(self.out_dir,
                                     f"bit_{model_name}_{model_size}_{dataset_name}_ternary_val_acc@{current*100:.2f}.pt")
            torch.save({"model": tern.state_dict(), "acc_tern": current}, tern_path)
            pl_module.print(f"[OK] exported ternary PoT -> {tern_path}")

# ----------------------------
# LightningModule: KD + hints + ternary eval/export
# ----------------------------
class CommonTrainConfig(BaseModel):
    data_dir: str = "./data"    
    export_dir:Optional[str]=''
    dataset_name: Literal["c100", "imnet", "timnet"] = Field(
        default="c100",
        description="Dataset to use (affects stems, classes, transforms)",
    )

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

class AccelLightningModule(nn.Module):
    """
    Lightning-like interface:
      - forward(x)
      - training_step(batch, batch_idx) -> (loss, logs)
      - validation_step(batch, batch_idx) -> logs
      - configure_optimizers() -> (optimizer, scheduler or None, scheduler_interval: "epoch"|"step")
    """

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> Tuple[torch.Tensor, Dict[str, Any]]:
        raise NotImplementedError

    @torch.no_grad()
    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> Dict[str, Any]:
        return {}

    def configure_optimizers(self):
        raise NotImplementedError

    # optional hooks
    def on_fit_start(self, accelerator: Accelerator): ...
    def on_train_epoch_start(self, epoch: int): ...
    def on_train_epoch_end(self, epoch: int): ...
    def on_validation_epoch_start(self, epoch: int): ...
    def on_validation_epoch_end(self, epoch: int): ...


@dataclass
class FitResult:
    train_loss: float
    train_acc: Optional[float] = None
    val_loss: Optional[float] = None
    val_acc: Optional[float] = None


class AccelTrainer:
    def __init__(
        self,
        max_epochs: int,
        mixed_precision: str = "no",              # "no" | "fp16" | "bf16"
        gradient_accumulation_steps: int = 1,
        log_every_n_steps: int = 10,
        max_grad_norm: Optional[float] = None,
        show_progress_bar: bool = True,
    ):
        self.max_epochs = int(max_epochs)
        self.accelerator = Accelerator(
            mixed_precision=mixed_precision,
            gradient_accumulation_steps=int(gradient_accumulation_steps),
        )
        self.log_every_n_steps = int(log_every_n_steps)
        self.max_grad_norm = max_grad_norm
        self.show_progress_bar = show_progress_bar

        self.model: Optional[AccelLightningModule] = None
        self.optimizer = None
        self.scheduler = None
        self.scheduler_interval = "epoch"

    def fit(
        self,
        model: AccelLightningModule,
        datamodule: Optional[Any] = None,
        train_dataloader: Optional[DataLoader] = None,
        val_dataloader: Optional[DataLoader] = None,
    ) -> FitResult:
        # --- dataloaders (LightningDataModule-compatible) ---
        if datamodule is not None:
            if hasattr(datamodule, "setup"):
                try:
                    datamodule.setup("fit")
                except TypeError:
                    datamodule.setup()
            if train_dataloader is None:
                train_dataloader = datamodule.train_dataloader()
            if val_dataloader is None and hasattr(datamodule, "val_dataloader"):
                val_dataloader = datamodule.val_dataloader()

        assert train_dataloader is not None, "Need train_dataloader or datamodule"

        # --- optimizers/schedulers ---
        optimizer, scheduler, interval = model.configure_optimizers()
        self.scheduler_interval = interval or "epoch"

        # --- prepare everything (device/DDP/etc) ---
        if scheduler is None:
            model, optimizer, train_dataloader = self.accelerator.prepare(model, optimizer, train_dataloader)
            if val_dataloader is not None:
                val_dataloader = self.accelerator.prepare(val_dataloader)
        else:
            if val_dataloader is None:
                model, optimizer, train_dataloader, scheduler = self.accelerator.prepare(
                    model, optimizer, train_dataloader, scheduler
                )
            else:
                model, optimizer, train_dataloader, val_dataloader, scheduler = self.accelerator.prepare(
                    model, optimizer, train_dataloader, val_dataloader, scheduler
                )

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler

        # --- hooks ---
        if hasattr(model, "on_fit_start"):
            model.on_fit_start(self.accelerator)

        last = FitResult(train_loss=float("nan"))

        for epoch in range(self.max_epochs):
            last = self._train_one_epoch(epoch, train_dataloader)
            if val_dataloader is not None:
                vloss, vacc = self._validate(epoch, val_dataloader)
                last.val_loss, last.val_acc = vloss, vacc

            # epoch scheduler step (like your working loop)
            if self.scheduler is not None and self.scheduler_interval == "epoch":
                self.scheduler.step()

            if self.accelerator.is_main_process:
                msg = f"[epoch {epoch+1}] train_loss={last.train_loss:.4f}"
                if last.train_acc is not None:
                    msg += f" train_acc={last.train_acc:.4f}"
                if last.val_loss is not None:
                    msg += f" val_loss={last.val_loss:.4f}"
                if last.val_acc is not None:
                    msg += f" val_acc={last.val_acc:.4f}"
                self.accelerator.print(msg)

        return last

    def _train_one_epoch(self, epoch: int, train_loader: DataLoader) -> FitResult:
        assert self.model is not None
        model = self.model
        model.train()
        if hasattr(model, "on_train_epoch_start"):
            model.on_train_epoch_start(epoch)

        loss_sum = 0.0
        n_sum = 0
        correct_sum = 0
        has_acc = False

        it = train_loader
        if self.show_progress_bar and self.accelerator.is_local_main_process:
            it = tqdm(train_loader, desc=f"train {epoch+1}", leave=False)

        for step, batch in enumerate(it):
            with self.accelerator.accumulate(model):  # automatic loss scaling + grad sync handling :contentReference[oaicite:1]{index=1}
                loss, logs = model.training_step(batch, step)
                self.accelerator.backward(loss)        # preferred over loss.backward() :contentReference[oaicite:2]{index=2}

                if self.max_grad_norm is not None:
                    self.accelerator.clip_grad_norm_(model.parameters(), self.max_grad_norm) # :contentReference[oaicite:3]{index=3}

                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

                if self.scheduler is not None and self.scheduler_interval == "step":
                    self.scheduler.step()

            # ---- cheap logging (mean loss across processes) ----
            bs = int(batch[0].size(0))
            loss_mean = self.accelerator.reduce(loss.detach(), reduction="mean")  # :contentReference[oaicite:4]{index=4}
            loss_sum += float(loss_mean.item()) * bs
            n_sum += bs

            # optional acc if module provides it
            if "train/acc" in logs:
                has_acc = True
                # logs['train/acc'] should be a scalar tensor/float for THIS batch
                acc_val = logs["train/acc"]
                if isinstance(acc_val, torch.Tensor):
                    acc_val = float(self.accelerator.reduce(acc_val.detach(), reduction="mean").item())
                correct_sum += int(round(acc_val * bs))

            if self.accelerator.is_main_process and self.log_every_n_steps > 0 and step % self.log_every_n_steps == 0:
                self.accelerator.print(f"epoch {epoch+1} step {step} loss {float(loss_mean.item()):.4f}")

        if hasattr(model, "on_train_epoch_end"):
            model.on_train_epoch_end(epoch)

        train_loss = loss_sum / max(n_sum, 1)
        train_acc = (correct_sum / max(n_sum, 1)) if has_acc else None
        return FitResult(train_loss=train_loss, train_acc=train_acc)

    @torch.no_grad()
    def _validate(self, epoch: int, val_loader: DataLoader) -> Tuple[float, float]:
        assert self.model is not None
        model = self.model
        model.eval()
        if hasattr(model, "on_validation_epoch_start"):
            model.on_validation_epoch_start(epoch)

        loss_sum = 0.0
        n_sum = 0
        correct = 0
        total = 0

        it = val_loader
        if self.show_progress_bar and self.accelerator.is_local_main_process:
            it = tqdm(val_loader, desc=f"val {epoch+1}", leave=False)

        for step, batch in enumerate(it):
            logs = model.validation_step(batch, step)

            # expect logs to include:
            #   "val/loss": scalar tensor
            #   "val/pred": LongTensor [B]
            #   "val/target": LongTensor [B]
            vloss = logs.get("val/loss", None)
            pred = logs.get("val/pred", None)
            target = logs.get("val/target", None)

            if vloss is not None:
                bs = int(batch[0].size(0))
                vloss_mean = self.accelerator.reduce(vloss.detach(), reduction="mean")
                loss_sum += float(vloss_mean.item()) * bs
                n_sum += bs

            if pred is not None and target is not None:
                # gather_for_metrics drops duplicates in the last batch on distributed setups :contentReference[oaicite:5]{index=5}
                pred_g, tgt_g = self.accelerator.gather_for_metrics((pred, target))
                correct += int((pred_g == tgt_g).sum().item())
                total += int(tgt_g.numel())

        if hasattr(model, "on_validation_epoch_end"):
            model.on_validation_epoch_end(epoch)

        val_loss = loss_sum / max(n_sum, 1)
        val_acc = correct / max(total, 1)
        return val_loss, val_acc

class LitCE(AccelLightningModule):
    def __init__(self, config):
        super().__init__()
        self.student = config.student
        self.criterion = nn.CrossEntropyLoss()
        self.lr, self.wd, self.epochs = float(config.lr), float(config.wd), int(config.epochs)
        self.momentum, self.nesterov = float(0.9), bool(True)

    def forward(self, x):
        return self.student(x)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)  # same as your loop

        # optional train acc (works for y=[B] or y=[B,C])
        y_idx = y.argmax(dim=1) if y.ndim == 2 else y
        acc = (logits.argmax(dim=1) == y_idx).float().mean()

        return loss, {"train/acc": acc}

    @torch.no_grad()
    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)

        y_idx = y.argmax(dim=1) if y.ndim == 2 else y
        pred = logits.argmax(dim=1)

        return {"val/loss": loss, "val/pred": pred, "val/target": y_idx}

    def configure_optimizers(self):
        opt = torch.optim.SGD(
            self.student.parameters(),
            lr=self.lr,
            momentum=self.momentum,
            weight_decay=self.wd,
            nesterov=self.nesterov,
        )
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.epochs)
        return opt, sched, "epoch"   # step scheduler once per epoch (like your loop)
    
class LitBitConfig(BaseModel):
    dataset: Optional[Any] = None  # DataModuleConfig
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

    model_name: str = ""
    model_size: str = ""

    hint_points: List[Union[str, Tuple]] = Field(default_factory=list)


class LitBit(AccelLightningModule):
    def __init__(self, config: LitBitConfig):
        super().__init__()
        if type(config) is not dict:
            config = config.model_dump()
        config = LitBitConfig.model_validate(config)

        # --- core ---
        self.scale_op = config.scale_op
        self.student: nn.Module = config.student
        self.teacher: Optional[nn.Module] = config.teacher
        self.has_teacher = True if config.teacher is not None else False

        # --- metadata ---
        self.dataset_name = getattr(config.dataset, "dataset_name", "")
        self.model_name = config.model_name
        self.model_size = config.model_size
        self.num_classes = getattr(config.dataset, "num_classes", 0)

        self.alpha_kd = float(config.alpha_kd)
        self.alpha_hint = float(config.alpha_hint)

        # --- CE selection (hard vs soft labels) ---
        mixup = bool(getattr(config.dataset, "mixup", False)) if config.dataset is not None else False
        cutmix = bool(getattr(config.dataset, "cutmix", False)) if config.dataset is not None else False

        if not (mixup or cutmix):
            self.ce_hard = nn.CrossEntropyLoss(label_smoothing=float(config.label_smoothing))
            self.ce_soft = None
        else:
            self.ce_hard = None
            self.ce_soft = nn.CrossEntropyLoss()

        # --- KD / Hint ---
        self.kd = KDLoss(T=float(config.T))
        self.hint = AdaptiveHintLoss()

        if not (self.alpha_kd > 0 and self.has_teacher):
            self.kd = None
        if not (self.alpha_hint > 0 and self.has_teacher):
            self.hint = None
        if self.kd is None and self.hint is None:
            self.has_teacher = False

        # --- hint plumbing ---
        self.hint_points = list(config.hint_points)
        self._t_feats: Dict[str, torch.Tensor] = {}
        self._s_feats: Dict[str, torch.Tensor] = {}
        self._t_handles = []
        self._s_handles = []

        # --- ternary snapshot & teacher acc cache ---
        self._ternary_snapshot: Optional[nn.Module] = None
        self.t_acc_fps: Dict[int, float] = {}

        # --- optim ---
        self.lr = float(config.lr)
        self.wd = float(config.wd)
        self.epochs = int(config.epochs)

        # --- teacher freeze / hint init / teacher snapshot ---
        if self.has_teacher:
            if self.hint is not None and len(self.hint_points) > 0:
                self.init_hint()
            for p in self.teacher.parameters():
                p.requires_grad_(False)

            # snapshot teacher on CPU to avoid device-mismatch headaches later
            self.teacher_params_snap, self.teacher_bufs_snap = snap_model(self.teacher)
            self.teacher_params_snap = {k: v.detach().cpu().clone() for k, v in self.teacher_params_snap.items()}
            self.teacher_bufs_snap = {k: v.detach().cpu().clone() for k, v in self.teacher_bufs_snap.items()}
        else:
            self.teacher = None
            self.kd = None
            self.hint = None
            self.alpha_kd = 0.0
            self.alpha_hint = 0.0
            self.teacher_params_snap, self.teacher_bufs_snap = None, None

        self._accel = None  # set in on_fit_start

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.student(x)

    @torch.no_grad()
    def on_fit_start(self, accelerator):
        self._accel = accelerator

        # NOTE: accelerate.prepare(model, ...) will already .to(device) the whole module,
        # including teacher if it is a registered submodule.
        # Still ensure eval/freeze status:
        if self.has_teacher and self.teacher is not None:
            self.teacher.eval()
            for p in self.teacher.parameters():
                p.requires_grad_(False)

            if self.alpha_hint > 0 and len(self.hint_points) > 0:
                # register feature hooks once
                self._s_handles = make_feature_hooks(self.student, self.hint_points, self._s_feats, idx=0)
                self._t_handles = make_feature_hooks(self.teacher, self.hint_points, self._t_feats, idx=1)

        # print model summary on main proc
        if accelerator.is_main_process:
            s_total = sum(p.numel() for p in self.student.parameters())
            s_train = sum(p.numel() for p in self.student.parameters() if p.requires_grad)
            t_total = sum(p.numel() for p in self.teacher.parameters()) if self.has_teacher and self.teacher else 0
            accelerator.print("=" * 80)
            accelerator.print(f"Dataset : {self.dataset_name} | num_classes: {self.num_classes}")
            accelerator.print(f"Student : {self.student.__class__.__name__} | params {s_total/1e6:.2f}M (train {s_train/1e6:.2f}M)")
            if self.has_teacher and self.teacher:
                accelerator.print(f"Teacher : {self.teacher.__class__.__name__} | params {t_total/1e6:.2f}M (frozen)")
            else:
                accelerator.print("Teacher : None")
            accelerator.print(f"KD      : alpha_kd={self.alpha_kd} | Hint: alpha_hint={self.alpha_hint} | points={self.hint_points}")
            accelerator.print(f"Optim   : lr={self.lr} wd={self.wd} epochs={self.epochs}")
            accelerator.print("=" * 80)

    def on_train_epoch_start(self, epoch: int):
        self.diff_from_init(f"on_train_epoch_start[{epoch}]")
        self.student.train()

    def on_validation_epoch_start(self, epoch: int):
        self.diff_from_init(f"on_validation_epoch_start[{epoch}]")
        self._ternary_snapshot = self._clone_student()

    # optional cleanup
    def on_validation_epoch_end(self, epoch: int):
        # keep hooks if you want; comment out if you prefer to remove each epoch
        pass

    # -------------------- hint / teacher utilities --------------------

    def init_hint(self):
        s_mods = dict(self.student.named_modules())
        t_mods = dict(self.teacher.named_modules())

        for n in self.hint_points:
            if isinstance(n, tuple):
                sn, tn = n
            else:
                sn, tn = n, n

            if sn not in s_mods:
                raise ValueError(f"Student hint point '{sn}' not found in student.named_modules().")
            if tn not in t_mods:
                raise ValueError(f"Teacher hint point '{tn}' not found in teacher.named_modules().")

            s_m = s_mods[sn]
            t_m = t_mods[tn]

            c_s = infer_out_channels(s_m)
            c_t = infer_out_channels(t_m)

            if c_s is None or c_t is None:
                raise ValueError(
                    f"Cannot infer channels for hint point {n}. "
                    f"Student module: {type(s_m)}, teacher module: {type(t_m)}"
                )

            self.hint.register_pair(sn, c_s, c_t)

    def get_loss_hint(self) -> torch.Tensor:
        loss_hint = 0.0
        for hint_name in self.hint_points:
            sn = tn = hint_name
            if isinstance(hint_name, tuple):
                sn, tn = hint_name

            if sn not in self._s_feats:
                raise ValueError(f"Hint point {sn} not found in student features keys: {list(self._s_feats.keys())}")
            if tn not in self._t_feats:
                raise ValueError(f"Hint point {tn} not found in teacher features keys: {list(self._t_feats.keys())}")

            loss_hint = loss_hint + self.hint(
                sn,
                self._s_feats[sn].float(),
                self._t_feats[tn].float().detach(),
            )
        return loss_hint

    def teacher_forward(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        if not self.has_teacher or self.teacher is None:
            return None
        with torch.no_grad():
            return self.teacher.eval()(x).detach()

    @torch.no_grad()
    def _clone_student(self) -> nn.Module:
        clone: nn.Module = self.student.clone()
        clone.load_state_dict(self.student.state_dict(), strict=True)
        clone = convert_to_ternary(clone)
        # device from actual params (works under accelerate/ddp)
        dev = next(self.student.parameters()).device
        return clone.to(dev)

    # -------------------- training / validation --------------------

    def _ce_training_step(self, x: torch.Tensor, y: torch.Tensor):
        logits = self.student(x)
        ce = self.ce_hard if self.ce_hard is not None else self.ce_soft
        loss = ce(logits, y)
        return loss, {"train/ce": loss.detach()}, logits

    def _ce_kd_training_step(self, x: torch.Tensor, y: torch.Tensor):
        z_t = self.teacher_forward(x)
        loss_ce, logd, logits = self._ce_training_step(x, y)
        alpha_kd = self.alpha_kd
        loss_kd = self.kd(logits.float(), z_t.float())
        loss = (1.0 - alpha_kd) * loss_ce + alpha_kd * loss_kd
        logd = {**logd, "train/kd": loss_kd.detach()}
        return loss, logd, logits

    def _ce_kd_hint_training_step(self, x: torch.Tensor, y: torch.Tensor):
        loss, logd, logits = self._ce_kd_training_step(x, y)
        loss_hint = self.get_loss_hint()
        loss = loss + loss_hint
        logd = {**logd, "train/hint": loss_hint.detach()}
        return loss, logd, logits

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        x, y = batch

        if self.kd is not None and self.hint is not None:
            loss, logd, logits = self._ce_kd_hint_training_step(x, y)
        elif self.kd is not None:
            loss, logd, logits = self._ce_kd_training_step(x, y)
        else:
            loss, logd, logits = self._ce_training_step(x, y)

        # optional acc (helps the AccelTrainer compute train_acc if you want)
        y_idx = y.argmax(dim=1) if y.ndim == 2 else y
        acc = (logits.argmax(dim=1) == y_idx).float().mean()
        logd["train/acc"] = acc.detach()

        return loss, logd

    @torch.no_grad()
    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        x, y = batch
        y_idx = y.argmax(dim=1) if y.ndim == 2 else y

        # snapshot must exist
        if self._ternary_snapshot is None:
            self._ternary_snapshot = self._clone_student()

        z_fp = self.student(x)
        z_tern = self._ternary_snapshot(x)

        # define val loss as CE(fp, y_idx) (hard labels)
        vloss = nn.functional.cross_entropy(z_fp, y_idx.long())

        acc_fp = (z_fp.argmax(dim=1) == y_idx).float().mean()
        acc_tern = (z_tern.argmax(dim=1) == y_idx).float().mean()

        out = {
            # what AccelTrainer uses
            "val/loss": vloss,
            "val/pred": z_fp.argmax(dim=1),
            "val/target": y_idx.long(),

            # extras (you can print/reduce if you want)
            "val/acc_fp": acc_fp,
            "val/acc_tern": acc_tern,
        }

        if self.has_teacher and self.teacher is not None and self.alpha_kd > 0:
            # optional teacher fp acc (cache per batch_idx like your old code)
            if self.t_acc_fps.get(batch_idx) is None:
                z_t = self.teacher_forward(x)
                t_acc = (z_t.argmax(dim=1) == y_idx).float().mean()
                self.t_acc_fps[batch_idx] = float(t_acc.item())
            out["val/t_acc_fp"] = torch.tensor(self.t_acc_fps[batch_idx], device=z_fp.device)

        return out

    # -------------------- optimizer --------------------

    def configure_optimizer_params(self):
        params = list(self.student.parameters())
        if self.hint is not None:
            params += list(self.hint.parameters())
        if self.kd is not None:
            params += list(self.kd.parameters())
        return params

    def configure_optimizers(self):
        opt = torch.optim.SGD(
            self.configure_optimizer_params(),
            lr=self.lr,
            momentum=0.9,
            weight_decay=self.wd,
            nesterov=True,
        )
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.epochs)
        return opt, sched, "epoch"

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

    def register_pair(self, name: str, c_s: int, c_t: int):
        """Create a 1x1 projection for this hint name."""
        k = self._k(name)
        self.proj[k] = nn.Conv2d(c_s, c_t, kernel_size=1, bias=True)
        # no .to(device) here; Lightning will move the whole module

    def forward(self, name, f_s, f_t):
        f_s = F.adaptive_avg_pool2d(f_s, f_t.shape[-2:])
        k = self._k(name)
        f_s = self.proj[k](f_s)
        return F.smooth_l1_loss(f_s, f_t)

class SaveOutputHook:
    """Picklable forward hook that stores outputs into a dict under a given key."""
    __slots__ = ("store", "key")
    def __init__(self, store: dict, key: str):
        self.store = store
        self.key = key
    def __call__(self, module, module_in, module_out:torch.Tensor):
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

def infer_out_channels(module: nn.Module):
    """Try to infer the number of output channels from a module.
    
    Works for Conv2d, BatchNorm2d, Linear, etc.,
    and recursively searches inside containers (Sequential, blocks, etc.).
    """
    # Direct attributes commonly used for "channels"
    for attr in ("out_channels", "num_features", "out_features"):
        if hasattr(module, attr):
            return getattr(module, attr)

    # If it's a container (Sequential, custom block, etc.), 
    # walk its children (from last to first to approximate "output")
    children = list(module.children())
    for child in reversed(children):
        c = infer_out_channels(child)
        if c is not None:
            return c

    # Give up
    return None

# ----------------------------
# Common CLI utilities
# ----------------------------
def setup_trainer(args:CommonTrainConfig, monitor="val/acc_tern"):
    """
    Setup common PyTorch Lightning training components.

    Args:
        args: Parsed command-line arguments
        lit_module: Lightning module to train

    Returns:
        tuple: (trainer, datamodule)
    """
    pl.seed_everything(args.seed, workers=True)
    os.makedirs(args.export_dir, exist_ok=True)
    logger = CSVLogger(save_dir=args.export_dir, name="logs")
    ckpt_cb = ModelCheckpoint(monitor=monitor, mode="max", save_top_k=1, save_last=True)
    lr_cb = LearningRateMonitor(logging_interval="epoch")
    export_cb = ExportBestTernary(args.export_dir, monitor=monitor, mode="max")
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
        num_sanity_val_steps=0
    )

    return trainer

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
        preprocess = transforms.ToTensor()

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

def set_export_mode(m, flag=True):
    if hasattr(m, "export_mode"):
        m.export_mode = flag
    for c in m.children():
        set_export_mode(c, flag)

def export_onnx(model, dummy_input, path="model.onnx"):
    model = model.eval()
    set_export_mode(model, True)
    exported = torch.onnx.dynamo_export(model, dummy_input)
    exported.save(path)

# if __name__ == '__main__':
#     freeze_support()
#     dm = TinyImageNetDataModule("./data",4)
#     dm.setup()
#     for data,label in dm.train_dataloader():
#         break
#     print(data)
#     print(label)
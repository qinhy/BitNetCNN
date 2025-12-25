"""
Common utilities for BitNetCNN implementations.
This module contains shared components used across different BitNet model implementations.
"""

import os
import re
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import warnings
from PIL import Image, ImageTk

import random
from typing import Any, Callable, List, Literal, Optional, Sequence, Tuple, Type, Union
from matplotlib import pyplot as plt
from pydantic import BaseModel, Field, PrivateAttr
import torch

from bitlayers.bit import Bit
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn

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
            dtype_str = "-" + " "*14

        out_chs = ""
        if hasattr(module,'out_channels'):
            out_chs = f"out_channels={module.out_channels}"
            
        row = (name, module.__class__.__name__, nparams, dtype_str)
        info.append(row)

        if verbose:
            print(f"{name:35} {module.__class__.__name__:25} "
                  f"params={nparams:8d}  dtypes={dtype_str:15} {out_chs}")
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

# ----------------------------
# Common CLI utilities
# ----------------------------
# def setup_trainer(args:CommonTrainConfig, monitor="val/acc_tern"):
#     """
#     Setup common PyTorch Lightning training components.

#     Args:
#         args: Parsed command-line arguments
#         lit_module: Lightning module to train

#     Returns:
#         tuple: (trainer, datamodule)
#     """
#     pl.seed_everything(args.seed, workers=True)
#     os.makedirs(args.export_dir, exist_ok=True)
#     logger = CSVLogger(save_dir=args.export_dir, name="logs")
#     ckpt_cb = ModelCheckpoint(monitor=monitor, mode="max", save_top_k=1, save_last=True)
#     lr_cb = LearningRateMonitor(logging_interval="epoch")
#     export_cb = ExportBestTernary(args.export_dir, monitor=monitor, mode="max")
#     callbacks = [ckpt_cb, lr_cb, export_cb]

#     accelerator = "cpu" if args.cpu else "auto"
#     precision = "16-mixed" if args.amp else "32-true"

#     # Multi-GPU setup
#     devices = args.gpus if hasattr(args, 'gpus') else 1
#     strategy = args.strategy if hasattr(args, 'strategy') else "auto"

#     # Use appropriate strategy for multi-GPU training
#     import sys
#     if devices > 1 or devices == -1:
#         if strategy == "auto":
#             # Check if NCCL is available (for CUDA GPUs)
#             if sys.platform == "win32":
#                 # Windows doesn't support NCCL, must use gloo backend
#                 from pytorch_lightning.strategies import DDPStrategy
#                 strategy = DDPStrategy(process_group_backend="gloo")
#                 print(f"[Multi-GPU] Windows detected, using DDP with gloo backend")
#             else:
#                 try:
#                     import torch.distributed
#                     if torch.cuda.is_available() and torch.distributed.is_nccl_available():
#                         strategy = "ddp"
#                     else:
#                         from pytorch_lightning.strategies import DDPStrategy
#                         strategy = DDPStrategy(process_group_backend="gloo")
#                         print(f"[Multi-GPU] NCCL not available, using DDP with gloo backend")
#                 except:
#                     from pytorch_lightning.strategies import DDPStrategy
#                     strategy = DDPStrategy(process_group_backend="gloo")
#                     print(f"[Multi-GPU] Using DDP with gloo backend")

#         print(f"[Multi-GPU] Training on {devices if devices > 0 else 'all'} GPUs")

#     trainer = pl.Trainer(
#         max_epochs=args.epochs,
#         accelerator=accelerator,
#         devices=devices,
#         strategy=strategy,
#         precision=precision,
#         logger=logger,
#         callbacks=callbacks,
#         log_every_n_steps=50,
#         deterministic=False,
#         sync_batchnorm=True if (devices > 1 or devices == -1) else False,
#         num_sanity_val_steps=0
#     )

#     return trainer

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
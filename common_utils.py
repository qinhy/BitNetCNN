"""
Common utilities for BitNetCNN implementations.
This module contains shared components used across different BitNet model implementations.
"""

from functools import partial
import os, math, copy, argparse
import torch
torch.set_float32_matmul_precision('high')
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models.resnet import ResNet, BasicBlock, Bottleneck
from huggingface_hub import hf_hub_download

# --- Lightning ---
import pytorch_lightning as pl
from pytorch_lightning import Callback
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from torchmetrics.classification import MulticlassAccuracy

import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

# Constants
EPS = 1e-12

# ----------------------------
# Quantization Utilities
# ----------------------------
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
        Wq is stored as int8 in {-1,0,+1}. s is float per output channel.
        """
        def __init__(self, w_q, s, bias, stride, padding, dilation, groups):
            super().__init__()
            self.register_buffer("w_q", w_q.to(torch.int8))
            self.register_buffer("s", s)  # [out,1,1]
            self.register_buffer("bias", None if bias is None else bias)
            self.stride, self.padding, self.dilation, self.groups = stride, padding, dilation, groups

        def forward(self, x):
            y = F.conv2d(x, self.w_q.float(), None, self.stride, self.padding, self.dilation, self.groups)
            y = y * self.s  # broadcast over H,W
            if self.bias is not None:
                y = y + self.bias.view(1, -1, 1, 1)
            return y

    class LinearInfer(nn.Module):
        """Frozen ternary linear: y = (x @ Wq^T) * s + b"""
        def __init__(self, w_q, s, bias):
            super().__init__()
            self.register_buffer("w_q", w_q.to(torch.int8))  # [out,in]
            self.register_buffer("s", s)                     # [out]
            self.register_buffer("bias", None if bias is None else bias)

        def forward(self, x):
            y = F.linear(x, self.w_q.float(), None)
            y = y * self.s
            if self.bias is not None:
                y = y + self.bias
            return y

    class Conv2dInferP2(nn.Module):
        """
        Ternary conv with power-of-two scales:
        - We store s_exp per output channel, so scaling is x * 2^s_exp (shift on integer backends).
        """
        def __init__(self, w_q, s_exp, bias, stride, padding, dilation, groups):
            super().__init__()
            self.register_buffer("w_q", w_q.to(torch.int8))
            self.register_buffer("s_exp", s_exp.to(torch.int8))      # [out,1,1]
            self.register_buffer("bias", None if bias is None else bias)
            self.stride, self.padding, self.dilation, self.groups = stride, padding, dilation, groups

        def forward(self, x):
            y = F.conv2d(x, self.w_q.float(), None, self.stride, self.padding, self.dilation, self.groups)
            y = torch.ldexp(y, self.s_exp.to(torch.int32))
            if self.bias is not None:
                y = y + self.bias.view(1, -1, 1, 1)
            return y

    class LinearInferP2(nn.Module):
        """Ternary linear with power-of-two output scales."""
        def __init__(self, w_q, s_exp, bias):
            super().__init__()
            self.register_buffer("w_q", w_q.to(torch.int8))  # [out,in]
            self.register_buffer("s_exp", s_exp.to(torch.int8))  # [out]
            self.register_buffer("bias", None if bias is None else bias)

        def forward(self, x):
            y = F.linear(x, self.w_q.float(), None)
            y = torch.ldexp(y, self.s_exp.to(torch.int32))
            if self.bias is not None:
                y = y + self.bias
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

def make_feature_hooks(module: nn.Module, names, store: dict):
    """Register picklable forward hooks; returns list of handles."""
    handles = []
    name_set = set(names)
    for n, sub in module.named_modules():
        if n in name_set:
            handles.append(sub.register_forward_hook(SaveOutputHook(store, n)))
    return handles

# ----------------------------
# CIFAR-100 DataModule (mixup/cutmix optional)
# ----------------------------
def mix_collate(batch, *, aug_cutmix: bool, aug_mixup: bool, alpha: float):
    xs, ys = zip(*batch)
    x = torch.stack(xs); y = torch.tensor(ys)
    if not (aug_cutmix or aug_mixup):
        return x, y

    import random
    lam = 1.0
    if aug_cutmix and random.random() < 0.5:
        lam = torch.distributions.Beta(alpha, alpha).sample().item()
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
        lam = torch.distributions.Beta(alpha, alpha).sample().item()
        idx = torch.randperm(x.size(0))
        x = lam * x + (1 - lam) * x[idx]
        return x, (y, y[idx], lam)
    
class CIFAR100DataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int, num_workers: int = 4,
                 aug_mixup: bool = False, aug_cutmix: bool = False, alpha: float = 1.0):
        super().__init__()
        self.data_dir, self.batch_size, self.num_workers = data_dir, batch_size, num_workers
        self.aug_mixup, self.aug_cutmix, self.alpha = aug_mixup, aug_cutmix, alpha

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
            aug_cutmix=self.aug_cutmix,
            aug_mixup=self.aug_mixup,
            alpha=self.alpha,
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


    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=256, shuffle=False,
                          num_workers=self.num_workers, pin_memory=True,
                          persistent_workers=True if self.num_workers > 0 else False)


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
            pl_module.print(f"✓ saved {fp_path} (val/acc_tern={current*100:.2f}%)")
            # save ternary PoT export
            tern = convert_to_ternary(copy.deepcopy(best_fp)).cpu().eval()
            tern_path = os.path.join(self.out_dir, f"bit_{model_name}_{model_size}_{dataset_name}_ternary.pt")
            torch.save({"model": tern.state_dict(), "acc_tern": current}, tern_path)
            pl_module.print(f"✓ exported ternary PoT → {tern_path}")

# ----------------------------
# LightningModule: KD + hints + ternary eval/export
# ----------------------------
class LitBit(pl.LightningModule):
    def __init__(self, lr, wd, epochs, label_smoothing=0.1,
                 alpha_kd=0.7, alpha_hint=0.05, T=4.0, scale_op="median",
                 width_mult=1.0, amp=True,
                 export_dir="./checkpoints_c100_mbv2",
                 student=None,
                 teacher=None,
                 dataset_name='',
                 model_name='',
                 model_size='',
                 hint_points=[]):
        super().__init__()
        self.save_hyperparameters(ignore=['student','teacher','_t_feats','_s_feats','_t_handles','_s_handles','_ternary_snapshot'])
        self.scale_op = scale_op
        self.student = student
        self.teacher = teacher
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.model_size = model_size
        if alpha_kd<=0 and alpha_hint<=0:
            self.teacher=None
        if self.teacher:
            for p in self.teacher.parameters(): p.requires_grad_(False)
        self.ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing).eval()
        self.kd = KDLoss(T=T).eval()
        self.hint = AdaptiveHintLoss().eval()
        self.hint_points = hint_points
        self.acc_fp = MulticlassAccuracy(num_classes=100).eval()
        self.acc_tern = MulticlassAccuracy(num_classes=100).eval()
        self._ternary_snapshot = None
        self._t_feats = {}
        self._s_feats = {}

    def setup(self, stage=None):
        if self.teacher:
            self.teacher = self.teacher.to(self.device).eval()
            self._t_handles = make_feature_hooks(self.teacher, self.hint_points, self._t_feats)
            self._s_handles = make_feature_hooks(self.student, self.hint_points, self._s_feats)

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
        clone = self.student.clone()
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
        use_amp = bool(self.hparams.amp and "cuda" in str(self.device))

        with torch.amp.autocast("cuda", enabled=use_amp):
            z_s = self.student(x)
            if is_mix:
                loss_ce = lam * self.ce(z_s, y_a) + (1 - lam) * self.ce(z_s, y_b)
            else:
                loss_ce = self.ce(z_s, y)

            if self.teacher:
                with torch.amp.autocast("cuda", enabled=use_amp):
                    z_t = self.teacher(x)

            loss_kd = 0.0
            if self.hparams.alpha_kd > 0:
                with torch.amp.autocast("cuda", enabled=False):
                    loss_kd = self.kd(z_s.float(), z_t.float())

            # Hints
            loss_hint = 0.0
            if self.hparams.alpha_hint>0 and len(self.hint_points)>0:
                for n in self.hint_points:
                    if n not in self._s_feats:
                        raise ValueError(f"Hint point {n} not found in student features of {self._s_feats}.")
                    if n not in self._t_feats:
                        raise ValueError(f"Hint point {n} not found in teacher features of {self._t_feats}.")
                    loss_hint = loss_hint + self.hint(n, self._s_feats[n].float(), self._t_feats[n].float())

            loss = (1.0 - self.hparams.alpha_kd) * loss_ce + self.hparams.alpha_kd * loss_kd + self.hparams.alpha_hint * loss_hint

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
        with torch.no_grad():
            logits_fp = self.student(x)
            acc_fp = self.acc_fp(logits_fp.softmax(1), y)
            self.log("val/acc_fp", acc_fp, on_step=False, on_epoch=True, prog_bar=True, batch_size=x.size(0))

            if self._ternary_snapshot is not None:
                logits_t = self._ternary_snapshot(x)
                acc_t = self.acc_tern(logits_t.softmax(1), y)
                self.log("val/acc_tern", acc_t, on_step=False, on_epoch=True, prog_bar=True, batch_size=x.size(0))

            if self.hparams.alpha_kd>0:
                # optional: show a static teacher acc estimate as in the other script
                self.log("val/t_acc_fp", 0.760, on_step=False, on_epoch=True, prog_bar=True, batch_size=x.size(0))

    def configure_optimizers(self):
        opt = torch.optim.SGD(
            list(self.student.parameters()) + list(self.hint.parameters()),
            lr=self.hparams.lr, momentum=0.9, weight_decay=self.hparams.wd, nesterov=True
        )
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.hparams.epochs)
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sched, "interval": "epoch", "monitor": "val/acc_tern"},
        }

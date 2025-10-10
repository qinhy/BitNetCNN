"""
Common utilities for BitNetCNN implementations.
This module contains shared components used across different BitNet model implementations.
"""

from functools import partial
from multiprocessing import freeze_support
import os
import math
import copy
import argparse
import random
from PIL import Image
import torch
torch.set_float32_matmul_precision('high')
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
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


    def val_dataloader(self,batch_size=256):
        return DataLoader(self.val_ds, batch_size=batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=True,
                          persistent_workers=True if self.num_workers > 0 else False)

# ----------------------------
# Mixup / CutMix Collate Function
# ----------------------------
def mix_collate(batch, *, aug_cutmix: bool, aug_mixup: bool, alpha: float):
    xs, ys = zip(*batch)
    x = torch.stack(xs)
    y = torch.tensor(ys)
    
    if not (aug_cutmix or aug_mixup):
        return x, y

    lam = 1.0
    idx = torch.randperm(x.size(0))

    if aug_cutmix and random.random() < 0.5:
        lam = torch.distributions.Beta(alpha, alpha).sample().item()
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
        lam = torch.distributions.Beta(alpha, alpha).sample().item()
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
                 aug_mixup: bool = False, aug_cutmix: bool = False, alpha: float = 1.0):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.aug_mixup = aug_mixup
        self.aug_cutmix = aug_cutmix
        self.alpha = alpha

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
    def __init__(self, data_dir: str, batch_size: int, num_workers: int = 4):
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
    - optional mixup/cutmix via your `mix_collate` and (aug_mixup, aug_cutmix, alpha)
    """
    def __init__(self,
                 data_dir: str,
                 batch_size: int,
                 num_workers: int = 8,
                 aug_mixup: bool = False,
                 aug_cutmix: bool = False,
                 alpha: float = 0.2,
                 image_size: int = 224,
                 val_resize: int = 256):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.aug_mixup = aug_mixup
        self.aug_cutmix = aug_cutmix
        self.alpha = alpha
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
                 hint_points=[],
                 num_classes=-1):
        super().__init__()
        self.save_hyperparameters(ignore=['student','teacher','_t_feats','_s_feats',
                                          '_t_handles','_s_handles','_ternary_snapshot'])
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
        self.acc_fp = MulticlassAccuracy(num_classes=num_classes).eval()
                        # average='micro', multidim_average='global', top_k=1).eval()

        self._ternary_snapshot = None
        self._t_feats = {}
        self._s_feats = {}
        self._t_handles = []
        self._s_handles = []
        self.t_acc_fp = None

    def setup(self, stage=None):
        if self.teacher:
            self.teacher = self.teacher.to(self.device).eval()
            if self.hparams.alpha_hint>0:
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

        # use_amp = bool(self.hparams.amp and "cuda" in str(self.device))

        z_s = self.student(x)
        
        if is_mix:
            loss_ce = lam * self.ce(z_s, y_a) + (1 - lam) * self.ce(z_s, y_b)
        else:
            loss_ce = self.ce(z_s, y)

        if self.teacher:
            z_t = self.teacher(x)

        loss_kd = 0.0
        if self.hparams.alpha_kd > 0:
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
        def log_val(n,model=None,acc=None,x=x,y=y):
            if acc is None:
                # acc = (model(x).argmax(1)==y).sum()/x.size(0)
                acc = self.acc_fp(model(x), y)
            self.log(f"val/{n}", acc, on_step=False, on_epoch=True, prog_bar=True, batch_size=x.size(0))
            return acc
        
        with torch.no_grad():
            acc_fp = log_val("acc_fp", self.student)
            acc_t = log_val("acc_tern", self._ternary_snapshot)
            if self.hparams.alpha_kd>0:
                if self.t_acc_fp is None and self.teacher:
                    self.t_acc_fp = log_val("t_acc_fp", self.teacher)
                else:
                    log_val("t_acc_fp", acc=self.t_acc_fp)

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

# ----------------------------
# Common CLI utilities
# ----------------------------
def add_common_args(parser):
    """Add common training arguments to an argument parser."""
    parser.add_argument("--data", type=str, default="./data")
    parser.add_argument("--out",  type=str, default="./ckpt_c100")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=2e-1)
    parser.add_argument("--wd", type=float, default=5e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--alpha-kd", type=float, default=0.3)
    parser.add_argument("--alpha-hint", type=float, default=0.05)
    parser.add_argument("--T", type=float, default=4.0)
    parser.add_argument("--scale-op", type=str, default="median", choices=["mean","median"])
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--mixup", action="store_true")
    parser.add_argument("--cutmix", action="store_true")
    parser.add_argument("--mix-alpha", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs to use (default: 1, use -1 for all available)")
    parser.add_argument("--strategy", type=str, default="auto", choices=["auto", "ddp", "ddp_spawn", "fsdp"],
                        help="Distributed training strategy (default: auto)")
    return parser

def setup_trainer(args, lit_module, dm = None):
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
            data_dir=args.data, batch_size=args.batch_size, num_workers=4,
            aug_mixup=args.mixup, aug_cutmix=args.cutmix, alpha=args.mix_alpha
        )

    os.makedirs(args.out, exist_ok=True)
    logger = CSVLogger(save_dir=args.out, name="logs")
    ckpt_cb = ModelCheckpoint(monitor="val/acc_tern", mode="max", save_top_k=1, save_last=True)
    lr_cb = LearningRateMonitor(logging_interval="epoch")
    export_cb = ExportBestTernary(args.out, monitor="val/acc_tern", mode="max")
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


if __name__ == '__main__':
    freeze_support()
    dm = TinyImageNetDataModule("./data",4)
    dm.setup()
    for data,label in dm.train_dataloader():
        break
    print(data)
    print(label)
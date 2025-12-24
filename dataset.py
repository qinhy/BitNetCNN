import json
import os
import math
from pathlib import Path
from PIL import Image, ImageOps
from typing import Any, Callable, Optional, Sequence, Tuple, Union, List, Dict

import cv2
from matplotlib import pyplot as plt
import numpy as np
from pydantic import BaseModel, Field, PrivateAttr, model_validator
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import default_collate

from torchvision import datasets
import torchvision.transforms.functional as vF
from torchvision.transforms import v2
from torchvision.transforms import InterpolationMode
from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import extract_archive, download_url, check_integrity

from bitlayers.aug import AutoRandomResizedCrop, MixupCutmix, SimpleImageTrainAugment, SimpleImageValAugment, ToTensor

import albumentations as A
from albumentations.core.composition import BaseCompose
from albumentations.core.transforms_interface import BasicTransform


# ----------------------------
# Augmentations
# ----------------------------
def get_val_tf(mean, std):
    return v2.Compose(
        [
            v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
            v2.Normalize(mean, std),
        ]
    )

# ----------------------------
# Config
# ----------------------------
class DataModuleConfig(BaseModel):
    data_dir: str
    dataset_name: str = ""
    num_classes: int = -1

    batch_size: int
    num_workers: int = 0

    mixup: bool = False
    cutmix: bool = False
    mix_alpha: float = 1.0

    _datasets: Dict[str, type] = PrivateAttr(default_factory=dict)

    def model_post_init(self, __context):
        self._datasets = {
            "c100": CIFAR100DataModule,
            "timnet": TinyImageNetDataModule,
            "imnet": ImageNetDataModule,
            "mnist": MNISTDataModule,
            "retinaface": RetinaFaceDataModule,
        }

        ds = self.dataset_name.lower()
        if ds in ["c100", "cifar100"]:
            self.num_classes = 100
            self.dataset_name = "c100"
        elif ds in ["timnet", "tiny", "tinyimagenet", "tiny-imagenet"]:
            self.num_classes = 200
            self.dataset_name = "timnet"
        elif ds in ["imnet", "imagenet", "in1k", "imagenet1k"]:
            self.num_classes = 1000
            self.dataset_name = "imnet"
        elif ds in ["mnist"]:
            self.num_classes = 10
            self.dataset_name = "mnist"
        elif ds in ["retinaface","wider-face","wider"]:
            self.num_classes = -1
            self.dataset_name = "retinaface"
        else:
            raise ValueError(f"Unsupported dataset: {ds}")

        return super().model_post_init(__context)

    def build(self) -> 'DataSetModule':
        print(f"[Dataset]: use {self.dataset_name}, {self.num_classes} classes.")
        return self._datasets[self.dataset_name](self)

# ----------------------------
# Base Data Module
# ----------------------------
class DataSetModule:
    def __init__(self, config: "DataModuleConfig"):
        self.data_dir = config.data_dir
        self.batch_size = config.batch_size
        self.num_workers = config.num_workers

        self.mixup = config.mixup
        self.cutmix = config.cutmix
        self.mix_alpha = config.mix_alpha

        self.train_tf = None
        self.p = 0.325
        self.val_tf = None
        self.dataset_cls = None  # must accept (root, train, download, transform)
        self.num_classes = -1

        self.train_ds = None
        self.val_ds = None

        # used by show_examples()
        self.mean = None
        self.std = None

        # collate-time transform (MixUp/CutMix)
        self._collate_transform = None

    def setup(self, stage: Optional[str] = None) -> None:
        if self.dataset_cls is None:
            raise ValueError("dataset_cls is not set")

        self.train_ds = self.dataset_cls(
            root=self.data_dir,
            train=True,
            download=True,
            transform=self.train_tf,
        )
        self.val_ds = self.dataset_cls(
            root=self.data_dir,
            train=False,
            download=True,
            transform=self.val_tf,
        )

        self._build_collate_transform()

    def _build_collate_transform(self):
        self._collate_transform = MixupCutmix(
            num_classes=self.num_classes,
            enable_mixup=self.mixup,
            enable_cutmix=self.cutmix,
            beta_alpha=self.mix_alpha,
            p=self.p
        ).build()

    def train_collate_fn(self, batch):
        x, y = default_collate(batch)
        if self._collate_transform is not None:
            x, y = self._collate_transform(x, y)
        return x, y

    def train_dataloader(self) -> DataLoader:
        collate_fn = self.train_collate_fn if self._collate_transform is not None else None
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=(self.num_workers > 0),
            collate_fn=collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=(self.num_workers > 0),
            collate_fn=None,
        )

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
        - If split='train', MixUp/CutMix will be shown if enabled (applied in collate_fn).
        """
        assert split in {"train", "val"}, "split must be 'train' or 'val'"
        if seed is not None:
            torch.manual_seed(seed)

        assert self.train_ds is not None and self.val_ds is not None, "Call setup() first."

        loader = self.train_dataloader() if split == "train" else self.val_dataloader()
        x, y = next(iter(loader))
        x = x[:n].cpu()
        try:
            y = y[:n].cpu()
        except:
            y = None

        # denormalize for display
        mean = torch.tensor(self.mean, dtype=x.dtype).view(1, x.shape[1], 1, 1)
        std = torch.tensor(self.std, dtype=x.dtype).view(1, x.shape[1], 1, 1)
        x_vis = (x * std + mean).clamp(0, 1)

        ds = self.train_ds if split == "train" else self.val_ds
        class_names = getattr(ds, "classes", None)

        def label_to_text(target):
            # target can be int tensor, or soft one-hot (MixUp/CutMix)
            if torch.is_tensor(target):
                if target.ndim == 0:
                    idx = int(target.item())
                    return class_names[idx] if class_names else str(idx)
                if target.ndim == 1:
                    topv, topi = torch.topk(target, k=min(2, target.numel()))
                    parts = []
                    for v, i in zip(topv, topi):
                        if float(v) <= 1e-3:
                            continue
                        name = class_names[int(i)] if class_names else str(int(i))
                        parts.append(f"{name}:{float(v):.2f}")
                    return " | ".join(parts) if parts else "mixed"
            return str(target)

        cols = max(1, min(cols, n))
        rows = math.ceil(n / cols)
        plt.figure(figsize=figsize)
        for i in range(n):
            plt.subplot(rows, cols, i + 1)
            img = x_vis[i].permute(1, 2, 0).numpy()
            if img.shape[2] == 1:
                plt.imshow(img[:, :, 0], cmap="gray")
            else:
                plt.imshow(img)
            plt.axis("off")
            plt.title(label_to_text(y[i])  if y else 'null', fontsize=8)
        plt.tight_layout()
        plt.show()
        return x, y


    # -----------------------------
    # Validation (Classification)
    # -----------------------------
    @staticmethod
    def _unwrap_model_output(output):
        """Handle models that return (logits, ...) or {'logits': ...}."""
        if isinstance(output, (tuple, list)):
            return output[0]
        if isinstance(output, dict):
            return output.get("logits", output.get("output", None)) if ("logits" in output or "output" in output) else next(iter(output.values()))
        return output

    @staticmethod
    def _soft_target_cross_entropy(logits: torch.Tensor, soft_targets: torch.Tensor) -> torch.Tensor:
        """
        Cross-entropy when targets are soft distributions (e.g., from MixUp/CutMix).
        soft_targets: [B, C], floats summing ~1 across C.
        """
        log_probs = torch.log_softmax(logits, dim=1)
        return -(soft_targets * log_probs).sum(dim=1).mean()

    @torch.no_grad()
    def validate(
        self,
        model: nn.Module,
        criterion: nn.Module,
        device: Optional[torch.device] = None,
        amp: bool = False,
        topk: Tuple[int, ...] = (1,),
        split: str = "val",
        max_batches: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Validate a classification model on hard labels.
        Returns: {'val_loss': ..., 'top1_acc': ..., ...}
        """
        # If you NEVER want train validation (because it may be mixup/cutmix),
        # uncomment the next line:
        # assert split == "val", "This validate() expects hard labels; use split='val' only."

        assert split in {"train", "val"}, "split must be 'train' or 'val'"

        model.eval()
        if device is None:
            device = next(model.parameters()).device

        loader = self.val_dataloader() if split == "val" else self.train_dataloader()

        total_loss = 0.0
        total_samples = 0
        correct_k = {k: 0.0 for k in topk}

        autocast_enabled = amp and (device.type == "cuda")

        for b_idx, (x, y) in enumerate(loader):
            if max_batches is not None and b_idx >= max_batches:
                break

            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)  # expected shape [B], dtype long for CE

            # AMP only on CUDA
            if device.type == "cuda":
                autocast_ctx = torch.amp.autocast(device_type="cuda", enabled=autocast_enabled)
            else:
                autocast_ctx = torch.amp.autocast(device_type="cpu", enabled=False)

            with autocast_ctx:
                out = model(x)
                logits = self._unwrap_model_output(out)
                loss = criterion(logits, y)

            bs = x.size(0)
            total_loss += float(loss.item()) * bs
            total_samples += bs

            # Top-k accuracy
            max_k = max(topk)
            _, pred = logits.topk(max_k, dim=1, largest=True, sorted=True)  # [B, max_k]
            pred = pred.t()  # [max_k, B]
            correct = pred.eq(y.view(1, -1).expand_as(pred))

            for k in topk:
                correct_k[k] += float(correct[:k].reshape(-1).float().sum().item())

        denom = max(total_samples, 1)
        metrics = {"val_loss": total_loss / denom}
        for k in topk:
            metrics[f"top{k}_acc"] = correct_k[k] / denom
        return metrics

    # -----------------------------
    # Validation (Regression)
    # -----------------------------
    @torch.no_grad()
    def validate_regression(
        self,
        model: nn.Module,
        criterion: nn.Module,
        device: Optional[torch.device] = None,
        amp: bool = True,
        split: str = "val",
        max_batches: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Validate a regression model.
        Returns dict: {'val_loss': ..., 'mae': ..., 'rmse': ...}
        """
        assert split in {"train", "val"}, "split must be 'train' or 'val'"

        model.eval()
        if device is None:
            device = next(model.parameters()).device

        loader = self.val_dataloader() if split == "val" else self.train_dataloader()

        total_loss = 0.0
        total_abs_err = 0.0
        total_sq_err = 0.0
        total_samples = 0

        autocast_enabled = amp and (device.type == "cuda")

        for b_idx, (x, y) in enumerate(loader):
            if max_batches is not None and b_idx >= max_batches:
                break

            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=autocast_enabled):
                out = model(x)
                preds = self._unwrap_model_output(out)
                loss = criterion(preds, y)

            bs = x.size(0)
            total_loss += float(loss.item()) * bs
            total_samples += bs

            preds_f = preds.detach().view(bs, -1)
            y_f = y.detach().view(bs, -1)

            abs_err = (preds_f - y_f).abs().mean(dim=1)        # per-sample MAE
            sq_err = ((preds_f - y_f) ** 2).mean(dim=1)        # per-sample MSE

            total_abs_err += float(abs_err.sum().item())
            total_sq_err += float(sq_err.sum().item())

        denom = max(total_samples, 1)
        mae = total_abs_err / denom
        rmse = (total_sq_err / denom) ** 0.5

        return {"val_loss": total_loss / denom, "mae": mae, "rmse": rmse}

# ----------------------------
# CIFAR-100
# ----------------------------
class CIFAR100Dataset(datasets.CIFAR100):
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img, target = self.data[index], self.targets[index]
        img:np.ndarray = img
        if self.transform is not None:
            if isinstance(self.transform, (BaseCompose, BasicTransform)):
                img = self.transform(image=img)["image"]
            else:
                img = self.transform(Image.fromarray(img))
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target
    
class CIFAR100DataModule(DataSetModule):
    def __init__(self, config: "DataModuleConfig"):
        super().__init__(config)
        self.num_classes = 100
        self.mean = (0.5071, 0.4867, 0.4408)
        self.std = (0.2675, 0.2565, 0.2761)
        self.p = 0.5
        self.train_tf = A.Compose([
            A.Rotate(limit=(-10,10), p=self.p),
            A.HorizontalFlip(p=self.p),
            AutoRandomResizedCrop(scale=(0.75, 1.0), ratio=(0.75, 1.33),p=self.p),
            # A.OneOf([
            #     A.ColorJitter(p=1.0,brightness=0.3, contrast=0.3, saturation=0.3, hue=0.2),
            #     A.ToGray(p=1.0),
            # ], p=self.p),
            # A.OneOf([
            #     A.GaussianBlur(p=1.0,blur_limit=(1, 3)),
            #     A.GaussNoise(p=1.0,std_range=self.noise_std_range),
            # ], p=self.p),
            #  A.CoarseDropout(num_holes_range=(1,4), 
            #                  hole_height_range=(rmin, rmax), 
            #                  hole_width_range=(rmin, rmax), p=self.p),
            A.Normalize(mean=self.mean, std=self.std),
            A.ToTensorV2(),
        ])
        self.train_tf = v2.Compose([
            v2.RandomCrop(32, padding=4),
            v2.RandomHorizontalFlip(p=self.p),

            v2.RandomApply(
                [v2.RandomChoice([
                    v2.AutoAugment(
                        policy=v2.AutoAugmentPolicy.CIFAR10,
                        interpolation=InterpolationMode.BILINEAR,
                    ),
                    v2.RandAugment(num_ops=2, magnitude=9),
                    v2.AugMix(severity=3, mixture_width=3, chain_depth=-1, alpha=1.0),
                ])],
                p=self.p
            ),

            ToTensor(mean=self.mean, std=self.std).build(),
            v2.Normalize(mean=self.mean, std=self.std),
        ])
        self.val_tf = SimpleImageValAugment(mean=self.mean,std=self.std).build()
        self.dataset_cls = CIFAR100Dataset


# ----------------------------
# TinyImageNet-200
# ----------------------------
class TinyImageNetDataModule(DataSetModule):
    def __init__(self, config: "DataModuleConfig"):
        super().__init__(config)
        self.num_classes = 200
        self.mean = (0.4802, 0.4481, 0.3975)
        self.std = (0.2302, 0.2265, 0.2262)
        self.train_tf = SimpleImageTrainAugment(mean=self.mean,std=self.std).build()
        self.val_tf = SimpleImageValAugment(mean=self.mean,std=self.std).build()
        self.dataset_cls = TinyImageNetDataset


# ----------------------------
# MNIST
# ----------------------------
class MNISTDataset(datasets.MNIST):
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img, target = self.data[index], self.targets[index]
        img:np.ndarray = img.numpy()
        img:np.ndarray = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
        if self.transform is not None:
            if isinstance(self.transform, (BaseCompose, BasicTransform)):
                img = self.transform(image=img)["image"]
            else:
                img = self.transform(Image.fromarray(img))
        if self.target_transform is not None:
            target = self.target_transform(target)
        img = img.mean(0,keepdim=True)
        return img, target
    
class MNISTDataModule(DataSetModule):
    def __init__(self, config: "DataModuleConfig"):
        super().__init__(config)
        self.num_classes = 10
        self.mean = (0.1307,)
        self.std = (0.3081,)
        # RandAugment can be a bit odd on grayscale; enable if you know it's fine in your env.
        self.train_tf = SimpleImageTrainAugment(mean=self.mean,std=self.std, flip=False).build()
        self.val_tf = SimpleImageValAugment(mean=self.mean,std=self.std).build()
        self.dataset_cls = MNISTDataset
        # CutMix on MNIST is typically not useful; keep it off by default.
        self.cutmix = False


# ----------------------------
# RetinaFace
# ----------------------------
class RetinaFaceDataModule(DataSetModule):
    def __init__(self, config: "DataModuleConfig"):
        super().__init__(config)
        self.num_classes = 100
        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)
        self.p = 0.5
        self.train_tf = A.Compose([
            A.Rotate(limit=(-10,10), p=self.p),
            A.HorizontalFlip(p=self.p),
            AutoRandomResizedCrop(scale=(0.75, 1.0), ratio=(0.75, 1.33),p=self.p),
            # A.OneOf([
            #     A.ColorJitter(p=1.0,brightness=0.3, contrast=0.3, saturation=0.3, hue=0.2),
            #     A.ToGray(p=1.0),
            # ], p=self.p),
            # A.OneOf([
            #     A.GaussianBlur(p=1.0,blur_limit=(1, 3)),
            #     A.GaussNoise(p=1.0,std_range=self.noise_std_range),
            # ], p=self.p),
            #  A.CoarseDropout(num_holes_range=(1,4), 
            #                  hole_height_range=(rmin, rmax), 
            #                  hole_width_range=(rmin, rmax), p=self.p),
            A.Normalize(mean=self.mean, std=self.std),
            A.ToTensorV2(),
        ])
        self.train_tf = v2.Compose([
            v2.RandomHorizontalFlip(p=self.p),

            v2.RandomApply(
                [v2.RandomChoice([
                    v2.AutoAugment(
                        policy=v2.AutoAugmentPolicy.IMAGENET,
                        interpolation=InterpolationMode.BILINEAR,
                    ),
                    v2.RandAugment(num_ops=2, magnitude=9),
                    v2.AugMix(severity=3, mixture_width=3, chain_depth=-1, alpha=1.0),
                ])],
                p=self.p
            ),
            v2.Resize((640,640)),
            ToTensor(mean=self.mean, std=self.std).build(),
            v2.Normalize(mean=self.mean, std=self.std),
        ])
        self.val_tf = SimpleImageValAugment(mean=self.mean,std=self.std).build()
        self.dataset_cls = RetinaFaceDataset

# ----------------------------
# ImageNet DataModule
# ----------------------------
class ImageNetDataModule(DataSetModule):
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
        train_tf = v2.Compose([
            v2.RandomResizedCrop(self.image_size, interpolation=InterpolationMode.BICUBIC),
            v2.RandomHorizontalFlip(),
            v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
            v2.Normalize(self.mean, self.std),
        ])
        val_tf = v2.Compose([
            v2.Resize(self.val_resize, interpolation=InterpolationMode.BICUBIC),
            v2.CenterCrop(self.image_size),
            v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
            v2.Normalize(self.mean, self.std),
        ])

        # ImageFolder expects subfolders per class
        self.train_ds = datasets.ImageFolder(root=f"{self.data_dir}/train", transform=train_tf)
        self.val_ds   = datasets.ImageFolder(root=f"{self.data_dir}/val",   transform=val_tf)

        # (Optional) you can inspect class count if needed:
        # self.num_classes = len(self.train_ds.classes)

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
               and returns a transformed version. E.g, ``v2.RandomCrop``
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

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        super().__init__(root, transform=transform, target_transform=target_transform)

        self.dataset_path = os.path.join(root, self.base_folder)
        self.loader = default_loader
        self.split = "train" if train else "val"

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

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img_path, target = self.data[index]
        img = self.loader(img_path)
        img:np.ndarray = np.asarray(img)
        if self.transform is not None:
            if isinstance(self.transform, (BaseCompose, BasicTransform)):
                img = self.transform(image=img)["image"]
            else:
                img = self.transform(Image.fromarray(img))
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target
    
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


# -----------------------------------------------------------------------------
# RetinaFace / DataModule
# -----------------------------------------------------------------------------
class RetinaFaceTensor(BaseModel):
    label:Optional[Any] = Field(default=None,exclude=True)
    bbox:Optional[Any] = Field(default=None,exclude=True)
    landmark:Optional[Any] = Field(default=None,exclude=True)

class RetinaFaceDataset(VisionDataset):
    
    """
    Yep — for *geometric* aug (resize/crop/flip/affine), you really want **joint transforms** that see **(image, boxes, landmarks)** together.

    The cleanest way in modern torchvision is:

    * treat `boxes` as `tv_tensors.BoundingBoxes`
    * treat `landmarks` as `tv_tensors.KeyPoints` (shape `[N, 5, 2]`)
    * run `torchvision.transforms.v2` so the same ops update everything consistently ([PyTorch Documentation][1])
    * note: **KeyPoints support landed in torchvision 0.23** (beta feature) ([PyTorch Documentation][2])

    Below is a practical pattern that also handles the one RetinaFace-specific quirk: **when you flip horizontally, you must swap left/right landmark indices** (eye/mouth corners). torchvision can flip keypoint coordinates, but it doesn’t know your semantic ordering.

    ## Joint transforms for RetinaFace (boxes + landmarks)

    ```py
    import torch
    from torchvision import tv_tensors
    from torchvision.transforms import v2
    from torchvision.transforms.v2 import functional as F


    class WrapRetinaFaceTargets:
        Convert plain tensors to TVTensors so v2 transforms update them.
        def __call__(self, img, target):
            H, W = map(int, F.get_size(img))  # [H, W]
            target = dict(target)

            boxes = target["boxes"]
            if boxes.numel() == 0:
                boxes_tv = tv_tensors.BoundingBoxes(boxes.view(-1, 4), format="XYXY", canvas_size=(H, W))
            else:
                boxes_tv = tv_tensors.BoundingBoxes(boxes.view(-1, 4), format="XYXY", canvas_size=(H, W))

            # landmarks: (N,10) with -1 for missing -> KeyPoints (N,5,2) + valid mask
            lm = target.get("landmarks", None)
            if lm is None or lm.numel() == 0:
                kp = torch.zeros((boxes_tv.shape[0], 5, 2), dtype=torch.float32)
                valid = torch.zeros((boxes_tv.shape[0], 5), dtype=torch.bool)
            else:
                kp = lm.view(-1, 5, 2).to(torch.float32).clone()
                valid = (kp[..., 0] >= 0) & (kp[..., 1] >= 0)
                kp[~valid] = 0.0  # placeholder; we keep "valid" separately

            kps_tv = tv_tensors.KeyPoints(kp, canvas_size=(H, W))

            target["boxes"] = boxes_tv
            target["keypoints"] = kps_tv
            target["landmarks_valid"] = valid
            return img, target


    class RandomFaceHorizontalFlip:
        Flip image/boxes/keypoints and also swap left/right landmark indices.
        def __init__(self, p=0.5):
            self.p = float(p)

        def __call__(self, img, target):
            if torch.rand(()) >= self.p:
                return img, target

            img = F.horizontal_flip(img)
            target = dict(target)

            target["boxes"] = F.horizontal_flip(target["boxes"])
            kps = F.horizontal_flip(target["keypoints"])      # (N,5,2), coords flipped
            kps = kps[:, [1, 0, 2, 4, 3], :]                  # swap le<->re and lm<->rm
            target["keypoints"] = kps

            if "landmarks_valid" in target:
                target["landmarks_valid"] = target["landmarks_valid"][:, [1, 0, 2, 4, 3]]

            return img, target


    class UnwrapAndNormalizeRetinaFace:
        Convert TVTensors back to your model format: boxes/landmarks normalized to [0,1] and -1 missing.
        def __call__(self, img, target):
            H, W = map(int, F.get_size(img))
            target = dict(target)

            boxes = torch.as_tensor(target["boxes"], dtype=torch.float32).view(-1, 4)
            if boxes.numel():
                boxes[:, [0, 2]] /= float(W)
                boxes[:, [1, 3]] /= float(H)
                boxes = boxes.clamp(0.0, 1.0)

            kps = torch.as_tensor(target["keypoints"], dtype=torch.float32)  # (N,5,2)
            valid = target.get("landmarks_valid", torch.ones(kps.shape[:2], dtype=torch.bool))

            # after crop/resize, some points may go out of bounds: mark them missing
            within = (kps[..., 0] >= 0) & (kps[..., 0] <= W) & (kps[..., 1] >= 0) & (kps[..., 1] <= H)
            valid = valid & within

            kps[..., 0] /= float(W)
            kps[..., 1] /= float(H)
            kps = kps.clamp(0.0, 1.0)
            kps[~valid] = -1.0

            target["boxes"] = boxes
            target["landmarks"] = kps.view(-1, 10)

            target.pop("keypoints", None)
            target.pop("landmarks_valid", None)
            return img, target
    ```

    ### Example train transform pipeline

    ```py
    mean = (0.485, 0.456, 0.406)
    std  = (0.229, 0.224, 0.225)

    train_tf = v2.Compose([
        v2.ToImage(),                            # makes image a tensor TVTensor
        WrapRetinaFaceTargets(),                 # boxes->BoundingBoxes, landmarks->KeyPoints
        v2.RandomResizedCrop((640, 640), antialias=True),
        RandomFaceHorizontalFlip(p=0.5),         # flip + swap landmark order
        v2.ClampBoundingBoxes(),                 # optional :contentReference[oaicite:2]{index=2}
        v2.ClampKeyPoints(),                     # optional :contentReference[oaicite:3]{index=3}
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=mean, std=std),
        UnwrapAndNormalizeRetinaFace(),          # back to your normalized [0,1] + -1 missing format
    ])
    ```

    ### Dataset usage

    Use the **joint** `transforms=` argument (not `transform=`), since `transform=` is image-only:

    ```py
    ds = RetinaFaceDataset(
        root=self.data_dir,
        train=True,
        download=True,
        transforms=train_tf,
    )
    ```

    If you paste your current `train_tf` (what ops you want: resize? mosaic? iou-crop? affine?), I can tailor the pipeline so every step correctly updates **boxes + landmarks** and matches RetinaFace’s expected input/output shapes.

    [1]: https://docs.pytorch.org/vision/stable/transforms.html "Transforming images, videos, boxes and more — Torchvision 0.24 documentation"
    [2]: https://docs.pytorch.org/vision/main/auto_examples/transforms/plot_keypoints_transforms.html "Transforms on KeyPoints — Torchvision main documentation"



    Expected layout (default):
      root/
        annotations/
          train.json
          val.json
        (images referenced by JSON paths)

    JSON entries:
      {
        "file": "...",
        "width": 1024, "height": 768,          # optional
        "boxes": [[x1,y1,x2,y2], ...],          # absolute pixels
        "landmarks": [[x1,y1, ..., x5,y5], ...] # optional, absolute pixels OR -1 for missing
      }

    Outputs:
      image: tensor or PIL (depending on transforms)
      target:
        boxes: (N,4) in [0,1]
        landmarks: (N,10) in [0,1] or -1 for missing
        labels: (N,) long (all ones)
    """

    # Hugging Face mirror (default)
    HF_WIDER_REPO = "https://huggingface.co/datasets/wider_face/resolve/main/data"
    DEFAULT_WIDER_URLS = {
        "train": f"{HF_WIDER_REPO}/WIDER_train.zip",
        "val": f"{HF_WIDER_REPO}/WIDER_val.zip",
        "test": f"{HF_WIDER_REPO}/WIDER_test.zip",  # unused here (no gt)
        "split": f"{HF_WIDER_REPO}/wider_face_split.zip",
    }

    def __init__(
        self,
        root: str,
        train: bool = True,
        download: bool = False,
        transform=None,
        target_transform=None,
        image_size: int = 640,  # kept for compatibility; unused internally
        annotation_file: Optional[str] = None,
        annotations_dir: str = "annotations",
        train_ann: str = "train.json",
        val_ann: str = "val.json",
        download_url: Optional[str] = None,  # optional override (single URL)
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)

        self.root = Path(root)
        self.train = bool(train)
        self.download = bool(download)
        self.image_size = int(image_size)  # unused
        self.download_url = download_url

        # pick annotation file
        if annotation_file is None:
            ann_name = train_ann if self.train else val_ann
            annotation_file = str(Path(annotations_dir) / ann_name)

        self.ann_path = Path(annotation_file)
        if not self.ann_path.is_absolute():
            self.ann_path = self.root / self.ann_path

        if self.download and not self.ann_path.exists():
            self._download()

        if not self.ann_path.exists():
            raise FileNotFoundError(f"Annotation file not found: {self.ann_path}")

        with open(self.ann_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(f"Annotation file {self.ann_path} must contain a list of samples.")
        self.items: List[Dict[str, Any]] = data

    def __len__(self) -> int:
        return len(self.items)

    def _resolve_path(self, file_path: str) -> Path:
        p = Path(file_path)
        return p if p.is_absolute() else (self.root / p)

    @staticmethod
    def _convert_wider_txt_to_json(txt_path: Path, images_prefix_rel: Path, out_json: Path) -> None:
        items: List[Dict[str, Any]] = []
        IMG_EXTS = (".jpg", ".jpeg", ".png")

        def is_image_line(s: str) -> bool:
            s = s.lower()
            return s.endswith(IMG_EXTS)

        with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
            while True:
                img_rel = f.readline()
                if not img_rel:
                    break
                img_rel = img_rel.strip()
                if not img_rel:
                    continue

                # Guard: sometimes a dummy bbox line can be encountered here if parsing got off.
                if not is_image_line(img_rel):
                    continue

                n_line = f.readline()
                if not n_line:
                    break
                n_line = n_line.strip()

                try:
                    n = int(n_line)
                except ValueError:
                    # If this happens, the file is desynced or it's a filelist-style txt.
                    # Treat as 0 faces and continue.
                    n = 0

                boxes: List[List[float]] = []
                for _ in range(n):
                    line = f.readline()
                    if not line:
                        break
                    parts = line.strip().split()
                    if len(parts) < 4:
                        continue
                    x, y, w, h = map(float, parts[:4])
                    boxes.append([x, y, x + w, y + h])

                # IMPORTANT: WIDER sometimes has an extra dummy bbox row even when n == 0.
                if n == 0:
                    pos = f.tell()
                    maybe = f.readline()
                    if maybe:
                        maybe = maybe.strip()
                        if is_image_line(maybe):
                            # It was actually the next image line; rewind so outer loop reads it.
                            f.seek(pos)
                        # else: it was the dummy "0 0 0 0 ..." line; we intentionally dropped it.

                file_rel = (images_prefix_rel / img_rel).as_posix()
                items.append({"file": file_rel, "boxes": boxes})

        out_json.parent.mkdir(parents=True, exist_ok=True)
        with open(out_json, "w", encoding="utf-8") as g:
            json.dump(items, g, ensure_ascii=False)

    def _download(self) -> None:
        """
        Default behavior (if download_url is None):
          - downloads WIDER_train.zip OR WIDER_val.zip depending on self.train
          - downloads wider_face_split.zip
          - extracts into: <root>/widerface/
          - converts split txt annotation into: <root>/annotations/train.json or val.json

        If download_url is provided, it is treated as a single archive/file override
        and no WIDER conversion is attempted.
        """
        from torchvision.datasets.utils import download_and_extract_archive

        # If already present, nothing to do.
        if self.ann_path.exists():
            return

        # If the user provided a custom URL override, do the simplest possible thing.
        if self.download_url:
            self.root.mkdir(parents=True, exist_ok=True)
            download_and_extract_archive(str(self.download_url), download_root=str(self.root))
            return

        # Default: WIDER FACE mirrors from Hugging Face
        urls = self.DEFAULT_WIDER_URLS

        # Use torchvision's conventional folder name to keep things tidy
        wider_root = self.root / "widerface"
        wider_root.mkdir(parents=True, exist_ok=True)

        # Always need split to build annotations
        split_dir = wider_root / "wider_face_split"
        if not split_dir.exists():
            download_and_extract_archive(urls["split"], download_root=str(wider_root))

        # Download only the image subset we need for this dataset instance
        subset = "train" if self.train else "val"
        subset_dir = wider_root / ("WIDER_train" if self.train else "WIDER_val")
        if not subset_dir.exists():
            download_and_extract_archive(urls[subset], download_root=str(wider_root))

        # Locate split txt files (robust to nesting)
        def find_one(name: str) -> Path:
            hits = list(wider_root.rglob(name))
            if len(hits) != 1:
                raise FileNotFoundError(
                    f"Expected exactly 1 '{name}' under {wider_root}, found {len(hits)}."
                )
            return hits[0]

        if self.train:
            gt_txt = find_one("wider_face_train_bbx_gt.txt")
            out_json = self.root / "annotations" / "train.json"
            images_prefix_rel = Path("widerface") / "WIDER_train" / "images"
        else:
            gt_txt = find_one("wider_face_val_bbx_gt.txt")
            out_json = self.root / "annotations" / "val.json"
            images_prefix_rel = Path("widerface") / "WIDER_val" / "images"

        if not out_json.exists():
            self._convert_wider_txt_to_json(gt_txt, images_prefix_rel, out_json)

        # If this dataset instance expects the default annotation path, it should now exist.
        # If the user passed a different annotation_file path, they should point it to out_json.
        # We only auto-switch if it matches the default name (train.json/val.json).
        if self.ann_path.name in ("train.json", "val.json") and out_json.exists():
            self.ann_path = out_json

    def __getitemdata__(self, idx: int) -> Tuple[Any, Dict[str, torch.Tensor]]:
        entry = self.items[idx]

        img_path = self._resolve_path(entry["file"])
        img = ImageOps.exif_transpose(Image.open(img_path)).convert("RGB")

        orig_w = int(entry.get("width", img.width))
        orig_h = int(entry.get("height", img.height))
        orig_w = max(orig_w, 1)
        orig_h = max(orig_h, 1)

        boxes = torch.tensor(entry.get("boxes", []), dtype=torch.float32).view(-1, 4)
        if boxes.numel() == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)

        if boxes.numel():
            boxes[:, [0, 2]] /= float(orig_w)
            boxes[:, [1, 3]] /= float(orig_h)
            boxes = boxes.clamp(0.0, 1.0)

        land = entry.get("landmarks", [])
        if not land:
            landmarks = torch.full((boxes.size(0), 10), -1.0, dtype=torch.float32)
        else:
            landmarks = torch.tensor(land, dtype=torch.float32).view(-1, 10)
            if landmarks.size(0) != boxes.size(0):
                raise ValueError(
                    f"Mismatch: {boxes.size(0)} boxes but {landmarks.size(0)} landmark rows "
                    f"for file={entry.get('file')}"
                )
            lm = landmarks.view(-1, 5, 2)
            valid = lm[..., 0] >= 0
            lm_x = lm[..., 0].clone()
            lm_y = lm[..., 1].clone()
            lm_x[valid] = (lm_x[valid] / float(orig_w)).clamp(0.0, 1.0)
            lm_y[valid] = (lm_y[valid] / float(orig_h)).clamp(0.0, 1.0)
            lm[..., 0] = lm_x
            lm[..., 1] = lm_y
            landmarks = lm.view(-1, 10)
        target = RetinaFaceTensor(label=torch.ones((boxes.size(0),), dtype=torch.long),
                                  bbox=boxes,
                                  landmark=landmarks).model_dump()
        return img, target

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img, target = self.__getitemdata__(index)
        img:np.ndarray = np.asarray(img)
        if self.transform is not None:
            if isinstance(self.transform, (BaseCompose, BasicTransform)):
                img = self.transform(image=img)["image"]
            else:
                img = self.transform(Image.fromarray(img))
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

if __name__ == "__main__":
    for n in ['retinaface']:#,'mnist','cifar100','timnet']:
        ds = DataModuleConfig(dataset_name=n,data_dir='./data',batch_size=32,mixup=False,cutmix=False).build()
        ds.setup()
        for i in range(4):
            ds.show_examples(32)
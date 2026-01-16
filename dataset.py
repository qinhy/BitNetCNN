import copy
import json
import os
import math
from pathlib import Path
from PIL import Image, ImageOps
from typing import Any, Callable, Optional, Sequence, Tuple, Union, List, Dict

import cv2
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
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
from torchvision.ops import nms, box_iou
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

    def collate_fn(self, batch):
        x, y = default_collate(batch)
        if self._collate_transform is not None:
            x, y = self._collate_transform(x, y)
        return x, y

    def train_dataloader(self) -> DataLoader:
        collate_fn = self.collate_fn if self._collate_transform is not None else None
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
        collate_fn = self.collate_fn if self._collate_transform is not None else None
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=(self.num_workers > 0),
            collate_fn=collate_fn,
        )

    @staticmethod
    def show_examples_static(
        x,
        y,
        mean,std,
        class_names=None,
        n: int = 16,
        cols: int = 8,
        figsize=(12, 6),
    ):

        # ---- handle x being Tensor or list of Tensors (variable-sized images) ----
        if torch.is_tensor(x):
            x = x[:n].detach().cpu()
            x_list = [x[i] for i in range(x.shape[0])]
        else:
            x_list = [xi.detach().cpu() for xi in list(x)[:n]]

        n = min(n, len(x_list))
        x_list = x_list[:n]

        # ---- move y to CPU if possible ----
        y_is_list = isinstance(y, (list, tuple))
        if y_is_list:
            y_list = list(y)[:n]
        else:
            try:
                y_list = y[:n].detach().cpu()
            except Exception:
                y_list = None

        # ---- dataset class names (if classification) ----

        def label_to_text(target):
            # for classification display
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

        def denorm_for_vis(img_t: torch.Tensor,mean=mean,std=std) -> torch.Tensor:
            # img_t: (C,H,W)
            if not torch.is_tensor(img_t) or img_t.ndim != 3:
                return img_t
            # if already uint8, just scale to [0,1] for matplotlib
            if img_t.dtype == torch.uint8:
                return (img_t.float() / 255.0).clamp(0, 1)

            mean = torch.tensor(mean, dtype=img_t.dtype).view(-1, 1, 1)
            std = torch.tensor(std, dtype=img_t.dtype).view(-1, 1, 1)
            return (img_t * std + mean).clamp(0, 1)

        def is_retina_target(t) -> bool:
            # supports RetinaFaceTensor or dict with bbox/landmark
            if t is None:
                return False
            if hasattr(t, "bbox") and hasattr(t, "landmark"):
                return True
            if isinstance(t, dict) and ("bbox" in t) and ("landmark" in t):
                return True
            return False

        # ---- decide whether this is detection-style batch ----
        detection = False
        if y_is_list and len(y_list) > 0 and is_retina_target(y_list[0]):
            detection = True

        cols = max(1, min(cols, n))
        rows = math.ceil(n / cols)
        plt.figure(figsize=figsize)

        for i in range(n):
            plt.subplot(rows, cols, i + 1)

            img_vis = denorm_for_vis(x_list[i])
            img_np = img_vis.permute(1, 2, 0).numpy()
            H, W = img_np.shape[:2]

            plt.imshow(img_np)
            plt.axis("off")

            title = "null"
            if detection and y_list is not None:
                t = y_list[i]
                if is_retina_target(t):
                    t:RetinaFaceTensor = t
                    boxes = torch.as_tensor(t.bbox).detach().cpu().view(-1, 4)

                    # draw boxes
                    for b in boxes:
                        x1, y1, x2, y2 = [float(v) for v in b.tolist()]
                        w = max(0.0, x2 - x1)
                        h = max(0.0, y2 - y1)
                        plt.gca().add_patch(Rectangle((x1, y1), w, h, fill=False, linewidth=1.0))

                    xs,ys = t.landmark_xs_ys()
                    if xs.size > 0:
                        plt.scatter(xs, ys, s=6)

                    title = f"faces:{boxes.shape[0]}"

            elif y_list is not None:
                # classification display
                title = label_to_text(y_list[i])

            plt.title(title, fontsize=8)

        plt.tight_layout()
        plt.show()

        # return something consistent with your old function
        if torch.is_tensor(x):
            x_out = x[:n]
        else:
            x_out = x_list
        return x_out, y_list
    @torch.no_grad()
    def show_examples(
        self,
        n: int = 16,
        split: str = "train",
        cols: int = 8,
        seed: int | None = None,
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

        batch = next(iter(loader))
        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
            x, y = batch[0], batch[1]
        else:
            raise ValueError("Dataloader must return (x, y)")
        ds = self.train_ds if split == "train" else self.val_ds
        self.show_examples_static(x, y, self.mean, self.std,
                                    getattr(ds, "classes", None),
                                    n, cols, figsize)
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
    file: str = ""
    img_h: int = Field(default=0)
    img_w: int = Field(default=0)
    img:Optional[Any] = Field(default=None,exclude=True)
    label:Optional[Any] = Field(default=None,exclude=True)
    # xyxy
    # tensor : Normalized coords (ratio, 0–1)
    # numpy : Pixel coords (up to image size)
    bbox:Optional[Any] = Field(default=None,exclude=True) 
    landmark:Optional[Any] = Field(default=None,exclude=True) # [lx, ly, rx, ry, nx, ny, mlx, mly, mrx, mry] as shape (N * 5,2)
    landmark_vis:Optional[Any] = Field(default=None,exclude=True) # [lv, rv, nv, mlv, mrv] as shape (N * 5)

    def model_post_init(self, context):
        if self.img is None:
            if not self.file:
                raise ValueError("RetinaFaceTensor requires file when img is None.")
            img_pil = ImageOps.exif_transpose(Image.open(self.file)).convert("RGB")
            self.img = np.asarray(img_pil)  # (H,W,3) uint8
            self.img_h, self.img_w = self.img.shape[:2]
            
            if self.bbox is None:
                raise ValueError(f"bbox is required for file={self.file}")
            self.bbox, keep = self._sanitize_xyxy(self.bbox, self.img_h, self.img_w, min_size=1.0)
            # if len(keep) != keep.sum():
            #     raise ValueError("invalid bbox included, for file={self.file}")
            # --- landmark: raw pixels (5,2), missing = -1 (numpy float32) ---
            land = self.landmark
            if land is None or len(land) == 0:
                num_boxes = int(self.bbox.shape[0]) if self.bbox is not None else 0
                self.landmark = np.full((num_boxes * 5, 2), -1.0, dtype=np.float32)
                self.landmark_vis = np.full((num_boxes * 5,), -1.0, dtype=np.float32)
            else:
                landmark = np.asarray(land, dtype=np.float32).reshape(-1, 15)[keep]
                if landmark.shape[0] != self.bbox.shape[0]:
                    raise ValueError(
                        f"Mismatch: {self.bbox.shape[0]} boxes but {landmark.shape[0]} landmark rows "
                        f"for file={self.file}"
                    )

                lm = landmark.reshape(-1, 5, 3).copy()
                valid = (lm[..., 0] >= 0) & (lm[..., 1] >= 0)

                # force invalid points to -1 exactly
                lm[..., 0][~valid] = -1.0
                lm[..., 1][~valid] = -1.0
                lm[..., 2][~valid] = -1.0

                lm = lm.reshape(-1, 5, 3).reshape(-1,3)
                self.landmark = lm[:,:2]
                self.landmark_vis = lm[:,2]
            
            self.label = np.ones((self.bbox.shape[0],), dtype=np.int64)
        return super().model_post_init(context)
    
    def _sanitize_xyxy(self,bboxes, h, w, min_size=1.0):
        b = np.asarray(bboxes, dtype=np.float32).reshape(-1, 4)
        x1, y1, x2, y2 = b.T

        # Fix swapped coords
        x1_, x2_ = np.minimum(x1, x2), np.maximum(x1, x2)
        y1_, y2_ = np.minimum(y1, y2), np.maximum(y1, y2)

        # Clip to image bounds (Albumentations allows x==w, y==h for pascal_voc)
        x1_ = np.clip(x1_, 0.0, float(w))
        x2_ = np.clip(x2_, 0.0, float(w))
        y1_ = np.clip(y1_, 0.0, float(h))
        y2_ = np.clip(y2_, 0.0, float(h))

        bw = x2_ - x1_
        bh = y2_ - y1_
        keep = (bw >= min_size) & (bh >= min_size) & np.isfinite(bw) & np.isfinite(bh)

        b2 = np.stack([x1_, y1_, x2_, y2_], axis=1)
        return b2[keep], keep
    
    def landmark_xs_ys(self):
        pts = self.landmark
        valid = (pts[..., 0] >= 0) & (pts[..., 1] >= 0)
        xs = pts[..., 0][valid]
        ys = pts[..., 1][valid]
        if type(pts) is torch.Tensor:
            xs = xs.numpy()
            ys = ys.numpy()
        return xs,ys
    
    def norm_to_0_1(self, clip: bool = True):    
        if type(self.img) is torch.Tensor:
            C, self.img_h, self.img_w = self.img.shape
        else:
            self.img_h, self.img_w = self.img.shape[:2]

        """Normalize bbox/landmarks from pixel coords to [0,1]. Keeps -1 as missing."""
        bbox,landmark = None,None
        # bbox: xyxy
        if self.bbox is not None:
            b_norm = np.asarray(self.bbox, dtype=np.float32)
            if b_norm.size == 0:
                b_norm = b_norm.reshape(0, 4)
            else:
                b_norm = b_norm.reshape(-1, 4)
            whwh = np.asarray([self.img_w, self.img_h, self.img_w, self.img_h], dtype=np.float32)
            b_norm /= whwh
            if clip:
                b_norm = np.clip(b_norm, 0.0, 1.0)
            bbox = b_norm
        else:
            bbox = np.zeros((0, 4), dtype=np.float32)

        # landmarks: (N,5,2)
        if self.landmark is not None:
            p_norm = np.asarray(self.landmark, dtype=np.float32)
            if p_norm.size == 0:
                p_norm = p_norm.reshape(0, 2)
            elif p_norm.ndim == 1:
                p_norm = p_norm.reshape(-1, 2)
            neg_mask = p_norm < 0
            p_norm[..., 0] /= float(self.img_w)
            p_norm[..., 1] /= float(self.img_h)
            if clip:
                p_norm = np.clip(p_norm, 0.0, 1.0)
                p_norm[neg_mask] = -1.0
            landmark = p_norm
        else:
            landmark = np.zeros((0, 2), dtype=np.float32)
        return bbox,landmark
    
    def clone(self, update={}, deep=False):
        if not deep:
            return self.model_copy(update=update)
        else:
            return copy.deepcopy(self)

    def as_tensor(self):
        if (
            torch.is_tensor(self.bbox)
            and torch.is_tensor(self.label)
            and torch.is_tensor(self.landmark)
            and torch.is_tensor(self.landmark_vis)
        ):
            return self
        bbox,landmark = self.norm_to_0_1()
        landmark_vis = self.landmark_vis
        if landmark_vis is None:
            if landmark is None:
                landmark_vis = np.zeros((0,), dtype=np.float32)
            else:
                lmk = np.asarray(landmark)
                count = lmk.reshape(-1, 2).shape[0] if lmk.size else 0
                landmark_vis = np.full((count,), -1.0, dtype=np.float32)
        res = self.clone(update=dict(bbox=bbox,landmark=landmark,
                                     landmark_vis=landmark_vis))
        res.bbox = torch.as_tensor(res.bbox)
        res.landmark = torch.as_tensor(res.landmark)
        res.landmark_vis = torch.as_tensor(res.landmark_vis)
        return res
        
    def norm_to_pixel(self):
        """Normalize bbox/landmarks from [0,1] to pixel coords."""
        bbox,landmark = None,None
        # bbox: xyxy
        if self.bbox is not None:
            b_norm = self.bbox.clone() #xyxy in pixel
            whwh = torch.Tensor([self.img_w, self.img_h, self.img_w, self.img_h]).to(dtype=torch.float32)
            b_norm *= whwh
            bbox = b_norm.detach().cpu().numpy()
        # landmarks: (N,5,2)
        if self.landmark is not None:
            p_norm = self.landmark.clone().reshape(-1, 2)
            wh = torch.Tensor([self.img_w, self.img_h]).to(dtype=torch.float32)
            p_norm *= wh
            landmark = p_norm.detach().cpu().numpy().reshape(-1,5,2) # (N,5,2)
        return bbox,landmark
    
    def as_numpy(self):
        bbox,landmark = self.norm_to_pixel()
        return self.clone(update=dict(bbox=bbox,landmark=landmark,
                                     landmark_vis=self.landmark_vis))

    @staticmethod
    def encode_boxes(
        anchors: torch.Tensor,
        gt_boxes_xyxy: torch.Tensor,
        variances: Tuple[float, float] = (0.1, 0.2),
        eps: float = 1e-6,
    ) -> torch.Tensor:
        """
        anchors:       [N,4] (cx,cy,w,h)
        gt_boxes_xyxy: [N,4] (x1,y1,x2,y2) in same frame as anchors
        returns:       [N,4] (dx,dy,dw,dh)
        """
        if anchors.ndim != 2 or anchors.shape[1] != 4:
            raise ValueError(f"anchors must be [N,4], got {tuple(anchors.shape)}")
        if gt_boxes_xyxy.ndim != 2 or gt_boxes_xyxy.shape[1] != 4:
            raise ValueError(f"gt_boxes_xyxy must be [N,4], got {tuple(gt_boxes_xyxy.shape)}")
        if anchors.shape[0] != gt_boxes_xyxy.shape[0]:
            raise ValueError("anchors and gt_boxes_xyxy must have same N")

        v0, v1 = variances

        a_xy = anchors[:, 0:2]               # (cx, cy)
        a_wh = anchors[:, 2:4].clamp_min(eps)  # (aw, ah) clamped exactly like before

        b1 = gt_boxes_xyxy[:, 0:2]           # (x1, y1)
        b2 = gt_boxes_xyxy[:, 2:4]           # (x2, y2)
        g_xy = (b1 + b2) * 0.5               # (gx, gy)
        g_wh = (b2 - b1).clamp_min(eps)      # (gw, gh) clamped exactly like before

        d_xy = (g_xy - a_xy) / a_wh / v0
        d_wh = torch.log(g_wh / a_wh) / v1

        return torch.cat((d_xy, d_wh), dim=1)

    @staticmethod
    def decode_boxes(
        anchors: torch.Tensor,
        deltas: torch.Tensor,
        variances: Tuple[float, float] = (0.1, 0.2),
        clip: bool = False,
    ) -> torch.Tensor:
        """
        anchors:  [N,4] (cx,cy,w,h)
        deltas:   [N,4] (dx,dy,dw,dh)
        returns:  [N,4] (x1,y1,x2,y2) in same frame as anchors (often normalized)
        """
        if anchors.ndim != 2 or anchors.shape[1] != 4:
            raise ValueError(f"anchors must be [N,4], got {tuple(anchors.shape)}")
        if deltas.ndim != 2 or deltas.shape[1] != 4:
            raise ValueError(f"deltas must be [N,4], got {tuple(deltas.shape)}")
        if anchors.shape[0] != deltas.shape[0]:
            raise ValueError("anchors and deltas must have same N")

        v0, v1 = variances

        a_xy = anchors[:, 0:2]  # (cx, cy)
        a_wh = anchors[:, 2:4]  # (aw, ah)

        d_xy = deltas[:, 0:2]   # (dx, dy)
        d_wh = deltas[:, 2:4]   # (dw, dh)

        g_xy = a_xy + d_xy * v0 * a_wh
        g_wh = a_wh * torch.exp(d_wh * v1)

        x1y1 = g_xy - 0.5 * g_wh
        x2y2 = g_xy + 0.5 * g_wh

        out = torch.cat((x1y1, x2y2), dim=1)
        if clip:
            out = out.clamp(0.0, 1.0)
        return out

    @staticmethod
    def encode_landmarks(
        anchors: torch.Tensor,
        gt_landmarks: torch.Tensor,
        variances: Tuple[float, float] = (0.1, 0.2),
    ) -> torch.Tensor:
        """
        anchors:      [N,4]  (cx,cy,w,h)
        gt_landmarks: [N,10] (x1,y1,...,x5,y5) same frame as anchors
        returns:      [N,10] (dx1,dy1,...,dx5,dy5)
        """
        if anchors.shape[0] != gt_landmarks.shape[0]:
            raise ValueError("anchors and gt_landmarks must have same N")

        v0 = variances[0]

        lm = gt_landmarks.reshape(-1, 5, 2)   # [N,5,2]
        a = anchors[:, None, :]               # [N,1,4] for broadcasting

        # dx = (x - cx) / (w * v0), dy = (y - cy) / (h * v0)
        d = (lm - a[..., 0:2]) / (a[..., 2:4] * v0)  # [N,5,2]
        return d.reshape(-1, 10)

    @staticmethod
    def decode_landmarks(
        anchors: torch.Tensor,
        landm_deltas: torch.Tensor,
        variances: Tuple[float, float] = (0.1, 0.2),
    ) -> torch.Tensor:
        """
        anchors:      [N,4]  (cx,cy,w,h)
        landm_deltas: [N,10] (dx1,dy1,...,dx5,dy5)
        returns:      [N,10] (x1,y1,...,x5,y5) same frame as anchors
        """
        if anchors.shape[0] != landm_deltas.shape[0]:
            raise ValueError("anchors and landm_deltas must have same N")

        v0 = variances[0]

        d = landm_deltas.reshape(-1, 5, 2)  # [N,5,2]
        a = anchors[:, None, :]             # [N,1,4]

        xy = a[..., 0:2] + d * v0 * a[..., 2:4]  # [N,5,2]
        return xy.reshape(-1, 10)

    @staticmethod
    def match_anchors(anchors:torch.Tensor, gt_boxes:torch.Tensor, gt_landmarks:torch.Tensor,
                      pos_iou, neg_iou, variances):
        """
        Matches anchors to Ground Truth.
        Returns:
            labels: [N] (-1: ignore, 0: bg, 1: face)
            bbox_targets: [N, 4]
            landmark_targets: [N, 10]
        """
        # landmark is [N, 10] or [N, 5, 2]. Ensure [N, 5, 2] for match_anchors
        if gt_landmarks.dim() == 2:
            gt_landmarks = gt_landmarks.view(-1, 5, 2)
        else:
            gt_landmarks = gt_landmarks

        if gt_boxes.numel() == 0:
            device = anchors.device
            return (torch.zeros(anchors.size(0), dtype=torch.long, device=device),
                    torch.zeros(anchors.size(0), 4, device=device),
                    torch.full((anchors.size(0), 10), -1.0, device=device))

        # Convert anchors to xyxy for IoU
        cx, cy, w, h = anchors.unbind(1)
        anchor_boxes = torch.stack([cx - w/2, cy - h/2, cx + w/2, cy + h/2], dim=1)

        # IoU Matrix [N_anchors, K_gt]
        ious = box_iou(anchor_boxes, gt_boxes)
        
        # 1. Best GT for each anchor
        best_target_iou, best_target_idx = ious.max(dim=1)
        
        # 2. Best anchor for each GT (Force match)
        best_anchor_iou, best_anchor_idx = ious.max(dim=0)
        
        # Force include the best anchor for every GT by artifically boosting IoU
        best_target_iou.index_fill_(0, best_anchor_idx, 2.0) 
        for i, anchor_idx in enumerate(best_anchor_idx):
            best_target_idx[anchor_idx] = i

        # Assign labels
        labels = torch.full((anchors.size(0),), -1, dtype=torch.long, device=anchors.device) # -1 ignore
        labels[best_target_iou < neg_iou] = 0  # BG
        labels[best_target_iou >= pos_iou] = 1 # Face

        # Prepare targets
        matched_gt_boxes = gt_boxes[best_target_idx]
        bbox_targets = RetinaFaceTensor.encode_boxes(anchors, matched_gt_boxes, variances)
        
        # Landmark encoding (only for positives)
        matched_landmarks = gt_landmarks[best_target_idx]
        landmark_targets = torch.full((anchors.size(0), 10), -1.0, device=anchors.device)
        
        pos_mask = labels == 1
        if pos_mask.any():
            # Check validity: [K, 5, 2] -> all coords >= 0
            valid_lm_mask = (matched_landmarks >= 0).all(dim=1).all(dim=1)
            
            # We need to process all positive anchors, but only write valid LMs
            # We can do this by masking the positive set
            p_anchors = anchors[pos_mask]
            p_lm = matched_landmarks[pos_mask]
            p_valid = valid_lm_mask[pos_mask]
            
            if p_valid.any():
                enc_lm = RetinaFaceTensor.encode_landmarks(p_anchors, p_lm, variances)
                
                # Apply to targets
                current_targets = landmark_targets[pos_mask]
                # Only update valid ones
                current_targets[p_valid] = enc_lm[p_valid]
                landmark_targets[pos_mask] = current_targets

        return labels, bbox_targets, landmark_targets
    
    def filter_landmarks(self, bbox_ids):
        # land: (num_boxes, 5, 2)
        land = self.landmark.reshape(-1, 5, 2)[bbox_ids]          # (M, 5, 2)
        vis  = self.landmark_vis.reshape(-1, 5)[bbox_ids]         # (M, 5)
        if type(self.img) is torch.Tensor:
            c, h, w = self.img.shape
        else:
            h, w, c = self.img.shape

        x = land[..., 0]
        y = land[..., 1]

        invisible = (x < 0) | (y < 0) | (x >= w) | (y >= h)        # (M, 5)

        land[invisible] = -1.0                                     # broadcasts into last dim (2)
        vis[invisible] = -1.0

        self.landmark = land.reshape(-1, 2)                        # (M*5, 2)
        self.landmark_vis = vis.reshape(-1)                        # (M*5,)



    def prepare(self, anchors, pos_iou, neg_iou, variances):

        if len(self.landmark)//5 != len(self.bbox):
            self.show()
            raise ValueError(f"{len(self.landmark)//5}(x5 points) landmarks not fit {len(self.bbox)} bboxes!")  
        
        t = self.as_tensor()
        device = anchors.device
        
        t.label, t.bbox, t.landmark = RetinaFaceTensor.match_anchors(
            anchors, t.bbox.to(device), t.landmark.to(device),
            pos_iou, neg_iou, variances
        )
        return t
    
    def show(
        self,mean=None,std=None,
        ax=None,
        figsize=(8, 8),
        title: str = "",
        show_landmarks: bool = True,
        show_boxes: bool = True,
        show_indices: bool = False,
        show_vis: bool = False,
        vis_threshold: float = 0.0,
        box_lw: float = 2.0,
        point_size: float = 20.0,
        assume_normalized: bool | None = None,  # None=auto, True=[0,1], False=pixels
    ):
        """
        Plot image with bbox (xyxy) and 5-point landmarks.

        - If assume_normalized is None, it auto-detects based on value ranges.
        - Missing landmarks are expected as -1 and will be skipped.
        - If landmark_vis exists and show_vis=True, will only plot points with vis>=vis_threshold (and vis>=0).
        """
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle

        # --------- get image (H,W,3) uint8 for display ----------
        img = self.img
        if img is None:
            raise ValueError("No image loaded. Provide `img` or `file`.")

        if hasattr(img, "detach"):  # torch tensor
            mean = torch.tensor(mean, dtype=img.dtype).view(-1, 1, 1)
            std = torch.tensor(std, dtype=img.dtype).view(-1, 1, 1)
            img = img * std + mean.clamp(0, 1)
            img_np = img.detach().cpu().numpy() *255
            # allow CHW or HWC
            if img_np.ndim == 3 and img_np.shape[0] in (1, 3) and img_np.shape[-1] not in (1, 3):
                img_np = np.transpose(img_np, (1, 2, 0))
        else:
            img_np = np.asarray(img)
        
        img_np = img_np.astype(np.uint8)

        if img_np.ndim == 2:
            img_np = np.stack([img_np] * 3, axis=-1)
        if img_np.shape[-1] == 1:
            img_np = np.repeat(img_np, 3, axis=-1)

        # infer H,W if needed
        H, W = img_np.shape[:2]
        self.img_h, self.img_w = H, W

        # --------- prepare bbox/landmarks as numpy ----------
        self_np = self.as_numpy()

        b = self_np.bbox
        lm = self_np.landmark
        lmv = self_np.landmark_vis

        # --------- build axes ----------
        created_ax = False
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
            created_ax = True

        ax.imshow(img_np.astype(np.uint8))
        ax.set_axis_off()
        if title:
            ax.set_title(title)

        # --------- draw boxes ----------
        if show_boxes and b.size:
            for i, (x1, y1, x2, y2) in enumerate(b):
                # skip invalid
                if not np.isfinite([x1, y1, x2, y2]).all():
                    continue
                w_box, h_box = (x2 - x1), (y2 - y1)
                if w_box <= 0 or h_box <= 0:
                    continue
                rect = Rectangle(
                    (x1, y1), w_box, h_box,
                    fill=False, linewidth=box_lw
                )
                ax.add_patch(rect)
                if show_indices:
                    ax.text(x1, y1, f"{i}", fontsize=10, va="top")

        # --------- draw landmarks (assumes 5 points per box if aligned) ----------
        if show_landmarks and lm.size:
            pts = lm.reshape(-1, 2)

            # If landmarks correspond to boxes: common layouts:
            # - stored as (num_boxes*5,2). We'll try to group by boxes if b exists.
            if b.shape[0] > 0 and pts.shape[0] == b.shape[0] * 5:
                pts = pts.reshape(b.shape[0], 5, 2)
                if lmv is not None and lmv.shape[0] == b.shape[0] * 5:
                    lmv_g = lmv.reshape(b.shape[0], 5)
                else:
                    lmv_g = None

                for bi in range(b.shape[0]):
                    for pi in range(5):
                        x, y = pts[bi, pi]
                        if x < 0 or y < 0:
                            continue
                        if lmv_g is not None and show_vis:
                            v = lmv_g[bi, pi]
                            if v < 0 or v < vis_threshold:
                                continue
                            ax.text(x + 2, y + 2, f"{v:.2f}", fontsize=8)
                        ax.scatter([x], [y], s=point_size)
                        if show_indices:
                            ax.text(x + 2, y - 2, f"{bi}:{pi}", fontsize=8)

            else:
                # fallback: plot all valid points as a flat list
                if lmv is not None and show_vis and lmv.shape[0] == pts.shape[0]:
                    mask = (pts[:, 0] >= 0) & (pts[:, 1] >= 0) & (lmv >= vis_threshold) & (lmv >= 0)
                else:
                    mask = (pts[:, 0] >= 0) & (pts[:, 1] >= 0)

                xs, ys = pts[mask, 0], pts[mask, 1]
                ax.scatter(xs, ys, s=point_size)

                if show_indices:
                    idxs = np.where(mask)[0]
                    for j, (x, y) in zip(idxs, pts[mask]):
                        ax.text(x + 2, y - 2, f"{j}", fontsize=8)

        if created_ax:
            plt.tight_layout()
            plt.show()

        return ax

class RetinaFaceDataModule(DataSetModule):
    def __init__(self, config: "DataModuleConfig", anchors=None, pos_iou=None, neg_iou=None, variances=None):
        super().__init__(config)
        self.anchors = anchors
        self.pos_iou = pos_iou
        self.neg_iou = neg_iou
        self.variances = variances
        self.num_classes = 100
        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)
        self.p = 0.5
        noise_std_range: Tuple[float, float] = (0.125, 0.25)
        rmin, rmax = (0.1, 0.2)
        self.train_tf = A.Compose([
            A.Rotate(limit=(-15,15), p=self.p),      
            A.RandomResizedCrop((640,640), scale=(0.6, 1.0)),
            A.HorizontalFlip(p=self.p),
            # A.Resize(640, 640),
            A.OneOf([
                A.ColorJitter(p=1.0,brightness=0.3, contrast=0.3, saturation=0.3, hue=0.2),
                A.ToGray(p=1.0),
            ], p=self.p),
            A.OneOf([
                A.GaussianBlur(p=1.0,blur_limit=(1, 3)),
                A.GaussNoise(p=1.0,std_range=noise_std_range),
            ], p=self.p),
            #  A.CoarseDropout(num_holes_range=(1,4), 
            #                  hole_height_range=(rmin, rmax), 
            #                  hole_width_range=(rmin, rmax), p=self.p),
            A.Normalize(mean=self.mean, std=self.std),
            A.ToTensorV2(),
            ],
            bbox_params=A.BboxParams(
                format="pascal_voc",
                label_fields=["bbox_ids"],
                min_area=1,          # drop boxes with area < 1 px^2
                min_visibility=0.0,  # or bump this to 0.1/0.2 to be stricter
                check_each_transform=True,
            ),
            keypoint_params=A.KeypointParams(
                format="xy",
                remove_invisible=False,      # keep keypoints even if they go outside; we’ll handle them
            ),
        )
        # self.train_tf = v2.Compose([
        #     v2.RandomHorizontalFlip(p=self.p),

        #     v2.RandomApply(
        #         [v2.RandomChoice([
        #             v2.AutoAugment(
        #                 policy=v2.AutoAugmentPolicy.IMAGENET,
        #                 interpolation=InterpolationMode.BILINEAR,
        #             ),
        #             v2.RandAugment(num_ops=2, magnitude=9),
        #             v2.AugMix(severity=3, mixture_width=3, chain_depth=-1, alpha=1.0),
        #         ])],
        #         p=self.p
        #     ),
        #     v2.Resize((640,640)),
        #     ToTensor(mean=self.mean, std=self.std).build(),
        #     v2.Normalize(mean=self.mean, std=self.std),
        # ])
        
        self.val_tf = A.Compose(
            [
                # Resize so the LONG side becomes img_size (keeps aspect ratio)
                A.LongestMaxSize(max_size=640, interpolation=cv2.INTER_LINEAR),

                # Pad to exactly img_size x img_size (stackable)
                A.PadIfNeeded(
                    min_height=640,
                    min_width=640,
                    position="top_left",          # pads on right/bottom only
                    border_mode=cv2.BORDER_CONSTANT,
                    fill=(0, 0, 0),
                ),
                A.Normalize(mean=self.mean, std=self.std),
                A.ToTensorV2(),
            ],
            bbox_params=A.BboxParams(format="pascal_voc"),
            keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
        )
        self.dataset_cls = lambda **kwargs: RetinaFaceDataset(**kwargs, anchors=anchors,
                                                              pos_iou=pos_iou, neg_iou=neg_iou,
                                                              variances=variances,)


    def _build_collate_transform(self):
        def collate_fn(batch):
            x, y = zip(*batch)
            x = torch.stack(x, dim=0)
            return x, list(y)
        self._collate_transform = collate_fn

    def collate_fn(self, batch):
        if self._collate_transform is not None:
            x, y = self._collate_transform(batch)
        return x, y

class RetinaFaceDataset(VisionDataset):
    # Hugging Face mirror (default)
    HF_WIDER_REPO = "https://huggingface.co/datasets/wider_face/resolve/main/data"
    DEFAULT_WIDER_URLS = {
        "train": f"{HF_WIDER_REPO}/WIDER_train.zip",
        "val": f"{HF_WIDER_REPO}/WIDER_val.zip",
        "test": f"{HF_WIDER_REPO}/WIDER_test.zip",  # unused here (no gt)
        "split": f"{HF_WIDER_REPO}/wider_face_split.zip",
    }

    # RetinaFace GT zip (annotations with bbox + 5 landmarks) used by common training repos.
    # We try dl=1 first (direct download). If that fails, dl=0 can still work in some setups.
    RETINAFACE_GT_V1_1_URLS = [
        "https://drive.usercontent.google.com/u/0/uc?id=1tU_IjyOwGQfGNUvZGwWWM4SwxKp2PUQ8&export=download",
    ]
    RETINAFACE_GT_FILENAME = "retinaface_gt_v1.1.zip"

    def __init__(
        self,
        root: str,
        train: bool = True,
        download: bool = False,
        transform=None,
        target_transform=None,
        image_size: int = 640,  # kept for compatibility; unused internally
        annotation_file: Optional[str] = None,
        annotations_dir: str = "widerface",
        train_ann: str = "train.json",
        val_ann: str = "val.json",
        download_url: Optional[str] = None,  # optional override (single URL)
        anchors=None, pos_iou=None, neg_iou=None, variances=None):
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.anchors = anchors.cpu() if anchors is not None else None
        self.pos_iou = pos_iou
        self.neg_iou = neg_iou
        self.variances = variances
        self.root = Path(root)
        self.train = bool(train)
        self.download = bool(download)
        self.image_size = int(image_size)  # unused
        self.download_url = download_url
        self.annotations_dir = annotations_dir

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
        """
        Convert official WIDER bbox GT txt into json:
          { "file": "...", "boxes": [[x1,y1,x2,y2], ...] }
        """
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
                            f.seek(pos)

                file_rel = (images_prefix_rel / img_rel).as_posix()
                items.append({"file": file_rel, "boxes": boxes})

        out_json.parent.mkdir(parents=True, exist_ok=True)
        with open(out_json, "w", encoding="utf-8") as g:
            json.dump(items, g, ensure_ascii=False)

    @staticmethod
    def _convert_retinaface_label_to_json(label_txt: Path, images_prefix_rel: Path, out_json: Path) -> None:
        """
        Convert RetinaFace train/label.txt format into json:

        {
            "file": "...",
            "boxes": [[x1,y1,x2,y2], ...],
            "landmarks": [[15 floats], ...],   # 5*(x,y,v)
            "scores": [float or null, ...]     # per-face final score (blur/quality) if present
        }

        Expected per-face row (common format):
        x y w h  (le_x le_y le_v) (re_x re_y re_v) (n_x n_y n_v) (lm_x lm_y lm_v) (rm_x rm_y rm_v)  score

        Visibility convention commonly used:
        0 = visible, 1 = invisible/occluded; sometimes -1 means missing.
        We store missing coords as -1, but we KEEP v as provided.
        """
        items: List[Dict[str, Any]] = []

        cur_img: Optional[str] = None
        cur_boxes: List[List[float]] = []
        cur_landmarks: List[List[float]] = []
        cur_scores: List[Optional[float]] = []

        def flush() -> None:
            nonlocal cur_img, cur_boxes, cur_landmarks, cur_scores
            if cur_img is None:
                return
            file_rel = (images_prefix_rel / cur_img).as_posix()
            items.append(
                {
                    "file": file_rel,
                    "boxes": cur_boxes,
                    "landmarks": cur_landmarks,
                    "scores": cur_scores,
                }
            )
            cur_img = None
            cur_boxes = []
            cur_landmarks = []
            cur_scores = []

        with open(label_txt, "r", encoding="utf-8", errors="ignore") as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    continue

                if line.startswith("#"):
                    flush()
                    cur_img = line.lstrip("#").strip()
                    continue

                if cur_img is None:
                    continue

                parts = line.split()
                if len(parts) < 4:
                    continue

                try:
                    x, y, w, h = map(float, parts[:4])
                except ValueError:
                    continue

                cur_boxes.append([x, y, x + w, y + h])

                # Default: 5*(x,y,v) all missing
                lm15 = [-1.0] * 15
                face_score: Optional[float] = None

                # Common case: bbox + 15 landmark vals (+ optional trailing score)
                if len(parts) >= 4 + 15:
                    # Parse the 15 landmark numbers
                    try:
                        vals15 = list(map(float, parts[4:4 + 15]))
                        for i in range(5):
                            px, py, vis = vals15[i * 3 : i * 3 + 3]

                            # Keep vis always; coords become -1 only if clearly missing
                            lm15[i * 3 + 2] = vis
                            if px >= 0.0 and py >= 0.0 and vis != -1.0:
                                lm15[i * 3 + 0] = px
                                lm15[i * 3 + 1] = py
                            else:
                                lm15[i * 3 + 0] = -1.0
                                lm15[i * 3 + 1] = -1.0
                    except ValueError:
                        pass

                    # Parse trailing score if present (immediately after the 15 vals)
                    if len(parts) >= 4 + 15 + 1:
                        try:
                            face_score = float(parts[4 + 15])
                        except ValueError:
                            face_score = None

                # Variant: bbox + 10 coords (+ optional trailing score), no vis flags
                elif len(parts) >= 4 + 10:
                    try:
                        vals10 = list(map(float, parts[4:4 + 10]))
                        for i in range(5):
                            px, py = vals10[i * 2 : i * 2 + 2]
                            # Unknown vis -> set to -1.0
                            lm15[i * 3 + 2] = -1.0
                            if px >= 0.0 and py >= 0.0:
                                lm15[i * 3 + 0] = px
                                lm15[i * 3 + 1] = py
                            else:
                                lm15[i * 3 + 0] = -1.0
                                lm15[i * 3 + 1] = -1.0
                    except ValueError:
                        pass

                    # Optional trailing score right after the 10 coords
                    if len(parts) >= 4 + 10 + 1:
                        try:
                            face_score = float(parts[4 + 10])
                        except ValueError:
                            face_score = None

                cur_landmarks.append(lm15)
                cur_scores.append(face_score)

        flush()

        out_json.parent.mkdir(parents=True, exist_ok=True)
        with open(out_json, "w", encoding="utf-8") as g:
            json.dump(items, g, ensure_ascii=False)

    def _download(self) -> None:
        """
        Default behavior (if download_url is None):
          - downloads WIDER_train.zip OR WIDER_val.zip depending on self.train
          - downloads wider_face_split.zip
          - extracts into: <root>/widerface/
          - tries to download retinaface_gt_v1.1.zip and build:
              train.json (bbox + 5 landmarks) from train/label.txt when available
              val.json from WIDER official bbox gt (landmarks absent) unless val/label.txt exists

        If download_url is provided, it is treated as a single archive/file override
        and no WIDER conversion is attempted.
        """
        from torchvision.datasets.utils import download_and_extract_archive, download_url, extract_archive

        # If already present, nothing to do.
        if self.ann_path.exists():
            return

        # If the user provided a custom URL override, do the simplest possible thing.
        if self.download_url:
            self.root.mkdir(parents=True, exist_ok=True)
            download_and_extract_archive(str(self.download_url), download_root=str(self.root))
            return

        urls = self.DEFAULT_WIDER_URLS

        wider_root = self.root / "widerface"
        wider_root.mkdir(parents=True, exist_ok=True)

        # Always need split to build fallback annotations
        split_dir = wider_root / "wider_face_split"
        if not split_dir.exists():
            download_and_extract_archive(urls["split"], download_root=str(wider_root))

        # Download only the image subset we need for this dataset instance
        subset = "train" if self.train else "val"
        subset_dir = wider_root / ("WIDER_train" if self.train else "WIDER_val")
        if not subset_dir.exists():
            download_and_extract_archive(urls[subset], download_root=str(wider_root))

        # Helpers
        def find_one(name: str) -> Path:
            hits = list(wider_root.rglob(name))
            if len(hits) != 1:
                raise FileNotFoundError(
                    f"Expected exactly 1 '{name}' under {wider_root}, found {len(hits)}."
                )
            return hits[0]

        def find_retinaface_label(which: str) -> Optional[Path]:
            # look for train/label.txt or val/label.txt anywhere under wider_root
            candidates = []
            for nm in ("label.txt", "labels.txt"):
                candidates.extend([p for p in wider_root.rglob(nm) if p.parent.name == which])
            if not candidates:
                return None
            # prefer paths that look like they came from retinaface_gt
            for p in candidates:
                if "retinaface_gt" in str(p).lower() or "retinaface" in str(p).lower():
                    return p
            return candidates[0]

        def ensure_retinaface_gt_present() -> None:
            # if we can already locate train label, assume extracted.
            if find_retinaface_label("train") is not None:
                return

            for url in self.RETINAFACE_GT_V1_1_URLS:
                try:
                    # IMPORTANT: override filename so query params don't break basename parsing.
                    download_url(url, root=str(wider_root), filename=self.RETINAFACE_GT_FILENAME)
                    archive_path = wider_root / self.RETINAFACE_GT_FILENAME
                    if archive_path.exists():
                        extract_archive(str(archive_path), to_path=str(wider_root))
                    # If extracted successfully, stop.
                    if find_retinaface_label("train") is not None:
                        return
                except Exception:
                    continue

        # Try to fetch RetinaFace GT (for train landmarks)
        ensure_retinaface_gt_present()
        train_label = find_retinaface_label("train")
        val_label = find_retinaface_label("val")

        if self.train:
            out_json = self.root / self.annotations_dir / "train.json"
            images_prefix_rel = Path("widerface") / "WIDER_train" / "images"
            if not out_json.exists():
                if train_label is not None:
                    self._convert_retinaface_label_to_json(train_label, images_prefix_rel, out_json)
                else:
                    # fallback: WIDER bbox only
                    gt_txt = find_one("wider_face_train_bbx_gt.txt")
                    self._convert_wider_txt_to_json(gt_txt, images_prefix_rel, out_json)
        else:
            out_json = self.root / self.annotations_dir / "val.json"
            images_prefix_rel = Path("widerface") / "WIDER_val" / "images"
            if not out_json.exists():
                if val_label is not None:
                    self._convert_retinaface_label_to_json(val_label, images_prefix_rel, out_json)
                else:
                    # fallback: WIDER bbox only (landmarks absent)
                    gt_txt = find_one("wider_face_val_bbx_gt.txt")
                    self._convert_wider_txt_to_json(gt_txt, images_prefix_rel, out_json)

        # Auto-switch ann_path if it matches default names.
        if self.ann_path.name in ("train.json", "val.json") and out_json.exists():
            self.ann_path = out_json
    
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        entry = self.items[index]
        img_path = self._resolve_path(entry["file"])
        target = RetinaFaceTensor(file=str(img_path),
                              bbox=np.asarray(entry.get("boxes", []), dtype=np.float32).reshape(-1, 4),
                              landmark=entry.get("landmarks", [])
                              )
        img:np.ndarray = target.img
        if self.transform is not None:
            if isinstance(self.transform, (BaseCompose, BasicTransform)):
                bbox_ids = np.arange(len(target.bbox))
                res = self.transform(image=img,
                                    bboxes=target.bbox,
                                    keypoints=target.landmark,
                                    bbox_ids=bbox_ids
                                )
                target.img = img = res["image"]
                target.bbox = res["bboxes"]
                target.landmark = res["keypoints"]
                bbox_ids = np.asarray(res["bbox_ids"]).astype(int)
                target.filter_landmarks(bbox_ids=bbox_ids)
                if self.anchors is not None:
                    target = target.prepare(self.anchors, self.pos_iou, self.neg_iou, self.variances)
            else:
                img = self.transform(Image.fromarray(img))
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

# if __name__ == "__main__":
#     for n in ['retinaface']:#,'mnist','cifar100','timnet']:
#         ds = DataModuleConfig(dataset_name=n,data_dir='./data',batch_size=32,mixup=False,cutmix=False).build()
#         ds.setup()
#         for i in range(4):
#             ds.show_examples(4)
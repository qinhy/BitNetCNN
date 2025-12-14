import os
import math
from typing import Optional, Tuple, Union, List, Dict

from matplotlib import pyplot as plt
from pydantic import BaseModel, PrivateAttr
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import default_collate

from torchvision import datasets
from torchvision.transforms import v2
from torchvision.transforms import InterpolationMode
from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import extract_archive, download_url, check_integrity

torch.set_float32_matmul_precision("high")


# ----------------------------
# Augmentations
# ----------------------------
class Cutout(nn.Module):
    """Simple Cutout transform for tensors [C, H, W].

    size:
        - int       -> square hole (size x size)
        - (h, w)    -> rectangular hole (height x width)
    """

    def __init__(self, size: Union[int, Tuple[int, int]] = 16):
        super().__init__()
        if isinstance(size, int):
            self.size = (size, size)
        else:
            if len(size) != 2:
                raise ValueError(f"Cutout size must be int or (h, w) tuple, got: {size}")
            self.size = (int(size[0]), int(size[1]))

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        # img is expected to be a Tensor after ToImage/ToDtype
        if not torch.is_tensor(img):
            return img

        _, h, w = img.shape
        if h <= 0 or w <= 0:
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

        img = img.clone()  # safer than in-place
        img[:, y1:y2, x1:x2] = 0.0
        return img


def get_train_tf(
    mean,
    std,
    flip: bool = True,
    crop: Tuple[int, int] = (32, 4),
    cout: Union[int, Tuple[int, int]] = (8, 8),
    p: float = 0.325,
    randaugment: bool = True,
):
    ra = v2.RandAugment(num_ops=2, magnitude=7)
    return v2.Compose(
        [
            v2.RandomCrop(crop[0], padding=crop[1]),
            v2.RandomHorizontalFlip() if flip else v2.Identity(),
            v2.RandomApply([ra], p=p) if randaugment else v2.Identity(),
            v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
            v2.RandomApply([Cutout(size=cout)], p=p),
            v2.Normalize(mean, std),
        ]
    )


def get_val_tf(mean, std):
    return v2.Compose(
        [
            v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
            v2.Normalize(mean, std),
        ]
    )


def get_mixup_or_cutmix(
    num_classes: int,
    mixup: bool,
    cutmix: bool,
    mix_alpha: float,
    p: float = 0.325,
):
    transforms_list = []
    if mixup:
        transforms_list.append(v2.MixUp(num_classes=num_classes, alpha=mix_alpha))
    if cutmix:
        transforms_list.append(v2.CutMix(num_classes=num_classes, alpha=mix_alpha))

    if len(transforms_list) == 0:
        return None

    mixup_or_cutmix = transforms_list[0] if len(transforms_list) == 1 else v2.RandomChoice(transforms_list)
    return v2.RandomApply([mixup_or_cutmix], p=p)



# ----------------------------
# Config
# ----------------------------
class DataModuleConfig(BaseModel):
    data_dir: str
    dataset_name: str = ""
    num_classes: int = -1

    batch_size: int
    num_workers: int = 1

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
        self._collate_transform = get_mixup_or_cutmix(
            num_classes=self.num_classes,
            mixup=self.mixup,
            cutmix=self.cutmix,
            mix_alpha=self.mix_alpha,
            p=0.325,
        )

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
        y = y[:n].cpu()

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
            plt.title(label_to_text(y[i]), fontsize=8)
        plt.tight_layout()
        plt.show()
        return x, y


# ----------------------------
# CIFAR-100
# ----------------------------
class CIFAR100DataModule(DataSetModule):
    def __init__(self, config: "DataModuleConfig"):
        super().__init__(config)
        self.num_classes = 100
        self.mean = (0.5071, 0.4867, 0.4408)
        self.std = (0.2675, 0.2565, 0.2761)
        self.train_tf = get_train_tf(self.mean, self.std, flip=True, crop=(32, 4), cout=(8, 8), p=0.325)
        self.val_tf = get_val_tf(self.mean, self.std)
        self.dataset_cls = datasets.CIFAR100


# ----------------------------
# TinyImageNet-200
# ----------------------------
class TinyImageNetDataModule(DataSetModule):
    def __init__(self, config: "DataModuleConfig"):
        super().__init__(config)
        self.num_classes = 200
        self.mean = (0.4802, 0.4481, 0.3975)
        self.std = (0.2302, 0.2265, 0.2262)
        self.train_tf = get_train_tf(self.mean, self.std, flip=True, crop=(64, 4), cout=(16, 16), p=0.325)
        self.val_tf = get_val_tf(self.mean, self.std)
        self.dataset_cls = TinyImageNetDataset


# ----------------------------
# MNIST
# ----------------------------
class MNISTDataModule(DataSetModule):
    def __init__(self, config: "DataModuleConfig"):
        super().__init__(config)
        self.num_classes = 10
        self.mean = (0.1307,)
        self.std = (0.3081,)
        # RandAugment can be a bit odd on grayscale; enable if you know it's fine in your env.
        self.train_tf = get_train_tf(
            self.mean, self.std, flip=False, crop=(28, 2), cout=(8, 8), p=0.325, randaugment=False
        )
        self.val_tf = get_val_tf(self.mean, self.std)
        self.dataset_cls = datasets.MNIST
        # CutMix on MNIST is typically not useful; keep it off by default.
        self.cutmix = False

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
        super(TinyImageNetDataset, self).__init__(root, transform=transform, target_transform=target_transform)

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


if __name__ == "__main__":
    for n in ['mnist','cifar100','timnet']:
        ds = DataModuleConfig(dataset_name=n,data_dir='./data',batch_size=32,mixup=True,cutmix=True).build()
        ds.setup()
        for i in range(4):
            ds.show_examples(32)
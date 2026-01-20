import torch
from torchvision import datasets
from torchvision.transforms import InterpolationMode
from torchvision.transforms import v2

from .base import DataSetModule


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

    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        num_workers: int = 1,
        mixup: bool = False,
        cutmix: bool = False,
        mix_alpha: float = 0.2,
        image_size: int = 224,
        val_resize: int = 256,
    ):
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
        self.std = (0.229, 0.224, 0.225)

        # Will be set in setup()
        self.train_ds = None
        self.val_ds = None

    def setup(self, stage=None):
        train_tf = v2.Compose(
            [
                v2.RandomResizedCrop(self.image_size, interpolation=InterpolationMode.BICUBIC),
                v2.RandomHorizontalFlip(),
                v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
                v2.Normalize(self.mean, self.std),
            ]
        )
        val_tf = v2.Compose(
            [
                v2.Resize(self.val_resize, interpolation=InterpolationMode.BICUBIC),
                v2.CenterCrop(self.image_size),
                v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
                v2.Normalize(self.mean, self.std),
            ]
        )

        # ImageFolder expects subfolders per class
        self.train_ds = datasets.ImageFolder(root=f"{self.data_dir}/train", transform=train_tf)
        self.val_ds = datasets.ImageFolder(root=f"{self.data_dir}/val", transform=val_tf)

        # (Optional) you can inspect class count if needed:
        # self.num_classes = len(self.train_ds.classes)

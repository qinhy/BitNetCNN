from typing import Any, Tuple

from PIL import Image
import numpy as np
import albumentations as A
from albumentations.core.composition import BaseCompose
from albumentations.core.transforms_interface import BasicTransform
from torchvision import datasets
from torchvision.transforms import InterpolationMode
from torchvision.transforms import v2

from bitlayers.aug import AutoRandomResizedCrop
from bitlayers.aug import SimpleImageTrainAugment
from bitlayers.aug import SimpleImageValAugment
from bitlayers.aug import ToTensor
from .base import DataSetModule


class CIFAR100Dataset(datasets.CIFAR100):
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img, target = self.data[index], self.targets[index]
        img: np.ndarray = img
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
        self.train_tf = A.Compose(
            [
                A.Rotate(limit=(-10, 10), p=self.p),
                A.HorizontalFlip(p=self.p),
                AutoRandomResizedCrop(scale=(0.75, 1.0), ratio=(0.75, 1.33), p=self.p),
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
            ]
        )
        self.train_tf = v2.Compose(
            [
                v2.RandomCrop(32, padding=4),
                v2.RandomHorizontalFlip(p=self.p),
                v2.RandomApply(
                    [
                        v2.RandomChoice(
                            [
                                v2.AutoAugment(
                                    policy=v2.AutoAugmentPolicy.CIFAR10,
                                    interpolation=InterpolationMode.BILINEAR,
                                ),
                                v2.RandAugment(num_ops=2, magnitude=9),
                                v2.AugMix(severity=3, mixture_width=3, chain_depth=-1, alpha=1.0),
                            ]
                        )
                    ],
                    p=self.p,
                ),
                ToTensor(mean=self.mean, std=self.std).build(),
                v2.Normalize(mean=self.mean, std=self.std),
            ]
        )
        self.val_tf = SimpleImageValAugment(mean=self.mean, std=self.std).build()
        self.dataset_cls = CIFAR100Dataset

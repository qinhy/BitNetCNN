from typing import Any, Tuple

import cv2
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.core.composition import BaseCompose
from albumentations.core.transforms_interface import BasicTransform
from torchvision import datasets

from bitlayers.aug import SimpleImageTrainAugment
from bitlayers.aug import SimpleImageValAugment
from .base import DataSetModule


class MNISTDataset(datasets.MNIST):
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img, target = self.data[index], self.targets[index]
        img = img.numpy()
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        if self.transform is not None:
            if isinstance(self.transform, (BaseCompose, BasicTransform)):
                img = self.transform(image=img)["image"]
            else:
                img = self.transform(Image.fromarray(img))
        if self.target_transform is not None:
            target = self.target_transform(target)
        img = img.mean(0, keepdim=True)
        return img, target


class MNISTDataModule(DataSetModule):
    def __init__(self, config: "DataModuleConfig"):
        super().__init__(config)
        self.num_classes = 10
        self.mean = (0.1307,)
        self.std = (0.3081,)
        # RandAugment can be a bit odd on grayscale; enable if you know it's fine in your env.
        self.train_tf = SimpleImageTrainAugment(mean=self.mean, std=self.std, flip=False).build()
        self.val_tf = SimpleImageValAugment(mean=self.mean, std=self.std).build()
        self.dataset_cls = MNISTDataset
        # CutMix on MNIST is typically not useful; keep it off by default.
        self.cutmix = False

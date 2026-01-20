import os
from typing import Any, Tuple

from PIL import Image
import numpy as np
from albumentations.core.composition import BaseCompose
from albumentations.core.transforms_interface import BasicTransform
from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import check_integrity
from torchvision.datasets.utils import download_url
from torchvision.datasets.utils import extract_archive

from bitlayers.aug import SimpleImageTrainAugment
from bitlayers.aug import SimpleImageValAugment
from .base import DataSetModule


class TinyImageNetDataModule(DataSetModule):
    def __init__(self, config: "DataModuleConfig"):
        super().__init__(config)
        self.num_classes = 200
        self.mean = (0.4802, 0.4481, 0.3975)
        self.std = (0.2302, 0.2265, 0.2262)
        self.train_tf = SimpleImageTrainAugment(mean=self.mean, std=self.std).build()
        self.val_tf = SimpleImageValAugment(mean=self.mean, std=self.std).build()
        self.dataset_cls = TinyImageNetDataset


# -----------------------------------------------------------------------------
# Tiny ImageNet Dataset Helper
# -----------------------------------------------------------------------------
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

    base_folder = "tiny-imagenet-200/"
    url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    filename = "tiny-imagenet-200.zip"
    md5 = "90528d7ca1a48142e341f4ef8d21d0de"

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        super().__init__(root, transform=transform, target_transform=target_transform)

        self.dataset_path = os.path.join(root, self.base_folder)
        self.loader = default_loader
        self.split = "train" if train else "val"

        if self._check_integrity():
            print("Files already downloaded and verified.")
        elif download:
            self._download()
        else:
            raise RuntimeError("Dataset not found. You can use download=True to download it.")
        if not os.path.isdir(self.dataset_path):
            print("Extracting...")
            extract_archive(os.path.join(root, self.filename))

        _, class_to_idx = TinyImageNetDataset.find_classes(os.path.join(self.dataset_path, "wnids.txt"))

        self.data = TinyImageNetDataset.make_dataset(self.root, self.base_folder, self.split, class_to_idx)
        self.targets = [s[1] for s in self.data]

    def _download(self):
        print("Downloading...")
        download_url(self.url, root=self.root, filename=self.filename)
        print("Extracting...")
        extract_archive(os.path.join(self.root, self.filename))

    def _check_integrity(self):
        return check_integrity(os.path.join(self.root, self.filename), self.md5)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img_path, target = self.data[index]
        img = self.loader(img_path)
        img = np.asarray(img)
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

        if dirname == "train":
            for fname in sorted(os.listdir(dir_path)):
                cls_fpath = os.path.join(dir_path, fname)
                if os.path.isdir(cls_fpath):
                    cls_imgs_path = os.path.join(cls_fpath, "images")
                    for imgname in sorted(os.listdir(cls_imgs_path)):
                        path = os.path.join(cls_imgs_path, imgname)
                        item = (path, class_to_idx[fname])
                        images.append(item)
        else:
            imgs_path = os.path.join(dir_path, "images")
            imgs_annotations = os.path.join(dir_path, "val_annotations.txt")

            with open(imgs_annotations) as r:
                data_info = map(lambda s: s.split("\t"), r.readlines())

            cls_map = {line_data[0]: line_data[1] for line_data in data_info}

            for imgname in sorted(os.listdir(imgs_path)):
                path = os.path.join(imgs_path, imgname)
                item = (path, class_to_idx[cls_map[imgname]])
                images.append(item)

        return images

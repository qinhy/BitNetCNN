from typing import Dict, Tuple, Union

from pydantic import BaseModel, Field, PrivateAttr


class DataModuleConfig(BaseModel):
    data_dir: str
    dataset_name: str = ""
    num_classes: int = -1
    img_size: int = -1
    color_channels: int = -1

    batch_size: Union[int,Tuple[int,int]]
    num_workers: int = 0

    mixup: bool = False
    cutmix: bool = False
    mix_alpha: float = 1.0

    _datasets: Dict[str, type] = PrivateAttr(default_factory=dict)

    def model_post_init(self, __context):
        from .cifar import CIFAR100DataModule
        from .tinyimagenet import TinyImageNetDataModule
        from .imagenet import ImageNetDataModule
        from .mnist import MNISTDataModule
        from .retinaface import RetinaFaceDataModule

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
            self.img_size = 32
            self.color_channels = 3
        elif ds in ["timnet", "tiny", "tinyimagenet", "tiny-imagenet"]:
            self.num_classes = 200
            self.dataset_name = "timnet"
            self.img_size = 64
            self.color_channels = 3
        elif ds in ["imnet", "imagenet", "in1k", "imagenet1k"]:
            self.num_classes = 1000
            self.dataset_name = "imnet"            
            self.img_size = -1
            self.color_channels = 3
        elif ds in ["mnist"]:
            self.num_classes = 10
            self.dataset_name = "mnist"            
            self.img_size = 28
            self.color_channels = 1
        elif ds in ["retinaface", "wider-face", "wider"]:
            self.num_classes = -1
            self.dataset_name = "retinaface"            
            self.img_size = -1
            self.color_channels = 3
        else:
            raise ValueError(f"Unsupported dataset: {ds}")

        return super().model_post_init(__context)

    def build(self) -> "DataSetModule":
        print(f"[Dataset]: use {self.dataset_name}, {self.num_classes} classes.")
        return self._datasets[self.dataset_name](self)

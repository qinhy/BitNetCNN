import argparse
from typing import Callable, Dict, Iterable, Literal, Optional, Tuple

from pydantic import Field
from pydanticV2_argparse import ArgumentParser
import torch
import torch.nn as nn
from torchvision.models.resnet import BasicBlock, Bottleneck, ResNet
from torchvision.models import (
    ResNet18_Weights,
    ResNet50_Weights,
    resnet18,
    resnet50,
)
from huggingface_hub import hf_hub_download

from bitlayers import convs
from bitlayers.bit import Bit
from bitlayers.acts import ActModels
from bitlayers.norms import NormModels
from common_utils import *


# ---------------------------------------------------------------------------
# Bit-blocks and core network
# ---------------------------------------------------------------------------

BasicBlockBit = convs.Conv2dModels.ResNetBasicBlock
BottleneckBit = convs.Conv2dModels.ResNetBottleneck


class BitResNet(nn.Module):
    def __init__(
        self,
        block: Callable[..., nn.Module],
        layers: Iterable[int],
        num_classes: int,
        expansion: int,
        scale_op: str = "median",
        in_ch: int = 3,
        small_stem: bool = True,
    ) -> None:
        super().__init__()
        self.block_cls = block
        self.layers = tuple(layers)
        self.num_classes = num_classes
        self.scale_op = scale_op
        self.in_ch = in_ch
        self.inplanes = 64
        self.small_stem = small_stem
        self.expansion = expansion

        if small_stem:
            # CIFAR / Tiny stem: 3x3 stride 1, no maxpool
            self.stem = nn.Sequential(
                Bit.Conv2d(
                    in_ch,
                    self.inplanes,
                    kernel_size=3,stride=1,padding=1,
                    bias=True,scale_op=scale_op,
                ),
                nn.BatchNorm2d(self.inplanes),
                nn.SiLU(inplace=True),
            )
        else:
            # ImageNet stem: 7x7 stride 2 + maxpool
            self.stem = nn.Sequential(
                Bit.Conv2d(
                    in_ch,
                    self.inplanes,
                    kernel_size=7,stride=2,padding=3,
                    bias=True,scale_op=scale_op,
                ),
                nn.BatchNorm2d(self.inplanes),
                nn.SiLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            )

        self.layer1 = self._make_layer(block, 64,  self.layers[0], stride=1, scale_op=scale_op)
        self.layer2 = self._make_layer(block, 128, self.layers[1], stride=2, scale_op=scale_op)
        self.layer3 = self._make_layer(block, 256, self.layers[2], stride=2, scale_op=scale_op)
        self.layer4 = self._make_layer(block, 512, self.layers[3], stride=2, scale_op=scale_op)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            Bit.Linear(512 * self.expansion, num_classes, bias=True, scale_op=scale_op),
        )

    @staticmethod
    def _act() -> ActModels.type:
        return ActModels.SiLU(inplace=True)

    @staticmethod
    def _norm() -> NormModels.type:
        return NormModels.BatchNorm2d(num_features=-1)

    def _make_layer(
        self,
        block: Callable[..., nn.Module],
        planes: int,
        blocks: int,
        stride: int,
        scale_op: str,
    ) -> nn.Sequential:
        out_ch = planes * self.expansion
        layers = []
        for block_idx in range(blocks):
            block_stride = stride if block_idx == 0 else 1
            layers.append(
                self._build_block(
                    block=block,
                    inplanes=self.inplanes,
                    out_channels=out_ch,
                    stride=block_stride,
                    scale_op=scale_op,
                )
            )
            self.inplanes = out_ch
        return nn.Sequential(*layers)

    def _build_block(
        self,
        block: Callable[..., nn.Module],
        inplanes: int,
        out_channels: int,
        stride: int,
        scale_op: str,
    ) -> nn.Module:
        shortcut_layer = None
        if stride != 1 or inplanes != out_channels:
            shortcut_layer = convs.Conv2dModels.Conv2dNorm(
                in_channels=-1,
                norm=self._norm(),
                scale_op=scale_op,
            )

        if block is BasicBlockBit:
            block_cfg = BasicBlockBit(
                in_channels=inplanes,
                out_channels=out_channels,
                stride=stride,
                padding=1,
                act_layer=self._act(),
                conv1_layer=convs.Conv2dModels.Conv2dNormAct(
                    in_channels=-1,
                    norm=self._norm(),
                    act=self._act(),
                    scale_op=scale_op,
                ),
                conv2_layer=convs.Conv2dModels.Conv2dNorm(
                    in_channels=-1,
                    norm=self._norm(),
                    scale_op=scale_op,
                ),
                shortcut_layer=shortcut_layer,
                scale_op=scale_op,
                bit=True,
            )
        elif block is BottleneckBit:
            block_cfg = BottleneckBit(
                in_channels=inplanes,
                out_channels=out_channels,
                stride=stride,
                padding=1,
                act_layer=self._act(),
                conv_reduce_layer=convs.Conv2dModels.Conv2dNormAct(
                    in_channels=-1,
                    norm=self._norm(),
                    act=self._act(),
                    scale_op=scale_op,
                ),
                conv_transform_layer=convs.Conv2dModels.Conv2dNormAct(
                    in_channels=-1,
                    norm=self._norm(),
                    act=self._act(),
                    scale_op=scale_op,
                ),
                conv_expand_layer=convs.Conv2dModels.Conv2dNorm(
                    in_channels=-1,
                    norm=self._norm(),
                    scale_op=scale_op,
                ),
                shortcut_layer=shortcut_layer,
                scale_op=scale_op,
            )
        else:
            raise ValueError(f"Unsupported block type: {block}")

        return block_cfg.build()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        return self.head(x)

    def clone(self) -> "BitResNet":
        return self.__class__(
            num_classes=self.num_classes,
            scale_op=self.scale_op,
            in_ch=self.in_ch,
            small_stem=self.small_stem,
        )


class BitResNet18(BitResNet):
    def __init__(
        self,
        num_classes: int,
        scale_op: str = "median",
        in_ch: int = 3,
        small_stem: bool = True,
    ) -> None:
        super().__init__(BasicBlockBit, [2, 2, 2, 2], num_classes, 1, scale_op, in_ch, small_stem)


class BitResNet50(BitResNet):
    def __init__(
        self,
        num_classes: int,
        scale_op: str = "median",
        in_ch: int = 3,
        small_stem: bool = True,
    ) -> None:
        super().__init__(BottleneckBit, [3, 4, 6, 3], num_classes, 4, scale_op, in_ch, small_stem)


# ---------------------------------------------------------------------------
# Teacher networks
# ---------------------------------------------------------------------------


class ResNet18CIFAR(ResNet):
    def __init__(self, num_classes: int = 100) -> None:
        super().__init__(block=BasicBlock, layers=[2, 2, 2, 2], num_classes=num_classes)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=True)
        self.maxpool = nn.Identity()


class ResNet50CIFAR(ResNet):
    def __init__(self, num_classes: int = 100) -> None:
        super().__init__(block=Bottleneck, layers=[3, 4, 6, 3], num_classes=num_classes)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=True)
        self.maxpool = nn.Identity()


def make_resnet18_cifar_teacher_from_hf(device: str = "cuda") -> ResNet:
    model = ResNet18CIFAR(num_classes=100)
    ckpt_path = hf_hub_download(
        repo_id="edadaltocg/resnet18_cifar100",
        filename="pytorch_model.bin",
    )
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[teacher][cifar100][rn18] Missing keys: {missing}")
    if unexpected:
        print(f"[teacher][cifar100][rn18] Unexpected keys: {unexpected}")
    return model.eval().to(device)


def make_resnet50_cifar_teacher_from_hf(device: str = "cuda") -> ResNet:
    model = ResNet50CIFAR(num_classes=100)
    ckpt_path = hf_hub_download(
        repo_id="edadaltocg/resnet50_cifar100",
        filename="pytorch_model.bin",
    )
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[teacher][cifar100][rn50] Missing keys: {missing}")
    if unexpected:
        print(f"[teacher][cifar100][rn50] Unexpected keys: {unexpected}")
    return model.eval().to(device)


def make_resnet18_imagenet_teacher(device: str = "cuda") -> ResNet:
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.eval()
    return model.to(device)


def make_resnet50_imagenet_teacher(device: str = "cuda") -> ResNet:
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    model.eval()
    return model.to(device)


def _strip_module_prefix(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}


def make_resnet_tiny_teacher_from_hf(
    model_size: str,
    epochs: int = 200,
    device: str = "cuda",
) -> ResNet:
    assert epochs in (50, 100, 200), "epochs must be 50/100/200"

    if model_size == "18":
        model = resnet18(num_classes=200)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
        fname = f"rn18_{epochs}ep/checkpoint_best.pth"
    elif model_size == "50":
        model = resnet50(num_classes=200)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
        fname = f"rn50_{epochs}ep/checkpoint_best.pth"
    else:
        raise ValueError(f"Unsupported model_size '{model_size}' for Tiny-ImageNet teacher.")

    ckpt_path = hf_hub_download(repo_id="zeyuanyin/tiny-imagenet", filename=fname)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict):
        if "model" in ckpt and isinstance(ckpt["model"], dict):
            sd = ckpt["model"]
        elif "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            sd = ckpt["state_dict"]
        else:
            sd = ckpt
    else:
        sd = ckpt

    sd = _strip_module_prefix(sd)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"[teacher][tiny][rn{model_size}_{epochs}ep] Missing keys: {missing}")
    if unexpected:
        print(f"[teacher][tiny][rn{model_size}_{epochs}ep] Unexpected keys: {unexpected}")
    return model.eval().to(device)


_TEACHER_MAP: Dict[str, Dict[str, Callable[..., ResNet]]] = {
    "18": {
        "c100": make_resnet18_cifar_teacher_from_hf,
        "imnet": make_resnet18_imagenet_teacher,
        "timnet": lambda device, epochs: make_resnet_tiny_teacher_from_hf(
            "18",
            epochs=epochs,
            device=device,
        ),
    },
    "50": {
        "c100": make_resnet50_cifar_teacher_from_hf,
        "imnet": make_resnet50_imagenet_teacher,
        "timnet": lambda device, epochs: make_resnet_tiny_teacher_from_hf(
            "50",
            epochs=epochs,
            device=device,
        ),
    },
}


# ---------------------------------------------------------------------------
# LightningModule wrappers
# ---------------------------------------------------------------------------

def _num_classes_and_stem(dataset_key: str) -> Tuple[int, bool]:
    if dataset_key == "c100":
        return 100, True
    if dataset_key == "imnet":
        return 1000, False
    if dataset_key == "timnet":
        return 200, True
    raise ValueError(f"Unsupported dataset key: {dataset_key}")


def _build_student(
    model_size: str,
    num_classes: int,
    scale_op: str,
    small_stem: bool,
    in_ch: int = 3,
) -> BitResNet:
    if model_size == "18":
        return BitResNet18(num_classes=num_classes, scale_op=scale_op, in_ch=in_ch, small_stem=small_stem)
    if model_size == "50":
        return BitResNet50(num_classes=num_classes, scale_op=scale_op, in_ch=in_ch, small_stem=small_stem)
    raise ValueError(f"Unsupported model_size: {model_size}")


def _build_teacher(
    model_size: str,
    dataset_name: str,
    device: str,
    timnet_teacher_epochs: int,
) -> ResNet:
    teachers = _TEACHER_MAP[model_size]
    if dataset_name in ("c100", "imnet"):
        return teachers[dataset_name](device=device)
    if dataset_name == "timnet":
        return teachers[dataset_name](device=device, epochs=timnet_teacher_epochs)
    raise ValueError(f"Unsupported dataset key for teachers: {dataset_name}")

class Config(CommonTrainConfig):
    dataset_name: str = "c100"
    model_size: Literal["18", "50"] = Field(
        default="18",
        description="Model size"
    )
    timnet_teacher_epochs: Literal[50, 100, 200] = Field(
        default=200,
        description="Which Tiny-ImageNet ResNet teacher to load from zeyuanyin/tiny-imagenet",
    )

# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------
def main() -> None:
    parser = ArgumentParser(model=Config)
    args = parser.parse_typed_args()

    dmargs = DataModuleConfig.model_validate(args.model_dump()).model_dump()
    if args.dataset_name == "c100":
        dm = CIFAR100DataModule(**dmargs)
    elif args.dataset_name == "imnet":
        dm = ImageNetDataModule(**dmargs)
    elif args.dataset_name == "timnet":
        dm = TinyImageNetDataModule(**dmargs)
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset_name}")

    config = LitBitConfig.model_validate(args.model_dump())

    config.model_size = str(config.model_size)
    if config.model_size not in ("18", "50"):
        raise ValueError(f"Unsupported model_size: {config.model_size}")

    config.num_classes, small_stem = _num_classes_and_stem(config.dataset_name)
    config.export_dir = args.export_dir = f"./ckpt_{config.dataset_name}_rn{config.model_size}"  

    config.student = _build_student(
        model_size=config.model_size,
        num_classes=config.num_classes,
        scale_op=config.scale_op,
        small_stem=small_stem,
    )
    config.teacher = _build_teacher(
        model_size=config.model_size,
        dataset_name=config.dataset_name,
        device="cpu",
        timnet_teacher_epochs=200,
    )
    config.model_name="resnet"
    config.hint_points=["layer1", "layer2", "layer3", "layer4"]
    
    lit = LitBit(config)
    trainer, dm = setup_trainer(args, lit, dm)
    trainer.fit(lit, datamodule=dm)
    trainer.validate(lit, datamodule=dm)


if __name__ == "__main__":
    main()

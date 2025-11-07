import argparse
from typing import Callable, Dict, Iterable, Optional

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

from common_utils import (  # noqa: F401 (re-exported for backwards compat)
    Bit,
    LitBit,
    TinyImageNetDataModule,
    CIFAR100DataModule,
    ImageNetDataModule,
    add_common_args,
    setup_trainer,
)


# ---------------------------------------------------------------------------
# Bit-blocks and core network
# ---------------------------------------------------------------------------

class BasicBlockBit(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        scale_op: str = "median",
    ) -> None:
        super().__init__()
        self.conv1 = Bit.Conv2d(
            inplanes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=True,
            scale_op=scale_op,
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.act = nn.SiLU(inplace=True)
        self.conv2 = Bit.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
            scale_op=scale_op,
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.act(out + identity)


class BottleneckBit(nn.Module):
    expansion = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        scale_op: str = "median",
    ) -> None:
        super().__init__()
        width = planes
        self.conv1 = Bit.Conv2d(
            inplanes,
            width,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
            scale_op=scale_op,
        )
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = Bit.Conv2d(
            width,
            width,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=True,
            scale_op=scale_op,
        )
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = Bit.Conv2d(
            width,
            planes * self.expansion,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
            scale_op=scale_op,
        )
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.act = nn.SiLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.act(self.bn1(self.conv1(x)))
        out = self.act(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.act(out + identity)


class BitResNet(nn.Module):
    def __init__(
        self,
        block: Callable[..., nn.Module],
        layers: Iterable[int],
        num_classes: int,
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

        if small_stem:
            # CIFAR / Tiny stem: 3x3 stride 1, no maxpool
            self.stem = nn.Sequential(
                Bit.Conv2d(
                    in_ch,
                    self.inplanes,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=True,
                    scale_op=scale_op,
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
                    kernel_size=7,
                    stride=2,
                    padding=3,
                    bias=True,
                    scale_op=scale_op,
                ),
                nn.BatchNorm2d(self.inplanes),
                nn.SiLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            )

        self.layer1 = self._make_layer(block, 64, self.layers[0], stride=1, scale_op=scale_op)
        self.layer2 = self._make_layer(block, 128, self.layers[1], stride=2, scale_op=scale_op)
        self.layer3 = self._make_layer(block, 256, self.layers[2], stride=2, scale_op=scale_op)
        self.layer4 = self._make_layer(block, 512, self.layers[3], stride=2, scale_op=scale_op)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            Bit.Linear(512 * block.expansion, num_classes, bias=True, scale_op=scale_op),
        )

    def _make_layer(
        self,
        block: Callable[..., nn.Module],
        planes: int,
        blocks: int,
        stride: int,
        scale_op: str,
    ) -> nn.Sequential:
        downsample = None
        out_ch = planes * block.expansion
        if stride != 1 or self.inplanes != out_ch:
            downsample = nn.Sequential(
                Bit.Conv2d(
                    self.inplanes,
                    out_ch,
                    kernel_size=1,
                    stride=stride,
                    bias=True,
                    scale_op=scale_op,
                ),
                nn.BatchNorm2d(out_ch),
            )
        layers = [
            block(
                self.inplanes,
                planes,
                stride=stride,
                downsample=downsample,
                scale_op=scale_op,
            )
        ]
        self.inplanes = out_ch
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    stride=1,
                    downsample=None,
                    scale_op=scale_op,
                )
            )
        return nn.Sequential(*layers)

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
        super().__init__(BasicBlockBit, [2, 2, 2, 2], num_classes, scale_op, in_ch, small_stem)


class BitResNet50(BitResNet):
    def __init__(
        self,
        num_classes: int,
        scale_op: str = "median",
        in_ch: int = 3,
        small_stem: bool = True,
    ) -> None:
        super().__init__(BottleneckBit, [3, 4, 6, 3], num_classes, scale_op, in_ch, small_stem)


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


def _canonical_dataset(dataset_name: str) -> str:
    mapping = {
        "c100": "c100",
        "cifar100": "c100",
        "imnet": "imnet",
        "imagenet": "imnet",
        "timnet": "timnet",
        "tiny-imagenet": "timnet",
        "tiny_imagenet": "timnet",
    }
    key = dataset_name.lower()
    if key not in mapping:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    return mapping[key]


def _num_classes_and_stem(dataset_key: str) -> (int, bool):
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
    dataset_key: str,
    device: str,
    timnet_teacher_epochs: int,
) -> ResNet:
    teachers = _TEACHER_MAP[model_size]
    if dataset_key in ("c100", "imnet"):
        return teachers[dataset_key](device=device)
    if dataset_key == "timnet":
        return teachers[dataset_key](device=device, epochs=timnet_teacher_epochs)
    raise ValueError(f"Unsupported dataset key for teachers: {dataset_key}")


class LitBitResNetKD(LitBit):
    def __init__(
        self,
        lr: float,
        wd: float,
        epochs: int,
        label_smoothing: float = 0.1,
        alpha_kd: float = 0.7,
        alpha_hint: float = 0.05,
        T: float = 4.0,
        scale_op: str = "median",
        width_mult: float = 1.0,
        amp: bool = True,
        export_dir: Optional[str] = None,
        dataset_name: str = "c100",
        timnet_teacher_epochs: int = 200,
        model_size: str = "18",
    ) -> None:
        model_size = str(model_size)
        if model_size not in ("18", "50"):
            raise ValueError(f"Unsupported model_size: {model_size}")

        dataset_key = _canonical_dataset(dataset_name)
        num_classes, small_stem = _num_classes_and_stem(dataset_key)

        student = _build_student(
            model_size=model_size,
            num_classes=num_classes,
            scale_op=scale_op,
            small_stem=small_stem,
        )
        teacher = _build_teacher(
            model_size=model_size,
            dataset_key=dataset_key,
            device="cpu",
            timnet_teacher_epochs=timnet_teacher_epochs,
        )

        export_dir = export_dir or f"./ckpt_kd_rn{model_size}"

        super().__init__(
            lr,
            wd,
            epochs,
            label_smoothing,
            alpha_kd,
            alpha_hint,
            T,
            scale_op,
            width_mult,
            amp,
            export_dir,
            dataset_name=dataset_key,
            model_name="resnet",
            model_size=model_size,
            hint_points=["layer1", "layer2", "layer3", "layer4"],
            student=student,
            teacher=teacher,
            num_classes=num_classes,
        )


class LitBitResNet18KD(LitBitResNetKD):
    def __init__(
        self,
        lr: float,
        wd: float,
        epochs: int,
        label_smoothing: float = 0.1,
        alpha_kd: float = 0.7,
        alpha_hint: float = 0.05,
        T: float = 4.0,
        scale_op: str = "median",
        width_mult: float = 1.0,
        amp: bool = True,
        export_dir: str = "./ckpt_kd_rn18",
        dataset_name: str = "c100",
        timnet_teacher_epochs: int = 200,
    ) -> None:
        super().__init__(
            lr,
            wd,
            epochs,
            label_smoothing,
            alpha_kd,
            alpha_hint,
            T,
            scale_op,
            width_mult,
            amp,
            export_dir,
            dataset_name,
            timnet_teacher_epochs,
            model_size="18",
        )


class LitBitResNet50KD(LitBitResNetKD):
    def __init__(
        self,
        lr: float,
        wd: float,
        epochs: int,
        label_smoothing: float = 0.1,
        alpha_kd: float = 0.7,
        alpha_hint: float = 0.05,
        T: float = 4.0,
        scale_op: str = "median",
        width_mult: float = 1.0,
        amp: bool = True,
        export_dir: str = "./ckpt_kd_rn50",
        dataset_name: str = "c100",
        timnet_teacher_epochs: int = 200,
    ) -> None:
        super().__init__(
            lr,
            wd,
            epochs,
            label_smoothing,
            alpha_kd,
            alpha_hint,
            T,
            scale_op,
            width_mult,
            amp,
            export_dir,
            dataset_name,
            timnet_teacher_epochs,
            model_size="50",
        )


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser = add_common_args(parser)
    parser.add_argument("--model-size", type=str, default="18", choices=["18", "50"])
    parser.add_argument(
        "--dataset",
        type=str,
        default="timnet",
        choices=["c100", "imnet", "timnet"],
        help="Dataset to use (affects stems, classes, transforms)",
    )
    parser.add_argument(
        "--timnet_teacher_epochs",
        type=int,
        default=200,
        choices=[50, 100, 200],
        help="Which Tiny-ImageNet ResNet teacher to load from zeyuanyin/tiny-imagenet",
    )
    parser.set_defaults(out=None)
    return parser


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = get_parser()
    args = parser.parse_args(argv)
    if args.out is None:
        args.out = f"./ckpt_{args.dataset}_rn{args.model_size}"
    return args


def run_training(args: argparse.Namespace) -> None:
    export_dir = f"{args.out}_{args.dataset}"
    lit = LitBitResNetKD(
        lr=args.lr,
        wd=args.wd,
        epochs=args.epochs,
        label_smoothing=args.label_smoothing,
        alpha_kd=args.alpha_kd,
        alpha_hint=args.alpha_hint,
        T=args.T,
        scale_op=args.scale_op,
        width_mult=args.width_mult,
        amp=args.amp,
        export_dir=export_dir,
        dataset_name=args.dataset,
        timnet_teacher_epochs=args.timnet_teacher_epochs,
        model_size=args.model_size,
    )

    dmargs = dict(
        data_dir=args.data,
        batch_size=args.batch_size,
        num_workers=4,
        aug_mixup=args.mixup,
        aug_cutmix=args.cutmix,
        alpha=args.mix_alpha,
    )

    if args.dataset == "c100":
        dm = CIFAR100DataModule(**dmargs)
    elif args.dataset == "imnet":
        dm = ImageNetDataModule(**dmargs)
    elif args.dataset == "timnet":
        dm = TinyImageNetDataModule(**dmargs)
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    trainer, dm = setup_trainer(args, lit, dm)
    trainer.fit(lit, datamodule=dm)
    trainer.validate(lit, datamodule=dm)


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = parse_args(argv)
    run_training(args)


if __name__ == "__main__":
    main()

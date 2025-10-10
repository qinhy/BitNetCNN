import argparse
from typing import Dict, Iterable, Optional

import torch
import torch.nn as nn

from common_utils import (
    Bit,
    LitBit,
    TinyImageNetDataModule,
    add_common_args,
    setup_trainer,
)


# ---------------------------------------------------------------------------
# Bit-based YOLOv8 classification backbone
# ---------------------------------------------------------------------------


def _make_divisible(v: float, divisor: int = 8) -> int:
    """
    Round channel counts in the same fashion as YOLO (and MobileNet) implementations.
    """
    new_v = max(divisor, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return int(new_v)


def _round_depth(n: int, depth_mult: float) -> int:
    return max(int(round(n * depth_mult)), 1)


class BitConv(nn.Module):
    def __init__(
        self,
        c1: int,
        c2: int,
        k: int = 1,
        s: int = 1,
        p: Optional[int] = None,
        g: int = 1,
        scale_op: str = "median",
    ) -> None:
        super().__init__()
        if p is None:
            p = k // 2
        self.conv = Bit.Conv2d(c1, c2, k, stride=s, padding=p, groups=g, bias=False, scale_op=scale_op)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class BitBottleneck(nn.Module):
    def __init__(
        self,
        c1: int,
        c2: int,
        shortcut: bool = True,
        expansion: float = 0.5,
        scale_op: str = "median",
    ) -> None:
        super().__init__()
        hidden = int(c2 * expansion)
        self.cv1 = BitConv(c1, hidden, k=1, s=1, p=0, scale_op=scale_op)
        self.cv2 = BitConv(hidden, c2, k=3, s=1, p=1, scale_op=scale_op)
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.cv2(self.cv1(x))
        return x + y if self.add else y


class BitC2f(nn.Module):
    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        shortcut: bool = True,
        expansion: float = 0.5,
        scale_op: str = "median",
    ) -> None:
        super().__init__()
        hidden = int(c2 * expansion)
        self.cv1 = BitConv(c1, hidden * 2, k=1, s=1, p=0, scale_op=scale_op)
        self.cv2 = BitConv(hidden * (n + 2), c2, k=1, s=1, p=0, scale_op=scale_op)
        self.m = nn.ModuleList(
            [
                BitBottleneck(hidden, hidden, shortcut=shortcut, expansion=1.0, scale_op=scale_op)
                for _ in range(n)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cv1(x)
        y1, y2 = torch.chunk(x, 2, dim=1)
        y = [y1, y2]
        for block in self.m:
            y.append(block(y[-1]))
        return self.cv2(torch.cat(y, dim=1))


class BitSPPF(nn.Module):
    def __init__(self, c1: int, c2: int, k: int = 5, scale_op: str = "median") -> None:
        super().__init__()
        hidden = c1 // 2
        self.cv1 = BitConv(c1, hidden, k=1, s=1, p=0, scale_op=scale_op)
        self.cv2 = BitConv(hidden * 4, c2, k=1, s=1, p=0, scale_op=scale_op)
        self.pool = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cv1(x)
        y1 = self.pool(x)
        y2 = self.pool(y1)
        y3 = self.pool(y2)
        return self.cv2(torch.cat([x, y1, y2, y3], dim=1))


class BitYOLOv8Classifier(nn.Module):
    """
    Tiny-ImageNet friendly Bit-based implementation of the YOLOv8 classification backbone.
    Defaults mimic the lightweight `yolov8n-cls` variant.
    """

    def __init__(
        self,
        num_classes: int = 200,
        width_mult: float = 0.25,
        depth_mult: float = 0.34,
        scale_op: str = "median",
        in_ch: int = 3,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.width_mult = width_mult
        self.depth_mult = depth_mult
        self.scale_op = scale_op
        self.in_ch = in_ch

        def c(channels: int) -> int:
            return _make_divisible(channels * width_mult)

        def d(repeats: int) -> int:
            return _round_depth(repeats, depth_mult)

        c1 = c(64)
        c2 = c(128)
        c3 = c(256)
        c4 = c(512)

        self.stem = BitConv(in_ch, c1, k=3, s=2, scale_op=scale_op)
        self.stage1 = nn.Sequential(
            BitConv(c1, c2, k=3, s=2, scale_op=scale_op),
            BitC2f(c2, c2, n=d(3), shortcut=True, scale_op=scale_op),
        )
        self.stage2 = nn.Sequential(
            BitConv(c2, c3, k=3, s=2, scale_op=scale_op),
            BitC2f(c3, c3, n=d(6), shortcut=True, scale_op=scale_op),
        )
        self.stage3 = nn.Sequential(
            BitConv(c3, c4, k=3, s=2, scale_op=scale_op),
            BitC2f(c4, c4, n=d(6), shortcut=True, scale_op=scale_op),
        )
        self.sppf = BitSPPF(c4, c4, k=5, scale_op=scale_op)

        self.head = nn.Sequential(
            BitConv(c4, c4, k=1, s=1, p=0, scale_op=scale_op),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            Bit.Linear(c4, num_classes, bias=True, scale_op=scale_op),
        )

        self.hint_points = ["stage1", "stage2", "stage3", "sppf"]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.sppf(x)
        return self.head(x)

    def clone(self) -> "BitYOLOv8Classifier":
        return self.__class__(
            num_classes=self.num_classes,
            width_mult=self.width_mult,
            depth_mult=self.depth_mult,
            scale_op=self.scale_op,
            in_ch=self.in_ch,
        )


# ---------------------------------------------------------------------------
# Teacher loading (ultralytics YOLOv8)
# ---------------------------------------------------------------------------


class YOLOv8ClassifierWrapper(nn.Module):
    """
    Thin wrapper to standardise outputs from ultralytics YOLOv8 classification models.
    """

    def __init__(self, base_model: nn.Module, num_classes: int) -> None:
        super().__init__()
        self.base_model = base_model
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.base_model(x)
        if isinstance(logits, (list, tuple)):
            logits = logits[0]
        nc = logits.shape[-1]
        if nc < self.num_classes:
            raise ValueError(
                f"YOLOv8 teacher produced {nc} output classes, "
                f"which is fewer than the required {self.num_classes}."
            )
        if nc > self.num_classes:
            logits = logits[..., : self.num_classes]
        return logits


def load_yolov8_teacher(
    variant: str,
    checkpoint: Optional[str],
    num_classes: int,
) -> nn.Module:
    """
    Load a YOLOv8 classification model via ultralytics.
    """
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError(
            "YOLOv8 distillation requires the 'ultralytics' package. "
            "Install it via `pip install ultralytics`."
        ) from exc

    source = checkpoint or variant
    model = YOLO(source)

    if not hasattr(model, "model"):
        raise RuntimeError("Unexpected YOLOv8 object structure; missing .model attribute.")

    return YOLOv8ClassifierWrapper(model.model, num_classes=num_classes)


# ---------------------------------------------------------------------------
# Lightning module
# ---------------------------------------------------------------------------


class LitBitYOLOv8KD(LitBit):
    """
    Knowledge distillation LightningModule that pairs our Bit YOLOv8 student with an Ultralytics YOLOv8 teacher.
    """

    def __init__(
        self,
        lr: float,
        wd: float,
        epochs: int,
        label_smoothing: float,
        alpha_kd: float,
        alpha_hint: float,
        T: float,
        scale_op: str,
        amp: bool,
        export_dir: str,
        teacher_variant: str,
        teacher_checkpoint: Optional[str],
        width_mult: float,
        depth_mult: float,
    ) -> None:
        num_classes = 200  # Tiny-ImageNet
        student = BitYOLOv8Classifier(
            num_classes=num_classes,
            width_mult=width_mult,
            depth_mult=depth_mult,
            scale_op=scale_op,
        )
        teacher = load_yolov8_teacher(
            variant=teacher_variant,
            checkpoint=teacher_checkpoint,
            num_classes=num_classes,
        )

        super().__init__(
            lr=lr,
            wd=wd,
            epochs=epochs,
            label_smoothing=label_smoothing,
            alpha_kd=alpha_kd,
            alpha_hint=alpha_hint,
            T=T,
            scale_op=scale_op,
            width_mult=1.0,
            amp=amp,
            export_dir=export_dir,
            student=student,
            teacher=teacher,
            dataset_name="timnet",
            model_name="yolov8",
            model_size=f"w{width_mult:.2f}_d{depth_mult:.2f}",
            hint_points=getattr(student, "hint_points", []),
            num_classes=num_classes,
        )


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser = add_common_args(parser)
    parser.add_argument(
        "--dataset",
        type=str,
        default="timnet",
        choices=["timnet"],
        help="Currently only Tiny-ImageNet is supported for YOLOv8 distillation.",
    )
    parser.add_argument(
        "--teacher-variant",
        type=str,
        default="yolov8n-cls.pt",
        help="YOLOv8 classification weight identifier understood by `ultralytics.YOLO`.",
    )
    parser.add_argument(
        "--teacher-checkpoint",
        type=str,
        default=None,
        help="Optional local checkpoint path for the YOLOv8 teacher (overrides --teacher-variant).",
    )
    parser.add_argument(
        "--width-mult",
        type=float,
        default=0.25,
        help="Channel width multiplier for the Bit YOLOv8 student (default: 0.25).",
    )
    parser.add_argument(
        "--depth-mult",
        type=float,
        default=0.34,
        help="Depth multiplier (number of bottlenecks per stage) for the Bit YOLOv8 student (default: 0.34).",
    )
    parser.set_defaults(
        alpha_hint=0.0,
        out="./ckpt_timnet_yolov8",
    )
    return parser


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = get_parser()
    args = parser.parse_args(argv)

    if args.dataset != "timnet":
        raise ValueError("Only Tiny-ImageNet is supported at the moment.")

    if not args.out:
        args.out = "./ckpt_timnet_yolov8"

    return args


def run_training(args: argparse.Namespace) -> None:
    export_dir = f"{args.out}_{args.dataset}"
    lit = LitBitYOLOv8KD(
        lr=args.lr,
        wd=args.wd,
        epochs=args.epochs,
        label_smoothing=args.label_smoothing,
        alpha_kd=args.alpha_kd,
        alpha_hint=args.alpha_hint,
        T=args.T,
        scale_op=args.scale_op,
        amp=args.amp,
        export_dir=export_dir,
        teacher_variant=args.teacher_variant,
        teacher_checkpoint=args.teacher_checkpoint,
        width_mult=args.width_mult,
        depth_mult=args.depth_mult,
    )

    dm_kwargs: Dict[str, object] = dict(
        data_dir=args.data,
        batch_size=args.batch_size,
        num_workers=4,
        aug_mixup=args.mixup,
        aug_cutmix=args.cutmix,
        alpha=args.mix_alpha,
    )
    dm = TinyImageNetDataModule(**dm_kwargs)

    trainer, dm = setup_trainer(args, lit, dm)
    trainer.fit(lit, datamodule=dm)
    trainer.validate(lit, datamodule=dm)


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = parse_args(argv)
    run_training(args)


if __name__ == "__main__":
    main()

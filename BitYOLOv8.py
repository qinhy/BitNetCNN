# refactored_yolov8_kd.py
import argparse
import json
import os
from typing import Dict, Iterable, Optional, Tuple, List

import torch
import torch.nn as nn

from common_utils import (
    Bit,
    LitBit,
    TinyImageNetDataModule,
    add_common_args,
    setup_trainer,
)

# -----------------------------------------------------------------------------
# YOLOv8 presets and helpers
# -----------------------------------------------------------------------------

SIZE_TO_MULT: Dict[str, Tuple[float, float]] = {
    "n": (0.25, 0.34), "nano":   (0.25, 0.34),
    "s": (0.50, 0.34), "small":  (0.50, 0.34),
    "m": (0.75, 0.67), "medium": (0.75, 0.67),
    "l": (1.00, 1.00), "large":  (1.00, 1.00),
    "x": (1.25, 1.00), "xlarge": (1.25, 1.00),
}

def _size_letter(name: str) -> str:
    return {"nano":"n","small":"s","medium":"m","large":"l","xlarge":"x"}.get(name, name)

def _make_divisible(v: float, divisor: int = 8) -> int:
    new_v = max(divisor, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return int(new_v)

def _round_depth(n: int, depth_mult: float) -> int:
    return max(int(round(n * depth_mult)), 1)

# -----------------------------------------------------------------------------
# Bit-based YOLOv8 classification backbone (parity with Ultralytics' cls head)
# -----------------------------------------------------------------------------

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
        hidden = _make_divisible(c2 * expansion)
        # AFTER : cv1 k=3, cv2 k=3  (Ultralytics YOLOv8-cls parity)
        self.cv1 = BitConv(c1, hidden, k=3, s=1, p=1, scale_op=scale_op)
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
        hidden = _make_divisible(c2 * expansion)
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
        hidden = _make_divisible(c1 * 0.5)
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
    Tiny-ImageNet friendly Bit-based YOLOv8 classification backbone aligned with Ultralytics YOLOv8-cls:
      stem → C2f(c2) → C2f(c3) → C2f(c4) → [optional SPPF] → 1x1 Conv to embed_dim → GAP → Dropout → Linear
    """
    def __init__(
        self,
        num_classes: int = 200,
        width_mult: float = 0.25,
        depth_mult: float = 0.34,
        scale_op: str = "median",
        in_ch: int = 3,
        expansion: float = 0.5,
        use_sppf: bool = False,     # default False (Ultralytics cls has no SPPF)
        embed_dim: int = 1280,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.width_mult = width_mult
        self.depth_mult = depth_mult
        self.scale_op = scale_op
        self.in_ch = in_ch
        self.expansion = expansion
        self.use_sppf = use_sppf
        self.embed_dim = embed_dim
        self.dropout = dropout

        def c(channels: int) -> int:
            return _make_divisible(channels * width_mult)

        def d(repeats: int) -> int:
            return _round_depth(repeats, depth_mult)

        # base channels (YOLOv8 style)
        c1 = c(64)
        c2 = c(128)
        c3 = c(256)
        c4 = c(512)
        c5 = c(1024)  # NEW: extra stage to reach 256 when width_mult=0.25

        self.stem = BitConv(in_ch, c1, k=3, s=2, scale_op=scale_op)

        self.stage1 = nn.Sequential(
            BitConv(c1, c2, k=3, s=2, scale_op=scale_op),
            BitC2f(c2, c2, n=d(3), shortcut=True, expansion=expansion, scale_op=scale_op),
        )
        self.stage2 = nn.Sequential(
            BitConv(c2, c3, k=3, s=2, scale_op=scale_op),
            BitC2f(c3, c3, n=d(6), shortcut=True, expansion=expansion, scale_op=scale_op),
        )
        self.stage3 = nn.Sequential(
            BitConv(c3, c4, k=3, s=2, scale_op=scale_op),
            BitC2f(c4, c4, n=d(6), shortcut=True, expansion=expansion, scale_op=scale_op),
        )

        # NEW: stage4 to match Ultralytics cls backbone (final feature = 256 in nano)
        self.stage4 = nn.Sequential(
            BitConv(c4, c5, k=3, s=2, scale_op=scale_op),
            BitC2f(c5, c5, n=d(2), shortcut=True, expansion=expansion, scale_op=scale_op),
        )

        # If you keep SPPF optional, apply it after stage4 (default off)
        if use_sppf:
            self.sppf = BitSPPF(c5, c5, k=5, scale_op=scale_op)
            c_head_in = c5
        else:
            self.sppf = None
            c_head_in = c5

        # Head: 1×1 to 1280 → GAP → Dropout → Linear
        self.head = nn.Sequential(
            BitConv(c_head_in, embed_dim, k=1, s=1, p=0, scale_op=scale_op),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(p=dropout),
            Bit.Linear(embed_dim, num_classes, bias=True, scale_op=scale_op),
        )

        # Update hint points
        self.hint_points = [("stage1","model.2"), ("stage2","model.4"), ("stage3","model.6"), ("stage4","model.8")] + (["sppf"] if use_sppf else [])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)      # NEW
        if self.sppf is not None:
            x = self.sppf(x)
        return self.head(x)


    def clone(self) -> "BitYOLOv8Classifier":
        return self.__class__(
            num_classes=self.num_classes,
            width_mult=self.width_mult,
            depth_mult=self.depth_mult,
            scale_op=self.scale_op,
            in_ch=self.in_ch,
            expansion=self.expansion,
            use_sppf=self.use_sppf,
            embed_dim=self.embed_dim,
            dropout=self.dropout,
        )

# -----------------------------------------------------------------------------
# Teacher wrapper (with optional class index mapping)
# -----------------------------------------------------------------------------

class YOLOv8ClassifierWrapper(nn.Module):
    def __init__(self, base_model: nn.Module, num_classes: int, class_indices: Optional[List[int]] = None) -> None:
        super().__init__()
        self.base_model = base_model
        self.num_classes = num_classes
        self.class_indices = None
        if class_indices is not None:
            idx = torch.tensor(class_indices, dtype=torch.long)
            if idx.ndim != 1:
                raise ValueError("class_indices must be a 1-D sequence of integers.")
            self.register_buffer("_idx", idx, persistent=False)
            self.class_indices = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.base_model(x)
        if isinstance(logits, (list, tuple)):
            logits = logits[0]
        if self.class_indices:
            if logits.shape[-1] <= int(self._idx.max()):
                raise ValueError(
                    f"Teacher produced {logits.shape[-1]} classes, but class_indices index up to {int(self._idx.max())}."
                )
            return logits.index_select(-1, self._idx.to(logits.device))
        if logits.shape[-1] != self.num_classes:
            raise ValueError(
                f"Teacher produced {logits.shape[-1]} classes, expected {self.num_classes}."
            )
        return logits

def _load_class_indices(path: str, expected: int) -> List[int]:
    with open(path, "r", encoding="utf-8") as f:
        txt = f.read().strip()
    indices: Optional[List[int]] = None
    # Try JSON
    try:
        obj = json.loads(txt)
        if isinstance(obj, list) and all(isinstance(x, int) for x in obj):
            indices = obj
        elif isinstance(obj, dict) and "class_indices" in obj and isinstance(obj["class_indices"], list):
            cand = obj["class_indices"]
            if all(isinstance(x, int) for x in cand):
                indices = cand
    except json.JSONDecodeError:
        pass
    # Fallback: newline-separated ints
    if indices is None:
        lines = [ln.strip() for ln in txt.splitlines() if ln.strip()]
        if lines and all(l.replace("-", "").isdigit() for l in lines):
            indices = [int(l) for l in lines]

    if indices is None:
        raise ValueError(f"Could not parse class indices from: {path}")
    if len(indices) != expected:
        raise ValueError(f"class_indices length {len(indices)} != expected {expected}")
    if min(indices) < 0:
        raise ValueError("class_indices contain negative entries.")
    return indices

def load_yolov8_teacher(
    variant: str,
    checkpoint: Optional[str],
    num_classes: int,
    class_indices: Optional[List[int]] = None,
) -> nn.Module:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError(
            "YOLOv8 distillation requires the 'ultralytics' package. Install via `pip install ultralytics`."
        ) from exc

    source = checkpoint or variant
    model = YOLO(source)
    if not hasattr(model, "model"):
        raise RuntimeError("Unexpected YOLOv8 object structure; missing .model attribute.")
    return YOLOv8ClassifierWrapper(model.model, num_classes=num_classes, class_indices=class_indices)

# -----------------------------------------------------------------------------
# Lightning module
# -----------------------------------------------------------------------------

class LitBitYOLOv8KD(LitBit):
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
        model_size_str: str,
        teacher_class_map: Optional[str],
        expansion: float = 0.5,
        use_sppf: bool = False,
        embed_dim: int = 1280,
        dropout: float = 0.0,
        print_summary: bool = True,
    ) -> None:
        num_classes = 200  # Tiny-ImageNet

        student = BitYOLOv8Classifier(
            num_classes=num_classes,
            width_mult=width_mult,
            depth_mult=depth_mult,
            scale_op=scale_op,
            expansion=expansion,
            use_sppf=use_sppf,
            embed_dim=embed_dim,
            dropout=dropout,
        )

        class_indices = None
        if teacher_class_map:
            class_indices = _load_class_indices(teacher_class_map, expected=num_classes)

        teacher = load_yolov8_teacher(
            variant=teacher_variant,
            checkpoint=teacher_checkpoint,
            num_classes=num_classes,
            class_indices=class_indices,
        )

        if print_summary:
            total = sum(p.numel() for p in student.parameters())
            trainable = sum(p.numel() for p in student.parameters() if p.requires_grad)
            print(f"[Student] params: {total/1e6:.3f}M (trainable {trainable/1e6:.3f}M) | size={model_size_str}, "
                  f"w={width_mult:.2f}, d={depth_mult:.2f}, embed={embed_dim}, sppf={use_sppf}")

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
            model_size=f"{model_size_str}",#_w{width_mult:.2f}_d{depth_mult:.2f}_e{expansion:.2f}_emb{embed_dim}",
            hint_points=getattr(student, "hint_points", []),
            num_classes=num_classes,
        )

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser = add_common_args(parser)

    parser.add_argument("--dataset", type=str, default="timnet", choices=["timnet"],
                        help="Currently only Tiny-ImageNet is supported for YOLOv8 distillation.")

    parser.add_argument("--teacher-variant", type=str, default=None,
                        help="Ultralytics YOLOv8 cls weight id (e.g., yolov8n-cls.pt). If omitted, inferred from --model-size.")
    parser.add_argument("--teacher-checkpoint", type=str, default=None,
                        help="Optional local checkpoint path for the YOLOv8 teacher (overrides --teacher-variant).")

    parser.add_argument("--model-size", type=str, default="nano",
                        choices=["n","s","m","l","x","nano","small","medium","large","xlarge"],
                        help="Student size preset (maps to YOLOv8 width/depth multipliers).")
    parser.add_argument("--width-mult", type=float, default=None,
                        help="Override width multiplier (if set, overrides --model-size for width).")
    parser.add_argument("--depth-mult", type=float, default=None,
                        help="Override depth multiplier (if set, overrides --model-size for depth).")

    parser.add_argument("--teacher-class-map", type=str, default='./timnet_to_imagenet1k_indices.txt',
                        help="Path to 200-length list (txt or JSON) mapping Tiny-ImageNet order to teacher indices.")

    parser.add_argument("--expansion", type=float, default=0.5,
                        help="Bottleneck expansion ratio (hidden=c2*expansion, rounded).")
    parser.add_argument("--use-sppf", action="store_true",
                        help="Use SPPF block before the head (Ultralytics YOLOv8-cls defaults to NO SPPF).")
    parser.add_argument("--embed-dim", type=int, default=1280,
                        help="Head embedding width (Ultralytics YOLOv8-cls uses 1280).")
    parser.add_argument("--dropout", type=float, default=0.0,
                        help="Dropout before classifier.")

    parser.add_argument("--print-summary", action="store_true",
                        help="Print student parameter summary on startup.")    
    parser.set_defaults(out=None)
    return parser

def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = get_parser()
    args = parser.parse_args(argv)

    if args.dataset != "timnet":
        raise ValueError("Only Tiny-ImageNet is supported at the moment.")

    if not args.out:
        args.out = "./ckpt_timnet_yolov8"

    if args.model_size not in SIZE_TO_MULT:
        raise ValueError(f"Unknown model size: {args.model_size}")
    size_w, size_d = SIZE_TO_MULT[args.model_size]
    if args.width_mult is None:
        args.width_mult = size_w
    if args.depth_mult is None:
        args.depth_mult = size_d

    if args.teacher_checkpoint is None and not args.teacher_variant:
        letter = _size_letter(args.model_size)
        args.teacher_variant = f"yolov8{letter}-cls.pt"

    return args

# -----------------------------------------------------------------------------
# Train / Validate
# -----------------------------------------------------------------------------

def run_training(args: argparse.Namespace) -> None:
    export_dir = args.out
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
        model_size_str=_size_letter(args.model_size),
        teacher_class_map=args.teacher_class_map,
        expansion=args.expansion,
        use_sppf=args.use_sppf,
        embed_dim=args.embed_dim,
        dropout=args.dropout,
        print_summary=args.print_summary,
    )

    dm_kwargs: Dict[str, object] = dict(
        data_dir=args.data,
        batch_size=args.batch_size,
        num_workers=min(8, os.cpu_count() or 4),
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

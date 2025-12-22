import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from pydantic import BaseModel, Field, field_validator
from pydanticV2_argparse import ArgumentParser
import torch

from trainer import LitBit

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.ops import box_iou
from PIL import Image

from bitlayers.convs import ActModels, Conv2dModels, NormModels, Bit
from common_utils import convert_to_ternary
from BitResNet import BitResNet

# -----------------------------------------------------------------------------
# Bit-based ResNet backbone that returns multi-scale features
# -----------------------------------------------------------------------------
BasicBlockBit = Conv2dModels.ResNetBasicBlock
BottleneckBit = Conv2dModels.ResNetBottleneck

class BitResNetBackbone(BitResNet):
    def __init__(self, model_size: str, scale_op: str = "median", in_ch: int = 3, small_stem: bool = True):
        model_size = str(model_size)

        if model_size == "18":
            block_cls, layers, expansion = BasicBlockBit, [2, 2, 2, 2], 1
        elif model_size == "50":
            block_cls, layers, expansion = BottleneckBit, [3, 4, 6, 3], 4
        else:
            raise ValueError(f"Unsupported model_size={model_size!r}. Expected '18' or '50'.")

        super().__init__(block_cls, layers, num_classes=1000, expansion=expansion,
                         scale_op=scale_op, in_ch=in_ch, small_stem=small_stem)

        del self.head
        self.feature_channels = (128 * expansion, 256 * expansion, 512 * expansion)

    def forward(self, x: torch.Tensor):
        _, c3, c4, c5 = self.forward_features(x)
        return c3, c4, c5


# -----------------------------------------------------------------------------
# RetinaFace-specific blocks
# -----------------------------------------------------------------------------
def _act(inplace=True):
    return ActModels.SiLU(inplace=inplace)

def _conv_block(
    in_ch: int, out_ch: int,
    k: int, stride: int, padding: int,
    scale_op: str,
    act: Any = ActModels.SiLU(inplace=True),
    norm: Any = NormModels.BatchNorm2d(num_features=-1),
    bias=False,
) -> nn.Sequential:
    conf = dict(
        in_channels=in_ch,
        out_channels=out_ch,
        kernel_size=k,
        stride=stride,
        padding=padding,
        bias=bias,
        bit=True,
        scale_op=scale_op,
        norm=norm,
    )
    if act and norm:
        return Conv2dModels.Conv2dNormAct(**conf,act=act).build()
    if not act and norm:
        return Conv2dModels.Conv2dNorm(**conf).build()
    if act and not norm:
        return Conv2dModels.Conv2dAct(**conf).build()
    return Conv2dModels.Conv2d(**conf).build()

class SSH(nn.Module):
    """Single Stage Headless (SSH) context module used in RetinaFace."""

    def __init__(self, in_ch: int, out_ch: int, scale_op: str = "median") -> None:
        super().__init__()
        if out_ch % 4 != 0:
            raise ValueError("SSH output channels must be divisible by 4.")

        half = out_ch // 2
        quart = out_ch // 4

        # 3x3 branch
        self.branch3 = _conv_block(in_ch, half, 3, 1, 1, scale_op, act=False)

        # shared stem for 5x5 and 7x7 branches
        self.stem5   = _conv_block(in_ch, quart, 3, 1, 1, scale_op, act=True)

        # 5x5 branch (two 3x3 convs total)
        self.branch5 = _conv_block(quart, quart, 3, 1, 1, scale_op, act=False)

        # 7x7 branch (three 3x3 convs total; last two are here)
        self.branch7 = nn.Sequential(
            _conv_block(quart, quart, 3, 1, 1, scale_op, act=True),
            _conv_block(quart, quart, 3, 1, 1, scale_op, act=False),
        )

        self.act = _act(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c3 = self.branch3(x)

        c5_1 = self.stem5(x)      # shared intermediate
        c5 = self.branch5(c5_1)   # 5x5 path output
        c7 = self.branch7(c5_1)   # 7x7 path output (must use c5_1)

        return self.act(torch.cat((c3, c5, c7), dim=1))


class FPN(nn.Module):
    def __init__(
        self, in_channels: Sequence[int], out_channels: int = 256, scale_op: str = "median"):
        super().__init__()

        self.lateral = nn.ModuleList(
            _conv_block(ch, out_channels, k=1, stride=1, padding=0, scale_op=scale_op, act=False)
            for ch in in_channels
        )
        self.output = nn.ModuleList(
            _conv_block(out_channels, out_channels, k=3, stride=1, padding=1, scale_op=scale_op, act=False)
            for _ in in_channels
        )

    def forward(self, feats: Sequence[torch.Tensor]) -> List[torch.Tensor]:
        results: List[torch.Tensor] = []
        last: torch.Tensor | None = None

        for idx in range(len(feats) - 1, -1, -1):
            lat = self.lateral[idx](feats[idx])

            if last is not None:
                lat = lat + F.interpolate(last, size=lat.shape[-2:], mode="nearest")

            last = lat
            results.append(self.output[idx](lat))

        results.reverse()
        return results

class PredictionHead(nn.Module):
    def __init__(
        self,
        num_levels: int,
        in_channels: int,
        out_per_anchor: int,
        num_priors_per_level: Sequence[int],
        scale_op: str = "median",
    ) -> None:
        super().__init__()
        if len(num_priors_per_level) != num_levels:
            raise ValueError("len(num_priors_per_level) must equal num_levels")

        self.out_per_anchor = out_per_anchor
        self.num_priors = tuple(num_priors_per_level)

        self.heads = nn.ModuleList(
            nn.Sequential(
                _conv_block(in_channels, in_channels, k=3, stride=1, padding=1, scale_op=scale_op),
                _conv_block(in_channels, priors * out_per_anchor,
                            k=1, stride=1, padding=0, scale_op=scale_op,
                            act=False,norm=False,bias=True),
            )
            for priors in self.num_priors
        )

    def forward(self, feats: Sequence[torch.Tensor]) -> torch.Tensor:
        preds = []
        for feat, head in zip(feats, self.heads):
            p = head(feat).permute(0, 2, 3, 1).contiguous()
            preds.append(p.view(p.size(0), -1, self.out_per_anchor))
        return torch.cat(preds, dim=1)

class RetinaFaceTensor(BaseModel):
    label:Optional[torch.Tensor] = Field(default=None,exclude=True)
    bbox:Optional[torch.Tensor] = Field(default=None,exclude=True)
    landmark:Optional[torch.Tensor] = Field(default=None,exclude=True)

class BitRetinaFace(nn.Module):
    """Bit-friendly RetinaFace with configurable ResNet backbone."""

    def __init__(
        self,
        backbone_size: str = "18",
        fpn_channels: int = 256,
        num_classes: int = 2,
        num_priors: Sequence[int] = (2, 2, 2),
        scale_op: str = "median",
        in_ch: int = 3,
        small_stem: bool = False,
    ) -> None:
        super().__init__()

        self.backbone_size = str(backbone_size)
        self.fpn_channels = fpn_channels
        self.num_classes = num_classes
        self.num_priors = tuple(num_priors)
        self.scale_op = scale_op
        self.in_ch = in_ch
        self.small_stem = small_stem

        num_levels = 3

        self.backbone = BitResNetBackbone(
            model_size=self.backbone_size,
            in_ch=in_ch,
            scale_op=scale_op,
            small_stem=small_stem,
        )

        self.fpn = FPN(self.backbone.feature_channels, fpn_channels, scale_op=scale_op)

        self.ssh = nn.ModuleList(
            SSH(fpn_channels, fpn_channels, scale_op=scale_op) for _ in range(num_levels)
        )
        head = lambda num_classes:PredictionHead(
            num_levels=num_levels,
            in_channels=fpn_channels,
            out_per_anchor=num_classes,
            num_priors_per_level=self.num_priors,
            scale_op=scale_op,
        )
        self.classification_head = head(num_classes)
        self.bbox_head = head(4)
        self.landmark_head = head(10)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        feats = self.fpn(self.backbone(x))
        feats = [m(f) for m, f in zip(self.ssh, feats)]
        return RetinaFaceTensor(label=self.classification_head(feats),
                                bbox=self.bbox_head(feats),
                                landmark=self.landmark_head(feats),)
        return {
            "cls": self.classification_head(feats),
            "bbox": self.bbox_head(feats),
            "landmark": self.landmark_head(feats),
        }

    def clone(self) -> "BitRetinaFace":
        model = BitRetinaFace(
            backbone_size=self.backbone_size,
            fpn_channels=self.fpn_channels,
            num_classes=self.num_classes,
            num_priors=self.num_priors,
            scale_op=self.scale_op,
            in_ch=self.in_ch,
            small_stem=self.small_stem,
        )
        model.load_state_dict(self.state_dict(), strict=True)
        return model

# -----------------------------------------------------------------------------
# Anchors and target assignment
# -----------------------------------------------------------------------------
class RetinaFaceAnchors:
    def __init__(
        self,
        min_sizes: Sequence[Sequence[int]] = ((16, 32), (64, 128), (256, 512)),
        steps: Sequence[int] = (8, 16, 32),
        clip: bool = False,
    ) -> None:
        assert len(min_sizes) == len(steps)
        self.min_sizes = tuple(tuple(m) for m in min_sizes)
        self.steps = tuple(int(s) for s in steps)
        self.clip = clip

    def __call__(
        self,
        image_size: Tuple[int, int],
        device: torch.device = torch.device("cpu"),
    ) -> torch.Tensor:
        img_h, img_w = image_size
        anchors = []
        for step, sizes in zip(self.steps, self.min_sizes):
            feature_h = math.ceil(img_h / step)
            feature_w = math.ceil(img_w / step)
            for i in range(feature_h):
                for j in range(feature_w):
                    cx = (j + 0.5) * step / img_w
                    cy = (i + 0.5) * step / img_h
                    for size in sizes:
                        s_kx = size / img_w
                        s_ky = size / img_h
                        anchors.append([cx, cy, s_kx, s_ky])
        anchors_tensor = torch.tensor(anchors, dtype=torch.float32, device=device)
        if self.clip:
            anchors_tensor.clamp_(0.0, 1.0)
        return anchors_tensor


def decode_anchors(anchors: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = anchors.unbind(dim=1)
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=1)


def encode_boxes(
    anchors: torch.Tensor,
    gt_boxes: torch.Tensor,
    variances: Tuple[float, float],
) -> torch.Tensor:
    cx = anchors[:, 0]
    cy = anchors[:, 1]
    w = anchors[:, 2]
    h = anchors[:, 3]

    gx = (gt_boxes[:, 0] + gt_boxes[:, 2]) / 2
    gy = (gt_boxes[:, 1] + gt_boxes[:, 3]) / 2
    gw = gt_boxes[:, 2] - gt_boxes[:, 0]
    gh = gt_boxes[:, 3] - gt_boxes[:, 1]

    eps = 1e-6
    encoded = torch.zeros_like(gt_boxes)
    encoded[:, 0] = (gx - cx) / (w + eps) / variances[0]
    encoded[:, 1] = (gy - cy) / (h + eps) / variances[0]
    encoded[:, 2] = torch.log((gw + eps) / (w + eps)) / variances[1]
    encoded[:, 3] = torch.log((gh + eps) / (h + eps)) / variances[1]
    return encoded


def encode_landmarks(
    anchors: torch.Tensor,
    gt_landmarks: torch.Tensor,
    variances: Tuple[float, float],
) -> torch.Tensor:
    cx = anchors[:, 0].unsqueeze(1)
    cy = anchors[:, 1].unsqueeze(1)
    w = anchors[:, 2].unsqueeze(1)
    h = anchors[:, 3].unsqueeze(1)

    # gt_landmarks expected shape: [N, 5, 2]
    eps = 1e-6
    encoded = torch.zeros(gt_landmarks.shape[0], 5, 2, device=gt_landmarks.device)
    encoded[:, :, 0] = (gt_landmarks[:, :, 0] - cx) / (w + eps) / variances[0]
    encoded[:, :, 1] = (gt_landmarks[:, :, 1] - cy) / (h + eps) / variances[0]
    return encoded.view(gt_landmarks.size(0), -1)


def match_anchors(
    anchors: torch.Tensor,
    gt_boxes: torch.Tensor,
    gt_landmarks: torch.Tensor,
    pos_iou: float,
    neg_iou: float,
    variances: Tuple[float, float],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num_anchors = anchors.size(0)
    labels = anchors.new_full((num_anchors,), -1, dtype=torch.long)
    bbox_targets = anchors.new_zeros((num_anchors, 4))
    land_targets = anchors.new_full((num_anchors, 10), -1.0)

    if gt_boxes.numel() == 0:
        labels.fill_(0)
        return labels, bbox_targets, land_targets

    anchor_boxes = decode_anchors(anchors)
    ious = box_iou(anchor_boxes, gt_boxes)
    best_anchor_iou, best_gt_idx = ious.max(dim=1)
    best_gt_iou, best_anchor_for_gt = ious.max(dim=0)

    labels[best_anchor_iou < neg_iou] = 0
    labels[best_anchor_iou >= pos_iou] = 1

    # Guarantee at least one positive anchor per GT box
    labels[best_anchor_for_gt] = 1
    best_gt_idx[best_anchor_for_gt] = torch.arange(gt_boxes.size(0), device=anchors.device)

    pos_mask = labels == 1
    if pos_mask.any():
        assigned_boxes = gt_boxes[best_gt_idx[pos_mask]]
        bbox_targets[pos_mask] = encode_boxes(anchors[pos_mask], assigned_boxes, variances)

        assigned_landmarks = gt_landmarks[best_gt_idx[pos_mask]]
        land_view = land_targets[pos_mask]
        valid_landmarks = (assigned_landmarks >= 0).all(dim=(1, 2))
        if valid_landmarks.any():
            encoded_landmarks = encode_landmarks(
                anchors[pos_mask][valid_landmarks],
                assigned_landmarks[valid_landmarks],
                variances,
            )
            land_view[valid_landmarks] = encoded_landmarks
        land_targets[pos_mask] = land_view

    return labels, bbox_targets, land_targets


# -----------------------------------------------------------------------------
# LightningModule
# -----------------------------------------------------------------------------
class LitRetinaFace(LitBit):
    def __init__(self, config: "RetinaFaceConfig") -> None:
        super().__init__(config)
        self.save_hyperparameters(config.model_dump())
        self.model = BitRetinaFace(
            backbone_size=config.backbone_size,
            fpn_channels=config.fpn_channels,
            num_priors=config.num_priors,
            scale_op=config.scale_op,
            in_ch=3,
            small_stem=config.small_stem,
        )
        self.student = self.model  # for compatibility with ternary export helper
        self.lr = config.lr
        self.wd = config.wd
        self.input_size = config.image_size
        self.variances = (config.variance_xy, config.variance_wh)
        self.pos_iou = config.pos_iou
        self.neg_iou = config.neg_iou
        self.box_weight = config.box_weight
        self.landmark_weight = config.landmark_weight
        self.dataset_name = "widerface"
        self.model_name = "retinaface"
        self.model_size = str(config.backbone_size)

        self.anchor_generator = RetinaFaceAnchors(
            min_sizes=config.anchor_sizes,
            steps=config.steps,
            clip=config.clip_anchors,
        )
        anchor_template = self.anchor_generator(
            (self.input_size, self.input_size),
            device=torch.device("cpu"),
        )
        self.register_buffer("anchors", anchor_template, persistent=False)
        self.epochs = config.epochs

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.model(images)

    def _build_targets(
        self,
        target_boxes: List[torch.Tensor],
        target_landmarks: List[torch.Tensor],):
        device = self.anchors.device
        labels = []
        bbox_t = []
        land_t = []
        for boxes, lands in zip(target_boxes, target_landmarks):
            boxes = boxes.to(device)
            lands = lands.to(device).view(-1, 5, 2)
            l, b, ld = match_anchors(
                self.anchors,
                boxes,
                lands,
                self.pos_iou,
                self.neg_iou,
                self.variances,
            )
            labels.append(l)
            bbox_t.append(b)
            land_t.append(ld)
        return RetinaFaceTensor(
            label=torch.stack(labels, dim=0),
            bbox=torch.stack(bbox_t, dim=0),
            landmark=torch.stack(land_t, dim=0),)


    def _cls_loss(
        self,
        cls_logits: torch.Tensor,
        target_labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Classification loss over anchors.
        Ignores labels < 0.
        """
        labels = target_labels.view(-1)
        valid = labels >= 0

        if valid.sum() == 0:
            return torch.tensor(0.0, device=self.device)

        logits = cls_logits.view(-1, cls_logits.size(-1))
        return F.cross_entropy(logits[valid], labels[valid])


    def _bbox_loss(
        self,
        bbox_preds: torch.Tensor,
        target_boxes: torch.Tensor,
        target_labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Box regression loss over positive anchors only (label == 1),
        normalized by number of positives (clamped to at least 1).
        """
        pos_mask = target_labels == 1
        pos_count = pos_mask.sum().clamp(min=1)

        if not pos_mask.any():
            return torch.tensor(0.0, device=self.device)

        return (
            F.smooth_l1_loss(
                bbox_preds[pos_mask],
                target_boxes[pos_mask],
                reduction="sum",
            )
            / pos_count
        )


    def _landmark_loss(
        self,
        land_preds: torch.Tensor,
        target_landmarks: torch.Tensor,
    ) -> torch.Tensor:
        """
        Landmark regression loss over anchors whose landmarks are fully valid.
        A landmark is considered valid if all coords for the 5 points are >= 0.
        """
        # shape assumed: [*, 5, 2] (or equivalent) so dim=2 is xy
        valid_landmarks = (target_landmarks >= 0).all(dim=2)

        if not valid_landmarks.any():
            return torch.tensor(0.0, device=self.device)

        denom = valid_landmarks.sum().clamp(min=1)
        return (
            F.smooth_l1_loss(
                land_preds[valid_landmarks],
                target_landmarks[valid_landmarks],
                reduction="sum",
            )
            / denom
        )
    
    def training_step(self, batch, batch_idx: int):
        images, targets = batch
        outputs:RetinaFaceTensor = self.student(images)
        targets = self._build_targets(targets["boxes"], targets["landmarks"])
        
        cls = self._cls_loss(outputs.label, targets.label)
        bbox = self._bbox_loss(outputs.bbox, targets.bbox, targets.label)
        landmark = self._landmark_loss(outputs.landmark, targets.landmark)

        total = cls + self.box_weight * bbox + self.landmark_weight * landmark
        stats = {
            "loss": total,
            "cls": cls,
            "bbox": bbox,
            "landmark": landmark,
        }
        self.log_dict(
            {f"train/{k}": v for k, v in stats.items()},
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=images.size(0),
        )
        return total

    def validation_step(self, batch, batch_idx: int):
        images, targets = batch
        outputs = self(images)
        labels, bbox_t, land_t = self._build_targets(targets["boxes"], targets["landmarks"])
        total, stats = self._compute_losses(outputs, labels, bbox_t, land_t)
        self.log_dict(
            {f"val/{k}": v for k, v in stats.items()},
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=images.size(0),
        )
        return total

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.configure_optimizer_params(), lr=self.lr, weight_decay=self.wd)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.epochs)
        return opt, sched, 'epoch'


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
class RetinaFaceConfig(BaseModel):
    data_dir: str = "./data"
    train_annotations: str = "widerface_train.json"
    val_annotations: str = "widerface_val.json"
    export_dir: str = "./ckpt_widerface_retinaface"

    epochs: int = 50
    batch_size: int = 16
    num_workers: int = 4
    lr: float = 1e-3
    wd: float = 1e-4

    image_size: int = 640
    backbone_size: str = Field(default="18", description="ResNet backbone depth (18 or 50)")
    fpn_channels: int = 256
    num_priors: Tuple[int, int, int] = (2, 2, 2)
    scale_op: str = "median"
    small_stem: bool = False

    pos_iou: float = 0.4
    neg_iou: float = 0.2
    variance_xy: float = 0.1
    variance_wh: float = 0.2
    box_weight: float = 1.0
    landmark_weight: float = 0.5

    anchor_sizes: Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]] = (
        (16, 32),
        (64, 128),
        (256, 512),
    )
    steps: Tuple[int, int, int] = (8, 16, 32)
    clip_anchors: bool = False

    amp: bool = False
    cpu: bool = False
    gpus: int = 1
    strategy: str = "auto"
    seed: int = 42

    @field_validator("backbone_size")
    def _validate_backbone(cls, value: str) -> str:
        if value not in ("18", "50"):
            raise ValueError("backbone_size must be '18' or '50'")
        return value


def main() -> None:
    parser = ArgumentParser(model=RetinaFaceConfig)
    config = parser.parse_typed_args()

    # dm = RetinaFaceDataModule(
    #     data_dir=config.data_dir,
    #     train_annotations=config.train_annotations,
    #     val_annotations=config.val_annotations,
    #     image_size=config.image_size,
    #     batch_size=config.batch_size,
    #     num_workers=config.num_workers,
    # )
    # model = LitRetinaFace(config)
    # trainer = setup_retinaface_trainer(config)
    # trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    main()

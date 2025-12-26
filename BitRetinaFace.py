import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from pydantic import BaseModel, Field, field_validator
from pydanticV2_argparse import ArgumentParser
import torch

from dataset import DataModuleConfig, RetinaFaceTensor
from trainer import AccelTrainer, LitBit, Metrics

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
    return ActModels.SiLU(inplace=inplace).build()

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
    act = ActModels.SiLU(inplace=True) if act else None
    norm = NormModels.BatchNorm2d(num_features=-1) if norm else None
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

    def forward(self, x: torch.Tensor) -> List[RetinaFaceTensor]:
        feats = self.fpn(self.backbone(x))
        feats = [m(f) for m, f in zip(self.ssh, feats)]
        imgs = x
        labels = self.classification_head(feats)
        bboxs = self.bbox_head(feats)
        landmarks = self.landmark_head(feats)
        return [
            RetinaFaceTensor(img=img, label=label, bbox=bbox, landmark=landmark,
                ) for img, label, bbox, landmark in zip(imgs, labels, bboxs, landmarks)
        ]
    
        # return (self.classification_head(feats),
        #         self.bbox_head(feats),
        #         self.landmark_head(feats))
        # return {
        #     "cls": self.classification_head(feats),
        #     "bbox": self.bbox_head(feats),
        #     "landmark": self.landmark_head(feats),
        # }

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


def make_resnet50_retinaface_teacher(device: str = "cpu"):
    model = torch.hub.load("qinhy/Pytorch_Retinaface", "retinaface_resnet50", pretrained=True)
    return model.eval().to(device)

# -----------------------------------------------------------------------------
# LightningModule
# -----------------------------------------------------------------------------
class LitRetinaFace(LitBit):
    def __init__(self, config: "RetinaFaceConfig") -> None:
        cfg = config.model_dump() if isinstance(config, BaseModel) else dict(config)
        if cfg.get("dataset") is None:
            cfg["dataset"] = DataModuleConfig(
                dataset_name="retinaface",
                data_dir=cfg.get("data_dir", "./data"),
                batch_size=cfg.get("batch_size", 16),
                num_workers=cfg.get("num_workers", 0),
                mixup=False,
                cutmix=False,
            )
        super().__init__(cfg)
        self.hparams = cfg
        self.model = BitRetinaFace(
            backbone_size=config.backbone_size,
            fpn_channels=config.fpn_channels,
            num_priors=config.num_priors,
            scale_op=config.scale_op,
            in_ch=3,
            small_stem=config.small_stem,
        )
        self.student:BitRetinaFace = self.model  # for compatibility with ternary export helper
        self.teacher = make_resnet50_retinaface_teacher()
        self.hint_points = [
            ("landmark_head","LandmarkHead"),
            ("bbox_head","BboxHead"),
            ("classification_head","ClassHead"),
            ("ssh.0","ssh1"),
            ("ssh.1","ssh2"),
            ("ssh.2","ssh3"),
            ("fpn","fpn"),
            ("backbone.layer1","body.layer1"),
            ("backbone.layer2","body.layer2"),
            ("backbone.layer3","body.layer3"),
            ("backbone.layer4","body.layer4"),
        ]
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
    
    def build_targets_by_anchors(self, targets: List[RetinaFaceTensor]) -> List[RetinaFaceTensor]:
        device = self.anchors.device
        out: List[RetinaFaceTensor] = []

        for t in targets:
            boxes = t.bbox.to(device)
            lands = t.landmark.to(device).view(-1, 5, 2)

            l, b, ld = match_anchors(
                self.anchors,
                boxes,
                lands,
                self.pos_iou,
                self.neg_iou,
                self.variances,
            )
            out.append(RetinaFaceTensor(img=t.img, label=l, bbox=b, landmark=ld))

        return out


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
            return torch.tensor(0.0, device=cls_logits.device)

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
            return torch.tensor(0.0, device=bbox_preds.device)

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
            return torch.tensor(0.0, device=land_preds.device)

        denom = valid_landmarks.sum().clamp(min=1)
        return (
            F.smooth_l1_loss(
                land_preds[valid_landmarks],
                target_landmarks[valid_landmarks],
                reduction="sum",
            )
            / denom
        )
    
    def _kd_loss(self, outputs, targets):
        pass

    def _hint_loss(self, outputs, targets):
        pass

    def _compute_losses(self, outputs: List[RetinaFaceTensor], targets: List[RetinaFaceTensor]):
        anchor_targets = self.build_targets_by_anchors(targets)  # now it's a list

        out_label  = torch.stack([o.label for o in outputs], dim=0)      # [B, N, 2]
        out_bbox   = torch.stack([o.bbox for o in outputs], dim=0)       # [B, N, 4]
        out_land   = torch.stack([o.landmark for o in outputs], dim=0)   # [B, N, 10]

        tgt_label  = torch.stack([t.label for t in anchor_targets], dim=0)     # [B, N]
        tgt_bbox   = torch.stack([t.bbox for t in anchor_targets], dim=0)      # [B, N, 4]
        tgt_land   = torch.stack([t.landmark for t in anchor_targets], dim=0)  # [B, N, 10]

        cls_loss = self._cls_loss(out_label, tgt_label)
        bbox_loss = self._bbox_loss(out_bbox, tgt_bbox, tgt_label)
        landmark_loss = self._landmark_loss(out_land, tgt_land)

        total = cls_loss + self.box_weight * bbox_loss + self.landmark_weight * landmark_loss
        stats = {"loss": total, "ce": cls_loss, "bb": bbox_loss, "lm": landmark_loss}
        return total, stats
    
    def _step_one(self, batch, batch_idx: int):
        images, targets = batch
        targets:List[RetinaFaceTensor] = [t.as_tensor() for t in targets]

        # if need
        # t = targets[0] 
        # if t.bbox.numel():
        #     assert 0.0 <= t.bbox.min() and t.bbox.max() <= 1.01
        # lm = t.landmark
        # valid = (lm >= 0)
        # if valid.any():
        #     assert lm[valid].max() <= 1.01

        outputs:List[RetinaFaceTensor] = self.student(images)
        total, stats = self._compute_losses(outputs, targets)
        return Metrics(loss=total, metrics=stats)

    def training_step(self, batch, batch_idx: int):return self._step_one(batch, batch_idx)

    def validation_step(self, batch, batch_idx: int):return self._step_one(batch, batch_idx)

    def configure_optimizers(self, trainer: 'AccelTrainer'=None):
        opt = torch.optim.AdamW(self.configure_optimizer_params(), lr=self.lr, weight_decay=self.wd)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.epochs)
        return opt, sched, "epoch"


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
    backbone_size: str = Field(default="50", description="ResNet backbone depth (18 or 50)")
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
    dm = DataModuleConfig(
        dataset_name="retinaface",
        data_dir=config.data_dir,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        mixup=False,        
        cutmix=False,)
    lit = LitRetinaFace(config)
    dm = dm.build()    
    trainer = AccelTrainer(
        max_epochs=config.epochs,
        mixed_precision="bf16" if config.amp else "no",
        gradient_accumulation_steps=1,
        log_every_n_steps=10,
    )
    trainer.fit(lit, datamodule=dm)


if __name__ == "__main__":
    # main()

    config = RetinaFaceConfig(batch_size=4)
    model = LitRetinaFace(config)
    # ds = DataModuleConfig(
    #     dataset_name="retinaface",
    #     data_dir=config.data_dir,
    #     batch_size=config.batch_size,
    #     num_workers=config.num_workers,
    #     mixup=False,
    #     cutmix=False,
    # ).build()
    # ds.setup()
    # loader = ds.train_dataloader()
    # batch = next(iter(loader))
    # metrics = model.training_step(batch, 0)
    # print(metrics.to_dict())

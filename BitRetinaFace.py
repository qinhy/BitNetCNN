import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

from pydantic import BaseModel, Field, field_validator
from pydanticV2_argparse import ArgumentParser
import torch

from common_utils import summ
from dataset import DataModuleConfig, RetinaFaceTensor
from trainer import AccelTrainer, CommonTrainConfig, LitBit, LitBitConfig, Metrics

import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import nms, box_iou

from bitlayers.convs import ActModels, Conv2dModels, NormModels
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

    def forward(self, x: torch.Tensor):
        feats = self.fpn(self.backbone(x))
        feats = [m(f) for m, f in zip(self.ssh, feats)]
        # imgs = x
        labels:torch.Tensor = self.classification_head(feats)
        bboxes:torch.Tensor = self.bbox_head(feats)
        landmarks:torch.Tensor = self.landmark_head(feats)
        # return RetinaFaceTensors(imgs=imgs, labels=labels, bboxes=bboxes, landmarks=landmarks)
        # return [
        #     RetinaFaceTensor(img=img, label=label, bbox=bbox, landmark=landmark,
        #         ) for img, label, bbox, landmark in zip(imgs, labels, bboxes, landmarks)
        # ]    
        return (bboxes,labels,landmarks)
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

    # decode_anchors    
    cx, cy, w, h = anchors.unbind(dim=1)
    anchor_boxes = torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=1)
    
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

def decode_boxes_from_deltas(
    anchors_cxcywh: torch.Tensor,  # [N,4] (cx,cy,w,h) normalized
    deltas: torch.Tensor,          # [N,4] (dx,dy,dw,dh)
    variances=(0.1, 0.2),
) -> torch.Tensor:
    v0, v1 = variances
    cx, cy, w, h = anchors_cxcywh.unbind(dim=1)
    dx, dy, dw, dh = deltas.unbind(dim=1)

    gx = cx + dx * v0 * w
    gy = cy + dy * v0 * h
    gw = w * torch.exp(dw * v1)
    gh = h * torch.exp(dh * v1)

    x1 = gx - gw / 2
    y1 = gy - gh / 2
    x2 = gx + gw / 2
    y2 = gy + gh / 2
    boxes = torch.stack([x1, y1, x2, y2], dim=1)
    return boxes

def detect_one_image(
    anchors: torch.Tensor,       # [N,4] cxcywh
    bbox_deltas: torch.Tensor,   # [N,4]
    cls_logits: torch.Tensor,    # [N,2]
    variances=(0.1, 0.2),
    score_thr=0.02,
    nms_iou=0.4,
    topk=5000,
):
    # face prob = softmax(...)[...,1]
    scores = cls_logits.softmax(dim=-1)[:, 1]

    boxes = decode_boxes_from_deltas(anchors, bbox_deltas, variances=variances)
    boxes = boxes.clamp(0.0, 1.0)

    keep = scores > score_thr
    boxes, scores = boxes[keep], scores[keep]
    if boxes.numel() == 0:
        return boxes, scores

    # optional topk before NMS (speed)
    if scores.numel() > topk:
        idx = scores.topk(topk).indices
        boxes, scores = boxes[idx], scores[idx]

    keep = nms(boxes, scores, nms_iou)
    return boxes[keep], scores[keep]

def compute_ap_voc_integral(prec: torch.Tensor, rec: torch.Tensor) -> float:
    # precision envelope + integration
    mpre = torch.cat([torch.tensor([0.0], device=prec.device), prec, torch.tensor([0.0], device=prec.device)])
    mrec = torch.cat([torch.tensor([0.0], device=rec.device),  rec,  torch.tensor([1.0], device=rec.device)])

    for i in range(mpre.numel() - 1, 0, -1):
        mpre[i - 1] = torch.maximum(mpre[i - 1], mpre[i])

    idx = (mrec[1:] != mrec[:-1]).nonzero(as_tuple=False).squeeze(1)
    ap = torch.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]).item()
    return ap

def ap_at_iou(
    preds,  # list of dicts: {"boxes": [Mi,4], "scores":[Mi]}
    gts,    # list of GT boxes: [Ki,4]
    iou_thr=0.5,
) -> float:
    device = preds[0]["boxes"].device if len(preds) else torch.device("cpu")

    # total GT faces
    total_gt = sum(int(gt.size(0)) for gt in gts)
    if total_gt == 0:
        return 0.0

    # flatten detections into (score, img_idx, box)
    det_scores = []
    det_img = []
    det_boxes = []
    for i, p in enumerate(preds):
        if p["boxes"].numel() == 0:
            continue
        det_scores.append(p["scores"])
        det_img.append(torch.full((p["scores"].numel(),), i, device=device, dtype=torch.long))
        det_boxes.append(p["boxes"])

    if len(det_scores) == 0:
        return 0.0

    det_scores = torch.cat(det_scores, dim=0)
    det_img    = torch.cat(det_img, dim=0)
    det_boxes  = torch.cat(det_boxes, dim=0)

    # sort descending by confidence
    order = det_scores.argsort(descending=True)
    det_scores = det_scores[order]
    det_img    = det_img[order]
    det_boxes  = det_boxes[order]

    # per-image matched GT bookkeeping
    matched = [torch.zeros((gt.size(0),), dtype=torch.bool, device=device) for gt in gts]

    tp = torch.zeros((det_scores.numel(),), device=device)
    fp = torch.zeros((det_scores.numel(),), device=device)

    for d in range(det_scores.numel()):
        i = int(det_img[d].item())
        gt_boxes = gts[i]
        if gt_boxes.numel() == 0:
            fp[d] = 1.0
            continue

        ious = box_iou(det_boxes[d].unsqueeze(0), gt_boxes).squeeze(0)  # [Ki]
        best_iou, best_j = ious.max(dim=0)

        if best_iou >= iou_thr and not matched[i][best_j]:
            tp[d] = 1.0
            matched[i][best_j] = True
        else:
            fp[d] = 1.0

    tp_cum = torch.cumsum(tp, dim=0)
    fp_cum = torch.cumsum(fp, dim=0)

    rec = tp_cum / max(total_gt, 1)
    prec = tp_cum / torch.clamp(tp_cum + fp_cum, min=1e-12)

    return compute_ap_voc_integral(prec, rec)

# -----------------------------------------------------------------------------
# LightningModule
# -----------------------------------------------------------------------------
class LitRetinaFace(LitBit):
    def __init__(self, config: "RetinaFaceConfig") -> None:
        super().__init__(config)
        self.hparams = config.model_dump()
        self.input_size = config.image_size
        self.variances = (config.variance_xy, config.variance_wh)
        self.pos_iou = config.pos_iou
        self.neg_iou = config.neg_iou
        self.box_weight = config.box_weight
        self.landmark_weight = config.landmark_weight

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

        # after model + anchors exist, sanity check once
        with torch.no_grad():
            x = torch.randn(1, 3, self.input_size, self.input_size, device="cpu")
            out_bbox, out_cls, out_land = self.student(x)
            assert out_bbox.shape[1] == self.anchors.shape[0], (out_bbox.shape, self.anchors.shape)

            tb, tc, tl = self.teacher_forward(x)  # whatever this returns
            assert tb.shape[1] == self.anchors.shape[0], (tb.shape, self.anchors.shape)

    
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
        target_labels: torch.Tensor, neg_pos_ratio=3) -> torch.Tensor:
        labels = target_labels.view(-1)
        logits = cls_logits.view(-1, cls_logits.size(-1))

        pos = labels == 1
        neg = labels == 0

        if pos.sum() == 0:
            # fallback: just background CE
            return F.cross_entropy(logits[neg], labels[neg])

        # per-anchor CE (no reduction)
        ce = F.cross_entropy(logits, labels.clamp(min=0), reduction="none")  # clamp for ignore
        ce[pos == 0] = ce[pos == 0]  # keep
        ce[labels < 0] = 0           # ignore

        num_pos = pos.sum()
        num_neg = torch.clamp(neg_pos_ratio * num_pos, max=neg.sum())

        # pick hardest negatives
        neg_ce = ce.clone()
        neg_ce[~neg] = -1
        _, idx = neg_ce.sort(descending=True)
        hard_neg = torch.zeros_like(neg, dtype=torch.bool)
        hard_neg[idx[:num_neg]] = True

        keep = pos | hard_neg
        return F.cross_entropy(logits[keep], labels[keep])


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
    
    def _compute_losses(self, x, outputs, targets: List[RetinaFaceTensor]):
        anchor_targets = self.build_targets_by_anchors(targets)  # now it's a list
        out_bbox,out_labe,out_land = outputs
        # out_labe [B, N, 2]
        # out_bbox [B, N, 4]
        # out_land [B, N, 10]

        tgt_label  = torch.stack([t.label for t in anchor_targets], dim=0)     # [B, N]
        tgt_bbox   = torch.stack([t.bbox for t in anchor_targets], dim=0)      # [B, N, 4]
        tgt_land   = torch.stack([t.landmark for t in anchor_targets], dim=0)  # [B, N, 10]

        cls_loss = self._cls_loss(out_labe, tgt_label)
        bbox_loss = self.box_weight * self._bbox_loss(out_bbox, tgt_bbox, tgt_label)
        landmark_loss = self.landmark_weight * self._landmark_loss(out_land, tgt_land)

        total = cls_loss + bbox_loss + landmark_loss
        stats = { "ce": cls_loss, "bb": bbox_loss, "lm": landmark_loss}

        if self.kd or self.hint:
            bbox_regressions, classifications, ldm_regressions = self.teacher_forward(x)
        if self.kd:
            loss_kd = self.alpha_kd * (
                        self.kd(out_labe.float(), classifications.float()
                    ) + self.kd(out_bbox.float(), bbox_regressions.float()
                    ) + self.kd(out_land.float(), ldm_regressions.float()))
            
            stats["kd"] = loss_kd
            total = (1.0 - self.alpha_kd) * total + loss_kd
        if self.hint:
            loss_hint = self.alpha_hint * self.get_loss_hint()
            stats["hint"] = loss_hint
            total += loss_hint            
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

        bboxes,labels,landmarks = self.student(images)
        total, stats = self._compute_losses(images, (bboxes,labels,landmarks), targets)
        return Metrics(loss=total, metrics=stats)

    def training_step(self, batch, batch_idx: int):return self._step_one(batch, batch_idx)


    def on_validation_epoch_start(self):
        self._ap_preds = []
        self._ap_gts = []

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        targets = [t.as_tensor() for t in targets]

        bboxes, labels, landmarks = self.student(images)

        # store detections + gts for AP
        anchors = self.anchors.to(images.device)  # [N,4]
        for i in range(images.size(0)):
            boxes_i, scores_i = detect_one_image(
                anchors=anchors,
                bbox_deltas=bboxes[i],
                cls_logits=labels[i],
                variances=self.variances,
                score_thr=0.02,
                nms_iou=0.4,
            )
            self._ap_preds.append({"boxes": boxes_i.detach(), "scores": scores_i.detach()})
            self._ap_gts.append(targets[i].bbox.to(images.device).detach())

        # keep your existing loss logging
        total, stats = self._compute_losses(images, (bboxes, labels, landmarks), targets)
        return Metrics(loss=total, metrics=stats)

    def on_validation_epoch_end(self):
        ap50 = ap_at_iou(self._ap_preds, self._ap_gts, iou_thr=0.5)
        # log however your framework logs
        # self.log("val/AP50", ap50, prog_bar=True)

    def configure_optimizers(self, trainer: 'AccelTrainer'=None):
        opt = torch.optim.AdamW(self.configure_optimizer_params(), lr=self.lr, weight_decay=self.wd)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.epochs)
        return opt, sched, "epoch"


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
class RetinaFaceConfig(CommonTrainConfig,LitBitConfig):
    dataset_name:str = "retinaface"
    data_dir: str = "./data"
    train_annotations: str = "widerface_train.json"
    val_annotations: str = "widerface_val.json"
    export_dir: str = "./ckpt_widerface_retinaface"
    dataset: Optional[Dict] = None

    epochs: int = 50
    batch_size: int = 4
    num_workers: int = 4
    mixup:bool=False
    cutmix:bool=False
    
    lr: float = 1e-3
    alpha_kd: float = 0.9
    alpha_hint: float = 0.1
    box_weight: float = 1.0
    landmark_weight: float = 0.001
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
    dm = DataModuleConfig.model_validate(config.model_dump())
    config = RetinaFaceConfig.model_validate(config.model_dump())
    config.dataset = dm.model_dump()
    
    config.student = BitRetinaFace(
        backbone_size=config.backbone_size,
        fpn_channels=config.fpn_channels,
        num_priors=config.num_priors,
        scale_op=config.scale_op,
        in_ch=3,
        small_stem=config.small_stem,
    )
    config.teacher = make_resnet50_retinaface_teacher()
    config.hint_points = [
        ("ssh.0","ssh1"),
        ("ssh.1","ssh2"),
        ("ssh.2","ssh3"),
        ("backbone.layer1","body.layer1"),
        ("backbone.layer2","body.layer2"),
        ("backbone.layer3","body.layer3"),
        ("backbone.layer4","body.layer4"),
    ]
    lit = LitRetinaFace(config)
    trainer = AccelTrainer(
        max_epochs=config.epochs,
        mixed_precision="bf16" if config.amp else "no",
        gradient_accumulation_steps=1,
        log_every_n_steps=10,
    )
    trainer.fit(lit, datamodule=dm.build())


if __name__ == "__main__":
    main()

    # config = RetinaFaceConfig(batch_size=4)
    # dm = DataModuleConfig.model_validate(config.model_dump())
    # config = RetinaFaceConfig.model_validate(config.model_dump())
    # config.dataset = dm.model_copy()
    
    # config.student = BitRetinaFace(
    #     backbone_size=config.backbone_size,
    #     fpn_channels=config.fpn_channels,
    #     num_priors=config.num_priors,
    #     scale_op=config.scale_op,
    #     in_ch=3,
    #     small_stem=config.small_stem,
    # )
    # config.teacher = make_resnet50_retinaface_teacher()
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

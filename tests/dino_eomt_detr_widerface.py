import copy
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from pydantic import Field, field_validator
from pydanticV2_argparse import ArgumentParser

from bitlayers.dinov3.layers.bitlayers import Linear as BitLinear
from bitlayers.dinov3.models.vision_transformer import DinoVisionTransformer, vit_base, vit_large, vit_small
from dataset import DataModuleConfig, RetinaFaceDataModule, RetinaFaceTensor
from trainer import (
    AccelTrainer,
    CommonTrainConfig,
    ExportBestTernary,
    LitBit,
    LitBitConfig,
    Metrics,
    MetricsManager,
    MetricsTracer,
)


# ----------------------------
# Box utils (DETR-style)
# ----------------------------
def box_cxcywh_to_xyxy(x: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = x.unbind(-1)
    return torch.stack([(cx - 0.5 * w), (cy - 0.5 * h), (cx + 0.5 * w), (cy + 0.5 * h)], dim=-1)


def box_xyxy_to_cxcywh(x: torch.Tensor) -> torch.Tensor:
    x0, y0, x1, y1 = x.unbind(-1)
    return torch.stack([(x0 + x1) * 0.5, (y0 + y1) * 0.5, (x1 - x0), (y1 - y0)], dim=-1)


def box_area_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    wh = (boxes[..., 2:] - boxes[..., :2]).clamp(min=0)
    return wh[..., 0] * wh[..., 1]


def box_iou_xyxy(boxes1: torch.Tensor, boxes2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    area1 = box_area_xyxy(boxes1)
    area2 = box_area_xyxy(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    union = area1[:, None] + area2[None, :] - inter
    iou = inter / union.clamp(min=1e-6)
    return iou, union


def generalized_box_iou_xyxy(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    iou, union = box_iou_xyxy(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    area_c = wh[..., 0] * wh[..., 1]

    return iou - (area_c - union) / area_c.clamp(min=1e-6)


# ----------------------------
# Hungarian matching
# ----------------------------
def _try_scipy_linear_sum_assignment(cost: torch.Tensor):
    try:
        from scipy.optimize import linear_sum_assignment  # type: ignore
    except Exception:
        return None

    cost_np = cost.detach().cpu().numpy()
    r, c = linear_sum_assignment(cost_np)
    return (
        torch.as_tensor(r, dtype=torch.int64, device=cost.device),
        torch.as_tensor(c, dtype=torch.int64, device=cost.device),
    )


def hungarian_fallback(cost: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # Pure-Python Hungarian fallback (O(n^3))
    n_rows, n_cols = cost.shape
    n = max(n_rows, n_cols)
    c = cost.detach().cpu()
    pad_val = float(c.max().item() + 1.0) if c.numel() > 0 else 1.0
    C = torch.full((n, n), pad_val, dtype=torch.float64)
    C[:n_rows, :n_cols] = c.to(torch.float64)

    u = torch.zeros(n + 1, dtype=torch.float64)
    v = torch.zeros(n + 1, dtype=torch.float64)
    p = torch.zeros(n + 1, dtype=torch.int64)
    way = torch.zeros(n + 1, dtype=torch.int64)

    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = torch.full((n + 1,), float("inf"), dtype=torch.float64)
        used = torch.zeros(n + 1, dtype=torch.bool)
        while True:
            used[j0] = True
            i0 = p[j0].item()
            delta = float("inf")
            j1 = 0
            for j in range(1, n + 1):
                if not used[j]:
                    cur = C[i0 - 1, j - 1].item() - u[i0].item() - v[j].item()
                    if cur < minv[j].item():
                        minv[j] = cur
                        way[j] = j0
                    if minv[j].item() < delta:
                        delta = minv[j].item()
                        j1 = j
            for j in range(0, n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0].item() == 0:
                break

        while True:
            j1 = way[j0].item()
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    assignment = torch.zeros(n + 1, dtype=torch.int64)
    for j in range(1, n + 1):
        assignment[p[j].item()] = j

    row_ind, col_ind = [], []
    for i in range(1, n_rows + 1):
        j = assignment[i].item()
        if 1 <= j <= n_cols:
            row_ind.append(i - 1)
            col_ind.append(j - 1)

    return (
        torch.tensor(row_ind, dtype=torch.int64, device=cost.device),
        torch.tensor(col_ind, dtype=torch.int64, device=cost.device),
    )


def linear_sum_assignment_torch(cost: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    out = _try_scipy_linear_sum_assignment(cost)
    if out is not None:
        return out
    return hungarian_fallback(cost)


# ----------------------------
# Model components
# ----------------------------
class MLP(nn.Module):
    def __init__(self, d_in: int, d_hid: int, d_out: int, num_layers: int, scale_op: str = "median"):
        super().__init__()
        assert num_layers >= 1
        layers: List[nn.Module] = []
        for i in range(num_layers):
            a = d_in if i == 0 else d_hid
            b = d_out if i == num_layers - 1 else d_hid
            layers.append(BitLinear(a, b, bias=True, scale_op=scale_op))
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = F.gelu(x)
        return x


class DinoEoMTDETR(nn.Module):
    """
    EoMT-style DETR detector (bbox-only):
      - Tokens from backbone (prefix + patch tokens)
      - Insert Q learnable queries before last `num_blocks` blocks
      - Run last blocks jointly over [Q | prefix | patches]
      - Predict cls + bbox from query tokens only
    """

    def __init__(
        self,
        encoder: Union[nn.Module, DinoVisionTransformer],
        num_classes: int = 1,
        num_queries: int = 200,
        num_blocks: int = 4,
        aux_loss: bool = True,
        freeze_backbone: bool = True,
        scale_op: str = "median",
    ):
        super().__init__()
        self.encoder = encoder
        self.backbone: DinoVisionTransformer = encoder.backbone if hasattr(encoder, "backbone") else encoder  # type: ignore

        self.num_classes = int(num_classes)
        self.num_queries = int(num_queries)
        self.num_blocks = int(num_blocks)
        self.aux_loss = bool(aux_loss)
        self.freeze_backbone = bool(freeze_backbone)

        if self.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad_(False)
            self.backbone.eval()

        embed_dim = getattr(self.backbone, "embed_dim", None) or getattr(self.backbone, "num_features", None)
        if embed_dim is None:
            raise AttributeError("Backbone has no embed_dim/num_features")

        self.query_embed = nn.Embedding(self.num_queries, embed_dim)
        self.class_head = BitLinear(embed_dim, self.num_classes + 1, bias=True, scale_op=scale_op)
        self.box_head = MLP(embed_dim, embed_dim, 4, num_layers=3, scale_op=scale_op)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def _prepare_tokens(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        if hasattr(self.backbone, "prepare_tokens_with_masks"):
            tokens, (h_p, w_p) = self.backbone.prepare_tokens_with_masks(x, masks=None)
            return tokens, (h_p, w_p)

        t = self.backbone.patch_embed(x)
        if t.dim() == 4:
            _, h_p, w_p, _ = t.shape
            t = t.flatten(1, 2)
        else:
            _, n, _ = t.shape
            h_p = w_p = int(math.sqrt(n))
        return t, (h_p, w_p)

    def _rope(self, h_p: int, w_p: int) -> Optional[torch.Tensor]:
        if hasattr(self.backbone, "rope_embed") and self.backbone.rope_embed is not None:
            return self.backbone.rope_embed(H=h_p, W=w_p)
        return None

    def _call_block(self, block: nn.Module, tokens: torch.Tensor, rope: Optional[torch.Tensor]) -> torch.Tensor:
        if rope is not None:
            try:
                return block(tokens, rope=rope)  # type: ignore
            except TypeError:
                try:
                    return block(tokens, rope)  # type: ignore
                except TypeError:
                    return block(tokens)
        return block(tokens)

    def _predict_from_queries(self, tokens: torch.Tensor) -> Dict[str, torch.Tensor]:
        x_norm = self.backbone.norm(tokens) if hasattr(self.backbone, "norm") else tokens
        q = x_norm[:, : self.num_queries, :]
        pred_logits = self.class_head(q)
        pred_boxes = self.box_head(q).sigmoid()
        return {"pred_logits": pred_logits, "pred_boxes": pred_boxes}

    def forward(self, x: torch.Tensor) -> Dict[str, Any]:
        tokens, (h_p, w_p) = self._prepare_tokens(x)
        rope = self._rope(h_p, w_p)

        blocks = self.backbone.blocks
        insert_at = len(blocks) - self.num_blocks
        if insert_at < 0:
            raise ValueError(f"num_blocks={self.num_blocks} > len(backbone.blocks)={len(blocks)}")

        aux: List[Dict[str, torch.Tensor]] = []

        # --- run early blocks without grad if backbone is frozen ---
        if self.freeze_backbone and insert_at > 0:
            with torch.no_grad():
                for i in range(insert_at):
                    tokens = self._call_block(blocks[i], tokens, rope)
            tokens = tokens.detach()
        else:
            for i in range(insert_at):
                tokens = self._call_block(blocks[i], tokens, rope)

        # insert queries
        q = self.query_embed.weight[None, :, :].expand(tokens.shape[0], -1, -1)
        tokens = torch.cat((q, tokens), dim=1)

        # last blocks (need grad for query_embed + heads)
        for i in range(insert_at, len(blocks)):
            tokens = self._call_block(blocks[i], tokens, rope)

            # aux losses for intermediate layers only (exclude final)
            if self.aux_loss and i < len(blocks) - 1:
                aux.append(self._predict_from_queries(tokens))

        out = self._predict_from_queries(tokens)
        if self.aux_loss:
            out["aux_outputs"] = aux
        return out
    
    def clone(self):
        return copy.deepcopy(self)

    @torch.no_grad()
    def show_predict_examples(self, images: torch.Tensor, score_thr: float = 0.5, topk: int = 200):
        self.eval()
        outputs = self(images)
        logits = outputs["pred_logits"]
        boxes = box_cxcywh_to_xyxy(outputs["pred_boxes"]).clamp(0.0, 1.0)
        probs = logits.softmax(-1)[..., : self.num_classes]
        scores, labels = probs.max(-1)

        preds: List[Dict[str, torch.Tensor]] = []
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        for i in range(images.size(0)):
            keep = scores[i] > score_thr
            b = boxes[i][keep]
            s = scores[i][keep]
            l = labels[i][keep]
            if s.numel() > topk:
                k = torch.topk(s, topk).indices
                b = b[k]
                s = s[k]
                l = l[k]

            y = RetinaFaceTensor(img=images[i], bbox=b, landmark=torch.zeros((0, 2), device=b.device))
            y.show(mean, std)
            preds.append({"boxes": b, "scores": s, "labels": l})
        return preds


# ----------------------------
# Matcher & Criterion (DETR)
# ----------------------------
@dataclass
class MatcherConfig:
    cost_class: float = 1.0
    cost_bbox: float = 5.0
    cost_giou: float = 2.0


class HungarianMatcher(nn.Module):
    def __init__(self, cfg: MatcherConfig):
        super().__init__()
        self.cfg = cfg

    @torch.no_grad()
    def forward(self, outputs: Dict[str, torch.Tensor], targets: List[Dict[str, torch.Tensor]]):
        out_prob = outputs["pred_logits"].softmax(-1)
        out_bbox = outputs["pred_boxes"]

        indices: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for b in range(out_bbox.shape[0]):
            tgt_ids = targets[b]["labels"]
            tgt_bbox = targets[b]["boxes"]
            if tgt_bbox.numel() == 0:
                indices.append(
                    (
                        torch.empty(0, dtype=torch.int64, device=out_bbox.device),
                        torch.empty(0, dtype=torch.int64, device=out_bbox.device),
                    )
                )
                continue

            cost_class = -out_prob[b, :, tgt_ids]
            cost_bbox = torch.cdist(out_bbox[b], tgt_bbox, p=1)
            cost_giou = -generalized_box_iou_xyxy(box_cxcywh_to_xyxy(out_bbox[b]), box_cxcywh_to_xyxy(tgt_bbox))

            C = self.cfg.cost_class * cost_class + self.cfg.cost_bbox * cost_bbox + self.cfg.cost_giou * cost_giou
            row_ind, col_ind = linear_sum_assignment_torch(C)
            indices.append((row_ind, col_ind))
        return indices


@dataclass
class CriterionConfig:
    eos_coef: float = 0.1
    loss_ce: float = 1.0
    loss_bbox: float = 5.0
    loss_giou: float = 2.0


class DETRCriterion(nn.Module):
    def __init__(self, num_classes: int, matcher: HungarianMatcher, cfg: CriterionConfig):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.cfg = cfg

        empty_weight = torch.ones(num_classes + 1)
        empty_weight[-1] = cfg.eos_coef
        self.register_buffer("empty_weight", empty_weight)

    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _num_boxes(self, targets: List[Dict[str, torch.Tensor]]) -> torch.Tensor:
        n = sum(t["boxes"].shape[0] for t in targets)
        return torch.as_tensor([n], dtype=torch.float, device=self.empty_weight.device)

    def loss_labels(self, outputs, targets, indices) -> torch.Tensor:
        src_logits = outputs["pred_logits"]
        B, Q, _ = src_logits.shape
        target_classes = torch.full((B, Q), self.num_classes, dtype=torch.int64, device=src_logits.device)
        for b, (src_idx, tgt_idx) in enumerate(indices):
            if src_idx.numel() > 0:
                target_classes[b, src_idx] = targets[b]["labels"][tgt_idx]
        return F.cross_entropy(src_logits.transpose(1, 2), target_classes, weight=self.empty_weight)

    def loss_boxes(self, outputs, targets, indices) -> Tuple[torch.Tensor, torch.Tensor]:
        src_boxes = outputs["pred_boxes"]
        batch_idx, src_idx = self._get_src_permutation_idx(indices)
        if src_idx.numel() == 0:
            z = torch.as_tensor(0.0, device=src_boxes.device)
            return z, z

        src = src_boxes[batch_idx, src_idx]
        tgt = torch.cat([t["boxes"][j] for t, (_, j) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src, tgt, reduction="none").sum()
        loss_giou = (1.0 - torch.diag(generalized_box_iou_xyxy(box_cxcywh_to_xyxy(src), box_cxcywh_to_xyxy(tgt)))).sum()
        return loss_bbox, loss_giou

    def forward(self, outputs: Dict[str, Any], targets: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        indices = self.matcher(outputs, targets)
        num_boxes = self._num_boxes(targets).clamp(min=1.0)

        loss_ce = self.loss_labels(outputs, targets, indices)
        loss_bbox, loss_giou = self.loss_boxes(outputs, targets, indices)
        losses = {
            "loss_ce": loss_ce * self.cfg.loss_ce,
            "loss_bbox": (loss_bbox / num_boxes) * self.cfg.loss_bbox,
            "loss_giou": (loss_giou / num_boxes) * self.cfg.loss_giou,
        }

        if "aux_outputs" in outputs:
            for i, aux in enumerate(outputs["aux_outputs"]):
                idx = self.matcher(aux, targets)
                l_ce = self.loss_labels(aux, targets, idx)
                l_bbox, l_giou = self.loss_boxes(aux, targets, idx)
                aux_w = 0.1  # try 0.1 first
                losses[f"loss_ce_{i}"]   = aux_w * (l_ce  * self.cfg.loss_ce)
                losses[f"loss_bbox_{i}"] = aux_w * ((l_bbox / num_boxes) * self.cfg.loss_bbox)
                losses[f"loss_giou_{i}"] = aux_w * ((l_giou / num_boxes) * self.cfg.loss_giou)

        return losses


# ----------------------------
# Postprocess (inference)
# ----------------------------
@torch.no_grad()
def postprocess_detr(
    outputs: Dict[str, torch.Tensor],
    orig_sizes: torch.Tensor,
    score_thresh: float = 0.3,
    topk: int = 300,
) -> List[Dict[str, torch.Tensor]]:
    logits = outputs["pred_logits"]
    boxes = outputs["pred_boxes"]
    B, _, Cp1 = logits.shape
    C = Cp1 - 1

    prob = logits.softmax(-1)[:, :, :C]
    scores, labels = prob.max(-1)

    results: List[Dict[str, torch.Tensor]] = []
    for b in range(B):
        s = scores[b]
        l = labels[b]
        bcxcywh = boxes[b]

        keep = s > score_thresh
        s = s[keep]
        l = l[keep]
        bcxcywh = bcxcywh[keep]

        if s.numel() > topk:
            top_idx = torch.topk(s, topk).indices
            s = s[top_idx]
            l = l[top_idx]
            bcxcywh = bcxcywh[top_idx]

        bxyxy = box_cxcywh_to_xyxy(bcxcywh).clamp(0, 1)
        H, W = orig_sizes[b].tolist()
        scale = torch.tensor([W, H, W, H], device=bxyxy.device, dtype=bxyxy.dtype)
        results.append({"boxes": bxyxy * scale, "scores": s, "labels": l})

    return results


# ----------------------------
# Training plumbing (Bit trainer stack)
# ----------------------------
def raw_target_to_detr_target(t, H: int, W: int, device: torch.device):
    bbox = getattr(t, "bbox", None)
    if bbox is None:
        bbox = t["bbox"]

    boxes_xyxy = torch.as_tensor(bbox, dtype=torch.float32, device=device)
    if boxes_xyxy.numel() == 0:
        boxes_xyxy = boxes_xyxy.reshape(0, 4)
    if boxes_xyxy.ndim == 1:
        boxes_xyxy = boxes_xyxy[None, :]

    # Accept either normalized [0,1] or absolute pixel boxes.
    if boxes_xyxy.numel() > 0 and float(boxes_xyxy.max()) <= 1.5:
        scale = torch.tensor([W, H, W, H], dtype=torch.float32, device=device)
        boxes_xyxy = boxes_xyxy * scale

    boxes_xyxy[:, 0::2] = boxes_xyxy[:, 0::2].clamp(0, W)
    boxes_xyxy[:, 1::2] = boxes_xyxy[:, 1::2].clamp(0, H)

    scale = torch.tensor([W, H, W, H], dtype=torch.float32, device=device)
    boxes_cxcywh = box_xyxy_to_cxcywh(boxes_xyxy) / scale
    boxes_cxcywh = boxes_cxcywh.clamp(0, 1)

    keep = (boxes_cxcywh[:, 2] > 1e-4) & (boxes_cxcywh[:, 3] > 1e-4)
    boxes_cxcywh = boxes_cxcywh[keep]
    det_labels = torch.zeros((boxes_cxcywh.shape[0],), dtype=torch.int64, device=device)

    return {"boxes": boxes_cxcywh, "labels": det_labels}


class DinoRetinaFaceConfig(CommonTrainConfig, LitBitConfig):
    dataset_name: str = "retinaface"
    data_dir: str = "./data"
    export_dir: str = "./ckpt_widerface_eomt_detr"
    dataset: Optional[Dict] = None

    model_name: str = Field(default="eomt_detr_face", description="Model family/name identifier.")
    model_size: str = Field(default="base", description="small/base/large")

    epochs: int = 50
    batch_size: int = 16
    num_workers: int = 0

    lr: float = 1e-4
    wd: float = 5e-2
    alpha_kd: float = 0.0
    alpha_hint: float = 0.0

    image_size: int = 640
    num_classes: int = 1
    num_queries: int = 200
    num_blocks: int = 4
    aux_loss: bool = True
    freeze_backbone: bool = True

    cost_class: float = 1.0
    cost_bbox: float = 5.0
    cost_giou: float = 2.0
    loss_ce: float = 1.0
    loss_bbox: float = 5.0
    loss_giou: float = 2.0
    eos_coef: float = 0.1

    score_thr: float = 0.5
    topk: int = 200

    @field_validator("model_size")
    def _validate_model_size(cls, v: str):
        v = str(v).lower()
        if v not in ("small", "base", "large"):
            raise ValueError("model_size must be one of: small, base, large")
        return v


def _build_encoder(model_size: str) -> DinoVisionTransformer:
    builders = {"small": vit_small, "base": vit_base, "large": vit_large}
    weights = {"small": "./data/dinov3_vits16_pretrain_lvd1689m-08c60483.pth",
                "base": "./data/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth",
                "large": "./data/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"}
    
    encoder = builders[model_size]()
    if hasattr(encoder, "init_weights") and weights[model_size] is not None:
        encoder.init_weights()
    if weights[model_size] is not None:
        p = weights[model_size]
        encoder = torch.hub.load(
            "../dinov3", p.split('/')[-1].split('_pretrain')[0],
            source='local', weights=p)        
    return encoder


class LitDinoEoMTDETR(LitBit):
    def __init__(self, config: DinoRetinaFaceConfig):
        super().__init__(config)
        self.hparams_cfg = config

        # This detector path is currently trained without teacher KD/hint.
        self.kd = None
        self.hint = None
        self.teacher = None
        self.has_teacher = False
        self.alpha_kd = 0.0
        self.alpha_hint = 0.0
        self.hint_points = []

        self.matcher = HungarianMatcher(
            MatcherConfig(
                cost_class=config.cost_class,
                cost_bbox=config.cost_bbox,
                cost_giou=config.cost_giou,
            )
        )
        self.criterion = DETRCriterion(
            num_classes=config.num_classes,
            matcher=self.matcher,
            cfg=CriterionConfig(
                eos_coef=config.eos_coef,
                loss_ce=config.loss_ce,
                loss_bbox=config.loss_bbox,
                loss_giou=config.loss_giou,
            ),
        )

    def _build_targets(self, images: torch.Tensor, raw_targets: List[Any]) -> List[Dict[str, torch.Tensor]]:
        _, _, H, W = images.shape
        device = images.device
        return [raw_target_to_detr_target(t, H=H, W=W, device=device) for t in raw_targets]

    @staticmethod
    def _reduce_losses(losses: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if not losses:
            z = torch.tensor(0.0)
            return z, z, z, z, z

        sample = next(iter(losses.values()))
        z = torch.zeros((), device=sample.device, dtype=sample.dtype)

        total = sum(losses.values(), z)
        l_ce = losses.get("loss_ce", z)
        l_bbox = losses.get("loss_bbox", z)
        l_giou = losses.get("loss_giou", z)
        l_aux = total - (l_ce + l_bbox + l_giou)
        return total, l_ce, l_bbox, l_giou, l_aux

    def training_step(self, batch: Tuple[torch.Tensor, List[RetinaFaceTensor]], batch_idx: int) -> Metrics:
        images, raw_targets = batch
        outputs = self.student(images)
        targets = self._build_targets(images, raw_targets)
        losses = self.criterion(outputs, targets)
        loss_total, l_ce, l_bbox, l_giou, l_aux = self._reduce_losses(losses)

        stats = {"ce": l_ce, "bb": l_bbox, "giou": l_giou, "aux": l_aux}
        return Metrics(loss=loss_total, metrics=stats)

    @torch.no_grad()
    def validation_step(self, batch: Tuple[torch.Tensor, List[RetinaFaceTensor]], batch_idx: int) -> Metrics:
        images, raw_targets = batch
        targets = self._build_targets(images, raw_targets)

        outputs_fp = self.student(images)
        losses_fp = self.criterion(outputs_fp, targets)
        fp_total, fp_ce, fp_bbox, fp_giou, fp_aux = self._reduce_losses(losses_fp)

        outputs_tern = self._ternary_snapshot(images)
        losses_tern = self.criterion(outputs_tern, targets)
        t_total, t_ce, t_bbox, t_giou, t_aux = self._reduce_losses(losses_tern)

        metrics: Dict[str, Any] = {
            "val/loss_fp": fp_total,
            "val/loss_tern": t_total,
            "val/ce_fp": fp_ce,
            "val/ce_tern": t_ce,
            "val/bb_fp": fp_bbox,
            "val/bb_tern": t_bbox,
            "val/giou_fp": fp_giou,
            "val/giou_tern": t_giou,
            "val/aux_fp": fp_aux,
            "val/aux_tern": t_aux,
        }
        return Metrics(loss=t_total, metrics=metrics)
    
    def configure_optimizer_params(self, trainer=None):
        params = [p for p in self.student.parameters() if p.requires_grad]
        return params
   
    def configure_optimizers(self, trainer=None):
        opt = torch.optim.AdamW(self.configure_optimizer_params(), lr=self.lr, weight_decay=self.wd)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.epochs)
        return opt, sched, "epoch"

    @torch.no_grad()
    def show_predict_examples(self, images: torch.Tensor, score_thr: Optional[float] = None, topk: Optional[int] = None):
        return self.student.show_predict_examples(
            images,
            score_thr=self.hparams_cfg.score_thr if score_thr is None else score_thr,
            topk=self.hparams_cfg.topk if topk is None else topk,
        )


def main(run_fit: bool = True):
    parser = ArgumentParser(model=DinoRetinaFaceConfig)
    config = parser.parse_typed_args()

    dm_conf = DataModuleConfig.model_validate(config.model_dump())
    config.dataset = dm_conf.model_dump()

    encoder = _build_encoder(config.model_size)
    config.student = DinoEoMTDETR(
        encoder=encoder,
        num_classes=config.num_classes,
        num_queries=config.num_queries,
        num_blocks=config.num_blocks,
        aux_loss=config.aux_loss,
        freeze_backbone=config.freeze_backbone,
        scale_op=config.scale_op,
    )
    config.teacher = None
    config.hint_points = []

    lit = LitDinoEoMTDETR(config)
    trainer = AccelTrainer(
        max_epochs=config.epochs,
        mixed_precision="bf16" if config.amp else "no",
        gradient_accumulation_steps=1,
        log_every_n_steps=10,
        metrics_manager=MetricsManager(
            epoch_metric_tracers=[
                MetricsTracer(
                    stage="val",
                    key="val/loss_tern_mean",
                    mode="min",
                    callback=ExportBestTernary(),
                )
            ]
        ),
    )
    dm = RetinaFaceDataModule(dm_conf)

    if run_fit:
        trainer.fit(lit, datamodule=dm)
    return lit, dm, trainer


if __name__ == "__main__":
    main(run_fit=True)

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from pydantic import BaseModel, Field, field_validator
from pydanticV2_argparse import ArgumentParser
from torchvision.ops import nms, box_iou

# -----------------------------------------------------------------------------
# Imports from your specific environment (BitNet & Trainer Stack)
# -----------------------------------------------------------------------------
from common_utils import summ
from dataset import DataModuleConfig, RetinaFaceDataModule, RetinaFaceTensor, RetinaFaceTensors
from trainer import AccelTrainer, CommonTrainConfig, LitBit, LitBitConfig, Metrics

from bitlayers.convs import ActModels, Conv2dModels, NormModels
from BitResNet import BitResNet

# =============================================================================
# 1. MODEL COMPONENTS (Backbone, FPN, SSH, Heads)
# =============================================================================

def _act(inplace=True):
    return ActModels.SiLU(inplace=inplace)

def _conv_block(
    in_ch: int, out_ch: int,
    k: int, stride: int, padding: int,
    scale_op: str,
    act: Any = _act(inplace=True),
    norm: Any = NormModels.BatchNorm2d(num_features=-1),
    bias=False,
) -> nn.Sequential:
    """Factory for creating 1.58-bit optimized convolution blocks."""
    conf = dict(
        in_channels=in_ch, out_channels=out_ch, kernel_size=k,
        stride=stride, padding=padding, bias=bias,
        bit=True, scale_op=scale_op, norm=norm,
    )
    # Handle explicit False/None
    act_layer = _act(inplace=True) if act is True else act
    norm_layer = NormModels.BatchNorm2d(num_features=-1) if norm is True else norm

    if act_layer and norm_layer:
        return Conv2dModels.Conv2dNormAct(**conf, act=act_layer).build()
    if not act_layer and norm_layer:
        return Conv2dModels.Conv2dNorm(**conf).build()
    if act_layer and not norm_layer:
        return Conv2dModels.Conv2dAct(**conf).build()
    return Conv2dModels.Conv2d(**conf).build()

# --- Backbone ---
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
            raise ValueError(f"Unsupported model_size={model_size}. Expected '18' or '50'.")

        super().__init__(block_cls, layers, num_classes=1000, expansion=expansion,
                         scale_op=scale_op, in_ch=in_ch, small_stem=small_stem)
        
        del self.head 
        self.feature_channels = (128 * expansion, 256 * expansion, 512 * expansion)

    def forward(self, x: torch.Tensor):
        # We need the features from stage 2, 3, and 4 (standard ResNet C3, C4, C5)
        _, c3, c4, c5 = self.forward_features(x)
        return c3, c4, c5

# --- Neck (FPN + SSH) ---
class FPN(nn.Module):
    def __init__(self, in_channels: Sequence[int], out_channels: int = 256, scale_op: str = "median"):
        super().__init__()
        self.lateral = nn.ModuleList([
            _conv_block(ch, out_channels, k=1, stride=1, padding=0, scale_op=scale_op, act=False)
            for ch in in_channels
        ])
        self.output = nn.ModuleList([
            _conv_block(out_channels, out_channels, k=3, stride=1, padding=1, scale_op=scale_op, act=False)
            for _ in in_channels
        ])

    def forward(self, feats: Sequence[torch.Tensor]) -> List[torch.Tensor]:
        results: List[torch.Tensor] = []
        last: Optional[torch.Tensor] = None

        # Top-down pathway
        for idx in range(len(feats) - 1, -1, -1):
            lat = self.lateral[idx](feats[idx])
            if last is not None:
                lat = lat + F.interpolate(last, size=lat.shape[-2:], mode="nearest")
            last = lat
            results.append(self.output[idx](lat))

        results.reverse()
        return results

class SSH(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, scale_op: str = "median") -> None:
        super().__init__()
        assert out_ch % 4 == 0
        half, quart = out_ch // 2, out_ch // 4

        self.branch3 = _conv_block(in_ch, half, 3, 1, 1, scale_op, act=False)
        self.stem5   = _conv_block(in_ch, quart, 3, 1, 1, scale_op, act=True)
        self.branch5 = _conv_block(quart, quart, 3, 1, 1, scale_op, act=False)
        self.branch7 = nn.Sequential(
            _conv_block(quart, quart, 3, 1, 1, scale_op, act=True),
            _conv_block(quart, quart, 3, 1, 1, scale_op, act=False),
        )
        self.act = _act(inplace=True).build()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c3 = self.branch3(x)
        c5_stem = self.stem5(x)
        c5 = self.branch5(c5_stem)
        c7 = self.branch7(c5_stem)
        return self.act(torch.cat((c3, c5, c7), dim=1))

# --- Heads & Main Model ---
class PredictionHead(nn.Module):
    def __init__(self, num_levels: int, in_channels: int, out_per_anchor: int, 
                 num_priors_per_level: Sequence[int], scale_op: str = "median"):
        super().__init__()
        self.heads = nn.ModuleList([
            nn.Sequential(
                _conv_block(in_channels, in_channels, k=3, stride=1, padding=1, scale_op=scale_op),
                _conv_block(in_channels, priors * out_per_anchor, k=1, stride=1, padding=0, 
                            scale_op=scale_op, act=False, norm=False, bias=True)
            ) for priors in num_priors_per_level
        ])
        self.out_per_anchor = out_per_anchor

    def forward(self, feats: Sequence[torch.Tensor]) -> torch.Tensor:
        preds = []
        for feat, head in zip(feats, self.heads):
            # Output: [B, A*C, H, W] -> [B, H, W, A*C]
            p = head(feat).permute(0, 2, 3, 1).contiguous()
            # Flatten to [B, N_anchors, C]
            preds.append(p.view(p.size(0), -1, self.out_per_anchor))
        return torch.cat(preds, dim=1)

class BitRetinaFace(nn.Module):
    def __init__(self, backbone_size: str = "50", fpn_channels: int = 256, 
                 num_classes: int = 2, num_priors: Sequence[int] = (2, 2, 2),
                 scale_op: str = "median", in_ch: int = 3, small_stem: bool = False):
        super().__init__()
        self.backbone = BitResNetBackbone(backbone_size, scale_op, in_ch, small_stem)
        self.fpn = FPN(self.backbone.feature_channels, fpn_channels, scale_op)
        self.ssh = nn.ModuleList([SSH(fpn_channels, fpn_channels, scale_op) for _ in range(3)])
        
        # Shared config for heads
        head_cfg = dict(num_levels=3, in_channels=fpn_channels, num_priors_per_level=num_priors, scale_op=scale_op)
        self.cls_head = PredictionHead(out_per_anchor=num_classes, **head_cfg)
        self.box_head = PredictionHead(out_per_anchor=4, **head_cfg)
        self.lm_head  = PredictionHead(out_per_anchor=10, **head_cfg)
        
        # Save initialization args for cloning/inference later
        self.cfg = locals(); del self.cfg['self']; del self.cfg['__class__']

    def forward(self, x: torch.Tensor):
        feats = self.backbone(x)
        feats = self.fpn(feats)
        feats = [m(f) for m, f in zip(self.ssh, feats)]
        
        # Order: box, cls, landmark (matching teacher logic usually, but consistent internally)
        return (
            self.box_head(feats),   # [B, N, 4]
            self.cls_head(feats),   # [B, N, 2]
            self.lm_head(feats)     # [B, N, 10]
        )

# =============================================================================
# 2. LOGIC (Anchors, Matching, Decoding, Loss)
# =============================================================================

class RetinaFaceAnchors:
    def __init__(self, min_sizes, steps, clip=False):
        self.min_sizes = min_sizes
        self.steps = steps
        self.clip = clip

    def __call__(self, image_size: Tuple[int, int], device="cpu") -> torch.Tensor:
        img_h, img_w = image_size
        anchors = []
        for step, sizes in zip(self.steps, self.min_sizes):
            feature_h, feature_w = math.ceil(img_h / step), math.ceil(img_w / step)
            
            # Vectorized grid generation
            shifts_x = torch.arange(0, feature_w, device=device) * step
            shifts_y = torch.arange(0, feature_h, device=device) * step
            shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing='ij')
            
            # Center points
            cx = (shift_x + 0.5 * step) / img_w
            cy = (shift_y + 0.5 * step) / img_h
            
            # Expand for each anchor size at this level
            for size in sizes:
                anchors.append(torch.stack([
                    cx, cy, 
                    torch.full_like(cx, size / img_w), 
                    torch.full_like(cy, size / img_h)
                ], dim=-1)) # [H, W, 4]

        # Flatten list of tensors
        anchors = torch.cat([a.view(-1, 4) for a in anchors], dim=0)
        if self.clip: anchors.clamp_(0.0, 1.0)
        return anchors

def decode_boxes(anchors, deltas, variances):
    """Decodes deltas back to [x1, y1, x2, y2]."""
    v0, v1 = variances
    cx, cy, w, h = anchors.unbind(1)
    dx, dy, dw, dh = deltas.unbind(1)

    gx = cx + dx * v0 * w
    gy = cy + dy * v0 * h
    gw = w * torch.exp(dw * v1)
    gh = h * torch.exp(dh * v1)

    return torch.stack([gx - gw/2, gy - gh/2, gx + gw/2, gy + gh/2], dim=1)

def detect_one_image(anchors, bbox_deltas, cls_logits, variances, score_thr=0.02, nms_iou=0.4):
    """Inference utility for a single image."""
    scores = cls_logits.softmax(dim=-1)[:, 1]
    boxes = decode_boxes(anchors, bbox_deltas, variances).clamp(0.0, 1.0)

    keep = scores > score_thr
    boxes, scores = boxes[keep], scores[keep]
    if boxes.numel() == 0:
        return boxes, scores

    keep_idx = nms(boxes, scores, nms_iou)
    return boxes[keep_idx], scores[keep_idx], keep_idx

def ap_at_iou(preds, gts, iou_thr=0.5):
    """Simplified AP calculation for validation logging."""
    # This is a placeholder for the integral AP function provided in your original snippet
    # Ensure compute_ap_voc_integral is defined if you need exact VOC calc.
    # For brevity, I'm including a simplified flow.
    
    device = preds[0]["boxes"].device if len(preds) else torch.device("cpu")
    total_gt = sum(len(g) for g in gts)
    if total_gt == 0: return 0.0

    # Flatten
    all_scores, all_img_ids, all_boxes = [], [], []
    for i, p in enumerate(preds):
        if p["boxes"].numel() == 0: continue
        all_scores.append(p["scores"])
        all_boxes.append(p["boxes"])
        all_img_ids.append(torch.full_like(p["scores"], i, dtype=torch.long))

    if not all_scores: return 0.0
    
    all_scores = torch.cat(all_scores)
    all_boxes = torch.cat(all_boxes)
    all_img_ids = torch.cat(all_img_ids)
    
    # Sort
    idx = torch.argsort(all_scores, descending=True)
    all_boxes, all_img_ids = all_boxes[idx], all_img_ids[idx]
    
    tp = torch.zeros_like(all_scores)
    fp = torch.zeros_like(all_scores)
    matched = [torch.zeros(len(g), dtype=torch.bool, device=device) for g in gts]
    
    for i in range(len(all_boxes)):
        img_id = all_img_ids[i]
        gt = gts[img_id]
        if len(gt) == 0:
            fp[i] = 1
            continue
            
        iou = box_iou(all_boxes[i].unsqueeze(0), gt).squeeze(0)
        max_iou, max_idx = iou.max(dim=0)
        
        if max_iou >= iou_thr and not matched[img_id][max_idx]:
            tp[i] = 1
            matched[img_id][max_idx] = True
        else:
            fp[i] = 1
            
    tp_cum = torch.cumsum(tp, dim=0)
    fp_cum = torch.cumsum(fp, dim=0)
    rec = tp_cum / total_gt
    prec = tp_cum / (tp_cum + fp_cum + 1e-12)
    
    # Simple AP integration (average precision)
    return torch.trapz(prec, rec).item() # Rough approximation

class MultiBoxLoss(nn.Module):
    def __init__(self, num_classes, overlap_thresh, neg_pos_ratio, box_weight, land_weight):
        super().__init__()
        self.num_classes = num_classes
        self.overlap_thresh = overlap_thresh
        self.neg_pos_ratio = neg_pos_ratio
        self.box_weight = box_weight
        self.land_weight = land_weight

    def forward(self, predictions, targets):
        """
        predictions: (loc_p, conf_p, land_p)
        targets: (loc_t, conf_t, land_t)
        """
        loc_p, conf_p, land_p = predictions
        loc_t, conf_t, land_t = targets

        # -----------------------------------------------------------
        # FIX: Flatten all tensors to [Batch*Anchors, ...] 
        # so they match the flattened mask length.
        # -----------------------------------------------------------
        loc_p = loc_p.view(-1, 4)      # [B, N, 4] -> [B*N, 4]
        loc_t = loc_t.view(-1, 4)      
        land_p = land_p.view(-1, 10)   # [B, N, 10] -> [B*N, 10]
        land_t = land_t.view(-1, 10)
        conf_p = conf_p.view(-1, self.num_classes) # [B*N, 2]
        conf_t = conf_t.view(-1)                   # [B*N]

        # 1. Classification Loss (OHEM)
        pos = conf_t > 0 # Label 1
        neg = conf_t == 0 # Label 0 (Background)

        # Calculate Loss for all (no reduction)
        # Handle ignore index (-1) by zeroing loss
        loss_c = F.cross_entropy(conf_p, conf_t.clamp(min=0), reduction='none')
        loss_c[conf_t < 0] = 0 
        
        # Hard Negative Mining
        num_pos = pos.sum()
        num_neg = torch.clamp(self.neg_pos_ratio * num_pos, max=neg.sum())
        
        # Sort negatives by loss
        loss_c_neg = loss_c.clone()
        loss_c_neg[~neg] = 0.0 
        _, neg_indices = loss_c_neg.sort(descending=True)
        
        # Create Hard Negative Mask
        hard_neg_mask = torch.zeros_like(conf_t, dtype=torch.bool)
        if num_neg > 0:
            hard_neg_mask[neg_indices[:int(num_neg.item())]] = True
        
        # Final Class Loss
        loss_cls = F.cross_entropy(conf_p[pos | hard_neg_mask], conf_t[pos | hard_neg_mask], reduction='sum')

        # 2. Box Regression Loss (Smooth L1)
        # Now loc_p and pos have matching first dimensions
        if num_pos > 0:
            loss_box = F.smooth_l1_loss(loc_p[pos], loc_t[pos], reduction='sum')
            
            # 3. Landmark Loss
            # Filter valid landmarks (where target is not -1)
            # land_t[pos] extracts only positive anchors
            pos_land_t = land_t[pos]
            pos_land_p = land_p[pos]
            
            valid_lm = (pos_land_t != -1.0).all(dim=1)
            
            if valid_lm.sum() > 0:
                loss_land = F.smooth_l1_loss(pos_land_p[valid_lm], pos_land_t[valid_lm], reduction='sum')
            else:
                loss_land = torch.tensor(0.0, device=loc_p.device)
        else:
            loss_box = torch.tensor(0.0, device=loc_p.device)
            loss_land = torch.tensor(0.0, device=loc_p.device)

        N = max(1.0, float(num_pos))
        return (loss_cls / N), (self.box_weight * loss_box / N), (self.land_weight * loss_land / N)
# =============================================================================
# 3. TRAINING & LIGHTNING MODULE
# =============================================================================

class RetinaFaceConfig(CommonTrainConfig, LitBitConfig):
    dataset_name: str = "retinaface"
    data_dir: str = "./data"
    export_dir: str = "./ckpt_widerface_retinaface"
    dataset: Optional[Dict] = None

    epochs: int = 50
    batch_size: int = 2
    num_workers: int = 0
    
    # Loss Weights
    alpha_kd: float = 0.9
    alpha_hint: float = 0.1
    box_weight: float = 2.0
    landmark_weight: float = 1.0
    wd: float = 1e-4
    lr: float = 1e-3

    # Model Params
    image_size: int = 640
    backbone_size: str = Field(default="50", description="18 or 50")
    fpn_channels: int = 256
    scale_op: str = "median"
    small_stem: bool = False
    
    # Anchor Params
    pos_iou: float = 0.35
    neg_iou: float = 0.35
    variance_xy: float = 0.1
    variance_wh: float = 0.2
    
    @field_validator("backbone_size")
    def _validate_backbone(cls, v):
        if v not in ("18", "50"): raise ValueError("backbone must be 18 or 50")
        return v

class LitRetinaFace(LitBit):
    def __init__(self, config: RetinaFaceConfig):
        super().__init__(config)
        self.hparams_cfg = config
        self.variances = (config.variance_xy, config.variance_wh)
        
        # 1. Anchors
        self.anchor_gen = RetinaFaceAnchors(
            min_sizes=[[16, 32], [64, 128], [256, 512]],
            steps=[8, 16, 32], clip=False
        )
        # Register buffer so it moves to GPU automatically
        self.register_buffer("anchors", self.anchor_gen((config.image_size, config.image_size)))
        
        # 2. Loss
        self.loss_fn = MultiBoxLoss(
            num_classes=2, overlap_thresh=config.pos_iou, neg_pos_ratio=7,
            box_weight=config.box_weight, land_weight=config.landmark_weight
        )

    def on_train_start(self):
        # Sanity check
        with torch.no_grad():
            dummy = torch.randn(1, 3, self.hparams_cfg.image_size, self.hparams_cfg.image_size, device=self.anchors.device)
            out = self.student(dummy) # returns (box, cls, lm)
            assert out[0].shape[1] == self.anchors.shape[0], \
                f"Anchor mismatch! Model: {out[0].shape[1]}, Anchors: {self.anchors.shape[0]}"

    def training_step(self, batch, batch_idx):
        images, raw_targets = batch
        raw_targets: List[RetinaFaceTensor]= [t.as_tensor() for t in raw_targets]
        raw_targets = [t.prepare(self.anchors, self.hparams_cfg.pos_iou,
                                 self.hparams_cfg.neg_iou, self.variances) for t in raw_targets]
        
        out_box, out_cls, out_land = self.student(images)
        tgt_box = torch.stack([t.bbox for t in raw_targets])
        tgt_cls = torch.stack([t.label for t in raw_targets])
        tgt_land = torch.stack([t.landmark for t in raw_targets])        
        
        # Compute Base Loss
        l_cls, l_box, l_land = self.loss_fn((out_box, out_cls, out_land), (tgt_box, tgt_cls, tgt_land))
        base_loss = l_cls + l_box + l_land
        stats = {"ce": l_cls, "bb": l_box, "lm": l_land}
        
        # Distillation
        if self.kd or self.hint:
            with torch.no_grad():
                # Teacher returns [loc, conf, land]
                t_box, t_cls, t_land = self.teacher_forward(images)

        if self.kd:
            loss_kd = self.alpha_kd * (
                self.kd(out_cls, t_cls) + self.kd(out_box, t_box) + self.kd(out_land, t_land)
            )
            stats["kd"] = loss_kd
            base_loss = (1.0 - self.alpha_kd) * base_loss + loss_kd

        if self.hint:
            loss_hint = self.alpha_hint * self.get_loss_hint()
            stats["hint"] = loss_hint
            base_loss += loss_hint

        return Metrics(loss=base_loss, metrics=stats)

    def validation_step(self, batch, batch_idx):
        images, raw_targets = batch
        raw_targets: List[RetinaFaceTensor]= [t.as_tensor() for t in raw_targets]
        raw_targets = [t.prepare(self.anchors, self.hparams_cfg.pos_iou,
                                 self.hparams_cfg.neg_iou, self.variances) for t in raw_targets]
        
        out_box, out_cls, out_land = self.student(images)
        tgt_box = torch.stack([t.bbox for t in raw_targets])
        tgt_cls = torch.stack([t.label for t in raw_targets])
        tgt_land = torch.stack([t.landmark for t in raw_targets])
        
        l_cls, l_box, l_land = self.loss_fn((out_box, out_cls, out_land), (tgt_box, tgt_cls, tgt_land))
        
        # Store for AP calc
        for i in range(images.size(0)):
            boxes, scores, keep_idx = detect_one_image(
                self.anchors, out_box[i], out_cls[i], self.variances, 
                score_thr=0.02, nms_iou=0.4
            )
            self._ap_preds.append({"boxes": boxes.cpu(), "scores": scores.cpu()})
            self._ap_gts.append(raw_targets[i].bbox.cpu())

        return Metrics(loss=l_cls + l_box + l_land, metrics={"val_ce": l_cls})

    def on_validation_epoch_start(self, epoch):
        self._ap_preds, self._ap_gts = [], []

    def on_validation_epoch_end(self, epoch):
        ap50 = ap_at_iou(self._ap_preds, self._ap_gts, iou_thr=0.5)
        # Log via print or trainer logger
        print(f"\nEpoch {epoch} | Val AP50: {ap50:.4f}")

    def configure_optimizers(self, trainer=None):
        opt = torch.optim.AdamW(self.configure_optimizer_params(), lr=self.lr, weight_decay=self.wd)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.epochs)
        return opt, sched, "epoch"
    
    def show_predict_examples(self, images):
        self.student.eval()
        preds = []
        with torch.no_grad():
            mean = (0.485, 0.456, 0.406)
            std = (0.229, 0.224, 0.225)
            out_box, out_cls, out_land = self.student(images)
            for i in range(images.size(0)):
                boxes, scores, keep_idx = detect_one_image(
                    self.anchors, out_box[i], out_cls[i], self.variances, 
                    score_thr=0.02, nms_iou=0.4
                )
                land  = out_land[i][keep_idx]
                y = RetinaFaceTensor(img=images[i], bbox=boxes, landmark=land, score=scores)
                preds.append(y)
        RetinaFaceDataModule.show_examples_static(images, preds, mean, std)


def make_resnet50_teacher(device="cpu"):
    m = torch.hub.load("qinhy/Pytorch_Retinaface", "retinaface_resnet50", pretrained=True)
    return m.eval().to(device)

def main():
    parser = ArgumentParser(model=RetinaFaceConfig)
    config = parser.parse_typed_args()
    dm_conf = DataModuleConfig.model_validate(config.model_dump())
    config.dataset = dm_conf.model_dump()

    config.student = BitRetinaFace(
        backbone_size=config.backbone_size, fpn_channels=config.fpn_channels,
        scale_op=config.scale_op, small_stem=config.small_stem
    )
    config.teacher = make_resnet50_teacher()
    
    # Feature hints for distillation
    config.hint_points = [
        ("ssh.0","ssh1"), ("ssh.1","ssh2"), ("ssh.2","ssh3"),
        ("backbone.layer1","body.layer1"), ("backbone.layer2","body.layer2"),
        ("backbone.layer3","body.layer3"), ("backbone.layer4","body.layer4"),
    ]

    lit = LitRetinaFace(config)
    
    # Use your Trainer stack
    trainer = AccelTrainer(
        max_epochs=config.epochs, 
        mixed_precision="bf16" if config.amp else "no",
        gradient_accumulation_steps=1,
        log_every_n_steps=10,
    )
    trainer.fit(lit, datamodule=dm_conf.build())
    return lit, dm_conf.build()

if __name__ == "__main__":
    lit, dm =  main()
    dm.setup()
    images, targets = next(iter(dm.val_dataloader()))
    lit.show_predict_examples(images)
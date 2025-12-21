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
# Dataset / DataModule
# -----------------------------------------------------------------------------
class RetinaFaceDataset(VisionDataset):
    
    """
    Yep — for *geometric* aug (resize/crop/flip/affine), you really want **joint transforms** that see **(image, boxes, landmarks)** together.

    The cleanest way in modern torchvision is:

    * treat `boxes` as `tv_tensors.BoundingBoxes`
    * treat `landmarks` as `tv_tensors.KeyPoints` (shape `[N, 5, 2]`)
    * run `torchvision.transforms.v2` so the same ops update everything consistently ([PyTorch Documentation][1])
    * note: **KeyPoints support landed in torchvision 0.23** (beta feature) ([PyTorch Documentation][2])

    Below is a practical pattern that also handles the one RetinaFace-specific quirk: **when you flip horizontally, you must swap left/right landmark indices** (eye/mouth corners). torchvision can flip keypoint coordinates, but it doesn’t know your semantic ordering.

    ## Joint transforms for RetinaFace (boxes + landmarks)

    ```py
    import torch
    from torchvision import tv_tensors
    from torchvision.transforms import v2
    from torchvision.transforms.v2 import functional as F


    class WrapRetinaFaceTargets:
        Convert plain tensors to TVTensors so v2 transforms update them.
        def __call__(self, img, target):
            H, W = map(int, F.get_size(img))  # [H, W]
            target = dict(target)

            boxes = target["boxes"]
            if boxes.numel() == 0:
                boxes_tv = tv_tensors.BoundingBoxes(boxes.view(-1, 4), format="XYXY", canvas_size=(H, W))
            else:
                boxes_tv = tv_tensors.BoundingBoxes(boxes.view(-1, 4), format="XYXY", canvas_size=(H, W))

            # landmarks: (N,10) with -1 for missing -> KeyPoints (N,5,2) + valid mask
            lm = target.get("landmarks", None)
            if lm is None or lm.numel() == 0:
                kp = torch.zeros((boxes_tv.shape[0], 5, 2), dtype=torch.float32)
                valid = torch.zeros((boxes_tv.shape[0], 5), dtype=torch.bool)
            else:
                kp = lm.view(-1, 5, 2).to(torch.float32).clone()
                valid = (kp[..., 0] >= 0) & (kp[..., 1] >= 0)
                kp[~valid] = 0.0  # placeholder; we keep "valid" separately

            kps_tv = tv_tensors.KeyPoints(kp, canvas_size=(H, W))

            target["boxes"] = boxes_tv
            target["keypoints"] = kps_tv
            target["landmarks_valid"] = valid
            return img, target


    class RandomFaceHorizontalFlip:
        Flip image/boxes/keypoints and also swap left/right landmark indices.
        def __init__(self, p=0.5):
            self.p = float(p)

        def __call__(self, img, target):
            if torch.rand(()) >= self.p:
                return img, target

            img = F.horizontal_flip(img)
            target = dict(target)

            target["boxes"] = F.horizontal_flip(target["boxes"])
            kps = F.horizontal_flip(target["keypoints"])      # (N,5,2), coords flipped
            kps = kps[:, [1, 0, 2, 4, 3], :]                  # swap le<->re and lm<->rm
            target["keypoints"] = kps

            if "landmarks_valid" in target:
                target["landmarks_valid"] = target["landmarks_valid"][:, [1, 0, 2, 4, 3]]

            return img, target


    class UnwrapAndNormalizeRetinaFace:
        Convert TVTensors back to your model format: boxes/landmarks normalized to [0,1] and -1 missing.
        def __call__(self, img, target):
            H, W = map(int, F.get_size(img))
            target = dict(target)

            boxes = torch.as_tensor(target["boxes"], dtype=torch.float32).view(-1, 4)
            if boxes.numel():
                boxes[:, [0, 2]] /= float(W)
                boxes[:, [1, 3]] /= float(H)
                boxes = boxes.clamp(0.0, 1.0)

            kps = torch.as_tensor(target["keypoints"], dtype=torch.float32)  # (N,5,2)
            valid = target.get("landmarks_valid", torch.ones(kps.shape[:2], dtype=torch.bool))

            # after crop/resize, some points may go out of bounds: mark them missing
            within = (kps[..., 0] >= 0) & (kps[..., 0] <= W) & (kps[..., 1] >= 0) & (kps[..., 1] <= H)
            valid = valid & within

            kps[..., 0] /= float(W)
            kps[..., 1] /= float(H)
            kps = kps.clamp(0.0, 1.0)
            kps[~valid] = -1.0

            target["boxes"] = boxes
            target["landmarks"] = kps.view(-1, 10)

            target.pop("keypoints", None)
            target.pop("landmarks_valid", None)
            return img, target
    ```

    ### Example train transform pipeline

    ```py
    mean = (0.485, 0.456, 0.406)
    std  = (0.229, 0.224, 0.225)

    train_tf = v2.Compose([
        v2.ToImage(),                            # makes image a tensor TVTensor
        WrapRetinaFaceTargets(),                 # boxes->BoundingBoxes, landmarks->KeyPoints
        v2.RandomResizedCrop((640, 640), antialias=True),
        RandomFaceHorizontalFlip(p=0.5),         # flip + swap landmark order
        v2.ClampBoundingBoxes(),                 # optional :contentReference[oaicite:2]{index=2}
        v2.ClampKeyPoints(),                     # optional :contentReference[oaicite:3]{index=3}
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=mean, std=std),
        UnwrapAndNormalizeRetinaFace(),          # back to your normalized [0,1] + -1 missing format
    ])
    ```

    ### Dataset usage

    Use the **joint** `transforms=` argument (not `transform=`), since `transform=` is image-only:

    ```py
    ds = RetinaFaceDataset(
        root=self.data_dir,
        train=True,
        download=True,
        transforms=train_tf,
    )
    ```

    If you paste your current `train_tf` (what ops you want: resize? mosaic? iou-crop? affine?), I can tailor the pipeline so every step correctly updates **boxes + landmarks** and matches RetinaFace’s expected input/output shapes.

    [1]: https://docs.pytorch.org/vision/stable/transforms.html "Transforming images, videos, boxes and more — Torchvision 0.24 documentation"
    [2]: https://docs.pytorch.org/vision/main/auto_examples/transforms/plot_keypoints_transforms.html "Transforms on KeyPoints — Torchvision main documentation"



    Expected layout (default):
      root/
        annotations/
          train.json
          val.json
        (images referenced by JSON paths)

    JSON entries:
      {
        "file": "...",
        "width": 1024, "height": 768,          # optional
        "boxes": [[x1,y1,x2,y2], ...],          # absolute pixels
        "landmarks": [[x1,y1, ..., x5,y5], ...] # optional, absolute pixels OR -1 for missing
      }

    Outputs:
      image: tensor or PIL (depending on transforms)
      target:
        boxes: (N,4) in [0,1]
        landmarks: (N,10) in [0,1] or -1 for missing
        labels: (N,) long (all ones)
    """

    def __init__(
        self,
        root: str,
        train: bool = True,
        download: bool = False,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,  # joint (img,target) transform
        image_size: int = 640,
        augment: Optional[bool] = None,
        annotation_file: Optional[str] = None,
        annotations_dir: str = "annotations",
        train_ann: str = "train.json",
        val_ann: str = "val.json",
        download_url: Optional[str] = None,  # optional hook
    ) -> None:
        super().__init__(root=root, transforms=transforms, transform=transform, target_transform=target_transform)

        self.root = Path(root)
        self.train = bool(train)
        self.download = bool(download)
        self.image_size = int(image_size)
        self.download_url = download_url

        if augment is None:
            augment = self.train
        self.augment = bool(augment)

        # pick annotation file
        if annotation_file is None:
            ann_name = train_ann if self.train else val_ann
            annotation_file = str(Path(annotations_dir) / ann_name)

        self.ann_path = Path(annotation_file)
        if not self.ann_path.is_absolute():
            self.ann_path = self.root / self.ann_path

        if self.download and not self.ann_path.exists():
            self._download()

        if not self.ann_path.exists():
            raise FileNotFoundError(f"Annotation file not found: {self.ann_path}")

        with open(self.ann_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(f"Annotation file {self.ann_path} must contain a list of samples.")
        self.items: List[Dict[str, Any]] = data

        # default transform only if user didn't pass transform/transforms
        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)
        self._default_tf = T.Compose(
            [
                T.Resize((self.image_size, self.image_size)),
                T.ToTensor(),
                T.Normalize(self.mean, self.std),
            ]
        )

        # built-in light aug (safe + we update targets for flip)
        self.color_aug = T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)

    def __len__(self) -> int:
        return len(self.items)

    def _resolve_path(self, file_path: str) -> Path:
        p = Path(file_path)
        return p if p.is_absolute() else (self.root / p)

    def _download(self) -> None:
        # Generic hook: you can implement this once you know your dataset URL/format.
        if not self.download_url:
            raise RuntimeError(
                "download=True was set but no download_url was provided, and the annotation file is missing.\n"
                "Either provide download_url or place the dataset under root."
            )
        # Example implementation (uncomment once you know the URL points to an archive):
        # from torchvision.datasets.utils import download_and_extract_archive
        # download_and_extract_archive(self.download_url, download_root=str(self.root))
        raise NotImplementedError("Provide a real download implementation for your dataset source.")

    @staticmethod
    def _flip_targets_horiz(
        boxes: torch.Tensor, landmarks: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if boxes.numel():
            x1 = boxes[:, 0].clone()
            x2 = boxes[:, 2].clone()
            boxes[:, 0] = 1.0 - x2
            boxes[:, 2] = 1.0 - x1

        if landmarks.numel():
            lm = landmarks.view(-1, 5, 2).clone()
            valid = lm[..., 0] >= 0  # missing is -1

            # flip x for valid points
            lm[..., 0][valid] = 1.0 - lm[..., 0][valid]

            # swap left/right: [le,re,nose,lm,rm] -> [re,le,nose,rm,lm]
            lm = lm[:, [1, 0, 2, 4, 3], :]

            landmarks = lm.view(-1, 10)

        return boxes, landmarks

    def __getitem__(self, idx: int) -> Tuple[Any, Dict[str, torch.Tensor]]:
        entry = self.items[idx]

        img_path = self._resolve_path(entry["file"])
        img = ImageOps.exif_transpose(Image.open(img_path)).convert("RGB")

        # Prefer actual image size (EXIF-safe). Keep JSON width/height as fallback.
        orig_w = int(entry.get("width", img.width))
        orig_h = int(entry.get("height", img.height))
        orig_w = max(orig_w, 1)
        orig_h = max(orig_h, 1)

        boxes = torch.tensor(entry.get("boxes", []), dtype=torch.float32).view(-1, 4)
        if boxes.numel() == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)

        if boxes.numel():
            boxes[:, [0, 2]] /= float(orig_w)
            boxes[:, [1, 3]] /= float(orig_h)
            boxes = boxes.clamp(0.0, 1.0)

        land = entry.get("landmarks", [])
        if not land:
            landmarks = torch.full((boxes.size(0), 10), -1.0, dtype=torch.float32)
        else:
            landmarks = torch.tensor(land, dtype=torch.float32).view(-1, 10)
            if landmarks.size(0) != boxes.size(0):
                raise ValueError(
                    f"Mismatch: {boxes.size(0)} boxes but {landmarks.size(0)} landmark rows for file={entry.get('file')}"
                )
            lm = landmarks.view(-1, 5, 2)

            # preserve missing points (-1): only normalize/clamp valid points
            valid = lm[..., 0] >= 0
            lm_x = lm[..., 0].clone()
            lm_y = lm[..., 1].clone()
            lm_x[valid] = (lm_x[valid] / float(orig_w)).clamp(0.0, 1.0)
            lm_y[valid] = (lm_y[valid] / float(orig_h)).clamp(0.0, 1.0)
            lm[..., 0] = lm_x
            lm[..., 1] = lm_y
            landmarks = lm.view(-1, 10)

        # built-in aug only if user did NOT provide a joint transform pipeline
        if self.augment and self.transforms is None:
            img = self.color_aug(img)
            if torch.rand(()) < 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                boxes, landmarks = self._flip_targets_horiz(boxes, landmarks)

        target: Dict[str, torch.Tensor] = {
            "boxes": boxes,
            "landmarks": landmarks,
            "labels": torch.ones((boxes.size(0),), dtype=torch.long),
        }

        # torchvision-style transform handling:
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        else:
            if self.transform is not None:
                img = self.transform(img)
            else:
                # default image pipeline (resize + tensor + normalize)
                img = self._default_tf(img)

            if self.target_transform is not None:
                target = self.target_transform(target)

        return img, target

def retinaface_collate(batch):
    images = torch.stack([item[0] for item in batch])
    boxes = [item[1]["boxes"] for item in batch]
    landmarks = [item[1]["landmarks"] for item in batch]
    return images, {"boxes": boxes, "landmarks": landmarks}


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
        target_landmarks: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        return (
            torch.stack(labels, dim=0),
            torch.stack(bbox_t, dim=0),
            torch.stack(land_t, dim=0),
        )

    def _compute_losses(
        self,
        outputs: Dict[str, torch.Tensor],
        target_labels: torch.Tensor,
        target_boxes: torch.Tensor,
        target_landmarks: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        cls_logits = outputs["cls"]
        bbox_preds = outputs["bbox"]
        land_preds = outputs["landmark"]

        labels = target_labels.view(-1)
        valid = labels >= 0
        if valid.sum() == 0:
            cls_loss = torch.tensor(0.0, device=self.device)
        else:
            cls_loss = F.cross_entropy(
                cls_logits.view(-1, cls_logits.size(-1))[valid],
                labels[valid],
            )

        pos_mask = target_labels == 1
        pos_count = pos_mask.sum().clamp(min=1)

        if pos_mask.any():
            bbox_loss = F.smooth_l1_loss(
                bbox_preds[pos_mask],
                target_boxes[pos_mask],
                reduction="sum",
            ) / pos_count
        else:
            bbox_loss = torch.tensor(0.0, device=self.device)

        valid_landmarks = (target_landmarks >= 0).all(dim=2)
        if valid_landmarks.any():
            landmark_loss = F.smooth_l1_loss(
                land_preds[valid_landmarks],
                target_landmarks[valid_landmarks],
                reduction="sum",
            ) / valid_landmarks.sum().clamp(min=1)
        else:
            landmark_loss = torch.tensor(0.0, device=self.device)

        total = cls_loss + self.box_weight * bbox_loss + self.landmark_weight * landmark_loss
        return total, {
            "loss": total,
            "cls": cls_loss,
            "bbox": bbox_loss,
            "landmark": landmark_loss,
        }

    def training_step(self, batch, batch_idx: int):
        images, targets = batch
        outputs = self(images)
        labels, bbox_t, land_t = self._build_targets(targets["boxes"], targets["landmarks"])
        total, stats = self._compute_losses(outputs, labels, bbox_t, land_t)
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

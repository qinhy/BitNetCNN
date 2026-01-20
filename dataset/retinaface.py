import copy
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageOps
from pydantic import BaseModel, Field
import torch
from albumentations.core.composition import BaseCompose
from albumentations.core.transforms_interface import BasicTransform
import albumentations as A
from torchvision.datasets import VisionDataset
from torchvision.datasets.utils import download_and_extract_archive
from torchvision.datasets.utils import download_url
from torchvision.datasets.utils import extract_archive
from torchvision.ops import box_iou

from .base import DataSetModule


class RetinaFaceTensor(BaseModel):
    file: str = ""
    img_h: int = Field(default=0)
    img_w: int = Field(default=0)
    img: Optional[Any] = Field(default=None, exclude=True)
    label: Optional[Any] = Field(default=None, exclude=True)
    # xyxy
    # tensor : Normalized coords (ratio, 0-1)
    # numpy : Pixel coords (up to image size)
    bbox: Optional[Any] = Field(default=None, exclude=True)
    landmark: Optional[Any] = Field(
        default=None, exclude=True
    )  # [lx, ly, rx, ry, nx, ny, mlx, mly, mrx, mry] as shape (N * 5,2)
    landmark_vis: Optional[Any] = Field(
        default=None, exclude=True
    )  # [lv, rv, nv, mlv, mrv] as shape (N * 5)

    def model_post_init(self, context):
        if self.img is None:
            if not self.file:
                raise ValueError("RetinaFaceTensor requires file when img is None.")
            img_pil = ImageOps.exif_transpose(Image.open(self.file)).convert("RGB")
            self.img = np.asarray(img_pil)  # (H,W,3) uint8
            self.img_h, self.img_w = self.img.shape[:2]

            if self.bbox is None:
                raise ValueError(f"bbox is required for file={self.file}")
            self.bbox, keep = self._sanitize_xyxy(self.bbox, self.img_h, self.img_w, min_size=1.0)
            # if len(keep) != keep.sum():
            #     raise ValueError("invalid bbox included, for file={self.file}")
            # --- landmark: raw pixels (5,2), missing = -1 (numpy float32) ---
            land = self.landmark
            if land is None or len(land) == 0:
                num_boxes = int(self.bbox.shape[0]) if self.bbox is not None else 0
                self.landmark = np.full((num_boxes * 5, 2), -1.0, dtype=np.float32)
                self.landmark_vis = np.full((num_boxes * 5,), -1.0, dtype=np.float32)
            else:
                landmark = np.asarray(land, dtype=np.float32).reshape(-1, 15)[keep]
                if landmark.shape[0] != self.bbox.shape[0]:
                    raise ValueError(
                        f"Mismatch: {self.bbox.shape[0]} boxes but {landmark.shape[0]} landmark rows "
                        f"for file={self.file}"
                    )

                lm = landmark.reshape(-1, 5, 3).copy()
                valid = (lm[..., 0] >= 0) & (lm[..., 1] >= 0)

                # force invalid points to -1 exactly
                lm[..., 0][~valid] = -1.0
                lm[..., 1][~valid] = -1.0
                lm[..., 2][~valid] = -1.0

                lm = lm.reshape(-1, 5, 3).reshape(-1, 3)
                self.landmark = lm[:, :2]
                self.landmark_vis = lm[:, 2]

            self.label = np.ones((self.bbox.shape[0],), dtype=np.int64)
        return super().model_post_init(context)

    def _sanitize_xyxy(self, bboxes, h, w, min_size=1.0):
        b = np.asarray(bboxes, dtype=np.float32).reshape(-1, 4)
        x1, y1, x2, y2 = b.T

        # Fix swapped coords
        x1_, x2_ = np.minimum(x1, x2), np.maximum(x1, x2)
        y1_, y2_ = np.minimum(y1, y2), np.maximum(y1, y2)

        # Clip to image bounds (Albumentations allows x==w, y==h for pascal_voc)
        x1_ = np.clip(x1_, 0.0, float(w))
        x2_ = np.clip(x2_, 0.0, float(w))
        y1_ = np.clip(y1_, 0.0, float(h))
        y2_ = np.clip(y2_, 0.0, float(h))

        bw = x2_ - x1_
        bh = y2_ - y1_
        keep = (bw >= min_size) & (bh >= min_size) & np.isfinite(bw) & np.isfinite(bh)

        b2 = np.stack([x1_, y1_, x2_, y2_], axis=1)
        return b2[keep], keep

    def landmark_xs_ys(self):
        pts = self.landmark
        valid = (pts[..., 0] >= 0) & (pts[..., 1] >= 0)
        xs = pts[..., 0][valid]
        ys = pts[..., 1][valid]
        if type(pts) is torch.Tensor:
            xs = xs.numpy()
            ys = ys.numpy()
        return xs, ys

    def norm_to_0_1(self, clip: bool = True):
        if type(self.img) is torch.Tensor:
            C, self.img_h, self.img_w = self.img.shape
        else:
            self.img_h, self.img_w = self.img.shape[:2]

        """Normalize bbox/landmarks from pixel coords to [0,1]. Keeps -1 as missing."""
        bbox, landmark = None, None
        # bbox: xyxy
        if self.bbox is not None:
            b_norm = np.asarray(self.bbox, dtype=np.float32)
            if b_norm.size == 0:
                b_norm = b_norm.reshape(0, 4)
            else:
                b_norm = b_norm.reshape(-1, 4)
            whwh = np.asarray([self.img_w, self.img_h, self.img_w, self.img_h], dtype=np.float32)
            b_norm /= whwh
            if clip:
                b_norm = np.clip(b_norm, 0.0, 1.0)
            bbox = b_norm
        else:
            bbox = np.zeros((0, 4), dtype=np.float32)

        # landmarks: (N,5,2)
        if self.landmark is not None:
            p_norm = np.asarray(self.landmark, dtype=np.float32)
            if p_norm.size == 0:
                p_norm = p_norm.reshape(0, 2)
            elif p_norm.ndim == 1:
                p_norm = p_norm.reshape(-1, 2)
            neg_mask = p_norm < 0
            p_norm[..., 0] /= float(self.img_w)
            p_norm[..., 1] /= float(self.img_h)
            if clip:
                p_norm = np.clip(p_norm, 0.0, 1.0)
                p_norm[neg_mask] = -1.0
            landmark = p_norm
        else:
            landmark = np.zeros((0, 2), dtype=np.float32)
        return bbox, landmark

    def clone(self, update={}, deep=False):
        if not deep:
            return self.model_copy(update=update)
        else:
            return copy.deepcopy(self)

    def as_tensor(self):
        if (
            torch.is_tensor(self.bbox)
            and torch.is_tensor(self.label)
            and torch.is_tensor(self.landmark)
            and torch.is_tensor(self.landmark_vis)
        ):
            return self
        bbox, landmark = self.norm_to_0_1()
        landmark_vis = self.landmark_vis
        if landmark_vis is None:
            if landmark is None:
                landmark_vis = np.zeros((0,), dtype=np.float32)
            else:
                lmk = np.asarray(landmark)
                count = lmk.reshape(-1, 2).shape[0] if lmk.size else 0
                landmark_vis = np.full((count,), -1.0, dtype=np.float32)
        res = self.clone(update=dict(bbox=bbox, landmark=landmark, landmark_vis=landmark_vis))
        res.bbox = torch.as_tensor(res.bbox)
        res.landmark = torch.as_tensor(res.landmark)
        res.landmark_vis = torch.as_tensor(res.landmark_vis)
        return res

    def norm_to_pixel(self):
        """Normalize bbox/landmarks from [0,1] to pixel coords."""
        bbox, landmark = None, None
        # bbox: xyxy
        if self.bbox is not None:
            b_norm = self.bbox.clone()  # xyxy in pixel
            whwh = torch.Tensor([self.img_w, self.img_h, self.img_w, self.img_h]).to(
                dtype=torch.float32
            )
            b_norm *= whwh
            bbox = b_norm.detach().cpu().numpy()
        # landmarks: (N,5,2)
        if self.landmark is not None:
            p_norm = self.landmark.clone().reshape(-1, 2)
            wh = torch.Tensor([self.img_w, self.img_h]).to(dtype=torch.float32)
            p_norm *= wh
            landmark = p_norm.detach().cpu().numpy().reshape(-1, 5, 2)  # (N,5,2)
        return bbox, landmark

    def as_numpy(self):
        bbox, landmark = self.norm_to_pixel()
        return self.clone(update=dict(bbox=bbox, landmark=landmark, landmark_vis=self.landmark_vis))

    @staticmethod
    def encode_boxes(
        anchors: torch.Tensor,
        gt_boxes_xyxy: torch.Tensor,
        variances: Tuple[float, float] = (0.1, 0.2),
        eps: float = 1e-6,
    ) -> torch.Tensor:
        """
        anchors:       [N,4] (cx,cy,w,h)
        gt_boxes_xyxy: [N,4] (x1,y1,x2,y2) in same frame as anchors
        returns:       [N,4] (dx,dy,dw,dh)
        """
        if anchors.ndim != 2 or anchors.shape[1] != 4:
            raise ValueError(f"anchors must be [N,4], got {tuple(anchors.shape)}")
        if gt_boxes_xyxy.ndim != 2 or gt_boxes_xyxy.shape[1] != 4:
            raise ValueError(f"gt_boxes_xyxy must be [N,4], got {tuple(gt_boxes_xyxy.shape)}")
        if anchors.shape[0] != gt_boxes_xyxy.shape[0]:
            raise ValueError("anchors and gt_boxes_xyxy must have same N")

        v0, v1 = variances

        a_xy = anchors[:, 0:2]  # (cx, cy)
        a_wh = anchors[:, 2:4].clamp_min(eps)  # (aw, ah) clamped exactly like before

        b1 = gt_boxes_xyxy[:, 0:2]  # (x1, y1)
        b2 = gt_boxes_xyxy[:, 2:4]  # (x2, y2)
        g_xy = (b1 + b2) * 0.5  # (gx, gy)
        g_wh = (b2 - b1).clamp_min(eps)  # (gw, gh) clamped exactly like before

        d_xy = (g_xy - a_xy) / a_wh / v0
        d_wh = torch.log(g_wh / a_wh) / v1

        return torch.cat((d_xy, d_wh), dim=1)

    @staticmethod
    def decode_boxes(
        anchors: torch.Tensor,
        deltas: torch.Tensor,
        variances: Tuple[float, float] = (0.1, 0.2),
        clip: bool = False,
    ) -> torch.Tensor:
        """
        anchors:  [N,4] (cx,cy,w,h)
        deltas:   [N,4] (dx,dy,dw,dh)
        returns:  [N,4] (x1,y1,x2,y2) in same frame as anchors (often normalized)
        """
        if anchors.ndim != 2 or anchors.shape[1] != 4:
            raise ValueError(f"anchors must be [N,4], got {tuple(anchors.shape)}")
        if deltas.ndim != 2 or deltas.shape[1] != 4:
            raise ValueError(f"deltas must be [N,4], got {tuple(deltas.shape)}")
        if anchors.shape[0] != deltas.shape[0]:
            raise ValueError("anchors and deltas must have same N")

        v0, v1 = variances

        a_xy = anchors[:, 0:2]  # (cx, cy)
        a_wh = anchors[:, 2:4]  # (aw, ah)

        d_xy = deltas[:, 0:2]  # (dx, dy)
        d_wh = deltas[:, 2:4]  # (dw, dh)

        g_xy = a_xy + d_xy * v0 * a_wh
        g_wh = a_wh * torch.exp(d_wh * v1)

        x1y1 = g_xy - 0.5 * g_wh
        x2y2 = g_xy + 0.5 * g_wh

        out = torch.cat((x1y1, x2y2), dim=1)
        if clip:
            out = out.clamp(0.0, 1.0)
        return out

    @staticmethod
    def encode_landmarks(
        anchors: torch.Tensor,
        gt_landmarks: torch.Tensor,
        variances: Tuple[float, float] = (0.1, 0.2),
    ) -> torch.Tensor:
        """
        anchors:      [N,4]  (cx,cy,w,h)
        gt_landmarks: [N,10] (x1,y1,...,x5,y5) same frame as anchors
        returns:      [N,10] (dx1,dy1,...,dx5,dy5)
        """
        if anchors.shape[0] != gt_landmarks.shape[0]:
            raise ValueError("anchors and gt_landmarks must have same N")

        v0 = variances[0]

        lm = gt_landmarks.reshape(-1, 5, 2)  # [N,5,2]
        a = anchors[:, None, :]  # [N,1,4] for broadcasting

        # dx = (x - cx) / (w * v0), dy = (y - cy) / (h * v0)
        d = (lm - a[..., 0:2]) / (a[..., 2:4] * v0)  # [N,5,2]
        return d.reshape(-1, 10)

    @staticmethod
    def decode_landmarks(
        anchors: torch.Tensor,
        landm_deltas: torch.Tensor,
        variances: Tuple[float, float] = (0.1, 0.2),
    ) -> torch.Tensor:
        """
        anchors:      [N,4]  (cx,cy,w,h)
        landm_deltas: [N,10] (dx1,dy1,...,dx5,dy5)
        returns:      [N,10] (x1,y1,...,x5,y5) same frame as anchors
        """
        if anchors.shape[0] != landm_deltas.shape[0]:
            raise ValueError("anchors and landm_deltas must have same N")

        v0 = variances[0]

        d = landm_deltas.reshape(-1, 5, 2)  # [N,5,2]
        a = anchors[:, None, :]  # [N,1,4]

        xy = a[..., 0:2] + d * v0 * a[..., 2:4]  # [N,5,2]
        return xy.reshape(-1, 10)

    @staticmethod
    def match_anchors(
        anchors: torch.Tensor,
        gt_boxes: torch.Tensor,
        gt_landmarks: torch.Tensor,
        pos_iou,
        neg_iou,
        variances,
    ):
        """
        Matches anchors to Ground Truth.
        Returns:
            labels: [N] (-1: ignore, 0: bg, 1: face)
            bbox_targets: [N, 4]
            landmark_targets: [N, 10]
        """
        # landmark is [N, 10] or [N, 5, 2]. Ensure [N, 5, 2] for match_anchors
        if gt_landmarks.dim() == 2:
            gt_landmarks = gt_landmarks.view(-1, 5, 2)
        else:
            gt_landmarks = gt_landmarks

        if gt_boxes.numel() == 0:
            device = anchors.device
            return (
                torch.zeros(anchors.size(0), dtype=torch.long, device=device),
                torch.zeros(anchors.size(0), 4, device=device),
                torch.full((anchors.size(0), 10), -1.0, device=device),
            )

        # Convert anchors to xyxy for IoU
        cx, cy, w, h = anchors.unbind(1)
        anchor_boxes = torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=1)

        # IoU Matrix [N_anchors, K_gt]
        ious = box_iou(anchor_boxes, gt_boxes)

        # 1. Best GT for each anchor
        best_target_iou, best_target_idx = ious.max(dim=1)

        # 2. Best anchor for each GT (Force match)
        best_anchor_iou, best_anchor_idx = ious.max(dim=0)

        # Force include the best anchor for every GT by artifically boosting IoU
        best_target_iou.index_fill_(0, best_anchor_idx, 2.0)
        for i, anchor_idx in enumerate(best_anchor_idx):
            best_target_idx[anchor_idx] = i

        # Assign labels
        labels = torch.full((anchors.size(0),), -1, dtype=torch.long, device=anchors.device)  # -1 ignore
        labels[best_target_iou < neg_iou] = 0  # BG
        labels[best_target_iou >= pos_iou] = 1  # Face

        # Prepare targets
        matched_gt_boxes = gt_boxes[best_target_idx]
        bbox_targets = RetinaFaceTensor.encode_boxes(anchors, matched_gt_boxes, variances)

        # Landmark encoding (only for positives)
        matched_landmarks = gt_landmarks[best_target_idx]
        landmark_targets = torch.full((anchors.size(0), 10), -1.0, device=anchors.device)

        pos_mask = labels == 1
        if pos_mask.any():
            # Check validity: [K, 5, 2] -> all coords >= 0
            valid_lm_mask = (matched_landmarks >= 0).all(dim=1).all(dim=1)

            # We need to process all positive anchors, but only write valid LMs
            # We can do this by masking the positive set
            p_anchors = anchors[pos_mask]
            p_lm = matched_landmarks[pos_mask]
            p_valid = valid_lm_mask[pos_mask]

            if p_valid.any():
                enc_lm = RetinaFaceTensor.encode_landmarks(p_anchors, p_lm, variances)

                # Apply to targets
                current_targets = landmark_targets[pos_mask]
                # Only update valid ones
                current_targets[p_valid] = enc_lm[p_valid]
                landmark_targets[pos_mask] = current_targets

        return labels, bbox_targets, landmark_targets

    def filter_landmarks(self, bbox_ids):
        # land: (num_boxes, 5, 2)
        land = self.landmark.reshape(-1, 5, 2)[bbox_ids]  # (M, 5, 2)
        vis = self.landmark_vis.reshape(-1, 5)[bbox_ids]  # (M, 5)
        if type(self.img) is torch.Tensor:
            c, h, w = self.img.shape
        else:
            h, w, c = self.img.shape

        x = land[..., 0]
        y = land[..., 1]

        invisible = (x < 0) | (y < 0) | (x >= w) | (y >= h)  # (M, 5)

        land[invisible] = -1.0  # broadcasts into last dim (2)
        vis[invisible] = -1.0

        self.landmark = land.reshape(-1, 2)  # (M*5, 2)
        self.landmark_vis = vis.reshape(-1)  # (M*5,)

    def prepare(self, anchors, pos_iou, neg_iou, variances):
        if len(self.landmark) // 5 != len(self.bbox):
            self.show()
            raise ValueError(f"{len(self.landmark)//5}(x5 points) landmarks not fit {len(self.bbox)} bboxes!")

        t = self.as_tensor()
        device = anchors.device

        t.label, t.bbox, t.landmark = RetinaFaceTensor.match_anchors(
            anchors, t.bbox.to(device), t.landmark.to(device), pos_iou, neg_iou, variances
        )
        return t

    def show(
        self,
        mean=None,
        std=None,
        ax=None,
        figsize=(8, 8),
        title: str = "",
        show_landmarks: bool = True,
        show_boxes: bool = True,
        show_indices: bool = False,
        show_vis: bool = False,
        vis_threshold: float = 0.0,
        box_lw: float = 2.0,
        point_size: float = 20.0,
        assume_normalized: bool | None = None,  # None=auto, True=[0,1], False=pixels
    ):
        """
        Plot image with bbox (xyxy) and 5-point landmarks.

        - If assume_normalized is None, it auto-detects based on value ranges.
        - Missing landmarks are expected as -1 and will be skipped.
        - If landmark_vis exists and show_vis=True, will only plot points with vis>=vis_threshold (and vis>=0).
        """
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle

        # --------- get image (H,W,3) uint8 for display ----------
        img = self.img
        if img is None:
            raise ValueError("No image loaded. Provide `img` or `file`.")

        if hasattr(img, "detach"):  # torch tensor
            mean_t = torch.tensor(mean, dtype=img.dtype).view(-1, 1, 1)
            std_t = torch.tensor(std, dtype=img.dtype).view(-1, 1, 1)
            img = img * std_t + mean_t.clamp(0, 1)
            img_np = img.detach().cpu().numpy() * 255
            # allow CHW or HWC
            if img_np.ndim == 3 and img_np.shape[0] in (1, 3) and img_np.shape[-1] not in (1, 3):
                img_np = np.transpose(img_np, (1, 2, 0))
        else:
            img_np = np.asarray(img)

        img_np = img_np.astype(np.uint8)

        if img_np.ndim == 2:
            img_np = np.stack([img_np] * 3, axis=-1)
        if img_np.shape[-1] == 1:
            img_np = np.repeat(img_np, 3, axis=-1)

        # infer H,W if needed
        H, W = img_np.shape[:2]
        self.img_h, self.img_w = H, W

        # --------- prepare bbox/landmarks as numpy ----------
        self_np = self.as_numpy()

        b = self_np.bbox
        lm = self_np.landmark
        lmv = self_np.landmark_vis

        # --------- build axes ----------
        created_ax = False
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
            created_ax = True

        ax.imshow(img_np.astype(np.uint8))
        ax.set_axis_off()
        if title:
            ax.set_title(title)

        # --------- draw boxes ----------
        if show_boxes and b.size:
            for i, (x1, y1, x2, y2) in enumerate(b):
                # skip invalid
                if not np.isfinite([x1, y1, x2, y2]).all():
                    continue
                w_box, h_box = (x2 - x1), (y2 - y1)
                if w_box <= 0 or h_box <= 0:
                    continue
                rect = Rectangle((x1, y1), w_box, h_box, fill=False, linewidth=box_lw)
                ax.add_patch(rect)
                if show_indices:
                    ax.text(x1, y1, f"{i}", fontsize=10, va="top")

        # --------- draw landmarks (assumes 5 points per box if aligned) ----------
        if show_landmarks and lm.size:
            pts = lm.reshape(-1, 2)

            # If landmarks correspond to boxes: common layouts:
            # - stored as (num_boxes*5,2). We'll try to group by boxes if b exists.
            if b.shape[0] > 0 and pts.shape[0] == b.shape[0] * 5:
                pts = pts.reshape(b.shape[0], 5, 2)
                if lmv is not None and lmv.shape[0] == b.shape[0] * 5:
                    lmv_g = lmv.reshape(b.shape[0], 5)
                else:
                    lmv_g = None

                for bi in range(b.shape[0]):
                    for pi in range(5):
                        x, y = pts[bi, pi]
                        if x < 0 or y < 0:
                            continue
                        if lmv_g is not None and show_vis:
                            v = lmv_g[bi, pi]
                            if v < 0 or v < vis_threshold:
                                continue
                            ax.text(x + 2, y + 2, f"{v:.2f}", fontsize=8)
                        ax.scatter([x], [y], s=point_size)
                        if show_indices:
                            ax.text(x + 2, y - 2, f"{bi}:{pi}", fontsize=8)

            else:
                # fallback: plot all valid points as a flat list
                if lmv is not None and show_vis and lmv.shape[0] == pts.shape[0]:
                    mask = (pts[:, 0] >= 0) & (pts[:, 1] >= 0) & (lmv >= vis_threshold) & (lmv >= 0)
                else:
                    mask = (pts[:, 0] >= 0) & (pts[:, 1] >= 0)

                xs, ys = pts[mask, 0], pts[mask, 1]
                ax.scatter(xs, ys, s=point_size)

                if show_indices:
                    idxs = np.where(mask)[0]
                    for j, (x, y) in zip(idxs, pts[mask]):
                        ax.text(x + 2, y - 2, f"{j}", fontsize=8)

        if created_ax:
            plt.tight_layout()
            plt.show()

        return ax


class RetinaFaceDataModule(DataSetModule):
    def __init__(self, config: "DataModuleConfig", anchors=None, pos_iou=None, neg_iou=None, variances=None):
        super().__init__(config)
        self.anchors = anchors
        self.pos_iou = pos_iou
        self.neg_iou = neg_iou
        self.variances = variances
        self.num_classes = 100
        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)
        self.p = 0.5
        noise_std_range: Tuple[float, float] = (0.125, 0.25)
        rmin, rmax = (0.1, 0.2)
        self.train_tf = A.Compose(
            [
                A.Rotate(limit=(-15, 15), p=self.p),
                A.RandomResizedCrop((640, 640), scale=(0.6, 1.0)),
                A.HorizontalFlip(p=self.p),
                # A.Resize(640, 640),
                A.OneOf(
                    [
                        A.ColorJitter(p=1.0, brightness=0.3, contrast=0.3, saturation=0.3, hue=0.2),
                        A.ToGray(p=1.0),
                    ],
                    p=self.p,
                ),
                A.OneOf(
                    [
                        A.GaussianBlur(p=1.0, blur_limit=(1, 3)),
                        A.GaussNoise(p=1.0, std_range=noise_std_range),
                    ],
                    p=self.p,
                ),
                #  A.CoarseDropout(num_holes_range=(1,4),
                #                  hole_height_range=(rmin, rmax),
                #                  hole_width_range=(rmin, rmax), p=self.p),
                A.Normalize(mean=self.mean, std=self.std),
                A.ToTensorV2(),
            ],
            bbox_params=A.BboxParams(
                format="pascal_voc",
                label_fields=["bbox_ids"],
                min_area=1,  # drop boxes with area < 1 px^2
                min_visibility=0.0,  # or bump this to 0.1/0.2 to be stricter
                check_each_transform=True,
            ),
            keypoint_params=A.KeypointParams(
                format="xy",
                remove_invisible=False,  # keep keypoints even if they go outside; we'll handle them
            ),
        )
        # self.train_tf = v2.Compose([
        #     v2.RandomHorizontalFlip(p=self.p),

        #     v2.RandomApply(
        #         [v2.RandomChoice([
        #             v2.AutoAugment(
        #                 policy=v2.AutoAugmentPolicy.IMAGENET,
        #                 interpolation=InterpolationMode.BILINEAR,
        #             ),
        #             v2.RandAugment(num_ops=2, magnitude=9),
        #             v2.AugMix(severity=3, mixture_width=3, chain_depth=-1, alpha=1.0),
        #         ])],
        #         p=self.p
        #     ),
        #     v2.Resize((640,640)),
        #     ToTensor(mean=self.mean, std=self.std).build(),
        #     v2.Normalize(mean=self.mean, std=self.std),
        # ])

        self.val_tf = A.Compose(
            [
                # Resize so the LONG side becomes img_size (keeps aspect ratio)
                A.LongestMaxSize(max_size=640, interpolation=cv2.INTER_LINEAR),
                # Pad to exactly img_size x img_size (stackable)
                A.PadIfNeeded(
                    min_height=640,
                    min_width=640,
                    position="top_left",  # pads on right/bottom only
                    border_mode=cv2.BORDER_CONSTANT,
                    fill=(0, 0, 0),
                ),
                A.Normalize(mean=self.mean, std=self.std),
                A.ToTensorV2(),
            ],
            bbox_params=A.BboxParams(format="pascal_voc"),
            keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
        )
        self.dataset_cls = lambda **kwargs: RetinaFaceDataset(
            **kwargs,
            anchors=anchors,
            pos_iou=pos_iou,
            neg_iou=neg_iou,
            variances=variances,
        )

    def _build_collate_transform(self):
        def collate_fn(batch):
            x, y = zip(*batch)
            x = torch.stack(x, dim=0)
            return x, list(y)

        self._collate_transform = collate_fn

    def collate_fn(self, batch):
        if self._collate_transform is not None:
            x, y = self._collate_transform(batch)
        return x, y


class RetinaFaceDataset(VisionDataset):
    # Hugging Face mirror (default)
    HF_WIDER_REPO = "https://huggingface.co/datasets/wider_face/resolve/main/data"
    DEFAULT_WIDER_URLS = {
        "train": f"{HF_WIDER_REPO}/WIDER_train.zip",
        "val": f"{HF_WIDER_REPO}/WIDER_val.zip",
        "test": f"{HF_WIDER_REPO}/WIDER_test.zip",  # unused here (no gt)
        "split": f"{HF_WIDER_REPO}/wider_face_split.zip",
    }

    # RetinaFace GT zip (annotations with bbox + 5 landmarks) used by common training repos.
    # We try dl=1 first (direct download). If that fails, dl=0 can still work in some setups.
    RETINAFACE_GT_V1_1_URLS = [
        "https://drive.usercontent.google.com/u/0/uc?id=1tU_IjyOwGQfGNUvZGwWWM4SwxKp2PUQ8&export=download",
    ]
    RETINAFACE_GT_FILENAME = "retinaface_gt_v1.1.zip"

    def __init__(
        self,
        root: str,
        train: bool = True,
        download: bool = False,
        transform=None,
        target_transform=None,
        image_size: int = 640,  # kept for compatibility; unused internally
        annotation_file: Optional[str] = None,
        annotations_dir: str = "widerface",
        train_ann: str = "train.json",
        val_ann: str = "val.json",
        download_url: Optional[str] = None,  # optional override (single URL)
        anchors=None,
        pos_iou=None,
        neg_iou=None,
        variances=None,
    ):
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.anchors = anchors.cpu() if anchors is not None else None
        self.pos_iou = pos_iou
        self.neg_iou = neg_iou
        self.variances = variances
        self.root = Path(root)
        self.train = bool(train)
        self.download = bool(download)
        self.image_size = int(image_size)  # unused
        self.download_url = download_url
        self.annotations_dir = annotations_dir

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

    def __len__(self) -> int:
        return len(self.items)

    def _resolve_path(self, file_path: str) -> Path:
        p = Path(file_path)
        return p if p.is_absolute() else (self.root / p)

    @staticmethod
    def _convert_wider_txt_to_json(txt_path: Path, images_prefix_rel: Path, out_json: Path) -> None:
        """
        Convert official WIDER bbox GT txt into json:
          { "file": "...", "boxes": [[x1,y1,x2,y2], ...] }
        """
        items: List[Dict[str, Any]] = []
        IMG_EXTS = (".jpg", ".jpeg", ".png")

        def is_image_line(s: str) -> bool:
            s = s.lower()
            return s.endswith(IMG_EXTS)

        with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
            while True:
                img_rel = f.readline()
                if not img_rel:
                    break
                img_rel = img_rel.strip()
                if not img_rel:
                    continue

                # Guard: sometimes a dummy bbox line can be encountered here if parsing got off.
                if not is_image_line(img_rel):
                    continue

                n_line = f.readline()
                if not n_line:
                    break
                n_line = n_line.strip()

                try:
                    n = int(n_line)
                except ValueError:
                    n = 0

                boxes: List[List[float]] = []
                for _ in range(n):
                    line = f.readline()
                    if not line:
                        break
                    parts = line.strip().split()
                    if len(parts) < 4:
                        continue
                    x, y, w, h = map(float, parts[:4])
                    boxes.append([x, y, x + w, y + h])

                # IMPORTANT: WIDER sometimes has an extra dummy bbox row even when n == 0.
                if n == 0:
                    pos = f.tell()
                    maybe = f.readline()
                    if maybe:
                        maybe = maybe.strip()
                        if is_image_line(maybe):
                            f.seek(pos)

                file_rel = (images_prefix_rel / img_rel).as_posix()
                items.append({"file": file_rel, "boxes": boxes})

        out_json.parent.mkdir(parents=True, exist_ok=True)
        with open(out_json, "w", encoding="utf-8") as g:
            json.dump(items, g, ensure_ascii=False)

    @staticmethod
    def _convert_retinaface_label_to_json(
        label_txt: Path, images_prefix_rel: Path, out_json: Path
    ) -> None:
        """
        Convert RetinaFace train/label.txt format into json:

        {
            "file": "...",
            "boxes": [[x1,y1,x2,y2], ...],
            "landmarks": [[15 floats], ...],   # 5*(x,y,v)
            "scores": [float or null, ...]     # per-face final score (blur/quality) if present
        }

        Expected per-face row (common format):
        x y w h  (le_x le_y le_v) (re_x re_y re_v) (n_x n_y n_v) (lm_x lm_y lm_v) (rm_x rm_y rm_v)  score

        Visibility convention commonly used:
        0 = visible, 1 = invisible/occluded; sometimes -1 means missing.
        We store missing coords as -1, but we KEEP v as provided.
        """
        items: List[Dict[str, Any]] = []

        cur_img: Optional[str] = None
        cur_boxes: List[List[float]] = []
        cur_landmarks: List[List[float]] = []
        cur_scores: List[Optional[float]] = []

        def flush() -> None:
            nonlocal cur_img, cur_boxes, cur_landmarks, cur_scores
            if cur_img is None:
                return
            file_rel = (images_prefix_rel / cur_img).as_posix()
            items.append(
                {
                    "file": file_rel,
                    "boxes": cur_boxes,
                    "landmarks": cur_landmarks,
                    "scores": cur_scores,
                }
            )
            cur_img = None
            cur_boxes = []
            cur_landmarks = []
            cur_scores = []

        with open(label_txt, "r", encoding="utf-8", errors="ignore") as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    continue

                if line.startswith("#"):
                    flush()
                    cur_img = line.lstrip("#").strip()
                    continue

                if cur_img is None:
                    continue

                parts = line.split()
                if len(parts) < 4:
                    continue

                try:
                    x, y, w, h = map(float, parts[:4])
                except ValueError:
                    continue

                cur_boxes.append([x, y, x + w, y + h])

                # Default: 5*(x,y,v) all missing
                lm15 = [-1.0] * 15
                face_score: Optional[float] = None

                # Common case: bbox + 15 landmark vals (+ optional trailing score)
                if len(parts) >= 4 + 15:
                    # Parse the 15 landmark numbers
                    try:
                        vals15 = list(map(float, parts[4 : 4 + 15]))
                        for i in range(5):
                            px, py, vis = vals15[i * 3 : i * 3 + 3]

                            # Keep vis always; coords become -1 only if clearly missing
                            lm15[i * 3 + 2] = vis
                            if px >= 0.0 and py >= 0.0 and vis != -1.0:
                                lm15[i * 3 + 0] = px
                                lm15[i * 3 + 1] = py
                            else:
                                lm15[i * 3 + 0] = -1.0
                                lm15[i * 3 + 1] = -1.0
                    except ValueError:
                        pass

                    # Parse trailing score if present (immediately after the 15 vals)
                    if len(parts) >= 4 + 15 + 1:
                        try:
                            face_score = float(parts[4 + 15])
                        except ValueError:
                            face_score = None

                # Variant: bbox + 10 coords (+ optional trailing score), no vis flags
                elif len(parts) >= 4 + 10:
                    try:
                        vals10 = list(map(float, parts[4 : 4 + 10]))
                        for i in range(5):
                            px, py = vals10[i * 2 : i * 2 + 2]
                            # Unknown vis -> set to -1.0
                            lm15[i * 3 + 2] = -1.0
                            if px >= 0.0 and py >= 0.0:
                                lm15[i * 3 + 0] = px
                                lm15[i * 3 + 1] = py
                            else:
                                lm15[i * 3 + 0] = -1.0
                                lm15[i * 3 + 1] = -1.0
                    except ValueError:
                        pass

                    # Optional trailing score right after the 10 coords
                    if len(parts) >= 4 + 10 + 1:
                        try:
                            face_score = float(parts[4 + 10])
                        except ValueError:
                            face_score = None

                cur_landmarks.append(lm15)
                cur_scores.append(face_score)

        flush()

        out_json.parent.mkdir(parents=True, exist_ok=True)
        with open(out_json, "w", encoding="utf-8") as g:
            json.dump(items, g, ensure_ascii=False)

    def _download(self) -> None:
        """
        Default behavior (if download_url is None):
          - downloads WIDER_train.zip OR WIDER_val.zip depending on self.train
          - downloads wider_face_split.zip
          - extracts into: <root>/widerface/
          - tries to download retinaface_gt_v1.1.zip and build:
              train.json (bbox + 5 landmarks) from train/label.txt when available
              val.json from WIDER official bbox gt (landmarks absent) unless val/label.txt exists

        If download_url is provided, it is treated as a single archive/file override
        and no WIDER conversion is attempted.
        """

        # If already present, nothing to do.
        if self.ann_path.exists():
            return

        # If the user provided a custom URL override, do the simplest possible thing.
        if self.download_url:
            self.root.mkdir(parents=True, exist_ok=True)
            download_and_extract_archive(str(self.download_url), download_root=str(self.root))
            return

        urls = self.DEFAULT_WIDER_URLS

        wider_root = self.root / "widerface"
        wider_root.mkdir(parents=True, exist_ok=True)

        # Always need split to build fallback annotations
        split_dir = wider_root / "wider_face_split"
        if not split_dir.exists():
            download_and_extract_archive(urls["split"], download_root=str(wider_root))

        # Download only the image subset we need for this dataset instance
        subset = "train" if self.train else "val"
        subset_dir = wider_root / ("WIDER_train" if self.train else "WIDER_val")
        if not subset_dir.exists():
            download_and_extract_archive(urls[subset], download_root=str(wider_root))

        # Helpers
        def find_one(name: str) -> Path:
            hits = list(wider_root.rglob(name))
            if len(hits) != 1:
                raise FileNotFoundError(
                    f"Expected exactly 1 '{name}' under {wider_root}, found {len(hits)}."
                )
            return hits[0]

        def find_retinaface_label(which: str) -> Optional[Path]:
            # look for train/label.txt or val/label.txt anywhere under wider_root
            candidates = []
            for nm in ("label.txt", "labels.txt"):
                candidates.extend([p for p in wider_root.rglob(nm) if p.parent.name == which])
            if not candidates:
                return None
            # prefer paths that look like they came from retinaface_gt
            for p in candidates:
                if "retinaface_gt" in str(p).lower() or "retinaface" in str(p).lower():
                    return p
            return candidates[0]

        def ensure_retinaface_gt_present() -> None:
            # if we can already locate train label, assume extracted.
            if find_retinaface_label("train") is not None:
                return

            for url in self.RETINAFACE_GT_V1_1_URLS:
                try:
                    # IMPORTANT: override filename so query params don't break basename parsing.
                    download_url(url, root=str(wider_root), filename=self.RETINAFACE_GT_FILENAME)
                    archive_path = wider_root / self.RETINAFACE_GT_FILENAME
                    if archive_path.exists():
                        extract_archive(str(archive_path), to_path=str(wider_root))
                    # If extracted successfully, stop.
                    if find_retinaface_label("train") is not None:
                        return
                except Exception:
                    continue

        # Try to fetch RetinaFace GT (for train landmarks)
        ensure_retinaface_gt_present()
        train_label = find_retinaface_label("train")
        val_label = find_retinaface_label("val")

        if self.train:
            out_json = self.root / self.annotations_dir / "train.json"
            images_prefix_rel = Path("widerface") / "WIDER_train" / "images"
            if not out_json.exists():
                if train_label is not None:
                    self._convert_retinaface_label_to_json(train_label, images_prefix_rel, out_json)
                else:
                    # fallback: WIDER bbox only
                    gt_txt = find_one("wider_face_train_bbx_gt.txt")
                    self._convert_wider_txt_to_json(gt_txt, images_prefix_rel, out_json)
        else:
            out_json = self.root / self.annotations_dir / "val.json"
            images_prefix_rel = Path("widerface") / "WIDER_val" / "images"
            if not out_json.exists():
                if val_label is not None:
                    self._convert_retinaface_label_to_json(val_label, images_prefix_rel, out_json)
                else:
                    # fallback: WIDER bbox only (landmarks absent)
                    gt_txt = find_one("wider_face_val_bbx_gt.txt")
                    self._convert_wider_txt_to_json(gt_txt, images_prefix_rel, out_json)

        # Auto-switch ann_path if it matches default names.
        if self.ann_path.name in ("train.json", "val.json") and out_json.exists():
            self.ann_path = out_json

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        entry = self.items[index]
        img_path = self._resolve_path(entry["file"])
        target = RetinaFaceTensor(
            file=str(img_path),
            bbox=np.asarray(entry.get("boxes", []), dtype=np.float32).reshape(-1, 4),
            landmark=entry.get("landmarks", []),
        )
        img = target.img
        if self.transform is not None:
            if isinstance(self.transform, (BaseCompose, BasicTransform)):
                bbox_ids = np.arange(len(target.bbox))
                res = self.transform(
                    image=img,
                    bboxes=target.bbox,
                    keypoints=target.landmark,
                    bbox_ids=bbox_ids,
                )
                target.img = img = res["image"]
                target.bbox = res["bboxes"]
                target.landmark = res["keypoints"]
                bbox_ids = np.asarray(res["bbox_ids"]).astype(int)
                target.filter_landmarks(bbox_ids=bbox_ids)
                if self.anchors is not None:
                    target = target.prepare(self.anchors, self.pos_iou, self.neg_iou, self.variances)
            else:
                img = self.transform(Image.fromarray(img))
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

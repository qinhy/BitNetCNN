import math
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from torch.utils.data import DataLoader
from torch.utils.data import default_collate

from bitlayers.aug import MixupCutmix


class DataSetModule:
    def __init__(self, config: "DataModuleConfig"):
        self.data_dir = config.data_dir
        self.batch_size = config.batch_size
        self.num_workers = config.num_workers

        self.mixup = config.mixup
        self.cutmix = config.cutmix
        self.mix_alpha = config.mix_alpha

        self.train_tf = None
        self.p = 0.325
        self.val_tf = None
        self.dataset_cls = None  # must accept (root, train, download, transform)
        self.num_classes = -1

        self.train_ds = None
        self.val_ds = None

        # used by show_examples()
        self.mean = None
        self.std = None

        # collate-time transform (MixUp/CutMix)
        self._collate_transform = None

    def setup(self, stage: Optional[str] = None) -> None:
        if self.dataset_cls is None:
            raise ValueError("dataset_cls is not set")

        self.train_ds = self.dataset_cls(
            root=self.data_dir,
            train=True,
            download=True,
            transform=self.train_tf,
        )
        self.val_ds = self.dataset_cls(
            root=self.data_dir,
            train=False,
            download=True,
            transform=self.val_tf,
        )

        self._build_collate_transform()

    def _build_collate_transform(self):
        self._collate_transform = MixupCutmix(
            num_classes=self.num_classes,
            enable_mixup=self.mixup,
            enable_cutmix=self.cutmix,
            beta_alpha=self.mix_alpha,
            p=self.p,
        ).build()

    def collate_fn(self, batch):
        x, y = default_collate(batch)
        if self._collate_transform is not None:
            x, y = self._collate_transform(x, y)
        return x, y

    def train_dataloader(self) -> DataLoader:
        collate_fn = self.collate_fn if self._collate_transform is not None else None
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=(self.num_workers > 0),
            collate_fn=collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        collate_fn = self.collate_fn if self._collate_transform is not None else None
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=(self.num_workers > 0),
            collate_fn=collate_fn,
        )

    @staticmethod
    def show_examples_static(
        x,
        y,
        mean,
        std,
        class_names=None,
        n: int = 16,
        cols: int = 8,
        figsize=(12, 6),
    ):
        # ---- handle x being Tensor or list of Tensors (variable-sized images) ----
        if torch.is_tensor(x):
            x = x[:n].detach().cpu()
            x_list = [x[i] for i in range(x.shape[0])]
        else:
            x_list = [xi.detach().cpu() for xi in list(x)[:n]]

        n = min(n, len(x_list))
        x_list = x_list[:n]

        # ---- move y to CPU if possible ----
        y_is_list = isinstance(y, (list, tuple))
        if y_is_list:
            y_list = list(y)[:n]
        else:
            try:
                y_list = y[:n].detach().cpu()
            except Exception:
                y_list = None

        # ---- dataset class names (if classification) ----

        def label_to_text(target):
            # for classification display
            if torch.is_tensor(target):
                if target.ndim == 0:
                    idx = int(target.item())
                    return class_names[idx] if class_names else str(idx)
                if target.ndim == 1:
                    topv, topi = torch.topk(target, k=min(2, target.numel()))
                    parts = []
                    for v, i in zip(topv, topi):
                        if float(v) <= 1e-3:
                            continue
                        name = class_names[int(i)] if class_names else str(int(i))
                        parts.append(f"{name}:{float(v):.2f}")
                    return " | ".join(parts) if parts else "mixed"
            return str(target)

        def denorm_for_vis(img_t: torch.Tensor, mean=mean, std=std) -> torch.Tensor:
            # img_t: (C,H,W)
            if not torch.is_tensor(img_t) or img_t.ndim != 3:
                return img_t
            # if already uint8, just scale to [0,1] for matplotlib
            if img_t.dtype == torch.uint8:
                return (img_t.float() / 255.0).clamp(0, 1)

            mean_t = torch.tensor(mean, dtype=img_t.dtype).view(-1, 1, 1)
            std_t = torch.tensor(std, dtype=img_t.dtype).view(-1, 1, 1)
            return (img_t * std_t + mean_t).clamp(0, 1)

        def is_retina_target(t) -> bool:
            # supports RetinaFaceTensor or dict with bbox/landmark
            if t is None:
                return False
            if hasattr(t, "bbox") and hasattr(t, "landmark"):
                return True
            if isinstance(t, dict) and ("bbox" in t) and ("landmark" in t):
                return True
            return False

        # ---- decide whether this is detection-style batch ----
        detection = False
        if y_is_list and len(y_list) > 0 and is_retina_target(y_list[0]):
            detection = True

        cols = max(1, min(cols, n))
        rows = math.ceil(n / cols)
        plt.figure(figsize=figsize)

        for i in range(n):
            plt.subplot(rows, cols, i + 1)

            img_vis = denorm_for_vis(x_list[i])
            img_np = img_vis.permute(1, 2, 0).numpy()
            H, W = img_np.shape[:2]

            plt.imshow(img_np)
            plt.axis("off")

            title = "null"
            if detection and y_list is not None:
                t = y_list[i]
                if is_retina_target(t):
                    boxes = torch.as_tensor(t.bbox).detach().cpu().view(-1, 4)

                    # draw boxes
                    for b in boxes:
                        x1, y1, x2, y2 = [float(v) for v in b.tolist()]
                        w = max(0.0, x2 - x1)
                        h = max(0.0, y2 - y1)
                        plt.gca().add_patch(Rectangle((x1, y1), w, h, fill=False, linewidth=1.0))

                    xs, ys = t.landmark_xs_ys()
                    if xs.size > 0:
                        plt.scatter(xs, ys, s=6)

                    title = f"faces:{boxes.shape[0]}"

            elif y_list is not None:
                # classification display
                title = label_to_text(y_list[i])

            plt.title(title, fontsize=8)

        plt.tight_layout()
        plt.show()

        # return something consistent with your old function
        if torch.is_tensor(x):
            x_out = x[:n]
        else:
            x_out = x_list
        return x_out, y_list

    @torch.no_grad()
    def show_examples(
        self,
        n: int = 16,
        split: str = "train",
        cols: int = 8,
        seed: int | None = None,
        figsize=(12, 6),
    ):
        """
        Randomly show examples from train/val.
        - If split='train', MixUp/CutMix will be shown if enabled (applied in collate_fn).
        """
        assert split in {"train", "val"}, "split must be 'train' or 'val'"
        if seed is not None:
            torch.manual_seed(seed)

        assert self.train_ds is not None and self.val_ds is not None, "Call setup() first."
        loader = self.train_dataloader() if split == "train" else self.val_dataloader()

        batch = next(iter(loader))
        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
            x, y = batch[0], batch[1]
        else:
            raise ValueError("Dataloader must return (x, y)")
        ds = self.train_ds if split == "train" else self.val_ds
        self.show_examples_static(
            x,
            y,
            self.mean,
            self.std,
            getattr(ds, "classes", None),
            n,
            cols,
            figsize,
        )

    # -----------------------------
    # Validation (Classification)
    # -----------------------------
    @staticmethod
    def _unwrap_model_output(output):
        """Handle models that return (logits, ...) or {'logits': ...}."""
        if isinstance(output, (tuple, list)):
            return output[0]
        if isinstance(output, dict):
            return (
                output.get("logits", output.get("output", None))
                if ("logits" in output or "output" in output)
                else next(iter(output.values()))
            )
        return output

    @staticmethod
    def _soft_target_cross_entropy(logits: torch.Tensor, soft_targets: torch.Tensor) -> torch.Tensor:
        """
        Cross-entropy when targets are soft distributions (e.g., from MixUp/CutMix).
        soft_targets: [B, C], floats summing ~1 across C.
        """
        log_probs = torch.log_softmax(logits, dim=1)
        return -(soft_targets * log_probs).sum(dim=1).mean()

    @torch.no_grad()
    def validate(
        self,
        model: nn.Module,
        criterion: nn.Module,
        device: Optional[torch.device] = None,
        amp: bool = False,
        topk: Tuple[int, ...] = (1,),
        split: str = "val",
        max_batches: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Validate a classification model on hard labels.
        Returns: {'val_loss': ..., 'top1_acc': ..., ...}
        """
        # If you NEVER want train validation (because it may be mixup/cutmix),
        # uncomment the next line:
        # assert split == "val", "This validate() expects hard labels; use split='val' only."

        assert split in {"train", "val"}, "split must be 'train' or 'val'"

        model.eval()
        if device is None:
            device = next(model.parameters()).device

        loader = self.val_dataloader() if split == "val" else self.train_dataloader()

        total_loss = 0.0
        total_samples = 0
        correct_k = {k: 0.0 for k in topk}

        autocast_enabled = amp and (device.type == "cuda")

        for b_idx, (x, y) in enumerate(loader):
            if max_batches is not None and b_idx >= max_batches:
                break

            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)  # expected shape [B], dtype long for CE

            # AMP only on CUDA
            if device.type == "cuda":
                autocast_ctx = torch.amp.autocast(device_type="cuda", enabled=autocast_enabled)
            else:
                autocast_ctx = torch.amp.autocast(device_type="cpu", enabled=False)

            with autocast_ctx:
                out = model(x)
                logits = self._unwrap_model_output(out)
                loss = criterion(logits, y)

            bs = x.size(0)
            total_loss += float(loss.item()) * bs
            total_samples += bs

            # Top-k accuracy
            max_k = max(topk)
            _, pred = logits.topk(max_k, dim=1, largest=True, sorted=True)  # [B, max_k]
            pred = pred.t()  # [max_k, B]
            correct = pred.eq(y.view(1, -1).expand_as(pred))

            for k in topk:
                correct_k[k] += float(correct[:k].reshape(-1).float().sum().item())

        denom = max(total_samples, 1)
        metrics = {"val_loss": total_loss / denom}
        for k in topk:
            metrics[f"top{k}_acc"] = correct_k[k] / denom
        return metrics

    # -----------------------------
    # Validation (Regression)
    # -----------------------------
    @torch.no_grad()
    def validate_regression(
        self,
        model: nn.Module,
        criterion: nn.Module,
        device: Optional[torch.device] = None,
        amp: bool = True,
        split: str = "val",
        max_batches: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Validate a regression model.
        Returns dict: {'val_loss': ..., 'mae': ..., 'rmse': ...}
        """
        assert split in {"train", "val"}, "split must be 'train' or 'val'"

        model.eval()
        if device is None:
            device = next(model.parameters()).device

        loader = self.val_dataloader() if split == "val" else self.train_dataloader()

        total_loss = 0.0
        total_abs_err = 0.0
        total_sq_err = 0.0
        total_samples = 0

        autocast_enabled = amp and (device.type == "cuda")

        for b_idx, (x, y) in enumerate(loader):
            if max_batches is not None and b_idx >= max_batches:
                break

            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=autocast_enabled):
                out = model(x)
                preds = self._unwrap_model_output(out)
                loss = criterion(preds, y)

            bs = x.size(0)
            total_loss += float(loss.item()) * bs
            total_samples += bs

            preds_f = preds.detach().view(bs, -1)
            y_f = y.detach().view(bs, -1)

            abs_err = (preds_f - y_f).abs().mean(dim=1)  # per-sample MAE
            sq_err = ((preds_f - y_f) ** 2).mean(dim=1)  # per-sample MSE

            total_abs_err += float(abs_err.sum().item())
            total_sq_err += float(sq_err.sum().item())

        denom = max(total_samples, 1)
        mae = total_abs_err / denom
        rmse = (total_sq_err / denom) ** 0.5

        return {"val_loss": total_loss / denom, "mae": mae, "rmse": rmse}

import os
import math
import copy
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple, Union

from pydantic import BaseModel, Field, PrivateAttr

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from accelerate import Accelerator
from accelerate.utils import gather_object, broadcast_object_list
from tqdm.auto import tqdm

from bitlayers.bit import Bit
from common_utils import DataModuleConfig, DataSetModule

torch.set_float32_matmul_precision("high")


# ----------------------------
# Model-wide conversion helpers
# ----------------------------
@torch.no_grad()
def convert_to_ternary(module: nn.Module) -> nn.Module:
    """
    Recursively replace Bit.Conv2d/Bit.Linear with Ternary*Infer modules.
    Returns the mutated module.
    """
    for name, child in list(module.named_children()):
        if hasattr(child, "to_ternary"):
            setattr(module, name, child.to_ternary())
        else:
            convert_to_ternary(child)
    return module


# ----------------------------
# AccelLightningModule
# ----------------------------
class AccelLightningModule(nn.Module):
    """
    Lightning-like interface:
      - forward(x)
      - training_step(batch, batch_idx) -> Metrics
      - validation_step(batch, batch_idx) -> Metrics
      - configure_optimizers() -> (optimizer, scheduler or None, scheduler_interval: "epoch"|"step")
    """

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> "Metrics":
        raise NotImplementedError

    @torch.no_grad()
    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> "Metrics":
        return Metrics(stage="val")

    def configure_optimizers(self):
        raise NotImplementedError

    # optional hooks
    def on_fit_start(self, accelerator: Accelerator): ...
    def on_train_epoch_start(self, epoch: int): ...
    def on_train_epoch_end(self, epoch: int): ...
    def on_validation_epoch_start(self, epoch: int): ...
    def on_validation_epoch_end(self, epoch: int): ...


# ----------------------------
# Metrics
# ----------------------------
class Metrics(BaseModel):
    stage: Literal["train", "val"] = "train"
    epoch: int = -1
    step: Optional[int] = None
    batch_size: Optional[int] = None

    loss: Optional[Any] = Field(default=None, exclude=True)
    metrics: Dict[str, Any] = Field(default_factory=dict)

    def set(self, key: str, value: float) -> None:
        self.metrics[key] = float(value)

    def get(self, key: str, default: Optional[float] = None) -> Any:
        return self.metrics.get(key, default)

    def to_dict(self) -> Dict[str, Any]:
        res: Dict[str, Any] = {}
        res[f"{self.stage}/loss"] = self.loss
        res.update(self.metrics)

        out: Dict[str, Any] = {}
        for k, v in res.items():
            if v is None:continue
            if torch.is_tensor(v):
                out[k] = float(v.detach().cpu().item())
            else:
                out[k] = v
        return out


def _as_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    if torch.is_tensor(x):
        x = x.detach().float().cpu().item()
    try:
        x = float(x)
    except Exception:
        return None
    if math.isnan(x):
        return None
    return x


class MetricsTracer(BaseModel):
    stage: Literal["train", "val"]
    key: str
    best: Optional[float] = None
    mode: Literal["max", "min"] = "max"
    callback: Optional[Callable[..., None]] = Field(default=None, exclude=True)

    def is_better(self, val: float) -> bool:
        if self.best is None:
            return True
        return (val > self.best) if self.mode == "max" else (val < self.best)

    def compare(self, val: Any) -> bool:
        v = _as_float(val)
        if v is None:
            return False
        if self.is_better(v):
            self.best = float(v)
            return True
        return False


class MetricsManager(BaseModel):
    metrics_list: List[Metrics] = Field(default_factory=list)
    epoch_metric_tracers: List[MetricsTracer] = Field(default_factory=list)
    _last_epoch_by_stage: Dict[str, int] = PrivateAttr(default_factory=lambda: {"train": -1, "val": -1})

    def to_list(self):
        return [m.to_dict() for m in self.metrics_list]
    
    def log(self, metrics: Metrics, trainer: "AccelTrainer" = None, model: "LitBit" = None) -> None:
        self.metrics_list.append(metrics)

        last_epoch = self._last_epoch_by_stage.get(metrics.stage, -1)
        if metrics.epoch > last_epoch:
            self._last_epoch_by_stage[metrics.stage] = metrics.epoch

        for t in self.epoch_metric_tracers:
            if t.stage != metrics.stage:
                continue
            val = metrics.get(t.key)
            if t.compare(val):
                if t.callback is not None:
                    t.callback(metrics=metrics, trainer=trainer, model=model)

    def last_epoch_mean(self, stage: str = "train", epoch: int = -1) -> Metrics:
        last_epoch = self._last_epoch_by_stage.get(stage, -1)
        if last_epoch < 0:
            return Metrics(stage=stage, epoch=epoch, loss=None, metrics={})

        rows = [m for m in self.metrics_list if m.stage == stage and m.epoch == last_epoch]
        if not rows:
            return Metrics(stage=stage, epoch=epoch, loss=None, metrics={})

        dict_rows = [r.to_dict() for r in rows]
        keys = sorted({k for d in dict_rows for k in d.keys()})

        mean_metrics: Dict[str, float] = {}
        for k in keys:
            vals = [_as_float(d.get(k)) for d in dict_rows]
            vals = [v for v in vals if v is not None]
            if vals:
                mean_metrics[f"{k}_mean"] = sum(vals) / len(vals)

        return Metrics(stage=stage, epoch=epoch, loss=None, metrics=mean_metrics)


# ----------------------------
# Export best ternary checkpoint
# ----------------------------
class ExportBestTernary:
    def __call__(self, metrics: Metrics, trainer: "AccelTrainer", model: "LitBit"):
        # avoid multi-proc races
        if not trainer.accelerator.is_local_main_process:
            return

        acc_tern = _as_float(metrics.get("val/acc_tern_mean"))
        if acc_tern is None:
            return

        # unwrap accelerate wrapper
        unwrapped:LitBit = trainer.accelerator.unwrap_model(model)

        config = unwrapped.config
        export_dir = config.export_dir
        dataset_name = config.dataset.dataset_name
        model_name = config.model_name
        model_size = config.model_size

        os.makedirs(export_dir, exist_ok=True)

        # save FP student
        best_fp = copy.deepcopy(unwrapped.student).cpu()
        fp_path = os.path.join(export_dir, f"bit_{model_name}_{model_size}_{dataset_name}_best_fp.pt")
        torch.save({"model": best_fp.state_dict(), "acc_tern": acc_tern}, fp_path)
        trainer.print(f"[OK] saved {fp_path} (val/acc_tern={acc_tern*100:.2f}%)")

        # save ternary PoT export
        tern = convert_to_ternary(copy.deepcopy(best_fp)).cpu()
        tern_path = os.path.join(
            export_dir,
            f"bit_{model_name}_{model_size}_{dataset_name}_ternary_val_acc@{acc_tern*100:.2f}.pt",
        )
        torch.save({"model": tern.state_dict(), "acc_tern": acc_tern}, tern_path)
        trainer.print(f"[OK] exported ternary PoT -> {tern_path}")


# ----------------------------
# AccelTrainer
# ----------------------------


class AccelTrainer:
    def __init__(
        self,
        max_epochs: int,
        mixed_precision: str = "no",  # "no" | "fp16" | "bf16"
        gradient_accumulation_steps: int = 1,
        max_grad_norm: Optional[float] = None,
        show_progress_bar: bool = True,
        metrics_manager: Optional["MetricsManager"] = None,
        log_every_n_steps: int = 10,
    ):
        self.max_epochs = int(max_epochs)
        self.accelerator = Accelerator(
            mixed_precision=mixed_precision,
            gradient_accumulation_steps=int(gradient_accumulation_steps),
        )
        self.max_grad_norm = max_grad_norm
        self.show_progress_bar = show_progress_bar

        self.model: Optional["AccelLightningModule"] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
        self.scheduler_interval: str = "epoch"

        # ✅ avoid shared mutable defaults across trainer instances
        if metrics_manager is None:
            metrics_manager = MetricsManager(
                epoch_metric_tracers=[
                    MetricsTracer(
                        stage="val",
                        key="val/acc_tern_mean",
                        mode="max",
                        callback=ExportBestTernary(),
                    )
                ]
            )
        self.metrics_manager = metrics_manager

        self.log_every_n_steps = int(log_every_n_steps)

        self.last_train_summary: Optional["Metrics"] = None
        self.last_val_summary: Optional["Metrics"] = None

    # -----------------------------
    # Public API
    # -----------------------------
    def fit(
        self,
        model: "AccelLightningModule",
        datamodule: Optional["DataSetModule"] = None,
        train_dataloader: Optional[DataLoader] = None,
        val_dataloader: Optional[DataLoader] = None,
    ):
        train_dataloader, val_dataloader = self._resolve_dataloaders(datamodule, train_dataloader, val_dataloader)

        optimizer, scheduler, interval = self._configure_optimizers(model)
        self.scheduler_interval = interval or "epoch"

        model, optimizer, train_dataloader, val_dataloader, scheduler = self._prepare(
            model, optimizer, train_dataloader, val_dataloader, scheduler
        )

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler

        model.on_fit_start(self.accelerator)

        for epoch in range(self.max_epochs):
            self._train_one_epoch(epoch, train_dataloader)
            if val_dataloader is not None:
                self._validate_one_epoch(epoch, val_dataloader)
            if self.scheduler is not None and self.scheduler_interval == "epoch":
                self.scheduler.step()

        return self.metrics_manager.to_list()

    # -----------------------------
    # Fit helpers
    # -----------------------------
    def _resolve_dataloaders(
        self,
        datamodule: Optional["DataSetModule"],
        train_dataloader: Optional[DataLoader],
        val_dataloader: Optional[DataLoader],
    ) -> Tuple[DataLoader, Optional[DataLoader]]:
        if datamodule is not None:
            datamodule.setup("fit")
            train_dataloader = datamodule.train_dataloader()
            val_dataloader = datamodule.val_dataloader()

        if train_dataloader is None:
            raise ValueError("Need train_dataloader or datamodule")

        return train_dataloader, val_dataloader

    def _configure_optimizers(
        self,
        model: "AccelLightningModule",
    ) -> Tuple[torch.optim.Optimizer, Optional[torch.optim.lr_scheduler._LRScheduler], str]:
        optimizer, scheduler, interval = model.configure_optimizers()
        if optimizer is None:
            raise ValueError("model.configure_optimizers() must return an optimizer")
        return optimizer, scheduler, (interval or "epoch")

    def _prepare(
        self,
        model: "AccelLightningModule",
        optimizer: torch.optim.Optimizer,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader],
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    ):
        objs = [model, optimizer, train_dataloader]
        has_val = val_dataloader is not None
        has_sched = scheduler is not None

        if has_val:
            objs.append(val_dataloader)
        if has_sched:
            objs.append(scheduler)

        prepared = self.accelerator.prepare(*objs)

        idx = 0
        model_p = prepared[idx]; idx += 1
        optim_p = prepared[idx]; idx += 1
        train_p = prepared[idx]; idx += 1

        val_p = None
        if has_val:
            val_p = prepared[idx]; idx += 1

        sched_p = None
        if has_sched:
            sched_p = prepared[idx]; idx += 1

        return model_p, optim_p, train_p, val_p, sched_p

    # -----------------------------
    # Logging / tqdm
    # -----------------------------
    def print(self, msg: str) -> None:
        if self.accelerator.is_local_main_process:
            tqdm.write(msg)

    def log_step(
        self,
        stage: str,
        epoch: int,
        step: int,
        metrics: "Metrics",
        model,
        pbar: tqdm,
        force_print: bool = False,
    ):
        if not self.accelerator.is_local_main_process:
            return

        self.metrics_manager.log(metrics, self, model)

        postfix = metrics.to_dict()
        if hasattr(pbar, "set_postfix"):
            pbar.set_postfix(postfix, refresh=False)

        if (step % self.log_every_n_steps == 0) or force_print:
            self.print(f"[{stage}] epoch={epoch} step={step} {postfix}")

    def tqdm(self, data_loader, stage: str, epoch: int):
        is_main = self.accelerator.is_local_main_process
        return tqdm(data_loader, desc=f"{stage} {epoch+1}", leave=False, dynamic_ncols=True) if (
            self.show_progress_bar and is_main
        ) else data_loader

    # -----------------------------
    # Utilities
    # -----------------------------
    def _infer_batch_size(self, batch: Any) -> int:
        if isinstance(batch, (tuple, list)) and len(batch) > 0 and hasattr(batch[0], "size"):
            return int(batch[0].size(0))
        if isinstance(batch, dict):
            for v in batch.values():
                if hasattr(v, "size"):
                    return int(v.size(0))
        return 1

    def _to_device_scalar(self, v: Any) -> Optional[torch.Tensor]:
        if v is None:
            return None
        if torch.is_tensor(v):
            t = v.detach()
            if t.numel() != 1:
                t = t.float().mean()
            else:
                t = t.float()
            return t.to(self.accelerator.device)
        try:
            return torch.tensor(float(v), device=self.accelerator.device, dtype=torch.float32)
        except Exception:
            return None

    def _accum_weighted(self, sums: Dict[str, torch.Tensor], metrics: "Metrics", batch_size: int, stage: str) -> torch.Tensor:
        bs = torch.tensor(float(batch_size), device=self.accelerator.device, dtype=torch.float32)

        if metrics.loss is not None:
            l = self._to_device_scalar(metrics.loss)
            if l is not None:
                k = f"{stage}/loss"
                sums[k] = sums.get(k, torch.zeros((), device=self.accelerator.device)) + l * bs

        for k, v in metrics.metrics.items():
            t = self._to_device_scalar(v)
            if t is None:
                continue
            sums[k] = sums.get(k, torch.zeros((), device=self.accelerator.device)) + t * bs

        return bs

    def _union_metric_keys_across_ranks(self, local_keys: List[str]) -> List[str]:
        """
        Ensures every rank reduces the same set of metric keys, even if some ranks
        didn't log certain keys in an epoch.
        """
        local_keys = list(local_keys)

        if self.accelerator.num_processes == 1:
            return sorted(set(local_keys))

        gathered = gather_object(local_keys)

        # accelerate.utils.gather_object() often returns a *flat* list across ranks for lists,
        # but we defensively handle nested structures too.
        if gathered is None:
            gathered_keys: List[str] = local_keys
        elif isinstance(gathered, (list, tuple)):
            if len(gathered) > 0 and isinstance(gathered[0], (list, tuple)):
                gathered_keys = [k for sub in gathered for k in sub]
            else:
                gathered_keys = list(gathered)
        else:
            gathered_keys = [str(gathered)]

        union = sorted(set(map(str, gathered_keys)))

        # Make ordering identical on all ranks (safe even if union already matches)
        obj = [union]
        broadcast_object_list(obj, from_process=0)
        return obj[0]

    def _reduce_epoch_means(self, stage: str, epoch: int, sums: Dict[str, torch.Tensor], n: torch.Tensor) -> "Metrics":
        n_tot = self.accelerator.reduce(n, reduction="sum")

        if float(n_tot.item()) <= 0:
            return Metrics(stage=stage, epoch=epoch, step=None, batch_size=0, loss=None, metrics={})

        # ✅ union keys across processes so we don't "drop" metrics that only appeared on some ranks
        keys = self._union_metric_keys_across_ranks(list(sums.keys()))

        out: Dict[str, Any] = {}
        zero = torch.zeros((), device=self.accelerator.device)

        for k in keys:
            s = sums.get(k, zero)
            s_tot = self.accelerator.reduce(s, reduction="sum")
            out[f"{k}_mean"] = (s_tot / n_tot).detach()

        return Metrics(stage=stage, epoch=epoch, step=None, batch_size=int(n_tot.item()), loss=None, metrics=out)

    def _maybe_set_dataloader_epoch(self, loader: Any, epoch: int) -> None:
        """
        Supports both standard PyTorch DistributedSampler and Accelerate's wrapped dataloaders.
        """
        if hasattr(loader, "set_epoch"):
            try:
                loader.set_epoch(epoch)
                return
            except TypeError:
                pass

        sampler = getattr(loader, "sampler", None)
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)

    # -----------------------------
    # Training / Validation
    # -----------------------------
    def _train_one_epoch(self, epoch: int, train_loader: DataLoader) -> None:
        assert self.model is not None
        assert self.optimizer is not None

        self.model.on_train_epoch_start(epoch)

        # ✅ correct shuffling for DistributedSampler / wrapped loaders
        self._maybe_set_dataloader_epoch(train_loader, epoch)

        model = self.model.train()
        stage = "train"
        pbar = self.tqdm(train_loader, stage, epoch)

        sums: Dict[str, torch.Tensor] = {}
        n = torch.zeros((), device=self.accelerator.device)

        step = -1
        for step, batch in enumerate(pbar):
            bs = self._infer_batch_size(batch)

            with self.accelerator.accumulate(model):
                m = model.training_step(batch, step).model_copy(
                    update=dict(stage=stage, epoch=epoch, step=step, batch_size=bs)
                )

                self.log_step(stage=stage, epoch=epoch, step=step, metrics=m, model=model, pbar=pbar)

                # global epoch mean accumulation
                n = n + self._accum_weighted(sums, m, bs, stage)

                if m.loss is None:
                    raise ValueError("training_step must return Metrics with loss (got loss=None)")
                self.accelerator.backward(m.loss)

                if self.accelerator.sync_gradients:
                    if self.max_grad_norm is not None:
                        self.accelerator.clip_grad_norm_(model.parameters(), self.max_grad_norm)

                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)

                    if self.scheduler is not None and self.scheduler_interval == "step":
                        self.scheduler.step()

        summary = self._reduce_epoch_means(stage=stage, epoch=epoch, sums=sums, n=n)

        self.log_step(
            stage=stage,
            epoch=epoch,
            step=max(step, 0),
            metrics=summary,
            model=model,
            pbar=pbar,
            force_print=True,
        )

        model.on_train_epoch_end(epoch)
        return summary

    @torch.no_grad()
    def _validate_one_epoch(self, epoch: int, val_loader: DataLoader) -> "Metrics":
        assert self.model is not None
        model = self.model.eval()
        stage = "val"

        model.on_validation_epoch_start(epoch)

        pbar = self.tqdm(val_loader, stage, epoch)

        sums: Dict[str, torch.Tensor] = {}
        n = torch.zeros((), device=self.accelerator.device)

        step = -1
        for step, batch in enumerate(pbar):
            bs = self._infer_batch_size(batch)

            m = model.validation_step(batch, step).model_copy(
                update=dict(stage=stage, epoch=epoch, step=step, batch_size=bs)
            )

            self.log_step(stage=stage, epoch=epoch, step=step, metrics=m, model=model, pbar=pbar)

            n = n + self._accum_weighted(sums, m, bs, stage)

        summary = self._reduce_epoch_means(stage=stage, epoch=epoch, sums=sums, n=n)

        self.log_step(
            stage=stage,
            epoch=epoch,
            step=max(step, 0),
            metrics=summary,
            model=model,
            pbar=pbar,
            force_print=True,
        )

        model.on_validation_epoch_end(epoch)
        return summary
    
# ----------------------------
# KD + hints + ternary eval/export
# ----------------------------
class CommonTrainConfig(BaseModel):
    data_dir: str = "./data"
    export_dir: Optional[str] = ""
    dataset_name: Literal["c100", "imnet", "timnet"] = Field(
        default="c100", description="Dataset to use (affects stems, classes, transforms)"
    )

    epochs: int = Field(200, ge=1)
    batch_size: int = Field(512, ge=1)

    lr: float = Field(2e-1, gt=0)
    wd: float = Field(5e-4, ge=0)

    label_smoothing: float = Field(0.1, ge=0.0, le=1.0)
    alpha_kd: float = 0.3
    alpha_hint: float = 0.05
    T: float = Field(4.0, gt=0)

    scale_op: Literal["mean", "median"] = "median"

    amp: bool = False
    cpu: bool = False
    mixup: bool = False
    cutmix: bool = False

    mix_alpha: float = Field(1.0, ge=0.0)

    seed: int = 42
    gpus: int = Field(1, description="Number of GPUs to use (1 = default, -1 = all available)")
    strategy: Literal["auto", "ddp", "ddp_spawn", "fsdp"] = "auto"


class LitBitConfig(BaseModel):
    dataset: Optional[DataModuleConfig] = None
    lr: float
    wd: float
    epochs: int

    label_smoothing: float = 0.1
    alpha_kd: float = 0.7
    alpha_hint: float = 0.05
    T: float = 4.0
    scale_op: str = "median"

    width_mult: float = 1.0
    amp: bool = True
    export_dir: str = "./ckpt_c100_mbv2"

    student: Optional[Any] = None
    teacher: Optional[Any] = None

    model_name: str = ""
    model_size: str = ""

    hint_points: List[Union[str, Tuple]] = Field(default_factory=list)


class LitBit(AccelLightningModule):
    def __init__(self, config: LitBitConfig):
        super().__init__()
        if type(config) is not dict:
            config = config.model_dump()
        self.config = config = LitBitConfig.model_validate(config)

        # --- core ---
        self.scale_op = config.scale_op
        self.student: nn.Module = config.student
        self.teacher: Optional[nn.Module] = config.teacher
        self.has_teacher = True if config.teacher is not None else False

        # --- metadata ---
        self.dataset_name = config.dataset.dataset_name
        self.model_name = config.model_name
        self.model_size = config.model_size
        self.num_classes = config.dataset.num_classes

        self.alpha_kd = float(config.alpha_kd)
        self.alpha_hint = float(config.alpha_hint)

        # --- CE selection (hard vs soft labels) ---
        mixup = bool(getattr(config.dataset, "mixup", False)) if config.dataset is not None else False
        cutmix = bool(getattr(config.dataset, "cutmix", False)) if config.dataset is not None else False

        if not (mixup or cutmix):
            self.ce_hard = nn.CrossEntropyLoss(label_smoothing=float(config.label_smoothing))
            self.ce_soft = None
        else:
            self.ce_hard = None
            self.ce_soft = nn.CrossEntropyLoss()

        # --- KD / Hint ---
        self.kd = KDLoss(T=float(config.T))
        self.hint = AdaptiveHintLoss()

        if not (self.alpha_kd > 0 and self.has_teacher):
            self.kd = None
        if not (self.alpha_hint > 0 and self.has_teacher):
            self.hint = None
        if self.kd is None and self.hint is None:
            self.has_teacher = False

        # --- hint plumbing ---
        self.hint_points = list(config.hint_points)
        self._t_feats: Dict[str, torch.Tensor] = {}
        self._s_feats: Dict[str, torch.Tensor] = {}
        self._t_handles = []
        self._s_handles = []

        # --- ternary snapshot & teacher acc cache ---
        self._ternary_snapshot: Optional[nn.Module] = None
        self.t_acc_fps: Dict[int, float] = {}

        # --- optim ---
        self.lr = float(config.lr)
        self.wd = float(config.wd)
        self.epochs = int(config.epochs)

        # --- teacher freeze / hint init ---
        if self.has_teacher:
            if self.hint is not None and len(self.hint_points) > 0:
                self.init_hint()
            for p in self.teacher.parameters():
                p.requires_grad_(False)
        else:
            self.teacher = None
            self.kd = None
            self.hint = None
            self.alpha_kd = 0.0
            self.alpha_hint = 0.0

        self._accel = None  # set in on_fit_start

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.student(x)

    @torch.no_grad()
    def on_fit_start(self, accelerator: Accelerator):
        self._accel = accelerator

        if self.has_teacher and self.teacher is not None:
            self.teacher.eval()
            for p in self.teacher.parameters():
                p.requires_grad_(False)

            if self.alpha_hint > 0 and len(self.hint_points) > 0:
                self._s_handles = make_feature_hooks(self.student, self.hint_points, self._s_feats, idx=0)
                self._t_handles = make_feature_hooks(self.teacher, self.hint_points, self._t_feats, idx=1)

        class_name = lambda c: c.__class__.__name__

        if accelerator.is_main_process:
            s_total = sum(p.numel() for p in self.student.parameters())
            s_train = sum(p.numel() for p in self.student.parameters() if p.requires_grad)
            t_total = sum(p.numel() for p in self.teacher.parameters()) if self.has_teacher and self.teacher else 0
            t_train = sum(p.numel() for p in self.teacher.parameters() if p.requires_grad) if self.teacher else 0
            accelerator.print("=" * 80)
            accelerator.print(f"Dataset : {self.dataset_name} | num_classes: {self.num_classes}")
            accelerator.print(f"Student : {class_name(self.student)} | params {s_total/1e6:.2f}M (train {s_train/1e6:.2f}M)")
            if self.has_teacher:
                accelerator.print(f"Teacher : {class_name(self.teacher)} | params {t_total/1e6:.2f}M ({'frozen' if t_train==0 else t_train})")
            if self.ce_hard is not None:
                accelerator.print("ce_hard : enable")
            if self.ce_soft is not None:
                accelerator.print("ce_soft : enable")
            if self.kd is not None:
                accelerator.print(f"KD      : alpha_kd={self.alpha_kd}")
            if self.hint is not None:
                accelerator.print(f"Hint    : alpha_hint={self.alpha_hint} | points={self.hint_points}")
            accelerator.print(f"Optim(SGD): lr={self.lr} wd={self.wd} epochs={self.epochs}")
            accelerator.print("=" * 80)

    def on_validation_epoch_start(self, epoch: int):
        self._ternary_snapshot = self._clone_student()

    # -------------------- hint / teacher utilities --------------------
    def init_hint(self):
        s_mods = dict(self.student.named_modules())
        t_mods = dict(self.teacher.named_modules())

        for n in self.hint_points:
            if isinstance(n, tuple):
                sn, tn = n
            else:
                sn, tn = n, n

            if sn not in s_mods:
                raise ValueError(f"Student hint point '{sn}' not found in student.named_modules().")
            if tn not in t_mods:
                raise ValueError(f"Teacher hint point '{tn}' not found in teacher.named_modules().")

            s_m = s_mods[sn]
            t_m = t_mods[tn]

            c_s = infer_out_channels(s_m)
            c_t = infer_out_channels(t_m)

            if c_s is None or c_t is None:
                raise ValueError(
                    f"Cannot infer channels for hint point {n}. "
                    f"Student module: {type(s_m)}, teacher module: {type(t_m)}"
                )

            self.hint.register_pair(sn, c_s, c_t)

    def get_loss_hint(self) -> torch.Tensor:
        loss_hint = 0.0
        for hint_name in self.hint_points:
            sn = tn = hint_name
            if isinstance(hint_name, tuple):
                sn, tn = hint_name

            if sn not in self._s_feats:
                raise ValueError(f"Hint point {sn} not found in student features keys: {list(self._s_feats.keys())}")
            if tn not in self._t_feats:
                raise ValueError(f"Hint point {tn} not found in teacher features keys: {list(self._t_feats.keys())}")

            loss_hint = loss_hint + self.hint(
                sn,
                self._s_feats[sn].float(),
                self._t_feats[tn].float().detach(),
            )
        return loss_hint

    def teacher_forward(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        if not self.has_teacher or self.teacher is None:
            return None
        with torch.no_grad():
            return self.teacher.eval()(x).detach()

    @torch.no_grad()
    def _clone_student(self) -> nn.Module:
        # robust clone: prefer .clone() if provided by Bit models, else deepcopy
        if hasattr(self.student, "clone") and callable(getattr(self.student, "clone")):
            clone: nn.Module = self.student.clone()
            clone.load_state_dict(self.student.state_dict(), strict=True)
        else:
            clone = copy.deepcopy(self.student)
        clone = convert_to_ternary(clone)

        dev = next(self.student.parameters()).device
        return clone.to(dev)

    # -------------------- training / validation --------------------
    def _ce_training_step(self, x: torch.Tensor, y: torch.Tensor):
        logits = self.student(x)
        ce = self.ce_hard if self.ce_hard is not None else self.ce_soft
        loss = ce(logits, y)
        return loss, {"train/ce": loss.detach()}, logits

    def _ce_kd_training_step(self, x: torch.Tensor, y: torch.Tensor):
        z_t = self.teacher_forward(x)
        loss_ce, logd, logits = self._ce_training_step(x, y)
        alpha_kd = self.alpha_kd
        loss_kd = self.kd(logits.float(), z_t.float())
        loss = (1.0 - alpha_kd) * loss_ce + alpha_kd * loss_kd
        logd = {**logd, "train/kd": loss_kd.detach()}
        return loss, logd, logits

    def _ce_kd_hint_training_step(self, x: torch.Tensor, y: torch.Tensor):
        loss, logd, logits = self._ce_kd_training_step(x, y)
        loss_hint = self.get_loss_hint()
        loss = loss + loss_hint
        logd = {**logd, "train/hint": loss_hint.detach()}
        return loss, logd, logits

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> Metrics:
        x, y = batch

        if self.kd is not None and self.hint is not None:
            loss, logd, logits = self._ce_kd_hint_training_step(x, y)
        elif self.kd is not None:
            loss, logd, logits = self._ce_kd_training_step(x, y)
        else:
            loss, logd, logits = self._ce_training_step(x, y)

        y_idx = y.argmax(dim=1) if y.ndim == 2 else y
        acc = (logits.argmax(dim=1) == y_idx).float().mean()
        logd["train/acc"] = acc.detach()

        return Metrics(loss=loss, metrics=logd)

    @torch.no_grad()
    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> Metrics:
        x, y = batch
        y_idx = y.argmax(dim=1) if y.ndim == 2 else y

        z_fp = self.student(x)
        z_tern = self._ternary_snapshot(x)

        vloss = F.cross_entropy(z_fp, y_idx.long())

        acc_fp = (z_fp.argmax(dim=1) == y_idx).float().mean()
        acc_tern = (z_tern.argmax(dim=1) == y_idx).float().mean()

        metrics: Dict[str, Any] = {"val/acc_fp": acc_fp, "val/acc_tern": acc_tern}

        if self.has_teacher and self.teacher is not None and self.alpha_kd > 0:
            if self.t_acc_fps.get(batch_idx) is None:
                z_t = self.teacher_forward(x)
                t_acc = (z_t.argmax(dim=1) == y_idx).float().mean()
                self.t_acc_fps[batch_idx] = float(t_acc.item())
            metrics["val/t_acc_fp"] = torch.tensor(self.t_acc_fps[batch_idx], device=z_fp.device)

        return Metrics(loss=vloss, metrics=metrics)

    # -------------------- optimizer --------------------
    def configure_optimizer_params(self):
        params = list(self.student.parameters())
        if self.hint is not None:
            params += list(self.hint.parameters())
        if self.kd is not None:
            params += list(self.kd.parameters())
        return params

    def configure_optimizers(self):
        opt = torch.optim.SGD(
            self.configure_optimizer_params(),
            lr=self.lr,
            momentum=0.9,
            weight_decay=self.wd,
            nesterov=True,
        )
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.epochs)
        return opt, sched, "epoch"


# ----------------------------
# KD losses & feature hints
# ----------------------------
class KDLoss(nn.Module):
    def __init__(self, T: float = 4.0):
        super().__init__()
        self.T = T

    def forward(self, z_s: torch.Tensor, z_t: torch.Tensor) -> torch.Tensor:
        T = self.T
        return F.kl_div(F.log_softmax(z_s / T, 1), F.softmax(z_t / T, 1), reduction="batchmean") * (T * T)


class AdaptiveHintLoss(nn.Module):
    """Learnable 1x1 per hint; auto matches spatial size then SmoothL1."""
    def __init__(self):
        super().__init__()
        self.proj = nn.ModuleDict()

    @staticmethod
    def _k(name: str) -> str:
        return name.replace(".", "\u2027")

    def register_pair(self, name: str, c_s: int, c_t: int):
        k = self._k(name)
        self.proj[k] = nn.Conv2d(c_s, c_t, kernel_size=1, bias=True)

    def forward(self, name: str, f_s: torch.Tensor, f_t: torch.Tensor) -> torch.Tensor:
        f_s = F.adaptive_avg_pool2d(f_s, f_t.shape[-2:])
        k = self._k(name)
        f_s = self.proj[k](f_s)
        return F.smooth_l1_loss(f_s, f_t)


class SaveOutputHook:
    """Picklable forward hook that stores outputs into a dict under a given key."""
    __slots__ = ("store", "key")

    def __init__(self, store: dict, key: str):
        self.store = store
        self.key = key

    def __call__(self, module, module_in, module_out: torch.Tensor):
        self.store[self.key] = module_out


def make_feature_hooks(module: nn.Module, names: Sequence[Union[str, Tuple]], feats: dict, idx: int = 0):
    """Register picklable forward hooks; returns list of handles."""
    handles = []
    if len(names) == 0:
        return handles

    if isinstance(names[0], tuple):
        names = [n[idx] for n in names]  # type: ignore

    for n, sub in module.named_modules():
        for key in names:  # ordered, deterministic
            if n.endswith(key):
                handles.append(sub.register_forward_hook(SaveOutputHook(feats, key)))
                break
    return handles


def infer_out_channels(module: nn.Module):
    """Try to infer the number of output channels from a module."""
    for attr in ("out_channels", "num_features", "out_features"):
        if hasattr(module, attr):
            return getattr(module, attr)

    children = list(module.children())
    for child in reversed(children):
        c = infer_out_channels(child)
        if c is not None:
            return c

    return None

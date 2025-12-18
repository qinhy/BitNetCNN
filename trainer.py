from datetime import datetime
import os
import math
import copy
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
from pydantic import BaseModel, Field, PrivateAttr, model_validator

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from accelerate import Accelerator, DistributedType
from accelerate.utils import gather_object, broadcast_object_list
from tqdm.auto import tqdm
from dataset import DataModuleConfig, DataSetModule

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
# Lightning-like base
# ----------------------------
class AccelLightningModule(nn.Module):
    """
    Lightning-like interface:
      - forward(x)
      - training_step(batch, batch_idx) -> Metrics (must include loss)
      - validation_step(batch, batch_idx) -> Metrics
      - configure_optimizers() -> (optimizer, scheduler or None, scheduler_interval: "epoch"|"step")
    """

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> "Metrics":
        raise NotImplementedError

    @torch.no_grad()
    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> "Metrics":
        return Metrics(stage="val")

    def configure_optimizers(self, trainer: 'AccelTrainer'=None):
        raise NotImplementedError

    # optional hooks
    def on_fit_start(self, trainer: 'AccelTrainer'): ...
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

    # excluded from pydantic serialization; still used in to_dict()
    loss: Optional[Any] = Field(default=None, exclude=True)
    metrics: Dict[str, Any] = Field(default_factory=dict)

    def set(self, key: str, value: float) -> None:
        self.metrics[key] = float(value)

    def get(self, key: str, default: Optional[float] = None) -> Any:
        return self.metrics.get(key, default)

    def to_dict(self,digits=7) -> Dict[str, Any]:
        res: Dict[str, Any] = {f"{self.stage}/loss": self.loss}
        res.update(self.metrics)

        out: Dict[str, Any] = {}
        for k, v in res.items():
            if v is None:
                continue
            if torch.is_tensor(v):
                out[k] = float(v.detach().cpu().item())
            else:
                out[k] = v
            try:                
                out[k] = float((f'{out[k]}')[:digits])
            except:
                pass
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

    def _is_better(self, val: float) -> bool:
        if self.best is None:
            return True
        return (val > self.best) if self.mode == "max" else (val < self.best)

    def compare(self, val: Any) -> bool:
        v = _as_float(val)
        if v is None:
            return False
        if self._is_better(v):
            self.best = float(v)
            return True
        return False


class MetricsManager(BaseModel):
    metrics_list: List[Metrics] = Field(default_factory=list)
    epoch_metric_tracers: List[MetricsTracer] = Field(default_factory=list)
    _last_epoch_by_stage: Dict[str, int] = PrivateAttr(default_factory=lambda: {"train": -1, "val": -1})

    def to_list(self) -> List[Dict[str, Any]]:
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
            if t.compare(val) and t.callback is not None:
                t.callback(metrics=metrics, trainer=trainer, model=model)

    def last_epoch_mean(self, stage: str = "train", epoch: int = -1) -> Metrics:
        """
        Convenience helper: mean over all logged rows for the latest epoch of a stage.
        Avoids producing keys like "x_mean_mean".
        """
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
            if not vals:
                continue

            avg = sum(vals) / len(vals)
            out_key = k if k.endswith("_mean") else f"{k}_mean"
            mean_metrics[out_key] = avg

        return Metrics(stage=stage, epoch=epoch, loss=None, metrics=mean_metrics)


# ----------------------------
# Export best ternary checkpoint
# ----------------------------
class ExportBestTernary:
    """
    Exports:
      - best FP student checkpoint
      - best ternary PoT checkpoint (via convert_to_ternary)

    Triggered by MetricsTracer when "val/acc_tern_mean" improves.
    """

    def __call__(self, metrics: Metrics, trainer: "AccelTrainer", model: "LitBit"):
        # IMPORTANT: global main process only (prevents multi-node duplicates/overwrites)
        if not trainer.accelerator.is_main_process:
            return

        acc_tern = _as_float(metrics.get("val/acc_tern_mean"))
        if acc_tern is None:
            return

        unwrapped: "LitBit" = trainer.accelerator.unwrap_model(model)
        config = unwrapped.config

        export_dir = config.export_dir
        dataset_name = config.dataset.dataset_name
        model_name = config.model_name
        model_size = config.model_size

        os.makedirs(export_dir, exist_ok=True)

        # ---- save FP state_dict on CPU (cheap & safe) ----
        fp_path = os.path.join(export_dir, f"bit_{model_name}_{model_size}_{dataset_name}_best_fp.pt")
        fp_state = {k: v.detach().cpu() for k, v in unwrapped.student.state_dict().items()}
        torch.save({"model": fp_state, "acc_tern": acc_tern}, fp_path)
        trainer.print(f"[OK] saved {fp_path} (val/acc_tern={acc_tern*100:.2f}%)")

        # ---- export ternary PoT (convert CPU copy) ----
        ternary_model = convert_to_ternary(copy.deepcopy(unwrapped.student).cpu()).cpu()
        tern_path = os.path.join(
            export_dir,
            f"bit_{model_name}_{model_size}_{dataset_name}_ternary_val_acc@{acc_tern*100:.2f}.pt",
        )
        torch.save({"model": ternary_model.state_dict(), "acc_tern": acc_tern}, tern_path)
        trainer.print(f"[OK] exported ternary PoT -> {tern_path}")


# ----------------------------
# AccelTrainer
# ----------------------------
class TextAppendLogger(BaseModel):
    filepath: Optional[Path] = None  # if None -> auto-generate
    log_dir: Path = Field(default=Path("logs"))
    name_prefix: str = "run"
    ext: str = ".txt"
    prefix_ts: bool = True
    encoding: str = "utf-8"

    def resolve_path(self):
        if self.filepath is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            ext = self.ext if self.ext.startswith(".") else f".{self.ext}"
            self.filepath = Path(self.log_dir) / f"{self.name_prefix}_{ts}{ext}"
        else:
            self.filepath = Path(self.filepath)

        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        return self

    def __call__(self, text: str) -> None:
        self.resolve_path()
        line = text.rstrip("\n")
        if self.prefix_ts:
            ts = datetime.now().isoformat(timespec="seconds")
            line = f"[{ts}] {line}"

        with self.filepath.open("a", encoding=self.encoding) as f:
            f.write(line + "\n")
            f.flush()

class StepMetricsAccumulator:
    """
    Accumulates (value * batch_size) and batch_size across micro-steps.
    On compute: all-reduces numerator/denominator and returns GLOBAL weighted means.
    """

    def __init__(self, accelerator: Accelerator):
        self.acc = accelerator
        self.reset()

    def reset(self) -> None:
        self.num: Dict[str, torch.Tensor] = {}
        self.den: torch.Tensor = torch.zeros((), device=self.acc.device)

    def _to_scalar(self, v: Any) -> Optional[torch.Tensor]:
        if v is None:
            return None
        if torch.is_tensor(v):
            t = v.detach()
            t = t.float().mean() if t.numel() != 1 else t.float()
            return t.to(self.acc.device)
        try:
            return torch.tensor(float(v), device=self.acc.device).float()
        except Exception:
            return None

    @staticmethod
    def _stage_prefix(stage: str, key: str) -> str:
        return key if "/" in key else f"{stage}/{key}"

    def update(self, metrics_obj: "Metrics", batch_size: int, stage: str) -> None:
        bs = torch.tensor(float(batch_size), device=self.acc.device).float()
        self.den = self.den + bs

        # stable loss key
        loss = metrics_obj.loss
        if loss is not None:
            t = self._to_scalar(loss)
            if t is not None:
                k = f"{stage}/loss"
                self.num[k] = self.num.get(k, torch.zeros((), device=self.acc.device)) + t * bs

        # arbitrary metric keys (stage-prefixed)
        for k, v in metrics_obj.metrics.items():
            t = self._to_scalar(v)
            if t is None:
                continue
            k = self._stage_prefix(stage, str(k))
            self.num[k] = self.num.get(k, torch.zeros((), device=self.acc.device)) + t * bs

    def _union_keys_across_ranks(self, local_keys) -> List[str]:
        local_keys = list(map(str, list(local_keys)))

        if self.acc.num_processes == 1:
            return sorted(set(local_keys))

        gathered = gather_object(local_keys)

        if self.acc.is_main_process:
            if gathered is None:
                flat = local_keys
            elif isinstance(gathered, (list, tuple)):
                if gathered and isinstance(gathered[0], (list, tuple)):
                    flat = [k for sub in gathered for k in sub]
                else:
                    flat = list(gathered)
            else:
                flat = [str(gathered)]
            union = sorted(set(map(str, flat)))
        else:
            union = None

        obj = [union]
        broadcast_object_list(obj, from_process=0)
        return obj[0]

    def compute_global_means_count_and_reset(self) -> Tuple[Dict[str, torch.Tensor], int]:
        """
        IMPORTANT: call on ALL ranks (no main-process guard), otherwise you can deadlock.
        Returns: (means_dict, global_sample_count)
        """
        den_tot = self.acc.reduce(self.den, reduction="sum")
        if float(den_tot.item()) <= 0:
            self.reset()
            return {}, 0

        keys = self._union_keys_across_ranks(self.num.keys())
        zero = torch.zeros((), device=self.acc.device)

        out: Dict[str, torch.Tensor] = {}
        for k in keys:
            num_k = self.num.get(k, zero)
            num_tot = self.acc.reduce(num_k, reduction="sum")
            out[k] = (num_tot / den_tot).detach()

        count = int(den_tot.detach().cpu().item())
        self.reset()
        return out, count

class AccelTrainer:
    def __init__(
        self,
        max_epochs: int,
        mixed_precision: str = "no",  # "no" | "fp16" | "bf16" | "fp8"
        gradient_accumulation_steps: int = 1,
        max_grad_norm: Optional[float] = None,
        show_progress_bar: bool = True,
        metrics_manager: Optional["MetricsManager"] = None,
        log_every_n_steps: int = 10,
        logger: Optional[Callable[..., None]] = None,
        # FP8 extras (optional)
        fp8_backend: Optional[Literal["te", "msamp", "ao"]] = None,
        fp8_kwargs: Optional[dict] = None,
        # Advanced: pass custom Accelerate kwargs_handlers directly
        kwargs_handlers: Optional[list] = None,
    ):
        self.max_epochs = int(max_epochs)
        self.max_grad_norm = max_grad_norm
        self.show_progress_bar = show_progress_bar
        self.log_every_n_steps = int(log_every_n_steps)

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
        self.logger=logger

        # ---- FP8 recipe selection (optional) ----
        handlers = list(kwargs_handlers) if kwargs_handlers is not None else []

        if mixed_precision == "fp8" and fp8_backend is not None:
            fp8_kwargs = fp8_kwargs or {}
            try:
                # Newer Accelerate uses recipe kwargs classes
                from accelerate.utils import TERecipeKwargs, MSAMPRecipeKwargs, AORecipeKwargs
            except Exception as e:
                raise RuntimeError(
                    "FP8 backend recipe requested, but your accelerate version "
                    "doesn't expose TERecipeKwargs/MSAMPRecipeKwargs/AORecipeKwargs. "
                    "Upgrade accelerate or pass kwargs_handlers yourself."
                ) from e

            if fp8_backend == "te":
                handlers.append(TERecipeKwargs(**fp8_kwargs))
            elif fp8_backend == "msamp":
                handlers.append(MSAMPRecipeKwargs(**fp8_kwargs))
            elif fp8_backend == "ao":
                handlers.append(AORecipeKwargs(**fp8_kwargs))
            else:
                raise ValueError(f"Unknown fp8_backend={fp8_backend!r}")

        self.accelerator = Accelerator(
            mixed_precision=mixed_precision,
            gradient_accumulation_steps=int(gradient_accumulation_steps),
            kwargs_handlers=handlers or None,
        )

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
        # optional logger dir convention
        if self.logger is not None and hasattr(model, "config") and hasattr(model.config, "export_dir"):
            try:
                self.logger.log_dir = model.config.export_dir
            except Exception:
                pass

        train_dataloader, val_dataloader = self._resolve_dataloaders(datamodule, train_dataloader, val_dataloader)

        optimizer, scheduler, interval = model.configure_optimizers(self)
        self.scheduler_interval = interval or "epoch"

        model, optimizer, train_dataloader, val_dataloader, scheduler = self._prepare(
            model, optimizer, train_dataloader, val_dataloader, scheduler
        )

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler

        # hooks on unwrapped model are fine
        self.accelerator.unwrap_model(model).on_fit_start(self)

        for epoch in range(self.max_epochs):
            self._run_epoch(stage="train", epoch=epoch, loader=train_dataloader)
            if val_dataloader is not None:
                self._run_epoch(stage="val", epoch=epoch, loader=val_dataloader)

            if self.scheduler is not None and self.scheduler_interval == "epoch":
                self.scheduler.step()

        return self.metrics_manager.to_list() if self.metrics_manager is not None else []

    # -----------------------------
    # Setup helpers
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

    def _prepare(self, model: nn.Module, optimizer, train_dataloader, val_dataloader=None, scheduler=None):
        # Only do SyncBN in multi-GPU DDP-style training on CUDA
        if (self.accelerator.distributed_type == DistributedType.MULTI_GPU
            and self.accelerator.device.type == "cuda"
            and hasattr(model, "student")):
            model.student = nn.SyncBatchNorm.convert_sync_batchnorm(model.student)

        objs = [model, optimizer, train_dataloader]
        if val_dataloader is not None:
            objs.append(val_dataloader)
        if scheduler is not None:
            objs.append(scheduler)

        prepared = self.accelerator.prepare(*objs)

        idx = 0
        model_p = prepared[idx]; idx += 1
        optim_p = prepared[idx]; idx += 1
        train_p = prepared[idx]; idx += 1
        val_p = prepared[idx] if val_dataloader is not None else None
        idx += 1 if val_dataloader is not None else 0
        sched_p = prepared[idx] if scheduler is not None else None

        return model_p, optim_p, train_p, val_p, sched_p

    # -----------------------------
    # Printing / tqdm / logging
    # -----------------------------
    def print(self, msg: str) -> None:
        if self.accelerator.is_main_process:
            tqdm.write(msg)
            if self.logger:
                self.logger(msg)

    def _pbar(self, loader: DataLoader, stage: str, epoch: int):
        if not (self.show_progress_bar and self.accelerator.is_main_process):
            return loader
        return tqdm(loader, desc=f"{stage} {epoch+1}", leave=False, dynamic_ncols=True)

    def _log(self, metrics: "Metrics", trainer_model: nn.Module, pbar, *, force_print: bool = False) -> None:
        # only main logs
        if not self.accelerator.is_main_process:
            return

        if self.metrics_manager is not None:
            self.metrics_manager.log(metrics, self, trainer_model)

        postfix = metrics.to_dict(7)
        if hasattr(pbar, "set_postfix"):
            pbar.set_postfix(postfix, refresh=False)

        step = metrics.step or 0
        if (step % self.log_every_n_steps == 0) or force_print:
            h = f"[{metrics.stage}]epoch={metrics.epoch} step={step}"
            self.print(f"{h} {str(postfix).replace(metrics.stage+'/','')}")

    # -----------------------------
    # Utilities
    # -----------------------------
    def _infer_batch_size(self, batch: Any) -> int:
        if isinstance(batch, (tuple, list)) and batch and hasattr(batch[0], "size"):
            return int(batch[0].size(0))
        if isinstance(batch, dict):
            for v in batch.values():
                if hasattr(v, "size"):
                    return int(v.size(0))
        return 1

    def _maybe_set_dataloader_epoch(self, loader: Any, epoch: int) -> None:
        if hasattr(loader, "set_epoch"):
            try:
                loader.set_epoch(epoch)
                return
            except TypeError:
                pass
        sampler = getattr(loader, "sampler", None)
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)

    @staticmethod
    def _mean_key(k: str) -> str:
        return k if k.endswith("_mean") else f"{k}_mean"

    # -----------------------------
    # Epoch runner (FP16/BF16/FP8-ready)
    # -----------------------------
    def _run_epoch(self, stage: Literal["train", "val"], epoch: int, loader: DataLoader) -> "Metrics":
        assert self.model is not None
        if stage == "train":
            assert self.optimizer is not None

        model = self.model                       # ✅ prepared/wrapped model
        raw_model = self.accelerator.unwrap_model(model)  # ✅ only for hooks/export
        optimizer = self.optimizer
        accelerator = self.accelerator

        # hooks + mode
        if stage == "train":
            raw_model.on_train_epoch_start(epoch)
            model.train()
        else:
            raw_model.on_validation_epoch_start(epoch)
            model.eval()

        self._maybe_set_dataloader_epoch(loader, epoch)
        pbar = self._pbar(loader, stage, epoch)

        epoch_accum = StepMetricsAccumulator(accelerator)
        step_accum = StepMetricsAccumulator(accelerator) if stage == "train" else None

        opt_step = -1
        last_micro_step = -1

        for micro_step, batch in enumerate(pbar):
            last_micro_step = micro_step
            bs = self._infer_batch_size(batch)

            if stage == "train":
                with accelerator.accumulate(model):
                    # ✅ This is the key for fp16/bf16/fp8:
                    with accelerator.autocast():
                        m = model.training_step(batch, micro_step).model_copy(
                            update=dict(stage=stage, epoch=epoch, step=micro_step, batch_size=bs)
                        )

                    if optimizer is not None:
                        try:
                            m.metrics["lr"] = float(optimizer.param_groups[0]["lr"])
                        except Exception:
                            pass

                    epoch_accum.update(m, batch_size=bs, stage=stage)
                    step_accum.update(m, batch_size=bs, stage=stage)

                    if m.loss is None:
                        raise ValueError("training_step must return Metrics with loss (got loss=None)")

                    accelerator.backward(m.loss)

                    if accelerator.sync_gradients:
                        if self.max_grad_norm is not None:
                            accelerator.clip_grad_norm_(model.parameters(), self.max_grad_norm)

                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)

                        # ✅ If fp16 overflow caused step to be skipped, don’t advance LR
                        if self.scheduler is not None and self.scheduler_interval == "step":
                            if not accelerator.optimizer_step_was_skipped:
                                self.scheduler.step()

                        # log ONE global mean row per optimizer step
                        opt_step += 1
                        step_means, step_count = step_accum.compute_global_means_count_and_reset()

                        loss_key = f"{stage}/loss"
                        step_loss = step_means.pop(loss_key, None)

                        step_row = Metrics(
                            stage=stage,
                            epoch=epoch,
                            step=opt_step,
                            batch_size=step_count,
                            loss=step_loss,
                            metrics=step_means,
                        )
                        self._log(step_row, trainer_model=model, pbar=pbar)

            else:
                with torch.no_grad():
                    with accelerator.autocast():
                        m = model.validation_step(batch, micro_step).model_copy(
                            update=dict(stage=stage, epoch=epoch, step=micro_step, batch_size=bs)
                        )
                epoch_accum.update(m, batch_size=bs, stage=stage)

        epoch_means, epoch_count = epoch_accum.compute_global_means_count_and_reset()
        summary_metrics = {self._mean_key(k): v for k, v in epoch_means.items()}

        summary = Metrics(
            stage=stage,
            epoch=epoch,
            step=max(last_micro_step, 0),
            batch_size=epoch_count,
            loss=None,
            metrics=summary_metrics,
        )
        self._log(summary, trainer_model=model, pbar=pbar, force_print=True)

        # end hooks
        if stage == "train":
            raw_model.on_train_epoch_end(epoch)
        else:
            raw_model.on_validation_epoch_end(epoch)

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

    model_name: str = Field(default="mbv2", description="Model family/name identifier.")
    model_size: str = Field(default="", description="Optional model size preset (empty = default).")
    model_weights: str = Field(default="", description="Optional path/name for pretrained weights (empty = none).")

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
        if os.path.exists(self.config.model_weights):
            stats = torch.load(self.config.model_weights)
            stats = stats['model'] if 'model' in stats else stats
            self.student.load_state_dict(state_dict=stats)
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
    def on_fit_start(self, trainer: 'AccelTrainer'):
        self._accel = accel = trainer.accelerator

        if self.has_teacher and self.teacher is not None:
            self.teacher.eval()
            for p in self.teacher.parameters():
                p.requires_grad_(False)

            if self.alpha_hint > 0 and len(self.hint_points) > 0:
                self._s_handles = make_feature_hooks(self.student, self.hint_points, self._s_feats, idx=0)
                self._t_handles = make_feature_hooks(self.teacher, self.hint_points, self._t_feats, idx=1)

        class_name = lambda c: c.__class__.__name__

        if accel.is_main_process:
            s_total = sum(p.numel() for p in self.student.parameters())
            s_train = sum(p.numel() for p in self.student.parameters() if p.requires_grad)
            t_total = sum(p.numel() for p in self.teacher.parameters()) if self.has_teacher and self.teacher else 0
            t_train = sum(p.numel() for p in self.teacher.parameters() if p.requires_grad) if self.teacher else 0
            trainer.print("=" * 80)
            trainer.print(f"Dataset : {self.dataset_name} | num_classes: {self.num_classes} | batch_size: {self.config.dataset.batch_size}")
            trainer.print(f"Student : {class_name(self.student)} | params {s_total/1e6:.2f}M (train {s_train/1e6:.2f}M)")
            if self.has_teacher:
                trainer.print(f"Teacher : {class_name(self.teacher)} | params {t_total/1e6:.2f}M ({'frozen' if t_train==0 else t_train})")
            if self.ce_hard is not None:
                trainer.print("ce_hard : enable")
            if self.ce_soft is not None:
                trainer.print("ce_soft : enable")
            if self.kd is not None:
                trainer.print(f"KD      : alpha_kd={self.alpha_kd}")
            if self.hint is not None:
                trainer.print(f"Hint    : alpha_hint={self.alpha_hint} | points={self.hint_points}")
                
            opt,sch,inv = self.configure_optimizers()
            trainer.print(f"Optim({class_name(opt)}): lr={self.lr} wd={self.wd} epochs={self.epochs}")
            if sch:
                trainer.print(f"Scheduler({class_name(sch)}): enable")
            trainer.print("=" * 80)

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

    @torch.no_grad()
    def teacher_forward(self, x):
        if not self.has_teacher or self.teacher is None:
            return None
        return self.teacher(x)

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
        loss_ce = (1.0 - self.alpha_kd) * ce(logits, y)
        return loss_ce, {"train/ce": loss_ce.detach()}, logits

    def _ce_kd_training_step(self, x: torch.Tensor, y: torch.Tensor):
        z_t = self.teacher_forward(x)
        loss_ce, logd, logits = self._ce_training_step(x, y)
        loss_kd = self.alpha_kd * self.kd(logits.float(), z_t.float())
        loss = loss_ce + loss_kd
        logd = {**logd, "train/kd": loss_kd.detach()}
        return loss, logd, logits

    def _ce_hint_training_step(self, x: torch.Tensor, y: torch.Tensor):
        z_t = self.teacher_forward(x)
        loss_ce, logd, logits = self._ce_training_step(x, y)
        loss_hint = self.alpha_hint * self.get_loss_hint()
        loss = loss_ce + loss_hint
        logd = {**logd, "train/hint": loss_hint.detach()}
        return loss, logd, logits
    
    def _ce_kd_hint_training_step(self, x: torch.Tensor, y: torch.Tensor):
        loss, logd, logits = self._ce_kd_training_step(x, y)
        loss_hint = self.alpha_hint * self.get_loss_hint()
        loss = loss + loss_hint
        logd = {**logd, "train/hint": loss_hint.detach()}
        return loss, logd, logits

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> Metrics:
        x, y = batch

        if self.kd is not None and self.hint is not None:
            loss, logd, logits = self._ce_kd_hint_training_step(x, y)

        elif self.kd is not None and self.hint is None:
            loss, logd, logits = self._ce_kd_training_step(x, y)

        elif self.kd is None and self.hint is not None:
            loss, logd, logits = self._ce_hint_training_step(x, y)
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

    def configure_optimizers(self, trainer: 'AccelTrainer'=None):
        opt = torch.optim.SGD(
            self.configure_optimizer_params(),
            lr=self.lr,
            momentum=0.9,
            weight_decay=self.wd,
            nesterov=True,
        )

        def auto_n_restarts_T0(max_epochs: int, T_mult: int = 2, min_T0: int = 20, n_restarts_max: int = 10):
            for n_restarts in range(n_restarts_max, -1, -1):
                n_cycles = n_restarts + 1
                if T_mult == 1:
                    T0 = math.ceil(max_epochs / n_cycles)
                else:
                    T0 = math.ceil(max_epochs * (T_mult - 1) / (T_mult**n_cycles - 1))
                if T0 >= min_T0:
                    return n_restarts, T0
            return 0, max_epochs
        
        def cosine_warm_restarts_lr(max_epochs, base_lr=1.0, eta_min=1e-6, T_mult=2, min_T0=20, n_restarts_max=10):
            n_restarts, T0 = auto_n_restarts_T0(max_epochs, T_mult=T_mult, min_T0=min_T0, n_restarts_max=n_restarts_max)
            # Build cycle lengths
            n_cycles = n_restarts + 1
            cycle_lengths = [T0 * (T_mult**k) for k in range(n_cycles)]
            boundaries = np.cumsum([0] + cycle_lengths)  # start indices of each cycle; last is end
            total = boundaries[-1]
            # epochs we will plot: 0..max_epochs inclusive
            epochs = np.arange(0, max_epochs + 1, dtype=float)
            lrs = np.empty_like(epochs)

            # For each epoch, find which cycle it is in using boundaries
            # boundaries: [0, end1, end2, ..., total]
            for i, e in enumerate(epochs):
                # If training ends before schedule total, clamp to e
                # Find cycle idx such that boundaries[idx] <= e < boundaries[idx+1]
                idx = np.searchsorted(boundaries[1:], e, side='right')
                idx = min(idx, len(cycle_lengths) - 1)
                start = boundaries[idx]
                Ti = cycle_lengths[idx]
                t_cur = e - start
                # In case e falls past planned total (unlikely), wrap into last cycle by mod
                if t_cur > Ti:
                    t_cur = t_cur % Ti
                lrs[i] = eta_min + (base_lr - eta_min) * (1.0 + math.cos(math.pi * (t_cur / Ti))) / 2.0

            restart_epochs = list(boundaries[1:-1].astype(int))  # where new cycle starts (excluding 0 and final end)
            return epochs, lrs, n_restarts, T0, cycle_lengths, restart_epochs, int(total)
        
        planned_epochs = cosine_warm_restarts_lr(self.epochs)[-1]
        if planned_epochs!=self.epochs and trainer:
            trainer.print(f'[!!warning!!]: scheduler planned epochs={planned_epochs}, but now epochs={self.epochs}.')
        
        # Good for [102,300,510,1023,1512,2016,4064] epoch
        sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            opt, T_0=auto_n_restarts_T0(self.epochs)[1], T_mult=2, eta_min=0.
        )
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
        self.proj[k] = nn.Identity() if c_s==c_t else nn.Conv2d(c_s, c_t, kernel_size=1, bias=True)

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

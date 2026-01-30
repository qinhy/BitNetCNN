
import os
from typing import Literal, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from pydantic import BaseModel, Field
from pydanticV2_argparse import ArgumentParser

from bitlayers.dinov3.models.vision_transformer import DinoVisionTransformer, vit_small, vit_large
from common_utils import summ
from dataset import DataModuleConfig, RetinaFaceDataModule
from trainer import AccelTrainer, AccelLightningModule, CommonTrainConfig, LitBit, LitBitConfig, Metrics, MetricsManager


# -----------------------------------------------------------------------------
# Distillation model: vit_large -> vit_small
# -----------------------------------------------------------------------------
class DinoV3Distill(LitBit):
    def __init__(self, config: "LitBitConfig"):        
        config.hint_points =[('blocks.1','blocks.2', 'seq'),
                            ('blocks.2','blocks.4', 'seq'),
                            ('blocks.3','blocks.6', 'seq'),
                            ('blocks.4','blocks.8', 'seq'),
                            ('blocks.5','blocks.10', 'seq'),
                            ('blocks.6','blocks.12', 'seq'),
                            ('blocks.7','blocks.14', 'seq'),
                            ('blocks.8','blocks.16', 'seq'),
                            ('blocks.9','blocks.18', 'seq'),
                            ('blocks.10','blocks.20', 'seq'),]
        super().__init__(config)
        student:DinoVisionTransformer = self.student
        teacher:DinoVisionTransformer = self.teacher
        # summ(self.student)
        # summ(self.teacher)
        student.init_weights()

        self.proj = nn.Linear(teacher.embed_dim, student.embed_dim, bias=True)
        self.alpha_kd = float(config.alpha_kd)
        self.lr = float(config.lr)
        self.wd = float(config.wd)
        self.epochs = int(config.epochs)

    @torch.no_grad()
    def teacher_forward(self, x):
        if not self.has_teacher or self.teacher is None:
            return None
        z_t = self.teacher(x)
        return self.proj(z_t)
    
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

        return Metrics(loss=loss, metrics=logd)
    
    @torch.no_grad()
    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> Metrics:
        x, _ = batch
        return Metrics(loss={}, metrics=[])
    
    def configure_optimizer_params(self):
        params = list(self.student.parameters()) + list(self.proj.parameters())
        if self.hint is not None:
            params += list(self.hint.parameters())
        if self.kd is not None:
            params += list(self.kd.parameters())
        return params

    def configure_optimizers(self, trainer: AccelTrainer = None):
        opt = torch.optim.AdamW(self.configure_optimizer_params(), lr=self.lr, weight_decay=self.wd)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.epochs)
        return opt, sched, "epoch"


# -----------------------------------------------------------------------------
# Config + CLI
# -----------------------------------------------------------------------------
class DinoV3DistillConfig(CommonTrainConfig):
    data_dir: str = "./data"
    dataset_name: str = "retinaface"

    epochs: int = Field(50, ge=1)
    batch_size: int = Field(8, ge=1)
    num_workers: int = Field(0, ge=0)

    lr: float = Field(1e-4, gt=0)
    wd: float = Field(5e-2, ge=0)
    amp: bool = True
    alpha_kd: float = 1.0
    alpha_hint: float = 0.001

    image_size: int = 640
    patch_size: int = 16

    model_name: str = "dinov3"
    model_size: Literal["vitl16", "vitb16", "vits16"] = "vits16"
    student_weights: str = ""
    teacher_weights: str = ""


def main() -> None:
    parser = ArgumentParser(model=DinoV3DistillConfig)
    cfg = parser.parse_typed_args()

    dm_conf = DataModuleConfig.model_validate(cfg.model_dump())
    config = LitBitConfig.model_validate(cfg.model_dump())
    config.dataset = dm_conf.model_copy()
    config.export_dir = f"./ckpt_{config.dataset.dataset_name}_dinov3_{config.model_size}"
    dm = RetinaFaceDataModule(dm_conf, anchors=None, pos_iou=None, neg_iou=None, variances=None)


    config.student = vit_small(patch_size=cfg.patch_size, img_size=cfg.image_size)
    config.teacher = torch.hub.load('../dinov3', 'dinov3_vitl16', source='local',
                                    weights='./data/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth')
    model = DinoV3Distill(config)

    trainer = AccelTrainer(
        max_epochs=cfg.epochs,
        mixed_precision="bf16" if cfg.amp else "no",
        gradient_accumulation_steps=1,
        log_every_n_steps=10,
        metrics_manager=MetricsManager(epoch_metric_tracers=[]),
    )

    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    main()



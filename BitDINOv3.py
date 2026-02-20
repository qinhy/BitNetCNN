
import os
import random
from typing import Literal, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from pydantic import BaseModel, Field
from pydanticV2_argparse import ArgumentParser

from bitlayers.dinov3.models.vision_transformer import DinoVisionTransformer, DinoVisionTransformerTRM, vit_small, vit_large
from common_utils import summ
from dataset import DataModuleConfig, RetinaFaceDataModule
from trainer import AccelTrainer, AccelLightningModule, CommonTrainConfig, LitBit, LitBitConfig, Metrics, MetricsManager

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

class DinoPatchSimilarityViewer:
    """
    Interactive viewer for patch-level self-similarity maps.

    Inputs:
      image:  HxWx3 numpy array (uint8 or float in [0,1])
      feats:  torch.Tensor of shape (B, C, Gh, Gw)
    """
    def __init__(self, image, feats, batch_idx=0, title="DINO patch similarity"):
        if not isinstance(image, np.ndarray):
            raise TypeError("image must be a numpy array (H, W, 3)")
        if image.ndim != 3 or image.shape[2] not in (3, 4):
            raise ValueError("image must have shape (H, W, 3) or (H, W, 4)")
        if not torch.is_tensor(feats):
            raise TypeError("feats must be a torch.Tensor")
        if feats.ndim != 4:
            raise ValueError("feats must have shape (B, C, Gh, Gw)")

        self.image = image[..., :3].copy()
        self.H, self.W = self.image.shape[:2]

        # keep features on original device (GPU is fine)
        self.feats = feats
        self.B, self.C, self.Gh, self.Gw = feats.shape
        self.batch_idx = int(np.clip(batch_idx, 0, self.B - 1))

        # normalize once for cosine similarity
        self.feats_n = F.normalize(self.feats, dim=1)

        # state
        self.gx = self.Gw // 2
        self.gy = self.Gh // 2
        self.alpha = 0.45
        self.show_overlay = True
        self.percentile_clip = (2, 98)  # for nicer visualization

        # figure
        self.fig, (self.ax_img, self.ax_map) = plt.subplots(1, 2, figsize=(12, 6))
        self.fig.canvas.manager.set_window_title(title) if hasattr(self.fig.canvas.manager, "set_window_title") else None
        self.fig.suptitle(title)

        self.ax_img.set_title("Image (click to select patch)")
        self.ax_map.set_title("Similarity map")

        # base image
        self.im_img = self.ax_img.imshow(self.image)
        self.im_overlay = self.ax_img.imshow(
            np.zeros((self.H, self.W), dtype=np.float32),
            cmap="jet",
            alpha=self.alpha,
            vmin=0.0,
            vmax=1.0,
        )
        self.im_overlay.set_visible(self.show_overlay)

        # patch grid marker
        self.marker = self.ax_img.plot([], [], marker="o", markersize=8, markeredgecolor="white",
                                       markerfacecolor="none", markeredgewidth=2)[0]

        # map panel
        self.im_map = self.ax_map.imshow(
            np.zeros((self.Gh, self.Gw), dtype=np.float32),
            cmap="jet",
            vmin=0.0,
            vmax=1.0,
            interpolation="nearest"
        )
        self.marker_map = self.ax_map.plot([], [], marker="s", markersize=9, markeredgecolor="white",
                                           markerfacecolor="none", markeredgewidth=2)[0]
        self.cbar = self.fig.colorbar(self.im_map, ax=self.ax_map, fraction=0.046, pad=0.04)
        self.cbar.set_label("relative score")

        self.ax_img.set_xlim([0, self.W - 1])
        self.ax_img.set_ylim([self.H - 1, 0])
        self.ax_map.set_xlim([-0.5, self.Gw - 0.5])
        self.ax_map.set_ylim([self.Gh - 0.5, -0.5])

        # help text
        help_text = (
            "Click image/map: select patch\n"
            "←/→: batch   +/-: overlay alpha\n"
            "o: toggle overlay   c: reset contrast"
        )
        self.text = self.fig.text(0.02, 0.02, help_text, fontsize=9)

        # connect events
        self.cid_click = self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        self.cid_key = self.fig.canvas.mpl_connect("key_press_event", self._on_key)

        # initial draw
        self._update_view()

    def _grid_to_image_xy(self, gx, gy):
        # patch center in image pixel coords
        x = (gx + 0.5) * (self.W / self.Gw)
        y = (gy + 0.5) * (self.H / self.Gh)
        return x, y

    def _image_xy_to_grid(self, x, y):
        gx = int(np.clip(np.floor(x * self.Gw / self.W), 0, self.Gw - 1))
        gy = int(np.clip(np.floor(y * self.Gh / self.H), 0, self.Gh - 1))
        return gx, gy

    def _compute_score(self):
        """
        Returns:
          score_grid: (Gh, Gw) float tensor in [-1,1]
          score_vis_grid: (Gh, Gw) float tensor in [0,1] (for display)
          score_vis_img: (H, W) float tensor in [0,1] (upsampled)
        """
        b = self.batch_idx
        # feats_n[b]: (C, Gh, Gw)
        ref = self.feats_n[b, :, self.gy, self.gx]  # (C,)
        score = (self.feats_n[b] * ref[:, None, None]).sum(dim=0)  # (Gh, Gw), cosine sim

        # robust normalization for visualization
        s = score.detach().float()
        lo = torch.quantile(s.flatten(), self.percentile_clip[0] / 100.0)
        hi = torch.quantile(s.flatten(), self.percentile_clip[1] / 100.0)
        s_vis = (s - lo) / (hi - lo + 1e-6)
        s_vis = s_vis.clamp(0, 1)

        # upscale to image size for overlay
        s_img = F.interpolate(
            s_vis[None, None],
            size=(self.H, self.W),
            mode="bilinear",
            align_corners=False
        )[0, 0]

        return score, s_vis, s_img

    def _update_view(self):
        score, s_vis, s_img = self._compute_score()

        # update overlay on image
        self.im_overlay.set_data(s_img.detach().cpu().numpy())
        self.im_overlay.set_alpha(self.alpha)
        self.im_overlay.set_visible(self.show_overlay)

        # update map panel
        self.im_map.set_data(s_vis.detach().cpu().numpy())

        # markers
        x, y = self._grid_to_image_xy(self.gx, self.gy)
        self.marker.set_data([x], [y])
        self.marker_map.set_data([self.gx], [self.gy])

        # titles
        raw_val = float(score[self.gy, self.gx].item())  # should be ~1.0
        self.ax_img.set_title(
            f"Image (B={self.batch_idx}/{self.B-1}) | patch=({self.gx},{self.gy})"
        )
        self.ax_map.set_title(f"Similarity map (self={raw_val:.3f})")

        self.fig.canvas.draw_idle()

    def _on_click(self, event):
        if event.inaxes is None or event.xdata is None or event.ydata is None:
            return

        if event.inaxes == self.ax_img:
            gx, gy = self._image_xy_to_grid(event.xdata, event.ydata)
            self.gx, self.gy = gx, gy
            self._update_view()
        elif event.inaxes == self.ax_map:
            gx = int(np.clip(round(event.xdata), 0, self.Gw - 1))
            gy = int(np.clip(round(event.ydata), 0, self.Gh - 1))
            self.gx, self.gy = gx, gy
            self._update_view()

    def _on_key(self, event):
        key = event.key
        if key in ("left", "right"):
            if key == "left":
                self.batch_idx = (self.batch_idx - 1) % self.B
            else:
                self.batch_idx = (self.batch_idx + 1) % self.B
            self._update_view()

        elif key in ("+", "="):
            self.alpha = min(1.0, self.alpha + 0.05)
            self._update_view()

        elif key == "-":
            self.alpha = max(0.0, self.alpha - 0.05)
            self._update_view()

        elif key == "o":
            self.show_overlay = not self.show_overlay
            self._update_view()

        elif key == "c":
            self.percentile_clip = (2, 98)
            self._update_view()

    def show(self):
        plt.tight_layout()
        plt.show()

# -----------------------------------------------------------------------------
# Distillation model: vit_small <- vit_large
# -----------------------------------------------------------------------------
def cosine_sim_matrix(z: torch.Tensor) -> torch.Tensor:
    # z: (B, N, D)
    z = F.normalize(z, dim=-1)
    return z @ z.transpose(-1, -2)  # (B, N, N)

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
    
    def relational_kd_kl(self,
        z_s: torch.Tensor,        # (B,N,D) student tokens
        z_t: torch.Tensor = None, # (B,N,D) teacher tokens (optional if you already have K_t)
        K_t: torch.Tensor = None, # (B,N,N) teacher sim matrix
        T: float = 0.1,
        drop_cls: bool = True,
        mask_diag: bool = True,
    ) -> torch.Tensor:
        if K_t is None:
            assert z_t is not None
            K_t = cosine_sim_matrix(z_t)

        if drop_cls:
            z_s = z_s[:, 1:, :]
            K_t = K_t[:, 1:, 1:]

        K_s = cosine_sim_matrix(z_s)

        if mask_diag:
            N = K_s.size(-1)
            diag = torch.eye(N, device=K_s.device, dtype=torch.bool)[None]  # (1,N,N)
            K_s = K_s.masked_fill(diag, -1e9)
            K_t = K_t.masked_fill(diag, -1e9)

        # teacher probs, student log-probs
        p_t = F.softmax(K_t.detach() / T, dim=-1)
        log_p_s = F.log_softmax(K_s / T, dim=-1)

        return F.kl_div(log_p_s, p_t, reduction="batchmean") * (T * T)
    
    def training_step_old(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> Metrics:
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
    
    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> Metrics:
        x, y = batch
        logd = {}
        # loss_hint, logd, logits = self._ce_hint_training_step(x, y)
        N_supervision_max = 2
        N_supervision = random.randint(1, N_supervision_max)
        loss_kd_kl = 0.0
        z_t = self.teacher.forward_features(x)["x_norm_patchtokens"]
        solution, latent = None, None
        for step in range(N_supervision):            
            T = random.randint(1, 2)
            n = random.randint(1, 2)    
            out = self.student.forward_features(
                    x, solution=solution, latent=latent, n=n, T=T,
                    # track_latent_grads=True,  # if your patched TRM class supports it
                )
            solution = out["x_prenorm"]   # solution state
            latent   = out["z_latent"]    # latent state
            z_s = out["x_norm_patchtokens"]

            loss_kd_kl += self.relational_kd_kl(z_s, z_t)
        loss_kd_kl /= N_supervision
        logd = {**logd,**{"train/kd_kl": loss_kd_kl.detach()}}
        return Metrics(loss=loss_kd_kl, metrics=logd)
    
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
    batch_size: int = Field(2, ge=1)
    num_workers: int = Field(0, ge=0)

    lr: float = Field(1e-4, gt=0)
    wd: float = Field(5e-2, ge=0)
    amp: bool = True
    alpha_kd: float = 1.0
    alpha_hint: float = 0.0

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

    config.student = vit_small(patch_size=cfg.patch_size, img_size=cfg.image_size, cls=DinoVisionTransformerTRM)
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

    # trainer.fit(model, datamodule=dm)
    return model,dm


def denorm_to_numpy(img_chw, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    """
    img_chw: torch.Tensor (3, H, W), normalized
    returns: numpy (H, W, 3), float32 in [0,1]
    """
    img = img_chw.detach().cpu().float().clone()
    mean_t = torch.tensor(mean, dtype=img.dtype).view(3, 1, 1)
    std_t = torch.tensor(std, dtype=img.dtype).view(3, 1, 1)
    img = img * std_t + mean_t
    img = img.clamp(0, 1)
    img = img.permute(1, 2, 0).numpy()  # HWC
    return img

def patchtokens_to_featmap(patchtokens, H, W, patch_size=16):
    """
    patchtokens: (B, N, C) where N = (H/P)*(W/P)
    returns:     (B, C, H/P, W/P)
    """
    B, N, C = patchtokens.shape
    Gh, Gw = H // patch_size, W // patch_size
    assert N == Gh * Gw, f"N={N}, expected {Gh*Gw} for H={H}, W={W}, P={patch_size}"
    featmap = patchtokens.reshape(B, Gh, Gw, C).permute(0, 3, 1, 2).contiguous()
    return featmap

if __name__ == "__main__":
    lit,dm = main()
    dm.setup()
    student,teacher = lit.student, lit.teacher

    # Example placeholders (replace with your real data):
    # image: numpy array HxWx3
    # feats: torch tensor (B, C, H/P, W/P)
    H, W = 640, 640
    Gh, Gw = 640//16, 640//16  # e.g., patch size ~16
    it = iter(dm.val_dataloader())
    for images, targets in it:
        teacher.eval()        
        with torch.no_grad():
            out = teacher.forward_features(images)
            patchtokens = out["x_norm_patchtokens"]   # usually (B, N, C)

        # Get image size from tensor
        _, _, H, W = images.shape

        # Reshape tokens -> (B, C, Gh, Gw)
        featmaps = patchtokens_to_featmap(patchtokens, H, W, patch_size=16)

        # Visualize the first sample
        image_np = denorm_to_numpy(images[0])        # (H, W, 3)
        feats_1 = featmaps[0:1]                      # keep batch dim: (1, C, Gh, Gw)

        viewer = DinoPatchSimilarityViewer(image_np, feats_1)
        viewer.show()
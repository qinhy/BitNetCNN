# PyTorch Lightning: All-in-one Ternary CNN for MNIST
# ---------------------------------------------------

# Features
# - Ternary weights (−1/0/+1) with per-channel scaling (STE)
# - Sub-layer normalization (GroupNorm) → ReLU^2 → PACT 8-bit activation → Conv
# - FP first & last layers
# - LightningModule + LightningDataModule
# - Quantized vs FP evaluation using **two validation dataloaders** when --eval-both is set
# - Cosine LR, AdamW, AMP, checkpoints, and TorchScript export

# Usage:
#   python ternary_cnn_mnist_lightning.py --max-epochs 5 --batch-size 128 --eval-both

# Author: ChatGPT
# License: MIT
import argparse
import math
import os
from dataclasses import dataclass
from typing import Tuple, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger


# -----------------------------
# Straight-Through Estimators
# -----------------------------
class TernaryQuantSTE(torch.autograd.Function):
    """
    Ternarize real-valued weights to {-1, 0, +1} with channel-wise thresholding and scaling.
    """
    @staticmethod
    def forward(ctx, w_r: torch.Tensor, delta: float = 0.05, eps: float = 1e-8):
        oc = w_r.shape[0]
        w_view = w_r.view(oc, -1)
        mean_abs = w_view.abs().mean(dim=1, keepdim=True)  # [OC,1]
        t = delta * mean_abs
        m = (w_view.abs() > t).float()
        s = w_view.sign()
        num = (w_view.abs() * m).sum(dim=1, keepdim=True)
        den = m.sum(dim=1, keepdim=True) + eps
        alpha = num / den
        w_q = alpha * s * m
        w_q = w_q.view_as(w_r)
        ctx.save_for_backward(m.view_as(w_r))
        return w_q

    @staticmethod
    def backward(ctx, grad_output):
        (m,) = ctx.saved_tensors
        grad_input = grad_output * (m + 0.5*(1.0 - m))
        return grad_input, None, None


class UniformActQuantSTE(torch.autograd.Function):
    """
    Uniform k-bit activation quantization with learnable clip (PACT-like).
    """
    @staticmethod
    def forward(ctx, x: torch.Tensor, a: torch.Tensor, k_bits: int):
        a_clamped = torch.relu(a) + 1e-8  # ensure positive
        S = (2 ** k_bits - 1) / a_clamped
        y = torch.clamp(x, 0.0, a_clamped.item())
        q = torch.round(y * S) / S
        ctx.save_for_backward(x, a_clamped)
        ctx.k_bits = k_bits
        return q

    @staticmethod
    def backward(ctx, grad_output):
        x, a = ctx.saved_tensors
        pass_through = (x >= 0.0) & (x <= a)
        grad_x = grad_output * pass_through.float()
        grad_a = (grad_output * (x > a).float()).sum().unsqueeze(0)
        return grad_x, grad_a, None


# -----------------------------
# Building Blocks
# -----------------------------
class PACT(nn.Module):
    """Learnable clipping for activations with k-bit uniform quantization."""
    def __init__(self, k_bits: int = 8, init_clip: float = 6.0, enabled: bool = True):
        super().__init__()
        self.k_bits = k_bits
        self.a = nn.Parameter(torch.tensor(init_clip, dtype=torch.float32))
        self.enabled = enabled

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return F.relu(x)
        return UniformActQuantSTE.apply(x, self.a, self.k_bits)


class ReLU2(nn.Module):
    """Squared ReLU activation."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x) ** 2


class BitConv2d(nn.Module):
    """
    Ternary convolution:
      - Pre-norm (GroupNorm) -> ReLU^2 -> PACT quant -> Conv with ternary weights.
      - Supports toggling quantization off for FP evaluation.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        groups: int = 1,
        gn_groups: int = 8,
        act_bits: int = 8,
        use_quant: bool = True,
    ):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups=min(gn_groups, out_channels), num_channels=in_channels, eps=1e-5, affine=True)
        self.act = ReLU2()
        self.pact = PACT(k_bits=act_bits, init_clip=6.0, enabled=use_quant)
        self.weight_fp = nn.Parameter(torch.empty(out_channels, in_channels // groups, kernel_size, kernel_size))
        nn.init.kaiming_normal_(self.weight_fp, nonlinearity='relu')
        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.use_quant = use_quant

    def set_quant(self, enabled: bool):
        self.use_quant = enabled
        self.pact.enabled = enabled

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = self.act(x)
        x = self.pact(x)
        if self.use_quant:
            w = TernaryQuantSTE.apply(self.weight_fp)
        else:
            w = self.weight_fp
        return F.conv2d(x, w, self.bias, stride=self.stride, padding=self.padding, groups=self.groups)


class FirstConv(nn.Module):
    """High-precision first conv (FP) with GN + ReLU^2."""
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, stride: int = 1, padding: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=True)
        nn.init.kaiming_normal_(self.conv.weight, nonlinearity='relu')
        self.norm = nn.GroupNorm(num_groups=min(8, out_ch), num_channels=out_ch, eps=1e-5, affine=True)
        self.act = ReLU2()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class LastFC(nn.Module):
    """High-precision classifier head."""
    def __init__(self, in_features: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.fc(x)


class TernaryCNN(nn.Module):
    """
    A compact ternary CNN for MNIST (1x28x28):
      Stem: FP conv
      Blocks: BitConv2d x N
      Head: GAP -> FP FC
    """
    def __init__(self, channels: Tuple[int, ...] = (32, 64, 64), num_classes: int = 10):
        super().__init__()
        c1, c2, c3 = channels
        self.stem = FirstConv(1, c1, kernel_size=3, stride=1, padding=1)
        self.block1 = BitConv2d(c1, c2, kernel_size=3, stride=2, padding=1)  # 14x14
        self.block2 = BitConv2d(c2, c3, kernel_size=3, stride=2, padding=1)  # 7x7
        self.block3 = BitConv2d(c3, c3, kernel_size=3, stride=1, padding=1)  # 7x7
        self.head_norm = nn.GroupNorm(num_groups=min(8, c3), num_channels=c3, eps=1e-5, affine=True)
        self.head_act = ReLU2()
        self.classifier = LastFC(c3, num_classes)

    def set_quantization(self, enabled: bool):
        self.block1.set_quant(enabled)
        self.block2.set_quant(enabled)
        self.block3.set_quant(enabled)

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.head_norm(x)
        x = self.head_act(x)
        x = x.mean(dim=[2, 3])
        logits = self.classifier(x)
        return logits


# -----------------------------
# Lightning Modules
# -----------------------------
@dataclass
class TrainConfig:
    lr: float = 1e-3
    wd: float = 1e-4
    label_smoothing: float = 0.0
    cosine: bool = True


class LitTernaryMNIST(pl.LightningModule):
    def __init__(self, channels=(32, 64, 64), num_classes=10, cfg: TrainConfig = TrainConfig(), eval_both: bool = False):
        super().__init__()
        self.save_hyperparameters(ignore=['cfg'])
        self.model = TernaryCNN(channels=tuple(channels), num_classes=num_classes)
        self.cfg = cfg
        self.eval_both = eval_both
        # ensure quantization is on during training
        self.model.set_quantization(True)

    def forward(self, x):
        return self.model(x)

    # ------------- training / validation / test -------------
    def training_step(self, batch, batch_idx):
        x, y = batch
        # enforce quantized path during training
        self.model.set_quantization(True)
        logits = self(x)
        loss = F.cross_entropy(logits, y, label_smoothing=self.cfg.label_smoothing)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log('train/loss', loss, prog_bar=False, on_step=True, on_epoch=True)
        self.log('train/acc', acc, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        # dataloader_idx=0 => quantized; 1 => FP
        quantized = True if (not self.eval_both or dataloader_idx == 0) else False
        if hasattr(self.model, 'set_quantization'):
            self.model.set_quantization(quantized)
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        tag = 'val_q' if quantized else 'val_fp'
        self.log(f'{tag}/loss', loss, prog_bar=False, on_step=False, on_epoch=True, sync_dist=False)
        self.log(f'{tag}/acc', acc, prog_bar=True, on_step=False, on_epoch=True, sync_dist=False)

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        quantized = True if (not self.eval_both or dataloader_idx == 0) else False
        if hasattr(self.model, 'set_quantization'):
            self.model.set_quantization(quantized)
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        tag = 'test_q' if quantized else 'test_fp'
        self.log(f'{tag}/loss', loss, prog_bar=False, on_step=False, on_epoch=True)
        self.log(f'{tag}/acc', acc, prog_bar=True, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        optim = torch.optim.AdamW(self.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.wd, betas=(0.9, 0.95))
        if self.cfg.cosine:
            # Step-wise cosine over total training steps
            return {
                "optimizer": optim,
                "lr_scheduler": {
                    "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(
                        optim, T_max=self.trainer.estimated_stepping_batches, eta_min=self.cfg.lr * 0.01
                    ),
                    "interval": "step",
                    "frequency": 1,
                },
            }
        return optim


class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "./data", batch_size: int = 128, num_workers: int = 2, eval_both: bool = False):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.eval_both = eval_both

    def prepare_data(self):
        datasets.MNIST(self.data_dir, train=True, download=True)
        datasets.MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage: Optional[str] = None):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        full_train = datasets.MNIST(self.data_dir, train=True, transform=transform)
        self.train_set, self.val_set = random_split(full_train, [55000, 5000], generator=torch.Generator().manual_seed(123))
        self.test_set = datasets.MNIST(self.data_dir, train=False, transform=transform)

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        val_loader = DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False,
                                num_workers=self.num_workers, pin_memory=True)
        if self.eval_both:
            # second loader used for FP eval; model toggles based on dataloader_idx
            return [val_loader, val_loader]
        return val_loader

    def test_dataloader(self):
        test_loader = DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False,
                                 num_workers=self.num_workers, pin_memory=True)
        if self.eval_both:
            return [test_loader, test_loader]
        return test_loader


# -----------------------------
# Main / CLI
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Lightning Ternary CNN for MNIST")
    parser.add_argument("--channels", type=str, default="32,64,64")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--no-cosine", action="store_true")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--max-epochs", type=int, default=5)
    parser.add_argument("--eval-both", action="store_true", help="evaluate quantized and full-precision in parallel")
    parser.add_argument("--precision", type=str, default="16-mixed", help="Lightning precision (e.g., 16-mixed, 32-true)")
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--accelerator", type=str, default="auto")
    args = parser.parse_args()

    channels = tuple(int(x) for x in args.channels.split(","))
    cfg = TrainConfig(lr=args.lr, wd=args.wd, label_smoothing=args.label_smoothing, cosine=(not args.no_cosine))

    # Data & model
    dm = MNISTDataModule(data_dir=args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers, eval_both=args.eval_both)
    model = LitTernaryMNIST(channels=channels, cfg=cfg, eval_both=args.eval_both)

    # Logging & checkpoints
    logger = CSVLogger(save_dir="logs", name="ternary_mnist")
    ckpt_cb = ModelCheckpoint(
        dirpath="checkpoints",
        filename="ternary-mnist-{epoch:02d}-{val_q_acc:.4f}",
        monitor="val_q/acc",
        mode="max",
        save_top_k=1
    )
    lr_cb = LearningRateMonitor(logging_interval='step')

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        precision=args.precision,
        logger=logger,
        callbacks=[ckpt_cb, lr_cb],
        log_every_n_steps=50
    )

    # Train
    trainer.fit(model, datamodule=dm)

    # Test on best checkpoint if available
    best_path = ckpt_cb.best_model_path
    if best_path and os.path.exists(best_path):
        print(f"Testing best checkpoint: {best_path}")
        best_model = LitTernaryMNIST.load_from_checkpoint(best_path, channels=channels, cfg=cfg, eval_both=args.eval_both)
        trainer.test(best_model, datamodule=dm)
        model_to_export = best_model
    else:
        trainer.test(model, datamodule=dm)
        model_to_export = model

    # Export TorchScript (quantized + FP)
    os.makedirs("exports", exist_ok=True)
    model_to_export.eval()
    example = torch.randn(1, 1, 28, 28)
    # Quantized export
    if hasattr(model_to_export.model, 'set_quantization'):
        model_to_export.model.set_quantization(True)
    ts_q = torch.jit.trace(model_to_export.model, example)
    ts_q.save("exports/ternary_mnist_quantized.ts")
    # FP export
    if hasattr(model_to_export.model, 'set_quantization'):
        model_to_export.model.set_quantization(False)
    ts_fp = torch.jit.trace(model_to_export.model, example)
    ts_fp.save("exports/ternary_mnist_fp.ts")
    print("Saved TorchScript models to exports/")

if __name__ == "__main__":
    main()
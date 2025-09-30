# mnist_bitnet_lightning.py
import argparse, math, os, copy
import torch
torch.set_float32_matmul_precision('high')
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# --- Lightning imports ---
import pytorch_lightning as pl
from pytorch_lightning import Callback
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from torchmetrics.classification import MulticlassAccuracy

EPS = 1e-12

# ----------------------------
# Utilities
# ----------------------------
@torch.no_grad()
def _pow2_quantize_scale(s: torch.Tensor, min_exp: int = -32, max_exp: int = 31):
    """
    s > 0 tensor -> (s_q, k) where s_q = 2^k and k is int8 clamped to [min_exp, max_exp].
    """
    s = s.clamp_min(EPS).float()
    k = torch.round(torch.log2(s)).clamp(min_exp, max_exp).to(torch.int8)
    s_q = torch.pow(2.0, k.to(torch.float32))
    return s_q, k

def _reduce_abs(x, keep_dim, op="mean"):
    dims = [d for d in range(x.dim()) if d != keep_dim]
    a = x.abs()
    if op == "mean":
        s = a.mean(dim=dims, keepdim=True)
    elif op == "median":
        # median over flattened other dims
        perm = (keep_dim,) + tuple(d for d in range(x.dim()) if d != keep_dim)
        flat = a.permute(perm).contiguous().view(a.size(keep_dim), -1)
        s = flat.median(dim=1).values.view([a.size(keep_dim)] + [1]*(x.dim()-1))
        inv = [0]*x.dim()
        for i,p in enumerate(perm): inv[p] = i
        s = s.permute(*inv).contiguous()
    else:
        raise ValueError("op must be 'mean' or 'median'")
    return s.clamp_min(EPS)

class Bit:
    # ----------------------------
    # Fake-quant building blocks (QAT) — activation quantization removed
    # ----------------------------
    class Bit1p58Weight(nn.Module):
        """1.58-bit (ternary) weight quantizer with per-out-channel scaling."""
        def __init__(self, dim=0, scale_op="median"):
            super().__init__()
            self.dim = dim
            self.scale_op = scale_op

        def forward(self, w):
            s = _reduce_abs(w, keep_dim=self.dim, op=self.scale_op)
            w_bar = (w / s).detach()
            w_q = torch.round(w_bar).clamp_(-1, 1)
            return w + (w_q * s - w).detach()

    # ----------------------------
    # Inference (frozen) ternary modules — no activation quantization
    # ----------------------------
    class Conv2dInfer(nn.Module):
        """
        Frozen ternary conv:
        y = (Conv(x, Wq) * s_per_out) + b
        Wq is stored as int8 in {-1,0,+1}. s is float per output channel.
        """
        def __init__(self, w_q, s, bias, stride, padding, dilation, groups):
            super().__init__()
            self.register_buffer("w_q", w_q.to(torch.int8))
            self.register_buffer("s", s)  # [out,1,1]
            self.register_buffer("bias", None if bias is None else bias)
            self.stride, self.padding, self.dilation, self.groups = stride, padding, dilation, groups

        def forward(self, x):
            y = F.conv2d(x, self.w_q.float(), None, self.stride, self.padding, self.dilation, self.groups)
            y = y * self.s  # broadcast over H,W
            if self.bias is not None:
                y = y + self.bias.view(1, -1, 1, 1)
            return y

    class LinearInfer(nn.Module):
        """Frozen ternary linear: y = (x @ Wq^T) * s + b"""
        def __init__(self, w_q, s, bias):
            super().__init__()
            self.register_buffer("w_q", w_q.to(torch.int8))  # [out,in]
            self.register_buffer("s", s)                     # [out]
            self.register_buffer("bias", None if bias is None else bias)

        def forward(self, x):
            y = F.linear(x, self.w_q.float(), None)
            y = y * self.s
            if self.bias is not None:
                y = y + self.bias
            return y

    class Conv2dInferP2(nn.Module):
        """
        Ternary conv with power-of-two scales:
        - We store s_exp per output channel, so scaling is x * 2^s_exp (shift on integer backends).
        """
        def __init__(self, w_q, s_exp, bias, stride, padding, dilation, groups):
            super().__init__()
            self.register_buffer("w_q", w_q.to(torch.int8))
            self.register_buffer("s_exp", s_exp.to(torch.int8))      # [out,1,1]
            self.register_buffer("bias", None if bias is None else bias)
            self.stride, self.padding, self.dilation, self.groups = stride, padding, dilation, groups

        def forward(self, x):
            y = F.conv2d(x, self.w_q.float(), None, self.stride, self.padding, self.dilation, self.groups)
            y = torch.ldexp(y, self.s_exp.to(torch.int32))
            if self.bias is not None:
                y = y + self.bias.view(1, -1, 1, 1)
            return y

    class LinearInferP2(nn.Module):
        """Ternary linear with power-of-two output scales."""
        def __init__(self, w_q, s_exp, bias):
            super().__init__()
            self.register_buffer("w_q", w_q.to(torch.int8))  # [out,in]
            self.register_buffer("s_exp", s_exp.to(torch.int8))  # [out]
            self.register_buffer("bias", None if bias is None else bias)

        def forward(self, x):
            y = F.linear(x, self.w_q.float(), None)
            y = torch.ldexp(y, self.s_exp.to(torch.int32))
            if self.bias is not None:
                y = y + self.bias
            return y

    # ----------------------------
    # Train-time modules (no BatchNorm), activation quantization removed
    # ----------------------------
    class Conv2d(nn.Module):
        """
        Conv2d with ternary weights (fake-quant for training).
        No BatchNorm inside. Add your own nonlinearity outside if desired.
        """
        def __init__(self, in_c, out_c, kernel_size, stride=1, padding=0, dilation=1, groups=1,
                    bias=True, scale_op="median"):
            super().__init__()
            if isinstance(kernel_size, int):
                kh = kw = kernel_size
            else:
                kh, kw = kernel_size
            self.weight = nn.Parameter(torch.empty(out_c, in_c // groups, kh, kw))
            nn.init.kaiming_normal_(self.weight, nonlinearity="relu")
            self.bias = nn.Parameter(torch.zeros(out_c)) if bias else None
            self.w_q = Bit.Bit1p58Weight(dim=0, scale_op=scale_op)
            self.stride, self.padding, self.dilation, self.groups = stride, padding, dilation, groups
            self.scale_op = scale_op

        def forward(self, x):
            wq = self.w_q(self.weight)
            return F.conv2d(x, wq, self.bias, self.stride, self.padding, self.dilation, self.groups)

        @torch.no_grad()
        def to_ternary(self):
            """
            Convert this layer into a frozen Bit.Conv2dInfer, carrying over:
            - per-out-channel weight scale s and Wq in {-1,0,+1}
            """
            w = self.weight.data
            s_vec = _reduce_abs(w, keep_dim=0, op=self.scale_op).squeeze()   # [out]
            s = s_vec.view(-1, 1, 1)                                         # [out,1,1] for conv broadcast
            w_bar = w / s_vec.view(-1, 1, 1, 1)
            w_q = torch.round(w_bar).clamp_(-1, 1).to(w.dtype)

            return Bit.Conv2dInfer(
                w_q=w_q, s=s,
                bias=(None if self.bias is None else self.bias.data.clone()),
                stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups
            )

        @torch.no_grad()
        def to_ternary_p2(self):
            # Per-out-channel scale from your chosen op
            w = self.weight.data
            s_vec = _reduce_abs(w, keep_dim=0, op=self.scale_op).squeeze()  # [out]
            w_bar = w / s_vec.view(-1,1,1,1)
            w_q = torch.round(w_bar).clamp_(-1, 1).to(w.dtype)

            # Quantize weight scale to power-of-two (save exponents)
            _, s_exp = _pow2_quantize_scale(s_vec)           # int8 exponents
            s_exp = s_exp.view(-1, 1, 1)

            return Bit.Conv2dInferP2(
                w_q=w_q,
                s_exp=s_exp,
                bias=(None if self.bias is None else self.bias.data.clone()),
                stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups
            )

    class Linear(nn.Module):
        def __init__(self, in_f, out_f, bias=True, scale_op="median"):
            super().__init__()
            self.weight = nn.Parameter(torch.empty(out_f, in_f))
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
            self.bias = nn.Parameter(torch.zeros(out_f)) if bias else None
            self.w_q = Bit.Bit1p58Weight(dim=0, scale_op=scale_op)
            self.scale_op = scale_op

        def forward(self, x):
            wq = self.w_q(self.weight)
            return F.linear(x, wq, self.bias)

        @torch.no_grad()
        def to_ternary(self):
            w = self.weight.data
            s = _reduce_abs(w, keep_dim=0, op=self.scale_op).squeeze()  # [out]
            w_q = torch.round(w / s.view(-1,1)).clamp_(-1, 1).to(w.dtype)
            return Bit.LinearInfer(
                w_q=w_q, s=s, bias=(None if self.bias is None else self.bias.data.clone())
            )

        @torch.no_grad()
        def to_ternary_p2(self):
            w = self.weight.data
            s = _reduce_abs(w, keep_dim=0, op=self.scale_op).squeeze()   # [out]
            w_q = torch.round((w / s.view(-1,1))).clamp_(-1, 1).to(w.dtype)

            # Quantize to power-of-two
            _, s_exp = _pow2_quantize_scale(s)   # [out] int8
            return Bit.LinearInferP2(
                w_q=w_q, s_exp=s_exp, bias=(None if self.bias is None else self.bias.data.clone())
            )

# ----------------------------
# Model-wide conversion helper
# ----------------------------
@torch.no_grad()
def convert_to_ternary(module: nn.Module) -> nn.Module:
    """
    Recursively replace Bit.Conv2d/Bit.Linear with Ternary*Infer modules.
    Returns a new nn.Module (original left untouched if you deepcopy before).
    """
    for name, child in list(module.named_children()):
        if hasattr(child, 'to_ternary'):
            setattr(module, name, child.to_ternary())
        else:
            convert_to_ternary(child)
    return module

@torch.no_grad()
def convert_to_ternary_p2(module: nn.Module) -> nn.Module:
    """
    Recursively replace Bit.Conv2d/Bit.Linear with their PoT inference counterparts.
    """
    for name, child in list(module.named_children()):
        if hasattr(child, 'to_ternary_p2'):
            setattr(module, name, child.to_ternary_p2())
        else:
            convert_to_ternary_p2(child)
    return module

# ----------------------------
# Simple BitNet block & model
# ----------------------------
class InvertedResidualBit(nn.Module):
    def __init__(self, in_c, out_c, expand, stride, scale_op="median", conv2d=Bit.Conv2d):
        super().__init__()
        hid = in_c * expand
        self.use_res = (stride == 1 and in_c == out_c)

        self.pw1 = nn.Sequential(
            conv2d(in_c, hid, kernel_size=1, bias=True, scale_op=scale_op),
            nn.BatchNorm2d(hid),
            nn.SiLU(inplace=True),
        )
        self.dw = nn.Sequential(
            conv2d(hid, hid, kernel_size=3, stride=stride, padding=1, groups=hid,
                      bias=True, scale_op=scale_op),
            nn.BatchNorm2d(hid),
            nn.SiLU(inplace=True),
        )
        # no activation here
        self.pw2 = nn.Sequential(
            conv2d(hid, out_c, kernel_size=1, bias=True, scale_op=scale_op),
            nn.BatchNorm2d(out_c),
        )

    def forward(self, x):
        y = self.pw2(self.dw(self.pw1(x)))
        return x + y if self.use_res else y


class NetCNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=10, expand_ratio = 5,
                 drop2d_p=0.05, drop_p=0.1, mod=nn):
        super().__init__()
        if mod==nn:
            self.is_bit = False
        elif mod==Bit:
            self.is_bit = True
            
        
        self.stem = nn.Sequential(
            mod.Conv2d(in_channels, 2**expand_ratio, kernel_size=3, stride=1, padding=1,
                      bias=True),
            nn.BatchNorm2d(2**expand_ratio),
            nn.SiLU(inplace=True),
        )

        self.stage1 = InvertedResidualBit(2**expand_ratio, 2**(expand_ratio+1), expand=2, stride=2)
        self.sd1 = nn.Dropout2d(p=drop2d_p)

        self.stage2 = InvertedResidualBit(2**(expand_ratio+1), 2**(expand_ratio+2), expand=2, stride=2)
        self.sd2 = nn.Dropout2d(p=drop2d_p)

        self.stage3 = InvertedResidualBit(2**(expand_ratio+2), 2**(expand_ratio+3), expand=2, stride=2)

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(p=drop_p),
            mod.Linear(2**(expand_ratio+3), num_classes, bias=True),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x); x = self.sd1(x)
        x = self.stage2(x); x = self.sd2(x)
        x = self.stage3(x)
        return self.head(x)
    
    def ternary(self):
        if not self.is_bit:return copy.deepcopy(self)
        return convert_to_ternary(copy.deepcopy(self))
    
    def ternary_p2(self):
        if not self.is_bit:return copy.deepcopy(self)
        return convert_to_ternary_p2(copy.deepcopy(self))


# ----------------------------
# Lightning DataModule
# ----------------------------
class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int, num_workers: int = 4):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        train_tfm = transforms.Compose([
            transforms.RandomCrop(28, padding=2),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        test_tfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.train_ds = datasets.MNIST(root=self.data_dir, train=True, download=True, transform=train_tfm)
        self.val_ds   = datasets.MNIST(root=self.data_dir, train=False, download=True, transform=test_tfm)
        self.test_ds  = self.val_ds  # same for MNIST

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers, pin_memory=True,persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=True,persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=True,persistent_workers=True)

# ----------------------------
# LightningModule
# ----------------------------
class LitNetCNN(pl.LightningModule):
    def __init__(self,lr, wd, epochs, expand_ratio=5, eval_ternary=True, mod=Bit):
        super().__init__()
        self.save_hyperparameters(ignore=['mod'])
        self.model = NetCNN(in_channels=1, num_classes=10, mod=mod, expand_ratio=expand_ratio)
        self.crit = nn.CrossEntropyLoss()
        self.acc_fp = MulticlassAccuracy(num_classes=10)
        self.acc_tern = MulticlassAccuracy(num_classes=10) if eval_ternary else None
        self.best_ternary_acc = -1.0
        self._ternary_snapshot = None  # set each val epoch

    def forward(self, x):
        return self.model(x)

    def on_fit_start(self):
        n_params = sum(p.numel() for p in self.model.parameters())
        acc_name = self.trainer.accelerator.__class__.__name__
        strategy_name = self.trainer.strategy.__class__.__name__
        num_devices = getattr(self.trainer, "num_devices", None) or len(self.trainer.devices or [])
        dev_str = str(self.device)
        # Optional CUDA name for nicer logs
        cuda_name = ""
        if torch.cuda.is_available() and "cuda" in dev_str:
            try:
                cuda_name = f" | CUDA: {torch.cuda.get_device_name(self.device.index or 0)}"
            except Exception:
                pass
        self.print(
            f"Model params: {n_params/1e6:.2f}M | Accelerator: {acc_name} "
            f"| Devices: {num_devices} | Strategy: {strategy_name} | Device: {dev_str}{cuda_name}"
        )


    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.crit(logits, y)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def on_validation_epoch_start(self):
        print()
        # Build one frozen ternary snapshot per epoch (like your evaluate(model.ternary()))
        if self.hparams.eval_ternary:
            self._ternary_snapshot = self.model.ternary().to(self.device).eval()
        else:
            self._ternary_snapshot = None

    def validation_step(self, batch, batch_idx):
        x, y = batch
        # FP eval (training graph model)
        logits_fp = self(x)
        loss_fp = self.crit(logits_fp, y)
        acc_fp = self.acc_fp(logits_fp.softmax(dim=1), y)
        self.log("val_loss", loss_fp, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc_fp, on_step=False, on_epoch=True, prog_bar=True)

        # Ternary-PoT eval snapshot
        if self._ternary_snapshot is not None:
            with torch.no_grad():
                logits_t = self._ternary_snapshot(x)
                loss_t = self.crit(logits_t, y)
                acc_t = self.acc_tern(logits_t.softmax(dim=1), y)
            self.log("val_loss_ternary", loss_t, on_step=False, on_epoch=True, prog_bar=False)
            self.log("val_acc_ternary", acc_t, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.wd)
        # cosine schedule across full training (per-epoch)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.hparams.epochs, eta_min=self.hparams.lr*0.01)
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": sched,
                "interval": "epoch",
                "monitor": "val_acc_ternary" if self.hparams.eval_ternary else "val_acc",
            },
        }

# ----------------------------
# Callback to export frozen ternary whenever we get a new best
# ----------------------------
class ExportBestTernary(Callback):
    def __init__(self, out_dir: str, monitor: str = "val_acc_ternary", mode: str = "max"):
        super().__init__()
        self.out_dir = out_dir
        self.monitor = monitor
        self.mode = mode
        self.best = None

    def _is_better(self, current, best):
        if best is None:
            return True
        return (current > best) if self.mode == "max" else (current < best)

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: LitNetCNN):
        metrics = trainer.callback_metrics
        if self.monitor not in metrics:
            return
        current = metrics[self.monitor].item()
        if self._is_better(current, self.best):
            self.best = current
            os.makedirs(self.out_dir, exist_ok=True)
            # Save Lightning checkpoint (already handled by ModelCheckpoint too)
            # Export ternary snapshot
            ternary = pl_module.model.ternary().cpu().eval()
            ternary_path = os.path.join(self.out_dir, "mnist_bitnet_ternary.pt")
            torch.save({"model": ternary.state_dict(), "acc": current}, ternary_path)
            pl_module.print(f"✓ Exported frozen ternary model to {ternary_path} (val_acc_ternary={current*100:.2f}%)")

# ----------------------------
# CLI / main
# ----------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default="./data", help="MNIST data dir")
    p.add_argument("--out",  type=str, default="./mnist_ckpt", help="where to save checkpoints")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--scale-op", type=str, default="median", choices=["mean", "median"])
    p.add_argument("--amp", action="store_true", help="enable mixed precision (Lightning precision=16-mixed)")
    p.add_argument("--cpu", action="store_true", help="force CPU")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-eval-ternary", action="store_true", help="skip ternary validation/export")
    return p.parse_args()

def train(mod=Bit):
    args = parse_args()
    pl.seed_everything(args.seed, workers=True)

    # Data
    dm = MNISTDataModule(data_dir=args.data, batch_size=args.batch_size, num_workers=4)

    # Model
    lit_model = LitNetCNN(
        lr=args.lr, wd=args.wd, epochs=args.epochs,
        eval_ternary=not args.no_eval_ternary,
        mod=mod,
        expand_ratio=5,
    )

    # Logging & callbacks
    os.makedirs(args.out, exist_ok=True)
    logger = CSVLogger(save_dir=args.out, name="logs")
    monitor_metric = "val_acc_ternary" if not args.no_eval_ternary else "val_acc"
    ckpt_cb = ModelCheckpoint(
        monitor=monitor_metric, mode="max", save_top_k=1, save_last=True
    )
    lr_cb = LearningRateMonitor(logging_interval="epoch")
    export_cb = ExportBestTernary(args.out, monitor=monitor_metric, mode="max") if not args.no_eval_ternary else None
    callbacks = [ckpt_cb, lr_cb] + ([export_cb] if export_cb is not None else [])

    # Trainer
    accelerator = "cpu" if args.cpu else "auto"
    precision = "16-mixed" if args.amp else "32-true"
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator=accelerator,
        devices=1,
        precision=precision,
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=50,
        deterministic=False,
    )

    # Fit + (optional) test
    trainer.fit(lit_model, datamodule=dm)
    trainer.validate(lit_model, datamodule=dm)
    return lit_model

    # Example: manual conversion after training
    # ternary_model = convert_to_ternary_p2(copy.deepcopy(lit_model.model)).cpu().eval()
    # for n, b in ternary_model.named_buffers():
    #     print(n, b.flatten()[:10], tuple(b.shape), b.dtype)

if __name__ == "__main__":
    lit_model = train(mod=Bit)

# uv run BitNetCNN.py
# Seed set to 42
# GPU available: True (cuda), used: True
# TPU available: False, using: 0 TPU cores
# HPU available: False, using: 0 HPUs
# LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]

#   | Name     | Type               | Params | Mode
# --------------------------------------------------------
# 0 | model    | NetCNN             | 140 K  | train
# 1 | crit     | CrossEntropyLoss   | 0      | train
# 2 | acc_fp   | MulticlassAccuracy | 0      | train
# 3 | acc_tern | MulticlassAccuracy | 0      | train
# --------------------------------------------------------
# 140 K     Trainable params
# 0         Non-trainable params
# 140 K     Total params
# 0.560     Total estimated model params size (MB)
# 62        Modules in train mode
# 0         Modules in eval mode
# ✓ Exported frozen ternary model to ./mnist_ckpt\mnist_bitnet_ternary.pt (val_acc_ternary=9.90%)
# Epoch 0: 100%|█████████| 30/30 [00:09<00:00,  3.08it/s, v_num=18]
# ✓ Exported frozen ternary model to ./mnist_ckpt\mnist_bitnet_ternary.pt (val_acc_ternary=15.14%)
# Epoch 1: 100%|█████████| 30/30 [00:06<00:00,  4.45it/s, v_num=18, val_loss=2.840, val_acc=0.152, val_acc_ternary=0.151, train_loss=1.760]
# ✓ Exported frozen ternary model to ./mnist_ckpt\mnist_bitnet_ternary.pt (val_acc_ternary=85.76%)
# Epoch 2: 100%|█████████| 30/30 [00:06<00:00,  4.52it/s, v_num=18, val_loss=0.444, val_acc=0.858, val_acc_ternary=0.858, train_loss=0.863]
# ✓ Exported frozen ternary model to ./mnist_ckpt\mnist_bitnet_ternary.pt (val_acc_ternary=91.30%)
# Epoch 3: 100%|█████████| 30/30 [00:06<00:00,  4.52it/s, v_num=18, val_loss=0.290, val_acc=0.913, val_acc_ternary=0.913, train_loss=0.392]
# Epoch 4: 100%|█████████| 30/30 [00:06<00:00,  4.49it/s, v_num=18, val_loss=0.290, val_acc=0.909, val_acc_ternary=0.909, train_loss=0.240]
# ✓ Exported frozen ternary model to ./mnist_ckpt\mnist_bitnet_ternary.pt (val_acc_ternary=95.99%)
# Epoch 5: 100%|█████████| 30/30 [00:06<00:00,  4.49it/s, v_num=18, val_loss=0.128, val_acc=0.960, val_acc_ternary=0.960, train_loss=0.185]
# ✓ Exported frozen ternary model to ./mnist_ckpt\mnist_bitnet_ternary.pt (val_acc_ternary=96.89%)
# Epoch 6: 100%|████████| 30/30 [00:06<00:00,  4.45it/s, v_num=18, val_loss=0.0958, val_acc=0.969, val_acc_ternary=0.969, train_loss=0.149]
# ✓ Exported frozen ternary model to ./mnist_ckpt\mnist_bitnet_ternary.pt (val_acc_ternary=97.69%)
# Epoch 7: 100%|████████| 30/30 [00:06<00:00,  4.53it/s, v_num=18, val_loss=0.0767, val_acc=0.977, val_acc_ternary=0.977, train_loss=0.130]
# Epoch 8: 100%|████████| 30/30 [00:06<00:00,  4.48it/s, v_num=18, val_loss=0.0817, val_acc=0.975, val_acc_ternary=0.975, train_loss=0.119]
# Epoch 9: 100%|████████| 30/30 [00:06<00:00,  4.37it/s, v_num=18, val_loss=0.0928, val_acc=0.971, val_acc_ternary=0.971, train_loss=0.106]
# Epoch 10: 100%|███████| 30/30 [00:06<00:00,  4.58it/s, v_num=18, val_loss=0.0788, val_acc=0.975, val_acc_ternary=0.975, train_loss=0.102]
# ✓ Exported frozen ternary model to ./mnist_ckpt\mnist_bitnet_ternary.pt (val_acc_ternary=97.70%)
# Epoch 11: 100%|██████| 30/30 [00:06<00:00,  4.58it/s, v_num=18, val_loss=0.0681, val_acc=0.977, val_acc_ternary=0.977, train_loss=0.0913]
# ✓ Exported frozen ternary model to ./mnist_ckpt\mnist_bitnet_ternary.pt (val_acc_ternary=98.30%)
# Epoch 12: 100%|██████| 30/30 [00:06<00:00,  4.58it/s, v_num=18, val_loss=0.0517, val_acc=0.983, val_acc_ternary=0.983, train_loss=0.0872]
# Epoch 13: 100%|██████| 30/30 [00:06<00:00,  4.56it/s, v_num=18, val_loss=0.0523, val_acc=0.982, val_acc_ternary=0.982, train_loss=0.0827]
# Epoch 14: 100%|███████| 30/30 [00:06<00:00,  4.57it/s, v_num=18, val_loss=0.0559, val_acc=0.982, val_acc_ternary=0.982, train_loss=0.080]
# Epoch 15: 100%|██████| 30/30 [00:06<00:00,  4.52it/s, v_num=18, val_loss=0.0556, val_acc=0.982, val_acc_ternary=0.982, train_loss=0.0739]
# Epoch 16: 100%|██████| 30/30 [00:06<00:00,  4.59it/s, v_num=18, val_loss=0.0746, val_acc=0.977, val_acc_ternary=0.977, train_loss=0.0738]
# Epoch 17: 100%|███████| 30/30 [00:06<00:00,  4.58it/s, v_num=18, val_loss=0.0554, val_acc=0.981, val_acc_ternary=0.981, train_loss=0.072]
# ✓ Exported frozen ternary model to ./mnist_ckpt\mnist_bitnet_ternary.pt (val_acc_ternary=98.44%)
# Epoch 18: 100%|████████| 30/30 [00:06<00:00,  4.52it/s, v_num=18, val_loss=0.046, val_acc=0.985, val_acc_ternary=0.984, train_loss=0.067]
# Epoch 19: 100%|██████| 30/30 [00:06<00:00,  4.55it/s, v_num=18, val_loss=0.0504, val_acc=0.984, val_acc_ternary=0.984, train_loss=0.0668]
# ✓ Exported frozen ternary model to ./mnist_ckpt\mnist_bitnet_ternary.pt (val_acc_ternary=98.78%)
# Epoch 20: 100%|██████| 30/30 [00:06<00:00,  4.48it/s, v_num=18, val_loss=0.0386, val_acc=0.988, val_acc_ternary=0.988, train_loss=0.0647]
# Epoch 21: 100%|██████| 30/30 [00:06<00:00,  4.53it/s, v_num=18, val_loss=0.0571, val_acc=0.982, val_acc_ternary=0.982, train_loss=0.0622]
# Epoch 22: 100%|██████| 30/30 [00:06<00:00,  4.57it/s, v_num=18, val_loss=0.0476, val_acc=0.985, val_acc_ternary=0.985, train_loss=0.0595]
# Epoch 23: 100%|███████| 30/30 [00:06<00:00,  4.56it/s, v_num=18, val_loss=0.0644, val_acc=0.978, val_acc_ternary=0.978, train_loss=0.059]
# Epoch 24: 100%|██████| 30/30 [00:06<00:00,  4.54it/s, v_num=18, val_loss=0.0508, val_acc=0.982, val_acc_ternary=0.982, train_loss=0.0606]
# Epoch 25: 100%|██████| 30/30 [00:06<00:00,  4.55it/s, v_num=18, val_loss=0.0406, val_acc=0.985, val_acc_ternary=0.986, train_loss=0.0537]
# Epoch 26: 100%|██████| 30/30 [00:06<00:00,  4.50it/s, v_num=18, val_loss=0.0642, val_acc=0.979, val_acc_ternary=0.979, train_loss=0.0541]
# Epoch 27: 100%|██████| 30/30 [00:06<00:00,  4.60it/s, v_num=18, val_loss=0.0659, val_acc=0.978, val_acc_ternary=0.978, train_loss=0.0538]
# Epoch 28: 100%|██████| 30/30 [00:06<00:00,  4.55it/s, v_num=18, val_loss=0.0394, val_acc=0.986, val_acc_ternary=0.986, train_loss=0.0541]
# Epoch 29: 100%|██████| 30/30 [00:06<00:00,  4.58it/s, v_num=18, val_loss=0.0498, val_acc=0.983, val_acc_ternary=0.983, train_loss=0.0504]
# Epoch 30: 100%|██████| 30/30 [00:06<00:00,  4.29it/s, v_num=18, val_loss=0.0503, val_acc=0.982, val_acc_ternary=0.982, train_loss=0.0518]
# Epoch 31: 100%|██████| 30/30 [00:06<00:00,  4.38it/s, v_num=18, val_loss=0.0559, val_acc=0.983, val_acc_ternary=0.983, train_loss=0.0532]
# Epoch 32: 100%|██████| 30/30 [00:06<00:00,  4.36it/s, v_num=18, val_loss=0.0683, val_acc=0.979, val_acc_ternary=0.979, train_loss=0.0502]
# Epoch 33: 100%|██████| 30/30 [00:06<00:00,  4.50it/s, v_num=18, val_loss=0.0409, val_acc=0.987, val_acc_ternary=0.987, train_loss=0.0522]
# ✓ Exported frozen ternary model to ./mnist_ckpt\mnist_bitnet_ternary.pt (val_acc_ternary=98.79%)
# Epoch 34: 100%|██████| 30/30 [00:06<00:00,  4.47it/s, v_num=18, val_loss=0.0372, val_acc=0.988, val_acc_ternary=0.988, train_loss=0.0493]
# ✓ Exported frozen ternary model to ./mnist_ckpt\mnist_bitnet_ternary.pt (val_acc_ternary=98.94%)
# Epoch 35: 100%|██████| 30/30 [00:06<00:00,  4.43it/s, v_num=18, val_loss=0.0339, val_acc=0.989, val_acc_ternary=0.989, train_loss=0.0471]
# Epoch 36: 100%|██████| 30/30 [00:06<00:00,  4.46it/s, v_num=18, val_loss=0.0361, val_acc=0.989, val_acc_ternary=0.988, train_loss=0.0473]
# ✓ Exported frozen ternary model to ./mnist_ckpt\mnist_bitnet_ternary.pt (val_acc_ternary=99.08%)
# Epoch 37: 100%|██████| 30/30 [00:06<00:00,  4.50it/s, v_num=18, val_loss=0.0265, val_acc=0.991, val_acc_ternary=0.991, train_loss=0.0452]
# Epoch 38: 100%|██████| 30/30 [00:06<00:00,  4.47it/s, v_num=18, val_loss=0.0607, val_acc=0.982, val_acc_ternary=0.982, train_loss=0.0451]
# Epoch 39: 100%|██████| 30/30 [00:06<00:00,  4.52it/s, v_num=18, val_loss=0.0336, val_acc=0.990, val_acc_ternary=0.990, train_loss=0.0482]
# Epoch 40: 100%|███████| 30/30 [00:06<00:00,  4.43it/s, v_num=18, val_loss=0.0502, val_acc=0.985, val_acc_ternary=0.985, train_loss=0.046]
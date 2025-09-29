import argparse
import math
import os
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# -----------------------------
# Utilities
# -----------------------------
def exists(x):
    return x is not None


def make_dir(path: str):
    os.makedirs(path, exist_ok=True)


# -----------------------------
# Straight-Through Estimators
# -----------------------------
class TernaryQuantSTE(torch.autograd.Function):
    """
    Ternarize real-valued weights to {-1, 0, +1} with channel-wise thresholding and scaling.

    Forward:
      w_r: (out_channels, in_channels, kH, kW)
      Compute threshold per-out-channel: t = delta * mean(|w_r|)
      Mask m = (|w_r| > t)
      Sign s = sign(w_r)
      α (scale) per-out-channel = mean(|w_r| * m) / (m.sum() + eps)
      w_q = α * s * m
    Backward:
      Straight-through estimator: pass gradients to w_r as if identity in the quantization range.
    """
    @staticmethod
    def forward(ctx, w_r: torch.Tensor, delta: float = 0.05, eps: float = 1e-8):
        # w_r: shape [OC, IC, kH, kW]
        oc = w_r.shape[0]
        w_view = w_r.view(oc, -1)
        mean_abs = w_view.abs().mean(dim=1, keepdim=True)  # [OC, 1]
        t = delta * mean_abs  # threshold per channel
        # expand threshold to weight shape
        t_full = t.expand_as(w_view)
        m = (w_view.abs() > t_full).float()  # [OC, *]
        s = w_view.sign()
        # per-channel scale alpha
        num = (w_view.abs() * m).sum(dim=1, keepdim=True)
        den = m.sum(dim=1, keepdim=True) + eps
        alpha = num / den  # [OC,1]
        w_q = alpha * s * m  # [-alpha, 0, +alpha]
        # reshape back
        w_q = w_q.view_as(w_r)
        # Save context for backward (only masks needed for sat gradient)
        ctx.save_for_backward(m.view_as(w_r))
        return w_q

    @staticmethod
    def backward(ctx, grad_output):
        (m,) = ctx.saved_tensors
        # Pass gradients through where mask is active; this is a simple STE variant.
        grad_input = grad_output * (m + 0.5*(1.0 - m))  # smaller grad where pruned to zero
        return grad_input, None, None


class UniformActQuantSTE(torch.autograd.Function):
    """
    Uniform k-bit activation quantization with learnable clip (PACT-like).
    y = clamp(x, 0, a)
    q = round(y * S) / S, S = (2^k - 1) / a
    STE in backward: gradient of clamp + identity through round.
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
        # gradient through clamp
        pass_through = (x >= 0.0) & (x <= a)
        grad_x = grad_output * pass_through.float()
        # gradient for 'a' (learnable clip): d clamp(x,0,a)/da = 1_{x > a}
        grad_a = (grad_output * (x > a).float()).sum().unsqueeze(0)
        return grad_x, grad_a, None


# -----------------------------
# Building Blocks
# -----------------------------
class PACT(nn.Module):
    """Learnable clipping for activations with k-bit uniform quantization."""
    def __init__(self, k_bits: int = 8, init_clip: float = 6.0):
        super().__init__()
        self.k_bits = k_bits
        self.a = nn.Parameter(torch.tensor(init_clip, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return UniformActQuantSTE.apply(x, self.a, self.k_bits)


class ReLU2(nn.Module):
    """Squared ReLU activation."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x) ** 2


class BitConv2d(nn.Module):
    """
    Ternary convolution:
      - Pre-norm (GroupNorm) -> Act (ReLU^2) -> Act Quant (PACT) -> Conv with ternary weights.
      - Per-out-channel ternarization + scale via TernaryQuantSTE.
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
    ):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups=min(gn_groups, out_channels), num_channels=in_channels, eps=1e-5, affine=True)
        self.act = ReLU2()
        self.pact = PACT(k_bits=act_bits, init_clip=6.0)
        self.weight_fp = nn.Parameter(torch.empty(out_channels, in_channels // groups, kernel_size, kernel_size))
        nn.init.kaiming_normal_(self.weight_fp, nonlinearity='relu')
        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.stride = stride
        self.padding = padding
        self.groups = groups

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = self.act(x)
        x = self.pact(x)
        w_q = TernaryQuantSTE.apply(self.weight_fp)  # [-α,0,+α] per-channel
        return F.conv2d(x, w_q, self.bias, stride=self.stride, padding=self.padding, groups=self.groups)


class FirstConv(nn.Module):
    """High-precision first conv (FP) with GN + ReLU^2 to stabilize early features."""
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
      Head: GlobalAvgPool -> FP FC
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

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.head_norm(x)
        x = self.head_act(x)
        # Global average pooling
        x = x.mean(dim=[2, 3])
        logits = self.classifier(x)
        return logits


# -----------------------------
# Training & Evaluation
# -----------------------------
@dataclass
class TrainConfig:
    epochs: int = 5
    batch_size: int = 128
    lr: float = 1e-3
    wd: float = 1e-4
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 2
    label_smoothing: float = 0.0
    cosine: bool = True
    amp: bool = True


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def get_data(batch_size: int, num_workers: int):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_ds = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_ds = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader


def accuracy(logits, targets):
    preds = torch.argmax(logits, dim=1)
    return (preds == targets).float().mean().item()


def train_one_epoch(model, loader, optimizer, scaler, cfg: TrainConfig, epoch: int, scheduler=None):
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    n = 0
    for x, y in loader:
        x = x.to(cfg.device, non_blocking=True)
        y = y.to(cfg.device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=cfg.amp and (cfg.device.startswith("cuda"))):
            logits = model(x)
            loss = F.cross_entropy(logits, y, label_smoothing=cfg.label_smoothing)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        if scheduler is not None:
            scheduler.step()
        total_loss += loss.item() * x.size(0)
        total_acc += accuracy(logits.detach(), y) * x.size(0)
        n += x.size(0)
    return total_loss / n, total_acc / n


@torch.no_grad()
def evaluate(model, loader, cfg: TrainConfig):
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    n = 0
    for x, y in loader:
        x = x.to(cfg.device, non_blocking=True)
        y = y.to(cfg.device, non_blocking=True)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        total_loss += loss.item() * x.size(0)
        total_acc += accuracy(logits, y) * x.size(0)
        n += x.size(0)
    return total_loss / n, total_acc / n


def main():
    parser = argparse.ArgumentParser(description="All-in-one Ternary CNN for MNIST")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--channels", type=str, default="32,64,64", help="comma-separated channel sizes")
    parser.add_argument("--no-cosine", action="store_true", help="disable cosine LR schedule")
    parser.add_argument("--no-amp", action="store_true", help="disable mixed precision")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()

    channels = tuple(int(x) for x in args.channels.split(","))

    cfg = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        wd=args.wd,
        seed=args.seed,
        num_workers=args.workers,
        cosine=not args.no_cosine,
        amp=not args.no_amp,
    )

    set_seed(cfg.seed)
    make_dir("./checkpoints")

    print("Loading data...")
    train_loader, test_loader = get_data(cfg.batch_size, cfg.num_workers)

    print("Building model...")
    model = TernaryCNN(channels=channels).to(cfg.device)

    # Optimizer & LR schedule
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd, betas=(0.9, 0.95))
    steps_per_epoch = math.ceil(len(train_loader.dataset) / cfg.batch_size)
    total_steps = steps_per_epoch * cfg.epochs
    if cfg.cosine:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=cfg.lr * 0.01)
    else:
        scheduler = None

    scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp and (cfg.device.startswith("cuda")))

    best_acc = 0.0
    for epoch in range(cfg.epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, scaler, cfg, epoch, scheduler)
        val_loss, val_acc = evaluate(model, test_loader, cfg)
        print(f"Epoch {epoch+1:02d}/{cfg.epochs} | train loss {train_loss:.4f} acc {train_acc*100:.2f}% | "
              f"val loss {val_loss:.4f} acc {val_acc*100:.2f}%")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({"model": model.state_dict(), "cfg": cfg.__dict__}, "./checkpoints/best.pt")

    # Export TorchScript
    model.eval()
    example = torch.randn(1, 1, 28, 28).to(cfg.device)
    traced = torch.jit.trace(model, example)
    ts_path = "./checkpoints/model.ts"
    traced.save(ts_path)
    print(f"Best val acc: {best_acc*100:.2f}%")
    print("Saved best checkpoint to ./checkpoints/best.pt and TorchScript model to ./checkpoints/model.ts")


if __name__ == "__main__":
    main()
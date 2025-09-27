import argparse, math, time, os, copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

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

# ----------------------------
# Fake-quant building blocks (QAT) — activation quantization removed
# ----------------------------
class Bit1p58Weight(nn.Module):
    """1.58-bit (ternary) weight quantizer with per-out-channel scaling."""
    def __init__(self, dim=0, scale_op="mean"):
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
class BitConv2dInfer(nn.Module):
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

class BitLinearInfer(nn.Module):
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

class BitConv2dInferP2(nn.Module):
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
        y = y * torch.pow(2.0, self.s_exp.float())  # shift on integer kernels
        if self.bias is not None:
            y = y + self.bias.view(1, -1, 1, 1)
        return y

class BitLinearInferP2(nn.Module):
    """Ternary linear with power-of-two output scales."""
    def __init__(self, w_q, s_exp, bias):
        super().__init__()
        self.register_buffer("w_q", w_q.to(torch.int8))  # [out,in]
        self.register_buffer("s_exp", s_exp.to(torch.int8))  # [out]
        self.register_buffer("bias", None if bias is None else bias)

    def forward(self, x):
        y = F.linear(x, self.w_q.float(), None)
        y = y * torch.pow(2.0, self.s_exp.float())
        if self.bias is not None:
            y = y + self.bias
        return y

# ----------------------------
# Train-time modules (no BatchNorm), activation quantization removed
# ----------------------------
class BitConv2d(nn.Module):
    """
    Conv2d with ternary weights (fake-quant for training).
    No BatchNorm inside. Add your own nonlinearity outside if desired.
    """
    def __init__(self, in_c, out_c, kernel_size, stride=1, padding=0, dilation=1, groups=1,
                 bias=False, scale_op="mean"):
        super().__init__()
        if isinstance(kernel_size, int):
            kh = kw = kernel_size
        else:
            kh, kw = kernel_size
        self.weight = nn.Parameter(torch.empty(out_c, in_c // groups, kh, kw))
        nn.init.kaiming_normal_(self.weight, nonlinearity="relu")
        self.bias = nn.Parameter(torch.zeros(out_c)) if bias else None
        self.w_q = Bit1p58Weight(dim=0, scale_op=scale_op)
        self.stride, self.padding, self.dilation, self.groups = stride, padding, dilation, groups
        self.scale_op = scale_op

    def forward(self, x):
        wq = self.w_q(self.weight)
        return F.conv2d(x, wq, self.bias, self.stride, self.padding, self.dilation, self.groups)

    @torch.no_grad()
    def to_ternary(self):
        """
        Convert this layer into a frozen BitConv2dInfer, carrying over:
          - per-out-channel weight scale s and Wq in {-1,0,+1}
        """
        w = self.weight.data
        s_vec = _reduce_abs(w, keep_dim=0, op=self.scale_op).squeeze()   # [out]
        s = s_vec.view(-1, 1, 1)                                         # [out,1,1] for conv broadcast
        w_bar = w / s_vec.view(-1, 1, 1, 1)
        w_q = torch.round(w_bar).clamp_(-1, 1).to(w.dtype)

        return BitConv2dInfer(
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

        return BitConv2dInferP2(
            w_q=w_q,
            s_exp=s_exp,
            bias=(None if self.bias is None else self.bias.data.clone()),
            stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups
        )

class BitLinear(nn.Module):
    def __init__(self, in_f, out_f, bias=False, scale_op="mean"):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_f, in_f))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.bias = nn.Parameter(torch.zeros(out_f)) if bias else None
        self.w_q = Bit1p58Weight(dim=0, scale_op=scale_op)
        self.scale_op = scale_op

    def forward(self, x):
        wq = self.w_q(self.weight)
        return F.linear(x, wq, self.bias)

    @torch.no_grad()
    def to_ternary(self):
        w = self.weight.data
        s = _reduce_abs(w, keep_dim=0, op=self.scale_op).squeeze()  # [out]
        w_q = torch.round(w / s.view(-1,1)).clamp_(-1, 1).to(w.dtype)
        return BitLinearInfer(
            w_q=w_q, s=s, bias=(None if self.bias is None else self.bias.data.clone())
        )

    @torch.no_grad()
    def to_ternary_p2(self):
        w = self.weight.data
        s = _reduce_abs(w, keep_dim=0, op=self.scale_op).squeeze()   # [out]
        w_q = torch.round((w / s.view(-1,1))).clamp_(-1, 1).to(w.dtype)

        # Quantize to power-of-two
        _, s_exp = _pow2_quantize_scale(s)   # [out] int8
        return BitLinearInferP2(
            w_q=w_q, s_exp=s_exp, bias=(None if self.bias is None else self.bias.data.clone())
        )

# ----------------------------
# Simple BitNet block & model (optional)
# ----------------------------
class InvertedResidualBit(nn.Module):
    def __init__(self, in_c, out_c, expand, stride, scale_op="mean"):
        super().__init__()
        hid = in_c * expand
        self.use_res = (stride == 1 and in_c == out_c)

        self.pw1 = nn.Sequential(
            BitConv2d(in_c, hid, kernel_size=1, bias=False, scale_op=scale_op),
            nn.ReLU(inplace=True),
        )
        self.dw = nn.Sequential(
            BitConv2d(hid, hid, kernel_size=3, stride=stride, padding=1, groups=hid,
                      bias=False, scale_op=scale_op),
            nn.ReLU(inplace=True),
        )
        # no activation here
        self.pw2 = nn.Sequential(
            BitConv2d(hid, out_c, kernel_size=1, bias=False, scale_op=scale_op),
            nn.BatchNorm2d(out_c),
        )

    def forward(self, x):
        y = self.pw2(self.dw(self.pw1(x)))
        return x + y if self.use_res else y


class BitNetCNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=10, scale_op="mean",
                 drop2d_p=0.3, drop_p=0.3):
        super().__init__()
        self.stem = nn.Sequential(
            BitConv2d(in_channels, 32, kernel_size=3, stride=1, padding=1,
                      bias=False, scale_op=scale_op),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        self.stage1 = InvertedResidualBit(32,   64,  expand=2, stride=2, scale_op=scale_op)
        self.sd1 = nn.Dropout2d(p=drop2d_p)

        self.stage2 = InvertedResidualBit(64,   128, expand=2, stride=2, scale_op=scale_op)
        self.sd2 = nn.Dropout2d(p=drop2d_p)

        self.stage3 = InvertedResidualBit(128,  256, expand=2, stride=2, scale_op=scale_op)

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(p=drop_p),
            BitLinear(256, num_classes, bias=False, scale_op=scale_op),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x); x = self.sd1(x)
        x = self.stage2(x); x = self.sd2(x)
        x = self.stage3(x)
        return self.head(x)

# ----------------------------
# Model-wide conversion helper
# ----------------------------
@torch.no_grad()
def convert_to_ternary(module: nn.Module) -> nn.Module:
    """
    Recursively replace BitConv2d/BitLinear with Ternary*Infer modules.
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
    Recursively replace BitConv2d/BitLinear with their PoT inference counterparts.
    """
    for name, child in list(module.named_children()):
        if hasattr(child, 'to_ternary_p2'):
            setattr(module, name, child.to_ternary_p2())
        else:
            convert_to_ternary_p2(child)
    return module

# -------------------------
# MNIST training/eval
# -------------------------
def get_loaders(data_dir, batch_size):
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_ds = datasets.MNIST(root=data_dir, train=True, download=True, transform=tfm)
    test_ds  = datasets.MNIST(root=data_dir, train=False, download=True, transform=tfm)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=512, shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, test_loader


# Int8 loader: quantizes to [-Q, Q] (default Q=14 for 3x3 ternary with no post-shift)
def get_loaders_int8(data_dir, batch_size, Q=14, num_workers=0, pin_memory=False):
    # Keep transform lightweight: just ToTensor(), then quantize in collate
    tfm = transforms.Compose([
        transforms.ToTensor(),                      # [0,1]
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    def int8_collate(batch):
        xs, ys = zip(*batch)                         # xs: list of [1,H,W] float in [-1,1]
        x = torch.stack(xs, 0)                       # [B,1,H,W]
        # Scale to [-Q, Q], round, clamp, cast to int8
        x_int8 = torch.round(x * 2 * Q).subtract(Q).clamp(-Q, Q)#.to(torch.int8)
        y = torch.tensor(ys, dtype=torch.long)
        return x_int8, y

    train_ds = datasets.MNIST(root=data_dir, train=True,  download=True, transform=tfm)
    test_ds  = datasets.MNIST(root=data_dir, train=False, download=True, transform=tfm)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory,
                              collate_fn=int8_collate)
    test_loader  = DataLoader(test_ds,  batch_size=512, shuffle=False,
                              num_workers=num_workers, pin_memory=pin_memory,
                              collate_fn=int8_collate)
    return train_loader, test_loader

def evaluate(model, loader, device):
    # Build an eval copy so we never mutate the training graph
    model_eval = model.eval()
    total_loss, total_acc, n = 0.0, 0.0, 0
    crit = nn.CrossEntropyLoss()
    with torch.inference_mode():
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            logits = model_eval(x)
            loss = crit(logits, y)
            bs = y.size(0)
            total_loss += loss.item() * bs
            total_acc  += (logits.argmax(1) == y).float().sum().item()
            n += bs
    return total_loss / n, total_acc / n

def train(args,model=None):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    torch.manual_seed(args.seed)
    train_loader, test_loader = get_loaders_int8(args.data, args.batch_size)

    # 1-channel in, 10 classes out
    if model is None:
        model = BitNetCNN(in_channels=1, num_classes=10, scale_op=args.scale_op).to(device)
    best_model = copy.deepcopy(model)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {num_params/1e6:.2f}M  | Device: {device}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    crit = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and args.amp))

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, total, total_loss = time.time(), 0, 0.0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda" and args.amp)):
                logits = model(x)
                loss = crit(logits, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            total += y.size(0)
            total_loss += loss.item() * y.size(0)

        train_loss = total_loss / total
        test_loss, test_acc = evaluate(convert_to_ternary_p2(copy.deepcopy(model)).to(device), test_loader, device)
        train_val_loss, train_val_acc = evaluate(model, test_loader, device)
        

        print(f"Epoch {epoch:02d} | train_loss {train_loss:.4f} | test_loss {test_loss:.4f} | "
              f"test_acc {test_acc*100:.2f}% | train_val_acc {train_val_acc*100:.2f}% | epoch_time {time.time()-t0:.1f}s")

        # Save the best
        if test_acc > best_acc:
            best_model = copy.deepcopy(model)
            best_acc = test_acc
            os.makedirs(args.out, exist_ok=True)
            ckpt_path = os.path.join(args.out, "mnist_bitnet.pt")
            torch.save({"model": model.state_dict(),
                        "args": vars(args),
                        "acc": best_acc}, ckpt_path)
            print(f"✓ Saved checkpoint to {ckpt_path} (acc={best_acc*100:.2f}%)")
            # Also export a frozen ternary snapshot for pure inference
            if not args.no_eval_ternary:
                ternary = convert_to_ternary_p2(copy.deepcopy(model)).cpu().eval()
                ternary_path = os.path.join(args.out, "mnist_bitnet_ternary.pt")
                torch.save({"model": ternary.state_dict(), "acc": best_acc}, ternary_path)
                print(f"✓ Exported frozen ternary model to {ternary_path}")

    print(f"Best test acc: {best_acc*100:.2f}%")
    return best_model

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default="./data", help="MNIST data dir")
    p.add_argument("--out",  type=str, default="./mnist_ckpt", help="where to save checkpoints")
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--scale-op", type=str, default="mean", choices=["mean", "median"])
    p.add_argument("--amp", action="store_true", help="enable mixed precision on CUDA")
    p.add_argument("--cpu", action="store_true", help="force CPU")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-eval-ternary", action="store_true", help="evaluate using frozen ternary model")
    args = p.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    best_model = train(args)
    # Example: manual conversion after training (commented)
    # ternary_model = convert_to_ternary_p2(copy.deepcopy(best_model)).cpu().eval()
    # for n, b in ternary_model.named_buffers():
    #     print(n, b.flatten()[:10], tuple(b.shape), b.dtype)
